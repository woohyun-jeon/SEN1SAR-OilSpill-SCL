import os
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import get_supervised_datasets
from models import get_model
from utils import load_dataset_ids, load_config, set_seed, EarlyStopping
from loss import FocalLoss, SupConLoss, calculate_class_weights
from metrics import get_metrics


# execute supervised training
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config):
    model.to(device)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config['params']['early_stopping_patience'], min_delta=config['params']['early_stopping_min_delta'])
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(config['params']['sup_epochs']):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch [{epoch + 1}/{config['params']['sup_epochs']}]")):
            # for SupCon models
            if config['model']['type'] in ['SupConUNet', 'SupConDeepLabV3Plus']:
                (img1, img2), (label1, label2), _ = batch
                img1, img2 = img1.to(device), img2.to(device)
                label1, label2 = label1.to(device), label2.to(device)

                optimizer.zero_grad()

                seg1, feat1 = model(img1)
                seg2, feat2 = model(img2)

                if label1.dim() == 2:
                    label1 = label1.unsqueeze(0)
                if label2.dim() == 2:
                    label2 = label2.unsqueeze(0)
                if label1.dim() == 3:
                    label1 = label1.unsqueeze(1)
                if label2.dim() == 3:
                    label2 = label2.unsqueeze(1)

                label1 = label1.float()
                label2 = label2.float()

                # segmentation loss
                seg_loss = (loss_fn['seg'](seg1, label1) + loss_fn['seg'](seg2, label2)) / 2

                # contrastive loss
                label1_small = F.interpolate(label1, size=(64, 64), mode='nearest')
                label2_small = F.interpolate(label2, size=(64, 64), mode='nearest')

                features = torch.cat([feat1, feat2], dim=0)
                pixel_labels = torch.cat([label1_small, label2_small], dim=0)

                features = features.permute(0, 2, 3, 1).reshape(-1, features.size(1))
                pixel_labels = pixel_labels.reshape(-1)

                con_loss = loss_fn['con'](features, pixel_labels)

                # total loss
                loss = seg_loss + config['params'].get('lambda_con', 1.0) * con_loss
            else:
                img, label, _ = batch
                img = img.float().to(device)
                label = label.long().to(device)

                optimizer.zero_grad()

                outputs = model(img)

                if label.dim() == 2:
                    label = label.unsqueeze(0)
                if label.dim() == 3:
                    label = label.unsqueeze(1)

                label = label.float()

                loss = loss_fn(outputs, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['params']['sup_epochs']}], Train Loss: {avg_train_loss:.4f}")

        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, loss_fn, device, config)
        avg_val_loss = val_metrics['loss']
        print(f"Epoch [{epoch + 1}/{config['params']['sup_epochs']}], Val Loss: {avg_val_loss:.4f}")

        if early_stopping.step(avg_val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"Best model saved at epoch {epoch + 1}")

    return best_model_state, val_metrics


def evaluate_model(model, dataloader, loss_fn, device, config):
    model.eval()
    all_metrics = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if config['model']['type'] in ['SupConUNet', 'SupConDeepLabV3Plus']:
                (images, _), (labels, _), _ = batch
            else:
                images, labels, _ = batch

            if isinstance(images, list):
                images = images[0]
            if isinstance(labels, list):
                labels = labels[0]

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            if config['model']['type'] in ['SupConUNet', 'SupConDeepLabV3Plus']:
                outputs, _ = model(images)
                loss = loss_fn['seg'](outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            if outputs is not None and labels is not None:
                metrics = get_metrics(outputs, labels)
                all_metrics.append(metrics)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {'loss': avg_loss}

    if all_metrics:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


def save_predictions(model, dataloader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch[0], tuple):
                (images, _), _, filenames = batch
            else:
                images, _, filenames = batch

            if isinstance(images, list):
                images = images[0]

            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.cpu().numpy()

            for j, output in enumerate(outputs):
                save_path = os.path.join(save_dir, f"{filenames[j]}.tif")
                output = output.squeeze()
                if output.ndim == 2:
                    height, width = output.shape
                    count = 1
                elif output.ndim == 3:
                    count, height, width = output.shape
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")

                with rasterio.open(save_path, 'w', driver='GTiff', height=height, width=width, count=count, dtype=output.dtype) as dst:
                    if count == 1:
                        dst.write(output, 1)
                    else:
                        dst.write(output)


def prepare_model_and_loss(config, train_dataset, device):
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    pos_weight = class_weights[1] / (class_weights[0] + class_weights[1])
    pos_weight = torch.tensor([pos_weight]).to(device)

    if config['model']['pretrained'] == 'ImageNet':
        model = get_model(config['model']['type'], config['model']['encoder'], pretrained=True)
        criterion = FocalLoss(pos_weight=pos_weight).to(device)
    elif config['model']['pretrained'] == 'SupCon':
        model = get_model(config['model']['type'], config['model']['encoder'], pretrained=False)
        criterion = {
            'seg': FocalLoss(pos_weight=pos_weight).to(device),
            'con': SupConLoss(temperature=0.07).to(device)
        }
    elif config['model']['pretrained'] is None:
        model = get_model(config['model']['type'], config['model']['encoder'], pretrained=False)
        criterion = FocalLoss(pos_weight=pos_weight).to(device)

    return model.to(device), criterion


def main():
    config_path = 'configs.yaml'
    cfgs = load_config(config_path)
    set_seed(seed=cfgs['params']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset IDs
    train_ids = load_dataset_ids(os.path.join(cfgs['path']['sup_path'], 'train.txt'))
    val_ids = load_dataset_ids(os.path.join(cfgs['path']['sup_path'], 'valid.txt'))
    test_ids = load_dataset_ids(os.path.join(cfgs['path']['sup_path'], 'test.txt'))

    models = [
        {'name': 'SupConUNet', 'encoder': 'resnet50', 'pretrained': 'SupCon'},
        {'name': 'UNet', 'encoder': 'resnet50', 'pretrained': 'ImageNet'},
        {'name': 'UNet', 'encoder': 'resnet50', 'pretrained': None},
        {'name': 'SupConUNet', 'encoder': 'resnet101', 'pretrained': 'SupCon'},
        {'name': 'UNet', 'encoder': 'resnet101', 'pretrained': 'ImageNet'},
        {'name': 'UNet', 'encoder': 'resnet101', 'pretrained': None},
        {'name': 'SupConDeepLabV3Plus', 'encoder': 'resnet50', 'pretrained': 'SupCon'},
        {'name': 'DeepLabV3Plus', 'encoder': 'resnet50', 'pretrained': 'ImageNet'},
        {'name': 'DeepLabV3Plus', 'encoder': 'resnet50', 'pretrained': None},
        {'name': 'SupConDeepLabV3Plus', 'encoder': 'resnet101', 'pretrained': 'SupCon'},
        {'name': 'DeepLabV3Plus', 'encoder': 'resnet101', 'pretrained': 'ImageNet'},
        {'name': 'DeepLabV3Plus', 'encoder': 'resnet101', 'pretrained': None},
    ]

    results = {}

    for model_config in models:
        model_name = model_config['name']
        encoder_name = model_config['encoder']
        pretraining = model_config['pretrained']

        print(f"\nTraining {model_name} with {encoder_name} ({pretraining if pretraining else 'No Pretraining'})")

        cfgs['model'] = {
            'type': model_name,
            'encoder': encoder_name,
            'pretrained': pretraining
        }

        use_supcon = model_name in ['SupConUNet', 'SupConDeepLabV3Plus']

        # get datasets and dataloaders
        train_dataset, val_dataset, test_dataset = get_supervised_datasets(
            cfgs['path']['sup_path'], train_ids, val_ids, test_ids, supcon=use_supcon
        )
        train_loader = DataLoader(train_dataset, batch_size=cfgs['params']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        model, criterion = prepare_model_and_loss(cfgs, train_dataset, device)

        optimizer = optim.Adam(model.parameters(), lr=cfgs['params']['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfgs['params']['sup_epochs'])

        # train
        best_model_state, best_val_metrics = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=criterion, device=device, config=cfgs
        )

        # evaluate
        model.load_state_dict(best_model_state)
        train_metrics = evaluate_model(model, train_loader, criterion, device, cfgs)
        val_metrics = best_val_metrics
        test_metrics = evaluate_model(model, test_loader, criterion, device, cfgs)

        results[f"{model_name}_{pretraining}"] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

        print(f"\n{model_name} with {pretraining} pretraining - Final Results:")
        for split in ['train', 'val', 'test']:
            print(f"  {split.capitalize()} Metrics:")
            metrics = results[f"{model_name}_{pretraining}"][split]
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

        # save model and predictions
        output_dir = os.path.join(cfgs['path']['out_path'], f'{model_name}_{encoder_name}_{pretraining}')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_{encoder_name}_{pretraining}.pth'))

        for split, loader in dataloaders.items():
            save_predictions(model=model, dataloader=loader, save_dir=os.path.join(output_dir, f'{split}_predictions'), device=device)

    # save final results summary
    print("\nFinal Results Summary:")
    with open(os.path.join(cfgs['path']['out_path'], 'result_summary.txt'), 'w') as f:
        for model_name, result in results.items():
            print(f"{model_name}:")
            f.write(f"{model_name}:\n")
            for split in ['train', 'val', 'test']:
                print(f"  {split.capitalize()} Metrics:")
                f.write(f"  {split.capitalize()} Metrics:\n")
                metrics = result[split]
                for metric in ['iou', 'precision', 'recall', 'f1_score', 'accuracy']:
                    value = metrics.get(metric, 0.0)
                    print(f"    {metric}: {value:.4f}")
                    f.write(f"    {metric}: {value:.4f}\n")
                print()
                f.write("\n")
            print()
            f.write("\n")


if __name__ == "__main__":
    main()