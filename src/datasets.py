import os
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import torch
from torch.utils.data import Dataset

import albumentations as A


# 4 channels :intensity, entropy, anisotropy, alpha
class Sen1OilDataset(Dataset):
    def __init__(self, data_dir, dataset_ids, transform=None, load_labels=True, supcon=False):
        self.load_labels = load_labels
        self.supcon = supcon
        self.image_dir = os.path.join(data_dir, 'image')
        self.label_dir = os.path.join(data_dir, 'label')
        self.dataset_ids = dataset_ids
        self.transform = transform

        self.image_files = []
        self.label_files = []

        for dataset_id in self.dataset_ids:
            image_file = os.path.join(self.image_dir, dataset_id)
            label_file = os.path.join(self.label_dir, dataset_id)

            if os.path.exists(image_file) and os.path.exists(label_file):
                self.image_files.append(image_file)
                self.label_files.append(label_file)
            else:
                print(f"Warning: Missing files for dataset ID {dataset_id}")
                if not os.path.exists(image_file):
                    print(f"  Image file not found: {image_file}")
                if not os.path.exists(label_file):
                    print(f"  Mask file not found: {label_file}")

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found for the given dataset IDs in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        img = self.load_image(image_path)
        label = self.load_label(label_path) if self.load_labels else None

        img /= 255.0

        if self.transform:
            if self.supcon:
                aug1 = self.transform(image=img, mask=label)
                aug2 = self.transform(image=img, mask=label)
                img1, label1 = aug1['image'], aug1['mask']
                img2, label2 = aug2['image'], aug2['mask']

                img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
                img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
                label1 = torch.from_numpy(label1).long()
                label2 = torch.from_numpy(label2).long()

                return (img1, img2), (label1, label2), os.path.basename(image_path)
            else:
                augmented = self.transform(image=img, mask=label)
                img, label = augmented['image'], augmented['mask']

                img = torch.from_numpy(img).float().permute(2, 0, 1)
                label = torch.from_numpy(label).long()

        return img, label, os.path.basename(image_path)

    def load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read([1,2,4]).astype(np.float32)  # use only intensity, entropy, alpha feature
            img = np.transpose(img, (1, 2, 0))  # CxHxW -> HxWxC
        return img

    def load_label(self, path):
        with rasterio.open(path) as src:
            label = src.read(1).astype(np.int64)
        return label


def get_supervised_datasets(data_path, train_ids, val_ids, test_ids, supcon=False):
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.4471, 0.6659, 0.3395], std=[0.1112, 0.2183, 0.1733]), # use only intensity, entropy, alpha feature
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.4471, 0.6659, 0.3395], std=[0.1112, 0.2183, 0.1733]), # use only intensity, entropy, alpha feature
    ])

    test_transform = val_transform

    train_dataset = Sen1OilDataset(data_path, train_ids, transform=train_transform, supcon=supcon)
    val_dataset = Sen1OilDataset(data_path, val_ids, transform=val_transform, supcon=supcon)
    test_dataset = Sen1OilDataset(data_path, test_ids, transform=test_transform, supcon=supcon)

    return train_dataset, val_dataset, test_dataset