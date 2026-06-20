import yaml
import random
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score

import torch


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif abs(self.best_score - current_score) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        return self.early_stop


def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def get_metrics(preds, targets, threshold=0.5, epsilon=1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float().cpu().numpy()
    targets_bin = (targets > threshold).float().cpu().numpy()

    precision = precision_score(targets_bin.flatten(), preds_bin.flatten(), zero_division=1)
    recall = recall_score(targets_bin.flatten(), preds_bin.flatten(), zero_division=1)
    f1 = f1_score(targets_bin.flatten(), preds_bin.flatten(), zero_division=1)
    accuracy = accuracy_score(targets_bin.flatten(), preds_bin.flatten())
    ap = average_precision_score(targets_bin.flatten(), probs.cpu().numpy().flatten())

    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'average_precision': ap,
        'iou': iou
    }
