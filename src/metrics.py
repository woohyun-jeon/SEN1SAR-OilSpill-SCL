import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score


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