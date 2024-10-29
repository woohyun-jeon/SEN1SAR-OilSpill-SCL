import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(dataset):
    total_pixels = 0
    fire_pixels = 0
    for item in dataset:
        if isinstance(item, tuple) and len(item) >= 2:
            _, mask = item[:2]
        else:
            mask = item

        if isinstance(mask, tuple):
            mask = mask[0]

        if isinstance(mask, torch.Tensor):
            total_pixels += mask.numel()
            fire_pixels += torch.sum(mask).item()
        elif isinstance(mask, np.ndarray):
            total_pixels += mask.size
            fire_pixels += np.sum(mask)
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

    fire_ratio = fire_pixels / total_pixels

    return torch.tensor([1 / (1 - fire_ratio), 1 / fire_ratio])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt + self.eps) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)

        # get mask of positive pairs based on labels
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        # compute similarity matrix
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature  # [B, B]

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # create mask that excludes self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # weight mask based on oil spill ratio if available
        if labels.sum() > 0:  # if there are positive samples
            pos_samples = (labels == 1).float()
            weight = pos_samples / (pos_samples.sum() + 1e-6)
            mask = mask * weight

        # compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log probabilities for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # scale loss by base temperature
        loss = -(self.base_temperature / self.temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss