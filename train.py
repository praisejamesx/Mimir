from __future__ import annotations

import argparse
import gc
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
SEED = 0
D_MODEL = 64
TCN_LAYERS = 7           # dilations 1..64 -> RF = 255 samples = 1.02 s @250 Hz
DROPOUT = 0.1
CAUSAL = False           # offline classification: symmetric two-sided kernel
USE_WHITEN = True        # per-trial channel decorrelation front-end
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Anti-overfitting augmentation (data_pipeline.TrialAugmentation). Applied only
# in train mode; evaluate() forces eval() so validation stays clean.
AUGMENT = True
AUGMENT_P_CHANNEL = 0.10
AUGMENT_WARP = (0.95, 1.05)
AUGMENT_NOISE = 0.02

# Local smoke configuration.
BATCH_SIZE = 8           # small batches keep peak activation memory tiny
MAX_PRINT_BATCHES = 5

# Cloud configuration (used only when data_pipeline.USE_REAL_DATA is True).
CLOUD_BATCH_SIZE = 32
CLOUD_EPOCHS = 20
CLOUD_PATIENCE = 3
CLOUD_LR = 1e-3
PROGRESS_EVERY = 200
CLOUD_TRAIN_METRIC_BATCHES = 300

# Resolved at main() time from --device (auto|cpu|cuda); everything (model,
# batches, labels, augmentation) moves here. The code stays device-agnostic:
# the same commands run on CPU or GPU.
DEVICE = torch.device("cpu")

# ----------------------------------------------------------------------------
# Class weighting & focal loss
# ----------------------------------------------------------------------------
def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def compute_class_weights(dataset, num_classes: int, ignore_index: int = -100) -> torch.Tensor:
    """Compute sqrt-inverse-frequency class weights, clamped to [0.5, 5.0].

    Used for focal loss to counter Zipfian class imbalance in the 50-word vocabulary.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    valid_total = 0
    for _, y_p, _ in torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=0):
        v = y_p != ignore_index
        if v.any():
            counts.index_add_(0, y_p[v].long(), torch.ones_like(y_p[v], dtype=torch.float32))
            valid_total += v.sum().item()
    if valid_total == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    freq = counts / (valid_total + 1e-6)
    weights = (freq + 1e-6).pow(-0.5)  # sqrt-inverse
    weights = weights.clamp(0.5, 5.0)
    weights = weights / weights.mean()  # normalize
    return weights


class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing.

    Args:
        weight: per-class loss weight (None = no weighting)
        gamma: focal modulating factor (gamma=0 reduces to CE)
        label_smoothing: label smoothing factor (0 = no smoothing)
        ignore_index: target value to ignore
    """
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.1,
                 ignore_index: int = -100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def get_scheduler(optimizer, epochs: int, warmup_epochs: int = 5, lr: float = 1e-3):
    """Cosine LR with linear warmup at the beginning.

    Warmup: LinearLR from lr*0.01 to lr over warmup_epochs.
    Cosine: CosineAnnealingLR from lr to lr/50 over remaining epochs.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr / 50.0)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs])
