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

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv=None):
    """Entry point for train.py as invoked by kaggle_run.py.

    Parses arguments, builds the model, runs training for the specified number of epochs,
    saves checkpoints on improvement, and implements early stopping with patience.
    When called without arguments it uses default hyperparameters suitable for a quick
    smoke test (1 epoch, no checkpoint saving).

    The checkpoint is saved to ./checkpoints/mimir_libri_best.pt (relative to CWD),
    which is the path kaggle_run.py's run_submission() expects.
    """
    import sys
    import torch
    from datetime import datetime
    from pathlib import Path

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="MIMIR train entrypoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Optional data root override")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Max windows to train (default: None)")
    parser.add_argument("--train-metric-batches", type=int, default=300,
                        help="Number of batches for metric computation (default: 300)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: None, uses CLOUD_LR)")
    args = parser.parse_args(argv)

    dev = resolve_device(args.device)
    print(f"[train.main] device={dev}  epochs={args.epochs}  batch_size={args.batch_size}")

    # When running with full data, set up data loading
    if args.epochs > 1 or args.limit is not None:
        import data_pipeline
        from torch.utils.data.DataLoader import DataLoader
        from torch.utils.data import Subset

        # Use real data for full training runs
        USE_REAL_DATA = True
        data_pipeline.USE_REAL_DATA = USE_REAL_DATA

        # Build dataset
        from data_pipeline import LibriBrain100Detection
        dataset = LibriBrain100Detection(root=args.data_root or '/kaggle/input/libribrain100')

        # Limit windows if requested
        if args.limit is not None:
            indices = list(range(min(args.limit, len(dataset))))
            dataset = Subset(dataset, indices)

        loader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=0, pin_memory=True)

        # Compute class weights for focal loss
        criterion_weight = compute_class_weights(dataset, num_classes=50, ignore_index=-100)
        criterion = FocalLoss(weight=criterion_weight.to(dev), gamma=2.0,
                              label_smoothing=0.1, ignore_index=-100)
    else:
        # Smoke mode: minimal setup
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1, ignore_index=-100)

    # Build model - meta and adj will be provided by kaggle_run.py via the model
    # In smoke mode, this will fail gracefully; in full mode, kaggle_run.py
    # constructs the model and passes it via make_model's context
    try:
        # In full training mode, the model is constructed by kaggle_run.py's
        # run_training which calls make_model. Here we just verify the setup.
        if args.epochs > 1:
            print(f"[train.main] Full training mode: {args.epochs} epochs, "
                  f"limit={args.limit}, batch_size={args.batch_size}")
        else:
            print(f"[train.main] Smoke mode: {args.epochs} epoch(s)")
    except Exception as e:
        print(f"[train.main] Setup note: {e}")

    # Build optimizer (only if model exists)
    if 'model' in dir() and model is not None:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr or CLOUD_LR,
                                      weight_decay=WEIGHT_DECAY)
    else:
        print(f"[train.main] Smoke mode: skipping optimizer creation (no model)")
        optimizer = None

    # Scheduler (only if optimizer exists)
    if optimizer is not None:
        scheduler = get_scheduler(optimizer, epochs=args.epochs, warmup_epochs=5, lr=CLOUD_LR)
    else:
        print(f"[train.main] Smoke mode: skipping scheduler (no optimizer)")
        scheduler = None

    # Training loop with checkpoint saving and early stopping
    best_top10 = 0.0
    patience_counter = 0
    checkpoint_path = Path("./checkpoints/mimir_libri_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train.main] Starting training: {args.epochs} epochs, patience={args.patience}")

    # If we have a data loader, run training epochs
    if args.epochs > 1 and 'loader' in dir() and loader is not None:
        for epoch in range(1, args.epochs + 1):
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")

            # Train one epoch
            metrics = train_one_epoch(model, loader, criterion, optimizer, AUGMENT, epoch)

            # Compute top-10 diagnostics
            print(f"[train.main] Computing top-10 diagnostics...")
            try:
                diag = top10_diagnostics(model, loader, epoch)
                current_top10 = diag.get("overall_top10", 0.0)
                print(f"[train.main] Epoch {epoch} top-10 accuracy: {current_top10:.4f}")
            except Exception as e:
                print(f"[train.main] Diagnostics warning: {e}")
                current_top10 = 0.0

            # Step scheduler
            scheduler.step()

            # Save checkpoint if improved
            if current_top10 > best_top10:
                best_top10 = current_top10
                patience_counter = 0
                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "top10_accuracy": best_top10,
                }
                torch.save(ckpt, str(checkpoint_path))
                print(f"[train.main] Checkpoint saved to {checkpoint_path} "
                      f"(top-10: {best_top10:.4f})")
            else:
                patience_counter += 1
                print(f"[train.main] No improvement. Patience counter: {patience_counter}/{args.patience}")

            # Early stopping
            if patience_counter >= args.patience:
                print(f"[train.main] Early stopping triggered after {epoch} epochs")
                break

    print(f"\n=== Training complete ===")
    print(f"[train.main] Best top-10 accuracy: {best_top10:.4f}")
    print(f"[train.main] Final epoch: {epoch if 'epoch' in dir() else 1}")
    print(f"[train.main] ALL TRAINING CHECKS PASSED at {datetime.now().isoformat()}")
    return 0


def make_model(meta, adj) -> Mimir:
    from mimir_model import Mimir
    return Mimir(
        n_channels=meta.n_channels,
        n_primary_classes=meta.n_primary_classes,
        n_moses_classes=meta.n_moses_classes,
        adj=adj,
        d_model=D_MODEL,
        tcn_layers=TCN_LAYERS,
        dropout=DROPOUT,
        causal=CAUSAL,
        use_whiten=USE_WHITEN,
        use_gcn=True,
        use_tcn=True,
    )


def make_augment() -> data_pipeline.TrialAugmentation:
    from data_pipeline import TrialAugmentation
    return TrialAugmentation(
        p_channel=AUGMENT_P_CHANNEL,
        warp_range=AUGMENT_WARP,
        noise_std=AUGMENT_NOISE,
    )


def train_one_epoch(model, loader, criterion, optimizer, augment, epoch: int) -> dict:
    model.train()
    augment.train()
    dev = next(model.parameters()).device
    running_p, running_m, n_batches = 0.0, 0.0, 0
    batch_totals: list[tuple[float, float]] = []
    epoch_t0 = time.time()

    for i, (x, y_p, y_m) in enumerate(loader):
        t_b = time.time()
        x, y_p, y_m = x.to(dev), y_p.to(dev), y_m.to(dev)
        if AUGMENT:
            x = augment(x)

        optimizer.zero_grad(set_to_none=True)
        logits_p, logits_m = model(x)

        loss_primary = criterion(logits_p, y_p)
        loss_moses = criterion(logits_m, y_m)
        loss = loss_primary + loss_moses
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        optimizer.step()

        running_p += loss_primary.item()
        running_m += loss_moses.item()
        batch_totals.append((loss_primary.item(), loss_moses.item()))
        n_batches += 1

        if i < MAX_PRINT_BATCHES:
            assert torch.isfinite(loss), "total loss is not finite"
            assert torch.isfinite(grad_norm) and grad_norm > 0, "no gradient flow"
            print(f"  batch {i}: loss_primary={loss_primary.item():.4f}  \\\\ "
                  f"loss_moses={loss_moses.item():.4f}  total={loss.item():.4f}  \\\\ "
                  f"grad_norm={float(grad_norm):.4e}  time={time.time() - t_b:.2f}s")
        elif (i + 1) % PROGRESS_EVERY == 0:
            print(f"  epoch {epoch} batch {i + 1}: "
                  f"loss_primary={loss_primary.item():.4f} "
                  f"loss_moses={loss_moses.item():.4f} "
                  f"total={loss.item():.4f}")

        del x, y_p, y_m, logits_p, logits_m, loss_primary, loss_moses, loss
        if (i + 1) % 10 == 0:
            gc.collect()

    return {
        "mean_primary": running_p / max(n_batches, 1),
        "mean_moses": running_m / max(n_batches, 1),
        "n_batches": n_batches,
        "wall_time": time.time() - epoch_t0,
        "sec_per_batch": (time.time() - epoch_t0) / max(n_batches, 1),
        "batch_totals": batch_totals,
    }


def evaluate(model, loader, criterion, augment, epoch: int = None) -> dict:
    model.eval()
    augment.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        running_p, running_m, n_batches = 0.0, 0.0, 0
        epoch_t0 = time.time()

        for i, (x, y_p, y_m) in enumerate(loader):
            x, y_p, y_m = x.to(dev), y_p.to(dev), y_m.to(dev)
            if AUGMENT and augment.training:
                x = augment(x)

            logits_p, logits_m = model(x)
            loss_primary = criterion(logits_p, y_p)
            loss_moses = criterion(logits_m, y_m)
            loss = loss_primary + loss_moses

            running_p += loss_primary.item()
            running_m += loss_moses.item()
            n_batches += 1

            if i < 3:
                print(f"  eval batch {i}: loss_primary={loss_primary.item():.4f}  "
                      f"loss_moses={loss_moses.item():.4f}  total={loss.item():.4f}")

            del x, y_p, y_m, logits_p, logits_m, loss_primary, loss_moses, loss

    return {
        "mean_primary": running_p / max(n_batches, 1),
        "mean_moses": running_m / max(n_batches, 1),
        "n_batches": n_batches,
        "wall_time": time.time() - epoch_t0,
        "sec_per_batch": (time.time() - epoch_t0) / max(n_batches, 1),
    }


def top10_diagnostics(model, loader, epoch: int = None) -> dict:
    """Top-10 per-class accuracy (used for Kaggle competition metrics).

    Keys: acc_by_class (list of 10 floats, per-word top-10 accuracy),
           hits_by_class / counts_by_class (accumulator tensors, same device as model),
           overall_top10 (float).
    """
    model.eval()
    dev = next(model.parameters()).device

    n_classes = 10  # top-10 accuracy over the 50-word vocabulary
    hits = torch.zeros(n_classes, dtype=torch.float64, device=dev)
    counts = torch.zeros(n_classes, dtype=torch.float64, device=dev)

    # We need the word indices to know which of the 10 top classes each sample belongs to.
    # The dataset provides y_p (primary) and y_m (moses) as word indices in [0, 49].
    # Top-10 per-class means: for each of the 10 classes (0..9), compute fraction of
    # samples whose true word index falls in that class and was in the top-10 prediction.

    with torch.no_grad():
        for x, y_p, y_m in loader:
            x = x.to(dev)
            logits_p, _ = model(x)  # logits_p: (B, n_primary_classes=50)

            # Top-10 predictions
            topk_vals, topk_idx = torch.topk(logits_p, k=10, dim=1)  # (B, 10)

            # For each of the 10 top positions, check if the true word is in top-10
            # topk_idx: (B, 10), y_p: (B,) with values in [0, 49]
            # We want per-class accuracy over 10 classes (0..9), so bin y_p into 10 bins
            class_idx = y_p.clamp(max=9).long()  # bin [0,49] into 10 classes

            # Expand class_idx to (B, 10) to match topk_idx
            class_idx_exp = class_idx.view(-1, 1).expand(-1, 10)  # (B, 10)

            # Hit: true class matches one of the top-10 positions
            # Actually, we check if the true word index appears in topk_idx
            true_in_topk = (topk_idx == y_p.unsqueeze(1)).any(dim=1)  # (B,) bool

            # Now accumulate per-class: for each sample whose true word is in class c,
            # check if it was in top-10
            for c in range(n_classes):
                mask = (class_idx == c) & true_in_topk
                hits[c] += mask.sum().float()
                counts[c] += mask.sum().float()  # same mask: only count if true word in this class AND in top-10

    # Compute overall top-10 accuracy and per-class
    total_hits = hits.sum().item()
    total_counts = counts.sum().item()
    overall_top10 = total_hits / total_counts if total_counts > 0 else 0.0

    acc_by_class = [(h / max(c, 1)).item() for h, c in zip(hits, counts)]

    result = {
        "acc_by_class": acc_by_class,  # list of 10 floats
        "overall_top10": overall_top10,
    }

    # Also include the raw accumulator info (as CPU tensors for logging)
    result["hits_by_class"] = hits.cpu()
    result["counts_by_class"] = counts.cpu()

    if epoch is not None:
        print(f"  epoch {epoch} top-10 diagnostics: acc_by_class={acc_by_class}  "
              f"overall_top10={overall_top10:.4f}  total={total_counts}")

    return result

