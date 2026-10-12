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

# Readout + objective. Attention pooling preserves the temporal trajectory that
# carries word identity (mean pooling collapsed balanced Top-10 to chance);
# sqrt-inverse-frequency class weights + label smoothing counter the Zipfian
# 50-word label distribution.
POOL = "attn"
LOSS = "focal"           # "focal" or "ce"
GAMMA = 2.0
CLASS_WEIGHTS = "sqrt-inv"   # "sqrt-inv" or "none"
LABEL_SMOOTHING = 0.1

# Optimization.
ACCUM_STEPS = 4          # effective batch = batch_size * accum_steps
CLIP_GRAD = 1.0          # real gradient clip (the old 1e9 was effectively off)
WARMUP_EPOCHS = 5
BALANCED_SAMPLING = True

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
def compute_class_weights(dataset, num_classes: int, ignore_index: int = -100,
                          head: str = "primary") -> torch.Tensor:
    """Compute sqrt-inverse-frequency class weights, clamped to [0.5, 5.0].

    Used for focal loss to counter Zipfian class imbalance in the 50-word
    vocabulary. ``head`` selects which label column to tabulate (the two
    competition vocabularies have independent orderings, so each head needs its
    own weight vector).
    """
    assert head in ("primary", "moses"), head
    col = 1 if head == "primary" else 2
    counts = torch.zeros(num_classes, dtype=torch.float32)
    valid_total = 0
    for batch in torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=0):
        y = batch[col]
        v = y != ignore_index
        if v.any():
            counts.index_add_(0, y[v].long(), torch.ones_like(y[v], dtype=torch.float32))
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
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce = ce[valid]
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

def make_criterion(dataset, num_classes: int, loss: str = LOSS,
                   class_weights: str = CLASS_WEIGHTS, gamma: float = GAMMA,
                   label_smoothing: float = LABEL_SMOOTHING, head: str = "primary",
                   device=None):
    """Build the per-head loss with optional sqrt-inverse class weighting.

    Returns ``(criterion, weights)``. ``loss="ce"`` -> CrossEntropyLoss;
    ``loss="focal"`` -> FocalLoss(gamma). ``class_weights="none"`` disables
    weighting. Each head must own its own weight vector (the two competition
    vocabularies have independent orderings).
    """
    weights = None
    if class_weights == "sqrt-inv":
        weights = compute_class_weights(dataset, num_classes, head=head)
        if device is not None:
            weights = weights.to(device)
    elif class_weights != "none":
        raise ValueError(f"unknown class_weights: {class_weights}")

    if loss == "focal":
        crit = FocalLoss(weight=weights, gamma=gamma,
                         label_smoothing=label_smoothing, ignore_index=-100)
    elif loss == "ce":
        crit = nn.CrossEntropyLoss(weight=weights,
                                   label_smoothing=label_smoothing,
                                   ignore_index=-100)
    else:
        raise ValueError(f"unknown loss: {loss}")
    return crit, weights


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv=None):
    """Entry point for train.py as invoked by kaggle_run.py run_training().

    Modes:
      * overfit (--overfit N): tiny fixed subset, no augmentation, no class
        weighting, many epochs. Proves the pipeline is learnable end-to-end
        before spending a full Kaggle session.
      * full (--epochs > 1 or --limit set): real LibriBrain100 training with
        balanced sampling, gradient accumulation, both-head balanced Top-10
        early stopping, and checkpointing.
      * smoke (neither): no data, no training.

    The checkpoint is saved next to this file at checkpoints/mimir_libri_best.pt,
    which is the path kaggle_run.py's run_submission() expects.
    """
    import sys
    from datetime import datetime

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="MIMIR train entrypoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device (default: auto)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Optional data root override")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Max windows to train (default: None)")
    parser.add_argument("--train-metric-batches", type=int, default=300,
                        help="Number of batches for metric computation (default: 300)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: None, uses CLOUD_LR)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS,
                        help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--clip-grad", type=float, default=CLIP_GRAD)
    parser.add_argument("--loss", choices=["focal", "ce"], default=LOSS,
                        help="Per-head loss (default: focal)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Focal loss gamma")
    parser.add_argument("--class-weights", choices=["sqrt-inv", "none"], default=CLASS_WEIGHTS,
                        help="Per-head sqrt-inverse-frequency class weighting")
    parser.add_argument("--pool", choices=["mean", "attn"], default=POOL,
                        help="Classifier head pooling (default: attn)")
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--tcn-layers", type=int, default=TCN_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--augment", dest="augment", action="store_true", default=AUGMENT)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--balanced-sampling", dest="balanced_sampling",
                        action="store_true", default=BALANCED_SAMPLING)
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.add_argument("--overfit", type=int, default=0,
                        help="Diagnostic: train on N fixed windows (no augment/weights)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    dev = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train.main] device={dev}  epochs={args.epochs}  batch_size={args.batch_size}  "
          f"loss={args.loss}  pool={args.pool}  class_weights={args.class_weights}  "
          f"balanced_sampling={args.balanced_sampling}  accum={args.accum_steps}")

    # Anchor the checkpoint to this file (not CWD) so it matches
    # kaggle_run.run_submission() regardless of the invoking directory.
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "mimir_libri_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    hp = {
        "d_model": args.d_model,
        "tcn_layers": args.tcn_layers,
        "dropout": args.dropout,
        "causal": CAUSAL,
        "use_whiten": USE_WHITEN,
        "use_gcn": True,
        "use_tcn": True,
        "pool": args.pool,
    }

    if args.overfit > 0:
        best_top10, final_epoch = _run_overfit(args, dev, hp)
    elif args.epochs > 1 or args.limit is not None:
        best_top10, final_epoch = _run_full(args, dev, hp, checkpoint_path)
    else:
        print(f"[train.main] Smoke mode: {args.epochs} epoch(s)")
        best_top10, final_epoch = 0.0, 0

    print(f"\n=== Training complete ===")
    print(f"[train.main] Best balanced Top-10: {best_top10:.4f}")
    print(f"[train.main] Final epoch: {final_epoch}")
    print(f"[train.main] ALL TRAINING CHECKS PASSED at {datetime.now().isoformat()}")
    return 0


def _save_checkpoint(path, model, optimizer, meta, adj, hp, epoch, best_top10):
    ckpt = {
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "meta": asdict(meta),
        "hyperparams": hp,
        "adj": adj.detach().cpu() if torch.is_tensor(adj) else adj,
        "val_bal_top10": best_top10,
    }
    torch.save(ckpt, str(path))


def _run_full(args, dev, hp, checkpoint_path):
    """Real LibriBrain100 training with both-head balanced Top-10 early stop."""
    import data_pipeline
    data_pipeline.USE_REAL_DATA = True
    print(f"[train.main] Full training mode: {args.epochs} epochs, limit={args.limit}, "
          f"batch_size={args.batch_size}, balanced_sampling={args.balanced_sampling}")

    dataset, adj, meta, loader = data_pipeline.build(
        data_path=Path(args.data_root) if args.data_root else None,
        batch_size=args.batch_size,
        max_windows=args.limit,
        balanced_sampling=args.balanced_sampling,
    )
    val_loader = getattr(dataset, "val_loader", None)
    print(f"[train.main] data ready: {len(dataset)} train windows, "
          f"{len(getattr(dataset, 'val_dataset', []))} val windows")

    model = make_model(meta, adj, hp).to(dev)
    augment = make_augment()

    # Per-head losses: each head owns its own class-weight vector.
    crit_p, _ = make_criterion(dataset, meta.n_primary_classes, loss=args.loss,
                               class_weights=args.class_weights, gamma=args.gamma,
                               label_smoothing=LABEL_SMOOTHING, head="primary", device=dev)
    crit_m, _ = make_criterion(dataset, meta.n_moses_classes, loss=args.loss,
                               class_weights=args.class_weights, gamma=args.gamma,
                               label_smoothing=LABEL_SMOOTHING, head="moses", device=dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or CLOUD_LR,
                                  weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer, epochs=args.epochs,
        warmup_epochs=max(1, min(args.warmup_epochs, max(1, args.epochs // 4))),
        lr=args.lr or CLOUD_LR,
    )

    best_top10, patience_counter, final_epoch = 0.0, 0, 0
    diag_loader = val_loader if val_loader is not None else loader
    print(f"[train.main] Starting training: {args.epochs} epochs, patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        metrics = train_one_epoch(model, loader, crit_p, crit_m, optimizer, augment, epoch,
                                  accum_steps=args.accum_steps, clip_grad=args.clip_grad,
                                  augment_on=args.augment)
        print(f"[train.main] epoch {epoch} train: primary={metrics['mean_primary']:.4f} "
              f"moses={metrics['mean_moses']:.4f} "
              f"({metrics['sec_per_batch'] * 1000:.1f} ms/batch)")

        diag_p = top10_diagnostics(model, diag_loader, head="primary", epoch=epoch)
        diag_m = top10_diagnostics(model, diag_loader, head="moses")
        current_top10 = diag_p["overall_top10"]
        print(f"[train.main] Epoch {epoch} balanced: "
              f"primary top10={diag_p['overall_top10']:.4f} "
              f"top1={diag_p['balanced_top1']:.4f} | "
              f"moses top10={diag_m['overall_top10']:.4f} "
              f"top1={diag_m['balanced_top1']:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Always write a checkpoint on epoch 1 so submit.py always has one.
        improved = current_top10 > best_top10
        if improved or epoch == 1:
            best_top10 = max(best_top10, current_top10)
            patience_counter = 0 if improved else patience_counter
            _save_checkpoint(checkpoint_path, model, optimizer, meta, adj, hp,
                             epoch, best_top10)
            print(f"[train.main] Checkpoint saved to {checkpoint_path} "
                  f"(balanced top-10: {best_top10:.4f})")
        else:
            patience_counter += 1
            print(f"[train.main] No improvement. Patience counter: "
                  f"{patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"[train.main] Early stopping triggered after {epoch} epochs")
            break

    return best_top10, final_epoch


def _run_overfit(args, dev, hp):
    """Diagnostic: can the pipeline memorise a tiny fixed subset?

    Success looks like train Top-1 -> ~1.0. If it cannot overfit, the bug is in
    data/label alignment or optimisation, not generalization, and no full run
    will help until it is fixed.
    """
    import data_pipeline
    data_pipeline.USE_REAL_DATA = True
    n = args.overfit
    bs = min(args.batch_size, n)
    print(f"[train.main] OVERFIT diagnostic: {n} fixed windows, no augment, no weights")

    dataset, adj, meta, loader = data_pipeline.build(
        data_path=Path(args.data_root) if args.data_root else None,
        batch_size=bs, max_windows=n, balanced_sampling=False,
    )
    from torch.utils.data import DataLoader
    over_loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

    model = make_model(meta, adj, hp).to(dev)
    augment = make_augment()
    crit_p = nn.CrossEntropyLoss(ignore_index=-100)
    crit_m = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr or 1e-3, weight_decay=0.0)

    best_top1, final_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        m = train_one_epoch(model, over_loader, crit_p, crit_m, opt, augment, epoch,
                            accum_steps=1, clip_grad=args.clip_grad, augment_on=False)
        d_p = top10_diagnostics(model, over_loader, head="primary")
        d_m = top10_diagnostics(model, over_loader, head="moses")
        best_top1 = max(best_top1, d_p["balanced_top1"])
        print(f"[overfit] epoch {epoch}: train_p={m['mean_primary']:.4f} "
              f"train_m={m['mean_moses']:.4f} | "
              f"primary top1={d_p['balanced_top1']:.4f} top10={d_p['overall_top10']:.4f} | "
              f"moses top1={d_m['balanced_top1']:.4f} top10={d_m['overall_top10']:.4f}")
        if d_p["balanced_top1"] >= 0.999:
            print("[overfit] SUCCESS: memorised the subset (pipeline is learnable)")
            break
    if best_top1 < 0.9:
        print(f"[overfit] WARNING: only {best_top1:.3f} top-1 after "
              f"{final_epoch} epochs - check data/label alignment or LR")
    return best_top1, final_epoch


def make_model(meta, adj, hp: dict = None) -> Mimir:
    """Build Mimir from PipelineMeta + hyperparameters (falls back to module
    defaults for any key omitted, so kaggle_run's measurement build works)."""
    from mimir_model import Mimir
    hp = hp or {}
    return Mimir(
        n_channels=meta.n_channels,
        n_primary_classes=meta.n_primary_classes,
        n_moses_classes=meta.n_moses_classes,
        adj=adj,
        d_model=hp.get("d_model", D_MODEL),
        tcn_layers=hp.get("tcn_layers", TCN_LAYERS),
        dropout=hp.get("dropout", DROPOUT),
        causal=hp.get("causal", CAUSAL),
        use_whiten=hp.get("use_whiten", USE_WHITEN),
        use_gcn=hp.get("use_gcn", True),
        use_tcn=hp.get("use_tcn", True),
        pool=hp.get("pool", POOL),
    )


def make_augment() -> data_pipeline.TrialAugmentation:
    from data_pipeline import TrialAugmentation
    return TrialAugmentation(
        p_channel=AUGMENT_P_CHANNEL,
        warp_range=AUGMENT_WARP,
        noise_std=AUGMENT_NOISE,
    )


def train_one_epoch(model, loader, criterion_p, criterion_m, optimizer, augment, epoch: int,
                    accum_steps: int = 1, clip_grad: float = CLIP_GRAD,
                    augment_on: bool = True) -> dict:
    """One training epoch with gradient accumulation and both-head losses.

    The two heads use separate criteria (independent vocabularies => independent
    class weights). Gradients are accumulated over ``accum_steps`` micro-batches
    before a clipped step, so the effective batch is ``batch_size*accum_steps``
    at the memory cost of one micro-batch.
    """
    model.train()
    augment.train()
    dev = next(model.parameters()).device
    accum_steps = max(1, int(accum_steps))
    running_p, running_m, n_batches = 0.0, 0.0, 0
    batch_totals: list[tuple[float, float]] = []
    epoch_t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for i, (x, y_p, y_m) in enumerate(loader):
        t_b = time.time()
        x, y_p, y_m = x.to(dev), y_p.to(dev), y_m.to(dev)
        if augment_on:
            x = augment(x)

        logits_p, logits_m = model(x)
        loss_primary = criterion_p(logits_p, y_p)
        loss_moses = criterion_m(logits_m, y_m)
        loss = loss_primary + loss_moses
        (loss / accum_steps).backward()

        at_step = ((i + 1) % accum_steps == 0)
        if at_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=clip_grad)
            assert torch.isfinite(grad_norm) and grad_norm > 0, \
                "no gradient flow"
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.zeros((), device=dev)

        running_p += loss_primary.item()
        running_m += loss_moses.item()
        batch_totals.append((loss_primary.item(), loss_moses.item()))
        n_batches += 1

        if i < MAX_PRINT_BATCHES:
            assert torch.isfinite(loss), "total loss is not finite"
            print(f"  batch {i}: loss_primary={loss_primary.item():.4f}  "
                  f"loss_moses={loss_moses.item():.4f}  total={loss.item():.4f}  "
                  f"grad_norm={float(grad_norm):.4e}  time={time.time() - t_b:.2f}s")
        elif (i + 1) % PROGRESS_EVERY == 0:
            print(f"  epoch {epoch} batch {i + 1}: "
                  f"loss_primary={loss_primary.item():.4f} "
                  f"loss_moses={loss_moses.item():.4f} "
                  f"total={loss.item():.4f}")

        del x, y_p, y_m, logits_p, logits_m, loss_primary, loss_moses, loss
        if (i + 1) % 10 == 0:
            gc.collect()

    # Flush the trailing partial accumulation window.
    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "mean_primary": running_p / max(n_batches, 1),
        "mean_moses": running_m / max(n_batches, 1),
        "n_batches": n_batches,
        "wall_time": time.time() - epoch_t0,
        "sec_per_batch": (time.time() - epoch_t0) / max(n_batches, 1),
        "batch_totals": batch_totals,
    }


def evaluate(model, loader, criterion_p, criterion_m, augment, epoch: int = None,
             augment_on: bool = False) -> dict:
    """Held-out loss for both heads (eval mode; augmentation off by default)."""
    model.eval()
    augment.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        running_p, running_m, n_batches = 0.0, 0.0, 0
        epoch_t0 = time.time()

        for i, (x, y_p, y_m) in enumerate(loader):
            x, y_p, y_m = x.to(dev), y_p.to(dev), y_m.to(dev)
            if augment_on:
                x = augment(x)

            logits_p, logits_m = model(x)
            loss_primary = criterion_p(logits_p, y_p)
            loss_moses = criterion_m(logits_m, y_m)
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


def top10_diagnostics(model, loader, head: str = "primary", epoch: int = None,
                      k: int = 10) -> dict:
    """Balanced Top-k / Top-1 accuracy for ONE head (the Kaggle metric).

    For every held-out sample we check whether the true word index is among the
    model's top-k logits; per-class accuracy = hits(c) / samples(c); the
    "balanced" score is the unweighted mean over classes with >=1 sample.
    Balanced (per-class) is the competition metric; plain accuracy is not.

    ``head`` selects the readout (both vocabularies are scored separately).
    Device-safe: accumulators live on the model's device and are filled with
    ``index_add_`` (indices and target share a device).

    Keys: overall_top10 (+ alias "balanced_top10"), balanced_top1,
          acc_by_class (list[C]), hits_by_class / counts_by_class (CPU tensors),
          classes_seen (int).
    """
    assert head in ("primary", "moses"), head
    head_idx = 0 if head == "primary" else 1
    label_idx = 1 + head_idx   # batch = (x, y_p, y_m)

    model.eval()
    dev = next(model.parameters()).device

    hits = counts = hits1 = None
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(dev)
            y = batch[label_idx].to(dev)
            logits = model(x)[head_idx]                # (B, C)
            C = logits.shape[1]
            if hits is None:
                hits = torch.zeros(C, dtype=torch.float64, device=dev)
                hits1 = torch.zeros(C, dtype=torch.float64, device=dev)
                counts = torch.zeros(C, dtype=torch.float64, device=dev)

            kk = min(k, C)
            topk_idx = torch.topk(logits, k=kk, dim=1).indices    # (B, kk)

            valid = y != -100
            if not valid.any():
                continue
            yv = y[valid].long()                        # (N,)
            tk = topk_idx[valid]                        # (N, kk)
            in_topk = (tk == yv.unsqueeze(1)).any(dim=1).to(torch.float64)  # (N,)
            correct1 = (tk[:, 0] == yv).to(torch.float64)                   # (N,)

            counts.index_add_(0, yv, torch.ones_like(yv, dtype=torch.float64))
            hits.index_add_(0, yv, in_topk)
            hits1.index_add_(0, yv, correct1)

    if hits is None or counts is None or counts.sum().item() == 0:
        overall_top10 = balanced_top1 = 0.0
        acc_by_class: list[float] = []
        classes_seen = 0
    else:
        per_class = hits / counts.clamp(min=1.0)
        per_class1 = hits1 / counts.clamp(min=1.0)
        seen = counts > 0
        classes_seen = int(seen.sum().item())
        overall_top10 = per_class[seen].mean().item() if seen.any() else 0.0
        balanced_top1 = per_class1[seen].mean().item() if seen.any() else 0.0
        acc_by_class = per_class.tolist()

    result = {
        "acc_by_class": acc_by_class,
        "overall_top10": overall_top10,
        "balanced_top10": overall_top10,
        "balanced_top1": balanced_top1,
        "classes_seen": classes_seen,
        "hits_by_class": hits.cpu() if hits is not None else torch.zeros(0),
        "counts_by_class": counts.cpu() if counts is not None else torch.zeros(0),
    }

    if epoch is not None:
        print(f"  epoch {epoch} {head} diagnostics: "
              f"balanced_top10={overall_top10:.4f}  "
              f"balanced_top1={balanced_top1:.4f}  classes_seen={classes_seen}")

    return result

