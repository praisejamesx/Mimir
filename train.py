"""train.py - Training loop for MIMIR (local smoke test / cloud run).

Two modes, chosen by data_pipeline.USE_REAL_DATA:

  MNE sample (default): one epoch over ~120 training trials, verifying the
  full contract end-to-end (tensor shapes, dual heads, augmentation gating,
  gradient flow, RAM) and reporting a clean validation loss afterwards. The
  validation loss comes from the held-out split whose z-score statistics were
  fit on the training trials only - no leakage.

  LibriBrain100: the competition training set (Deep Track). Up to CLOUD_EPOCHS
  epochs with a seeded 90/10 validation split, early stopping (patience 3) and
  best-checkpoint saving for submit.py.

The architecture blocks are toggled by the USE_GCN / USE_TCN flags defined in
mimir_model.py and re-read at construction time, so an ablation run is a
single-flag edit in mimir_model.py.
"""

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

import data_pipeline
from mimir_model import Mimir, USE_GCN, USE_TCN

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
CLOUD_TRAIN_METRIC_BATCHES = 300   # plain Top-10 recall on this many train batches

# Resolved at main() time from --device (auto|cpu|cuda); everything (model,
# batches, labels, augmentation) moves here. The code stays device-agnostic:
# the same commands run on CPU or GPU.
DEVICE = torch.device("cpu")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_model(meta, adj) -> Mimir:
    return Mimir(
        n_channels=meta.n_channels,                  # 305 (sample) / 306 (PNPL)
        n_primary_classes=meta.n_primary_classes,    # 2 local / 50 in PNPL
        n_moses_classes=meta.n_moses_classes,        # 2 local / 50 Moses-50
        adj=adj,                                     # (C, C) sensor graph
        d_model=D_MODEL,
        tcn_layers=TCN_LAYERS,
        dropout=DROPOUT,
        causal=CAUSAL,
        use_whiten=USE_WHITEN,
        use_gcn=USE_GCN,
        use_tcn=USE_TCN,
    )


def make_augment() -> data_pipeline.TrialAugmentation:
    return data_pipeline.TrialAugmentation(
        p_channel=AUGMENT_P_CHANNEL,
        warp_range=AUGMENT_WARP,
        noise_std=AUGMENT_NOISE,
    )


def train_one_epoch(model, loader, criterion, optimizer, augment, epoch: int) -> dict:
    """One full training pass; returns diagnostics for the epoch summary."""
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

        # Competition objective: two independent vocabularies, summed CE.
        # ignore_index=-100: a window whose word is absent from one vocab
        # simply trains the other head (both may be masked, defensively).
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

    return {
        "mean_primary": running_p / max(n_batches, 1),
        "mean_moses": running_m / max(n_batches, 1),
        "n_batches": n_batches,
        "wall_time": time.time() - epoch_t0,
        "sec_per_batch": (time.time() - epoch_t0) / max(n_batches, 1),
        "batch_totals": batch_totals,
    }


def evaluate(model, loader, criterion, augment) -> float:
    """Mean validation loss (primary + moses) over a holdout split.

    augment.eval() is an exact identity, so the number is provably clean -
    augmentation cannot leak into early stopping / reporting.
    """
    model.eval()
    augment.eval()
    dev = next(model.parameters()).device
    running, n = 0.0, 0
    with torch.no_grad():
        for x, y_p, y_m in loader:
            x, y_p, y_m = x.to(dev), y_p.to(dev), y_m.to(dev)
            logits_p, logits_m = model(x)
            running += (criterion(logits_p, y_p) + criterion(logits_m, y_m)).item()
            n += 1
    model.train()
    augment.train()
    return running / max(n, 1)


@torch.no_grad()
def top10_diagnostics(model, loader, num_classes: int, max_batches=None) -> dict:
    """Primary-head Top-10 recall over a loader.

    Returns both the plain top-10 recall and the balanced version (per-class
    recall averaged over the classes present). The competition metric is the
    balanced one; chance is ~1.0/50 for balanced and ~10/50 for plain recall
    when a model emits the same top-10 list for every row. Vectorized per
    batch, so it is cheap even on the full validation partition.
    """
    model.eval()
    # Derive the device from the model itself (not the module global) so the
    # metric is correct wherever the model lives - CPU, CUDA, whatever.
    dev = next(model.parameters()).device
    hits_total = n_total = 0
    hits_by_class = torch.zeros(num_classes, dtype=torch.int64, device=dev)
    counts_by_class = torch.zeros(num_classes, dtype=torch.int64, device=dev)
    n_batches = 0
    for x, y_p, y_m in loader:
        x, y_p = x.to(dev), y_p.to(dev)
        logits_p, _ = model(x)
        top = torch.topk(logits_p, 10, dim=-1).indices          # (B, 10)
        yp = y_p
        valid = yp != -100
        if (n_valid := int(valid.sum().item())) > 0:
            hit_any = (top == yp.unsqueeze(-1)).any(-1)         # (B,)
            yv = yp[valid]
            hv = hit_any[valid]
            hits_by_class.index_add_(0, yv.long(), hv.long())
            counts_by_class.index_add_(0, yv.long(), torch.ones_like(yv.long()))
            hits_total += int(hv.sum().item())
            n_total += n_valid
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()

    seen = counts_by_class > 0
    per_class = (hits_by_class[seen].float() / counts_by_class[seen].clamp(min=1))
    balanced = float(per_class.mean().item()) if int(seen.sum()) > 0 else 0.0
    plain = hits_total / max(n_total, 1)
    return {
        "plain": plain,
        "balanced": balanced,
        "n_samples": n_total,
        "n_classes_seen": int(seen.sum().item()),
    }


# ----------------------------------------------------------------------------
# Cloud: multi-epoch training with early stopping + best-checkpoint saving
# ----------------------------------------------------------------------------
def _cloud_value(cli_value, env_name: str, module_default):
    """CLI flag > $ENV > module constant (Kaggle sets CLOUD_EPOCHS/PATIENCE)."""
    if cli_value is not None:
        return cli_value
    if env_name in os.environ:
        return type(module_default)(os.environ[env_name])
    return module_default


def run_cloud_training(opts=None) -> None:
    epochs = _cloud_value(getattr(opts, "epochs", None), "CLOUD_EPOCHS", CLOUD_EPOCHS)
    patience = _cloud_value(getattr(opts, "patience", None), "CLOUD_PATIENCE", CLOUD_PATIENCE)
    max_windows = getattr(opts, "limit", None)
    lr = _cloud_value(getattr(opts, "lr", None), "CLOUD_LR", CLOUD_LR)
    batch = _cloud_value(getattr(opts, "batch", None), "CLOUD_BATCH_SIZE",
                         CLOUD_BATCH_SIZE)
    train_metric_batches = _cloud_value(
        getattr(opts, "train_metric_batches", None), "CLOUD_TRAIN_METRIC_BATCHES",
        CLOUD_TRAIN_METRIC_BATCHES)

    out_root = Path(__file__).resolve().parent / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "mimir_libri_best.pt"

    dataset, adj, meta, loader = data_pipeline.build(
        batch_size=batch, max_windows=max_windows,
    )
    val_loader = dataset.val_loader

    model = make_model(meta, adj).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # Cosine annealing over the full epoch budget: warm at CLOUD_LR, decay to
    # near zero at the last epoch (better than a constant LR when epoch count
    # is small and the curve is dominated by a Zipfian prior fit).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20.0)
    augment = make_augment()

    best_val, best_bal, best_epoch, no_improve = float("inf"), 0.0, 0, 0
    print(f"[train] cloud run: epochs={epochs} patience={patience} "
          f"batch={batch} train={len(dataset)} "
          f"val={len(val_loader.dataset)} lr={lr} max_windows={max_windows}")

    for epoch in range(1, epochs + 1):
        diag = train_one_epoch(model, loader, criterion, optimizer, augment, epoch)
        val_loss = evaluate(model, val_loader, criterion, augment)
        # Compute metrics defensively; any metric failure should not kill the run
        # (it would waste an epoch before saving a checkpoint).
        try:
            vd = top10_diagnostics(model, val_loader, meta.n_primary_classes)
            td = top10_diagnostics(
                model, loader, meta.n_primary_classes,
                max_batches=train_metric_batches)
        except Exception as exc:
            print(f"  [warning] metric computation failed ({exc}); metrics set to 0.0")
            vd = {"plain": float("nan"), "balanced": 0.0, "n_samples": 0,
                  "n_classes_seen": 0}
            td = {"plain": float("nan"), "balanced": 0.0, "n_samples": 0,
                  "n_classes_seen": 0}
        cur_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        first5 = "/".join(
            f"{p:.3f}/{m:.3f}" for p, m in diag["batch_totals"][:MAX_PRINT_BATCHES]
        )
        print(f"epoch {epoch}: mean_primary={diag['mean_primary']:.4f} "
              f"mean_moses={diag['mean_moses']:.4f} val_loss={val_loss:.4f} "
              f"train_Top10={td['plain']:.3f} val_Top10={vd['plain']:.3f} "
              f"val_bal_Top10={vd['balanced']:.3f} lr={cur_lr:.2e} "
              f"(first-5 batches: {first5}) time={diag['wall_time']:.0f}s "
              f"RSS={peak_rss_mb():.0f} MB")

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_bal = vd["balanced"]
            no_improve = 0
            torch.save(
                {
                    # Always persist CPU tensors so the checkpoint loads on any
                    # machine, regardless of the device it was trained on.
                    "model_state": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "adj": adj,
                    "meta": asdict(meta),
                    "hyperparams": {
                        "d_model": D_MODEL,
                        "tcn_layers": TCN_LAYERS,
                        "dropout": DROPOUT,
                        "causal": CAUSAL,
                        "use_whiten": USE_WHITEN,
                        "use_gcn": USE_GCN,
                        "use_tcn": USE_TCN,
                        "lr": lr,
                        "batch": batch,
                        "epochs": epochs,
                        "weight_decay": WEIGHT_DECAY,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_bal_top10": best_bal,
                    "class_to_id": meta.class_to_id,
                    "class_names": meta.class_names,
                },
                ckpt_path,
            )
            print(f"  * saved best checkpoint to {ckpt_path} (val_loss={val_loss:.4f}, "
                  f"val_bal_Top10={best_bal:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping after {patience} epochs without "
                      f"improvement (best val_loss={best_val:.4f} @ epoch {best_epoch})")
                break

        del diag
        if epoch % 2 == 0:
            gc.collect()

    print("\n=== CLOUD SUMMARY ===")
    print(f"best val_loss = {best_val:.4f} @ epoch {best_epoch} of {epoch} run, "
          f"val_bal_Top10 = {best_bal:.3f} (chance = 1/50 = 0.020 per class)")
    print(f"checkpoint: {ckpt_path} (feed to submit.py)")
    print(f"peak RSS = {peak_rss_mb():.0f} MB "
          f"({'OK' if peak_rss_mb() < 20000 else 'TOO HIGH'})")


# ----------------------------------------------------------------------------
# Local: one-epoch smoke test
# ----------------------------------------------------------------------------
def run_smoke_training() -> None:
    t0 = time.time()

    # ---- 1. Data -------------------------------------------------------------
    dataset, adj, meta, loader = data_pipeline.build(batch_size=BATCH_SIZE)
    xb, yp_b, ym_b = next(iter(loader))
    print(f"[train] batch: x={tuple(xb.shape)} ({xb.dtype})  "
          f"y_primary={tuple(yp_b.shape)} ({yp_b.dtype})  "
          f"y_moses={tuple(ym_b.shape)} ({ym_b.dtype})  "
          f"heads=({meta.n_primary_classes}, {meta.n_moses_classes})-way")

    # ---- 2. Augmentation: prove train-modifies / eval-identity --------------
    augment = make_augment()
    augment.train()
    x_aug = augment(xb)
    changed = not torch.equal(x_aug, xb)
    dropped = float((x_aug == 0).all(dim=-1).float().mean())
    augment.eval()
    identity = torch.equal(augment(xb), xb)
    if AUGMENT:
        augment.train()
    else:
        augment.eval()
    print(f"[train] TrialAugmentation: train-mode changed={changed} "
          f"(dropped-channel fraction={dropped:.3f}, target ~"
          f"{AUGMENT_P_CHANNEL:.2f}), eval-mode identity={identity}")
    assert changed and identity, "augmentation train/eval gating broken"
    del xb, yp_b, ym_b, x_aug
    gc.collect()

    # ---- 3. Model -----------------------------------------------------------
    model = make_model(meta, adj).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"[train] RSS after setup = {peak_rss_mb():.0f} MB")

    # ---- 4. One epoch (dual-head) -------------------------------------------
    diag = train_one_epoch(model, loader, criterion, optimizer, augment, epoch=1)

    # ---- 5. Validation on the held-out split (z-scored on train stats only) --
    val_loader = dataset.val_loader
    val_loss = evaluate(model, val_loader, criterion, augment)

    # ---- 6. Verdict ---------------------------------------------------------
    mean_p, mean_m = diag["mean_primary"], diag["mean_moses"]
    mean_total = mean_p + mean_m
    chance_total = float(np.log(meta.n_primary_classes) + np.log(meta.n_moses_classes))
    n_head = min(MAX_PRINT_BATCHES, len(diag["batch_totals"]))
    bt = diag["batch_totals"]
    first5_p = float(np.mean([t[0] for t in bt[:n_head]]))
    first5_m = float(np.mean([t[1] for t in bt[:n_head]]))
    last5_p = float(np.mean([t[0] for t in bt[-n_head:]]))
    last5_m = float(np.mean([t[1] for t in bt[-n_head:]]))

    print("\n=== EPOCH SUMMARY (dual head) ===")
    print(f"batches={diag['n_batches']}  train trials={len(dataset)}  "
          f"val trials={len(val_loader.dataset)}")
    print(f"mean loss_primary = {mean_p:.4f}  "
          f"(chance ln K_p = {np.log(meta.n_primary_classes):.4f})")
    print(f"mean loss_moses   = {mean_m:.4f}  "
          f"(chance ln K_m = {np.log(meta.n_moses_classes):.4f})")
    print(f"mean total        = {mean_total:.4f}  (chance total = {chance_total:.4f})")
    print(f"first {n_head} -> last {n_head} batches: "
          f"primary {first5_p:.4f} -> {last5_p:.4f}, "
          f"moses {first5_m:.4f} -> {last5_m:.4f}")
    print(f"clean validation loss = {val_loss:.4f} (augmentation off, "
          f"z-score stats from train)")

    droppool = last5_p < first5_p and last5_m < first5_m
    tag = "SMOKE PASSED (total loss below chance)" if mean_total < chance_total else (
        "pipeline OK (losses decreasing; single augmented epoch)" if droppool
        else "inspect the per-batch losses above")
    print(f"verdict: {tag}")
    print(f"epoch wall time = {diag['wall_time']:.1f}s  "
          f"({diag['sec_per_batch']:.2f}s/batch on CPU)")
    print(f"peak RSS = {peak_rss_mb():.0f} MB (8 GB budget -> "
          f"{'OK' if peak_rss_mb() < 6000 else 'TOO HIGH'})")
    print(f"total wall time = {time.time() - t0:.1f}s")


def parse_args() -> argparse.Namespace:
    """Cloud-run overrides (all optional; env vars/constants are the default).

    The Kaggle notebook already exports CLOUD_EPOCHS/CLOUD_PATIENCE, so those
    still win unless a CLI flag is passed.
    """
    p = argparse.ArgumentParser(description="MIMIR trainer (PNPL 2026 deep track)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the vocab-filtered window count (probe runs)")
    p.add_argument("--train-metric-batches", type=int, default=None,
                   help="train-side Top-10 batches per epoch (default 800)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="run device (auto = cuda if available)")
    return p.parse_args()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, int(os.cpu_count() or 4)))

    opts = parse_args()
    global DEVICE
    DEVICE = resolve_device(opts.device)
    print(f"[train] device = {DEVICE} "
          f"(cuda available={torch.cuda.is_available()})")
    if data_pipeline.USE_REAL_DATA:
        print("[train] REAL-DATA mode (LibriBrain100) - cloud training "
              f"(gcn={USE_GCN}, tcn={USE_TCN})")
        run_cloud_training(opts)
    else:
        print(f"[train] MNE-SAMPLE mode - smoke test (gcn={USE_GCN}, tcn={USE_TCN})")
        run_smoke_training()


if __name__ == "__main__":
    main()