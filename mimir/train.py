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
    running_p, running_m, n_batches = 0.0, 0.0, 0
    batch_totals: list[tuple[float, float]] = []
    epoch_t0 = time.time()

    for i, (x, y_p, y_m) in enumerate(loader):
        t_b = time.time()
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
    running, n = 0.0, 0
    with torch.no_grad():
        for x, y_p, y_m in loader:
            logits_p, logits_m = model(x)
            running += (criterion(logits_p, y_p) + criterion(logits_m, y_m)).item()
            n += 1
    model.train()
    augment.train()
    return running / max(n, 1)


# ----------------------------------------------------------------------------
# Cloud: multi-epoch training with early stopping + best-checkpoint saving
# ----------------------------------------------------------------------------
def run_cloud_training() -> None:
    out_root = Path(__file__).resolve().parent / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "mimir_libri_best.pt"

    dataset, adj, meta, loader = data_pipeline.build(batch_size=CLOUD_BATCH_SIZE)
    val_loader = dataset.val_loader

    model = make_model(meta, adj)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLOUD_LR, weight_decay=WEIGHT_DECAY)
    augment = make_augment()

    best_val, best_epoch, no_improve = float("inf"), 0, 0
    print(f"[train] cloud run: epochs={CLOUD_EPOCHS} patience={CLOUD_PATIENCE} "
          f"batch={CLOUD_BATCH_SIZE} train={len(dataset)} "
          f"val={len(val_loader.dataset)}")

    for epoch in range(1, CLOUD_EPOCHS + 1):
        diag = train_one_epoch(model, loader, criterion, optimizer, augment, epoch)
        val_loss = evaluate(model, val_loader, criterion, augment)

        first5 = "/".join(
            f"{p:.3f}/{m:.3f}" for p, m in diag["batch_totals"][:MAX_PRINT_BATCHES]
        )
        print(f"epoch {epoch}: mean_primary={diag['mean_primary']:.4f} "
              f"mean_moses={diag['mean_moses']:.4f} val_loss={val_loss:.4f} "
              f"(first-5 batches: {first5}) time={diag['wall_time']:.0f}s "
              f"RSS={peak_rss_mb():.0f} MB")

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
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
                        "lr": CLOUD_LR,
                        "weight_decay": WEIGHT_DECAY,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "class_to_id": meta.class_to_id,
                    "class_names": meta.class_names,
                },
                ckpt_path,
            )
            print(f"  * saved best checkpoint to {ckpt_path} (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= CLOUD_PATIENCE:
                print(f"  early stopping after {CLOUD_PATIENCE} epochs without "
                      f"improvement (best val_loss={best_val:.4f} @ epoch {best_epoch})")
                break

        del diag
        if epoch % 2 == 0:
            gc.collect()

    print("\n=== CLOUD SUMMARY ===")
    print(f"best val_loss = {best_val:.4f} @ epoch {best_epoch} of {epoch} run")
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
    model = make_model(meta, adj)
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


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, int(os.cpu_count() or 4)))

    if data_pipeline.USE_REAL_DATA:
        print("[train] REAL-DATA mode (LibriBrain100) - cloud training "
              f"(gcn={USE_GCN}, tcn={USE_TCN})")
        run_cloud_training()
    else:
        print(f"[train] MNE-SAMPLE mode - smoke test (gcn={USE_GCN}, tcn={USE_TCN})")
        run_smoke_training()


if __name__ == "__main__":
    main()