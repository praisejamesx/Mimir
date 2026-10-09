"""kaggle_run.py - one-shot self-sizing Kaggle run (PNPL 2026 deep track).

Pipeline: clone -> install -> build -> MEASURE real per-window cost ->
auto-pick (windows x epochs) to fit the wall-clock budget -> train ->
inference on the holdout -> write submission with a data-blind guard.

Why auto-size: the deep subject holds far more in-vocab word windows than a
CPU session can chew through. Caching that dataset would need ~30 GB+, and at
~0.1-0.17 s/window (measured on a Kaggle CPU session) a full-data epoch is
multiple hours. So this script measures the live throughput once, then solves
for the largest window count AND epoch count whose combined epoch time fits
budget hours (leaving headroom for install/preload/submit), and prints the
plan before committing.

Usage (from the repo root, USE_REAL_DATA defaults handled here):

    python -u kaggle_run.py [--budget-hours 5.0] [--limit 25000] [--epochs 12]
                            [--patience 4] [--max-epochs 20] [--batch 32]
                            [--measure-windows 8192] [--measure-batches 64]
                            [--submission submission_deep.csv] [--no-submit]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import data_pipeline
import submit
import train as train_mod

# One channel-times-times window in float32 bytes.
WINDOW_BYTES = 306 * 250 * 4
METRIC_FORWARD_FACTOR = 0.4      # metric/val forwards are ~2.5x cheaper than train steps
HOLD_OUT_VAL_FRACTION = 0.10     # train.py's seeded 90/10 split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-sizing Kaggle deep-track run")
    p.add_argument("--budget-hours", type=float, default=5.0,
                   help="total wall-clock budget for the whole run (train+submit)")
    p.add_argument("--limit", type=int, default=None,
                   help="hard cap on training windows (overrides auto-sizing high side)")
    p.add_argument("--epochs", type=int, default=None,
                   help="hard cap on epochs (auto-computed if omitted)")
    p.add_argument("--min-epochs", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--measure-windows", type=int, default=8192)
    p.add_argument("--measure-batches", type=int, default=64)
    p.add_argument("--train-metric-batches", type=int, default=250)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="run device (auto = cuda if available); falls back to "
                        "cpu if a CUDA step fails")
    p.add_argument("--submission", type=str, default="submission_deep.csv")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="local wiring test on the MNE sample path (no pnpl)")
    p.add_argument("--keep", action="store_true",
                   help="do not delete the introspect build's dataset objects")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------
def measure_sec_per_window(args, device: torch.device) -> float:
    """Freeze a (train-augmented, backward) step rate from a capped build."""
    ds, adj, meta, loader = data_pipeline.build(
        batch_size=args.batch, max_windows=args.measure_windows,
    )
    model = train_mod.make_model(meta, adj).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or train_mod.CLOUD_LR,
                           weight_decay=train_mod.WEIGHT_DECAY)
    aug = train_mod.make_augment()
    model.train()
    aug.train()

    it = iter(loader)
    n_done = windows_seen = 0
    t0 = time.time()
    while n_done < args.measure_batches:
        try:
            x, y_p, y_m = next(it)
        except StopIteration:
            it = iter(loader)
            continue
        x, y_p, y_m = x.to(device), y_p.to(device), y_m.to(device)
        x = aug(x)
        logits_p, logits_m = model(x)
        loss = crit(logits_p, y_p) + crit(logits_m, y_m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n_done += 1
        windows_seen += int(x.shape[0])
        del x, y_p, y_m, logits_p, logits_m, loss

    elapsed = time.time() - t0
    if not args.keep:
        del ds, model, opt, loader
        gc.collect()
    sec = elapsed / max(windows_seen, 1)
    print(f"[kaggle_run] measured {windows_seen} windows in {elapsed:.1f}s -> "
          f"{sec * 1000:.1f} ms/window (train step, aug+backward, device {device})")
    return sec


def count_available_windows(args) -> tuple[int, dict]:
    """Cheap pass: full vocab-filtered window count, no data preload (cache
    disabled for this introspect build only, so it returns in seconds)."""
    saved = data_pipeline.LIBRI_CACHE_WINDOWS
    data_pipeline.LIBRI_CACHE_WINDOWS = False
    try:
        ds, adj, meta, loader = data_pipeline.build(batch_size=args.batch)
        n_full = len(ds) + len(ds.val_dataset)
        info = {
            "n_train": len(ds), "n_val": len(ds.val_dataset),
            "adj": adj, "meta": meta,
        }
    finally:
        data_pipeline.LIBRI_CACHE_WINDOWS = saved
    return n_full, info


# ----------------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------------
def solve_plan(args, sec_per_window: float, n_full: int) -> dict:
    """Largest (windows, epochs) that honours every constraint at once."""
    overhead_h = 0.45           # install/clone/build/preload/buffer + submit
    budget_h = max(0.5, args.budget_hours)
    avail_h = max(0.0, budget_h - overhead_h)

    cache_cap = data_pipeline.LIBRI_CACHE_MEMORY_LIMIT // WINDOW_BYTES

    # Windows that fit: budget > cache cap > user limit > full dataset.
    n_budget = int(avail_h * 3600.0 / sec_per_window)
    candidates: list[int] = [n_budget, cache_cap, n_full]
    if args.limit is not None:
        candidates.append(args.limit)
    n_use = min(candidates)
    if n_use < 2000:
        print(f"[kaggle_run] WARNING: available budget/RAM allows only {n_use} "
              f"windows (< 2000); expect a weak model")

    # Epoch cost = train windows + hold-out val + bounded train-metric pass.
    metric_windows = args.train_metric_batches * args.batch
    val_windows = int(n_use * HOLD_OUT_VAL_FRACTION)
    epoch_windows = n_use + val_windows * METRIC_FORWARD_FACTOR + \
        metric_windows * METRIC_FORWARD_FACTOR
    epoch_h = epoch_windows * sec_per_window / 3600.0

    # Budget-available epochs are the binding constraint; the user's --epochs
    # (or --max-epochs) acts only as an upper bound on top of that.
    epochs_avail = int(avail_h / epoch_h) if epoch_h > 0 else args.max_epochs
    epochs_cap = args.epochs or args.max_epochs
    epochs = min(epochs_avail, epochs_cap)
    epochs = max(args.min_epochs, min(epochs, args.max_epochs))

    patience = min(args.patience, max(1, epochs - 1))
    train_h = epochs * epoch_h
    if train_h > avail_h:
        print(f"[kaggle_run] WARNING: plan uses {train_h:.1f}h of the "
              f"{avail_h:.1f}h train budget; raise --budget-hours if you can.")
    return {
        "n_use": n_use,
        "n_full": n_full,
        "epochs": epochs,
        "patience": patience,
        "epoch_h": epoch_h,
        "train_h": train_h,
        "budget_h": budget_h,
        "avail_h": avail_h,
        "cache_n_cap": cache_cap,
        "cache_gb": data_pipeline.LIBRI_CACHE_MEMORY_LIMIT / 2**30,
    }


def run_training(plan: dict, args) -> None:
    sys.argv = [
        "train.py",
        "--epochs", str(plan["epochs"]),
        "--patience", str(plan["patience"]),
        "--batch", str(args.batch),
        "--limit", str(plan["n_use"]),
        "--train-metric-batches", str(args.train_metric_batches),
        "--device", args.device,
    ]
    if args.lr is not None:
        sys.argv += ["--lr", str(args.lr)]
    train_mod.main()


def run_submission(args) -> None:
    ckpt = Path(__file__).resolve().parent / "checkpoints" / "mimir_libri_best.pt"
    sys.argv = [
        "submit.py",
        "--checkpoint", str(ckpt),
        "--track", "deep",
        "--output", args.submission,
        "--device", args.device,
    ]
    try:
        submit.main()
    except RuntimeError as exc:
        print("\n[kaggle_run] REFUSED to write a data-blind submission:", exc)
        print("[kaggle_run] the sanity guard fired; check the epoch metrics above.")
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.smoke:
        data_pipeline.USE_REAL_DATA = False
        print("[kaggle_run] SMOKE mode (MNE sample path, no pnpl)")
    else:
        data_pipeline.USE_REAL_DATA = True
        data_pipeline.LIBRI_CACHE_MEMORY_LIMIT = int(os.environ.get(
            "LIBRI_CACHE_MEMORY_LIMIT_BYTES", 14_000_000_000))

    print("[kaggle_run] deep-track run  "
          f"budget={args.budget_hours}h  cache-limit="
          f"{data_pipeline.LIBRI_CACHE_MEMORY_LIMIT / 2**30:.1f}GiB")

    device = train_mod.resolve_device(args.device)
    print(f"[kaggle_run] device = {device} "
          f"(cuda available={torch.cuda.is_available()})")
    try:
        sec = measure_sec_per_window(args, device)
    except RuntimeError as exc:
        print(f"[kaggle_run] measurement failed on {device}: {exc}")
        if device.type == "cuda":
            print("[kaggle_run] falling back to cpu; the whole run will use cpu")
            device = torch.device("cpu")
            args.device = "cpu"
            sec = measure_sec_per_window(args, device)
        else:
            raise
    n_full, _info = count_available_windows(args)
    print(f"[kaggle_run] full vocab-filtered windows available: {n_full}")

    plan = solve_plan(args, sec, n_full)
    print("\n[kaggle_run] PLAN")
    print(f"  windows : {plan['n_use']} (of {n_full} available) "
          f"[cache cap {plan['cache_n_cap']}, budget cap "
          f"{int(args.budget_hours * 3600 / max(sec, 1e-9))}]")
    print(f"  epochs  : {plan['epochs']}  (patience {plan['patience']}, "
          f"~{plan['epoch_h']:.2f}h/epoch)")
    print(f"  total   : ~{plan['train_h']:.1f}h train + ~0.4h build/submit "
          f"vs {plan['budget_h']:.1f}h budget")
    print("  NOTE: if windows dropped below full, they are capped by "
          "RAM/budget; more budget (--budget-hours) or a faster box changes it.\n")

    run_training(plan, args)

    if not args.no_submit:
        run_submission(args)
        print(f"\n[kaggle_run] DONE - submission at {args.submission}")
        print("[kaggle_run] upload it from the notebook Output tab or the "
              "competition page.")
    else:
        print("\n[kaggle_run] DONE - training only (--no-submit); "
              "run submit.py manually.")


if __name__ == "__main__":
    main()