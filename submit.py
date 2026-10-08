"""submit.py - Build the PNPL 2026 submission CSV from a trained checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from mimir_model import Mimir

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "mimir_libri_best.pt"
SUBMIT_BATCH_SIZE = 64
SEED = 0


def build_model_from_checkpoint(ckpt: dict) -> Mimir:
    """Reconstruct the exact architecture the checkpoint was trained with."""
    meta = ckpt["meta"]
    hp = ckpt["hyperparams"]
    model = Mimir(
        n_channels=meta["n_channels"],
        n_primary_classes=meta["n_primary_classes"],
        n_moses_classes=meta["n_moses_classes"],
        adj=ckpt["adj"],
        d_model=hp.get("d_model", 64),
        tcn_layers=hp.get("tcn_layers", 7),
        dropout=hp.get("dropout", 0.1),
        causal=hp.get("causal", False),
        use_whiten=hp.get("use_whiten", True),
        use_gcn=hp.get("use_gcn", True),
        use_tcn=hp.get("use_tcn", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PNPL 2026 submission builder")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--track", choices=["deep", "broad"], default="deep")
    return p.parse_args()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, int(os.cpu_count() or 4)))

    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    output = args.output or Path(f"submission_{args.track}.csv")

    # ---- 1. Model ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    print(f"[submit] checkpoint: epoch {ckpt.get('epoch')} val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    # ---- 2. Holdout ----------------------------------------------------------
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary
    from pnpl.competition import write_submission

    # CRITICAL FIX 1: Tell pnpl to standardize using the GLOBAL training stats
    holdout = LibriBrainCompetitionHoldout(track=args.track, standardize=True)
    counts = holdout.counts()
    n = len(holdout)
    print(f"[submit] holdout: {counts['total']} examples, track={args.track}")

    # ---- 3. Predict (softmax over both heads) --------------------------------
    primary_blocks: list[np.ndarray] = []
    moses_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch, _ in holdout.iter_windows(batch_size=SUBMIT_BATCH_SIZE):
            x = torch.from_numpy(batch).float()          # (B, 306, 250)
            
            # CRITICAL FIX 2: DO NOT apply per-window z-scoring. 
            # pnpl already standardized this using the global training stats.
            # Per-window z-scoring destroys the global variance structure the model expects.
                                
            logits_p, logits_m = model(x)                # (B, 50), (B, 50)
            primary_blocks.append(torch.softmax(logits_p, dim=1).numpy())
            moses_blocks.append(torch.softmax(logits_m, dim=1).numpy())

    primary_probs = np.concatenate(primary_blocks, axis=0)
    moses_probs = np.concatenate(moses_blocks, axis=0)
    
    # ---- 4. Write ------------------------------------------------------------
    written = write_submission(
        path=output,
        indices=holdout.indices,
        primary_probs=primary_probs,
        secondary_probs=moses_probs,
    )
    print(f"[submit] WROTE {written} rows to {output}")

if __name__ == "__main__":
    main()