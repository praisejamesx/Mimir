"""submit.py - Build the PNPL 2026 submission CSV from a trained checkpoint.

Pipeline
--------
1. Load the best checkpoint from train.py cloud mode (weights, meta,
   adjacency, hyperparameters - including the use_gcn / use_tcn flags).
2. Rebuild the exact Mimir architecture used for training.
3. Run the competition holdout (LibriBrainCompetitionHoldout, deep track)
   in canonical order; for every (B, 306, 250) window, take softmax over BOTH
   heads: (B, 50) primary probabilities and (B, 50) Moses-50 probabilities.
4. Write the submission via pnpl.write_submission with the canonical columns
   (index + 50 primary-vocab probs + moses_<word> columns).

Head class order == vocabulary order by construction: both derive from
load_vocabulary (primary / secondary), which is also the column order the
scoring reads, so column k corresponds to head logits[k].

The primary distribution is what the Top-10 balanced accuracy is computed on;
the Moses columns are scored in the secondary analysis block. They are filled
with the Moses head's softmax - a real distribution over the secondary
vocabulary, not a uniform guess - matching the dual-head training objective.
"""

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
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Best checkpoint from train.py cloud mode")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default: submission_<track>.csv)")
    p.add_argument("--track", choices=["deep", "broad"], default="deep",
                   help="Competition track (deep = subject 0)")
    return p.parse_args()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, int(os.cpu_count() or 4)))

    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}. Run train.py in cloud "
            "mode first."
        )
    output = args.output or Path(f"submission_{args.track}.csv")

    # ---- 1. Model ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    print(f"[submit] checkpoint: epoch {ckpt.get('epoch')} "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    # ---- 2. Holdout ----------------------------------------------------------
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary
    from pnpl.competition import write_submission

    holdout = LibriBrainCompetitionHoldout(track=args.track)
    counts = holdout.counts()
    n = len(holdout)
    print(f"[submit] holdout: {counts['total']} examples "
          f"(sentence={counts['sentence']}, word={counts['word']}), "
          f"track={args.track}")

    # ---- 3. Predict (softmax over both heads) --------------------------------
    primary_blocks: list[np.ndarray] = []
    moses_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch, _ in holdout.iter_windows(batch_size=SUBMIT_BATCH_SIZE):
            x = torch.from_numpy(batch).float()          # (B, 306, 250)

            # Apply per-channel z-scoring to match training preprocessing
            # Raw MEG data causes activations to vanish; the model expects whitened data.
            x_mean = x.mean(dim=2, keepdim=True)
            x_std = x.std(dim=2, keepdim=True)
            x = (x - x_mean) / (x_std + 1e-6)
                                
            logits_p, logits_m = model(x)                # (B, 50), (B, 50)
            primary_blocks.append(torch.softmax(logits_p, dim=1).numpy())
            moses_blocks.append(torch.softmax(logits_m, dim=1).numpy())

    primary_probs = np.concatenate(primary_blocks, axis=0)   # (N, 50)
    moses_probs = np.concatenate(moses_blocks, axis=0)       # (N, 50)
    assert primary_probs.shape == (n, 50) and moses_probs.shape == (n, 50)
    assert np.isfinite(primary_probs).all() and np.isfinite(moses_probs).all()

    # ---- 4. Write ------------------------------------------------------------
    written = write_submission(
        path=output,
        indices=holdout.indices,
        primary_probs=primary_probs,
        secondary_probs=moses_probs,
    )
    print(f"[submit] WROTE {written} rows to {output} "
          f"(index + 50 primary + 50 moses_ columns)")

    # Diagnostic (not scored): how often the two heads' argmax words agree.
    primary_vocab = load_vocabulary("primary")
    moses_vocab = load_vocabulary("secondary")
    agree = float(
        (
            np.array([primary_vocab[i] for i in primary_probs.argmax(1)])
            == np.array([moses_vocab[i] for i in moses_probs.argmax(1)])
        ).mean()
    )
    print(f"[submit] argmax-word agreement between heads: {agree:.3f}")


if __name__ == "__main__":
    main()