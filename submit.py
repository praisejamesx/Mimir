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


def build_model_from_checkpoint(ckpt: dict):
    """Reconstruct the exact architecture the checkpoint was trained with."""
    meta = ckpt["meta"]
    hp = ckpt["hyperparams"]
    if hp.get("model", "v1") == "v3":
        from mimir_v3 import MimirV3
        model = MimirV3(
            n_channels=meta["n_channels"],
            n_primary_classes=meta["n_primary_classes"],
            n_moses_classes=meta["n_moses_classes"],
            f1=hp.get("f1", 16),
            f2=hp.get("f2", 128),
            dropout=hp.get("dropout", 0.25),
        )
    else:
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
            pool=hp.get("pool", "mean"),
        )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PNPL 2026 submission builder")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--track", choices=["deep", "broad"], default="deep")
    p.add_argument("--no-sanity", action="store_true",
                   help="skip the bias-only (data-blind) CSV check")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="run device (auto = cuda if available)")
    return p.parse_args()

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, int(os.cpu_count() or 4)))

    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    output = args.output or Path(f"submission_{args.track}.csv")
    device = resolve_device(args.device)
    print(f"[submit] device = {device}")

    # ---- 1. Model ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    retrieval = "t5_primary" in ckpt or "centroids_primary" in ckpt
    if "t5_primary" in ckpt:
        from retrieval import ProjectionHead
        from mimir_v4 import MimirV4
        hp = ckpt["hyperparams"]
        encoder = MimirV4(
            n_channels=ckpt["meta"]["n_channels"],
            n_primary_classes=ckpt["meta"]["n_primary_classes"],
            n_moses_classes=ckpt["meta"]["n_moses_classes"],
            f1=hp.get("f1", 16),
            f2=hp.get("f2", 128),
            dropout=hp.get("dropout", 0.25),
        )
        encoder.load_state_dict(ckpt["model_state"])
        proj = ProjectionHead(encoder.feat_dim, ckpt["t5_primary"].shape[1])
        proj.load_state_dict(ckpt["proj_state"])
        encoder.to(device).eval()
        proj.to(device).eval()
        cent_p = torch.as_tensor(ckpt["t5_primary"]).float().to(device)
        cent_m = torch.as_tensor(ckpt["t5_moses"]).float().to(device)
        submit_tau = float(ckpt.get("submit_tau", 0.2))
        model = None
        print(f"[submit] t5-retrieval checkpoint: epoch {ckpt.get('epoch')} "
              f"val_bal_Top10 {ckpt.get('val_bal_top10', float('nan')):.3f} "
              f"tau={submit_tau}")
    elif retrieval:
        from retrieval import ProjectionHead
        from mimir_v3 import MimirV3
        hp = ckpt["hyperparams"]
        assert hp.get("model", "v3") == "v3", "retrieval ckpt needs v3 encoder"
        encoder = MimirV3(
            n_channels=ckpt["meta"]["n_channels"],
            n_primary_classes=ckpt["meta"]["n_primary_classes"],
            n_moses_classes=ckpt["meta"]["n_moses_classes"],
            f1=hp.get("f1", 16),
            f2=hp.get("f2", 128),
            dropout=hp.get("dropout", 0.25),
        )
        encoder.load_state_dict(ckpt["model_state"])
        proj = ProjectionHead(2 * hp.get("f2", 128))
        proj.load_state_dict(ckpt["proj_state"])
        encoder.to(device).eval()
        proj.to(device).eval()
        cent_p = torch.as_tensor(ckpt["centroids_primary"]).float().to(device)
        cent_m = torch.as_tensor(ckpt["centroids_moses"]).float().to(device)
        submit_tau = float(ckpt.get("submit_tau", 0.2))
        model = None
        print(f"[submit] retrieval checkpoint: epoch {ckpt.get('epoch')} "
              f"val_bal_Top10 {ckpt.get('val_bal_top10', float('nan')):.3f} "
              f"tau={submit_tau}")
    else:
        model = build_model_from_checkpoint(ckpt).to(device)
        encoder = proj = cent_p = cent_m = None
        submit_tau = None
    meta = ckpt["meta"]
    print(f"[submit] checkpoint: epoch {ckpt.get('epoch')} "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f} "
          f"val_bal_Top10 {ckpt.get('val_bal_top10', float('nan')):.3f}")

    # ---- 2. Exact inference transform (must mirror training) -----------------
    mean, std, clip = meta.get("mean"), meta.get("std"), meta.get("clip")
    if mean is None or std is None:
        raise RuntimeError(
            "checkpoint predates train-stat persistence (meta.mean/std are None); "
            "without them the holdout would be scored untransformed (bias-only). "
            "Retrain with the current pipeline before submitting."
        )
    mean = torch.as_tensor(np.asarray(mean, dtype=np.float32))   # (C,)
    std = torch.as_tensor(np.asarray(std, dtype=np.float32))     # (C,)
    clip = float(clip) if clip is not None else None
    print(f"[submit] inference transform: (x - mean) / std "
          f"{'then clip +/-%.1f' % clip if clip else ''} "
          f"({len(mean)} channels)")
    mean = mean.to(device)                                        # (C,)
    std = std.to(device)                                          # (C,)

    # ---- 3. Holdout ----------------------------------------------------------
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary
    from pnpl.competition import write_submission

    # pnpl serves RAW SI units (no standardize kwarg exists on this class).
    holdout = LibriBrainCompetitionHoldout(track=args.track)
    counts = holdout.counts()
    n = len(holdout)
    print(f"[submit] holdout: {counts['total']} examples, track={args.track}")

    # ---- 4. Predict ----------------------------------------------------------
    # Legacy: softmax over both classifier heads. Retrieval: cosine to
    # per-word centroids + softmax (diverse by construction: different trials
    # land near different centroids).
    primary_blocks: list[np.ndarray] = []
    moses_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for batch, _ in holdout.iter_windows(batch_size=SUBMIT_BATCH_SIZE):
            x = torch.from_numpy(batch).float().to(device)   # (B, 306, 250) SI
            x = (x - mean[None, :, None]) / std[None, :, None]
            if clip is not None:
                x = x.clamp(-clip, clip)                 # pnpl training clip
            if retrieval:
                z = proj(encoder.encode(x))              # (B, D) L2-normed
                primary_blocks.append(
                    torch.softmax(z @ cent_p.T / submit_tau, dim=1).cpu().numpy())
                moses_blocks.append(
                    torch.softmax(z @ cent_m.T / submit_tau, dim=1).cpu().numpy())
            else:
                logits_p, logits_m = model(x)            # (B, 50), (B, 50)
                primary_blocks.append(torch.softmax(logits_p, dim=1).cpu().numpy())
                moses_blocks.append(torch.softmax(logits_m, dim=1).cpu().numpy())

    primary_probs = np.concatenate(primary_blocks, axis=0)
    moses_probs = np.concatenate(moses_blocks, axis=0)

    # ---- 5. Sanity: refuse a data-blind (bias-only) CSV ----------------------
    argmaxes = primary_probs.argmax(axis=1)
    col_std = primary_probs.std(axis=0).mean()
    mean_max = float(primary_probs.max(axis=1).mean())
    n_unique_argmax = int(len(np.unique(argmaxes)))
    print(f"[submit] diversity: {n_unique_argmax}/{primary_probs.shape[1]} "
          f"distinct argmax rows, mean-max-p={mean_max:.4f}, "
          f"mean column-std={col_std:.2e}")
    if not args.no_sanity:
        if n_unique_argmax < 2 or col_std < 1e-3:
            raise RuntimeError(
                "predictions look bias-only: every row collapses to the same "
                f"top-10 (unique argmax={n_unique_argmax}, "
                f"column-std={col_std:.2e}). Refuse to write it; pass "
                "--no-sanity to force."
            )

    # ---- 6. Write ------------------------------------------------------------
    written = write_submission(
        path=output,
        indices=holdout.indices,
        primary_probs=primary_probs,
        secondary_probs=moses_probs,
    )
    print(f"[submit] WROTE {written} rows to {output}")


if __name__ == "__main__":
    main()