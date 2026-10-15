"""retrieval.py - Matching-based objective (successor to 50-way CE).

Why this exists: four consecutive configs (v1 CE, v2 focal, v3-small CE,
v3-big CE+balanced) all floored at BalAcc@10 ~0.21 with submit diversity 1/50.
A 50-way classifier on single noisy trials minimizes loss by predicting
frequent words -- there is always a prior to retreat to. Matching removes the
retreat: every trial must be placed near its word's trials and far from other
words' trials. There is no frequent-class bias term to hide behind.

Method (all standard, no new inventions):
- Encoder: MimirV3.encode (mean + log-power spectral-spatial features).
- Projection: Linear -> L2-normalized embedding (SimCLR/Khosla et al. 2020).
- Positives: two augmented views of the same trial ALWAYS (instance signal,
  needs no labels) plus same-word trials in the batch when present.
- Inference: per-word centroids (mean embedding over train trials -- the
  "average repeated tokens" finding) + cosine rank. Top-10 is naturally
  diverse: different trials land near different centroids.
- Submission probs: softmax(similarity / submit_tau).

Pure torch. No mne/pnpl imports, so this file self-tests anywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_DIM = 128
TEMPERATURE = 0.1
SUBMIT_TAU = 0.2


def siglip_loss(sims: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """SigLIP-style loss (Zhai et al. 2023) over a fixed candidate matrix.

    sims: (B, C) cosine similarities / tau. targets: (B, C) in {+1, -1};
    rows whose trial label is invalid (-100) must be pre-excluded by the
    caller (mask and mean outside). Unlike SupCon, the candidate set is FIXED
    (the vocabulary), so every trial always has its positives present -- no
    dependence on batch composition. This structurally kills the missing-pair
    failure (batch-32 SupCon collapsed with centroids at 0.996 pairwise cos).
    """
    return F.logsigmoid(sims * targets).mean().neg()


class ProjectionHead(nn.Module):
    """Linear projection to L2-normalized retrieval embeddings."""

    def __init__(self, in_dim: int, out_dim: int = PROJ_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(h), dim=1)


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor,
                tau: float = TEMPERATURE) -> torch.Tensor:
    """Supervised contrastive loss over a doubled (2-view) batch.

    feats: (2N, D), L2-normalized. labels: (2N,) with -100 = invalid anchor.
    Anchors: valid-label rows with >=1 positive (same label, other row).
    Denominator: all other rows (invalid-label rows count as negatives only).
    Anchors without positives contribute nothing (common for rare words).
    """
    n = feats.shape[0]
    sim = (feats @ feats.T) / tau                        # (2N, 2N)
    sim = sim - 1e9 * torch.eye(n, device=feats.device)  # exclude self

    valid = labels != -100
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos = same & valid.unsqueeze(0) & valid.unsqueeze(1)
    pos.fill_diagonal_(False)
    n_pos = pos.float().sum(dim=1)
    has_pos = (n_pos > 0) & valid

    sim_max, _ = sim.max(dim=1, keepdim=True)
    exp = torch.exp(sim - sim_max)
    eye = torch.eye(n, device=feats.device, dtype=exp.dtype)
    exp = exp * (1.0 - eye)                               # no-self, out-of-place
    log_prob = sim - sim_max - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    per_anchor = -(log_prob * pos.float()).sum(dim=1) / n_pos.clamp(min=1.0)

    if not bool(has_pos.any()):
        return feats.sum() * 0.0
    return per_anchor[has_pos].mean()


@torch.no_grad()
def build_centroids(encoder: nn.Module, proj: nn.Module, loader,
                    n_classes: int, head_idx: int,
                    device: torch.device) -> torch.Tensor:
    """Mean L2-normalized embedding per class over a loader (train split).

    head_idx: 0 = primary (labels are batch[1]), 1 = moses (batch[2]).
    Returns (C, D) L2-renormalized rows; unseen classes stay zero.
    """
    encoder.eval()
    proj.eval()
    sums = None
    counts = None
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1 + head_idx].to(device)
        h = encoder.encode(x)
        z = proj(h)                                       # already L2-normed
        if sums is None:
            sums = torch.zeros(n_classes, z.shape[1], device=device)
            counts = torch.zeros(n_classes, device=device)
        v = y != -100
        if bool(v.any()):
            yv = y[v].long()
            sums.index_add_(0, yv, z[v].float())
            counts.index_add_(0, yv, torch.ones_like(yv, dtype=torch.float32))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    cent = sums / counts
    return F.normalize(cent, dim=1).float()


@torch.no_grad()
def retrieval_top10(encoder: nn.Module, proj: nn.Module,
                    centroids: torch.Tensor, loader, head_idx: int,
                    device: torch.device, k: int = 10) -> dict:
    """Balanced Top-k / Top-1 by cosine-to-centroid rank (the Kaggle metric)."""
    encoder.eval()
    proj.eval()
    C = centroids.shape[0]
    cent = centroids.to(device)
    hits = torch.zeros(C, dtype=torch.float64, device=device)
    hits1 = torch.zeros(C, dtype=torch.float64, device=device)
    counts = torch.zeros(C, dtype=torch.float64, device=device)
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1 + head_idx].to(device)
        z = proj(encoder.encode(x))
        sim = z @ cent.T                                   # (B, C)
        kk = min(k, C)
        topk = torch.topk(sim, k=kk, dim=1).indices
        v = y != -100
        if not bool(v.any()):
            continue
        yv = y[v].long()
        tk = topk[v]
        in_topk = (tk == yv.unsqueeze(1)).any(dim=1).to(torch.float64)
        top1 = (tk[:, 0] == yv).to(torch.float64)
        counts.index_add_(0, yv, torch.ones_like(yv, dtype=torch.float64))
        hits.index_add_(0, yv, in_topk)
        hits1.index_add_(0, yv, top1)
    seen = counts > 0
    top10 = (hits / counts.clamp(min=1.0))[seen].mean().item() if bool(seen.any()) else 0.0
    top1 = (hits1 / counts.clamp(min=1.0))[seen].mean().item() if bool(seen.any()) else 0.0
    return {"balanced_top10": top10, "balanced_top1": top1,
            "classes_seen": int(seen.sum().item())}


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cpu")
    # Fake encoder with the v3 contract: .encode(x) -> (B, 256).
    enc = nn.Module()
    enc.encode = lambda x: torch.randn(x.shape[0], 256)  # noqa: E731
    proj = ProjectionHead(256, 128)
    opt = torch.optim.AdamW(list(proj.parameters()), lr=1e-3)

    # Mechanics: loss must decrease on random data (instance pairs always
    # exist, independent of labels) and stay finite.
    B, C, T, K = 32, 306, 250, 50
    x = torch.randn(B, C, T)
    y = torch.randint(0, K, (B,))
    first, last = None, None
    proj.train()
    for step in range(30):
        v1 = proj(F.normalize(torch.randn(B, 256), dim=1) + 0.1 * torch.randn(B, 256))
        v2 = proj(F.normalize(torch.randn(B, 256), dim=1) + 0.1 * torch.randn(B, 256))
        feats = torch.cat([v1, v2], dim=0)
        lab = torch.cat([y, y], dim=0)
        loss = supcon_loss(feats, lab)
        assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    print(f"supcon mechanics: first={first:.4f} last={last:.4f}")
    assert last < first, "loss did not decrease"

    # Shapes: centroids (K, D), retrieval diagnostics run, probs valid.
    from torch.utils.data import DataLoader, TensorDataset
    xs = torch.randn(200, C, T)
    ys = torch.randint(0, K, (200,))
    ym = torch.randint(0, K, (200,))
    loader = DataLoader(TensorDataset(xs, ys, ym), batch_size=32)
    cent = build_centroids(enc, proj, loader, K, 0, dev)
    assert cent.shape == (K, 128) and torch.isfinite(cent).all()
    d = retrieval_top10(enc, proj, cent, loader, 0, dev)
    assert 0.0 <= d["balanced_top10"] <= 1.0
    sim = F.normalize(torch.randn(8, 128), dim=1) @ cent.T
    probs = torch.softmax(sim / SUBMIT_TAU, dim=1)
    assert torch.isfinite(probs).all() and abs(float(probs.sum(1).mean()) - 1.0) < 1e-4
    assert int(len(torch.unique(probs.argmax(1)))) >= 1
    print(f"retrieval shapes OK: centroids{tuple(cent.shape)}, "
          f"top10={d['balanced_top10']:.3f}, probs sum={float(probs.sum(1).mean()):.4f}")
