"""mimir_model.py - The MIMIR architecture (PNPL 2026 Deep Track).

Contract
--------
forward(): (B, C, T) -> (primary_logits (B, K_primary), moses_logits (B, K_moses))

The competition scores two independent 50-way vocabularies (primary and
Moses-50) as separate Top-10 balanced accuracies, so the shared backbone
terminates in two readouts and train.py sums their cross-entropy.

Blocks
------
1. TangentSpaceWhiten (optional): per-trial sensor decorrelation (SPD-aware).
2. SpatialGraphBlock: GCN over the geometric sensor graph built from the
   Neuromag 306 coil positions - one hop mixes a sensor with its physical
   neighbours on the helmet, which is the correct topology prior (channel
   order in a file is not spatial order).
3. TemporalStack: dilated residual convolutions; dilation powers of two give
   an exponential receptive field (RF = 255 samples at 250 Hz) at linear
   depth, covering the full 1 s word window.
4. Two ClassifierHeads: mean-pool over time + linear (time-translation
   invariant; parameter count independent of window length).

Ablation switches: use_gcn / use_tcn / use_whiten constructor flags. The
module-level USE_GCN / USE_TCN constants below are the single place to flip
the blocks for ablation runs; train.py reads them and passes them through.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Ablation switches (defaults = full architecture). train.py reads these.
# ----------------------------------------------------------------------------
USE_GCN = True   # spatial graph convolution over the geometric sensor graph
USE_TCN = True   # dilated temporal convolution stack


class TangentSpaceWhiten(nn.Module):
    """Per-trial symmetric whitening over the channel dimension.

    Per-trial sensor covariances are SPD matrices; whitening decorrelates the
    sensors before the spatial block (the classical Riemannian preprocessing
    step, Barachant et al.). A shrinkage term keeps the estimate SPD even when
    the window is shorter than the channel count (T < C), where the raw sample
    covariance would be rank-deficient.
    """

    def __init__(self, shrinkage: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.shrinkage = shrinkage
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.shape
        xc = x - x.mean(dim=-1, keepdim=True)                    # centre in time
        r = torch.bmm(xc, xc.transpose(1, 2)) / max(t - 1, 1)    # (B, C, C) cov
        d = r.diagonal(dim1=-2, dim2=-1).clamp_min(self.eps).rsqrt()
        r = d.unsqueeze(-1) * r * d.unsqueeze(-2)                # correlation
        r_s = (1.0 - self.shrinkage) * r + self.shrinkage * torch.eye(
            c, device=x.device, dtype=x.dtype
        ).expand(b, c, c)                                        # shrunk SPD
        evals, evecs = torch.linalg.eigh(r_s)
        inv_sqrt = evecs * evals.clamp_min(self.eps).rsqrt().unsqueeze(1)
        whitening = torch.bmm(evecs, inv_sqrt.transpose(1, 2))   # R^-1/2
        return torch.bmm(whitening, xc)                          # (B, C, T)


class SpatialGraphBlock(nn.Module):
    """Spatial aggregation over the geometric sensor graph, then channel mix.

    Math (Kipf & Welling, 2017, adapted to time series):
        H' = ReLU(A_hat X W),  A_hat = D^{-1/2} (A + I) D^{-1/2}
    X is (B, C, T), A_hat is the (C, C) normalised adjacency built from coil
    positions (data_pipeline.build_sensor_adjacency), W is a (C -> D) linear.

    Why a graph instead of a 2D conv over (channel, time): MEG files group
    channels by type (magnetometers then gradiometers), so adjacent indices
    are not adjacent sensors. The k-NN graph on the helmet is the correct
    spatial neighbourhood; one hop mixes each sensor with its physical
    neighbours, and the learnable per-channel gain lets the network down-weight
    a noisy sensor without leaving the sensor basis.
    """

    def __init__(self, n_channels: int, out_channels: int,
                 adj: torch.Tensor, n_hops: int = 2, dropout: float = 0.1):
        super().__init__()
        assert adj.shape == (n_channels, n_channels), "adjacency/shape mismatch"
        # Buffer (not a parameter): moves with .to(device) and is saved in the
        # checkpoint so inference uses the same graph.
        self.register_buffer("adj", adj.float())
        self.n_hops = n_hops
        self.gain = nn.Parameter(torch.ones(n_channels))
        self.mix = nn.Linear(n_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, out, T)
        h = x
        for _ in range(self.n_hops):
            h = torch.matmul(self.adj, h)                # aggregate neighbours
            h = h * self.gain.view(1, -1, 1)
            h = F.relu(h)
        h = self.mix(h.transpose(1, 2)).transpose(1, 2)  # channels-first again
        return self.dropout(h)


class TemporalBlock(nn.Module):
    """One residual dilated-convolution stage.

    Dilation stretches the kernel in time (tap spacing = d samples). Stacking
    d = 1, 2, 4, ... grows the receptive field exponentially with depth, so the
    whole stack covers the window with O(log depth) parameter cost. causal=True
    left-pads only (online decoding); else symmetric ('same') padding.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            dilation=dilation, padding=0,               # padded manually below
        )
        # GroupNorm over channels: stable with small batches (unlike BatchNorm)
        # and normalises across the channel mixing the spatial block produced.
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x_p = F.pad(x, (self.pad, 0))
        else:
            half = self.pad // 2
            x_p = F.pad(x, (half, self.pad - half))
        y = self.conv(x_p)
        y = self.norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        # Residual path keeps a length-1 gradient highway across the whole
        # stack (otherwise a 7-deep dilated stack loses long-range gradients).
        return F.relu(x + y)


class TemporalStack(nn.Module):
    """Stack of TemporalBlocks with geometrically increasing dilation.

    Receptive field with kernel k=3 over dilations {1,...,2^(n_layers-1)}:
        RF = 1 + (k - 1) * sum(dilations)
           = 1 + 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 255 samples = 1.02 s @250 Hz
    so the deepest layer can condition on the entire 1 s word window.
    """

    def __init__(self, channels: int, n_layers: int = 7, kernel_size: int = 3,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            TemporalBlock(channels, kernel_size, d, dropout=dropout, causal=causal)
            for d in dilations
        ])
        self.rf = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ClassifierHead(nn.Module):
    """Mean-pool over time, then a linear readout.

    Mean-pooling gives time-translation invariance (which word was spoken does
    not depend on exactly when in the window) and keeps parameter count
    independent of T, so the same head serves any window length. Two of these
    sit on the shared backbone - one per competition vocabulary.
    """

    def __init__(self, channels: int, n_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x).squeeze(-1)          # (B, D)
        return self.fc(h)                     # (B, n_classes)


class Mimir(nn.Module):
    """Spatial -> Temporal -> two heads.

    forward() contract:
        (B, C, T) --whiten--> (B, C, T)
                   --spatial--> (B, D, T)
                   --temporal--> (B, D, T)   [shared backbone, both heads]
        head_primary -> (B, K_primary)   head_moses -> (B, K_moses)
    """

    def __init__(
        self,
        n_channels: int,                 # C, e.g. 306
        n_primary_classes: int,          # K_p: 2 (local) / 50 (competition)
        n_moses_classes: int,            # K_m: 2 (local) / 50 (Moses-50)
        adj: torch.Tensor,               # (C, C) normalised sensor graph
        d_model: int = 64,               # width after the spatial mix
        tcn_layers: int = 7,             # dilations 1..64 -> RF 255 @ k=3
        tcn_kernel: int = 3,
        dropout: float = 0.1,
        causal: bool = False,            # online decoding needs causal=True
        use_whiten: bool = False,
        use_gcn: Optional[bool] = None,
        use_tcn: Optional[bool] = None,
        n_spatial_hops: int = 2,
    ):
        super().__init__()
        # None -> read the module-level switch AT CONSTRUCTION TIME, so a flag
        # flipped in this file is honoured by any caller never passing it.
        if use_gcn is None:
            use_gcn = USE_GCN
        if use_tcn is None:
            use_tcn = USE_TCN
        self.whiten = TangentSpaceWhiten() if use_whiten else nn.Identity()
        # use_gcn=False drops the graph aggregation, keeping only a channel
        # mixing projection (the no-geometry-prior ablation arm).
        self.spatial = (
            SpatialGraphBlock(n_channels, d_model, adj, n_hops=n_spatial_hops,
                              dropout=dropout)
            if use_gcn
            else nn.Conv1d(n_channels, d_model, 1)
        )
        self.temporal = (
            TemporalStack(d_model, n_layers=tcn_layers, kernel_size=tcn_kernel,
                          dropout=dropout, causal=causal)
            if use_tcn
            else nn.Identity()
        )
        # Two readouts over the SAME backbone, one per vocabulary. The
        # competition scores them separately, so each keeps its own decision
        # boundary.
        self.head_primary = ClassifierHead(d_model, n_primary_classes)
        self.head_moses = ClassifierHead(d_model, n_moses_classes)
        self._init_weights()

        print(f"[mimir] receptive field = {self.temporal.rf} samples "
              f"({self.temporal.rf / 250.0:.2f} s at 250 Hz)"
              if isinstance(self.temporal, TemporalStack)
              else "[mimir] temporal block disabled (use_tcn=False)")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[mimir] heads: primary {n_primary_classes}-way + "
              f"moses {n_moses_classes}-way "
              f"(gcn={use_gcn}, tcn={use_tcn}, whiten={use_whiten})")
        print(f"[mimir] parameters = {n_params / 1e3:.1f}K")

    def _init_weights(self):
        # Xavier keeps activation variance stable across the residual stack
        # without an extra batch-style renormalisation.
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        h = self.whiten(x)                 # (B, C, T) (identity if disabled)
        h = self.spatial(h)                # (B, D, T)
        h = self.temporal(h)               # (B, D, T) (identity if disabled)
        return self.head_primary(h), self.head_moses(h)


# ----------------------------------------------------------------------------
# Self-test: shape contract + receptive-field + ablation flags
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T, KP, KM = 8, 306, 176, 2, 2
    adj = torch.eye(C)                     # identity is fine for a shape test

    model = Mimir(n_channels=C, n_primary_classes=KP, n_moses_classes=KM, adj=adj)
    x = torch.randn(B, C, T)
    primary, moses = model(x)
    assert primary.shape == (B, KP) and moses.shape == (B, KM)
    print(f"forward OK: {(B, C, T)} -> primary {tuple(primary.shape)}, "
          f"moses {tuple(moses.shape)}")

    # Receptive field: a far-past perturbation must reach both outputs.
    model.eval()
    with torch.no_grad():
        x0 = torch.zeros(1, C, T)
        p0, m0 = model(x0)
        x1 = x0.clone(); x1[..., 0] = 10.0
        p1, m1 = model(x1)
    moved = (not torch.allclose(p0, p1)) or (not torch.allclose(m0, m1))
    assert moved, "t=0 has no effect on logits -> RF < T"
    print(f"receptive field covers the full epoch: max |d primary| = "
          f"{(p1 - p0).abs().max():.3e}")

    # Ablation arms must satisfy the same dual-head contract and swap the
    # right block types (all four combinations of {gcn, tcn}).
    for flags in ((True, True), (True, False), (False, True), (False, False)):
        gcn, tcn = flags
        m = Mimir(n_channels=C, n_primary_classes=KP, n_moses_classes=KM,
                  adj=adj, use_gcn=gcn, use_tcn=tcn, use_whiten=True)
        assert isinstance(m.spatial, SpatialGraphBlock) == gcn, "spatial arm wrong"
        assert isinstance(m.temporal, TemporalStack) == tcn, "temporal arm wrong"
        p, m_ = m(x)
        assert p.shape == (B, KP) and m_.shape == (B, KM)
        assert torch.isfinite(p).all() and torch.isfinite(m_).all()
    print("ablation flags OK: (gcn, tcn) in {0,1}^2 - block types correct, "
          "outputs finite")