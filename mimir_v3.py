"""mimir_v3.py - Successor to mimir_model.py (PNPL 2026 Deep Track).

Why this supersedes v1, part by part (each part is forced; removing it
reintroduces a demonstrated failure):

1. No per-trial whiten. v1's TangentSpaceWhiten does a 306x306 eigh per trial
   with T=250<C=306 (rank-deficient) and centers away the evoked DC offset
   that carries word identity. Global train-fit z-scoring (already in
   data_pipeline meta.mean/std, applied identically at train and submit) is
   the correct decorrelation; per-trial whitening is dropped, not tuned.

2. No fixed kNN graph. The Kaggle log shows
   `WARNING: could not read H5 channel_names; will fall back to canonical
   order` (data_pipeline.py). A fixed 306x306 graph built on assumed order
   mixes the wrong sensors when H5 order differs. Depthwise spatial convs
   learn the unmixing from data with no channel-name dependency. Topology
   prior is kept as *learned* spatial filters (EEGNet-style), not a fixed
   matrix.

3. No deep dilated stack. v1's 7-8 residual dilated layers (RF 255-511) plus
   2-hop ReLU GCN smoothing plus softmax-attention over 250 steps plus
   LayerNorm collapsed inter-trial variance to col-std ~8e-08 (constant
   predictor, submit diversity 1/50). Shallow separable convs with RF ~200ms
   + context preserve transient amplitude. Pooling is mean + log-power, both
   amplitude-preserving; no softmax attention, no pre-logit LayerNorm.

4. Explicit spectral bias. Word identity in listening MEG lives partly in the
   high-gamma power envelope, not raw phase. v1's raw-waveform TCN had to
   invent squaring/logging from scratch. v3 builds it in (ShallowConvNet
   result): parallel mean (ERP amplitude) and log-power (oscillatory) readouts.

5. Single-head objective. v1 summed primary + moses equally with -100
   masking; logs show moses 0.6 vs primary 2.4 all run, backbone serving two
   masters, primary stuck at chance top1 0.02. v3 trains primary; moses is a
   0.2-weight auxiliary (submission still needs both 50-col blocks).

Contract (same as v1):
  forward(): (B, C, T) -> (primary_logits (B, Kp), moses_logits (B, Km))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MimirV3(nn.Module):
    """Spectral-spatial single-head decoder.

    Shapes for Deep track (C=306, T=250):
      x (B,306,250) -> temporal bank (B,8,306,250) -> depthwise spatial
      (B,16,1,250) -> pool/4 (B,16,1,62) -> separable (B,32,1,62)
      -> pool/5 (B,32,1,12) -> mean+logpower (B,64) -> fc -> (B,50)x2
    """

    def __init__(
        self,
        n_channels: int,
        n_primary_classes: int,
        n_moses_classes: int,
        f1: int = 8,          # temporal filter-bank width
        d: int = 2,           # spatial filters per temporal filter
        f2: int = 32,         # separable summary width
        dropout: float = 0.25,
    ):
        super().__init__()
        f_spatial = f1 * d  # 16

        # Temporal filter-bank: 51 samples = 204ms @250Hz, covers N100-P200.
        self.temporal = nn.Conv2d(1, f1, (1, 51), padding=(0, 25), bias=False)
        # Depthwise spatial: learns unmixing per band, no channel names needed.
        self.spatial = nn.Conv2d(f1, f_spatial, (n_channels, 1), groups=f1,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(f_spatial)
        self.pool1 = nn.AvgPool2d((1, 4))  # 250 -> 62
        self.drop1 = nn.Dropout(dropout)

        # Separable temporal summary.
        self.depthwise = nn.Conv2d(f_spatial, f_spatial, (1, 15),
                                   padding=(0, 7), groups=f_spatial, bias=False)
        self.pointwise = nn.Conv2d(f_spatial, f2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 5))  # 62 -> 12
        self.drop2 = nn.Dropout(dropout)

        self.fc_primary = nn.Linear(f2 * 2, n_primary_classes)
        self.fc_moses = nn.Linear(f2 * 2, n_moses_classes)
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[mimir_v3] params={n_params/1e3:.1f}K "
              f"(f1={f1}, d={d}, f2={f2}, dropout={dropout})")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        h = x.unsqueeze(1)
        h = self.temporal(h)                    # (B, f1, C, T)
        h = self.spatial(h)                     # (B, 16, 1, T)
        h = self.bn1(h)
        h = F.elu(h, inplace=False)
        h = self.pool1(h)                       # (B, 16, 1, 62)
        h = self.drop1(h)
        h = self.depthwise(h)
        h = self.pointwise(h)                   # (B, f2, 1, 62)
        h = self.bn2(h)
        h = F.elu(h, inplace=False)
        h = self.pool2(h)                       # (B, f2, 1, 12)
        h = self.drop2(h)
        h = h.squeeze(2)                        # (B, f2, 12)
        h_mean = h.mean(dim=-1)                                   # ERP amplitude
        h_pow = torch.log((h ** 2).mean(dim=-1) + 1e-6)           # bandpower
        return torch.cat([h_mean, h_pow], dim=1)                  # (B, 2*f2)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.fc_primary(h), self.fc_moses(h)


# Compatibility alias so train.py can call one name.
Mimir = MimirV3


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T, KP, KM = 4, 306, 250, 50, 50
    m = MimirV3(n_channels=C, n_primary_classes=KP, n_moses_classes=KM)
    m.eval()
    x = torch.randn(B, C, T)
    with torch.no_grad():
        p, mo = m(x)
    assert p.shape == (B, KP) and mo.shape == (B, KM)
    assert torch.isfinite(p).all() and torch.isfinite(mo).all()
    # Input-dependence: different inputs must give different outputs.
    with torch.no_grad():
        p2, _ = m(x + 2.0)
    assert not torch.allclose(p, p2, atol=1e-4), "constant output"
    # Diversity smoke: random batch should not collapse to one argmax.
    assert len(torch.unique(p.argmax(1))) >= 2, "single argmax on noise"
    print(f"v3 OK: {(B,C,T)} -> {tuple(p.shape)}, "
          f"unique-argmax={len(torch.unique(p.argmax(1)))}, "
          f"col-std={p.std(0).mean():.2e}")
