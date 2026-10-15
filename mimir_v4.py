"""mimir_v4.py - Physics-informed successor (PNPL 2026 Deep Track).

Each part is forced by measurement or literature, not taste:

1. Fixed 4-band FIR filterbank (1-8 / 8-30 / 30-70 / 70-120 Hz @250Hz).
   Heard-word identity lives in slow evoked fields (N100m/P200m, delta/theta
   envelope tracking) AND gamma power envelopes. A raw broadband conv must
   invent spectral separation from scratch; v3's single broadband power mixed
   all bands. Windowed-sinc FIRs in pure torch (no scipy needed on Kaggle).

2. 306 -> bottleneck spatial projection (16 dims per band, 64 total).
   Post-SSS data has rank ~65 (SSS keeps <=80 inside components; MVPA work
   shows rank reduction 306->65 and saturation by ~25 sensors). Learning
   spatial filters in 306-dim is learning in a 5x inflated, degenerate
   space. The bottleneck enforces the true dimensionality. It is LEARNED
   (not fixed PCA) so no extra Kaggle fitting step can fail, and it needs no
   channel names (the H5-name failure mode that poisoned v1's graph).

3. Broadband stream = v3 encoder verbatim (proven: memorized the 256 gate).
   Temporal bank f1 -> depthwise spatial -> separable -> mean+logpower.

4. Onset/offset attention (2 queries over pooled time steps).
   A 1s word window holds the onset complex (~50-250ms: P50m/N100m/P200m)
   AND the offset response (word end, duration-dependent) AND often the next
   word's onset (continuous speech, ~3 words/s). Mean pooling dilutes the
   200ms evoked complex in 800ms of context. Two learnable queries separate
   early (onset) from late (offset/sustained) evidence. Global stats are kept
   alongside, so the model works even if attention contributes nothing.

5. encode() contract: (B, 306, 250) -> (B, D) with D = 4*f2 (broadband
   mean+logpower + onset/offset attention) + 64 (bandpower stream).

Contract matches v3: forward() -> (primary_logits, moses_logits) for the
legacy CE path (linear heads over encode()).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# (low, high) Hz per band at TARGET_SFREQ=250. High edge 120 < Nyquist 125.
BANDS = ((1.0, 8.0), (8.0, 30.0), (30.0, 70.0), (70.0, 120.0))
FIR_KERNEL = 65
BAND_BOTTLENECK = 16   # spatial dims per band; 4x16=64 ~ SSS rank
N_QUERIES = 2          # onset query + offset/sustained query


def fir_bandpass(f1: float, f2: float, kernel: int = FIR_KERNEL,
                 fs: float = 250.0) -> torch.Tensor:
    """Windowed-sinc FIR bandpass (unit L2 norm, zero-phase via centering)."""
    n = torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2.0
    h_high = (2.0 * f2 / fs) * torch.sinc(2.0 * f2 * n / fs)
    h_low = (2.0 * f1 / fs) * torch.sinc(2.0 * f1 * n / fs)
    h = (h_high - h_low) * torch.hamming_window(kernel, periodic=False)
    return h / h.norm().clamp(min=1e-9)


class MimirV4(nn.Module):
    """Two-stream spectral-spatial encoder + onset/offset attention."""

    def __init__(self, n_channels: int, n_primary_classes: int,
                 n_moses_classes: int, f1: int = 16, d: int = 2,
                 f2: int = 128, dropout: float = 0.25):
        super().__init__()
        self.f2 = f2

        # ---- fixed spectral filterbank: 4 bands x C depthwise ----
        fir = torch.stack([fir_bandpass(lo, hi) for lo, hi in BANDS])  # (4, K)
        self.register_buffer("fir", fir)
        self.band_pad = FIR_KERNEL // 2

        # ---- broadband stream (v3 verbatim: proven on the gate) ----
        self.temporal = nn.Conv2d(1, f1, (1, 51), padding=(0, 25), bias=False)
        self.spatial = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depthwise = nn.Conv2d(f1 * d, f1 * d, (1, 15), padding=(0, 7),
                                   groups=f1 * d, bias=False)
        self.pointwise = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 5))
        self.drop2 = nn.Dropout(dropout)

        # ---- bandpower stream: per-band 306 -> 16 bottleneck ----
        self.band_mix = nn.ModuleList(
            [nn.Conv1d(n_channels, BAND_BOTTLENECK, 1) for _ in BANDS])

        # ---- onset/offset attention over pooled broadband time ----
        self.q = nn.Parameter(torch.randn(N_QUERIES, 64) * 0.1)
        self.k_proj = nn.Linear(f2, 64, bias=False)
        self.v_proj = nn.Linear(f2, 64, bias=False)

        # 2*f2 (broadband mean+logpower) + 128 (bandpower: 4 bands x
        # (16 mean + 16 logpower)) + 128 (2 queries x 64)
        self.feat_dim = (2 * f2 + 2 * len(BANDS) * BAND_BOTTLENECK
                         + N_QUERIES * 64)
        self.head_primary = nn.Linear(self.feat_dim, n_primary_classes)
        self.head_moses = nn.Linear(self.feat_dim, n_moses_classes)
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[mimir_v4] params={n_params/1e3:.1f}K feat={self.feat_dim} "
              f"(f1={f1}, f2={f2}, bands={len(BANDS)})")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if m.weight.requires_grad:
                    nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None and m.bias.requires_grad:
                    nn.init.zeros_(m.bias)

    def _broadband(self, x: torch.Tensor):
        h = x.unsqueeze(1)                        # (B, 1, C, T)
        h = self.temporal(h)
        h = self.spatial(h)                       # (B, f1*d, 1, T)
        h = self.drop1(self.pool1(F.elu(self.bn1(h), inplace=False)))
        h = self.depthwise(h)
        h = self.pointwise(h)                     # (B, f2, 1, T/4)
        h = self.drop2(self.pool2(F.elu(self.bn2(h), inplace=False)))
        return h.squeeze(2)                       # (B, f2, Tp)

    def _bandpower(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> per-band depthwise FIR -> bottleneck -> stats
        B, C, T = x.shape
        K = self.fir.shape[1]
        outs = []
        for b, mix in enumerate(self.band_mix):
            w = self.fir[b].expand(C, 1, K)                    # (C, 1, K)
            f = F.conv1d(x, w, groups=C, padding=self.band_pad)  # (B, C, T)
            h = mix(f)                                         # (B, 8, T)
            outs.append(h.mean(dim=-1))                        # ERP energy
            outs.append(torch.log((h ** 2).mean(dim=-1) + 1e-6))  # bandpower
        return torch.cat(outs, dim=1)                          # (B, 64)

    def _attend(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, f2, Tp) -> 2 queries separate early vs late evidence
        k = self.k_proj(h.transpose(1, 2))                     # (B, Tp, 64)
        v = self.v_proj(h.transpose(1, 2))
        q = self.q.unsqueeze(0).expand(h.shape[0], -1, -1)      # (B, 2, 64)
        w = torch.softmax(q @ k.transpose(1, 2) / 8.0, dim=-1)  # (B, 2, Tp)
        return (w @ v).reshape(h.shape[0], -1)                 # (B, 128)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self._broadband(x)                                 # (B, f2, Tp)
        h_mean = h.mean(dim=-1)
        h_pow = torch.log((h ** 2).mean(dim=-1) + 1e-6)
        a = self._attend(h)
        b = self._bandpower(x)
        return torch.cat([h_mean, h_pow, a, b], dim=1)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.head_primary(h), self.head_moses(h)


Mimir = MimirV4


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T, KP, KM = 4, 306, 250, 50, 50
    m = MimirV4(n_channels=C, n_primary_classes=KP, n_moses_classes=KM)
    m.eval()
    # FIR sanity: band filters must be unit-norm and near-orthogonal-ish.
    fb = m.fir
    assert abs(float(fb.norm(dim=1).mean()) - 1.0) < 1e-5, "FIR not unit norm"
    x = torch.randn(B, C, T)
    with torch.no_grad():
        p, mo = m(x)
    assert p.shape == (B, KP) and mo.shape == (B, KM)
    assert torch.isfinite(p).all() and torch.isfinite(mo).all()
    # Power-envelope sensitivity (the signal the band stream reads).
    x2 = x * torch.linspace(0.5, 2.0, B).view(B, 1, 1)
    with torch.no_grad():
        p2, _ = m(x2)
    assert not torch.allclose(p, p2, atol=1e-4), "power envelope ignored"
    assert float(p2.std(0).mean()) > 1e-3, "collapsed outputs"
    print(f"v4 OK: {(B,C,T)} -> {tuple(p.shape)}, feat={m.feat_dim}, "
          f"col-std={p2.std(0).mean():.2e}")
