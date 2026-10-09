"""data_pipeline.py - Data loaders for MIMIR (PNPL 2026 Deep Track).

Two sources behind one 4-tuple contract (dataset, adj, meta, loader):
  1. MNE sample (build_mne_sample): local smoke-test data.
  2. LibriBrain100 (build_libribrain100): the competition training set
     (Deep Track, subject 0); ~20 GB on first use.

Every batch element is (x: float32 (C, T), y_primary: int64, y_moses: int64).
Swap USE_REAL_DATA to route build() to either source.

z-scoring (no leakage): per-channel stats are always fit on the TRAIN
partition only. The MNE path splits train/val first, fits its statistics on
the train trials, and applies the same statistics to both splits. The
LibriBrain path uses pnpl's standardization, which computes channel stats
over the pnpl train partition and caches them in the H5 metadata.
"""

from __future__ import annotations

import functools
import gc
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import mne

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Bandpass 1-100 Hz: removes slow drift without killing the evoked response,
# and keeps the gamma band (30-100 Hz) whose amplitude envelope carries the
# theta->gamma cross-frequency coupling used for decoding. 100 Hz sits under
# the 250 Hz Nyquist (125 Hz), so resampling cannot alias.
L_FREQ = 1.0
H_FREQ = 100.0

# Mains frequency of the MNE sample recording (US). The competition
# preprocessing chain already includes a notch step.
LINE_FREQ = 60.0

# Window locked to stimulus onset: a short pre-stimulus baseline plus the
# speech-evoked response. Matches the competition task convention.
TMIN = -0.2
TMAX = 0.5

TARGET_SFREQ = 250.0

EVENT_ID = {"auditory/left": 1, "auditory/right": 2}

# Peak-to-peak threshold per channel type (SI units); drops gross artifacts
# (blinks, head jumps) inside the epoch window.
REJECT = {"mag": 4e-12, "grad": 4e-10}

# k-NN degree of the sensor graph. After two GCN hops the neighbourhood is
# roughly 5-10 cm on the helmet - the spatial scale of a cortical patch's
# field spread through the skull.
N_NEIGHBORS = 8

# ----------------------------------------------------------------------------
# Real (competition) data - LibriBrain100
# ----------------------------------------------------------------------------
# THE single switch: build() routes to LibriBrain100 when True, else MNE.
USE_REAL_DATA = False

# 1.0 s word windows (tmin=0, tmax=1) => (306, 250) at 250 Hz, matching the
# holdout contract (WINDOW_SECONDS == 1.0), so train time-axis == inference.
LIBRI_TMIN = 0.0
LIBRI_TMAX = 1.0
LIBRI_BATCH_SIZE = 32
# num_workers=0: the pnpl dataset holds open H5 handles and a per-sample
# cache; workers would copy them (or fail to pickle) for zero gain on a
# 4-vCPU box where the model is the bottleneck.
LIBRI_NUM_WORKERS = 0

# Preload all accepted windows into one (N, 306, 250) tensor up-front instead
# of re-reading the H5 files on every __getitem__. This removes the disk-bound
# per-step H5 reads that dominated Kaggle epoch time (hours -> minutes).
# Memory guard: cache only if it fits under CACHE_MEMORY_LIMIT bytes, else
# fall back to lazy sampling. Override with $CACHE_WINDOWS=0/1.
LIBRI_CACHE_WINDOWS = int(os.environ.get("CACHE_WINDOWS", 1)) == 1
LIBRI_CACHE_MEMORY_LIMIT = 6_500_000_000   # ~6.5 GB headroom on 16 GB boxes
LIBRI_CLIP_BOUNDARY = 10.0                 # pnpl StandardizationMixin default

# Validation holdout of the training split.
VAL_FRACTION = 0.1
LIBRI_SEED = 1337

# MNE local smoke run split (stratified by class).
MNE_VAL_FRACTION = 0.15
MNE_SEED = 0


# ----------------------------------------------------------------------------
# Data root (Kaggle override)
# ----------------------------------------------------------------------------
# Point $MIMIR_DATA_ROOT (or $KAGGLE_DATA_PATH) at the bulky data location
# instead of the repo data/ folder - e.g. a Kaggle dataset mount such as
# /kaggle/input/libribrain100. The path is re-read at each call so a runtime
# export in the Kaggle notebook takes effect immediately.
def _data_root() -> Path:
    env = os.environ.get("MIMIR_DATA_ROOT") or os.environ.get("KAGGLE_DATA_PATH")
    return Path(env) if env else (Path(__file__).resolve().parent / "data")


# ----------------------------------------------------------------------------
# Sensor graph: geometric k-NN adjacency over Neuromag 306 positions
# ----------------------------------------------------------------------------
def build_sensor_adjacency_from_coords(
    ch_names: Sequence[str],
    coords: np.ndarray,
    n_neighbors: int = N_NEIGHBORS,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Gaussian-kNN sensor adjacency from explicit (C, 3) channel coordinates.

    The 306 Neuromag sensors lie on a helmet (a curved 2D surface in 3D).
    Channel *index* order in a file is not spatial order (magnetometers are
    stored before gradiometers), so we build a k-NN graph on the actual coil
    positions - the physical topology prior used by the GCN block.

    Co-located coils: each helmet site mounts one magnetometer and two
    orthogonal planar gradiometers at identical XYZ coordinates (concentric
    triplets), so the 306 channels occupy only 102 distinct positions. k-NN
    over the raw 306 points would return the two co-located partners at
    distance 0 first, driving the kernel bandwidth sigma -> 0 and nulling the
    Gaussian weight exp(-d^2/2 sigma^2) for every real spatial neighbour.
    We therefore run the k-NN search and the sigma estimate over the UNIQUE
    locations only, then cover each location by its channels: spatial edges
    keep their Gaussian weights, and co-located triplet members link at
    exp(-0) = 1 (the field plus its two orthogonal spatial derivatives at a
    point - exactly what the concentric hardware measures).

    Weights: w_ij = exp(-||p_i - p_j||^2 / 2 sigma^2), sigma = median kNN
    distance, then symmetric normalisation D^{-1/2} (A + I) D^{-1/2} (Kipf &
    Welling, 2017) to keep the aggregation operator's spectral radius in
    check over multiple hops.

    Returns a dense (C, C) float32 tensor (dense 306x306 matmuls beat sparse
    kernels at this size). Channels with a missing/custom position get a bare
    self-loop.
    """
    ch_names = [str(n) for n in ch_names]
    n_channels = len(ch_names)
    locs = np.asarray(coords, dtype=np.float64).reshape(n_channels, 3)

    valid = np.isfinite(locs).all(axis=1) & (np.abs(locs).sum(axis=1) > 1e-9)
    if not valid.any():
        raise ValueError("Sensor graph is empty - check channel coords.")

    # Deduplicate to unique locations (concentric triplets fold together).
    keys = np.round(locs[valid], 6)
    uniq, loc_id = np.unique(keys, axis=0, return_inverse=True)
    n_locs = len(uniq)
    if n_locs < 2:
        # No spatial edges possible between fewer than two distinct sites.
        return torch.eye(n_channels, dtype=torch.float32)

    dloc = np.linalg.norm(uniq[:, None, :] - uniq[None, :, :], axis=-1)
    dloc[~np.isfinite(dloc)] = np.inf
    np.fill_diagonal(dloc, np.inf)                       # no self-edges
    loc_nn = np.argsort(dloc, axis=1)[:, :n_neighbors]
    loc_d = dloc[np.arange(n_locs)[:, None], loc_nn]

    if sigma is None:
        sigma = float(np.median(loc_d))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sensor graph is empty - check channel coords.")

    # Gaussian graph over locations, symmetrised (edge union), plus self-loops.
    Aloc = np.zeros((n_locs, n_locs), dtype=np.float64)
    src = np.repeat(np.arange(n_locs), n_neighbors)
    Aloc[src, loc_nn.ravel()] = np.exp(-(loc_d.ravel() ** 2) / (2.0 * sigma ** 2))
    Aloc = np.maximum(Aloc, Aloc.T)
    Aloc = Aloc + np.eye(n_locs)

    # Expand locations to channels: co-located triplets share the self-loop
    # weight (1) -- the concentric cross-links.
    valid_idx = np.arange(n_channels)[valid]
    adj = np.zeros((n_channels, n_channels), dtype=np.float64)
    for rank, ch_i in enumerate(valid_idx):
        adj[ch_i, valid_idx] = Aloc[loc_id[rank], loc_id]
    invalid = np.invert(valid)
    adj[invalid, invalid] = 1.0

    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    return torch.from_numpy(adj.astype(np.float32))


def build_sensor_adjacency(
    info: mne.Info,
    n_neighbors: int = N_NEIGHBORS,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """(C, C) geometric kNN adjacency from an ``mne.Info``.

    Uses each channel's device-frame coil position (``ch["loc"][:3]``). This
    is the MNE-sample path of the geometry prior.
    """
    return build_sensor_adjacency_from_coords(
        [ch["ch_name"] for ch in info["chs"]],
        np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float64),
        n_neighbors=n_neighbors,
        sigma=sigma,
    )


@functools.lru_cache(maxsize=1)
def _neuromag306_names_coords() -> tuple[list[str], np.ndarray]:
    """Canonical Neuromag 306 sensor geometry (from MNE core).

    ``read_meg_canonical_info("neuromag")`` (MNE >= 1.12) ships the canonical
    Neuromag/Vectorview system: 102 helmet sites, each carrying one
    magnetometer (coil_type 3024) and two orthogonal planar gradiometers
    (3012), with device-frame XYZ position and orientation. LibriBrain100 was
    recorded with a MEGIN TRIUX Neo - the same 306-channel concentric-triplet
    helmet (Ozdogan et al., NeurIPS 2025 Datasets & Benchmarks,
    arXiv:2506.02098; pnpl docs/libribrain100.md) - so this is the correct
    geometry prior.

    Note: the canonical CSV stores each triplet grad-first (MEG 0113, MEG
    0112, then magnetometer MEG 0111); positions are resolved BY NAME below,
    never by row order.
    """
    from mne.channels import read_meg_canonical_info

    info = read_meg_canonical_info("neuromag")
    names = list(info.ch_names)
    coords = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float64)
    return names, coords


def _norm_ch(name: object) -> str:
    """Canonical channel-name key for NAME-based position lookup.

    Drops every non-alphanumeric character and uppercases, so ``MEG 0111``,
    ``MEG0111``, ``meg_0111``, ``MEG-0111``, ``meg.0111`` all resolve to the
    same key; any channel whose key is not ``MEG <number>`` (EEG/EOG/STI/
    custom) is returned as-is and later maps to a self-loop.
    """
    s = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    m = re.match(r"^MEG(\d+)$", s)
    return f"MEG {m.group(1)}" if m else s


@functools.lru_cache(maxsize=1)
def _neuromag306_positions() -> dict[str, np.ndarray]:
    """Canonical Neuromag 306 (C, 3) positions, keyed by normalised name."""
    names, coords = _neuromag306_names_coords()
    return {_norm_ch(n): c for n, c in zip(names, coords)}


def build_neuromag306_adjacency(
    ch_names: Sequence[str],
    n_neighbors: int = N_NEIGHBORS,
) -> torch.Tensor:
    """(C, C) geometric kNN adjacency for a Neuromag-306 channel list.

    Maps each channel (in the dataset's order) to its canonical Neuromag 306
    position BY NAME and builds the Gaussian kNN graph. Robust to any ordering
    or naming difference between the H5 and the canonical CSV (folding is in
    ``_norm_ch``), because geometry is looked up through a name-keyed
    dictionary. Channels with no canonical position (renamed/custom/non-MEG)
    get a bare self-loop, with a warning.
    """
    canon_lookup = _neuromag306_positions()
    raw_names = [str(n) for n in ch_names]
    coords = np.array(
        [canon_lookup.get(_norm_ch(n), np.zeros(3)) for n in raw_names],
        dtype=np.float64,
    )
    missing = [n for n in raw_names if _norm_ch(n) not in canon_lookup]
    n_missing = len(missing)
    if n_missing:
        fraction = n_missing / max(1, len(raw_names))
        degree = ">5% unmapped - the sensor graph is likely incorrect" \
            if fraction > 0.05 else "unmapped channels isolated as self-loops"
        print(
            f"[data_pipeline] WARNING ({degree}): {n_missing}/{len(raw_names)} "
            f"channels have no canonical Neuromag306 position"
        )
        shown: list[str] = missing if n_missing <= 5 else missing[:5] + ["..."]
        print(f"[data_pipeline]   unmapped: {shown}")
    return build_sensor_adjacency_from_coords(raw_names, coords, n_neighbors)


def _h5_channel_names(h5) -> list[str]:
    """Real channel names from a LibriBrain H5 (attr or root dataset).

    pnpl stores the run's channel names as H5 metadata (``attrs["channel_names"]``,
    or a legacy root ``channel_names`` dataset) but does not forward them onto
    the dataset object, so we read them from the first run's cached H5 handle.
    """
    names = []
    if "channel_names" in h5.attrs:
        names = h5.attrs["channel_names"]
    elif "channel_names" in h5:
        names = h5["channel_names"][:]
        if getattr(names, "ndim", 0) == 2:        # legacy serialisation: (1, C)
            names = names.reshape(-1)
    if isinstance(names, bytes):
        names = names.decode("utf-8", "replace").split(",")
    elif isinstance(names, str):
        names = names.split(",")
    return [
        s.strip()
        for s in (
            x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
            for x in np.asarray(names).ravel()
        )
        if s.strip()
    ]


# ----------------------------------------------------------------------------
# Training-time augmentation
# ----------------------------------------------------------------------------
class TrialAugmentation(nn.Module):
    """Anti-overfitting augmentation for (B, C, T) MEG batches, train-only.

    Each transform encodes an invariance a real decoder must have, so the
    model cannot memorise trial-specific noise:
      1. Channel dropout (p=0.10): zero a fraction of sensors per trial, so
         the decoder cannot anchor on any single sensor and must rely on
         neighbourhood structure (robustness to bad/flat sensors).
      2. Time warp (+-5%): linear stretch/compress with crop back to T,
         mimicking response-latency jitter (~+-35 ms on a 0.7 s window).
      3. Gaussian noise (2% of signal std): small sensor-noise perturbation.

    Order: noise is added BEFORE channel dropout so dropped channels end up
    exactly zero. In eval mode forward() is an exact identity (same contract
    as nn.Dropout), so validation metrics are provably clean.
    """

    def __init__(
        self,
        p_channel: float = 0.10,
        warp_range: tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.p_channel = p_channel
        self.warp_range = warp_range
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, t = x.shape

        # --- time warp (per-trial factor, crop/pad back to T) ----------------
        # Vectorized equivalent of per-sample resample + random view: one
        # batched interpolate to a common grid, then a per-sample gather with
        # clamped (replicated) boundaries.
        lo, hi = self.warp_range
        scale = torch.rand(b, device=x.device) * (hi - lo) + lo
        t2 = torch.clamp(torch.round(t * scale), min=2).long()
        g = int(max(t2.max().item(), t))
        if g != t:
            x = F.interpolate(x, size=g, mode="linear", align_corners=False)
        # random view start per sample: crop (stretched) or pad-left (compressed)
        stretched = t2 > t
        crop_start = torch.rand(b, device=x.device) * (t2 - t).clamp(min=0)
        pad_left = torch.rand(b, device=x.device) * (t - t2).clamp(min=0)
        off = torch.where(stretched, crop_start, -pad_left)           # (b,)
        # output j -> source coordinate (off + j) * (T / t2): the stretched
        # window sub-samples the t2 grid, the compressed view over-samples it.
        rate = (t / t2.float())
        pos = off[:, None] + torch.arange(t, device=x.device)[None, :] * \
              rate[:, None]
        idx = pos.floor().long().clamp(0, g - 1)[:, None, :]         # replicate edge
        x = torch.gather(x, dim=2, index=idx.expand(b, c, t))

        # --- gaussian noise (z-scored units: 2% of signal std) ---
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # --- channel dropout LAST so dropped channels are exact zeros ---
        if self.p_channel > 0:
            mask = (
                torch.rand(b, c, 1, device=x.device) >= self.p_channel
            ).to(x.dtype)
            x = x * mask
        return x


# ----------------------------------------------------------------------------
# Real-data helpers
# ----------------------------------------------------------------------------
def normalize_word(word: object) -> Optional[str]:
    """Canonicalise a word for vocabulary matching.

    events.tsv words are case-as-is, while the competition vocabularies ship
    with a unicode apostrophe (``it's`` is ``it\\u2019s``). NFKC does not map
    U+2019 to ASCII, so we fold every curly/quasi apostrophe to ASCII ``'`` on
    both sides before lowercasing, or ``it's`` would never match its vocab row.
    """
    if not isinstance(word, str):
        return None
    s = unicodedata.normalize("NFKC", word).strip().lower()
    return (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
    )


class MEGCompetitionDataset(Dataset):
    """pnpl.LibriBrain100 -> the same (x, y_primary, y_moses) contract.

    Wraps a prescripted index list over a pnpl dataset:
      - x: (C, T) float32 brain response (z-scored by pnpl over its TRAIN
           partition - no leakage by construction).
      - y_primary / y_moses: int64 index into the corresponding 50-word
           vocabulary, or -100 if the window's word is absent from it.
    -100 (CrossEntropyLoss ignore_index) is the single target. A window can
    teach the primary head, the Moses head, both, or (defensive fallback on a
    clipped recording) neither.
    """

    def __init__(
        self,
        base,
        indices: Sequence[int],
        word_to_primary: dict[str, int],
        word_to_moses: dict[str, int],
        ignore_index: int = -100,
        preloaded: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.word_to_primary = word_to_primary
        self.word_to_moses = word_to_moses
        self.ignore_index = ignore_index
        self.n_channels = int(base.n_channels)     # 306 (MEGIN TRIUX Neo)
        self.n_times = int(base.n_times)           # 250 (1 s @ 250 Hz)
        # Optional RAM cache: (N, C, T) pre-standardized windows. When present,
        # __getitem__ stops touching the H5 files entirely (the Kaggle epoch
        # time bottleneck). Kept as a plain attribute so DataLoader collation
        # (num_workers=0 here) stays cheap.
        self._preloaded = preloaded

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = int(self.indices[i])
        word = normalize_word(self.base.samples[j][5])   # sample tuple (.., word, ..)
        y_primary = self.word_to_primary.get(word, self.ignore_index)
        y_moses = self.word_to_moses.get(word, self.ignore_index)
        if self._preloaded is not None:
            x = self._preloaded[i]                  # (C, T); labels still derived
        else:
            try:
                x, _ = self.base[j]                  # (C, T) + unused task label
            except Exception:
                # A word onset near a recording's end may lack the 1-s tail (pnpl
                # raises rather than pad). Mask the window out of BOTH losses so
                # one stray run cannot kill a long job.
                x = torch.zeros(self.n_channels, self.n_times, dtype=torch.float32)
                y_primary = y_moses = self.ignore_index
        return (
            x,
            torch.tensor(y_primary, dtype=torch.int64),
            torch.tensor(y_moses, dtype=torch.int64),
        )


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
class MEGEpochDataset(Dataset):
    """In-memory (N, C, T) float32 array + labels -> Tensors.

    The MNE sample container; the LibriBrain path replaces it with the
    HDF5-backed pnpl dataset, but the __getitem__ contract is identical:
    (x: float32 (C, T), y_primary: int64, y_moses: int64). Both targets exist
    because the competition scores two separate vocabularies; on the sample
    data both carry the same auditory L/R label to exercise the plumbing.
    """

    def __init__(
        self,
        x: np.ndarray,
        y_primary: np.ndarray,
        y_moses: np.ndarray,
        ch_names: list[str],
    ):
        assert x.dtype == np.float32 and x.ndim == 3
        assert y_primary.shape == y_moses.shape == (x.shape[0],)
        self.x = torch.from_numpy(x)
        self.y_primary = torch.from_numpy(y_primary.astype(np.int64))
        self.y_moses = torch.from_numpy(y_moses.astype(np.int64))
        self.ch_names = ch_names

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_primary[idx], self.y_moses[idx]


@dataclass
class PipelineMeta:
    """Everything the model and trainer need to know about the data."""

    n_channels: int
    n_times: int
    n_primary_classes: int
    n_moses_classes: int
    class_names: list[str]
    sfreq: float
    class_to_id: dict[str, int] = field(default_factory=dict)
    n_rejected: int = 0
    mean: Optional[np.ndarray] = None      # per-channel train stats
    std: Optional[np.ndarray] = None
    clip: Optional[float] = None           # pnpl clips standardized data to +/-clip


# ----------------------------------------------------------------------------
# LibriBrain100 memory cache + collate
# ----------------------------------------------------------------------------
def _preload_windows(base, indices: Sequence[int], n_channels: int, n_times: int):
    """Materialise every accepted window as one (N, C, T) float32 tensor.

    The one-time cost is a single pass over the pnpl dataset — exactly the disk
    work the original code paid on EVERY __getitem__. Standardized on load by
    pnpl's mixin (train-partition stats), so the cache holds train units.
    """
    cache = torch.empty(len(indices), n_channels, n_times, dtype=torch.float32)
    for k, j in enumerate(indices):
        try:
            x, _ = base[j]
        except Exception:
            x = torch.zeros(n_channels, n_times, dtype=torch.float32)
        cache[k] = torch.as_tensor(x, dtype=torch.float32).view(
            n_channels, n_times
        )
        if (k + 1) % 5000 == 0:
            print(f"[data_pipeline] preloaded {k + 1}/{len(indices)} windows ...")
    return cache


def _competition_collate(batch):
    """Stack a batch, dropping fully-ignored (-100/-100) windows.

    CrossEntropyLoss(ignore_index=-100) already ignores their loss, but keeping
    them would feed zero rows into the batch-level TangentSpaceWhiten
    covariance. Dropped rows shrink the batch marginally; never empty.
    """
    keep = [
        (x, yp, ym) for (x, yp, ym) in batch
        if not (int(yp) == -100 and int(ym) == -100)
    ]
    if not keep:
        keep = [batch[0]]
    return (
        torch.stack([s[0] for s in keep]),
        torch.stack([s[1] for s in keep]),
        torch.stack([s[2] for s in keep]),
    )


# ----------------------------------------------------------------------------
# MNE sample pipeline
# ----------------------------------------------------------------------------
def load_filtered_raw(data_path: Optional[Path] = None, verbose: str = "error"):
    """Download (first run), load, pick, and filter the MNE sample raw run.

    Picks MEG channels before loading so the ~70 non-MEG channels are never
    read off disk. Filtering is in place on the Raw. NOT resampled here: events
    are stored in the raw's native 600.615 Hz samples, so we epoch first and
    resample the small Epochs object afterwards (see build_epochs).
    """
    if data_path is None:
        # Keep bulky data outside the source tree; override via $MIMIR_DATA_ROOT.
        default_root = _data_root() / "mne"
        default_root.mkdir(parents=True, exist_ok=True)
        data_path = Path(
            mne.datasets.sample.data_path(path=default_root, verbose=verbose)
        )

    def _find(*names: str) -> Path:
        for name in names:
            hits = sorted(data_path.rglob(name))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"none of {names} found under {data_path}")

    raw = mne.io.read_raw_fif(
        _find("sample_audvis_raw.fif"), preload=False, verbose=verbose
    )
    raw.pick("meg")
    bads = [b for b in list(raw.info["bads"]) if b in raw.ch_names]
    if bads:
        raw.drop_channels(bads)
    raw.load_data()
    # Notch the mains line BEFORE the bandpass; 60 Hz sits inside the gamma
    # band that must survive for cross-frequency coupling decoding.
    raw.notch_filter(LINE_FREQ, picks="meg", verbose=verbose)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks="meg", verbose=verbose)
    events = mne.read_events(_find("sample_audvis_raw-eve.fif", "sample_audvis-eve.fif"))
    return raw, events


def build_epochs(raw, events) -> mne.Epochs:
    """Epoch around auditory events, THEN resample to 250 Hz.

    Order matters: events are in the raw's native samples, so moving the data
    first would silently misalign the us. Epochs.resample() then only touches
    the small epoched array; the 100 Hz lowpass already prevents aliasing.
    """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,       # 1 Hz highpass already removed DC drift
        reject=REJECT,
        preload=True,
        verbose="error",
    )
    epochs.resample(TARGET_SFREQ, verbose="error")
    return epochs


def build_mne_sample(
    data_path: Optional[Path] = None,
    batch_size: int = 8,
) -> tuple[MEGEpochDataset, torch.Tensor, PipelineMeta, DataLoader]:
    """MNE sample: download -> filter -> epoch -> split -> normalise -> loader.

    Returns (dataset, adj, meta, loader). The training split is stratified by
    class and per-channel z-score statistics are fit on the TRAIN trials only,
    then applied to both splits and stored in meta (for checkpoint reuse).
    """
    raw, events = load_filtered_raw(data_path)
    epochs = build_epochs(raw, events)
    del raw
    gc.collect()

    x = epochs.get_data(copy=True).astype(np.float32)      # (N, C, T) SI units
    y = epochs.events[:, 2].astype(np.int64)               # codes {1, 2}
    n_rejected = sum(1 for r in epochs.drop_log if r and r != ("IGNORED",))
    ch_names = epochs.ch_names
    n = x.shape[0]

    assert set(np.unique(y)) == {1, 2}, \
        f"a class was fully rejected; surviving codes={sorted(set(y))}"
    y = y - y.min()                                        # {1,2} -> {0,1}
    y_moses = y.copy()

    # Stratified train/val split so every fold keeps both classes.
    rng = np.random.RandomState(seed=MNE_SEED)
    per_class = max(1, int(round(n * MNE_VAL_FRACTION / 2.0)))
    val_idx = np.concatenate([
        rng.choice(np.where(y == 0)[0], per_class, replace=False),
        rng.choice(np.where(y == 1)[0], per_class, replace=False),
    ])
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    train_mask = ~val_mask

    # Per-channel z-scoring, fit on the training trials ONLY (no leakage).
    # Magnetometers (~1e-13 T) and gradiometers (~1e-11 T/m) differ by orders
    # of magnitude; standardisation puts every sensor in std-of-that-sensor
    # units. Dead channels (std == 0) get a scale factor of 1.
    # mu/sd are moments of the train-mask rows; the validation rows never
    # enter these reductions. The same (mu, sd) then rescale BOTH splits below,
    # so normalisation cannot move validation information into training.
    mu = x[train_mask].mean(axis=(0, 2))
    sd = x[train_mask].std(axis=(0, 2))
    sd = np.where(sd > 0.0, sd, 1.0)
    x = (x - mu[None, :, None]) / sd[None, :, None]
    # Leakage proof: the train split is standard-normal by construction with
    # its own moments. The validation split is only standard-normal to the
    # extent that the TRAIN-fit (mu, sd) described it - the val rows never
    # touched the statistics, and were rescaled by the identical values.
    assert abs(float(x[train_mask].mean())) < 1e-2, "train z-mean drifted"
    assert abs(float(x[val_mask].mean())) < 1e-2, "val not in train-fit units"
    assert 0.9 < float(x[train_mask].std()) < 1.1, "train z-std drifted"
    assert 0.9 < float(x[val_mask].std()) < 1.1, "val not in train-fit units"

    meta = PipelineMeta(
        n_channels=x.shape[1],
        n_times=x.shape[2],
        n_primary_classes=len(EVENT_ID),
        n_moses_classes=len(EVENT_ID),
        class_names=list(EVENT_ID.keys()),
        sfreq=TARGET_SFREQ,
        class_to_id={name: i for i, name in enumerate(EVENT_ID)},
        n_rejected=n_rejected,
        mean=mu,
        std=sd,
    )

    train_ds = MEGEpochDataset(x[train_mask], y[train_mask], y_moses[train_mask], ch_names)
    val_ds = MEGEpochDataset(x[val_mask], y[val_mask], y_moses[val_mask], ch_names)
    adj = build_sensor_adjacency(epochs.info)              # same channel set
    del epochs
    gc.collect()

    loader, val_loader = _make_splits(train_ds, val_ds, batch_size)
    print(
        f"[data_pipeline] MNE sample: channels={meta.n_channels}, "
        f"train={len(train_ds)} trials, val={len(val_ds)} trials "
        f"(artifact-rejected {n_rejected}), sfreq={meta.sfreq} Hz, "
        f"bp=[{L_FREQ:.0f}-{H_FREQ:.0f}] Hz + notch {LINE_FREQ:.0f} Hz"
    )
    return train_ds, adj, meta, loader


def _make_splits(train_ds, val_ds, batch_size: int, collate_fn=None):
    """Train (shuffled) + val (sequential) loaders; val exposed on train_ds."""
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=False, pin_memory=False, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, pin_memory=False, collate_fn=collate_fn,
    )
    # Early-stopping handle used by train.py in both modes.
    train_ds.val_loader = val_loader
    train_ds.val_dataset = val_ds
    return loader, val_loader


def build(
    data_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> tuple:
    """Dispatcher: same (dataset, adj, meta, loader) tuple from either source.

    USE_REAL_DATA=False (local smoke) -> MNE loader.
    USE_REAL_DATA=True  (cloud)       -> LibriBrain100 loader (also attaches
    dataset.val_loader/val_dataset for early stopping).

    max_windows caps the vocab-filtered window count (probe runs).
    """
    if USE_REAL_DATA:
        return build_libribrain100(
            data_path=data_path,
            batch_size=batch_size or LIBRI_BATCH_SIZE,
            max_windows=max_windows,
        )
    return build_mne_sample(data_path=data_path, batch_size=batch_size or 8)


# ----------------------------------------------------------------------------
# LibriBrain100 (competition) pipeline
# ----------------------------------------------------------------------------
def _as_mask_stats(stats, n_channels: int) -> np.ndarray:
    """pnpl channel stats -> (C,) float32; dead channels get variance 1.

    pnpl's StandardizationMixin never leaves a channel at zero variance when
    standardizing (it masks), so inverting the transform must reproduce that:
    std of 0 would otherwise blow up at inference. Mirroring the MNE path's
    ``sd = where(sd > 0, sd, 1.0)`` keeps both pipelines symmetric.
    """
    arr = np.asarray(stats, dtype=np.float64)
    if arr.shape != (n_channels,):
        raise ValueError(
            f"channel stats shape {arr.shape} != expected ({n_channels},)"
        )
    out = arr.astype(np.float32).copy()
    if np.any(arr <= 0):
        out = np.where(arr > 0, out, 1.0)
    return out


def build_libribrain100(
    data_path: Optional[Path] = None,
    batch_size: int = LIBRI_BATCH_SIZE,
    max_windows: Optional[int] = None,
) -> tuple:
    """LibriBrain100 (Deep Track, subject 0) -> the 4-tuple.

    Data requirements (first run only): ~20 GB of LibriBrain100 downloads
    into ``<data root>/libribrain100`` (the data root is ``$MIMIR_DATA_ROOT``
    or ``$KAGGLE_DATA_PATH`` if set, else ``<repo>/data``); pnpl must be
    importable.

    The word of every window is matched (normalised + lowercased) against the
    two competition vocabularies; windows whose word is in NEITHER vocabulary
    are discarded (they would give the shared backbone zero gradient). Labels
    for the other vocabulary are -100, so each window teaches whichever
    vocabularies its word belongs to.

    z-scoring is pnpl's StandardizationMixin: channel stats over its TRAIN
    partition, computed once and cached in the H5 metadata - train-only by
    construction.

    Sensor graph: built from the canonical Neuromag 306 positions, name-matched
    to the run's real channel names read from the H5 metadata (pnpl does not
    expose them on the dataset object).
    """
    from pnpl.competition import load_vocabulary
    from pnpl.datasets import LibriBrain100
    from pnpl.datasets.libribrain100.constants import DEFAULT_PREPROCESSING_STR
    from pnpl.tasks import WordClassification

    if data_path is None:
        data_path = _data_root() / "libribrain100"
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Vocabulary order == submission column order == head class index order:
    # load_vocabulary is the single source of that ordering (mirrored by
    # write_submission in pnpl.competition).
    primary_vocab = [normalize_word(w) for w in load_vocabulary("primary")]
    moses_vocab = [normalize_word(w) for w in load_vocabulary("secondary")]
    word_to_primary = {w: i for i, w in enumerate(primary_vocab)}
    word_to_moses = {w: i for i, w in enumerate(moses_vocab)}
    vocab_union = set(primary_vocab) | set(moses_vocab)

    # 1-s word windows (matches the holdout (306, 250)) on the canonical
    # `bads+headpos+sss+notch+bp+ds` preprocessing chain.
    ds = LibriBrain100(
        data_path=data_path,
        task=WordClassification(tmin=LIBRI_TMIN, tmax=LIBRI_TMAX),
        partition="train",
        subjects="deep",
        preprocessing_str=DEFAULT_PREPROCESSING_STR,
        standardize=True,
        download=True,
        preload_h5=False,      # lazy HDF5 page reads keep RAM flat
    )
    print(f"[data_pipeline] LibriBrain100 loaded: {len(ds.samples)} word samples "
          f"(subject deep, {len(ds.records)} runs, {ds.n_channels} ch x "
          f"{ds.n_times} samples)")

    keep = [
        i for i, s in enumerate(ds.samples)
        if (w := normalize_word(s[5])) is not None and w in vocab_union
    ]
    if not keep:
        raise ValueError(
            "no training windows map into the competition vocabularies"
        )
    print(f"[data_pipeline] vocab filter: kept {len(keep)} / {len(ds.samples)} "
          f"windows (dropped {len(ds.samples) - len(keep)} out-of-vocab words)")
    if max_windows is not None:
        keep = keep[:max_windows]
        print(f"[data_pipeline] probe limit: restricted to first "
              f"{max_windows} windows")

    # Seeded 90/10 split on the index space; both wrapper datasets share the
    # one pnpl dataset (one standardization pass + file cache).
    rng = torch.Generator().manual_seed(LIBRI_SEED)
    perm = torch.randperm(len(keep), generator=rng).tolist()
    n_val = max(1, int(round(len(keep) * VAL_FRACTION)))
    train_idx = [keep[p] for p in perm[n_val:]]
    val_idx = [keep[p] for p in perm[:n_val]]

    # RAM cache: preload the pre-standardized windows once, then never touch
    # the H5 files again (the Kaggle epoch bottleneck). Only when it fits.
    preloaded = None
    if LIBRI_CACHE_WINDOWS:
        n_cache = len(ds.samples)
        n_bytes = n_cache * ds.n_channels * ds.n_times * 4
        if n_bytes <= LIBRI_CACHE_MEMORY_LIMIT:
            print(f"[data_pipeline] caching standardized windows ({n_cache} x "
                  f"{ds.n_channels} x {ds.n_times} = {n_bytes / 2**30:.1f} GiB) ...")
            cache_all = _preload_windows(ds, train_idx + val_idx,
                                         ds.n_channels, ds.n_times)
            preloaded = {
                "train": cache_all[:len(train_idx)],
                "val": cache_all[len(train_idx):],
            }
            del cache_all
            gc.collect()
        else:
            print(f"[data_pipeline] CACHE DISABLED: {n_bytes / 2**30:.1f} GiB "
                  f"> limit {LIBRI_CACHE_MEMORY_LIMIT / 2**30:.1f} GiB; falling "
                  f"back to lazy H5 reads (slow on Kaggle CPU)")

    train_ds = MEGCompetitionDataset(
        ds, train_idx, word_to_primary, word_to_moses,
        preloaded=preloaded["train"] if preloaded else None,
    )
    val_ds = MEGCompetitionDataset(
        ds, val_idx, word_to_primary, word_to_moses,
        preloaded=preloaded["val"] if preloaded else None,
    )

    # GEOMETRIC sensor graph: the run's real channel names (from H5 metadata)
    # name-matched onto the canonical Neuromag 306 positions. Fall back to the
    # canonical 306 name order only if the metadata is absent/unreadable.
    try:
        ch_names = _h5_channel_names(ds.get_h5_dataset(ds.run_keys[0]))
    except Exception as exc:
        ch_names = None
        print(f"[data_pipeline] WARNING: could not read H5 channel_names "
              f"({type(exc).__name__}); will fall back to canonical order")
    if ch_names is None or len(ch_names) != ds.n_channels:
        if ch_names is not None:
            print(f"[data_pipeline] WARNING: H5 lists {len(ch_names)} channels, "
                  f"dataset expects {ds.n_channels}; using canonical 306 order")
        ch_names = list(_neuromag306_names_coords()[0])
    adj = build_neuromag306_adjacency(ch_names)
    canon_lookup = _neuromag306_positions()
    n_mapped = sum(1 for n in ch_names if _norm_ch(n) in canon_lookup)
    n_spatial = (int((adj > 0).sum().item()) - adj.shape[0]) // 2
    print(f"[data_pipeline] adjacency: geometric Neuromag 306 kNN "
          f"({n_spatial} spatial edges, {n_mapped}/{len(ch_names)} channels "
          f"name-matched to canonical positions, {N_NEIGHBORS}-nn "
          f"median-sigma kernel incl. concentric triplets)")

    meta = PipelineMeta(
        n_channels=ds.n_channels,                # 306
        n_times=ds.n_times,                      # 250
        n_primary_classes=len(primary_vocab),
        n_moses_classes=len(moses_vocab),
        class_names=primary_vocab,               # head order == submission cols
        sfreq=250.0,
        class_to_id=word_to_primary,
        n_rejected=0,                            # pnpl pre-filtered
        mean=_as_mask_stats(ds.channel_means, ds.n_channels),
        std=_as_mask_stats(ds.channel_stds, ds.n_channels),
        clip=LIBRI_CLIP_BOUNDARY,
    )

    loader, val_loader = _make_splits(
        train_ds, val_ds, batch_size, collate_fn=_competition_collate
    )
    print(
        f"[data_pipeline] LibriBrain100 ready: train {len(train_ds)} windows, "
        f"val {len(val_ds)} windows, batch {batch_size}, "
        f"window=[{LIBRI_TMIN}, {LIBRI_TMAX}] s, "
        f"heads=({meta.n_primary_classes}, {meta.n_moses_classes})-way"
    )
    return train_ds, adj, meta, loader


# ----------------------------------------------------------------------------
# Self-test: batch contract, augmentation gating, full sweep
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dataset, adj, meta, loader = build(batch_size=8)
    xb, yp, ym = next(iter(loader))
    print("\n=== BATCH CONTRACT (what mimir_model.forward() receives) ===")
    print(f"x.shape={tuple(xb.shape)}  dtype={xb.dtype}  "
          f"(Batch, {meta.n_channels} channels, {meta.n_times} samples)")
    print(f"y_primary.shape={tuple(yp.shape)}  labels={yp.tolist()}")
    print(f"y_moses.shape={tuple(ym.shape)}  labels={ym.tolist()}")
    print(f"adj={tuple(adj.shape)} {adj.dtype} "
          f"symmetric={bool(torch.allclose(adj, adj.T, atol=1e-5))} "
          f"rowsum~1={float(adj.sum(1).mean()):.3f}")
    print(f"x stats: mean={xb.mean():.3f} std={xb.std():.3f} "
          f"heads=(primary {meta.n_primary_classes}-way, "
          f"moses {meta.n_moses_classes}-way)")
    print(f"val split: {len(dataset.val_dataset)} / "
          f"{len(dataset) + len(dataset.val_dataset)} trials")

    # Augmentation contract: transforms in train mode, exact identity in eval.
    aug = TrialAugmentation()
    aug.train()
    x_aug = aug(xb)
    changed = not torch.equal(x_aug, xb)
    dropped = float((x_aug == 0).all(dim=-1).float().mean())
    aug.eval()
    identity = torch.equal(aug(xb), xb)
    print(f"TrialAugmentation: train-mode changed={changed} "
          f"(dropped-channel fraction={dropped:.3f}, target ~0.10), "
          f"eval-mode identity={identity}")
    assert changed and identity, "augmentation train/eval gating broken"

    n_total = sum(1 for _ in loader)
    print(f"[data_pipeline] swept train loader: {n_total} batches OK.")


if __name__ == "__main__":
    main()