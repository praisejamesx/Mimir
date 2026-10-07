# MIMIR - MEG Neural Decoding for PNPL 2026

MIMIR is a compact dual-head MEG decoder for the PNPL 2026 Deep Track (LibriBrain100 word classification, 306-channel MEGIN TRIUX Neo). It pairs a **spatial GCN** that mixes each sensor with its physical neighbours on the geometric k-NN helmet graph (built from canonical Neuromag 306 coil positions) with a **temporal TCN** whose exponentially dilated residual convolutions cover the full 1 s word window (receptive field 255 samples @ 250 Hz); the shared backbone terminates in two independent 50-way readouts (primary vocabulary + Moses-50), trained with summed cross-entropy and the competition's `-100` ignore-index convention.

## Repository layout

| File | Purpose |
|------|---------|
| `mimir_model.py` | Architecture (whiten - GCN - TCN - two heads). Ablation flags `USE_GCN`, `USE_TCN` at the top of the file. |
| `data_pipeline.py` | Dual loader (MNE sample locally, LibriBrain100 on Kaggle) + geometric sensor adjacency. Per-channel z-scoring is fit on the **train** split only (no leakage). |
| `train.py` | Training loop; reads `USE_GCN` / `USE_TCN` from `mimir_model.py`. |
| `submit.py` | Rebuilds the model from a checkpoint and writes the submission CSV (full softmax over both heads). |

## Setup

```bash
conda create -n mimir python=3.10 -y && conda activate mimir
pip install torch numpy mne>=1.12
pip install "pnpl @ git+https://github.com/neural-processing-lab/pnpl"  # competition SDK; preinstalled on Kaggle
```

`pnpl` is only needed for the LibriBrain100 paths (`train.py` cloud mode,
`submit.py`); the local smoke test runs on `mne` + `torch` alone.

## How to run

### Local smoke test

Verifies the whole contract (tensor shapes, dual heads, augmentation gating, gradient flow, RAM) with the MNE `sample` auditory dataset, about 2 minutes on the current machine:

```bash
conda activate mimir
cd mimir
python -u train.py          # one epoch on the MNE sample data
python -u data_pipeline.py  # optional: batch-contract self-test
python -u mimir_model.py    # optional: shape/RF/ablation self-test
```

### Kaggle run (LibriBrain100 deep track)

1. Set `USE_REAL_DATA = True` at the top of `data_pipeline.py` (a single switch; `train.py` reads it).
2. Point the data root at the head of the run. By default data lives in `<repo>/data`; override with `MIMIR_DATA_ROOT` (or `KAGGLE_DATA_PATH`) - e.g. an exported Kaggle dataset mount such as `/kaggle/input/libribrain100`.
3. Train. The first run downloads ~20 GB of LibriBrain100 into the data root and caches pnpl standardization stats in the H5 metadata. Up to 20 epochs with a seeded 90/10 validation split, early stopping (patience 3), best checkpoint at `mimir/checkpoints/mimir_libri_best.pt`.

```bash
cd mimir
python -u train.py           # cloud mode
python -u submit.py          # writes submission_deep.csv (full softmax, both heads)
```

### Ablation runs

Toggle the flags at the top of `mimir_model.py` (or pass `use_gcn=...` / `use_tcn=...` to the `Mimir` constructor) and re-run `train.py`; the flags are recorded in the checkpoint so `submit.py` reproduces the exact architecture. Typical arms: full model, no spatial graph (`use_gcn=False`), no temporal stack (`use_tcn=False`), no whitening.

## Ablation study

| Ablation | Params | Train epochs | Val loss | Primary Top-10 | Moses acc | Notes |
|----------|-------:|-------------:|---------:|---------------:|----------:|-------|
| Full (GCN + TCN + whiten) | 107.5K | | | | | baseline |
| No spatial graph (`use_gcn=False`) | | | | | | no geometry prior |
| No temporal stack (`use_tcn=False`) | | | | | | envelope only |
| No whitening (`use_whiten=False`) | | | | | | raw sensor scale |

> Fill in results after each run. Primary Top-10 balanced accuracy is the scored metric; Moses-50 is the secondary block.

## Notes

- **No leakage**: z-score statistics are fit on train trials/split only (MNE path: split-then-fit; LibriBrain path: pnpl standardizes over its train partition and caches the stats).
- **Sensor graph**: LibriBrain runs carry their real channel names in the H5 metadata; `data_pipeline` name-matches them onto MNE's canonical Neuromag 306 positions (concentric mag + 2 planar-grad triplets handled by location-deduplication in the k-NN). Fallback to the canonical 306 order if metadata is unreadable.
- **num_workers=0** throughout: the pnpl dataset holds open HDF5 handles and a per-sample cache; worker processes would fork them for no throughput gain on a 4-vCPU box.
- `data/`, `mimir/checkpoints/`, `*.pt`, `*.csv` and `__pycache__` are git-ignored (see `.gitignore`).