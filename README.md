# MIMIR - MEG Neural Decoding for PNPL 2026

MIMIR is a compact dual-head MEG decoder for the PNPL 2026 Deep Track (LibriBrain100 word classification, 306-channel MEGIN TRIUX Neo). It pairs a **spatial GCN** that mixes each sensor with its physical neighbours on the geometric k-NN helmet graph (built from canonical Neuromag 306 coil positions) with a **temporal TCN** whose exponentially dilated residual convolutions cover the full 1 s word window (receptive field 255 samples @ 250 Hz); the shared backbone terminates in two independent 50-way readouts (primary vocabulary + Moses-50), trained with summed cross-entropy and the competition's `-100` ignore-index convention.

## Repository layout

| File | Purpose |
|------|---------|
| `mimir_model.py` | Architecture (whiten - GCN - TCN - two heads). Ablation flags `USE_GCN`, `USE_TCN` at the top of the file. |
| `data_pipeline.py` | Dual loader (MNE sample locally, LibriBrain100 on Kaggle) + geometric sensor adjacency. Per-channel z-scoring is fit on the **train** split only (no leakage); optional `CACHE_WINDOWS=1` RAM preload that removes the per-step H5 reads that dominated Kaggle epoch time. |
| `train.py` | Training loop; reads `USE_GCN` / `USE_TCN` from `mimir_model.py`. Logs **train/val Top-10 recall** (plain + balanced) and anneals the LR cosinely. |
| `submit.py` | Rebuilds the model from a checkpoint, applies the exact training transform (`(x - mean)/std` then clip) from the checkpoint meta, and writes the submission CSV. Refuses to write a data-blind (bias-only) CSV. |

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
python -u train.py          # one epoch on the MNE sample data
python -u data_pipeline.py  # optional: batch-contract self-test
python -u mimir_model.py    # optional: shape/RF/ablation self-test
```

### Kaggle run (LibriBrain100 deep track)

1. Set `USE_REAL_DATA = True` at the top of `data_pipeline.py` (a single switch; `train.py` reads it).
2. Point the data root at the head of the run. By default data lives in `<repo>/data`; override with `MIMIR_DATA_ROOT` (or `KAGGLE_DATA_PATH`) - e.g. an exported Kaggle dataset mount such as `/kaggle/input/libribrain100`.
3. Train. The first run downloads ~20 GB of LibriBrain100 into the data root and caches pnpl standardization stats in the H5 metadata. The standardized windows are then **RAM-preloaded once** (`CACHE_WINDOWS=1`, the default) so every epoch reads from memory instead of the H5 files - this is what makes multiple epochs feasible on Kaggle CPU.

```bash
export CACHE_WINDOWS=1                          # default; RAM-cache standardized windows
python -u train.py --epochs 8 --patience 4      # probe: watch train_Top10 / val_bal_Top10
python -u train.py --epochs 20 --patience 5     # full run (cosine LR) -> best checkpoint
python -u submit.py                             # writes submission_deep.csv (exact transform)
```

The checkpoint stores the training `mean`/`std`/`clip` so `submit.py` reproduces the 
exact inference transform on the raw holdout signal (no per-window z-scoring, no 
`standardize=` kwarg - that path produced a data-blind bias-only CSV with a balanced 
Top-10 of 0.200, exactly chance).

> Diagnostics: `train_Top10` (plain, first 800 batches) and `val_bal_Top10` (balanced,
> chance = 0.020 per class average). If `train_Top10` climbs well past 0.2 while
> `val_bal_Top10` follows, the model is genuinely decoding; extend epochs. If both stay
> ~0.2 the run is still fitting the word-frequency prior - keep training.

### Kaggle one-shot (recommended)

`kaggle_run.py` measures the live per-window cost once, then picks the largest
window count + epoch count that fit a wall-clock budget and drives the whole
pipeline (train -> exact-transform inference -> diversity-guarded CSV):

```bash
python -u kaggle_run.py --budget-hours 5 --limit 25000 --epochs 12 --patience 4
```

It is what the notebook's single "master cell" calls. Flags: `--budget-hours`
(total wall-clock, leaves ~0.4 h headroom), `--limit` (hard cap on training
windows; the deep subject has far too many for a CPU session even with the RAM
cache), `--epochs`/`--patience`, `--batch`, `--lr`, `--submission` (output
path), `--no-submit`, and `--smoke` (local MNE wiring test, no pnpl).

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
- `data/`, `checkpoints/`, `*.pt`, `*.csv` and `__pycache__` are git-ignored (see `.gitignore`).