# Kaggle master cell - PNPL 2026 Deep Track (MIMIR)

Copy the single code cell below into a new Kaggle notebook and run it.

**Two ways to bring the code in:**

1. **Upload (recommended):** in the notebook, drag
   `data_pipeline.py`, `mimir_model.py`, `train.py`, `submit.py`,
   `kaggle_run.py` into the file panel so they end up under
   `/kaggle/working/Mimir/`. The cell detects them, skips git entirely, and
   runs from your uploaded copy.
2. **Repository clone:** add a Kaggle Secret named `GH_TOKEN` (a GitHub
   Personal Access Token with `repo` scope) and the cell will clone the
   repository with it.

**Accelerator:** set the notebook to `GPU P100` (optionally `GPU T4 x2`; the
pipeline auto-detects CUDA). On CPU-only notebooks the *same* cell works
unchanged, just far slower.

**How to run (two runs, one cell):**
1. Paste the code below into a notebook set to GPU P100. Run it once.
2. The first run is a **3-minute check** that the pipeline can learn on real data
   (it trains on 256 windows, no submission, takes ~3 min).
3. If the printed result shows `primary top1` printed above 0.9, flip the switch
   `GATE_ONLY = False` at the top of the cell and press Run again. That is the
   full training run (hours), which writes `submission_deep.csv`.
4. Download it and submit on the competition page.

```python
import os, subprocess, sys, glob

# ---- 0. ONE switch: True = 3-min learnability check; False = full run --------
GATE_ONLY = True

# ---- 1. Dependencies ---------------------------------------------------------
for pkg in ("mne", "pnpl"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "pnpl", "mne"], check=True)

# ---- 2. Source (uploaded files take priority; git only as a fallback) --------
REPO = "/kaggle/working/Mimir"
have_files = os.path.isdir(REPO) and os.path.exists(os.path.join(REPO, "kaggle_run.py"))
if have_files:
    print("[master cell] using uploaded Mimir files in /kaggle/working/Mimir")
    for name in ("data_pipeline.py", "mimir_model.py", "train.py",
                 "submit.py", "kaggle_run.py"):
        src = f"/kaggle/working/{name}"
        if os.path.exists(src):
            subprocess.run(["cp", src, os.path.join(REPO, name)], check=True)
else:
    try:
        subprocess.run(["git", "clone",
                        "https://github.com/praisejamesx/Mimir.git", REPO],
                       check=True)
    except subprocess.CalledProcessError:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GH_TOKEN")
        except Exception as exc:
            raise SystemExit(
                "no Mimir files uploaded and no GH_TOKEN secret set - see the "
                "'Two ways to bring the code in' notes") from exc
        subprocess.run(["git", "clone",
                        f"https://{token}@github.com/praisejamesx/Mimir.git",
                        REPO], check=True)
    subprocess.run(["git", "-C", REPO, "config", "safe.directory", REPO], check=True)

# ---- 3. Data root -------------------------------------------------------------
comp_inputs = glob.glob("/kaggle/input/**/libribrain100", recursive=True)
if comp_inputs and os.listdir(comp_inputs[0]):
    os.environ["MIMIR_DATA_ROOT"] = os.path.dirname(comp_inputs[0])
else:
    os.environ["MIMIR_DATA_ROOT"] = "/kaggle/tmp"
os.environ["CACHE_WINDOWS"] = "1"
os.environ["LIBRI_CACHE_MEMORY_LIMIT_BYTES"] = "24000000000"   # 24 GiB cache

# ---- 4. Run: gate (3 min) or full training (hours) ----------------------------
cmd = ["kaggle_run.py", "--overfit", "256", "--no-submit"] if GATE_ONLY else [
    "kaggle_run.py",
    "--budget-hours", "6.5",
    "--limit", "60000",
    "--epochs", "90",
    "--patience", "8",
    "--pool", "attn",
    "--loss", "ce",
    "--class-weights", "none",
    "--balanced-sampling",
    "--dropout", "0.25",
    "--f1", "16",
    "--f2", "128",
    "--objective", "supcon",
    "--submission", "/kaggle/working/submission_deep.csv"]
rc = subprocess.run([sys.executable, "-u"] + cmd, cwd=REPO, check=False).returncode
if rc != 0:
    raise SystemExit(f"kaggle_run.py exited with rc={rc}")

if GATE_ONLY:
    print("\n[gate] PASS if 'primary top1' printed near 1.0. Then set")
    print("[gate] GATE_ONLY = False at the top of this cell and Run again")
    print("[gate] to do the full training run (hours).")
else:
    print("\n[master cell] submission ready - download submission_deep.csv from")
    print("[master cell] the notebook's right-hand file panel / Output tab and")
    print("[master cell] submit it on the competition page.")
```

---

## Audit cell (label-mapping check) — fully self-contained

Paste into ANY notebook (fresh or not), press Run once, walk away. Installs
its own packages, pulls the repo code, uses mounted data if present else
downloads it (~20GB, unattended). Takes ~2 min with mounted data, longer if
downloading. Paste the whole output back.

```python
import os, subprocess, sys, glob
from collections import Counter

# ---- 1. Packages ----
for pkg in ("mne", "pnpl"):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[audit] installing {pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "pnpl", "mne"], check=True)

# ---- 2. Repo code (same word-reading training uses) ----
REPO = "/kaggle/working/Mimir"
if not (os.path.isdir(REPO) and os.path.exists(os.path.join(REPO, "data_pipeline.py"))):
    print("[audit] cloning repo ...", flush=True)
    subprocess.run(["git", "clone",
                    "https://github.com/praisejamesx/Mimir.git", REPO], check=True)
sys.path.insert(0, REPO)
from data_pipeline import normalize_word, _h5_channel_names

# ---- 3. Data: mounted if present, else download (unattended) ----
roots = [r for r in glob.glob("/kaggle/input/**/libribrain100", recursive=True)
         if os.path.isdir(r) and os.listdir(r)]
if roots:
    DATA = roots[0]
    print("[audit] using mounted data:", DATA, flush=True)
else:
    DATA = "/kaggle/tmp/libribrain100"
    print("[audit] no mounted data; downloading to", DATA, flush=True)
    print("[audit] this is ~20GB and takes a while; nothing needed from you.", flush=True)

# ---- 4. Dataset (downloads on first use if DATA is empty) ----
from pnpl.datasets import LibriBrain100
from pnpl.datasets.libribrain100.constants import DEFAULT_PREPROCESSING_STR
from pnpl.tasks import WordClassification
ds = LibriBrain100(data_path=DATA, task=WordClassification(tmin=0.0, tmax=1.0),
                   partition="train", subjects="deep",
                   preprocessing_str=DEFAULT_PREPROCESSING_STR,
                   standardize=True, download=True, preload_h5=False)
print("[audit] loaded:", len(ds.samples), "samples,", len(ds.records),
      "runs,", ds.n_channels, "ch x", ds.n_times, "samples", flush=True)

# ---- 5. Schema: what is field #6? ----
s0 = ds.samples[0]
print("record type:", type(s0), "n fields:", len(s0), flush=True)
for i, f in enumerate(list(s0)[:8]):
    print(f"  field[{i}] =", repr(f)[:120], flush=True)
for s in ds.samples[:10]:
    print("  field[5] =", repr(s[5])[:40], "->", repr(normalize_word(s[5])), flush=True)

# ---- 6. Vocabularies ----
from pnpl.competition import load_vocabulary
pv, mv = list(load_vocabulary("primary")), list(load_vocabulary("secondary"))
print("primary:", len(pv), list(pv)[:5], flush=True)
print("moses:", len(mv), list(mv)[:5], flush=True)

# ---- 7. Coverage over first 50k samples ----
w2p, w2m, vu = set(pv), set(mv), set(pv) | set(mv)
cnt, npp, nmm, neither = Counter(), 0, 0, 0
for s in ds.samples[:50000]:
    w = normalize_word(s[5])
    if w is None or w not in vu:
        neither += 1
        continue
    cnt[w] += 1
    npp += w in w2p
    nmm += w in w2m
print("top-10 words:", cnt.most_common(10), flush=True)
print(f"valid primary: {npp}, valid moses: {nmm}, dropped: {neither}", flush=True)

# ---- 8. Channel names readable? (training log showed a fallback warning) ----
try:
    names = _h5_channel_names(ds.get_h5_dataset(ds.run_keys[0]))
    print(f"H5 channel_names: {len(names)} read, e.g. {names[:3]}", flush=True)
except Exception as exc:
    print(f"H5 channel_names UNREADABLE ({type(exc).__name__}: {exc})", flush=True)
print("[audit] DONE. Paste everything above back.", flush=True)
```
