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
   full training run (hours), which writes `submission_deep.csv` and calls
   `submit.py` with `--prior-tau 0.5`.
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
    "--limit", "20000",
    "--epochs", "90",
    "--patience", "18",
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
