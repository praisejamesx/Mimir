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

**Before the full run, prove the pipeline learns** (2 min, no submission):
run `kaggle_run.py --overfit 256 --no-submit` and check that `primary top1`
climbs toward 1.0. If it does not, stop — a full run cannot help and the bug is
in data/label alignment or optimisation.

**Expected (GPU P100):** measure (~1 min) -> plan -> train (20+ epochs on ~78k
windows) -> inference on the holdout -> `submission_deep.csv`. A sanity guard
refuses to write a bias-only CSV if training went wrong.

```python
import os, subprocess, sys, glob

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
    # always refresh to the latest uploaded copy when the session is reused:
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
        # restricted/private repo -> clone with the GH_TOKEN Kaggle secret
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GH_TOKEN")
        except Exception as exc:
            raise SystemExit(
                "no Mimir files uploaded and no GH_TOKEN secret set - see the "
                "'Two ways to bring the code in' notes") from exc
        proc = subprocess.run(["git", "clone",
                               f"https://{token}@github.com/praisejamesx/Mimir.git",
                               REPO], check=True)
        print("[master cell] private-repo clone OK (GH_TOKEN)")
    subprocess.run(["git", "-C", REPO, "config", "safe.directory", REPO], check=True)

# ---- 3. Data root -------------------------------------------------------------
# Use the competition input mount if it carries libribrain100, else download
# the ~20 GB into the writable working area (network is available on Kaggle).
comp_inputs = glob.glob("/kaggle/input/**/libribrain100", recursive=True)
if comp_inputs and os.listdir(comp_inputs[0]):
    os.environ["MIMIR_DATA_ROOT"] = os.path.dirname(comp_inputs[0])
else:
    os.environ["MIMIR_DATA_ROOT"] = "/kaggle/tmp"
os.environ["CACHE_WINDOWS"] = "1"

# ---- 4. GPU run (auto-sizes windows x epochs to fit the wall-clock budget) ----
os.environ["LIBRI_CACHE_MEMORY_LIMIT_BYTES"] = "24000000000"   # 24 GiB cache
rc = subprocess.run(
    [sys.executable, "-u", "kaggle_run.py",
     "--budget-hours", "6.5", "--limit", "80000",
     "--epochs", "20", "--patience", "5",
     "--pool", "attn", "--loss", "focal", "--class-weights", "sqrt-inv",
     "--submission", "/kaggle/working/submission_deep.csv"],
    cwd=REPO, check=False,
).returncode
if rc != 0:
    raise SystemExit(f"kaggle_run.py exited with {rc}")

print("\n[master cell] submission ready - download submission_deep.csv from")
print("[master cell] the notebook's right-hand file panel / Output tab and")
print("[master cell] submit it on the competition page.")
```

CPU-only / extra-spare-time variants (change one line, everything else the same):

- `--budget-hours 6.5` -> `--budget-hours 11` (a full CPU session can train far
  more, but stays under the 12 h notebook cap).
- `--limit 80000` / `--epochs 20` are only ceilings; the auto-sizer picks the
  largest (windows, epochs) that fit whatever device and budget you gave it.
- Lower `LIBRI_CACHE_MEMORY_LIMIT_BYTES` to `14000000000` (14 GiB, the default)
  if your notebook is short on RAM.