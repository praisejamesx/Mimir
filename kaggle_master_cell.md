export MIMIR_DATA_ROOT=/kaggle/input/libribrain100
export LIBRI_CACHE_MEMORY_LIMIT_BYTES=50000000000
export CACHE_WINDOWS=1

python3 -u kaggle_run.py \
  --budget-hours 12 \
  --device cuda \
  --epochs 15 \
  --seed 42

python3 submit.py \
  --checkpoint /home/praisejamesx/Documents/Documents/Research/pnpl-2026/Mimir/checkpoints/mimir_libri_best.pt \
  --prior-tau 0.5 \
  --output /home/praisejamesx/submission_deep.csv
