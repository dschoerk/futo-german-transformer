#!/usr/bin/env bash
# Build the per-epoch training shards (CPU, thermally guarded) and then train (GPU, self-throttling).
# Run detached so an editor/terminal crash does not kill it:
#   setsid nohup scripts/run_all.sh > logs/run_all.log 2>&1 &
# Re-running is safe: existing shards are skipped and training resumes from models/ckpt/last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/tok
EPOCHS=${EPOCHS:-3}
WORKERS=${WORKERS:-10}

for ((s = 0; s < EPOCHS; s++)); do
  if [ -f "data/tok/train_s$s.npy" ]; then echo "shard $s exists"; continue; fi
  echo "$(date +%T) building shard $s"
  nice -n 10 uv run scripts/03_build_dataset.py --workers "$WORKERS" --seed "$s" > "logs/build_dataset_s$s.log" 2>&1 &
  build=$!
  sleep 5
  scripts/thermal_guard.sh "scripts/03_build_dataset.py" 85 logs/thermal.log 75 &
  guard=$!
  wait "$build"
  kill "$guard" 2>/dev/null || true
  grep -E "seqs" "logs/build_dataset_s$s.log"
done

echo "$(date +%T) training"
uv run scripts/04_train.py --epochs "$EPOCHS" "$@" > logs/train.log 2>&1
echo "$(date +%T) done"
