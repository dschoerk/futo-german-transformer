#!/usr/bin/env bash
# After training: snapshot final weights, export + validate GGUF, evaluate, plot.
#   setsid nohup scripts/finalize.sh > logs/finalize.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-v1.0}
until grep -q "training finished" logs/train.log; do
  if grep -q -E "Traceback|Error" logs/train.log; then echo "training failed"; exit 1; fi
  sleep 30
done
echo "$(date +%T) training finished; finalizing $TAG"
rm -rf models/final && cp -r models/ckpt/hf_latest models/final
uv run scripts/05_export_gguf.py --model-dir models/final --out "models/gguf/de_keyboard_${TAG}_f16.gguf" --name "German ${TAG}"
uv run scripts/check_gguf.py "models/gguf/de_keyboard_${TAG}_q6_k.gguf" --model-dir models/final
uv run scripts/06_eval.py --model-dir models/final -n 500 --seed 1 --out "results/eval_${TAG}_seed1.json"
uv run scripts/06_eval.py --model-dir models/final -n 2400 --seed 2 --out "results/eval_${TAG}_seed2_n2000.json"
uv run scripts/plot_training.py
echo "$(date +%T) FINALIZE_DONE"
