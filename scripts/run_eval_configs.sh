#!/usr/bin/env bash
# Run every eval config in a folder written by `python -m finetune.make_eval_configs`.
#   bash scripts/run_eval_configs.sh configs/finetuned/shared_lora-best
#   GPU=1 bash scripts/run_eval_configs.sh configs/finetuned/dora-best
# Runs resume (outputs are keyed by generation_version), so rerunning skips finished samples.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:?usage: run_eval_configs.sh <folder of eval configs>}
for config in "$dir"/*.yaml; do
  echo "== $config"
  # DLAMI: its CUDA libraries break cuDNN, hence env -u LD_LIBRARY_PATH (as in finetune_data/README.md 6.1)
  CUDA_VISIBLE_DEVICES=${GPU:-0} env -u LD_LIBRARY_PATH uv run --extra finetune visionbench run --config "$config"  # peft: adapter loading
done
