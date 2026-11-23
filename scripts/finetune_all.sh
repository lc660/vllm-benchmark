#!/usr/bin/env bash
# Run the fine-tuning experiment matrix, one run per GPU at a time.
#
#   bash scripts/finetune_all.sh                  # every experiment in EXPERIMENTS on all visible GPUs
#   GPUS="0 1" bash scripts/finetune_all.sh       # only these GPUs
#   EXPERIMENTS="shared_lora lora_depth" bash scripts/finetune_all.sh
#
# Names are finetune/configs/<name>.yaml, or lora_<task> for the per-task LoRA ablation (shared_lora's recipe on
# one task). Each run first fills the encode cache (shared between runs: identical samples are encoded once), then
# trains, then writes eval configs for its best checkpoint. Logs: runs/<name>/log.txt. Rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=${EXPERIMENTS:-"shared_lora lora_segmentation lora_depth lora_normals lora_pose lora_attn dora lokr partial_ft onestep_lora"}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
PY=${PY:-"uv run --extra finetune python"}

run_one() {  # $1 = experiment, $2 = GPU index
  local name=$1 gpu=$2 config args=()
  case $name in
    lora_segmentation|lora_depth|lora_normals|lora_pose)
      config=shared_lora; args=(--set "name=$name" --set "data.tasks=[${name#lora_}]") ;;
    *) config=$name ;;
  esac
  mkdir -p "runs/$name"
  {
    echo "== $name on GPU $gpu, $(date)"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m finetune.encode "$config" "${args[@]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY -m finetune.train "$config" "${args[@]}"
    $PY -m finetune.make_eval_configs "runs/$name/best" --tag "$name-best"
    echo "== $name finished, $(date)"
  } >>"runs/$name/log.txt" 2>&1
}

queue=($EXPERIMENTS)
declare -A busy=()
while ((${#queue[@]})) || ((${#busy[@]})); do
  for gpu in $GPUS; do
    pid=${busy[$gpu]:-}
    if [[ -n $pid ]] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || echo "run on GPU $gpu failed; see its runs/<name>/log.txt"
      unset "busy[$gpu]"; pid=
    fi
    if [[ -z $pid ]] && ((${#queue[@]})); then
      name=${queue[0]}; queue=("${queue[@]:1}")
      echo "start $name on GPU $gpu (log: runs/$name/log.txt)"
      run_one "$name" "$gpu" & busy[$gpu]=$!
    fi
  done
  sleep 20
done
echo "all runs finished; eval configs are in configs/finetuned/"
