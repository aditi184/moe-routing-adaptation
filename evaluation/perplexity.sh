#!/usr/bin/env bash
# Thin wrapper around adaptation/eval_peft.py for per-language validation perplexity.
#
# Usage:
#   evaluation/perplexity.sh \
#       --checkpoint   ${RUN_DIR}/peft/peft_stage2_catalan/ssft_k5_lr4e4 \
#       --validation   "${DATA_DIR}/evaluation/tokenized/en/part-0-00000.npy,..." \
#       --tokenizer    allenai/OLMoE-1B-7B-0924-Instruct \
#       --max-length   2048 --num-samples 250 --batch-size 4
#
# Or sweep a checkpoint dir (one perplexity report per subdir):
#   evaluation/perplexity.sh \
#       --checkpoints-dir ${RUN_DIR}/peft/peft_stage2_catalan \
#       --validation "${DATA_DIR}/evaluation/tokenized/ca/part-0-00000.npy" \
#       --tokenizer  allenai/OLMoE-1B-7B-0924-Instruct
#
# Calls into adaptation/eval_peft.py with the args you provide.

set -euo pipefail

CHECKPOINT=""
CHECKPOINTS_DIR=""
VALIDATION=""
TOKENIZER="${TOKENIZER:-allenai/OLMoE-1B-7B-0924-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_SAMPLES="${NUM_SAMPLES:-250}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/../adaptation/eval_peft.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
    --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --validation)      VALIDATION="$2"; shift 2 ;;
    --tokenizer)       TOKENIZER="$2"; shift 2 ;;
    --max-length)      MAX_LENGTH="$2"; shift 2 ;;
    --num-samples)     NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)         sed -n '1,18p' "$0" | sed 's/^# *//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$VALIDATION" ]]; then
  echo "ERROR: --validation is required (comma-separated .npy paths)" >&2
  exit 1
fi

run_one() {
  local ckpt=$1
  echo "=== $ckpt ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$EVAL_PY" \
    --checkpoint_path "$ckpt" \
    --validation_path "$VALIDATION" \
    --tokenizer_path  "$TOKENIZER" \
    --max_length      "$MAX_LENGTH" \
    --num_samples     "$NUM_SAMPLES" \
    --batch_size      "$BATCH_SIZE"
}

if [[ -n "$CHECKPOINT" ]]; then
  run_one "$CHECKPOINT"
elif [[ -n "$CHECKPOINTS_DIR" ]]; then
  shopt -s nullglob
  for ckpt in "$CHECKPOINTS_DIR"/*; do
    [[ -d "$ckpt" ]] || continue
    run_one "$ckpt"
  done
else
  echo "ERROR: pass --checkpoint or --checkpoints-dir" >&2
  exit 1
fi
