#!/usr/bin/env bash
# Sweep MultiBLiMP across every subdirectory of a checkpoint dir.
#
# Usage:
#   evaluation/multiblimp.sh \
#       --checkpoints-dir ${RUN_DIR}/peft \
#       --results-dir     ${RESULTS_DIR}/multiblimp \
#       --tokenizer       allenai/OLMoE-1B-7B-0924-Instruct
#
# Or evaluate a single checkpoint:
#   evaluation/multiblimp.sh \
#       --checkpoint  /path/to/single/ckpt \
#       --results-dir ${RESULTS_DIR}/multiblimp \
#       --tokenizer   allenai/OLMoE-1B-7B-0924-Instruct
#
# Env-var defaults (CLI flags override):
#   CHECKPOINTS_DIR   sweep root
#   RESULTS_DIR       eval output root
#   TOKENIZER         HF model id or local tokenizer dir
#   TASKS             comma-separated lm-eval task list (defaults to all 14 paper langs)
#   PATTERN           glob inside CHECKPOINTS_DIR (default: '*')
#   DEVICE            cuda:0 / cpu (default cuda:0)
#   NUM_FEWSHOT       (default 0)

set -euo pipefail

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
CHECKPOINT=""
RESULTS_DIR="${RESULTS_DIR:-}"
TOKENIZER="${TOKENIZER:-allenai/OLMoE-1B-7B-0924-Instruct}"
TASKS="${TASKS:-multiblimp_eng,multiblimp_rus,multiblimp_hin,multiblimp_arb,multiblimp_ces,multiblimp_spa,multiblimp_fin,multiblimp_nld,multiblimp_ukr,multiblimp_mar,multiblimp_urd,multiblimp_cat,multiblimp_est,multiblimp_slk}"
PATTERN="${PATTERN:-*}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
    --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
    --tokenizer)       TOKENIZER="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --pattern)         PATTERN="$2"; shift 2 ;;
    --device)          DEVICE="$2"; shift 2 ;;
    --num-fewshot)     NUM_FEWSHOT="$2"; shift 2 ;;
    -h|--help)         sed -n '1,24p' "$0" | sed 's/^# *//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$RESULTS_DIR" ]]; then
  echo "ERROR: --results-dir (or RESULTS_DIR env var) is required" >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"

run_one() {
  local ckpt=$1
  local name=$2
  local out="$RESULTS_DIR/$name"
  if [[ -d "$out" ]] && find "$out" -name 'results_*.json' | grep -q .; then
    echo "  skip $name (already evaluated)"
    return 0
  fi
  echo "  eval $name"
  lm_eval --model hf \
    --model_args "pretrained=$ckpt,tokenizer=$TOKENIZER,dtype=auto" \
    --tasks "$TASKS" \
    --batch_size auto \
    --device "$DEVICE" \
    --num_fewshot "$NUM_FEWSHOT" \
    --output_path "$out"
}

if [[ -n "$CHECKPOINT" ]]; then
  run_one "$CHECKPOINT" "$(basename "$CHECKPOINT")"
elif [[ -n "$CHECKPOINTS_DIR" ]]; then
  shopt -s nullglob
  for ckpt in "$CHECKPOINTS_DIR"/$PATTERN; do
    [[ -d "$ckpt" ]] || continue
    run_one "$ckpt" "$(basename "$ckpt")"
  done
else
  echo "ERROR: provide --checkpoint or --checkpoints-dir" >&2
  exit 1
fi
