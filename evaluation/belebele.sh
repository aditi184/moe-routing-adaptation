#!/usr/bin/env bash
# Sweep Belebele across one or more checkpoint paths (default 4-shot).
#
# Usage (sweep a directory):
#   evaluation/belebele.sh \
#       --checkpoints-dir ${RUN_DIR}/peft \
#       --results-dir     ${RESULTS_DIR}/belebele_4_shot \
#       --num-fewshot     4
#
# Usage (named checkpoints):
#   evaluation/belebele.sh \
#       --results-dir ${RESULTS_DIR}/belebele_4_shot \
#       --num-fewshot 4 \
#       /path/to/ckpt1 /path/to/ckpt2 ...
#
# Env defaults: same set as multiblimp.sh (TOKENIZER, TASKS, DEVICE, NUM_FEWSHOT).

set -euo pipefail

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-}"
TOKENIZER="${TOKENIZER:-allenai/OLMoE-1B-7B-0924-Instruct}"
TASKS="${TASKS:-belebele_eng_Latn,belebele_rus_Cyrl,belebele_hin_Deva,belebele_arb_Arab,belebele_ces_Latn,belebele_spa_Latn,belebele_fin_Latn,belebele_nld_Latn,belebele_ukr_Cyrl,belebele_mar_Deva,belebele_urd_Arab,belebele_cat_Latn,belebele_est_Latn,belebele_slk_Latn}"
PATTERN="${PATTERN:-*}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FEWSHOT="${NUM_FEWSHOT:-4}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
    --tokenizer)       TOKENIZER="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --pattern)         PATTERN="$2"; shift 2 ;;
    --device)          DEVICE="$2"; shift 2 ;;
    --num-fewshot)     NUM_FEWSHOT="$2"; shift 2 ;;
    -h|--help)         sed -n '1,18p' "$0" | sed 's/^# *//'; exit 0 ;;
    --) shift; POSITIONAL+=("$@"); break ;;
    -*) echo "Unknown arg: $1" >&2; exit 1 ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ -z "$RESULTS_DIR" ]]; then
  echo "ERROR: --results-dir (or RESULTS_DIR env var) is required" >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"

run_one() {
  local ckpt=$1
  local name
  name="$(basename "$ckpt")"
  local out="$RESULTS_DIR/$name"
  if [[ -d "$out" ]] && find "$out" -name 'results_*.json' | grep -q .; then
    echo "  skip $name (already evaluated)"
    return 0
  fi
  echo "  eval $name (${NUM_FEWSHOT}-shot)"
  lm_eval --model hf \
    --model_args "pretrained=$ckpt,tokenizer=$TOKENIZER,dtype=auto" \
    --tasks "$TASKS" \
    --batch_size auto \
    --device "$DEVICE" \
    --num_fewshot "$NUM_FEWSHOT" \
    --output_path "$out"
}

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  for ckpt in "${POSITIONAL[@]}"; do
    run_one "$ckpt"
  done
elif [[ -n "$CHECKPOINTS_DIR" ]]; then
  shopt -s nullglob
  for ckpt in "$CHECKPOINTS_DIR"/$PATTERN; do
    [[ -d "$ckpt" ]] || continue
    run_one "$ckpt"
  done
else
  echo "ERROR: pass --checkpoints-dir or positional checkpoint paths" >&2
  exit 1
fi
