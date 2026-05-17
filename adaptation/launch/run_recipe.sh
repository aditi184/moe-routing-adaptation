#!/bin/bash
# Generic launcher: pass a recipe YAML, get a torchrun call.
#
# Usage:
#   bash adaptation/launch/run_recipe.sh adaptation/recipes/catalan_ssft_k5_lr4e4.yaml
#
# Required env vars (export them in your shell or a .env loaded by the SLURM script):
#   CKPT_DIR, DATA_DIR, RUN_DIR, WANDB_ENTITY
# Optional:
#   NPROC (default 4), WANDB_PROJECT (overrides recipe value)

set -euo pipefail

RECIPE=${1:?"usage: $0 <recipe.yaml>"}
NPROC=${NPROC:-4}

# Minimal YAML parser: extracts "key: value" pairs, strips quotes from value if surrounded.
parse() {
  local key=$1
  grep -E "^${key}:" "$RECIPE" | head -1 | sed -E "s/^${key}:[[:space:]]*//; s/^'(.*)'$/\\1/; s/^\"(.*)\"$/\\1/; s/[[:space:]]+#.*$//"
}

MODEL_PATH=$(parse model_path)
TOKENIZER_PATH=$(parse tokenizer_path)
OUTPUT_DIR=$(parse output_dir)
TARGET_EXPERTS=$(parse target_experts)
DATASET_PATH=$(parse dataset_path)
LR=$(parse learning_rate)
NUM_EPOCHS=$(parse num_epochs)
BATCH_SIZE=$(parse batch_size)
SAVE_STEPS=$(parse save_steps)
MAX_TOKENS=$(parse max_tokens)
WANDB_PROJECT_REC=$(parse wandb_project)
WANDB_GROUP=$(parse wandb_group)
WANDB_NAME=$(parse wandb_name)
WANDB_TAGS=$(parse wandb_tags)

# Env var expansion (eval the strings since they may contain ${CKPT_DIR} etc.)
MODEL_PATH=$(eval echo "$MODEL_PATH")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
DATASET_PATH=$(eval echo "$DATASET_PATH")
WANDB_GROUP=$(eval echo "$WANDB_GROUP")

torchrun --nproc_per_node="$NPROC" "$(dirname "$0")/../train_peft.py" \
  --model_path     "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_dir     "$OUTPUT_DIR" \
  --target_experts "$TARGET_EXPERTS" \
  --train_routers \
  --dataset_path   "$DATASET_PATH" \
  --num_epochs     "$NUM_EPOCHS" \
  --batch_size     "$BATCH_SIZE" \
  --save_steps     "$SAVE_STEPS" \
  --max_tokens     "$MAX_TOKENS" \
  --learning_rate  "$LR" \
  --wandb_project  "${WANDB_PROJECT:-$WANDB_PROJECT_REC}" \
  --wandb_entity   "${WANDB_ENTITY}" \
  --wandb_group    "$WANDB_GROUP" \
  --wandb_name     "$WANDB_NAME" \
  --wandb_tags     "$WANDB_TAGS"
