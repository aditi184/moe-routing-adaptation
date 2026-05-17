#!/bin/bash
#SBATCH --job-name=peft
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --output=logs/peft_%j.out

# Usage:
#   sbatch adaptation/launch/slurm_peft.sh adaptation/recipes/catalan_ssft_k5_lr4e4.yaml
#
# Required env vars (set them before sbatch, or in your shell rc):
#   CKPT_DIR DATA_DIR RUN_DIR WANDB_ENTITY SLURM_ACCOUNT
# Optional:
#   PYTHON_VENV (default: ../.venv_olmoe), CUDA_MODULE (default: cuda/12.2)

set -euo pipefail
RECIPE=${1:?"usage: sbatch $0 <recipe.yaml>"}

module load "${CUDA_MODULE:-cuda/12.2}"
source "${PYTHON_VENV:-../.venv_olmoe}/bin/activate"
module load httpproxy 2>/dev/null || true

bash "$(dirname "$0")/run_recipe.sh" "$RECIPE"
