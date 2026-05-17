#!/bin/bash
#SBATCH --job-name=seven_langs
#SBATCH --time=11:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/seven_langs.out

# srun bash starter_srun.sh
srun bash run_training.sh
