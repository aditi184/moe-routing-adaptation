#!/bin/bash
#SBATCH --job-name=seven_langs
#SBATCH --time=02:30:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/seven_langs.out

# srun bash starter_srun.sh
srun bash starter_low_resource.sh
