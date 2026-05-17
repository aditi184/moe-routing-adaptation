#!/bin/bash
#SBATCH --job-name=pretrain_mid_resource
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/pretrain_mid_resource.out

# srun bash starter_srun.sh
srun bash starter_low_resource.sh