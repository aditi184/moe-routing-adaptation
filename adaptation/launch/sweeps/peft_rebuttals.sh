#!/bin/bash
#SBATCH --job-name=catalan
#SBATCH --time=10:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/peft_catalan.out


module load cuda/12.2
source ../.venv_olmoe/bin/activate
module load httpproxy

torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_urdu/ssft_long_lr1e4_strictalpha \
  --target_experts '{"15": [44]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/ur/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --save_steps 10000 \
  --max_tokens 800000000 \
  --learning_rate 1e-4 \
  --wandb_project "peft_stage2_urdu" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_urdu_ssft_long_lr1e4_rebuttal_three" \
  --wandb_tags "experiment1"

torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_slovak/ssft_long_lr1e4_strictalpha \
  --target_experts '{"15": [1], "14": [25,26]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --save_steps 10000 \
  --max_tokens 800000000 \
  --learning_rate 1e-4 \
  --wandb_project "peft_stage2_slovak" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_slovak_ssft_long_lr1e4_strictalpha" \
  --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/ssft_long_lr1e4_strictalpha \
#   --target_experts '{"15": [6,8,63,7,49,58]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 10000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr1e4_strictalpha" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_estonian/ssft_long_lr1e4_strictalpha \
#   --target_experts '{"15": [41], "14": [27,7,19]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/et/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 10000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --wandb_project "peft_stage2_estonian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_estonian_ssft_long_lr1e4_strictalpha" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/ssft_long_lr1e4_strictalpha \
#   --target_experts '{"14": [13]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 10000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_ssft_long_lr1e4_strictalpha" \
#   --wandb_tags "experiment1"
