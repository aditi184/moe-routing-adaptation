#!/bin/bash
#SBATCH --job-name=dutch
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/peft_dutch.out


module load cuda/12.2
source ../.venv_olmoe/bin/activate
module load httpproxy

# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/seft_top20_lr1e5 \
#   --target_experts '{"15": [51, 2, 25, 18, 21, 56, 5, 20, 10, 52, 61, 13, 46, 23, 54, 15, 19, 27, 48, 55], "14": [23, 29, 43, 63, 34, 62, 31, 44, 48, 20, 38, 15, 16, 45, 10, 60, 57, 0, 18, 54]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_seft_top20_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/seft_top20_lr1e4 \
#   --target_experts '{"15": [51, 2, 25, 18, 21, 56, 5, 20, 10, 52, 61, 13, 46, 23, 54, 15, 19, 27, 48, 55], "14": [23, 29, 43, 63, 34, 62, 31, 44, 48, 20, 38, 15, 16, 45, 10, 60, 57, 0, 18, 54]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/seft_top20_lr4e4 \
#   --target_experts '{"15": [51, 2, 25, 18, 21, 56, 5, 20, 10, 52, 61, 13, 46, 23, 54, 15, 19, 27, 48, 55], "14": [23, 29, 43, 63, 34, 62, 31, 44, 48, 20, 38, 15, 16, 45, 10, 60, 57, 0, 18, 54]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/seft_top20_lr1e3 \
#   --target_experts '{"15": [51, 2, 25, 18, 21, 56, 5, 20, 10, 52, 61, 13, 46, 23, 54, 15, 19, 27, 48, 55], "14": [23, 29, 43, 63, 34, 62, 31, 44, 48, 20, 38, 15, 16, 45, 10, 60, 57, 0, 18, 54]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_seft_top20_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_dutch/seft_top20_lr4e3 \
  --target_experts '{"15": [51, 2, 25, 18, 21, 56, 5, 20, 10, 52, 61, 13, 46, 23, 54, 15, 19, 27, 48, 55], "14": [23, 29, 43, 63, 34, 62, 31, 44, 48, 20, 38, 15, 16, 45, 10, 60, 57, 0, 18, 54]}' \
  --train_routers \
  --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
  --num_epochs 1 \
  --batch_size 16 \
  --learning_rate 4e-3 \
  --save_steps 3000 \
  --wandb_project "peft_stage2_dutch" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_dutch_seft_top20_lr4e3" \
  --wandb_tags "experiment1"
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/all_experts_lr1e5 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/nl/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_lr1e5" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/all_experts_lr4e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/nl/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_lr4e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/all_experts_lr1e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/nl/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_lr1e3" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/all_experts_lr4e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/nl/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch_lr4e3" \
#   --wandb_tags "experiment1"





# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_dutch/all_experts_lr1e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/nl/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_dutch" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_dutch" \
#   --wandb_tags "experiment1"

