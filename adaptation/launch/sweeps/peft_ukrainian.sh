#!/bin/bash
#SBATCH --job-name=ukrainian
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/peft_ukrainian.out


module load cuda/12.2
source ../.venv_olmoe/bin/activate
module load httpproxy





torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_ukrainian/ssft_longer_lr1e4_rebuttal_three \
  --target_experts '{"15": [57,53,17,1,43,31,0,36,58], "14": [13,36,28,59,33,17,39,7]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 10000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_ukrainian" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_ukrainian_ssft_longer_lr1e4_rebuttal_three" \
  --wandb_tags "experiment1"




torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_ukrainian/ssft_longer_lr1e4_rebuttal_one \
  --target_experts '{"15": [57,53,17,1,43,31,0,36], "14": [13,36,28,59,33,17,39]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 10000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_ukrainian" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_ukrainian_ssft_longer_lr1e4_rebuttal_one" \
  --wandb_tags "experiment1"




# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/ssft_longer_lr4e4 \
#   --target_experts '{"15": [57,53,17,1,43,31,0,36,58,60,6], "14": [13,36,28,59,33,17,39,7,27,47,19]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_ssft_longer_lr4e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/ssft_longer_lr1e4 \
#   --target_experts '{"15": [57,53,17,1,43,31,0,36,58,60,6], "14": [13,36,28,59,33,17,39,7,27,47,19]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_ssft_longer_lr1e4" \
#   --wandb_tags "experiment1"




# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_longer_lr4e4 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17,39]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_longer_lr4e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_longer_lr1e4 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17,39]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/uk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_longer_lr1e4" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_shared_lr4e4 \
#   --target_experts '{"15": [57,53,17,1,43,31,0,36,58,60,6], "14": [13,36,28,59,33,17,39,7,27,47,19]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_shared_lr4e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_shared_lr1e4 \
#   --target_experts '{"15": [57,53,17,1,43,31,0,36,58,60,6], "14": [13,36,28,59,33,17,39,7,27,47,19]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_shared_lr1e4" \
#   --wandb_tags "experiment1"






torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_ukrainian/train_embeddings_lr1e5 \
  --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
  --train_routers \
  --train_embeddings \
  --gradient_checkpointing \
  --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
  --num_epochs 1 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --save_steps 3000 \
  --wandb_project "peft_stage2_ukrainian" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_ukrainian_train_embeddings_lr1e5" \
  --wandb_tags "experiment1"

# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_top20_lr1e5 \
#   --target_experts '{"15": [57, 27, 23, 53, 17, 2, 18, 35, 30, 3, 11, 54, 20, 15, 51, 56, 52, 21, 28, 37], "14": [28, 13, 36, 59, 33, 23, 57, 29, 17, 43, 60, 62, 10, 20, 30, 45, 22, 54, 31, 0]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_top20_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_top20_lr1e4 \
#   --target_experts '{"15": [57, 27, 23, 53, 17, 2, 18, 35, 30, 3, 11, 54, 20, 15, 51, 56, 52, 21, 28, 37], "14": [28, 13, 36, 59, 33, 23, 57, 29, 17, 43, 60, 62, 10, 20, 30, 45, 22, 54, 31, 0]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_top20_lr4e4 \
#   --target_experts '{"15": [57, 27, 23, 53, 17, 2, 18, 35, 30, 3, 11, 54, 20, 15, 51, 56, 52, 21, 28, 37], "14": [28, 13, 36, 59, 33, 23, 57, 29, 17, 43, 60, 62, 10, 20, 30, 45, 22, 54, 31, 0]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_top20_lr1e3 \
#   --target_experts '{"15": [57, 27, 23, 53, 17, 2, 18, 35, 30, 3, 11, 54, 20, 15, 51, 56, 52, 21, 28, 37], "14": [28, 13, 36, 59, 33, 23, 57, 29, 17, 43, 60, 62, 10, 20, 30, 45, 22, 54, 31, 0]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_top20_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/seft_top20_lr4e3 \
#   --target_experts '{"15": [57, 27, 23, 53, 17, 2, 18, 35, 30, 3, 11, 54, 20, 15, 51, 56, 52, 21, 28, 37], "14": [28, 13, 36, 59, 33, 23, 57, 29, 17, 43, 60, 62, 10, 20, 30, 45, 22, 54, 31, 0]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_seft_top20_lr4e3" \
#   --wandb_tags "experiment1"
# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/train_embeddings_lr1e5 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_train_embeddings_lr1e5" \
# #   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/train_embeddings_lr1e4 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_train_embeddings_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/train_embeddings_lr1e3 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_train_embeddings_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/train_embeddings_lr4e3 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_train_embeddings_lr4e3" \
#   --wandb_tags "experiment1"

# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/all_experts_lr1e5 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/all_experts_lr4e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_lr4e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/all_experts_lr1e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/all_experts_lr4e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian_lr4e3" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/lr_4e4 \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/all_experts \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/random_experts \
#   --target_experts '{"15": [4, 9, 22, 38, 46, 60], "14": [2, 7, 19, 41, 50, 62]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_ukrainian/train_embeddings \
#   --target_experts '{"15": [57,53,17,1,43,31], "14": [13,36,28,59,33,17]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_ukrainian" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_ukrainian" \
#   --wandb_tags "experiment1"