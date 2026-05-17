#!/bin/bash
#SBATCH --job-name=slovak
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/peft_slovak.out


module load cuda/12.2
source ../.venv_olmoe/bin/activate
module load httpproxy





torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_slovak/ssft_longer_1e4_rebuttal_three \
  --target_experts '{"15": [1,60,30,38,26,57,0,43], "14": [25,26,3,27,7,19,39,13,47]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_slovak" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_slovak_ssft_longer_1e4_rebuttal_three" \
  --wandb_tags "experiment1"

torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_slovak/ssft_longer_1e4_rebuttal_one \
  --target_experts '{"15": [1,60,30,38,26,57], "14": [25,26,3,27,7,19,39]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_slovak" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_slovak_ssft_longer_1e4_rebuttal_one" \
  --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_longer_lr4e4 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_ssft_longer" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/ssft_longer_1e4 \
#   --target_experts '{"15": [1,60,30,38,26,57,0,43,36,31], "14": [25,26,3,27,7,19,39,13,47,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_ssft_longer" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_longer \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_ssft_longer" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/ssft_longer \
#   --target_experts '{"15": [1,60,30,38,26,57,0,43,36,31], "14": [25,26,3,27,7,19,39,13,47,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_ssft_longer" \
#   --wandb_tags "experiment1"








# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/ssft_longer \
#   --target_experts '{"15": [1,60,30,38,26,57,0,43,36,31], "14": [25,26,3,27,7,19,39,13,47,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/sk/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_ssft_longer" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_shared_lr4e4 \
#   --target_experts '{"15": [1,60,30,38,26,57,0,43,36,31], "14": [25,26,3,27,7,19,39,13,47,8,43]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_shared_lr4e4" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_shared_lr1e4 \
#   --target_experts '{"15": [1,60,30,38,26,57,0,43,36,31], "14": [25,26,3,27,7,19,39,13,47,8,43]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_shared_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_top20_lr1e5 \
#   --target_experts '{"15": [1, 60, 38, 30, 26, 59, 51, 53, 23, 52, 39, 48, 27, 18, 3, 20, 35, 2, 47,11], "14": [25, 26, 62, 3, 10, 60, 48, 57, 31, 30, 23, 54, 37, 43, 45, 61, 44, 22, 18, 52]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_top20_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_top20_lr1e4 \
#   --target_experts '{"15": [1, 60, 38, 30, 26, 59, 51, 53, 23, 52, 39, 48, 27, 18, 3, 20, 35, 2, 47,11], "14": [25, 26, 62, 3, 10, 60, 48, 57, 31, 30, 23, 54, 37, 43, 45, 61, 44, 22, 18, 52]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_top20_lr4e4 \
#   --target_experts '{"15": [1, 60, 38, 30, 26, 59, 51, 53, 23, 52, 39, 48, 27, 18, 3, 20, 35, 2, 47,11], "14": [25, 26, 62, 3, 10, 60, 48, 57, 31, 30, 23, 54, 37, 43, 45, 61, 44, 22, 18, 52]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_top20_lr1e3 \
#   --target_experts '{"15": [1, 60, 38, 30, 26, 59, 51, 53, 23, 52, 39, 48, 27, 18, 3, 20, 35, 2, 47,11], "14": [25, 26, 62, 3, 10, 60, 48, 57, 31, 30, 23, 54, 37, 43, 45, 61, 44, 22, 18, 52]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_top20_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/seft_top20_lr4e3 \
#   --target_experts '{"15": [1, 60, 38, 30, 26, 59, 51, 53, 23, 52, 39, 48, 27, 18, 3, 20, 35, 2, 47,11], "14": [25, 26, 62, 3, 10, 60, 48, 57, 31, 30, 23, 54, 37, 43, 45, 61, 44, 22, 18, 52]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_seft_top20_lr4e3" \
#   --wandb_tags "experiment1"


# # LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/train_embeddings_lr1e5 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_train_embeddings_lr1e5" \
#   --wandb_tags "experiment1"

# LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/train_embeddings_lr1e4 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_train_embeddings_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/train_embeddings_lr1e3 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_train_embeddings_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/train_embeddings_lr4e3 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_train_embeddings_lr4e3" \
#   --wandb_tags "experiment1"






# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/all_experts_lr1e5 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/all_experts_lr4e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_lr4e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/all_experts_lr1e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/all_experts_lr4e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak_lr4e3" \
  # --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/lr_4e4 \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/all_experts \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/random_experts \
#   --target_experts '{"15": [4, 11, 33, 47, 59], "14": [0, 8, 14, 32, 41, 58]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_slovak/train_embeddings \
#   --target_experts '{"15": [1,60,30,38,26], "14": [25,26,3,27,7,19]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/sk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_slovak" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_slovak" \
#   --wandb_tags "experiment1"