#!/bin/bash
#SBATCH --job-name=marathi
#SBATCH --time=02:59:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --account=
#SBATCH --output=logs/peft_marathi.out


module load cuda/12.2
source ../.venv_olmoe/bin/activate
module load httpproxy



torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_marathi/ssft_longer_lr1e4_rebuttal_three \
  --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43], "14": [9,4,11,42,55,21,35,39,13]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_marathi" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_marathi_ssft_longer_lr1e4_rebuttal_three" \
  --wandb_tags "experiment1"

torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_marathi/ssft_longer_lr1e4_rebuttal_one \
  --target_experts '{"15": [33,50,14,45,34,32,40,4,57], "14": [9,4,11,42,55,21,35,39]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --wandb_project "peft_stage2_marathi" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_marathi_ssft_longer_lr1e4_rebuttal_one" \
  --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_longer_lr1e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_longer_lr1e4" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/ssft_longer_lr4e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43,36,31], "14": [9,4,11,42,55,21,35,39,13,7,27,47]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_ssft_longer_lr4e4" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_longer \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_ssft_longer" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/ssft_longer \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43,36,31], "14": [9,4,11,42,55,21,35,39,13,7,27,47]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/mr/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_ssft_longer" \
#   --wandb_tags "experiment1"





# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_shared_lr1e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43,36,31], "14": [9,4,11,42,55,21,35,39,13,7,27,47]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_shared_lr1e4" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_shared_lr1e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43,36,31], "14": [9,4,11,42,55,21,35,39,13,7,27,47]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_shared_lr1e4" \
#   --wandb_tags "experiment1"



# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_shared_lr4e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4,57,0,43,36,31], "14": [9,4,11,42,55,21,35,39,13,7,27,47]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_shared_lr4e4" \
#   --wandb_tags "experiment1"


# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_top20_lr1e5 \
#   --target_experts '{"15": [33, 50, 25, 55, 46, 32, 45, 13, 4, 62, 14, 28, 10, 21, 34, 56, 15, 40, 61, 37], "14": [9, 21, 4, 35, 11, 42, 55, 18, 49, 22, 30, 5, 54, 12, 60, 17, 51, 61, 29, 37]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_top20_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_top20_lr1e4 \
#   --target_experts '{"15": [33, 50, 25, 55, 46, 32, 45, 13, 4, 62, 14, 28, 10, 21, 34, 56, 15, 40, 61, 37], "14": [9, 21, 4, 35, 11, 42, 55, 18, 49, 22, 30, 5, 54, 12, 60, 17, 51, 61, 29, 37]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_top20_lr4e4 \
#   --target_experts '{"15": [33, 50, 25, 55, 46, 32, 45, 13, 4, 62, 14, 28, 10, 21, 34, 56, 15, 40, 61, 37], "14": [9, 21, 4, 35, 11, 42, 55, 18, 49, 22, 30, 5, 54, 12, 60, 17, 51, 61, 29, 37]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_top20_lr1e3 \
#   --target_experts '{"15": [33, 50, 25, 55, 46, 32, 45, 13, 4, 62, 14, 28, 10, 21, 34, 56, 15, 40, 61, 37], "14": [9, 21, 4, 35, 11, 42, 55, 18, 49, 22, 30, 5, 54, 12, 60, 17, 51, 61, 29, 37]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_top20_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/seft_top20_lr4e3 \
#   --target_experts '{"15": [33, 50, 25, 55, 46, 32, 45, 13, 4, 62, 14, 28, 10, 21, 34, 56, 15, 40, 61, 37], "14": [9, 21, 4, 35, 11, 42, 55, 18, 49, 22, 30, 5, 54, 12, 60, 17, 51, 61, 29, 37]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_seft_top20_lr4e3" \
#   --wandb_tags "experiment1"

# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/train_embeddings_lr1e5 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_train_embeddings_lr1e5" \
#   --wandb_tags "experiment1"

# LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/train_embeddings_lr1e4 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_train_embeddings_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/train_embeddings_lr1e3 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_train_embeddings_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/train_embeddings_lr4e3 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_train_embeddings_lr4e3" \
#   --wandb_tags "experiment1"





# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/all_experts_lr1e5 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/all_experts_lr4e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_lr4e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/all_experts_lr1e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/all_experts_lr4e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi_lr4e3" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/lr_1e3 \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/all_experts \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/random_experts \
#   --target_experts '{"15": [2, 7, 18, 26, 41, 47, 56, 61], "14": [1, 6, 17, 28, 39, 48, 60]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_marathi/train_embeddings \
#   --target_experts '{"15": [33,50,14,45,34,32,40,4], "14": [9,4,11,42,55,21,35]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/mr/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_marathi" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_marathi" \
#   --wandb_tags "experiment1"