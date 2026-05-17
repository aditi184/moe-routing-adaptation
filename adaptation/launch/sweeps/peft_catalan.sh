#!/bin/bash
#SBATCH --job-name=catalan
#SBATCH --time=02:59:00
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
  --output_dir /peft/peft_stage2_catalan/ssft_long_lr1e4_rebuttal_three \
  --target_experts '{"15": [6,8,63,7,49,58,57,0,43], "14": [46,0,27,7,19,47,39,13,26]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --learning_rate 1e-4 \
  --wandb_project "peft_stage2_catalan" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_catalan_ssft_long_lr1e4_rebuttal_three" \
  --wandb_tags "experiment1"

torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_catalan/ssft_long_lr1e4_rebuttal_one \
  --target_experts '{"15": [6,8,63,7,49,58,57], "14": [46,0,27,7,19,47,39]}' \
  --train_routers \
  --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
  --num_epochs 10 \
  --batch_size 16 \
  --save_steps 5000 \
  --max_tokens 800000000 \
  --learning_rate 1e-4 \
  --wandb_project "peft_stage2_catalan" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_catalan_ssft_long_lr1e4_rebuttal_one" \
  --wandb_tags "experiment1"






#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/ssft_long_lr1e4 \
#   --target_experts '{"15": [6,8,63,7,49,58,57,0,43,36,31], "14": [46,0,27,7,19,47,39,13,26,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr1e4" \
#   --wandb_tags "experiment1"




# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/ssft_long_lr1e4 \
#   --target_experts '{"15": [6,8,63,7,49,58,57,0,43,36,31], "14": [46,0,27,7,19,47,39,13,26,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr1e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/ssft_long_lr4e4 \
#   --target_experts '{"15": [6,8,63,7,49,58,57,0,43,36,31], "14": [46,0,27,7,19,47,39,13,26,8,43]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --learning_rate 4e-4 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr4e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_longer_lr1e4 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr1e4" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_longer_lr4e4 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --dataset_path "/low_resource/tokenized/ca/part-0-00000.npy" \
#   --num_epochs 10 \
#   --batch_size 16 \
#   --save_steps 5000 \
#   --max_tokens 800000000 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_ssft_long_lr4e4" \
#   --wandb_tags "experiment1"






# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_shared_lr1e4 \
#   --target_experts '{"15": [6,8,63,7,49,58,57,0,43,36,31], "14": [46,0,27,7,19,47,39,13,26,8,43]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_shared_lr1e4" \
#   --wandb_tags "experiment1"


torchrun --nproc_per_node=4 train_peft_hf.py \
  --model_path /seven-langs-stage1 \
  --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
  --output_dir /peft/peft_stage2_catalan/seft_shared_lr4e4 \
  --target_experts '{"15": [6,8,63,7,49,58,57,0,43,36,31], "14": [46,0,27,7,19,47,39,13,26,8,43]}' \
  --train_routers \
  --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
  --num_epochs 1 \
  --batch_size 16 \
  --learning_rate 4e-4 \
  --save_steps 3000 \
  --wandb_project "peft_stage2_catalan" \
  --wandb_entity "" \
  --wandb_group "" \
  --wandb_name "peft_stage2_catalan_seft_shared_lr4e4" \
  --wandb_tags "experiment1"


# # LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_top20_lr1e5 \
#   --target_experts '{"15": [6, 7, 8, 63, 49, 58, 42, 47, 52, 26, 51, 2, 48, 23, 18, 20, 30, 38, 3, 54], "14": [8, 38, 41, 32, 53, 46, 0, 15, 44, 14, 31, 20, 43, 48, 10, 45, 3, 37, 60, 18]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_top20_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_top20_lr1e4 \
#   --target_experts '{"15": [6, 7, 8, 63, 49, 58, 42, 47, 52, 26, 51, 2, 48, 23, 18, 20, 30, 38, 3, 54], "14": [8, 38, 41, 32, 53, 46, 0, 15, 44, 14, 31, 20, 43, 48, 10, 45, 3, 37, 60, 18]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_top20_lr4e4 \
#   --target_experts '{"15": [6, 7, 8, 63, 49, 58, 42, 47, 52, 26, 51, 2, 48, 23, 18, 20, 30, 38, 3, 54], "14": [8, 38, 41, 32, 53, 46, 0, 15, 44, 14, 31, 20, 43, 48, 10, 45, 3, 37, 60, 18]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_top20_lr1e4" \
#   --wandb_tags "experiment1"


# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_top20_lr1e3 \
#   --target_experts '{"15": [6, 7, 8, 63, 49, 58, 42, 47, 52, 26, 51, 2, 48, 23, 18, 20, 30, 38, 3, 54], "14": [8, 38, 41, 32, 53, 46, 0, 15, 44, 14, 31, 20, 43, 48, 10, 45, 3, 37, 60, 18]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_top20_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/seft_top20_lr4e3 \
#   --target_experts '{"15": [6, 7, 8, 63, 49, 58, 42, 47, 52, 26, 51, 2, 48, 23, 18, 20, 30, 38, 3, 54], "14": [8, 38, 41, 32, 53, 46, 0, 15, 44, 14, 31, 20, 43, 48, 10, 45, 3, 37, 60, 18]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_seft_top20_lr4e3" \
#   --wandb_tags "experiment1"




# # LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/train_embeddings_lr1e5 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_train_embeddings_lr1e5" \
#   --wandb_tags "experiment1"

# LR = 1e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/train_embeddings_lr1e4 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_train_embeddings_lr1e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/train_embeddings_lr1e3 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_train_embeddings_lr1e3" \
#   --wandb_tags "experiment1"

# LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/train_embeddings_lr4e3 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_train_embeddings_lr4e3" \
#   --wandb_tags "experiment1"




# LR = 1e-5
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/all_experts_lr1e5 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-5 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_lr1e5" \
#   --wandb_tags "experiment1"

# # LR = 4e-4
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/all_experts_lr4e4 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_lr4e4" \
#   --wandb_tags "experiment1"

# # LR = 1e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/all_experts_lr1e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_lr1e3" \
#   --wandb_tags "experiment1"

# # LR = 4e-3
# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/all_experts_lr4e3 \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-3 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan_lr4e3" \
#   --wandb_tags "experiment1"



# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/lr_4e4 \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan" \
#   --wandb_tags "experiment1"


# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/all_experts \
#   --target_experts '{"14": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63], "15": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 1e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/random_experts \
#   --target_experts '{"15": [12, 34, 51, 2, 41, 60], "14": [5, 38, 22, 61, 9, 55]}' \
#   --train_routers \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan" \
#   --wandb_tags "experiment1"

# torchrun --nproc_per_node=4 train_peft_hf.py \
#   --model_path /seven-langs-stage1 \
#   --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
#   --output_dir /peft/peft_stage2_catalan/train_embeddings \
#   --target_experts '{"15": [6,8,63,7,49,58], "14": [46,0,27,7,19,47]}' \
#   --train_routers \
#   --train_embeddings \
#   --gradient_checkpointing \
#   --dataset_path "/peft_sets/ca/tokenized/part-0-00000.npy" \
#   --num_epochs 1 \
#   --batch_size 16 \
#   --learning_rate 4e-4 \
#   --save_steps 3000 \
#   --wandb_project "peft_stage2_catalan" \
#   --wandb_entity "" \
#   --wandb_group "" \
#   --wandb_name "peft_stage2_catalan" \
#   --wandb_tags "experiment1"
