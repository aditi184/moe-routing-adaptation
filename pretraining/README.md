# Continual pre-training: OLMoE-Base → OLMoE-M7

Hyperparameters (Table 4 of the paper):

| | |
|---|---|
| Initial checkpoint | [`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924) (English-centric) |
| Architecture | 16 decoder layers, 64 experts/layer, top-8 routing, d_model=2048 |
| Total training tokens | 35B |
| Languages (uniform sampling) | en, ar, cs, es, fi, hi, ru |
| Optimizer | AdamW (β₁=0.9, β₂=0.95), weight decay 0.1 |
| Learning rate | 1e-4 with cosine schedule, warmup 10.5B tokens |
| Global batch size | 2048 |
| Sequence length | 4096 |
| Precision | bfloat16 AMP |
| Distributed | FSDP, full shard, by-block wrapping |
| Hardware (paper) | 4 nodes × 4 × H100 |

## Setup

Required environment variables:

```bash
export DATA_DIR=/path/to/tokenized/multilang_7   # from data/ step 3
export CKPT_DIR=/path/to/checkpoints
export RUN_DIR=/path/to/runs
export SLURM_ACCOUNT=your-slurm-account-name
```

## 1. Get the OLMoE-Base unsharded checkpoint

```bash
# Download from HuggingFace and unshard into OLMo's native format:
python pretraining/launch/convert_olmo_to_hf.py \
       --download   allenai/OLMoE-1B-7B-0924 \
       --unshard_to ${CKPT_DIR}/base-0924-unsharded
```

(Or use `unshard.py` directly if you already have a sharded checkpoint.)

## 2. Edit the config

`configs/seven_langs.yml` has placeholder paths. Edit:
- `load_path:` → `${CKPT_DIR}/base-0924-unsharded`
- `save_folder:` → `${RUN_DIR}/seven-langs`
- `data.paths:` → your tokenized `${DATA_DIR}/tokenized/part-*.npy`
- `evaluators[].datasets:` → your tokenized held-out splits

## 3. Launch

```bash
sbatch pretraining/launch/slurm_pretrain.sh
```

This runs `starter_low_resource.sh` which parses the SLURM nodelist, sets `MASTER_ADDR` / `NODE_RANK` / `WORLD_SIZE`, and invokes:

```bash
torchrun --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$NODE_RANK --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR --master_port=29500 \
         train.py configs/seven_langs.yml
```

## 4. Convert the final checkpoint to HuggingFace format

The analysis and adaptation pipelines load `OlmoeForCausalLM` from HuggingFace transformers, so the OLMo-native sharded checkpoint needs to be converted:

```bash
python pretraining/launch/convert_olmo_to_hf.py \
       --input_dir  ${RUN_DIR}/seven-langs/stepFINAL-unsharded \
       --output_dir ${CKPT_DIR}/seven-langs-stage1
```

`${CKPT_DIR}/seven-langs-stage1` is the **OLMoE-M7** checkpoint referenced everywhere downstream.

## Low-resource continuation configs

`configs/seven_langs_<lang>_cont.yml` (catalan/estonian/marathi/slovak/ukrainian/urdu/dutch) extend continual pre-training onto a target low-resource language on top of OLMoE-M7. These are not the paper's primary adaptation method (use `adaptation/` for that) — they correspond to the **Full-FT** baseline in Tables 1–3.
