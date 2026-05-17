# Adaptation to low-resource languages (SEFT / SSFT / baselines)

Parameter-efficient adaptation of a continually-pretrained multilingual OLMoE. The main method is **SSFT** — Selective and Shared Expert Finetuning — which updates only the activation-gap-selected experts and a small set of globally-shared experts in the final two MoE layers (~1.5–2% of model parameters).

## Setup

```bash
export CKPT_DIR=/path/to/checkpoints       # contains your OLMoE-M7 checkpoint
export DATA_DIR=/path/to/tokenized/data    # contains tokenized target-language .npy shards
export RUN_DIR=/path/to/adaptation/runs
export WANDB_ENTITY=your-wandb-entity
export SLURM_ACCOUNT=your-slurm-account
```

## Run a recipe

There's one annotated template at [`recipes/example.yaml`](recipes/example.yaml). Copy it, edit the fields you care about (`target_experts`, `dataset_path`, `learning_rate`, `output_dir`), and launch:

```bash
cp adaptation/recipes/example.yaml adaptation/recipes/my_run.yaml
# edit my_run.yaml ...

# Locally:
bash adaptation/launch/run_recipe.sh adaptation/recipes/my_run.yaml

# Via SLURM:
sbatch adaptation/launch/slurm_peft.sh adaptation/recipes/my_run.yaml
```

## Choosing `target_experts`

The JSON dict in `target_experts` selects which experts to unfreeze. Common choices:

| Strategy | What goes in `target_experts` |
|----------|-------------------------------|
| **SEFT** | activation-gap-selected experts for one target language (~6 per final layer) |
| **SSFT** (recommended) | SEFT set + top-k globally-shared experts per layer (paper uses k = 5) |
| **SEFT-Top20** | top 20 experts per layer by activation gap (control for expert count) |
| **AEFT** | all 64 experts in both final layers (~800M params) |
| **Full-FT** | drop `target_experts` and use `pretraining/configs/...` with a low LR instead |

Generate per-language SEFT/SSFT expert sets from your own routing pickle:

```bash
python -m analysis.analysis select-experts \
    --counts analysis/data/olmoe_m7_counts.pkl \
    --layers 14 15 \
    --alpha  0.01 \
    --ks     1 3 5 \
    --out    experts.json
```

The output JSON has one entry per (language, strategy); paste the relevant `target_experts` value into your recipe.

## Learning-rate sweep

Sweep `learning_rate` over `{1e-5, 1e-4, 4e-4, 1e-3, 4e-3}` (one recipe per LR) and pick the checkpoint with the best held-out validation perplexity:

```bash
CUDA_VISIBLE_DEVICES=0 python adaptation/eval_peft.py \
  --checkpoint_path  ${RUN_DIR}/peft/my_run \
  --validation_path  "${DATA_DIR}/evaluation/tokenized/<lang>/part-0-00000.npy,..." \
  --tokenizer_path   allenai/OLMoE-1B-7B-0924-Instruct \
  --max_length 2048 --num_samples 250 --batch_size 4
```

## What `train_peft.py` actually does

1. Loads `OlmoeForCausalLM` from `--model_path`.
2. Freezes every parameter.
3. Re-enables gradients on the experts listed in `--target_experts` (`{"layer_idx": [expert_idx, ...]}`).
4. If `--train_routers`, re-enables router params for the listed layers.
5. If `--train_embeddings`, re-enables embedding params.
6. Trains with HuggingFace `Trainer` on the pre-tokenized `.npy` shards, with a perplexity callback for monitoring.

## Original sweep scripts (archived)

The full LR / α / k grids ran during the paper are preserved verbatim under [`launch/sweeps/`](launch/sweeps/) for transparency. Each one expects the same env vars as the template above.
