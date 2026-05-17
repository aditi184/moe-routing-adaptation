# MoE routing analysis and parameter-efficient multilingual adaptation

A study of multilingual routing dynamics in OLMoE-class Mixture-of-Experts language models, and a parameter-efficient adaptation method for low-resource languages that updates only ~2% of model parameters.

We continually pre-train [OLMoE-Base](https://huggingface.co/allenai/OLMoE-1B-7B-0924) on a balanced 7-language corpus (→ **OLMoE-M7**), analyze how multilingual routing evolves layer-by-layer, and use the analysis to drive a parameter-efficient adaptation method called **SSFT** (Selective and Shared Expert Finetuning). SSFT updates only ~2% of model parameters and matches the performance of full-model finetuning on six low-resource languages.

```
data ──┐
       ├──► pretraining (OLMoE-Base → OLMoE-M7, 35B tokens, 7 langs)
       │      │
       │      ▼                            ┌──► analysis  (routing entropy, JSD, vocab overlap)
       │      seven-langs-stage1 ──────────┤                          │
       │                                   │                          ▼
       │                                   └──► selected_experts.json (activation-gap selection)
       │                                                              │
       │                                                              ▼
       └──► adaptation (SEFT / SSFT / AEFT / Full-FT on 6 low-res langs)
              │
              ▼
            evaluation (MultiBLiMP zero-shot, Belebele 4-shot, perplexity)
              │
              ▼
            consolidated CSVs (one row per checkpoint)
```

Every script takes its inputs via CLI flags or environment variables (`${CKPT_DIR}`, `${DATA_DIR}`, etc.). 

---

## 1. Setup

```bash
git clone https://github.com/aditi184/moe-routing-adaptation.git
cd moe-routing-adaptation

# Pretraining env (uses upstream OLMo's pinned dependencies):
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: separate env for adaptation (HF transformers, accelerate, peft, datasets, wandb):
python3.11 -m venv .venv_olmoe
source .venv_olmoe/bin/activate
pip install "transformers>=4.45" "accelerate>=0.33" datasets wandb dolma "lm-eval>=0.4"
```

### Required environment variables

The scripts use env vars instead of hardcoded paths. Export these in your shell (or `.env`):

```bash
export DATA_DIR=/path/to/data            # raw + tokenized text shards
export CKPT_DIR=/path/to/checkpoints     # OLMoE-M7 + base + tmp ckpts
export RUN_DIR=/path/to/runs             # training run outputs
export RESULTS_DIR=/path/to/eval_results # lm_eval JSON outputs
export TOKENIZER_DIR=/path/to/tokenizer  # local OLMoE tokenizer cache
export WANDB_ENTITY=your-wandb-entity    # Weights & Biases entity
export SLURM_ACCOUNT=your-slurm-account  # SLURM account for sbatch
export SCRATCH_DIR=/path/to/scratch      # fallback
```

### Hardware used in the paper

- **Continual pre-training (OLMoE-M7):** 4 nodes × 4 × H100, ~35B tokens, bf16 AMP, FSDP, ~2–3 days.
- **Adaptation (SSFT/SEFT):** 1 node × 4 × H100, ~300M target-language tokens, 45–50 min per run.
- **Evaluation (MultiBLiMP, Belebele):** 1 × H100 per checkpoint via `lm_eval`.

---

## 2. End-to-end pipeline

See each sub-directory's `README.md` for step-by-step instructions:

| Step | Directory | What |
|------|-----------|------|
| **Data prep** | [`data/`](data/) | Download CulturaX (7 HR + 7 LR languages), preprocess, tokenize with Dolma |
| **Pre-training** | [`pretraining/`](pretraining/) | OLMoE-Base → OLMoE-M7 (35B tokens, balanced 7-language mix) |
| **Routing analysis** | [`analysis/`](analysis/) | Entropy, JSD, vocab-overlap scatter, activation-gap expert selection (CLI subcommands over any OLMoE-class checkpoint) |
| **Adaptation** | [`adaptation/`](adaptation/) | SEFT / SSFT / SEFT-Top20 (+ AEFT / Full-FT) on 6 low-resource languages |
| **Evaluation** | [`evaluation/`](evaluation/) | MultiBLiMP zero-shot + Belebele 4-shot + perplexity |

### Quick start: reproduce SSFT for one language

If you already have the OLMoE-M7 checkpoint at `${CKPT_DIR}/seven-langs-stage1` and tokenized Catalan data at `${DATA_DIR}/low_resource/tokenized/ca/`:

```bash
cp adaptation/recipes/example.yaml adaptation/recipes/my_run.yaml
# edit my_run.yaml to set target_experts, dataset_path, learning_rate
bash adaptation/launch/run_recipe.sh adaptation/recipes/my_run.yaml
```

Then evaluate:

```bash
evaluation/multiblimp.sh \
    --checkpoints-dir ${RUN_DIR}/peft \
    --results-dir     ${RESULTS_DIR}/multiblimp
python evaluation/consolidate_multiblimp.py \
    --results-dir ${RESULTS_DIR}/multiblimp \
    --output      multiblimp_consolidated.csv
```

---

## 3. Analysis subcommands

`analysis/analysis.py` is a single CLI with one subcommand per figure / output:

```
entropy        English vs. avg-non-English routing entropy per layer
jsd-heatmap    pairwise JSD heatmap at one layer
jsd-per-layer  average pairwise JSD across layers for two model stages
jsd-vs-vocab   scatter of pairwise JSD vs token-vocab overlap
specialization normalized IG(L;E)/H(L) per layer for one or more models
select-experts activation-gap expert selection -> selected_experts.json
export-jsd     dump all-pair JSD at one layer to CSV
all            run every subcommand into one output directory
```

Routing counts (the input to all subcommands above) come from `analysis/collect_routing.py`, which does a forward pass through any OLMoE-class HF checkpoint and dumps per-(language, layer, expert) activation counts. See [`analysis/README.md`](analysis/README.md) for full usage.

---

## 4. Repository layout

```
moe-routing-adaptation/
├── README.md                    you are here
├── LICENSE                      Apache 2.0 (inherited from allenai/OLMo)
├── NOTICE                       attribution to upstream OLMo + EleutherAI
├── pyproject.toml               pip-installable
├── requirements.txt             pinned deps for pre-training env
│
├── data/                        raw download + preprocessing + tokenization
├── pretraining/                 OLMoE-M7 continual pre-training (vendored olmo/ lib + train.py + configs + SLURM)
├── analysis/                    routing entropy / JSD / vocab overlap / activation-gap selection (CLI + selected_experts.json)
├── adaptation/                  SEFT / SSFT / SEFT-Top20 recipes + HF trainer
└── evaluation/                  MultiBLiMP, Belebele, perplexity via lm-eval
```

---

## 5. Acknowledgements

Built on top of [allenai/OLMo](https://github.com/allenai/OLMo), [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924), and [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Pre-training data from [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) (Nguyen et al., 2023). Evaluation benchmarks: [MultiBLiMP](https://huggingface.co/datasets/jumelet/multiblimp) (Jumelet et al., 2025) and [Belebele](https://huggingface.co/datasets/facebook/belebele) (Bandarkar et al., 2024).

## 6. License

Apache 2.0, inherited from upstream [allenai/OLMo](https://github.com/allenai/OLMo). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
