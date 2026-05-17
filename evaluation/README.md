# Evaluation

Downstream evaluation on **MultiBLiMP** (syntactic/morphological minimal pairs, zero-shot) and **Belebele** (multiple-choice reading comprehension, 4-shot), plus per-language perplexity. All three use [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).

All scripts take their inputs via CLI flags or environment variables (`${RUN_DIR}`, `${RESULTS_DIR}`, etc.).

## Setup

```bash
pip install lm-eval

# Optional: pre-cache the OLMoE tokenizer locally (otherwise the scripts pull
# from the Hub on first use):
python evaluation/download_tokenizer.py \
       --model_id allenai/OLMoE-1B-7B-0924-Instruct \
       --output_dir ${TOKENIZER_DIR}
```

## MultiBLiMP (zero-shot, all 14 paper languages)

Sweep over every subdir of a runs directory:

```bash
evaluation/multiblimp.sh \
    --checkpoints-dir ${RUN_DIR}/peft \
    --results-dir     ${RESULTS_DIR}/multiblimp
```

Or evaluate one checkpoint:

```bash
evaluation/multiblimp.sh \
    --checkpoint  ${RUN_DIR}/peft/peft_stage2_catalan/ssft_k5_lr4e4 \
    --results-dir ${RESULTS_DIR}/multiblimp
```

Useful flags: `--tokenizer` (HF id or local dir), `--tasks` (override task list), `--pattern` (glob inside the sweep dir, e.g. `peft_stage2_marathi_*`), `--device cuda:1`. The default task list is the 14 paper languages.

## Belebele (4-shot by default)

```bash
evaluation/belebele.sh \
    --checkpoints-dir ${RUN_DIR}/peft \
    --results-dir     ${RESULTS_DIR}/belebele_4_shot
```

Or pass checkpoint paths positionally:

```bash
evaluation/belebele.sh \
    --results-dir ${RESULTS_DIR}/belebele_4_shot \
    /path/to/ckpt1 /path/to/ckpt2 ...
```

To run zero-shot: `--num-fewshot 0` and a different output dir.

## Perplexity

Thin wrapper around `adaptation/eval_peft.py`. For one checkpoint against a list of pre-tokenized `.npy` validation shards:

```bash
evaluation/perplexity.sh \
    --checkpoint  ${RUN_DIR}/peft/peft_stage2_catalan/ssft_k5_lr4e4 \
    --validation  "${DATA_DIR}/evaluation/tokenized/en/part-0-00000.npy,${DATA_DIR}/evaluation/tokenized/ca/part-0-00000.npy" \
    --tokenizer   allenai/OLMoE-1B-7B-0924-Instruct
```

Use `--checkpoints-dir` to sweep an entire sweep directory.

## Consolidate per-checkpoint JSONs into a single CSV

After all `lm_eval` runs finish:

```bash
# MultiBLiMP: appends new rows to the CSV; pass --overwrite to truncate
python evaluation/consolidate_multiblimp.py \
       --results-dir ${RESULTS_DIR}/multiblimp \
       --output      multiblimp_consolidated.csv

# Belebele: merges multiple --results-dir + --shot pairs into one CSV
python evaluation/consolidate_belebele.py \
       --results-dir ${RESULTS_DIR}/belebele_4_shot   --shot 4-shot \
       --results-dir ${RESULTS_DIR}/belebele_zero_shot --shot 0-shot \
       --output      belebele_consolidated.csv
```

## Task definitions

The `multiblimp_*` and `belebele_*` task YAMLs are part of the standard `lm-evaluation-harness` distribution. The paper used a slightly-customized fork; we recommend pinning to upstream `main` and only using local overrides if a task is missing. The script defaults work against current `lm-eval` versions.
