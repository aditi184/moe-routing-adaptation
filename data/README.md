# Data preparation

We use the [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) multilingual corpus (Nguyen et al., 2023) for both continual pre-training and low-resource adaptation.

| Stage | Languages | Tokens (approx.) |
|-------|-----------|------------------|
| Continual pre-training of OLMoE-M7 | en, ar, cs, es, fi, hi, ru | 35B (5B/language, uniform sampling) |
| Low-resource adaptation            | uk, mr, nl, ca, et, sk, ur | 300M per target (paper main experiments) |

## Required environment variables

```bash
export DATA_DIR=/path/to/data   # root for raw + tokenized text
```

The scripts will write to `${DATA_DIR}/high_resource/`, `${DATA_DIR}/low_resource/`, `${DATA_DIR}/multilang/`, etc.

## 1. Download raw text

```bash
# High-resource (7 languages, 5M docs each ≈ enough for 35B-token budget after tokenization):
python data/download_raw.py \
       --languages en ar cs es fi hi ru \
       --samples 5000000 \
       --output_dir ${DATA_DIR}/high_resource/raw

# Low-resource (7 languages):
python data/download_raw.py \
       --languages uk mr nl ca et sk ur \
       --samples 200000 \
       --output_dir ${DATA_DIR}/low_resource/raw
```

## 2. Preprocess

Two scripts depending on whether you want per-language splits (`preprocess_raw.py`) or uniformly-sampled multi-language shards (`preprocess_multilingual.py`):

```bash
# Per-language preprocessing (test split = first 5k docs, rest is train, rotated every 1M docs):
for lang in en ar cs es fi hi ru uk mr nl ca et sk ur; do
  python data/preprocess_raw.py --language $lang \
         --raw_dir    ${DATA_DIR}/high_resource/raw \
         --output_dir ${DATA_DIR}/high_resource
done

# Multi-language uniform sampling for the 35B M7 corpus:
python data/preprocess_multilingual.py \
       --languages en ar cs es fi hi ru \
       --samples 98326775 \
       --output_dir ${DATA_DIR}/multilang_7

# (Optional) Combined high+low-resource shuffled corpus:
python data/preprocess_combined.py \
       --multilang_dir     ${DATA_DIR}/multilang_7 \
       --low_resource_file ${DATA_DIR}/low_resource/combined.jsonl \
       --output_dir        ${DATA_DIR}/combined
```

## 3. Tokenize with the OLMoE/Dolma tokenizer

Requires `dolma` installed (`pip install dolma` — already in `requirements.txt`).

```bash
# Single-shot (high-resource combined corpus):
bash data/tokenize_dolma.sh        # edit base_input / base_output inside

# Loop over the 7 low-resource languages:
bash data/loop_tokenize.sh         # edit base_input / base_output inside
```

Outputs are Dolma `.npy` memmap shards under `<output_dir>/tokenized/<lang>/part-*.npy`.

## Validation splits used for routing analysis

The paper uses 500 held-out documents per language at layer-level analysis. After running `preprocess_raw.py`, the first 5000 docs per language are written to `<lang>_test.jsonl`; you can take the first 500 of those for analysis (or all 5000 for sturdier estimates).
