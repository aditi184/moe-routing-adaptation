# Routing analysis

Two scripts, both fully CLI-driven (no module-level paths to edit):

| File | What it does |
|------|--------------|
| `collect_routing.py` | Forward-pass over per-language held-out text through an OLMoE checkpoint; dumps per-(layer, expert) token activation counts to a pickle. One pickle per model. |
| `analysis.py` | Consumes the pickles from `collect_routing.py` and produces routing entropy, pairwise JSD, JSD-vs-vocab-overlap scatter, language-specialization score, and activation-gap expert selection. |

The `jsd-vs-vocab` subcommand needs a pairwise token-vocab overlap CSV (a square matrix indexed by language full name). Compute it for your tokenizer and pass the path via `--vocab-overlap`.

## Inputs

- `--model`: any OLMoE-family HuggingFace checkpoint (HF id or local path). The paper uses [`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924) as the base and the continually-pretrained OLMoE-M7 (output of `pretraining/`) as the second model.
- `--data-dir`: a directory containing one `{lang_code}.txt` per language. ~500 docs per language is enough.
- `--vocab-overlap` (only `jsd-vs-vocab` / `all`): a square CSV indexed by language *full name* (`english`, `arabic`, etc. — see `LANG_FULL` in `analysis.py`).

## 1. Collect routing counts

```bash
# OLMoE-Base
python -m analysis.collect_routing \
    --model      allenai/OLMoE-1B-7B-0924 \
    --data-dir   ${DATA_DIR}/validation_text \
    --languages  en,ar,cs,es,fi,hi,ru,nl,ur,sk,ca,et,mr,uk \
    --output     analysis/data/olmoe_base_counts.pkl

# OLMoE-M7 (your continually-pretrained HF checkpoint)
python -m analysis.collect_routing \
    --model      ${CKPT_DIR}/seven-langs-stage1 \
    --data-dir   ${DATA_DIR}/validation_text \
    --languages  en,ar,cs,es,fi,hi,ru,nl,ur,sk,ca,et,mr,uk \
    --output     analysis/data/olmoe_m7_counts.pkl
```

Pickle layout:
```
counts[lang_code: str][layer_idx: int] -> Counter({expert_id: token_count})
```

## 2. Make the figures and expert assignments

Subcommands let you run only what you need:

```bash
# Fig. 2 - English vs avg-non-English routing entropy per layer
python -m analysis.analysis entropy \
    --base   analysis/data/olmoe_base_counts.pkl \
    --stage1 analysis/data/olmoe_m7_counts.pkl \
    --out    figs/entropy

# Fig. 4 - pairwise JSD heatmap at layer 15
python -m analysis.analysis jsd-heatmap \
    --counts analysis/data/olmoe_m7_counts.pkl \
    --layer  15 \
    --title  "OLMoE-M7" \
    --out    figs/jsd_heatmap_m7_layer15.pdf

# Fig. 5 - average pairwise JSD across layers, both models with/without English
python -m analysis.analysis jsd-per-layer \
    --base   analysis/data/olmoe_base_counts.pkl \
    --stage1 analysis/data/olmoe_m7_counts.pkl \
    --out    figs/avg_jsd_per_layer.pdf

# Fig. 6 - JSD vs token-vocab overlap (needs the pairwise overlap CSV)
python -m analysis.analysis jsd-vs-vocab \
    --counts        analysis/data/olmoe_m7_counts.pkl \
    --vocab-overlap analysis/data/pairwise_vocab_overlap.csv \
    --layer         15 \
    --out           figs/jsd_vs_vocab_overlap.pdf

# Activation-gap expert selection -> JSON of per-language expert IDs
python -m analysis.analysis select-experts \
    --counts analysis/data/olmoe_m7_counts.pkl \
    --layers 14 15 \
    --alpha  0.01 \
    --ks     1 3 5 \
    --out    experts.json

# Or fire everything at once into one output directory
python -m analysis.analysis all \
    --base   analysis/data/olmoe_base_counts.pkl \
    --stage1 analysis/data/olmoe_m7_counts.pkl \
    --vocab-overlap analysis/data/pairwise_vocab_overlap.csv \
    --out-dir figs
```

Run `python -m analysis.analysis <subcommand> --help` for the full per-subcommand option list (you can override the HR/LR language sets, layer numbers, heatmap layer list, etc.).

## Formulas (for reference)

Per-token routing probability at layer k for the t-th token of document i in language ℓ:

  p^k_{i,t}(ℓ) ∈ Δ^{E-1}   (post-softmax over E=64 experts)

Document- and language-level expert usage:

  q^k_i(ℓ) = (1 / T_i) Σ_t p^k_{i,t}(ℓ)
  q^k(ℓ)   = (1 / N_ℓ) Σ_i q^k_i(ℓ)

Router entropy:

  H_k(ℓ) = − Σ_e q^k(ℓ)[e] · log q^k(ℓ)[e]

Pairwise cross-lingual divergence:

  JSD_k(ℓ_i, ℓ_j) = JSD( q^k(ℓ_i), q^k(ℓ_j) )

Activation gap (for expert selection):

  For each expert e in layer k:
    a_ℓ(e) = C_{ℓ,k}(e) / Σ_{e'} C_{ℓ,k}(e')         (normalized counts)
    (ℓ₁, ℓ₂) = top-2 languages by a_ℓ(e)
    gap(e) = a_{ℓ₁}(e) − a_{ℓ₂}(e)
  Select experts where gap(e) ≥ α (paper: α = 0.01)
  Augment with top-k shared experts by mean activation across all HR langs (paper: k = 5)

Note: `collect_routing.py` collects post-softmax top-k *counts* rather than the full probability vector. With OLMoE's top-8 router each token contributes 8 to the layer total, which is what `analysis.py` normalizes when it computes `q^k(ℓ)`.
