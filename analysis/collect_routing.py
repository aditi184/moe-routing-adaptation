"""Run an OLMoE forward pass and dump per-(language, layer, expert) token counts.

Distilled from the original ``OLMoE/scripts/expert_routing_analysis.py`` +
``OLMoE/scripts/utils.py``. Pure-PyTorch, no wandb, no global state.

Output format (one pickle per model, consumed by ``analysis.py``)::

    counts[lang: str][layer: int] -> collections.Counter({expert_id: count, ...})

The Counter values are *token-level activation counts* aggregated across all
documents for that language. With OLMoE's top-8 router this means each token
contributes 8 to the layer's total count (one per chosen expert).

Usage
-----
    python -m analysis.collect_routing \\
        --model         allenai/OLMoE-1B-7B-0924 \\
        --data-dir      ${DATA_DIR}/validation_text \\
        --output        analysis/data/olmoe_base_counts.pkl \\
        --languages     en,ar,cs,es,fi,hi,ru,nl,ur,sk,ca,et,mr,uk \\
        --max-tokens    1024000 \\
        --batch-tokens  2048

For ``--data-dir`` provide a folder with one ``{lang}.txt`` per language code.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, OlmoeForCausalLM


def load_model(model_name: str, device: str):
    """Return a ready-to-eval OLMoE model + its tokenizer."""
    model = OlmoeForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def iter_token_windows(tokens: list[int], batch_tokens: int):
    """Yield non-overlapping length-``batch_tokens`` windows; drop the tail."""
    for start in range(0, len(tokens) - batch_tokens + 1, batch_tokens):
        yield tokens[start:start + batch_tokens]


@torch.no_grad()
def collect_one_language(
    model, tokenizer, text: str, *,
    batch_tokens: int, max_tokens: int, device: str, top_k: int,
) -> dict[int, Counter]:
    """Return ``{layer_index: Counter(expert_id -> activation_count)}`` for one language."""
    layer_counts: dict[int, Counter] = defaultdict(Counter)
    tokens = tokenizer(text, truncation=False)["input_ids"]
    processed = 0

    pbar = tqdm(iter_token_windows(tokens, batch_tokens),
                total=min(len(tokens), max_tokens) // batch_tokens,
                desc="forward", leave=False)
    for window in pbar:
        input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=input_ids, output_router_logits=True)
        # router_logits is a tuple of (num_layers,) tensors, each (B*T, num_experts).
        for layer_idx, logits in enumerate(out.router_logits):
            top_experts = torch.topk(logits, top_k, dim=-1).indices  # (B*T, top_k)
            for expert_id in top_experts.flatten().tolist():
                layer_counts[layer_idx][expert_id] += 1
        processed += len(window)
        if processed >= max_tokens:
            break
    return dict(layer_counts)


def collect_all_languages(
    model_name: str, data_dir: Path, languages: list[str], *,
    batch_tokens: int = 2048, max_tokens: int = 1_024_000,
    top_k: int = 8, device: str | None = None,
) -> dict[str, dict[int, Counter]]:
    """Run ``collect_one_language`` for each language and stitch the results."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name, device)

    results: dict[str, dict[int, Counter]] = {}
    for lang in tqdm(languages, desc="languages"):
        text_path = data_dir / f"{lang}.txt"
        if not text_path.exists():
            print(f"[warn] missing {text_path}, skipping")
            continue
        text = text_path.read_text(encoding="utf-8")
        results[lang] = collect_one_language(
            model, tokenizer, text,
            batch_tokens=batch_tokens, max_tokens=max_tokens,
            device=device, top_k=top_k,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HF model id, e.g. allenai/OLMoE-1B-7B-0924 or your OLMoE-M7 checkpoint path")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with {lang}.txt held-out text files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output pickle path")
    parser.add_argument("--languages", required=True,
                        help="Comma-separated language codes (must match filenames in --data-dir)")
    parser.add_argument("--max-tokens", type=int, default=1_024_000,
                        help="Cap per language (default ~500 docs * 2k tokens)")
    parser.add_argument("--batch-tokens", type=int, default=2048,
                        help="Tokens per forward pass (default OLMoE seq len)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of experts per token to count (OLMoE default 8)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (autodetected by default)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    counts = collect_all_languages(
        args.model, args.data_dir, languages,
        batch_tokens=args.batch_tokens, max_tokens=args.max_tokens,
        top_k=args.top_k, device=args.device,
    )

    with open(args.output, "wb") as f:
        pickle.dump(counts, f)
    print(f"wrote {args.output}  ({len(counts)} languages)")


if __name__ == "__main__":
    main()
