"""Routing-analysis figures and activation-gap expert selection.

All inputs come from the CLI; nothing is hard-coded. Run ``python -m
analysis.analysis --help`` (or ``python analysis/analysis.py --help``) for
the list of subcommands.

Inputs
------
* ``--counts`` pickles produced by ``analysis/collect_routing.py``. Each has
  the shape::

      counts[lang: str][layer: int] -> collections.Counter({expert_id: count, ...})

  with ``NUM_LAYERS`` MoE layers and ``NUM_EXPERTS`` experts per layer
  (defaults: 16 / 64, matching OLMoE-1B-7B-0924).

* ``--vocab-overlap`` (only for the ``jsd-vs-vocab`` subcommand): a square CSV
  of pairwise token-vocab overlap, rows and columns indexed by language
  *full name* (see ``LANG_FULL``).

Subcommands
-----------
    entropy        Fig. 2: English vs. avg-non-English routing entropy per layer.
    jsd-heatmap    Fig. 4: pairwise JSD heatmap at one layer.
    jsd-per-layer  Fig. 5: average pairwise JSD over layers for two model stages.
    jsd-vs-vocab   Fig. 6: scatter of pairwise JSD vs token-vocab overlap.
    specialization Normalized IG(L;E)/H(L) per layer for one or more models.
    select-experts Activation-gap expert selection -> selected_experts.json.
    export-jsd     Dump all-pair JSD at one layer to CSV.
    all            Run every subcommand with sensible defaults.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, linregress

NUM_LAYERS = 16
NUM_EXPERTS = 64

HIGH_RESOURCE = ["en", "ar", "cs", "es", "fi", "hi", "ru"]
LOW_RESOURCE = ["nl", "ur", "sk", "ca", "et", "mr", "uk"]
ALL_LANGS = HIGH_RESOURCE + LOW_RESOURCE

LANG_FULL = {
    "en": "english", "ar": "arabic", "cs": "czech", "es": "spanish",
    "fi": "finnish", "hi": "hindi", "ru": "russian",
    "nl": "dutch", "ur": "urdu", "sk": "slovak", "ca": "catalan",
    "et": "estonian", "mr": "marathi", "uk": "ukrainian",
}


# -------- IO helpers ------------------------------------------------------

def load_counts(path: Path) -> dict[str, dict[int, Counter]]:
    """Load a routing-count pickle. See module docstring for layout."""
    with open(path, "rb") as f:
        return pickle.load(f)


def to_distribution(counts: Counter, num_experts: int = NUM_EXPERTS) -> np.ndarray:
    """Normalize a Counter(expert_id -> count) into a length-E probability vector."""
    dist = np.zeros(num_experts)
    total = sum(counts.values()) or 1
    for expert_id, value in counts.items():
        if expert_id < num_experts:
            dist[expert_id] = value / total
    return dist


def lang_layer_matrix(counts: dict[int, Counter]) -> np.ndarray:
    """Stack a single language's per-layer Counters into a (num_layers, num_experts) matrix."""
    matrix = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    for layer in range(NUM_LAYERS):
        matrix[layer] = to_distribution(counts[layer])
    return matrix


# -------- Entropy (Fig. 2) ------------------------------------------------

def entropy_per_layer(counts_one_lang: dict[int, Counter]) -> np.ndarray:
    """H_k(l) = - sum_e q^k(l)[e] * log q^k(l)[e], for k = 0..NUM_LAYERS-1."""
    matrix = lang_layer_matrix(counts_one_lang)
    return np.array([entropy(matrix[layer]) for layer in range(NUM_LAYERS)])


def plot_entropy_english_vs_avg(
    base: dict, stage1: dict,
    english_lang: str = "en", stage1_label: str = "OLMoE-M7",
    save_path: Path | None = None,
):
    """Reproduce the English-vs-non-English entropy curves for two models."""
    layers = np.arange(1, NUM_LAYERS + 1)

    def en_and_avg(data):
        en_h = entropy_per_layer(data[english_lang])
        others = [l for l in data if l != english_lang]
        avg_h = np.mean(np.stack([entropy_per_layer(data[l]) for l in others]), axis=0)
        return en_h, avg_h

    en_base, avg_base = en_and_avg(base)
    en_stage1, avg_stage1 = en_and_avg(stage1)

    sns.set_context("talk")
    plt.figure(figsize=(7, 7))
    plt.plot(layers, en_base, linewidth=2.5, marker="o", label="English (Base)", color="#1f77b4")
    plt.plot(layers, avg_base, linewidth=2.5, marker="o", label="Avg non-English (Base)", color="#6baed6")
    plt.plot(layers, en_stage1, linestyle="--", linewidth=2.5, marker="o",
             label=f"English ({stage1_label})", color="#d62728")
    plt.plot(layers, avg_stage1, linestyle="--", linewidth=2.5, marker="o",
             label=f"Avg non-English ({stage1_label})", color="#fb6a4a")

    plt.xlabel("Layer Number")
    plt.ylabel("Mean Routing Entropy")
    plt.xticks([1, 6, 11, 16])
    plt.xlim(1, NUM_LAYERS)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()


# -------- Pairwise JSD heatmap (Fig. 4) -----------------------------------

def pairwise_jsd(counts1: Counter, counts2: Counter) -> float:
    """Squared Jensen-Shannon divergence between two expert distributions."""
    expert_ids = sorted(set(counts1) | set(counts2))
    total1 = sum(counts1.values()) or 1
    total2 = sum(counts2.values()) or 1
    dist1 = np.array([counts1.get(e, 0) / total1 for e in expert_ids])
    dist2 = np.array([counts2.get(e, 0) / total2 for e in expert_ids])
    return jensenshannon(dist1, dist2) ** 2


def plot_jsd_heatmap(
    data: dict, languages: list[str], layer: int,
    title: str, save_path: Path | None = None,
):
    """Lower-triangular heatmap of pairwise JSD between languages at one layer."""
    n = len(languages)
    jsd = np.zeros((n, n))
    for i, lang_i in enumerate(languages):
        for j, lang_j in enumerate(languages):
            if i == j:
                continue
            jsd[i, j] = pairwise_jsd(data[lang_i][layer], data[lang_j][layer])

    mask = np.triu(np.ones_like(jsd, dtype=bool), k=1)

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2, style="white")
    sns.heatmap(
        jsd, mask=mask, annot=True, fmt=".2f",
        xticklabels=languages, yticklabels=languages,
        cmap="GnBu_r", vmin=0, vmax=0.4,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Jensen-Shannon Divergence"},
    )
    plt.title(f"{title} - Layer {layer}", fontsize=18, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------- Average JSD per layer (Fig. 5) ----------------------------------

def avg_pairwise_jsd_per_layer(
    data: dict, languages: list[str], skip_langs: list[str] | None = None,
) -> np.ndarray:
    """Mean pairwise JSD over (languages choose 2) at each layer."""
    skip = set(skip_langs or [])
    langs = [l for l in languages if l not in skip]
    out = np.zeros(NUM_LAYERS)
    for layer in range(NUM_LAYERS):
        pairs = [
            pairwise_jsd(data[a][layer], data[b][layer])
            for a, b in combinations(langs, 2)
        ]
        out[layer] = float(np.mean(pairs))
    return out


def plot_avg_jsd_two_stages(
    base: dict, stage1: dict, languages: list[str],
    save_path: Path | None = None,
):
    """Layer-wise average pairwise JSD for two model stages, with/without English."""
    layers = np.arange(NUM_LAYERS)
    series = {
        "OLMoE-Base": avg_pairwise_jsd_per_layer(base, languages),
        "OLMoE-M7": avg_pairwise_jsd_per_layer(stage1, languages),
        "OLMoE-Base (excl. en)": avg_pairwise_jsd_per_layer(base, languages, skip_langs=["en"]),
        "OLMoE-M7 (excl. en)": avg_pairwise_jsd_per_layer(stage1, languages, skip_langs=["en"]),
    }
    styles = [
        ("OLMoE-Base", "o", "-"),
        ("OLMoE-M7", "s", "-"),
        ("OLMoE-Base (excl. en)", "^", "--"),
        ("OLMoE-M7 (excl. en)", "v", "--"),
    ]

    plt.figure(figsize=(8, 5))
    for label, marker, ls in styles:
        plt.plot(layers, series[label], marker=marker, linestyle=ls, label=label)
    plt.xlabel("Layer")
    plt.ylabel("Average pairwise JSD")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------- JSD vs vocab overlap scatter (Fig. 6) ---------------------------

def jsd_vs_vocab_overlap_scatter(
    data: dict, vocab_overlap_csv: Path, layer: int = 15,
    save_path: Path | None = None,
):
    """Scatter JSD against pairwise token-vocab overlap for HR-LR and HR-HR pairs."""
    vocab_df = pd.read_csv(vocab_overlap_csv, index_col=0)

    high_low = list(product(HIGH_RESOURCE, LOW_RESOURCE))
    high_high = list(combinations(HIGH_RESOURCE, 2))

    rows = []
    for lang1, lang2 in high_low + high_high:
        f1, f2 = LANG_FULL[lang1], LANG_FULL[lang2]
        try:
            overlap = vocab_df.loc[f1, f2]
        except KeyError:
            overlap = vocab_df.loc[f2, f1]
        jsd = pairwise_jsd(data[lang1][layer], data[lang2][layer])
        pair_type = "High-High" if (lang1 in HIGH_RESOURCE and lang2 in HIGH_RESOURCE) else "High-Low"
        rows.append({"lang1": lang1, "lang2": lang2,
                     "vocab_overlap": overlap, "jsd": jsd, "pair_type": pair_type})

    df = pd.DataFrame(rows).sort_values("vocab_overlap")

    plt.figure(figsize=(14, 9))
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"High-Low": "#2E86AB", "High-High": "#A23B72"}
    labels = {"High-Low": "HR-LR", "High-High": "HR-HR"}
    for pair_type, color in colors.items():
        sub = df[df["pair_type"] == pair_type]
        plt.scatter(sub["vocab_overlap"], sub["jsd"],
                    s=150, color=color, alpha=0.75, label=labels[pair_type], zorder=3)

    sister_pairs = [("ru", "uk"), ("hi", "mr"), ("en", "nl"), ("es", "ca"),
                    ("fi", "et"), ("cs", "sk"), ("ar", "ur")]
    for lang1, lang2 in sister_pairs:
        row = df[((df["lang1"] == lang1) & (df["lang2"] == lang2)) |
                 ((df["lang1"] == lang2) & (df["lang2"] == lang1))]
        if len(row) == 0:
            continue
        x, y = row.iloc[0]["vocab_overlap"], row.iloc[0]["jsd"]
        plt.scatter(x, y, s=500, marker="*", color="#F18F01", zorder=6, linewidths=3)
        plt.annotate(f"{lang1.capitalize()}-{lang2.capitalize()} ({x * 100:.1f}%)",
                     xy=(x, y), xytext=(5, 5), textcoords="offset points",
                     fontsize=18, fontweight="bold", color="#1a1a1a")

    slope, intercept, r, _, _ = linregress(df["vocab_overlap"], df["jsd"])
    x_line = np.linspace(df["vocab_overlap"].min(), df["vocab_overlap"].max(), 100)
    plt.plot(x_line, slope * x_line + intercept, color="#333333", linestyle="--",
             linewidth=4.5, alpha=0.85, label=f"Linear fit (r={r:.3f})", zorder=2)

    plt.legend(fontsize=22, loc="best")
    plt.xlabel("Token Vocabulary Overlap", fontsize=28, fontweight="bold")
    plt.ylabel("Jensen-Shannon Divergence", fontsize=28, fontweight="bold")
    plt.title(f"JSD vs Vocab Overlap - Layer {layer}", fontsize=30, fontweight="bold")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(bottom=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"r(all)      = {df['vocab_overlap'].corr(df['jsd']):.3f}  (n={len(df)})")
    for pt in ["High-Low", "High-High"]:
        sub = df[df["pair_type"] == pt]
        print(f"r({pt:<9}) = {sub['vocab_overlap'].corr(sub['jsd']):.3f}  (n={len(sub)})")
    return df


def export_pairwise_jsd_csv(data: dict, layer: int, out_csv: Path) -> pd.DataFrame:
    """Cache all-pair JSD values at a single layer for downstream plotting."""
    langs = sorted(data.keys())
    rows = [{"lang1": a, "lang2": b, "jsd": pairwise_jsd(data[a][layer], data[b][layer])}
            for a, b in combinations(langs, 2)]
    df = pd.DataFrame(rows).sort_values("jsd")
    df.to_csv(out_csv, index=False)
    return df


# -------- Language-specialization score per layer -------------------------

def _safe_entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def layer_specialization(
    data: dict, layer: int, exclude_langs: list[str] | None = None,
) -> float:
    """Normalized information gain IG(L; E) / H(L) at this layer.

    Treats per-(language, expert) counts as a contingency table and asks how
    much of H(L) is explained by knowing the expert that fired.
    """
    skip = set(exclude_langs or [])
    langs = [l for l in data if l not in skip]
    experts = sorted({e for l in langs for e in data[l][layer]})

    counts_le = np.zeros((len(langs), len(experts)))
    for i, lang in enumerate(langs):
        for j, expert in enumerate(experts):
            counts_le[i, j] = data[lang][layer].get(expert, 0)

    total = counts_le.sum()
    if total == 0:
        return 0.0

    p_l = counts_le.sum(axis=1) / total
    p_e = counts_le.sum(axis=0) / total
    p_l_given_e = np.divide(
        counts_le, counts_le.sum(axis=0, keepdims=True),
        where=counts_le.sum(axis=0, keepdims=True) > 0,
    )

    h_l = _safe_entropy_bits(p_l)
    h_l_given_e = sum(p_e[j] * _safe_entropy_bits(p_l_given_e[:, j])
                      for j in range(len(experts)))
    return float((h_l - h_l_given_e) / h_l) if h_l > 0 else 0.0


def plot_specialization_curves(
    runs: list[tuple[str, dict]], save_path: Path | None = None,
):
    """One curve per model: normalized language-specialization score vs layer."""
    plt.figure(figsize=(12, 7))
    colors = ["#0f5a9e", "#e4572e", "#84cfc1", "#fec260", "#6a4c93"]
    for i, (label, data) in enumerate(runs):
        scores = [layer_specialization(data, layer) for layer in range(NUM_LAYERS)]
        plt.plot(range(NUM_LAYERS), scores, marker="o", linewidth=3,
                 markersize=10, label=label, color=colors[i % len(colors)])
    plt.xlabel("Layer", fontsize=20)
    plt.ylabel("Language Specialization Score", fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------- Activation-gap expert selection (Fig. 3) ------------------------

def top_languages_per_expert(
    data: dict, layer: int, languages: list[str], top_k: int = 2,
) -> pd.DataFrame:
    """For every expert at one layer, return its top-k activating languages."""
    experts = sorted({e for l in languages for e in data[l][layer]})
    rows = []
    for expert in experts:
        acts = []
        for lang in languages:
            counts = data[lang][layer]
            total = sum(counts.values()) or 1
            acts.append((lang, counts.get(expert, 0) / total))
        acts.sort(key=lambda x: x[1], reverse=True)
        row = {"expert": expert}
        for rank, (lang, act) in enumerate(acts[:top_k], start=1):
            row[f"top_{rank}_lang"] = lang
            row[f"top_{rank}_act"] = act
        rows.append(row)
    return pd.DataFrame(rows)


def select_experts_by_activation_gap(
    data: dict, languages: list[str], layer: int, alpha: float = 0.01,
) -> dict[str, list[int]]:
    """Per-language list of experts whose (top1 - top2 across languages) >= alpha."""
    df = top_languages_per_expert(data, layer, languages, top_k=2)
    df["gap"] = df["top_1_act"] - df["top_2_act"]
    selected: dict[str, list[int]] = {l: [] for l in languages}
    for _, row in df[df["gap"] >= alpha].iterrows():
        selected[row["top_1_lang"]].append(int(row["expert"]))
    return selected


def shared_experts_by_mean_activation(
    data: dict, languages: list[str], layer: int, k: int = 5,
) -> list[int]:
    """Top-k experts by mean normalized activation across all `languages`."""
    experts = sorted({e for l in languages for e in data[l][layer]})
    mean_acts = []
    for expert in experts:
        acts = []
        for lang in languages:
            counts = data[lang][layer]
            total = sum(counts.values()) or 1
            acts.append(counts.get(expert, 0) / total)
        mean_acts.append((expert, float(np.mean(acts))))
    mean_acts.sort(key=lambda x: x[1], reverse=True)
    return [expert for expert, _ in mean_acts[:k]]


def build_selected_experts_json(
    data: dict, hr_languages: list[str], lr_languages: list[str],
    target_layers: tuple[int, ...] = (14, 15),
    alpha: float = 0.01, ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Assemble the selected_experts.json structure used by the SEFT/SSFT recipes.

    The HR pool defines the activation-gap selection AND the shared-expert pool;
    LR target languages are then routed through one of the HR experts via the
    parent-language mapping established in the paper.
    """
    seft_per_layer = {
        str(layer): select_experts_by_activation_gap(data, hr_languages, layer, alpha)
        for layer in target_layers
    }
    shared_per_layer = {
        str(layer): {f"k{k}": shared_experts_by_mean_activation(data, hr_languages, layer, k=k)
                     for k in ks}
        for layer in target_layers
    }

    out = {
        "_meta": {
            "alpha": alpha,
            "target_layers": list(target_layers),
            "hr_languages": hr_languages,
        },
        "shared_experts": {
            f"k{k}": {str(layer): shared_per_layer[str(layer)][f"k{k}"] for layer in target_layers}
            for k in ks
        },
    }
    for lr in lr_languages:
        # The paper maps every LR language to its HR parent; replace this lookup
        # with the actual parent (see analysis/selected_experts.json _meta).
        parent = {"ca": "es", "et": "fi", "mr": "hi", "sk": "cs",
                  "uk": "ru", "ur": "ar", "nl": "en"}.get(lr)
        if parent is None:
            continue
        seft = {str(layer): seft_per_layer[str(layer)].get(parent, []) for layer in target_layers}
        out[lr] = {
            "seft": seft,
            **{
                f"ssft_k{k}": {
                    str(layer): seft[str(layer)] + shared_per_layer[str(layer)][f"k{k}"]
                    for layer in target_layers
                }
                for k in ks
            },
        }
    return out


# -------- CLI -------------------------------------------------------------

def _split_langs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _add_lang_args(p: argparse.ArgumentParser, *, default_langs: str | None = None):
    p.add_argument("--high-resource", type=_split_langs,
                   default=",".join(HIGH_RESOURCE) if default_langs is None else default_langs,
                   help=f"Comma-separated HR language codes (default: {','.join(HIGH_RESOURCE)})")
    p.add_argument("--low-resource", type=_split_langs,
                   default=",".join(LOW_RESOURCE),
                   help=f"Comma-separated LR language codes (default: {','.join(LOW_RESOURCE)})")


def cmd_entropy(args):
    base = load_counts(args.base)
    stage1 = load_counts(args.stage1)
    plot_entropy_english_vs_avg(base, stage1, english_lang=args.english,
                                stage1_label=args.stage1_label, save_path=args.out)


def cmd_jsd_heatmap(args):
    data = load_counts(args.counts)
    langs = args.languages or sorted(data.keys())
    plot_jsd_heatmap(data, langs, args.layer, title=args.title, save_path=args.out)


def cmd_jsd_per_layer(args):
    base = load_counts(args.base)
    stage1 = load_counts(args.stage1)
    plot_avg_jsd_two_stages(base, stage1, args.languages, save_path=args.out)


def cmd_jsd_vs_vocab(args):
    data = load_counts(args.counts)
    df = jsd_vs_vocab_overlap_scatter(data, args.vocab_overlap, layer=args.layer,
                                      save_path=args.out)
    if args.csv_out:
        df.to_csv(args.csv_out, index=False)


def cmd_specialization(args):
    runs = []
    for pair in args.run:
        if ":" not in pair:
            raise SystemExit(f"--run expects LABEL:PATH, got: {pair}")
        label, path = pair.split(":", 1)
        runs.append((label, load_counts(Path(path))))
    plot_specialization_curves(runs, save_path=args.out)


def cmd_select_experts(args):
    data = load_counts(args.counts)
    selected = build_selected_experts_json(
        data, hr_languages=args.high_resource, lr_languages=args.low_resource,
        target_layers=tuple(args.layers), alpha=args.alpha, ks=tuple(args.ks),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"wrote {args.out}")


def cmd_export_jsd(args):
    data = load_counts(args.counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_pairwise_jsd_csv(data, layer=args.layer, out_csv=args.out)
    print(f"wrote {args.out}")


def cmd_all(args):
    """Run every subcommand with one shared output directory."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_counts(args.base)
    stage1 = load_counts(args.stage1)
    languages = args.high_resource + args.low_resource

    plot_entropy_english_vs_avg(
        base, stage1, save_path=args.out_dir / "routing_entropy_en_vs_avg",
    )
    for layer in args.heatmap_layers:
        plot_jsd_heatmap(stage1, languages, layer, title="OLMoE-M7",
                         save_path=args.out_dir / f"jsd_heatmap_m7_layer{layer}.pdf")
        plot_jsd_heatmap(base, languages, layer, title="OLMoE-Base",
                         save_path=args.out_dir / f"jsd_heatmap_base_layer{layer}.pdf")
    plot_avg_jsd_two_stages(
        base, stage1, args.high_resource,
        save_path=args.out_dir / "avg_jsd_per_layer.pdf",
    )
    export_pairwise_jsd_csv(stage1, layer=args.scatter_layer,
                            out_csv=args.out_dir / f"jsd_layer{args.scatter_layer}_all_pairs.csv")
    if args.vocab_overlap is not None:
        jsd_vs_vocab_overlap_scatter(
            stage1, args.vocab_overlap, layer=args.scatter_layer,
            save_path=args.out_dir / "jsd_vs_vocab_overlap.pdf",
        )
    plot_specialization_curves(
        [("OLMoE-Base", base), ("OLMoE-M7", stage1)],
        save_path=args.out_dir / "specialization_score_per_layer.pdf",
    )
    selected = build_selected_experts_json(
        stage1, hr_languages=args.high_resource, lr_languages=args.low_resource,
        target_layers=tuple(args.layers), alpha=args.alpha,
    )
    with open(args.out_dir / "selected_experts.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"wrote outputs under {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # entropy
    p = sub.add_parser("entropy", help="Fig. 2 -- entropy: English vs. avg-non-English per layer")
    p.add_argument("--base", type=Path, required=True, help="Pickle of routing counts for the base model")
    p.add_argument("--stage1", type=Path, required=True, help="Pickle of counts for the continually-pretrained model")
    p.add_argument("--english", default="en", help="Language code treated as English (default: en)")
    p.add_argument("--stage1-label", default="OLMoE-M7", help="Legend label for the stage-1 model")
    p.add_argument("--out", type=Path, required=True, help="Output path (suffix .pdf+.png written)")
    p.set_defaults(func=cmd_entropy)

    # jsd-heatmap
    p = sub.add_parser("jsd-heatmap", help="Fig. 4 -- pairwise JSD heatmap at one layer")
    p.add_argument("--counts", type=Path, required=True, help="Pickle of routing counts for one model")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--languages", type=_split_langs, default=None,
                   help="Comma-separated subset; default = every language present in the pickle")
    p.add_argument("--title", default="", help="Plot title prefix (e.g. 'OLMoE-M7')")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_jsd_heatmap)

    # jsd-per-layer
    p = sub.add_parser("jsd-per-layer", help="Fig. 5 -- average pairwise JSD over layers")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--languages", type=_split_langs, default=HIGH_RESOURCE,
                   help=f"Comma-separated language codes (default: {','.join(HIGH_RESOURCE)})")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_jsd_per_layer)

    # jsd-vs-vocab
    p = sub.add_parser("jsd-vs-vocab", help="Fig. 6 -- JSD vs token-vocab overlap scatter")
    p.add_argument("--counts", type=Path, required=True, help="Routing pickle (typically stage-1 / M7)")
    p.add_argument("--vocab-overlap", type=Path, required=True,
                   help="Square CSV of pairwise vocab overlap, indexed by language full name")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--csv-out", type=Path, default=None, help="Optional: write per-pair JSD+overlap to CSV")
    p.set_defaults(func=cmd_jsd_vs_vocab)

    # specialization
    p = sub.add_parser("specialization", help="Normalized IG(L;E)/H(L) per layer")
    p.add_argument("--run", action="append", required=True,
                   metavar="LABEL:PATH",
                   help="One per model; e.g. --run 'OLMoE-Base:base.pkl' --run 'OLMoE-M7:m7.pkl'")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_specialization)

    # select-experts
    p = sub.add_parser("select-experts", help="Activation-gap expert selection -> selected_experts.json")
    p.add_argument("--counts", type=Path, required=True,
                   help="Routing pickle (use your OLMoE-M7 counts, not OLMoE-Base)")
    _add_lang_args(p)
    p.add_argument("--layers", type=int, nargs="+", default=[14, 15],
                   help="Layers to select experts from (default: 14 15)")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Activation-gap threshold (default: 0.01 = 1%%)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5],
                   help="Shared-expert pool sizes to compute (default: 1 3 5)")
    p.add_argument("--out", type=Path, default=Path("analysis/selected_experts.json"),
                   help="Output JSON path (default: analysis/selected_experts.json)")
    p.set_defaults(func=cmd_select_experts)

    # export-jsd
    p = sub.add_parser("export-jsd", help="Dump all-pair JSD at one layer to CSV")
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_export_jsd)

    # all
    p = sub.add_parser("all", help="Run every subcommand with shared output dir")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--vocab-overlap", type=Path, default=None,
                   help="Optional: enables the jsd-vs-vocab scatter")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--heatmap-layers", type=int, nargs="+", default=[13, 14, 15])
    p.add_argument("--scatter-layer", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.01)
    _add_lang_args(p)
    p.add_argument("--layers", type=int, nargs="+", default=[14, 15],
                   help="Layers for activation-gap selection (default: 14 15)")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
