"""Collect per-checkpoint Belebele JSONs into a single CSV.

Usage:
    python evaluation/consolidate_belebele.py \
        --results-dir ${RESULTS_DIR}/belebele_4_shot --shot 4-shot \
        --results-dir ${RESULTS_DIR}/belebele_zero_shot --shot 0-shot \
        --output results/belebele_consolidated.csv

You can pass multiple ``--results-dir`` / ``--shot`` pairs (in order) to merge
zero-shot and 4-shot runs into a single CSV. One row per (Model, Shot)
combination; rows are deduplicated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

ALL_LANGS = [
    "eng", "arb", "ces", "spa", "fin", "hin", "rus",
    "nld", "urd", "slk", "cat", "est", "mar", "ukr",
]
HIGH_RESOURCE = ["eng", "arb", "ces", "spa", "fin", "hin", "rus"]
LOW_RESOURCE = ["nld", "urd", "slk", "cat", "est", "mar", "ukr"]

LANG_NAME_TO_CODE = {
    "catalan": "cat", "dutch": "nld", "estonian": "est",
    "marathi": "mar", "slovak": "slk", "ukrainian": "ukr", "urdu": "urd",
}


def extract_lang_from_task(task: str) -> str | None:
    """``belebele_eng_Latn`` -> ``eng``."""
    if not task.startswith("belebele_"):
        return None
    rest = task[len("belebele_"):]
    parts = rest.split("_")
    return parts[0] if parts else None


def extract_model_name(dir_path: Path) -> str:
    """Best-effort canonicalization of a checkpoint dir name."""
    dir_name = dir_path.name
    clean = dir_name.lstrip("_").replace("__", "_")
    patterns = [
        (r"peft_stage2_(\w+)_seft_shared_lr(.+)",
         lambda m: f"{LANG_NAME_TO_CODE.get(m.group(1), m.group(1))}_seft_shared_lr{m.group(2)}"),
        (r"peft_stage2_(\w+)_lr_(.+)",
         lambda m: f"{LANG_NAME_TO_CODE.get(m.group(1), m.group(1))}_lr_{m.group(2)}"),
        (r"peft_stage2_(\w+)_all_experts_lr(.+)",
         lambda m: f"{LANG_NAME_TO_CODE.get(m.group(1), m.group(1))}_all_experts_lr{m.group(2)}"),
        (r"full_(\w+)",
         lambda m: f"full_{LANG_NAME_TO_CODE.get(m.group(1), m.group(1))}"),
    ]
    for pattern, formatter in patterns:
        match = re.search(pattern, clean)
        if match:
            return formatter(match)
    return clean


def process_results_dir(base_dir: Path, shot_type: str) -> list[dict]:
    """Walk one results dir, return one row per checkpoint subdir."""
    all_json = glob.glob(str(base_dir / "**" / "results_*.json"), recursive=True)
    dir_to_files: dict[str, list[str]] = {}
    for jf in all_json:
        rel = os.path.relpath(jf, base_dir)
        first = rel.split(os.sep)[0]
        dir_to_files.setdefault(str(base_dir / first), []).append(jf)

    rows: list[dict] = []
    for result_dir, json_files in dir_to_files.items():
        if not json_files:
            continue
        model_name = extract_model_name(Path(result_dir))
        latest = sorted(json_files)[-1]
        try:
            with open(latest) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  error reading {latest}: {e}")
            continue

        acc_norm: dict[str, float] = {}
        for task, vals in data.get("results", {}).items():
            lang = extract_lang_from_task(task)
            if lang and "acc_norm,none" in vals:
                acc_norm[lang] = vals["acc_norm,none"]

        if not acc_norm:
            continue

        row: dict = {"Model": model_name, "Shot": shot_type}
        for lang in ALL_LANGS:
            row[lang] = round(acc_norm[lang], 3) if lang in acc_norm else ""

        def avg(langs: list[str]) -> float | str:
            vals = [acc_norm[l] for l in langs if l in acc_norm]
            return round(sum(vals) / len(vals), 3) if vals else ""

        row["High Resource Average"] = avg(HIGH_RESOURCE)
        row["Low Resource Average"] = avg(LOW_RESOURCE)
        row["Overall Avg"] = avg(ALL_LANGS)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, action="append", required=True,
                        help="Directory of per-checkpoint lm_eval outputs; may be passed multiple times")
    parser.add_argument("--shot", action="append", required=True,
                        help="Label for each --results-dir (e.g. '4-shot' or '0-shot'). One per --results-dir.")
    parser.add_argument("--output", type=Path, required=True, help="CSV path (overwritten)")
    args = parser.parse_args()

    if len(args.results_dir) != len(args.shot):
        raise SystemExit("Got "
                         f"{len(args.results_dir)} --results-dir and {len(args.shot)} --shot; counts must match")

    all_rows: list[dict] = []
    for base_dir, shot in zip(args.results_dir, args.shot):
        if not base_dir.exists():
            print(f"skip {base_dir} (does not exist)")
            continue
        print(f"Processing {base_dir} ({shot})")
        rows = process_results_dir(base_dir, shot)
        all_rows.extend(rows)
        print(f"  {len(rows)} models")

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for row in all_rows:
        key = (row["Model"], row["Shot"])
        if key in seen:
            print(f"  duplicate skipped: {key}")
            continue
        seen.add(key)
        deduped.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "Shot", *ALL_LANGS,
                  "High Resource Average", "Low Resource Average", "Overall Avg"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)
    print(f"Wrote {len(deduped)} rows to {args.output}")


if __name__ == "__main__":
    main()
