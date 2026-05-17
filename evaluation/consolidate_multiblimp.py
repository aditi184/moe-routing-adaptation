"""Collect per-checkpoint MultiBLiMP JSONs into a single CSV.

Usage:
    python evaluation/consolidate_multiblimp.py \
        --results-dir ${RESULTS_DIR} \
        --output results/multiblimp_consolidated.csv

The script walks every immediate subdirectory of ``--results-dir`` and looks
for ``results_*.json`` files (the format ``lm_eval`` writes when run via
``evaluation/multiblimp.sh``). One CSV row per checkpoint dir.
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

# Renaming patterns for known sweep variants (kept as a best-effort cosmetic step).
SEFT_SHARED_PATTERN = re.compile(r"^(.+?)_seft_shared_lr(.+)$")


def load_existing_models(csv_path: Path) -> set[str]:
    """Read the Model column from an existing CSV (if any) to skip already-done runs."""
    out: set[str] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if "Model" in row:
                    out.add(row["Model"])
    return out


def process_result_dir(result_dir: Path, model_name: str) -> dict | None:
    """Pull all multiblimp_* acc_norm values out of one checkpoint's JSON results."""
    json_files = glob.glob(str(result_dir / "**" / "results_*.json"), recursive=True)
    if not json_files:
        return None

    acc_norm: dict[str, float] = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        for task, vals in data.get("results", {}).items():
            if not task.startswith("multiblimp_"):
                continue
            lang = task.replace("multiblimp_", "")
            if "acc_norm,none" in vals:
                acc_norm[lang] = vals["acc_norm,none"]

    if not acc_norm:
        return None

    row: dict = {"Model": model_name}
    for lang in ALL_LANGS:
        row[lang] = round(acc_norm[lang], 3) if lang in acc_norm else ""

    def avg(langs: list[str]) -> float | str:
        vals = [acc_norm[l] for l in langs if l in acc_norm]
        return round(sum(vals) / len(vals), 3) if vals else ""

    row["High Resource Average"] = avg(HIGH_RESOURCE)
    row["Low Resource Average"] = avg(LOW_RESOURCE)
    row["Overall Avg"] = avg(ALL_LANGS)
    return row


def canonical_model_name(dir_name: str) -> str:
    """Rewrite known sweep dir patterns to a tidier name; otherwise pass through."""
    m = SEFT_SHARED_PATTERN.match(dir_name)
    if m:
        lang_name, lr = m.group(1), m.group(2)
        return f"{LANG_NAME_TO_CODE.get(lang_name, lang_name)}_seft_shared_lr{lr}"
    return dir_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing per-checkpoint subdirs of lm_eval JSON outputs")
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV path to append to (created if missing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Truncate the output CSV instead of appending")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing = load_existing_models(args.output)
    print(f"Found {len(existing)} existing rows in {args.output}")

    rows: list[dict] = []
    for child in sorted(args.results_dir.iterdir()):
        if not child.is_dir():
            continue
        model_name = canonical_model_name(child.name)
        if model_name in existing:
            print(f"  skip {child.name} (already in CSV as {model_name})")
            continue
        row = process_result_dir(child, model_name)
        if row is not None:
            rows.append(row)
            existing.add(model_name)

    print(f"Collected {len(rows)} new models")

    file_exists = args.output.exists() and args.output.stat().st_size > 0
    fieldnames = ["Model", *ALL_LANGS, "High Resource Average",
                  "Low Resource Average", "Overall Avg"]
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} new rows to {args.output}")


if __name__ == "__main__":
    main()
