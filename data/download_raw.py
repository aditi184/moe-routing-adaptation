# python download_raw.py --languages uk mr nl ca et sk ur --samples 200000 --output_dir /low_resource

import argparse
from datasets import load_dataset
import json
import os
from itertools import islice

def create_raw_record(example, idx):
    return {
        "id": idx,
        "text": example.get("text", ""),
        "source": example.get("source", ""),
        "timestamp": example.get("timestamp", ""),
        "url": example.get("url", ""),
    }

def download_raw(languages, output_dir, num_samples):
    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:
        print(f"Downloading first {num_samples} samples for {lang}")
        ds_iter = iter(load_dataset("uonlp/CulturaX", lang, streaming=True)["train"])
        samples = islice(ds_iter, num_samples)

        out_path = os.path.join(output_dir, f"raw_{lang}.jsonl")
        with open(out_path, "w", encoding="utf8") as f:
            for idx, ex in enumerate(samples):
                record = create_raw_record(ex, idx)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved raw data for {lang} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()

    download_raw(args.languages, args.output_dir, args.samples)
