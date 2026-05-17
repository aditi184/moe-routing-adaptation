#!/usr/bin/env python3
"""
Download and save a HuggingFace tokenizer locally for offline use.

This script downloads a tokenizer from HuggingFace and saves it to a local directory
so it can be used on nodes without internet access.

Usage:
    python download_tokenizer.py \
        --model_id allenai/OLMoE-1B-7B-0924-Instruct \
        --output_dir s/OLMoE-1B-7B-0924-Instruct
"""

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def download_tokenizer(model_id: str, output_dir: str):
    """
    Download a tokenizer from HuggingFace and save it locally.
    
    Args:
        model_id: HuggingFace model ID (e.g., "allenai/OLMoE-1B-7B-0924-Instruct")
        output_dir: Local directory path to save the tokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Downloading tokenizer from: {model_id}")
    log.info(f"Saving to: {output_path}")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save tokenizer locally
        tokenizer.save_pretrained(str(output_path))
        
        log.info(f"✓ Tokenizer successfully downloaded and saved to {output_path}")
        log.info(f"✓ Tokenizer files:")
        for file in sorted(output_path.glob("*")):
            if file.is_file():
                log.info(f"    - {file.name}")
        
        return True
    except Exception as e:
        log.error(f"✗ Failed to download tokenizer: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and save a HuggingFace tokenizer locally for offline use"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'allenai/OLMoE-1B-7B-0924-Instruct')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory path to save the tokenizer"
    )
    
    args = parser.parse_args()
    
    download_tokenizer(args.model_id, args.output_dir)
    
    log.info("")
    log.info("=" * 80)
    log.info("Next steps:")
    log.info("=" * 80)
    log.info(f"1. Use the local tokenizer path in your commands:")
    log.info(f"   tokenizer={args.output_dir}")
    log.info("")
    log.info("2. Example for lm_eval:")
    log.info(f"   lm_eval --model hf \\")
    log.info(f"     --model_args pretrained=...,tokenizer={args.output_dir},dtype=auto \\")
    log.info(f"     ...")
    log.info("=" * 80)


if __name__ == "__main__":
    main()




