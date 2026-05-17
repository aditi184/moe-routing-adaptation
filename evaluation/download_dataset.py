#!/usr/bin/env python3
"""
Download and cache a HuggingFace dataset locally for offline use.

This script downloads a dataset from HuggingFace and caches it locally
so it can be used on nodes without internet access.

Usage:
    # Download a specific config (for multiblimp_eng):
    python download_dataset.py \
        --dataset_name jumelet/multiblimp \
        --config eng \
        --cache_dir /datasets_cache
    
    # Download all configs:
    python download_dataset.py \
        --dataset_name jumelet/multiblimp \
        --download_all \
        --cache_dir /datasets_cache
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def download_dataset(dataset_name: str, cache_dir: str, config_name: str = None, download_all_configs: bool = False):
    """
    Download a dataset from HuggingFace and cache it locally.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "jumelet/multiblimp")
        cache_dir: Local directory path to cache the dataset
        config_name: Optional config name to download (e.g., "eng" for multiblimp)
        download_all_configs: If True, download all available configs
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Downloading dataset: {dataset_name}")
    if config_name:
        log.info(f"Config: {config_name}")
    if download_all_configs:
        log.info("Downloading all available configs...")
    log.info(f"Cache directory: {cache_path}")
    
    try:
        # Set the cache directory environment variable
        os.environ['HF_DATASETS_CACHE'] = str(cache_path)
        
        if download_all_configs:
            # Get all available configs first
            from datasets import get_dataset_config_names
            try:
                configs = get_dataset_config_names(dataset_name)
                log.info(f"Found {len(configs)} configs: {configs[:10]}{'...' if len(configs) > 10 else ''}")
            except Exception as e:
                log.warning(f"Could not list configs: {e}")
                log.info("Trying to download without config (may fail if config is required)")
                configs = [None]
            else:
                # Download each config
                for config in configs:
                    log.info(f"Downloading config: {config}")
                    try:
                        dataset = load_dataset(dataset_name, config)
                        log.info(f"✓ Config '{config}' downloaded successfully")
                    except Exception as e:
                        log.warning(f"✗ Failed to download config '{config}': {e}")
        else:
            # Download the dataset (this will cache it)
            log.info("Loading dataset (this will download and cache it)...")
            if config_name:
                dataset = load_dataset(dataset_name, config_name)
            else:
                dataset = load_dataset(dataset_name)
            
            log.info(f"✓ Dataset successfully downloaded and cached")
            log.info(f"✓ Dataset splits: {list(dataset.keys())}")
        
        # Show cache location
        # The actual cache is usually in cache_path/datasets/<dataset_name>
        dataset_cache = cache_path / "datasets" / dataset_name.replace("/", "___")
        if dataset_cache.exists():
            log.info(f"✓ Dataset cached at: {dataset_cache}")
            # Show size
            total_size = sum(f.stat().st_size for f in dataset_cache.rglob('*') if f.is_file())
            log.info(f"✓ Cache size: {total_size / (1024**3):.2f} GB")
        
        return True
    except Exception as e:
        log.error(f"✗ Failed to download dataset: {e}")
        # If config is missing, show helpful message
        if "Config name is missing" in str(e) or "Please pick one" in str(e):
            log.error("")
            log.error("This dataset requires a config name. Use --config <config_name>")
            log.error("For multiblimp_eng, use: --config eng")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache a HuggingFace dataset locally for offline use"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'jumelet/multiblimp')"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Local directory path to cache the dataset (default: uses HF_DATASETS_CACHE or ~/.cache/huggingface/datasets)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config name to download (e.g., 'eng' for multiblimp_eng). Required if dataset has multiple configs."
    )
    parser.add_argument(
        "--download_all",
        action="store_true",
        help="Download all available configs (useful for datasets with multiple language configs)"
    )
    
    args = parser.parse_args()
    
    # Use provided cache_dir or default
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        # Use default HuggingFace cache location
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        log.info(f"No cache_dir specified, using default: {cache_dir}")
    
    download_dataset(args.dataset_name, cache_dir, config_name=args.config, download_all_configs=args.download_all)
    
    log.info("")
    log.info("=" * 80)
    log.info("Next steps:")
    log.info("=" * 80)
    log.info(f"1. Set the cache directory environment variable before running lm_eval:")
    log.info(f"   export HF_DATASETS_CACHE={cache_dir}")
    log.info("")
    log.info("2. Or set it in your command:")
    log.info(f"   HF_DATASETS_CACHE={cache_dir} lm_eval --model hf ...")
    log.info("")
    log.info("3. The dataset will be loaded from the cache automatically")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

