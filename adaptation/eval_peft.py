"""Evaluation script for HuggingFace OLMoE model checkpoints.

This script:
1. Loads a HuggingFace OLMoE model checkpoint
2. Evaluates on validation data (pre-tokenized .npy format)
3. Computes perplexity and other metrics
4. Supports multiple validation datasets

Usage (single dataset):
    python eval_peft_hf.py \
        --checkpoint_path /peft/peft_stage2_catalan/lr_4e4 \
        --validation_path /evaluation/tokenized/ca/part-0-00000.npy,/evaluation/tokenized/ar/part-0-00000.npy \
        --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
        --max_length 2048 \
        --num_samples 250 \
        --batch_size 4

Usage (multiple datasets):
    python eval_peft_hf.py \
        --checkpoint_path /path/to/checkpoint \
        --validation_path /path/to/dataset1.npy,/path/to/dataset2.npy,/path/to/dataset3.npy \
        --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
        --num_samples 250 \
        --batch_size 4

    python eval_peft_hf.py \
        --checkpoint_path /peft_stage2_slovak-4e4-all-experts/checkpoint-2290 \
        --validation_path /olmoe-data/test/uk/part-0-00000.npy,/olmoe-data/test/sk/part-0-00000.npy,/olmoe-data/test/ru/part-0-00000.npy,/olmoe-data/test/cs/part-0-00000.npy,/olmoe-data/test/en/part-0-00000.npy,/olmoe-data/test/ar/part-0-00000.npy,/olmoe-data/test/es/part-0-00000.npy,/olmoe-data/test/fi/part-0-00000.npy,/olmoe-data/test/hi/part-0-00000.npy,/olmoe-data/test/ru/part-0-00000.npy,/olmoe-data/test/mr/part-0-00000.npy,/olmoe-data/test/ca/part-0-00000.npy,/olmoe-data/test/et/part-0-00000.npy,/olmoe-data/test/nl/part-0-00000.npy,/olmoe-data/test/ur/part-0-00000.npy \
        --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
        --num_samples 250 \
        --batch_size 4


    python eval_peft_hf.py \
        --checkpoint_path allenai/OLMoE-1B-7B-0924 \
        --validation_path /olmoe-data/test/uk/part-0-00000.npy,/olmoe-data/test/sk/part-0-00000.npy,/olmoe-data/test/ru/part-0-00000.npy,/olmoe-data/test/cs/part-0-00000.npy,/olmoe-data/test/en/part-0-00000.npy,/olmoe-data/test/ar/part-0-00000.npy,/olmoe-data/test/es/part-0-00000.npy,/olmoe-data/test/fi/part-0-00000.npy,/olmoe-data/test/hi/part-0-00000.npy,/olmoe-data/test/ru/part-0-00000.npy,/olmoe-data/test/mr/part-0-00000.npy,/olmoe-data/test/ca/part-0-00000.npy,/olmoe-data/test/et/part-0-00000.npy,/olmoe-data/test/nl/part-0-00000.npy,/olmoe-data/test/ur/part-0-00000.npy \
        --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
        --num_samples 250 \
        --batch_size 4
"""

import argparse
import csv
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    OlmoeForCausalLM,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized .npy files (memory-mapped for efficiency)."""
    
    def __init__(self, npy_paths: List[str], max_length: int = 2048, pad_token_id: int = 0, max_samples: Optional[int] = None):
        """
        Args:
            npy_paths: List of paths to .npy files containing token IDs
            max_length: Maximum sequence length (chunk size)
            pad_token_id: Token ID to use for padding
            max_samples: Maximum number of samples to use (None = use all)
        """
        import numpy as np
        
        self.npy_paths = [Path(p) for p in npy_paths]
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.max_samples = max_samples
        
        # Load memory-mapped arrays
        self.memmaps = []
        self.offsets = []  # (start_idx, end_idx) for each memmap
        total_instances = 0
        
        for npy_path in self.npy_paths:
            if not npy_path.exists():
                raise ValueError(f"Tokenized file not found: {npy_path}")
            
            # Try to load as raw binary array first (like OLMo's MemMapDataset)
            # If that fails, try np.load with allow_pickle
            try:
                # Try reading as raw binary (uint16, like OLMo uses)
                file_size = npy_path.stat().st_size
                num_tokens = file_size // 2  # uint16 = 2 bytes per token
                # Create memory-mapped view of raw binary file
                memmap = np.memmap(str(npy_path), dtype=np.uint16, mode='r', shape=(num_tokens,))
            except (ValueError, OSError):
                # If raw binary doesn't work, try loading as .npy file
                try:
                    memmap = np.load(str(npy_path), mmap_mode='r', allow_pickle=True)
                    num_tokens = len(memmap)
                except Exception as e:
                    raise ValueError(f"Failed to load {npy_path}: {e}. "
                                   f"Expected raw binary array (uint16) or .npy file.")
            
            num_instances = num_tokens // max_length
            
            self.memmaps.append(memmap)
            start_idx = total_instances
            end_idx = total_instances + num_instances
            self.offsets.append((start_idx, end_idx))
            total_instances += num_instances
            
            log.info(f"Loaded {npy_path.name}: {num_tokens:,} tokens, {num_instances:,} instances")
        
        self._num_instances = total_instances
        if max_samples is not None and max_samples < self._num_instances:
            self._num_instances = max_samples
            log.info(f"Limited dataset to {self._num_instances:,} instances (from {total_instances:,})")
        else:
            log.info(f"Total dataset: {self._num_instances:,} instances")
    
    def __len__(self):
        return self._num_instances
    
    def __getitem__(self, idx):
        import numpy as np
        import torch
        
        idx = int(idx)
        if idx < 0:
            idx = len(self) + idx
        
        if idx >= self._num_instances:
            raise IndexError(f"Index {idx} out of range (max: {self._num_instances-1})")
        
        # Find which memmap this index belongs to
        memmap_idx = None
        local_idx = None
        for i, (start, end) in enumerate(self.offsets):
            if start <= idx < end:
                memmap_idx = i
                local_idx = idx - start
                break
        
        if memmap_idx is None:
            raise IndexError(f"Index {idx} out of range")
        
        # Read chunk from memory-mapped array
        memmap = self.memmaps[memmap_idx]
        start_pos = local_idx * self.max_length
        end_pos = start_pos + self.max_length
        
        # Get the chunk and convert to int64 (long) for PyTorch
        chunk = memmap[start_pos:end_pos]
        if isinstance(chunk, np.memmap):
            # For memmap, convert to regular array first
            input_ids = torch.from_numpy(np.array(chunk, dtype=np.int64)).long()
        else:
            # For regular numpy array
            input_ids = torch.from_numpy(chunk.astype(np.int64)).long()
        
        # Create attention mask (all ones since we're not padding)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def compute_loss(model, inputs, device):
    """
    Compute loss using the same logic as train_peft_hf.py.
    This ensures proper handling of padding tokens and matches the original OLMo training logic.
    
    When using device_map="auto" or DataParallel, the model handles device placement automatically,
    but inputs should still be moved to the primary device (respects CUDA_VISIBLE_DEVICES).
    """
    labels = inputs.get("labels")
    attention_mask = inputs.get("attention_mask")
    
    # Move inputs to device
    # For device_map="auto" or DataParallel, move to primary device (respects CUDA_VISIBLE_DEVICES)
    # The model will handle distribution across GPUs automatically
    input_ids = inputs["input_ids"].to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    labels = labels.to(device) if labels is not None else None
    
    # Forward pass
    # DataParallel wrapper handles device placement automatically
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get logits from model output
    logits = outputs.get("logits")
    if logits is None:
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # Fallback to model's loss if logits not available
            return outputs.loss
        else:
            raise ValueError("Model must return logits or loss")
    
    # Align with train.py's model_forward logic:
    # 1. Shift logits: logits[..., :-1, :] to predict next token
    # 2. Get labels: shift input_ids and mask padding tokens to -100
    # 3. Flatten and compute loss with ignore_index=-100
    
    # Shift logits so that tokens < n predict n (same as train.py line 756)
    logits_for_loss = logits[..., :-1, :].contiguous()
    # Flatten: (batch_size * seq_len, vocab_size)
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
    
    # Get labels - align with train.py's get_labels() method
    if labels is not None:
        # Clone labels to avoid in-place modification issues
        labels_for_loss = labels.clone()
        
        # Mask padding tokens BEFORE shifting (same as train.py line 740)
        # This ensures proper alignment with attention_mask
        if attention_mask is not None:
            # Mask padding positions (where attention_mask == 0) to -100
            labels_for_loss.masked_fill_(attention_mask == 0, -100)
        
        # Now shift labels to match shifted logits (same as train.py line 743)
        labels_for_loss = labels_for_loss[..., 1:].contiguous()
        
        # Flatten: (batch_size * seq_len,) (same as train.py line 762)
        labels_for_loss = labels_for_loss.view(-1)
    else:
        raise ValueError("Labels are required for loss computation")
    
    # Compute loss with ignore_index=-100 using reduction='sum' (same as train.py line 777)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    loss_sum = loss_fct(logits_for_loss, labels_for_loss)
    
    # Calculate total number of tokens from original input (BEFORE shifting)
    # This matches train.py's batch_size_in_tokens normalization exactly
    batch_size_in_tokens = input_ids.numel()  # Total tokens in original batch
    loss = loss_sum / batch_size_in_tokens
    
    return loss


def evaluate(model, dataloader, device):
    """Evaluate model on dataset and return metrics."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Create labels from input_ids (same as DataCollatorForLanguageModeling)
            labels = batch["input_ids"].clone()
            
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels,
            }
            
            # Compute loss
            loss = compute_loss(model, inputs, device)
            
            # Accumulate
            batch_size = batch["input_ids"].shape[0]
            batch_tokens = batch["input_ids"].numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                log.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Compute average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_batches": num_batches,
        "total_tokens": total_tokens,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace OLMoE model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to HuggingFace model checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (if not in checkpoint_path). Can be HF checkpoint or OLMo tokenizer.json")
    parser.add_argument("--validation_path", type=str, required=True,
                       help="Path(s) to validation dataset(s). Can be:\n"
                            "- Single .npy file\n"
                            "- Comma-separated list of .npy files (e.g., path1.npy,path2.npy,path3.npy)\n"
                            "- Each path will be evaluated separately")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of samples to evaluate on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). Auto-detect if not specified. Use 'auto' to use all GPUs")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for model loading: 'auto' (use all GPUs with device_map), 'dataparallel' (use DataParallel), or None (single GPU, default)")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (optional)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional)")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional)")
    
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("HUGGINGFACE MODEL EVALUATION")
    log.info("=" * 80)
    log.info(f"Checkpoint path: {args.checkpoint_path}")
    log.info(f"Validation path(s): {args.validation_path}")
    log.info(f"Max length: {args.max_length}")
    log.info(f"Num samples per dataset: {args.num_samples}")
    log.info(f"Batch size: {args.batch_size}")
    log.info("=" * 80)
    
    # Determine device and device_map
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info(f"Available GPUs: {num_gpus}")
    
    # Determine device_map strategy
    # Default to single GPU for stability - device_map="auto" can cause issues with evaluation
    use_data_parallel = False
    if args.device_map == "auto" and num_gpus > 1:
        # Use device_map="auto" only if explicitly requested
        device_map = "auto"
        device = torch.device("cuda")  # Primary device for inputs (respects CUDA_VISIBLE_DEVICES)
        log.info(f"Using device_map='auto' to distribute model across {num_gpus} GPU(s)")
    elif args.device_map == "dataparallel" and num_gpus > 1:
        # Use DataParallel for multi-GPU (better for evaluation)
        device_map = None
        device = torch.device("cuda")
        use_data_parallel = True
        log.info(f"Using DataParallel across {num_gpus} GPU(s)")
    else:
        # Default: single GPU (most stable for evaluation)
        device_map = None
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        log.info(f"Using single device: {device}")
    
    # Load model
    log.info(f"Loading model from {args.checkpoint_path}...")
    if device_map == "auto":
        model = OlmoeForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across all GPUs
        )
    else:
        model = OlmoeForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=None,  # We'll move to device manually
        )
        model.to(device)
    
    # Wrap with DataParallel if requested
    if use_data_parallel:
        model = torch.nn.DataParallel(model)
    
    model.eval()
    log.info("Model loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    log.info(f"Loading tokenizer from {tokenizer_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        log.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        log.info("Trying to create tokenizer from OLMo tokenizer.json...")
        
        # Try to load from OLMo tokenizer.json format
        tokenizer_json_path = Path(tokenizer_path) / "tokenizer.json"
        if not tokenizer_json_path.exists():
            # Try parent directory
            tokenizer_json_path = Path(tokenizer_path).parent / "tokenizer.json"
        
        if tokenizer_json_path.exists():
            try:
                from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
                from tokenizers import Tokenizer as HFTokenizer
                
                base_tokenizer = HFTokenizer.from_file(str(tokenizer_json_path))
                eos_token_id = model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 50279
                pad_token_id = model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else eos_token_id
                
                tokenizer = GPTNeoXTokenizerFast(
                    tokenizer_object=base_tokenizer,
                    eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
                    pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
                    unk_token=None,
                    bos_token=None,
                )
                log.info(f"Successfully loaded tokenizer from {tokenizer_json_path}")
            except Exception as e2:
                log.error(f"Failed to load tokenizer from {tokenizer_json_path}: {e2}")
                raise
        else:
            log.error(f"Could not find tokenizer.json at {tokenizer_json_path}")
            raise ValueError(f"Could not load tokenizer. Please provide --tokenizer_path or ensure tokenizer.json exists")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def extract_dataset_name(path: str) -> str:
        """Extract a meaningful dataset name from the path.
        
        Tries to extract language code from path structure like:
        - /path/to/test/uk/part-0-00000.npy -> "uk"
        - /path/to/test/sk/part-0-00000.npy -> "sk"
        Falls back to parent directory name or filename if language not found.
        """
        path_obj = Path(path)
        parts = path_obj.parts
        
        # Try to find language code in path (common patterns)
        # Look for 2-letter codes in common positions
        for i, part in enumerate(parts):
            # Check if it's a 2-letter code (likely language)
            if len(part) == 2 and part.isalpha() and part.islower():
                # Check if it's in a reasonable position (not too deep, not filename)
                if i < len(parts) - 1:  # Not the last part (filename)
                    return part
        
        # Fallback: use parent directory name
        if len(parts) >= 2:
            return parts[-2]  # Parent directory
        
        # Last resort: use filename without extension
        return path_obj.stem
    
    # Parse validation paths - support comma-separated list
    if ',' in args.validation_path:
        validation_paths = [p.strip() for p in args.validation_path.split(',')]
    else:
        validation_paths = [args.validation_path]
    
    log.info(f"Found {len(validation_paths)} validation dataset(s) to evaluate")
    
    # Store results for each dataset
    all_results = {}
    aggregate_loss_sum = 0.0
    aggregate_tokens = 0
    aggregate_batches = 0
    
    # Evaluate each dataset separately
    for dataset_idx, validation_path in enumerate(validation_paths):
        dataset_name = extract_dataset_name(validation_path)
        log.info("=" * 80)
        log.info(f"Evaluating dataset {dataset_idx + 1}/{len(validation_paths)}: {dataset_name}")
        log.info(f"Path: {validation_path}")
        log.info("=" * 80)
        
        # Load validation dataset
        log.info(f"Loading validation dataset from {validation_path}...")
        
        # For a single path, treat it as a single .npy file or directory
        val_dataset = PreTokenizedDataset(
            [validation_path],
            max_length=args.max_length,
            pad_token_id=pad_token_id if pad_token_id is not None else 0,
            max_samples=args.num_samples
        )
        
        log.info(f"Loaded {len(val_dataset)} validation examples")
        
        # Create dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if device.type == 'cuda' else False,
        )
        
        # Run evaluation
        log.info("Starting evaluation...")
        metrics = evaluate(model, val_dataloader, device)
        
        # Store results
        all_results[dataset_name] = {
            "path": validation_path,
            **metrics
        }
        
        # Accumulate for aggregate metrics
        aggregate_loss_sum += metrics['loss'] * metrics['total_tokens']
        aggregate_tokens += metrics['total_tokens']
        aggregate_batches += metrics['num_batches']
        
        # Print results for this dataset
        log.info("=" * 80)
        log.info(f"RESULTS for {dataset_name}")
        log.info("=" * 80)
        log.info(f"Loss: {metrics['loss']:.6f}")
        log.info(f"Perplexity: {metrics['perplexity']:.4f}")
        log.info(f"Number of batches: {metrics['num_batches']}")
        log.info(f"Total tokens: {metrics['total_tokens']:,}")
        log.info("=" * 80)
    
    # Print summary of all results
    log.info("")
    log.info("=" * 80)
    log.info("EVALUATION SUMMARY - ALL DATASETS")
    log.info("=" * 80)
    for dataset_name, results in all_results.items():
        log.info(f"{dataset_name}:")
        log.info(f"  Loss: {results['loss']:.6f}")
        log.info(f"  Perplexity: {results['perplexity']:.4f}")
        log.info(f"  Tokens: {results['total_tokens']:,}")
        log.info("")
    
    # Compute aggregate metrics (weighted average by tokens)
    aggregate_loss = None
    aggregate_perplexity = None
    if aggregate_tokens > 0:
        aggregate_loss = aggregate_loss_sum / aggregate_tokens
        aggregate_perplexity = math.exp(aggregate_loss)
        log.info("=" * 80)
        log.info("AGGREGATE METRICS (weighted by tokens)")
        log.info("=" * 80)
        log.info(f"Average Loss: {aggregate_loss:.6f}")
        log.info(f"Average Perplexity: {aggregate_perplexity:.4f}")
        log.info(f"Total Batches: {aggregate_batches}")
        log.info(f"Total Tokens: {aggregate_tokens:,}")
        log.info("=" * 80)
    
    # Log to wandb if specified
    if args.wandb_project:
        try:
            import wandb
            
            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or f"eval_{Path(args.checkpoint_path).name}",
                config={
                    "checkpoint_path": args.checkpoint_path,
                    "validation_paths": validation_paths,
                    "num_datasets": len(validation_paths),
                    "num_samples": args.num_samples,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                }
            )
            
            # Log metrics for each dataset
            for dataset_name, results in all_results.items():
                wandb.log({
                    f"eval/{dataset_name}/loss": results['loss'],
                    f"eval/{dataset_name}/perplexity": results['perplexity'],
                    f"eval/{dataset_name}/num_batches": results['num_batches'],
                    f"eval/{dataset_name}/total_tokens": results['total_tokens'],
                })
            
            # Log aggregate metrics
            if aggregate_tokens > 0 and aggregate_loss is not None and aggregate_perplexity is not None:
                wandb.log({
                    "eval/aggregate/loss": aggregate_loss,
                    "eval/aggregate/perplexity": aggregate_perplexity,
                    "eval/aggregate/num_batches": aggregate_batches,
                    "eval/aggregate/total_tokens": aggregate_tokens,
                })
            
            wandb.finish()
            log.info("Results logged to wandb")
        except ImportError:
            log.warning("wandb not installed, skipping wandb logging")
        except Exception as e:
            log.warning(f"Failed to log to wandb: {e}")
    
    # Save results to output text file
    # Map 2-letter language codes to 3-letter ISO codes
    lang_code_mapping = {
        'ar': 'arb',
        'cs': 'ces',
        'en': 'eng',
        'fi': 'fin',
        'hi': 'hin',
        'ru': 'rus',
        'es': 'spa',
        'ca': 'cat',
        'et': 'est',
        'mr': 'mar',
        'nl': 'nld',
        'sk': 'slk',
        'uk': 'ukr',
        'ur': 'urd',
    }
    
    # Define the order of languages to output
    output_languages = ['arb', 'ces', 'eng', 'fin', 'hin', 'rus', 'spa', 'cat', 'est', 'mar', 'nld', 'slk', 'ukr', 'urd']
    
    # Create reverse mapping (3-letter to 2-letter) for lookup
    reverse_mapping = {v: k for k, v in lang_code_mapping.items()}
    
    # Determine output file path
    checkpoint_path_obj = Path(args.checkpoint_path)
    if checkpoint_path_obj.exists() and checkpoint_path_obj.is_dir():
        # Local checkpoint path
        output_file = checkpoint_path_obj.parent / "evaluation_results.txt"
    else:
        # HuggingFace model name or non-existent path - save to current directory
        output_file = Path("evaluation_results.txt")
    
    # Extract model/checkpoint name for header
    # Try to extract language from checkpoint path (e.g., "catalan" from "peft_stage2_catalan")
    def extract_language_from_path(path_str: str) -> Optional[str]:
        """Extract language name from checkpoint path.
        
        Looks for patterns like:
        - peft_stage2_catalan -> "catalan"
        - peft_stage2_urdu -> "urdu"
        - peft_stage2_ukrainian -> "ukrainian"
        """
        path_parts = Path(path_str).parts
        for part in path_parts:
            # Check if part contains "peft_stage2_" followed by language
            if "peft_stage2_" in part:
                lang = part.replace("peft_stage2_", "")
                if lang:
                    return lang
            # Also check if part is a language name (common patterns)
            if part in ["catalan", "urdu", "ukrainian", "slovak", "russian", "czech", 
                       "english", "arabic", "spanish", "finnish", "hindi", "estonian",
                       "marathi", "dutch", "hindi"]:
                return part
        return None
    
    language = extract_language_from_path(args.checkpoint_path)
    
    if checkpoint_path_obj.exists() and checkpoint_path_obj.is_dir():
        # Use the checkpoint directory name
        base_model_name = checkpoint_path_obj.name
    else:
        # Use the full path/name as-is
        base_model_name = args.checkpoint_path
    
    # Prepend language to model name if found
    if language:
        model_name = f"{language}_{base_model_name}"
    else:
        model_name = base_model_name
    
    log.info(f"Saving results to {output_file}")
    
    # Append to file (open in append mode)
    with open(output_file, 'a') as f:
        # Write header with model/checkpoint name
        f.write(f"\n{model_name}\n")
        
        for lang_3letter in output_languages:
            # Find the 2-letter code
            lang_2letter = reverse_mapping.get(lang_3letter)
            
            # Try to find results by 2-letter code or 3-letter code
            perplexity = None
            if lang_2letter and lang_2letter in all_results:
                perplexity = all_results[lang_2letter]['perplexity']
            elif lang_3letter in all_results:
                perplexity = all_results[lang_3letter]['perplexity']
            else:
                # Try case-insensitive search
                for key, results in all_results.items():
                    if key.lower() == lang_2letter.lower() if lang_2letter else False:
                        perplexity = results['perplexity']
                        break
            
            # Write language and perplexity
            if perplexity is not None:
                f.write(f"{lang_3letter}:{perplexity:.4f}\n")
            else:
                # If language not found, write placeholder
                log.warning(f"Language {lang_3letter} not found in results")
                f.write(f"{lang_3letter}\nN/A\n")
    
    log.info(f"Results saved to {output_file}")
    
    # Save results to CSV file (multiblimp_consolidated.csv)
    # Define language categories
    HIGH_RESOURCE = ["eng", "arb", "ces", "spa", "fin", "hin", "rus"]
    LOW_RESOURCE = ["nld", "urd", "slk", "cat", "est", "mar", "ukr"]
    ALL_LANGS = ["eng", "arb", "ces", "spa", "fin", "hin", "rus", "nld", "urd", "slk", "cat", "est", "mar", "ukr"]
    
    # Determine CSV file path (in scripts directory)
    script_dir = Path(__file__).parent
    csv_file = script_dir / "perplexity_consolidated.csv"
    
    # Prepare row data
    row = {"Model": model_name}
    
    # Map results to 3-letter language codes
    for lang_3letter in ALL_LANGS:
        lang_2letter = reverse_mapping.get(lang_3letter)
        
        # Try to find results by 2-letter code or 3-letter code
        perplexity = None
        if lang_2letter and lang_2letter in all_results:
            perplexity = all_results[lang_2letter]['perplexity']
        elif lang_3letter in all_results:
            perplexity = all_results[lang_3letter]['perplexity']
        else:
            # Try case-insensitive search
            for key, results in all_results.items():
                if key.lower() == lang_2letter.lower() if lang_2letter else False:
                    perplexity = results['perplexity']
                    break
        
        # Store perplexity (round to 3 decimal places like in consolidate_multiblimp.py)
        if perplexity is not None:
            row[lang_3letter] = round(perplexity, 3)
        else:
            row[lang_3letter] = ""  # Empty string for missing languages
    
    # Calculate averages
    def avg(langs):
        vals = [row[l] for l in langs if row.get(l) != ""]
        return round(sum(vals) / len(vals), 3) if vals else ""
    
    row["High Resource Average"] = avg(HIGH_RESOURCE)
    row["Low Resource Average"] = avg(LOW_RESOURCE)
    row["Overall Avg"] = avg(ALL_LANGS)
    
    # Write to CSV (append mode, create file with header if it doesn't exist)
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ["Model", *ALL_LANGS, "High Resource Average", "Low Resource Average", "Overall Avg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            log.info(f"Created new CSV file: {csv_file}")
        
        # Write row
        writer.writerow(row)
    
    log.info(f"Results appended to CSV: {csv_file}")
    
    log.info("Evaluation complete!")


if __name__ == "__main__":
    main()

