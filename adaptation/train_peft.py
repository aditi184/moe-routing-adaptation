"""Simple HuggingFace training script for fine-tuning specific experts in MoE layers.

This script:
1. Loads a HuggingFace OLMoE model checkpoint
2. Freezes all parameters except target experts and their routers (and optionally embeddings)
3. Fine-tunes only the selected experts

Usage:
    # Using YAML config file (recommended):
    torchrun --nproc_per_node=4 train_peft_hf.py \
        --config scripts/configs/peft_olmoe_stage2.yml \
        --output_dir /peft_stage2_ukrainian
    
    # Or with command-line arguments (all parameters from YAML):
    torchrun --nproc_per_node=4 train_peft_hf.py \
        --model_path /retrain-5lang \
        --tokenizer_path allenai/OLMoE-1B-7B-0924-Instruct \
        --output_dir /peft_stage2_ukrainian \
        --target_experts '{"14": [27, 61, 13, 40, 42, 44, 1, 34], "15": [35, 24, 7, 42, 36, 44, 37, 0]}' \
        --train_routers \
        --train_embeddings \
        --dataset_path "/peft_sets/uk/tokenized/part-0-00000.npy,/peft_sets/uk/tokenized/part-1-00000.npy" \
        --num_epochs 5 \
        --batch_size 2 \
        --global_batch_size 1024 \
        --learning_rate 4.0e-4 \
        --max_length 2048 \
        --save_steps 100 \
        --logging_steps 10 \
        --optimizer_eps 1.0e-8 \
        --weight_decay 0.1 \
        --beta1 0.9 \
        --beta2 0.95 \
        --warmup_tokens 10485760000 \
        --max_tokens 5000000000000 \
        --max_grad_norm 1.0 \
        --seed 6198 \
        --wandb_project "peft_stage2_ukrainian" \
        --wandb_entity "" \
        --wandb_group "" \
        --wandb_name "peft_stage2_ukrainian" \
        --wandb_tags "experiment1" \
        --eval_dataset_path "/olmoe-data/test/uk/part-0-00000.npy"
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    OlmoeForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from transformers.trainer_utils import set_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training.
    
    Explicitly initialize process group to avoid deadlocks with Trainer.
    """
    import torch.distributed as dist
    from datetime import timedelta
    
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set CUDA device for this process
        torch.cuda.set_device(f"cuda:{local_rank}")
        
        # Explicitly initialize process group BEFORE Trainer sees it
        # This prevents Trainer from trying to initialize it and causing deadlocks
        if not dist.is_initialized():
            if rank == 0:
                log.info("Initializing process group explicitly...")
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=30),
                init_method="env://"  # Use environment variables set by torchrun
            )
            if rank == 0:
                log.info("Process group initialized successfully")
        else:
            if rank == 0:
                log.info("Process group already initialized")
        
        # Add a barrier to ensure all processes are synchronized after init
        dist.barrier()
        if rank == 0:
            log.info("All processes synchronized after process group initialization")
        
        return rank, world_size, local_rank
    return 0, 1, 0


def freeze_all_parameters(model):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False
    log.info("Frozen all parameters")


def unfreeze_embeddings(model, train_embeddings: bool = True):
    """Unfreeze embedding parameters (input and output embeddings).
    
    Args:
        model: HuggingFace OLMoE model
        train_embeddings: Whether to train embedding parameters
    """
    if not train_embeddings:
        return
    
    log.info("=" * 80)
    log.info("UNFREEZING EMBEDDING PARAMETERS")
    log.info("=" * 80)
    
    unfrozen_count = 0
    total_trainable_params = 0
    
    # Get input embeddings
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    
    # Check if embeddings are tied
    are_tied = input_emb.weight is output_emb.weight if hasattr(input_emb, 'weight') and hasattr(output_emb, 'weight') else False
    
    # Unfreeze input embeddings
    if hasattr(input_emb, 'weight'):
        for param in input_emb.parameters():
            param.requires_grad = True
            unfrozen_count += 1
            total_trainable_params += param.numel()
        input_param_count = sum(p.numel() for p in input_emb.parameters())
        log.info(f"  Input embeddings: unfrozen ({input_param_count:,} params)")
    
    # Unfreeze output embeddings (only if not tied)
    if not are_tied and hasattr(output_emb, 'weight'):
        for param in output_emb.parameters():
            param.requires_grad = True
            unfrozen_count += 1
            total_trainable_params += param.numel()
        output_param_count = sum(p.numel() for p in output_emb.parameters())
        log.info(f"  Output embeddings (lm_head): unfrozen ({output_param_count:,} params)")
    elif are_tied:
        log.info(f"  Output embeddings: tied with input embeddings (already unfrozen)")
    
    log.info(f"Unfrozen {unfrozen_count} embedding parameter tensors")
    log.info(f"Total trainable embedding parameters: {total_trainable_params:,}")
    
    # Performance warning
    log.warning("=" * 80)
    log.warning("PERFORMANCE WARNING: Training embeddings significantly slows down training!")
    log.warning("=" * 80)
    log.warning("Embeddings are large matrices accessed on every forward/backward pass.")
    log.warning("This causes:")
    log.warning("  - Slower forward passes (embedding lookup + gradient computation)")
    log.warning("  - Slower backward passes (gradients through entire model)")
    log.warning("  - Higher memory usage (embedding gradients)")
    log.warning("")
    log.warning("RECOMMENDATIONS to improve speed:")
    log.warning("  1. Use gradient checkpointing (--gradient_checkpointing)")
    log.warning("  2. Reduce batch size or sequence length")
    log.warning("  3. Consider training embeddings with a lower learning rate")
    log.warning("  4. Only train embeddings if absolutely necessary for your task")
    log.warning("=" * 80)
    
    # Log summary
    log.info("")
    log.info("SUMMARY OF TRAINABLE COMPONENTS:")
    log.info("  - Input embeddings: TRAINABLE")
    if are_tied:
        log.info("  - Output embeddings: TRAINABLE (tied with input)")
    else:
        log.info("  - Output embeddings: TRAINABLE (separate)")
    log.info("=" * 80)


def unfreeze_target_experts(model, target_experts: Dict[int, List[int]], train_routers: bool = True):
    """Unfreeze only target expert parameters and routers.
    
    Args:
        model: HuggingFace OLMoE model
        target_experts: Dict mapping layer_idx -> list of expert indices to train
        train_routers: Whether to also train router parameters
    """
    log.info("=" * 80)
    log.info("UNFREEZING TARGET EXPERT PARAMETERS")
    log.info("=" * 80)
    
    unfrozen_count = 0
    total_trainable_params = 0
    
    for layer_idx, expert_indices in target_experts.items():
        layer_idx = int(layer_idx)
        expert_indices = [int(e) for e in expert_indices]
        
        if layer_idx >= len(model.model.layers):
            log.warning(f"Layer {layer_idx} out of range (max: {len(model.model.layers)-1})")
            continue
        
        layer = model.model.layers[layer_idx]
        
        # Check if this is an MoE layer
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts'):
            log.warning(f"Layer {layer_idx} is not an MoE layer")
            continue
        
        # Unfreeze target expert parameters
        for expert_idx in expert_indices:
            if expert_idx >= len(layer.mlp.experts):
                log.warning(f"Layer {layer_idx} expert {expert_idx} out of range")
                continue
            
            expert = layer.mlp.experts[expert_idx]
            
            # Unfreeze expert parameters (gate_proj, up_proj, down_proj)
            for param_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(expert, param_name):
                    module = getattr(expert, param_name)
                    if module is not None:
                        # Unfreeze all parameters in this module (weight, bias, etc.)
                        for param in module.parameters():
                            param.requires_grad = True
                            unfrozen_count += 1
                            total_trainable_params += param.numel()
                        param_count = sum(p.numel() for p in module.parameters())
                        log.info(f"  Layer {layer_idx} expert {expert_idx} {param_name}: "
                                f"unfrozen ({param_count:,} params)")
        
        # Unfreeze router if requested
        if train_routers and hasattr(layer.mlp, 'gate'):
            router_module = layer.mlp.gate
            if router_module is not None:
                # Unfreeze all parameters in router module
                for param in router_module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
                    total_trainable_params += param.numel()
                router_param_count = sum(p.numel() for p in router_module.parameters())
                log.info(f"  Layer {layer_idx} router: unfrozen ({router_param_count:,} params)")
    
    log.info(f"Unfrozen {unfrozen_count} parameter tensors")
    log.info(f"Total trainable parameters: {total_trainable_params:,}")
    log.info("=" * 80)
    
    # Verify other parameters are frozen
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Frozen parameters: {frozen_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log summary of what's being trained
    log.info("")
    log.info("SUMMARY OF TRAINABLE COMPONENTS:")
    if train_routers:
        log.info("  - Routers: TRAINABLE")
    else:
        log.info("  - Routers: FROZEN")
    log.info("  - Target experts: TRAINABLE")
    log.info("  - Other experts: FROZEN")
    log.info("  - Attention layers: FROZEN")
    log.info("  - Layer norms: FROZEN")
    log.info("=" * 80)


def enforce_expert_routing(model, target_experts: Dict[int, List[int]], training_mode: bool = True):
    """
    Add forward hooks to enforce routing to ONLY target experts during training.
    This masks non-target experts by setting their router logits to a very negative value.
    
    Args:
        model: HuggingFace OLMoE model
        target_experts: Dict mapping layer_idx -> list of expert indices to route to
        training_mode: If True, enforce routing; if False, remove hooks
    """
    if not training_mode:
        # Remove hooks if any exist
        if hasattr(model, '_expert_routing_hooks'):
            for hook in model._expert_routing_hooks:
                hook.remove()
            delattr(model, '_expert_routing_hooks')
        return
    
    # Store hooks for later removal
    model._expert_routing_hooks = []
    
    for layer_idx, expert_indices in target_experts.items():
        layer_idx = int(layer_idx)
        expert_indices = [int(e) for e in expert_indices]
        
        if layer_idx >= len(model.model.layers):
            log.warning(f"Layer {layer_idx} out of range (max: {len(model.model.layers)-1})")
            continue
        
        layer = model.model.layers[layer_idx]
        
        # Check if this is an MoE layer
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            log.warning(f"Layer {layer_idx} is not an MoE layer")
            continue
        
        # Get the router (gate) module
        router = layer.mlp.gate
        num_experts = router.weight.shape[0]  # Number of experts
        
        # Create a mask for target experts
        target_expert_mask = torch.zeros(num_experts, dtype=torch.bool)
        for expert_idx in expert_indices:
            if expert_idx < num_experts:
                target_expert_mask[expert_idx] = True
            else:
                log.warning(f"Expert {expert_idx} out of range (max: {num_experts-1})")
        
        def make_routing_hook(layer_id, target_mask, target_list):
            """Create a hook function for this layer."""
            def routing_hook(module, input, output):
                """
                Hook to mask non-target experts in router logits.
                This only runs during training (when model.training is True).
                
                By setting non-target expert logits to a very negative value,
                we ensure that top-k selection will only choose target experts.
                """
                if not module.training:
                    return output
                
                # output is the router logits: can be 2D (batch*seq_len, num_experts) or 3D (batch, seq_len, num_experts)
                router_logits = output
                
                # Create a mask: True for target experts, False for others
                # Shape: (num_experts,)
                mask = target_mask.to(router_logits.device)
                
                # Handle both 2D and 3D router logits
                if router_logits.dim() == 2:
                    # 2D case: (batch*seq_len, num_experts)
                    mask = mask.unsqueeze(0)  # (1, num_experts)
                    mask = mask.expand_as(router_logits)  # (batch*seq_len, num_experts)
                elif router_logits.dim() == 3:
                    # 3D case: (batch_size, seq_len, num_experts)
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, num_experts)
                    mask = mask.expand_as(router_logits)  # (batch_size, seq_len, num_experts)
                else:
                    # Unexpected shape, return as-is
                    log.warning(f"Unexpected router logits shape: {router_logits.shape}")
                    return output
                
                # Mask non-target experts: set their logits to a very negative value
                # This ensures they won't be selected by top-k
                masked_logits = router_logits.clone()
                masked_logits[~mask] = float('-inf')  # or use a large negative value like -1e9
                
                return masked_logits
            
            return routing_hook
        
        # Register forward hook on the router
        hook = router.register_forward_hook(make_routing_hook(layer_idx, target_expert_mask, expert_indices))
        model._expert_routing_hooks.append(hook)
        
        log.info(f"Registered routing hook for layer {layer_idx} to enforce routing to experts {expert_indices} only")
    
    log.info(f"Registered {len(model._expert_routing_hooks)} routing hooks for expert enforcement")


class TokenTrackingCallback(TrainerCallback):
    """Callback to track and log the number of tokens processed during training."""
    
    def __init__(self, max_length: int, world_size: int = 1, max_tokens: Optional[int] = None):
        self.max_length = max_length
        self.world_size = world_size
        self.max_tokens = max_tokens  # Maximum tokens to process before stopping
        self.total_tokens = 0
        self.last_logged_step = -1
        self.initialized = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize token tracking at the start of training."""
        self.initialized = True
        self.last_logged_step = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log tokens processed at each logging step."""
        if logs is None:
            return
        
        # Calculate tokens per step
        # tokens_per_step = batch_size * max_length * gradient_accumulation_steps * world_size
        tokens_per_step = (
            args.per_device_train_batch_size * 
            self.max_length * 
            args.gradient_accumulation_steps * 
            self.world_size
        )
        
        # Calculate tokens processed since last log
        current_step = state.global_step
        if self.last_logged_step < 0:
            # First log - count tokens from step 0 to current step
            steps_since_last_log = current_step + 1
        else:
            steps_since_last_log = current_step - self.last_logged_step
        
        # Update total tokens
        tokens_this_interval = steps_since_last_log * tokens_per_step
        self.total_tokens += tokens_this_interval
        self.last_logged_step = current_step
        
        # Add to logs (will be logged to wandb automatically)
        logs["tokens_processed"] = self.total_tokens
        logs["tokens_per_step"] = tokens_per_step
        
        # Check if we've reached the maximum token limit
        if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
            if state.is_world_process_zero:
                log.info("=" * 80)
                log.info(f"Reached maximum token limit: {self.total_tokens:,} >= {self.max_tokens:,}")
                log.info("Stopping training and saving checkpoint...")
                log.info("=" * 80)
            # Stop training
            control.should_training_stop = True
            # Trigger a save
            control.should_save = True
        
        # Log to console (only on rank 0)
        if state.is_world_process_zero:
            log.info(f"Step {current_step}: Total tokens processed: {self.total_tokens:,} "
                    f"({tokens_per_step:,} tokens/step)")
            if self.max_tokens is not None:
                remaining = max(0, self.max_tokens - self.total_tokens)
                log.info(f"  Remaining tokens until limit: {remaining:,}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final token count at end of training."""
        if state.is_world_process_zero:
            # Calculate final tokens if training ended between logging steps
            current_step = state.global_step
            if current_step > self.last_logged_step:
                tokens_per_step = (
                    args.per_device_train_batch_size * 
                    self.max_length * 
                    args.gradient_accumulation_steps * 
                    self.world_size
                )
                final_steps = current_step - self.last_logged_step
                self.total_tokens += final_steps * tokens_per_step
            
            log.info("=" * 80)
            log.info(f"Training complete! Total tokens processed: {self.total_tokens:,}")
            log.info("=" * 80)


class SimpleTextDataset(Dataset):
    """Simple dataset for text data."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized .npy files (memory-mapped for efficiency)."""
    
    def __init__(self, npy_paths: List[str], max_length: int = 2048, pad_token_id: int = 0):
        """
        Args:
            npy_paths: List of paths to .npy files containing token IDs
            max_length: Maximum sequence length (chunk size)
            pad_token_id: Token ID to use for padding
        """
        import numpy as np
        
        self.npy_paths = [Path(p) for p in npy_paths]
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
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
        log.info(f"Total dataset: {self._num_instances:,} instances")
    
    def __len__(self):
        return self._num_instances
    
    def __getitem__(self, idx):
        import numpy as np
        import torch
        
        idx = int(idx)
        if idx < 0:
            idx = len(self) + idx
        
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


def load_dataset(dataset_path: str, tokenizer, max_length: int = 2048, pad_token_id: int = 0):
    """Load dataset from file, directory, or list of pre-tokenized .npy files.
    
    Args:
        dataset_path: Path to dataset. Can be:
            - A single .npy file (pre-tokenized)
            - A directory containing .npy files (pre-tokenized)
            - A single text file (will be tokenized on the fly)
            - A directory containing .jsonl files (will be tokenized on the fly)
            - A comma-separated list of .npy file paths (pre-tokenized)
    """
    dataset_path_str = str(dataset_path)
    
    # Check if it's a comma-separated list of .npy files
    if ',' in dataset_path_str:
        npy_paths = [p.strip() for p in dataset_path_str.split(',')]
        log.info(f"Loading pre-tokenized dataset from {len(npy_paths)} .npy files")
        return PreTokenizedDataset(npy_paths, max_length=max_length, pad_token_id=pad_token_id)
    
    dataset_path = Path(dataset_path_str)
    
    # Check if it's a single .npy file
    if dataset_path.is_file() and dataset_path.suffix == '.npy':
        log.info(f"Loading pre-tokenized dataset from {dataset_path}")
        return PreTokenizedDataset([str(dataset_path)], max_length=max_length, pad_token_id=pad_token_id)
    
    # Check if it's a directory with .npy files
    if dataset_path.is_dir():
        npy_files = sorted(dataset_path.glob("*.npy"))
        if npy_files:
            log.info(f"Loading pre-tokenized dataset from {len(npy_files)} .npy files in {dataset_path}")
            return PreTokenizedDataset([str(f) for f in npy_files], max_length=max_length, pad_token_id=pad_token_id)
        
        # Otherwise, try loading as text/JSONL files
        texts = []
        for jsonl_file in dataset_path.glob("*.jsonl"):
            log.info(f"Loading from {jsonl_file}")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Try common keys for text
                        text = data.get('text', data.get('content', data.get('input', '')))
                        if text:
                            texts.append(text)
        if texts:
            return SimpleTextDataset(texts, tokenizer, max_length)
    
    # Check if it's a single text file
    if dataset_path.is_file():
        log.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return SimpleTextDataset(texts, tokenizer, max_length)
    
    raise ValueError(f"Dataset path {dataset_path} does not exist or is not a valid format")


def load_config_from_yaml(yaml_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune specific experts in HuggingFace OLMoE model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file (overrides other arguments)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to HuggingFace model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (if not in model_path). Can be HF checkpoint or OLMo tokenizer.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--target_experts", type=str, default=None,
                       help='JSON string mapping layer_idx to expert indices, e.g. \'{"14": [0, 1], "15": [2, 3]}\'')
    parser.add_argument("--train_routers", action="store_true", default=None,
                       help="Whether to train router parameters")
    parser.add_argument("--train_embeddings", action="store_true", default=None,
                       help="Whether to train embedding parameters (input and output embeddings). "
                            "WARNING: This significantly slows down training!")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None,
                       help="Enable gradient checkpointing to save memory (trades compute for memory). "
                            "Automatically enabled when --train_embeddings is used.")
    parser.add_argument("--train_all_parameters", action="store_true", default=False,
                       help="Train all model parameters (full fine-tuning). If set, --target_experts is ignored.")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to training dataset. Can be:\n"
                            "- Pre-tokenized: .npy file, directory with .npy files, or comma-separated list of .npy paths\n"
                            "- Text: .txt file or directory with .jsonl files (will be tokenized on the fly)")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                       help="Path to evaluation dataset (same format as --dataset_path). If provided, evaluation will run during training.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--global_batch_size", type=int, default=None, help="Global batch size (overrides batch_size calculation)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=6198, help="Random seed (default: 6198)")
    # Optimizer arguments
    parser.add_argument("--optimizer_eps", type=float, default=1e-8, help="Optimizer epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    # Scheduler arguments
    parser.add_argument("--lr_scheduler_type", type=str, default=None,
                       choices=["constant", "linear", "cosine", "polynomial", "constant_with_warmup"],
                       help="Learning rate scheduler type. Use 'constant' to keep LR constant throughout training.")
    parser.add_argument("--warmup_tokens", type=int, default=None, help="Warmup tokens for scheduler")
    parser.add_argument("--max_tokens", type=int, default=None, 
                       help="Max tokens for scheduler and training stopping. If set, training will stop when this many tokens are processed.")
    # Gradient clipping
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)
        log.info(f"Loaded config from {args.config}")
        
        # Override args with config values if not explicitly provided
        if args.model_path is None and "load_path" in config:
            args.model_path = config["load_path"]
        # if args.output_dir is None and "save_folder" in config:
        #     # Replace ${run_name} if present
        #     save_folder = config["save_folder"]
        #     if "${run_name}" in save_folder and "run_name" in config:
        #         save_folder = save_folder.replace("${run_name}", config["run_name"])
        #     args.output_dir = save_folder
        if not args.train_all_parameters:
            if args.target_experts is None and "selective_expert_training" in config:
                sel_exp = config["selective_expert_training"]
                args.target_experts = json.dumps(sel_exp.get("target_experts", {}))
                if args.train_routers is None:
                    args.train_routers = sel_exp.get("train_routers", True)
                if args.train_embeddings is None:
                    args.train_embeddings = sel_exp.get("train_embeddings", False)
        elif "selective_expert_training" in config and config["selective_expert_training"].get("train_all_parameters", False):
            # Allow config to set train_all_parameters
            args.train_all_parameters = True
        if args.num_epochs is None and "max_duration" in config:
            max_dur = config["max_duration"]
            if isinstance(max_dur, str) and max_dur.endswith("ep"):
                args.num_epochs = int(max_dur[:-2])
        if args.batch_size is None and "device_train_microbatch_size" in config:
            args.batch_size = config["device_train_microbatch_size"]
        if args.global_batch_size is None and "global_train_batch_size" in config:
            args.global_batch_size = config["global_train_batch_size"]
        # Note: gradient_accumulation_steps will be calculated after world_size is determined
        if args.learning_rate is None and "optimizer" in config:
            args.learning_rate = config["optimizer"].get("learning_rate", 4e-4)
        if "optimizer" in config:
            opt_config = config["optimizer"]
            args.optimizer_eps = opt_config.get("eps", args.optimizer_eps)
            args.weight_decay = opt_config.get("weight_decay", args.weight_decay)
            if "betas" in opt_config:
                args.beta1 = opt_config["betas"][0]
                args.beta2 = opt_config["betas"][1]
        if "scheduler" in config:
            sched_config = config["scheduler"]
            if args.lr_scheduler_type is None:
                args.lr_scheduler_type = sched_config.get("type", args.lr_scheduler_type)
            args.warmup_tokens = sched_config.get("t_warmup", args.warmup_tokens)
            args.max_tokens = sched_config.get("t_max", args.max_tokens)
        if args.max_grad_norm is None and "max_grad_norm" in config:
            args.max_grad_norm = config["max_grad_norm"]
        if args.seed is None and "seed" in config:
            args.seed = config["seed"]
        if "wandb" in config:
            wandb_config = config["wandb"]
            if args.wandb_project is None:
                args.wandb_project = wandb_config.get("project")
            if args.wandb_entity is None:
                args.wandb_entity = wandb_config.get("entity")
            if args.wandb_group is None:
                args.wandb_group = wandb_config.get("group")
            if args.wandb_name is None:
                args.wandb_name = wandb_config.get("name")
            if args.wandb_tags is None and "tags" in wandb_config:
                args.wandb_tags = ",".join(wandb_config["tags"])
        if args.dataset_path is None and "data" in config and "paths" in config["data"]:
            # Join paths with comma
            args.dataset_path = ",".join(config["data"]["paths"])
    
    # Validate required arguments
    if args.model_path is None:
        raise ValueError("--model_path is required (or provide --config)")
    if args.output_dir is None:
        raise ValueError("--output_dir is required (or provide --config)")
    if not args.train_all_parameters and args.target_experts is None:
        raise ValueError("--target_experts is required (or provide --config) unless --train_all_parameters is set")
    if args.dataset_path is None:
        raise ValueError("--dataset_path is required (or provide --config)")
    
    # Set defaults if still None
    if args.num_epochs is None:
        args.num_epochs = 5
    if args.batch_size is None:
        args.batch_size = 2
    if args.learning_rate is None:
        args.learning_rate = 4e-4
    # gradient_accumulation_steps will be calculated after world_size is determined
    if args.train_routers is None:
        args.train_routers = True
    if args.train_embeddings is None:
        args.train_embeddings = False
    
    # Auto-enable gradient checkpointing when training embeddings (helps with memory)
    if args.train_embeddings and args.gradient_checkpointing is None:
        args.gradient_checkpointing = True
    elif args.gradient_checkpointing is None:
        args.gradient_checkpointing = False
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Log auto-enabled gradient checkpointing after rank is available
    if args.train_embeddings and args.gradient_checkpointing:
        if rank == 0:
            log.info("Auto-enabled gradient checkpointing because embeddings are being trained")
    
    # Calculate gradient accumulation steps if global_batch_size is specified
    if args.gradient_accumulation_steps is None:
        if args.global_batch_size and args.batch_size:
            args.gradient_accumulation_steps = args.global_batch_size // (args.batch_size * world_size)
            if args.gradient_accumulation_steps < 1:
                args.gradient_accumulation_steps = 1
            if rank == 0:
                log.info(f"Calculated gradient_accumulation_steps: {args.gradient_accumulation_steps} "
                        f"(global_batch_size={args.global_batch_size}, per_device_batch_size={args.batch_size}, world_size={world_size})")
        else:
            args.gradient_accumulation_steps = 1
    
    if rank == 0:
        log.info("=" * 80)
        log.info("HUGGINGFACE EXPERT FINE-TUNING")
        log.info("=" * 80)
        log.info(f"Model path: {args.model_path}")
        log.info(f"Output dir: {args.output_dir}")
        log.info(f"Target experts: {args.target_experts}")
        log.info(f"Train routers: {args.train_routers}")
        log.info(f"Train embeddings: {args.train_embeddings}")
        log.info("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Load model and tokenizer
    if rank == 0:
        log.info(f"Loading model from {args.model_path}...")
    
    model = OlmoeForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,  # We'll handle device placement
    )
    
    # Load tokenizer - try model_path first, then tokenizer_path if provided
    tokenizer_path = args.tokenizer_path or args.model_path
    if rank == 0:
        log.info(f"Loading tokenizer from {tokenizer_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        if rank == 0:
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
    
    # Don't move model to device - let Trainer handle it
    
    # Parse target experts (only needed if not training all parameters)
    target_experts = None
    if not args.train_all_parameters:
        target_experts = json.loads(args.target_experts)
        target_experts = {int(k): [int(x) for x in (v if isinstance(v, list) else [v])] 
                         for k, v in target_experts.items()}
        
        # Freeze all parameters
        freeze_all_parameters(model)
        
        # Unfreeze embeddings if requested
        if args.train_embeddings:
            unfreeze_embeddings(model, train_embeddings=True)
        
        # Unfreeze target experts
        unfreeze_target_experts(model, target_experts, train_routers=args.train_routers)
        
        # Enforce routing to ONLY target experts during training
        enforce_expert_routing(model, target_experts, training_mode=True)
    else:
        # Train all parameters - don't freeze anything
        log.info("=" * 80)
        log.info("TRAINING ALL PARAMETERS (FULL FINE-TUNING)")
        log.info("=" * 80)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable parameters: {trainable_params:,}")
        log.info("=" * 80)
    
    # Load dataset
    log.info(f"Loading dataset from {args.dataset_path}...")
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_dataset = load_dataset(
        args.dataset_path, 
        tokenizer, 
        max_length=args.max_length,
        pad_token_id=pad_token_id if pad_token_id is not None else 0
    )
    
    log.info(f"Loaded {len(train_dataset)} training examples")
    
    # Load eval dataset if provided
    eval_dataset = None
    if args.eval_dataset_path:
        log.info(f"Loading eval dataset from {args.eval_dataset_path}...")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eval_dataset = load_dataset(
            args.eval_dataset_path,
            tokenizer,
            max_length=args.max_length,
            pad_token_id=pad_token_id if pad_token_id is not None else 0
        )
        log.info(f"Loaded {len(eval_dataset)} eval examples")
    
    # Setup data collator
    # Note: DataCollatorForLanguageModeling with mlm=False creates labels by copying input_ids
    # The model's forward method will compute loss from these labels
    # We need to ensure padding tokens are properly ignored (using ignore_index=-100)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=None,  # Don't pad to multiple
    )
    
    # TEMPORARILY DISABLE WANDB COMPLETELY to debug deadlock
    # Set WANDB_MODE=disabled for all processes
    if args.wandb_project:
        os.environ["WANDB_MODE"] = "disabled"
        if rank == 0:
            log.warning("Wandb is TEMPORARILY DISABLED to debug NCCL deadlock")
    
    # Patch Accelerate optimizer wrapper BEFORE creating trainer
    # This is a workaround for Accelerate trying to call .train() on PyTorch optimizers
    try:
        from accelerate.optimizer import AcceleratedOptimizer
        
        # Monkey-patch AcceleratedOptimizer.train() to handle PyTorch optimizers
        original_train = AcceleratedOptimizer.train
        
        def patched_train(self):
            # If the underlying optimizer has .train(), call it
            if hasattr(self.optimizer, 'train'):
                return original_train(self)
            # Otherwise, just return (PyTorch optimizers don't need .train())
            return None
        
        AcceleratedOptimizer.train = patched_train
        log.info("Patched Accelerate optimizer.train() to handle PyTorch optimizers")
    except (ImportError, AttributeError) as e:
        log.warning(f"Could not patch Accelerate optimizer: {e}")
    
    
    # Training arguments - matching YAML config
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,  # Set to 0 to avoid deadlocks in DDP mode
        # Optimizer settings (matching YAML config)
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.optimizer_eps,
        weight_decay=args.weight_decay,
        # Gradient clipping (matching YAML config)
        max_grad_norm=args.max_grad_norm,
        # Wandb logging configuration
        # TEMPORARILY DISABLED COMPLETELY to debug deadlock
        report_to=[],  # Disable wandb completely
        # Force DDP instead of Accelerate
        ddp_backend="nccl" if world_size > 1 else None,
        # Disable deepspeed/fsdp to force DDP
        deepspeed=None,
        fsdp=[],  # Empty list instead of None to avoid TypeError
        # Scheduler settings (will be updated if warmup_tokens is specified)
        # Default: cosine if warmup_tokens/max_tokens specified, else linear
        # Can be overridden with --lr_scheduler_type (e.g., "constant" for constant LR)
        lr_scheduler_type=args.lr_scheduler_type or ("cosine" if args.warmup_tokens or args.max_tokens else "linear"),
        warmup_steps=0,  # Will be calculated from warmup_tokens if specified
        # Evaluation settings (if eval_dataset is provided)
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,  # Evaluate at same interval as saving
        per_device_eval_batch_size=args.batch_size if eval_dataset is not None else None,
        # Gradient checkpointing (helps with memory, especially when training embeddings)
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Create token tracking callback
    # Use args.max_tokens if provided, otherwise None (no token limit)
    # Note: args.max_tokens is also used for scheduler, but we use it here for stopping training
    callback_max_tokens = args.max_tokens if args.max_tokens else None
    if rank == 0:
        if callback_max_tokens:
            log.info(f"Token tracking enabled: Training will stop after processing {callback_max_tokens:,} tokens")
        else:
            log.info("Token tracking enabled: No token limit (training will stop based on num_epochs)")
    token_tracking_callback = TokenTrackingCallback(
        max_length=args.max_length,
        world_size=world_size,
        max_tokens=callback_max_tokens
    )
    
    # If max_tokens is set, we should prioritize token limit over epochs
    # Set a very high num_epochs so token limit takes precedence
    if callback_max_tokens is not None:
        # Calculate approximate max epochs needed (with large safety margin)
        # This ensures token limit is the stopping condition, not epochs
        tokens_per_epoch = len(train_dataset) * args.max_length
        estimated_epochs_for_max_tokens = (callback_max_tokens // tokens_per_epoch) + 10  # Add safety margin
        if args.num_epochs and args.num_epochs < estimated_epochs_for_max_tokens:
            if rank == 0:
                log.warning(f"max_tokens={callback_max_tokens:,} is set. Training will stop at token limit, "
                          f"not at num_epochs={args.num_epochs}. Adjusting num_epochs to {estimated_epochs_for_max_tokens} "
                          f"to ensure token limit is reached.")
            # Temporarily increase num_epochs to ensure token limit is reached
            training_args.num_train_epochs = estimated_epochs_for_max_tokens
    
    # Calculate scheduler steps if needed
    # Let Trainer handle optimizer creation to ensure proper DDP integration
    warmup_steps = 0
    scheduler_type = args.lr_scheduler_type or ("cosine" if args.warmup_tokens or args.max_tokens else "linear")
    
    # Only calculate warmup if scheduler supports it (not for "constant" without warmup)
    if scheduler_type != "constant" and (args.warmup_tokens or args.max_tokens):
        # Calculate total training steps for scheduler
        # We'll approximate based on dataset size and batch size
        total_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps * world_size) * args.num_epochs
        
        # Convert token-based warmup to step-based
        # Approximate: tokens_per_step = batch_size * max_length * gradient_accumulation_steps * world_size
        tokens_per_step = args.batch_size * args.max_length * args.gradient_accumulation_steps * world_size
        warmup_steps = int(args.warmup_tokens / tokens_per_step) if args.warmup_tokens else 0
        
        if warmup_steps > 0:
            # Update training_args with calculated warmup steps
            training_args.warmup_steps = warmup_steps
            if rank == 0:
                log.info(f"Calculated warmup steps: {warmup_steps} out of {total_steps} total steps")
    elif scheduler_type == "constant":
        # Ensure warmup is 0 for constant scheduler (unless constant_with_warmup)
        training_args.warmup_steps = 0
        if rank == 0:
            log.info("Using constant learning rate scheduler (no warmup, no decay)")
    
    # Create trainer with custom compute_loss to ensure proper loss scaling
    # Note: We don't use load balancing loss or router z-loss when training only specific experts:
    # - Load balancing loss encourages even distribution across ALL experts, which is counterproductive
    #   when we want the router to route more tokens to the target experts we're training.
    # - Router z-loss can help stabilize router training but is optional and not critical here.
    # The model's forward pass computes standard cross-entropy loss, which is sufficient.
    # Let Trainer create optimizer internally to ensure proper DDP integration
    
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            # Extract target_experts before passing kwargs to parent
            self.target_experts = kwargs.pop('target_experts', {})
            super().__init__(*args, **kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Custom loss computation aligned with train.py's model_forward.
            This ensures proper handling of padding tokens and matches the original OLMo training logic.
            
            Args:
                model: The model to compute loss for
                inputs: Input dictionary containing input_ids, attention_mask, labels, etc.
                return_outputs: Whether to return model outputs along with loss
                num_items_in_batch: Number of items in batch (optional, for compatibility with newer transformers versions)
            """
            labels = inputs.get("labels")
            attention_mask = inputs.get("attention_mask")
            outputs = model(**inputs)
           
            # Get logits from model output
            logits = outputs.get("logits")
            if logits is None:
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    # Fallback to model's loss if logits not available
                    loss = outputs.loss
                    return (loss, outputs) if return_outputs else loss
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
            # DataCollatorForLanguageModeling creates labels as copies of input_ids (not shifted)
            # So we need to mask BEFORE shifting, then shift to match shifted logits
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
            # Then divide by total tokens (including padding positions) to match train.py normalization
            # train.py uses: reduction='sum' then divides by batch_size_in_tokens (line 779)
            # batch_size_in_tokens = batch["input_ids"].numel() (line 807) - computed BEFORE shifting
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss_sum = loss_fct(logits_for_loss, labels_for_loss)
            
            # Calculate total number of tokens from original input (BEFORE shifting)
            # This matches train.py's batch_size_in_tokens normalization exactly
            # train.py computes this from the original batch before shifting
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                batch_size_in_tokens = input_ids.numel()  # Total tokens in original batch
            else:
                # Fallback: use shifted labels size (shouldn't happen but safer)
                batch_size_in_tokens = labels_for_loss.numel()
            loss = loss_sum / batch_size_in_tokens
            
            return (loss, outputs) if return_outputs else loss
        
        def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
            """
            Override log method to add perplexity metric.
            Perplexity = exp(cross_entropy_loss)
            Aligned with train.py line 965: metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
            """
            # Calculate perplexity from loss if loss is in logs (training loss)
            if "loss" in logs:
                loss_value = logs["loss"]
                # Handle both float and tensor cases
                if isinstance(loss_value, torch.Tensor):
                    logs["perplexity"] = torch.exp(loss_value).item()
                else:
                    # loss_value is already a float, use math.exp for efficiency (same as train.py)
                    logs["perplexity"] = math.exp(loss_value)
                
                # Log warning if perplexity is unusually high (suggesting potential issues)
            
            # Calculate perplexity from eval_loss if eval_loss is in logs (evaluation loss)
            if "eval_loss" in logs:
                eval_loss_value = logs["eval_loss"]
                # Handle both float and tensor cases
                if isinstance(eval_loss_value, torch.Tensor):
                    logs["eval_perplexity"] = torch.exp(eval_loss_value).item()
                else:
                    # eval_loss_value is already a float, use math.exp for efficiency
                    logs["eval_perplexity"] = math.exp(eval_loss_value)
            
            # Call parent log method with start_time
            super().log(logs, start_time)
        
        def on_train_end(self, args, state, control, **kwargs):
            """Remove routing hooks after training."""
            if self.target_experts is not None:
                enforce_expert_routing(self.model, self.target_experts, training_mode=False)
            super().on_train_end(args, state, control, **kwargs)
        
        def on_eval_begin(self, args, state, control, **kwargs):
            """Disable routing enforcement during evaluation."""
            if self.target_experts is not None:
                enforce_expert_routing(self.model, self.target_experts, training_mode=False)
            super().on_eval_begin(args, state, control, **kwargs)
        
        def on_train_begin(self, args, state, control, **kwargs):
            """Re-enable routing enforcement at start of training."""
            if self.target_experts is not None:
                enforce_expert_routing(self.model, self.target_experts, training_mode=True)
            super().on_train_begin(args, state, control, **kwargs)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[token_tracking_callback],
        target_experts=target_experts,  # Pass target_experts to trainer
    )
    
    # Enable gradient checkpointing on the model if requested
    if args.gradient_checkpointing:
        if rank == 0:
            log.info("Enabling gradient checkpointing on model...")
        model.gradient_checkpointing_enable()
        if rank == 0:
            log.info("Gradient checkpointing enabled")
    
    # Train
    log.info("Starting training...")
    log.info("=" * 80)
    
    trainer.train()
    
    # Save final model
    log.info("Saving final model...")
    trainer.save_model()
    log.info(f"Model saved to {args.output_dir}")
    log.info("Training complete!")


if __name__ == "__main__":
    main()

