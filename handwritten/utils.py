"""
Shared utilities for LaTeX OCR training and inference.
Contains common imports, distributed training setup, evaluation functions, and helper functions.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup,
    logging as transformers_logging
)

import evaluate
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import os
import json
import time
import warnings


def setup_logging():
    """Setup logging and warnings."""
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_warning()
    transformers_logging.set_verbosity_error()


def setup_distributed_training():
    """
    Setup distributed training environment.
    
    Returns:
        Tuple containing (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process)
    """
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # Single GPU or CPU setup
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizers(config, device):
    """
    Load model, tokenizer, and feature extractor.
    
    Args:
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Tuple containing (model, tokenizer, feature_extractor)
    """
    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor)
    
    # Load model
    model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to(device)
    
    return model, tokenizer, feature_extractor


def setup_lora_config():
    """
    Setup LoRA configuration for parameter-efficient fine-tuning.
    
    Returns:
        LoRA configuration object
    """
    from peft import LoraConfig
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            # Encoder (Swin Transformer) modules
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
            # Decoder (GPT-2) modules
            "c_attn",
            "c_proj",
            "c_fc",
            "attn.c_proj",
        ],
        lora_dropout=0.1,
        bias="none",
    )
    
    return lora_config


def setup_ia3_config():
    """
    Setup IA3 configuration for parameter-efficient fine-tuning.
    
    Returns:
        IA3 configuration object
    """
    from peft import IA3Config
    
    ia3_config = IA3Config(
        target_modules=[
            # Encoder (Swin Transformer) modules
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
            # Decoder (GPT-2) modules
            "c_attn",
            "c_proj",
            "c_fc",
            "attn.c_proj",
        ],
        feedforward_modules=[
            "mlp.fc1",
            "mlp.fc2",
            "c_fc",
        ],
    )
    
    return ia3_config


def create_data_loaders(train_dataset, val_dataset, config, ddp=False):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object
        ddp: Whether to use distributed data parallel
        
    Returns:
        Tuple containing (train_dataloader, val_dataloader)
    """
    from .dataset import data_collator
    
    if ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size_train, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=data_collator, 
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size_val, 
        sampler=val_sampler,
        collate_fn=data_collator, 
        drop_last=True
    )
    
    return train_dataloader, val_dataloader


def setup_optimizer_and_scheduler(model, config, total_steps):
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration object
        total_steps: Total number of training steps
        
    Returns:
        Tuple containing (optimizer, scheduler)
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def evaluate_model(model, val_dataloader, device, tokenizer, bleu_metric, max_batches=None, stage="val"):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        device: Device to run evaluation on
        tokenizer: Tokenizer for decoding
        bleu_metric: BLEU metric for evaluation
        max_batches: Maximum number of batches to evaluate (None for all)
        stage: Stage of evaluation ("val", "test", "final")
        
    Returns:
        Tuple containing (average_loss, bleu_scores)
    """
    model.eval()
    total_loss = 0
    predictions = []
    references = []
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc=f"Evaluating {stage}")):
            if max_batches and i >= max_batches:
                break
                
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predictions and references
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Filter out -100 values from labels before decoding
            filtered_labels = []
            for label_seq in labels:
                filtered_seq = label_seq[label_seq != -100]
                filtered_labels.append(filtered_seq)
            
            ref_texts = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            
            predictions.extend(pred_texts)
            references.extend(ref_texts)
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Calculate BLEU scores
    try:
        # BLEU metric expects references as list of lists
        formatted_references = [[ref] for ref in references]
        bleu_scores = bleu_metric.compute(predictions=predictions, references=formatted_references)
        print(f"BLEU calculation successful: {bleu_scores}")
    except Exception as e:
        print(f"BLEU calculation failed: {e}")
        bleu_scores = {"bleu": 0.0}
    
    model.train()
    return avg_loss, bleu_scores


def save_checkpoint(model, tokenizer, step, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        step: Current training step
        checkpoint_dir: Base directory for checkpoints
        is_best: Whether this is the best checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if is_best:
        save_dir = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
    else:
        save_dir = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Checkpoint saved at {save_dir}")


def log_metrics(metrics, step, stage="train"):
    """
    Log training metrics.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        stage: Stage of training ("train", "val", "test")
    """
    log_str = f"Step {step} | {stage.capitalize()} | "
    log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(log_str)


def load_datasets(config, dataset_name=None):
    """
    Load datasets for training.
    
    Args:
        config: Configuration object
        dataset_name: Name of dataset to load (if different from config)
        
    Returns:
        Tuple containing datasets (train_ds, val_ds, test_ds)
    """
    if dataset_name:
        dataset = load_dataset(dataset_name)
    else:
        # Load based on config
        dataset = load_dataset(config.train_dataset_path, config.split_dataset_name)
        
        # Split dataset
        train_val_split = dataset["train"].train_test_split(test_size=config.val_test_size, seed=42)
        train_ds = train_val_split["train"]
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]
        
        return train_ds, val_ds, test_ds
    
    # For predefined splits
    train_ds = dataset.get('train')
    val_ds = dataset.get('validation') or dataset.get('val')
    test_ds = dataset.get('test')
    
    return train_ds, val_ds, test_ds