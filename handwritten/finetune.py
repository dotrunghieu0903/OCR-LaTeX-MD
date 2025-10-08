"""
Fine-tuning script for LaTeX OCR model using LoRA/IA3 on handwritten dataset.
Refactored to use shared components.
"""

from dataset import LatexDataset, data_collator, filter_dataset
from utils import (
    setup_logging, setup_distributed_training, set_seed,
    setup_lora_config, setup_ia3_config, 
    setup_optimizer_and_scheduler, evaluate_model, save_checkpoint,
    log_metrics
)
from train_config import Config

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from peft import get_peft_model
import evaluate
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np

# Setup logging and distributed training
setup_logging()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed_training()

torch.set_float32_matmul_precision(Config.float32_matmul_precision)

# Setting a seed for reproducibility
set_seed(Config.seed)

# Getting pre-trained tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained(Config.feature_extractor)

# Loading new dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")

# Combining train, val, and test splits and then splitting into train and val
train_ds = new_dataset['train']
val_ds = new_dataset['validation']
test_ds = new_dataset['test']

train_ds = filter_dataset(train_ds)
val_ds = filter_dataset(val_ds)

if master_process:
    print("Length of train set after splitting:", len(train_ds))
    print("Length of val set after splitting:", len(val_ds))

# Setting up the model
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to(device)

# Setup LoRA configuration
lora_config = setup_lora_config()

# Setup IA3 configuration  
ia3_config = setup_ia3_config()

# Choose which PEFT method to use (LoRA or IA3)
USE_LORA = True  # Set to False to use IA3

if USE_LORA:
    model = get_peft_model(model, lora_config)
    if master_process:
        print("Using LoRA fine-tuning")
        model.print_trainable_parameters()
else:
    model = get_peft_model(model, ia3_config)
    if master_process:
        print("Using IA3 fine-tuning")
        model.print_trainable_parameters()

# Wrap model with DDP for distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank, find_unused_parameters=False)

# Creating datasets and dataloader
train_dataset = LatexDataset(train_ds, tokenizer, feature_extractor, phase='train')
val_dataset = LatexDataset(val_ds, tokenizer, feature_extractor, phase='val')
test_dataset = LatexDataset(test_ds, tokenizer, feature_extractor, phase='test')

train_sampler = DistributedSampler(train_dataset) if ddp else None
val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None
test_sampler = DistributedSampler(test_dataset, shuffle=False) if ddp else None

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=Config.batch_size_train, 
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    collate_fn=data_collator, 
    drop_last=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=Config.batch_size_val, 
    sampler=val_sampler,
    collate_fn=data_collator, 
    drop_last=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=Config.batch_size_val, 
    sampler=test_sampler,
    collate_fn=data_collator, 
    drop_last=True
)

# Training parameters
learning_rate = 2e-4  # Fine-tuning specific learning rate
num_epochs = 40       # Increased epochs for fine-tuning
eval_steps = 400      # More frequent evaluation

# Calculate total steps
total_steps = len(train_dataloader) * num_epochs

# Setup optimizer and scheduler
optimizer, scheduler = setup_optimizer_and_scheduler(model, Config, total_steps)

# Setup evaluation metric
bleu_metric = evaluate.load(Config.bleu)

# Variables for tracking best model
best_val_bleu = 0.0
best_checkpoint_step = None

def train_lora(model, train_dataloader, optimizer, scheduler, device, num_epochs, eval_steps, val_dataloader, tokenizer, bleu_metric, local_rank=0):
    """Training loop for LoRA/IA3 fine-tuning."""
    model.train()
    global best_val_bleu, best_checkpoint_step
    
    step = 0
    train_losses = []
    
    for epoch in range(num_epochs):
        if ddp:
            train_dataloader.sampler.set_epoch(epoch)
            
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=local_rank != 0)
        
        for batch in epoch_iterator:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            step += 1
            
            # Update progress bar
            if local_rank == 0:
                epoch_iterator.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
            # Evaluate and save checkpoint
            if step % eval_steps == 0:
                if master_process:
                    print(f"\nEvaluating at step {step}...")
                
                val_loss, val_bleu_scores = evaluate_model(
                    model, val_dataloader, device, tokenizer, bleu_metric, 
                    max_batches=50, stage="val"
                )
                
                if master_process:
                    metrics = {
                        "train_loss": np.mean(train_losses[-eval_steps:]),
                        "val_loss": val_loss,
                        "val_bleu": val_bleu_scores.get("bleu", 0.0)
                    }
                    log_metrics(metrics, step, "validation")
                    
                    # Save checkpoint if best model
                    current_bleu = val_bleu_scores.get("bleu", 0.0)
                    if current_bleu > best_val_bleu:
                        best_val_bleu = current_bleu
                        best_checkpoint_step = step
                        save_checkpoint(model, tokenizer, step, Config.checkpoint_dir, is_best=True)
                        print(f"New best model saved with BLEU: {current_bleu:.4f}")
    
    return train_losses

# Training loop for LoRA fine-tuning
train_losses = train_lora(
    model, train_dataloader, optimizer, scheduler, device, 
    num_epochs, eval_steps, val_dataloader, tokenizer, bleu_metric, 
    local_rank=ddp_local_rank
)

# Evaluating on the final test dataset
if master_process:
    print("\nEvaluating on test set...")
    
    if best_checkpoint_step:
        checkpoint_dir = f"checkpoints/checkpoint_step_{best_checkpoint_step}"
        best_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)
        best_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    else:
        best_model = model
        best_tokenizer = tokenizer
    
    # Evaluating on test set
    test_loss, test_bleu_scores = evaluate_model(
        best_model, test_dataloader, device, best_tokenizer, bleu_metric, stage='test'
    )
    
    print(f"Test Loss: {test_loss}")
    print(f"Test BLEU: {test_bleu_scores}")
    
    # Save final results
    results = {
        "best_checkpoint_step": best_checkpoint_step,
        "best_val_bleu": best_val_bleu,
        "test_loss": test_loss,
        "test_bleu": test_bleu_scores
    }
    
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    import json
    with open(os.path.join(Config.checkpoint_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("Training completed!")

if ddp:
    torch.distributed.destroy_process_group()