"""
Inference script for LaTeX OCR model.
Refactored to use shared components.
"""

from dataset import LatexDataset, data_collator
from utils import setup_logging, set_seed
from train_config import Config

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset
import numpy as np

# Setup logging
setup_logging()

# Set seed for reproducibility
set_seed(Config.seed)

# Load pre-trained model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("DGurgurov/im2latex")

# Load dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")
train_dataset = new_dataset["train"]
test_dataset = new_dataset["test"]

# Creating datasets and dataloader
latex_dataset = LatexDataset(test_dataset, tokenizer, feature_extractor, phase='val')
test_dataloader = DataLoader(latex_dataset, batch_size=1, collate_fn=data_collator)

# Making inference on test images
print("Starting inference on test images...")
inferences = []

model.eval()
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        if i >= 10:  # Limit to first 10 samples for demo
            break
            
        pixel_values = batch["pixel_values"].to("cuda")
        
        # Generate prediction
        generated_ids = model.generate(
            pixel_values,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode prediction
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Get ground truth
        ground_truth = tokenizer.decode(batch["labels"][0], skip_special_tokens=True)
        
        inferences.append({
            "prediction": pred_text,
            "ground_truth": ground_truth
        })
        
        print(f"Sample {i+1}:")
        print(f"Prediction: {pred_text}")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 50)

print(f"Completed inference on {len(inferences)} samples.")

# Optionally save results to file
import json
with open("inference_results.json", "w") as f:
    json.dump(inferences, f, indent=2)

print("Results saved to inference_results.json")