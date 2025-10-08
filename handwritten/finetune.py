import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup
)

from peft import LoraConfig, IA3Config, get_peft_model

from PIL import Image
import evaluate
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
import json
import time

from train_config import Config

import warnings
warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

ddp = int(os.environ.get('RANK', -1)) != -1
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

torch.set_float32_matmul_precision(Config.float32_matmul_precision)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# setting a seed for reproducibility
set_seed(Config.seed)

# getting pre-trained tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained(Config.feature_extractor)

# loading new dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")

# combining train, val, and test splits and then splitting into train and val
train_ds = new_dataset['train']
val_ds = new_dataset['validation']
test_ds = new_dataset['test']

def filter_dataset(dataset):
    def is_valid_sample(sample):
        try:
            image = sample['image']
            latex = sample['text']
            return image is not None and latex is not None and len(latex) > 0
        except:
            return False

    return dataset.filter(is_valid_sample)

train_ds = filter_dataset(train_ds)
val_ds = filter_dataset(val_ds)

if master_process:
    print("Length of train set after splitting:", len(train_ds))
    print("Length of val set after splitting:", len(val_ds))

# setting up the model
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to(device)

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
    init_ia3_weights=True,
)

# applying lora
model = get_peft_model(model, lora_config)
if master_process:  
    model.print_trainable_parameters() # print trainable parameters

torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank, find_unused_parameters=False)

class LatexDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=Config.image_size,
        max_length=512
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def __len__(self):
        return len(self.dataset)

    def get_train_transform(self):
        def train_transform(image):
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image
        return train_transform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        latex_sequence = item['text']
        image = item['image']

        # converting RGBA to RGB for the test set --> some images have alphas
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # image processing
        try:
            pixel_values = self.feature_extractor(
                images=image.resize(self.image_size),
                return_tensors="pt",
            ).pixel_values.squeeze()
            if pixel_values.ndim == 0:
                raise ValueError("Processed image has no dimensions")
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # tokenization
        try:
            latex_tokens = self.tokenizer(
                latex_sequence,
                padding=False,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.squeeze()
            if latex_tokens.ndim == 0:
                raise ValueError("Tokenized latex has no dimensions")
        except Exception as e:
            print(f"Error tokenizing latex at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            latex_tokens = torch.zeros(1, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": latex_tokens
        }