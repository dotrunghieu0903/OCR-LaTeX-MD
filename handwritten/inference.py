from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# Load pre-trained model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("DGurgurov/im2latex")

# Load dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")
train_dataset = new_dataset["train"]
test_dataset = new_dataset["test"]

class LatexDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=(224, 468),
        max_length=512
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.train_transform(image)
        text = item["latex"]
        return {"pixel_values": image, "input_ids": text}