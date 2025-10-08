"""
Shared dataset utilities for LaTeX OCR training and inference.
Contains the LatexDataset class, data collator, and preprocessing utilities.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image


class LatexDataset(Dataset):
    """
    Dataset class for LaTeX OCR training and inference.
    Handles image preprocessing and LaTeX tokenization.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=(224, 468),
        max_length=512
    ):
        """
        Initialize the LatexDataset.
        
        Args:
            dataset: The dataset containing images and LaTeX text
            tokenizer: Tokenizer for LaTeX sequences
            feature_extractor: Feature extractor for images
            phase: Phase of training ('train', 'val', 'test')
            image_size: Target size for image resizing
            max_length: Maximum length for tokenized sequences
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def get_train_transform(self):
        """
        Get the image transformation function.
        
        Returns:
            Function for transforming images
        """
        def train_transform(image):
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image
        return train_transform

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing pixel_values and labels
        """
        item = self.dataset[idx]
        latex_sequence = item['text']
        image = item['image']

        # Convert RGBA to RGB for some images that have alpha channels
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Image processing
        try:
            pixel_values = self.feature_extractor(
                images=image.resize(self.image_size),
                return_tensors="pt",
            ).pixel_values.squeeze()
            if pixel_values.ndim == 0:
                raise ValueError("Processed image has no dimensions")
        except Exception as e:
            print(f"Error when processing image at index {idx}: {str(e)}")
            # Provide a default tensor in case of error
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # Tokenization
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
            # Provide a default tensor in case of error
            latex_tokens = torch.zeros(1, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": latex_tokens
        }


def data_collator(batch):
    """
    Collate function for batching data.
    
    Args:
        batch: List of data items from the dataset
        
    Returns:
        Dictionary containing batched pixel_values and labels
    """
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Stack pixel values (they should all be the same size)
    pixel_values = torch.stack(pixel_values)
    
    # Pad labels to the same length
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def filter_dataset(dataset):
    """
    Filter dataset to remove invalid samples.
    
    Args:
        dataset: Dataset to filter
        
    Returns:
        Filtered dataset
    """
    def is_valid_sample(sample):
        try:
            # Check if image exists and is valid
            if sample['image'] is None:
                return False
            
            # Check if text exists and is not empty
            if not sample['text'] or len(sample['text'].strip()) == 0:
                return False
                
            # Check if text is reasonable length (not too long or too short)
            if len(sample['text']) < 3 or len(sample['text']) > 1000:
                return False
                
            return True
        except:
            return False
    
    return dataset.filter(is_valid_sample)


def preprocess_image(image, image_size=(224, 468)):
    """
    Preprocess a single image.
    
    Args:
        image: PIL Image to preprocess
        image_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(image_size)
    
    # Convert to numpy array and normalize
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    
    return image