# OCR-LaTeX-MD

A Vision Encoder-Decoder model for converting mathematical formula images to LaTeX code. This project implements a state-of-the-art OCR system specifically designed for handwritten and printed mathematical formulas.

## 🚀 Features

- **High Accuracy**: Achieves 66.6% BLEU score on test datasets
- **Multi-Modal**: Supports both handwritten and printed mathematical formulas
- **Distributed Training**: Built-in support for multi-GPU training with DDP
- **Easy Inference**: Simple API for batch and single image processing
- **Comprehensive Evaluation**: Built-in metrics and evaluation tools

## 🏗️ Model Architecture

- **Encoder**: Swin Transformer (`microsoft/swin-base-patch4-window7-224-in22k`)
- **Decoder**: GPT-2
- **Framework**: PyTorch with Transformers
- **Training**: Distributed Data Parallel (DDP) support

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: CUDA-compatible GPU (recommended: RTX 3080+ or V100+)
- **RAM**: 16GB+ system RAM
- **VRAM**: 8GB+ GPU memory for training
- **Storage**: 50GB+ free space for datasets and checkpoints

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Anaconda or Miniconda

## 🔧 Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/dotrunghieu0903/OCR-LaTeX-MD.git
cd OCR-LaTeX-MD
```

### Step 2: Create Conda Environment
```bash
# Create a new conda environment
conda create -n ocr-latex python=3.9 -y

# Activate the environment
conda activate ocr-latex
```

### Step 3: Install PyTorch with CUDA
```bash
# For CUDA 11.8 (adjust based on your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only (not recommended for training)
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 4: Install Dependencies
```bash
# Navigate to handwritten directory
cd handwritten

# Install required packages
pip install -r requirements.txt

# Additional conda packages for better performance
conda install pillow numpy scipy -c conda-forge
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Training Data

The model is trained on high-quality mathematical formula datasets:

- **Primary**: [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) (cleaned_formulas split)
- **Evaluation**: [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR) (human_handwrite split)

### Data Split Configuration
- **Training**: 80% of the dataset (~240K samples)
- **Validation**: 10% of the dataset (~30K samples)  
- **Testing**: 10% of the dataset (~30K samples)

```python
dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = train_val_split["train"]
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]
```

## 🎯 Quick Start

### Training from Scratch
```bash
# Activate conda environment
conda activate ocr-latex

# Navigate to training directory
cd handwritten

# Single GPU training
python train.py

# Multi-GPU training (if available)
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### Fine-tuning Existing Model
```bash
# Fine-tune a pre-trained model
python finetune.py
```

### Running Inference
```bash
# Run inference on test samples
python inference.py

# For custom images (modify inference.py)
python inference.py --image_path "path/to/your/formula.png"
```

## 📈 Performance Metrics

### Current Model Performance
- **Test Loss**: 0.1047
- **Test BLEU Score**: 0.6662
- **Inference Speed**: ~0.1-0.5 seconds per image
- **Accuracy**: 85-95% exact match on simple formulas

### Training Configuration
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Learning Rate**: 1e-4 with warmup
- **Epochs**: 10 (increase for better results)
- **Image Size**: 224×468 pixels
- **Max Sequence Length**: 512 tokens

## 🚀 Usage Examples

### Basic Inference
```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image

# Load model components
model = VisionEncoderDecoderModel.from_pretrained("path/to/your/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/checkpoint")
feature_extractor = AutoFeatureExtractor.from_pretrained("path/to/your/checkpoint")

# Process image
image = Image.open("formula_image.png")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Generate LaTeX
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    latex_formula = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated LaTeX: {latex_formula}")
```

### Batch Processing
```python
# Process multiple images
images = [Image.open(f"formula_{i}.png") for i in range(5)]
pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, max_length=512, num_beams=4)
formulas = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for i, formula in enumerate(formulas):
    print(f"Image {i+1}: {formula}")
```

## 📁 Project Structure

```
OCR-LaTeX-MD/
├── README.md                          # This file
├── LICENSE                           # License information
├── Finetune_OCR_AllYouNeeded.ipynb  # Jupyter notebook for fine-tuning
├── Multimodal_OCR.ipynb             # Multimodal OCR experiments
├── TrOCR_Math_Retraining.ipynb      # TrOCR retraining notebook
├── multimodal_ocr.py                # Multimodal OCR script
└── handwritten/                     # Main training directory
    ├── train.py                     # Training from scratch
    ├── finetune.py                  # Fine-tuning script
    ├── inference.py                 # Inference script
    ├── dataset.py                   # Dataset handling
    ├── utils.py                     # Utility functions
    ├── train_config.py              # Training configuration
    ├── requirements.txt             # Python dependencies
    └── ...
```

## ⚙️ Configuration

### Key Training Parameters (in `train_config.py`)
```python
class Config:
    # Model parameters
    encoder_name = "microsoft/swin-base-patch4-window7-224-in22k"
    decoder_name = "gpt2"
    
    # Training parameters
    num_epochs = 10
    batch_size_train = 32
    learning_rate = 1e-4
    max_grad_norm = 1.0
    
    # Image parameters
    image_size = (224, 468)
    max_length = 512
    
    # Checkpoint parameters
    checkpoint_dir = "checkpoints"
    eval_steps = 200
```

### Adjusting for Your Hardware
```python
# For limited GPU memory
batch_size_train = 16        # Reduce batch size
batch_size_val = 16

# For faster training (with more GPUs)
batch_size_train = 64        # Increase batch size
learning_rate = 2e-4         # Increase learning rate
```

## 🔍 Monitoring Training

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting  
- **BLEU Score**: Should increase (target: >0.6)
- **Learning Rate**: Follows warmup schedule

### Checkpoints
- Saved automatically every 200 steps
- Best model saved based on validation BLEU score
- Located in `checkpoints/` directory

### Expected Training Timeline
- **10 epochs**: 6-12 hours (depending on GPU)
- **Convergence**: Usually around epoch 7-10
- **Evaluation**: Every 200 steps

## 🧪 Evaluation & Testing

### Automatic Evaluation
```bash
# The training script automatically evaluates on test set
python train.py  # Includes final evaluation

# Standalone evaluation
python -c "
from utils import evaluate_model
# Evaluation code here
"
```

### Manual Testing
```bash
# Test on sample images
python inference.py

# Check results
cat inference_results.json
```

### Evaluation Metrics
- **BLEU Score**: Measures translation quality
- **Exact Match**: Percentage of perfectly predicted formulas
- **Character-level Accuracy**: Fine-grained accuracy measure

## 🔧 Troubleshooting

### Common Training Issues

#### Out of Memory (OOM)
```python
# Reduce batch size in train_config.py
batch_size_train = 8  # or even smaller
batch_size_val = 8
```

#### Slow Convergence
```python
# Increase learning rate
learning_rate = 2e-4

# Or train for more epochs
num_epochs = 20
```

#### CUDA Issues
```bash
# Reinstall PyTorch with correct CUDA version
conda uninstall pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Environment Issues
```bash
# Reset conda environment
conda deactivate
conda remove -n ocr-latex --all
# Then follow installation steps again
```

## 📚 Additional Resources

- **Jupyter Notebooks**: Use `Finetune_OCR_AllYouNeeded.ipynb` for interactive training
- **Pre-trained Models**: Available on Hugging Face Hub
- **Datasets**: Links provided in training data section
- **Documentation**: Check individual script docstrings

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Base Architecture**: Inspired by [im2latex](https://github.com/d-gurgurov/im2latex)
- **Datasets**: Thanks to OleehyO and linxy for providing high-quality datasets
- **Framework**: Built with 🤗 Transformers and PyTorch

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/dotrunghieu0903/OCR-LaTeX-MD/issues) page
2. Create a new issue with detailed description
3. Include your conda environment info: `conda list`
4. Include error logs and system specifications

---

**Happy Training! 🚀**
