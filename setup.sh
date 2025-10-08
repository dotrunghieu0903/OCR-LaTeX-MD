#!/bin/bash

# OCR-LaTeX-MD Setup Script
# This script sets up the conda environment for the OCR-LaTeX-MD project

echo "🚀 Setting up OCR-LaTeX-MD environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found"

# Check if environment already exists
if conda env list | grep -q "ocr-latex"; then
    echo "⚠️  Environment 'ocr-latex' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ocr-latex -y
    else
        echo "❌ Setup cancelled"
        exit 1
    fi
fi

# Create environment from yml file
echo "📦 Creating conda environment from environment.yml..."
if conda env create -f environment.yml; then
    echo "✅ Environment created successfully"
else
    echo "❌ Failed to create environment from yml file"
    echo "🔄 Trying manual installation..."
    
    # Fallback manual installation
    conda create -n ocr-latex python=3.12 -y
    conda activate ocr-latex
    
    # Install PyTorch with CUDA
    echo "🔥 Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other conda packages
    echo "📚 Installing conda packages..."
    conda install pillow numpy scipy -c conda-forge -y
    
    # Install pip packages
    echo "🐍 Installing pip packages..."
    pip install datasets==2.20.0 evaluate==0.4.2 transformers==4.32.0 peft==0.11.1 tqdm==4.66.4
    pip install huggingface-hub accelerate tensorboard wandb matplotlib seaborn
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "   conda activate ocr-latex"
echo ""
echo "To verify the installation, run:"
echo "   conda activate ocr-latex"
echo "   python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "To start training, run:"
echo "   cd handwritten"
echo "   python train.py"
echo ""
echo "📖 For more information, see README.md"