# OCR-LaTeX-MD Quick Start Guide

## 🚀 Ultra-Fast Setup (Windows)

### Option 1: Automated Setup
```powershell
# Run the setup script
.\setup.bat
```

### Option 2: Manual Setup
```powershell
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ocr-latex

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Quick Training

```powershell
# Activate environment
conda activate ocr-latex

# Navigate to training directory
cd handwritten

# Start training
python train.py
```

## 🔍 Quick Inference

```powershell
# Run inference on test samples
python inference.py

# View results
Get-Content inference_results.json
```

## 📊 Monitor Training

Training automatically saves:
- **Checkpoints**: `checkpoints/checkpoint_step_*`
- **Best model**: `checkpoints/best_model/`
- **Logs**: Console output shows progress

**Key metrics to watch:**
- Training Loss: Should decrease
- Validation BLEU: Should increase (target >0.6)
- Training time: ~6-12 hours for 10 epochs

## 🛠️ Quick Troubleshooting

### Out of Memory
```python
# Edit handwritten/train_config.py
batch_size_train = 8  # Reduce from 32
```

### CUDA Issues
```powershell
# Reinstall PyTorch
conda uninstall pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Environment Issues
```powershell
# Reset environment
conda deactivate
conda env remove -n ocr-latex
.\setup.bat
```

## 📁 Key Files

- `handwritten/train.py` - Main training script
- `handwritten/inference.py` - Inference script  
- `handwritten/train_config.py` - Configuration
- `environment.yml` - Conda environment
- `setup.bat` - Automated setup

## 🎯 Expected Results

- **Training time**: 6-12 hours (10 epochs)
- **BLEU Score**: 0.65+ for good models
- **Memory usage**: 6-8GB GPU memory
- **Checkpoint size**: ~1GB per checkpoint

Ready to start? Run `.\setup.bat` and then `python handwritten/train.py`! 🚀