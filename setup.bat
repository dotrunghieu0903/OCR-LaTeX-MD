@echo off
REM OCR-LaTeX-MD Setup Script for Windows
REM This script sets up the conda environment for the OCR-LaTeX-MD project

echo 🚀 Setting up OCR-LaTeX-MD environment...

REM Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda is not installed. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ Conda found

REM Check if environment already exists
conda env list | findstr "ocr-latex" >nul
if %errorlevel% equ 0 (
    echo ⚠️  Environment 'ocr-latex' already exists
    set /p "choice=Do you want to remove and recreate it? (y/N): "
    if /i "%choice%"=="y" (
        echo 🗑️  Removing existing environment...
        conda env remove -n ocr-latex -y
    ) else (
        echo ❌ Setup cancelled
        pause
        exit /b 1
    )
)

REM Create environment from yml file
echo 📦 Creating conda environment from environment.yml...
conda env create -f environment.yml
if %errorlevel% equ 0 (
    echo ✅ Environment created successfully
    goto :success
) else (
    echo ❌ Failed to create environment from yml file
    echo 🔄 Trying manual installation...
    
    REM Fallback manual installation
    conda create -n ocr-latex python=3.9 -y
    call conda activate ocr-latex
    
    REM Install PyTorch with CUDA
    echo 🔥 Installing PyTorch with CUDA support...
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    REM Install other conda packages
    echo 📚 Installing conda packages...
    conda install pillow numpy scipy -c conda-forge -y
    
    REM Install pip packages
    echo 🐍 Installing pip packages...
    pip install datasets==2.20.0 evaluate==0.4.2 transformers==4.32.0 peft==0.11.1 tqdm==4.66.4
    pip install huggingface-hub accelerate tensorboard wandb matplotlib seaborn
)

:success
echo.
echo 🎉 Setup completed successfully!
echo.
echo To activate the environment, run:
echo    conda activate ocr-latex
echo.
echo To verify the installation, run:
echo    conda activate ocr-latex
echo    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.
echo To start training, run:
echo    cd handwritten
echo    python train.py
echo.
echo 📖 For more information, see README.md

pause