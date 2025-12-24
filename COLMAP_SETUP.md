# COLMAP Environment Setup

This document describes how to set up a conda environment to run the `video_to_colmap.py` script.

## Quick Setup

```bash
# Create conda environment with Python 3.10
conda create -n colmap python=3.10 -y

# Activate the environment
conda activate colmap

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install pycolmap with CUDA support (recommended for GPU acceleration)
pip install pycolmap-cuda12

# Or install CPU-only version (slower)
# pip install pycolmap
```

## Verification

Verify the installation:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check pycolmap and CUDA device support
python -c "import pycolmap; print(f'pycolmap version: {pycolmap.__version__}'); print(f'CUDA device: {pycolmap.Device.cuda}')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA available: True
pycolmap version: 3.13.0
CUDA device: Device.cuda
```

## GPU Acceleration

The `video_to_colmap.py` script automatically uses GPU acceleration when `pycolmap-cuda12` is installed.

To verify GPU is being used during feature extraction, look for this log message:
```
Creating SIFT GPU feature extractor
```

If you see `Creating SIFT CPU feature extractor` instead, GPU is not being used. Ensure you have:
1. Installed `pycolmap-cuda12` (not `pycolmap`)
2. A compatible NVIDIA GPU with CUDA drivers installed

## Usage

Run the COLMAP reconstruction pipeline:

```bash
conda activate colmap

# From images directory
export IMAGE_DIR="/path/to/your/images"
export OUTPUT_DIR="/path/to/output"
python scripts/video_to_colmap.py ${IMAGE_DIR} ${OUTPUT_DIR}

# From video file (extracts frames at specified fps)
python scripts/video_to_colmap.py input.mp4 output_dir --fps 2

# Specify GPU
python scripts/video_to_colmap.py input_dir output_dir --gpu 0
```
