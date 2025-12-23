# COLMAP Environment Setup

This document describes how to set up a conda environment to run the `video_to_colmap.py` script.

## Prerequisites

- NVIDIA GPU with CUDA support
- Miniconda or Anaconda installed
- NVIDIA drivers installed (tested with CUDA 12.x compatible drivers)

## Quick Setup

```bash
# Create conda environment with Python 3.10
conda create -n colmap python=3.10 -y

# Activate the environment
conda activate colmap

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install pycolmap
pip install pycolmap
```

## Verification

Verify the installation:

```bash
conda activate colmap

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check pycolmap
python -c "import pycolmap; print(f'pycolmap version: {pycolmap.__version__}')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA available: True
pycolmap version: 3.13.0
```

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

## Example

```bash
conda activate colmap

export IMAGE_DIR="${DROPBOX}/cobot_office_raw_images/008/perspective_images"
export OUTPUT_DIR="/home/yanbing/fortyfive/cobot/datasets/colmap/cobot_office/008"
python scripts/video_to_colmap.py ${IMAGE_DIR} ${OUTPUT_DIR}_colmap
```

## Installed Packages

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| pycolmap | 3.13.0 |
| numpy | 2.2.6 |

## Troubleshooting

### CUDA not available
If `torch.cuda.is_available()` returns `False`:
1. Ensure NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA version compatibility with PyTorch
3. Try reinstalling PyTorch with a different CUDA version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### pycolmap import errors
If you encounter issues with pycolmap:
```bash
pip uninstall pycolmap
pip install pycolmap --no-cache-dir
```
