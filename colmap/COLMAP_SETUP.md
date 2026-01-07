# COLMAP Environment Setup


## System-Level COLMAP Installation

The scripts require the COLMAP binary with CUDA support. Download the pre-built binary from GitHub releases:

```bash
# Download COLMAP 3.11.1 with CUDA support (Ubuntu 24.04)
wget https://github.com/colmap/colmap/releases/download/3.11.1/colmap-3.11.1-linux-cuda.tar.gz

# Extract and install
tar -xzf colmap-3.11.1-linux-cuda.tar.gz
cp colmap-3.11.1-linux-cuda/bin/colmap ~/.local/bin/
chmod +x ~/.local/bin/colmap

# Verify installation
colmap --help
```

**Requirements:**
- CUDA 12.x installed (the binary links against libcudart.so.12, libcusolver.so.11, libcublas.so.12, etc.)
- System dependencies (Ubuntu 24.04):
  ```bash
  sudo apt install libgmp10 libglew2.2 libboost-program-options1.83.0 \
      libspqr4 libcholmod5 libmetis5 liblapack3 libblas3 libsqlite3-0 \
      libgoogle-glog0v6t64 libfreeimage3 qtbase5-dev
  ```

**Note:** The COLMAP binary must be available in your system PATH. The example above installs to `~/.local/bin/`, which should already be in PATH on most systems.


## Quick Setup

```bash
# Create conda environment with Python 3.10
conda create -n colmap python=3.10 -y

# Activate the environment
conda activate colmap

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install pycolmap with CUDA support (recommended for GPU acceleration)
pip install pycolmap==3.13.0

# Install additional dependencies
pip install opencv-python tqdm
```

## Usage

There are two main scripts depending on your input type:

### video_to_colmap.py

The main script (`colmap/video_to_colmap.py:1`) processes video files and extracts frames automatically:

**Single Video:**

```bash
python colmap/video_to_colmap.py \
  --inputs input.mp4 \
  --output output_dir \
  --fps 2 \
  --gpu 0 \
  --set-config colmap/configs/exhaustive.json
```

**Multiple Videos (Multi-Camera):**

```bash
python colmap/video_to_colmap.py \
  --inputs video1.mp4 video2.mp4 video3.mp4 \
  --output output_dir \
  --fps 4 \
  --gpu 0 \
  --set-config colmap/configs/exhaustive.json
```
**Test Video Processing (Skip COLMAP):**

```bash
# Useful for testing frame extraction and mask generation
python colmap/video_to_colmap.py \
  --inputs video1.mp4 video2.mp4 \
  --output output_dir \
  --fps 4 \
  --skip-colmap
```

### image_to_colmap.py

**Single Image Directory:**

```bash
python colmap/image_to_colmap.py \
  --inputs /path/to/images \
  --output output_dir \
  --gpu 0 \
  --original-fps 30 \
  --target-fps 4 \
  --set-config colmap/configs/exhaustive.json
```

**Multiple Image Directories (Multi-Camera):**

```bash
python colmap/image_to_colmap.py \
  --inputs "/path/to/images1,/path/to/images2,/path/to/images3" \
  --output output_dir \
  --gpu 0 \
  --original-fps 30 \
  --target-fps 4 \
  --set-config colmap/configs/exhaustive.json
```

## Output Structure

After running either script, the output directory contains:

```
output_dir/
├── images/           # Extracted/linked frames
├── depths/           # Linked depth images (if found by image_to_colmap.py)
├── masks/            # Auto-generated masks (if applicable)
└── sparse/
    └── 0/            # COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
```

**Note:** The `depths/` directory is only created when using `image_to_colmap.py` with source directories that have corresponding depth data.
