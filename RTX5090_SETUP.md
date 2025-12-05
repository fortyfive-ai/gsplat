# gsplat Setup Guide for RTX 5090 (Blackwell Architecture)

This guide provides step-by-step instructions to set up gsplat on machines with NVIDIA RTX 5090 GPUs.

## Prerequisites

- NVIDIA RTX 5090 GPU
- CUDA Toolkit 12.9+ installed at `/usr/local/cuda-12.9`
- Python 3.12+
- GCC 13 or 14 (not GCC 15, which is incompatible with CUDA 12.9)

## Step 1: Clone the Repository

```bash
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git submodule update --init --recursive
```

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv_gsplat
source venv_gsplat/bin/activate
```

## Step 3: Install PyTorch Nightly with CUDA 12.8 Support

The RTX 5090 uses Blackwell architecture (sm_120 / compute capability 12.0), which requires PyTorch nightly builds:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Step 4: Install gsplat

Set environment variables and install gsplat:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

pip install --no-build-isolation -e .
```

## Step 5: Apply RTX 5090 Compatibility Patch

The RTX 5090 has a known issue with `torch.einsum` operations that use batched CUBLAS calls. You need to modify `gsplat/strategy/ops.py`:

### Patch 1: Fix `split()` function (around line 144-154)

Find this code:
```python
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]
```

Replace with:
```python
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    # Compute samples using broadcasting instead of einsum to avoid CUBLAS issues
    # on newer GPU architectures (e.g., RTX 5090 / Blackwell)
    # Original: samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, rand)
    rand = torch.randn(2, len(scales), 3, device=device)
    scaled_rand = rand * scales.unsqueeze(0)  # [2, N, 3]
    # Expand for broadcasting: rotmats[N, 3, 3], scaled_rand[2, N, 1, 3]
    # Result after element-wise multiplication: [2, N, 3, 3], sum over last dim -> [2, N, 3]
    samples = (rotmats.unsqueeze(0) * scaled_rand.unsqueeze(2)).sum(dim=-1)  # [2, N, 3]
```

### Patch 2: Fix `inject_noise_to_position()` function (around line 365-370)

Find this code:
```python
    noise = (
        torch.randn_like(params["means"])
        * (op_sigmoid(1 - opacities)).unsqueeze(-1)
        * scaler
    )
    noise = torch.einsum("bij,bj->bi", covars, noise)
    params["means"].add_(noise)
```

Replace with:
```python
    noise = (
        torch.randn_like(params["means"])
        * (op_sigmoid(1 - opacities)).unsqueeze(-1)
        * scaler
    )
    # Use broadcasting instead of einsum to avoid CUBLAS issues on newer GPUs
    # Original: noise = torch.einsum("bij,bj->bi", covars, noise)
    # covars: [N, 3, 3], noise: [N, 3] -> result: [N, 3]
    noise = (covars * noise.unsqueeze(1)).sum(dim=-1)
    params["means"].add_(noise)
```

## Step 6: Install Example Dependencies (Optional)

If you want to run the training examples:

```bash
cd examples
pip install viser imageio[ffmpeg] "numpy<2.0.0" scikit-learn tqdm "torchmetrics[image]" \
    opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib splines packaging

# Install git-based dependencies with --no-build-isolation
pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
pip install git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787

# These require CUDA compilation
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
pip install --no-build-isolation git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe
```

## Step 7: Create Activation Script (Recommended)

Create a convenience script `activate_gsplat.sh` in the repo root:

```bash
#!/bin/bash
# Activation script for gsplat environment on RTX 5090

source /path/to/gsplat/venv_gsplat/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

echo "gsplat environment activated for RTX 5090!"
```

Make it executable:
```bash
chmod +x activate_gsplat.sh
```

## Verification

Test that everything works:

```bash
source activate_gsplat.sh

python -c "
import torch
import gsplat
from gsplat import rasterization

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'gsplat: {gsplat.__version__}')

# Test rasterization
N = 1000
device = 'cuda'
means = torch.randn(N, 3, device=device)
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=device) * 0.1
opacities = torch.ones(N, device=device)
colors = torch.rand(N, 3, device=device)
viewmat = torch.eye(4, device=device)[None]
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32, device=device)[None]

renders, alphas, meta = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmat, Ks=K, width=640, height=480,
)
print(f'Rasterization test: SUCCESS (output shape: {renders.shape})')
"
```

## Running Training

Download dataset and run training:

```bash
source activate_gsplat.sh
cd examples

# Download mipnerf360 dataset
python datasets/download_dataset.py --dataset mipnerf360

# Extract (if unzip is not available)
cd data/360_v2
python -c "import zipfile; zipfile.ZipFile('360_v2.zip', 'r').extractall('.')"
rm 360_v2.zip
cd ../..

# Run training on a scene
python simple_trainer.py default \
    --disable_viewer \
    --data_factor 2 \
    --data_dir data/360_v2/bonsai/ \
    --result_dir results/bonsai/
```

## Troubleshooting

### Error: `nvcc not found`
Make sure CUDA_HOME is set correctly:
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
```

### Error: `unsupported GNU version`
CUDA 12.9 requires GCC 14 or earlier. Use system GCC instead of conda GCC:
```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
```

### Error: `Unsupported gpu architecture 'compute_120'`
Make sure you're using CUDA 12.9 which supports sm_120:
```bash
/usr/local/cuda-12.9/bin/nvcc --list-gpu-arch | grep 120
```

### Error: `CUBLAS_STATUS_INVALID_VALUE` during training
This means you haven't applied the patches in Step 5. The `torch.einsum` operations need to be replaced with broadcasting operations.

### Error: `No module named 'glm'` or missing headers
Initialize git submodules:
```bash
git submodule update --init --recursive
```
