#!/bin/bash
# Activation script for gsplat environment on RTX 5090
# Usage: source activate_gsplat.sh

# Activate virtual environment
source /home/yanbinghan/fortyfive/gsplat/venv_gsplat/bin/activate

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Required for building extensions with CUDA 12.9 on RTX 5090 (sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

echo "gsplat environment activated!"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo "  gsplat: $(python -c 'import gsplat; print(gsplat.__version__)')"
