#!/bin/bash
# Activation script for gsplat training environment on RTX 3090
# Usage: source activate_env.sh
# Asset: Q

export CUDA_HOME=/home/yanbinghan/fortyfive/cobot/gsplat/builds/cuda_nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"

echo "=========================================="
echo "gsplat Environment Activated (RTX 3090)"
echo "=========================================="
echo "  CUDA_HOME: $CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo ""
echo "Installed packages:"
echo "  - PyTorch 2.6.0 (CUDA 12.4)"
echo "  - gsplat 1.5.3"
echo "  - fused-ssim"
echo "  - fused-bilagrid"
echo ""
echo "Example training commands:"
echo "  cd examples"
echo "  # Default 3DGS training:"
echo "  python simple_trainer.py default --data_dir <path_to_colmap_data>"
echo "  # MCMC strategy:"
echo "  python simple_trainer.py mcmc --data_dir <path_to_colmap_data>"
echo "=========================================="

