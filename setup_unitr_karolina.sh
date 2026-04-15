#!/bin/bash
set -e
# Init conda
source /opt/sw/el8/x86_64/anaconda3/2023.03/etc/profile.d/conda.sh || module load Anaconda3
export CONDA_ALWAYS_YES="true"

# Create conda environment
conda create -n unitr python=3.10 -y || true
source activate unitr

# 1. Core PyTorch & Build Tools
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-nvcc=12.1 cuda-toolkit=12.1 cuda-cccl=12.1 cuda-libraries-dev=12.1 "mkl<2024" "intel-openmp<2024" -c pytorch -c nvidia -y

# Use the cluster's GCC 11 instead of conda-forge to avoid dependency conflicts
module load GCC/11.3.0

# 2. Standard Dependencies
pip install "numpy<2.0.0"
pip install llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray opencv-python pyquaternion av2 kornia

# 3. Hardware-Specific CUDA Libraries
# The A100 also uses SpConv
pip install spconv-cu121

# Output the torch version so we can install torch-scatter next
python -c "import torch; print('TORCH_VERSION_FOR_SCATTER=' + torch.__version__.split('+')[0])"
