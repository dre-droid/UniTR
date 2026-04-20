#!/bin/bash
#SBATCH --job-name=unitr_ssl_fusion
#SBATCH --partition=gpu-heavy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/ssl_fusion_%j.out
#SBATCH --error=slurm_logs/ssl_fusion_%j.err

# --- This v3 is run with fusion modules enabled ---

# Set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitr
cd ~/UniTR/tools

# We run single-GPU SSL training natively
srun --kill-on-bad-exit=1 \
    python -u train_ssl.py \
    --extra_tag full_dataset_v3_fusion \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_fusion.yaml \
    --batch_size 8 \
    --epochs 50 \
    --workers 8
