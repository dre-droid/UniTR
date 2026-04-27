#!/bin/bash
#SBATCH --job-name=ft_v3_e20
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/ft_v3_e20_%j.out
#SBATCH --error=slurm_logs/ft_v3_e20_%j.err

# 1. Environment Setup
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
module load squashfuse
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
cd ~/UniTR/tools

# 2. Ensure nuScenes Mount
mkdir -p /home/it4i-andreaam/UniTR/data/nuscenes
fusermount -uz /home/it4i-andreaam/UniTR/data/nuscenes 2>/dev/null || true
squashfuse /mnt/proj1/eu-25-122/andreaam/nuscenes.sqsh /home/it4i-andreaam/UniTR/data/nuscenes

# 3. Launch Fine-tuning
PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python

$PYTHON train.py \
    --cfg_file cfgs/nuscenes_models/unitr_mini.yaml \
    --pretrained_model /home/it4i-andreaam/UniTR/checkpoints/v3/checkpoint_epoch_20.pth \
    --extra_tag ft_v3_e20_mini \
    --batch_size 3 \
    --epochs 10 \
    --workers 8