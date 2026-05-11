#!/bin/bash
#SBATCH --job-name=ssl_split_head
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/ssl_split_%j.out
#SBATCH --error=slurm_logs/ssl_split_%j.err
#SBATCH --time=12:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
export OMP_NUM_THREADS=1
cd ~/UniTR/tools

PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

# iBOT with Split Projection Heads (separate LiDAR/Camera heads)
$PYTHON -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    train_ssl.py \
    --launcher pytorch \
    --extra_tag split_head_v1 \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_split.yaml \
    --workers 8 \
    --sync_bn
