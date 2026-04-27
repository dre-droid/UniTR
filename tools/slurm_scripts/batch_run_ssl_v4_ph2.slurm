#!/bin/bash
#SBATCH --job-name=unitr_ssl_fusion
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_logs/ssl_fusion_%j.out
#SBATCH --error=slurm_logs/ssl_fusion_%j.err
#SBATCH --time=18:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
module load squashfuse
cd ~/UniTR/tools

# 2. Ensure Mount is active
mkdir -p /home/it4i-andreaam/UniTR/data/nuscenes
# Clear any stale/broken mounts first
fusermount -uz /home/it4i-andreaam/UniTR/data/nuscenes 2>/dev/null || true
squashfuse /mnt/proj1/eu-25-122/andreaam/nuscenes.sqsh /home/it4i-andreaam/UniTR/data/nuscenes

# Verify mount
if [ ! -d "/home/it4i-andreaam/UniTR/data/nuscenes/v1.0-trainval" ]; then
    echo "ERROR: nuScenes mount failed!"
    exit 1
fi

# 3. Launch Training
# Using explicit python path to bypass activation issues
PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

$PYTHON -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    train_ssl.py \
    --launcher pytorch \
    --extra_tag v4_ph2 \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_fusion_ph2.yaml \
    --ckpt /home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion_ph2/v4_ph2/ckpt/latest_model.pth \
    --epochs 50 \
    --workers 4
