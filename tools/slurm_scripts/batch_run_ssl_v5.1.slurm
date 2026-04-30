#!/bin/bash
#SBATCH --job-name=unitr_ssl_fusion
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/ssl_fusion_%j.out
#SBATCH --error=slurm_logs/ssl_fusion_%j.err
#SBATCH --time=6:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
module load squashfuse
export OMP_NUM_THREADS=1
cd ~/UniTR/tools

# 2. Ensure Mount is active
mkdir -p /home/it4i-andreaam/UniTR/data/nuscenes
# Clear any stale/broken mounts first
fusermount -uz /home/it4i-andreaam/UniTR/data/nuscenes 2>/dev/null || true
squashfuse /scratch/project/eu-25-122/nuscenes.sqsh /home/it4i-andreaam/UniTR/data/nuscenes

# Verify mount
if [ ! -d "/home/it4i-andreaam/UniTR/data/nuscenes/v1.0-trainval" ]; then
    echo "ERROR: nuScenes mount failed!"
    exit 1
fi

# 3. Launch Training
# Using explicit python path to bypass activation issues
PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

$PYTHON -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    train_ssl.py \
    --launcher pytorch \
    --extra_tag full_dataset_v5.1_mask \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_fusion_v5.1.yaml \
    --workers 8 \
    --sync_bn