#!/bin/bash
#SBATCH --job-name=linear_probe_pretrained
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm_logs/linear_probe_pretrained_%j.out
#SBATCH --error=slurm_logs/linear_probe_pretrained_%j.err
#SBATCH --time=12:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
export OMP_NUM_THREADS=1
cd ~/UniTR/tools


PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

# Launch Linear Probe with Pretrained Checkpoint
$PYTHON -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    train.py \
    --launcher pytorch \
    --cfg_file cfgs/nuscenes_models/unitr_strict_probe.yaml \
    --pretrained_model /home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion_v5.1/full_dataset_v5.3/ckpt/checkpoint_epoch_50.pth \
    --extra_tag pretrained_epoch_50 \
    --workers 8 \
    --sync_bn
