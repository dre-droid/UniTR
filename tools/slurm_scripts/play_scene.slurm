#!/bin/bash
#SBATCH --job-name=play_scene
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/play_scene_%j.out
#SBATCH --error=slurm_logs/play_scene_%j.err

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
module load squashfuse
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
cd ~/UniTR/tools

# 2. Ensure Mount is active (FUSE mounts are local to the node!)
mkdir -p /home/it4i-andreaam/UniTR/data/nuscenes
fusermount -uz /home/it4i-andreaam/UniTR/data/nuscenes 2>/dev/null || true
squashfuse /mnt/proj1/eu-25-122/andreaam/nuscenes.sqsh /home/it4i-andreaam/UniTR/data/nuscenes

# 3. Launch Visualization
PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
OUT_DIR="/home/it4i-andreaam/UniTR/output/visualizations/scene_play/v4_epoch10_20pct"

# Run BEV
$PYTHON play_camera_scene.py \
    --sample_idx 5 \
    --split train \
    --view_mode bev \
    --checkpoint /home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion/full_dataset_v4_fusion/ckpt/latest_model.pth \
    --output_dir $OUT_DIR

# Run Camera
$PYTHON play_camera_scene.py \
    --sample_idx 5 \
    --split train \
    --view_mode camera \
    --checkpoint /home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion/full_dataset_v4_fusion/ckpt/latest_model.pth \
    --output_dir $OUT_DIR