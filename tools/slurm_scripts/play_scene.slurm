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

# 3. Launch Visualization
PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
OUT_DIR="/home/it4i-andreaam/UniTR/output/visualizations/scene_play/v5.2_epoch33_16pct"
CKPT="/home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_ibot_fusion_v5.1/full_dataset_v5.3/ckpt/latest_model.pth"

# Run BEV
$PYTHON play_camera_scene.py \
    --sample_idx 5 \
    --split train \
    --view_mode bev \
    --pca_start 0 \
    --checkpoint $CKPT \
    --output_dir $OUT_DIR

# Run Camera
$PYTHON play_camera_scene.py \
    --sample_idx 5 \
    --split train \
    --view_mode camera \
    --checkpoint $CKPT \
    --output_dir $OUT_DIR

# # Run 3D PCA Sequence (GIF) - Sharp & Stable
# $PYTHON play_camera_scene.py \
#     --cfg_file cfgs/nuscenes_models/unitr_ibot_fusion.yaml \
#     --sample_idx 5 \
#     --split train \
#     --view_mode 3d \
#     --stride 2 \
#     --pca_start 0 \
#     --checkpoint $CKPT \
#     --output_dir $OUT_DIR