#!/bin/bash
#SBATCH --job-name=play_scene
#SBATCH --partition=gpu-heavy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/play_scene_%j.out
#SBATCH --error=slurm_logs/play_scene_%j.err

# Set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitr
cd ~/UniTR/tools

srun python play_camera_scene.py \
    --sample_idx 5 \
    --split train \
    --view_mode bev \
    --checkpoint /home/andrea_mastroberti/UniTR/output/nuscenes_models/unitr_ibot_fusion/full_dataset_v3_fusion/ckpt/latest_model.pth \
    --output_dir /home/andrea_mastroberti/UniTR/output/visualizations/scene_play/v3_epoch10