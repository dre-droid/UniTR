#!/bin/bash
#SBATCH --job-name=unitr_ft_v2
#SBATCH --partition=gpu-light
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/finetune_v2_%j.out
#SBATCH --error=slurm_logs/finetune_v2_%j.err

# Set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitr
cd ~/UniTR/tools

# We run single-GPU finetuning natively
srun --kill-on-bad-exit=1 \
    python -u train.py \
    --extra_tag v2_finetune_mini \
    --cfg_file cfgs/nuscenes_models/unitr_mini.yaml \
    --pretrained_model ../output/nuscenes_models/unitr_ibot/full_dataset_v2/ckpt/latest_model.pth \
    --batch_size 3 \
    --epochs 20 \
    --workers 8
