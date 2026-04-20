#!/bin/bash
#SBATCH --job-name=unitr_ft_v2
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/finetune_v2_%j.out
#SBATCH --error=slurm_logs/finetune_v2_%j.err

# Set environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate unitr

cd ~/UniTR/tools
mkdir -p slurm_logs

# We run single-GPU finetuning natively
srun --kill-on-bad-exit=1 \
    python -u train.py \
    --extra_tag v2.1_finetune_mini \
    --cfg_file cfgs/nuscenes_models/unitr_mini.yaml \
    --pretrained_model ../checkpoints/v2/latest_model.pth \
    --batch_size 3 \
    --epochs 20 \
    --workers 8
