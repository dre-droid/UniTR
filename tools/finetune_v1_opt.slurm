#!/bin/bash
#SBATCH --job-name=unitr_ft_v1
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/finetune_v1_%j.out
#SBATCH --error=slurm_logs/finetune_v1_%j.err

# Set environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate unitr

cd ~/UniTR/tools
mkdir -p slurm_logs

# We run single-GPU finetuning natively
srun --kill-on-bad-exit=1 \
    python -u train.py \
    --extra_tag v1_opt \
    --cfg_file cfgs/nuscenes_models/unitr_mini.yaml \
    --pretrained_model ../checkpoints/v1/latest_model.pth \
    --batch_size 4 \
    --epochs 10 \
    --workers 8 \
    --use_amp
