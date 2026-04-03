#!/bin/bash
#SBATCH --job-name=unitr_baseline
#SBATCH --partition=gpu-light
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --error=slurm_logs/evaluate_%j.err
#SBATCH --output=slurm_logs/evaluate_%j.out

# Load Anaconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitr
cd ~/UniTR/tools

# Train the downstream 3D Detection Head using the pretrained multi-modal backbone
srun --kill-on-bad-exit=1 \
    python -u train.py \
    --cfg_file cfgs/nuscenes_models/unitr.yaml \
    --batch_size 1 \
    --epochs 10 \
    --workers 16 \
    --extra_tag v1_mini_baseline \
    --set \
    DATA_CONFIG.VERSION v1.0-mini \
    DATA_CONFIG.INFO_PATH.train nuscenes_mini_infos_train.pkl \
    DATA_CONFIG.INFO_PATH.test nuscenes_mini_infos_val.pkl \
    DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST "['gt_sampling']"
