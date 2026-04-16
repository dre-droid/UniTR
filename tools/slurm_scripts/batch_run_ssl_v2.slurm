#!/bin/bash
#SBATCH --job-name=unitr_ssl
#SBATCH --partition=gpu-heavy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/ssl_pretrain_%j.out
#SBATCH --error=slurm_logs/ssl_pretrain_%j.err

# --- This v2 is run with image masking patch loss ---

# Set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitr
cd ~/UniTR/tools

# # Automatically find a free port for distributed training
# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done

# We run single-GPU SSL training natively
srun --kill-on-bad-exit=1 \
    python -u train_ssl.py \
    --extra_tag full_dataset_v2 \
    --cfg_file cfgs/nuscenes_models/unitr_ibot.yaml \
    --batch_size 4 \
    --epochs 25 \
    --workers 8