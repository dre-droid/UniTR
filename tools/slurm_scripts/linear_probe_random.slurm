#!/bin/bash
#SBATCH --job-name=linear_probe_random
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm_logs/linear_probe_random_%j.out
#SBATCH --error=slurm_logs/linear_probe_random_%j.err
#SBATCH --time=12:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
export OMP_NUM_THREADS=1
cd ~/UniTR/tools


PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

# Launch Linear Probe with Random Init
$PYTHON -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    train.py \
    --launcher pytorch \
    --cfg_file cfgs/nuscenes_models/unitr_strict_probe.yaml \
    --extra_tag random_init \
    --workers 8 \
    --sync_bn
