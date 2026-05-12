#!/bin/bash
#SBATCH --job-name=eval_linear_probe
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm_logs/eval_linear_probe_%j.out
#SBATCH --error=slurm_logs/eval_linear_probe_%j.err
#SBATCH --time=1:00:00

# Set environment
module load Anaconda3
source /apps/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr
export OMP_NUM_THREADS=1
cd ~/UniTR/tools


PYTHON=/mnt/proj1/eu-25-122/andreaam/.conda/envs/unitr/bin/python
PORT=$((29500 + $RANDOM % 100))

# Launch Linear Probe Evaluation
# Note: Using test.py with --eval_map for BEV segmentation evaluation
$PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    test.py \
    --launcher pytorch \
    --cfg_file cfgs/nuscenes_models/unitr_strict_probe.yaml \
    --ckpt /home/it4i-andreaam/UniTR/output/nuscenes_models/unitr_strict_probe/pretrained_epoch_50/ckpt/checkpoint_epoch_13.pth \
    --extra_tag linear_probe_v5.3 \
    --eval_map \
    --workers 8
