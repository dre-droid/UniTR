#!/bin/bash
#SBATCH --job-name=unitr_data_prep
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/data_prep_%j.out
#SBATCH --error=slurm_logs/data_prep_%j.err

# Set environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate unitr

# Set compilers and paths as we did for installation
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CPATH

cd /home/it4i-andreaam/UniTR

# Create dataset info file, lidar and image gt database
# We use v1.0-mini
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-mini \
    --with_cam \
    --with_cam_gt

# After generation, rename the files to match the unitr_mini.yaml config if necessary
# The script usually saves them in data/nuscenes/v1.0-mini/
cd data/nuscenes/v1.0-mini

if [ -f "nuscenes_infos_10sweeps_train.pkl" ]; then
    mv nuscenes_infos_10sweeps_train.pkl nuscenes_mini_infos_train.pkl
fi

if [ -f "nuscenes_infos_10sweeps_val.pkl" ]; then
    mv nuscenes_infos_10sweeps_val.pkl nuscenes_mini_infos_val.pkl
fi

echo "Data preparation complete."
