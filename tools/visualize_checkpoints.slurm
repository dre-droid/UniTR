#!/bin/bash
#SBATCH --job-name=unitr_vis
#SBATCH --account=eu-25-122
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/vis_%j.out
#SBATCH --error=slurm_logs/vis_%j.err

# Set environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate unitr

cd ~/UniTR/tools
mkdir -p ../output/visualizations/v1
mkdir -p ../output/visualizations/v2

echo "Starting Visualization for V1 checkpoint..."
python visualize_ssl_features.py \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_mini.yaml \
    --checkpoint ../checkpoints/v1/latest_model.pth \
    --mode all \
    --output_dir ../output/visualizations/v1 \
    --no_overlay

echo "Starting Visualization for V2 checkpoint..."
python visualize_ssl_features.py \
    --cfg_file cfgs/nuscenes_models/unitr_ibot_mini.yaml \
    --checkpoint ../checkpoints/v2/latest_model.pth \
    --mode all \
    --output_dir ../output/visualizations/v2 \
    --no_overlay

echo "All visualizations complete. Results are in output/visualizations/"
