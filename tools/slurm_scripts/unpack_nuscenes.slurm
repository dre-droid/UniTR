#!/bin/bash
#SBATCH --job-name=unpack_nusc
#SBATCH --account=eu-25-122
#SBATCH --partition=qcpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_logs/unpack_nusc_%j.out
#SBATCH --time=12:00:00

echo "Starting unpack..."
time unsquashfs -f -d /scratch/project/eu-25-122/nuscenes_unpacked /scratch/project/eu-25-122/nuscenes.sqsh
echo "Done!"
