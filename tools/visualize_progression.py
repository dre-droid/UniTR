"""
SSL Feature Progression Visualizer for UniTR.

Scans a directory for epoch checkpoints, fits a consistent (global) PCA
basis from the final epoch, and generates periodic snapshots to assess
learned feature quality over time.

Usage:
    python visualize_progression.py \
        --cfg_file cfgs/nuscenes_models/unitr_ibot.yaml \
        --ckpt_dir ../output/nuscenes_models/unitr_ibot/full_dataset_v1/ckpt \
        --sample_idx 0 \
        --output_dir ../output/visualizations/progression
"""

import argparse
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from visualize_ssl_features import (
    load_model_and_data, 
    extract_features_with_image_patches,
    visualize_bev_pca,
    visualize_camera_pca,
    common_utils
)
from pcdet.models import load_data_to_gpu

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SSL feature progression')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/unitr_ibot.yaml')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing epoch checkpoints')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--query_xy', type=float, nargs=2, default=[0.0, 0.0],
                        help='Query point [x, y] in meters for bev_similarity mode')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='List of specific epochs to visualize (e.g. 5 10 20)')
    parser.add_argument('--output_dir', type=str, default='../output/visualizations/progression')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def get_checkpoints(ckpt_dir):
    """Find all checkpoint_epoch_N.pth files and sort them."""
    files = os.listdir(ckpt_dir)
    pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    
    ckpts = []
    for f in files:
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            ckpts.append((epoch, os.path.join(ckpt_dir, f)))
    
    # Sort by epoch
    ckpts.sort(key=lambda x: x[0])
    return ckpts

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = common_utils.create_logger()
    
    # 1. Get all checkpoints
    ckpts = get_checkpoints(args.ckpt_dir)
    if not ckpts:
        logger.error(f"No epoch checkpoints found in {args.ckpt_dir}")
        return
    
    # Filter by user-requested epochs if specified
    if args.epochs is not None:
        ckpts = [c for c in ckpts if c[0] in args.epochs]
        if not ckpts:
            logger.error(f"None of the requested epochs {args.epochs} were found.")
            return

    logger.info(f"Processing {len(ckpts)} checkpoints: {[c[0] for c in ckpts]}")
    
    # 2. Load model and data (initially with no checkpoint to get the sample)
    # We use a dummy checkpoint for first load
    model, dataset, dataloader, logger = load_model_and_data(argparse.Namespace(
        cfg_file=args.cfg_file,
        checkpoint=ckpts[-1][1], # Load last checkpoint first for PCA fitting
        set_cfgs=args.set_cfgs
    ))
    
    # Get the sample
    batch_dict = None
    for i, data in enumerate(dataloader):
        if i == args.sample_idx:
            batch_dict = data
            break
    
    load_data_to_gpu(batch_dict)
    
    # 3. Fit Global PCA using the LAST epoch (highest variance/separation)
    logger.info(f"--- Fitting Global PCA using Epoch {ckpts[-1][0]} ---")
    result_final = extract_features_with_image_patches(model, batch_dict, logger)
    
    # Fit LiDAR PCA
    from visualize_ssl_features import pca_to_rgb
    _, pca_bev = pca_to_rgb(result_final['pillar_features'])
    
    # Fit Camera PCA
    # (6*32*88, 128)
    D = result_final['patch_features'].shape[1]
    all_patches = result_final['patch_features'][:6].transpose(0, 2, 3, 1).reshape(-1, D)
    _, pca_cam = pca_to_rgb(all_patches)
    
    logger.info("Global PCA basis established.")

    # 4. Loop through ALL checkpoints and generate images
    for epoch, ckpt_path in tqdm(ckpts, desc="Processing Epochs"):
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Load weights for this epoch
        ckpt = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(ckpt['model_state'], strict=False)
        model.eval()
        
        # Extract features
        result = extract_features_with_image_patches(model, batch_dict, logger)
        
        # Visualize with consistent PCA
        # Overwrite the output path in result/args if needed, or just pass epoch_dir
        # We'll modify the visualizing functions to accept a custom filename or use the output_dir
        
        # BEV PCA
        visualize_bev_pca(result, epoch_dir, logger, pca_obj=pca_bev)
        
        # Camera PCA
        visualize_camera_pca(result, epoch_dir, logger, pca_obj=pca_cam)
        
    logger.info(f"Done! All progression images saved to {args.output_dir}")

if __name__ == '__main__':
    main()
