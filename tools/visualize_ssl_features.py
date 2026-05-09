"""
Visualize SSL-learned features from the UniTR backbone.

Extracts 128-D per-element features from the teacher backbone and projects
them to RGB via PCA for visual inspection.

Modes:
  bev_pca       – BEV birds-eye-view colored by PCA of pillar features
  camera_pca    – Camera images overlaid with PCA of image patch features
  bev_similarity – BEV cosine similarity map from a query point

Usage:
  srun -p gpu-light --gres=gpu:1 python visualize_ssl_features.py \
      --checkpoint ../output/nuscenes_models/unitr_ibot/full_dataset_v1/ckpt/latest_model.pth \
      --mode bev_pca \
      --sample_idx 0 \
      --output_dir ../output/visualizations
"""

import argparse
import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models.ssl.ibot_unitr import iBOTUniTR
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SSL features from UniTR backbone')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/unitr_ibot.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SSL checkpoint (latest_model.pth or pretrained_unitr.pth)')
    parser.add_argument('--mode', type=str, default='bev_pca',
                        choices=['bev_pca', 'camera_pca', 'bev_similarity', 'lidar_3d_pca', 'all'],
                        help='Visualization mode')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample in the dataset to visualize')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--query_xy', type=float, nargs=2, default=[0.0, 0.0],
                        help='Query point [x, y] in meters for bev_similarity mode')
    parser.add_argument('--output_dir', type=str, default='../output/visualizations')
    parser.add_argument('--no_overlay', action='store_true', help='Do not overlay features on original images')
    parser.add_argument('--pca_start', type=int, default=0,
                        help='Starting PCA component index (e.g. 1 to skip first gradient)')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_model_and_data(args):
    """Load the iBOTUniTR model and a single data sample."""
    # Config
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        from pcdet.config import cfg_from_list
        cfg_from_list(args.set_cfgs, cfg)

    logger = common_utils.create_logger()
    logger.info('Loading dataset...')

    # Build dataloader
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=2,
        logger=logger,
        training=(args.split == 'train'),
    )
    logger.info(f'Dataset size: {len(dataset)}')

    # Build model
    logger.info('Building iBOTUniTR model...')
    model = iBOTUniTR(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_state = ckpt['model_state']

    # Flexible loading (handles both raw SSL ckpt and converted ckpt)
    current_state = model.state_dict()
    matched = {k: v for k, v in model_state.items()
               if k in current_state and current_state[k].shape == v.shape}
    current_state.update(matched)
    model.load_state_dict(current_state)
    logger.info(f'Loaded {len(matched)}/{len(current_state)} parameters')

    model.cuda().eval()
    return model, dataset, dataloader, logger

@torch.no_grad()
def extract_features_with_image_patches(model, batch_dict, logger):
    """
    Run the teacher backbone and capture both LiDAR and image patch features
    by enabling out_indices in the backbone.
    """
    # Temporarily modify out_indices to capture image features at the last block
    backbone = model.teacher_backbone
    old_indices = backbone.out_indices
    # NuScenes UniTR usually has 4 blocks (idx 0 to 3)
    backbone.out_indices = [3]  
    
    # Ensure out_norm3 exists (it might be missing if trained with out_indices=[])
    if not hasattr(backbone, 'out_norm3'):
        logger.info("Adding dummy out_norm3 to backbone for visualization")
        import torch.nn as nn
        backbone.add_module('out_norm3', nn.LayerNorm(backbone.d_model[-1]).cuda())
    
    # Step 1: VFE
    batch_dict = model.teacher_vfe(batch_dict)
    voxel_features = batch_dict['voxel_features']
    voxel_coords = batch_dict['voxel_coords']
    voxel_num = voxel_features.shape[0]

    # Step 2: Teacher backbone (unmasked, clean features)
    teacher_batch = copy.copy(batch_dict)
    teacher_batch.update({
        'voxel_features': voxel_features,
        'voxel_coords': voxel_coords,
        'voxel_num': voxel_num,
    })

    # Run full backbone forward
    teacher_out = backbone(teacher_batch)
    
    # Restore old indices
    backbone.out_indices = old_indices

    # Extract LiDAR pillar features
    pillar_features = teacher_out['pillar_features']  # (N_voxels, 128)
    pillar_coords = teacher_out['voxel_coords']       # (N_voxels, 4)

    # Extract image patch features (after transformer processing)
    # image_features is a list containing output from requested indices
    if len(teacher_out.get('image_features', [])) > 0:
        patch_feats = teacher_out['image_features'][0]  # (B*6, 128, 32, 88)
    else:
        raise RuntimeError(
            "Failed to extract 'image_features' from the backbone. "
            "Ensure that 'out_indices' is set correctly (e.g., [3]) and that the "
            "backbone is correctly populating the 'image_features' list in its forward pass."
        )

    result = {
        'pillar_features': pillar_features.cpu().numpy(),
        'pillar_coords': pillar_coords.cpu().numpy(),
        'patch_features': patch_feats.cpu().numpy(),   # (B*6, 128, 32, 88)
        'camera_imgs': batch_dict['camera_imgs'].cpu().numpy(),
        'points': batch_dict['points'].cpu().numpy(),
        'voxel_num': voxel_num,
        'batch_size': batch_dict['batch_size'],
        'pc_range': model.dataset.point_cloud_range,
        'voxel_size': model.dataset.voxel_size,
    }

    return result


def get_concerto_pca_color(feat, brightness=1.0, center=True, v_matrix=None):
    """
    Concerto-style PCA coloring: Weighted blend of top 12 components.
    Supports a fixed v_matrix for temporal stability in videos.
    """
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat).float().cuda()
    
    if v_matrix is None:
        # Calculate basis if not provided
        u, s, v_matrix = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    
    # Project using either the new or provided basis
    projection = feat @ v_matrix
    
    # Weighted blend of components (0.2, 0.2, 0.1, 0.5)
    rgb = projection[:, :3] * 0.2 + \
          projection[:, 3:6] * 0.2 + \
          projection[:, 6:9] * 0.1 + \
          projection[:, 9:12] * 0.5
    
    # Robust normalization [0, 1]
    rgb_np = rgb.cpu().numpy()
    vmin = np.percentile(rgb_np, 2, axis=0)
    vmax = np.percentile(rgb_np, 98, axis=0)
    rgb_np = np.clip((rgb_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    # Multiply by brightness
    rgb_np = np.clip(rgb_np * brightness, 0, 1)
    
    return rgb_np, v_matrix


def pca_to_rgb(features, pca_obj=None, start_component=0):
    """
    Standard wrapper for BEV/Camera modes using Concerto blend.
    """
    rgb_np, _ = get_concerto_pca_color(features, v_matrix=pca_obj)
    return rgb_np, pca_obj


def visualize_bev_pca(result, output_dir, logger, pca_obj=None, pca_start=0):
    """
    Mode 1: BEV feature map colored by PCA of pillar features.
    """
    logger.info('--- Mode: BEV PCA Feature Map ---')
    
    pillar_features = result['pillar_features']  # (N, 128)
    pillar_coords = result['pillar_coords']      # (N, 4) [batch, z, y, x]
    
    logger.info(f'Pillar features shape: {pillar_features.shape}')
    
    # PCA → RGB
    rgb, pca_obj = pca_to_rgb(pillar_features, pca_obj=pca_obj, start_component=pca_start)
    
    # Build BEV grid image
    # Grid is 360x360 (from sparse_shape config), coords are [batch, z, y, x]
    grid_h, grid_w = 360, 360
    bev_image = np.ones((grid_h, grid_w, 3), dtype=np.float32) * 0.15  # dark background
    
    # Filter to batch 0
    mask_b0 = pillar_coords[:, 0] == 0
    coords_b0 = pillar_coords[mask_b0]
    rgb_b0 = rgb[mask_b0]
    
    y_idx = coords_b0[:, 2].astype(int)  # y grid index
    x_idx = coords_b0[:, 3].astype(int)  # x grid index
    
    # Clip to grid bounds
    valid = (y_idx >= 0) & (y_idx < grid_h) & (x_idx >= 0) & (x_idx < grid_w)
    y_idx, x_idx, rgb_b0 = y_idx[valid], x_idx[valid], rgb_b0[valid]
    
    bev_image[y_idx, x_idx] = rgb_b0
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
    ax.imshow(bev_image, origin='lower')
    ax.set_title('BEV PCA Feature Map (SSL-Pretrained UniTR)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X grid (0.3m per cell)')
    ax.set_ylabel('Y grid (0.3m per cell)')
    
    # Add range annotations
    ax.text(0.02, 0.98, f'Range: -54m to +54m\n'
            f'Voxel size: 0.3m\n'
            f'Pillars: {mask_b0.sum()}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white')
    
    out_path = os.path.join(output_dir, 'bev_pca_features.png')
    fig.savefig(out_path, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    logger.info(f'Saved BEV PCA visualization to: {out_path}')
    return out_path, pca_obj


def visualize_camera_pca(result, output_dir, logger, pca_obj=None, no_overlay=False, pca_start=0):
    """
    Mode 2: Camera images overlaid with PCA of image patch features.
    """
    logger.info('--- Mode: Camera PCA Feature Map ---')
    
    patch_features = result['patch_features']  # (B*6, 128, 32, 88)
    camera_imgs = result['camera_imgs']        # (B, 6, 3, 256, 704)
    
    B_N, D, pH, pW = patch_features.shape
    N_cam = 6
    
    # Use batch 0 only
    patch_feats_b0 = patch_features[:N_cam]  # (6, 128, 32, 88)
    imgs_b0 = camera_imgs[0]                 # (6, 3, 256, 704)
    
    # Re-order features for global PCA fit across all cameras
    # (6, 128, 32, 88) -> (6*32*88, 128)
    all_patches = patch_feats_b0.transpose(0, 2, 3, 1).reshape(-1, D)
    rgb_all, pca_obj = pca_to_rgb(all_patches, pca_obj=pca_obj, start_component=pca_start)
    
    rgb_maps = rgb_all.reshape(N_cam, pH, pW, 3)
    
    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), dpi=150)
    fig.suptitle('Camera PCA Feature Map (SSL-Pretrained UniTR)', fontsize=16, fontweight='bold')
    
    for cam_i in range(N_cam):
        ax = axes[cam_i // 2, cam_i % 2]
        
        # Original image
        img = imgs_b0[cam_i]  # (3, 256, 704)
        img = img * std + mean
        img = np.clip(img, 0, 1).transpose(1, 2, 0)  # (256, 704, 3)
        
        rgb_map = rgb_maps[cam_i]  # (32, 88, 3)
        
        # Upsample to image resolution
        from PIL import Image
        rgb_upsampled = np.array(Image.fromarray(
            (rgb_map * 255).astype(np.uint8)
        ).resize((704, 256), Image.BILINEAR)) / 255.0
        
        # Alpha blend
        if no_overlay:
            blended = rgb_upsampled
        else:
            alpha = 0.5
            blended = img * (1 - alpha) + rgb_upsampled * alpha
        
        blended = np.clip(blended, 0, 1)
        
        ax.imshow(blended)
        ax.set_title(cam_names[cam_i], fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'camera_pca_features.png')
    fig.savefig(out_path, bbox_inches='tight', facecolor='#1a1a2e')
    # Also return the PCA object used
    return out_path, pca_obj


def visualize_bev_similarity(result, query_xy, output_dir, logger):
    """
    Mode 3: BEV cosine similarity map from a query point.
    """
    logger.info(f'--- Mode: BEV Cosine Similarity (query={query_xy}) ---')
    
    pillar_features = result['pillar_features']  # (N, 128)
    pillar_coords = result['pillar_coords']      # (N, 4) [batch, z, y, x]
    
    # Filter to batch 0
    mask_b0 = pillar_coords[:, 0] == 0
    feats_b0 = pillar_features[mask_b0]
    coords_b0 = pillar_coords[mask_b0]
    
    # Convert query xy (meters) to grid indices
    # Grid: 360x360, range: [-54, 54], voxel_size: 0.3
    voxel_size = 0.3
    pc_range_min = np.array([-54.0, -54.0])
    
    query_grid = ((np.array(query_xy) - pc_range_min) / voxel_size).astype(int)
    logger.info(f'Query grid index: [{query_grid[0]}, {query_grid[1]}]')
    
    # Find the nearest occupied pillar to the query
    grid_xy = coords_b0[:, [3, 2]]  # [x, y] grid indices
    distances = np.linalg.norm(grid_xy - query_grid[np.newaxis, :], axis=1)
    nearest_idx = np.argmin(distances)
    query_feat = feats_b0[nearest_idx]  # (128,)
    
    logger.info(f'Nearest pillar at grid [{grid_xy[nearest_idx, 0]}, {grid_xy[nearest_idx, 1]}], '
                f'distance={distances[nearest_idx]:.1f} cells')
    
    # Compute cosine similarity of all pillars against query
    norms = np.linalg.norm(feats_b0, axis=1, keepdims=True) + 1e-8
    query_norm = np.linalg.norm(query_feat) + 1e-8
    cosine_sim = (feats_b0 @ query_feat) / (norms.squeeze() * query_norm)
    
    # Build BEV grid
    grid_h, grid_w = 360, 360
    bev_sim = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    
    y_idx = coords_b0[:, 2].astype(int)
    x_idx = coords_b0[:, 3].astype(int)
    valid = (y_idx >= 0) & (y_idx < grid_h) & (x_idx >= 0) & (x_idx < grid_w)
    
    bev_sim[y_idx[valid], x_idx[valid]] = cosine_sim[valid]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)
    
    # Background
    bev_bg = np.full((grid_h, grid_w, 3), 0.1)
    ax.imshow(bev_bg, origin='lower')
    
    # Overlay cosine similarity with a colormap
    cmap = cm.get_cmap('coolwarm')
    sim_rgba = cmap((bev_sim + 1) / 2)  # map [-1, 1] → [0, 1]
    sim_rgba[np.isnan(bev_sim)] = [0.1, 0.1, 0.1, 1.0]  # dark for empty
    
    ax.imshow(sim_rgba, origin='lower')
    
    # Mark query point
    q_y = int((query_xy[1] - (-54.0)) / voxel_size)
    q_x = int((query_xy[0] - (-54.0)) / voxel_size)
    ax.plot(q_x, q_y, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_title(f'BEV Cosine Similarity Map (query=({query_xy[0]:.1f}, {query_xy[1]:.1f})m)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X grid')
    ax.set_ylabel('Y grid')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    out_path = os.path.join(output_dir, 'bev_similarity_features.png')
    fig.savefig(out_path, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    logger.info(f'Saved BEV similarity visualization to: {out_path}')
    return out_path


def visualize_lidar_3d_pca(result, output_dir, logger, pca_obj=None, pca_start=0):
    """
    Mode 4: 3D point cloud colored by Concerto-style PCA features.
    Improved Density: Uses interpolation to color ALL points in the scene.
    """
    logger.info('--- Mode: 3D LiDAR PCA (Dense Concerto-Style) ---')
    
    points = result['points']
    pillar_features = result['pillar_features']
    pillar_coords = result['pillar_coords']
    pc_range = result['pc_range']
    voxel_size = result['voxel_size']
    
    # 1. Extract Batch 0 points
    mask_pts = (points[:, 0] == 0)
    pts_b0 = points[mask_pts]
    
    # Preferred Crop: 40m wide, 90m deep
    crop_mask = (pts_b0[:, 1] > -35) & (pts_b0[:, 1] < 35) & \
                (pts_b0[:, 2] > -25) & (pts_b0[:, 2] < 65) & \
                (pts_b0[:, 3] > -5) & (pts_b0[:, 3] < 5)
    
    pts_b0 = pts_b0[crop_mask]
    if len(pts_b0) == 0:
        pts_b0 = points[mask_pts]

    raw_x, raw_y, raw_z = pts_b0[:, 1], pts_b0[:, 2], pts_b0[:, 3]
    raw_i = pts_b0[:, 4] if pts_b0.shape[1] > 4 else np.ones_like(raw_x)

    # 2. Get Concerto PCA colors
    mask_pillars = (pillar_coords[:, 0] == 0)
    feats_b0 = pillar_features[mask_pillars]
    coords_b0 = pillar_coords[mask_pillars]
    rgb_pillars = get_concerto_pca_color(feats_b0, brightness=1.1)
    
    # 3. Dense Mapping (Nearest Neighbor Interpolation)
    from scipy.interpolate import NearestNDInterpolator
    train_x = (coords_b0[:, 3] * voxel_size[0]) + pc_range[0]
    train_y = (coords_b0[:, 2] * voxel_size[1]) + pc_range[1]
    interp = NearestNDInterpolator(np.stack([train_x, train_y], axis=1), rgb_pillars)
    final_colors = interp(np.stack([raw_x, raw_y], axis=1))
    
    # Fallback for far points
    nan_mask = np.isnan(final_colors).any(axis=1)
    if nan_mask.any():
        gray = np.clip(raw_i[nan_mask] / 255.0, 0.1, 0.3)
        final_colors[nan_mask] = np.stack([gray, gray, gray], axis=1)

    # 4. Plotting
    fig = plt.figure(figsize=(18, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050508')
    fig.patch.set_facecolor('#050508')
    
    # Plot points
    ax.scatter(raw_x, raw_y, raw_z, c=final_colors, s=2.2, alpha=0.95, edgecolors='none')
    
    # Correct aspect ratio
    ax.set_box_aspect((70, 90, 25)) 
    
    # View: Behind and tilted down
    ax.view_init(elev=32, azim=-90)
    
    # View bounds (Zoomed)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-25, 65)
    ax.set_zlim(-4, 6)
    
    ax.axis('off')
    ax.set_title(f'3D SSL Features - Dense Concerto View (Sample {result.get("sample_idx", "?")})', 
                 color='white', fontsize=18, pad=-30)
    
    out_path = os.path.join(output_dir, 'lidar_3d_pca_concerto_dense.png')
    fig.savefig(out_path, bbox_inches='tight', facecolor='#050508', pad_inches=0)
    plt.close(fig)
    logger.info(f'Saved dense Concerto-style 3D plot to: {out_path}')
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, dataset, dataloader, logger = load_model_and_data(args)
    
    # Get a single sample
    logger.info(f'Extracting features for sample index {args.sample_idx}...')
    
    # Iterate to the requested sample
    batch_dict = None
    for i, data in enumerate(dataloader):
        if i == args.sample_idx:
            batch_dict = data
            break
    
    if batch_dict is None:
        logger.error(f'Could not find sample index {args.sample_idx}. Dataloader length is {len(dataloader)}. Check your dataset config and paths.')
        return

    # Move to GPU
    from pcdet.models import load_data_to_gpu
    load_data_to_gpu(batch_dict)
    
    # Run modes
    modes = ['bev_pca', 'camera_pca', 'bev_similarity', 'lidar_3d_pca'] if args.mode == 'all' else [args.mode]
    
    for mode in modes:
        if mode == 'camera_pca':
            result = extract_features_with_image_patches(model, batch_dict, logger)
            visualize_camera_pca(result, args.output_dir, logger, no_overlay=args.no_overlay, pca_start=args.pca_start)
        elif mode == 'bev_pca':
            result = extract_features_with_image_patches(model, batch_dict, logger) # capture joint
            visualize_bev_pca(result, args.output_dir, logger, pca_start=args.pca_start)
        elif mode == 'bev_similarity':
            result = extract_features_with_image_patches(model, batch_dict, logger)
            visualize_bev_similarity(result, args.query_xy, args.output_dir, logger)
        elif mode == 'lidar_3d_pca':
            result = extract_features_with_image_patches(model, batch_dict, logger)
            visualize_lidar_3d_pca(result, args.output_dir, logger, pca_start=args.pca_start)
    
    logger.info('Done! All visualizations saved.')


if __name__ == '__main__':
    main()
