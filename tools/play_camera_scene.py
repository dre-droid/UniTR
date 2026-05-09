
import argparse
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models.ssl.ibot_unitr import iBOTUniTR
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu

from visualize_ssl_features import extract_features_with_image_patches, pca_to_rgb, get_concerto_pca_color
from scipy.interpolate import NearestNDInterpolator

def main():
    parser = argparse.ArgumentParser(description='Play camera feeds for a nuScenes scene')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/unitr_ibot_fusion.yaml')
    parser.add_argument('--checkpoint', type=str, default='output/nuscenes_models/unitr_ibot_fusion/full_dataset_v4_fusion/ckpt/latest_model.pth')
    parser.add_argument('--sample_idx', type=int, default=5)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val'])
    parser.add_argument('--output_dir', type=str, default='output/visualizations/scene_play')
    parser.add_argument('--view_mode', type=str, default='camera', choices=['camera', 'bev', '3d'])
    parser.add_argument('--pca_start', type=int, default=0, help='Starting PCA component index')
    parser.add_argument('--stride', type=int, default=1, help='Skip frames in the sequence')
    args = parser.parse_args()

    # Use split-specific output folder
    args.output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = common_utils.create_logger()

    # 1. Config and Model
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # training=True loads the 'train' split, training=False loads the 'test' split (val)
    is_training = (args.split == 'train')
    
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=2,
        logger=logger,
        training=is_training,
    )
    # CRITICAL: Force training=False on the dataset object even if we loaded the 'train' split.
    # This disables random augmentations and random cropping (shakiness) for the video.
    dataset.training = False
    
    model = iBOTUniTR(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.cuda()
    model.eval()

    # 2. Find samples in the same scene as sample_idx
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg.DATA_CONFIG.VERSION, dataroot=str(dataset.root_path), verbose=False)
    
    target_info = dataset.infos[args.sample_idx]
    target_sample = nusc.get('sample', target_info['token'])
    scene_token = target_sample['scene_token']
    scene = nusc.get('scene', scene_token)
    
    # Get all tokens in scene
    current_token = scene['first_sample_token']
    scene_tokens = []
    while current_token != '':
        scene_tokens.append(current_token)
        current_token = nusc.get('sample', current_token)['next']
    
    logger.info(f"Scene {scene['name']} has {len(scene_tokens)} samples. Index 5 belongs here.")
    
    # Map tokens back to dataset indices
    token_to_idx = {info['token']: i for i, info in enumerate(dataset.infos)}
    scene_indices = [token_to_idx[t] for t in scene_tokens if t in token_to_idx]
    
    if not scene_indices:
        logger.error("No scene samples found in the current dataset split.")
        return

    # 3. Fit Global PCA basis
    # Extract features for first and last sample to anchor the PCA
    all_features = []
    with torch.no_grad():
        # Sample start, middle, and end frames for a robust basis
        indices = [scene_indices[0], scene_indices[len(scene_indices)//2], scene_indices[-1]]
        for i in indices:
            batch_dict = dataset[i]
            batch_dict = dataset.collate_batch([batch_dict])
            load_data_to_gpu(batch_dict)
            res = extract_features_with_image_patches(model, batch_dict, logger)
            all_features.append(res['pillar_features'])
    
    global_features = np.concatenate(all_features, axis=0)
    # Fit the global Concerto basis (top 12 components)
    _, global_v = get_concerto_pca_color(global_features, brightness=1.1)
    logger.info("Global Concerto basis fitted.")

    # 4. Generate frames
    frames = []
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cam_names = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']

    for i in tqdm(scene_indices, desc="Generating Frames"):
        batch_dict = dataset[i]
        batch_dict = dataset.collate_batch([batch_dict])
        load_data_to_gpu(batch_dict)
        
        with torch.no_grad():
            res = extract_features_with_image_patches(model, batch_dict, logger)
        
        if args.view_mode == 'camera':
            patch_feats = res['patch_features'] # (6, 128, 32, 88)
            imgs = res['camera_imgs'][0]        # (6, 3, 256, 704)
            _, _, pH, pW = patch_feats.shape
            
            # Grid plot 3x2
            fig, axes = plt.subplots(3, 2, figsize=(15, 10), dpi=100)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)

            for cam_i in range(6):
                # PCA for this camera
                cam_feats = patch_feats[cam_i].transpose(1, 2, 0).reshape(-1, 128)
                rgb_map, _ = pca_to_rgb(cam_feats, pca_obj=global_v, start_component=args.pca_start)
                rgb_map = rgb_map.reshape(pH, pW, 3)
                
                # Upsample
                rgb_upsampled = np.array(Image.fromarray((rgb_map * 255).astype(np.uint8)).resize((704, 256), Image.BILINEAR)) / 255.0
                
                # Original Image
                img = imgs[cam_i] * std[:, None, None] + mean[:, None, None]
                img = np.clip(img, 0, 1).transpose(1, 2, 0)
                
                # No Alpha Blend
                blended = rgb_upsampled
                blended = np.clip(blended, 0, 1)
                
                ax = axes[cam_i // 2, cam_i % 2]
                ax.imshow(blended)
                ax.axis('off')
                ax.text(5, 20, cam_names[cam_i], color='white', fontsize=12, backgroundcolor='black')

            plt.suptitle(f"Scene: {scene['name']} | Sample: {i}", fontsize=16, color='white', y=0.98)
            fig.patch.set_facecolor('#1a1a2e')
        elif args.view_mode == 'bev':
            pillar_features = res['pillar_features']
            pillar_coords = res['pillar_coords']
            
            rgb_b0, _ = pca_to_rgb(pillar_features, pca_obj=global_v, start_component=args.pca_start)
            
            grid_h, grid_w = 360, 360
            bev_image = np.ones((grid_h, grid_w, 3), dtype=np.float32) * 0.15
            
            mask_b0 = pillar_coords[:, 0] == 0
            coords_b0 = pillar_coords[mask_b0]
            rgb_b0 = rgb_b0[mask_b0]
            
            y_idx = coords_b0[:, 2].astype(int)
            x_idx = coords_b0[:, 3].astype(int)
            
            valid = (y_idx >= 0) & (y_idx < grid_h) & (x_idx >= 0) & (x_idx < grid_w)
            y_idx, x_idx, rgb_b0 = y_idx[valid], x_idx[valid], rgb_b0[valid]
            
            bev_image[y_idx, x_idx] = rgb_b0
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
            ax.imshow(bev_image, origin='lower')
            ax.set_title(f"Scene: {scene['name']} | Sample: {i} | BEV", fontsize=16, color='white', pad=20)
            ax.axis('off')
            fig.patch.set_facecolor('#1a1a2e')
        elif args.view_mode == '3d':
            pillar_features = res['pillar_features']
            pillar_coords = res['pillar_coords']
            points = res['points']
            pc_range = res['pc_range']
            voxel_size = res['voxel_size']
            
            # 1. Concerto PCA for pillars using GLOBAL basis
            mask_pillars = (pillar_coords[:, 0] == 0)
            feats_b0 = pillar_features[mask_pillars]
            coords_b0 = pillar_coords[mask_pillars]
            rgb_pillars, _ = get_concerto_pca_color(feats_b0, brightness=1.1, v_matrix=global_v)
            
            # 2. Extract and Crop points
            mask_pts = (points[:, 0] == 0)
            pts_b0 = points[mask_pts]
            crop_mask = (pts_b0[:, 1] > -35) & (pts_b0[:, 1] < 35) & \
                        (pts_b0[:, 2] > -25) & (pts_b0[:, 2] < 65) & \
                        (pts_b0[:, 3] > -5) & (pts_b0[:, 3] < 5)
            pts_b0 = pts_b0[crop_mask]
            if len(pts_b0) == 0:
                pts_b0 = points[mask_pts]
            
            raw_x, raw_y, raw_z = pts_b0[:, 1], pts_b0[:, 2], pts_b0[:, 3]
            
            # 3. Dense Interpolation
            from scipy.interpolate import NearestNDInterpolator
            train_x = (coords_b0[:, 3] * voxel_size[0]) + pc_range[0]
            train_y = (coords_b0[:, 2] * voxel_size[1]) + pc_range[1]
            interp = NearestNDInterpolator(np.stack([train_x, train_y], axis=1), rgb_pillars)
            final_colors = interp(np.stack([raw_x, raw_y], axis=1))
            
            # 4. Plot (Sharper 3D)
            fig = plt.figure(figsize=(16, 10), dpi=150) # Higher DPI
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('#050508')
            fig.patch.set_facecolor('#050508')
            
            ax.scatter(raw_x, raw_y, raw_z, c=final_colors, s=2.5, alpha=0.9, edgecolors='none')
            
            ax.set_box_aspect((70, 90, 25)) 
            ax.view_init(elev=32, azim=-90)
            
            ax.set_xlim(-35, 35)
            ax.set_ylim(-25, 65)
            ax.set_zlim(-4, 6)
            
            ax.axis('off')
            ax.set_title(f"Scene: {scene['name']} | Sample: {i}", 
                         fontsize=16, color='white', pad=-20)
        
        # Save to buffer
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert('RGB'))
        plt.close(fig)

    # 5. Save GIF
    gif_path = os.path.join(args.output_dir, f"scene_{scene['name']}_{args.view_mode}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    logger.info(f"Animation saved to {gif_path}")

if __name__ == '__main__':
    main()
