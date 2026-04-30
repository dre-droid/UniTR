import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from PIL import Image

def pca_to_rgb(features, n_components=3, pca_obj=None):
    if features.shape[0] < n_components:
        return np.zeros((features.shape[0], 3)), None
    
    if pca_obj is None:
        pca_obj = PCA(n_components=n_components)
        rgb = pca_obj.fit_transform(features)
    else:
        rgb = pca_obj.transform(features)
    
    for i in range(n_components):
        col = rgb[:, i]
        vmin, vmax = np.percentile(col, [2, 98])
        if vmax - vmin > 1e-8:
            col = np.clip(col, vmin, vmax)
            rgb[:, i] = (col - vmin) / (vmax - vmin)
        else:
            rgb[:, i] = 0.5
    
    return rgb, pca_obj

@torch.no_grad()
def extract_vis_features(model, batch_dict):
    # Unwrap DDP if necessary
    if hasattr(model, 'module'):
        model = model.module
    
    backbone = model.teacher_backbone
    old_indices = backbone.out_indices
    backbone.out_indices = [3]  
    
    if not hasattr(backbone, 'out_norm3'):
        import torch.nn as nn
        backbone.add_module('out_norm3', nn.LayerNorm(backbone.d_model[-1]).cuda())
    
    # Run VFE and Backbone
    vfe_batch = model.teacher_vfe(batch_dict)
    voxel_features = vfe_batch['voxel_features']
    voxel_coords = vfe_batch['voxel_coords']
    voxel_num = voxel_features.shape[0]

    teacher_batch = batch_dict.copy()
    teacher_batch.update({
        'voxel_features': voxel_features,
        'voxel_coords': voxel_coords,
        'voxel_num': voxel_num,
    })

    teacher_out = backbone(teacher_batch)
    backbone.out_indices = old_indices

    result = {
        'pillar_features': teacher_out['pillar_features'].cpu().numpy(),
        'pillar_coords': teacher_out['voxel_coords'].cpu().numpy(),
        'patch_features': teacher_out['image_features'][0].cpu().numpy() if 'image_features' in teacher_out else None,
        'camera_imgs': batch_dict['camera_imgs'].cpu().numpy(),
        'voxel_num': voxel_num,
        'batch_size': batch_dict['batch_size'],
    }
    return result

def visualize_ssl(model, train_loader, cfg, epoch, output_dir, logger):
    vis_cfg = cfg.get('VISUALIZATION', None)
    if vis_cfg is None:
        return

    sample_idx = vis_cfg.get('SAMPLE_IDX', 0)
    modes = vis_cfg.get('MODES', ['bev_pca'])
    query_xy = vis_cfg.get('QUERY_XY', [0.0, 0.0])
    
    vis_output_dir = output_dir / 'visualizations' / f'epoch_{epoch}'
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # Get the specific sample
    dataset = train_loader.dataset
    if hasattr(dataset, 'dataset'): # handling wrapper
        dataset = dataset.dataset
    
    # Disable augmentations for deterministic visualization
    old_training = dataset.training
    dataset.training = False
    try:
        data_dict = dataset[sample_idx]
    finally:
        dataset.training = old_training

    batch_dict = dataset.collate_batch([data_dict])
    
    from pcdet.models import load_data_to_gpu
    load_data_to_gpu(batch_dict)

    model.eval()
    result = extract_vis_features(model, batch_dict)
    model.train()

    if 'bev_pca' in modes:
        visualize_bev_pca(result, vis_output_dir, logger)
    if 'camera_pca' in modes:
        visualize_camera_pca(result, vis_output_dir, logger)
    if 'bev_similarity' in modes:
        visualize_bev_similarity(result, query_xy, vis_output_dir, logger)

def visualize_bev_pca(result, output_dir, logger):
    pillar_features = result['pillar_features']
    pillar_coords = result['pillar_coords']
    rgb, _ = pca_to_rgb(pillar_features)
    
    grid_h, grid_w = 360, 360
    bev_image = np.ones((grid_h, grid_w, 3), dtype=np.float32) * 0.15
    mask_b0 = pillar_coords[:, 0] == 0
    coords_b0 = pillar_coords[mask_b0]
    rgb_b0 = rgb[mask_b0]
    
    y_idx, x_idx = coords_b0[:, 2].astype(int), coords_b0[:, 3].astype(int)
    valid = (y_idx >= 0) & (y_idx < grid_h) & (x_idx >= 0) & (x_idx < grid_w)
    bev_image[y_idx[valid], x_idx[valid]] = rgb_b0[valid]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bev_image, origin='lower')
    ax.set_title('BEV PCA Features')
    fig.savefig(output_dir / 'bev_pca.png', bbox_inches='tight')
    plt.close(fig)

def visualize_camera_pca(result, output_dir, logger):
    if result['patch_features'] is None: return
    patch_features = result['patch_features']
    camera_imgs = result['camera_imgs']
    
    B_N, D, pH, pW = patch_features.shape
    all_patches = patch_features[:6].transpose(0, 2, 3, 1).reshape(-1, D)
    rgb_all, _ = pca_to_rgb(all_patches)
    rgb_maps = rgb_all.reshape(6, pH, pW, 3)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for i in range(6):
        ax = axes[i // 2, i % 2]
        rgb_map = rgb_maps[i]
        rgb_upsampled = np.array(Image.fromarray((rgb_map * 255).astype(np.uint8)).resize((704, 256), Image.BILINEAR)) / 255.0
        ax.imshow(rgb_upsampled)
        ax.axis('off')
    fig.savefig(output_dir / 'camera_pca.png', bbox_inches='tight')
    plt.close(fig)

def visualize_bev_similarity(result, query_xy, output_dir, logger):
    pillar_features = result['pillar_features']
    pillar_coords = result['pillar_coords']
    
    mask_b0 = pillar_coords[:, 0] == 0
    feats_b0, coords_b0 = pillar_features[mask_b0], pillar_coords[mask_b0]
    
    query_grid = ((np.array(query_xy) - np.array([-54.0, -54.0])) / 0.3).astype(int)
    grid_xy = coords_b0[:, [3, 2]]
    nearest_idx = np.argmin(np.linalg.norm(grid_xy - query_grid[np.newaxis, :], axis=1))
    query_feat = feats_b0[nearest_idx]
    
    norms = np.linalg.norm(feats_b0, axis=1) + 1e-8
    cosine_sim = (feats_b0 @ query_feat) / (norms * (np.linalg.norm(query_feat) + 1e-8))
    
    grid_h, grid_w = 360, 360
    bev_sim = np.full((grid_h, grid_w), np.nan)
    y_idx, x_idx = coords_b0[:, 2].astype(int), coords_b0[:, 3].astype(int)
    valid = (y_idx >= 0) & (y_idx < grid_h) & (x_idx >= 0) & (x_idx < grid_w)
    bev_sim[y_idx[valid], x_idx[valid]] = cosine_sim[valid]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = cm.get_cmap('coolwarm')
    sim_rgba = cmap((bev_sim + 1) / 2)
    sim_rgba[np.isnan(bev_sim)] = [0.1, 0.1, 0.1, 1.0]
    ax.imshow(sim_rgba, origin='lower')
    fig.savefig(output_dir / 'bev_similarity.png', bbox_inches='tight')
    plt.close(fig)
