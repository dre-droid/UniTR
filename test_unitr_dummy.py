import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.mm_backbone.unitr import UniTR

def test_unitr_encoder():
    cfg_file = 'tools/cfgs/nuscenes_models/unitr.yaml'
    cfg_from_yaml_file(cfg_file, cfg)

    model_cfg = cfg.MODEL.MM_BACKBONE

    # Disable FUSE_BACKBONE to avoid requiring complex dataset camera extrinsics
    if 'FUSE_BACKBONE' in model_cfg:
        model_cfg.pop('FUSE_BACKBONE')

    model = UniTR(model_cfg=model_cfg).cuda()
    model.eval()

    batch_size = 1
    num_cams = 6
    C, H, W = 3, 256, 704
    num_voxels = 500  
    voxel_feat_dim = 128  

    print("Generating dummy test vectors...")
    print(f" -> Camera Images: {batch_size}x{num_cams}x{C}x{H}x{W}")
    print(f" -> Voxel Features: {num_voxels}x{voxel_feat_dim}")

    dummy_imgs = torch.randn(batch_size, num_cams, C, H, W).cuda()
    dummy_voxel_features = torch.randn(num_voxels, voxel_feat_dim).cuda()
    
    dummy_voxel_coords = torch.zeros((num_voxels, 4), dtype=torch.int32).cuda()
    dummy_voxel_coords[:, 0] = 0
    dummy_voxel_coords[:, 1] = 0
    dummy_voxel_coords[:, 2] = torch.randint(0, 360, (num_voxels,))
    dummy_voxel_coords[:, 3] = torch.randint(0, 360, (num_voxels,))

    batch_dict = {
        'batch_size': batch_size,
        'camera_imgs': dummy_imgs,
        'voxel_features': dummy_voxel_features,
        'voxel_coords': dummy_voxel_coords,
        'voxel_num': num_voxels
    }

    print("Running forward pass through UniTR...")
    with torch.no_grad():
        output_dict = model(batch_dict)

    print("\nInference Complete! Final output embeddings:")
    if output_dict['image_features']:
        print(f" -> Image Features Shape: {output_dict['image_features'][0].shape}")
    else:
        print(f" -> Image Features: [] (out_indices is empty, no image features extracted)")
    print(f" -> Pillar Features Shape: {output_dict['pillar_features'].shape}")

if __name__ == '__main__':
    test_unitr_encoder()
