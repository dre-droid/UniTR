
import sys
import os
import torch
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/andrea_mastroberti/UniTR')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

def get_scene_info(sample_idx):
    cfg_file = '/home/andrea_mastroberti/UniTR/tools/cfgs/nuscenes_models/unitr_ibot_fusion.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Build dataset
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        training=False,
        logger=None
    )
    
    # Get total samples
    total_samples = len(dataset)
    if sample_idx >= total_samples:
        print(f"Index {sample_idx} is out of range (total {total_samples})")
        return
    
    # Get sample and navigate to its scene
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg.DATA_CONFIG.VERSION, dataroot=str(dataset.root_path), verbose=False)
    
    sample_info = dataset.infos[sample_idx]
    sample_token = sample_info['token']
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    
    print(f"SAMPLE_INDEX: {sample_idx}")
    print(f"SAMPLE_TOKEN: {sample_token}")
    print(f"SCENE_TOKEN: {scene_token}")
    print(f"SCENE_NAME: {scene['name']}")
    
    # Get all samples in this scene
    first_sample_token = scene['first_sample_token']
    current_sample_token = first_sample_token
    scene_samples = []
    while current_sample_token != '':
        scene_samples.append(current_sample_token)
        current_sample_token = nusc.get('sample', current_sample_token)['next']
    
    print(f"TOTAL_SAMPLES_IN_SCENE: {len(scene_samples)}")
    print(f"SCENE_SAMPLE_TOKENS: {scene_samples}")

if __name__ == '__main__':
    get_scene_info(5)
