import os
import argparse
from PIL import Image
from tqdm import tqdm

def create_gif(source_dir, mode='bev', duration=200):
    """
    source_dir: path to progression study (e.g. ../output/visualizations/progression_study_v2.1)
    mode: 'bev' or 'camera'
    duration: ms per frame
    """
    file_name = 'bev_pca_features.png' if mode == 'bev' else 'camera_pca_features.png'
    output_name = f"{mode}_progression.gif"
    
    # 1. Gather all epoch directories
    epochs = []
    for d in os.listdir(source_dir):
        if d.startswith('epoch_') and os.path.isdir(os.path.join(source_dir, d)):
            epoch_num = int(d.split('_')[1])
            img_path = os.path.join(source_dir, d, file_name)
            if os.path.exists(img_path):
                epochs.append((epoch_num, img_path))
    
    if not epochs:
        print(f"No images found for mode {mode} in {source_dir}")
        return
    
    # 2. Sort by epoch number
    epochs.sort()
    
    # 3. Load images
    frames = []
    for _, img_path in tqdm(epochs, desc=f"Creating {mode} GIF"):
        img = Image.open(img_path)
        # Convert to RGB if necessary (GIFs need a palette or RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frames.append(img)
    
    # 4. Save GIF
    output_path = os.path.join(source_dir, output_name)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to study directory')
    parser.add_argument('--mode', type=str, default='both', choices=['bev', 'camera', 'both'])
    parser.add_argument('--duration', type=int, default=300, help='ms per frame')
    args = parser.parse_args()
    
    if args.mode == 'both':
        create_gif(args.dir, 'bev', args.duration)
        create_gif(args.dir, 'camera', args.duration)
    else:
        create_gif(args.dir, args.mode, args.duration)
