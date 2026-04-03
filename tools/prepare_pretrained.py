import torch
import sys

checkpoint = torch.load('../output/nuscenes_models/unitr_ibot/full_dataset_v1/ckpt/latest_model.pth', map_location='cpu')
old_state = checkpoint['model_state']
new_state = {}

for k, v in old_state.items():
    if k.startswith('vfe.'):
        new_state[k] = v
    elif k.startswith('student_backbone.'):
        new_k = k.replace('student_backbone.', 'mm_backbone.', 1)
        new_state[new_k] = v

checkpoint['model_state'] = new_state
# Remove optimizer and scheduler states since we just want the model weights
if 'optimizer_state' in checkpoint:
    del checkpoint['optimizer_state']

torch.save(checkpoint, 'pretrained_unitr.pth')
print(f"Successfully converted checkpoint! Mapped {len(new_state)} keys and saved to tools/pretrained_unitr.pth")
