import sys, copy
sys.path.insert(0, "../")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import torch
import warnings

warnings.filterwarnings('ignore')

cfg_from_yaml_file("cfgs/nuscenes_models/unitr_ibot.yaml", cfg)
logger = common_utils.create_logger()

from pcdet.datasets import build_dataloader
train_set, train_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
    batch_size=1, dist=False, workers=0, logger=logger, training=True,
)
print("Dataset OK:", len(train_set))

from pcdet.models.ssl import iBOTUniTR
print("Building iBOTUniTR...")
model = iBOTUniTR(model_cfg=cfg.MODEL, num_class=0, dataset=train_set)
model.cuda()
model.train()

param_count = sum(p.numel() for p in model.parameters())
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {param_count:,}, Trainable: {trainable_count:,}")
print("Model built successfully!")

# Test forward pass
from pcdet.models import load_data_to_gpu
print("Fetching batch...")
batch = next(iter(train_loader))
load_data_to_gpu(batch)
print("Running forward pass...")
with torch.cuda.amp.autocast(enabled=True):
    ret_dict, tb_dict, disp_dict = model(batch)
print("Loss:", ret_dict["loss"].item())
print("TB dict:", tb_dict)
print("SMOKE TEST PASSED!")
