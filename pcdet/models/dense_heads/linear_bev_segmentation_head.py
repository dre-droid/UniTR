import torch
from torch import nn
from ...utils import loss_utils
from .bev_segmentation_head import BEVGridTransform

class LinearBEVSegmentationHead(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.classes = class_names
        input_scope = self.model_cfg.GRID_TRANSFORM.INPUT_SCOPE
        output_scope = self.model_cfg.GRID_TRANSFORM.OUTPUT_SCOPE
        self.transform = BEVGridTransform(input_scope, output_scope)
        
        # STRICT LINEAR PROBE: No hidden layers, no non-linearities.
        self.classifier = nn.Conv2d(input_channels, len(self.classes), 1)
        
        loss_cfg = self.model_cfg.LOSS_CONFIG
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cfg.gamma,alpha=loss_cfg.alpha)

    def forward(self, batch_dict):
        # Strict probe takes raw spatial_features directly from PointPillarScatter3d.
        if 'spatial_features_2d' in batch_dict:
            x = batch_dict['spatial_features_2d']
        else:
            x = batch_dict['spatial_features']
            
        target = batch_dict['gt_masks_bev']
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        x = self.classifier(x)

        if self.training:
            tb_dict = {}
            loss_all = 0
            for index, name in enumerate(self.classes):
                pred = x[:, index].flatten(1,2).unsqueeze(-1)
                label = target[:, index].flatten(1,2).unsqueeze(-1)
                label_weight = torch.ones_like(label)
                loss = self.loss_cls(pred,label,label_weight).mean()
                tb_dict[f"loss_{name}"] = loss
                loss_all += loss
            batch_dict['loss'] = loss_all
            batch_dict['tb_dict'] = tb_dict
        else:
            batch_dict['masks_bev'] = torch.sigmoid(x)
        return batch_dict
