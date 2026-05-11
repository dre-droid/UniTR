import torch
from .detector3d_template import Detector3DTemplate
from .unitr_map import UniTRMAP

class UniTRStrictProbe(UniTRMAP):
    def __init__(self, model_cfg, num_class, dataset):
        Detector3DTemplate.__init__(self, model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe','mm_backbone', 'map_to_bev_module',
            'neck','vtransform', 'fuser',
            'dense_head',
        ]
        self.module_list = self.build_networks()
        self.time_list = []
        
        # FREEZE PRETRAINED BACKBONE
        frozen_modules = ['vfe', 'mm_backbone']
        for name, module in self.named_children():
            if name in frozen_modules:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
                
    def _load_state_dict(self, model_state_disk, *, strict=True):
        new_model_state_disk = {}
        for key, val in model_state_disk.items():
            if key.startswith('student_backbone.'):
                new_key = key.replace('student_backbone.', 'mm_backbone.', 1)
                new_model_state_disk[new_key] = val
            elif key.startswith('student.mm_backbone.'):
                new_key = key.replace('student.mm_backbone.', 'mm_backbone.', 1)
                new_model_state_disk[new_key] = val
            elif key.startswith('student_vfe.'):
                new_key = key.replace('student_vfe.', 'vfe.', 1)
                new_model_state_disk[new_key] = val
            else:
                new_model_state_disk[key] = val
        return super()._load_state_dict(new_model_state_disk, strict=strict)

    def train(self, mode=True):
        super().train(mode)
        frozen_modules = ['vfe', 'mm_backbone']
        for name, module in self.named_children():
            if name in frozen_modules:
                module.eval()

    def forward(self, batch_dict):
        # Run frozen modules (everything except the head) in no_grad
        with torch.no_grad():
            for cur_module in self.module_list[:-1]:
                batch_dict = cur_module(batch_dict)
        
        # Run the trainable head with autograd
        batch_dict = self.module_list[-1](batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            ret_dict = [{
                'masks_bev': masks_bev,
                'gt_masks_bev': gt_masks_bev
            } for masks_bev,gt_masks_bev in zip(batch_dict['masks_bev'],batch_dict['gt_masks_bev'])]
            return ret_dict
