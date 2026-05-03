"""
iBOT-style self-supervised pretraining wrapper for the UniTR multi-modal backbone.

Architecture:
  - Student: VFE + UniTR backbone (with masked inputs) + projection head. Trained via backprop.
  - Teacher: VFE + UniTR backbone (unmasked inputs) + projection head. Updated via EMA.
  - Loss: L_MIM (cross-entropy distillation on masked token positions).
"""

import copy
import math
import torch
import torch.nn as nn

from pcdet.models.backbones_3d import vfe
from pcdet.models.mm_backbone import unitr as mm_unitr_module
from pcdet.models.ssl.masking import VoxelMasker, PatchMasker
from pcdet.models.ssl.projection_head import iBOTProjectionHead, iBOTLoss


class iBOTUniTR(nn.Module):
    """
    iBOT self-supervised pretraining wrapper for UniTR.

    Wraps student/teacher UniTR backbones with:
    - Independent VFEs for student and teacher
    - Independent masking for voxels and patches
    - Shared-architecture MLP projection heads (separate weights for student/teacher)
    - EMA teacher updates (parameters + BN buffers)
    - L_MIM distillation loss with separate centers per modality
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.ssl_cfg = model_cfg.SSL

        # ---- Build VFE (Student & Teacher) ----
        vfe_cfg = model_cfg.VFE
        vfe_kwargs = dict(
            model_cfg=vfe_cfg,
            num_point_features=dataset.point_feature_encoder.num_point_features,
            point_cloud_range=dataset.point_cloud_range,
            voxel_size=dataset.voxel_size,
            grid_size=dataset.grid_size,
            depth_downsample_factor=dataset.depth_downsample_factor,
        )
        self.student_vfe = vfe.__all__[vfe_cfg.NAME](**vfe_kwargs)
        self.teacher_vfe = vfe.__all__[vfe_cfg.NAME](**vfe_kwargs)
        for p in self.teacher_vfe.parameters():
            p.requires_grad = False

        # ---- Build UniTR backbone (student) ----
        mm_cfg = copy.deepcopy(model_cfg.MM_BACKBONE)
        backbone_name = mm_cfg.pop('NAME')
        self.student_backbone = getattr(mm_unitr_module, backbone_name)(model_cfg=mm_cfg)

        # ---- Build UniTR backbone (teacher) — deep copy, no gradients ----
        mm_cfg_t = copy.deepcopy(model_cfg.MM_BACKBONE)
        mm_cfg_t.pop('NAME')
        self.teacher_backbone = getattr(mm_unitr_module, backbone_name)(model_cfg=mm_cfg_t)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # ---- Projection heads (shared architecture, separate student/teacher weights) ----
        d_model = model_cfg.MM_BACKBONE.d_model[-1]  # 128
        proj_hidden = self.ssl_cfg.PROJ_HIDDEN_DIM
        proj_out = self.ssl_cfg.PROJ_OUT_DIM

        self.student_proj = iBOTProjectionHead(d_model, proj_hidden, proj_out)
        self.teacher_proj = iBOTProjectionHead(d_model, proj_hidden, proj_out)
        for p in self.teacher_proj.parameters():
            p.requires_grad = False

        # ---- Maskers ----
        self.voxel_masker = VoxelMasker(d_model, self.ssl_cfg.MASK_RATIO_VOXEL)
        self.patch_masker = PatchMasker(d_model, self.ssl_cfg.MASK_RATIO_PATCH)

        # ---- Loss (separate per modality to prevent center cross-contamination) ----
        loss_kwargs = dict(
            out_dim=proj_out,
            teacher_temp=self.ssl_cfg.TEACHER_TEMP,
            student_temp=self.ssl_cfg.STUDENT_TEMP,
            center_momentum=self.ssl_cfg.CENTER_MOMENTUM,
        )
        self.loss_fn_voxel = iBOTLoss(**loss_kwargs)
        self.loss_fn_patch = iBOTLoss(**loss_kwargs)

        # ---- EMA state ----
        self.ema_momentum_start = self.ssl_cfg.EMA_MOMENTUM_START
        self.ema_momentum_end = self.ssl_cfg.EMA_MOMENTUM_END

        # Initialize teacher as exact copy of student (params + buffers)
        self._copy_student_to_teacher()

        # Global step counter for EMA schedule
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

    def train(self, mode=True):
        """Override to keep teacher modules permanently in eval mode.

        Teacher BN stats are updated via EMA from student, not from forward passes.
        Allowing teacher to run in train mode would pollute its BN running stats
        with batch-level statistics independent of the EMA trajectory.
        """
        super().train(mode)
        self.teacher_vfe.eval()
        self.teacher_backbone.eval()
        self.teacher_proj.eval()
        return self

    def update_global_step(self):
        self.global_step += 1

    @torch.no_grad()
    def _copy_student_to_teacher(self):
        """Initialize teacher weights and buffers as exact copy of student."""
        for t_param, s_param in zip(self.teacher_vfe.parameters(),
                                     self.student_vfe.parameters()):
            t_param.data.copy_(s_param.data)
        for t_param, s_param in zip(self.teacher_backbone.parameters(),
                                     self.student_backbone.parameters()):
            t_param.data.copy_(s_param.data)
        for t_param, s_param in zip(self.teacher_proj.parameters(),
                                     self.student_proj.parameters()):
            t_param.data.copy_(s_param.data)

        # Also copy buffers (BN running_mean, running_var, etc.)
        self._copy_buffers(self.teacher_vfe, self.student_vfe)
        self._copy_buffers(self.teacher_backbone, self.student_backbone)
        self._copy_buffers(self.teacher_proj, self.student_proj)

    @staticmethod
    @torch.no_grad()
    def _copy_buffers(target_module, source_module):
        """Copy all buffers from source to target module."""
        for t_buf, s_buf in zip(target_module.buffers(), source_module.buffers()):
            t_buf.data.copy_(s_buf.data)

    @torch.no_grad()
    def update_teacher(self, total_steps: int):
        """EMA update of teacher from student weights and buffers.

        Runs in FP32 to prevent precision loss from autocast context.
        """
        momentum = self._get_momentum(total_steps)

        with torch.cuda.amp.autocast(enabled=False):
            # EMA update parameters
            for t_param, s_param in zip(self.teacher_vfe.parameters(),
                                         self.student_vfe.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)
            for t_param, s_param in zip(self.teacher_backbone.parameters(),
                                         self.student_backbone.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)
            for t_param, s_param in zip(self.teacher_proj.parameters(),
                                         self.student_proj.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)

            # EMA update buffers (BN running_mean/running_var)
            # Integer buffers (e.g. num_batches_tracked) are copied directly
            self._ema_buffers(self.teacher_vfe, self.student_vfe, momentum)
            self._ema_buffers(self.teacher_backbone, self.student_backbone, momentum)
            self._ema_buffers(self.teacher_proj, self.student_proj, momentum)

    @staticmethod
    @torch.no_grad()
    def _ema_buffers(target_module, source_module, momentum):
        """EMA update floating-point buffers; direct copy for integer buffers."""
        for t_buf, s_buf in zip(target_module.buffers(), source_module.buffers()):
            if t_buf.dtype.is_floating_point:
                t_buf.data.mul_(momentum).add_(s_buf.data.float(), alpha=1.0 - momentum)
            else:
                t_buf.data.copy_(s_buf.data)

    def _get_momentum(self, total_steps: int) -> float:
        """Cosine schedule for EMA momentum: starts low (0.996), anneals to 1.0."""
        step = self.global_step.item()
        progress = min(step / max(total_steps, 1), 1.0)
        momentum = self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * \
            (math.cos(math.pi * progress) + 1) / 2
        return momentum

    def forward(self, batch_dict):
        """
        Forward pass for iBOT pretraining.

        Args:
            batch_dict: dict from dataloader containing 'points', 'camera_imgs', etc.

        Returns:
            ret_dict: {'loss': scalar}
            tb_dict: tensorboard logging dict
            disp_dict: display dict
        """
        # ---- Step 1: Independent VFE for Student and Teacher ----
        student_batch = dict(batch_dict)
        student_batch = self.student_vfe(student_batch)
        student_voxel_features = student_batch['voxel_features']       # (N, 128)
        voxel_num = student_voxel_features.shape[0]

        with torch.no_grad():
            teacher_batch = dict(batch_dict)
            teacher_batch = self.teacher_vfe(teacher_batch)

        # ---- Step 2: Mask voxel features (student only) ----
        masked_voxel_features, voxel_mask = self.voxel_masker(student_voxel_features)

        # ---- Step 3: Student forward (masked) ----
        student_batch.update({
            'voxel_features': masked_voxel_features,
        })
        student_out = self.student_backbone(student_batch, patch_masker=self.patch_masker)
        student_voxel_out = student_out['pillar_features']  # (N, 128)
        student_patch_out = student_out['patch_features']   # (P, 128)
        patch_mask = student_out['patch_mask']              # (P,)

        # ---- Step 4: Teacher forward (unmasked) ----
        with torch.no_grad():
            teacher_out = self.teacher_backbone(teacher_batch)
            teacher_voxel_out = teacher_out['pillar_features']  # (N, 128)
            teacher_patch_out = teacher_out['patch_features']   # (P, 128)

        # ---- Step 5: Project ----
        student_voxel_proj = self.student_proj(student_voxel_out)   # (N, proj_out)
        with torch.no_grad():
            teacher_voxel_proj = self.teacher_proj(teacher_voxel_out)  # (N, proj_out)

        student_patch_proj = self.student_proj(student_patch_out)   # (P, proj_out)
        with torch.no_grad():
            teacher_patch_proj = self.teacher_proj(teacher_patch_out)  # (P, proj_out)

        # ---- Step 6: Compute L_MIM loss on masked positions (separate centers) ----
        loss_voxel = self.loss_fn_voxel(student_voxel_proj, teacher_voxel_proj, voxel_mask)
        loss_patch = self.loss_fn_patch(student_patch_proj, teacher_patch_proj, patch_mask)

        # ---- Step 7: Update centers per modality ----
        with torch.no_grad():
            self.loss_fn_voxel.update_center(teacher_voxel_proj.detach())
            self.loss_fn_patch.update_center(teacher_patch_proj.detach())

        # ---- Total loss ----
        loss = loss_voxel + loss_patch

        tb_dict = {
            'loss_mim_voxel': loss_voxel.item(),
            'loss_mim_patch': loss_patch.item(),
            'loss_total': loss.item(),
            'ema_momentum': self._get_momentum(1),
            'num_voxels': voxel_num,
            'num_masked_voxels': voxel_mask.sum().item(),
            'num_masked_patches': patch_mask.sum().item(),
        }
        disp_dict = {}

        ret_dict = {'loss': loss}
        return ret_dict, tb_dict, disp_dict

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        """Load parameters from checkpoint (compatibility with training utils)."""
        import os
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        state_dict = self.state_dict()
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val

        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        """Load parameters along with optimizer state."""
        import os
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading checkpoint %s' % filename)
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        model_state_disk = checkpoint['model_state']
        state_dict = self.state_dict()
        update_state = {k: v for k, v in model_state_disk.items()
                        if k in state_dict and state_dict[k].shape == v.shape}
        state_dict.update(update_state)
        self.load_state_dict(state_dict)

        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info('==> Done')
        return it, epoch
