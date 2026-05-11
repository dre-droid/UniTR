"""
iBOT-style self-supervised pretraining for UniTR with SPLIT projection heads.

Key difference from iBOTUniTR:
  - Separate projection heads per modality (voxel vs. patch) to prevent
    LiDAR gradient dominance from collapsing camera features.
  - Teacher temperature warmup schedule (linear warmup from 0.04 to final value).
  - Configurable loss weighting between modalities.
  - Comprehensive per-step diagnostics for collapse detection.

Architecture:
  - Student: VFE + UniTR backbone (masked) + proj_voxel + proj_patch. Trained via backprop.
  - Teacher: VFE + UniTR backbone (unmasked) + proj_voxel + proj_patch. Updated via EMA.
  - Loss: L_MIM per modality with separate centers, separate heads, and loss weighting.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.backbones_3d import vfe
from pcdet.models.mm_backbone import unitr as mm_unitr_module
from pcdet.models.ssl.masking import VoxelMasker, PatchMasker
from pcdet.models.ssl.projection_head import iBOTProjectionHead, iBOTLoss


class iBOTUniTRSplit(nn.Module):
    """
    iBOT SSL wrapper for UniTR with modality-specific projection heads.

    Changes vs. iBOTUniTR:
    1. Separate student/teacher projection heads per modality:
       - student_proj_voxel / teacher_proj_voxel  (LiDAR)
       - student_proj_patch / teacher_proj_patch  (Camera)
       This prevents LiDAR gradients from dominating the shared MLP weights
       and collapsing camera features.
    2. Configurable PATCH_LOSS_WEIGHT to rebalance modality contributions.
    3. Teacher temperature warmup schedule (optional).
    4. Fixed EMA momentum logging (passes real total_steps).
    5. Comprehensive per-step diagnostics for debugging collapse.
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

        # ---- SPLIT Projection heads (separate per modality) ----
        d_model = model_cfg.MM_BACKBONE.d_model[-1]  # 128
        proj_hidden = self.ssl_cfg.PROJ_HIDDEN_DIM
        proj_out = self.ssl_cfg.PROJ_OUT_DIM
        proj_bottleneck = self.ssl_cfg.get('PROJ_BOTTLENECK_DIM', 256)

        # Voxel (LiDAR) projection heads
        self.student_proj_voxel = iBOTProjectionHead(d_model, proj_hidden, proj_bottleneck, proj_out)
        self.teacher_proj_voxel = iBOTProjectionHead(d_model, proj_hidden, proj_bottleneck, proj_out)
        for p in self.teacher_proj_voxel.parameters():
            p.requires_grad = False

        # Patch (Camera) projection heads
        self.student_proj_patch = iBOTProjectionHead(d_model, proj_hidden, proj_bottleneck, proj_out)
        self.teacher_proj_patch = iBOTProjectionHead(d_model, proj_hidden, proj_bottleneck, proj_out)
        for p in self.teacher_proj_patch.parameters():
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

        # ---- Loss weighting ----
        self.patch_loss_weight = self.ssl_cfg.get('PATCH_LOSS_WEIGHT', 1.0)

        # ---- Teacher temperature warmup (optional) ----
        self.teacher_temp_start = self.ssl_cfg.get('TEACHER_TEMP_START', self.ssl_cfg.TEACHER_TEMP)
        self.teacher_temp_end = self.ssl_cfg.get('TEACHER_TEMP_END', self.ssl_cfg.TEACHER_TEMP)
        self.teacher_temp_warmup_epochs = self.ssl_cfg.get('TEACHER_TEMP_WARMUP_EPOCHS', 0)

        # ---- EMA state ----
        self.ema_momentum_start = self.ssl_cfg.EMA_MOMENTUM_START
        self.ema_momentum_end = self.ssl_cfg.EMA_MOMENTUM_END

        # Initialize teacher as exact copy of student (params + buffers)
        self._copy_student_to_teacher()

        # Global step counter for EMA schedule
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        # Store total_steps for correct logging (set by model_fn_ssl)
        self.register_buffer('_total_steps', torch.LongTensor([1]))

    def update_global_step(self):
        self.global_step += 1

    @torch.no_grad()
    def _copy_student_to_teacher(self):
        """Initialize teacher weights as exact copy of student."""
        for t_param, s_param in zip(self.teacher_vfe.parameters(),
                                     self.student_vfe.parameters()):
            t_param.data.copy_(s_param.data)
        for t_param, s_param in zip(self.teacher_backbone.parameters(),
                                     self.student_backbone.parameters()):
            t_param.data.copy_(s_param.data)
        # Split heads: copy each modality independently
        for t_param, s_param in zip(self.teacher_proj_voxel.parameters(),
                                     self.student_proj_voxel.parameters()):
            t_param.data.copy_(s_param.data)
        for t_param, s_param in zip(self.teacher_proj_patch.parameters(),
                                     self.student_proj_patch.parameters()):
            t_param.data.copy_(s_param.data)

    @torch.no_grad()
    def update_teacher(self, total_steps: int):
        """EMA update of teacher parameters.

        Teacher stays in train mode (uses batch BN stats for normalization).
        Only learned parameters (weights/biases) are EMA-updated.
        BN running stats are naturally updated during the teacher's forward pass.
        """
        # Store total_steps for correct logging
        self._total_steps.fill_(total_steps)

        momentum = self._get_momentum(total_steps)

        with torch.amp.autocast('cuda', enabled=False):
            for t_param, s_param in zip(self.teacher_vfe.parameters(),
                                         self.student_vfe.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)
            for t_param, s_param in zip(self.teacher_backbone.parameters(),
                                         self.student_backbone.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)
            # Split heads: EMA each modality independently
            for t_param, s_param in zip(self.teacher_proj_voxel.parameters(),
                                         self.student_proj_voxel.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)
            for t_param, s_param in zip(self.teacher_proj_patch.parameters(),
                                         self.student_proj_patch.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data.float(), alpha=1.0 - momentum)

    def _get_momentum(self, total_steps: int) -> float:
        """Cosine schedule for EMA momentum: starts low (0.996), anneals to 1.0."""
        step = self.global_step.item()
        progress = min(step / max(total_steps, 1), 1.0)
        momentum = self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * \
            (math.cos(math.pi * progress) + 1) / 2
        return momentum

    def _get_teacher_temp(self) -> float:
        """Linear warmup of teacher temperature, then constant."""
        if self.teacher_temp_warmup_epochs <= 0:
            return self.teacher_temp_end
        # Approximate epoch from global step (rough, but sufficient for a smooth schedule)
        total_steps = self._total_steps.item()
        step = self.global_step.item()
        if total_steps <= 0:
            return self.teacher_temp_end
        num_epochs = self.model_cfg.get('_num_epochs_', 100)
        steps_per_epoch = max(total_steps / num_epochs, 1)
        current_epoch = step / steps_per_epoch
        if current_epoch < self.teacher_temp_warmup_epochs:
            # Linear warmup
            alpha = current_epoch / self.teacher_temp_warmup_epochs
            return self.teacher_temp_start + alpha * (self.teacher_temp_end - self.teacher_temp_start)
        return self.teacher_temp_end

    def forward(self, batch_dict):
        """
        Forward pass for iBOT pretraining with split projection heads.

        Args:
            batch_dict: dict from dataloader containing 'points', 'camera_imgs', etc.

        Returns:
            ret_dict: {'loss': scalar}
            tb_dict: tensorboard logging dict (comprehensive diagnostics)
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

        # ---- Step 5: Project through SPLIT heads ----
        student_voxel_proj = self.student_proj_voxel(student_voxel_out)   # (N, proj_out)
        student_patch_proj = self.student_proj_patch(student_patch_out)   # (P, proj_out)

        with torch.no_grad():
            teacher_voxel_proj = self.teacher_proj_voxel(teacher_voxel_out)  # (N, proj_out)
            teacher_patch_proj = self.teacher_proj_patch(teacher_patch_out)  # (P, proj_out)

        # ---- Step 6: Compute L_MIM loss on masked positions (separate centers) ----
        loss_voxel = self.loss_fn_voxel(student_voxel_proj, teacher_voxel_proj, voxel_mask)
        loss_patch = self.loss_fn_patch(student_patch_proj, teacher_patch_proj, patch_mask)

        # ---- Step 7: Update centers per modality ----
        with torch.no_grad():
            self.loss_fn_voxel.update_center(teacher_voxel_proj.detach())
            self.loss_fn_patch.update_center(teacher_patch_proj.detach())

        # ---- Total loss with modality weighting ----
        loss = loss_voxel + self.patch_loss_weight * loss_patch

        # ---- Comprehensive Diagnostics (no_grad, minimal overhead) ----
        with torch.no_grad():
            # Use real total_steps for correct momentum logging
            real_total_steps = self._total_steps.item()
            ema_momentum = self._get_momentum(real_total_steps)
            teacher_temp = self._get_teacher_temp()

            # -- Teacher logit scale (should be O(1-10), not growing unboundedly) --
            t_vox_scale = teacher_voxel_proj.float().abs().mean().item()
            t_pat_scale = teacher_patch_proj.float().abs().mean().item()
            s_vox_scale = student_voxel_proj.float().abs().mean().item()
            s_pat_scale = student_patch_proj.float().abs().mean().item()

            # -- Feature norms BEFORE projection (backbone output health) --
            s_vox_feat_norm = student_voxel_out.float().norm(dim=-1).mean().item()
            s_pat_feat_norm = student_patch_out.float().norm(dim=-1).mean().item()
            t_vox_feat_norm = teacher_voxel_out.float().norm(dim=-1).mean().item()
            t_pat_feat_norm = teacher_patch_out.float().norm(dim=-1).mean().item()

            # -- Collapse detection: fraction of tokens voting for the mode bucket --
            # Healthy: < 0.05. Concerning: > 0.10. Collapsed: > 0.50
            t_vox_top1 = teacher_voxel_proj.float().argmax(dim=-1)
            t_vox_collapse = (t_vox_top1 == t_vox_top1.mode().values).float().mean().item()
            t_pat_top1 = teacher_patch_proj.float().argmax(dim=-1)
            t_pat_collapse = (t_pat_top1 == t_pat_top1.mode().values).float().mean().item()

            # -- Student-side collapse (important: if student collapses, teacher follows via EMA) --
            s_vox_top1 = student_voxel_proj.float().argmax(dim=-1)
            s_vox_collapse = (s_vox_top1 == s_vox_top1.mode().values).float().mean().item()
            s_pat_top1 = student_patch_proj.float().argmax(dim=-1)
            s_pat_collapse = (s_pat_top1 == s_pat_top1.mode().values).float().mean().item()

            # -- Center norm (large center → imbalanced distribution) --
            center_vox_norm = self.loss_fn_voxel.center.float().norm().item()
            center_pat_norm = self.loss_fn_patch.center.float().norm().item()

            # -- Teacher output entropy (bits). Collapsed → 0, healthy → high --
            # Use a random subsample to keep compute low
            _n_sample = min(4096, teacher_voxel_proj.shape[0])
            t_vox_probs = F.softmax(teacher_voxel_proj[:_n_sample].float() / max(self.ssl_cfg.TEACHER_TEMP, 1e-6), dim=-1)
            t_vox_entropy = -(t_vox_probs * (t_vox_probs + 1e-8).log()).sum(dim=-1).mean().item()
            _n_sample_p = min(4096, teacher_patch_proj.shape[0])
            t_pat_probs = F.softmax(teacher_patch_proj[:_n_sample_p].float() / max(self.ssl_cfg.TEACHER_TEMP, 1e-6), dim=-1)
            t_pat_entropy = -(t_pat_probs * (t_pat_probs + 1e-8).log()).sum(dim=-1).mean().item()

            # -- Projection head weight norms (per modality, detect divergence) --
            proj_vox_weight_norm = sum(p.data.float().norm().item()
                                       for p in self.student_proj_voxel.mlp.parameters() if p.dim() >= 2) \
                                  / max(sum(1 for p in self.student_proj_voxel.mlp.parameters() if p.dim() >= 2), 1)
            proj_pat_weight_norm = sum(p.data.float().norm().item()
                                       for p in self.student_proj_patch.mlp.parameters() if p.dim() >= 2) \
                                  / max(sum(1 for p in self.student_proj_patch.mlp.parameters() if p.dim() >= 2), 1)

            # -- Projection head gradient norms (per modality) --
            grad_vox = [p.grad.float().norm().item()
                        for p in self.student_proj_voxel.parameters() if p.grad is not None]
            proj_vox_grad_norm = sum(grad_vox) / max(len(grad_vox), 1) if grad_vox else float('nan')
            grad_pat = [p.grad.float().norm().item()
                        for p in self.student_proj_patch.parameters() if p.grad is not None]
            proj_pat_grad_norm = sum(grad_pat) / max(len(grad_pat), 1) if grad_pat else float('nan')

            # -- Student/teacher param distance per modality --
            vox_param_dists = [
                (tp.data.float() - sp.data.float()).norm().item()
                for tp, sp in zip(self.teacher_proj_voxel.parameters(),
                                  self.student_proj_voxel.parameters())
            ]
            proj_vox_param_dist = sum(vox_param_dists) / max(len(vox_param_dists), 1)
            pat_param_dists = [
                (tp.data.float() - sp.data.float()).norm().item()
                for tp, sp in zip(self.teacher_proj_patch.parameters(),
                                  self.student_proj_patch.parameters())
            ]
            proj_pat_param_dist = sum(pat_param_dists) / max(len(pat_param_dists), 1)

            # -- Backbone S/T distance --
            bb_param_dists = [
                (tp.data.float() - sp.data.float()).norm().item()
                for tp, sp in zip(self.teacher_backbone.parameters(),
                                  self.student_backbone.parameters())
            ]
            bb_param_dist = sum(bb_param_dists) / max(len(bb_param_dists), 1)

            # -- Backbone weight norm --
            bb_norms = [p.data.float().norm().item()
                        for p in self.student_backbone.parameters() if p.dim() >= 2]
            bb_weight_norm = sum(bb_norms) / max(len(bb_norms), 1)

        tb_dict = {
            # ---- Losses ----
            'loss_mim_voxel': loss_voxel.item(),
            'loss_mim_patch': loss_patch.item(),
            'loss_total': loss.item(),

            # ---- Training state ----
            'ema_momentum': ema_momentum,
            'teacher_temp': teacher_temp,

            # ---- Token counts ----
            'num_voxels': voxel_num,
            'num_masked_voxels': voxel_mask.sum().item(),
            'num_masked_patches': patch_mask.sum().item(),

            # ---- Logit scales (should not grow unboundedly) ----
            'diag/teacher_voxel_logit_scale': t_vox_scale,
            'diag/teacher_patch_logit_scale': t_pat_scale,
            'diag/student_voxel_logit_scale': s_vox_scale,
            'diag/student_patch_logit_scale': s_pat_scale,

            # ---- Backbone feature norms (pre-projection) ----
            'diag/student_voxel_feat_norm': s_vox_feat_norm,
            'diag/student_patch_feat_norm': s_pat_feat_norm,
            'diag/teacher_voxel_feat_norm': t_vox_feat_norm,
            'diag/teacher_patch_feat_norm': t_pat_feat_norm,

            # ---- Collapse detection (fraction voting for mode bucket) ----
            # Healthy: < 0.05. Concerning: > 0.10. Collapsed: > 0.50.
            'diag/teacher_voxel_collapse_ratio': t_vox_collapse,
            'diag/teacher_patch_collapse_ratio': t_pat_collapse,
            'diag/student_voxel_collapse_ratio': s_vox_collapse,
            'diag/student_patch_collapse_ratio': s_pat_collapse,

            # ---- Teacher output entropy (bits, higher=healthier) ----
            'diag/teacher_voxel_entropy': t_vox_entropy,
            'diag/teacher_patch_entropy': t_pat_entropy,

            # ---- Center norms ----
            'diag/center_voxel_norm': center_vox_norm,
            'diag/center_patch_norm': center_pat_norm,

            # ---- Per-modality projection head health ----
            'diag/proj_voxel_weight_norm': proj_vox_weight_norm,
            'diag/proj_patch_weight_norm': proj_pat_weight_norm,
            'diag/proj_voxel_grad_norm': proj_vox_grad_norm,
            'diag/proj_patch_grad_norm': proj_pat_grad_norm,

            # ---- Student/Teacher distances (detect EMA stalling or collapse) ----
            'diag/st_dist_proj_voxel': proj_vox_param_dist,
            'diag/st_dist_proj_patch': proj_pat_param_dist,
            'diag/st_dist_backbone': bb_param_dist,

            # ---- Backbone health ----
            'diag/backbone_weight_norm': bb_weight_norm,
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
