"""
Projection head and distillation loss for iBOT-style self-supervised pretraining.

Architecture follows the official iBOT DINOHead design:
  MLP → bottleneck → L2-normalize → weight_norm(Linear) → raw logits

The weight-normalized last layer prevents logit magnitude explosion while
allowing the network to learn meaningful output directions. L2 normalization
in the bottleneck space bounds the input to the last layer, creating tight
control over the output scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.utils import commu_utils


class iBOTProjectionHead(nn.Module):
    """
    Projection head following the official iBOT/DINO design.

    Architecture: 3-layer MLP → bottleneck_dim → L2-norm → weight_norm(Linear → out_dim)

    The bottleneck + L2-norm + weight_norm system ensures:
      - Bottleneck compresses to a compact representation
      - L2-norm puts all features on the unit hypersphere
      - weight_norm(Linear) projects to output space with bounded magnitude
        (weight_g frozen at 1, so only direction is learned)
      - Output: raw logits (NOT L2-normalized), suitable for softmax distillation
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512,
                 bottleneck_dim: int = 256, out_dim: int = 4096,
                 norm_last_layer: bool = True):
        super().__init__()

        # 3-layer MLP projecting to bottleneck dimension
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Initialize weights with truncated normal (matching official iBOT)
        self._init_weights()

        # Weight-normalized last layer: direction is learned, magnitude is frozen at 1
        # This prevents logit magnitude explosion while allowing meaningful logits
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self):
        """Initialize MLP weights with truncated normal (std=0.02)."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) features from backbone.
        Returns:
            (N, out_dim) raw logits (not normalized).
        """
        x = self.mlp(x)
        # L2-normalize in bottleneck space (FP32 for numerical safety)
        x = F.normalize(x.float(), dim=-1, p=2)
        # Project to output space via weight-normalized linear layer
        x = self.last_layer(x)
        return x


class iBOTLoss(nn.Module):
    """
    iBOT distillation loss: cross-entropy between sharpened teacher
    and student output distributions on masked token positions.

    Includes teacher output centering to prevent mode collapse.
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Running center for teacher outputs (prevents collapse)
        self.register_buffer('center', torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update the running center with the mean of teacher outputs."""
        if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
            return

        if len(teacher_output) == 0:
            return

        # Force FP32 to prevent precision loss when summing many activations
        with torch.amp.autocast('cuda', enabled=False):
            teacher_output = teacher_output.float()
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

            # Synchronize across all GPUs: sum the features AND the count independently
            # to correctly handle variable-length inputs (e.g. voxel count differs per GPU)
            world_size = commu_utils.get_world_size()
            if world_size > 1:
                count = torch.tensor(len(teacher_output), dtype=batch_center.dtype,
                                     device=batch_center.device)
                torch.distributed.all_reduce(batch_center)
                torch.distributed.all_reduce(count)
                count = count.clamp(min=1.0)
                batch_center = batch_center / count
            else:
                batch_center = batch_center / max(len(teacher_output), 1)

            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss on masked positions only.

        Args:
            student_output: (N, out_dim) student projection output for all tokens.
            teacher_output: (N, out_dim) teacher projection output for all tokens.
            mask: (N,) boolean mask, True = masked positions (loss computed here).

        Returns:
            Scalar loss value.
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_output.device, requires_grad=True)

        # Select only masked positions
        s_out = student_output[mask]  # (M, out_dim)
        t_out = teacher_output[mask]  # (M, out_dim)

        # Force FP32 to prevent overflows in softmax/exp and cross-entropy sum
        t_out = t_out.detach().float()
        s_out = s_out.float()

        # Teacher: centered + sharpened (no gradient)
        t_probs = F.softmax((t_out - self.center) / self.teacher_temp, dim=-1)

        # Student: sharpened
        s_log_probs = F.log_softmax(s_out / self.student_temp, dim=-1)

        # Cross-entropy loss
        loss = -torch.sum(t_probs * s_log_probs, dim=-1).mean()

        return loss
