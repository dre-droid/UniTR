"""
Projection head and distillation loss for iBOT-style self-supervised pretraining.

The projection head maps backbone outputs to a space where the distillation loss
(cross-entropy between sharpened teacher and student distributions) is computed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.utils import commu_utils


class iBOTProjectionHead(nn.Module):
    """
    3-layer MLP projection head following DINO/iBOT design.
    Maps backbone features to a normalized output space.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) features from backbone.
        Returns:
            (N, out_dim) L2-normalized projected features.
        """
        out = self.mlp(x)
        # Force FP32 for normalization to prevent FP16 overflow in sum(x^2)
        return F.normalize(out.float(), dim=-1, p=2)


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

        # Force FP32 to prevent precision loss when summing many FP16 activations
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
