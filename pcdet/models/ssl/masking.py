"""
Masking utilities for iBOT-style self-supervised pretraining.

Provides random masking for both LiDAR voxel features and image patch features.
Masked positions are replaced with a learnable [MASK] token embedding.
"""

import torch
import torch.nn as nn


class VoxelMasker(nn.Module):
    """Randomly masks a fraction of voxel features with a learnable [MASK] token."""

    def __init__(self, d_model: int, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, voxel_features: torch.Tensor):
        """
        Args:
            voxel_features: (N, D) voxel features from VFE.

        Returns:
            masked_features: (N, D) features with masked positions replaced.
            mask: (N,) boolean tensor, True = masked.
        """
        N = voxel_features.shape[0]
        num_mask = int(N * self.mask_ratio)

        # Random permutation to select which voxels to mask
        perm = torch.randperm(N, device=voxel_features.device)
        mask_indices = perm[:num_mask]

        mask = torch.zeros(N, dtype=torch.bool, device=voxel_features.device)
        mask[mask_indices] = True

        masked_features = voxel_features.clone()
        masked_features[mask] = self.mask_token.to(voxel_features.dtype)

        return masked_features, mask


class PatchMasker(nn.Module):
    """Randomly masks a fraction of image patch features with a learnable [MASK] token."""

    def __init__(self, d_model: int, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patch_features: torch.Tensor):
        """
        Args:
            patch_features: (P, D) patch features from PatchEmbed.

        Returns:
            masked_features: (P, D) features with masked positions replaced.
            mask: (P,) boolean tensor, True = masked.
        """
        P = patch_features.shape[0]
        num_mask = int(P * self.mask_ratio)

        perm = torch.randperm(P, device=patch_features.device)
        mask_indices = perm[:num_mask]

        mask = torch.zeros(P, dtype=torch.bool, device=patch_features.device)
        mask[mask_indices] = True

        masked_features = patch_features.clone()
        masked_features[mask] = self.mask_token.to(patch_features.dtype)

        return masked_features, mask
