import os
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from cups.utils import normalize_min_max_m1_1

from ._smurf import ResidualBlock, _raft

__all__: Tuple[str, ...] = ("raft_smurf",)


class _RAFT(nn.Module):
    """This class wraps Torchvision's RAFT models and always ensures eval mode."""

    def __init__(self, raft: nn.Module) -> None:
        # Call super constructor
        super(_RAFT, self).__init__()
        # Save RAFT
        self.raft: nn.Module = raft

    def forward(
        self,
        images_1: Tensor,
        images_2: Tensor,
        disparity: bool = False,
        forward: bool = True,
        num_flow_updates: int = 16,
    ) -> Tensor:
        """Forward pass.

        Args:
            images_1 (Tensor): Batch of first images of the shape [B, 3, H, W] pixel range is [0, 1].
            images_2 (Tensor): Batch of second images of the shape [B, 3, H, W] pixel range is [0, 1].
            disparity (bool): If true disparity is returned
            forward (bool): If true forward disparity is returned
            num_flow_updates (int): Number of iterative optical flow updates (GRU passes). Default: 16.

        Returns:
            optical_float (Tensor): Predicted optical flow of the shape [B, 2, H, W] or disparity of shape [B, 1, H, W].
        """
        # Normalize input images to a pixel range of [-1, 1]
        images_normalized_1: Tensor = normalize_min_max_m1_1(images_1)
        images_normalized_2: Tensor = normalize_min_max_m1_1(images_2)
        # Perform forward pass
        optical_flow: Tensor = self.raft(
            images_normalized_1,
            images_normalized_2,
            num_flow_updates=num_flow_updates,
        )[-1]
        # Get disparity
        if disparity:
            if forward:
                disparity_map: Tensor = -optical_flow[:, :1]
                # We clip since we know the disparity must always be zero or larger
                return disparity_map.clip(min=0)
            else:
                disparity_map = optical_flow[:, :1]
                return disparity_map.clip(min=0)
        return optical_flow


def raft_smurf() -> nn.Module:
    """Build a pre-trained (unsupervised, SMURF) RAFT small model.

    Returns:
        model (nn.Module): RAFT small model.
    """
    raft = _RAFT(
        raft=_raft(
            weights=None,
            progress=False,
            # Feature encoder
            feature_encoder_layers=(64, 64, 96, 128, 256),
            feature_encoder_block=ResidualBlock,
            feature_encoder_norm_layer=nn.InstanceNorm2d,
            # Context encoder
            context_encoder_layers=(64, 64, 96, 128, 256),
            context_encoder_block=ResidualBlock,
            context_encoder_norm_layer=nn.InstanceNorm2d,
            # Correlation block
            corr_block_num_levels=4,
            corr_block_radius=4,
            # Motion encoder
            motion_encoder_corr_layers=(256, 192),
            motion_encoder_flow_layers=(128, 64),
            motion_encoder_out_channels=128,
            # Recurrent block
            recurrent_block_hidden_state_size=128,
            recurrent_block_kernel_size=((1, 5), (5, 1)),
            recurrent_block_padding=((0, 2), (2, 0)),
            # Flow head
            flow_head_hidden_size=256,
            # Mask predictor
            use_mask_predictor=True,
        )
    )
    # Load checkpoint from SMURF paper
    raft.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/raft_smurf.pt")))
    return raft
