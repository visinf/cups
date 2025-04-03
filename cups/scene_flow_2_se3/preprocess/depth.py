import torch

from cups.scene_flow_2_se3 import rearrange as o4rearr


def complete(depth: torch.Tensor, depth_valid: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """Given invalid points complete depth using recursive neighbor interpolation.

    Parameters
    ----------
    depth torch.Tensor: Bx1xHxW, float
    depth_valid torch.Tensor: Bx1xHxW, bool

    Returns
    -------
    depth torch.Tensor: Bx1xHxW, float
    depth_valid torch.Tensor: Bx1xHxW, bool
    """

    depth_valid_orig = depth_valid.clone()
    depth_orig = depth.clone()
    while True:
        if (~depth_valid).sum() == 0:
            break

        patch_size = 3
        depth_neighbors = o4rearr.neighbors_to_channels(depth, patch_size=patch_size, new_dim=False)
        depth_valid_neighbors = o4rearr.neighbors_to_channels(depth_valid, patch_size=patch_size, new_dim=False)
        depth = (depth_neighbors * depth_valid_neighbors).sum(dim=1, keepdim=True) / (
            depth_valid_neighbors.sum(dim=1, keepdim=True) + 1e-10
        )
        depth[depth_valid_orig] = depth_orig[depth_valid_orig]
        depth_valid = depth_valid_neighbors.sum(dim=1, keepdim=True) > 0

    return depth, depth_valid
