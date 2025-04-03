import torch


class VisualizationSettings:

    def __init__(self, intrinsics: torch.Tensor, extrinsics: torch.Tensor, H: int, W: int, pt3d_radius: float):
        """Create visualization settings object.

        Parameters
        ----------
        intrinsics torch.Tensor: 2x3, dtype float
        extrinsics torch.Tensor: 4x4, dtype float
        H int: window height
        W int: window width
        pt3d_radius float: radius of visualized sphere for each point in 3D
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.H = H
        self.W = W
        self.pt3d_radius = pt3d_radius
