import torch

import cups.scene_flow_2_se3.visual._2d as o4visual2d
from cups.scene_flow_2_se3.visual.classes import VisualizationSettings


class SceneFlow:
    pass


class DataModelRelation:

    def __init__(
        self, likelihood: torch.Tensor, posterior: torch.Tensor, inlier_soft: torch.Tensor, inlier_hard: torch.Tensor
    ):
        """Create relation between data/dataframe and K models likelihood torch.Tensor: KxHxW/KxN, dtype float, range[0,
        inf] posterior torch.Tensor: KxHxW/KxN, dtype float, range[0, 1] inlier_soft torch.Tensor: KxHxW/KxN, dtype
        float, range[0, 1] inlier_hard torch.Tensor: KxHxW/KxN, dtype bool."""


class DynamicRigidObjects:

    def __init__(
        self, K: int = 0, pts3d: torch.Tensor = None, pts3d_assign: torch.Tensor = None, se3s: torch.Tensor = None
    ):
        """Create collection of Dynamic Rigid Objects.

        Parameters
        ----------
        K int: number of objects
        pts3d torch.Tensor: 3xHxW / 3xN, dtype float
        pts3d_assign torch.Tensor: KxHxW / KxN, dtype bool
        se3s torch.Tensor: Kx4x4, dtype bool
        """
        self.K = K
        self.pts3d = pts3d
        self.pts3d_assign = pts3d_assign  # K x H x W / K x N
        self.se3s = se3s  # K x 4 x 4

    def visualize(self, visual_settings: VisualizationSettings):
        """Visualize 3D points in color.

        Parameters
        ----------
        visual_settings:
        intrinsics torch.Tensor: 2x3, dtype float
        extrinsics torch.Tensor: 4x4, dtype float
        H int: window height
        W int: window width
        pt3d_radius float: radius of visualized sphere for each point in 3D
        """
        list_pts3d = [self.pts3d[:, self.pts3d_assign[k]] for k in range(self.K)]
        o4visual2d.visualize_pts3d(
            list_pts3d,
            change_viewport=True,
            extrinsics=visual_settings.extrinsics,
            intrinsics=visual_settings.intrinsics,
            return_img=True,
            colors=o4visual2d.get_colors(self.K),
            H=visual_settings.H,
            W=visual_settings.W,
            radius=visual_settings.pt3d_radius,
        )
