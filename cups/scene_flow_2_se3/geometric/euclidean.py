import torch

from cups.scene_flow_2_se3.geometric import grid as o4geo_grid
from cups.scene_flow_2_se3.geometric import pinhole as o4geo_pin


def pts3d_2_dists_eucl(pts3d_1_down, pts3d_2_down=None):
    # pts3d_1_down:    B x 3 x H x W
    # mask_valid_down: B x 1 x H x W
    # B x 3 x N

    if pts3d_2_down == None:
        x1 = pts3d_1_down.flatten(2)

        # B x 3 x N x N
        v1 = x1[:, :, :, None] - x1[:, :, None, :]

        # B x N x N
        dist = torch.norm(v1, dim=1)

    else:
        # out: B x N
        dist = torch.norm(pts3d_2_down - pts3d_1_down, dim=1).flatten(1)

    return dist


def pt3d_2_dev_uvz_rel(pt3d, proj_mat=None, W=None):
    """Calculate deviation in pixel coordinates relative to width and relative depth.

    Parameters
    ----------
    pt3d:    B x 3 x H x W

    Returns
    -------
    dev_uvz_rel:    B x 3 x N x N
    """
    #
    # B x 3 x N

    dtype = pt3d.dtype
    device = pt3d.device

    if W is None or proj_mat is None:
        B, _, H, W = pt3d.shape
        pxl2d_rel_to_width = o4geo_grid.shape_2_pxl2d(B, H, W, dtype=dtype, device=device) / (W - 1)

    else:
        B, _, _, _ = pt3d.shape
        pxl2d_rel_to_width = o4geo_pin.pt3d_2_pxl2d(pt3d, proj_mat) / (W - 1)

    depth = pt3d[:, 2:]

    uv_rel_z = torch.cat((pxl2d_rel_to_width, depth), dim=1)
    uv_rel_z_flat = uv_rel_z.flatten(2)

    # B x 3 x N x N
    uv_rel_z_dev_flat = uv_rel_z_flat[:, :, :, None] - uv_rel_z_flat[:, :, None, :]

    z_avg_flat = (uv_rel_z_flat[:, 2:, :, None].abs() + uv_rel_z_flat[:, 2:, None, :].abs()) / 2.0

    uvz_rel_dev_flat = torch.cat((uv_rel_z_dev_flat[:, :2], uv_rel_z_dev_flat[:, 2:] / (z_avg_flat + 1e-10)), dim=1)
    uvz_rel_dev_flat = uvz_rel_dev_flat.abs()
    # B x N x N
    # dist = torch.norm(v1, dim=1)

    return uvz_rel_dev_flat


def pts_2_inner_product(pts1):

    x1 = pts1.flatten(2)

    inner_product = (x1[:, :, :, None] * x1[:, :, None, :]).sum(dim=1)

    return inner_product
