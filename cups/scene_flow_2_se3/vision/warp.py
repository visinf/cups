import torch

from cups.scene_flow_2_se3.geometric.grid import (
    oflow_2_pxl2d,
    pxl2d_2_pxl2d_normalized,
    pxl2d_normalized_2_mask_inside,
)


def interpolate2d(x, pxl2d, return_masks_flow_inside=False, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = x.size()
    pxl2d_normalized = pxl2d_2_pxl2d_normalized(pxl2d, H=H, W=W)
    mask_flow_inside = pxl2d_normalized_2_mask_inside(pxl2d_normalized)

    pxlcoords_normalized_perm = pxl2d_normalized.transpose(1, 2).transpose(2, 3)  # .permute(0, 2, 3, 1)

    dtype = x.dtype
    device = x.device
    # print(pxlcoords_normalized_perm.dtype)
    # print(x.dtype)
    # x = x.type_as(pxlcoords_normalized_perm)
    # pxlcoords_normalized_perm = pxlcoords_normalized_perm.type_as(x)
    # input: (B, C, Hin​, Win​) and grid: (1, Hout​, Wout​, 2)
    # output: (B, C, Hin, Win
    # TODO: delete float(), which is required for 16-bit precision (if pytorch version > 1.7.1 it might be fixed)
    x_warped = torch.nn.functional.grid_sample(
        input=x.float(),
        grid=pxlcoords_normalized_perm.float(),
        mode=mode,
        align_corners=True,
        padding_mode=padding_mode,
    )
    # BxCxHxW

    # x_warped = x_warped * mask_flow_inside

    if dtype == torch.bool:
        x_warped = x_warped == 1.0

    if return_masks_flow_inside:
        return x_warped, mask_flow_inside
    else:
        return x_warped


def warp(x, flow, return_masks_flow_inside=False, mode="bilinear"):
    B, C, H, W = x.size()
    dtype = x.dtype
    device = x.device

    pxl2d = oflow_2_pxl2d(flow)
    # input: (B, C, Hin​, Win​) and grid: (B, 2, Hout​, Wout​)
    # output: (B, C, Hin, Win

    return interpolate2d(x, pxl2d, return_masks_flow_inside=return_masks_flow_inside, mode=mode)


# def warp3d(x, flow, disp, proj_mats, reproj_mats):
#    pt3d = o4geo_pinhole.disp_2_pt3d(disp, proj_mats, reproj_mats)
#    pt3d_ftf = pt3d + flow
#    pxl3d_ftf = o4geo_pinhole.pt3d_2_pxl2d(pt3d_ftf, proj_mats)
#
#    return interpolate2d(x, pxl3d_ftf, return_masks_flow_inside=False)
