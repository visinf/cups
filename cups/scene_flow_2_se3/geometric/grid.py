import torch

import cups.scene_flow_2_se3.vision.resize as o4res


def shape_2_pxl2d(B, H, W, dtype, device):
    grid_v, grid_u = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ],
    )

    grid_uv = torch.stack((grid_u, grid_v), dim=0)
    # 2xHxW

    if B != 0:
        grid_uv = grid_uv.unsqueeze(0).repeat(repeats=(B, 1, 1, 1))
        # Bx2xHxW

    return grid_uv


def shape_2_pxl2d_normalized(B, H, W, dtype, device):
    grid_uv = shape_2_pxl2d(B, H, W, dtype, device)
    grid_uv_normalized = pxl2d_2_pxl2d_normalized(grid_uv)

    return grid_uv_normalized


def oflow_2_pxl2d(flow, orig_H=None, orig_W=None, resize_mode="bilinear"):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    if orig_H is None or orig_W is None:
        orig_H = H
        orig_W = W

    grid_uv = shape_2_pxl2d(B=B, H=orig_H, W=orig_W, dtype=dtype, device=device)

    grid_uv = o4res.resize(
        grid_uv,
        H_out=H,
        W_out=W,
        mode=resize_mode,
    )

    grid_uv = grid_uv + flow

    return grid_uv


def pxl2d_normalized_2_mask_inside(pxl2d_normalized):
    B, C, H, W = pxl2d_normalized.size()
    dtype = pxl2d_normalized.dtype
    device = pxl2d_normalized.device

    pxl2d_normalized = pxl2d_normalized.permute(0, 2, 3, 1)
    # B x H x W x C
    mask_valid_flow = torch.ones(size=(B, H, W), dtype=dtype, device=device)
    mask_valid_flow[pxl2d_normalized[:, :, :, 0] > 1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 0] < -1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 1] > 1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 1] < -1.0] = 0.0
    mask_valid_flow = mask_valid_flow.unsqueeze(1)
    # Bx1xHxW

    mask_valid_flow = mask_valid_flow == 1.0

    return mask_valid_flow


def pxl2d_2_pxl2d_normalized(grid_xy, H=None, W=None):
    # ensure normalize pxlcoords is no inplace
    grid_xy = grid_xy.clone()
    if H is None or W is None:
        B, C, H, W = grid_xy.shape

    grid_xy[:, 0] = grid_xy[:, 0] / (W - 1.0) * 2.0 - 1.0
    grid_xy[:, 1] = grid_xy[:, 1] / (H - 1.0) * 2.0 - 1.0

    return grid_xy
