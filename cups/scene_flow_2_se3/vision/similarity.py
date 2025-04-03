import torch

import cups.scene_flow_2_se3.vision.warp as o4warp


def calc_ddsim_per_pixel(x, y, keep_size=False):
    # patch size = 3x3
    # x = BxCxHxW
    B, C, H, W = x.shape

    c1 = 0.01**2
    c2 = 0.03**2
    # (c3 = c2 / 2)
    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = torch.nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = torch.nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    ssim_n = (2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d  # range [-1, 1]
    dssim = (1 - ssim) / 2  # range [ 0, 1]

    if keep_size:
        dssim = torch.nn.functional.pad(dssim, pad=(1, 1, 1, 1))

    return dssim.mean(dim=1, keepdim=True)


def oflow_2_mask_non_occl(flow_fwd, imgpair, max_ddsim=0.5):
    img1 = imgpair[:, :3]
    img2 = imgpair[:, 3:]

    img2_bwdwrpd, flow_inside = o4warp.warp(img2.clone(), flow_fwd.clone(), return_masks_flow_inside=True)

    # TODO: visualize backward warped img2

    ddsim = calc_ddsim_per_pixel(img1, img2_bwdwrpd, keep_size=True)

    return (ddsim < max_ddsim) * flow_inside


def disp_2_mask_non_occl(disp_fwd, imgpair, max_ddsim=0.5, fwd=True):
    B, C, H, W = disp_fwd.shape
    dtype = disp_fwd.dtype
    device = disp_fwd.device

    disp_fwd_oflow = torch.zeros(size=(B, 2, H, W), dtype=dtype, device=device)
    if fwd:
        disp_fwd_oflow[:, :1] = -disp_fwd[:, :1]
    else:
        disp_fwd_oflow[:, :1] = disp_fwd[:, :1]

    return oflow_2_mask_non_occl(disp_fwd_oflow, imgpair, max_ddsim)
