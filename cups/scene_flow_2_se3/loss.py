# from util.chamfer_distance import ChamferDistance
import pytorch3d.transforms as t3d
import torch

import cups.scene_flow_2_se3.differentiation as odiff
import cups.scene_flow_2_se3.geometric as ogeo
import cups.scene_flow_2_se3.rearrange as orearr
import cups.scene_flow_2_se3.vision.vision as ovis
import cups.scene_flow_2_se3.vision.warp as owarp


def census_transform(x, patch_size):
    """
    census transform:
    input: rgb image
    output: difference for each pixel to its neighbors 7x7
    1. rgb to gray: bxhxwxc -> bxhxwx1
    2. neighbor intensities as channels: bxhxwx1 -> bxhxwx7*7 (padding with zeros)
    3. difference calculation: L1 / sqrt(0.81 + L1^2): bxhxwx7*7 (coefficient from DDFlow)
    """

    # x: Bx3xHxW
    x = ovis.rgb_2_grayscale(x) * 255.0

    # x: Bx1xHxW
    x = orearr.neighbors_to_channels(x, patch_size=patch_size)

    # x: BxP^2xHxW - Bx1xHxW
    dist_per_pixel_per_neighbor = x - x[:, 24].unsqueeze(1)

    # L1: BxP^2xHxW
    dist_per_pixel_per_neighbor = dist_per_pixel_per_neighbor / torch.sqrt(0.81 + dist_per_pixel_per_neighbor**2)
    # neighbor_dist: BxP^2xHxW
    # neighbor_dist in [0, 1]

    return dist_per_pixel_per_neighbor


def soft_hamming_distance(x1, x2):
    """
    soft hamming distance:
    input: census transformed images bxhxwxk
    output: difference between census transforms per pixel
    1. difference calculation per pixel, per features: L2 / (0.1 + L2)
    2. summation over features: bxhxwxk -> bxhxwx1
    """

    # x1, x2: BxCxHxW

    squared_dist_per_pixel_per_feature = (x1 - x2) ** 2
    # squared_dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel_per_feature = squared_dist_per_pixel_per_feature / (0.1 + squared_dist_per_pixel_per_feature)
    # dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel = torch.sum(dist_per_pixel_per_feature, dim=1)
    # dist_per_pixel_per: BxHxW

    return dist_per_pixel


def calc_photo_loss(x1, x2, type="census", masks_flow_valid=None, fwdbwd=True):

    if not fwdbwd:
        B = x1.size(0) // 2
        x1 = x1[:B]
        x2 = x2[:B]
        if masks_flow_valid is not None:
            masks_flow_valid = masks_flow_valid[:B]

    if type == "census":
        return calc_census_loss(x1, x2, patch_size=7, masks_flow_valid=masks_flow_valid)

    elif type == "ssim":

        l1_per_pixel = calc_l1_per_pixel(x1, x2)
        dssim_per_pixel = calc_dssim_per_pixel(x1, x2)
        dssim_per_pixel = torch.nn.functional.pad(dssim_per_pixel, (1, 1, 1, 1), mode="constant", value=0)
        loss_per_pixel = (0.85 * dssim_per_pixel + 0.15 * l1_per_pixel).mean(dim=1, keepdim=True)

        loss_per_pixel = loss_per_pixel * masks_flow_valid

        return torch.sum(loss_per_pixel) / (torch.sum(masks_flow_valid) + 1e-8)


def calc_census_loss(x1, x2, patch_size, masks_flow_valid=None):
    """
    census loss:
    1. hamming distance from census transformed rgb images
    2. robust loss for per pixel hamming distance: (|diff|+0.01)^0.4   (as in DDFlow)
    3. per pixel multiplication with zero mask at border s.t. every loss value close to border = 0
    4. sum over all pixel and divide by number of pixel which were not zeroed out: sum(per_pixel_loss)/ (num_pixels + 1e-6)
    """

    dtype = x1.dtype
    device = x1.device

    # x1, x2: Bx3xHxW
    x1_census = census_transform(x1, patch_size)
    x2_census = census_transform(x2, patch_size)
    # x1_census, x2_census: Bxpatch_size^2xHxW

    soft_hamming_dist_per_pixel = soft_hamming_distance(x1_census, x2_census)
    # soft_hamming_dist: BxHxW

    robust_soft_hamming_dist_per_pixel = (soft_hamming_dist_per_pixel + 0.01) ** (0.4)
    # robust_soft_hamming_dist_per_pixel: BxHxW

    masks_valid_pixel = torch.zeros((robust_soft_hamming_dist_per_pixel.size()), dtype=dtype, device=device)

    pad = int((patch_size - 1) / 2)

    if masks_flow_valid is not None:
        masks_valid_pixel[:, pad:-pad, pad:-pad] = masks_flow_valid[:, 0, pad:-pad, pad:-pad]
    else:
        masks_valid_pixel[:, pad:-pad, pad:-pad] = 1.0

    # mask = mask.repeat(robust_soft_hamming_dist_per_pixel.size(0), 1, 1)
    # mask: BxHxW
    #  # * valid_warp_mask: adds if warping is outside of frame

    valid_pixel_mask_total_weight = torch.sum(masks_valid_pixel, dim=(0, 1, 2))

    # q: why does uflow stop gradient computation for mask in mask_total_weight, but not for mask in general?

    return torch.sum(robust_soft_hamming_dist_per_pixel * masks_valid_pixel, dim=(0, 1, 2)) / (
        valid_pixel_mask_total_weight + 1e-6
    )


def calc_fb_consistency(flow_forward, flow_backward, fb_sigma, masks_flow_inside_forward=None):
    # in : Bx2xHxW
    # out: Bx1xHxW
    H = flow_forward.size(2)
    W = flow_forward.size(3)

    if masks_flow_inside_forward == None:
        flow_backward_warped, masks_flow_inside_forward = owarp.warp(
            flow_backward, flow_forward, return_masks_flow_inside=True
        )
    else:
        flow_backward_warped = owarp.warp(flow_backward, flow_forward, return_masks_flow_inside=False)

    flows_diff_squared = torch.sum((flow_forward + flow_backward_warped) ** 2, dim=1, keepdim=True)

    fb_consistency = torch.exp(-(flows_diff_squared / (fb_sigma**2 * (H**2 + W**2))))

    fb_consistency = fb_consistency * masks_flow_inside_forward
    # flows_squared_sum = torch.sum((flow_forward ** 2 + flow_backward ** 2), dim=1, keepdim=True)

    return fb_consistency


def calc_selfsup_loss(
    student_flow_forward,
    student_flow_backward,
    teacher_flow_forward,
    teacher_flow_backward,
    crop_reduction_size=16,
):

    B, _, H, W = student_flow_forward.shape

    teacher_fb_sigma = 0.003
    student_fb_sigma = 0.03

    student_fb_consistency_forward = calc_fb_consistency(student_flow_forward, student_flow_backward, student_fb_sigma)
    teacher_fb_consistency_forward = calc_fb_consistency(teacher_flow_forward, teacher_flow_backward, teacher_fb_sigma)
    teacher_fb_consistency_forward = teacher_fb_consistency_forward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]
    teacher_fb_consistency_forward = torch.nn.functional.interpolate(
        teacher_fb_consistency_forward,
        size=(H, W),
        mode="bilinear",
        align_corners=True,
    )

    student_fb_consistency_backward = calc_fb_consistency(student_flow_backward, student_flow_forward, student_fb_sigma)
    teacher_fb_consistency_backward = calc_fb_consistency(teacher_flow_backward, teacher_flow_forward, teacher_fb_sigma)

    teacher_fb_consistency_backward = teacher_fb_consistency_backward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]
    teacher_fb_consistency_backward = torch.nn.functional.interpolate(
        teacher_fb_consistency_backward,
        size=(H, W),
        mode="bilinear",
        align_corners=True,
    )
    # Bx1xHxW

    teacher_flow_forward = teacher_flow_forward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]

    teacher_flow_forward = torch.nn.functional.interpolate(
        teacher_flow_forward, size=(H, W), mode="bilinear", align_corners=True
    )

    teacher_flow_forward[:, 0] = teacher_flow_forward[:, 0] * W / (W - 2 * crop_reduction_size)
    teacher_flow_forward[:, 1] = teacher_flow_forward[:, 1] * H / (H - 2 * crop_reduction_size)

    teacher_flow_backward = teacher_flow_backward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]

    teacher_flow_backward[:, 0] = teacher_flow_backward[:, 0] * W / (W - 2 * crop_reduction_size)
    teacher_flow_backward[:, 1] = teacher_flow_backward[:, 1] * H / (H - 2 * crop_reduction_size)

    teacher_flow_backward = torch.nn.functional.interpolate(
        teacher_flow_backward, size=(H, W), mode="bilinear", align_corners=True
    )

    # valid_warp_mask adds mask for if warping is outside of frame
    weights_forward = (1.0 - student_fb_consistency_forward) * (
        teacher_fb_consistency_forward
    )  # * valid_warp_mask (forward)
    weights_forward = weights_forward.detach()

    weights_backward = (1.0 - student_fb_consistency_backward) * (
        teacher_fb_consistency_backward
    )  # * valid_warp_mask (forward)
    weights_backward = weights_backward.detach()
    # weights.requires_grad = False
    # Bx1xHxW

    teacher_flow_forward = teacher_flow_forward.detach()
    teacher_flow_backward = teacher_flow_backward.detach()
    selfsup_loss_forward = calc_charbonnier_loss(student_flow_forward, teacher_flow_forward, weights_forward)
    selfsup_loss_backward = calc_charbonnier_loss(student_flow_backward, teacher_flow_backward, weights_backward)

    return (selfsup_loss_forward + selfsup_loss_backward) / 2.0


def calc_l1_loss(x1, x2):
    return torch.sum(torch.abs(x1 - x2)) / x1.size(0)


def calc_l1_per_pixel(x1, x2):
    return torch.norm(x1 - x2, p=1, dim=1, keepdim=True)


def calc_charbonnier_loss(x1, x2, weights):
    # x1, x2: BxCxHxW
    # weights: Bx1xHxW
    # return torch.sum((((x1 - x2) ** 2 + 0.001 ** 2) ** 0.5) * weights) / (torch.sum(weights) + 1e-16)

    return torch.sum((((x1 - x2) ** 2 + 0.001**2) ** 0.5) * weights) / (
        weights.size(0) * weights.size(2) * weights.size(3) + 1e-16
    )


def calc_dssim_per_pixel(x, y):
    # patch size = 3x3
    # x = BxCxHxW
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

    return dssim


def calc_mask_consistency_oflow_loss(
    pts3d,
    masks,
    se3s_mat,
    oflow_gt,
    proj_mat,
    reproj_mat,
    masks_valid,
    loss_cross3d_score_const_weight,
    loss_cross3d_score_linear_weight,
    loss_cross3d_score_exp_weight,
    loss_cross3d_score_exp_slope,
    loss_cross3d_outlier_slope,
    loss_cross3d_outlier_min,
    loss_cross3d_outlier_max,
    loss_cross3d_max,
    loss_cross3d_min,
    egomotion_addition=True,
    visualize=False,
    vwriter=None,
):
    """
    loss_cross3d_score_const_weight: 1.0
    loss_cross3d_score_linear_weight: 10.0
    loss_cross3d_score_exp_weight: 10.0
    loss_cross3d_score_exp_slope: 500.0
    loss_cross3d_outlier_slope: 0.1
    loss_cross3d_outlier_min: 0.1
    loss_cross3d_outlier_max: 0.9
    loss_cross3d_max: 0.1
    loss_cross3d_min: 0.001
    """
    masks_detached = masks.detach()
    B, K, H, W = masks.shape
    losses_single_pxl = []
    losses_single = []
    for k in range(K):

        if k == 0:
            masks_single_bin = 1.0 * (masks[:, :1] >= 0.0)  # >= min_prob)
            se3s_mat_single = se3s_mat[:, :1]
        else:
            if egomotion_addition:
                masks_single_bin = torch.cat((masks[:, :1] > 1.0, masks[:, k : k + 1] >= 0.0), dim=1)
                se3s_mat_single = se3s_mat[:, [0, k]]
                se3s_mat_single[:, 0] = se3s_mat_single[:, 0].clone()
                se3s_mat_single[:, 0] = se3s_mat_single[:, 0].detach()
            else:
                masks_single_bin = 1.0 * (masks[:, k : k + 1] >= 0.0)  # >= min_prob)
                se3s_mat_single = se3s_mat[
                    :,
                    k : k + 1,
                ]

        pts3d_ftf_bin = ogeo.pts3d_transform_obj_ego(
            pts3d.detach(),
            se3mat=se3s_mat_single,
            mask=masks_single_bin,
            egomotion_addition=egomotion_addition,
        )

        loss_single, loss_single_pxl = calc_consistency_oflow_corr3d_loss(
            pts3d_ftf_bin,
            oflow_gt,
            masks_valid,
            proj_mat,
            reproj_mat,
        )

        loss_single_pxl = loss_single_pxl.detach()
        losses_single_pxl.append(loss_single_pxl)
        losses_single.append(loss_single.detach())

    # prob_masks = torch.softmax(torch.cat(scores, dim=1), dim=1)
    # scores = torch.cat(scores, dim=1)

    losses_single = torch.stack(losses_single, dim=1)
    losses_single_pxl = torch.cat(losses_single_pxl, dim=1)
    masks_indices_arg_max = torch.argmax(masks, dim=1, keepdim=True)

    loss_single_pxl_max_mask = torch.gather(input=losses_single_pxl, index=masks_indices_arg_max, dim=1)

    # loss_single_pxl_max_mask_var, loss_single_pxl_max_mask_avg = torch.var_mean(
    #    loss_single_pxl_max_mask, dim=(2, 3), keepdim=True
    # )
    losses_single_avg = torch.sum(losses_single_pxl * masks_detached, dim=(2, 3), keepdim=True) / torch.sum(
        masks_detached, dim=(2, 3), keepdim=True
    )

    # losses_single_pxl_sqdev = (
    #    torch.clip(losses_single_pxl - losses_single_avg, min=0.0) ** 2
    # )

    losses_single_pxl_sqdev = losses_single_pxl**2

    losses_single_var = torch.sum(losses_single_pxl_sqdev * masks_detached, dim=(2, 3), keepdim=True) / torch.sum(
        masks_detached, dim=(2, 3), keepdim=True
    )
    # losses_single_var = torch.clamp(losses_single_var, 0.0, 1e-4)
    # print("losses_single_var", losses_single_var)
    pxl_outlier_max_mask = torch.clamp(
        1.0 - torch.exp(-losses_single_pxl_sqdev / (2 * losses_single_var) * loss_cross3d_outlier_slope),
        loss_cross3d_outlier_min,
        loss_cross3d_outlier_max,
    )
    # print("")
    # print("var", loss_single_pxl_max_mask_var)
    # print("avg", loss_single_pxl_max_mask_avg)
    # print("")
    # avg: 0.0526 -> 0.0115 (exp(-(x-0.01)* 100)):  0.05 -> 1.0
    # var: 0.0019 -> 0.0006
    # uncertainty_loss_min,
    # uncertainty_mask_slope,
    # uncertainty_pxl_slope,
    """
    loss_single_var_max_mask = torch.gather(
        input=losses_single_mask_avg, index=masks_indices_arg_max, dim=1
    )

    pxl_uncertainty_max_mask = torch.clamp(
        1.0
        - torch.clamp(
            loss_single_avg_max_mask / loss_single_pxl_max_mask * 2, 0.0, 1.0
        ),
        # / -torch.exp(
        #    -(losses_single_mask_avg - uncertainty_loss_min) * uncertainty_mask_slope
        # )
        # * (
        #    1.0
        #    - torch.exp(
        #        -(loss_single_pxl_max_mask - uncertainty_loss_min)
        #        * uncertainty_pxl_slope
        #    )
        # ),
        0.0,
        1.0,
    )
    """

    pxl_uncertainty_max_mask = torch.gather(
        input=pxl_outlier_max_mask,  # torch.max(pxl_outlier_max_mask, avg_uncertainty_max_mask),
        index=masks_indices_arg_max,
        dim=1,
    )

    # pxl_outlier_max_mask = torch.gather(
    #    input=pxl_outlier_max_mask,
    #    index=masks_indices_arg_max,
    #    dim=1,
    # )

    pxl_uncertainty_masks = pxl_uncertainty_max_mask.repeat(1, K, 1, 1)

    masks_indices_arg_max = masks_indices_arg_max == torch.arange(0, K).to(masks.device).reshape(1, K, 1, 1)
    pxl_uncertainty_masks[masks_indices_arg_max] = 1.0 - pxl_uncertainty_masks[masks_indices_arg_max]

    losses_single_pxl = torch.clamp(
        losses_single_pxl,
        min=loss_cross3d_min + 1e-8,
        max=loss_cross3d_max - 1e-8,
    )
    # print("losses_single", losses_single)
    scores = (
        (
            # 1.0 / (losses_single_pxl)
            # - 1 / uncertainty_loss_max
            torch.exp(-(losses_single_pxl - loss_cross3d_min) * loss_cross3d_score_exp_slope)
            * loss_cross3d_score_exp_weight
            + (1 - losses_single_pxl / loss_cross3d_max) * loss_cross3d_score_linear_weight
            + loss_cross3d_score_const_weight
        )
        # * torch.clamp(
        #    torch.exp(
        #        -losses_single[:, :, None, None] * 30,
        #    ),
        #    loss_cross3d_outlier_min,
        #    loss_cross3d_outlier_max,
        # )
        * (pxl_uncertainty_masks)
    )

    # scores = -losses_single_pxl * 100 * (pxl_uncertainty_masks)
    # scores = torch.clamp(
    #    -torch.log(losses_single_pxl * 30) * pxl_uncertainty_masks, min=0.01
    # )
    # scores = -losses_single_pxl * 1000 * pxl_uncertainty_masks
    # prob_masks = torch.softmax(scores, dim=1)

    prob_masks = (scores / torch.sum(scores, keepdim=True, dim=1)) / 1.0

    #  0.0012 -> 0.0191
    #  0.005 -> 0.1  -> factor: 200 , 10
    # var_masks = 10 * (
    #    torch.var(prob_masks.flatten(2), dim=2).sum(dim=1)[:, None, None, None] + 0.01
    # )
    # print("var", var_masks)

    # entropy = (-torch.sum(masks * torch.log(masks), dim=1)).mean()
    # print("entropy", entropy)
    loss_mask = (
        torch.norm(
            (masks - prob_masks),
            dim=1,
            p=2,
            keepdim=True,
        )
        * masks_valid
    ).sum() / (torch.sum(masks_valid) + 1e-8)
    """Scores = torch.clamp(1.0 / losses_single_pxl, min=0, max=1.0 / uncertainty_loss_min)

    prob_masks = (scores / torch.sum(scores, keepdim=True, dim=1) + 0.01) / 1.01
    """

    if visualize:
        masks_detached_bin = masks_detached.clone()
        masks_detached_bin[masks_indices_arg_max] = 1.0
        masks_detached_bin[~masks_indices_arg_max] = 0.0

        pts3d_ftf_bin = ogeo.pts3d_transform_obj_ego(
            pts3d.detach(),
            se3mat=se3s_mat,
            mask=masks_detached_bin,
            egomotion_addition=egomotion_addition,
        )

        flow_max = ogeo.pt3d_2_oflow(pts3d_ftf_bin, proj_mats=proj_mat)[0]

        ovis.visualize_hist(losses_single_pxl)

        ovis.visualize_img(
            torch.cat(
                (
                    # ovis.flow2rgb(oflow_gt[0], draw_arrows=True),
                    # ovis.flow2rgb(flow0, draw_arrows=True),
                    ovis.mask2rgb(masks[0]) * masks_valid[0],
                    ovis.flow2rgb(flow_max, draw_arrows=True),
                    pxl_uncertainty_max_mask.repeat(1, 3, 1, 1)[0] * masks_valid[0],
                    # pxl_outlier_max_mask.repeat(1, 3, 1, 1)[0] * masks_valid[0],
                    # ovis.mask2rgb(masks_indices_arg_max[0] * 1.0),
                    ovis.mask2rgb(prob_masks[0]) * masks_valid[0],
                ),
                dim=1,
            ),
            duration=1,
            vwriter=vwriter,
        )
    return loss_mask, prob_masks


def calc_consistency_oflow_corr3d_single_loss(
    pts3d,
    masks,
    se3s_mat,
    oflow_gt,
    proj_mat,
    reproj_mat,
    masks_valid,
    egomotion_addition=True,
):

    _, K, _, _ = masks.shape
    # masks = prob_masks
    loss_motion = None
    for k in range(K):
        if k == 0:
            masks_single = 1.0 * (masks[:, :1])  # >= min_prob)
            masks_single_bin = 1.0 * (masks[:, :1] >= 0.0)  # >= min_prob)
            se3s_mat_single = se3s_mat[:, :1]
        else:
            if egomotion_addition:
                masks_single = 1.0 * (masks[:, [0, k]])  # >= min_prob)
                masks_single_bin = torch.cat((masks[:, :1] > 1.0, masks[:, k : k + 1] >= 0.0), dim=1)
                se3s_mat_single = se3s_mat[:, [0, k]]
                se3s_mat_single[:, 0] = se3s_mat_single[:, 0].clone()
                se3s_mat_single[:, 0] = se3s_mat_single[:, 0].detach()
            else:
                masks_single = 1.0 * (masks[:, k : k + 1])  # >= min_prob)
                masks_single_bin = 1.0 * (masks[:, k : k + 1] >= 0.0)
                se3s_mat_single = se3s_mat[
                    :,
                    k : k + 1,
                ]

        pts3d_ftf = ogeo.pts3d_transform_obj_ego(
            pts3d.detach(),
            se3mat=se3s_mat_single,
            mask=masks_single_bin,  # > 0.5,
            egomotion_addition=egomotion_addition,
        )

        loss_single, _ = calc_consistency_oflow_corr3d_loss(
            pts3d_ftf,
            oflow_gt,
            masks_valid * masks[:, k : k + 1].detach(),
            proj_mat,
            reproj_mat,
        )
        loss_single = loss_single.mean()

        if k == 0:
            loss_motion = loss_single
        else:
            loss_motion += loss_single

    return loss_motion


"""
def calc_photo_single(pts3d, masks, se3s_mat, imgs1, imgs2):

    B, K, H, W = masks.shape

    for k in range(K):
        if k == 0:
            masks_single = 1. * (masks[:, :1] > 0.3)
            se3s_mat_single = se3s_mat[:, :1]
        else:
            masks_single = masks[:, [0, k]]
            se3s_mat_single = se3s_mat[:, [0, k]]

        pts3d_ftf = ogeo.pts3d_transform_obj_ego(
            pts3d.detach(),
            se3mat=se3s_mat_single,
            mask=masks_single)

        pxl2d_ftf = ogeo.pt3d_2_pxl2d(pts3d_ftf, proj_mats=proj_mats)

        imgs2_bwrpd, masks_l1_se3_ftf_inside = owarp.interpolate2d(
            imgs2,
            pxl2d_ftf,
            return_masks_flow_inside=True,
        )
        masks_l1_se3_ftf_valid = masks_l1_se3_ftf_inside * masks_flow_valid_l1

        calc_photo_loss(
            imgs_l1_rs,
            imgs_l2_rs_disp_se3_bwrpd,
            masks_flow_valid=masks_l1_se3_ftf_valid,
            type=self.args.loss_photo_type,
            fwdbwd=self.args.loss_disp_se3_photo_fwdbwd
        )
"""


def calc_consistency_oflow_corr3d_loss(pts3d_pred, oflow_gt, mask_valid, proj_mat, reproj_mat):
    B, _, H, W = pts3d_pred.shape
    # oflow_pred = ogeo.pt3d_2_oflow(pts3d_pred, proj_mats=proj_mat)
    vec3d_gt = ogeo.oflow_2_vec3d(oflow_gt, reproj_mats=reproj_mat)
    # vec3d_pred = ogeo.oflow_2_vec3d(oflow_pred, reproj_mats=reproj_mat)
    vec3d_pred = pts3d_pred

    vec3d_pred = vec3d_pred / (torch.norm(pts3d_pred, dim=1, keepdim=True) + 1e-8)
    loss_corr_pxl = torch.norm(torch.cross(vec3d_gt, vec3d_pred, dim=1), dim=1, p=2, keepdim=True)
    loss_corr = torch.sum(loss_corr_pxl * (mask_valid**2), dim=[1, 2, 3]) / (
        torch.sum(mask_valid**2, dim=[1, 2, 3]) + 1e-8
    )  #  torch.sum(mask_valid) (B * H * W + 1e-8)

    return loss_corr, loss_corr_pxl


def calc_mask_reg_loss(masks):
    # masks: BxKxHxW
    B, K, H, W = masks.shape
    area = H * W
    masks = masks.flatten(2)
    loss = -torch.log(torch.sum(masks, dim=2) / area).mean()

    return loss


def calc_consistency_mask_loss(masks1, pxls2d_1_ftf, masks_valid, fwdbwd=True):
    # forward_transformed
    # backward_warped
    # x1, x2: B x 3 x H x W

    # reorder
    masks2 = orearr.batches_swap_order(masks1)

    if not fwdbwd:
        B = masks1.size(0)
        masks1 = masks1[:B]
        masks2 = masks2[:B]
        pxls2d_1_ftf = pxls2d_1_ftf[:B]
        masks_valid = masks_valid[:B]

    masks2_bwrpd = owarp.interpolate2d(
        masks2,
        pxls2d_1_ftf,
        return_masks_flow_inside=False,
    )

    res = masks1 - masks2_bwrpd
    epe = torch.norm(res, p=2, dim=1, keepdim=True)

    cons_loss = torch.sum((masks_valid * epe)) / (torch.sum(masks_valid) + 1e-8)  # / (pts1_norm + 1e-8)) / (

    return cons_loss


def calc_consistency3d_loss(pts1_norm, pts1_ftf, pts1, pxlcoords1_ftf, mask_valid, type="smsf", fwdbwd=True):
    # forward_transformed
    # backward_warped
    # x1, x2: B x 3 x H x W

    # reorder
    pts2 = orearr.batches_swap_order(pts1)

    if not fwdbwd:
        B = pts1.size(0)
        pts1_norm = pts1_norm[:B]
        pts1 = pts1[:B]
        pts2 = pts2[:B]
        pts1_ftf = pts1_ftf[:B]
        pxlcoords1_ftf = pxlcoords1_ftf[:B]
        mask_valid = mask_valid[:B]

    if type == "smsf":
        pts2_bwrpd = owarp.interpolate2d(
            pts2,
            pxlcoords1_ftf,
            return_masks_flow_inside=False,
        )

        res = pts1_ftf - pts2_bwrpd
        epe = torch.norm(res, p=2, dim=1, keepdim=True)

        # * 2.0 because forward and backward reconstruction are summed in self-mono-sf
        rec_loss = torch.sum((mask_valid * epe / (pts1_norm + 1e-8))) / (torch.sum(mask_valid) + 1e-8)

    elif type == "chamfer":
        rec_loss = calc_chamfer_loss(pts1_ftf, pts2, mask_valid)

    return rec_loss


def calc_chamfer_dist(pts1, pts2):
    # pts1, pts2: Bx3xHxW
    # patch_size = kernel_size
    B, C, H, W = pts1.shape
    kernel_size = int(W * 0.3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(kernel_size)
    pts1_neighbors = orearr.neighbors_to_channels(pts1, patch_size=kernel_size).reshape(B, C, -1, H, W)
    dist, _ = torch.min(torch.norm(pts1_neighbors - pts2.unsqueeze(2), dim=1, keepdim=True), dim=2)
    # BxK*CxHxW
    return dist


def calc_chamfer_loss(
    pts1,
    pts2,
    masks_valid=None,
    img=None,
    edge_weight=150,
    order=2,
    lambda_smoothness=0.1,
):
    # B x 3 x H x W
    B, C, H, W = pts1.shape
    # chamfer_dist = ChamferDistance()
    chamfer_dist = 0
    # ...
    # points and points_reconstructed are n_points x 3 matrices
    pts1 = pts1.flatten(2).permute(0, 2, 1)
    pts2 = pts2.flatten(2).permute(0, 2, 1)
    # Bx N x 3
    dist1, dist2 = chamfer_dist(pts1, pts2)
    dist1 = dist1.reshape(B, 1, H, W)
    dist2 = dist2.reshape(B, 1, H, W)

    if masks_valid is not None:
        dist1 = dist1 * masks_valid
        dist2 = dist2 * masks_valid
    chamfer_loss = (torch.mean(dist1) + torch.mean(dist2)) / 2.0

    if img is not None:
        smoothness_loss = (
            calc_smoothness_loss(dist1, img, edge_weight=edge_weight, order=order, smooth_type="smsf")
            + calc_smoothness_loss(dist2, img, edge_weight=edge_weight, order=order, smooth_type="smsf")
        ) / 2.0

        # print('chamfer-dist', chamfer_loss)
        # print('chamfer-smooth', lambda_smoothness * smoothness_loss)
        chamfer_loss += lambda_smoothness * smoothness_loss

    return chamfer_loss


def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2) ** 0.5


def calc_smoothness_loss(flow, img1, edge_weight=150, order=1, weights_inv=None, smooth_type="uflow"):
    # flow: torch.tensor: Bx2xHxW
    # img1: torch.tensor: Bx3xHxW

    margin = 0
    flow_k_grad_x, flow_k_grad_y = odiff.calc_batch_gradients(flow)
    for k in range(order - 1):
        flow_k_grad_x, _ = odiff.calc_batch_gradients(flow_k_grad_x)
        _, flow_k_grad_y = odiff.calc_batch_gradients(flow_k_grad_y)

    if smooth_type == "uflow":
        # Bx2xHxW-(order * 2)
        flow_k_grad_x = robust_l1(flow_k_grad_x)
        flow_k_grad_y = robust_l1(flow_k_grad_y)

        img1_grad_x, img1_grad_y = odiff.calc_batch_gradients(img1, margin=order - 1)
    elif smooth_type == "smsf":
        flow_k_grad_x = torch.abs(flow_k_grad_x)
        flow_k_grad_y = torch.abs(flow_k_grad_y)

        img1_grad_x, img1_grad_y = odiff.calc_batch_gradients(img1, margin=0)
        img1_grad_x = img1_grad_x[:, :, :, order - 1 :]
        img1_grad_y = img1_grad_y[:, :, order - 1 :, :]
    else:
        print("error: unknown smooth-type: ", smooth_type)

    img1_grad_x = torch.abs(img1_grad_x)
    # Bx3xHxW-1
    img1_grad_y = torch.abs(img1_grad_y)
    # Bx3xH-1xW

    if weights_inv is not None:
        weights_inv_x = weights_inv[:, :, :, 1:-1]
        weights_inv_y = weights_inv[:, :, 1:-1, :]

        loss = (
            torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_x, dim=1, keepdim=True)) * flow_k_grad_x / weights_inv_x
            )
            + torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_y, dim=1, keepdim=True)) * flow_k_grad_y / weights_inv_y
            )
        ) / 2.0

    else:
        loss = (
            torch.mean(torch.exp(-edge_weight * torch.mean(img1_grad_x, dim=1, keepdim=True)) * flow_k_grad_x)
            + torch.mean(torch.exp(-edge_weight * torch.mean(img1_grad_y, dim=1, keepdim=True)) * flow_k_grad_y)
        ) / 2.0

    return loss


def dist_transls(transl1, transl2):

    dist = torch.norm((transl1 - transl2), dim=-1)

    return dist


def angle_rots(rot1, rot2):

    rot = torch.matmul(rot1.permute(0, 2, 1), rot2)

    rot_logs = t3d.so3_log_map(rot[:, :3, :3])
    angle = torch.norm(rot_logs, dim=1)

    return angle


def dist_angle_transfs(transf1, transf2):

    rot1 = transf1[:, :3, :3]
    transl1 = transf1[:, :3, 3]

    rot2 = transf2[:, :3, :3]
    transl2 = transf2[:, :3, 3]

    angle = angle_rots(rot1, rot2)
    dist = dist_transls(transl1, transl2)

    return dist, angle


def calc_disp_outlier_percentage(disps_pred, disps_gt, masks_disps_gt):
    diff_disps = torch.abs(disps_pred[masks_disps_gt] - disps_gt[masks_disps_gt])
    disp_outlier_percentage = (
        100
        * torch.sum((diff_disps >= 3.0) * ((diff_disps / disps_gt[masks_disps_gt]) >= 0.05))
        / torch.sum(masks_disps_gt)
    )

    return disp_outlier_percentage


def calc_flow_outlier_percentage(flows_pred, flows_gt, masks_flows_gt):
    masks_flows_gt = masks_flows_gt.unsqueeze(0)
    diff_flows = torch.norm(flows_pred - flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    norm_flows_gt = torch.norm(flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    flow_outlier_percentage = (
        100 * torch.sum((diff_flows >= 3.0) * ((diff_flows / norm_flows_gt) >= 0.05)) / torch.sum(masks_flows_gt)
    )

    return flow_outlier_percentage


def calc_sflow_outlier_percentage(
    disps0_pred,
    disps0_gt,
    masks_disps0_gt,
    disps1_pred,
    disps1_gt,
    masks_disps1_gt,
    flows_pred,
    flows_gt,
    masks_flows_gt,
):
    masks_flows_gt = masks_flows_gt.unsqueeze(0)
    diff_flows = torch.norm(flows_pred - flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    norm_flows_gt = torch.norm(flows_gt, dim=1, keepdim=True)[masks_flows_gt]

    diff_disps0 = torch.abs(disps0_pred[masks_disps0_gt] - disps0_gt[masks_disps0_gt])
    diff_disps1 = torch.abs(disps1_pred[masks_disps1_gt] - disps1_gt[masks_disps1_gt])

    flows_outls = (diff_flows >= 3.0) * ((diff_flows / norm_flows_gt) >= 0.05)
    disps0_outls = (diff_disps0 >= 3.0) * ((diff_disps0 / disps0_gt[masks_disps0_gt]) >= 0.05)
    disps1_outls = (diff_disps0 >= 3.0) * ((diff_disps0 / disps0_gt[masks_disps0_gt]) >= 0.05)

    sflow_outlier_percentage = (
        100
        * torch.sum(flows_outls | disps0_outls | disps1_outls)
        / torch.sum(masks_flows_gt | masks_disps0_gt | masks_disps1_gt)
    )

    return sflow_outlier_percentage
