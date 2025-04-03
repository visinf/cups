from __future__ import annotations

import math
from argparse import Namespace

import torch
from kornia.filters import canny
from kornia.morphology import dilation
from torch import Tensor

from cups.optical_flow.utils import backward_warp_flow, estimate_occlusions
from cups.scene_flow_2_se3.connectec_components import connected_components
from cups.scene_flow_2_se3.geometric.pinhole import (
    depth_2_pt3d,
    disp_2_depth,
    oflow_2_mask_valid,
)
from cups.scene_flow_2_se3.probabilistic.models.gaussian import estimate_std
from cups.scene_flow_2_se3.retrieval.sflow2se3.proposal_selection import (
    sflow2se3,
    sflow2se3_vanilla,
)
from cups.scene_flow_2_se3.utils import omit_too_small_object_proposals, remap_ids
from cups.scene_flow_2_se3.vision.similarity import oflow_2_mask_non_occl
from cups.scene_flow_2_se3.vision.warp import warp

default_arguments = Namespace(
    sflow2se3_downscale_factor=0.06,
    sflow2se3_downscale_mode="nearest_v2",
    sflow2se3_rigid_dist_dev_max=0.03,
    eval_visualize_paper=False,
    sflow2se3_depth_min=5.5,
    sflow2se3_depth_max=70.0,
    sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std=0.05,
    sflow2se3_model_inlier_hard_threshold=0.0455,
    sflow2se3_se3filter_prob_gain_min=0.0015,
    sflow2se3_se3filter_prob_same_mask_max=0.15,
    sflow2se3_oflow_disp_std_abs_max=10.0,
    sflow2se3_oflow_disp_std_abs_min=2.0,
    sflow2se3_oflow_disp_trusted_perc=0.5,
    sflow2se3_oflow_disp_std_valid_min_perc=0.1,
    sflow2se3_occl_warp_dssim_max=0.3,
    sflow2se3_background_th=0.02,
    sflow2se3_min_object_size=0.65,
    sflow2se3_oflow_disp_std_correct_truncation=True,
    sflow2se3_model_euclidean_nn_rel_depth_dev_std=None,  # Will be computed
    sflow2se3_model_se3_likelihood_disp_abs_std=None,  # Will be computed
    sflow2se3_model_se3_likelihood_oflow_abs_std=None,  # Will be computed
)


def sf2se3(
    image_1_l: Tensor,
    optical_flow_l_forward: Tensor,
    optical_flow_l_backward: Tensor,
    disparity_1_forward: Tensor,
    disparity_2_forward: Tensor,
    disparity_1_backward: Tensor,
    disparity_2_backward: Tensor,
    intrinsics: Tensor,
    baseline: Tensor,
    valid_pixels: Tensor | None = None,
    runs: int = 8,
) -> Tensor:
    """Function extracts SE(3)-motions from a given optical flow map and disparity maps.

    Args:
        image_1_l (Tensor): First left image of the shape [1, 3, H, W].
        optical_flow_l_forward (Tensor): Forward optical flow of the left images, shape is [1, 2, H, W].
        optical_flow_l_backward (Tensor): Backward optical flow of the left images, shape is [1, 2, H, W].
        disparity_1_forward (Tensor): Disparity (left to right, forward) of the first images, shape is [1, 1, H, W].
        disparity_2_forward (Tensor): Disparity (left to right, forward) of the second images, shape is [1, 1, H, W].
        disparity_1_backward (Tensor): Disparity (right to left, backward) of the first images, shape is [1, 1, H, W].
        disparity_1_backward (Tensor): Disparity (right to left, backward) of the second images, shape is [1, 1, H, W].
        intrinsics (Tensor): Camera intrinsics of the shape [1, 3, 3].
        baseline (Tensor): Stereo camera baseline of the shape [1, 1].
        valid_pixels (Tensor | None): Optional argument of valid pixels of the shape [1, 1, H, W].

    Returns:
        object_proposals_final (Tensor): Object proposals of the shape [H, W].
    """
    # Compute re-projection matrix
    reprojection = intrinsics.float().inverse().to(image_1_l.dtype)
    # Compute parameters
    default_arguments.sflow2se3_model_euclidean_nn_rel_depth_dev_std = (
        torch.tan(torch.tensor([89.999999 / 360.0 * 2 * torch.pi], device=intrinsics.device))
        * default_arguments.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std
        * image_1_l.shape[-1]
        / intrinsics[0, 0, 0]
    ).item()
    # Estimate std of disparity
    disparity_1_backward_wrap, valid_pixels_disparity_1_backward_wrap = backward_warp_flow(
        disparity_1_backward,
        torch.cat((-disparity_1_forward, torch.zeros_like(disparity_1_forward)), dim=1),
        normalized_coordinates=False,
        return_valid_pixels=True,
        padding_mode="zeros",
    )
    disparity_diff = (disparity_1_forward - disparity_1_backward_wrap).abs()
    default_arguments.sflow2se3_model_se3_likelihood_disp_abs_std = estimate_std(
        disparity_diff,
        valid=(
            valid_pixels_disparity_1_backward_wrap
            * (disparity_diff < default_arguments.sflow2se3_oflow_disp_std_abs_max * math.sqrt(2))
        ).bool(),
        dev_trusted_perc=default_arguments.sflow2se3_oflow_disp_trusted_perc,
        valid_min_perc=default_arguments.sflow2se3_oflow_disp_std_valid_min_perc,
        std_min=default_arguments.sflow2se3_oflow_disp_std_abs_min * math.sqrt(2),
        std_max=default_arguments.sflow2se3_oflow_disp_std_abs_max * math.sqrt(2),
        correct_trunc_factor=default_arguments.sflow2se3_oflow_disp_std_correct_truncation,
    )[0] / math.sqrt(2)
    # Estimate std of optical flow
    optical_flow_l_backward_wrap, valid_pixels_optical_flow_l_backward_wrap = backward_warp_flow(
        image=optical_flow_l_backward,
        flow=optical_flow_l_forward,
        return_valid_pixels=True,
        normalized_coordinates=False,
        padding_mode="zeros",
    )
    optical_flow_diff = optical_flow_l_forward + optical_flow_l_backward_wrap
    default_arguments.sflow2se3_model_se3_likelihood_oflow_abs_std = estimate_std(
        dev=optical_flow_diff,
        valid=(
            valid_pixels_optical_flow_l_backward_wrap
            * (
                (optical_flow_diff).norm(dim=1, keepdim=True)
                < default_arguments.sflow2se3_oflow_disp_std_abs_max * math.sqrt(2)
            )
        ).bool(),
        dev_trusted_perc=default_arguments.sflow2se3_oflow_disp_trusted_perc,
        valid_min_perc=default_arguments.sflow2se3_oflow_disp_std_valid_min_perc,
        std_min=default_arguments.sflow2se3_oflow_disp_std_abs_min * math.sqrt(2),
        std_max=default_arguments.sflow2se3_oflow_disp_std_abs_max * math.sqrt(2),
        correct_trunc_factor=default_arguments.sflow2se3_oflow_disp_std_correct_truncation,
    )[0] / math.sqrt(2)
    # Compute depth
    depth_1 = disp_2_depth(disparity_1_forward, intrinsics[0, 0, 0], baseline)
    depth_2 = disp_2_depth(disparity_2_forward, intrinsics[0, 0, 0], baseline)
    # Compute valid optical flow and disparity
    oflow_occ = estimate_occlusions(optical_flow_l_forward, optical_flow_l_backward)
    depth_occ_1 = estimate_occlusions(
        torch.cat((-disparity_1_forward, torch.zeros_like(disparity_1_forward)), dim=1),
        torch.cat((disparity_1_backward, torch.zeros_like(disparity_1_backward)), dim=1),
    )
    depth_occ_2 = estimate_occlusions(
        torch.cat((-disparity_2_forward, torch.zeros_like(disparity_2_forward)), dim=1),
        torch.cat((disparity_2_backward, torch.zeros_like(disparity_2_backward)), dim=1),
    )
    # Compute edges of disparity maps
    edges_1: Tensor = ~dilation(
        (canny((disparity_1_forward + 0.001).log())[0] > 0.2), torch.ones(5, 5, device=disparity_1_forward.device)
    ).bool()
    edges_2: Tensor = ~dilation(
        (canny((disparity_2_forward + 0.001).log())[0] > 0.2), torch.ones(5, 5, device=disparity_2_forward.device)
    ).bool()
    # Compute valid depth
    depth_valid_0 = (depth_1 >= default_arguments.sflow2se3_depth_min) * (
        depth_1 <= default_arguments.sflow2se3_depth_max
    )
    depth_valid_1 = (depth_2 >= default_arguments.sflow2se3_depth_min) * (
        depth_2 <= default_arguments.sflow2se3_depth_max
    )
    # Combine valid regions
    depth_valid_f0_1 = warp(torch.ones_like(depth_valid_1), optical_flow_l_forward, mode="nearest")
    depth_inbounds_f0_1 = warp(depth_valid_1 * depth_occ_2 * edges_2, optical_flow_l_forward, mode="nearest")
    valid_regions = depth_valid_0 * edges_1 * oflow_occ * depth_occ_1 * depth_valid_f0_1 * depth_inbounds_f0_1
    # To 3D
    pt3d_0 = depth_2_pt3d(depth_1, reproj_mats=reprojection)
    pt3d_1 = depth_2_pt3d(depth_2, reproj_mats=reprojection)
    pt3d_f0_1 = warp(pt3d_1, optical_flow_l_forward, mode="nearest")
    # Make data dict
    data = {}
    data["rgb_l_01"] = image_1_l.float()
    data["pt3d_0"] = pt3d_0.float()
    data["pt3d_f0_1"] = pt3d_f0_1.float()
    data["oflow"] = optical_flow_l_forward.float()
    data["pt3d_valid_0"] = valid_regions.bool()
    data["pt3d_valid_f0_1"] = valid_regions.bool()
    data["projection_matrix"] = intrinsics[:, :-1].float()
    data["baseline"] = baseline.float()
    # Apply valid pixels if given
    if valid_pixels is not None:
        data["pt3d_valid_0"] *= valid_pixels
        data["pt3d_valid_f0_1"] *= valid_pixels
    # Check if we have valid pixels
    if torch.logical_and(data["pt3d_valid_0"], data["pt3d_valid_f0_1"]).sum() < 512:
        return torch.zeros_like(image_1_l[0, 0]).long()
    # Init lists to store object proposals and SE(3)-motions
    object_proposals_runs = []
    logits_runs = []
    # Perform clustering (SF2SE3)
    for _ in range(runs):
        object_proposals, logits = sflow2se3(data, default_arguments, logger=None, iterations=6)
        object_proposals_runs.extend(object_proposals.split(1, dim=0))
        logits_runs.extend(logits.split(1, dim=0))
    # To tensor of the shape [N, H, W] and [N] (logits)
    object_proposals_runs = torch.cat(object_proposals_runs, dim=0)
    logits_runs = torch.cat(logits_runs, dim=0)
    assert object_proposals_runs.shape[0] == logits_runs.shape[0]
    # Aggregate object proposal regions and threshold to get object proposal regions
    object_counts = object_proposals_runs.sum(dim=0, keepdims=True)
    object_counts = (object_counts * object_proposals_runs).sum(dim=(1, 2)) / object_proposals_runs.sum(dim=(1, 2))
    objects_keep = object_counts >= (runs * 0.8)
    object_proposals_filtered = object_proposals_runs[objects_keep]
    logits_filtered = logits_runs[objects_keep]
    # Perform NMS
    _, _, object_proposals_nms, keep_nms = mask_matrix_nms(
        object_proposals_filtered,
        torch.zeros_like(logits_filtered),
        logits_filtered,
    )
    # Return empty map if no object proposal has survived
    if object_proposals_nms.shape[0] == 0:
        return torch.zeros_like(image_1_l[0, 0]).long()
    # Sort object proposal by size
    permutation = torch.argsort(object_proposals_nms.sum(dim=(-1, -2)), descending=False)
    object_proposals_nms = object_proposals_nms[permutation]
    # First full holes in each object proposal and then perform connected components
    object_proposals_final = torch.zeros(
        object_proposals_nms.shape[-2],
        object_proposals_nms.shape[-1],
        device=object_proposals_nms.device,
        dtype=torch.long,
    )
    for object_proposal in object_proposals_nms:
        # Perform connected components on background to fill holes and convert back
        object_proposal = connected_components(object_proposal == 0)
        value, count = object_proposal.unique(return_counts=True)
        background_value = value[torch.argmax(count)]
        object_proposal = (object_proposal != background_value).long()
        # Remap ids
        object_proposal = remap_ids(object_proposal)
        # Save object proposal(s)
        object_proposals_final[object_proposal > 0] = (
            object_proposal[object_proposal > 0] + object_proposals_final.amax()
        )
    # Finally omit too small object proposal
    object_proposals_final = omit_small_objects(
        object_proposals_final, pt3d_0[0], min_size=default_arguments.sflow2se3_min_object_size
    )
    return object_proposals_final


from scipy.spatial import Delaunay


def omit_small_objects(object_proposals: Tensor, points_3d: Tensor, min_size: float = 0.5) -> Tensor:
    """Function removes too small object proposals.

    Args:
        object_proposals (Tensor): Object proposals as a tensor of the shape [H, W].
        points_3d (Tensor): Depth map of the shape [3, H, W].
        min_size (float): Minimum object size. Default 1.0.

    Returns:
        object_proposals_filtered (Tensor): Filtered object proposals of the shape [H, W].
    """
    # Iterate over all object proposals
    for id in object_proposals.unique(sorted=True)[1:]:
        # Get pixel coordinates of object
        coordinates = torch.argwhere(object_proposals == id)
        # Get 3d coordinates of object
        coordinates_3d = points_3d[:, coordinates[:, 0], coordinates[:, 1]].permute(1, 0)
        # Assume we have enough points available to perform a meaningful triangulation
        if coordinates.shape[0] >= 64:
            # Perform triangulation
            triangulation = Delaunay(coordinates.cpu().detach().numpy())
            # Get triangles
            triangles = torch.tensor(triangulation.simplices, device=coordinates.device)
            # Get 3D triangles
            triangles_3d = coordinates_3d[triangles]
            # Calculate the vectors along the edges of each triangle
            v1 = triangles_3d[:, 1] - triangles_3d[:, 0]  # Vector from vertex 0 to vertex 1
            v2 = triangles_3d[:, 2] - triangles_3d[:, 0]  # Vector from vertex 0 to vertex 2
            # Compute surface area
            cross_prod = torch.cross(v1, v2, dim=1)
            triangle_areas = torch.norm(cross_prod, dim=1) * 0.5
            surface_area = triangle_areas.sum()
            if surface_area.item() < min_size:
                object_proposals[object_proposals == id] = 0
        else:
            object_proposals[object_proposals == id] = 0
    # Remap object proposals to consecutive ids
    object_proposals_filtered = remap_ids(object_proposals)
    return object_proposals_filtered


def mask_matrix_nms(
    masks,
    labels,
    scores,
    filter_thr=0.6,
    nms_pre=-1,
    max_num=-1,
    kernel="gaussian",
    sigma=2.0,
    mask_area=None,
):
    """Matrix NMS for multi-class masks.

    Notes:
        Code from MMDetection (MMDetection detection license applies)
        Copyright (c) OpenMMLab. All rights reserved.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
              the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(0, *masks.shape[-2:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix / (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)).triu(
        diagonal=1
    )
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(0, *masks.shape[-2:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds
