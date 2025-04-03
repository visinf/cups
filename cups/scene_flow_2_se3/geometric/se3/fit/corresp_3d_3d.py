import pytorch3d as t3d
import torch

from cups.scene_flow_2_se3.geometric.se3.fit.elemental import mask_points
from cups.scene_flow_2_se3.geometric.se3.fit.pt3d_oflow import (
    fit_se3_to_pt3d_oflow_and_masks,
)


def fit_se3_to_corresp_3d_3d_and_masks(
    masks_in, pts1, pts2, method="gpu-3d-3d-analytic", oflow=None, proj_mat=None, weights=None, corresp_count_max=None
):
    """Calculates se3 fit.

    Parameters
    ---------
    masks_in torch.Tensor: KxHxW / KxN, bool
    pts1 torch.Tensor: C1xHxW / C1xN, float
    pts2 torch.Tensor: C2xHxW / C2xN, float
    weights_in torch.Tensor: KxHxW / KxN, float

    Returns
    -------
    transf: Kx4x4, float
    transf_centroid1: Kx4x4, float
    """

    if corresp_count_max is not None:
        masks_in = masks_in.clone()
        masks_in_shape = masks_in.shape
        inlier_too_much = masks_in[masks_in.flatten(1).sum(dim=1) > corresp_count_max].flatten(1)
        K_too_much = inlier_too_much.shape[0]
        if K_too_much > 0:
            indices_default = torch.arange(masks_in_shape[1:].numel(), device=masks_in.device)
            for k in range(K_too_much):
                N_too_much = inlier_too_much[k].sum()
                # inlier_too_much_pos = inlier_too_much[k]
                indices_too_much = indices_default[inlier_too_much[k].flatten()]
                indices_too_much_nullified = indices_too_much[torch.randperm(N_too_much)[corresp_count_max:]]
                inlier_too_much[k, indices_too_much_nullified] = False
            inlier_too_much = inlier_too_much.reshape(-1, *masks_in_shape[1:])
            masks_in[masks_in.flatten(1).sum(dim=1) > corresp_count_max] = inlier_too_much

    if method == "gpu-3d-3d-analytic":
        pts1, pts2, weights = mask_points(masks_in, pts1, pts2, weights)

        use_pytorch3d = False
        if use_pytorch3d:
            K = len(pts1)
            dtype = pts1.dtype
            device = pts1.device
            se3_mats = torch.zeros(size=(K, 4, 4), dtype=dtype, device=device)
            se3s = t3d.ops.corresponding_points_alignment(
                pts1, pts2, weights, estimate_scale=False, allow_reflection=False
            )
            se3_mats[:, :3, :3] = se3s.R.permute(0, 2, 1)
            se3_mats[:, :3, 3] = se3s.T
            se3_mats[:, 3, 3] = 1.0
        else:
            se3_mats = fit_se3_to_corresp_3d_3d(pts1, pts2, weights)
    else:
        # pxl2 = o4geo_pinhole.pt3d_2_pxl2d(pts2[None], proj_mat[None])[0]
        # return fit_se3_to_corresp_3d_2d_and_masks(masks_in, pts1, pxl2, proj_mat, method=method, weights=weights, prev_se3_mats=None)

        return fit_se3_to_pt3d_oflow_and_masks(
            masks_in, pts1, oflow=oflow, proj_mat=proj_mat, method=method, weights=weights
        )
    return se3_mats


def fit_se3_to_corresp_3d_3d(pts1, pts2, weights=None):
    """Calculates se3 fit.

    Parameters
    ----------
    pts1 torch.Tensor: KxNxC1, float
    pts2 torch.Tensor: KxNxC2, float
    weights torch.Tensor: KxN, float

    Returns
    -------
    transf: Kx4x4, float
    transf_centroid1: Kx4x4, float
    """
    B = len(pts1)

    device = pts1.device
    dtype = pts1.dtype

    pts1_ftf = pts1.clone()

    # if weights == None:
    #    inverse_depth = 1. / ((pts1[2:, :, :] + pts2[2:, :, :]) / 2.)**2
    #    weights = inverse_depth # (1 / pixel_assigned_counts) *
    #    weights = torch.clamp(weights, 0., 1.)

    # pts1, pts2: B x N x 3
    if weights == None:
        centroid_pts1 = torch.mean(pts1_ftf, dim=1)
        centroid_pts2 = torch.mean(pts2, dim=1)
    else:
        centroid_pts1 = torch.sum(pts1_ftf * weights[:, :, None], dim=1) / torch.sum(weights[:, :, None], dim=1)
        centroid_pts2 = torch.sum(pts2 * weights[:, :, None], dim=1) / torch.sum(weights[:, :, None], dim=1)

    pts1_centered = pts1_ftf - centroid_pts1[:, None, :]
    pts1_norm = pts1_centered  # / (torch.norm(pts1_centered, dim=2).unsqueeze(2) + 1e-5)
    pts2_centered = pts2 - centroid_pts2[:, None, :]
    pts2_norm = pts2_centered  # / (torch.norm(pts2_centered, dim=2).unsqueeze(2) + 1e-5)

    if weights == None:
        U, S, V = torch.svd(torch.matmul(pts2_norm.permute(0, 2, 1), pts1_norm))
    else:
        U, S, V = torch.svd(torch.matmul(pts2_norm.permute(0, 2, 1), pts1_norm * weights[:, :, None]))

    # M = torch.diag(torch.Tensor([1., 1., 1.]).to(device))
    ones_B = torch.ones(B, dtype=dtype, device=device)
    M = torch.zeros(size=(B, 3, 3), dtype=dtype, device=device)
    M_diag = M.diagonal(dim1=1, dim2=2)
    M_diag[:, :] = torch.stack((ones_B, ones_B, torch.det(U) * torch.det(V)), dim=1)

    rot = torch.matmul(U, torch.matmul(M, V.permute(0, 2, 1)))

    transl = centroid_pts2 - torch.sum(rot * centroid_pts1[:, None, :], dim=2)
    # transl = centroid_pts2 - (rot @ centroid_pts1[:, :, None])[:, :, 0]
    transf = torch.eye(4, device=device).repeat(B, 1, 1)

    transf[:, :3, :3] = rot
    transf[:, :3, 3] = transl

    if transf.isnan().sum() > 0:
        print("error: transf contains nan values")

    return transf
