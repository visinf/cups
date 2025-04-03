import cv2
import numpy as np
import pytorch3d as t3d
import torch

from cups.scene_flow_2_se3.geometric.se3.fit.elemental import mask_points


def fit_se3_to_corresp_3d_2d_and_masks(
    masks_in, pts1, pxl2, proj_mat, method="cpu-epnp", weights=None, prev_se3_mats=None
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

    pts1, pxl2, weights = mask_points(masks_in, pts1, pxl2, weights)
    return fit_se3_to_corresp_3d_2d(pts1, pxl2, weights, proj_mat, method=method, prev_se3_mats=prev_se3_mats)


def fit_se3_to_corresp_3d_2d(pts1, pxl2, weights, proj_mat, method="cpu-epnp", prev_se3_mats=None):
    """Calculates se3 fit.

    Parameters
    ---------
    pts1 torch.Tensor: KxNxC1, float
    pts2 torch.Tensor: KxNxC2, float
    weights torch.Tensor: KxN, float
    proj_mat torch.Tensor: 2x3, float
    method str: describes which method should be used

    Returns
    -------
    transf: Kx4x4, float
    transf_centroid1: Kx4x4, float
    """

    if method.startswith("cpu"):
        # method2 = "cpu-epnp"
        se3_mats = fit_se3_to_corresp_3d_2d_opencv(pts1, pxl2, weights, proj_mat, method, prev_se3_mats)
        return se3_mats
    elif method == "gpu-epnp":
        K = len(pts1)
        dtype = pts1.dtype
        device = pts1.device
        se3_mats = torch.zeros(size=(K, 4, 4), dtype=dtype, device=device)
        se3s = t3d.ops.efficient_pnp(pts1, pxl2, weights, skip_quadratic_eq=False)
        se3_mats[:, :3, :3] = se3s.R.permute(0, 2, 1)
        se3_mats[:, :3, 3] = se3s.T
        se3_mats[:, 3, 3] = 1.0
        return se3_mats


def gpu_epnp(pts1, pxl2, weights, proj_mat):
    # 1. compute control points

    # 2. compute barycentric coordinates

    # 3. fill M

    # 4. solve M via SVD

    # 5. compute L 6x10
    # 6. compute rho
    # 7. compute R, t (perform 3 times and for betas1/2/3)
    # 7.1. find_betas_approx(L, rho, betas)
    # 7.2. gauss_newton(L, rho, betas)
    # 7.3. compute R,t
    # 7.4. reprojection error
    pass


def gpu_dlt(pts1, pxl2, weights, proj_mat):
    pass


def gpu_iterative(pts1, pxl2, weights, proj_mat):
    pass


def fit_se3_to_corresp_3d_2d_opencv(pts1, pxl2, weights, proj_mat, method, prev_se3_mats=None):
    """Calculates se3 fit.

    Parameters
    ---------
    pts1 torch.Tensor: KxNxC1, float
    pts2 torch.Tensor: KxNxC2, float
    weights torch.Tensor: KxN, float
    proj_mat torch.Tensor: 2x3, float
    method str: describes which method should be used

    Returns
    -------
    transf: Kx4x4, float
    transf_centroid1: Kx4x4, float
    """
    dtype = pts1.dtype
    device = pts1.device
    K = len(pts1)
    N_ver = 4
    se3_mats = []  # torch.zeros(size=(K, 4, 4), dtype=dtype, device=device)

    proj_mat_ext = torch.cat((proj_mat, torch.zeros(size=(1, 3), dtype=dtype, device=device)), dim=0)
    proj_mat_ext[2, 2] = 1.0
    proj_mat_ext = proj_mat_ext.detach().cpu().numpy()

    if prev_se3_mats is not None:
        prev_se3_mats_np = prev_se3_mats[:, :3, :3].detach().cpu().numpy()
        prev_transl_np = prev_se3_mats[:, :3, 3].detach().cpu().numpy()
        prev_so3_log_np = np.zeros(shape=(K, 3, 1), dtype=prev_transl_np.dtype)
        for k in range(K):
            prev_so3_log_np[k], _ = cv2.Rodrigues(prev_se3_mats_np[k])

    for k in range(K):
        pts1_k = pts1[k].permute(1, 0)
        pxl2_k = pxl2[k].permute(1, 0)
        weights_k = weights[k] > 0.5

        if weights_k.sum() < N_ver:
            transf_pred = torch.eye(n=4, dtype=dtype, device=device)
            se3_mats.append(transf_pred)
            continue
        # 3 x H*W

        pts1_k = pts1_k[:, weights_k]
        pxl2_k = pxl2_k[:, weights_k]

        pts1_k = pts1_k.permute(1, 0).detach().cpu().numpy()
        pxl2_k = pxl2_k.permute(1, 0).detach().cpu().numpy()

        if method == "cpu-iterative-continue":
            retval, r, t = cv2.solvePnP(
                pts1_k,
                pxl2_k,
                proj_mat_ext,
                0,
                prev_so3_log_np[k],
                prev_transl_np[k][:, None],
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        elif method == "cpu-iterative":
            retval, r, t = cv2.solvePnP(
                pts1_k, pxl2_k, proj_mat_ext, 0, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE
            )

        elif method == "cpu-epnp":
            retval, r, t = cv2.solvePnP(pts1_k, pxl2_k, proj_mat_ext, 0, flags=cv2.SOLVEPNP_EPNP)

        elif method == "cpu-ransac-iterative-continue":
            retval, r, t, mask_inliers = cv2.solvePnPRansac(
                pts1_k,
                pxl2_k,
                proj_mat_ext,
                0,
                prev_so3_log_np[k],
                prev_transl_np[k][:, None],
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        elif method == "cpu-ransac-iterative":
            retval, r, t, mask_inliers = cv2.solvePnPRansac(
                pts1_k, pxl2_k, proj_mat_ext, 0, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE
            )

        elif method == "cpu-ransac-epnp":
            retval, r, t, mask_inliers = cv2.solvePnPRansac(pts1_k, pxl2_k, proj_mat_ext, 0, flags=cv2.SOLVEPNP_EPNP)

        R, _ = cv2.Rodrigues(r)
        transf_pred = torch.eye(4, dtype=dtype, device=device)
        transf_pred[:3, :3] = torch.from_numpy(R).type(dtype).to(device)
        transf_pred[:3, 3] = torch.from_numpy(t[:, 0]).type(dtype).to(device)

        se3_mats.append(transf_pred)
    se3_mats = torch.stack(se3_mats)
    return se3_mats
