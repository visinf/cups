import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
from cups.scene_flow_2_se3.geometric.se3.fit.corresp_3d_2d import (
    fit_se3_to_corresp_3d_2d_and_masks,
)


def fit_se3_to_pt3d_oflow_and_masks(
    masks_in,
    pts1,
    oflow,
    proj_mat,
    orig_H=None,
    orig_W=None,
    resize_mode="bilinear",
    method="cpu-epnp",
    weights=None,
    prev_se3_mats=None,
):
    K, H, W = masks_in.shape
    dtype = proj_mat.dtype
    device = proj_mat.device

    pxl2 = o4geo_pinhole.oflow_2_pxl2d(oflow[None,], orig_H=orig_H, orig_W=orig_W, resize_mode=resize_mode)[0]
    se3 = fit_se3_to_corresp_3d_2d_and_masks(
        masks_in, pts1, pxl2, proj_mat, method=method, weights=weights, prev_se3_mats=prev_se3_mats
    )
    return se3
