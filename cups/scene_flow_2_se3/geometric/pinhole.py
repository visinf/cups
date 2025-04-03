import torch

import cups.scene_flow_2_se3.vision.resize as o4res
import cups.scene_flow_2_se3.vision.warp as o4warp
from cups.scene_flow_2_se3 import elemental as o4
from cups.scene_flow_2_se3.geometric.grid import (
    oflow_2_pxl2d,
    pxl2d_2_pxl2d_normalized,
    pxl2d_normalized_2_mask_inside,
    shape_2_pxl2d,
)
from cups.scene_flow_2_se3.probabilistic.models import gaussian as o4prob_gaussian


def rescale_proj_mats(proj_mats, sx, sy):
    proj_mats[:, 0, :] = proj_mats[:, 0, :].clone() * sx
    proj_mats[:, 1, :] = proj_mats[:, 1, :].clone() * sy


def rescale_reproj_mats(reproj_mats, sx, sy):
    reproj_mats[:, :, 0] = reproj_mats[:, :, 0].clone() / sx
    reproj_mats[:, :, 1] = reproj_mats[:, :, 1].clone() / sy


def rescale_intrinsics(proj_mats, reproj_mats, sx, sy):
    # sx = target_W / W
    # sy = target_H / H
    proj_mats = rescale_proj_mats(proj_mats, sx, sy)
    reproj_mats = rescale_reproj_mats(reproj_mats, sx, sy)

    return proj_mats, reproj_mats


def disp_2_depth(disp, fx, baseline=0.5326):  # , baseline=0.5326):
    # depth_min=1e-3, depth_max=1e8
    # disp: Bx1xHxW
    # note for kitti-dataset: baseline=0.54 -> 54 cm
    # depth = focal-length * baseline / disparity

    # depth_valid = disp >= 0.0
    # disp = torch.clamp(disp, 0)
    fx = fx[..., None, None, None]

    depth = fx * baseline / (disp.abs() + 1e-10) * disp.sign()

    # depth_valid *= (depth >= depth_min) * (depth <= depth_max)

    # depth = torch.clamp(depth, depth_min, depth_max)

    # https: // github.com / visinf / self - mono - sf / tree / master / models

    return depth


def depth_2_disp(depth, fx, baseline=0.5326):  # baseline=0.5326):
    fx = fx[..., None, None, None]

    disp = fx * baseline / (depth.abs() + 1e-10) * depth.sign()

    return disp


def pt3d_2_depth(pts3d):
    return pts3d[:, 2:]


def pt3d_2_disp(pts3d, proj_mats, baseline=0.5326):  # , baseline=0.5326):
    depth = pt3d_2_depth(pts3d)

    return depth_2_disp(depth, fx=proj_mats[:, 0, 0], baseline=baseline)


def disp_2_pt3d(disp, proj_mats, reproj_mats, oflow=None, baseline=0.5326):  # 5334 5326
    depth, depth_valid = disp_2_depth(disp, fx=proj_mats[:, 0, 0], baseline=baseline)
    xyz = depth_2_pt3d(depth, reproj_mats, oflow)

    # if depth_min is not None or depth_max is not None:
    #    return xyz, depth_valid
    # else:
    return xyz


def pt3d_2_pxl3d(pt3d, proj_mats, baseline):

    pxl2d = pt3d_2_pxl2d(pt3d, proj_mats)

    disp = pt3d_2_disp(pt3d, proj_mats, baseline=baseline)

    pxl3d = torch.cat((pxl2d, disp), dim=1)

    return pxl3d


def pt3d_2_pxl2d(pt3d, proj_mats):
    # 3D-2D Projection:
    # u = (fx*x + cx * z) / z
    # v = (fy*y + cy * z) / z
    # shift on plane: delta_x = (fx * bx) / z
    #                 delta_y = (fy * by) / z
    # uv = (P * xyz) / z
    # P = [ fx   0  cx]
    #     [ 0   fy  cy]

    dim_in = pt3d.dim()

    if dim_in == 4:
        B, _, H, W = pt3d.shape
        pt3d = pt3d.reshape(B, 3, -1)
    elif dim_in == 3:
        B, N, _ = pt3d.shape
        pt3d = pt3d.permute(0, 2, 1)
    # 3 x N
    # z = torch.abs(xyz[:, 2].clone()) + 1e-8
    # uv = torch.matmul(proj_mats, xyz[:, :2]) / z.unsqueeze(1)

    pt3d[:, 2] = torch.abs(pt3d.clone()[:, 2]) + 1e-10
    # z = torch.abs(pt3d[:, 2]) + 1e-8
    uv = torch.matmul(proj_mats, pt3d / (pt3d[:, 2]).unsqueeze(1))
    # uv = uv.type_as(pt3d)
    # uv = torch.div(torch.matmul(proj_mats, xyz), (xyz[:, 2] + 1e-8).unsqueeze(1))

    # 2xN
    if dim_in == 4:
        uv = uv.reshape(B, 2, H, W)
    elif dim_in == 3:
        uv = uv.permute(0, 2, 1)
    return uv


def pt3d_2_oflow(pt3d, proj_mats, orig_H=None, orig_W=None, resize_mode="bilinear"):
    pxl2d = pt3d_2_pxl2d(pt3d, proj_mats)
    oflow = pxl2d_2_oflow(pxl2d, orig_H=orig_H, orig_W=orig_W, resize_mode=resize_mode)
    return oflow


def oflow_2_vec3d(oflow, reproj_mats):
    B, _, H, W = oflow.shape
    depth = torch.ones(size=(B, 1, H, W), dtype=oflow.dtype, device=oflow.device)

    vec3d = depth_2_pt3d(depth, reproj_mats, oflow)

    vec3d = vec3d / torch.norm(vec3d, p=2, dim=1, keepdim=True)

    return vec3d


def pt_2_hpt(pt, cdim=1):
    pt_shape = pt.shape

    ones_buf_shape = list(pt_shape)
    ones_buf_shape[cdim] = 1
    ones_buf_shape = torch.Size(ones_buf_shape)

    ones_buf = torch.ones(size=ones_buf_shape, dtype=pt.dtype).to(pt.device)

    hpt = torch.cat((pt, ones_buf), dim=cdim)

    return hpt


def pt3d_2_hpt4d(pt3d, cdim=1):
    hpt4d = pt_2_hpt(pt3d, cdim)
    return hpt4d


def pxl2d_2_pxl3d(pxl2d, cdim=1):
    pxl3d = pt_2_hpt(pxl2d, cdim)
    return pxl3d


def pxl2d_2_hpt3d(pxl2d, intr_inv, cdim=1):
    # pxl2d torch.Tensor:    S1 x ... Sb x 2 x ... (with Sc=2 and c=cdim)
    # intr_inv torch.Tensor: S1 x ... Sb x  3 x 3

    # S1 x ... Sb x 3 x ...
    pxl3d = pxl2d_2_pxl3d(pxl2d, cdim=cdim)

    hpt3d = o4.multiply_matrix_vector(intr_inv, pxl3d, cdim=cdim)

    return hpt3d


def intr2x3_to_3x3(intr):
    """Copy values from 2x3 into 3x3 matrix.

    Parameters
    ----------
    intr torch.Tensor: ...x2x3

    Returns
    -------
    intr torch.Tensor: ...x3x3
    """

    r3 = torch.FloatTensor([0.0, 0.0, 1]).to(intr.device).reshape(1, 3)
    r3 = r3.repeat(*intr.shape[:-2], 1, 1)
    intr = torch.cat((intr, r3), dim=-2)

    return intr


def intr2x3_to_4x4(intr):
    """Copy values from 2x3 into 4x4 matrix.

    Parameters
    ----------
    intr torch.Tensor: ...x2x3

    Returns
    -------
    intr torch.Tensor: ...x4x4
    """

    intr = intr2x3_to_3x3(intr)

    c3 = torch.FloatTensor([0.0, 0.0, 0.0]).to(intr.device).reshape(3, 1)
    c3 = c3.repeat(*intr.shape[:-2], 1, 1)
    intr = torch.cat((intr, c3), dim=-1)

    r4 = torch.FloatTensor([0.0, 0.0, 0, 1]).to(intr.device).reshape(1, 4)
    r4 = r4.repeat(*intr.shape[:-2], 1, 1)
    intr = torch.cat((intr, r4), dim=-2)

    return intr


def proj4x4_to_3x4(proj):
    ids012 = torch.LongTensor([0, 1, 2]).to(proj.device)
    proj = torch.index_select(proj, dim=-2, index=ids012)

    return proj


def oflow_2_vec2d(oflow, reproj_mats):
    B, _, H, W = oflow.shape
    depth = torch.ones(size=(B, 1, H, W), dtype=oflow.dtype, device=oflow.device)

    vec2d = depth_2_pt3d(depth, reproj_mats, oflow)[:, :2]
    # vec2d = vec2d / torch.norm(vec2d, p=2, dim=1, keepdim=True)

    return vec2d


def depth_2_pt3d(depth, reproj_mats, oflow=None):
    B, _, H, W = depth.shape

    dtype = depth.dtype
    device = depth.device

    if oflow == None:
        grid_uv1 = shape_2_pxl3d(B=0, H=H, W=W, dtype=dtype, device=device)
        uv1 = grid_uv1.reshape(3, -1)
        # 3 x N
    else:
        grid_uv1 = oflow_2_pxl3d(oflow)
        uv1 = grid_uv1.reshape(B, 3, -1)

    xyz = torch.matmul(reproj_mats, uv1) * depth.flatten(2)
    # B x 3 x 3 * 3 x N = (B x 3 x N)

    xyz = xyz.reshape(B, 3, H, W)

    # 2D-3D Re-Projection:
    # x = (u/fx - cx/fx) * z
    # y = (v/fy - cy/fy) * z
    # z = z
    # xyz = (RP * uv1) * z
    # RP = [ 1/fx     0  -cx/fx ]
    #      [    0  1/fy  -cy/fy ]
    #      [    0      0      1 ]

    return xyz


def undistort_data(data, keys, proj_mat_2x3, reproj_mat_3x3, k1, k2, k3, p1, p2):

    dtype = data[keys[0]].dtype
    device = data[keys[0]].device
    B, C, H, W = data[keys[0]].shape
    """
    # x(1 +k1r2+k2r4+k3r6)+ 2p1 ̄x ̄y+p2(r2+ 2 ̄x2)
    grid_uv = shape_2_pxl2d(B, H, W, dtype, device)

    #
    grid_uv_centered = grid_uv -
    grid_uv_norm = pxl2d_2_pxl2d_normalized(grid_uv)

    grid_r = torch.grid_uv_norm
    #r = grid
    """
    # idea:
    # 1. reproject reproj_mat
    depth = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    homog_pt3d = depth_2_pt3d(depth, reproj_mat_3x3)

    # 2. remap:  x(1 +k1r2+k2r4+k3r6)+ 2p1 ̄x ̄y+p2(r2+ 2 ̄x2)
    r = torch.norm(homog_pt3d[:, :2], dim=1, keepdim=True)
    r2 = r**2
    homog_x = homog_pt3d[:, 0:1].clone()
    homog_y = homog_pt3d[:, 1:2].clone()
    kr = 1 + ((k3 * r2 + k2) * r2 + k1) * r2

    homog_pt3d_distorted = torch.ones_like(homog_pt3d)
    homog_pt3d_distorted[:, 0:1] = homog_x * kr + p1 * 2 * homog_x * homog_y + p2 * (r2 + 2 * homog_x**2)
    homog_pt3d_distorted[:, 1:2] = homog_y * kr + p1 * (r2 + 2 * homog_y**2) + p2 * 2 * homog_x * homog_y
    """
    d0: -0.073768415722531969	// distortion coefficients k_1 (OpenCV format)
    d1: 0.16408477247405578		// distortion coefficients k_2 (OpenCV format)
    d2: 0.0				// distortion coefficients p_1 (OpenCV format)
    d3: 0.0	 			// distortion coefficients p_2 (OpenCV format)
    d4: 0.0	 			// distortion coefficients k_3 (OpenCV format)
    """

    # homog_pxl2d_dist =
    # 3. project proj_mat
    pxl2d_distorted = pt3d_2_pxl2d(homog_pt3d_distorted, proj_mats=proj_mat_2x3)

    for key in keys:
        if key in data.keys():
            data[key] = o4warp.interpolate2d(data[key], pxl2d_distorted, mode="nearest", padding_mode="zeros")

    return data
    """proj_mat_2x3 = proj_mat_2x3.detach().cpu().numpy()

    # Define Camera Matrix proj_mat_3x3 = np.concatenate([proj_mat_2x3, np.array([[[0, 0, 1]]])], axis=1)

    # Define distortion coefficients dist = np.array([k1, k2, p1, p2, k3])

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(proj_mat_3x3[0], dist, (W, H), 1, (W, H)) # undistort mapx, mapy =
    cv2.initUndistortRectifyMap(proj_mat_3x3[0], dist, None, newcameramtx, (W, H), 5)

    for b in range(B):     img1 = img[b]     img1 = o4visual.tensor_to_cv_img(img1)     img1 = cv2.remap(img1, mapx,
    mapy, cv2.INTER_NEAREST)

    # crop the image x, y, w, h = roi img1 = img1[y:y + h, x:x + w]

    img[b] = o4visual.cv_img_to_tensor(img1, device=device)

    return img
    """


def shape_2_pxl3d(B, H, W, dtype, device):

    grid_uv = shape_2_pxl2d(B, H, W, dtype, device)

    if B != 0:
        grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)
        # B x 3 x H x W

    else:
        grid_1 = torch.ones(size=(1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=0)
        # 3 x H x W
    return grid_uv1


def disp_2_pxl2d(disp):
    # in: Bx1xHxW
    B, _, H, W = disp.shape
    dtype = disp.dtype
    device = disp.device

    grid_uv = shape_2_pxl2d(B=B, H=H, W=W, dtype=dtype, device=device)

    grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    grid_uv = grid_uv + torch.cat((disp, grid_1), dim=1)

    return grid_uv


def pxl2d_2_oflow(pxlcoords, orig_H=None, orig_W=None, resize_mode="bilinear"):

    B, _, H, W = pxlcoords.shape
    dtype = pxlcoords.dtype
    device = pxlcoords.device

    if orig_H is None or orig_W is None:
        orig_H = H
        orig_W = W

    grid_xy = shape_2_pxl2d(B=0, H=orig_H, W=orig_W, dtype=dtype, device=device)

    grid_xy = o4res.resize(
        grid_xy[None,],
        H_out=H,
        W_out=W,
        mode=resize_mode,
    )[0]

    flow = pxlcoords - grid_xy

    return flow


def oflow_2_pxl3d(flow):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    grid_uv = oflow_2_pxl2d(flow)

    grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)

    return grid_uv1


def oflow_2_pxl2d_normalized(flow):
    grid_xy = oflow_2_pxl2d(flow)

    grid_xy = pxl2d_2_pxl2d_normalized(grid_xy)

    return grid_xy


def pxl2d_2_mask_non_occl(grid_xy, binary=False):
    # input: B x 2 x H x W
    B, _, H, W = grid_xy.shape
    dtype = grid_xy.dtype
    device = grid_xy.device

    grid_xy_floor = torch.floor(grid_xy).long()
    # B x 2 x H x W

    grid_xy_offset = grid_xy - grid_xy_floor
    # B x 2 x H x W

    kernel_xy = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64, device=device)
    kernel_xy = kernel_xy.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    K = 4
    # 1 x 4 x 2 x 1 x 1
    grid_weights_offset_x = 1.0 - torch.abs(kernel_xy[:, :, 0] - grid_xy_offset[:, 0].unsqueeze(1))
    grid_weights_offset_y = 1.0 - torch.abs(kernel_xy[:, :, 1] - grid_xy_offset[:, 1].unsqueeze(1))
    #                                      1 x 4 x 1 x 1      - B x 1 x H x W

    grid_weights = grid_weights_offset_x * grid_weights_offset_y
    # B x 4 x H x W

    grid_weights = grid_weights.reshape(B, K, -1)
    # grid_weights = torch.ones_like(grid_weights)

    grid_indices_x = grid_xy_floor[:, 0].unsqueeze(1) + kernel_xy[:, :, 0]
    grid_indices_y = grid_xy_floor[:, 1].unsqueeze(1) + kernel_xy[:, :, 1]
    #                 B x 1 x H x W                   + 1 x 4 x 1 x 1
    # -> B x 4 x H x W

    grid_indices_x = grid_indices_x.reshape(B, K, -1)
    grid_indices_y = grid_indices_y.reshape(B, K, -1)
    # B x 4 x H*W

    grid_xy_indices = grid_indices_y * W + grid_indices_x
    # B x 4 x H*W

    grid_counts = torch.zeros((B, K, H * W), requires_grad=False, dtype=dtype, device=device)

    # problem not in parallel: the number of valid indices are different for each batch and kernel_offset
    for b in range(B):
        for i in range(K):
            valid_indices_mask = (
                (grid_indices_x[b, i] >= 0)
                & (grid_indices_x[b, i] < W)
                & (grid_indices_y[b, i] >= 0)
                & (grid_indices_y[b, i] < H)
            )
            # B x H*W

            grid_xy_indices_i = grid_xy_indices[b, i]
            # B x H*W
            grid_xy_indices_i = grid_xy_indices_i[valid_indices_mask].reshape(-1)
            # B x valid(H*W)

            grid_weights_i = grid_weights[b, i]
            # B x H*W
            grid_weights_i = grid_weights_i[valid_indices_mask].reshape(-1)
            # B x valid(H*W)

            grid_counts[b, i] = grid_counts[b, i].scatter_add(dim=0, index=grid_xy_indices_i, src=grid_weights_i)
            # B x 4 x H*W

    grid_counts = grid_counts.reshape((B, K, H, W))
    grid_counts = torch.sum(grid_counts, dim=1, keepdim=True)

    grid_counts = torch.clamp(grid_counts, 0.0, 1.0)
    masks_non_occlusion = grid_counts

    masks_non_occlusion = masks_non_occlusion.detach()

    if binary:
        masks_non_occlusion = masks_non_occlusion > 0.5

    return masks_non_occlusion


def oflow_2_mask_non_occl(flow_fwd, flow_bwd, std_pixel_dev, inlier_thresh, std_pixel_dev_rel=None, return_dev=False):

    flow_bwd_bwdwrpd, flow_inside = o4warp.warp(flow_bwd.clone(), flow_fwd.clone(), return_masks_flow_inside=True)
    # o4vis.visualize_img(torch.cat((o4vis.flow2rgb(flow_fwd[0]), o4vis.flow2rgb(flow_bwd[0]), o4vis.flow2rgb(flow_bwd_bwdwrpd[0])), dim=1), height=900)

    # flow_dev_abs = torch.norm(flow_fwd + flow_bwd_bwdwrpd, dim=1, keepdim=True)
    flow_dev_abs = (flow_fwd + flow_bwd_bwdwrpd).abs()
    mask_fwd_bwd_cons = (
        o4prob_gaussian.calc_inlier_prob(dev=flow_dev_abs, std=std_pixel_dev[None, :, None, None]).prod(
            dim=1, keepdim=True
        )
        > inlier_thresh
    ) * flow_inside

    # mask_fwd_bwd_cons = (
    #    flow_dev_abs < max_pixel_dev[None, :, None, None]
    # ).prod(dim=1, keepdim=True).bool() * flow_inside

    if std_pixel_dev_rel is not None:

        # flow_norm_avg = (
        #    torch.norm(flow_fwd, dim=1, keepdim=True)
        #    + torch.norm(flow_bwd_bwdwrpd, dim=1, keepdim=True)
        # ) / 2.0

        flow_norm_avg = (flow_fwd.abs() + flow_bwd_bwdwrpd.abs()) / 2.0

        # flow_dev_rel = torch.norm(flow_fwd + flow_bwd_bwdwrpd, dim=1, keepdim=True) / flow_norm_avg
        flow_dev_rel = (flow_fwd + flow_bwd_bwdwrpd).abs() / flow_norm_avg

        mask_fwd_bwd_cons_rel = ((flow_dev_rel) < std_pixel_dev_rel[None, :, None, None]).prod(
            dim=1, keepdim=True
        ).bool() * flow_inside

        mask_fwd_bwd_cons += mask_fwd_bwd_cons_rel

    # pxl2d = o4geo.oflow_2_pxl2d(flow_bwd)
    # mask_bwd = pxl2d_2_mask_non_occl(pxl2d)
    if return_dev:
        return mask_fwd_bwd_cons, flow_dev_abs
    else:
        return mask_fwd_bwd_cons


def oflow_2_mask_valid(flow_fwd, flow_bwd, std_pixel_dev, inlier_thresh, std_pixel_dev_rel=None, return_dev=False):
    if return_dev:
        mask_non_occl, dev = oflow_2_mask_non_occl(
            flow_fwd,
            flow_bwd,
            std_pixel_dev,
            inlier_thresh=inlier_thresh,
            std_pixel_dev_rel=std_pixel_dev_rel,
            return_dev=return_dev,
        )
        return mask_non_occl * oflow_2_mask_inside(flow_fwd), dev
    else:
        return oflow_2_mask_non_occl(
            flow_fwd,
            flow_bwd,
            std_pixel_dev,
            inlier_thresh=inlier_thresh,
            std_pixel_dev_rel=std_pixel_dev_rel,
            return_dev=return_dev,
        ) * oflow_2_mask_inside(flow_fwd)


def disp_2_mask_valid(disp_fwd, disp_bwd, std_pixel_dev, inlier_thresh, std_pixel_dev_rel=None, return_dev=False):
    B, C, H, W = disp_fwd.shape
    dtype = disp_fwd.dtype
    device = disp_fwd.device

    disp_fwd_oflow = torch.zeros(size=(B, 2, H, W), dtype=dtype, device=device)
    disp_fwd_oflow[:, :1] = -disp_fwd[:, :1]

    disp_bwd_oflow = torch.zeros(size=(B, 2, H, W), dtype=dtype, device=device)
    disp_bwd_oflow[:, :1] = disp_bwd[:, :1]

    return oflow_2_mask_valid(
        disp_fwd_oflow,
        disp_bwd_oflow,
        std_pixel_dev,
        inlier_thresh=inlier_thresh,
        std_pixel_dev_rel=std_pixel_dev_rel,
        return_dev=return_dev,
    )


def oflow_2_mask_inside(flow):

    pxl2d_normalized = oflow_2_pxl2d_normalized(flow)
    mask_inside = pxl2d_normalized_2_mask_inside(pxl2d_normalized)

    return mask_inside


def pxl2d_2_mask_inside(pxl2d):
    pxl2d_normalized = pxl2d_2_pxl2d_normalized(pxl2d)
    mask_inside = pxl2d_normalized_2_mask_inside(pxl2d_normalized)
    return mask_inside


def pxl2d_2_mask_valid(pxl2d_tf_fwd, pxl2d_tf_bwd):
    # in: Bx2xHxW
    # out: Bx1xHxW

    mask_inside = pxl2d_2_mask_inside(pxl2d_tf_fwd)
    mask_non_occl = pxl2d_2_mask_non_occl(pxl2d_tf_bwd)

    return mask_inside * mask_non_occl
