import numpy as np
import pytorch3d.transforms as t3d
import torch

import cups.scene_flow_2_se3.geometric.epipolar as o4geo_epi
import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
import cups.scene_flow_2_se3.vision.visualization as o4visual


def se3_mat_2_dist(se3_mat):
    # in B x 4 x 4
    return torch.norm(se3_mat[:, 0:3, 3])


def se3_mat_2_angle(se3_mat):
    # in B x 4 x 4
    # an invitation to 3-d vision, p 27
    # return torch.acos(min(1, max(-1, (torch.trace(se3_mat[:, 0:3, 0:3]) - 1) / 2)))

    rot_logs = t3d.so3_log_map(se3_mat[:, 0:3, 0:3], eps=2e-04)
    return torch.norm(rot_logs, dim=1)


def se3_mat_2_angle_deg(se3_mat):
    # in B x 4 x 4
    return se3_mat_2_angle(se3_mat) / (2 * np.pi) * 360


def dist_transls(transl1, transl2):

    dist = torch.norm((transl1 - transl2), dim=-1)

    return dist


def angle_rots(rot1, rot2):

    rot = torch.matmul(rot1.permute(0, 2, 1), rot2)
    # print("rot", rot)
    """
    for i in range(len(rot)):
        trace = torch.trace(rot[i, :3, :3])
        if trace > 3.1 or trace <= -1:
            print("trace", torch.trace(rot[i, :3, :3]))
            print(rot[i, :3, :3])
    """
    try:
        rot_logs = t3d.so3_log_map(rot[:, :3, :3])
    except:
        print("geometric.se3.registration.angle_rots(): failed")
        # import time
        # time.sleep(2)
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


# def calc_transf_from_opticalflow(flow, mask, projection_matrix):
def calc_optical_flow_registration(objects_masks, pts1, proj_mats, target_oflow, orig_H, orig_W, resize_mode):
    K, H, W = objects_masks.shape
    dtype = proj_mats.dtype
    device = proj_mats.device
    """
    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(0.0, H + 0.0, dtype=dtype, device=device),
            torch.arange(0.0, W + 0.0, dtype=dtype, device=device),
        ]
    )
    grid_x = (grid_x / (W - 1.0) - 0.5) * W
    grid_y = (grid_y / (H - 1.0) - 0.5) * H
    # note1: magnitude must fit to the optical flow
    # note2: centered is required for triangulation later on
    # TODO: check if reprojection_matrix must be used for these normalized coordinates

    grid_1 = torch.ones((H, W), dtype=dtype, device=device)
    coords1 = torch.stack((grid_x, grid_y, grid_1), dim=0)

    grid_x = grid_x + flow[0, :, :]
    grid_y = grid_y + flow[1, :, :]
    coords2 = torch.stack((grid_x, grid_y, grid_1), dim=0)
    # 3 x H x W
    """

    pxl3d_1 = None  # 3 x H x W
    pxl3d_2 = None  # 3 x H x W
    # number of points to choose R,t from four Options (R1, t1), (R1, -t1), (R2, t1), (R2, -t2)

    # 3 x H x W
    pxl3d_1 = o4geo_pinhole.shape_2_pxl3d(B=0, H=orig_H, W=orig_W, dtype=dtype, device=device)
    pxl3d_1 = o4visual.resize(pxl3d_1, H_out=H, W_out=W, mode=resize_mode)
    pxl3d_2 = pxl3d_1 + torch.cat((target_oflow, torch.zeros(1, H, W, dtype=dtype, device=device)), dim=0)
    # pts: 3 x H x W
    # objects_masks: K x H x W

    objects_pts3d_1, objects_pxl3d_2, objects_weights = mask_points(objects_masks, pts1, pxl3d_2)
    # pxl3d_1, pxl3d_2: K x N x 3
    # weights: K x N

    proj_mat = proj_mats
    N_ver = 6
    se3_mats = []  # torch.zeros(size=(K, 4, 4), dtype=dtype, device=device)

    for k in range(K):
        pts3d_1 = objects_pts3d_1[k].permute(1, 0)
        pxl3d_2 = objects_pxl3d_2[k].permute(1, 0)
        mask = objects_weights[k] > 0.5

        if mask.sum() < N_ver:
            transf_pred = torch.zeros(size=(4, 4), dtype=dtype, device=device)
            se3_mats.append(transf_pred)
            continue
        # 3 x H*W

        pts3d_1 = pts3d_1[:, mask]
        pxl3d_2 = pxl3d_2[:, mask]
        N = pxl3d_1.shape[1]
        print("using ", N, "out of", H * W)
        # 3 x N
        K = torch.cat((proj_mat, torch.zeros(size=(1, 3), dtype=dtype, device=device)), dim=0)
        K[2, 2] = 1.0
        transf_pred = o4geo_epi.se3_from_correspondences_3D_2D(pts3d_1, pxl3d_2[:2], K)

        # transf_pred = o4geo_epi.calc_essential_matrix(pxl3d_1[:2], pxl3d_2[:2], K)

        se3_mats.append(transf_pred)
        """coords1_ver = pxl3d_1[:, :N_ver] coords2_ver = pxl3d_2[:, :N_ver]

        # normalization
        var1, mean1 = torch.var_mean(pxl3d_1, dim=1)
        var2, mean2 = torch.var_mean(pxl3d_2, dim=1)

        T1 = meanvar2normtransf(mean1, var1)
        T2 = meanvar2normtransf(mean2, var2)
        # 3 x 3

        pxl3d_1 = torch.matmul(T1, pxl3d_1)
        pxl3d_2 = torch.matmul(T2, pxl3d_2)

        s = torch.stack(
            (
                pxl3d_1[0] * pxl3d_2[0],
                pxl3d_1[1] * pxl3d_2[0],
                pxl3d_2[0],
                pxl3d_1[0] * pxl3d_2[1],
                pxl3d_1[1] * pxl3d_2[1],
                pxl3d_2[1],
                pxl3d_1[0],
                pxl3d_1[1],
                torch.ones_like(pxl3d_1[1]),
            ),
            dim=0,
        )
        # 9 x N

        f = torch.arange(9, dtype=dtype, device=device)
        # 9

        # coords2.T * F * coords1 = s.T f with s = [x1*x2, y1*x2, x2, x1*y2 , y1*y2, y2, x1, y1, 1]
        # s.T * f # N x 9 * 9= N

        U, S, V = torch.svd(s.T)
        f = V[:, -1]
        # f = V[-1, :]
        print(f)
        # print('singular', S[-1])

        F = f.reshape(3, 3)
        # undo data normalization
        F = torch.matmul(T2.T, torch.matmul(F, T1))

        res = torch.sum(torch.abs(torch.sum(pxl3d_2 * torch.matmul(F, pxl3d_1), dim=0)))
        # print('res',res )

        intr_K = torch.cat(
            (
                proj_mat,
                torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device),
            ),
            dim=0,
        )
        E = torch.mm(intr_K.T, torch.mm(F, intr_K))

        U, S, V = torch.svd(E)

        # rank enforcement
        S[-1] = 0.0
        E = torch.mm(U, torch.mm(torch.diag(S), V.T))
        U, S, V = torch.svd(E)

        W = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )

        R1 = torch.mm(torch.mm(U, W), V.T)
        if torch.det(R1) < 0.0:
            R1 = -R1

        R2 = torch.mm(torch.mm(U, W.T), V.T)
        if torch.det(R2) < 0.0:
            R2 = -R2

        t1 = U[:, -1]
        t2 = -t1

        transf1 = translrot2transf(R1, t1)
        transf2 = translrot2transf(R1, t2)
        transf3 = translrot2transf(R2, t1)
        transf4 = translrot2transf(R2, t2)

        # x2' = R * x1
        coords2_pred1 = torch.matmul(transf1[:3, :3], coords1_ver)
        coords2_pred2 = torch.matmul(transf2[:3, :3], coords1_ver)
        coords2_pred3 = torch.matmul(transf3[:3, :3], coords1_ver)
        coords2_pred4 = torch.matmul(transf4[:3, :3], coords1_ver)
        # 3 x N_ver

        zT1 = torch.zeros(size=(N_ver, 2), dtype=dtype, device=device)
        zT2 = torch.zeros(size=(N_ver, 2), dtype=dtype, device=device)
        zT3 = torch.zeros(size=(N_ver, 2), dtype=dtype, device=device)
        zT4 = torch.zeros(size=(N_ver, 2), dtype=dtype, device=device)

        # solve triangulation: (x2_normalized * x2_z) =  R (x1_normalized * x1_z) + t
        for i in range(N_ver):
            A = torch.stack((coords2_ver[:, i], -coords2_pred1[:, i]), dim=1)
            sol, _ = torch.lstsq(transf1[:3, 3, None], A)
            zT1[i] = sol[:2, 0]

            A = torch.stack((coords2_ver[:, i], -coords2_pred2[:, i]), dim=1)
            sol, _ = torch.lstsq(transf2[:3, 3, None], A)
            zT2[i] = sol[:2, 0]

            A = torch.stack((coords2_ver[:, i], -coords2_pred3[:, i]), dim=1)
            sol, _ = torch.lstsq(transf3[:3, 3, None], A)
            zT3[i] = sol[:2, 0]

            A = torch.stack((coords2_ver[:, i], -coords2_pred4[:, i]), dim=1)
            sol, _ = torch.lstsq(transf4[:3, 3, None], A)
            zT4[i] = sol[:2, 0]

        zT1_pos = torch.sum(((zT1[:, 0] > 0.0) * (zT1[:, 1]) > 0.0))
        zT2_pos = torch.sum(((zT2[:, 0] > 0.0) * (zT2[:, 1]) > 0.0))
        zT3_pos = torch.sum(((zT3[:, 0] > 0.0) * (zT3[:, 1]) > 0.0))
        zT4_pos = torch.sum(((zT4[:, 0] > 0.0) * (zT4[:, 1]) > 0.0))

        zTs_pos = torch.stack((zT1_pos, zT2_pos, zT3_pos, zT4_pos), dim=0)
        T_index = torch.argmax(zTs_pos, dim=0)
        print("zTs_pos", zTs_pos)
        print("index", T_index)

        transf_pred = None
        if T_index == 0:
            transf_pred = transf1
        elif T_index == 1:
            transf_pred = transf2
        elif T_index == 2:
            transf_pred = transf3
        elif T_index == 3:
            transf_pred = transf4
        """
    se3_mats = torch.stack(se3_mats)
    return se3_mats


"""
f = tensor([ 6.1542e-08, -1.7981e-08, -1.0418e-08,  9.1291e-08, -1.0803e-07,
         6.7861e-01, -4.8542e-01, -4.7422e-01,  2.8102e-01])
res = singular tensor(6.7959e-06)
"""


def meanvar2normtransf(mean, var):
    std = torch.sqrt(var)
    dtype = mean.dtype
    device = mean.device

    normtransf = torch.tensor(
        [
            [1.0 / std[0], 0.0, -mean[0] / std[0] + 1.0],
            [0.0, 1.0 / std[1], -mean[1] / std[1] + 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    return normtransf


def translrot2transf(R, t):
    dtype = R.dtype
    device = R.device
    transf = torch.eye(4, dtype=dtype, device=device)
    transf[:3, :3] = R
    transf[:3, 3] = t

    return transf


"""
def calc_optical_flow_registration(objects_masks, pts1, proj_mats, target_oflow, orig_H, orig_W):
    # pts: 3 x H x W
    # objects_masks: K x H x W

    pts1, pts2, weights = mask_points(objects_masks, pts1, pts2)
    # B x 6
    K, H, W =
    se3_6d_to_se3mat
    se3s
    gauss_newton(x1, )

    import tensor_operations.geometric.se3.transform as o4geo_se3_transf
    import tensor_operations.geometric.pinhole as o4geo_pinhole
    K, H, W = objects_masks.shape
    device = objects_masks.device

    se3s = torch.ones(size=(K, 6), device=device, requires_grad=True)
    se3s_mat = o4geo_se3_transf.se3_6d_to_se3mat(se3s)
    #se3s = torch.eye(4, requires_grad=True, device=device).repeat(1, K, 1, 1)
    # B*K x 3 x H x W
    pts1_ftf = o4geo_se3_transf.pts3d_transform(
        pts1.repeat(K, 1, 1, 1), se3s_mat.reshape(K, 4, 4)
    )

    pred_oflows = o4geo_pinhole.pt3d_2_oflow(pts1_ftf.reshape(1 * K, 3, H, W), proj_mats, orig_H, orig_W).reshape(K, 2, H, W)

    a, b, c = mask_points(objects_masks, target_oflow[None], pred_oflows)

    target_oflow, pred_oflows, weights = mask_points(objects_masks, target_oflow[None], pred_oflows)
    target_oflow = target_oflow.detach()
    se3s = gauss_newton(target_oflow, pred_oflows, weights, se3s)

    return o4geo_se3_transf.se3_6d_to_se3mat(se3s)
"""


def calc_pointsets_registration(objects_masks, pts1, pts2, return_transf_centroid1=False):
    # 3 x H x W
    # mask: K x H x W
    pts1, pts2, weights = mask_points(objects_masks, pts1, pts2)

    return calc_pointsets_registration_from_corresp3d(pts1, pts2, weights, return_transf_centroid1)


def calc_pointsets_registration_from_corresp3d(pts1, pts2, weights=None, return_transf_centroid1=False):
    # pts1, pts2, K x N x 3
    # weights: K x N
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

    if return_transf_centroid1:
        transf_centroid1 = torch.eye(4, device=device).repeat(B, 1, 1)
        transf_centroid1[:, :3, :3] = rot
        transf_centroid1[:, :3, 3] = centroid_pts2 - centroid_pts1
        return transf, transf_centroid1
    else:
        return transf


def filter_sim_se3(objects_se3s, objects_scores, dist_thresh, angle_thresh):
    K = len(objects_se3s)

    objects_se3s_dists, objects_se3s_angles = dist_angle_transfs(
        objects_se3s.repeat(K, 1, 1), objects_se3s.repeat_interleave(K, dim=0)
    )
    objects_se3s_dists = objects_se3s_dists.reshape(K, K)
    objects_se3s_angles = objects_se3s_angles.reshape(K, K)

    # print('dists', objects_se3s_dists)
    # print('angles', objects_se3s_angles)

    objects_se3s_sim = (objects_se3s_dists < dist_thresh) * (objects_se3s_angles < angle_thresh)

    min_devs_ids = (-objects_scores).argsort()
    core_se3s_filtered = []
    unselected = torch.zeros_like(min_devs_ids) == 0
    ids_filtered = []
    for id in min_devs_ids:
        if unselected[id]:
            ids_filtered.append(id)
            unselected[objects_se3s_sim[id]] = False

    ids_filtered = torch.stack(ids_filtered)
    # core_se3s_filtered.append(objects_se3s[id])
    # objects_se3s = core_se3s_filtered
    # objects_se3s = torch.stack(objects_se3s)
    objects_se3s = objects_se3s[ids_filtered]

    return objects_se3s, ids_filtered


def gauss_newton(x1, x2, params, weights):
    # x1, x2: NxC, params: M
    err = torch.norm((x2 - x1) * weights, dim=1)
    print("error - mean:", err.mean())
    params.zero_grad()
    J = torch.autograd.grad(err, params)
    update = -torch.inverse(J.T * J) * J * err

    params += update

    return params


def mask_points(objects_masks, pts1, pts2):
    # pts: 3 x H x W
    # objects_masks: K x H x W
    C = pts1.shape[0]
    K, H, W = objects_masks.shape
    device = pts1.device
    objects_se3s = []
    objects_se3s_devs = []

    objects_masks_counts = objects_masks.flatten(1).sum(dim=1)
    objects_masks_counts_mean = objects_masks_counts.float().mean()

    # pixel_assigned_counts = objects_masks.sum(dim=0, keepdim=True)
    # inverse_depth = 1. / ((pts1[2:, :, :] + pts2[2:, :, :]) / 2.)
    # weights = inverse_depth # (1 / pixel_assigned_counts) *
    # if (inverse_depth.sum(dim=0) == 0.).sum() > 0:
    #    print('weights = 0')
    # weights[torch.isinf(weights)] = 1.0
    weights = torch.ones(size=(1, H, W), device=device)
    # weights = torch.clamp(weights, 0, 1)
    # weights *= inverse_depth
    # weights[:] = 1.0
    weights_list = []
    if (objects_masks_counts_mean == objects_masks_counts).sum() == K:
        objects_masks_counts_mean = objects_masks_counts_mean.int()
        pts1 = (
            pts1[:, None, :, :]
            .repeat(1, K, 1, 1)[:, objects_masks]
            .reshape(C, K, objects_masks_counts_mean)
            .permute(1, 2, 0)
        )
        pts2 = (
            pts2[:, None, :, :]
            .repeat(1, K, 1, 1)[:, objects_masks]
            .reshape(C, K, objects_masks_counts_mean)
            .permute(1, 2, 0)
        )
        weights = (
            weights[:, None, :, :]
            .repeat(1, K, 1, 1)[:, objects_masks]
            .reshape(1, K, objects_masks_counts_mean)
            .permute(1, 2, 0)
        )
    else:
        pts1_list = []
        pts2_list = []
        weights_list = []
        for k in range(K):
            N = objects_masks_counts.max()
            N_k = objects_masks_counts[k]
            pts1_k = pts1[:, objects_masks[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            pts2_k = pts2[:, objects_masks[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            weights_k = weights[:, objects_masks[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            pts1_k = torch.cat((pts1_k, torch.zeros(size=(N - N_k, C), device=device)), dim=0)
            pts2_k = torch.cat((pts2_k, torch.zeros(size=(N - N_k, C), device=device)), dim=0)
            weights_k = torch.cat((weights_k, torch.zeros(size=(N - N_k, 1), device=device)), dim=0)
            pts1_list.append(pts1_k)
            pts2_list.append(pts2_k)
            weights_list.append(weights_k)
        pts1 = torch.stack(pts1_list)
        pts2 = torch.stack(pts2_list)
        weights = torch.stack(weights_list)

    weights = weights[:, :, 0]

    return pts1, pts2, weights
    # pts: K x N x 3
    # se3s, se3s_centroid1 = calc_pointsets_registration(pts1, pts2, weights=weights, return_transf_centroid1=True)
    # pts1_ftf = pts3d_transform(pts1, se3s)

    # objects_masks_new = sflow2mask_rigid.sflow2mask_fit()
    # sflow_dist = torch.norm(pts2 - pts1, dim=2)
    # dev = torch.norm(pts1_ftf - pts2, dim=2)
    # dev_means = dev.mean(dim=1)
    # dev_rel = dev / sflow_dist
    # dev_rel_means = dev_rel.mean(dim=1)

    # for k in range(K):
    #    #if (dev_means[k] < thresh) or (dev_rel_means[k] < thresh_rel):
    #    objects_se3s_devs.append(torch.Tensor([0.]))#dev_means[k])
    #    objects_se3s.append(se3s[k])
    #    # ops_vis.visualize_img(core_mask[None,])

    # objects_se3s = torch.stack(objects_se3s)
    # objects_se3s_devs = torch.stack(objects_se3s_devs)

    # return se3s, se3s_centroid1
