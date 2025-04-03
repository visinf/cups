import numpy as np
import pytorch3d.transforms as t3d
import torch
from util import helpers

import cups.scene_flow_2_se3.loss as oloss
from cups.scene_flow_2_se3.geometric import (
    pt3d_2_oflow,
    pts3d_transform,
    se3_6d_to_se3mat,
)

# from pytorch3d.transforms.so3 import so3_exponential_map


def quat2mat(quaternion):
    w, x, y, z = quaternion
    matrix = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]], dtype=np.float32)

    return matrix


def quat2mat_3x3transp(quaternion):
    w, x, y, z = quaternion
    matrix = np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]], dtype=np.float32)

    return matrix


def calc_transformation_via_optical_flow(pts1, pts2, gt_oflow, mask_valid, proj_mat, reproj_mat):
    pts1_norm = torch.norm(pts1.detach(), dim=0, keepdim=True)
    pts1 = pts1.reshape(3, -1)
    gt_pts1_ftf = pts2
    # gt_pts1_ftf = owarp.warp(pts2.unsqueeze(0), gt_oflow.unsqueeze(0), return_masks_flow_inside=False).squeeze(0)

    CW = int(proj_mat[0, 2])
    CH = int(proj_mat[1, 2])
    # pts1: 3xN
    global_transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    # mask_valid = (pts1[2] < 10.) * (pts2[2] < 10.)
    translation_3d = torch.zeros(size=(3,), dtype=pts1.dtype, device=pts1.device, requires_grad=True)
    rotation_3d = torch.zeros(size=(3,), dtype=pts1.dtype, device=pts1.device, requires_grad=True)

    transform_6d = torch.zeros(size=(6,), dtype=pts1.dtype, device=pts1.device, requires_grad=True)
    # transform_6d[:3] = translation_3d.clone()
    # transform_6d[3:] = rotation_3d.clone()
    # transform_6d = torch.cat((translation_3d, rotation_3d))
    # transform_6d[:3].detach_()
    optimizer = torch.optim.SGD([transform_6d], lr=0.0002, momentum=0.95)
    # optimizer = torch.optim.Adam([transform_6d], lr=0.0002, weight_decay=0)

    pts1_ftf = pts1.clone()
    for it in range(150):
        transform_6d.data.fill_(0.0)

        for it2 in range(1):
            # transl_3d = transform_6d[3:]
            # rot_3d = transform_6d[:3] / 50.
            # transform_6d_passed = torch.cat((rot_3d, transl_3d))
            transform = se3_6d_to_se3mat(transform_6d.unsqueeze(0)).squeeze(0)

            pts1_ftf0 = (
                pts3d_transform(pts1_ftf.unsqueeze(0).permute(0, 2, 1), transform.unsqueeze(0))
                .permute(0, 2, 1)
                .squeeze(0)
            )

            optimizer.zero_grad()

            H, W = mask_valid.shape
            pts1_ftf0 = pts1_ftf0.reshape(1, -1, H, W).squeeze(0)
            oflow = pt3d_2_oflow(pts1_ftf0.unsqueeze(0), proj_mat.unsqueeze(0)).squeeze(0)

            # helpers.visualize_flow(torch.cat((oflow, gt_oflow), dim=1))

            # helpers.visualize_img(mask_valid.unsqueeze(0))
            num_pixels = torch.sum(mask_valid)
            weight_q1 = num_pixels / (4 * torch.sum(mask_valid[:CH, :CW]))
            weight_q2 = num_pixels / (4 * torch.sum(mask_valid[CH:, :CW]))
            weight_q3 = num_pixels / (4 * torch.sum(mask_valid[CH:, CW:]))
            weight_q4 = num_pixels / (4 * torch.sum(mask_valid[:CH, CW:]))
            weight = mask_valid * 1.0
            weight[:CH, :CW] = weight_q1
            weight[CH:, :CW] = weight_q2
            weight[CH:, CW:] = weight_q3
            weight[:CH, CW:] = weight_q4
            # helpers.visualize_img(weight.unsqueeze(0))

            # gt_vec3d = ogeo.oflow_2_vec3d(gt_oflow.unsqueeze(0), reproj_mats=reproj_mat.unsqueeze(0)).squeeze(0)
            # vec3d = ogeo.oflow_2_vec3d(oflow.unsqueeze(0), reproj_mats=reproj_mat.unsqueeze(0)).squeeze(0)
            # corr = 1 - torch.sum(gt_vec3d * vec3d, dim=0)
            loss_corr = 1 * oloss.calc_consistency_oflow_corr3d_loss(
                gt_oflow.unsqueeze(0),
                oflow.unsqueeze(0),
                mask_valid.unsqueeze(0).unsqueeze(0),
                reproj_mat.unsqueeze(0),
            )
            loss_epe = 0.1 * oloss.calc_consistency_oflow_epe2d_loss(
                gt_oflow.unsqueeze(0),
                oflow.unsqueeze(0),
                mask_valid.unsqueeze(0).unsqueeze(0),
            )
            loss = loss_epe  # + loss_corr  #
            # corr = corr.squeeze(0).squeeze(0)
            # loss =(
            #    torch.norm(
            #        #gt_pts1_ftf[:, mask_valid] - pts1_ftf0[:, mask_valid],
            #        #(gt_oflow[:, mask_valid] - oflow[:, mask_valid]) * weight[mask_valid], #/ (1+pts1_norm[:, mask_valid]),
            #        corr[mask_valid],
            #        dim=0,
            #        p=1,
            #    )
            # )
            # problem 1: large displacement leads to
            # problem 2: uneven distribution leads to accumulate error
            # #                       (2d gradient is more likely to fit than 3d ?)
            # loss = loss / (torch.abs(loss).detach() + 1.)
            # / torch.norm(pts1[2:3], dim=0, p=1)**2).mean()

            # loss_vis = torch.ones(size=(H, W), dtype=loss.dtype, device=loss.device) * mask_valid
            # loss_vis[mask_valid] = loss
            # helpers.visualize_img(loss_vis.unsqueeze(0) / torch.max(loss))
            # print('loss min max:', torch.min(loss), torch.max(loss))
            loss = loss.mean()
            loss.backward()

            # transform_6d.grad = torch.clamp(transform_6d.grad, -np.inf, 10)

            print(it2, "loss", loss, "grad", transform_6d.grad)
            optimizer.step()

        transform = se3_6d_to_se3mat(transform_6d.unsqueeze(0)).squeeze(0)
        global_transform = torch.matmul(transform, global_transform).detach()
        pts1_ftf = (
            pts3d_transform(pts1.unsqueeze(0).permute(0, 2, 1), global_transform.unsqueeze(0))
            .permute(0, 2, 1)
            .squeeze(0)
        )

        # visualize_pcds(pts1_ftf0, pts2, mask_valid)

    return global_transform


def calc_transform_between_pointclouds_v2(pts1, pts2, so3_type="exponential", img=None, mask_valid=None):

    # pts1: 3xN
    global_transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    # mask_valid = (pts1[2] < 10.) * (pts2[2] < 10.)
    transform_6d = torch.zeros(size=(6,), dtype=pts1.dtype, device=pts1.device, requires_grad=True)
    optimizer = torch.optim.SGD([transform_6d], lr=0.001, momentum=0.95)

    pts1_ftf = pts1.clone()
    for it in range(1):
        transform_6d.data.fill_(0.0)

        for it2 in range(50):
            transform = se3_6d_to_se3mat(transform_6d.unsqueeze(0), type=so3_type).squeeze(0)

            pts1_ftf0 = (
                pts3d_transform(pts1_ftf.unsqueeze(0).permute(0, 2, 1), transform.unsqueeze(0))
                .permute(0, 2, 1)
                .squeeze(0)
            )

            optimizer.zero_grad()
            if img is not None and mask_valid is not None:
                _, H, W = img.shape
                pts1_ftf0 = pts1_ftf0.reshape(1, -1, H, W)
                pts2 = pts2.reshape(1, -1, H, W)
                loss = helpers.calc_chamfer_loss(
                    pts1_ftf0,
                    pts2,
                    img=img.unsqueeze(0),
                    masks_valid=mask_valid.unsqueeze(0),
                    edge_weight=50,
                    lambda_smoothness=1,
                )
                pts1_ftf0 = pts1_ftf0.reshape(-1, H * W)
                pts2 = pts2.reshape(-1, H * W)
            else:
                # pts1_ftf0 = pts1_ftf0
                loss = (
                    torch.norm(
                        pts1_ftf0[:] - pts2[:],  # , mask_valid.flatten()]  # , mask_valid.flatten()],
                        dim=0,
                        p=2,
                    )
                ).mean()  # / torch.norm(pts1[2:3], dim=0, p=1)**2).mean()

            loss.backward()
            print(it2, transform_6d.grad)
            optimizer.step()

        transform = se3_6d_to_se3mat(transform_6d.unsqueeze(0), type=so3_type).squeeze(0)
        global_transform = torch.matmul(transform, global_transform).detach()
        pts1_ftf = (
            pts3d_transform(pts1.unsqueeze(0).permute(0, 2, 1), global_transform.unsqueeze(0))
            .permute(0, 2, 1)
            .squeeze(0)
        )

        # visualize_pcds(pts1_ftf0, pts2, mask_valid)

    return global_transform


def visualize_pcds(pts1, pts2, mask_valid=None):
    # pts1: 3xN, pts2: 3xN, mask_valid:
    import open3d as o3d

    if mask_valid is not None:
        pts1_ftf0_np = pts1[:, mask_valid.flatten()].detach().cpu().numpy().T
        pts2_np = pts2[:, mask_valid.flatten()].detach().cpu().numpy().T
    else:
        pts1_ftf0_np = pts1.detach().cpu().numpy().T
        pts2_np = pts2.detach().cpu().numpy().T

    pcd_pts2 = o3d.geometry.PointCloud()
    pcd_pts2.points = o3d.utility.Vector3dVector(pts2_np)
    pcd_pts2.paint_uniform_color(np.array([[255.0], [0.0], [0.0]]))

    pcd_pts1_ftf0 = o3d.geometry.PointCloud()
    pcd_pts1_ftf0.points = o3d.utility.Vector3dVector(pts1_ftf0_np)
    pcd_pts1_ftf0.paint_uniform_color(np.array([[0.0], [255.0], [0.0]]))

    o3d.visualization.draw_geometries(
        [pcd_pts1_ftf0, pcd_pts2],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )


def calc_transform_between_pointclouds(pts1, pts2):
    # mask_valid = (pts1[2] < 25.) * (pts2[2] < 25.)
    # pts1 = pts1[:, mask_valid].detach()
    # pts2 = pts2[:, mask_valid].detach()

    global_transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    pts1_ftf = pts1.clone()
    for i in range(20):

        # pts1, pts2: 3 x N
        centroid_pts1 = torch.mean(pts1_ftf, dim=1)
        centroid_pts2 = torch.mean(pts2, dim=1)

        pts1_norm = pts1_ftf - centroid_pts1.unsqueeze(1)  # / torch.norm(pts1, dim=0).unsqueeze(0)**2
        pts2_norm = pts2 - centroid_pts2.unsqueeze(1)  # / torch.norm(pts1, dim=0).unsqueeze(0)**2

        U, S, V = torch.svd(torch.matmul(pts2_norm, pts1_norm.T))

        rot = torch.matmul(U, V.T)

        rot = rot.detach().cpu().numpy()

        transl = centroid_pts2.detach().cpu().numpy() - np.dot(rot, centroid_pts1.detach().cpu().numpy())

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = transl

        transform = torch.from_numpy(transform.astype(np.float32)).to(pts1.device)

        global_transform = torch.matmul(transform, global_transform)
        pts1_ftf = (
            pts3d_transform(pts1.permute(1, 0).unsqueeze(0), global_transform.unsqueeze(0)).squeeze(0).permute(1, 0)
        )

        epe = torch.norm(pts1_ftf - pts2, dim=0)
        N = pts1.shape[1]
        epe_ind = torch.argsort(epe)
        # pts1_ftf = pts1_ftf[:, epe_ind[:int(0.95 * N)]]
        # pts1 = pts1[:, epe_ind[:int(0.95 * N)]]
        # pts2 = pts2[:, epe_ind[:int(0.95 * N)]]

    return global_transform


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


def ransac(pts1, pts2, num_splits=10, perc_outliers=50):
    # 3 x N (10x 10%)
    # 1. split 10 times
    # 2. calc transformations
    # 3. calc epe on best 90%
    # 4. sort transformations

    _, N = pts1.shape
    split_size = N // num_splits
    indices = torch.randperm(N)
    epe_sel_list = []
    transf_list = []
    for i in range(num_splits):
        pts1_split = pts1[:, indices[i * split_size : (i + 1) * split_size]]
        pts2_split = pts2[:, indices[i * split_size : (i + 1) * split_size]]

        transf = calc_transform_between_pointclouds(pts1_split, pts2_split)

        pts1_ftf = pts3d_transform(pts1.permute(1, 0).unsqueeze(0), transf.unsqueeze(0)).squeeze(0).permute(1, 0)

        epe = torch.norm(pts1_ftf - pts2, dim=0)
        epe_sel = epe[torch.argsort(epe)[: int(N * (1.0 - (perc_outliers / 100.0)))]].mean()

        transf_list.append(transf)
        epe_sel_list.append(epe_sel.item())

    transf = transf_list[np.argmin(epe_sel_list)]

    return transf


def calc_chance_no_outlier(
    num,
    num_outliers,
    num_els_split,
):
    chance_no_outlier_in_single_split = 1.0
    for num_drawn in range(num_els_split):
        chance_no_outlier_in_single_split *= (num - num_outliers - num_drawn) / num
    print("single", chance_no_outlier_in_single_split)

    chance_no_outlier_in_splits = 1.0 - (1.0 - chance_no_outlier_in_single_split) ** (num // num_els_split)
    print("total", chance_no_outlier_in_splits)
