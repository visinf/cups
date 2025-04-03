import numpy as np
import quaternion
import torch
from scipy.spatial.transform import Rotation


def calc_transf_from_opticalflow(flow, mask, projection_matrix):
    # number of points to choose R,t from four Options (R1, t1), (R1, -t1), (R2, t1), (R2, -t2)
    N_ver = 1000

    _, H, W = mask.shape
    dtype = projection_matrix.dtype
    device = projection_matrix.device

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

    coords1 = coords1.reshape(3, -1)
    coords2 = coords2.reshape(3, -1)
    mask = mask.reshape(-1)
    # 3 x H*W

    coords1 = coords1[:, mask]
    coords2 = coords2[:, mask]
    N = coords1.shape[1]
    print("using ", N, "out of", H * W)
    # 3 x N

    coords1_ver = coords1[:, :N_ver]
    coords2_ver = coords2[:, :N_ver]

    # normalization
    var1, mean1 = torch.var_mean(coords1, dim=1)
    var2, mean2 = torch.var_mean(coords2, dim=1)

    T1 = meanvar2normtransf(mean1, var1)
    T2 = meanvar2normtransf(mean2, var2)
    # 3 x 3

    coords1 = torch.matmul(T1, coords1)
    coords2 = torch.matmul(T2, coords2)

    s = torch.stack(
        (
            coords1[0] * coords2[0],
            coords1[1] * coords2[0],
            coords2[0],
            coords1[0] * coords2[1],
            coords1[1] * coords2[1],
            coords2[1],
            coords1[0],
            coords1[1],
            torch.ones_like(coords1[1]),
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

    res = torch.sum(torch.abs(torch.sum(coords2 * torch.matmul(F, coords1), dim=0)))
    # print('res',res )

    K = torch.cat(
        (
            projection_matrix,
            torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device),
        ),
        dim=0,
    )
    E = torch.mm(K.T, torch.mm(F, K))

    U, S, V = torch.svd(E)

    # rank enforcement
    S[-1] = 0.0
    E = torch.mm(U, torch.mm(torch.diag(S), V.T))
    U, S, V = torch.svd(E)

    W = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)

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
        sol, _ = torch.lstsq(transf1[:3, 3], A)
        zT1[i] = sol[:2, 0]

        A = torch.stack((coords2_ver[:, i], -coords2_pred2[:, i]), dim=1)
        sol, _ = torch.lstsq(transf2[:3, 3], A)
        zT2[i] = sol[:2, 0]

        A = torch.stack((coords2_ver[:, i], -coords2_pred3[:, i]), dim=1)
        sol, _ = torch.lstsq(transf3[:3, 3], A)
        zT3[i] = sol[:2, 0]

        A = torch.stack((coords2_ver[:, i], -coords2_pred4[:, i]), dim=1)
        sol, _ = torch.lstsq(transf4[:3, 3], A)
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

    return transf_pred


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
        ]
    )

    return normtransf


def translrot2transf(R, t):
    dtype = R.dtype
    device = R.device
    transf = torch.eye(4, dtype=dtype, device=device)
    transf[:3, :3] = R
    transf[:3, 3] = t

    return transf


def quat2mat(quaternion):
    w, x, y, z = quaternion
    matrix = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]], dtype=np.float32)

    return matrix


def quat2mat_3x3transp(quaternion):
    w, x, y, z = quaternion
    matrix = np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]], dtype=np.float32)

    return matrix


def calc_transform_between_pointclouds(pts1, pts2):
    # pts1, pts2: 3 x N
    centroid_pts1 = torch.mean(pts1, dim=1)
    centroid_pts2 = torch.mean(pts2, dim=1)

    pts1 = pts1 - centroid_pts1.unsqueeze(1)
    pts2 = pts2 - centroid_pts2.unsqueeze(1)

    S = torch.matmul(pts1, pts2.T)

    N = torch.tensor(
        [
            [
                S[0, 0] + S[1, 1] + S[2, 2],
                S[1, 2] - S[2, 1],
                S[2, 0] - S[0, 2],
                S[0, 1] - S[1, 0],
            ],
            [
                S[1, 2] - S[2, 1],
                S[0, 0] - S[1, 1] - S[2, 2],
                S[0, 1] + S[1, 0],
                S[2, 0] + S[0, 2],
            ],
            [
                S[2, 0] - S[0, 2],
                S[0, 1] + S[1, 0],
                -S[0, 0] + S[1, 1] - S[2, 2],
                S[1, 2] + S[2, 1],
            ],
            [
                S[0, 1] - S[1, 0],
                S[2, 0] + S[0, 2],
                S[1, 2] + S[2, 1],
                -S[0, 0] - S[1, 1] + S[2, 2],
            ],
        ]
    )
    # 4x4

    eigenvalues, eigenvectors = torch.eig(N, eigenvectors=True)
    q_torch = eigenvectors[:, 0]
    # print('q', q_torch)
    # print('||q||', torch.dot(q_torch, q_torch))

    Q = quat2mat(q_torch.detach().cpu().numpy())
    Q_bar = quat2mat_3x3transp(q_torch.detach().cpu().numpy())
    rot = np.matmul(Q_bar.T, Q)[1:, 1:]
    # print('rot', rot)

    transl = centroid_pts2.detach().cpu().numpy() - np.dot(rot, centroid_pts1.detach().cpu().numpy())
    # print('transl', transl)

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = transl

    return transform
