import cv2
import kornia.geometry.epipolar.triangulation as kornia4triang
import numpy as np
import torch

import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pin
from cups.scene_flow_2_se3 import elemental as o4
from cups.scene_flow_2_se3.geometric.se3 import elemental as o4geo_se3
from cups.scene_flow_2_se3.probabilistic.models import gaussian as o4prob_gauss


def se3_from_correspondences_3D_2D(pts1, pxl2, K):
    # pts1: 3xN
    # pts2: 2xN
    # K : 3x3
    device = pts1.device
    dtype = pts1.dtype

    # Nx3
    pts1 = pts1.permute(1, 0).detach().cpu().numpy()
    pxl2 = pxl2.permute(1, 0).detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    retval, r, t, mask_inliers = cv2.solvePnPRansac(pts1, pxl2, K, None)
    R, _ = cv2.Rodrigues(r)

    transf = torch.eye(4, dtype=dtype, device=device)
    transf[:3, :3] = torch.from_numpy(R).type(dtype).to(device)
    transf[:3, 3] = torch.from_numpy(t[:, 0]).type(dtype).to(device)

    return transf


def calc_essential_matrix(pxl1, pxl2, K):
    # 3xN
    device = pxl1.device
    dtype = pxl1.dtype
    pxl1 = pxl1.permute(1, 0).detach().cpu().numpy()
    # Nx3
    pxl2 = pxl2.permute(1, 0).detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    E, mask_inlier = cv2.findEssentialMat(pxl1, pxl2, K, cv2.RANSAC, 0.01, 1.0)
    mask_inlier = mask_inlier[:, 0]
    R1, R2, T = cv2.decomposeEssentialMat(E)

    transf = torch.eye(4, dtype=dtype, device=device)

    for rott in [(R1, T), (R2, T), (R1, -T), (R2, -T)]:
        if test_essential(K, rott[0], rott[1], pxl1[mask_inlier == 1].T, pxl2[mask_inlier == 1].T):
            print("found rotation/translation")
            # if testEss(K0,K1,rott[0],rott[1],hp0_cpu[0,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10], hp1_cpu[i,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10]):
            R = rott[0].T
            t = -R.dot(rott[1][:, 0])

            transf[:3, :3] = torch.from_numpy(R).type(dtype).to(device)
            transf[:3, 2] = torch.from_numpy(t).type(dtype).to(device)
    return transf


def test_essential(K, R, T, p1, p2):
    import cv2

    N = p1.shape[1]
    # p1 = np.concatenate((p1, np.ones((1, N))), 0)
    # p2 = np.concatenate((p2, np.ones((1, N))), 0)
    testP = cv2.triangulatePoints(
        K.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), -1)),
        K.dot(np.concatenate((R, T), -1)),
        p1[:2],
        p2[:2],
    )
    Z1 = testP[2, :] / testP[-1, :]

    p1 = np.concatenate((p1, np.ones((1, N))), 0)
    Z2 = (R.dot(Z1 * np.linalg.inv(K).dot(p1)) + T)[-1, :]
    if ((Z1 > 0).sum() > (Z1 <= 0).sum()) and ((Z2 > 0).sum() > (Z2 <= 0).sum()):
        # print(Z1)
        # print(Z2)
        return True
    else:
        return False


# triangulate depth points from transformation + optical flow + intrinsics
# def triangulate(se3, oflow, proj_mat):
#    pass


def triangulate_multiple_se3(proj_mat, se3_mats, oflow, framework="opencv"):
    K = se3_mats.shape[0]
    pts3d = []
    for k in range(K):
        pts3d_single = triangulate_single_se3(proj_mat, se3_mats[k], oflow, framework)
        pts3d.append(pts3d_single)
    pts3d = torch.stack(pts3d)
    return pts3d


def triangulate(proj_mat, se3_mats, oflow):
    # oflow: Bx2xHxW
    # se3_mat: Bx4x4
    # proj_mat: Bx2x3
    B, _, H, W = oflow.shape
    dtype = oflow.dtype
    device = oflow.device

    P1 = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    P1[:, :2, :3] = proj_mat
    P2 = torch.matmul(P1.clone(), se3_mats)

    P1 = P1[:, :3, :4]
    P2 = P2[:, :3, :4]

    pxl1 = o4geo_pin.shape_2_pxl2d(B=B, H=H, W=W, dtype=dtype, device=device)
    pxl2 = pxl1 + oflow

    # P1, P2: 3x4
    # pxl1, pxl2: Nx2 (note: not normalized)
    pxl1 = pxl1.flatten(2).permute(0, 2, 1)
    pxl2 = pxl2.flatten(2).permute(0, 2, 1)
    pts3d = kornia4triang.triangulate_points(P1, P2, pxl1, pxl2)
    # Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312
    pts3d = pts3d.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    return pts3d


def midpoint_triangulate(x, cam):
    """
    described in: Multiple view geometry in computer vision
    also helpful: https://imkaywu.github.io/blog/2017/07/triangulation/



    Args:
        x:   Set of 2D points in homogeneous coords, (3 x n x N) matrix
        cam: Collection of n objects, each containing member variables
                 cam.P - 3x4 camera matrix [0]
                 cam.R - 3x3 rotation matrix [1]
                 cam.T - 3x1 translation matrix [2]
    Returns:
        midpoint: 3D point in homogeneous coords, (4 x 1) matrix
    """
    n = len(cam)  # No. of cameras
    N = x.shape[-1]

    I = np.eye(3)  # 3x3 identity matrix
    A = np.zeros((3, n))
    B = np.zeros((3, n, N))
    sigma2 = np.zeros((3, N))

    for i in range(n):
        a = -np.linalg.inv(cam[i][:3, :3]).dot(cam[i][:3, -1:])  # ith camera position #
        A[:, i, None] = a
        if i == 0:
            b = np.linalg.pinv(cam[i][:3, :3]).dot(x[:, i])  # Directional vector # 4, N
        else:
            b = np.linalg.pinv(cam[i]).dot(x[:, i])  # Directional vector # 4, N
            b = b / b[3:]
            b = b[:3, :] - a  # 3,N
        b = b / np.linalg.norm(b, 2, 0)[np.newaxis]
        B[:, i, :] = b

        sigma2 = sigma2 + b * (b.T.dot(a).reshape(-1, N))  # 3,N

    Bo = B.transpose([2, 0, 1])
    Bt = B.transpose([2, 1, 0])

    Bo = torch.DoubleTensor(Bo)
    Bt = torch.DoubleTensor(Bt)
    A = torch.DoubleTensor(A)
    sigma2 = torch.DoubleTensor(sigma2)
    I = torch.DoubleTensor(I)

    BoBt = torch.matmul(Bo, Bt)
    C = (n * I)[np.newaxis] - BoBt  # N,3,3
    Cinv = C.inverse()
    sigma1 = torch.sum(A, axis=1)[:, None]
    m1 = I[np.newaxis] + torch.matmul(BoBt, Cinv)
    m2 = torch.matmul(Cinv, sigma2.T[:, :, np.newaxis])
    midpoint = (1 / n) * torch.matmul(m1, sigma1[np.newaxis]) - m2

    midpoint = np.asarray(midpoint)
    return midpoint[:, :, 0].T, np.asarray(Bo)


def cam_and_oflow_2_optical_centers_and_rays(
    extr_cam2, intr_inv_cam2, oflow, intr_inv_cam1=None, extr_cam1=None, cdim=1
):
    """For a given optical flow and.

    Parameters
    ----------
    intr_inv_cam1 torch.Tensor: S1 x ... x Sb x 3 x 3
    extr_cam1 torch.Tensor: S1 x ... x Sb x 4 x 4
    intr_inv_cam2 torch.Tensor: S1 x ... x Sb x 3 x 3
    extr_cam2 torch.Tensor: S1 x ... x Sb x 4 x 4
    oflow torch.Tensor: S1 x ... x Sb x 2 x H x W


    Returns
    -------
    o1 torch.Tensor: S1 x ... x Sb x 3, optical center of camera 1 in global coordinate system
    o2 torch.Tensor: S1 x ... x Sb x 3, optical center of camera 2 in global coordinate system
    v1 torch.Tensor: S1 x ... x Sb x 3 x H x W, rays of camera 1 in global coordinate system
    v2 torch.Tensor: S1 x ... x Sb x 3 x H x W, rays of camera 2 in global coordinate system
    """

    SBs = oflow.shape[:cdim]

    H, W = oflow.shape[cdim + 1 :]
    dtype = oflow.dtype
    device = oflow.device

    if intr_inv_cam1 is None:
        intr_inv_cam1 = intr_inv_cam2

    if extr_cam1 is None:
        extr_cam1 = torch.eye(4, dtype=dtype, device=device).repeat(*SBs, 1, 1)

    ids3 = torch.LongTensor([3]).to(device)
    ids012 = torch.LongTensor([0, 1, 2]).to(device)

    o1 = torch.index_select(torch.index_select(extr_cam1, dim=-1, index=ids3), dim=-2, index=ids012).reshape(*SBs, 3)
    o2 = torch.index_select(torch.index_select(extr_cam2, dim=-1, index=ids3), dim=-2, index=ids012).reshape(*SBs, 3)

    rot1 = torch.index_select(torch.index_select(extr_cam1, dim=-1, index=ids012), dim=-2, index=ids012).reshape(
        *SBs, 3, 3
    )
    rot2 = torch.index_select(torch.index_select(extr_cam2, dim=-1, index=ids012), dim=-2, index=ids012).reshape(
        *SBs, 3, 3
    )

    pxl2d_1 = o4geo_pin.shape_2_pxl2d(B=0, H=H, W=W, dtype=dtype, device=device).repeat(*SBs, 1, 1, 1)
    pxl2d_2 = pxl2d_1 + oflow

    hpt3d_1 = o4geo_pin.pxl2d_2_hpt3d(pxl2d_1, intr_inv_cam1)
    hpt3d_2 = o4geo_pin.pxl2d_2_hpt3d(pxl2d_2, intr_inv_cam2)

    # to get rays in the global coordinate system we have to take the rays from the camera coordinate system and rotate them
    # note: do not translate them!
    # S1 x ... x Sb x 3 x H x W
    rays3d_1 = o4.multiply_matrix_vector(rot1, hpt3d_1)
    rays3d_2 = o4.multiply_matrix_vector(rot2, hpt3d_2)

    return o1, o2, rays3d_1, rays3d_2


def triangulate_single_se3(
    extr_cam2,
    intr_cam2,
    intr_inv_cam2,
    oflow,
    intr_cam1=None,
    intr_inv_cam1=None,
    extr_cam1=None,
    cdim=1,
    pxl_std=None,
    method="dlt",
    dlt_framework="opencv",
    midpoint_z_positive=False,
):
    """

    method str: "dlt", "midpoint"
    """

    if method == "midpoint":
        pt3d_1_pred = triangulate_single_se3_midpoint(
            extr_cam2,
            intr_inv_cam2,
            oflow,
            intr_inv_cam1=intr_inv_cam1,
            extr_cam1=extr_cam1,
            cdim=cdim,
            z_positive=midpoint_z_positive,
        )
    elif method == "dlt":
        pt3d_1_pred = triangulate_single_se3_dlt(
            extr_cam2, intr_cam2, oflow, intr_cam1=intr_cam1, extr_cam1=extr_cam1, cdim=cdim, framework=dlt_framework
        )

    if pxl_std is not None:
        device = oflow.device
        dtype = oflow.dtype
        SBs = oflow.shape[:cdim]
        H, W = oflow.shape[cdim + 1 :]
        if intr_cam1 is None:
            intr_cam1 = intr_cam2

        pxl2d_1_evid = o4geo_pin.shape_2_pxl2d(B=0, H=H, W=W, dtype=dtype, device=device).repeat(*SBs, 1, 1, 1)
        pxl2d_2_evid = pxl2d_1_evid + oflow

        pxl2d_1_pred = o4geo_pin.pt3d_2_pxl2d(pt3d_1_pred, proj_mats=intr_cam1)

        pt3d_2_pred = o4geo_se3.transform_fwd(pt3d_1_pred, se3_mat=torch.linalg.inv(extr_cam2), cdim=cdim)
        pxl2d_2_pred = o4geo_pin.pt3d_2_pxl2d(pt3d_2_pred, proj_mats=intr_cam2)

        pxl2d_1_inlier_prob = o4prob_gauss.calc_inlier_prob(pxl2d_1_pred - pxl2d_1_evid, std=pxl_std)
        pxl2d_2_inlier_prob = o4prob_gauss.calc_inlier_prob(pxl2d_2_pred - pxl2d_2_evid, std=pxl_std)
        pxl2d_12_inlier_prob = pxl2d_1_inlier_prob.prod(dim=cdim, keepdim=True) * pxl2d_2_inlier_prob.prod(
            dim=cdim, keepdim=True
        )

        return pt3d_1_pred, pxl2d_12_inlier_prob
    else:
        return pt3d_1_pred


def triangulate_single_se3_midpoint(
    extr_cam2, intr_inv_cam2, oflow, intr_inv_cam1=None, extr_cam1=None, cdim=1, z_positive=False
):
    """
    described in: Multiple view geometry in computer vision
    also helpful: https://imkaywu.github.io/blog/2017/07/triangulation/
    problem:
        given:
            optical centers o1, o2
            homogeneous coordinates v1, v2
        wanted:
            point in 3D p

    approach:
        1. minimize E = ||(o1 + a1 * v1) - (o2 + a2 * v2)|| w.r.t. a1, a2
            1.1. partial derivatives yield
                    dE / da1 = v1^T 2 * [(o1 + a1 * v1) - (o2 + a2 * v2)] = 0
                    dE / da2 = v2^T 2 * [(o1 + a1 * v1) - (o2 + a2 * v2)] = 0
            1.2. inhomogeneous linear system
                |  v1^T * v1    -v1^T * v2 |   | a1 | =  | v1^T (o2 - o1) |
                | -v2^T * v1     v2^T * v2 |   | a2 | =  | v2^T (o1 - o2) |
            1.3. solution is given by inverting A
                det(A) = [v1^T v1 * v2^T v2 - (v1^T v2)^2]
                | a1 | = 1. / det(A) * | v2^T * v2   v1^T * v2 |  | v1^T (o2 - o1) |
                | a2 | = 1. / det(A) * | v2^T * v1   v1^T * v1 |  | v2^T (o1 - o2) |
        2. p = [(o1 + a1 * v1) + (o2 + a2 * v2)] / 2.

    Parameters
    ----------
    intr_inv_cam1 torch.Tensor: S1 x ... x Sb x 3 x 3
    extr_cam1 torch.Tensor: S1 x ... x Sb x 4 x 4
    intr_inv_cam2 torch.Tensor: S1 x ... x Sb x 3 x 3
    extr_cam2 torch.Tensor: S1 x ... x Sb x 4 x 4
    oflow torch.Tensor: S1 x ... x Sb x 2 x H x W

    Results
    -------
    pt3d torch.Tensor: S1 x ... x Sb x 3 x H x W

    """

    device = oflow.device

    o1, o2, v1, v2 = cam_and_oflow_2_optical_centers_and_rays(
        extr_cam2, intr_inv_cam2, oflow, intr_inv_cam1=intr_inv_cam1, extr_cam1=extr_cam1, cdim=cdim
    )

    v1_v1_dot = (v1 * v1).sum(dim=cdim, keepdim=True)
    v2_v2_dot = (v2 * v2).sum(dim=cdim, keepdim=True)
    v1_v2_dot = (v1 * v2).sum(dim=cdim, keepdim=True)

    oflow_shape = torch.LongTensor(list(oflow.shape))
    o1o2_op_shape = torch.cat(
        (oflow_shape[:cdim], torch.LongTensor([3]), torch.ones(len(oflow_shape[cdim + 1 :]), dtype=torch.long))
    )
    o1o2_op_size = torch.Size(o1o2_op_shape)

    v1_o2o1_dot = (v1 * (o2 - o1).reshape(o1o2_op_size)).sum(dim=cdim, keepdim=True)
    v2_o1o2_dot = (v2 * (o1 - o2).reshape(o1o2_op_size)).sum(dim=cdim, keepdim=True)

    det_A = v1_v1_dot * v2_v2_dot - v1_v2_dot * v1_v2_dot
    det_A = det_A.sign() * (det_A.abs() + 1e-10)
    a1 = (1.0 / det_A) * (v2_v2_dot * v1_o2o1_dot + v1_v2_dot * v2_o1o2_dot)
    a2 = (1.0 / det_A) * (v1_v2_dot * v1_o2o1_dot + v1_v1_dot * v2_o1o2_dot)

    a1_sign = (1.0 / det_A.sign()) * (v2_v2_dot.sign() * v1_o2o1_dot.sign() + v1_v2_dot.sign() * v2_o1o2_dot.sign())
    a2_sign = (1.0 / det_A.sign()) * (v1_v2_dot.sign() * v1_o2o1_dot.sign() + v1_v1_dot.sign() * v2_o1o2_dot.sign())
    a1_nan = a1.isnan()
    a2_nan = a2.isnan()
    a1[a1_nan] = 1e10 * a1_sign[a1_nan]
    a2[a2_nan] = 1e10 * a2_sign[a2_nan]

    # a1 = torch.clamp(a1, min=-1e+10, max=1e+10)
    # a2 = torch.clamp(a2, min=-1e+10, max=1e+10)

    if z_positive:
        mask_a1_negative = a1 < 0.0
        a1[mask_a1_negative] = -a1[mask_a1_negative]
        a2[mask_a1_negative] = -a2[mask_a1_negative]

    pt3d_1_pred = ((o1.reshape(o1o2_op_size) + a1 * v1) + (o2.reshape(o1o2_op_size) + a2 * v2)) / 2.0

    return pt3d_1_pred


# def triangulate_single_se3_dlt(proj_mat, se3_mat, oflow, framework="opencv", pxl_std=None):
def triangulate_single_se3_dlt(extr_cam2, intr_cam2, oflow, intr_cam1=None, extr_cam1=None, cdim=1, framework="opencv"):
    """For a given optical flow and.

    Parameters
    ----------
    extr_cam1 torch.Tensor: S1 x ... x Sb x 4 x 4
    intr_inv_cam2 torch.Tensor: S1 x ... x Sb x 3 x 3
    extr_cam2 torch.Tensor: S1 x ... x Sb x 4 x 4
    oflow torch.Tensor: S1 x ... x Sb x 2 x H x W


    Returns
    -------
    o1 torch.Tensor: S1 x ... x Sb x 3, optical center of camera 1 in global coordinate system
    o2 torch.Tensor: S1 x ... x Sb x 3, optical center of camera 2 in global coordinate system
    v1 torch.Tensor: S1 x ... x Sb x 3 x H x W, rays of camera 1 in global coordinate system
    v2 torch.Tensor: S1 x ... x Sb x 3 x H x W, rays of camera 2 in global coordinate system
    """

    SBs = oflow.shape[:cdim]

    H, W = oflow.shape[cdim + 1 :]
    dtype = oflow.dtype
    device = oflow.device

    if intr_cam1 is None:
        intr_cam1 = intr_cam2

    if extr_cam1 is None:
        extr_cam1 = torch.eye(4, dtype=dtype, device=device).repeat(*SBs, 1, 1)

    # P1: projection from global coordinate frame into coordinate frame from camera 1
    # P2: projection from global coordinate frame into coordinate frame from camera 2
    P1 = o4.multiply_matrix_matrix(o4geo_pin.intr2x3_to_4x4(intr_cam1), torch.linalg.inv(extr_cam1))
    P2 = o4.multiply_matrix_matrix(o4geo_pin.intr2x3_to_4x4(intr_cam2), torch.linalg.inv(extr_cam2))

    P1 = o4geo_pin.proj4x4_to_3x4(P1).expand(*SBs, 3, 4)
    P2 = o4geo_pin.proj4x4_to_3x4(P2).expand(*SBs, 3, 4)

    if framework == "opencv":

        oflow = oflow.reshape(-1, 2, H, W)
        P1 = P1.reshape(-1, 3, 4)
        P2 = P2.reshape(-1, 3, 4)

        B = oflow.shape[0]
        pt3d = []
        for b in range(B):
            oflow_b = oflow[b]
            P1_b = P1[b]
            P2_b = P2[b]
            pxl1_b = o4geo_pin.shape_2_pxl2d(B=0, H=H, W=W, dtype=dtype, device=device)
            pxl2_b = pxl1_b + oflow_b

            # P1, P2: 3x4
            # pxl1, pxl2: Nx2 (note: not normalized)
            P1_b = P1_b.detach().cpu().numpy()
            P2_b = P2_b.detach().cpu().numpy()
            pxl1_b = pxl1_b.flatten(1).detach().cpu().numpy()
            pxl2_b = pxl2_b.flatten(1).detach().cpu().numpy()
            # .reshape(3, H, W)
            pts4d_b = torch.from_numpy(cv2.triangulatePoints(P1_b, P2_b, pxl1_b, pxl2_b)).type(dtype).to(device)
            pts3d_b = (pts4d_b / pts4d_b[3:])[:3].reshape(3, H, W)

            pt3d.append(pts3d_b)

        pt3d = torch.stack(pt3d)
        pt3d.reshape(*SBs, 3, H, W)

    elif framework == "kornia":
        """Kornia batchwise pxl1_b = pxl1_b.flatten(1).permute(1, 0) pxl2_b = pxl2_b.flatten(1).permute(1, 0) pts4d_b =
        kornia4triang.triangulate_points(P1_b, P2_b, pxl1_b, pxl2_b).permute(1, 0) # pts3d_b = (pts4d_b /
        pts4d_b[3:])[:3].reshape(3, H, W) pts3d_b = pts4d_b.reshape(3, H, W) # x = PX, x'=P'X.

        # cross(x, PX) = 0 , cross(x', P'X) = 0
        # x_2 * (PX)_3 - x_3 * (PX)_2 = 0
        """
        pxl1 = o4geo_pin.shape_2_pxl2d(B=0, H=H, W=W, dtype=dtype, device=device).repeat(*SBs, 1, 1, 1)
        pxl2 = pxl1 + oflow
        pxl1 = pxl1.flatten(-2).swapaxes(-2, -1)
        pxl2 = pxl2.flatten(-2).swapaxes(-2, -1)
        pts4d = kornia4triang.triangulate_points(P1, P2, pxl1, pxl2)  # .permute(1, 0)
        pt3d = pts4d.swapaxes(-2, -1).reshape(*SBs, 3, H, W)

    else:
        print("error: unknown framework ", framework)

    return pt3d
