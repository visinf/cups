import numpy as np
import pytorch3d.transforms as t3d
import torch

from cups.scene_flow_2_se3 import elemental as o4
from cups.scene_flow_2_se3.geometric import pinhole as o4geo_pin


def transform_fwd(pt3d_1, se3_mat, cdim=1):
    hpt4d_1 = o4geo_pin.pt3d_2_hpt4d(pt3d_1, cdim=cdim)
    hpt4d_2 = o4.multiply_matrix_vector(se3_mat, hpt4d_1, cdim=cdim)

    ids012 = torch.LongTensor([0, 1, 2]).to(pt3d_1.device)

    pt3d_2 = torch.index_select(hpt4d_2, dim=cdim, index=ids012)

    return pt3d_2


def se3_mat_2_dist_angle_all_pairs(se3_mat, angle_unit="rad"):
    # in * x 4 x 4
    shape_in = se3_mat.shape
    se3_mat = se3_mat.reshape(-1, 4, 4)
    se3_mat_rel = torch.linalg.inv(se3_mat[None,]) @ se3_mat[:, None]
    se3_mat_rel = se3_mat_rel.reshape(*shape_in[:-2], *shape_in[:-2], 4, 4)
    dist, angle = se3_mat_2_dist_angle(se3_mat_rel, angle_unit=angle_unit)

    return dist, angle


def se3_mats_similar(se3_mat, rpe_transl_thresh, rpe_angle_thresh, angle_unit="deg"):
    rpe_transl, rpe_angle = se3_mat_2_dist_angle_all_pairs(se3_mat, angle_unit=angle_unit)
    # rpe_transl.flatten().sort()[0][:500]
    # rpe_angle.flatten().sort()[0][:500]
    connected = (rpe_transl.abs() <= rpe_transl_thresh) * (rpe_transl.abs() <= rpe_angle_thresh)

    return connected


def se3_mat_2_dist_angle(se3_mat, angle_unit="rad"):
    dist = se3_mat_2_dist(se3_mat)
    angle = se3_mat_2_angle(se3_mat, angle_unit=angle_unit)
    return dist, angle


def se3_mat_2_dist(se3_mat):
    # in * x 4 x 4
    shape_in = se3_mat.shape
    norm = torch.norm(se3_mat.reshape(-1, 4, 4)[:, 0:3, 3], dim=1)
    return norm.reshape(*shape_in[:-2])


def se3_mat_2_angle(se3_mat, angle_unit="rad"):
    # in B x 4 x 4
    # an invitation to 3-d vision, p 27
    # return torch.acos(min(1, max(-1, (torch.trace(se3_mat[:, 0:3, 0:3]) - 1) / 2)))
    shape_in = se3_mat.shape
    rot_logs = t3d.so3_log_map(se3_mat.reshape(-1, 4, 4)[:, 0:3, 0:3])
    rot_rads = torch.norm(rot_logs, dim=1).reshape(*shape_in[:-2])
    if angle_unit == "rad":
        return rot_rads
    else:
        rot_degs = rot_rads / (2 * np.pi) * 360
        return rot_degs
