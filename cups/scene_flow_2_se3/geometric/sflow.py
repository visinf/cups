import numpy as np
import torch


def get_sflow_relative_to_dist_bounded(pts3d_1, pts3d_2, rot_deg_max, cdim=1):
    # ... x 3 x ...
    sflow = pts3d_2 - pts3d_1
    device = pts3d_1.device
    size_1 = torch.Size(torch.cat([torch.LongTensor(list(pts3d_1.shape))[: cdim + 1], torch.LongTensor([1, -1])]))
    size_2 = torch.Size(torch.cat([torch.LongTensor(list(pts3d_1.shape))[: cdim + 1], torch.LongTensor([-1, 1])]))
    sflow_dev_norm = torch.norm(sflow.reshape(size_1) - sflow.reshape(size_2), dim=cdim)
    pt3d_dev_norm = torch.norm(pts3d_1.reshape(size_1) - pts3d_1.reshape(size_2), dim=cdim)
    sflow_relative_to_dist_bound = get_sflow_relative_to_dist_bounds(rot_deg_max, device=device)
    sflow_rel_bounded = (sflow_dev_norm / (pt3d_dev_norm + 1e-10)) < sflow_relative_to_dist_bound

    # ... x 3 x N x N
    return sflow_rel_bounded


def get_sflow_relative_to_dist_bounds(rot_max_deg, device="cpu"):
    if type(rot_max_deg) == float:
        rot_max_deg = torch.Tensor([rot_max_deg]).to(device)
    else:
        rot_max_deg = rot_max_deg.to(device)
    sflow_relative_to_dist_bound = torch.sqrt(
        (torch.sin(rot_max_deg / 360 * 2 * np.pi)) ** 2 + ((1 - torch.cos(rot_max_deg / 360 * 2 * np.pi)) ** 2)
    )

    return sflow_relative_to_dist_bound
