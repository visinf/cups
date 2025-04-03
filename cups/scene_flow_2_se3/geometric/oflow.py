import torch

from cups.scene_flow_2_se3.geometric.sflow import get_sflow_relative_to_dist_bounds


def get_oflow_relative_to_dist_bounded(oflow, pts3d_1, rot_deg_max, focal_length_max, cdim=1):
    # ... x 2 x ...
    device = pts3d_1.device
    shape_1 = torch.cat([torch.LongTensor(list(pts3d_1.shape))[: cdim + 1], torch.LongTensor([1, -1])])
    shape_2 = torch.cat([torch.LongTensor(list(pts3d_1.shape))[: cdim + 1], torch.LongTensor([-1, 1])])
    size_1 = torch.Size(shape_1)
    size_2 = torch.Size(shape_2)
    pt3d_dev_norm = torch.norm(pts3d_1.reshape(size_1) - pts3d_1.reshape(size_2), dim=cdim)
    shape_1[cdim] = 2
    shape_2[cdim] = 2
    size_1 = torch.Size(shape_1)
    size_2 = torch.Size(shape_2)
    oflow_dev_norm = torch.norm(oflow.reshape(size_1) - oflow.reshape(size_2), dim=cdim)

    shape_1 = torch.cat([torch.LongTensor(list(pts3d_1.shape))[:cdim], torch.LongTensor([1, -1])])
    shape_2 = torch.cat([torch.LongTensor(list(pts3d_1.shape))[:cdim], torch.LongTensor([-1, 1])])
    size_1 = torch.Size(shape_1)
    size_2 = torch.Size(shape_2)
    ids_0 = torch.LongTensor([0]).to(device)
    depth = torch.index_select(pts3d_1, dim=cdim, index=ids_0).abs()
    depth_min = torch.min(depth.reshape(size_1), depth.reshape(size_2))
    sflow_relative_to_dist_bound = get_sflow_relative_to_dist_bounds(rot_deg_max, device=device)
    oflow_rel_bounded = (
        oflow_dev_norm / (pt3d_dev_norm + 1e-10)
    ) * depth_min / focal_length_max <= sflow_relative_to_dist_bound

    # ... x N x N
    return oflow_rel_bounded
