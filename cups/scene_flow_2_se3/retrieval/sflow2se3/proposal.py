import torch

import cups.scene_flow_2_se3.geometric.se3.fit.corresp_3d_3d as o4geo_se3_fit_pt3d_pt3d
import cups.scene_flow_2_se3.visual._2d as o4vis2d


def proposals(sflow, args, drpcs=None, logger=None):
    """Create se3 proposals given scene flow data points and drpcs.

    Parameters
    ----------
        sflow SFlow: data about moving points in 3D
        drpcs DRPCs: models of already chosen Dynamic Rigid Point Clouds (DRPCs)

    Returns
    -------
        drpcs_proposed:
    """

    # 1. find potential points: K x H x W
    # outlier = ~drpcs.calc_inlier()

    # 2. candidates uniformly distributed on 2D image
    pts_dists_spatial_0 = torch.norm(sflow.pt3d_0[:, None, None, :, :] - sflow.pt3d_0[:, :, :, None, None], dim=0)
    pts_dists_spatial_1 = torch.norm(sflow.pt3d_1[:, None, None, :, :] - sflow.pt3d_1[:, :, :, None, None], dim=0)
    pts_dists_temporal = torch.abs(pts_dists_spatial_0 - pts_dists_spatial_1)
    pts_dists_spatial_0 = (pts_dists_spatial_0 > 5.0) * 1.0
    pts_dists_temporal = (pts_dists_temporal > args.sflow2se3_rigid_dist_dev_max) * 1.0

    pts_dists_spatial_0 = (
        pts_dists_spatial_0 + 1.0 * ~sflow.depth_reliable_0[0] + 1.0 * ~sflow.depth_reliable_0[0, None, None]
    )
    pts_dists_temporal = (
        pts_dists_temporal + 1.0 * ~sflow.depth_reliable_01[0] + 1.0 * ~sflow.depth_reliable_01[0, None, None]
    )

    if drpcs is not None:
        inlier = drpcs.sflow_inlier_hard.sum(dim=0) > 0.5
        pts_dists_spatial_0 = pts_dists_spatial_0 + 1.0 * inlier + 1.0 * inlier[None, None]
        pts_dists_temporal = pts_dists_temporal + 1.0 * inlier + 1.0 * inlier[None, None]

    clusters_spatial_temporal = clusters_accumulation(
        num_clusters=100, min_size=2, max_size=5, dists_single=pts_dists_spatial_0, dists_complete=pts_dists_temporal
    )

    if clusters_spatial_temporal is None or clusters_spatial_temporal.size(0) == 0:
        return None
    else:

        if logger is not None and args.eval_visualize_paper:
            if drpcs is not None:
                logger.log_image(
                    o4vis2d.draw_circles_in_rgb(sflow.depth_reliable_01 * ~inlier, img=sflow.rgb),
                    key="paper_points_reliable_outlier/img",
                )
            else:
                logger.log_image(
                    o4vis2d.draw_circles_in_rgb(sflow.depth_reliable_01, img=sflow.rgb),
                    key="paper_points_reliable_outlier/img",
                )

            logger.log_image(
                o4vis2d.draw_circles_in_rgb(clusters_spatial_temporal[:5], img=sflow.rgb),
                key="paper_points_rigid_clusters/img",
            )

        se3prop = o4geo_se3_fit_pt3d_pt3d.fit_se3_to_corresp_3d_3d_and_masks(
            clusters_spatial_temporal, sflow.pt3d_0, sflow.pt3d_1
        )

        return se3prop


def clusters_accumulation(num_clusters, min_size, max_size, dists_single=None, dists_complete=None):
    """Create clusters with samples that fulfill pairwise the distances constraints.

    Parameters
    ----------
    num_clusters int: number of clusters that are instantiated
    min_size int: minimum number of samples in one cluster
    max_size int: maximum number of samples in one cluster
    dists_single torch.Tensor: NxN/HxWxHxW distances that must be small enough to one other sample in the cluster
    dists_complete torch.Tensor: NxN/HxWxHxW distances that must be small enough to all other samples in the cluster

    Returns
    -------
    clusters torch.Tensor: num_clusters x N
    """
    N = None
    H, W = (None, None)
    if dists_single is not None:
        device = dists_single.device
        if dists_single.dim() == 2:
            N = dists_single.size(0)
        else:
            N = dists_single.size(0) * dists_single.size(1)
            H, W = (dists_single.size(0), dists_single.size(1))
    else:
        dists_single = torch.zeros_like(dists_complete)

    if dists_complete is not None:
        device = dists_complete.device
        if dists_complete.dim() == 2:
            N = dists_complete.size(0)
        else:
            N = dists_complete.size(0) * dists_complete.size(1)
            H, W = (dists_complete.size(0), dists_complete.size(1))
    else:
        dists_complete = torch.zeros_like(dists_single)

    dists_single = dists_single.reshape(N, N)
    dists_complete = dists_complete.reshape(N, N)

    clusters_pts_selected = torch.zeros(size=(num_clusters, N), dtype=torch.bool, device=device)

    potential_pts_ids = ((torch.max(dists_single, dists_complete) < 0.5).sum(dim=1) > 2.0).nonzero(as_tuple=True)[0]
    if len(potential_pts_ids) == 0:
        return None
    clusters_new_pts_ids = potential_pts_ids[
        (torch.rand(num_clusters, device=device) * len(potential_pts_ids)).floor().long()
    ]

    clusters_pts_dists_single = dists_single[clusters_new_pts_ids]
    clusters_pts_dists_complete = dists_complete[clusters_new_pts_ids]
    clusters_pts_dists = torch.max(clusters_pts_dists_single, clusters_pts_dists_complete)
    clusters_pts_selected[torch.arange(num_clusters, device=device), clusters_new_pts_ids] = True

    for i in range(max_size - 1):
        clusters_pts_inrange = (clusters_pts_dists < 0.5) * (~clusters_pts_selected)
        clusters_pts_in_range_num = clusters_pts_inrange.sum(dim=1)
        clusters_ids_extendable = clusters_pts_in_range_num.nonzero(as_tuple=True)[0]

        if len(clusters_ids_extendable) == 0:
            break
        clusters_new_pts_ids_offsets = torch.cat(
            (
                torch.tensor([0], device=device, dtype=torch.long),
                clusters_pts_in_range_num[clusters_ids_extendable].cumsum(dim=0)[:-1],
            )
        )
        clusters_new_pts_ids = clusters_pts_inrange.nonzero(as_tuple=True)[1][
            clusters_new_pts_ids_offsets
            + (
                torch.rand(len(clusters_ids_extendable), device=device)
                * clusters_pts_in_range_num[clusters_ids_extendable]
            )
            .floor()
            .long()
        ]
        clusters_pts_dists_single[clusters_ids_extendable] = torch.max(
            clusters_pts_dists_single[clusters_ids_extendable], dists_single[clusters_new_pts_ids]
        )
        clusters_pts_dists_complete[clusters_ids_extendable] = torch.min(
            clusters_pts_dists_complete[clusters_ids_extendable], dists_complete[clusters_new_pts_ids]
        )
        clusters_pts_dists[clusters_ids_extendable] = torch.max(
            clusters_pts_dists_single[clusters_ids_extendable], clusters_pts_dists_complete[clusters_ids_extendable]
        )
        clusters_pts_selected[clusters_ids_extendable, clusters_new_pts_ids] = True

    # ensure min_size
    clusters_pts_selected = clusters_pts_selected[clusters_pts_selected.sum(dim=1) >= min_size]

    if H is not None and W is not None:
        clusters_pts_selected = clusters_pts_selected.reshape(-1, H, W)
    return clusters_pts_selected
