import numpy as np
import tensor_operations.clustering.elemental as o4cluster
import tensor_operations.clustering.hierarchical as o4cluster_hierarch
import tensor_operations.geometric.euclidean as o4geo_eucl
import tensor_operations.geometric.pinhole as o4pinhole
import tensor_operations.masks.filter as o4masks_filter
import tensor_operations.probabilistic.elemental as o4prob
import tensor_operations.probabilistic.models.euclidean_nn as o4prob_eucl_nn
import tensor_operations.probabilistic.models.gaussian as o4prob_gauss
import tensor_operations.retrieval.sflow2rigid.clustering as o4ret_sflow2rigid_cluster
import tensor_operations.vision.visualization as o4visual
import torch
from tensor_operations.geometric import sflow as o4geo_sflow


def pts3d_2_depth_avg_temp(pts3d_1_down, pts3d_2_down):
    # in: B x 3 x H x W   or   B x 3 x N

    # out: B x H x W or B x N
    depth_avg_temp = (pts3d_1_down[:, 2] + pts3d_2_down[:, 2]) / 2.0
    depth_avg_temp = depth_avg_temp.flatten(1)

    # out B x N
    return depth_avg_temp


def pts3d_2_depth_avg_temp_spatial(pts3d_1_down, pts3d_2_down):
    # in: B x 3 x H x W   or   B x 3 x N
    depth_avg_temp = pts3d_2_depth_avg_temp(pts3d_1_down, pts3d_2_down)
    # B x H x W or B x N

    depth_avg_temp = depth_avg_temp.flatten(1)

    depth_avg_temp_spatial = (depth_avg_temp[:, :, None] + depth_avg_temp[:, None, :]) / 2.0

    # out: B x N x N
    return depth_avg_temp_spatial


def rigid_dists2dists_dev(dist1, dist2, depth_avg_temp, mask_valid):
    # dist1 : B x N x N   or   B x N
    # sflow_dist: B x N
    # mask_valid: B x 1 x H x W  or  B x N

    # B x N
    mask_invalid = ~mask_valid.flatten(1)
    dist_dev = torch.abs(dist2 - dist1)

    dist_dim_in = dist1.dim()

    if dist_dim_in == 3:
        dist_dev_rel = dist_dev / ((depth_avg_temp[:, None, :] + depth_avg_temp[:, :, None]) / 2)

        dist_dev[mask_invalid, :] = 999999
        dist_dev = dist_dev.permute(0, 2, 1)
        dist_dev[mask_invalid, :] = 999999
        dist_dev = dist_dev.permute(0, 2, 1)

        dist_dev_rel[mask_invalid, :] = 999999
        dist_dev_rel = dist_dev_rel.permute(0, 2, 1)
        dist_dev_rel[mask_invalid, :] = 999999
        dist_dev_rel = dist_dev_rel.permute(0, 2, 1)
    else:
        dist_dev_rel = None
        print(
            "error: retrieval.sflow2mask_rigid.elemental.distance2distances_dev - dim in: ",
            dist_dim_in,
        )

    return dist_dev, dist_dev_rel


def dev2mask_valid(dist_dev, dist_dev_rel, dist_dev_max, dist_dev_rel_max):
    # dist_dev: N x N

    mask_valid = (dist_dev < dist_dev_max) + (dist_dev_rel < dist_dev_rel_max)

    return mask_valid


def dev2prob(dist_dev, dist_dev_rel, dist_dev_max, dist_dev_rel_max):
    dev_std = dist_dev_max / 3.0 + 1e-8
    dev_rel_std = dist_dev_rel_max / 3.0 + 1e-8

    prob_dev = o4prob.dev_2_prob_gaussian(dist_dev, dev_std)
    prob_dev_rel = o4prob.dev_2_prob_gaussian(dist_dev_rel, dev_rel_std)

    if dist_dev_max > 0.0 and dist_dev_rel_max > 0.0:
        prob = torch.max(prob_dev, prob_dev_rel)
    elif dist_dev_max > 0.0:
        prob = prob_dev
    elif dist_dev_rel_max > 0.0:
        prob = prob_dev_rel
    else:
        print("error: no std > 0.")

    return prob


def rigid_dists2mask_valid(dist1, dist2, args):
    mask_valid = (
        (dist1 < args.sflow2se3_rigid_dist_max)
        * (dist1 > args.sflow2se3_rigid_dist_min)
        * (dist2 < args.sflow2se3_rigid_dist_max)
        * (dist2 > args.sflow2se3_rigid_dist_min)
    )

    return mask_valid


def sflow2rigid_pairwise(
    pts3d_0_down,
    pts3d_1_down,
    mask_valid_down,
    args,
    return_pair_neighbor=False,
    return_rigid_dist_dev=False,
):
    # pts3d_1_down:    B x 3 x H x W
    # mask_valid_down: B x 1 x H x W

    # B x N x N
    rigid_dist_0 = o4geo_eucl.pts3d_2_dists_eucl(pts3d_0_down)
    rigid_dist_1 = o4geo_eucl.pts3d_2_dists_eucl(pts3d_1_down)

    # sflow_dist = pts2sflow_distances(pts3d_1_down, pts3d_2_down)
    depth_avg_temp = pts3d_2_depth_avg_temp(pts3d_0_down, pts3d_1_down)

    rigid_dist_dev, rigid_dist_dev_rel = rigid_dists2dists_dev(
        rigid_dist_0, rigid_dist_1, depth_avg_temp, mask_valid_down
    )
    # rigid_dist_dev_valid = rigid_dist_dev[:, mask_valid_down.flatten()][:, :, mask_valid_down.flatten()]

    mask_connected = rigid_dists2mask_valid(rigid_dist_0, rigid_dist_1, args) * dev2mask_valid(
        rigid_dist_dev,
        rigid_dist_dev_rel,
        args.sflow2se3_rigid_dist_dev_max,
        args.sflow2se3_rigid_dist_dev_rel_max,
    )

    res = []
    # B x N x N
    res.append(mask_connected)
    if return_pair_neighbor:
        # depth_0_avg = \
        # ((torch.abs(pts3d_0_down[:, 2].flatten(1)[:, None, :]) + torch.abs(pts3d_0_down[:, 2].flatten(1)[:, :, None])) / 2)
        # inlier_hard = o4prob_eucl_nn.calc_inlier_hard(dist=rigid_dist_0 / (depth_0_avg + 1e-10),
        #                                              std=args.sflow2se3_model_euclidean_nn_dist_std_per_depth_m,
        #                                              hard_threshold=args.sflow2se3_model_euclidean_nn_dist_inlier_hard_threshold)
        uvz_rel_0 = o4geo_eucl.pt3d_2_dev_uvz_rel(pts3d_0_down)

        inlier_hard = o4prob_eucl_nn.calc_inlier_hard(
            dist=uvz_rel_0,
            std_depth=args.sflow2se3_model_euclidean_nn_rel_depth_dev_std,
            std_uv=args.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std,
            hard_threshold=args.sflow2se3_model_inlier_hard_threshold,
        )[:, 0]

        res.append(inlier_hard)
    if return_rigid_dist_dev:
        res.append(rigid_dist_dev)
        res.append(rigid_dist_dev_rel)
        # return mask_connected, rigid_dist_dev, rigid_dist_dev_rel

    if len(res) == 1:
        res = res[0]
    return res


def sflow2rigid_cluster(pts3d_1, pts3d_2, mask_valid_pairs, args):
    # pts3d_1, pts3d_2: Bx3xHxW

    B, _, H, W = pts3d_1.shape
    N = H * W
    pair_rigid, pair_neighbor, pair_dist_dev, pair_dist_dev_rel = sflow2rigid_pairwise(
        pts3d_1, pts3d_2, mask_valid_pairs, args, return_pair_neighbor=True, return_rigid_dist_dev=True
    )

    # pair_rigid: BxNxN
    # if args.sflow2se3_visualize_mask_se3_progression:
    #    o4visual.visualize_imgs(
    #        pair_rigid.reshape(B, N, H, W)[0, :, None],
    #        #mask_overlay=gt_mask_rgb_down,
    #    )

    if args.sflow2se3_rigid_clustering_add_sflow_bound:
        sflow_rel_bounded = o4geo_sflow.get_sflow_relative_to_dist_bounded(
            pts3d_1, pts3d_2, rot_deg_max=args.sflow2se3_rot_deg_max, cdim=1
        )
        pair_rigid = pair_rigid * sflow_rel_bounded

    if args.sflow2se3_rigid_clustering_method == "agglomerative":
        # B x N
        assocs_rigid = o4ret_sflow2rigid_cluster.agglomerative(
            pair_rigid,
            linkage=args.sflow2se3_rigid_clustering_agglomerative_linkage,
            max_dist=args.sflow2se3_rigid_clustering_agglomerative_max_dist,
        )

    elif args.sflow2se3_rigid_clustering_method == "accumulation":
        # args.sflow2se3_min_samples = args.sflow2se3_rigid_clustering_accumulation_max_samples

        if args.sflow2se3_rigid_clustering_accumulation_req_neighbor:
            assocs_rigid = o4ret_sflow2rigid_cluster.accumulation_core_complete(
                pair_rigid=pair_rigid,
                num_steps=args.sflow2se3_rigid_clustering_accumulation_max_samples,
                min_samples=args.sflow2se3_min_samples,
                return_core_range=args.sflow2se3_rigid_clustering_accumulation_use_range,
                largest_k=args.sflow2se3_rigid_clustering_accumulation_largest_k,
                pair_neighbor=pair_neighbor,
            )
        else:
            assocs_rigid = o4ret_sflow2rigid_cluster.accumulation_core_complete(
                pair_rigid=pair_rigid,
                num_steps=args.sflow2se3_rigid_clustering_accumulation_max_samples,
                min_samples=args.sflow2se3_min_samples,
                return_core_range=args.sflow2se3_rigid_clustering_accumulation_use_range,
                largest_k=args.sflow2se3_rigid_clustering_accumulation_largest_k,
                pair_neighbor=None,
            )

    elif args.sflow2se3_rigid_clustering_method == "pairs":
        assocs_rigid = []
        for b in range(B):
            b_assocs_rigid_ids = torch.tril(pair_rigid[b], diagonal=-1).nonzero()
            K = b_assocs_rigid_ids.shape[0]
            b_assocs_rigid = torch.zeros(size=(K, N), dtype=pair_rigid.dtype, device=pair_rigid.device)
            mask_id1 = o4cluster.label_2_onehot(b_assocs_rigid_ids[:, 0], label_min=0, label_max=N - 1).permute(1, 0)
            mask_id2 = o4cluster.label_2_onehot(b_assocs_rigid_ids[:, 1], label_min=0, label_max=N - 1).permute(1, 0)
            b_assocs_rigid += mask_id1
            b_assocs_rigid += mask_id2
            assocs_rigid.append(b_assocs_rigid)
        assocs_rigid = torch.stack(assocs_rigid)
        # b_assocs_cores_rigid = torch.eye(N).bool().to(device)

        # models_eucl_likelihood, objects_origin_id = o4sflow2se3_fusion_se3_eucl.fusion_se3_eucl(rigid_clusters[None], pts3d_1_down, max_eucl_dist=3.0)
        # rigid_clusters = models_eucl_likelihood[0]
        # rigid_clusters = o4masks_filter.filter_size(rigid_clusters, min_samples=min_samples)
        # K = len(rigid_clusters)
        # objs_cores = torch.eye(K).bool().to(rigid_clusters.device)

        # rigid_clusters = init_j_linkage(objs_masks=rigid_clusters, objs_cores=objs_cores, objs_connected=pts_connected, min_samples=min_samples)

        # rigid_clusters = init_random_core_accumulation(objs_masks=rigid_clusters, objs_cores=rigid_clusters, objs_connected=rigid_clusters, min_samples=min_samples)

        # rigid_clusters = init_dbscan(rigid_clusters[None], H = H_down, W = W_down, min_samples=min_samples)[0]

        # rigid_clusters = init_max_association_seq(rigid_clusters, min_samples=min_samples)

    else:
        print("error: unknown rigid clustering method ", args.sflow2se3_rigid_clustering)
        return None

    assocs_rigid = assocs_rigid.reshape(B, -1, H, W)

    assocs_rigid = o4masks_filter.batch_filter_size(assocs_rigid, min_samples=args.sflow2se3_min_samples)

    return assocs_rigid
