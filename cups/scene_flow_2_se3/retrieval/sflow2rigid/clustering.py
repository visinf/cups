import tensor_operations.clustering.core_accumulation as o4cluster_core_accumulation
import tensor_operations.clustering.elemental as o4cluster
import tensor_operations.clustering.hierarchical as o4cluster_hierarch
import tensor_operations.masks.filter as o4masks_filter
import torch


def agglomerative(pair_rigid, linkage, max_dist):
    # pair_rigid: B x N x N

    dists = 1.0 - pair_rigid * 1.0
    labels_rigid = o4cluster_hierarch.agglomerative(dists, dist_max=max_dist, linkage=linkage)
    # labels_rigid: B x N

    # in: B x 1 x N
    # out: B x K x 1 x N
    assocs_rigid = o4cluster.label_2_onehot(labels_rigid[:, None], negative_handling="ignore")[:, :, 0]

    # clusters_rigid: BxKxN

    return assocs_rigid


def accumulation_core_complete(
    pair_rigid, min_samples, num_steps, return_core_range=False, largest_k=-1, pair_neighbor=None
):
    # pair_rigid: B x N x N
    B, N, N = pair_rigid.shape
    device = pair_rigid.device

    assocs_clusters_rigid = []
    assocs_cores_rigid = []

    for b in range(B):
        b_assocs_cores_rigid = torch.eye(N).bool().to(device)
        b_pair_rigid = pair_rigid[b]

        b_assocs_cores_range_rigid = b_pair_rigid.clone()
        if pair_neighbor is not None:
            b_pair_neighbor = pair_neighbor[b]
            b_assocs_cores_range_neighbor = b_pair_neighbor.clone()
        else:
            b_pair_neighbor = None
            b_assocs_cores_range_neighbor = None

        # b_assocs_cores_range_rigid, = o4masks_filter.filter_size(b_assocs_cores_range_rigid, min_samples=min_samples, return_filter_valid=True)
        b_assocs_masks_filter_size = torch.sum(b_assocs_cores_range_rigid.flatten(1), dim=1) >= min_samples
        b_assocs_cores_range_rigid = b_assocs_cores_range_rigid[b_assocs_masks_filter_size]
        if b_assocs_cores_range_neighbor is not None:
            b_assocs_cores_range_neighbor = b_assocs_cores_range_neighbor[b_assocs_masks_filter_size]
        b_assocs_cores_rigid = b_assocs_cores_rigid[b_assocs_masks_filter_size]
        # b_assocs_cores_rigid, b_links_clusters_rigid
        # objs_cores = b_assocs_cores_rigid, \
        # objs_connected = b_links_clusters_rigid
        K = len(b_assocs_cores_range_rigid)
        print("masks after filter size", K)

        # pair_rigid, objs_connected, ids_filtered = o4masks_filter.filter_overlap(pair_rigid, max_overlap=1.0, masks_connected=objs_connected)
        # K = len(pair_rigid)
        # print("masks after filter overlap", K)

        b_assocs_cores_range_rigid, b_assocs_cores_rigid = o4cluster_core_accumulation.cluster(
            b_pair_rigid,
            b_assocs_cores_range_rigid,
            masks_cores=b_assocs_cores_rigid,
            num_steps=num_steps,
            pair_neighbor=b_pair_neighbor,
            masks_neighbor_range=b_assocs_cores_range_neighbor,
        )
        K = len(b_assocs_cores_range_rigid)
        print("masks after filter random_core_accumulation", K)

        # pair_rigid, objs_cores, objs_connected = o4masks_filter.filter_size(pair_rigid, min_samples=min_samples, objs_cores=objs_cores, objs_connected=objs_connected)
        # K = len(pair_rigid)
        # print("masks after filter size", K)

        # objs_masks, ids_filtered = ops_mask.filter_overlap(objs_masks, max_overlap=0.9)
        # K = len(objs_masks)
        # print("masks after filter overlap", K)

        assocs_clusters_rigid.append(b_assocs_cores_range_rigid)
        assocs_cores_rigid.append(b_assocs_cores_rigid)

    assocs_clusters_rigid = torch.stack(assocs_clusters_rigid)
    assocs_cores_rigid = torch.stack(assocs_cores_rigid)

    if largest_k > 0:
        assocs_rigid_largest_idx = assocs_clusters_rigid.sum(dim=2).argsort(dim=1, descending=True)[:, :largest_k]
        K = assocs_rigid_largest_idx.shape[1]

        if return_core_range:

            assocs_rigid_largest = torch.zeros(
                size=(B, K, N), dtype=assocs_clusters_rigid.dtype, device=assocs_clusters_rigid.device
            )
            for b in range(B):
                assocs_rigid_largest[b] = assocs_clusters_rigid[b, assocs_rigid_largest_idx[b]]
            assocs_clusters_rigid = assocs_rigid_largest

        else:
            assocs_rigid_largest = torch.zeros(
                size=(B, K, N), dtype=assocs_cores_rigid.dtype, device=assocs_cores_rigid.device
            )
            for b in range(B):
                assocs_rigid_largest[b] = assocs_cores_rigid[b, assocs_rigid_largest_idx[b]]
            assocs_cores_rigid = assocs_rigid_largest

    # else:
    #    assocs_rigid_largest = torch.zeros(size=(B, 0, N), dtype=assocs_rigid.dtype, device=assocs_rigid.device)

    if return_core_range:
        return assocs_clusters_rigid
    else:
        return assocs_cores_rigid
