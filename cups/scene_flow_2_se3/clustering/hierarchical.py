import torch
from sklearn.cluster import AgglomerativeClustering


def agglomerative(dists, dist_max=0.5, linkage="single", dists2d=False):
    # dists: * x N x N / * x H x W x H x W

    if dists2d:
        H, W = dists.size(-2), dists.size(-1)
        dists = dists.reshape(*dists.shape[:-4], H * W, H * W)
    N = dists.shape[-1]
    device = dists.device

    dists_shape_in = dists.shape
    labels_shape_out = dists_shape_in[:-1]

    dists = dists.reshape(dists_shape_in[:-2].numel(), N, N)
    labels_shape_between = dists.shape[:-1]
    labels = torch.zeros(size=labels_shape_between, device=device, dtype=torch.long)
    S = dists.shape[0]

    for s in range(S):
        mask_connected = (dists[s] <= dist_max).sum(dim=1) > 1.0
        # labels = torch.zeros(size=(N,), device=device, dtype=torch.long)
        if mask_connected.sum() > 1:
            dists_connected = dists[s][mask_connected][:, mask_connected]

            dists_connected = dists_connected.detach().cpu().numpy()

            clustering = AgglomerativeClustering(
                affinity="precomputed",
                linkage=linkage,
                distance_threshold=dist_max,
                n_clusters=None,
            ).fit(dists_connected)
            # clustering.labels_

            labels[s][mask_connected] = torch.from_numpy(clustering.labels_).to(device)
            # labels[s][~mask_connected] = labels[s][mask_connected].max() + torch.arange((~mask_connected).sum()).to(device)
            labels[s][~mask_connected] = -1
        else:
            # labels[s][~mask_connected] = labels[s][mask_connected].max() + torch.arange((~mask_connected).sum()).to(device)
            labels[s][:] = -1

    labels = labels.reshape(labels_shape_out)

    if dists2d:
        labels = labels.reshape()
    return labels
