import torch
from sklearn.cluster import DBSCAN


def fit(dists, dist_max, core_min_samples):
    # dists = N x N

    shape_in = dists.shape
    device = dists.device
    N = shape_in[-1]
    dists = dists.reshape(-1, N, N)
    K = dists.shape[0]

    labels = torch.zeros(size=(K, N), dtype=torch.int64, device=device)
    dists = dists.detach().cpu().numpy()
    for k in range(K):

        clustering = DBSCAN(eps=dist_max, min_samples=core_min_samples, metric="precomputed").fit(dists[k])

        labels[k] = torch.from_numpy(clustering.labels_).to(device)

    labels = labels.reshape(*shape_in[:-2], N)
    # outlier = (labels == -1)

    return labels  # , outlier
