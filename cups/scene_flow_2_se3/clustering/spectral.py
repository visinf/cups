import torch
from sklearn.cluster import SpectralClustering


def spectral_clustering(dist, k):
    device = dist.device
    affinity = torch.exp(-(dist**2) / 0.5)
    affinity = affinity.detach().cpu().numpy()
    # discretize / kmeans
    print("type(affinity)", type(affinity))
    print("affinity.shape", affinity.shape)
    clustering = SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=0, affinity="precomputed").fit(
        affinity
    )
    labels = torch.from_numpy(clustering.labels_).to(device)

    return labels
