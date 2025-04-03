import torch.linalg

import cups.scene_flow_2_se3.vision.visualization as o4visual


def fit(data, models_posterior):
    # pts: B x C x H x W / B x C x N
    # models_likelihood: B x K x H x W
    # means, stds: B x K x C
    # out: argmax(P(H|E)) B x K x C

    pts = data["pts"]
    B, K, H, W = models_posterior.shape
    # models_posterior = (models_posterior > 0.5) * 1.0
    # B x K x 3
    centroid = (models_posterior[:, :, None] * pts[:, None]).flatten(3).sum(dim=3) / (
        models_posterior[:, :, None].flatten(3).sum(dim=3) + 1e-8
    )
    # B x K x 3 x N
    dev = (models_posterior[:, :, None] * (pts[:, None] - centroid[:, :, :, None, None])).flatten(3)
    dev = dev.permute(0, 1, 3, 2)
    # B x K x N x 3

    U, stds, axesT = torch.linalg.svd(dev)
    # B x K x 3
    axes = axesT.permute(0, 1, 3, 2)

    # B x K x N x 3
    dev_proj = dev @ axes
    # B x K x 3
    dev_max, _ = dev_proj.abs().max(dim=2)
    measures = dev_max * 2
    params = {}
    params["center"] = centroid
    params["axes"] = axes
    params["measures"] = measures
    params["stds"] = stds
    params["pts"] = []

    for b in range(B):
        params["pts"].append([])
        for k in range(K):
            params["pts"][b].append(pts[b, :, models_posterior[b, k] > 0.5])

    o4visual.visualize_boxes3d(params)
    # +- 1 std: 68.27%
    # +- 2 std: 95.45%
    # +- 3 std: 99.73%
    return params
