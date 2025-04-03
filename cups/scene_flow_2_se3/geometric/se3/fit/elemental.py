import torch


def mask_points(masks_in, pts1, pts2, weights_in=None):
    """Mask points and weights so that they are in the required forms for se3 fits.

    Parameters
    ---------
    masks_in torch.Tensor: KxHxW / KxN, bool
    pts1 torch.Tensor: C1xHxW / C1xN, float
    pts2 torch.Tensor: C2xHxW / C2xN, float
    weights_in torch.Tensor: KxHxW / KxN, float

    Returns
    -------
    pts1 torch.Tensor: KxNxC1, float
    pts2 torch.Tensor: KxNxC2, float
    weights_out torch.Tensor: KxN, float
    """

    C1 = pts1.shape[0]
    C2 = pts2.shape[0]

    if masks_in.dim() == 2:
        K, H = masks_in.shape
        W = 1
        masks_in = masks_in.reshape(K, H, W)
        pts1 = pts1.reshape(C1, H, W)
        pts2 = pts2.reshape(C2, H, W)
        if weights_in is not None:
            weights_in = weights_in.reshape(K, H, W)
    else:
        K, H, W = masks_in.shape

    device = pts1.device

    masks_counts = masks_in.flatten(1).sum(dim=1)
    masks_counts_mean = masks_counts.float().mean()

    # pixel_assigned_counts = objects_masks.sum(dim=0, keepdim=True)
    # inverse_depth = 1. / ((pts1[2:, :, :] + pts2[2:, :, :]) / 2.)
    # weights = inverse_depth # (1 / pixel_assigned_counts) *
    # if (inverse_depth.sum(dim=0) == 0.).sum() > 0:
    #    print('weights = 0')
    # weights[torch.isinf(weights)] = 1.0
    if weights_in is None:
        weights_in = torch.ones(size=(K, H, W), device=device)
    # weights = torch.clamp(weights, 0, 1)
    # weights *= inverse_depth
    # weights[:] = 1.0
    weights_list = []
    if (masks_counts_mean == masks_counts).sum() == K:
        masks_counts_mean = masks_counts_mean.int()
        pts1 = pts1[:, None, :, :].repeat(1, K, 1, 1)[:, masks_in].reshape(C1, K, masks_counts_mean).permute(1, 2, 0)
        pts2 = pts2[:, None, :, :].repeat(1, K, 1, 1)[:, masks_in].reshape(C2, K, masks_counts_mean).permute(1, 2, 0)
        weights_out = (
            weights_in[None, :, :, :].repeat(1, 1, 1, 1)[:, masks_in].reshape(1, K, masks_counts_mean).permute(1, 2, 0)
        )
    else:
        pts1_list = []
        pts2_list = []
        weights_list = []
        for k in range(K):
            N = masks_counts.max()
            N_k = masks_counts[k]
            pts1_k = pts1[:, masks_in[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            pts2_k = pts2[:, masks_in[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            weights_k = weights_in[k : k + 1, masks_in[k]].permute(1, 0)  # .repeat(int(torch.ceil(N/N_k)), 1)[:N, :]
            pts1_k = torch.cat((pts1_k, torch.zeros(size=(N - N_k, C1), device=device)), dim=0)
            pts2_k = torch.cat((pts2_k, torch.zeros(size=(N - N_k, C2), device=device)), dim=0)
            weights_k = torch.cat((weights_k, torch.zeros(size=(N - N_k, 1), device=device)), dim=0)
            pts1_list.append(pts1_k)
            pts2_list.append(pts2_k)
            weights_list.append(weights_k)
        pts1 = torch.stack(pts1_list)
        pts2 = torch.stack(pts2_list)
        weights_out = torch.stack(weights_list)

    weights_out = weights_out[:, :, 0]

    return pts1, pts2, weights_out
