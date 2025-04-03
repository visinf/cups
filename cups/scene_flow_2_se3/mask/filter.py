import torch


def mask_ensure_non_zero(masks, thresh_perc=0.5):

    B, _, H, W = masks.shape

    if torch.sum(masks) < thresh_perc * (B * H * W):
        masks[:] = 1.0

    return masks


# def noise_canceling_via_min_neighbors(masks, kernel_size=3, )
