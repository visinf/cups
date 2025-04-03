import torch


def pairs_ids_from_mask(mask_pair):
    # in: x: K x XXX  mask: K x K
    # out: pair_id1, pair_id2: #number-pairs

    K = len(mask_pair)
    grid_id1 = torch.arange(K)[:, None].repeat(1, K)
    grid_id2 = torch.arange(K)[None,].repeat(K, 1)

    pairs_id1 = grid_id1[mask_pair]
    pairs_id2 = grid_id2[mask_pair]

    return pairs_id1, pairs_id2


def select_pairs_from_mask(x, mask_pair):

    pairs_id1, pairs_id2 = pairs_ids_from_mask(mask_pair)

    return x[pairs_id1], x[pairs_id2]


def calc_intersect_cross(masks1, masks2, masks_connected=None):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK
    K1 = len(masks1)
    K2 = len(masks2)

    if masks_connected == None:
        masks1 = masks1[:, None]
        masks2 = masks2[None, :]

        intersect = torch.sum(
            masks1.flatten(2) * masks2.flatten(2),
            dim=2,
        )

    else:
        pairs_id1, pairs_id2 = pairs_ids_from_mask(masks_connected)
        masks_pair1 = masks1[pairs_id1]
        masks_pair2 = masks2[pairs_id2]

        pairs_intersect = torch.sum(
            masks_pair1.flatten(1) * masks_pair2.flatten(1),
            dim=1,
        )
        intersect = torch.zeros(size=(K1, K2), dtype=pairs_intersect.dtype, device=pairs_intersect.device)
        intersect[masks_connected] = pairs_intersect

    # K x K
    if masks_connected is not None:
        intersect *= masks_connected

    return intersect


def calc_union_cross(masks1, masks2, masks_connected=None):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK
    K1 = len(masks1)
    K2 = len(masks2)

    if masks_connected == None:
        masks1 = masks1[:, None]
        masks2 = masks2[None, :]

        union = torch.sum(
            masks1.flatten(2) + masks2.flatten(2),
            dim=2,
        )
    else:

        pairs_id1, pairs_id2 = pairs_ids_from_mask(masks_connected)
        masks_pair1 = masks1[pairs_id1]
        masks_pair2 = masks2[pairs_id2]

        pairs_union = torch.sum(
            masks_pair1.flatten(1) + masks_pair2.flatten(1),
            dim=1,
        )
        union = torch.zeros(size=(K1, K2), dtype=pairs_union.dtype, device=pairs_union.device)
        union[masks_connected] = pairs_union

    return union


def calc_prob_same_mask(prob1, prob2, dim_K=0):
    # masks in: K x N or K x H x W

    if prob1.dtype == torch.bool:
        prob1 = prob1.float()

    if prob2.dtype == torch.bool:
        prob2 = prob2.float()

    prob1 = prob1.clone().clamp(0, 1)
    prob2 = prob2.clone().clamp(0, 1)

    intersect = calc_intersect_cross(prob1, prob2)
    union = intersect + calc_intersect_cross(prob1, 1.0 - prob2) + calc_intersect_cross(1.0 - prob1, prob2)

    return intersect / union


def calc_iou_cross(masks1, masks2, masks_connected=None, min_intersect=0):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK

    # size_masks1 = torch.sum(masks1.flatten(1), dim=1)
    # size_masks2 = torch.sum(masks2.flatten(1), dim=1)

    intersect = calc_intersect_cross(masks1, masks2, masks_connected)
    union = calc_union_cross(masks1, masks2, masks_connected)

    # K x K
    iou = intersect / union
    if masks_connected is not None:
        iou *= masks_connected

    iou[intersect < min_intersect] = 0.0

    return iou


def calc_sizes(masks):
    return torch.sum(masks.flatten(1), dim=1)


def calc_ios1(masks1, masks2, masks_connected=None):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK
    K1 = len(masks1)
    K2 = len(masks2)
    intersect = calc_intersect_cross(masks1, masks2, masks_connected)
    size_masks1 = torch.sum(masks1.flatten(1), dim=1)

    # K x K
    ios1 = intersect / size_masks1[:, None].repeat(1, K2)
    ios1[(size_masks1[:, None] == 0).repeat(1, K2)] = 0.0

    if masks_connected is not None:
        ios1 *= masks_connected

    return ios1


def calc_ios2(masks1, masks2, masks_connected=None):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK
    K1 = len(masks1)
    K2 = len(masks2)

    intersect = calc_intersect_cross(masks1, masks2, masks_connected)
    size_masks2 = torch.sum(masks2.flatten(1), dim=1)

    # K x K
    ios2 = intersect / size_masks2[None, :].repeat(K1, 1)
    ios2[(size_masks2[None, :] == 0).repeat(K1, 1)] = 0.0

    if masks_connected is not None:
        ios2 *= masks_connected

    return ios2


def calc_iou(masks, masks_connected=None, min_intersect=0):
    # masks in: K x N or K x H x W
    # masks_connected in: KxK

    return calc_iou_cross(masks, masks, masks_connected=masks_connected, min_intersect=min_intersect)


def random_subset_for_max(masks, N_max):
    """Selects random subset for masks that contain of too many points.

    Parameters
    ----------
    masks torch.Tensor: KxHxW / KxN, bool

    Returns
    -------
    masks torch.Tensor: KxHxW / KxN, bool
    """

    masks = masks.clone()

    shape_in = masks.shape

    N = shape_in[1:].numel()
    masks_too_much = masks[masks.flatten(1).sum(dim=1) > N_max].flatten(1)
    K_too_much = masks_too_much.shape[0]
    indices_default = torch.arange(N, device=masks.device)
    for k in range(K_too_much):
        N_too_much = masks_too_much[k].sum()
        # inlier_too_much_pos = inlier_too_much[k]
        indices_too_much = indices_default[masks_too_much[k]]
        indices_too_much_nullified = indices_too_much[torch.randperm(N_too_much)[N_max:]]
        masks_too_much[k, indices_too_much_nullified] = False
    masks_too_much = masks_too_much.reshape(-1, *shape_in[1:])
    masks[masks.flatten(1).sum(dim=1) > N_max] = masks_too_much

    return masks
