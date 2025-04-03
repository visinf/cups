import torch

import cups.scene_flow_2_se3.clustering.elemental as o4cluster
import cups.scene_flow_2_se3.masks.elemental as o4masks
import cups.scene_flow_2_se3.vision.vision as o4vis
import cups.scene_flow_2_se3.vision.visualization as o4visual


def batch_filter_size(
    masks,
    min_samples,
    objs_cores=None,
    objs_connected=None,
    objects_se3s=None,
    weight_inverse_counts=False,
    return_filter_valid=False,
):
    B = masks.shape[0]

    out = []
    for b in range(B):
        out.append(
            filter_size(
                masks[b],
                min_samples,
                objs_cores=objs_cores,
                objs_connected=objs_connected,
                objects_se3s=objects_se3s,
                weight_inverse_counts=weight_inverse_counts,
                return_filter_valid=return_filter_valid,
            )
        )

    return out


def filter_size(
    masks,
    min_samples,
    objs_cores=None,
    objs_connected=None,
    objects_se3s=None,
    weight_inverse_counts=False,
    return_filter_valid=False,
):
    # in KxN or KxHxW

    if weight_inverse_counts:
        pixel_weight = 1.0 / masks.sum(dim=0, keepdim=True)
        pixel_weight[torch.isinf(pixel_weight)] = 1.0
        # pixel_weight[:] = 1.
        masks_counts = torch.sum((masks * pixel_weight).flatten(1), dim=1)
    else:
        masks_counts = torch.sum(masks.flatten(1), dim=1)
    masks_valid = masks_counts >= min_samples

    out = []
    out.append(masks[masks_valid])
    K = len(out[0])

    if objs_cores is not None:
        out.append(objs_cores[masks_valid])

    if objs_connected is not None:
        out.append(
            objs_connected[
                masks_valid[None,]
                * masks_valid[
                    :,
                    None,
                ]
            ].reshape(K, K)
        )

    if objects_se3s is not None:
        out.append(objects_se3s[masks_valid])

    if len(out) == 1:
        if return_filter_valid:
            return out[0], masks_valid
        else:
            return out[0]
    else:
        if return_filter_valid:
            return out, masks_valid
        else:
            return out


def filter_masks_valid(masks, valid_masks, connected_masks=None):

    masks = masks[valid_masks]
    K_out = len(masks)

    if connected_masks is not None:
        # connected_in: KxK
        # connected_out: K_filtered x K_filtered
        connected = connected_masks[
            valid_masks[None,]
            * valid_masks[
                :,
                None,
            ]
        ].reshape(K_out, K_out)
        return masks, connected
    else:
        return masks


def filter_masks_ids(masks, valid_ids, connected_masks=None):
    K_in = len(masks)
    valid_masks = torch.zeros(K_in, device=masks.device).to(torch.bool)
    valid_masks[valid_ids] = True

    return filter_masks_valid(masks, valid_masks, connected_masks)


def filter_multiple_assignment(masks, min_samples, max_assignments=1):
    # in: K x H x W
    # out: K x H x W
    # filter multiple assignments

    mask_multiple_assignments = torch.sum(masks, dim=0) > max_assignments
    masks[:, mask_multiple_assignments] = False

    # masks = torch.cat((masks, mask_multiple_assignments[None, ]), dim=0)

    masks = filter_size(masks, min_samples)

    return masks


def filter_match_max_iou(masks):
    ious, sizes = o4masks.calc_iou(masks)

    ious.fill_diagonal_(0.0)
    ious_max_ids = torch.argmax(ious, dim=1)

    masks = masks * masks[ious_max_ids]

    return masks


def filter_intersection(masks, connected, thresh_intersect=0.5):
    # iou: K x K,

    # K1 x K1 x H x W or K1 x K1 x N
    # if masks.dim() == 2:
    #    masks_intersect_valid = masks.repeat(295, 1, 1)[connected * (masks_iou < thresh_intersect)]
    # elif masks.dim() == 3:
    #    masks_intersect_valid = masks.repeat(295, 1, 1, 1)[connected * (masks_iou < thresh_intersect)]

    overlaps, masks_sizes = o4masks.calc_iou(masks)

    pairs_id1, pairs_id2 = o4masks.pairs_ids_from_mask(connected * (overlaps > 0.01))
    masks_pair1 = masks[pairs_id1]
    masks_pair2 = masks[pairs_id2]
    # masks_pair1, masks_pair2 = select_pairs_from_mask(masks, connected)

    masks_intersect = masks_pair1 * masks_pair2
    # masks1_minus_intersect = masks_pair1 - masks_intersect
    # masks2_minus_intersect = masks_pair2 - masks_intersect

    # masks_intersect_connected = connected[pairs_id1] + connected[pairs_id2]

    masks = torch.cat((masks, masks_intersect), dim=0)
    return masks


def filter_unique(masks, min_samples):
    K = len(masks)
    mask_unique = (torch.max(masks, dim=0, keepdim=True)[0] == 1) * (torch.sum(masks, dim=0, keepdim=True) == 1)
    mask_unique = mask_unique.repeat(K, 1, 1)

    masks_unique = masks.clone()
    masks_unique[~mask_unique] = False
    masks_unique_sizes = o4masks.calc_sizes(masks_unique)

    filter_mask = masks_unique_sizes >= min_samples
    return masks[filter_mask], filter_mask


def seq_maxprob(masks_pxl_prob, prob_gain_min, prob_same_mask_max):
    K = len(masks_pxl_prob)
    masks_pxl_prob_in = masks_pxl_prob.clone()
    # masks_pxl_prob = masks_pxl_prob.clone().clamp(0, 1)
    mask_pxl_prob = torch.zeros_like(masks_pxl_prob[0])

    # same_masks_prob = o4masks.calc_prob_same_mask(masks_pxl_prob, masks_pxl_prob)

    mask_unfiltered = torch.ones(size=(K,), device=masks_pxl_prob.device, dtype=torch.bool)
    filter_ids = []
    while True:
        # .clamp(0, 99999)
        prob_gain_max_val, prob_gain_max_id = (
            (((masks_pxl_prob > 0.5) * (mask_pxl_prob[None] <= 0.5))).flatten(1).sum(dim=1).max(dim=0)
        )

        # print(prob_gain_max_val)
        if prob_gain_max_val < prob_gain_min:
            break

        if len(filter_ids) > 0:
            # prob_same_mask = same_masks_prob[prob_gain_max_id, torch.stack(filter_ids)].max()
            prob_same_mask = (
                o4masks.calc_prob_same_mask(
                    (masks_pxl_prob_in[prob_gain_max_id][None,] > 0.5).float(),
                    (masks_pxl_prob_in[torch.stack(filter_ids)] > 0.5).float(),
                )
                .max()
                .item()
            )

        else:
            prob_same_mask = 0.0
        # print("prob_same_mask", prob_same_mask)
        if prob_same_mask <= prob_same_mask_max:

            # o4visual.visualize_img((masks_pxl_prob.clamp(0, 1) - mask_pxl_prob[None].clamp(0, 1)).clamp(0, 1)[prob_gain_max_id].reshape(1, 18, 62))
            filter_ids.append(prob_gain_max_id)
            mask_unfiltered[prob_gain_max_id] = False

            mask_filtered = (masks_pxl_prob[prob_gain_max_id].clone() > 0.5) * (mask_pxl_prob <= 0.5)

            mask_pxl_prob = torch.max(mask_pxl_prob, masks_pxl_prob[prob_gain_max_id])

            masks_pxl_prob[:, mask_filtered] = 0.0
            masks_pxl_prob[prob_gain_max_id, mask_filtered] = 1.0

        else:
            masks_pxl_prob[prob_gain_max_id] = 0.0

    # TODO: could be empty!
    # if len(filter_ids) == 0:
    #    filter_ids = [0]
    filter_ids = torch.stack(filter_ids)
    masks_pxl_prob = masks_pxl_prob[filter_ids]
    return filter_ids, masks_pxl_prob


def filter_max_area_recall(masks, max_recall=0.9, scores=None, masks_connected=None):
    # in: K x N or K x H x W
    # ios1: K x K,
    masks_ios1 = o4masks.calc_ios1(masks, masks)
    masks_sizes = o4masks.calc_sizes(masks)

    # K, such that largest masks are preferred
    if scores == None:
        masks_ids_sorted_largest = (-masks_sizes).argsort()
    else:
        masks_ids_sorted_largest = (-scores).argsort()
    ids_filtered = []
    unselected = torch.zeros_like(masks_ids_sorted_largest) == 0

    total_mask = torch.zeros_like(masks[0])
    for id in masks_ids_sorted_largest:

        if (
            unselected[id] and (torch.sum(masks[id] * total_mask) / masks_sizes[id]) <= max_recall
        ):  # and scores[id] > 10:
            # print('score', scores[id])
            o4visual.visualize_img(o4visual.mask2rgb(masks[id : id + 1]))
            ids_filtered.append(id)
            unselected[masks_ios1[:, id] > max_recall] = False
            total_mask += masks[id]

    ids_filtered = torch.stack(ids_filtered)
    if masks_connected == None:
        masks = filter_masks_ids(masks, valid_ids=ids_filtered, connected_masks=masks_connected)
        return masks, ids_filtered
    else:
        masks, masks_connected = filter_masks_ids(masks, valid_ids=ids_filtered, connected_masks=masks_connected)
        return masks, masks_connected, ids_filtered


def filter_overlap(masks, max_overlap=0.5, masks_connected=None):
    # in: K x N or K x H x W
    # iou: K x K,
    masks_iou = o4masks.calc_iou(masks, masks_connected=masks_connected)
    masks_sizes = o4masks.calc_sizes(masks)

    # K, such that largest masks are preferred
    masks_ids_sorted_largest = (-masks_sizes).argsort()

    ids_filtered = []
    unselected = torch.zeros_like(masks_ids_sorted_largest) == 0
    for id in masks_ids_sorted_largest:
        if unselected[id]:
            ids_filtered.append(id)
        unselected[masks_iou[id] > max_overlap] = False

    ids_filtered = torch.stack(ids_filtered)
    if masks_connected == None:
        masks = filter_masks_ids(masks, valid_ids=ids_filtered, connected_masks=masks_connected)
        return masks, ids_filtered
    else:
        masks, masks_connected = filter_masks_ids(masks, valid_ids=ids_filtered, connected_masks=masks_connected)
        return masks, masks_connected, ids_filtered


def filter_erode(objects_masks, erode_patchsize, erode_threshold, min_samples, valid_pts):
    objects_masks = o4vis.erode(
        objects_masks[
            :,
            None,
        ],
        patch_size=erode_patchsize,
        thresh=erode_threshold,
    )[:, 0]
    # objects_masks = ops_vis.dilate(objects_masks[:, None,], patch_size=3)[:, 0]
    objects_masks = objects_masks * valid_pts
    objects_masks = filter_size(objects_masks, min_samples)

    return objects_masks


"""
def filter_interconnected(objects_masks, dists_div, rigid_dists_max_div, percentage_thresh):
    K, H_down, W_down = objects_masks.shape
    objects_masks = objects_masks.reshape(K, H_down * W_down)
    for k in range(K):
        objects_masks[k] = objects_masks[k]
        dists_div_masked = dists_div[
            objects_masks[k, None, :] * objects_masks[k, :, None]
        ]
        num_pixel = int(dists_div_masked.shape[0] ** 0.5)
        dists_div_masked = dists_div_masked.reshape(num_pixel, num_pixel)
        interconnected_masked = (
            torch.mean(1.0 * (dists_div_masked < rigid_dists_max_div), dim=1) > percentage_thresh
        )
        interconnected = torch.ones_like(objects_masks[0], dtype=torch.bool)
        interconnected[objects_masks[k]] = interconnected_masked
        objects_masks[k] *= interconnected
    objects_masks = objects_masks.reshape(K, H_down, W_down)

    return objects_masks
"""


def filter_interconnected(objs_masks, objs_masks_rigid1, percentage_thresh):
    in_dim = objs_masks.dim()
    if in_dim == 3:
        K, H_down, W_down = objs_masks.shape
        objs_masks = objs_masks.reshape(K, H_down * W_down)

    K = len(objs_masks)
    for k in range(K):
        objs_masks_rigid1_masked = objs_masks_rigid1[objs_masks[k, None, :] * objs_masks[k, :, None]]
        num_pixel = int(objs_masks_rigid1_masked.shape[0] ** 0.5)
        objs_masks_rigid1_masked = objs_masks_rigid1_masked.reshape(num_pixel, num_pixel)
        interconnected_masked = torch.mean(1.0 * objs_masks_rigid1_masked, dim=1) > percentage_thresh
        interconnected = torch.ones_like(objs_masks[0], dtype=torch.bool)
        interconnected[objs_masks[k]] = interconnected_masked
        objs_masks[k] *= interconnected

    if in_dim == 3:
        objs_masks = objs_masks.reshape(K, H_down, W_down)

    return objs_masks
