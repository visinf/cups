def j_linkage(masks, masks_cores, masks_connected=None, iou_min_thresh=0.5, min_samples=5):
    K_in = len(masks)
    K_out = 2
    if masks_connected is not None:
        masks_connected = masks_connected.tril(diagonal=-1)
    ious, sizes = calc_iou(masks, masks_connected=masks_connected, min_intersect=min_samples)
    ious.fill_diagonal_(0.0)

    masks_done = torch.Tensor()

    while K_out < K_in and K_out > 1:
        K_in = len(masks)

        # print('K_in', K_in)
        masks, masks_cores, ious = j_linkage_single_step(masks, masks_cores, ious, iou_min_thresh, min_samples)

        done = (ious > iou_min_thresh).sum(dim=1) == 0

        masks_done = torch.cat((masks[done], masks_done.bool()), dim=0)
        # print('done.sum()', done.sum())
        masks = masks[~done]
        masks_cores = masks_cores[~done]
        ious = ious[~done][:, ~done]
        # print('len masks', len(masks), 'len ious', len(ious))
        K_out = len(masks)

    masks = torch.cat((masks_done.bool(), masks.bool()), dim=0)
    return masks


def j_linkage_single_step(masks, masks_cores, ious, iou_min_thresh, min_samples):
    K_in = len(masks)
    ious_max_vals, ious_max_ids = torch.max(ious, dim=1)
    ious_max_val, ious_max_id1 = torch.max(ious_max_vals, dim=0)

    if ious_max_val > iou_min_thresh:
        ious_max_id2 = ious_max_ids[ious_max_id1]

        if ious_max_id2 < ious_max_id1:
            buf = ious_max_id1
            ious_max_id1 = ious_max_id2
            ious_max_id2 = buf
            # print('id1', ious_max_id1, 'id2', ious_max_id2)
        elif ious_max_id1 == ious_max_id2:
            print("error: id1==id2", ious_max_id1)

        masks[ious_max_id1] *= masks[ious_max_id2]
        masks_cores[ious_max_id1] += masks_cores[ious_max_id2]

        masks_connected = torch.zeros_like(ious).bool()
        masks_connected[ious_max_id1] = True
        ious_new, _ = calc_iou(masks, masks_connected=masks_connected, min_intersect=min_samples)
        ious_core_new = calc_ios1(masks_cores, masks, masks_connected=masks_connected)
        ious_new *= ious_core_new == 1.0
        ious[ious_max_id1, :] = ious_new[ious_max_id1]
        ious[:, ious_max_id1] = ious_new[ious_max_id1]
        ious[ious_max_id1, ious_max_id1] = 0.0

        masks = torch.cat((masks[:ious_max_id2], masks[ious_max_id2 + 1 :]), dim=0)
        masks_cores = torch.cat((masks_cores[:ious_max_id2], masks_cores[ious_max_id2 + 1 :]), dim=0)
        # print('cores-sum', masks_cores.sum(dim=1).sort(descending=True)[0][:10])
        ious = torch.cat((ious[:ious_max_id2], ious[ious_max_id2 + 1 :]), dim=0)
        ious = torch.cat((ious[:, :ious_max_id2], ious[:, ious_max_id2 + 1 :]), dim=1)

    return masks, masks_cores, ious
