import torch

import cups.scene_flow_2_se3.masks.elemental as o4masks


def greedy_search(
    pot_masks_pxl_prob, prob_gain_min, prob_same_mask_max=1.0, prev_extr_masks_pxl_prob=None, max_new_masks=-1
):
    """Search greedily for new masks without overlap, minimum probability gain and maximum number of new masks.

    Parameters
    ----------
    pot_masks_pxl_prob torch.Tensor: KxHxW
    extracted_masks_pxl_prob torch.Tensor: K0xHxW
    prob_gain_min float: min gain of the sum over all probabilities for each mask

    Returns
    -------
    sel_masks_ids_onehot torch.Tensor: K, dtype=bool
    """

    if len(pot_masks_pxl_prob) == 0:
        sel_masks_ids = torch.zeros(size=(0,), dtype=torch.int, device=pot_masks_pxl_prob.device)
        return sel_masks_ids

    if prev_extr_masks_pxl_prob is None or len(prev_extr_masks_pxl_prob) == 0:
        K0 = 0
        mask_pxl_prob = torch.zeros_like(pot_masks_pxl_prob[0])
    else:
        K0 = len(prev_extr_masks_pxl_prob)
        mask_pxl_prob, _ = torch.max(prev_extr_masks_pxl_prob, dim=0)

    K = len(pot_masks_pxl_prob)
    pot_masks_pxl_prob = pot_masks_pxl_prob.clone()
    masks_pxl_prob_in = pot_masks_pxl_prob.clone()

    sel_masks_ids = []
    sel_prob_gains_pxlwse = []
    while True:
        # .clamp(0, 99999)
        prob_gain_max_val, prob_gain_max_id = (
            ((((pot_masks_pxl_prob.float() - mask_pxl_prob[None,].float())).clamp(0.0, 1.0)))
            .flatten(1)
            .sum(dim=1)
            .max(dim=0)
        )

        # print(prob_gain_max_val)
        if prob_gain_max_val < prob_gain_min or (max_new_masks > 0 and len(sel_masks_ids) >= max_new_masks):
            break

        if len(sel_masks_ids) > 0 or K0 > 0:
            # prob_same_mask = same_masks_prob[prob_gain_max_id, torch.stack(filter_ids)].max()
            if len(sel_masks_ids) > 0:
                total_extr_masks_pxl_prob = masks_pxl_prob_in[torch.stack(sel_masks_ids)]

            if K0 > 0:
                if len(sel_masks_ids) > 0:
                    total_extr_masks_pxl_prob = torch.cat((prev_extr_masks_pxl_prob, total_extr_masks_pxl_prob), dim=0)
                else:
                    total_extr_masks_pxl_prob = prev_extr_masks_pxl_prob

            prob_same_mask = (
                o4masks.calc_prob_same_mask((masks_pxl_prob_in[prob_gain_max_id][None,]), total_extr_masks_pxl_prob)
                .max()
                .item()
            )
        else:
            prob_same_mask = 0.0
        # print("prob_same_mask", prob_same_mask)
        if prob_same_mask <= prob_same_mask_max:

            # o4visual.visualize_img((masks_pxl_prob.clamp(0, 1) - mask_pxl_prob[None].clamp(0, 1)).clamp(0, 1)[prob_gain_max_id].reshape(1, 18, 62))
            sel_masks_ids.append(prob_gain_max_id)

            # mask_filtered = (pot_masks_pxl_prob[prob_gain_max_id].clone() > 0.5) * (mask_pxl_prob <= 0.5)

            mask_pxl_prob = torch.max(mask_pxl_prob, pot_masks_pxl_prob[prob_gain_max_id])

            # pot_masks_pxl_prob[:, mask_filtered] = 0.0
            # pot_masks_pxl_prob[prob_gain_max_id, mask_filtered] = 1.0
            """
            if len(sel_masks_ids) > 1:
                total_extr_masks_pxl_prob = masks_pxl_prob_in[torch.stack(sel_masks_ids)]
                count_masks = len(sel_masks_ids)
                no_contrib_ids = []
                for i in range(count_masks):

                    not_sel_masks_ids = list(range(0, i)) + list(range(i+1, count_masks))
                    not_sel_masks_ids = [id for id in not_sel_masks_ids if id not in no_contrib_ids]
                    mask_pxl_prob_without_sel = total_extr_masks_pxl_prob[not_sel_masks_ids].float().max(dim=0)[0]
                    #not_sel_masks_ids = torch.LongTensor(list(range(0, i)) + list(range(i+1, len(sel_masks_ids)))).to(total_extr_masks_pxl_prob.device)
                    prob_gain = (total_extr_masks_pxl_prob[i].float() - mask_pxl_prob_without_sel).clamp(0., 1.).flatten(1).sum()
                    if prob_gain < prob_gain_min:
                        no_contrib_ids.append(i)
                        mask_pxl_prob = mask_pxl_prob_without_sel
                no_contrib_ids.reverse()
                for i in no_contrib_ids:
                    sel_masks_ids = sel_masks_ids[:i] + sel_masks_ids[i + 1:]
            """

        else:
            pot_masks_pxl_prob[prob_gain_max_id] = 0.0

    # TODO: could be empty!
    # if len(filter_ids) == 0:
    #    filter_ids = [0]

    if len(sel_masks_ids) > 0:
        sel_masks_ids = torch.stack(sel_masks_ids)
        # pot_masks_pxl_prob = pot_masks_pxl_prob[filter_ids]

        sel_masks_ids_onehot = torch.nn.functional.one_hot(sel_masks_ids, num_classes=K)[0].bool()
    else:
        sel_masks_ids = torch.zeros(size=(0,), dtype=torch.int, device=pot_masks_pxl_prob.device)
        # sel_masks_ids_onehot = torch.zeros(size=(K,), dtype=torch.bool, device=pot_masks_pxl_prob.device)
    return sel_masks_ids


def breadth_first_search(pot_masks_pxl_prob, prob_gain_min, prob_same_mask_max=1.0):
    """Search greedily for new masks without overlap, minimum probability gain and maximum number of new masks.

    Parameters
    ----------
    pot_masks_pxl_prob torch.Tensor: KxHxW
    extracted_masks_pxl_prob torch.Tensor: K0xHxW
    prob_gain_min float: min gain of the sum over all probabilities for each mask

    Returns
    -------
    sel_masks_ids_onehot torch.Tensor: K, dtype=bool
    """
    probs_same_mask = o4masks.calc_prob_same_mask(pot_masks_pxl_prob, pot_masks_pxl_prob)

    bools_same_mask = probs_same_mask > prob_same_mask_max

    K_pot, H, W = pot_masks_pxl_prob.shape
    device = pot_masks_pxl_prob.device
    dtype = pot_masks_pxl_prob.dtype

    sels = {}
    sels["K"] = 0
    sels["K_pot"] = K_pot
    sels["sel_ids_onehot"] = torch.zeros(size=(0, K_pot), dtype=torch.bool, device=device)
    sels["pot_ids_onehot"] = torch.zeros(size=(0, K_pot), dtype=torch.bool, device=device)
    sels["inliers"] = torch.zeros(size=(H, W), dtype=torch.bool, device=device)
    for i in range(30):
        sels = breadth_first_search_single_step(pot_masks_pxl_prob, prob_gain_min, bools_same_mask, sels)

        if sels["pot_ids_onehot"].sum() == 0:
            break

    sel_max_inlier_id = torch.argmax(sels["inliers"].flatten(1).sum(dim=1))
    sel_ids_onehot = sels["sel_ids_onehot"][sel_max_inlier_id]
    sel_ids = torch.where(sel_ids_onehot)[0]
    return sel_ids


def breadth_first_search_single_step(pot_masks_pxl_prob, prob_gain_min, bools_same_mask, sels):
    print("INFO :: BFS selected", sels["K"])
    if sels["K"] == 0:
        sels["sel_ids_onehot"] = torch.diag(pot_masks_pxl_prob.flatten(1).sum(dim=1) >= prob_gain_min)
        sels["sel_ids_onehot"] = sels["sel_ids_onehot"][sels["sel_ids_onehot"].sum(dim=1) > 0]
    else:
        new_sels_ids_onehot = torch.zeros(size=(0, sels["K_pot"]), dtype=torch.bool, device=pot_masks_pxl_prob.device)
        for k in range(sels["K"]):
            new_sels_ids_onehot_k = torch.diag(sels["pot_ids_onehot"][k])
            new_sels_ids_onehot_k = new_sels_ids_onehot_k[new_sels_ids_onehot_k.sum(dim=1) > 0]
            if len(new_sels_ids_onehot_k) == 0:
                new_sels_ids_onehot_k = sels["sel_ids_onehot"][k][None]
            else:
                new_sels_ids_onehot_k = sels["sel_ids_onehot"][k][None,] + new_sels_ids_onehot_k
            if len(new_sels_ids_onehot) > 0:
                duplicates_ids_onehot = (
                    torch.logical_xor(new_sels_ids_onehot_k[:, None], new_sels_ids_onehot[None, :])
                    .max(dim=2)[0]
                    .min(dim=1)[0]
                )
                new_sels_ids_onehot_k = new_sels_ids_onehot_k[~duplicates_ids_onehot]

            new_sels_ids_onehot = torch.cat((new_sels_ids_onehot, new_sels_ids_onehot_k), dim=0)
        sels["sel_ids_onehot"] = new_sels_ids_onehot

    sels["inliers"] = (sels["sel_ids_onehot"][:, :, None, None] * pot_masks_pxl_prob[None,]).sum(dim=1) > 0
    sels["pot_ids_onehot"] = (
        (~sels["sel_ids_onehot"][:, :, None] + (sels["sel_ids_onehot"][:, :, None] * (~bools_same_mask)[None, :]))
        .prod(dim=1)
        .bool()
    )
    sels["pot_ids_onehot"] *= (
        pot_masks_pxl_prob[None, :].float()
        - sels["inliers"][
            :,
            None,
        ].float()
    ).clamp(
        0, 1
    ).flatten(2).sum(dim=2) >= prob_gain_min
    sels["K"] = len(sels["sel_ids_onehot"])

    return sels
