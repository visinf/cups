import tensor_operations.visual._2d as o4vis2d
import torch


def selection(sflow, drpcs_prop, args, drpcs_prev=None, max_count=1, logger=None):
    sel_ids = []
    for k in range(max_count):
        sel_id = selection_single(
            sflow, drpcs_prop, args=args, drpcs_prev=drpcs_prev, drpcs_sel_ids=sel_ids, logger=logger
        )
        if sel_id is not None:
            sel_ids.append(sel_id)
        else:
            break

    if len(sel_ids) == 0:
        return None
    drpcs_prop.select_drpcs(torch.stack(sel_ids))
    return drpcs_prop


def selection_single(sflow, drpcs_prop, args, drpcs_prev=None, drpcs_sel_ids=[], logger=None):
    # drpcs_prop.sflow_inlier_soft
    # drpcs_prop.sflow_inlier_hard
    # drpcs_prop.sflow_log_likelihood

    p_contribute_pxl = drpcs_prop.sflow_inlier_soft
    p_coverage_pxl = torch.zeros_like(drpcs_prop.sflow_inlier_soft)

    if drpcs_prev is not None:
        p_coverage_pxl = torch.max(p_coverage_pxl, drpcs_prev.sflow_inlier_soft.max(dim=0, keepdim=True)[0])
        p_overlap = calc_p_overlap(drpcs_prop, drpcs_prev, sflow.depth_reliable_0)
        drpcs_prop.sflow_inlier_soft[p_overlap > args.sflow2se3_se3filter_prob_same_mask_max] = 0.0
    if len(drpcs_sel_ids) > 0:
        p_coverage_pxl = torch.max(
            p_coverage_pxl, drpcs_prop.sflow_inlier_soft[torch.stack(drpcs_sel_ids)].max(dim=0, keepdim=True)[0]
        )

    p_contribute_pxl = (drpcs_prop.sflow_inlier_soft - p_coverage_pxl).clamp(0, 1)

    p_contribute = (sflow.depth_reliable_0 * p_contribute_pxl).sum(dim=(1, 2)) / (
        sflow.depth_reliable_0.sum(dim=(1, 2)) + 1e-10
    )
    # p_overlap = 0.0

    p_contribute_max_val, p_contribute_max_id = p_contribute.max(dim=0)

    if logger is not None and args.eval_visualize_paper:
        logger.log_image(
            o4vis2d.mask2rgb(
                torch.cat(
                    [
                        drpcs_prop.sflow_inlier_soft[:4],
                        drpcs_prop.sflow_inlier_soft[p_contribute_max_id : p_contribute_max_id + 1],
                    ]
                ),
                img=sflow.rgb,
            ),
            key="paper_inlier_pixel/img",
        )

    if p_contribute_max_val > args.sflow2se3_se3filter_prob_gain_min:
        return p_contribute_max_id
    else:
        return None


def calc_p_overlap(drpcs_prop, drpcs_prev, depth_reliable_0):
    drpcs_prop.sflow_inlier_soft = drpcs_prop.sflow_inlier_soft.clamp(0.0, 1.0)
    drpcs_prev.sflow_inlier_soft = drpcs_prev.sflow_inlier_soft.clamp(0.0, 1.0)
    inter_true_true = (
        depth_reliable_0[None,] * drpcs_prop.sflow_inlier_soft[:, None] * drpcs_prev.sflow_inlier_soft[None, :]
    ).sum(dim=(2, 3))
    inter_true_false = (
        depth_reliable_0[None,] * (1.0 - drpcs_prop.sflow_inlier_soft[:, None]) * drpcs_prev.sflow_inlier_soft[None, :]
    ).sum(dim=(2, 3))
    inter_false_true = (
        depth_reliable_0[None,] * drpcs_prop.sflow_inlier_soft[:, None] * (1.0 - drpcs_prev.sflow_inlier_soft[None, :])
    ).sum(dim=(2, 3))
    # inter_false_false = ((1.0 - drpcs_prop.sflow_inlier_soft[:, None]) * (1.0 - drpcs_prev.sflow_inlier_soft[None, :])).sum(dim=(2, 3))

    p_overlap = inter_true_true / (inter_true_true + inter_true_false + inter_false_true + 1e-10)
    p_overlap = p_overlap.max(dim=1)[0]
    return p_overlap
