import torch
from torch import Tensor

import cups.scene_flow_2_se3.visual._2d as o4vis2d
from cups.scene_flow_2_se3.connectec_components import connected_components
from cups.scene_flow_2_se3.retrieval.sflow2se3.drpc import DRPCs
from cups.scene_flow_2_se3.retrieval.sflow2se3.proposal import proposals
from cups.scene_flow_2_se3.retrieval.sflow2se3.selection import selection
from cups.scene_flow_2_se3.retrieval.sflow2se3.sflow import SFlow
from cups.scene_flow_2_se3.utils import omit_too_small_object_proposals, remap_ids


def sflow2se3(data, args, logger=None, iterations: int = 10):
    """Extracts se3 transformations from scene flow in a greedy way.

    Parameters
    ----------
    data dict: scene flow BxCxHxW
    objs dict:
        se3 dict:
            se3 torch.Tensor: Kx4x4
            se3_centroid1 torch.Tensor: Kx4x4
            inlier torch.Tensor: KxHxW, dtype bool
            log_likelihood torch.Tensor: KxHxW, dtype float
        inlier_joint torch.Tensor: 1xHxW, dtype bool
        log_likelhood_max torch.Tensor: 1xHxW, dtype bool
        K int: number of objects

    args argparse.Namespace: args from configs

    Returns
    -------
    objs dict:
        se3 dict:
            se3 torch.Tensor: Kx4x4
            se3_centroid1 torch.Tensor: Kx4x4
            inlier torch.Tensor: KxHxW, dtype bool
            log_likelihood torch.Tensor: KxHxW, dtype float
        inlier_joint torch.Tensor: 1xHxW, dtype bool
        log_likelhood_max torch.Tensor: 1xHxW, dtype bool
        K int: number of objects
    """
    drpcs = None
    sflow = SFlow(data, args)

    sflow_down = sflow.resizeToNewObject(
        scale_factor=args.sflow2se3_downscale_factor, mode=args.sflow2se3_downscale_mode
    )
    finish = False
    for k in range(iterations):
        se3_prop = proposals(sflow_down, args=args, drpcs=drpcs, logger=logger)
        if se3_prop is None:
            continue

        drpcs_prop = DRPCs(se3=se3_prop)
        drpcs_prop.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)
        drpcs_sel = selection(sflow_down, drpcs_prop, args=args, drpcs_prev=drpcs, logger=logger)

        if drpcs_sel is None:
            if finish:
                break
            finish = True
            continue

        drpcs_sel.add_spatial_model(sflow_down, drpcs_prev=drpcs)
        if logger is not None and args.eval_visualize_paper:
            logger.log_image(
                o4vis2d.draw_circles_in_rgb(drpcs_sel.pt3d_assign, img=sflow.rgb),
                key="paper_selected_objects_points_assign/img",
            )

        if drpcs_sel.K > 0:
            drpcs_sel.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)
            drpcs_sel = selection(sflow_down, drpcs_sel, args=args, drpcs_prev=drpcs, max_count=3)

            if drpcs_sel is None:
                continue

            drpcs_sel.update_se3(sflow_down, args=args)
            drpcs_sel.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)

            if drpcs is None:
                drpcs = drpcs_sel
            else:
                drpcs.fuse_drpcs(drpcs_sel)

            if logger is not None and args.eval_visualize_paper:
                if drpcs is not None:
                    logger.log_image(
                        o4vis2d.draw_circles_in_rgb(drpcs.pt3d_assign, img=sflow.rgb),
                        key="paper_fused_objects_points_assign/img",
                    )

    drpcs = selection(sflow_down, drpcs, args=args, drpcs_prev=None, max_count=10)
    drpcs.calc_sflow_consensus(sflow, update_pt3d_0=False, update_pt3d_1=True)

    models_params = {}
    models_params["se3"] = {}
    models_params["geo"] = {}
    models_params["se3"]["se3"] = drpcs.se3[None]
    models_params["geo"]["pts"] = drpcs.pt3d[None]
    models_params["geo"]["pts_assign"] = drpcs.pt3d_assign[None]
    labels_objs = drpcs.max_log_likelihood_label.int()
    # Get background SE(3)
    background_se3 = models_params["se3"]["se3"][0, 0]
    # Filter object proposals
    for id in range(1, models_params["se3"]["se3"].shape[1]):
        object_se3 = models_params["se3"]["se3"][0, id]
        # Compute similarity
        similarity = se3_similarity(background_se3, object_se3)
        # Set object to background if close to the background
        if similarity >= args.sflow2se3_background_th:
            labels_objs[labels_objs == id] = 0
    # Compute dense logits
    logits = drpcs.sflow_log_likelihood.softmax(dim=0)
    # Compute logits per object
    logits_objs = []
    object_masks = []
    for id in labels_objs.unique(sorted=True)[1:]:
        # Get object mask
        object_mask = labels_objs == id
        # Perform connected components
        object_mask = connected_components(object_mask.squeeze())
        # Omit too small objects
        object_mask_cc = remap_ids(omit_too_small_object_proposals(object_mask, min_object_size=128))
        # Iterate over objects in object mask after connected components
        for id_cc in object_mask_cc.unique(sorted=True)[1:]:
            mask = object_mask_cc == id_cc
            object_masks.append(mask.long())
            # Get logits for the object
            logits_object = logits[id]
            logits_objs.append(logits_object[mask].mean())
    # Make tensor
    logits_objs = torch.stack(logits_objs, dim=0) if len(logits_objs) > 0 else torch.zeros(0, device=logits.device)
    object_masks = (
        torch.stack(object_masks, dim=0)
        if len(object_masks) > 0
        else torch.zeros(0, logits.shape[-2], logits.shape[-1], device=logits.device)
    )
    return object_masks, logits_objs


def sflow2se3_vanilla(data, args, logger=None, iterations: int = 10):
    """Extracts se3 transformations from scene flow in a greedy way.

    Parameters
    ----------
    data dict: scene flow BxCxHxW
    objs dict:
        se3 dict:
            se3 torch.Tensor: Kx4x4
            se3_centroid1 torch.Tensor: Kx4x4
            inlier torch.Tensor: KxHxW, dtype bool
            log_likelihood torch.Tensor: KxHxW, dtype float
        inlier_joint torch.Tensor: 1xHxW, dtype bool
        log_likelhood_max torch.Tensor: 1xHxW, dtype bool
        K int: number of objects

    args argparse.Namespace: args from configs

    Returns
    -------
    objs dict:
        se3 dict:
            se3 torch.Tensor: Kx4x4
            se3_centroid1 torch.Tensor: Kx4x4
            inlier torch.Tensor: KxHxW, dtype bool
            log_likelihood torch.Tensor: KxHxW, dtype float
        inlier_joint torch.Tensor: 1xHxW, dtype bool
        log_likelhood_max torch.Tensor: 1xHxW, dtype bool
        K int: number of objects
    """
    drpcs = None
    sflow = SFlow(data, args)

    sflow_down = sflow.resizeToNewObject(
        scale_factor=args.sflow2se3_downscale_factor, mode=args.sflow2se3_downscale_mode
    )
    finish = False
    for k in range(iterations):
        se3_prop = proposals(sflow_down, args=args, drpcs=drpcs, logger=logger)
        if se3_prop is None:
            continue

        drpcs_prop = DRPCs(se3=se3_prop)
        drpcs_prop.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)
        drpcs_sel = selection(sflow_down, drpcs_prop, args=args, drpcs_prev=drpcs, logger=logger)

        if drpcs_sel is None:
            if finish:
                break
            finish = True
            continue

        drpcs_sel.add_spatial_model(sflow_down, drpcs_prev=drpcs)
        if logger is not None and args.eval_visualize_paper:
            logger.log_image(
                o4vis2d.draw_circles_in_rgb(drpcs_sel.pt3d_assign, img=sflow.rgb),
                key="paper_selected_objects_points_assign/img",
            )

        if drpcs_sel.K > 0:
            drpcs_sel.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)
            drpcs_sel = selection(sflow_down, drpcs_sel, args=args, drpcs_prev=drpcs, max_count=3)

            if drpcs_sel is None:
                continue

            drpcs_sel.update_se3(sflow_down, args=args)
            drpcs_sel.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)

            if drpcs is None:
                drpcs = drpcs_sel
            else:
                drpcs.fuse_drpcs(drpcs_sel)

            if logger is not None and args.eval_visualize_paper:
                if drpcs is not None:
                    logger.log_image(
                        o4vis2d.draw_circles_in_rgb(drpcs.pt3d_assign, img=sflow.rgb),
                        key="paper_fused_objects_points_assign/img",
                    )

    drpcs = selection(sflow_down, drpcs, args=args, drpcs_prev=None, max_count=10)
    drpcs.calc_sflow_consensus(sflow, update_pt3d_0=False, update_pt3d_1=True)

    models_params = {}
    models_params["se3"] = {}
    models_params["geo"] = {}
    models_params["se3"]["se3"] = drpcs.se3[None]
    models_params["geo"]["pts"] = drpcs.pt3d[None]
    models_params["geo"]["pts_assign"] = drpcs.pt3d_assign[None]
    labels_objs = drpcs.max_log_likelihood_label.int()
    return labels_objs


def se3_similarity(se3_1: Tensor, se3_2: Tensor) -> Tensor:
    """This function computes the similarity between two SE(3) transformations.

    Args:
        se3_1 (Tensor): First SE(3) transformation of the shape [4, 4].
        se3_2 (Tensor): Second SE(3) transformation of the shape [4, 4].

    Returns:
        similarity (Tensor): Similarity composed of angular distance of the rotation and the norm between translation.
    """
    # Compute angular distance
    angular_distance: Tensor = (torch.arccos((torch.trace(se3_1[:3, :3].T @ se3_2[:3, :3]) - 1.0) / 2.0)).abs()
    # Compute norm between the translations
    translation_norm: Tensor = torch.norm(se3_1[:, -1] - se3_1[:, -1], p=2, dim=-1)
    # Compute final similarity
    similarity: Tensor = 0.5 * angular_distance + translation_norm
    return similarity
