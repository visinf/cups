import tensor_operations.visual._2d as o4vis2d
from tensor_operations.retrieval.sflow2se3.drpc import DRPCs
from tensor_operations.retrieval.sflow2se3.proposal import proposals
from tensor_operations.retrieval.sflow2se3.selection import selection
from tensor_operations.retrieval.sflow2se3.sflow import SFlow


def sflow2se3(data, args, logger=None):
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

    for k in range(10):
        se3_prop = proposals(sflow_down, args=args, drpcs=drpcs, logger=logger)
        if se3_prop is None:
            continue

        drpcs_prop = DRPCs(se3=se3_prop)
        drpcs_prop.calc_sflow_consensus(sflow_down, update_pt3d_0=True, update_pt3d_1=False)
        drpcs_sel = selection(sflow_down, drpcs_prop, args=args, drpcs_prev=drpcs, logger=logger)

        if drpcs_sel is None:
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

    pts3d_0 = sflow.pt3d_0[None,]
    pts3d_0_ftf = drpcs.pt3d_1[None,]
    models_params = {}
    models_params["se3"] = {}
    models_params["geo"] = {}
    models_params["se3"]["se3"] = drpcs.se3[None]
    models_params["geo"]["pts"] = drpcs.pt3d[None]
    models_params["geo"]["pts_assign"] = drpcs.pt3d_assign[None]
    masks_objs = drpcs.max_log_likelihood_onehot.bool()
    labels_objs = drpcs.max_log_likelihood_label.int()

    return labels_objs, masks_objs, models_params, pts3d_0, pts3d_0_ftf
