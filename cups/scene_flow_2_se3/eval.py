import torch

import cups.scene_flow_2_se3.clustering.elemental as o4cluster
import cups.scene_flow_2_se3.geometric.se3.registration as o4geo_se3_reg
import cups.scene_flow_2_se3.geometric.se3.transform as o4geo_se3_transf
import cups.scene_flow_2_se3.masks.elemental as o4masks
import cups.scene_flow_2_se3.match as o4match
import cups.scene_flow_2_se3.vision.visualization as o4visual


def calc_outlier_pixelwise(pred, gt, mask=None):
    dev = torch.norm(pred - gt, dim=1, keepdim=True)
    dev_rel = dev / torch.norm(gt, dim=1, keepdim=True)

    outlier = (dev >= 3.0) * (dev_rel >= 0.05)

    if mask is not None:
        outlier *= mask
    return outlier


def calc_std_from_inlier(pred, gt, mask=None):
    dev = torch.norm(pred - gt, dim=1, keepdim=True)
    dev_rel = dev / torch.norm(gt, dim=1, keepdim=True)
    outlier = (dev >= 3.0) * (dev_rel >= 0.05)
    inlier = ~outlier
    if mask is not None:
        inlier = inlier * mask
    std_abs = (dev[inlier] ** 2).mean().sqrt()
    std_rel = (dev_rel[inlier] ** 2).mean().sqrt()

    return std_abs, std_rel


def calc_outlier_percentage_from_pixelwise(outlier, mask):

    return (outlier * mask).sum() / (mask.sum() + 1e-10) * 100


def calc_outlier_percentage(pred, gt, masks):

    return (calc_outlier_pixelwise(pred, gt) * masks).sum() / (masks.sum() + 1e-10) * 100


def calc_epe(pred, gt, masks, rel=False):
    # C = pred.size(1)
    dev = pred - gt
    if rel:
        gt_norm = torch.norm(gt, dim=1, keepdim=True)
        dev = dev / (gt_norm + 1e-10)
    return torch.norm(dev * masks, dim=1).sum() / (masks.sum() + 1e-10)


def calc_depth_inlier_perc(pred, gt, masks):
    rel1 = torch.abs(pred / gt)
    rel2 = torch.abs(gt / pred)
    rel = torch.max(rel1, rel2)
    inlier_perc = ((rel < 1.25) * masks).sum() / (masks.sum() + 1e-10) * 100
    return inlier_perc


def calc_f_measure(pred_masks, gt_masks):
    precision = o4masks.calc_ios1(pred_masks, gt_masks)
    recall = o4masks.calc_ios2(pred_masks, gt_masks)

    f_measure = 2 * precision * recall / (precision + recall)
    f_measure[(precision == 0.0) * (recall == 0.0)] = 0.0
    return f_measure * 100


def calc_f_measure_avg(pred_masks, gt_masks):
    K1 = len(pred_masks)
    K2 = len(gt_masks)
    _, H, W = pred_masks.shape
    dtype = torch.float32
    device = pred_masks.device

    # if K2 > K1:
    #    placeholder = torch.zeros(
    #        size=(K2 - K1, H, W), dtype=pred_masks.dtype, device=device
    #    )
    #    pred_masks = torch.cat((pred_masks, placeholder), dim=0)
    # K1 = len(pred_masks)
    # o4visual.visualize_img(gt_masks[2][None])

    f_measure = torch.zeros(size=(K1, K2), dtype=dtype, device=device)
    for k1 in range(K1):
        for k2 in range(K2):
            f_measure[k1, k2] = calc_f_measure(pred_masks[k1][None], gt_masks[k2][None])[0, 0]
    # requires too much GPU memory: K1, K2 ~ 50, H=540, W=960 -> 6 GB
    # f_measure = calc_f_measure(pred_masks, gt_masks)

    row_ids, col_ids = o4match.hungarian_one_to_one(-f_measure)
    # 1 on 1 matching
    # pred_masks: K1 x H x W, gt_masks: K2 x H x W
    # -> K1 x N, K2 x N
    # 1. calculate TP
    # 2. F1 = 2TP / (2TP + FN + FP)
    TP = (pred_masks[row_ids[:K1]] * gt_masks[col_ids[:K1]]).sum()
    f_measure_avg = TP * 2 / (TP + gt_masks.sum()) * 100

    # ACC_{i,j} = TP_{i,j} / OMEGA_i
    # ACC = TP / OMEGA =  (\sum_{(i,j)} TP_{i,j}) / (\sum(OMEGA))

    # f_measure_matched = f_measure[row_ind, col_ind]
    # weights = gt_masks.flatten(1).sum(dim=1) / gt_masks.sum()
    # f_measure_avg = (f_measure_matched * weights).sum()

    return f_measure_avg


def calc_accuracy_multilabel_segmentation(pred_masks, gt_masks):
    K1 = len(pred_masks)
    K2 = len(gt_masks)
    _, H, W = pred_masks.shape
    dtype = torch.float32
    device = pred_masks.device

    # if K2 > K1:
    #    placeholder = torch.zeros(
    #        size=(K2 - K1, H, W), dtype=pred_masks.dtype, device=device
    #    )
    #    pred_masks = torch.cat((pred_masks, placeholder), dim=0)
    # K1 = len(pred_masks)
    # o4visual.visualize_img(gt_masks[2][None])

    tp_ij = torch.zeros(size=(K1, K2), dtype=dtype, device=device)
    for k1 in range(K1):
        for k2 in range(K2):
            tp_ij[k1, k2] = (pred_masks[k1] * gt_masks[k2]).sum()
    # requires too much GPU memory: K1, K2 ~ 50, H=540, W=960 -> 6 GB
    # f_measure = calc_f_measure(pred_masks, gt_masks)

    row_ids, col_ids = o4match.hungarian_one_to_one(-tp_ij)
    # print(row_ids, col_ids)
    # 1 on 1 matching
    # pred_masks: K1 x H x W, gt_masks: K2 x H x W
    # -> K1 x N, K2 x N
    # 1. calculate TP
    # 2. F1 = 2TP / (2TP + FN + FP)
    TP = (pred_masks[row_ids[:K1]] * gt_masks[col_ids[:K1]]).sum()
    acc = TP / (gt_masks.sum()) * 100

    K_min = min(K1, K2)
    gt_ids = torch.arange(K2, device=device)
    gt_ids[col_ids[:K_min]] = -1
    gt_ids = gt_ids.sort()[0]
    gt_ids[:K_min] = col_ids[:K_min]

    pred_ids = torch.arange(K1, device=device)
    pred_ids[row_ids[:K_min]] = -1
    pred_ids = pred_ids.sort()[0]
    pred_ids[:K_min] = row_ids[:K_min]
    # gt_masks = gt_masks[gt_ids]
    # ACC_{i,j} = TP_{i,j} / OMEGA_i
    # ACC = TP / OMEGA =  (\sum_{(i,j)} TP_{i,j}) / (\sum(OMEGA))
    return acc, gt_ids, pred_ids


def eval_data(data_pred, data_gt, visual_dir=None):
    metrics = {}
    outlier = {}
    pred_keys = data_pred.keys()
    gt_keys = data_gt.keys()

    oflow_outlier_pxl = None
    disp_0_outlier_pxl = None
    disp_f0_1_outlier_pxl = None
    if "oflow" in pred_keys and "oflow" in gt_keys:

        data_pred["oflow"] = o4visual.resize(
            data_pred["oflow"],
            H_out=data_gt["oflow"].shape[2],
            W_out=data_gt["oflow"].shape[3],
            mode="bilinear",
            vals_rescale=True,
        )

        oflow_outlier_pxl = calc_outlier_pixelwise(data_pred["oflow"], data_gt["oflow"], data_gt["oflow_valid"])
        outlier["oflow"] = oflow_outlier_pxl
        metrics["oflow_outlier_perc"] = calc_outlier_percentage_from_pixelwise(
            oflow_outlier_pxl, data_gt["oflow_valid"]
        )

        metrics["oflow_epe"] = calc_epe(data_pred["oflow"], data_gt["oflow"], data_gt["oflow_valid"])

    if "disp_0" in pred_keys and "disp_0" in gt_keys:
        data_pred["disp_0"] = o4visual.resize(
            data_pred["disp_0"],
            H_out=data_gt["disp_0"].shape[2],
            W_out=data_gt["disp_0"].shape[3],
            mode="bilinear",
            vals_rescale=True,
        )

        disp_0_outlier_pxl = calc_outlier_pixelwise(data_pred["disp_0"], data_gt["disp_0"], data_gt["disp_valid_0"])
        outlier["disp_0"] = disp_0_outlier_pxl
        metrics["disp_0_outlier_perc"] = calc_outlier_percentage_from_pixelwise(
            disp_0_outlier_pxl, data_gt["disp_valid_0"]
        )

        metrics["disp_0_epe"] = calc_epe(data_pred["disp_0"], data_gt["disp_0"], data_gt["disp_valid_0"])

    if "depth_0" in pred_keys and "depth_0" in gt_keys:
        data_pred["depth_0"] = o4visual.resize(
            data_pred["depth_0"],
            H_out=data_gt["depth_0"].shape[2],
            W_out=data_gt["depth_0"].shape[3],
            mode="bilinear",
            vals_rescale=True,
        )

        metrics["depth_0_abs"] = calc_epe(data_pred["depth_0"], data_gt["depth_0"], data_gt["depth_valid_0"])
        metrics["depth_0_rel"] = calc_epe(data_pred["depth_0"], data_gt["depth_0"], data_gt["depth_valid_0"], rel=True)
        metrics["depth_0_inlier_perc"] = calc_depth_inlier_perc(
            data_pred["depth_0"], data_gt["depth_0"], data_gt["depth_valid_0"]
        )

    # calc_depth_inlier
    if "ego_pose_0" in gt_keys and "ego_pose_1" in gt_keys and "ego_se3" in pred_keys:
        data_gt["ego_se3"] = torch.matmul(torch.linalg.inv(data_gt["ego_pose_0"]), data_gt["ego_pose_1"])
        # data_pred['ego_se3'] = t3d.se3_exp_map(t3d.se3_log_map(data_pred['ego_se3'].permute(0, 2, 1)), eps=1e-04).permute(0, 2, 1)
        # data_gt['ego_se3'] = t3d.se3_exp_map(t3d.se3_log_map(data_gt['ego_se3'].permute(0, 2, 1)), eps=1e-04).permute(0, 2, 1)

        rel_se3 = torch.matmul(torch.linalg.inv(data_pred["ego_se3"]), data_gt["ego_se3"])

        # print('pred \n ', data_pred['ego_se3'].dtype)
        # print('pred \n ', data_pred['ego_se3'])
        # print('gt \n ', data_gt['ego_se3'])
        # print('gt \n ', data_gt['ego_se3'].dtype)

        # rel_se3 = t3d.se3_exp_map(t3d.se3_log_map(rel_se3.permute(0, 2, 1))).permute(0, 2, 1)

        # print('rel \n ', rel_se3.shape)
        rpe_dist = o4geo_se3_reg.se3_mat_2_dist(rel_se3) * data_gt["fps"]
        rpe_angle = o4geo_se3_reg.se3_mat_2_angle_deg(rel_se3) * data_gt["fps"]
        metrics["ego_se3_rpe_dist"] = rpe_dist
        metrics["ego_se3_rpe_angle"] = rpe_angle

        print("TRANSLATION PRED", data_pred["ego_se3"][:, :3, 3])
        print("TRANSLATION GT", data_gt["ego_se3"][:, :3, 3])

        if data_gt["seq_el_id"] + 1 == data_gt["seq_len"]:
            data_pred_seq_poses = data_pred["seq_ego_poses_0"] + [data_pred["seq_ego_poses_0"][-1]]
            data_gt_seq_poses = data_gt["seq_ego_poses_0"] + [data_gt["seq_ego_poses_0"][-1]]
            data_pred_seq_poses = torch.cat(data_pred_seq_poses, dim=0)
            data_gt_seq_poses = torch.cat(data_gt_seq_poses, dim=0)
            data_gt_seq_poses = torch.linalg.inv(data_gt_seq_poses[:1]) @ data_gt_seq_poses

            data_pred_seq_poses_xyz = data_pred_seq_poses[:, :3, 3]
            data_gt_seq_poses_xyz = data_gt_seq_poses[:, :3, 3]

            se3_mat = o4geo_se3_reg.calc_pointsets_registration_from_corresp3d(
                data_pred_seq_poses_xyz[None,], data_gt_seq_poses_xyz[None,]
            )[0]
            # reset se3_mat for debug purpose
            # se3_mat[:, :] = torch.eye(4, dtype=se3_mat.dtype, device=se3_mat.device)
            data_pred_seq_poses_xyz_ftf = o4geo_se3_transf.pts3d_transform(
                data_pred_seq_poses_xyz[None,], se3_mat[None,]
            )[0]
            metrics["ego_se3_ate"] = torch.norm(data_gt_seq_poses_xyz - data_pred_seq_poses_xyz_ftf, dim=1).norm()
            # if visual_dir is not None:
            #    o4visual.visualize_pts3d([data_gt_seq_poses_xyz_centered, data_pred_seq_poses_xyz_ftf_centered, data_gt_seq_poses_xyz_centered[:, :1], data_pred_seq_poses_xyz_ftf_centered[:, :1]],
            #                              fpath=os.path.join(visual_dir, 'seq_poses.gif'), visualize_rot_x=True)

    if "disp_f0_1" in pred_keys and "disp_f0_1" in gt_keys:
        data_pred["disp_f0_1"] = o4visual.resize(
            data_pred["disp_f0_1"],
            H_out=data_gt["disp_f0_1"].shape[2],
            W_out=data_gt["disp_f0_1"].shape[3],
            mode="bilinear",
            vals_rescale=True,
        )
        disp_f0_1_outlier_pxl = calc_outlier_pixelwise(
            data_pred["disp_f0_1"], data_gt["disp_f0_1"], data_gt["disp_valid_f0_1"]
        )
        outlier["disp_f0_1"] = disp_f0_1_outlier_pxl
        metrics["disp_f0_1_outlier_perc"] = calc_outlier_percentage_from_pixelwise(
            disp_f0_1_outlier_pxl, data_gt["disp_valid_f0_1"]
        )

        # o4visual.visualize_img(o4visual.disp2rgb(data_pred['disp_f0_1'][0]) + 100 * (~data_gt['disp_valid_f0_1'][0]))
        metrics["disp_f0_1_epe"] = calc_epe(data_pred["disp_f0_1"], data_gt["disp_f0_1"], data_gt["disp_valid_f0_1"])

    if "depth_f0_1" in pred_keys and "depth_f0_1" in gt_keys:
        data_pred["depth_f0_1"] = o4visual.resize(
            data_pred["depth_f0_1"],
            H_out=data_gt["depth_f0_1"].shape[2],
            W_out=data_gt["depth_f0_1"].shape[3],
            mode="bilinear",
            vals_rescale=True,
        )

        metrics["depth_f0_1_abs"] = calc_epe(
            data_pred["depth_f0_1"], data_gt["depth_f0_1"], data_gt["depth_valid_f0_1"]
        )
        metrics["depth_f0_1_rel"] = calc_epe(
            data_pred["depth_f0_1"], data_gt["depth_f0_1"], data_gt["depth_valid_f0_1"], rel=True
        )
        metrics["depth_f0_1_inlier_perc"] = calc_depth_inlier_perc(
            data_pred["depth_f0_1"], data_gt["depth_f0_1"], data_gt["depth_valid_f0_1"]
        )

    if "objs_masks" in pred_keys and "objs_masks" in gt_keys:
        data_pred["objs_masks"] = o4visual.resize(
            data_pred["objs_masks"],
            H_out=data_gt["objs_masks"].shape[1],
            W_out=data_gt["objs_masks"].shape[2],
            mode="nearest_v2",
            vals_rescale=False,
        )
        # metrics['f_measure'] = calc_f_measure_avg(data_pred['objs_masks'], data_gt['objs_masks'])
        # metrics['f_measure_outlier_perc'] = 100 - metrics['f_measure']
        metrics["seg_acc"], gt_ids, pred_ids = calc_accuracy_multilabel_segmentation(
            data_pred["objs_masks"], data_gt["objs_masks"]
        )
        data_gt["objs_masks"] = data_gt["objs_masks"][gt_ids]
        data_pred["objs_masks"] = data_pred["objs_masks"][pred_ids]

        if "objs_params" in data_gt.keys():
            data_gt["objs_params"]["se3"]["se3"] = data_gt["objs_params"]["se3"]["se3"][:, gt_ids]
        if "objs_params" in data_pred.keys():
            data_pred["objs_params"]["se3"]["se3"] = data_pred["objs_params"]["se3"]["se3"][:, pred_ids]
            if "objs_center_3d_0" in data_pred.keys():
                data_pred["objs_center_3d_0"] = data_pred["objs_center_3d_0"][pred_ids]
                data_pred["objs_center_3d_1"] = data_pred["objs_center_3d_1"][pred_ids]
                data_pred["objs_center_2d_0"] = data_pred["objs_center_2d_0"][pred_ids]
                data_pred["objs_center_2d_1"] = data_pred["objs_center_2d_1"][pred_ids]
            # if 'ego_se3' in data_gt.keys():
            #    data_gt['objs_params']['se3']['se3'][:, 0] = torch.linalg.inv(data_gt['ego_se3'])
        data_gt["objs_labels"] = o4cluster.onehot_2_label(data_gt["objs_masks"][None])
        data_pred["objs_labels"] = o4cluster.onehot_2_label(data_pred["objs_masks"][None])

    if oflow_outlier_pxl is not None and disp_0_outlier_pxl is not None and disp_f0_1_outlier_pxl is not None:
        sflow_outlier_pxl = oflow_outlier_pxl + disp_0_outlier_pxl + disp_f0_1_outlier_pxl
        outlier["sflow"] = sflow_outlier_pxl
        metrics["sflow_outlier_perc"] = calc_outlier_percentage_from_pixelwise(
            sflow_outlier_pxl, data_gt["oflow_valid"] * data_gt["disp_valid_0"] * data_gt["disp_valid_f0_1"]
        )

    if "pt3d_0" in pred_keys and "pt3d_0" in gt_keys and "pt3d_f0_1" in pred_keys and "pt3d_f0_1" in gt_keys:
        data_pred["sflow"] = data_pred["pt3d_f0_1"] - data_pred["pt3d_0"]
        data_gt["sflow"] = data_gt["pt3d_f0_1"] - data_gt["pt3d_0"]
        metrics["sflow_epe"] = calc_epe(
            data_pred["sflow"],
            data_gt["sflow"],
            data_gt["oflow_valid"] * data_gt["disp_valid_0"] * data_gt["disp_valid_f0_1"],
        )

    # if 'ego_se3' in pred_keys and 'ego_se3' in gt_keys:
    # TODO: add ego_se3 evaluation
    # TODO: add visualization / return visualization if want to compare pred_sflow/pred_se3

    return metrics, outlier
