import torch

import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
import cups.scene_flow_2_se3.geometric.se3.fit.corresp_3d_3d as o4geo_se3_fit_pt3d_pt3d
import cups.scene_flow_2_se3.geometric.se3.fit.pt3d_oflow as o4geo_se3_fit_pt3d_oflow
import cups.scene_flow_2_se3.geometric.se3.transform as o4geo_se3_transf


def fit_and_likelihood(data, models_posterior, args, models_params_prev=None):
    models_params = fit(data, models_posterior, args, models_params_prev)

    models_likelihood = likelihood(data, models_params, args)

    # fits_precision =  (models_posterior * models_likelihood).flatten(2).sum(dim=2) / (models_likelihood.flatten(2).sum(dim=2) + 1e-8)
    # fits_bin = fits_precision > 0.2
    # models_likelihood = models_likelihood[fits_bin].reshape(B, -1, H, W)
    # models_params['se3'] = models_params['se3'][fits_bin].reshape(B, -1, 4, 4)
    # models_params['se3_centroid1'] = models_params['se3_centroid1'][fits_bin].reshape(B, -1, 4, 4)

    return models_params, models_likelihood


def fit(data, models_posterior, args, models_params_prev=None):
    # models_posterior: B x K x H x W

    dtype = data["pt3d_0"].dtype
    device = models_posterior.device

    if "B" in data and data["B"] == -1:
        K, H, W = models_posterior.shape
        B = 1
        models_posterior = models_posterior[None,]
        pts1 = data["pt3d_0"][None,]
        pts2 = data["pt3d_f0_1"][None,]
        pt3d_pair_valid = data["pt3d_pair_valid"][None,]
        pt3d_valid_0 = data["pt3d_valid_0"][None,]
        oflow = data["oflow"][None,]
        proj_mat = data["proj_mat"][None,]
        baseline = data["baseline"][None,]
        if models_params_prev is not None:
            models_params_prev["se3"] = models_params_prev["se3"][None,]
            models_params_prev["se3_centroid1"] = models_params_prev["se3_centroid1"][None,]
    else:
        B, K, H, W = models_posterior.shape
        pts1 = data["pt3d_0"]
        pts2 = data["pt3d_f0_1"]
        pt3d_pair_valid = data["pt3d_pair_valid"]
        pt3d_valid_0 = data["pt3d_valid_0"]
        oflow = data["oflow"]
        proj_mat = data["proj_mat"]
        baseline = data["baseline"]

    models_se3_cam = []  # torch.zeros(size=(B, K, 4, 4), dtype=dtype, device=device)
    models_se3_centroid1 = []  # torch.zeros(size=(B, K, 4, 4), dtype=dtype, device=device)
    models_posterior = models_posterior * pt3d_valid_0
    for b in range(B):
        for k in range(K):
            if (models_posterior[b][k] * pt3d_pair_valid[b][0]).sum() > 10:
                models_posterior[b][k] = models_posterior[b][k] * pt3d_pair_valid[b][0]
            # if "disp_occ_0" in data.keys():
            #    if (models_posterior[b][k] * (~data["disp_occ_0"][b][0])).sum() > 10:
            #        models_posterior[b][k] = models_posterior[b][k] * (~data["disp_occ_0"][b][0])

        if args["fit"]:
            if args["fit_data"] == "pt3d-pt3d":
                # points3d: N x 3
                # pxl2d_f0_1: N x 2
                # K0: 3x3 [[fx, 0, cx],[0, fy, cy], [0, 0, 1]]
                models_posterior = models_posterior * pt3d_pair_valid

                b_models_se3_cam = o4geo_se3_fit_pt3d_pt3d.fit_se3_to_corresp_3d_3d_and_masks(
                    models_posterior[b] > 0.5, pts1[b], pts2[b], weights=models_posterior[b]
                )

            elif args["fit_data"] == "pt3d-oflow":

                b_models_se3_cam = o4geo_se3_fit_pt3d_oflow.fit_se3_to_pt3d_oflow_and_masks(
                    models_posterior[b] > 0.5,
                    pts1[b],
                    oflow[b],
                    proj_mat[b],
                    orig_H=data["orig_H"],
                    orig_W=data["orig_W"],
                    resize_mode=args["resize_mode"],
                    method=args["fit_pt3d_oflow_method"],
                    weights=models_posterior[b],
                )
                # print(b_models_se3_cam)
            b_models_se3_centroid1 = b_models_se3_cam

        else:
            b_models_se3_cam = models_params_prev["se3"][b]
            b_models_se3_centroid1 = models_params_prev["se3_centroid1"][b]

        if args["refine"]:
            b_models_se3_cam = o4geo_se3_fit_pt3d_oflow.fit_se3_to_pt3d_oflow_and_masks(
                models_posterior[b] > 0.5,
                pts1[b],
                oflow[b],
                proj_mat[b],
                orig_H=data["orig_H"],
                orig_W=data["orig_W"],
                resize_mode=args["resize_mode"],
                method=args["refine_pt3d_oflow_method"],
                weights=models_posterior[b],
                prev_se3_mats=b_models_se3_cam,
            )
            # b_models_se3_cam.is_inf().sum()

        invalid_se3 = (b_models_se3_cam.isinf().flatten(1).sum(dim=1) > 0) + (
            b_models_se3_cam.isnan().flatten(1).sum(dim=1) > 0
        )
        if invalid_se3.sum() > 0:
            b_models_se3_cam[invalid_se3] = torch.eye(4, dtype=dtype).to(device)[None].repeat(invalid_se3.sum(), 1, 1)
        models_se3_cam.append(b_models_se3_cam)
        models_se3_centroid1.append(b_models_se3_centroid1)

    models_se3_cam = torch.stack(models_se3_cam)
    models_se3_centroid1 = torch.stack(models_se3_centroid1)

    models_params = {}

    if data["B"] == -1:
        models_params["se3"] = models_se3_cam[0]
        models_params["se3_centroid1"] = models_se3_centroid1[0]
    else:
        models_params["se3"] = models_se3_cam
        models_params["se3_centroid1"] = models_se3_centroid1

    return models_params


def likelihood(data, models_params, args):
    # pts3d_1_down:    B x 3 x H x W
    # pairs_valid: B x 1 x H x W
    # pts3d_1_ftf_down: B x K x H x W

    if "B" in data and data["B"] == -1:
        K, _, _ = models_params["se3"].shape
        B = 1
        pts1 = data["pt3d_0"][None,]
        pts2 = data["pt3d_f0_1"][None,]
        pt3d_pair_valid = data["pt3d_pair_valid"][None,]
        pt3d_valid_0 = data["pt3d_valid_0"][None,]
        oflow = data["oflow"][None,]
        proj_mat = data["proj_mat"][None,]
        baseline = data["baseline"][None,]
    else:
        _, K, _, _ = models_params["se3"].shape
        pts1 = data["pt3d_0"]
        pts2 = data["pt3d_f0_1"]
        pt3d_pair_valid = data["pt3d_pair_valid"]
        pt3d_valid_0 = data["pt3d_valid_0"]
        oflow = data["oflow"]
        proj_mat = data["proj_mat"]
        baseline = data["baseline"]

    B, _, H, W = pt3d_pair_valid.shape
    dtype = pts1.dtype
    device = pts1.device
    pt3d_pair_valid_rep = pt3d_pair_valid.repeat(B, K, 1, 1)
    pt3d_valid_0_rep = pt3d_valid_0.repeat(B, K, 1, 1)
    # o4eval.calc_outlier_pixelwise(pred_disps2_fwd_bwdwrpd, gt_disps2, gt_masks_disps2_valid)

    # B*K x 3 x H x W
    pts1_ftf = o4geo_se3_transf.pts3d_transform(pts1.repeat(K, 1, 1, 1), models_params["se3"].reshape(B * K, 4, 4))
    pts1_ftf = pts1_ftf.reshape(B, K, 3, H, W)

    # args["std_factor_occ"], args["disp_std_use_fwdbwd_dev_if_larger"], args["oflow_std_use_fwdbwd_dev_if_larger"]
    # std_factor_occ = 3
    # disp_std_use_fwdbwd_dev_if_larger = False
    # oflow_std_use_fwdbwd_dev_if_larger = True

    if args["likelihood_use_sflow"]:

        # B x K x 3 x H x W
        sflow_dev_abs = torch.norm(pts1_ftf - pts2[:, None], dim=2)
        sflow_dev_abs[~pt3d_valid_0_rep] = 0.0
        sflow_norm = torch.norm(pts2 - pts1, dim=1, keepdim=True)
        sflow_dev_rel = sflow_dev_abs / (sflow_norm + 1e-10)

        # B x K x H x W
        sflow_dev_abs_distr = torch.distributions.Normal(loc=0.0, scale=args["likelihood_sflow_abs_std"])

        sflow_dev_rel_distr = torch.distributions.Normal(loc=0.0, scale=args["likelihood_sflow_rel_std"])

        sflow_log_likelihood = sflow_dev_abs_distr.log_prob(sflow_dev_abs)

        sflow_dev_rel_inlier = 2 * (1.0 - sflow_dev_rel_distr.cdf(sflow_dev_rel))
        sflow_dev_abs_inlier = 2 * (1.0 - sflow_dev_abs_distr.cdf(sflow_dev_abs))
        sflow_inlier_soft = torch.max(sflow_dev_rel_inlier, sflow_dev_abs_inlier)
        sflow_inlier_hard = sflow_inlier_soft > args["inlier_sflow_hard_threshold"]
        sflow_log_likelihood_min = sflow_dev_abs_distr.log_prob(1e10 * torch.ones(size=(1,), device=device))
        if args["likelihood_sflow_invalid_pairs"] == 0.0:
            sflow_inlier_hard[~pt3d_pair_valid_rep] = False
            sflow_inlier_soft[~pt3d_pair_valid_rep] = 0.0
            sflow_log_likelihood[~pt3d_pair_valid_rep] = sflow_log_likelihood_min
        elif args["likelihood_sflow_invalid_pairs"] == 1.0:
            sflow_inlier_hard[~pt3d_pair_valid_rep] = True
            sflow_inlier_soft[~pt3d_pair_valid_rep] = 1.0
            sflow_log_likelihood_max = sflow_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))
            sflow_log_likelihood[~pt3d_pair_valid_rep] = sflow_log_likelihood_max

        sflow_inlier_soft[~pt3d_valid_0_rep] = False
        sflow_log_likelihood[~pt3d_valid_0_rep] = sflow_log_likelihood_min

    if args["likelihood_use_oflow"]:
        pred_oflows = o4geo_pinhole.pt3d_2_oflow(
            pts1_ftf.reshape(B * K, 3, H, W),
            proj_mat,
            orig_H=data["orig_H"],
            orig_W=data["orig_W"],
            resize_mode=args["resize_mode"],
        ).reshape(B, K, 2, H, W)

        # oflows_norm = torch.norm(oflow, dim=1, keepdim=True)
        # B x 1 x H x W

        if "oflow_fwdbwd_dev" in data.keys() and args["oflow_std_use_fwdbwd_dev_if_larger"]:
            # oflow_occ = data['oflow_occ']
            # std_x = torch.max(args["likelihood_oflow_abs_std"][0], data['oflow_fwdbwd_dev'][:, 0:1].abs() / np.sqrt(2.))

            # std_x[data['oflow_occ']] = args["std_factor_occ"] * args["likelihood_oflow_abs_std"][0]
            # std_y = torch.max(args["likelihood_oflow_abs_std"][1], data['oflow_fwdbwd_dev'][:, 1:2].abs() / np.sqrt(2.))
            # std_y[data['oflow_occ']] = args["std_factor_occ"] * args["likelihood_oflow_abs_std"][1]

            valid_pxl = (~data["oflow_occ"]) * (~data["disp_occ_0"]) * pt3d_valid_0 * pt3d_pair_valid

            std_y = args["likelihood_oflow_abs_std"][1] * (valid_pxl + (~valid_pxl) * args["std_factor_occ"])
            std_x = args["likelihood_oflow_abs_std"][0] * (valid_pxl + (~valid_pxl) * args["std_factor_occ"])

        else:
            std_x = args["likelihood_oflow_abs_std"][0] * (
                pt3d_pair_valid + (~pt3d_pair_valid) * args["std_factor_occ"]
            )
            std_y = args["likelihood_oflow_abs_std"][1] * (
                pt3d_pair_valid + (~pt3d_pair_valid) * args["std_factor_occ"]
            )

        # B x K x 2 x H x W -> B x K x H x W
        # oflow_dev_abs = torch.norm(pred_oflows - oflow[:, None], dim=2)
        oflow_dev_abs = (pred_oflows - oflow[:, None]).abs()

        # oflow_dev_abs[~pt3d_valid_0_rep] = 0.
        oflow_dev_abs[~pt3d_valid_0_rep[:, :, None].repeat(1, 1, 2, 1, 1)] = 0.0

        # oflow_dev_abs[:, 0:1] = torch.clamp(oflow_dev_abs[:, 0:1], 0. * std_x, 3 * std_x)
        # oflow_dev_abs[:, 1:2] = torch.clamp(oflow_dev_abs[:, 1:2], 0. * std_y, 3 * std_y)

        # oflow_dev_rel = oflow_dev_abs / (oflows_norm + 1e-10)
        # oflow_dev_rel = oflow_dev_abs / (oflows_norm[:, :, None] + 1e-10)

        # oflow_dev_abs_distr = torch.distributions.Normal(
        #    loc=0.0, scale=args["likelihood_oflow_abs_std"]
        # )
        # oflow_dev_rel_distr = torch.distributions.Normal(
        #    loc=0.0, scale=args["likelihood_oflow_rel_std"]
        # )

        # o4visual.visualize_img(data['oflow_occ'][0])
        # o4visual.visualize_img(data['disp_occ_0'][0])
        # o4visual.visualize_img(std_x[0] / 10)
        # o4visual.visualize_img(std_y[0] / 10)
        oflow_x_dev_abs_distr = torch.distributions.Normal(loc=0.0, scale=std_x)
        # oflow_x_dev_rel_distr = torch.distributions.Normal(
        #    loc=0.0, scale=args["likelihood_oflow_rel_std"][0]
        # )
        oflow_y_dev_abs_distr = torch.distributions.Normal(loc=0.0, scale=std_y)
        # oflow_y_dev_rel_distr = torch.distributions.Normal(
        #    loc=0.0, scale=args["likelihood_oflow_rel_std"][1]
        # )

        # oflow_log_likelihood = oflow_dev_abs_distr.log_prob(oflow_dev_abs)
        # data["oflow_fwdbwd_dev"]
        oflow_log_likelihood = oflow_x_dev_abs_distr.log_prob(oflow_dev_abs[:, :, 0]) + oflow_y_dev_abs_distr.log_prob(
            oflow_dev_abs[:, :, 1]
        )

        # oflow_dev_abs_inlier = 2 * (1.0 - oflow_dev_abs_distr.cdf(oflow_dev_abs))
        # oflow_dev_rel_inlier = 2 * (1.0 - oflow_dev_rel_distr.cdf(oflow_dev_rel))
        oflow_dev_abs_inlier = (
            2
            * (1.0 - oflow_x_dev_abs_distr.cdf(oflow_dev_abs[:, :, 0]))
            * 2
            * (1.0 - oflow_y_dev_abs_distr.cdf(oflow_dev_abs[:, :, 1]))
        )
        # oflow_dev_rel_inlier = 2 * (1.0 - oflow_x_dev_rel_distr.cdf(oflow_dev_rel[:, :, 0])) * 2 * (1.0 - oflow_y_dev_rel_distr.cdf(oflow_dev_rel[:, :, 1]))

        oflow_inlier_soft = oflow_dev_abs_inlier
        # oflow_inlier_soft = torch.max(
        #    oflow_dev_abs_inlier, oflow_dev_rel_inlier
        # )

        # oflow_log_likelihood_min = oflow_dev_abs_distr.log_prob(
        #        1e10 * torch.ones(size=(1,), device=device)
        # )
        oflow_log_likelihood_min = oflow_x_dev_abs_distr.log_prob(
            1e10 * torch.ones(size=(1,), device=device)
        ) * oflow_y_dev_abs_distr.log_prob(1e10 * torch.ones(size=(1,), device=device))

        oflow_inlier_hard = oflow_inlier_soft > args["inlier_oflow_hard_threshold"]

        if args["likelihood_oflow_invalid_pairs"] == 0.0:
            oflow_inlier_hard[~pt3d_pair_valid_rep] = False
            oflow_inlier_soft[~pt3d_pair_valid_rep] = 0.0
            oflow_log_likelihood[~pt3d_pair_valid_rep] = oflow_log_likelihood_min.repeat(1, K, 1, 1)[
                ~pt3d_pair_valid_rep
            ]

        elif args["likelihood_oflow_invalid_pairs"] == 1.0:
            oflow_inlier_hard[~pt3d_pair_valid_rep] = True
            oflow_inlier_soft[~pt3d_pair_valid_rep] = 1.0
            # oflow_log_likelihood_max = oflow_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))
            oflow_log_likelihood_max = oflow_x_dev_abs_distr.log_prob(
                torch.zeros(size=(1,), device=device)
            ) * oflow_y_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))
            oflow_log_likelihood[~pt3d_pair_valid_rep] = oflow_log_likelihood_max.repeat(1, K, 1, 1)[
                ~pt3d_pair_valid_rep
            ]

        oflow_inlier_hard[~pt3d_valid_0_rep] = False
        oflow_inlier_soft[~pt3d_valid_0_rep] = 0.0
        oflow_log_likelihood[~pt3d_valid_0_rep] = oflow_log_likelihood_min.repeat(1, K, 1, 1)[~pt3d_valid_0_rep]

    if args["likelihood_use_disp"]:

        pred_disps = o4geo_pinhole.pt3d_2_disp(
            pts1_ftf.reshape(B * K, 3, H, W),
            proj_mats=proj_mat,
            baseline=baseline,
        ).reshape(B, K, 1, H, W)
        gt_disps2 = o4geo_pinhole.pt3d_2_disp(
            pts2.reshape(B, 3, H, W),
            proj_mats=proj_mat,
            baseline=baseline,
        ).reshape(B, 1, H, W)
        # disps_norm = torch.norm(gt_disps, dim=1, keepdim=True)

        if "disp_fwdbwd_dev" in data.keys() and args["disp_std_use_fwdbwd_dev_if_larger"]:
            # std_z = torch.max(args["likelihood_disp_abs_std"], data['disp_fwdbwd_dev'][:, 0:1].abs() / np.sqrt(2.))
            # std_z[data['disp_occ_0']] = args["std_factor_occ"] * args["likelihood_disp_abs_std"]
            valid_pxl = (~data["oflow_occ"]) * (~data["disp_occ_0"]) * pt3d_valid_0 * pt3d_pair_valid
            std_z = args["likelihood_disp_abs_std"] * (
                valid_pxl + (~valid_pxl) * args["std_factor_occ"]
            )  # * 10 # * np.sqrt(2.)
        else:
            # transl_z = models_params["se3"].reshape(B * K, 4, 4)[:, 2:3, 3][None, :, :, None]
            # z = pts1[:, 2:]
            std_z = args["likelihood_disp_abs_std"] * (
                pt3d_valid_0 + (~pt3d_valid_0) * args["std_factor_occ"]
            )  # * 10 #* np.sqrt(2.)
            # std_z = (((transl_z / (720 * 0.54)).abs()**2 + 1 + 1).sqrt()).repeat(1, 1, H, W) * args["likelihood_disp_abs_std"]
        # o4visual.visualize_img(std_z[0] / 10)
        # o4visual.visualize_img(data['disp_occ_0'][0])

        # B x K x 2 x H x W -> B x K x H x W
        disp_dev_abs = torch.norm(pred_disps - gt_disps2[:, None], dim=2)
        # disp_dev_abs = torch.clamp(disp_dev_abs, std_z*0, std_z*3)

        # f_x * b / t - (gt_disps - disps_0) ~ N(0, sqrt(2)*disp)
        # models_params["se3"]
        disp_dev_abs[~pt3d_valid_0_rep] = 0.0
        # disp_dev_rel = disp_dev_abs / (disps_norm + 1e-10)

        disp_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=std_z,  # args["likelihood_disp_abs_std"]
        )
        # disp_dev_rel_distr = torch.distributions.Normal(
        #    loc=0.0, scale=args["likelihood_disp_rel_std"]
        # )

        disp_dev_abs_inlier = 2 * (1.0 - disp_dev_abs_distr.cdf(disp_dev_abs))
        # disp_dev_rel_inlier = 2 * (1.0 - disp_dev_rel_distr.cdf(disp_dev_rel))
        disp_inlier_soft = disp_dev_abs_inlier
        # disp_inlier_soft = torch.max(disp_dev_abs_inlier, disp_dev_rel_inlier)

        disp_log_likelihood = disp_dev_abs_distr.log_prob(disp_dev_abs)
        disp_log_likelihood_min = disp_dev_abs_distr.log_prob(1e10 * torch.ones(size=(1,), device=device))

        disp_inlier_hard = disp_inlier_soft > args["inlier_disp_hard_threshold"]

        if args["likelihood_disp_invalid_pairs"] == 0.0:
            disp_inlier_hard[~pt3d_pair_valid_rep] = False
            disp_inlier_soft[~pt3d_pair_valid_rep] = 0.0
            # disp_log_likelihood[~pt3d_pair_valid_rep] = disp_log_likelihood_min[~pt3d_pair_valid_rep]
            disp_log_likelihood[~pt3d_pair_valid_rep] = disp_log_likelihood_min.repeat(1, K, 1, 1)[~pt3d_pair_valid_rep]
        elif args["likelihood_disp_invalid_pairs"] == 1.0:
            disp_inlier_hard[~pt3d_pair_valid_rep] = True
            disp_inlier_soft[~pt3d_pair_valid_rep] = 1.0
            disp_log_likelihood_max = disp_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))
            # disp_log_likelihood[~pt3d_pair_valid_rep] = disp_log_likelihood_max[~pt3d_pair_valid_rep]
            disp_log_likelihood[~pt3d_pair_valid_rep] = disp_log_likelihood_max.repeat(1, K, 1, 1)[~pt3d_pair_valid_rep]

        disp_inlier_hard[~pt3d_valid_0_rep] = False
        disp_inlier_soft[~pt3d_valid_0_rep] = 0.0
        # disp_log_likelihood[~pt3d_valid_0_rep] = disp_log_likelihood_min[~pt3d_valid_0_rep]
        disp_log_likelihood[~pt3d_valid_0_rep] = disp_log_likelihood_min.repeat(1, K, 1, 1)[~pt3d_valid_0_rep]

    inlier_soft = torch.ones_like(pt3d_pair_valid_rep)
    inlier_hard = torch.ones_like(pt3d_pair_valid_rep)
    log_likelihood = torch.zeros_like(pt3d_pair_valid_rep)
    if args["likelihood_use_oflow"]:
        log_likelihood = log_likelihood + oflow_log_likelihood
        inlier_soft = inlier_soft * oflow_inlier_soft
        inlier_hard = inlier_hard * oflow_inlier_hard
    if args["likelihood_use_sflow"]:
        log_likelihood = log_likelihood + sflow_log_likelihood
        inlier_soft = inlier_soft * sflow_inlier_soft
        inlier_hard = inlier_hard * sflow_inlier_hard
    if args["likelihood_use_disp"]:
        log_likelihood = log_likelihood + disp_log_likelihood
        inlier_soft = inlier_soft * disp_inlier_soft
        inlier_hard = inlier_hard * disp_inlier_hard

    inlier_hard = inlier_soft > args["inlier_hard_threshold"]  # 0.0455
    if not args["likelihood_use_oflow"] and not args["likelihood_use_sflow"] and not args["likelihood_use_disp"]:
        print("error: for se3-likelihood whether oflow, disp or sflow must be used.")

    if inlier_hard.isnan().sum() > 0:
        print("error: nans in se3 inlier hard")

    if inlier_soft.isnan().sum() > 0:
        print("error: nans in se3 inlier soft")

    if log_likelihood.isnan().sum() > 0:
        print("error: nans in se3 likelihood")

    log_likelihood = torch.clamp(log_likelihood, torch.log(torch.Tensor([1e-10])).item(), 99999.0)

    if data["B"] == -1:
        return inlier_hard[0], inlier_soft[0], log_likelihood[0]
    else:
        return inlier_hard, inlier_soft, log_likelihood
