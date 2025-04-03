import torch.linalg

import cups.scene_flow_2_se3.visual._2d as o4visual2d
from cups.scene_flow_2_se3.clustering import elemental as o4cluster
from cups.scene_flow_2_se3.geometric import grid as o4geo_grid

# from sklearn import svm


def fit(data, models_posterior, pts_per_object_count_max=1000):
    # pts: B x C x H x W / B x C x N
    # models_likelihood: B x K x H x W
    # means, stds: B x K x C
    # out: argmax(P(H|E)) B x K x C

    if data["B"] == -1:
        pts = data["pts"][None,]
        pts_valid = data["pts_valid"][None,]
        K, H, W = models_posterior.shape
        B = 1
        models_posterior = models_posterior[None,]

    else:
        pts = data["pts"]
        pts_valid = data["pts_valid"]
        B, K, H, W = models_posterior.shape

    inlier = models_posterior > 0.5
    N_max = pts_per_object_count_max  # 600 for FT3D 1200 for KITTI
    N = H * W
    inlier_too_much = inlier[inlier.flatten(2).sum(dim=2) > N_max].flatten(1)
    K_too_much = inlier_too_much.shape[0]
    indices_default = torch.arange(N, device=models_posterior.device)
    for k in range(K_too_much):
        N_too_much = inlier_too_much[k].sum()
        # inlier_too_much_pos = inlier_too_much[k]
        indices_too_much = indices_default[inlier_too_much[k]]
        indices_too_much_nullified = indices_too_much[torch.randperm(N_too_much)[N_max:]]
        inlier_too_much[k, indices_too_much_nullified] = False
    inlier_too_much = inlier_too_much.reshape(-1, H, W)
    inlier[inlier.flatten(2).sum(dim=2) > N_max] = inlier_too_much
    # N = inlier_b_k.sum()
    # if N > 1000:
    #    perm = torch.randperm(N)[:1000]

    C = pts.shape[1]
    params = {}
    # if pts.shape[1] ~= model_posterior.shape[1]:
    #    print("error :: euclidean-nn :: fit :: s")
    params["pts"] = pts
    params["pts_assign"] = inlier
    """For b in range(B): params["pts"].append([]) for k in range(K): inlier_b_k = models_posterior[b, k] > 0.5 pts_b_k
    = pts[b, :, inlier_b_k].flatten(1) inlier_b_k = inlier_b_k.flatten() N = inlier_b_k.sum()

    #o4visual2d.visualize_projected_pts3d(pts_b_k, data["proj_mat"], data["orig_H"], data["orig_W"])
    #o4visual2d.visualize_pts3d(pts_b_k)

    if N > 1000:     perm = torch.randperm(N)[:1000]

        inlier_b_k = inlier_b_k[perm]     pts_b_k = pts_b_k[:, perm] # clf = svm.SVC(C = 10) #
    clf.fit(pts[b].flatten(1).permute(1, 0).detach().cpu().numpy(), inlier_b_k.flatten().detach().cpu().numpy()) # svs =
    clf.support_vectors_ params["pts"][b].append(pts_b_k) #print("collected points", inlier_b_k.sum())
    """
    # o4visual.visualize_pts3d(pts3d=params['pts'][0])
    # +- 1 std: 68.27%
    # +- 2 std: 95.45%
    # +- 3 std: 99.73%
    if data["B"] == -1:
        params["pts"] = params["pts"][0]
        params["pts_assign"] = params["pts_assign"][0]

    return params


def likelihood(data, params, inlier_hard_thresh, std_depth, std_uv, max=2.0):
    # B x 3 x H x W / B x 3 x N

    if data["B"] == -1:
        raw_pts = data["pts"][None,]
        depth_valid = data["depth_valid_0"][None,]
        model_pts = params["pts"][None,]
        model_pts_assign = params["pts_assign"][None,]
        # model_pts = [params["pts"]]
    else:
        raw_pts = data["pts"]
        depth_valid = data["depth_valid_0"]
        model_pts = params["pts"]
        model_pts_assign = params["pts_assign"]
    B, K, H_model, W_model = model_pts_assign.shape  # [1]
    B, _, H, W = raw_pts.shape
    dtype = raw_pts.dtype
    device = raw_pts.device
    raw_pts = raw_pts.flatten(2)
    depth_valid = depth_valid.flatten(2)
    B = len(model_pts)

    data_pxl2d = o4geo_grid.shape_2_pxl2d(B, H=H, W=W, dtype=dtype, device=device) / (W - 1)
    model_pxl2d = o4visual2d.resize(data_pxl2d, H_out=H_model, W_out=W_model, mode="nearest_v2")

    data_depth = raw_pts[:, 2:].clone()
    model_depth = model_pts[:, 2:].clone()

    # fx = torch.Tensor([721.5377]).to(device)
    # baseline = torch.Tensor([0.54]).to(device)
    # data_depth = o4geo_pin.depth_2_disp(data_depth.reshape(B, 1, H, W), fx=fx, baseline=baseline).reshape(B, 1, H, W)
    # data_depth = 1./ data_depth
    # model_depth = o4geo_pin.depth_2_disp(model_depth.reshape(B, 1, H_model, W_model), fx=fx, baseline=baseline).reshape(B, 1, H_model, W_model)

    if raw_pts.flatten(2).shape[2] == model_pts.flatten(2).shape[2]:
        dists3d = torch.norm(raw_pts.flatten(2)[:, :, :, None] - model_pts.flatten(2)[:, :, None, :], dim=1)
        dists3d[~depth_valid[:, 0]] = torch.norm(
            raw_pts.flatten(2).permute(0, 2, 1)[~depth_valid[:, 0]][:, None]
            / (raw_pts.flatten(2).permute(0, 2, 1)[~depth_valid[:, 0]][:, None, 2:].abs() + 1e-10)
            - model_pts.flatten(2).permute(0, 2, 1) / (model_pts.flatten(2).permute(0, 2, 1)[:, :, 2:].abs() + 1e-10),
            dim=2,
        )

        # dists_pxl2d = torch.norm(data_pxl2d.flatten(2)[:, : , :, None] - model_pxl2d.flatten(2)[:, :, None, :], dim=1)

        dists_pxl2d_x = torch.norm(
            data_pxl2d.flatten(2)[:, 0:1, :, None] - model_pxl2d.flatten(2)[:, 0:1, None, :], dim=1
        )
        dists_pxl2d_y = torch.norm(
            data_pxl2d.flatten(2)[:, 1:2, :, None] - model_pxl2d.flatten(2)[:, 1:2, None, :], dim=1
        )
        dists_depth = torch.norm(
            data_depth.flatten(2)[:, :, :, None] - model_depth.flatten(2)[:, :, None, :], dim=1
        ) / (
            torch.norm(data_depth.flatten(2)[:, :, :, None] + model_depth.flatten(2)[:, :, None, :], dim=1) / 2.0
            + 1e-10
        )

    else:
        dists3d = []
        # dists_pxl2d = []
        dists_pxl2d_x = []
        dists_pxl2d_y = []
        dists_depth = []
        for b in range(B):
            dists3d.append([])
            # dists_pxl2d.append([])
            dists_pxl2d_x.append([])
            dists_pxl2d_y.append([])
            dists_depth.append([])
            for k in range(K):
                dists3d[b].append(
                    torch.norm(
                        raw_pts.flatten(2)[b, :, :, None]
                        - model_pts.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                        dim=0,
                    )
                )
                dists3d[b][k][~depth_valid[b, 0]] = torch.norm(
                    raw_pts.flatten(2).permute(0, 2, 1)[b][~depth_valid[b, 0]][:, None]
                    / (raw_pts.flatten(2).permute(0, 2, 1)[b, ~depth_valid[b, 0]][:, None, 2:].abs() + 1e-10)
                    - model_pts.flatten(2)[b, :, model_pts_assign[b][k].flatten()].permute(1, 0)
                    / (model_pts.flatten(2)[b, :, model_pts_assign[b][k].flatten()].permute(1, 0)[:, 2:].abs() + 1e-10)[
                        None,
                    ],
                    dim=2,
                )
                """
                #dists_pxl2d[b].append(torch.norm(data_pxl2d.flatten(2)[b, :, :, None] - model_pxl2d.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                #                         dim=0))
                dists_pxl2d_x[b].append(torch.norm(data_pxl2d.flatten(2)[b, 0:1, :, None] - model_pxl2d.flatten(2)[b, 0:1,
                                                                                        model_pts_assign[b][
                                                                                            k].flatten()][:, None, :],
                                                 dim=0))
                dists_pxl2d_y[b].append(torch.norm(data_pxl2d.flatten(2)[b, 1:2, :, None] - model_pxl2d.flatten(2)[b, 1:2,
                                                                                        model_pts_assign[b][
                                                                                            k].flatten()][:, None, :],
                                                 dim=0))
                dists_depth[b].append(torch.norm(data_depth.flatten(2)[b, :, :, None] - model_depth.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                                         dim=0) / (torch.norm(data_depth.flatten(2)[b, :, :, None] + model_depth.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                                         dim=0)) / 2. + 1e-10)
                """
            # dists[b] = torch.stack(dists[b])
        # dists = torch.stack(dists)
    # dists = torch.norm(raw_pts[:, :, :, None] - model_pts[b][k][:, None, :], dim=0)
    # pts_dist3d_distr = torch.distributions.Normal(loc=0.0, scale=std)
    distr_dist_pxl2d = torch.distributions.Normal(loc=0.0, scale=std_uv)  # 0.12 25 # 150 / 1242
    distr_dist_depth = torch.distributions.Normal(loc=0.0, scale=std_depth)  # 0.13 1.5

    dists3d_min = []
    # dists_2d_min = []
    dists_2d_min_x = []
    dists_2d_min_y = []
    dists_depth_min = []

    for b in range(B):
        dists3d_min.append([])
        # dists_2d_min.append([])
        dists_2d_min_x.append([])
        dists_2d_min_y.append([])
        dists_depth_min.append([])
        for k in range(K):
            # print("geo pts shape", model_pts[b][k].shape)
            # dists_b_k = dists[b][]
            if model_pts_assign[b][k].sum() == 0:
                dist3d_min = torch.ones(size=(H * W,), dtype=dtype, device=device) * float("inf")
                # dist_min_2d = torch.ones(size=(H*W,), dtype=dtype, device=device) * float('inf')
                dist_min_2d_x = torch.ones(size=(H * W,), dtype=dtype, device=device) * float("inf")
                dist_min_2d_y = torch.ones(size=(H * W,), dtype=dtype, device=device) * float("inf")
                dist_min_depth = torch.ones(size=(H * W,), dtype=dtype, device=device) * float("inf")
            else:
                # dist_min, _ = torch.min(
                #    torch.norm(raw_pts[b, :, :, None] - model_pts[b][k][:, None, :], dim=0),
                #    dim=1,
                # )
                if raw_pts.flatten(2).shape[2] == model_pts.flatten(2).shape[2]:
                    dist3d_min, dist_min_ids = torch.min(dists3d[b, :, model_pts_assign[b][k].flatten()], dim=1)
                    id_max = model_pts_assign[b][k].sum() - 1
                    dist_min_ids_onehot = o4cluster.label_2_onehot(dist_min_ids, label_max=id_max).permute(1, 0)
                    # dist_min_2d = dists_pxl2d[b][:, model_pts_assign[b][k].flatten()][dist_min_ids_onehot]
                    dist_min_2d_x = dists_pxl2d_x[b][:, model_pts_assign[b][k].flatten()][dist_min_ids_onehot]
                    dist_min_2d_y = dists_pxl2d_y[b][:, model_pts_assign[b][k].flatten()][dist_min_ids_onehot]

                    dist_min_depth = dists_depth[b][:, model_pts_assign[b][k].flatten()][dist_min_ids_onehot]
                else:
                    dist3d_min, dist_min_ids = torch.min(dists3d[b][k], dim=1)
                    id_max = dists3d[b][k].shape[1] - 1
                    dist_min_ids_onehot = o4cluster.label_2_onehot(dist_min_ids, label_max=id_max).permute(1, 0)
                    # dist_min_2d = dists_pxl2d[b][k][dist_min_ids_onehot]
                    # dist_min_2d_x = dists_pxl2d_x[b][k][dist_min_ids_onehot]
                    # dist_min_2d_y = dists_pxl2d_y[b][k][dist_min_ids_onehot]
                    # dist_min_depth = dists_depth[b][k][dist_min_ids_onehot]

                    dist_min_2d_x = torch.norm(
                        data_pxl2d.flatten(2)[b, 0:1, :, None]
                        - model_pxl2d.flatten(2)[b, 0:1, model_pts_assign[b][k].flatten()][:, None, :],
                        dim=0,
                    )[dist_min_ids_onehot]

                    dist_min_2d_y = torch.norm(
                        data_pxl2d.flatten(2)[b, 1:2, :, None]
                        - model_pxl2d.flatten(2)[b, 1:2, model_pts_assign[b][k].flatten()][:, None, :],
                        dim=0,
                    )[dist_min_ids_onehot]

                    dist_min_depth = (
                        torch.norm(
                            data_depth.flatten(2)[b, :, :, None]
                            - model_depth.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                            dim=0,
                        )
                        / (
                            torch.norm(
                                data_depth.flatten(2)[b, :, :, None]
                                + model_depth.flatten(2)[b, :, model_pts_assign[b][k].flatten()][:, None, :],
                                dim=0,
                            )
                            / 2.0
                        )
                        + 1e-10
                    )[dist_min_ids_onehot]
                    #
            dist3d_min = dist3d_min.reshape(H, W)
            # dist_min_2d = dist_min_2d.reshape(H, W)
            dist_min_2d_x = dist_min_2d_x.reshape(H, W)
            dist_min_2d_y = dist_min_2d_y.reshape(H, W)
            dist_min_depth[~depth_valid[0, 0]] = 0.0
            dist_min_depth = dist_min_depth.reshape(H, W)

            dists3d_min[b].append(dist3d_min)
            # dists_2d_min[b].append(dist_min_2d)
            dists_2d_min_x[b].append(dist_min_2d_x)
            dists_2d_min_y[b].append(dist_min_2d_y)
            dists_depth_min[b].append(dist_min_depth)

        dists3d_min[b] = torch.stack(dists3d_min[b])
        # dists_2d_min[b] = torch.stack(dists_2d_min[b])
        dists_2d_min_x[b] = torch.stack(dists_2d_min_x[b])
        dists_2d_min_y[b] = torch.stack(dists_2d_min_y[b])
        dists_depth_min[b] = torch.stack(dists_depth_min[b])
    dists3d_min = torch.stack(dists3d_min)
    # dists_2d_min = torch.stack(dists_2d_min)
    dists_2d_min_x = torch.stack(dists_2d_min_x)
    dists_2d_min_y = torch.stack(dists_2d_min_y)
    dists_depth_min = torch.stack(dists_depth_min)

    raw_pts = raw_pts.reshape(B, 3, H, W)

    # inlier_soft = ((2 * (1.0 - distr_dist_pxl2d.cdf(dists_2d_min))) * (2 * (1.0 - distr_dist_depth.cdf(dists_depth_min))))
    inlier_2d_x_soft = 2 * (1.0 - distr_dist_pxl2d.cdf(dists_2d_min_x))
    inlier_2d_y_soft = 2 * (1.0 - distr_dist_pxl2d.cdf(dists_2d_min_y))
    inlier_depth_soft = 2 * (1.0 - distr_dist_depth.cdf(dists_depth_min))
    inlier_soft = inlier_depth_soft * inlier_2d_x_soft * inlier_2d_y_soft
    # (2 * (1.0 - distr_dist_pxl2d.cdf(dists_2d_min))) > 0.0455) * ((2 * (1.0 - distr_dist_depth.cdf(dists_depth_min)))
    inlier_hard = (
        inlier_soft > inlier_hard_thresh
    )  # (inlier_depth_soft > 0.0455) * (inlier_2d_x_soft > 0.0455) * (inlier_2d_y_soft > 0.0455)
    # log_likelihood = distr_dist_pxl2d.log_prob(dists_2d_min) + distr_dist_depth.log_prob(dists_depth_min)
    log_likelihood = (
        distr_dist_pxl2d.log_prob(dists_2d_min_x)
        + distr_dist_pxl2d.log_prob(dists_2d_min_y)
        + distr_dist_depth.log_prob(dists_depth_min)
    )

    log_likelihood = torch.clamp(log_likelihood, torch.log(torch.Tensor([1e-10])).item(), 99999.0)

    # dist_min_per_depth_m = dists3d_min / (raw_pts[:, 2:] + 1e-10)
    # inlier_soft = 2 * (1.0 - pts_dist3d_distr.cdf(dist_min_per_depth_m))
    # inlier_hard = inlier_soft > max
    # log_likelihood = pts_dist3d_distr.log_prob(dist_min_per_depth_m)
    # log_likelihood = torch.clamp(log_likelihood, torch.log(torch.Tensor([1e-10])).item(), 99999.)
    # inlier_soft = 2 * (1.0 - pts_dist3d_distr.cdf(dists3d_min))

    if data["B"] == -1:
        log_likelihood = log_likelihood[0]
        inlier_hard = inlier_hard[0]
        inlier_soft = inlier_soft[0]

    return inlier_hard, inlier_soft, log_likelihood


# def calc_inlier_hard(dist, std, hard_threshold):
def calc_inlier_hard(dist, std_depth, std_uv, hard_threshold):
    inlier_soft = calc_inlier_soft(dist, std_depth, std_uv)
    inlier_hard = inlier_soft > hard_threshold
    # inlier_hard = dist < 2*std
    return inlier_hard


# def calc_inlier_soft(dist, std):
def calc_inlier_soft(dist, std_depth, std_uv):
    distr_dist_pxl2d = torch.distributions.Normal(loc=0.0, scale=std_uv)
    distr_dist_depth = torch.distributions.Normal(loc=0.0, scale=std_depth)

    inlier_2d_x_soft = 2 * (1.0 - distr_dist_pxl2d.cdf(dist[:, 0:1].abs()))
    inlier_2d_y_soft = 2 * (1.0 - distr_dist_pxl2d.cdf(dist[:, 1:2].abs()))
    inlier_depth_soft = 2 * (1.0 - distr_dist_depth.cdf(dist[:, 2:3].abs()))

    inlier_soft = inlier_2d_x_soft * inlier_2d_y_soft * inlier_depth_soft
    return inlier_soft
