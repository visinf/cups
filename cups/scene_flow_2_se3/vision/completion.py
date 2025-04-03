import torch

import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
import cups.scene_flow_2_se3.rearrange as o4rearr
import cups.scene_flow_2_se3.vision.visualization as o4visual


def neighbor_vote(mask, img, pts3d):
    B, K, H, W = mask.shape
    scale_factor = 0.2
    patch_size = 3
    mask_down = o4visual.resize(mask, scale_factor=scale_factor, mode="nearest")
    img_down = o4visual.resize(img, scale_factor=scale_factor, mode="nearest")
    pts3d_down = o4visual.resize(pts3d, scale_factor=scale_factor, mode="nearest")

    while True:
        # o4vis.visualize_img(o4vis.mask2rgb(mask_down[0]))
        mask_down_fix = (torch.max(mask_down, dim=1, keepdim=True)[0] == 1) * (
            torch.sum(mask_down, dim=1, keepdim=True) == 1
        )
        # o4vis.visualize_img(mask_down_fix[0, :1])
        if (~mask_down_fix).sum() == 0:
            break

        mask_down = neighbor_vote_step(mask_down, img_down, pts3d_down, mask_down_fix, patch_size=patch_size)

    mask_fix = (torch.max(mask, dim=1, keepdim=True)[0] == 1) * (torch.sum(mask, dim=1, keepdim=True) == 1)
    mask_fix = mask_fix.repeat(1, K, 1, 1)

    mask_filled = o4visual.resize(mask_down, H_out=H, W_out=W, mode="nearest")

    mask[~mask_fix] = mask_filled[~mask_fix]
    return mask


def neighbor_vote_step(mask, img, pts3d, mask_fix, patch_size):
    # img: B x C x H x W
    # pts3d: B x C x H x W
    # mask: B x K x H x W
    K = mask.shape[1]
    # mask_fix: B x 1 x H x W
    mask_fix = mask_fix.repeat(1, K, 1, 1)

    # neighbors: B x C x P x H x W (N = patch_size**2)
    # dist: B x 1 x P x H x W
    img_neighbors = o4rearr.neighbors_to_channels(img, patch_size=patch_size, new_dim=True)
    img_dist = torch.norm(img[:, :, None, :, :] - img_neighbors, dim=1, keepdim=True)
    pts3d_neighbors = o4rearr.neighbors_to_channels(pts3d, patch_size=patch_size, new_dim=True)
    pts3d_dist = torch.norm(pts3d[:, :, None, :, :] - pts3d_neighbors, dim=1, keepdim=True)

    mask_neighbors = o4rearr.neighbors_to_channels(mask * mask_fix, patch_size=patch_size, new_dim=True)

    dist = 1 * img_dist / img_dist.mean() + 1 * pts3d_dist / pts3d_dist.mean() + 1e-8
    weight = 1.0 / dist

    # for i in range(81):
    #    o4vis.visualize_img(weight[0, :, i])

    mask_new = (mask_neighbors * weight).sum(dim=2, keepdim=False)  # / weight.sum(dim=2)
    mask_new_vals, mask_new_inds = torch.max(mask_new, dim=1)
    mask_new = torch.nn.functional.one_hot(mask_new_inds, num_classes=K).permute(0, 3, 1, 2).bool()
    mask_new_valid = (~mask_fix) * (mask_new_vals[:, None] > 0.0)

    # o4vis.visualize_img(mask_new_vals > 0.0)
    # o4vis.visualize_img(mask_new_valid[:, 0])

    mask[mask_new_valid] = mask_new[mask_new_valid]

    return mask


def batchwise_maskwise_vars_means(x, mask, pixel_weights=None):
    # x: B x C x H x W
    # mask: B x K x H x W
    # pixel_weights = B x 1 x H x W
    B, C, H, W = x.shape
    B, K, H, W = mask.shape

    vars = torch.zeros(size=(B, K, C), device=x.device)
    means = torch.zeros(size=(B, K, C), device=x.device)

    if pixel_weights is not None:
        mask = pixel_weights * mask

    means = (mask[:, :, None] * x[:, None]).flatten(3).sum(dim=3) / (mask[:, :, None].flatten(3).sum(dim=3) + 1e-8)
    vars = (mask[:, :, None] * (x[:, None] - means[:, :, :, None, None]) ** 2).flatten(3).sum(dim=3) / (
        mask[:, :, None].flatten(3).sum(dim=3) + 1e-8
    )
    # for k in range(K):
    #    for b in range(B):
    #        x_batch_mask = x[b, :, mask[b, k]]
    #        if pixel_weights is not None:
    #            w_batch_mask = pixel_weights[b, :, mask[b, k]]
    #            x_mean = w_batch_mask
    #        var, mean = torch.var_mean(x[b, :, mask[b, k]], dim=1)
    #        # C
    #        vars[b, k, :] = var[:]
    #        means[b, k, :] = mean[:]

    return vars, means


def kmeans(
    mask,
    img,
    pts3d,
    pred_flow,
    gt_flow,
    proj_mats,
    depth_valid,
    visualize_progression=False,
):
    # img: B x C x H x W
    # pts3d: B x C x H x W
    # pts3d_ftf: B x K x C x H x W
    # mask: B x K x H x W

    B, K, H, W = mask.shape

    dev_flow = torch.norm(pred_flow - gt_flow, dim=1)[None,]
    # corr_flow = (torch.sum(pred_flow * gt_flow, dim=1) / torch.sqrt(torch.norm(pred_flow, dim=1)**2 + torch.norm(gt_flow, dim=1)**2))[None,]
    # prob_flow = torch.softmax( torch.exp(-dev_flow) + (1 - torch.exp(-dev_flow)) * corr_flow, dim=1)
    prob_flow = torch.softmax(-dev_flow, dim=1)
    # *
    # mask_fix: B x 1 x H x W
    mask_fix = (torch.max(mask, dim=1, keepdim=True)[0] == 1) * (torch.sum(mask, dim=1, keepdim=True) == 1)
    mask_fix = mask_fix.repeat(1, K, 1, 1)

    mask_rgb = o4visual.mask2rgb(mask[0])

    mask_count = mask.sum(dim=1, keepdim=True)
    mask_count[mask_count == 0] = 1
    #
    # mask_empty = torch.sum(mask, dim=1, keepdim=True) == 0
    for i in range(1):
        # 1 x B x H x W
        feats = torch.cat((img, pts3d), dim=1)  # flow
        mask_binary = mask.clone()
        mask_binary[~mask_fix] = 0.0
        pixel_weights = (1.0 / pts3d[:, 2:]) * (1.0 / mask_count)
        cluster_vars, cluster_means = batchwise_maskwise_vars_means(feats, mask, pixel_weights=pixel_weights)

        cluster_geo_means = cluster_means[:, :, 3:]
        cluster_geo_means_proj2d = o4geo_pinhole.pt3d_2_pxl2d(cluster_geo_means, proj_mats)
        mask_rgb = o4visual.draw_pixels(mask_rgb, cluster_geo_means_proj2d.reshape(-1, 2))

        if visualize_progression:
            o4visual.visualize_img(mask_rgb)
        pts_cluster_dev = torch.abs(feats[:, None] - cluster_means[:, :, :, None, None])

        pts_cluster_geo_dev = torch.norm(pts_cluster_dev[:, :, 3:], dim=2)
        pts_cluster_geo_var = torch.norm(cluster_vars[:, :, 3:], dim=2)
        # pts_cluster_geo_dist = pts_cluster_geo_dist / pts_cluster_geo_dist.mean() * 1
        pts_cluster_vis_dev = torch.norm(pts_cluster_dev[:, :, :3], dim=2)
        pts_cluster_vis_var = torch.norm(cluster_vars[:, :, :3], dim=2)
        # pts_cluster_vis_dist = pts_cluster_vis_dist / pts_cluster_vis_dist.mean() / 100
        # cluster_vars[:, :, 6:] = 1
        # cluster_vars[:, :, 3:] = torch.mean(cluster_vars[:, :, 3:], dim=2, keepdim=True)
        # cluster_vars[:, :, :3] = torch.mean(cluster_vars[:, :, :3], dim=2, keepdim=True)

        cluster_vars[:, :, :3] = cluster_vars[:, :, :3]  # + 200
        cluster_vars[:, :, 3:] = cluster_vars[:, :, 3:]  # + 2

        pts_cluster_feats_prob = (1.0 / torch.sqrt(2 * 3.14 * (cluster_vars[:, :, :, None, None]))) * torch.exp(
            -(pts_cluster_dev**2) / (2 * cluster_vars[:, :, :, None, None])
        )

        pts_cluster_geo_prob = torch.prod(pts_cluster_feats_prob[:, :, 3:], dim=2)
        # pts_cluster_geo_prob = torch.softmax(pts_cluster_geo_prob + 0., dim=1)
        pts_cluster_vis_prob = torch.prod(pts_cluster_feats_prob[:, :, :3], dim=2)
        pts_cluster_vis_prob = torch.softmax(pts_cluster_vis_prob + 0.1, dim=1)

        pts_cluster_feats_prob = torch.prod(pts_cluster_feats_prob[:, :, 3:], dim=2)
        pts_cluster_feats_prob = torch.softmax(pts_cluster_feats_prob, dim=1)

        # pts_cluster_geo_prob = torch.softmax(-pts_cluster_geo_dist ** 2, dim=1)
        # pts_cluster_vis_prob = torch.softmax(-pts_cluster_vis_dist ** 2, dim=1)
        # pts_cluster_prob = torch.prod(pts_cluster_feats_prob, dim=2)
        pts_cluster_mask_prob = torch.softmax(mask * 1.0, dim=1)

        pts_cluster_prob = (
            prob_flow * pts_cluster_mask_prob * pts_cluster_geo_prob
        )  # * pts_cluster_vis_prob # pts_cluster_feats_prob #
        # pts_cluster_prob[pts_cluster_prob < 0.1] = 0.
        # should have something like minimum distance to random pts for rgb/

        # o4vis.visualize_img(o4vis.mask2rgb(pts_cluster_prob[0]))
        mask_new_vals, mask_new_inds = torch.max(pts_cluster_prob, dim=1)
        mask_new = torch.nn.functional.one_hot(mask_new_inds, num_classes=K).permute(0, 3, 1, 2).bool()

        for b in range(B):
            # max_var_id = torch.argmax(torch.norm(cluster_vars[:, :, 3:], dim=2), dim=1)
            max_pts_id = mask_new[b, :, depth_valid[b, 0]].sum(dim=1).argmax(dim=0)
            mask_new_inds[b, ~depth_valid[b, 0]] = max_pts_id
        mask_new = torch.nn.functional.one_hot(mask_new_inds, num_classes=K).permute(0, 3, 1, 2).bool()

        mask = mask_new

        # mask = pts_cluster_prob
        # o4vis.visualize_img(o4vis.mask2rgb(mask[0]))
        # mask[~mask_fix] = mask_new[~mask_fix]

    return mask
