import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
import cups.scene_flow_2_se3.geometric.se3.fit.pt3d_oflow as o4geo_se3_fit_pt3d_oflow
import cups.scene_flow_2_se3.geometric.se3.transform as o4geo_se3_transf
import cups.scene_flow_2_se3.visual._2d as o4visual2d
from cups.scene_flow_2_se3.retrieval.sflow2se3.sflow import SFlow


class DRPCs:
    def __init__(self, se3, pt3d=None, pt3d_assign=None, depth_reliable=None):
        """Create new Dynamic Rigid Point Clouds (DRPCs) from parameters.

        Parameters
        ----------
        pt3d torch.Tensor: 3xHxW/3xN, points in 3D which are possibly assigned to one DRPC
        pt3d_assign torch.Tensor: KxHxW/KxN, assignment of points in 3D to DRPCs
        depth_reliable torch.Tensor: 1xHxW/1xN, indicator if depth of 3D point is reliable
        se3 torch.Tensor: Kx4x4, se(3)-motions corresponding to DRPCs
        """

        self.se3 = se3
        self.device = self.se3.device
        self.pt3d = pt3d
        self.K = self.se3.size(0)
        self.pt3d_assign = pt3d_assign
        self.depth_reliable = depth_reliable
        if pt3d is not None:
            self.model = "joint"
        else:
            self.model = "motion"

        self.sflow_log_likelihood = None
        self.sflow_inlier_soft = None
        self.sflow_inlier_hard = None

    def calc_se3_pt3d_1(self, sflow):

        self.calc_sflow_consensus(sflow)

        se3_pt3d_1 = o4geo_se3_transf.pts3d_transform(sflow.pt3d_0.repeat(self.K, 1, 1, 1), self.se3)[:, :]
        # se3_pt3d_1_id = self.sflow_log_likelihood.argmax()
        return se3_pt3d_1

    def calc_sflow_consensus(self, sflow, update_pt3d_0=False, update_pt3d_1=False):
        if update_pt3d_0:
            self.pt3d = sflow.pt3d_0
        if self.model == "motion":
            self.sflow_log_likelihood, self.sflow_inlier_soft = self.calc_sflow_motion_consensus(
                sflow, update_pt3d_1=update_pt3d_1
            )
            self.sflow_inlier_hard = self.sflow_inlier_soft > sflow.inlier_hard_thresh
            # return sflow_log_likelihood, sflow_inlier_soft, sflow_inlier_hard
        else:
            sflow_motion_log_likelihood, sflow_motion_inlier_soft = self.calc_sflow_motion_consensus(
                sflow, update_pt3d_1=update_pt3d_1
            )
            sflow_spatial_log_likelihood, sflow_spatial_inlier_soft = self.calc_sflow_spatial_consensus(sflow)
            self.sflow_log_likelihood = sflow_motion_log_likelihood + sflow_spatial_log_likelihood
            self.sflow_inlier_soft = sflow_motion_inlier_soft * sflow_spatial_inlier_soft
            self.sflow_inlier_hard = self.sflow_inlier_soft > sflow.inlier_hard_thresh
            # return sflow_log_likelihood, sflow_inlier_soft, sflow_inlier_hard

        if update_pt3d_1:
            self.max_log_likelihood_label = self.sflow_log_likelihood.argmax(dim=0)[None]
            self.max_log_likelihood_label[:, self.sflow_inlier_soft.max(dim=0)[0] == 0.0] = 0
            self.max_log_likelihood_label[~sflow.depth_reliable_0] = 0
            self.max_log_likelihood_onehot = F.one_hot(self.max_log_likelihood_label[0], num_classes=self.K).permute(
                2, 0, 1
            )
            se3_pts3d_1 = o4geo_se3_transf.pts3d_transform(sflow.pt3d_0.repeat(self.K, 1, 1, 1), self.se3)[:, :]
            self.pt3d_1 = (se3_pts3d_1 * self.max_log_likelihood_onehot[:, None]).sum(dim=0)

        if update_pt3d_0:
            # self.max_log_likelihood_label = self.sflow_log_likelihood.argmax(dim=0)[None]
            # self.max_log_likelihood_label[:, self.sflow_inlier_soft.max(dim=0)[0] == 0.] = 0
            # self.max_log_likelihood_onehot = F.one_hot(self.max_log_likelihood_label[0], num_classes=self.K).permute(2, 0, 1)
            self.pt3d_assign = self.sflow_inlier_hard * sflow.depth_reliable_0

    def calc_sflow_motion_consensus(self, sflow: SFlow, update_pt3d_1=False):
        device = self.device

        # print(self.K)
        # print(sflow.pt3d_0.shape)
        se3_pts3d_1 = o4geo_se3_transf.pts3d_transform(sflow.pt3d_0.repeat(self.K, 1, 1, 1), self.se3)[:, :]

        # print(se3_pts3d_1.shape)
        se3_oflow = o4geo_pinhole.pt3d_2_oflow(
            se3_pts3d_1,
            sflow.cam_int[None,],
            orig_H=sflow.cam_H,
            orig_W=sflow.cam_W,
            resize_mode=sflow.resize_mode,
        )

        oflow_x_dev_abs_distr = torch.distributions.Normal(loc=0.0, scale=sflow.std_oflow_x)
        oflow_y_dev_abs_distr = torch.distributions.Normal(loc=0.0, scale=sflow.std_oflow_y)

        oflow_dev_abs = (sflow.oflow[None,] - se3_oflow).abs()

        oflow_log_likelihood = oflow_x_dev_abs_distr.log_prob(oflow_dev_abs[:, 0]) + oflow_y_dev_abs_distr.log_prob(
            oflow_dev_abs[:, 1]
        )

        oflow_inlier_soft = (
            2
            * (1.0 - oflow_x_dev_abs_distr.cdf(oflow_dev_abs[:, 0]))
            * 2
            * (1.0 - oflow_y_dev_abs_distr.cdf(oflow_dev_abs[:, 1]))
        )

        oflow_log_likelihood_max = oflow_x_dev_abs_distr.log_prob(
            torch.zeros(size=(1,), device=device)
        ) * oflow_y_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))

        oflow_inlier_soft[:, ~sflow.depth_reliable_0[0]] = 0.0
        oflow_log_likelihood[:, ~sflow.depth_reliable_0[0]] = oflow_log_likelihood_max

        se3_disps_1 = o4geo_pinhole.pt3d_2_disp(
            se3_pts3d_1,
            proj_mats=sflow.cam_int[None,],
            baseline=sflow.cam_baseline,
        )
        sflow_disps_1 = o4geo_pinhole.pt3d_2_disp(
            sflow.pt3d_1[None,],
            proj_mats=sflow.cam_int[None,],
            baseline=sflow.cam_baseline,
        )[0]

        # B x K x 2 x H x W -> B x K x H x W
        disp_dev_abs = torch.norm(
            se3_disps_1 - sflow_disps_1[None,],
            dim=1,
        )

        disp_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=sflow.std_disp_temp,  # args["likelihood_disp_abs_std"]
        )

        disp_inlier_soft = 2 * (1.0 - disp_dev_abs_distr.cdf(disp_dev_abs))

        disp_log_likelihood = disp_dev_abs_distr.log_prob(disp_dev_abs)
        disp_log_likelihood_max = disp_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=device))

        disp_inlier_soft[:, ~sflow.depth_reliable_0[0]] = 0.0
        disp_inlier_soft[:, ~sflow.depth_reliable_1[0]] = 1.0
        disp_log_likelihood[:, ~sflow.depth_reliable_0[0]] = disp_log_likelihood_max
        disp_log_likelihood[:, ~sflow.depth_reliable_1[0]] = disp_log_likelihood_max

        inlier_soft = oflow_inlier_soft * disp_inlier_soft
        log_likelihood = oflow_log_likelihood + disp_log_likelihood
        # log_likelihood = torch.clamp(log_likelihood, torch.log(torch.Tensor([1e-10])).item(), 99999.)

        return log_likelihood, inlier_soft

    def calc_dev_nn_spatial_consensus(self, dev_nn_x, dev_nn_y, sflow):
        dev_nn_x_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=sflow.std_nn_x,
        )
        dev_nn_y_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=sflow.std_nn_y,
        )

        # dev_nn_x_likelihood_max = dev_nn_x_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=dev_nn_x.device))
        # dev_nn_y_likelihood_max = dev_nn_y_dev_abs_distr.log_prob(torch.zeros(size=(1,), device=dev_nn_x.device))

        dev_nn_x_log_likelihood = dev_nn_x_dev_abs_distr.log_prob(dev_nn_x)
        dev_nn_x_inlier_soft = 2 * (1.0 - dev_nn_x_dev_abs_distr.cdf(dev_nn_x))
        dev_nn_y_log_likelihood = dev_nn_y_dev_abs_distr.log_prob(dev_nn_y)
        dev_nn_y_inlier_soft = 2 * (1.0 - dev_nn_y_dev_abs_distr.cdf(dev_nn_y))

        # dev_nn_x_inlier_soft[:, ~sflow.depth_reliable_0[0]] = 0.0
        # dev_nn_x_log_likelihood[:, ~sflow.depth_reliable_0[0]] = dev_nn_x_likelihood_max
        # dev_nn_y_inlier_soft[:, ~sflow.depth_reliable_0[0]] = 0.0
        # dev_nn_y_log_likelihood[:, ~sflow.depth_reliable_0[0]] = dev_nn_y_likelihood_max

        dev_nn_inlier_soft = dev_nn_x_inlier_soft * dev_nn_y_inlier_soft
        dev_nn_log_likelihood = dev_nn_x_log_likelihood + dev_nn_y_log_likelihood

        return dev_nn_log_likelihood, dev_nn_inlier_soft

    def calc_sflow_spatial_consensus(self, sflow):
        # std_depth = args.sflow2se3_model_euclidean_nn_rel_depth_dev_std,
        # std_uv = args.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std,

        device = sflow.pt3d_0.device

        sflow_pt3d_hom = sflow.pt3d_0 / (sflow.pt3d_0[2:] + 1e-10)

        # self.pt3d / self.pt3d_assign : 3xHxW/KxHxW , range 3
        # 1. construct 3x3**2xHxW
        range = 9
        kernels = (
            F.one_hot(torch.arange(range**2, device=device))
            .type(dtype=self.pt3d_hom.dtype)
            .reshape(range**2, range, range)
        )
        pt3d_hom_conv2d = F.conv2d(self.pt3d_hom[:, None], weight=kernels[:, None], padding=range // 2)
        # 2. construct Kx3**2xHxW
        pt3d_assign_conv2d = F.conv2d(
            self.pt3d_assign[:, None].type(kernels.dtype), weight=kernels[:, None], padding=range // 2
        )
        # 3. upsample
        # o4visual2d.visualize_imgs(self.pt3d_assign[:, None])
        # o4visual2d.visualize_imgs(pt3d_assign_conv2d[0][:, None])
        pt3d_hom_conv2d_up = o4visual2d.resize(pt3d_hom_conv2d, H_out=sflow.H, W_out=sflow.W, mode=sflow.resize_mode)
        pt3d_assign_conv2d_up = o4visual2d.resize(
            pt3d_assign_conv2d, H_out=sflow.H, W_out=sflow.W, mode=sflow.resize_mode
        )
        # 4. calc: dev_nn_x, dev_nn_y : KxHxWx3x3
        dev_nn_x = (pt3d_hom_conv2d_up[0] - sflow_pt3d_hom[0, None]).abs()
        dev_nn_y = (pt3d_hom_conv2d_up[1] - sflow_pt3d_hom[1, None]).abs()
        # 5. calc: log_likelihood, inlier: KxHxWx3x3
        dev_nn_log_likelihood, dev_nn_inlier_soft = self.calc_dev_nn_spatial_consensus(dev_nn_x, dev_nn_y, sflow)
        # 6. choose min if available: KxHxW
        N = pt3d_assign_conv2d_up.size(1)
        dev_nn_inlier_soft, dev_nn_inlier_max_indices = (dev_nn_inlier_soft[None] * pt3d_assign_conv2d_up).max(dim=1)

        dev_nn_max_mask = F.one_hot(dev_nn_inlier_max_indices, num_classes=N).permute(0, 3, 1, 2).bool()
        dev_nn_log_likelihood = (dev_nn_log_likelihood[None,] * dev_nn_max_mask).sum(dim=1)

        dev_nn_x_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=sflow.std_nn_x,
        )
        dev_nn_y_dev_abs_distr = torch.distributions.Normal(
            loc=0.0,
            scale=sflow.std_nn_y,
        )
        dev_nn_x_likelihood_min = dev_nn_x_dev_abs_distr.log_prob(torch.ones(size=(1,), device=dev_nn_x.device) * 10e05)
        dev_nn_y_likelihood_min = dev_nn_y_dev_abs_distr.log_prob(torch.ones(size=(1,), device=dev_nn_y.device) * 10e05)
        dev_nn_log_likelihood[dev_nn_inlier_soft == 0.0] = dev_nn_x_likelihood_min + dev_nn_y_likelihood_min

        return dev_nn_log_likelihood, dev_nn_inlier_soft

    def add_spatial_model(self, sflow, drpcs_prev=None):
        """Specify pt3d and pt3d_assign for inliers and make sure that they are connected."""

        self.pt3d_hom = self.pt3d / (self.pt3d[2:] + 1e-10)
        # dist_z = (self.pt3d[2, None, None, :, :] - self.pt3d[2, :, :, None, None]).abs()[None,]
        dev_nn_x = (self.pt3d_hom[0, None, None, :, :] - self.pt3d_hom[0, :, :, None, None]).abs()[None,]
        dev_nn_y = (self.pt3d_hom[1, None, None, :, :] - self.pt3d_hom[1, :, :, None, None]).abs()[None,]

        dev_nn_log_likelihood, dev_nn_inlier_soft = self.calc_dev_nn_spatial_consensus(dev_nn_x, dev_nn_y, sflow)

        dist = 1.0 - 1.0 * (
            (dev_nn_inlier_soft * self.sflow_inlier_soft[:, :, :, None, None]) > sflow.inlier_hard_thresh
        )

        if drpcs_prev is not None:
            prev_inlier_hard = drpcs_prev.sflow_inlier_hard.sum(dim=0, keepdim=True)
            dist = dist + 1.0 * prev_inlier_hard[:, :, :, None, None] + 1.0 * prev_inlier_hard[:, None, None, :, :]

        # dist = dist - 2.0 * ~sflow.depth_reliable_0[:, :, :, None, None] - 2.0 * ~sflow.depth_reliable_0[:, None, None, :, :]
        # import tensor_operations.visual._2d as o4visual2d
        # o4visual2d.visualize_imgs(sflow.depth_reliable_0[None])
        # o4visual2d.visualize_imgs(dist[0, 0, 0, None, None])

        clusters_per_se3, clusters = clusters_agglomerative(dists=dist, dists2d=True)

        self.K = clusters_per_se3.sum()
        if self.K > 0:
            clusters = clusters * sflow.depth_reliable_0
        self.se3 = self.se3.repeat_interleave(repeats=clusters_per_se3, dim=0)
        self.pt3d_assign = clusters
        # import tensor_operations.visual._2d as o4visual2d
        # o4visual2d.visualize_imgs(clusters[:, None])
        self.model = "joint"

    def fuse_drpcs(self, drpcs_sel):
        if drpcs_sel is not None:
            self.K = self.K + drpcs_sel.K
            self.se3 = torch.cat((self.se3, drpcs_sel.se3), dim=0)
            self.pt3d_assign = torch.cat((self.pt3d_assign, drpcs_sel.pt3d_assign), dim=0)
            # import tensor_operations.visual._2d as o4visual2d
            # o4visual2d.visualize_imgs(self.pt3d_assign[:, None])
            self.sflow_log_likelihood = torch.cat((self.sflow_log_likelihood, drpcs_sel.sflow_log_likelihood), dim=0)
            self.sflow_inlier_soft = torch.cat((self.sflow_inlier_soft, drpcs_sel.sflow_inlier_soft), dim=0)
            self.sflow_inlier_hard = torch.cat((self.sflow_inlier_hard, drpcs_sel.sflow_inlier_hard), dim=0)

    def select_drpcs(self, drpcs_sel_ids):
        """Select subset with ids to reduce number of drpcs.

        Parameters
        ----------
        drpcs_sel_ids torch.Tensor: K_new, ids which should remain
        """
        if drpcs_sel_ids.dim() == 0:
            drpcs_sel_ids = drpcs_sel_ids[None,]

        self.K = len(drpcs_sel_ids)

        if self.model == "joint":
            self.pt3d_assign = self.pt3d_assign[drpcs_sel_ids]

        self.se3 = self.se3[drpcs_sel_ids]

        self.sflow_log_likelihood = self.sflow_log_likelihood[drpcs_sel_ids]
        self.sflow_inlier_soft = self.sflow_inlier_soft[drpcs_sel_ids]
        self.sflow_inlier_hard = self.sflow_inlier_hard[drpcs_sel_ids]

    def update_se3(self, sflow, args, req_inlier=True, req_max_likelihood=True):
        """self.se3 = o4geo_se3_fit_pt3d_oflow.fit_se3_to_pt3d_oflow_and_masks( self.sflow_inlier_hard, sflow.pt3d_0,
        sflow.oflow, sflow.cam_int, orig_H=sflow.cam_H, orig_W=sflow.cam_W, resize_mode=sflow.resize_mode,

        method="cpu-ransac-epnp", weights=self.sflow_inlier_hard )
        """
        self.se3 = o4geo_se3_fit_pt3d_oflow.fit_se3_to_pt3d_oflow_and_masks(
            self.sflow_inlier_hard,
            sflow.pt3d_0,
            sflow.oflow,
            sflow.cam_int,
            orig_H=sflow.cam_H,
            orig_W=sflow.cam_W,
            resize_mode=sflow.resize_mode,
            method="cpu-iterative-continue",
            weights=self.sflow_inlier_hard,
            prev_se3_mats=self.se3,
        )


def clusters_agglomerative(dists, thresh=0.5, dists2d=False):
    """Construct clusters which are connected in agglomerative way.

    Parameters
    ----------
    dists torch.Tensor: *xNxN/*xHxWxHxW
    thresh float
    dists2d bool: indicates if input is in *xNxN or *xHxWxHxW

    Returns
    -------
    clusters torch.Tensor: *xN/*xHxW
    """
    device = dists.device
    if dists2d:
        H, W = dists.size(-2), dists.size(-1)
        dists = dists.reshape(*dists.size()[:-4], H * W, H * W)

    N = dists.size(-1)
    dists = dists.reshape(-1, N, N)
    dists_np = dists.detach().cpu().numpy()
    clusters = []
    clusters_per_batch = torch.zeros(size=(dists.size(0),), device=device, dtype=torch.int)
    for k in range(dists.size(0)):
        pxl_connected = (dists[k] <= thresh).sum(dim=1) > 0.5
        if pxl_connected.sum() > 0.0:
            clustering = AgglomerativeClustering(
                metric="precomputed",
                linkage="single",
                distance_threshold=thresh,
                n_clusters=None,
            ).fit(dists_np[k])

            labels_k = torch.from_numpy(clustering.labels_).to(device)
            labels_k[~pxl_connected] = -1
            labels_unique, labels_k = labels_k.unique(return_inverse=True)
            labels_k_onehot = F.one_hot(labels_k).permute(1, 0)
            if labels_unique[0] == -1:
                labels_k_onehot = labels_k_onehot[1:]
            clusters.append(labels_k_onehot)
            clusters_per_batch[k] = labels_k_onehot.size(0)
        else:
            clusters_per_batch[k] = 0

    if len(clusters) > 0:
        # print("add geo - num clusters", len(clusters))
        clusters = torch.cat(clusters, dim=0)
        if dists2d:
            clusters = clusters.reshape(-1, H, W)

    return clusters_per_batch, clusters


class ConsensusSFlowDRPCs:
    def __init__(self, sflow, drpcs):
        # self.inlier_soft
        # self.inlier_hard

        pass
