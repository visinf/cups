import sys

sys.path.append(".")

import argparse

# from third_party.RAFT3D.raft3d.raft3d import RAFT3D
from argparse import Namespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.transforms as t3d
import tensor_operations.clustering.dbscan as o4cluster_dbscan
import tensor_operations.clustering.elemental as o4cluster
import tensor_operations.geometric.euclidean as o4geo_eucl
import tensor_operations.geometric.se3.elemental as o4geo_se3
import tensor_operations.geometric.se3.transform as o4geo_se3_transf
import tensor_operations.probabilistic.elemental as o4prob
import tensor_operations.probabilistic.models.euclidean_nn as o4prob_euclidean_nn
import tensor_operations.probabilistic.models.se3 as o4prob_se3
import tensor_operations.string as o4string
import tensor_operations.visual._2d as o4visual2d
import tensor_operations.visual._3d as o4visual3d

# requires pip install git+https://github.com/princeton-vl/lietorch.git
import third_party.RAFT3D.raft3d.projective_ops as pops
import torch
import torch.nn.functional as F
from third_party.RAFT3D.data_readers import frame_utils
from third_party.RAFT3D.scripts.utils import normalize_image, show_image


class RAFT3D_Wrapper:

    def __init__(self, args):
        # super(RAFT3D_Wrapper, self).__init__()

        self.model_path = "third_party/RAFT3D/models/raft3d_laplacian.pth"  #''
        self.model_path = "third_party/RAFT3D/models/raft3d_kitti.pth"
        # self.model_path = 'third_party/RAFT3D/models/raft3d.pth'

        self.network_path = "third_party.RAFT3D.raft3d.raft3d"
        self.network_path = "third_party.RAFT3D.raft3d.raft3d_bilaplacian"

        #'sflow2se3-raft3d-train-dataset: kitti # flyingthings3d kitti
        # sflow2se3-raft3d-architecture-bilaplacian: True
        # sflow2se3-rigidmask-train-dataset: kitti # flyingthings3d kitti'

        if args.sflow2se3_raft3d_train_dataset == "kitti":
            if not args.sflow2se3_raft3d_architecture_bilaplacian:
                print("net raft3d: there is only bilaplacian model for kitti train dataset available")
            self.network_path = "third_party.RAFT3D.raft3d.raft3d_bilaplacian"
            self.model_path = "third_party/RAFT3D/models/raft3d_kitti.pth"
            print("net raft3d: load kitti weights...")
        elif args.sflow2se3_raft3d_train_dataset == "flyingthings3d":
            if args.sflow2se3_raft3d_architecture_bilaplacian:
                self.network_path = "third_party.RAFT3D.raft3d.raft3d_bilaplacian"
                print("net raft3d: use bilaplacian model.")
                self.model_path = "third_party/RAFT3D/models/raft3d_laplacian.pth"
                print("net raft3d: load flyingthings3d weights...")
            else:
                self.network_path = "third_party.RAFT3D.raft3d.raft3d"
                print("net raft3d: do not use bilaplacian model.")
                self.model_path = "third_party/RAFT3D/models/raft3d.pth"
                print("net raft3d: load flyingthings3d weights...")
        else:
            print("net raft3d: unknown train dataset", args.sflow2se3_raft3d_train_dataset)
            self.network_path = "third_party.RAFT3D.raft3d.raft3d_bilaplacian"
            print("net raft3d: use bilaplacian model.")
            self.model_path = "third_party/RAFT3D/models/raft3d_laplacian.pth"
            print("net raft3d: load flyingthings3d weights...")

        print("model_path", self.model_path)
        print("network_path", self.network_path)
        # prog='.myprogram',
        self.model_args = Namespace(network=self.network_path, model=self.model_path)

        print("model", self.model_args.model)
        print("network", self.model_args.network)
        import importlib

        RAFT3D = importlib.import_module(self.model_args.network).RAFT3D
        self.depth_scale = 0.2
        # self.depth_scale = 0.1 # maybe for kitt use this

        self.model = torch.nn.DataParallel(RAFT3D(self.model_args), device_ids=[0])
        # self.model = RAFT3D(self.model_args)
        self.model.load_state_dict(torch.load(self.model_args.model), strict=False)

        self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def forward(self, data_pred_sflow, args):
        image1 = data_pred_sflow["rgb_l_01"][:, :3] * 255
        image2 = data_pred_sflow["rgb_l_01"][:, 3:] * 255
        depth1 = torch.clamp(data_pred_sflow["depth_0"], min=1e-10)
        depth2 = torch.clamp(data_pred_sflow["depth_1"], min=1e-10)

        # print("depth0", depth1)
        # print("depth1", depth2)

        # disp1 = data_pred_sflow['disp_0']
        # disp2 = data_pred_sflow['disp_1']
        # depth1 = self.depth_scale * (self.intrinsics[0,0] / disp1)
        # depth2 = self.depth_scale * (self.intrinsics[0,0] / disp2)

        projection_matrix = data_pred_sflow["projection_matrix"]
        self.fx = projection_matrix[:, 0, 0]
        self.fy = projection_matrix[:, 1, 1]
        self.cx = projection_matrix[:, 0, 2]
        self.cy = projection_matrix[:, 1, 2]
        self.intrinsics = torch.stack([self.fx, self.fy, self.cx, self.cy], dim=1)
        # B x 4

        _, _, in_H, in_W = image1.shape
        pad_W = (((in_W // 8) + 1) * 8 - in_W) % 8
        pad_H = (((in_H // 8) + 1) * 8 - in_H) % 8

        depth1 = depth1[:, 0]
        depth2 = depth2[:, 0]
        image1, image2, depth1, depth2 = self.prepare_images_and_depths(image1, image2, depth1, depth2, pad_H, pad_W)

        Ts, mask = self.model(image1, image2, depth1, depth2, self.intrinsics, iters=16)

        # coords, _ = pops.projective_transform(Ts, depth1, self.intrinsics)
        # coords = coords.permute(0, 3, 1, 2)
        # coords = coords[:, :, :in_H, :in_W] #/  self.depth_scale
        # coords[:, 2:] = coords[:, 2:] / self.depth_scale
        # compute 2d and 3d from from SE3 field (Ts)
        flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, self.intrinsics)
        flow3d = flow3d / self.depth_scale
        flow3d = flow3d.permute(0, 3, 1, 2)
        flow3d = flow3d[:, :, :in_H, :in_W]

        if args.sflow2se3_raft3d_cluster:

            # B x H x W x 4 x 4

            se3s = Ts.matrix()
            se3s[:, :, :, :3, 3] /= self.depth_scale
            se3s_logs = Ts.log().permute(0, 3, 1, 2)
            se3s_logs[:, :3] /= self.depth_scale
            (
                B,
                H,
                W,
                _,
                _,
            ) = se3s.shape
            H_down = int(H / 20.0)
            W_down = int(W / 20.0)
            se3s_down = o4visual2d.resize(
                se3s.reshape(B, H, W, 16).permute(0, 3, 1, 2), H_out=H_down, W_out=W_down, mode="nearest_v2"
            )
            se3s_down = se3s_down.permute(0, 2, 3, 1).reshape(B, -1, 4, 4)
            se3s_log_down = o4visual2d.resize(se3s_logs, H_out=H_down, W_out=W_down, mode="nearest_v2")
            pt3d_0_down = o4visual2d.resize(data_pred_sflow["pt3d_0"], H_out=H_down, W_out=W_down, mode="nearest_v2")
            pt3d_0_down_dist_apair = o4geo_eucl.pts3d_2_dists_eucl(pts3d_1_down=pt3d_0_down)
            # B x H x W x 4 x 4

            # o4visual3d.visualize_se3s(se3s_down.reshape(-1, 4, 4))
            # returns indices of lower triangualr
            # se3s_down = se3s_down.reshape(H_down, W_down, 4, 4)[torch.tril_indices(H_down, W_down).unbind()]
            se3s_down_dist_apair, se3s_down_angle_apair = o4geo_se3.se3_mat_2_dist_angle_all_pairs(
                se3s_down, angle_unit="deg"
            )

            # (pt3d_0_down_dist_apair[0] < 1.0)
            dists = (
                1.0 - (pt3d_0_down_dist_apair < 10.0) * (se3s_down_dist_apair < 0.3) * (se3s_down_angle_apair < 3) * 1.0
            )
            labels = o4cluster_dbscan.fit(dists=dists, dist_max=0.5, core_min_samples=2)
            models_se3_inlier = o4cluster.label_2_onehot(labels, ignore_negative=True).reshape(B, -1, H_down, W_down)
            # o4visual2d.visualize_img(o4visual2d.mask2rgb(models_se3_inlier[0]))
            # print('hello')

            models_params = {}
            models_params["se3"] = {}
            data = {}
            data["pts"] = pt3d_0_down
            models_params["pts"] = o4prob_euclidean_nn.fit(data, models_se3_inlier)
            models_params["se3"]["se3_log"] = (models_se3_inlier[:, :, None] * se3s_log_down[:, None, :]).flatten(
                3
            ).sum(dim=3) / models_se3_inlier[:, :, None].flatten(3).sum(dim=3)
            # lietorch.SE3.exp(models_params["se3"]["se3_log"])
            import lietorch

            # models_params["se3"]["se3"] = t3d.se3_exp_map(models_params["se3"]["se3_log"].reshape(-1, 6)).reshape(B, -1, 4, 4)\
            models_params["se3"]["se3"] = lietorch.SE3.exp(models_params["se3"]["se3_log"]).matrix()
            data = {}
            # data["pts"] = data_pred_sflow['pt3d_0']
            # data["pts1"] = data_pred_sflow['pt3d_0']
            # data["pts2"] = data_pred_sflow['pt3d_f0_1']
            # data["oflows"] = data_pred_sflow['oflow']
            # data["pairs_valid"] = data_pred_sflow['pt3d_pair_valid']
            # data["proj_mats"] = data_pred_sflow["projection_matrix"]
            # data["baseline"] = data_pred_sflow["baseline"]
            # data["orig_H"] = H
            # data["orig_W"] = W

            # models_likelihood = o4prob_se3_eucl.likelihood(data, models_params, prob_se3_eucl_args)
            data["pts"] = data_pred_sflow["pt3d_0"]
            data["proj_mats"] = data_pred_sflow["projection_matrix"]
            data["orig_H"] = H
            data["orig_W"] = W
            data["baseline"] = data_pred_sflow["baseline"]
            data["pts1"] = data_pred_sflow["pt3d_0"]
            data["pts2"] = data_pred_sflow["pt3d_f0_1"]
            data["oflows"] = data_pred_sflow["oflow"]
            data["pairs_valid"] = data_pred_sflow["pt3d_pair_valid"]

            prob_se3_eucl_args = {}
            prob_se3_eucl_args["se3"] = {}
            for key, val in vars(args).items():
                if key.startswith("sflow2se3_model_se3_"):
                    prob_se3_eucl_args["se3"][o4string.remove_prefix(key, "sflow2se3_model_se3_")] = val
            prob_se3_eucl_args["se3"]["resize_mode"] = args.sflow2se3_downscale_mode
            prob_se3_eucl_args["se3"]["likelihood_use_oflow"] = True
            prob_se3_eucl_args["se3"]["likelihood_oflow_invalid_pairs"] = -1.0
            models_se3_inlier, models_se3_log_likelihood = o4prob_se3.likelihood(
                data, models_params["se3"], prob_se3_eucl_args["se3"]
            )
            (
                models_euclidean_nn_inlier,
                models_euclidean_nn_log_likelihood,
            ) = o4prob_euclidean_nn.likelihood(
                data, models_params["pts"], std=args.sflow2se3_model_euclidean_nn_dist_std
            )
            if args.sflow2se3_se3_inlier_req_pt3d_0_valid:
                models_se3_inlier *= data_pred_sflow["pt3d_valid_0"]
            # o4visual2d.visualize_imgs(models_euclidean_nn_inlier[0, :, None])
            models_inlier = models_se3_inlier * models_euclidean_nn_inlier
            models_log_prior = torch.log(
                (models_inlier.flatten(2).sum(dim=2) / models_inlier.flatten(1).sum(dim=1, keepdim=True))
            )  # ood[0] #

            # o4visual2d.visualize_imgs(models_se3_inlier[0, :, None])
            if args.sflow2se3_labels_source_argmax == "prior*inlier":
                objects_masks = o4prob.argmax_prob_2_binary(
                    torch.log(models_inlier + 1e-10) + models_log_prior[:, :, None, None]
                )[0]

            elif args.sflow2se3_labels_source_argmax == "prior*likelihood":
                models_log_likelihood = models_euclidean_nn_log_likelihood + models_se3_log_likelihood
                objects_masks = o4prob.argmax_prob_2_binary(models_log_likelihood + models_log_prior[:, :, None, None])[
                    0
                ]

            elif args.sflow2se3_labels_source_argmax == "likelihood":
                models_log_likelihood = models_se3_log_likelihood + models_euclidean_nn_log_likelihood
                objects_masks = o4prob.argmax_prob_2_binary(models_log_likelihood)[0]

            else:
                print("error: unknown labels source ", args.sflow2se3_labels_source_argmax)

            K1 = len(objects_masks)
            labels = torch.argmax(objects_masks * 1.0, dim=0, keepdim=True)
            # vals, labels = torch.max(objects_masks * 1.0, dim=0, keepdim=True)
            B, K, _, _ = models_params["se3"]["se3"].shape
            if K1 != K:
                print("STOP")

            data_pred_se3 = {}
            data_pred_se3["pt3d_0"] = data_pred_sflow["pt3d_0"]  # pt3d_0
            data_pred_se3["objs_labels"] = (
                labels  # polarmask_label[None, ].to(device) # torch.zeros_like(data_pred_sflow['pt3d_0'][0, :1])
            )
            data_pred_se3["objs_masks"] = (
                objects_masks  # polarmask_onehot.to(device) # torch.zeros_like(data_pred_sflow['pt3d_0'][0, :1])
            )
            data_pred_se3["objs_params"] = {}
            data_pred_se3["objs_params"]["se3"] = {}
            data_pred_se3["objs_params"]["se3"]["se3"] = models_params["se3"]["se3"]

            pts1_ftf = o4geo_se3_transf.pts3d_transform(
                data_pred_se3["pt3d_0"].repeat(K, 1, 1, 1), data_pred_se3["objs_params"]["se3"]["se3"].reshape(K, 4, 4)
            )

            pts1_ftf = (
                o4cluster.label_2_onehot(data_pred_se3["objs_labels"], label_min=0, label_max=K - 1)[:, :, None].repeat(
                    1, 1, 3, 1, 1
                )
                * pts1_ftf
            ).sum(dim=1)

            data_pred_se3["pt3d_f0_1"] = pts1_ftf

        else:

            # extract rotational and translational components of Ts
            # tau, phi = Ts.log().split([3, 3], dim=-1)
            # tau = tau[0].cpu().numpy()
            # phi = phi[0].cpu().numpy()

            # undo depth scaling

            # display(img1, tau, phi)

            data_pred_se3 = {}
            data_pred_se3["pt3d_0"] = data_pred_sflow["pt3d_0"]
            data_pred_se3["pt3d_f0_1"] = data_pred_sflow["pt3d_0"] + flow3d  # coords
            data_pred_se3["objs_labels"] = torch.zeros_like(data_pred_sflow["pt3d_0"][0, :1])
            data_pred_se3["objs_masks"] = torch.zeros_like(data_pred_sflow["pt3d_0"][0, :1])
            # data_pred_se3['objs_params'] = None
            # data_pred_se3['objs_params']['se3']['se3'][:, 0, :, :]

        return data_pred_se3

    def prepare_images_and_depths(self, image1, image2, depth1, depth2, pad_H, pad_W):
        """Padding, normalization, and scaling."""
        # (padding_left,padding_right, padding_top,padding_bottom)
        # valid resolution: 960 x 544

        image1 = F.pad(image1, [0, pad_W, 0, pad_H], mode="replicate")
        image2 = F.pad(image2, [0, pad_W, 0, pad_H], mode="replicate")
        depth1 = F.pad(depth1[:, None], [0, pad_W, 0, pad_H], mode="replicate")[:, 0]
        depth2 = F.pad(depth2[:, None], [0, pad_W, 0, pad_H], mode="replicate")[:, 0]

        depth1 = (self.depth_scale * depth1).float()
        depth2 = (self.depth_scale * depth2).float()
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)

        return image1, image2, depth1, depth2


def normalize_image(image):
    image = image[:, [2, 1, 0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image / 255.0).sub_(mean[:, None, None]).div_(std[:, None, None])


def display(img, tau, phi):
    """Display se3 fields."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()
