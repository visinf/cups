import time

import cv2
import numpy as np
import tensor_operations.clustering.elemental as o4cluster
import tensor_operations.geometric.pinhole as o4geo_pinhole
import tensor_operations.geometric.se3.transform as o4geo_se3_transf
import tensor_operations.vision.visualization as o4visual
import torch

# modify module according to inputs
from third_party.rigidmask.models.VCNplus import VCN, WarpModule, flow_reg
from third_party.rigidmask.utils import dydepth as ddlib


class RigidMaskWrapper:

    def __init__(self, args):

        self.args = args

        if "kitti" in self.args.data_dataset_tags:
            # kitti 1.2 sintel 1.5
            self.maxh = int(384)
            self.maxw = int(1280)
        elif "sintel" in self.args.data_dataset_tags:
            self.maxh = 436
            self.maxw = 1024
        elif "tum_rgbd" in self.args.data_dataset_tags:
            self.maxh = 480
            self.maxw = 640
        elif "bonn_rgbd" in self.args.data_dataset_tags:
            self.maxh = 480
            self.maxw = 640
        elif "sceneflow" in self.args.data_dataset_tags:
            self.maxh = 540
            self.maxw = 960
        elif "flyingthings3d" in self.args.data_dataset_tags:
            self.maxh = 540
            self.maxw = 960
        else:
            print("net rigidmask: unknown dataset", self.args.data_dataset_tags)
            print("set max_H", 540)
            self.maxh = 540
            print("set max_W", 960)
            self.maxh = 960

        self.fac = 1
        self.maxdisp = 256

        self.max_h = int(self.maxh // 64 * 64)
        self.max_w = int(self.maxw // 64 * 64)
        if self.max_h < self.maxh:
            self.max_h += 64
        if self.max_w < self.maxw:
            self.max_w += 64
        self.maxh = self.max_h
        self.maxw = self.max_w

        if args.sflow2se3_rigidmask_train_dataset == "kitti":
            print("net rigidmask: load kitti weights..")
            self.model_path = "third_party/rigidmask/weights/kitti.pth"
        elif args.sflow2se3_rigidmask_train_dataset == "sceneflow":
            print("net rigidmask: load sceneflow weights..")
            self.model_path = "third_party/rigidmask/weights/sf.pth"
        else:
            print("net rigidmask: unknown train dataset", args.sflow2se3_rigidmask_train_dataset)
            print("net rigidmask: load sceneflow weights..")
            self.model_path = "third_party/rigidmask/weights/sf.pth"

        self.mean_L = torch.from_numpy(np.asarray([[0.33, 0.33, 0.33]]).mean(0))[None,].cuda()
        self.mean_R = torch.from_numpy(np.asarray([[0.33, 0.33, 0.33]]).mean(0))[None,].cuda()
        # torch.Tensor([0.485, 0.456, 0.406]).cuda()[np.newaxis, :, np.newaxis, np.newaxis]) / \
        #        torch.Tensor([0.229, 0.224, 0.225]).cuda()[np.newaxis, :, np.newaxis, np.newaxis]
        # construct model, VCN-expansion
        # TODO: exp_unc=('kitti' in args.sflow2se3_rigidmask_train_dataset) (problem with shapes)
        model = VCN(
            [1, self.maxw, self.maxh],
            md=[int(4 * (self.maxdisp / 256)), 4, 4, 4, 4],
            fac=self.fac,
            exp_unc=not "kitti" in self.model_path,
        )
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.cuda()

        pretrained_dict = torch.load(self.model_path, map_location="cpu")
        # np.asarray(mean_L).mean(0)
        self.mean_L = torch.from_numpy(np.asarray(pretrained_dict["mean_L"]).astype(np.float32).mean(0))[None,].cuda()
        self.mean_R = torch.from_numpy(np.asarray(pretrained_dict["mean_R"]).astype(np.float32).mean(0))[None,].cuda()
        pretrained_dict["state_dict"] = {k: v for k, v in pretrained_dict["state_dict"].items()}
        model.load_state_dict(pretrained_dict["state_dict"], strict=False)
        model.eval()
        self.model = model

        for i in range(len(self.model.module.reg_modules)):
            self.model.module.reg_modules[i] = flow_reg(
                [1, self.max_w // (2 ** (6 - i)), self.max_h // (2 ** (6 - i))],
                ent=getattr(self.model.module, "flow_reg%d" % 2 ** (6 - i)).ent,
                maxdisp=getattr(self.model.module, "flow_reg%d" % 2 ** (6 - i)).md,
                fac=getattr(self.model.module, "flow_reg%d" % 2 ** (6 - i)).fac,
            ).cuda()
        for i in range(len(self.model.module.warp_modules)):
            self.model.module.warp_modules[i] = WarpModule(
                [1, self.max_w // (2 ** (6 - i)), self.max_h // (2 ** (6 - i))]
            ).cuda()

        print("loaded rigidmask")

    @torch.no_grad()
    def forward(self, data_pred_sflow):
        # imgL, imgR: BxCxHxW
        # projection_mats: Bx2x4
        device = data_pred_sflow["rgb_l_01"].device
        dtype = data_pred_sflow["rgb_l_01"].dtype

        imgL = data_pred_sflow["rgb_l_01"][:, :3]
        B, C, H, W = imgL.shape
        imgR = data_pred_sflow["rgb_l_01"][:, 3:]
        disp0 = data_pred_sflow[
            "disp_0"
        ].abs()  # / (data_pred_sflow['baseline'][0])# * data_pred_sflow['projection_matrix'][0, 0, 0])
        # note: 1/ disp != depth
        # -> multiply disp by 1/(fx * baseline)

        # oflow = data_pred_sflow['oflow']
        # oflow_valid = data_pred_sflow['oflow_valid']
        projection_mats = data_pred_sflow["projection_matrix"]
        baseline = data_pred_sflow["baseline"]

        # repeat for gray images
        if C == 1:
            imgL = imgL.repeat(1, 3, 1, 1)
            imgR = imgR.repeat(1, 3, 1, 1)

        imgL = o4visual.resize(imgL, H_out=self.max_h, W_out=self.max_w)
        imgR = o4visual.resize(imgR, H_out=self.max_h, W_out=self.max_w)
        # disp0 = o4visual.resize(disp0, H_out=self.max_h, W_out=self.max_w, mode='nearest_v2', vals_rescale=True)
        # oflow_valid = o4visual.resize(oflow_valid, H_out=self.max_h, W_out=self.max_w, mode='nearest_v2')
        # oflow = o4visual.resize(oflow, H_out=self.max_h, W_out=self.max_w, vals_rescale=True)

        # imgL = imgL.flip(dims=[1])
        # imgR = imgR.flip(dims=[1])
        imgL_noaug = imgL.clone()
        imgL = imgL.flip(dims=[1])
        imgR = imgR.flip(dims=[1])
        imgL = imgL - self.mean_L[:, :, None, None]
        imgR = imgR - self.mean_R[:, :, None, None]

        # get intrinsics
        fl = projection_mats[0, 0, 0]
        fl_next = fl
        cx = projection_mats[0, 0, 2]
        cy = projection_mats[0, 1, 2]
        bl = baseline[0]
        intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl], [cx], [cy], [bl], [1], [0], [0], [1], [0], [0]]]
        K0 = torch.eye(3, device=baseline.device)
        K0[:2, :3] = projection_mats[0, :2, :3]
        K1 = K0

        intr_list.append(torch.Tensor([W / self.max_w]).cuda())  # delta fx
        intr_list.append(torch.Tensor([H / self.max_h]).cuda())  # delta fy
        intr_list.append(torch.Tensor([fl_next]).cuda())

        disp_input = disp0  # .permute(0, 2, 3, 1)

        imgLR = torch.cat([imgL, imgR], 0)
        # None
        # TODO: consider using midas for depth of rgb
        # torch.cuda.synchronize()
        # disc_aux = [None, None, None, intr_list, imgL_noaug, None]
        disc_aux = [None, None, None, intr_list, imgL_noaug, None, None, None, 0]
        output = self.model(imgLR, disc_aux, disp_input=disp_input)
        # torch.cuda.synchronize()

        # logmid / logexp = dchange2 (from optical expansion)
        polarmask_label_np = {}
        flow, occ, logmid, logexp, fgmask, heatmap, polarmask_label_np, disp = output
        bbox = polarmask_label_np["bbox"]
        polarmask_label_np = polarmask_label_np["mask"]
        polarcontour = polarmask_label_np[: polarmask_label_np.shape[0] // 2]
        polarmask_label_np = polarmask_label_np[polarmask_label_np.shape[0] // 2 :]

        # upsampling
        occ = cv2.resize(occ.data.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        logexp = cv2.resize(logexp.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        logmid = cv2.resize(logmid.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        fgmask = cv2.resize(fgmask.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.resize(heatmap.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        polarcontour = cv2.resize(polarcontour, (W, H), interpolation=cv2.INTER_NEAREST)
        polarmask_label_np = cv2.resize(polarmask_label_np, (W, H), interpolation=cv2.INTER_NEAREST).astype(int)
        polarmask_label_np[np.logical_and(fgmask > 0, polarmask_label_np == 0)] = -1

        # if args.disp_path == '':
        #    disp = cv2.resize(disp.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        # else:
        disp = np.asarray(disp_input.cpu())[0, 0]
        # disp = disp.cpu()
        flow = torch.squeeze(flow).data.cpu().numpy()
        flow = np.concatenate(
            [cv2.resize(flow[0], (W, H))[:, :, np.newaxis], cv2.resize(flow[1], (W, H))[:, :, np.newaxis]], -1
        )
        flow[:, :, 0] *= W / self.max_w
        flow[:, :, 1] *= H / self.max_h
        flow = np.concatenate((flow, np.ones([flow.shape[0], flow.shape[1], 1])), -1)
        bbox[:, 0] *= W / self.max_w
        bbox[:, 2] *= W / self.max_w
        bbox[:, 1] *= H / self.max_h
        bbox[:, 3] *= H / self.max_h
        """# draw instance center and motion in 2D ins_center_vis = np.zeros(flow.shape[:2]) for k in
        range(bbox.shape[0]): from third_party.rigidmask.utils.detlib import draw_umich_gaussian

        draw_umich_gaussian(ins_center_vis, bbox[k, :4].reshape(2, 2).mean(0), 15) ins_center_vis = 256 *
        np.stack([ins_center_vis, np.zeros(ins_center_vis.shape), np.zeros(ins_center_vis.shape)], -1)
        """

        ## depth and scene flow estimation
        # save initial disp and flow
        bgmask_np = polarmask_label_np == 0

        # polarmask_onehot = o4cluster.label_2_onehot(torch.from_numpy(polarmask_label_np)[None, None], negative_handling="ignore")[0]
        # polarmask_label = o4cluster.onehot_2_label(polarmask_onehot[None, ])[0, 0]
        # labels_unique, labels_counts  = torch.unique(polarmask_label, return_counts=True)
        # print('labels', labels_unique)
        # print('count', labels_counts)
        # print(labels_unique[labels_counts > 5])
        # if K > 1:
        #    print('stop: more than ego mask')

        K0 = K0.cpu().numpy()
        K1 = K1.cpu().numpy()
        bl = bl.cpu().numpy()
        scene_type, T01_c, R01, RTs = ddlib.rb_fitting(
            bgmask_np,
            polarmask_label_np,
            disp,
            flow,
            occ,
            K0,
            K1,
            bl,
            parallax_th=4,
            mono=False,
            sintel="sintel" in self.args.data_dataset_tags,
        )
        # scene_type 'F'
        disp, flow, disp1 = ddlib.mod_flow(
            bgmask_np,
            polarmask_label_np,
            disp,
            disp / np.exp(logmid),
            flow,
            occ,
            bl,
            K0,
            K1,
            scene_type,
            T01_c,
            R01,
            RTs,
            fgmask,
            mono=False,
            sintel="sintel" in self.args.data_dataset_tags,
        )
        # logmid = np.clip(np.log(disp / disp1), -1, 1)
        # disp_f0_1 = torch.from_numpy(disp1).type(dtype)[None, None].to(device)
        # pt3d_f0_1, _ = o4geo_pinhole.disp_2_pt3d(disp_f0_1, data_pred_sflow['projection_matrix'], data_pred_sflow['reprojection_matrix'], baseline=data_pred_sflow['baseline'])
        oflow = torch.from_numpy(flow)[None, :, :, :2].to(device).type(dtype).permute(0, 3, 1, 2)

        # polarmask: H x W
        polarmask_not_assigned = torch.from_numpy(polarmask_label_np == -1)[None, None].to(device)
        # if scene_type == 'H':
        #    polarmask_not_assigned[:, 0] += polarmask_onehot[None, 0].to(device)

        print("label -1", polarmask_not_assigned.sum())
        polarmask_label_np[polarmask_label_np == -1] = 0
        # not assigned pixel: assign to background
        # data_pred_se3['objs_labels'][data_pred_se3['objs_labels'] == -1] = 0
        polarmask_onehot = o4cluster.label_2_onehot(
            torch.from_numpy(polarmask_label_np)[None, None], negative_handling="ignore"
        )[0]
        # polarmask_onehot: K x H x W
        polarmask_label = o4cluster.onehot_2_label(polarmask_onehot[None,])[0, 0]
        # polarmask_label: H x W
        # -1: not background, not foreground
        #  0: background
        # >0: foreground

        se3_ego = torch.eye(4, dtype=dtype, device=device)
        se3_ego[:3, :3] = torch.from_numpy(R01).type(dtype).to(device)
        se3_ego[:3, 3] = torch.from_numpy(np.array(T01_c)).type(dtype=dtype).to(device)
        se3_ego = torch.linalg.inv(se3_ego)

        se3_objs = []
        se3_objs.append(se3_ego)
        for RT in RTs:
            if RT is None or np.isinf(np.linalg.norm(RT[1])):
                # or np.linalg.norm(RT[1]) == 0.
                # if RT is not None and np.linalg.norm(RT[1]) == 0.:
                #   se3_objs.append(torch.eye(4, dtype=dtype, device=device))
                # else:
                se3_objs.append(se3_objs[0])
                polarmask_not_assigned[:, 0] += polarmask_onehot[None, len(se3_objs) - 1].to(device)
            else:
                se3_obj = torch.eye(4, dtype=dtype, device=device)
                se3_obj[:3, :3] = torch.from_numpy(RT[0]).type(dtype).to(device)
                se3_obj[:3, 3] = torch.from_numpy(np.array(RT[1])).type(dtype=dtype).to(device)
                se3_obj = torch.linalg.inv(se3_obj)
                se3_objs.append(se3_obj)
        K = len(se3_objs)
        se3_objs = torch.stack(se3_objs, dim=0)
        se3_objs = se3_objs[None,]
        print("camera trans: ")
        print(T01_c)

        data_pred_se3 = {}
        data_pred_se3["pt3d_0"] = data_pred_sflow["pt3d_0"]  # pt3d_0
        data_pred_se3["objs_labels"] = polarmask_label[None,].to(
            device
        )  # torch.zeros_like(data_pred_sflow['pt3d_0'][0, :1])
        data_pred_se3["objs_masks"] = polarmask_onehot.to(device)  # torch.zeros_like(data_pred_sflow['pt3d_0'][0, :1])
        data_pred_se3["objs_params"] = {}
        data_pred_se3["objs_params"]["se3"] = {}
        data_pred_se3["objs_params"]["se3"]["se3"] = se3_objs

        pts1_ftf = o4geo_se3_transf.pts3d_transform(
            data_pred_se3["pt3d_0"].repeat(K, 1, 1, 1), data_pred_se3["objs_params"]["se3"]["se3"].reshape(K, 4, 4)
        )

        pts1_ftf = (
            o4cluster.label_2_onehot(data_pred_se3["objs_labels"], negative_handling="ignore", label_max=K - 1)[
                :, :, None
            ].repeat(1, 1, 3, 1, 1)
            * pts1_ftf
        ).sum(dim=1)

        data_pred_se3["pt3d_f0_1"] = pts1_ftf

        data_pred_se3["oflow"] = o4geo_pinhole.pt3d_2_oflow(
            data_pred_se3["pt3d_f0_1"], data_pred_sflow["projection_matrix"]
        )
        data_pred_se3["oflow"][polarmask_not_assigned.repeat(1, 2, 1, 1)] = oflow[
            polarmask_not_assigned.repeat(1, 2, 1, 1)
        ]

        return data_pred_se3
