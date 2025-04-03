import os

import numpy as np
import tensor_operations.rearrange as o4rearr
import tensor_operations.vision.resize as o4vis_res
import torch
from third_party.LEAStereo.config_utils.predict_args import obtain_predict_args
from third_party.LEAStereo.retrain.LEAStereo import LEAStereo
from third_party.LEAStereo.utils.colorize import get_color_map
from third_party.LEAStereo.utils.multadds_count import (
    comp_multadds,
    count_parameters_in_MB,
)


class LEASTEREOWrapperCPU(torch.nn.Module):
    def __init__(self, module):
        super(LEASTEREOWrapperCPU, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x, y):
        return self.module(x, y)


class LEASTStereoWrapper(torch.nn.Module):
    def __init__(self, args):
        super(LEASTStereoWrapper, self).__init__()

        # args = Namespace()

        #### KITTI
        # required
        if "kitti" in args.data_dataset_tags:
            self.downsample = 1.0
            args.maxdisp = 192
            args.crop_height = 384  # KITTI original: 375x1242
            args.crop_width = 1248  # KITTI original: 375x1242
        elif "flyingthings3d" in args.data_dataset_tags:
            self.downsample = 1.0
            args.maxdisp = 192
            args.crop_height = 576  # 544 # sceneflow original: 540x960
            args.crop_width = 960  # sceneflow original: 540x960
        elif "sintel" in args.data_dataset_tags:
            self.downsample = 0.5
            args.maxdisp = 192
            args.crop_height = 576  # 576 #448 # sintel original: 436x1024
            args.crop_width = 1056  # 1056 #1248 #1024 # sintel original: 436x1024
            # inf_width: 480
            # inf_height: 192
        else:
            print("error: unknown dataset for leastereo")
            self.downsample = 1.0
            args.maxdisp = 192
            args.crop_height = 576  # sceneflow original: 540x960
            args.crop_width = 960  # sceneflow original: 540x960

        args.maxdisp = 192
        args.crop_width = int(np.ceil(args.data_res_width / 96.0) * 96)
        args.crop_height = int(np.ceil(args.data_res_height / 192.0) * 192)

        args.inf_width = int(np.ceil(args.data_res_width / 96.0) * self.downsample) * 96
        args.inf_height = int(np.ceil(args.data_res_height / 192.0) * self.downsample) * 192

        args.fea_step = 3
        args.fea_num_layers = 6
        args.fea_block_multiplier = 4
        args.fea_filter_multiplier = 8

        args.mat_step = 3
        args.mat_num_layers = 12
        args.mat_block_multiplier = 4
        args.mat_filter_multiplier = 8

        args.net_arch_fea = "third_party/LEAStereo/run/sceneflow/best/architecture/feature_network_path.npy"
        args.cell_arch_fea = "third_party/LEAStereo/run/sceneflow/best/architecture/feature_genotype.npy"
        args.net_arch_mat = "third_party/LEAStereo/run/sceneflow/best/architecture/matching_network_path.npy"
        args.cell_arch_mat = "third_party/LEAStereo/run/sceneflow/best/architecture/matching_genotype.npy"

        if args.sflow2se3_leaststereo_train_dataset == "kitti":
            print("leaststereo: loading kitti")
            args.resume = "third_party/LEAStereo/run/Kitti15/best/best.pth"
        elif args.sflow2se3_leaststereo_train_dataset == "sceneflow":
            print("leaststereo: loading sceneflow")
            # args.resume = "third_party/LEAStereo/run/Kitti15/best/checkpoint/best.pth"
            args.resume = "third_party/LEAStereo/run/sceneflow/best/checkpoint/best.pth"
        else:
            print(
                "error: unknown sflow2se3_leaststereo_train_dataset",
                args.sflow2se3_leaststereo_train_dataset,
            )
            args.resume = "third_party/LEAStereo/run/sceneflow/best/checkpoint/best.pth"
            print("using ", args.resume)

        # probably not required
        if args.setup_dataloader_device != "cpu":
            args.cuda = True
        else:
            args.cuda = False
        # args.kitti2015 = 1
        # args.kitti2012 = 0
        # args.middlebury = 0
        # args.sceneflow = 0
        # args.data_path = './dataset/kitti2015/testing/'
        # args.test_list = './dataloaders/lists/kitti2015_test.list'
        # args.save_path = './predict/kitti2015/images/'

        self.args = args

        torch.backends.cudnn.benchmark = True

        cuda = args.cuda
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        # print('===> Building LEAStereo model')
        import copy

        model_args = copy.deepcopy(self.args)
        model_args.crop_width = model_args.inf_width
        model_args.crop_height = model_args.inf_height
        self.model = LEAStereo(args)

        # print('Total Params = %.2fMB' % count_parameters_in_MB(self.model))
        # print('Feature Net Params = %.2fMB' % count_parameters_in_MB(self.model.feature))
        # print('Matching Net Params = %.2fMB' % count_parameters_in_MB(self.model.matching))

        # mult_adds = comp_multadds(self.model, input_size=(3, args.crop_height, args.crop_width))  # (3,192, 192))
        # print("compute_average_flops_cost = %.2fMB" % mult_adds)

        # self.model =
        if cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
            # self.model = self.model.cuda()
        else:
            self.model = LEASTEREOWrapperCPU(self.model).cpu()

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=torch.device(args.setup_dataloader_device))
                self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        self.model.to(args.setup_dataloader_device)
        self.model.eval()

    def forward_backward(self, imgpairs_fwd, disp=True):
        B_fwd, _, _, _ = imgpairs_fwd.shape
        imgpairs_bwd = o4rearr.imgpairs_swap_order(imgpairs_fwd)

        # imgpairs = torch.cat((imgpairs_fwd, imgpairs_bwd), dim=0)
        # disps = self.forward(imgpairs)
        # disps_fwd = disps[:B_fwd]
        # disps_bwd = disps[B_fwd:]

        disps_fwd = self.forward(imgpairs_fwd)
        disps_bwd = torch.flip(self.forward(torch.flip(imgpairs_bwd, dims=[3])), dims=[3])

        return disps_fwd, disps_bwd

    @torch.no_grad()
    def forward(self, imgpairs, disp=True):

        B, C, orig_H, orig_W = imgpairs.shape

        imgpairs = self.imgpairs_normalize(imgpairs)
        imgpairs = self.imgpairs_adapt_size(imgpairs, int(self.args.crop_height), int(self.args.crop_width))

        if self.args.cuda:
            imgpairs = imgpairs.cuda()

        img1 = imgpairs[:, :3]
        img2 = imgpairs[:, 3:]

        if self.args.inf_height != self.args.crop_height or self.args.inf_width != self.args.crop_width:
            img1 = o4vis_res.resize(img1, H_out=self.args.inf_height, W_out=self.args.inf_width, mode="bilinear")
            img2 = o4vis_res.resize(img2, H_out=self.args.inf_height, W_out=self.args.inf_width, mode="bilinear")

        pred = self.model(img1, img2)[:, None, :, :]

        if self.args.inf_height != self.args.crop_height or self.args.inf_width != self.args.crop_width:
            pred = o4vis_res.resize(
                pred, H_out=self.args.crop_height, W_out=self.args.crop_width, vals_rescale=True, mode="bilinear"
            )

        pred = self.pred_inverse_adapt_size(pred, orig_H, orig_W)

        return pred

    def imgpairs_adapt_size(self, imgpairs, target_H, target_W):
        # crop or pad
        B, C, H, W = imgpairs.shape
        device = imgpairs.device
        dtype = imgpairs.dtype

        if H <= target_H and W <= target_W:
            # pad zeros
            imgpairs_adapted = torch.zeros(size=(B, C, target_H, target_W), dtype=dtype, device=device)
            imgpairs_adapted[:, :, target_H - H :, target_W - W :] = imgpairs
        else:
            # crop
            start_x = int((W - target_W) / 2)
            start_y = int((H - target_H) / 2)

            imgpairs_adapted = imgpairs[:, :, start_y : start_y + target_H, start_x : start_x + target_W]

        return imgpairs_adapted

    def pred_inverse_adapt_size(self, pred, orig_H, orig_W):
        B, C, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype

        if orig_H <= H and orig_W <= W:
            # crop
            pred_inv_adapted = pred[:, :, H - orig_H :, W - orig_W :]
        else:
            # pad
            pred_inv_adapted = torch.zeros(size=(B, C, orig_H, orig_W), dtype=dtype, device=device)
            pred_inv_adapted[:, :, orig_H - H :, orig_W - W :] = pred

        return pred_inv_adapted

    def imgpairs_normalize(self, imgpairs):
        # in: B x C x H x W or B x C x N

        std, mean = torch.std_mean(imgpairs.flatten(2), dim=2)

        imgpairs = (imgpairs - mean[:, :, None, None]) / std[:, :, None, None]

        return imgpairs
