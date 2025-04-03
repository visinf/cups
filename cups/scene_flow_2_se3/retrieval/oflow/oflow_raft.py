import argparse
import sys
from argparse import Namespace

import torch
from third_party.RAFT.core.raft import RAFT
from third_party.RAFT.core.utils.utils import InputPadder


class RAFT_Wrapper:
    def __init__(self, args):
        """sys.argv = sys.argv[:1] parser = argparse.ArgumentParser() parser.add_argument('--model', help="restore
        checkpoint") parser.add_argument('--path', help="dataset for evaluation") parser.add_argument('--small',
        action='store_true', help='use small model') parser.add_argument('--mixed_precision', action='store_true',
        help='use mixed precision') parser.add_argument('--alternate_corr', action='store_true', help='use efficient
        correlation implementation') args = parser.parse_args()"""
        self.model_args = Namespace(small=False, mixed_precision=True, alternate_corr=False)

        if args.sflow2se3_raft_train_dataset == "small":
            self.model_args.small = True

        self.model_args.data_dataset_tags = args.data_dataset_tags

        self.model = RAFT(self.model_args)
        self.model = torch.nn.DataParallel(self.model)

        # model.load_state_dict(torch.load('models/raft/raft-things.pth'))
        if self.model.module.args.small:
            print("raft: loading small")
            self.model.load_state_dict(torch.load("third_party/RAFT/models/raft-small.pth"))
        else:
            # if 'kitti' in self.model.module.args.data_dataset_tags:
            if args.sflow2se3_raft_train_dataset == "kitti":
                print("raft: loading kitti")
                self.model.load_state_dict(torch.load("third_party/RAFT/models/raft-kitti.pth"))
                # self.model.load_state_dict(torch.load('models/raft/raft-things.pth'))
            elif args.sflow2se3_raft_train_dataset == "flyingthings3d":
                print("raft: loading flyingthings3d")
                self.model.load_state_dict(torch.load("third_party/RAFT/models/raft-things.pth"))
            elif args.sflow2se3_raft_train_dataset == "sintel":
                print("raft: loading sintel")
                self.model.load_state_dict(torch.load("third_party/RAFT/models/raft-sintel.pth"))
            else:
                print(
                    "error unknown sflow2se3_raft_train_dataset ",
                    args.sflow2se3_raft_train_dataset,
                )
                print("raft: loading flyingthings3d")
                self.model.load_state_dict(torch.load("third_party/RAFT/models/raft-things.pth"))

        # or things or sintel

        self.device = args.setup_dataloader_device
        self.model.to(self.device)  # .cuda()
        self.model.eval()
        self.model = self.model.module
        self.model.eval()

    def forward(self, imgpair, disp=False, fwd=True):
        torch.cuda.empty_cache()
        image1 = imgpair[:, :3] * 255
        image2 = imgpair[:, 3:] * 255
        if "kitti" in self.model.args.data_dataset_tags:
            print("raft: pad kitti")
            padder = InputPadder(image1.shape, mode="kitti")
            # padder = InputPadder(image1.shape, mode='things')
        elif "flyingthings3d" in self.model.args.data_dataset_tags:
            print("raft: pad things")
            padder = InputPadder(image1.shape, mode="things")
        else:
            print("raft: pad sintel")
            padder = InputPadder(image1.shape)

        # for kitti : mode = 'kitti'
        image1, image2 = padder.pad(image1.to(self.device), image2.to(self.device))  # image1[None]

        with torch.no_grad():
            # iters: kitti/chairs: 24, sintel: 32 (in sintel use prev flow as initialization for next flow)
            if "sintel" in self.model.args.data_dataset_tags:
                flow_low, flow_pr = self.model(
                    image1, image2, iters=32, test_mode=True
                )  # model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            else:
                flow_low, flow_pr = self.model(image1, image2, iters=24, test_mode=True)

        flow = padder.unpad(flow_pr)  # [None] #permute(1, 2, 0).cpu().numpy()

        if disp:
            if fwd:
                flow = -flow[:, :1]
            else:
                flow = flow[:, :1]

        return flow

    def forward_backward(self, imgpair_fwd, disp=False):
        B = imgpair_fwd.size(0)

        imgpair_bwd = torch.cat((imgpair_fwd[:, 3:], imgpair_fwd[:, :3]), dim=1)
        # imgpair = torch.cat((imgpair_fwd, imgpair_bwd), dim=0)

        flow_fwd = self.forward(imgpair_fwd, disp=disp, fwd=True)

        flow_bwd = self.forward(imgpair_bwd, disp=disp, fwd=False)

        return flow_fwd, flow_bwd
