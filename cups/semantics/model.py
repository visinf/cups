import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "external", "depthg", "src"))
from external.depthg.src.train_segmentation import (
    LitUnsupervisedSegmenter as LitUnsupervisedSegmenterDepthg,
)

sys.path.remove(os.path.join(os.getcwd(), "external", "depthg", "src"))


class DepthG:
    def __init__(
        self,
        device,
        checkpoint_root,
        call_type: None = None,
        img_shape: Tuple = (640, 1280),
        stride: Tuple = (160, 160),
        crop: Tuple = (320, 320),
    ):
        self.call_type = call_type
        self.model = LitUnsupervisedSegmenterDepthg.load_from_checkpoint(checkpoint_root)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)

        # sliding window parameters
        self.img_shape = img_shape
        self.stride = stride
        self.h_stride = stride[0]
        self.w_stride = stride[1]
        self.crop = crop
        self.bottom_pad = img_shape[0] % stride[0]
        self.right_pad = img_shape[1] % stride[1]
        if self.bottom_pad != 0:
            self.bottom_pad = self.h_stride - self.bottom_pad
        if self.right_pad != 0:
            self.right_pad = self.w_stride - self.right_pad

    def slide_segment(self, img):
        unfolded = F.unfold(img, self.crop, stride=self.stride, padding=(self.bottom_pad, self.right_pad))
        unfolded = rearrange(unfolded, "B (C H W) N -> (B N) C H W", H=self.crop[0], W=self.crop[1])

        _, code = self.model.net(unfolded)
        code = F.interpolate(code, (self.crop[0], self.crop[1]), mode="bilinear", align_corners=False)
        crop_seg_logits = self.model.cluster_probe(code, 2, log_probs=True)
        c = crop_seg_logits.size(1)
        crop_seg_logits = rearrange(crop_seg_logits, "(B N) C H W -> B (C H W) N", B=img.size(0))

        preds = F.fold(
            crop_seg_logits,
            (img.size(-2), img.size(-1)),
            self.crop,
            stride=self.stride,
            padding=(self.bottom_pad, self.right_pad),
        )
        count_mat = F.fold(
            torch.ones(
                (
                    crop_seg_logits.size(0),
                    crop_seg_logits.size(1) // c,
                    crop_seg_logits.size(2),
                ),
                device=img.device,
            ),
            (img.size(-2), img.size(-1)),
            self.crop,
            stride=self.stride,
            padding=(self.bottom_pad, self.right_pad),
        )

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def depth_guided_sliding_window(self, img, depth_weight):
        # sliding window forward pass
        out_slidingw = self.slide_segment(img)

        # single image and flipped image forward pass
        img_small = F.interpolate(
            img, (img.shape[-2] // 2, img.shape[-1] // 2), mode="bilinear", align_corners=False
        ).float()
        code = self(img_small)
        code2 = self(img_small.flip(dims=[3]))
        code = (code + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False)
        out_singleimg = self.model.cluster_probe(code, 2, log_probs=True)

        weight = depth_weight.expand_as(out_slidingw)
        out = out_singleimg * weight + out_slidingw * (1 - weight)

        return out

    def single_image_prediction(self, img):
        # # normalize image
        # img = normalize(img)
        # resize image to have height of 320
        with torch.no_grad():
            img_small = F.interpolate(
                img,
                size=(320, int(((320 / img.shape[-2]) * img.shape[-1] // 8) * 8)),
                mode="bilinear",
                align_corners=False,
            )
            # get feature embeddings
            feats = self.model(img_small)
            feats = F.interpolate(feats, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
            # get semantic prediction
            prediction = self.model.cluster_probe(feats, 2, log_probs=True)
        return prediction.argmax(dim=1).long(), feats, prediction

    def __call__(self, x):
        if self.call_type == "single_image_prediction":
            return self.single_image_prediction(x)
        else:
            return self.model.net(x)[-1]
