#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as pycrf_utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from cups.utils import crop_object_proposal, normalize, reverse_crop_object_proposal


class UnNormalize(object):
    def __init__(self, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    # print('check hyperparameters!')
    image = np.array(VF.to_pil_image(UnNormalize()(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = pycrf_utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def dense_crf_object_proposals(image_tensor: torch.FloatTensor, output_probabilities: torch.FloatTensor):
    # print('check hyperparameters!')
    output_probabilities, bounding_box, original_shape = crop_object_proposal(output_probabilities)
    image_tensor = image_tensor[:, bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]]
    image = (image_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    if (output_probabilities.shape[0] != H) or (output_probabilities.shape[1] != W):
        output_probabilities = F.interpolate(
            output_probabilities[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
    output_probabilities = torch.stack((1 - output_probabilities, output_probabilities), dim=0)

    c = output_probabilities.shape[0]
    h = output_probabilities.shape[1]
    w = output_probabilities.shape[2]

    U = pycrf_utils.unary_from_softmax(output_probabilities.cpu().numpy())
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))

    Q = torch.from_numpy(Q).to(image_tensor.device).argmax(dim=0)
    Q = reverse_crop_object_proposal(Q, bounding_box, original_shape)  # type: ignore
    return Q


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


# CRF applied to predictions
def batched_crf_predictions(pool, img_tensor, pred_tensor, void_id=255):
    img_tensor = normalize(img_tensor)
    # get irgnore mask
    unknown_mask = pred_tensor == void_id
    # set irgnore to 0 zero and get one hot encoding
    pred_tensor[unknown_mask] = 0
    pred_tensor = torch.nn.functional.one_hot(pred_tensor.long()).float()
    # assign constant class distribution to unknown unknown pixels
    pred_tensor[unknown_mask, :] = float(1 / pred_tensor.argmax())
    # apply log softmax
    prob_tensor = torch.nn.functional.log_softmax(pred_tensor.permute(0, 3, 1, 2) * 2, dim=1)
    # map to multiprocessing crf
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    # stitch output back together
    outputs = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)
    return outputs.argmax(1).long().to(pred_tensor.device)


def _apply_dense_crf_object_proposals(tup):
    return dense_crf_object_proposals(tup[0], tup[1])


def batched_dense_crf_object_proposals(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_dense_crf_object_proposals, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.stack(outputs, dim=0)


MAX_ITER_MASKCUT = 10
POS_W_MASKCUT = 7
POS_XY_STD_MASKCUT = 3
Bi_W_MASKCUT = 10
Bi_XY_STD_MASKCUT = 50
Bi_RGB_STD_MASKCUT = 5


def densecrf_maskcut(image, mask):
    image = np.array(VF.to_pil_image(UnNormalize()(image)))[:, :, ::-1]
    h, w = mask.shape
    mask = mask.reshape(1, h, w)
    fg = mask.astype(float)
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = pycrf_utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD_MASKCUT, compat=POS_W_MASKCUT)
    d.addPairwiseBilateral(sxy=Bi_XY_STD_MASKCUT, srgb=Bi_RGB_STD_MASKCUT, rgbim=image, compat=Bi_W_MASKCUT)

    Q = d.inference(MAX_ITER_MASKCUT)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
    return MAP


def _apply_dense_crf_maskcut(tup):
    return densecrf_maskcut(tup[0], tup[1])


def batched_dense_crf_maskcut(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_dense_crf_maskcut, [(img_tensor[0], p) for p in prob_tensor])
    return torch.stack(outputs, dim=0)
