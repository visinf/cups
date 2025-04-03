import numpy as np
import torch

import cups.scene_flow_2_se3.clustering.elemental as o4cluster


def bayesian_inference(likelihood, prior=None):
    # prior P(H): B x K x H x W , B x K x N
    # likelihood P(E|H): B x K x H x W, B x K x N
    # P(E, H_k) = P(E|H_k) P(H_k)
    # P(H_k|E) = [P(E|H_k) * P(H_k)] / [sum_j P(E|H_j) P(H_j)]
    # posterior: B x K x H x W, B x K x N
    if prior is None:
        posterior = (likelihood + 1e-8) / ((likelihood + 1e-8).sum(dim=1, keepdim=True))
    else:
        posterior = (likelihood + 1e-8) * prior / (((likelihood + 1e-8) * prior).sum(dim=1, keepdim=True))

    return posterior


def argmax_prob_2_binary(prob):

    K = prob.shape[1]
    ids = torch.argmax(prob, dim=1, keepdim=True)

    # label_min = 0 #label_min=label_min
    label_max = K - 1
    binary = o4cluster.label_2_onehot(ids, negative_handling="ignore", label_max=label_max)

    return binary


def thresh_prob_2_binary(prob, thresh):

    binary = prob > thresh

    return binary


def set_actual_2_prob_gaussian(set, actual, std):
    # set, actual: B x C x H x W, B x C x N
    dev = torch.norm(set - actual, dim=1, keepdim=True)

    return dev_2_prob_gaussian(dev, std)


def models_vars_means(x, models_prob_pxl, pixel_weights=None):
    # x: B x C x H x W
    # masks: B x K x H x W
    # pixel_weights = B x 1 x H x W

    if pixel_weights is not None:
        models_prob_pxl = pixel_weights * models_prob_pxl

    means = (models_prob_pxl[:, :, None] * x[:, None]).flatten(3).sum(dim=3) / (
        models_prob_pxl[:, :, None].flatten(3).sum(dim=3) + 1e-8
    )
    vars = (models_prob_pxl[:, :, None] * (x[:, None] - means[:, :, :, None, None]) ** 2).flatten(3).sum(dim=3) / (
        models_prob_pxl[:, :, None].flatten(3).sum(dim=3) + 1e-8
    )

    # vars, means: B x K x C
    return vars, means


def models_stds_means(x, masks, pixel_weights=None):
    # x: B x C x H x W
    # masks: B x K x H x W
    # pixel_weights = B x 1 x H x W

    vars, means = models_vars_means(x, masks, pixel_weights=pixel_weights)

    stds = torch.sqrt(vars)
    # stds, means: B x K x C
    return stds, means


def dev_2_prob_gaussian(dev, std):
    # std: B x 1, B x 1 x H x W, B x 1 x N
    if not isinstance(std, float):
        while len(dev.shape) > len(std.shape):
            std = std.unsqueeze()

    prob = 1.0 / (std * np.sqrt(2.0 * np.pi)) * torch.exp(-((dev / std) ** 2) / 2.0)
    return prob
