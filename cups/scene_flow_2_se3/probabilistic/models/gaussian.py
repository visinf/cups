import numpy as np
import torch


def fit(data, models_posterior, fix_first_to_0=False):
    # pts: B x C x H x W / B x C x N
    # models_likelihood: B x K x H x W
    # means, stds: B x K x C
    # out: argmax(P(H|E)) B x K x C

    pts = data["pts"]

    means = (models_posterior[:, :, None] * pts[:, None]).flatten(3).sum(dim=3) / (
        models_posterior[:, :, None].flatten(3).sum(dim=3) + 1e-10
    )
    if fix_first_to_0:
        means[:, :1] = 0.0
        means[:, 1:] = data["pts"].max()
    vars = (models_posterior[:, :, None] * (pts[:, None] - means[:, :, :, None, None]) ** 2).flatten(3).sum(dim=3) / (
        models_posterior[:, :, None].flatten(3).sum(dim=3) + 1e-10
    )

    params = {}
    params["mean"] = means
    params["std"] = torch.sqrt(vars)
    # +- 1 std: 68.27% inlier
    # +- 2 std: 95.45% inlier
    # +- 3 std: 99.73% inlier
    return params
    #   int_{d(err)} (p(inlier) p(err) d(err) #/ [p(inlier| err) p(err) + p(err | not inlier)]
    #


def likelihood(data, models_params):
    # pts: B x C x H x W / B x C x N
    # means, stds: B x K x C
    # out: P(E|H) B x K x H x W

    pts = data["pts"]

    # B x K x C x H x W
    dev = pts[:, None] - models_params["mean"][:, :, :, None, None]

    stds = models_params["std"][:, :, :, None, None]

    models_likelihood = likelihood_dev(dev, stds)

    return models_likelihood


def posterior(data, models_params):
    models_likelihood = likelihood(data, models_params)
    models_posterior = models_likelihood / (torch.sum(models_likelihood, dim=1, keepdim=True) + 1e-10)
    return models_posterior


def em(data, models_posterior, rounds=3, fix_first_to_0=False):
    # pts: B x C x H x W / B x C x N
    # models_likelihood: B x K x H x W
    # means, stds: B x K x C
    # out: argmax(P(H|E)) B x K x C

    for r in range(rounds):
        models_params = fit(data, models_posterior, fix_first_to_0=fix_first_to_0)
        models_posterior = posterior(data, models_params)

    return models_params


def likelihood_dev(dev, std):
    # means, vars: B x K x C
    # dev: B x K x C x H x W
    # vars: B x K x C x 1 x 1, B x K x C x H x W
    # 1. / (std * np.sqrt(2. * np.pi)) *
    models_likelihood = torch.exp(-(dev**2 / std**2) / 2.0)
    models_likelihood = torch.prod(models_likelihood, dim=2) + 1e-10

    return models_likelihood


def calc_inlier_prob(dev, std):
    distr = torch.distributions.Normal(loc=0.0, scale=std)
    dev[dev.isnan()] = 1e10
    inlier_prob = 2 * (1.0 - distr.cdf(dev.abs().clamp(0, 1e10)))
    return inlier_prob


def calc_inlier(dev_abs, dev_rel, distr_abs, distr_rel, inlier_hard_threshold=None):
    inlier_prob_rel = torch.log(2 * (1.0 - distr_rel.cdf(dev_rel)) + 1e-10)
    inlier_prob_abs = torch.log(2 * (1.0 - distr_rel.cdf(dev_rel)) + 1e-10)
    inlier_prob = torch.max(inlier_prob_rel, inlier_prob_abs)
    if inlier_hard_threshold is not None:
        inlier_prob = inlier_prob > inlier_hard_threshold  # 0.0455 # equals dev < 2 * std

    return inlier_prob


def calc_log_likelihood(dev_abs, distr_abs):
    log_likelihood = distr_abs.log_prob(dev_abs)
    return log_likelihood


def visualize_estimate_std(dev, norm, valid, std_min, std_max, tag="pixel-shift", correct_trunc_factor=True):
    """Visualize estimation of standard deviation.

    Parameters
    ----------
    dev torch.Tensor: BxCxHxW, float
    norm torch.Tensor: Bx1xHxW, float
    valid torch.Tensor: BxCxHxW / Bx1xHxW, bool
    dev_trusted_perc float: not used
    valid_min_perc float:

    Returns
    -------
    std torch.Tensor: BxC, float
    """

    B, C, H, W = dev.shape
    device = dev.device
    dtype = dev.dtype

    if valid is None:
        valid = torch.ones(size=dev.shape, dtype=torch.bool, device=device)

    valid = valid * (dev.abs() <= std_max)

    bounds_min = 0
    bounds_max = 80
    bins_count = 8

    bounds_count = bins_count + 1
    bounds = torch.linspace(bounds_min, bounds_max, bounds_count)
    lower_bounds = bounds[:-1]
    upper_bounds = bounds[1:]
    mean_bounds = (lower_bounds + upper_bounds) / 2.0

    for b in range(B):
        # valid[b] = o4masks.random_subset_for_max(valid[b], 10000)
        for c in range(C):
            dev_valid = dev[b, c, valid[b, c]]
            norm_valid = norm[b, 0, valid[b, c]]

            stds_bounds = []
            for bin_id in range(bins_count):
                std_bounds = (
                    (dev_valid[(norm_valid > lower_bounds[bin_id]) * (norm_valid < upper_bounds[bin_id])] ** 2)
                    .mean()
                    .sqrt()
                )
                if correct_trunc_factor:
                    # calc_sigma_optimum(delta, sigma_trunc,
                    std_bounds = calc_sigma_optimum(delta=std_max, sigma_trunc=std_bounds)
                stds_bounds.append(std_bounds)
            stds_bounds = torch.stack(stds_bounds)

            import matplotlib.pyplot as plt

            # plt.plot(norm_valid.detach().cpu().numpy(), dev_valid.detach().cpu().numpy(), "*", alpha=0.005)
            # set seaborn style
            # sns.set_style("white")
            # Custom the color, add shade and bandwidth
            # a_plt = sns.kdeplot(x=norm_valid.detach().cpu().numpy(), y=dev_valid.detach().cpu().numpy(), cmap="Reds", shade=True, bw_adjust=0.3)
            plt.plot(norm_valid.detach().cpu().numpy(), dev_valid.detach().cpu().numpy(), "*", alpha=0.01)
            plt.xlim(0, 80)
            plt.ylim(-20, 20)

            plt.xticks(fontsize=14)  # , rotation=90)
            plt.yticks(fontsize=14)

            plt.errorbar(
                mean_bounds.detach().cpu().numpy(),
                torch.zeros_like(mean_bounds).detach().cpu().numpy(),
                yerr=stds_bounds.detach().cpu().numpy(),
                fmt="-ko",
                capsize=5.0,
            )
            if tag == "oflow":
                plt.xlabel("norm - optical flow", fontsize=20)
                if c == 0:
                    plt.ylabel("error - optical flow x-axis", fontsize=20)
                    plt.tight_layout()
                    plt.savefig("results/visuals/optical_flow_dev_x_over_norm.png")
                if c == 1:
                    plt.ylabel("error - optical flow y-axis", fontsize=20)
                    plt.tight_layout()
                    plt.savefig("results/visuals/optical_flow_dev_y_over_norm.png")

            if tag == "disp":
                plt.ylabel("error - disparity", fontsize=20)
                plt.xlabel("norm - disparity", fontsize=20)
                plt.tight_layout()
                plt.savefig("results/visuals/disparity_dev_over_norm.png")
            plt.show()

            # plt.set(xlim=(0, 60))
            # plt.set(ylim=(-2, 2))
            # if tag == "oflow":
            #    a_plt.set(xlabel="norm", ylabel="dev")
            # plt.set_title(tag + " " + str(c))

            # plt.show()
            # erfinv(dev_trusted_perc) = a
            # a = (dev_at_trusted_perc) / (sigma * sqrt(2))
            # sigma = (dev_at_trusted_perc) / (sqrt(2) * a)

            # dev_at_trusted_perc = dev_valid.quantile(dev_trusted_perc)
            # sqrt_2 = torch.Tensor([dev_trusted_perc]).type(dtype).to(device).sqrt()
            # aerf = torch.special.erfinv(dev_trusted_perc)
            # std = dev_at_trusted_perc / (sqrt_2 * aerf)
            std = (dev_valid**2).mean().sqrt()
            std = torch.clamp(std, std_min, std_max)


def calc_sigma_rel_optimum(delta, sigma_trunc, sigma_rel_min=0.1, sigma_rel_max=1.0, resolution=100, thresh=1e-2):
    sigma_rel, sigma_rel_implicit_proposals = calc_sigma_rel_implicit_proposals(
        delta, sigma_trunc, sigma_rel_min=sigma_rel_min, sigma_rel_max=sigma_rel_max, resolution=resolution
    )

    mask_thresh = sigma_rel_implicit_proposals.abs() < thresh

    if mask_thresh.sum() == 0:
        sigma_rel_optimum = torch.Tensor([sigma_rel_min]).to(sigma_trunc.device)
    else:
        sigma_rel_optimum = sigma_rel[mask_thresh].max()

    return sigma_rel_optimum


def calc_sigma_optimum(delta, sigma_trunc, sigma_rel_min=0.1, sigma_rel_max=1.0, resolution=100, thresh=1e-2):
    sigma_rel_optimum = calc_sigma_rel_optimum(
        delta,
        sigma_trunc,
        sigma_rel_min=sigma_rel_min,
        sigma_rel_max=sigma_rel_max,
        resolution=resolution,
        thresh=thresh,
    )

    return sigma_trunc / (sigma_rel_optimum + 1e-10)


def calc_sigma_rel_implicit_proposals(delta, sigma_trunc, sigma_rel_min=0.1, sigma_rel_max=1.0, resolution=100):
    sigma_rel = torch.linspace(sigma_rel_min, sigma_rel_max, resolution).to(sigma_trunc.device)

    sigma = sigma_trunc / (sigma_rel + 1e-10)
    sigma_rel_implicit_proposals = calc_sigma_rel_implicit(delta, sigma_trunc, sigma)

    return sigma_rel, sigma_rel_implicit_proposals


def calc_sigma_rel_implicit(delta, sigma_trunc, sigma):  # ratio_delta_sigma):
    sqrt_2 = torch.Tensor([2]).to(sigma.device).sqrt()
    sqrt_pi = torch.Tensor([np.pi]).to(sigma.device).sqrt()

    alpha = delta / (sigma_trunc * sqrt_2 + 1e-10)
    sigma_rel = sigma_trunc / (sigma + 1e-10)

    sigma_rel_implicit = (
        1
        - (2.0 / sqrt_pi)
        * alpha
        * sigma_rel
        * torch.exp(-(alpha.square()) * (sigma_rel.square()))
        / (torch.erf(alpha * sigma_rel) + 1e-10)
        - sigma_rel.square()
    )

    return sigma_rel_implicit


def calc_ratio_sigma_sigma_part(delta, sigma):  # ratio_delta_sigma):
    sqrt_2 = torch.Tensor([2]).to(sigma.device).sqrt()
    sqrt_pi = torch.Tensor([np.pi]).to(sigma.device).sqrt()
    ratio_delta_sigma_sqrt_2 = delta / (sigma * sqrt_2 + 1e-10)
    ratio_sigma_sigma_part = (
        1.0
        / (
            (
                1
                - (2.0 / sqrt_pi)
                * ratio_delta_sigma_sqrt_2
                * torch.exp(-ratio_delta_sigma_sqrt_2.square())
                * (1.0 / (torch.erf(ratio_delta_sigma_sqrt_2) + 1e-10))
            )
            + 1e-10
        ).sqrt()
    )
    return ratio_sigma_sigma_part


def estimate_std(dev, valid, dev_trusted_perc, valid_min_perc, std_min, std_max, correct_trunc_factor=True):
    """Estimates standard deviation.

    Parameters
    ----------
    dev torch.Tensor: BxCxHxW, float
    valid torch.Tensor: BxCxHxW / Bx1xHxW, bool
    dev_trusted_perc float: not used
    valid_min_perc float:

    Returns
    -------
    std torch.Tensor: BxC, float
    """

    B, C, H, W = dev.shape
    device = dev.device
    dtype = dev.dtype
    dev_trusted_perc = torch.Tensor([dev_trusted_perc]).type(dtype).to(device)

    if valid is None:
        valid = torch.ones(size=dev.shape, dtype=torch.bool, device=device)

    valid = valid * (dev.abs() <= std_max)
    if valid.sum() / valid.numel() < valid_min_perc:
        return torch.Tensor([std_min]).to(device).repeat(B, C)
    else:
        stds = []
        for b in range(B):
            for c in range(C):
                dev_valid = dev[b, c, valid[b, c]]
                # erfinv(dev_trusted_perc) = a
                # a = (dev_at_trusted_perc) / (sigma * sqrt(2))
                # sigma = (dev_at_trusted_perc) / (sqrt(2) * a)

                # dev_at_trusted_perc = dev_valid.quantile(dev_trusted_perc)
                # sqrt_2 = torch.Tensor([dev_trusted_perc]).type(dtype).to(device).sqrt()
                # aerf = torch.special.erfinv(dev_trusted_perc)
                # std = dev_at_trusted_perc / (sqrt_2 * aerf)
                std = (dev_valid**2).mean().sqrt()

                if correct_trunc_factor:
                    std = calc_sigma_optimum(delta=std_max, sigma_trunc=std)

                std = torch.clamp(std, std_min, std_max)
                stds.append(std)
        stds = torch.stack(stds).reshape(B, C)
        return stds
