import cups.scene_flow_2_se3.probabilistic.elemental as o4prob
import cups.scene_flow_2_se3.probabilistic.models.gaussian as o4prob_gauss
import cups.scene_flow_2_se3.probabilistic.models.se3 as o4prob_se3
import cups.scene_flow_2_se3.vision.visualization as o4visual


def em(models_params, data, args):
    models_likelihood = likelihood(data, models_params, args)

    models_posterior = o4prob.bayesian_inference(models_likelihood)
    print("posterior")
    o4visual.visualize_imgs(models_posterior[0, :, None])
    print("likelihood")
    o4visual.visualize_imgs(models_likelihood[0, :, None])

    for k in range(3):

        models_params = fit(data, models_posterior, models_likelihood, args)

        models_likelihood = likelihood(data, models_params, args)
        models_posterior = o4prob.bayesian_inference(models_likelihood)

        print("posterior")
        o4visual.visualize_imgs(models_posterior[0, :, None])
        print("likelihood")
        o4visual.visualize_imgs(models_likelihood[0, :, None])

    return models_params


def fit_and_likelihood(data, models_posterior, models_likelihood, args):
    B, _, H, W = models_posterior.shape
    models_params = fit(data, models_posterior, models_likelihood, args)

    models_likelihood = likelihood(data, models_params, args)

    # fits_precision =  (models_posterior * models_likelihood).flatten(2).sum(dim=2) / (models_likelihood.flatten(2).sum(dim=2) + 1e-8)
    # fits_bin = fits_precision > 0.2

    # models_likelihood = models_likelihood[fits_bin].reshape(B, -1, H, W)
    # models_params['se3']['se3'] = models_params['se3']['se3'][fits_bin].reshape(B, -1, 4, 4)
    # models_params['se3']['se3_centroid1'] = models_params['se3']['se3_centroid1'][fits_bin].reshape(B, -1, 4, 4)

    # models_params['eucl']['mean'] = models_params['eucl']['mean'][fits_bin].reshape(B, -1, 3)
    # models_params['eucl']['std'] = models_params['eucl']['std'][fits_bin].reshape(B, -1, 3 )

    return models_params, models_likelihood


def likelihood(data, models_params, args):
    # out: models_prob: P(E|H)
    # args.sflow2se3_model_se3_pt3d_std

    pts1 = data["pts1"]
    B, K, H, W = pts1.shape
    dtype = pts1.dtype
    device = pts1.device

    models_se3_likelihood = o4prob_se3.likelihood(data=data, models_params=models_params["se3"], args=args["se3"])

    if models_se3_likelihood.isnan().sum() > 0:
        print("error: nans in se3 likelihood")

    models_eucl_likelihood = o4prob_gauss.likelihood(data=data, models_params=models_params["eucl"])
    if models_eucl_likelihood.isnan().sum() > 0:
        print("error: nans in eucl likelihood")

    models_likelihood = models_se3_likelihood  # * models_eucl_likelihood

    return models_likelihood


def fit(data, models_posterior, models_likelihood, args):
    # out: models_se3, models_means, models_stds, P(H|E)
    B, K, H, W = models_posterior.shape
    dtype = models_posterior.dtype
    device = models_posterior.device

    # prob_bin = o4prob.argmax_prob_2_binary(models_posterior)
    # prob = (models_posterior * models_likelihood)
    prob = models_likelihood
    # prob = models_posterior
    prob_bin = o4prob.thresh_prob_2_binary(prob, thresh=0.5)

    # B x K x 3
    models_eucl_params = o4prob_gauss.fit(data, prob)

    models_se3_params = o4prob_se3.fit(data, prob_bin, args["se3"])

    models_params = {}

    models_params["eucl"] = models_eucl_params
    models_params["se3"] = models_se3_params

    models_params["eucl"]["std"] = models_params["eucl"]["std"] + 999999.0
    return models_params
