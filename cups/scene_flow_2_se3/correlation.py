import torch


def normalize_features(x1, x2):
    # over channel and spatial dimensions
    # and moments average over batch size and features
    # x1 : B x C x H x W
    """X1 = torch.rand((12, 3, 9, 16))

    import tensorflow as tf
    x1 = tf.convert_to_tensor(x1.detach().cpu().numpy())
    #x2 = tf.convert_to_tensor(x2.detach().cpu().numpy())
    mean1, var1 = tf.nn.moments(x1, axes=[-1, -2, -3])
    mean1 = torch.from_numpy(mean1.numpy()).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    var1 = torch.from_numpy(var1.numpy()).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    """

    var1, mean1 = torch.var_mean(x1, dim=(1, 2, 3), keepdim=True, unbiased=False)
    var2, mean2 = torch.var_mean(x2, dim=(1, 2, 3), keepdim=True, unbiased=False)

    mean = torch.mean(torch.cat((mean1, mean2)))
    var = torch.mean(torch.cat((var1, var2)))

    # 1e-16 for robustness of division by std, this can be found in uflow implementation
    std = torch.sqrt(var + 1e-16)
    x1_normalized = (x1 - mean) / std
    x2_normalized = (x2 - mean) / std

    return x1_normalized, x2_normalized


def compute_cost_volume(x1, x2, max_displacement):
    # x1 : B x C x H x W
    # x2 : B x C x H x W

    dtype = x1.dtype
    device = x1.device

    B, C, H, W = x1.size()

    padding_module = torch.nn.ConstantPad2d(max_displacement, 0.0)
    x2_pad = padding_module(x2)

    num_shifts = 2 * max_displacement + 1

    cost_volume_channel_dim = num_shifts**2
    cost_volume = torch.zeros((B, cost_volume_channel_dim, H, W), dtype=dtype, device=device)

    for i in range(num_shifts):
        for j in range(num_shifts):
            cost_volume_single_layer = torch.mean(x1 * x2_pad[:, :, i : i + H, j : j + W], dim=1)
            cost_volume[:, i * num_shifts + j, :, :] = cost_volume_single_layer

    return cost_volume
