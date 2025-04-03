import torch


def mask_select_broadcasted(x_in, mask):
    """Select partial x depending on mask.

    Parameters
    ----------
    x_in torch.Tensor:  D1xD2x...xDR
    mask torch.Tensor: D1xD2x...xDR / each Dr can be exchanged with 1

    Results
    -------
    x_out torch.Tensor:
    """

    x_in_shape = torch.Tensor(list(x_in.shape))
    mask_shape = torch.Tensor(list(mask.shape))

    if len(x_in_shape) != len(mask_shape):
        print("error: x and shape do not contain same number of dimensions", x_in_shape, mask_shape)
        return x_in

    if (~((x_in_shape == mask_shape) + (mask_shape == 1))).sum() > 0:
        print("error: shapes are not broadcastable", x_in_shape, mask_shape)
        return x_in
    mask_rep = mask.repeat(list((x_in_shape / mask_shape).int()))
    x_out = x_in[mask_rep]
    return x_out


def batches_swap_order(x_in):
    B = x_in.size(0)
    x_out = torch.cat(
        (
            x_in[B // 2 :],
            x_in[: B // 2],
        ),
        dim=0,
    )
    return x_out


def imgpairs_swap_order(imgpairs_fwd):
    imgpairs_bwd = torch.cat((imgpairs_fwd[:, 3:, :, :], imgpairs_fwd[:, :3, :, :]), dim=1)

    return imgpairs_bwd


def neighbors_to_channels(x, patch_size, new_dim=False):
    # input: BxCxHxW
    # output: BxP*P*CxHxW
    # first channel upper-left
    # for top-to-bottom
    #   for left-to-right
    # note: second channel = first row, second element
    # last channel lower-right

    B, C, H, W = x.shape
    dtype_in = x.dtype
    dtype = x.dtype
    device = x.device

    if dtype_in == torch.bool:
        x = x * 1.0
        dtype = torch.float

    kernels = torch.eye(patch_size**2, dtype=dtype, device=device).view(patch_size**2, 1, patch_size, patch_size)

    # kernels: P*Px1xPxP : out_channels x in_channels x H x W
    kernels = kernels.repeat(x.size(1), 1, 1, 1)
    # kernels: P*Px1xPxP : out_channels x in_channels x H x W

    x = torch.nn.functional.conv2d(input=x, weight=kernels, padding=int((patch_size - 1) / 2), groups=x.size(1))

    if new_dim:
        x = x.reshape(B, C, patch_size**2, H, W)

    if dtype_in == torch.bool:
        x = x == 1.0

    return x


def neighbors_to_channels_v2(x, patch_size):
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    num_channels_out = patch_size**2 * C

    max_displacement = int((patch_size - 1) / 2)

    x_volume = torch.zeros(size=(B, num_channels_out, H, W), device=device, dtype=dtype)

    padding_module = torch.nn.ConstantPad2d(max_displacement, 0.0)
    x_pad = padding_module(x)

    for i in range(patch_size):
        for j in range(patch_size):
            channel_index = (i * patch_size + j) * C
            x_volume[:, channel_index : channel_index + C, :, :] = x_pad[:, :, i : i + H, j : j + W]

    return x_volume


def neighbors_to_channels_v3(x, patch_size):
    # input: BxCxHxW
    # output: BxP*P*CxHxW

    B, C, H, W = x.shape
    dtype = x.dtype
    device = x.device

    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=torch.int32, device=device),
            torch.arange(0.0, W, dtype=torch.int32, device=device),
        ]
    )
    grid_yx = torch.stack((grid_y, grid_x), dim=0)
    grid_yx = grid_yx.unsqueeze(1)

    patch_grid_y, patch_grid_x = torch.meshgrid(
        [
            torch.arange(0.0, patch_size, dtype=torch.long, device=device),
            torch.arange(0.0, patch_size, dtype=torch.long, device=device),
        ]
    )
    patch_grid_yx = torch.stack((patch_grid_y, patch_grid_x), dim=0) - ((patch_size - 1) // 2)
    patch_grid_yx = patch_grid_yx.flatten(1).unsqueeze(2).unsqueeze(3)

    grid_yx = grid_yx + patch_grid_yx
    indices = (grid_yx[0] * W + grid_yx[1]).flatten(1).unsqueeze(0)
    x = x.flatten(2)

    y = torch.zeros(size=(B, patch_size**2, H * W), device=device, dtype=dtype)

    # mask = (indices >= 0) * (indices < H*W)

    y.scatter_(dim=2, src=x, index=indices[:, :1])

    return x
