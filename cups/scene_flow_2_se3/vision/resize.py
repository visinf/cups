import torch


def resize_nearest_v2(x, H_out, W_out, align_corners=False):
    """Resizes the input tensor by using the nearest neighbor.

    Parameters
    ----------
    x torch.Tensor: BxCxHxW
    H_out int: determines the height for the resulting resolution
    W_out int: determines the width for the resulting resolution
    align_corners bool: if set, the outer values are preserved

    Returns
    -------
    x_out torch.Tensor: BxCxH_out, W_out
    """

    B, C, H_in, W_in = x.shape
    dtype = x.dtype
    device = x.device

    if not align_corners:
        step_x = W_in / W_out
        step_y = H_in / H_out

        # -step_y/2.0
        grid_y, grid_x = torch.meshgrid(
            [
                torch.arange(
                    step_y / 2.0 - 0.5,
                    step_y / 2.0 - 0.5 + (H_out - 1) * step_y + 1e-10,
                    step_y,
                    dtype=dtype,
                    device=device,
                ),
                torch.arange(
                    step_x / 2.0 - 0.5,
                    step_x / 2.0 - 0.5 + (W_out - 1) * step_x + 1e-10,
                    step_x,
                    dtype=dtype,
                    device=device,
                ),
            ]
        )
    else:
        step_x = (W_in - 1) / (W_out - 1)
        step_y = (H_in - 1) / (H_out - 1)

        # -step_y/2.0
        grid_y, grid_x = torch.meshgrid(
            [
                torch.arange(0.0, (H_out - 1) * step_y + 1e-10, step_y, dtype=dtype, device=device),
                torch.arange(0.0, (W_out - 1) * step_x + 1e-10, step_x, dtype=dtype, device=device),
            ]
        )

    grid_x = (grid_x - (W_in - 1) / 2.0) / (W_in - 1) * 2.0
    grid_y = (grid_y - (H_in - 1) / 2.0) / (H_in - 1) * 2.0

    grid_xy = torch.stack((grid_x, grid_y), dim=-1)

    return torch.nn.functional.grid_sample(
        x, grid_xy[None,].repeat(B, 1, 1, 1), padding_mode="border", mode="nearest", align_corners=True
    )


def resize(x, H_out=None, W_out=None, scale_factor=None, mode="bilinear", vals_rescale=False, align_corners=False):
    """Resizes the input tensor depending on the mode. If target resolution is not specified with (H_out, W_out), then
    the scale_factor is used to calculate it.

    Parameters
    ----------
    x torch.Tensor: Tensor which is resized.
    H_out int: Determines the height of the target resolution.
    W_out int: Determines the width of the target resolution.
    scale_factor float: Determines the target resolution in case (H_out, W_out) are not specified.
    mode="bilinear" str: Determines the weighted sampling process. Options are 'bilinear', 'nearest', and 'nearest_v2'.
    vals_rescale bool: Determines if the values are rescaled according to the scale factor between output and input resolution.
    align_corners bool: Determines if the outer values are preserved.

    Returns
    -------
    x_out torch.Tensor: Resized tensor.
    """
    # mode 'nearest' or 'bilinear'
    # in: BxCxHxW
    # out: BxCxHxW
    dtype_in = x.dtype
    if dtype_in == torch.bool:
        x = x * 1.0

    x_dim_in = x.dim()
    if x_dim_in == 3:
        x = x[None,]

    B, C, H_in, W_in = x.shape

    if H_out != None and W_out != None:
        if mode != "nearest":
            if mode == "nearest_v2":
                x_out = resize_nearest_v2(x, H_out=H_out, W_out=W_out, align_corners=align_corners)
            else:
                x_out = torch.nn.functional.interpolate(
                    x,
                    size=(H_out, W_out),
                    mode=mode,
                    align_corners=align_corners,
                )
        else:
            x_out = torch.nn.functional.interpolate(x, size=(H_out, W_out), mode=mode)

    elif scale_factor != None:
        H_out = int(H_in * scale_factor)
        W_out = int(W_in * scale_factor)
        if mode != "nearest":
            if mode == "nearest_v2":
                x_out = resize_nearest_v2(x, H_out=H_out, W_out=W_out, align_corners=align_corners)
            else:
                x_out = torch.nn.functional.interpolate(
                    x,
                    size=(H_out, W_out),
                    mode=mode,
                    align_corners=align_corners,
                )
        else:
            x_out = torch.nn.functional.interpolate(x, size=(H_out, W_out), mode=mode, align_corners=align_corners)

            _, _, H_out, W_out = x_out.shape

    else:
        print("warning: could not resize. pls specify H_out/W_out or scale")
        x_out = x

    if vals_rescale:
        # print('W', W_out / W_in, 'H', H_out / H_in)
        if C > 0:
            x_out[:, 0:1] = x_out[:, 0:1] * (W_out / W_in)
        if C > 1:
            x_out[:, 1:2] = x_out[:, 1:2] * (H_out / H_in)

    if x_dim_in == 3:
        x_out = x_out[0]

    if dtype_in == torch.bool:
        x_out = x_out == 1.0
    return x_out
