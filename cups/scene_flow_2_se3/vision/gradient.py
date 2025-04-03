import torch


def calc_batch_gradients(batch, shift=1, pad_zeros=False):
    # batch: torch.tensor: BxCxHxW
    # shift = 1 + margin
    batch_grad_x = batch[:, :, :, shift:] - batch[:, :, :, :-shift]
    # B x 2 x H x W-shift

    batch_grad_y = batch[:, :, shift:, :] - batch[:, :, :-shift, :]
    # B x 2 x H-shift x W

    if pad_zeros:
        pad1 = shift // 2
        pad2 = shift - pad1
        batch_grad_x = torch.nn.functional.pad(batch_grad_x, pad=[pad1, pad2, 0, 0], mode="constant")
        batch_grad_y = torch.nn.functional.pad(batch_grad_y, pad=[0, 0, pad1, pad2], mode="constant")

    return batch_grad_x, batch_grad_y


def calc_batch_gradients_norm(batch, shift=1, pad_zeros=False):
    batch_grad_x, batch_grad_y = calc_batch_gradients(batch, shift=shift, pad_zeros=pad_zeros)
    batch_grad_x = torch.norm(batch_grad_x, dim=1, keepdim=True)
    batch_grad_y = torch.norm(batch_grad_y, dim=1, keepdim=True)

    return batch_grad_x, batch_grad_y


def calc_img_gradients(img, margin=1):
    margin = 1
    _, H, W = img.shape
    batch = torch.nn.functional.pad(img.unsqueeze(0), pad=[margin, margin, margin, margin], mode="replicate")
    batch_grad_x, batch_grad_y = calc_batch_gradients(batch=batch, shift=margin + 1)
    batch_grad_x = batch_grad_x[:, :, margin : H + margin, :]
    batch_grad_y = batch_grad_y[:, :, :, margin : W + margin]

    img_grad_x = batch_grad_x[0]
    img_grad_y = batch_grad_y[0]

    return img_grad_x, img_grad_y
