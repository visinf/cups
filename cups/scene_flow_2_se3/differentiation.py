def calc_batch_gradients(batch, margin=0):
    # batch: torch.tensor: BxCxHxW
    shift = 1 + margin
    batch_grad_x = batch[:, :, :, shift:] - batch[:, :, :, :-shift]
    # B x 2 x H x W-shift

    batch_grad_y = batch[:, :, shift:, :] - batch[:, :, :-shift, :]
    # B x 2 x H-shift x W

    return batch_grad_x, batch_grad_y


def calc_batch_k_gradients(batch, order):

    print(batch.size())
    batch_k_grad_x, batch_k_grad_y = calc_batch_gradients(batch)
    print(batch_k_grad_x.size())
    for k in range(order - 1):
        batch_k_grad_x, _ = calc_batch_gradients(batch_k_grad_x)
        print(batch_k_grad_x.size())
        _, batch_k_grad_y = calc_batch_gradients(batch_k_grad_y)

    return batch_k_grad_x, batch_k_grad_y
