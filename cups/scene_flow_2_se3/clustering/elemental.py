import torch


def onehot_2_label(onehot):
    # onehot: BxKxHxW

    value, label = torch.max(onehot, dim=1, keepdim=True)
    label[value < 0.5] = -1
    return label


def label_2_unique(labels):

    unique, labels = torch.unique(labels, return_inverse=True)
    return labels


def label_2_onehot(labels, label_max=None, negative_handling="ignore"):
    """Encodes labels with onehot masks.

    Parameters
    ----------
    labels torch.Tensor: N / BxN / BxHxW / Bx1xHxW, int
    negative_handling string: "ignore" / "join" / "separate"
    label_max int: ensures that
        negative_handling=="ignore"   : K == label_max+1
        negative_handling=="join"     : K == label_max+2
        negative_handling=="separate" : K >= label_max+1

    Returns
    -------
    onehot torch.Tensor: KxN / BxKxN / BxKxHxW, bool
    """

    device = labels.device

    labels_dim_in = labels.dim()

    # if label_min is None:
    #    if labels.numel() == 0:
    #        label_min = -1
    #    else:
    label_min = 0

    if label_max is None:
        if labels.numel() == 0:
            label_max = -1
        else:
            label_max = labels.max().long()

    if negative_handling == "ignore":
        labels[labels < 0] = -1
    elif negative_handling == "join":
        labels[labels < 0] = label_max
        label_max += 1
    elif negative_handling == "separate":
        labels[labels < 0] = label_max + torch.arange(labels[labels < 0], device=device) + 1
        label_max += len(labels < 0)

    if labels_dim_in == 2 or labels_dim_in == 3 or labels_dim_in == 4:
        if labels_dim_in == 2:
            B, N = labels.shape
            labels = labels.reshape(B, 1, N, 1)
        elif labels_dim_in == 3:
            B, H, W = labels.shape
            labels = labels.reshape(B, 1, H, W)

        onehot = torch.arange(label_min, label_max + 1).to(device)[None, :, None, None] == labels
        # B x K x H x W  | B x K x N x 1

        if labels_dim_in == 2:
            onehot = onehot.reshape(B, -1, N)

    elif labels_dim_in == 1:
        onehot = torch.arange(label_min, label_max + 1).to(device)[:, None] == labels[None, :]
        # K x N
    else:
        print("error: vision.label2onehot - input dim is not 1, 2, 3 or 4")
        return labels
    # onehot = onehot * 1.0

    # if ignore_negative and label_min < 0:
    #    onehot = onehot[:, abs(label_min) :]
    return onehot


def label_2_unique_2_onehot(labels):
    labels = label_2_unique(labels)
    onehot = label_2_onehot(labels)
    return onehot
