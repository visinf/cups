import torch


def label2unique(labels):
    # sorted: sorts in ascending order
    unique, labels = torch.unique(labels, sorted=True, return_inverse=True)

    if unique[0] == -1:
        labels = labels - 1
    return labels


def label2onehot(labels, label_min=0):

    labels_min = labels.min()
    if labels_min > label_min:
        label_min = labels_min
    device = labels.device

    if labels.dim() == 3 or labels.dim() == 4:
        onehot = torch.arange(label_min, labels.max() + 1).to(device)[None, :, None, None] == labels
    elif labels.dim() == 1:
        onehot = torch.arange(label_min, labels.max() + 1).to(device)[:, None] == labels[None, :]
    else:
        print("error: vision.label2onehot - input dim is not 1, 3 or 4")
        return labels
    # onehot = onehot * 1.0
    return onehot


def label2unique2onehot(labels):

    labels = label2unique(labels)
    onehot = label2onehot(labels)
    return onehot
