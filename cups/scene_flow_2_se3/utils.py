import torch
from torch import Tensor


def omit_too_small_object_proposals(labels: Tensor, min_object_size: int) -> Tensor:
    """Omits too small object proposals by setting them to background.

    Args:
        labels (Tensor): Labels of the shape [M].
        min_samples (int): Minimum number of samples. Default 400.

    Returns:
        labels_filtered (Tensor): Filtered labels of the shape [M].
    """
    # Omit too small clusters
    ids, counts = labels.unique(return_counts=True)
    ids[counts < min_object_size] = 0
    labels_filtered: Tensor = torch.embedding(indices=labels, weight=ids.view(-1, 1))[..., 0]
    return labels_filtered


def remap_ids(input: Tensor) -> Tensor:
    """Remaps IDs of cluster [0, N].

    Args:
        input (Tensor): ID tensor of the shape [M].

    Returns:
        output (Tensor): Remapped tensor of the shape [M].
    """
    # Get dtype and device
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    # Remap ids to -1 to N
    ids = input.unique()
    weight = torch.ones(ids.amax() + 1, dtype=dtype, device=device)
    weight[ids] = torch.arange(start=0, end=ids.shape[0], dtype=dtype, device=device)
    output: Tensor = torch.embedding(indices=input, weight=weight.view(-1, 1))[..., 0]
    return output
