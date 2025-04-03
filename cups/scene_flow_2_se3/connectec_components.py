import torch
from cc_torch import connected_components_labeling
from scipy.ndimage import label
from torch import Tensor


def connected_components(input: Tensor) -> Tensor:
    """Performs connected components. If tensor is on the GPU CC is performed on the GPU if not Scipy (CPU) is used.

    Args:
        input (Tensor): Tensor of the shape [H, W].

    Returns:
        output (Tensor): Connected components as a long tensor of the shape [H, W].
    """
    # Perform connected components
    if input.is_cpu:
        output: Tensor = torch.from_numpy(label(input.detach().numpy())[0]).long()
    else:
        output = connected_components_labeling(input.byte()).long()
        for index, value in enumerate(output.unique(sorted=True)):
            output[output == value] = index
    return output
