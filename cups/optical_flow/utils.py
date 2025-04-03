from typing import Tuple, Union

import kornia
import torch
from torch import Tensor

__all__: Tuple[str, ...] = (
    "backward_warp_flow",
    "backward_warp_grid",
    "estimate_occlusions",
    "threshold_valid_pixels",
)


def backward_warp_flow(
    image: Tensor,
    flow: Tensor,
    return_valid_pixels: bool = True,
    normalized_coordinates: bool = False,
    padding_mode: str = "border",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Function performs backward warping using optical flow.

    Args:
        image (Tensor): Image to be warped of the shape [B, C, H, W].
        flow (Tensor): Flow to be applied of the shape [B, 2, H, W].
        return_valid_pixels (bool): If true a binary map of valid pixels is also returned. Default: True.
        normalized_coordinates (bool): If true coordinates are assumed to be normalized to [-1, 1]. Default: False.
        padding_mode (str): Padding mode to be used. Default: "border".

    Returns:
        image_warped (Tensor): Warped image of the shape [B, C, H, W].
        valid_pixels (Optional[Tensor}): Map of valid pixels with shape [B, 1, H, W] on if return_valid_pixels is true.
    """
    # Get shapes
    B, _, H, W = image.shape  # type: int, int, int, int
    # Init default grid
    default_grid: Tensor = kornia.create_meshgrid(
        height=H, width=W, normalized_coordinates=normalized_coordinates, device=image.device
    ).permute(0, 3, 1, 2)
    return backward_warp_grid(
        image=image,
        grid=flow + default_grid,
        return_valid_pixels=return_valid_pixels,
        normalized_coordinates=normalized_coordinates,
        padding_mode=padding_mode,
    )


def backward_warp_grid(
    image: Tensor,
    grid: Tensor,
    return_valid_pixels: bool = True,
    normalized_coordinates: bool = False,
    padding_mode: str = "border",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Function performs backward warping using a full grid.

    Args:
        image (Tensor): Image to be warped of the shape [B, C, H, W].
        grid (Tensor): Grid to be applied of the shape [B, 2, H, W].
        return_valid_pixels (bool): If true a binary map of valid pixels is also returned. Default: True.
        normalized_coordinates (bool): If true coordinates are assumed to be normalized to [-1, 1]. Default: False.
        padding_mode (str): Padding mode to be used. Default: "border".

    Returns:
        image_warped (Tensor): Warped image of the shape [B, C, H, W].
        valid_pixels (Optional[Tensor}): Map of valid pixels with shape [B, 1, H, W] on if return_valid_pixels is true.
    """
    # Get shapes
    B, _, H, W = image.shape  # type: int, int, int, int
    # Get dtype
    dtype: torch.dtype = image.dtype
    # Warp image
    image_warped: Tensor = kornia.geometry.transform.remap(
        image=image.float(),
        map_x=grid[:, 0].float(),
        map_y=grid[:, 1].float(),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
        normalized_coordinates=normalized_coordinates,
    ).to(dtype)
    if not return_valid_pixels:
        return image_warped
    # Compute valid pixels
    valid_pixels: Tensor = kornia.geometry.transform.remap(
        image=torch.ones(B, 1, H, W, dtype=torch.float, device=image.device),
        map_x=grid[:, 0].float(),
        map_y=grid[:, 1].float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        normalized_coordinates=normalized_coordinates,
    ).to(dtype)
    valid_pixels: Tensor = threshold_valid_pixels(valid_pixels=valid_pixels)
    return image_warped, valid_pixels


def threshold_valid_pixels(valid_pixels: Tensor) -> Tensor:
    """Function thresholds a map of valid pixels.

    Args:
        valid_pixels (Tensor): Any tensor.

    Returns:
        valid_pixels (Tensor): Thresholded tensor with the same shape as input.
    """
    valid_pixels[valid_pixels < 0.999] = 0.0
    valid_pixels[valid_pixels > 0.0] = 1.0
    return valid_pixels


def estimate_occlusions(
    flow_forward: Tensor,
    flow_backward: Tensor,
    alpha_1: float = 0.01,
    alpha_2: float = 0.5,
    stop_gradients: bool = True,
) -> Tensor:
    """Function estimates occlusions based on forward-backward consistency of optical flow estimate.

    References:
        https://lmb.informatik.uni-freiburg.de/Publications/2010/Bro10e/sundaram_eccv10.pdf
        https://arxiv.org/pdf/2006.04902.pdf

    Notes:
        Note that occluded pixels entail a value of 0 and valid pixels a value of 1.

    Args:
        flow_forward (Tensor): Optical flow from t to t + 1 with the shape [B, 2, H, W].
        flow_backward (Tensor): Optical flow from t + 1 to t with the shape [B, 2, H, W].
        alpha_1 (float): Fist constant for tolerance interval. Default 0.01.
        alpha_2 (float): Second constant for tolerance interval. Default 0.5.
        stop_gradients (bool): If true gradients are stopped. Default True.

    Returns:
        occlusions (Tensor): Binary (same dtype as flow_forward) occlusion map of the shape [B, 1, H, W].
    """
    # Stop gradients if utilized
    if stop_gradients:
        flow_forward = flow_forward.detach()
        flow_backward = flow_backward.detach()
    # Warp backward flow to first image
    flow_backward_warped, valid_pixels = backward_warp_flow(  # type: ignore
        image=flow_backward, flow=flow_forward, return_valid_pixels=True, normalized_coordinates=False
    )
    # Compute binary occlusion map
    first_term: Tensor = torch.norm(flow_forward + flow_backward_warped, dim=1) ** 2
    second_term: Tensor = (torch.norm(flow_forward, dim=1) ** 2) + (torch.norm(flow_backward_warped, dim=1) ** 2)
    occlusions: Tensor = first_term < (alpha_1 * second_term + alpha_2)
    # Add dimension and cast to dtype of flow
    occlusions = occlusions.unsqueeze(dim=1).to(flow_forward.dtype) * valid_pixels.to(flow_forward.dtype)
    return occlusions


def estimate_occlusions_ssim(
    image_1: Tensor,
    image_2: Tensor,
    flow_forward: Tensor,
    max_ssim: float = 0.3,
    stop_gradients: bool = True,
) -> Tensor:
    """Function estimates occlusions based on SSIM concistency.

    Notes:
        Note that occluded pixels entail a value of 0 and valid pixels a value of 1.

    Args:
        flow_forward (Tensor): Optical flow from t to t + 1 with the shape [B, 2, H, W].
        flow_backward (Tensor): Optical flow from t + 1 to t with the shape [B, 2, H, W].
        alpha_1 (float): Fist constant for tolerance interval. Default 0.01.
        alpha_2 (float): Second constant for tolerance interval. Default 0.5.
        stop_gradients (bool): If true gradients are stopped. Default True.

    Returns:
        occlusions (Tensor): Binary (same dtype as flow_forward) occlusion map of the shape [B, 1, H, W].
    """
    # Stop gradients if utilized
    if stop_gradients:
        flow_forward = flow_forward.detach()
        image_1 = image_1.detach()
        image_2 = image_2.detach()
    # Warp backward flow to first image
    image_2_warp, valid_pixels = backward_warp_flow(  # type: ignore
        image=image_2, flow=flow_forward, return_valid_pixels=True, normalized_coordinates=False
    )
    # Compute ssim between original and wrapped image
    ssim = kornia.metrics.ssim(image_1, image_2_warp, window_size=3).mean(dim=1, keepdim=True)
    # Estimate occlusions
    occlusions = ssim < max_ssim
    # Add dimension and cast to dtype of flow
    occlusions = occlusions.to(flow_forward.dtype) * valid_pixels.to(flow_forward.dtype)
    return occlusions  # type: ignore
