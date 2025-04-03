from __future__ import annotations

import os
import pathlib
from typing import Tuple

import kornia.enhance
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from cv2 import CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, drawContours, findContours
from kornia.color import lab_to_rgb
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.io import write_png
from torchvision.utils import flow_to_image

import cups.utils
from cups.data.utils import get_bounding_boxes

__all__: Tuple[str, ...] = (
    "plot_panoptic_segmentation_overlay",
    "plot_panoptic_segmentation",
    "plot_semantic_segmentation",
    "plot_semantic_segmentation_overlay",
    "plot_optical_flow",
    "plot_disparity",
    "plot_disparity_overlay",
    "plot_image",
    "plot_occlusions",
    "plot_object_proposals",
    "plot_object_proposals_with_noise",
    "plot_object_proposals_overlay",
    "plot_scene_flow",
    "save_panoptic_segmentation",
    "save_panoptic_segmentation_overlay",
    "save_semantic_segmentation_overlay",
    "save_semantic_segmentation",
    "save_disparity",
    "save_disparity_overlay",
    "save_image",
    "save_optical_flow",
    "save_occlusions",
    "save_object_proposals",
    "save_object_proposals_with_noise",
    "save_object_proposals_overlay",
    "scene_flow_to_image",
    "save_scene_flow",
    "semantic_segmentation_to_rgb",
    "semantic_segmentation_overlay_to_rgb",
    "panoptic_segmentation_to_rgb",
    "panoptic_segmentation_overlay_to_rgb",
    "object_proposals_to_rgb",
)

COS_45: float = 1.0 / np.sqrt(2.0)
SIN_45: float = 1.0 / np.sqrt(2.0)

CLASS_TO_RGB_COCO: Tuple[Tuple[int, int, int], ...] = (
    (220, 20, 60),  # Background
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
    (0, 0, 0),  # Ignore is black
)

CLASS_TO_RGB_MOTS: Tuple[Tuple[int, int, int], ...] = (
    (220, 20, 60),  # Background
    (220, 20, 60),  # person
    (0, 0, 0),  # Ignore is black
)

CLASS_TO_RGB_CITYSCAPES: Tuple[Tuple[int, int, int], ...] = (
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (250, 170, 160),  # parking
    (230, 150, 140),  # rail track
    (70, 70, 70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (180, 165, 180),  # guard rail
    (150, 100, 100),  # bridge
    (150, 120, 90),  # tunnel
    (153, 153, 153),  # pole
    (153, 153, 153),  # polegroup
    (250, 170, 30),  # traffic light
    (220, 220, 0),  # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person
    (255, 0, 0),  # rider
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 0, 90),  # caravan
    (0, 0, 110),  # trailer
    (0, 80, 100),  # train
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
    (0, 0, 0),  # Ignore is black
)

CLASS_TO_RGB_CITYSCAPES_19: Tuple[Tuple[int, int, int], ...] = (
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (0, 0, 0),  # Ignore is black
)

CLASS_TO_RGB_CITYSCAPES_7: Tuple[Tuple[int, int, int], ...] = (
    (128, 64, 128),
    (70, 70, 70),
    (250, 170, 30),
    (107, 142, 35),
    (70, 130, 180),
    (220, 20, 60),
    (0, 0, 142),
    (0, 0, 0),  # Ignore is black
)

RANDOM_COLORS = (
    (128, 64, 128),
    (244, 35, 232),
    (250, 170, 160),
    (230, 150, 140),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (180, 165, 180),
    (150, 100, 100),
    (150, 120, 90),
    (153, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 0, 90),
    (0, 0, 110),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (54, 11, 42),
    (75, 64, 94),
    (173, 106, 88),
    (217, 190, 249),
    (30, 103, 25),
    (222, 68, 199),
    (239, 120, 57),
    (139, 177, 205),
    (174, 50, 126),
    (229, 207, 198),
    (137, 55, 63),
    (62, 219, 23),
    (7, 180, 74),
    (221, 223, 180),
    (166, 189, 227),
    (12, 212, 156),
    (69, 212, 128),
    (47, 80, 175),
    (117, 229, 14),
    (161, 160, 172),
    (233, 159, 176),
    (168, 158, 236),
    (171, 104, 136),
    (180, 109, 117),
    (245, 155, 192),
    (56, 174, 178),
    (243, 207, 7),
    (56, 37, 29),
    (232, 42, 39),
    (239, 0, 193),
    (163, 71, 181),
    (254, 248, 49),
    (57, 63, 7),
    (188, 121, 156),
    (150, 109, 229),
    (217, 33, 99),
    (123, 6, 49),
    (78, 186, 91),
    (175, 247, 98),
    (50, 212, 57),
    (46, 125, 198),
    (138, 206, 221),
    (14, 100, 223),
    (214, 179, 133),
    (134, 136, 127),
    (122, 157, 16),
    (67, 211, 11),
    (242, 247, 196),
    (1, 244, 183),
    (44, 193, 156),
    (19, 28, 181),
    (202, 90, 22),
    (247, 23, 144),
    (20, 236, 140),
    (45, 235, 214),
    (233, 153, 129),
    (65, 32, 33),
    (210, 18, 195),
    (148, 79, 66),
    (240, 53, 204),
    (122, 136, 217),
    (109, 179, 7),
    (80, 34, 221),
    (159, 46, 141),
    (205, 157, 62),
    (231, 146, 100),
    (211, 23, 200),
    (157, 177, 173),
    (86, 74, 57),
    (86, 213, 129),
    (21, 44, 10),
    (104, 130, 77),
    (228, 58, 108),
    (82, 29, 143),
    (152, 77, 209),
    (56, 225, 234),
    (244, 221, 73),
    (112, 239, 6),
    (136, 105, 76),
    (234, 222, 174),
    (251, 149, 241),
    (65, 5, 41),
    (185, 73, 251),
    (76, 48, 158),
    (209, 233, 30),
    (240, 242, 143),
    (177, 64, 155),
    (216, 201, 51),
    (147, 103, 126),
    (15, 221, 26),
    (166, 43, 138),
    (93, 194, 254),
    (208, 99, 118),
    (27, 14, 159),
    (194, 53, 3),
    (88, 142, 148),
    (21, 147, 244),
    (118, 151, 215),
    (4, 40, 231),
    (77, 137, 43),
    (223, 114, 34),
    (143, 210, 217),
    (58, 38, 230),
    (87, 69, 29),
    (123, 225, 154),
    (124, 150, 23),
    (164, 252, 112),
    (52, 113, 68),
    (46, 50, 26),
    (159, 52, 44),
    (105, 245, 214),
    (114, 242, 209),
    (178, 160, 115),
    (209, 185, 159),
    (13, 23, 140),
    (45, 192, 142),
    (194, 10, 59),
    (7, 111, 137),
    (52, 54, 9),
    (41, 98, 107),
    (142, 241, 173),
    (70, 39, 89),
    (120, 6, 178),
    (5, 74, 156),
    (195, 39, 54),
    (228, 177, 128),
    (185, 186, 95),
    (84, 5, 31),
    (45, 40, 171),
    (107, 140, 174),
    (33, 210, 43),
    (243, 82, 17),
    (151, 169, 127),
    (116, 234, 117),
    (251, 44, 191),
    (133, 200, 208),
    (154, 244, 101),
    (175, 101, 229),
    (232, 151, 203),
    (61, 104, 175),
    (11, 90, 208),
    (20, 76, 198),
    (189, 103, 111),
    (161, 32, 252),
    (71, 49, 8),
    (18, 39, 209),
    (171, 15, 104),
    (250, 115, 166),
    (24, 12, 65),
    (141, 74, 3),
    (145, 121, 96),
    (200, 28, 238),
    (22, 134, 173),
    (182, 97, 60),
    (250, 204, 227),
    (47, 15, 36),
    (247, 59, 169),
    (67, 32, 92),
    (17, 48, 65),
    (163, 227, 179),
    (254, 41, 12),
    (42, 92, 104),
    (109, 145, 124),
    (16, 163, 230),
    (26, 60, 163),
    (191, 41, 70),
    (54, 174, 165),
    (69, 36, 249),
    (172, 54, 5),
    (238, 133, 127),
    (161, 118, 246),
    (168, 94, 109),
    (191, 232, 242),
    (220, 127, 14),
    (35, 2, 77),
    (57, 30, 97),
    (189, 38, 215),
    (67, 181, 209),
    (166, 253, 107),
    (201, 67, 234),
    (113, 235, 234),
    (168, 176, 43),
    (198, 27, 10),
    (3, 234, 173),
    (202, 211, 254),
    (127, 130, 154),
    (66, 11, 191),
    (100, 74, 202),
    (200, 27, 67),
    (235, 94, 159),
    (4, 235, 93),
    (22, 159, 122),
    (37, 166, 203),
    (23, 74, 42),
    (192, 221, 8),
    (226, 94, 62),
    (121, 96, 16),
    (157, 189, 136),
    (2, 87, 223),
    (152, 167, 13),
    (45, 239, 81),
    (99, 39, 41),
    (174, 141, 223),
    (67, 86, 178),
    (177, 170, 159),
    (187, 13, 59),
    (145, 247, 206),
    (74, 151, 214),
    (88, 27, 233),
    (41, 112, 163),
    (108, 250, 66),
    (235, 206, 106),
    (163, 66, 137),
    (207, 252, 244),
    (149, 15, 253),
    (147, 0, 151),
    (68, 85, 10),
    (13, 150, 210),
    (83, 12, 167),
    (185, 162, 79),
    (110, 178, 164),
    (0, 39, 107),
    (188, 243, 204),
    (251, 31, 122),
    (213, 59, 10),
    (223, 159, 210),
    (15, 13, 84),
    (34, 130, 44),
    (0, 0, 0),  # Ignore is black
)


def _tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Converts a given torch tensor to a numpy array.

    Args:
        tensor (Tensor): PyTorch tensor of any shape.

    Returns:
        tensor_np (np.ndarray): Numpy array.
    """
    # Get dtype
    dtype = tensor.dtype if tensor.dtype in [torch.long, torch.uint8] else torch.float
    return tensor.to(dtype).cpu().detach().numpy()  # type: ignore


def plot_image(image: Tensor, denormalize: bool = False) -> None:
    """Plots a given image using matplotlib.

    Args:
        image (Tensor): Image (RGB) of the shape [3, H, W].
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Denormalize if needed
    if denormalize:
        image = cups.utils.denormalize(image[None])[0]
    # Init figure
    fig, ax = plt.subplots()
    # Plot image
    ax.imshow(_tensor_to_numpy(image.permute(1, 2, 0)))
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def save_image(image: Tensor, path: str | None, denormalize: bool = False) -> None | Tensor:
    """Save a given RGB image to the provided path.

    Args:
        image (Tensor): Image (RGB) of the shape [3, H, W] pixel range should be [0, 1].
        path (str | None): Path to store the image to.
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Denormalize if needed
    if denormalize:
        image = cups.utils.denormalize(image[None])[0]
    # Check png is used
    if path is not None:
        assert ".png" in path, "Use png not something else!"
    # Save image as a png
    if path is not None:
        write_png((255.0 * image).round().clip(0, 255).byte().cpu(), path)
    return (255.0 * image).round().clip(0, 255).byte().cpu()


def plot_optical_flow(optical_flow: Tensor) -> None:
    """Plots a given optical flow map.

    Args:
        optical_flow (Tensor): Optical flow of the shape [2, H, W].
    """
    # Optical flow to rgb
    optical_flow_rgb: Tensor = flow_to_image(optical_flow.float())
    # Plot optical flow
    plot_image(image=optical_flow_rgb)


def save_optical_flow(optical_flow: Tensor, path: str | None) -> None | Tensor:
    """Save a given optical flow map to the provided path.

    Args:
        optical_flow (Tensor): Optical flow of the shape [2, H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Optical flow to rgb
    optical_flow_rgb: Tensor = flow_to_image(optical_flow.float())
    # Save optical flow as an image
    if path is not None:
        save_image(image=optical_flow_rgb / 255.0, path=path)
    return optical_flow_rgb / 255.0


def plot_occlusions(occlusions: Tensor) -> None:
    """Plots a binary occlusion map.

    Notes:
        Can be also used to plot valid pixels.

    Args:
        occlusions (Tensor): Occlusion map of the shape [1, H, W].
    """
    # Init figure
    fig, ax = plt.subplots()
    # Plot image
    ax.imshow(_tensor_to_numpy(occlusions[0]), cmap="gray", vmin=0.0, vmax=1.0)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def save_occlusions(occlusions: Tensor, path: str | None) -> None | Tensor:
    """Save a binary occlusion map to the provided path.

    Notes:
        Can be also used to save valid pixels.

    Args:
        occlusions (Tensor): Occlusion map of the shape [1, H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Occlusions to RGB
    occlusions_rgb: Tensor = occlusions.repeat_interleave(3, dim=0)
    # Save occlusion map as an image
    if path is not None:
        save_image(image=occlusions_rgb, path=path)
    return occlusions_rgb.float()


def plot_disparity(disparity: Tensor) -> None:
    """Plots a given disparity map.

    Args:
        disparity (Tensor): Disparity maps of the shape [1, H, W].
    """
    # Init figure
    _, ax = plt.subplots()
    # Plot image
    ax.imshow(_tensor_to_numpy(-disparity[0]), cmap="magma")
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def save_disparity(disparity: Tensor, path: str | None) -> None | Tensor:
    """Saves a given disparity to the provided path.

    Args:
        disparity (Tensor): Disparity maps of the shape [1, H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Get colormap
    color_map = cm.get_cmap("magma")
    # Normalize disparity to a pixel range of [0, 1]
    disparity = kornia.enhance.normalize_min_max(-disparity[None])[0]
    # Apply color map
    disparity_rgb = color_map(_tensor_to_numpy(disparity[0]))[..., :3]
    # Save disparity map as an image
    if path is not None:
        save_image(image=torch.from_numpy(disparity_rgb).permute(2, 0, 1), path=path)
    return torch.from_numpy(disparity_rgb).permute(2, 0, 1)


def plot_disparity_overlay(disparity: Tensor, image: Tensor, alpha: float = 0.7) -> None:
    """Plots a given disparity map.

    Args:
        disparity (Tensor): Disparity maps of the shape [1, H, W].
        image (Tensor): Image of the shape [3, H, W].
        alpha (float): Overlay factor. Default 0.4.
    """
    # Get colormap
    color_map = cm.get_cmap("magma")
    # Normalize disparity to a pixel range of [0, 1]
    disparity = kornia.enhance.normalize_min_max(-disparity[None])[0]
    # Apply color map
    disparity_rgb = color_map(_tensor_to_numpy(disparity[0]))[..., :3]
    # Overlay image and disparity
    overlay = _overlay(torch.from_numpy(disparity_rgb).permute(2, 0, 1), image, alpha=alpha)
    plot_image(overlay)


def save_disparity_overlay(disparity: Tensor, image: Tensor, path: str | None, alpha: float = 0.7) -> None | Tensor:
    """Saves a given disparity to the provided path.

    Args:
        disparity (Tensor): Disparity maps of the shape [1, H, W].
        image (Tensor): Image of the shape [3, H, W].
        path (str | None): Path to store the image to.
        alpha (float): Overlay factor. Default 0.4.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Get colormap
    color_map = cm.get_cmap("magma")
    # Normalize disparity to a pixel range of [0, 1]
    disparity = kornia.enhance.normalize_min_max(-disparity[None])[0]
    # Apply color map
    disparity_rgb = color_map(_tensor_to_numpy(disparity[0]))[..., :3]
    # Overlay image and disparity
    overlay = _overlay(torch.from_numpy(disparity_rgb).permute(2, 0, 1), image, alpha=alpha)
    # Save disparity map as an image
    if path is not None:
        save_image(image=overlay, path=path)
    return overlay


def semantic_segmentation_to_rgb(semantic_segmentation: Tensor, dataset: str = "cityscapes") -> Tensor:
    """Converts a given semantic segmentation to RGB.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".

    Returns:
        semantic_segmentation_rgb (Tensor): RGB semantic segmentation of the shape [3, H, W] pix. range [0, 255].
    """
    # Check dataset
    assert dataset in (
        "cityscapes",
        "cityscapes_19",
        "cityscapes_7",
        "coco",
        "mots",
        "waymo",
        "pseudo",
    ), f"Dataset {dataset} not supported!"
    # Ensure semantic segmentation is on CPU
    semantic_segmentation = semantic_segmentation.cpu().detach()
    # Get color encoding
    if dataset == "cityscapes":
        color_encoding: Tensor = torch.tensor(CLASS_TO_RGB_CITYSCAPES)
    elif dataset == "cityscapes_19":
        color_encoding = torch.tensor(CLASS_TO_RGB_CITYSCAPES_19)
    elif dataset == "cityscapes_7":
        color_encoding = torch.tensor(CLASS_TO_RGB_CITYSCAPES_7)
    elif dataset == "coco":
        color_encoding = torch.tensor(CLASS_TO_RGB_COCO)
    elif dataset == "mots":
        color_encoding = torch.tensor(CLASS_TO_RGB_MOTS)
    else:
        color_encoding = torch.tensor(RANDOM_COLORS)
    # map void to last color embedding
    semantic_segmentation[semantic_segmentation == 255] = len(color_encoding) - 1
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = torch.embedding(indices=semantic_segmentation.long(), weight=color_encoding)
    return semantic_segmentation_rgb.permute(2, 0, 1).long()


def plot_semantic_segmentation(semantic_segmentation: Tensor, dataset: str = "cityscapes") -> None:
    """Plots a given semantic segmentation map.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
    """
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # Plot semantic segmentation map
    plot_image(semantic_segmentation_rgb)


def save_semantic_segmentation(
    semantic_segmentation: Tensor,
    path: str | None,
    dataset: str = "cityscapes",
) -> None | Tensor:
    """Saves a given semantic segmentation map.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        path (str | None): Path to store the image to.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # Plot semantic segmentation map
    if path is not None:
        save_image(semantic_segmentation_rgb / 255.0, path=path)
    return semantic_segmentation_rgb / 255.0


def _overlay(segmentation: Tensor, image: Tensor, alpha: float = 0.4, ignore_background: bool = False) -> Tensor:
    """Overlays a segmentation with an image.

    Args:
        segmentation (Tensor): RGB segmentation of the shape [3, H, W].
        image (Tensor): Image of the shape [3, H, W].
        alpha (float): Overlay strength. Default 0.4.
        ignore_background (bool): If true background is ignored. Default False.

    Returns:
        overlay (Tensor): Overlay of image and segmentation of the shape [3, H, W].
    """
    if ignore_background:
        overlay: Tensor = torch.where(
            (segmentation != 0.0).all(dim=0, keepdim=True), alpha * segmentation + (1.0 - alpha) * image, image
        )
    else:
        overlay = alpha * segmentation + (1.0 - alpha) * image
    return overlay


def semantic_segmentation_overlay_to_rgb(
    semantic_segmentation: Tensor,
    image: Tensor,
    alpha: float = 0.4,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> Tensor:
    """Generates a semantic segmentation map overlaid with an image as a tensor.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        image (Tensor): Image of the shape [3, H, W].
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.

    Returns:
        overlay (Tensor): Overlay as an RGB image of the shape [3, H, W], pixel range is [0, 255] (long).
    """
    # Denormalize if needed
    if denormalize:
        image = cups.utils.denormalize(image[None])[0]
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # Overlay segmentation
    overlay: Tensor = _overlay(semantic_segmentation_rgb, 255.0 * image, alpha=alpha).long()
    return overlay


def plot_semantic_segmentation_overlay(
    semantic_segmentation: Tensor,
    image: Tensor,
    alpha: float = 0.4,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> None:
    """Plots a given a semantic segmentation map overlaid with an image.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        image (Tensor): Image of the shape [3, H, W].
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Make overlay
    overlay: Tensor = semantic_segmentation_overlay_to_rgb(semantic_segmentation, image, alpha, dataset, denormalize)
    # Plot semantic segmentation map
    plot_image(overlay / 255.0)


def save_semantic_segmentation_overlay(
    semantic_segmentation: Tensor,
    image: Tensor,
    path: str | None,
    alpha: float = 0.4,
    dataset: str = ["cityscapes", "waymo", "pseudo"][0],
    denormalize: bool = False,
) -> None | Tensor:
    """Saves a given a semantic segmentation map overlaid with an image.

    Args:
        semantic_segmentation (Tensor): Semantic segmentation map of the shape [H, W].
        image (Tensor): Image of the shape [3, H, W].
        path (str | None): Path to store the image to.
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Make overlay
    overlay: Tensor = semantic_segmentation_overlay_to_rgb(semantic_segmentation, image, alpha, dataset, denormalize)
    # Plot semantic segmentation map
    if path is not None:
        save_image(overlay / 255.0, path=path)
    return overlay / 255.0


def _get_object_centers(instance_segmentation: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes the object centers of each instance.

    Args:
        instance_segmentation (Tensor): Instance segmentation of the shape [H, W].

    Returns:
        ids (Tensor): Instance IDs of the shape [N] (excluding background).
        object_centers (Tensor): Rounded object centers of the shape [N, 2].
    """
    # Get instance IDs
    ids: Tensor = torch.unique(instance_segmentation)[1:]
    # Compute centers
    object_centers: Tensor = torch.empty((ids.shape[0], 2), dtype=torch.long, device=ids.device)
    for index_id, id in enumerate(ids):
        indexes: Tensor = torch.argwhere(instance_segmentation == id).float()
        object_centers[index_id] = indexes.mean(dim=0).round()
    return ids, object_centers


def _draw_instance_contours(semantic_segmentation_rgb: Tensor, instance_segmentation: Tensor) -> Tensor:
    """Draw instance contours on the RGB semantic segmentation map.

    Args:
        semantic_segmentation_rgb (Tensor): RGB semantic segmentation of the shape [3, H, W].
        instance_segmentation (Tensor): Instance segmentation of the shape [H, W].
    """
    # convert to numpy
    segmentation_rgb: Tensor = np.array(  # type: ignore
        semantic_segmentation_rgb.detach().clone(), dtype=np.uint8  # type: ignore
    ).transpose(
        1, 2, 0  # type: ignore
    )  # type: ignore
    # dynamic line width
    line_width = int(3 * segmentation_rgb.shape[0] / 1024)
    for instance_id in instance_segmentation.unique():
        # Skip background
        if instance_id == 0:
            continue
        # Create a binary mask for the current instance id
        binary_mask = np.uint8(instance_segmentation == instance_id)
        # Find contours for the current instance
        contours, _ = findContours(binary_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        # Draw the contours on the contour_image
        drawContours(segmentation_rgb, contours, -1, (255, 255, 255), line_width)

    return torch.Tensor(segmentation_rgb).permute(2, 0, 1)


def _add_ids(segmentation_rgb: Tensor, ids: Tensor, object_centers: Tensor, font_type: str = "Helvetica.ttf") -> Tensor:
    """Adds instance IDs to RGB segmentation.

    Args:
        segmentation_rgb (Tensor): RGB segmentation of the shape [3, H, W].
        ids (Tensor): Instance IDs of the shape [N].
        object_centers (Tensor): Object centers of the shape [N].
        font_size (int): Font size to be used. Default 10.

    Returns:
        segmentation_rgb_text (Tensor): RGB segmentation with text of the shape [3, H, W].
    """
    # dynamic font size
    font_size = int(42 * segmentation_rgb.shape[1] / 1024)
    # Get font
    font = ImageFont.truetype(os.path.join(pathlib.Path(__file__).parent.resolve(), font_type), font_size)
    # Segmentation to RGB image
    segmentation_rgb_image = Image.fromarray(_tensor_to_numpy(segmentation_rgb.permute(1, 2, 0)).astype(np.uint8))
    # Make drawer
    draw = ImageDraw.Draw(segmentation_rgb_image)
    # Add text
    for id, object_center in zip(ids, object_centers):
        draw.text((object_center[1], object_center[0]), str(id.item()), font=font, align="center", anchor="mm")
    # Back to torch
    segmentation_rgb_text: Tensor = torch.from_numpy(np.array(segmentation_rgb_image)).permute(2, 0, 1)
    return segmentation_rgb_text


def panoptic_segmentation_to_rgb(
    panoptic_segmentation: Tensor, dataset: str = "cityscapes", instance_contour: bool = True
) -> Tensor:
    """Plots a given panoptic segmentation map.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".

    Returns:
        segmentation_rgb (Tensor): Panoptic segmentation as an RGB image [3, B, W], pixel range os [0, 255].
    """
    # Data to cpu
    panoptic_segmentation = panoptic_segmentation.cpu()
    # Get semantic and instance segmentation
    semantic_segmentation, instance_segmentation = panoptic_segmentation[..., 0], panoptic_segmentation[..., 1]
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # Apply contours around instance masks
    if instance_contour:
        segmentation_rgb: Tensor = _draw_instance_contours(semantic_segmentation_rgb, instance_segmentation)
    else:
        # Make a random color shift for each instance
        weight: Tensor = 60.0 * (2.0 * torch.rand(instance_segmentation.amax() + 1) - 1.0)  # type: ignore
        weight[0] = 0.0
        color_shift: Tensor = torch.embedding(indices=instance_segmentation, weight=weight[..., None]).permute(2, 0, 1)
        # Apply color shift
        segmentation_rgb = (semantic_segmentation_rgb + color_shift).clip(min=0, max=255).long()
    # Get object centers
    ids, object_centers = _get_object_centers(instance_segmentation)
    # Put IDs into RGB segmentation
    segmentation_rgb = _add_ids(segmentation_rgb, ids, object_centers)
    return segmentation_rgb


def plot_panoptic_segmentation(panoptic_segmentation: Tensor, dataset: str = "cityscapes") -> None:
    """Plots a given panoptic segmentation map.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
    """
    # Make panoptic segmentation RGB
    segmentation_rgb = panoptic_segmentation_to_rgb(panoptic_segmentation, dataset)
    # Plot semantic segmentation map
    plot_image(segmentation_rgb)


def save_panoptic_segmentation(
    panoptic_segmentation: Tensor,
    path: str | None,
    dataset: str = "cityscapes",
) -> None | Tensor:
    """Plots a given panoptic segmentation map.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        path (str | None): Path to store the image to.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Make panoptic segmentation RGB
    segmentation_rgb = panoptic_segmentation_to_rgb(panoptic_segmentation, dataset)
    # Plot panoptic segmentation map
    if path is not None:
        save_image(segmentation_rgb / 255.0, path=path)
    return segmentation_rgb / 255.0


def panoptic_segmentation_overlay_to_rgb(
    panoptic_segmentation: Tensor,
    image: Tensor,
    alpha: float = 0.4,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> Tensor:
    """Plots a given panoptic segmentation map overlaid with an image.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        image (Tensor): Image of the shape [3, H, W].
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.

    Returns:
        overlay (Tensor): Seg. overlay as an RGB image of the shape [3, B, W], pix. range is [0, 255].
    """
    # Denormalize if needed
    if denormalize:
        image = cups.utils.denormalize(image[None])[0]
    # Get semantic and instance segmentation
    semantic_segmentation, instance_segmentation = panoptic_segmentation[..., 0], panoptic_segmentation[..., 1]
    # Semantic segmentation to RGB
    semantic_segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # Make a random color shift for each instance
    weight: Tensor = 60.0 * (2.0 * torch.rand(instance_segmentation.amax() + 1) - 1.0)  # type: ignore
    weight[0] = 0.0
    color_shift: Tensor = torch.embedding(indices=instance_segmentation, weight=weight[..., None]).permute(2, 0, 1)
    # Apply color shift
    segmentation_rgb: Tensor = (semantic_segmentation_rgb + color_shift).clip(min=0, max=255).long()
    # Get object centers
    ids, object_centers = _get_object_centers(instance_segmentation)
    # Overlay segmentation
    overlay: Tensor = _overlay(segmentation_rgb, image * 255.0, alpha=alpha)
    # Put IDs into RGB segmentation
    overlay = _add_ids(overlay, ids, object_centers).long()
    return overlay


def plot_panoptic_segmentation_overlay(
    panoptic_segmentation: Tensor,
    image: Tensor,
    bounding_boxes: bool = False,
    alpha: float = 0.4,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> None:
    """Plots a given panoptic segmentation map overlaid with an image.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        image (Tensor): Image of the shape [3, H, W].
        bounding_boxes (Tensor | None= None): Bounding boxes of the shape [N, 4 (XYXY_ABS)]. Default None.
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Get RGB panoptic segmentation overlay
    overlay: Tensor = panoptic_segmentation_overlay_to_rgb(panoptic_segmentation, image, alpha, dataset, denormalize)
    # Plot bounding boxes if given
    if bounding_boxes:
        bounding_boxes = get_bounding_boxes(panoptic_segmentation[..., 1])  # type: ignore
        bounding_boxes = bounding_boxes.cpu().detach().clone()  # type: ignore
        overlay = torchvision.utils.draw_bounding_boxes(overlay.byte(), bounding_boxes, colors=(255, 0, 0))
    # Plot semantic segmentation map
    plot_image(overlay)


def save_panoptic_segmentation_overlay(
    panoptic_segmentation: Tensor,
    image: Tensor,
    path: str | None,
    bounding_boxes: bool | Tensor = False,
    alpha: float = 0.6,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> None | Tensor:
    """Plots a given panoptic segmentation map overlaid with an image.

    Args:
        semantic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W, 2].
        image (Tensor): Image of the shape [3, H, W].
        path (str | None): Path to store the image to.
        bounding_boxes (Tensor | bool): Bounding boxes of the shape [N, 4 (XYXY_ABS)]. Default None.
        alpha (float): Overlay strength. Default 0.4.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Get RGB panoptic segmentation overlay
    overlay: Tensor = panoptic_segmentation_overlay_to_rgb(panoptic_segmentation, image, alpha, dataset, denormalize)
    # Plot bounding boxes if given
    if bounding_boxes:
        bounding_boxes = get_bounding_boxes(panoptic_segmentation[..., 1])  # type: ignore
        bounding_boxes = bounding_boxes.cpu().detach().clone()  # type: ignore
        overlay = torchvision.utils.draw_bounding_boxes(overlay.byte(), bounding_boxes, colors=(255, 0, 0))
    # Plot semantic segmentation map
    if path is not None:
        save_image(overlay.float() / 255.0, path=path)
    return overlay.float() / 255.0


def save_panoptic_segmentation_imggtpred_seperated(
    panoptic_segmentation: Tensor,
    image: Tensor,
    gt: Tensor,
    path: str | None = "",
    bounding_boxes: bool = True,
    dataset: str = "cityscapes",
    denormalize: bool = False,
) -> None | Tensor:
    """Plots a given what?

    Args:
        panoptic_segmentation (Tensor): Panoptic segmentation map of the shape [H, W].
        image (Tensor): Image of the shape [3, H, W].
        gt (Tensor): ???
        path (str | None): Path to store the image to.
        dataset (str): Dataset to be used (cityscapes, coco(?)). Default: "cityscapes".
        denormalize (bool): Set to true if image should be denormalized (w/ ImageNet stats). Default: False.
    """
    # Denormalize if needed
    if denormalize:
        image = cups.utils.denormalize(image[None])[0]
    # Get semantic and instance segmentation
    semantic_segmentation, instance_segmentation = panoptic_segmentation[..., 0], panoptic_segmentation[..., 1]
    # Semantic segmentation to RGB
    segmentation_rgb: Tensor = semantic_segmentation_to_rgb(semantic_segmentation, dataset=dataset)
    # # # Make a random color shift for each instance
    # # weight: Tensor = 60.0 * (2.0 * torch.rand(instance_segmentation.amax() + 1) - 1.0)  # type: ignore
    # # weight[0] = 0.0
    # # color_shift: Tensor = torch.embedding(indices=instance_segmentation, weight=weight[..., None]).permute(2, 0, 1)
    # # # Apply color shift
    # # segmentation_rgb: Tensor = (semantic_segmentation_rgb + color_shift).clip(min=0, max=255).long()
    # Get object centers
    ids, object_centers = _get_object_centers(instance_segmentation)
    # Put IDs into RGB segmentation
    segmentation_rgb_ids = _add_ids(segmentation_rgb, ids, object_centers)
    if bounding_boxes:
        b_boxes = get_bounding_boxes(instance_segmentation)
        b_boxes = b_boxes.cpu().detach().clone()
        segmentation_rgb_ids = torchvision.utils.draw_bounding_boxes(
            segmentation_rgb_ids.byte(), b_boxes, colors=(255, 0, 0)
        )
    # Get groud truth to RGB
    ground_truth_rgb: Tensor = semantic_segmentation_to_rgb(gt[..., 0], dataset=dataset)
    # Get object centers
    ids, object_centers = _get_object_centers(gt[..., 1])
    # Put IDs into RGB groud truth
    ground_truth_rgb_ids = _add_ids(ground_truth_rgb, ids, object_centers)
    # Plot bounding boxes if given
    if bounding_boxes:
        b_boxes = get_bounding_boxes(gt[..., 1])
        b_boxes = b_boxes.cpu().detach().clone()
        ground_truth_rgb_ids = torchvision.utils.draw_bounding_boxes(
            ground_truth_rgb_ids.byte(), b_boxes, colors=(255, 0, 0)
        )
    # stack all -- image, ground truth and prediction
    combined_output: Tensor = torch.cat((image * 255.0, ground_truth_rgb_ids, segmentation_rgb_ids), dim=2)
    # Plot semantic segmentation map
    if path != "" and path is not None:
        save_image(combined_output / 255.0, path=path)
    return combined_output / 255.0


def object_proposals_to_rgb(instance_map: Tensor) -> Tensor:
    """

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].

    Returns:
        instance_map_rgb (Tensor): Instance map as an RGB image of the shape [3, B, H], pix. range is [0, 255].
    """
    # Make a random color shift for each instance
    weight: Tensor = torch.rand(instance_map.amax() + 1, 3)  # type: ignore
    weight[0] = 0.0
    instance_map_rgb: Tensor = 255.0 * torch.embedding(indices=instance_map, weight=weight).permute(2, 0, 1)
    return instance_map_rgb.long()


def plot_object_proposals(instance_map: Tensor, show_ids: bool = False) -> None:
    """Plots a given instance map.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
    """
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map.cpu().detach().clone())
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map.cpu().detach().clone())
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Plot instance map
    plot_image(instance_map_rgb)


def plot_object_proposals_overlay(
    instance_map: Tensor,
    image: Tensor,
    show_ids: bool = False,
    alpha: float = 0.4,
) -> None:
    """Plots a given instance map overlaid with an image.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
        image (Tensor): Image as a tensor of the shape [3, H, W].
    """
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map.cpu().detach().clone())
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map.cpu().detach().clone())
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Overlay segmentation
    overlay: Tensor = _overlay(instance_map_rgb.cpu(), image.cpu() * 255.0, alpha=alpha, ignore_background=True).long()
    # Plot instance map
    plot_image(overlay)


def save_object_proposals_overlay(
    instance_map: Tensor,
    image: Tensor,
    path: str | None,
    show_ids: bool = False,
    alpha: float = 0.4,
) -> None | Tensor:
    """Save a given instance map overlaid with an image.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
        image (Tensor): Image as a tensor of the shape [3, H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map.cpu().detach().clone())
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map.cpu().detach().clone())
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Overlay segmentation
    overlay: Tensor = _overlay(instance_map_rgb.cpu(), image.cpu() * 255.0, alpha=alpha, ignore_background=True)
    # Plot instance map
    if path is not None:
        save_image(overlay / 255.0, path=path)
    return overlay / 255.0


def save_object_proposals(instance_map: Tensor, path: str | None, show_ids: bool = False) -> None | Tensor:
    """Save a given an instance map.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map.cpu().detach().clone())
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map.cpu().detach().clone())
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Plot instance map
    if path is not None:
        save_image(instance_map_rgb / 255.0, path=path)
    return instance_map_rgb / 255.0


def plot_object_proposals_with_noise(instance_map: Tensor, show_ids: bool = False) -> None:
    """Plots a given instance map with noise.

    Notes:
        Noisy pixels are indicated with a value of -1.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
    """
    # Clone tensor
    instance_map_copy = instance_map.cpu().detach().clone()
    # Make noise map
    noise = instance_map_copy == -1
    # Remove noise from instance map
    instance_map_copy[noise] = 0
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map_copy)
    # Add noise
    instance_map_rgb[noise.repeat(3, 1, 1)] = 255
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map_copy)
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Plot instance map
    plot_image(instance_map_rgb)


def save_object_proposals_with_noise(instance_map: Tensor, path: str | None, show_ids: bool = False) -> None | Tensor:
    """Save a given an instance map with noise.

    Notes:
        Noisy pixels are indicated with a value of -1.

    Args:
        instance_map (Tensor): Instance map of the shape [H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Clone tensor
    instance_map_copy = instance_map.cpu().detach().clone()
    # Make noise map
    noise = instance_map_copy == -1
    # Remove noise from instance map
    instance_map_copy[noise] = 0
    # Make RGB instance map
    instance_map_rgb: Tensor = object_proposals_to_rgb(instance_map_copy)
    # Add noise
    instance_map_rgb[noise.repeat(3, 1, 1)] = 255
    # Add IDs is utilized
    if show_ids:
        # Get object centers
        ids, object_centers = _get_object_centers(instance_map_copy)
        # Put IDs into RGB segmentation
        instance_map_rgb = _add_ids(instance_map_rgb, ids, object_centers)
    # Plot instance map
    if path is not None:
        save_image(instance_map_rgb / 255.0, path=path)
    return instance_map_rgb / 255.0


def scene_flow_to_image(scene_flow: Tensor) -> Tensor:
    """Converts a given scene flow to the RGB color encoding as proposed by Hur.

    Args:
        scene_flow (Tensor): Scene flow of the shape [B, 3, H, W] or [3, H, W].

    Returns:
        scene_flow_rgb (Tensor): Color coded scene flow of the shape [B, 3, H, W] or [3, H, W].
    """
    # Permute to [B, H, W, 3] or [H, W, 3]
    scene_flow = scene_flow.permute(0, 2, 3, 1) if scene_flow.ndim == 4 else scene_flow.permute(1, 2, 0)
    # Normalize scene flow
    scene_flow = scene_flow / torch.sqrt(torch.sum(torch.square(scene_flow), dim=-1)).max()
    # Get components
    scene_flow_x: Tensor = scene_flow[..., 0]
    scene_flow_y: Tensor = scene_flow[..., 1]
    scene_flow_z: Tensor = scene_flow[..., 2]
    # Rotating x and y by 45 degrees
    scene_flow_x_rotated: Tensor = scene_flow_x * COS_45 + scene_flow_z * SIN_45
    scene_flow_z_rotated: Tensor = -scene_flow_x * SIN_45 + scene_flow_z * COS_45
    scene_flow_lab: Tensor = torch.stack((scene_flow_y, scene_flow_x_rotated, scene_flow_z_rotated), dim=2)
    # Normalize to lab space
    scene_flow_lab[:, :, 0] = scene_flow_lab[:, :, 0] * 50 + 50
    scene_flow_lab[:, :, 1] = scene_flow_lab[:, :, 1] * 127
    scene_flow_lab[:, :, 2] = scene_flow_lab[:, :, 2] * 127
    # Back to original shape [B, 3, H, W] or [3, H, W]
    scene_flow_lab = scene_flow_lab.permute(0, 3, 1, 2) if scene_flow_lab.ndim == 4 else scene_flow_lab.permute(2, 0, 1)
    # To RGB
    scene_flow_rgb: Tensor = (
        255.0 * torch.flip(lab_to_rgb(scene_flow_lab), dims=(1 if scene_flow_lab.ndim == 4 else 0,))
    ).byte()
    return scene_flow_rgb


def plot_scene_flow(scene_flow: Tensor) -> None:
    """Function to plot scene flow.

    Args:
        scene_flow (Tensor): Scene flow of the shape [3, H, W].
    """
    # Scene flow to RGB
    scene_flow_rgb: Tensor = scene_flow_to_image(scene_flow)
    # Plot scene flow
    plot_image(scene_flow_rgb)


def save_scene_flow(scene_flow: Tensor, path: str | None) -> None | Tensor:
    """Saves a given scene flow as an RGB image.

    Args:
        scene_flow (Tensor): Scene flow of the shape [3, H, W].
        path (str | None): Path to store the image to.

    Returns:
        image (Tensor): Image of the shape [3, H, W].
    """
    # Scene flow to RGB
    scene_flow_rgb: Tensor = scene_flow_to_image(scene_flow)
    # Plot scene flow
    if path is not None:
        save_image(scene_flow_rgb / 255.0, path=path)
    return scene_flow_rgb / 255.0
