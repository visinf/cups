import os
import sys
from typing import Dict, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, PadTo
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from cups.data.utils import load_image
from cups.scene_flow_2_se3.utils import remap_ids

MOTS_STUFF_CLASSES: Set[int] = {0}
MOTS_THING_CLASSES: Set[int] = {1}


def mots2cs_class_mapping(num_classes: int = 2, void_id: int = 255) -> Tensor:
    """FUnction returns the mapping from raw Cityscapes classes to the supported number of classes (27, 19, and 7).

    Args:
        num_classes (int): Number of classes to be utilized. Default is 27.
        void_id (int): Void id class to ignore parts.

    Returns:
        mapping_weights (Tensor): Mapping weights of the shape [34, 1]
    """
    # Check input
    assert num_classes == 2, f"{num_classes} classes is not supported."
    weights: Tensor = torch.zeros(27)  # * void_id
    weights = weights.to(torch.long)
    if num_classes == 2:
        # weights[1] = 1  # car -- not existent in this dataset
        weights[2] = 1  # person

    # Reshape weights
    weights = weights[..., None]
    return weights


class MOTS(Dataset):
    """This class implements the KITTI panoptic validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: None = None,
        void_id: int = 255,
        num_classes: int = 3,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to the dataset.
            resize_scale (float): Scale factor to be applied to the images and labels.
            crop_resolution (Tuple[int, int]): Crop resolution to be utilized after resizing. Default (368, 1240).
            void_id (int): Void ID to be utilized. Default 255.
            num_classes (int): Number of classes to be utilized. Default 27.
        """
        # Call super constructor
        super(MOTS, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        if crop_resolution is None:
            self.crop_module = None
        else:
            self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)  # type: ignore
            # Init padding module
            self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
            self.pad_module_semantic: nn.Module = PadTo(
                size=crop_resolution, pad_mode="constant", pad_value=self.void_id
            )
        # Get class mapping
        self.class_mapping: Tensor = mots2cs_class_mapping(num_classes=num_classes, void_id=self.void_id)
        # Init list to store paths
        self.images = []
        # Get image paths
        scene_paths = [
            os.path.join(root, "train", "images", d) for d in os.listdir(os.path.join(root, "train", "images"))
        ]
        for scene_path in scene_paths:
            for frame_path in sorted(os.listdir(scene_path)):
                if "jpg" in frame_path:
                    self.images.append(os.path.join(scene_path, frame_path))

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.images)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Method returns an instances of the dataset given its index.

        Args:
            index (int): Index of the sample.

        Returns:
            output (Dict[str, Tensor]): Sample containing the image, semantic, and instance segmentation.
        """
        # Get image left t=0 path
        image_0_l_path = self.images[index]
        # Load images
        image_0_l: Tensor = load_image(path=image_0_l_path)[None]
        # Resize images
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        # Load labels
        label = Image.open(image_0_l_path.replace("images", "instances").replace(".jpg", ".png"))
        label = torch.Tensor(np.array(label, dtype=int)).long()
        semantic_label = label.clone() // 1000
        instance_label = label.clone() % 1000
        del label, image_0_l_path
        semantic_label = self.class_mapping[semantic_label].squeeze().long()
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None][None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None][None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        if self.crop_module is not None:
            # Pad data to ensure min size
            image_0_l = self.pad_module(image_0_l)
            semantic_label = self.pad_module_semantic(semantic_label.float()).long()
            instance_label = self.pad_module(instance_label.float()).long()
            # Crop data
            image_0_l = self.crop_module(image_0_l)
            semantic_label = self.crop_module(semantic_label.float()).long()
            instance_label = self.crop_module(instance_label.float()).long()
        # Ensure we don't have an object mask for ignore regions
        instance_label[semantic_label == self.void_id] = 0
        # Remap instance IDs to 0, 1, 2, ..., N
        instance_label = remap_ids(instance_label[0, 0])[None, None]
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": image_0_l,
            "semantic_gt": semantic_label,
            "instance_gt": instance_label,
            "image_name": self.images[index].split("/")[-2]  # type: ignore
            + self.images[index].split("/")[-1].replace(".jpg", ""),  # type: ignore
        }
        return output
