import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, PadTo
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from cups.data.cityscapes import get_class_mapping
from cups.data.utils import load_image
from cups.scene_flow_2_se3.utils import remap_ids

__all__: Tuple[str, ...] = ("MUSESPanopticValidation",)

MUSES_STUFF: Tuple[int, ...] = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23)
MUSES_THINGS: Tuple[int, ...] = (24, 25, 26, 27, 28, 31, 32, 33)


class MUSESPanopticValidation(Dataset):
    """This class implements the KITTI panoptic validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 0.6,
        crop_resolution: Tuple[int, int] = (648, 1152),
        void_id: int = 255,
        num_classes: int = 27,
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
        super(MUSESPanopticValidation, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        self.pad_module_semantic: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=self.void_id)
        # Get class mapping
        self.class_mapping: Tensor = get_class_mapping(num_classes=num_classes, void_id=self.void_id)
        # Init list to store paths
        self.images = []
        self.labels = []
        # Get image paths
        image_directory = os.path.join(root, "frame_camera", "val", "clear", "day")
        for file in sorted(os.listdir(image_directory)):
            if ".png" in file:
                self.images.append(os.path.join(image_directory, file))
        # Get annotation paths
        label_directory_instance = os.path.join(root, "gt_panoptic", "val", "clear", "day")
        for file in sorted(os.listdir(label_directory_instance)):
            if ".png" in file:
                self.labels.append(os.path.join(label_directory_instance, file))
        # Load label file
        with open(os.path.join(root, "gt_panoptic", "gt_panoptic_by_condition", "val_clear_day.json"), "r") as file:
            json_file = json.load(file)
        self.label_file = {sample["image_id"]: sample for sample in json_file["annotations"]}

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
        # Get file name
        image_name = self.images[index].split("/")[-1].replace("_frame_camera.png", "")
        # Load images
        image_0_l: Tensor = load_image(path=image_0_l_path)[None]
        # Load label
        label = torch.from_numpy(_rgb2id(np.array(Image.open(self.labels[index]))).astype(np.int64))
        # Construct semantic and instance label
        label_info = self.label_file[image_name]
        semantic_label_raw = torch.zeros(image_0_l.shape[-2:], dtype=torch.long)
        instance_label = torch.zeros(image_0_l.shape[-2:], dtype=torch.long)
        for id in label.unique():
            if id != 0:
                pass
            for segment in label_info["segments_info"]:
                if (segment["id"] == id) and (segment["iscrowd"] == 0):
                    semantic_label_raw[label == id] = segment["category_id"]
                    if segment["category_id"] in MUSES_THINGS:
                        instance_label[label == id] = 1 + instance_label.amax()
        # Remap semantic classes to N classes plus void class
        semantic_label = torch.embedding(indices=semantic_label_raw.clip(min=0), weight=self.class_mapping).squeeze(
            dim=-1
        )
        # Take care of -1 class
        semantic_label[semantic_label_raw == -1] = self.void_id
        # Resize images and labels
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        semantic_label = F.interpolate(
            semantic_label[None, None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None, None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        # Crop data
        image_0_l = self.crop_module(image_0_l)
        semantic_label = self.crop_module(semantic_label.float()).long()
        instance_label = self.crop_module(instance_label.float()).long()
        # Remap instance IDs to 0, 1, 2, ..., N
        instance_label = remap_ids(instance_label[0, 0])[None, None]
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": image_0_l,
            "semantic_gt": semantic_label,
            "instance_gt": instance_label,
            "image_name": self.images[index].split("/")[-1].replace(".png", ""),  # type: ignore
        }
        return output


def _rgb2id(color: np.ndarray) -> np.ndarray:
    """RGB 2 ID (taken from MUSES SDK).

    Args:
        color (np.ndarray): Color tensor of the shape [H, W, 3].

    Returns:
        id (np.ndarray): ID tensor of the shape [H, W].
    """
    color = color.astype(np.uint32)
    id: np.ndarray = color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return id
