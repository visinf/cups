from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import BitMasks, Boxes, Instances
from kornia.augmentation import (
    AugmentationSequential,
    CenterCrop,
    PadTo,
    RandomCrop,
    Resize,
)
from kornia.geometry import scale_intrinsics
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor

sys.path.append(os.getcwd())
from cups.data.utils import (
    CITYSCAPES_TRAINING_FILES,
    get_bounding_boxes,
    instances_to_masks,
    load_image,
)
from cups.scene_flow_2_se3 import remap_ids

__all__: Tuple[str, ...] = (
    "CityscapesPanoptic",
    "CityscapesPanopticValidation",
    "CityscapesStereoVideo",
    "CityscapesStereoVideoPanoptic",
    "CITYSCAPES_CLASSNAMES",
    "CITYSCAPES_CLASSNAMES_19",
    "CITYSCAPES_CLASSNAMES_7",
    "CITYSCAPES_STUFF_CLASSES",
    "CITYSCAPES_THING_CLASSES",
    "CITYSCAPES_VOID_CLASS",
    "collate_function_detectron2_train",
    "collate_function_validation",
)

CITYSCAPES_CLASSNAMES: List[str] = [c.name for c in Cityscapes.classes if (6 < c.id)]
CITYSCAPES_CLASSNAMES_19: List[str] = [c.name for c in Cityscapes.classes if not c.ignore_in_eval]
CITYSCAPES_CLASSNAMES_7: List[str] = list({c.category for c in Cityscapes.classes if not c.ignore_in_eval})
CITYSCAPES_STUFF_CLASSES: Set[int] = {c.id - 7 for c in Cityscapes.classes if (not c.has_instances and 6 < c.id)}
CITYSCAPES_THING_CLASSES: Set[int] = {c.id - 7 for c in Cityscapes.classes if (c.has_instances and 6 < c.id)}
CITYSCAPES_STUFF_CLASSES_19: Set[int] = {
    c.train_id for c in Cityscapes.classes if (not c.has_instances and not c.ignore_in_eval)
}
CITYSCAPES_THING_CLASSES_19: Set[int] = {
    c.train_id for c in Cityscapes.classes if (c.has_instances and not c.ignore_in_eval)
}
CITYSCAPES_STUFF_CLASSES_7: Set[int] = {
    c.category_id - 1 for c in Cityscapes.classes if (not c.has_instances and not c.ignore_in_eval)
}
CITYSCAPES_THING_CLASSES_7: Set[int] = {
    c.category_id - 1 for c in Cityscapes.classes if (c.has_instances and not c.ignore_in_eval)
}
CITYSCAPES_VOID_CLASS: int = 255


class CityscapesPanoptic(Cityscapes):
    """This class implements the Cityscapes dataset (panoptic segmentation)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: Tuple[int, int] = (320, 320),
        void_id: int = 255,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset.
            split (str): Split to be used (train, val, or test). Default: "train".
            resolution (Tuple[int, int]): Resolution to be utilized. Default: 320 x 320.
        """
        # Call super constructor
        super(CityscapesPanoptic, self).__init__(
            root=root,
            split=split,
            target_type=["semantic", "instance"],
            transform=ToTensor(),
            target_transform=lambda labels: (to_tensor(label) for label in labels),
        )
        # set void id
        self.void_id = void_id
        # Init augmentation pipeline
        self.augmentation: nn.Module = AugmentationSequential(
            Resize(size=min(resolution)),
            RandomCrop(size=resolution) if split == "train" else CenterCrop(size=resolution),
            data_keys=["input", "mask", "mask"],
        )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Method loads a single instance of the dataset by a given index.

        Args:
            index (int): Index of dataset sample to be loaded.

        Returns:
            image (Tensor): Image of the shape [3, H, W], pixel range is [0, 1].
            panoptic_segmentation (Tensor): Panoptic segmentation of the shape [H, W, 2]. First dimension contains the
                semantic categories ranging from 0 to 26. 255 corresponds to the void class. The second dimension
                contains instance IDs. Instance IDs are normalized for each image. An ID of 0 means no instance.
        """
        # Load data
        image, (semantic_label, instance_label) = super(CityscapesPanoptic, self).__getitem__(index=index)
        # Perform augmentations
        image, semantic_label, instance_label = self.augmentation(
            image, semantic_label[None].float(), instance_label[None].float()
        )
        # Back to 3D and scale
        image = image[0]
        semantic_label = (255.0 * semantic_label).long()[0]
        instance_label = instance_label[0].long()
        # Remap semantic classes to 27 classes plus void class
        weight: Tensor = torch.ones(34, device=semantic_label.device, dtype=torch.long) * self.void_id
        weight[7:] = torch.arange(start=0, end=27)
        semantic_label = torch.embedding(indices=semantic_label, weight=weight[..., None]).squeeze(dim=0)
        # Omit non-instance IDs, instance IDs are constructed by semantic ID (two digits and >= 24)
        # + instance ID (three digits starting from 000). We shift the instance label by 23999 to prevent building a
        # large weight matrix
        instance_label = torch.where(instance_label >= 24000, instance_label - 23999, torch.zeros_like(instance_label))
        # Remap instance IDs to 0, 1, 2, ..., N
        current_instance_ids: Tensor = torch.unique(instance_label)
        new_instance_ids: Tensor = torch.arange(
            start=0, end=current_instance_ids.shape[0], device=current_instance_ids.device
        )
        weight: Tensor = torch.zeros(  # type: ignore
            current_instance_ids.amax() + 1,
            dtype=torch.long,
            device=instance_label.device,
        )
        weight[current_instance_ids] = new_instance_ids
        instance_label = torch.embedding(indices=instance_label, weight=weight[..., None]).squeeze(dim=0)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image": image,
            "semantic_gt": semantic_label.squeeze(),
            "instance_gt": instance_label.squeeze(),
        }
        return output


class CityscapesStereoVideo(Dataset):
    """This class implements the unsupervised cityscapes stereo video dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_scale: float = 0.625,
        crop_resolution: Tuple[int, int] = (640, 1280),
        temporal_stride: int = 1,
        void_id: int = 255,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset folders (i.e. path to left/rightImg8bit_sequence)
            split (str): Split to be used.
            resize_scale (Tuple[int, int]): Scale to which the images are resized.
            crop_resolution (Tuple[int, int]): Resolution to which the images are cropped after resizing.
            temporal_stride (int): Temporal stride to be utilized (stride one is 17 FPS). Default: 1.
        """
        # Call super constructor
        super(CityscapesStereoVideo, self).__init__()
        self.void_id: int = void_id
        # Check parameters
        assert split in ["train", "test", "val"]
        # Save parameters
        self.resize_scale: float = resize_scale
        self.temporal_stride: int = temporal_stride
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init list to store path to data
        self.sample_path: List[Dict[str, str]] = []
        # Get cities
        cities = os.listdir(os.path.join(root, "leftImg8bit_sequence", split))
        # Remove everything that is not a folder
        cities = [city for city in cities if os.path.isdir(os.path.join(root, "leftImg8bit_sequence", split, city))]
        # Get clips and frames
        clips: Dict = {}
        for city in cities:
            clips[city] = {}
            frames = [
                frame
                for frame in os.listdir(os.path.join(root, "leftImg8bit_sequence", split, city))
                if ".png" in frame
            ]
            for frame in sorted(frames):
                # Get clip ID
                clip_ids = frame.split("_")[1]
                # Save frame
                if clip_ids in clips[city].keys():
                    clips[city][clip_ids].append(frame)
                else:
                    clips[city][clip_ids] = [frame]
        # Construct paths
        for city in clips.keys():
            # Get all calibration files
            calibration_files = os.listdir(os.path.join(root, "camera", split, city))
            for clip_id in clips[city].keys():
                frames = clips[city][clip_id]
                assert len(frames) % 30 == 0
                frames_chunks = [frames[index : index + 30] for index in range(0, len(frames), 30)]
                for chunk_index, frames_chunk in enumerate(frames_chunks):
                    for index in range(len(frames_chunk) - temporal_stride):
                        # Get possible calibration_files
                        calibration_files_possible = [
                            file for file in calibration_files if file.split("_")[1] == clip_id
                        ]
                        # If we only found one file we take this
                        if len(calibration_files_possible) == 1:
                            calibration_file = calibration_files_possible[0]
                        # If not we find the one closest to the target frame
                        else:
                            min_distance = 1e20
                            calibration_file = None
                            target_frame_index = int(frames_chunk[index].split("_")[2])
                            for file in calibration_files_possible:
                                # Get frame index
                                frame_index = int(file.split("_")[2])
                                if abs(frame_index - target_frame_index) < min_distance:
                                    min_distance = abs(frame_index - target_frame_index)
                                    calibration_file = file
                            assert min_distance < 20
                        assert calibration_file is not None
                        self.sample_path.append(
                            {
                                "left_0": os.path.join(root, "leftImg8bit_sequence", split, city, frames_chunk[index]),
                                "right_0": os.path.join(
                                    root,
                                    "rightImg8bit_sequence",
                                    split,
                                    city,
                                    frames_chunk[index].replace("leftImg8bit", "rightImg8bit"),
                                ),
                                "left_1": os.path.join(
                                    root, "leftImg8bit_sequence", split, city, frames_chunk[index + temporal_stride]
                                ),
                                "right_1": os.path.join(
                                    root,
                                    "rightImg8bit_sequence",
                                    split,
                                    city,
                                    frames_chunk[index + temporal_stride].replace("leftImg8bit", "rightImg8bit"),
                                ),
                                "calibration": os.path.join(
                                    os.path.join(root, "camera", split, city, calibration_file)
                                ),
                            }
                        )

    def __len__(self) -> int:
        """Returns length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.sample_path)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (images (0, 1, l, r), valid_pixels, baseline, intrinsics).
        """
        image_0_l_path = self.sample_path[index]["left_0"]
        # Load data
        image_0_l: Tensor = load_image(path=self.sample_path[index]["left_0"])[None]
        image_0_r: Tensor = load_image(path=self.sample_path[index]["right_0"])[None]
        image_1_l: Tensor = load_image(path=self.sample_path[index]["left_1"])[None]
        image_1_r: Tensor = load_image(path=self.sample_path[index]["right_1"])[None]
        # Make map of valid pixels (some pixels are not valid due to rectification)
        valid_pixels: Tensor = torch.ones(1, 1, *image_0_l.shape[-2:], device=image_0_l.device)
        valid_pixels[..., :96] = 0.0
        valid_pixels[..., -96:] = 0.0
        valid_pixels[:, :, : round(valid_pixels.shape[2] * 0.05)] = 0.0
        valid_pixels[:, :, -round(valid_pixels.shape[2] * 0.2) :] = 0.0
        # Resize images
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        image_0_r = F.interpolate(image_0_r, scale_factor=self.resize_scale, mode="bilinear")
        image_1_l = F.interpolate(image_1_l, scale_factor=self.resize_scale, mode="bilinear")
        image_1_r = F.interpolate(image_1_r, scale_factor=self.resize_scale, mode="bilinear")
        valid_pixels = F.interpolate(valid_pixels, scale_factor=self.resize_scale, mode="nearest")
        # Load calibration
        baseline, intrinsics = read_calibration_file(self.sample_path[index]["calibration"])  # type: Tensor, Tensor
        # Scale intrinsics
        intrinsics = scale_intrinsics(intrinsics, scale_factor=self.resize_scale)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": self.crop_module(image_0_l),
            "image_0_r": self.crop_module(image_0_r),
            "image_1_l": self.crop_module(image_1_l),
            "image_1_r": self.crop_module(image_1_r),
            "valid_pixels": self.crop_module(valid_pixels).bool(),
            "baseline": baseline.reshape(1),
            "intrinsics": intrinsics.reshape(1, 3, 3),
            "stuff_classes": torch.tensor(tuple(CITYSCAPES_STUFF_CLASSES), dtype=torch.long),
            "thing_classes": torch.tensor(tuple(CITYSCAPES_THING_CLASSES), dtype=torch.long),
            "void_id": torch.tensor(self.void_id, dtype=torch.long),
            "image_0_l_path": image_0_l_path,  # type: ignore
        }
        return output


class CityscapesSelfTraining(Dataset):
    """This class implements the self-training dataset with Cityscapes data."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_scale: float = 0.625,
        crop_resolution: Tuple[int, int] = (640, 1280),
        only_train_samples: bool = True,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset folders (i.e. path to left/rightImg8bit_sequence)
            split (str): Split to be used.
            resize_scale (Tuple[int, int]): Scale to which the images are resized.
            crop_resolution (Tuple[int, int]): Resolution to which the images are cropped after resizing.
            temporal_stride (int): Temporal stride to be utilized (stride one is 17 FPS). Default: 1.
            only_train_samples (bool): If true only training images (not the full seq.) are used.
        """
        # Call super constructor
        super(CityscapesSelfTraining, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init list to store path to data
        self.sample_path: List[str] = []
        # Get all image paths
        for city in os.listdir(os.path.join(root, "leftImg8bit_sequence", split)):
            # Get all images
            for image in os.listdir(os.path.join(root, "leftImg8bit_sequence", split, city)):
                if ".png" in image:
                    # Only get training samples if utilized
                    if only_train_samples:
                        if image.replace("_leftImg8bit.png", "") in CITYSCAPES_TRAINING_FILES:
                            self.sample_path.append(os.path.join(root, "leftImg8bit_sequence", split, city, image))
                    else:
                        self.sample_path.append(os.path.join(root, "leftImg8bit_sequence", split, city, image))

    def __len__(self) -> int:
        """Returns length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.sample_path)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (images (0, 1, l, r), valid_pixels, baseline, intrinsics).
        """
        # Load data
        image_0_l: Tensor = load_image(path=self.sample_path[index])[None]
        # Resize images
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        # Crop image
        image_0_l = self.crop_module(image_0_l)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image": image_0_l.squeeze(),
            "height": image_0_l.shape[-2],
            "width": image_0_l.shape[-1],
        }
        return output


class CityscapesStereoVideoPanoptic(Dataset):
    """This class implements the unsupervised cityscapes stereo video dataset with panoptic labels."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_scale: float = 0.625,
        crop_resolution: Tuple[int, int] = (640, 1280),
        temporal_stride: int = 1,
        void_id: int = 255,
        num_classes: int = 27,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset folders (i.e. path to left/rightImg8bit_sequence)
            split (str): Split to be used.
            resize_scale (Tuple[int, int]): Scale to which the images are resized.
            crop_resolution (Tuple[int, int]): Resolution to which the images are cropped after resizing.
            temporal_stride (int): Temporal stride to be utilized (stride one is 17 FPS). Default: 2.
            void_id (int): Class of ignore pixels. Default: 255.
        """
        # Call super constructor
        super(CityscapesStereoVideoPanoptic, self).__init__()
        # Check parameters
        assert split in ["train", "test", "val"]
        # Save parameters
        self.resize_scale: float = resize_scale
        self.temporal_stride: int = temporal_stride
        self.void_id: int = void_id
        self.classnames = [c.name for c in Cityscapes.classes if (6 < c.id)]
        self.stuff_classes: Set[int] = {c.id - 7 for c in Cityscapes.classes if (not c.has_instances and 6 < c.id)}
        self.thing_classes: Set[int] = {c.id - 7 for c in Cityscapes.classes if (c.has_instances and 6 < c.id)}
        # Remap semantic classes to 27 classes plus void class
        self.remap_classes = torch.ones(34, dtype=torch.long) * self.void_id
        self.remap_classes[7:] = torch.arange(start=0, end=27)
        # remap classes to 19 class setup
        if num_classes == 19:
            self.classnames = []  # type: ignore
            self.thing_classes = []  # type: ignore
            runnning_idx = 0
            for idx, cls in enumerate(Cityscapes.classes[1:]):
                if cls.ignore_in_eval:
                    self.remap_classes[idx] = self.void_id
                else:
                    self.classnames.append(cls.name)
                    if cls.has_instances:
                        self.thing_classes.append(runnning_idx)  # type: ignore
                    self.remap_classes[idx] = runnning_idx
                    runnning_idx += 1
            self.thing_classes = set(self.thing_classes)
            self.stuff_classes = set(range(0, 19)) - self.thing_classes
        # remap classes to 7 class setup
        elif num_classes == 7:
            self.classnames = []  # type: ignore
            self.thing_classes = []  # type: ignore
            for idx, cls in enumerate(Cityscapes.classes[1:]):
                if cls.ignore_in_eval:
                    self.remap_classes[idx] = self.void_id
                else:
                    if self.classnames == [] or cls.category not in self.classnames:
                        self.classnames.append(cls.category)
                    if cls.has_instances:
                        self.thing_classes.append(cls.category_id)  # type: ignore
                    self.remap_classes[idx] = cls.category_id
            self.thing_classes = set(self.thing_classes)
            self.stuff_classes = set(range(0, 7)) - self.thing_classes
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Get path to labels
        labels_path = os.path.join(root, "gtFine", split)
        # Init list to store paths
        self.images = []
        self.labels = []
        # Iterate over cities
        for city in os.listdir(labels_path):
            if os.path.isdir(os.path.join(labels_path, city)):
                # Get semantic label paths
                semantic_label_paths = [
                    file for file in sorted(os.listdir(os.path.join(labels_path, city))) if "_labelIds.png" in file
                ]
                # Get instance label paths
                instance_label_paths = [
                    file for file in sorted(os.listdir(os.path.join(labels_path, city))) if "instanceIds.png" in file
                ]
                # Add to label paths
                for semantic_label_path, instance_label_path in zip(semantic_label_paths, instance_label_paths):
                    self.labels.append(
                        [
                            os.path.join(labels_path, city, semantic_label_path),
                            os.path.join(labels_path, city, instance_label_path),
                        ]
                    )
        # Get image paths
        for path_pair in self.labels:
            # Get semantic label path
            semantic_label_path = path_pair[0]
            # Make image path
            image_path = os.path.join(
                *semantic_label_path.split("/")[:-1],
                semantic_label_path.split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png"),
            )
            # Save path
            self.images.append("/" + image_path.replace("gtFine", "leftImg8bit_sequence"))

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.images)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        # Get image left t=0 path
        image_0_l_path = self.images[index]
        frame_index = self.images[index].split("_")[-2]
        image_1_l_path = self.images[index].replace(
            "_" + frame_index + "_l", f"_{str(int(frame_index) + self.temporal_stride).zfill(6)}_l"
        )
        image_0_r_path = self.images[index].replace("leftImg8bit", "rightImg8bit")
        image_1_r_path = image_0_r_path.replace(
            "_" + frame_index + "_r", f"_{str(int(frame_index) + self.temporal_stride).zfill(6)}_r"
        )
        # Load images
        image_0_l: Tensor = load_image(path=image_0_l_path)[None]
        image_0_r: Tensor = load_image(path=image_0_r_path)[None]
        image_1_l: Tensor = load_image(path=image_1_l_path)[None]
        image_1_r: Tensor = load_image(path=image_1_r_path)[None]
        # Make map of valid pixels (some pixels are not valid due to rectification)
        valid_pixels: Tensor = torch.ones(1, 1, *image_0_l.shape[-2:], device=image_0_l.device)
        valid_pixels[..., :96] = 0.0
        valid_pixels[:, :, : round(valid_pixels.shape[2] * 0.05)] = 0.0
        valid_pixels[:, :, -round(valid_pixels.shape[2] * 0.2) :] = 0.0
        # Resize images
        image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        image_0_r = F.interpolate(image_0_r, scale_factor=self.resize_scale, mode="bilinear")
        image_1_l = F.interpolate(image_1_l, scale_factor=self.resize_scale, mode="bilinear")
        image_1_r = F.interpolate(image_1_r, scale_factor=self.resize_scale, mode="bilinear")
        valid_pixels = F.interpolate(valid_pixels, scale_factor=self.resize_scale, mode="nearest")
        # Get calibration path
        calibration_path = image_0_l_path.replace("leftImg8bit_sequence", "camera").replace(
            "_leftImg8bit.png", "_camera.json"
        )
        # Load calibration
        baseline, intrinsics = read_calibration_file(calibration_path)  # type: Tensor, Tensor
        # Scale intrinsics
        intrinsics = scale_intrinsics(intrinsics, scale_factor=self.resize_scale)
        # Load semantic and instance label
        semantic_label = to_tensor(Image.open(self.labels[index][0]))
        instance_label = to_tensor(Image.open(self.labels[index][1]))
        # Scale labels
        semantic_label = (255.0 * semantic_label).long()
        instance_label = instance_label.long()
        semantic_label = torch.embedding(indices=semantic_label, weight=self.remap_classes[..., None]).squeeze(dim=0)[
            ..., 0
        ]
        # Omit non-instance IDs, instance IDs are constructed by semantic ID (two digits and >= 24)
        # + instance ID (three digits starting from 000). We shift the instance label by 23999 to prevent building a
        # large weight matrix
        instance_label = torch.where(instance_label >= 24000, instance_label - 23999, torch.zeros_like(instance_label))
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None, None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        # Crop data
        image_0_l = self.crop_module(image_0_l)
        image_0_r = self.crop_module(image_0_r)
        image_1_l = self.crop_module(image_1_l)
        image_1_r = self.crop_module(image_1_r)
        valid_pixels = self.crop_module(valid_pixels).bool()
        semantic_label = self.crop_module(semantic_label.float()).long()
        instance_label = self.crop_module(instance_label.float()).long()
        # Remap instance IDs to 0, 1, 2, ..., N
        instance_label = remap_ids(instance_label[0, 0])[None, None]
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": image_0_l,
            "image_0_r": image_0_r,
            "image_1_l": image_1_l,
            "image_1_r": image_1_r,
            "valid_pixels": valid_pixels,
            "baseline": baseline.reshape(1),
            "intrinsics": intrinsics.reshape(1, 3, 3),
            "semantic_gt": semantic_label,
            "instance_gt": instance_label,
            "stuff_classes": torch.tensor(tuple(CITYSCAPES_STUFF_CLASSES), dtype=torch.long),
            "thing_classes": torch.tensor(tuple(CITYSCAPES_THING_CLASSES), dtype=torch.long),
            "void_id": torch.tensor(self.void_id, dtype=torch.long),
            "image_0_l_path": image_0_l_path,  # type: ignore
        }
        return output


class CityscapesPanopticValidation(Dataset):
    """This class implements the cityscapes validation dataset with panoptic labels."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 0.625,
        crop_resolution: Tuple[int, int] = (640, 1280),
        void_id: int = 255,
        num_classes: int = 27,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset folders (i.e. path to left/rightImg8bit_sequence)
            resize_scale (Tuple[int, int]): Scale to which the images are resized.
            crop_resolution (Tuple[int, int]): Resolution to which the images are cropped after resizing.
            void_id (int): Class of ignore pixels. Default: 255.
            num_classes (int): Number of classes to be utilized. Default 27.
        """
        # Call super constructor
        super(CityscapesPanopticValidation, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Get class mapping
        self.class_mapping: Tensor = get_class_mapping(num_classes=num_classes, void_id=self.void_id)
        # Get path to labels
        labels_path = os.path.join(root, "gtFine", "val")
        # Init list to store paths
        self.images = []
        self.labels = []
        # Iterate over cities
        for city in os.listdir(labels_path):
            if os.path.isdir(os.path.join(labels_path, city)):
                # Get instance label paths
                instance_label_paths = [
                    file for file in sorted(os.listdir(os.path.join(labels_path, city))) if "instanceIds.png" in file
                ]
                # Add to label paths
                self.labels.extend([os.path.join(labels_path, city, path) for path in instance_label_paths])
        # Get image paths
        for path in self.labels:
            # Make image path
            image_path = os.path.join(
                *path.split("/")[:-1],
                path.split("/")[-1].replace("_gtFine_instanceIds.png", "_leftImg8bit.png"),
            )
            # Save path
            self.images.append("/" + image_path.replace("gtFine", "leftImg8bit_sequence"))

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
        # Load semantic and instance label
        semantic_label, instance_label = load_panoptic_cityscapes_labels(
            self.labels[index], self.class_mapping, self.void_id
        )
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None].float(), scale_factor=self.resize_scale, mode="nearest"
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


class CityscapesLabelEfficient(Dataset):
    """This class implements the panoptic pseudo label dataset."""

    def __init__(
        self,
        root: str,
        crop_resolution: Tuple[int, int] = (608, 1104),
        return_detectron2_format: bool = True,
        dataset: str = "cityscapes_5_0",
        ground_truth_scale: float = 0.625,
        void_id: int = 255,
        augmentations: AugmentationSequential | None = None,
    ) -> None:
        """Constructor method.

        Notes:
            return_ground_truth is not supported for Detectron2 format.

        Args:
            root (str): Path to dataset.
            crop_resolution (Tuple[int, int]): Crop target resolution.
            return_detectron2_format (bool): If true we return the Detectron2 training format.
            dataset (str): Dataset to be used. Either cityscapes or kitti. Default: cityscapes.
            ground_truth_scale (float): Scale used to rescale the ground truth labels to image resolution.
            void_id (int): Void ID to be used. Default 255.0
            augmentations (AugmentationSequential): Standard photometric augmentations.
        """
        # Call super constructor
        super(CityscapesLabelEfficient, self).__init__()
        # Save parameters
        self.ground_truth_root: str = root
        self.return_detectron2_format: bool = return_detectron2_format
        self.dataset: str = dataset
        self.ground_truth_scale: float = ground_truth_scale
        self.void_id: int = void_id
        self.augmentations: AugmentationSequential | None = augmentations
        # Init crop module
        self.crop_module: nn.Module = RandomCrop(size=crop_resolution, keepdim=True)
        self.pad_module: nn.Module = (
            PadTo(size=crop_resolution, pad_mode="constant", pad_value=0) if dataset == "kitti" else None
        )
        # Get class mapping
        self.class_mapping: Tensor = get_class_mapping(num_classes=19, void_id=self.void_id)
        # Get paths
        self.image_paths = []
        if dataset != "full_dataset":
            set_name = "CITYSCAPES_LABEL_EFFICIENT_TRAINING_FILES_" + dataset.split("_")[-2]
            set_image_names = getattr(importlib.import_module("cups.data.utils"), set_name)[int(dataset.split("_")[-1])]
            for image_name in set_image_names:
                image_path = os.path.join(
                    root, "leftImg8bit_sequence", "train", image_name.split("_")[0], image_name + ".png"
                )
                self.image_paths.append(image_path)
        else:
            for city in os.listdir(os.path.join(root, "gtFine", "train")):
                for sample in os.listdir(os.path.join(root, "gtFine", "train", city)):
                    if "_gtFine_instanceIds.png" in sample:
                        sample = sample.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")
                        self.image_paths.append(os.path.join(root, "leftImg8bit_sequence", "train", city, sample))
        # thing and stuff classes
        self.things_classes = tuple(CITYSCAPES_THING_CLASSES_19)
        self.stuff_classes = tuple(CITYSCAPES_STUFF_CLASSES_19)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.image_paths)
        return length

    def load_cityscapes_ground_truth(self, image_path: str) -> Tuple[Tensor | None, Tensor | None]:
        """Function loads ground truth Cityscapes label if available.

        Args:
            image_path (str): Image path.

        Returns:
            semantic_gt (Tensor): Semantic segmentation GT of the shape [H, W].
            instance_gt (Tensor): Instance segmentation GT of the shape [H, W].
        """
        # Get unique file name
        file_name: str = image_path.split("/")[-1].replace("_leftImg8bit_semantic.png", "")
        # Get city name
        city: str = file_name.split("_")[0]
        # Get split based on city
        split: str = "train"
        train_cities: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", "train"))
        if city not in train_cities:
            split = "val"
            val_cities: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", "val"))
            if city not in val_cities:
                split = "test"
        # Get all labels for the city
        label_paths: List[str] = os.listdir(os.path.join(self.ground_truth_root, "gtFine", split, city))
        # Check if label is available
        if not any([file_name in file for file in label_paths]):
            return None, None
        # Make file paths
        semantic_label_path: str = os.path.join(
            self.ground_truth_root, "gtFine", split, city, file_name + "_gtFine_labelIds.png"
        )
        instance_label_path: str = os.path.join(
            self.ground_truth_root, "gtFine", split, city, file_name + "_gtFine_instanceIds.png"
        )
        # Load semantic and instance label
        semantic_label = to_tensor(Image.open(semantic_label_path))
        instance_label = to_tensor(Image.open(instance_label_path))
        # Scale labels
        semantic_label = (255.0 * semantic_label).long()
        instance_label = instance_label.long()
        # Remap semantic classes to 27 classes plus void class
        weight: Tensor = torch.ones(34, dtype=torch.long) * self.void_id
        weight[7:] = torch.arange(start=0, end=27)
        semantic_label = torch.embedding(indices=semantic_label, weight=weight[..., None]).squeeze(dim=0)[..., 0]
        # Omit non-instance IDs, instance IDs are constructed by semantic ID (two digits and >= 24)
        # + instance ID (three digits starting from 000). We shift the instance label by 23999 to prevent building a
        # large weight matrix
        instance_label = torch.where(instance_label > 24000, instance_label - 23999, 0)
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None, None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None].float(), scale_factor=self.ground_truth_scale, mode="nearest"
        ).long()
        return semantic_label, instance_label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (optionally in Detectron2 format).
        """
        # Load data
        image: Tensor = load_image(self.image_paths[index])
        # Scale image
        image = F.interpolate(image[None], scale_factor=self.ground_truth_scale, mode="bilinear")[0]
        # Crop both image and pseudo label
        if self.pad_module is not None:
            image = self.pad_module(image[None])[0]
        image = self.crop_module(image[None])

        # Load GT
        semantic_gt, instance_gt = load_panoptic_cityscapes_labels(
            self.image_paths[index]
            .replace("leftImg8bit_sequence", "gtFine")
            .replace("_leftImg8bit.png", "_gtFine_instanceIds.png"),
            self.class_mapping,
            self.void_id,
        )
        # Crop GT
        semantic_gt = self.crop_module(semantic_gt.float(), params=self.crop_module._params).long().squeeze()
        instance_gt = self.crop_module(instance_gt.float(), params=self.crop_module._params).long().squeeze()
        # Remap instance IDs to 0, 1, 2, ..., N
        instance_gt = remap_ids(instance_gt)
        # Perform augmentations
        if self.augmentations is not None:
            image, semantic_gt, instance_gt, valid_mask = self.augmentations(
                image, semantic_gt.float(), instance_gt.float(), torch.ones_like(instance_gt).float()
            )
            # Clip image to valid pixel range
            image = image.clip(min=0.0, max=1.0)
            # Ignore invalid regions
            semantic_gt[valid_mask != 1.0] = self.void_id
            # Ensure long tensor again
            semantic_gt = semantic_gt.long().squeeze()
            instance_gt = instance_gt.long()
            # Remap ids
            instance_gt = remap_ids(instance_gt[0, 0]).squeeze()
        # return in standard format
        if not self.return_detectron2_format:
            output: Dict[str, Tensor] = {
                "image_0_l": image.squeeze(),
                "semantic_gt": semantic_gt,
                "instance_gt": instance_gt,
                "image_name": self.image_paths[index],  # type: ignore
            }
            return output

        # Get stuff semantic segmentation
        weight = torch.ones(256, dtype=torch.long) * self.void_id
        weight[torch.tensor(self.things_classes)] = 0
        weight[torch.tensor(self.stuff_classes)] = torch.arange(len(self.stuff_classes), dtype=torch.long) + 1
        semantic_gt_stuff: Tensor = torch.embedding(weight.reshape(-1, 1), semantic_gt).squeeze()
        # Get binary instance masks
        instance_masks: Tensor = instances_to_masks(instance_gt)
        # Get semantic classes of objects
        object_semantics: Tensor = (semantic_gt * instance_masks).amax(dim=(-1, -2))
        # Catch case for which the object semantic is set to ignore
        valid_masks = object_semantics != self.void_id
        instance_masks = instance_masks[valid_masks]
        object_semantics = object_semantics[valid_masks]
        # Remap object semantics
        weight = torch.zeros(len(self.stuff_classes) + len(self.things_classes), dtype=torch.long)
        weight[torch.tensor(self.things_classes)] = torch.arange(len(self.things_classes), dtype=torch.long)
        object_semantics = torch.embedding(weight.reshape(-1, 1), object_semantics)[..., 0]
        # Remap instance pseudo label since we need it for getting the bounding boxes
        instance_gt[~instance_masks.any(dim=0)] = 0
        # Make Detectron2 dict
        output = {
            "image": image.squeeze(),
            "sem_seg": semantic_gt_stuff,
            "instances": Instances(
                image_size=tuple(image.shape[1:]),
                gt_masks=BitMasks(instance_masks),
                gt_boxes=Boxes(get_bounding_boxes(instance_gt)),
                gt_classes=object_semantics,
            ),
        }
        return output


def load_panoptic_cityscapes_labels(path: str, class_mapping: Tensor, void_id: int = 255) -> Tuple[Tensor, Tensor]:
    """Loads Cityscapes panoptic labels.

    Args:
        path (str): Path to data.
        class_mapping (str): Semantic class mapping of the shape [34, 1]
        void_id (int): Void ID. Default 255.

    Returns:
        semantic_label (Tensor): Semantic label of the shape [1, H, W]
        instance_label (Tensor): Instance label of the shape [1, H, W]
    """
    # Load semantic and instance label
    instance_map = torch.from_numpy(np.array(Image.open(path)).astype(np.int64)).long()
    # Get semantic segmentation
    semantic_label_raw = torch.where(instance_map > 1000, instance_map // 1000, instance_map)
    # Remap semantic classes to N classes plus void class
    semantic_label = torch.embedding(indices=semantic_label_raw.clip(min=0), weight=class_mapping).squeeze(dim=-1)
    # Take care of -1 class
    semantic_label[semantic_label_raw == -1] = void_id
    # Get instance map
    instance_label = torch.where(instance_map >= 24000, instance_map - (24000 - 1), 0)
    # Remap ids
    instance_label = remap_ids(instance_label)
    return semantic_label[None], instance_label[None]


def read_calibration_file(path: str) -> Tuple[Tensor, Tensor]:
    """Function reads a Cityscapes calibration file and returns the baseline and intrinsics.

    Args:
        path (str): Path to calibration file.

    Returns:
        baseline (Tensor): Baseline tensor of the shape [1].
        intrinsics (Tensor): Intrinsics as a tensor of the shape [3, 3].
    """
    # Load file
    with open(path) as file:
        # Load json format
        calibration = json.load(file)
        # Make baseline tensor
        baseline: Tensor = torch.tensor(calibration["extrinsic"]["baseline"])
        # Make intrinsic tensor
        intrinsics: Tensor = torch.tensor(
            [
                [calibration["intrinsic"]["fx"], 0.0, calibration["intrinsic"]["u0"]],
                [0.0, calibration["intrinsic"]["fy"], calibration["intrinsic"]["v0"]],
                [0.0, 0.0, 1.0],
            ],
        )
    return baseline, intrinsics


def collate_function_validation(data: List[Dict[str, Tensor]]) -> Tuple[List[Dict[str, Tensor]], Tensor, List[str]]:
    """Collate function for validation.

    Args:
        data (List[Dict[str, Tensor]]): Batch of data as a list in standard format.

    Returns:
        images (List[Dict[str, Tensor]]): List of image in dict (Detrctron2 validation input).
        panoptic_label (Tensor): Panoptic label as a tensor of the shape [B, H, W, 2].
        image_names (List[str]): List of image names.
    """
    # Make list of image dict
    images: List[Dict[str, Tensor]] = [{"image": sample["image_0_l"].squeeze()} for sample in data]
    # Get panoptic label
    panoptic_label: Tensor = torch.stack(
        (
            torch.stack([sample["semantic_gt"] for sample in data], dim=0),
            torch.stack([sample["instance_gt"] for sample in data], dim=0),
        ),
        dim=-1,
    )
    # Omit empty dimensions
    panoptic_label = panoptic_label.squeeze()
    # Add empty batch dimension if needed
    if panoptic_label.ndim == 3:
        panoptic_label = panoptic_label[None]
    # Get image names
    image_names: List[str] = [sample["image_name"] for sample in data]  # type: ignore
    return images, panoptic_label, image_names


def mixed_dtype_collate_fn(batch):
    # Assume that all items in the batch are dictionaries
    batch_dict = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            # Stack all tensors along the first dimension
            batch_dict[key] = torch.stack([item[key] for item in batch], dim=0).squeeze()
        else:
            # Handle non-tensor values (e.g., strings)
            batch_dict[key] = [item[key] for item in batch]

    return batch_dict


def collate_function_detectron2_train(data: List[Dict[str, Tensor]]) -> List[Dict]:
    # Init output list
    output = []
    # Iterate over samples
    for sample in data:
        # Get image shape
        image_size = sample["image_0_l"].shape[-2:]
        # Get semantic segmentation
        semantic_segmentation = sample["semantic_gt"][0, 0]
        # Make semantic segmentation of stuff + other (things)
        weights = torch.arange(sample["stuff_classes"].shape[0] + sample["thing_classes"].shape[0] + 1)
        weights[sample["thing_classes"]] = sample["stuff_classes"].shape[0]
        weights[sample["void_id"]] = sample["stuff_classes"].shape[0] + 1
        semantic_segmentation_stuff = torch.embedding(weights.view(-1, 1), semantic_segmentation)[..., 0]
        # Get binary instance masks
        instance_masks = instances_to_masks(sample["instance_gt"][0, 0])
        # Get semantic classes of objects
        object_semantics = (semantic_segmentation[None] * instance_masks).amax(dim=(-1, -2))
        # Map object semantics to [0, ..., N]
        object_semantics = object_semantics - sample["thing_classes"].amin()
        # Add dict to output
        output.append(
            {
                "image": sample["image_0_l"][0],
                "sem_seg": semantic_segmentation_stuff,
                "instances": Instances(
                    image_size=tuple(image_size),
                    gt_masks=BitMasks(instances_to_masks(sample["instance_gt"][0, 0])),
                    gt_boxes=Boxes(sample["bounding_boxes"]),
                    gt_classes=object_semantics,
                ),
            }
        )
    return output


def get_class_mapping(num_classes: int = 27, void_id: int = 255) -> Tensor:
    """FUnction returns the mapping from raw Cityscapes classes to the supported number of classes (27, 19, and 7).

    Args:
        num_classes (int): Number of classes to be utilized. Default is 27.
        void_id (int): Void id class to ignore parts.

    Returns:
        mapping_weights (Tensor): Mapping weights of the shape [34, 1]
    """
    # Check input
    assert num_classes in (27, 19, 7), f"{num_classes} classes is not supported."
    # Init weights
    weights: Tensor = torch.ones(34, dtype=torch.long) * void_id
    # 27 classes case
    if num_classes == 27:
        weights[7:] = torch.arange(start=0, end=27)
    # 19 classes case
    elif num_classes == 19:
        for cityscapes_class in Cityscapes.classes:
            if cityscapes_class.id >= 0:
                if cityscapes_class.ignore_in_eval:
                    weights[cityscapes_class.id] = void_id
                else:
                    weights[cityscapes_class.id] = cityscapes_class.train_id
    # 7 parent classes
    else:
        for cityscapes_class in Cityscapes.classes:
            if cityscapes_class.id >= 0:
                if cityscapes_class.category == "void":
                    weights[cityscapes_class.id] = void_id
                else:
                    weights[cityscapes_class.id] = cityscapes_class.category_id - 1
    # Reshape weights to [34, 1]
    weights = weights[..., None]
    return weights
