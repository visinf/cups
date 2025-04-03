from __future__ import annotations

import random
from copy import deepcopy
from typing import Dict, List, Tuple

import kornia.augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import BitMasks, Boxes, Instances
from kornia.augmentation import (
    AugmentationSequential,
    ColorJitter,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
)
from kornia.constants import DataKey
from numpy.random import uniform
from torch import Tensor

from cups.data.utils import get_bounding_boxes

__all__: Tuple[str, ...] = (
    "get_pseudo_label_augmentations",
    "get_label_efficient_augmentations",
    "CopyPasteAugmentation",
    "PhotometricAugmentations",
    "ResolutionJitter",
)


class CopyPasteAugmentation(nn.Module):
    """This class implements batch-wise copy-paste augmentation."""

    def __init__(
        self,
        thing_class: int,
        max_num_pasted_objects: int = 7,
        scale_range: Tuple[float, float] = (0.25, 1.5),
        use_random_horizontal_flipping: bool = True,
        min_bounding_box_size: Tuple[int, int] = (32, 32),
        copy_scale_border: float = 0.0,
    ) -> None:
        """Constructor method.

        Args:
            thing_class (int): Thing class number in semantic segmentation (Detectron2 format).
            max_num_pasted_objects (int): Number of pasted objects per images.
            scale_range (Tuple[float, float]): Scale range used to randomly rescale objects.
            use_random_horizontal_flipping (bool): If true random horizontal flipping is used.
            min_bounding_box_size (Tuple[int, int]): Min box size to be considered to be pasted.
            copy_scale_border (float): Scale factor of object border regions (within bounding box). Default 0.3.
        """
        # Call super constructor
        super(CopyPasteAugmentation, self).__init__()
        # Save parameters
        self.thing_class: int = thing_class
        self.max_num_pasted_objects: int = max_num_pasted_objects
        self.scale_range: Tuple[float, float] = scale_range
        self.use_random_horizontal_flipping: bool = use_random_horizontal_flipping
        self.min_bounding_box_size: Tuple[int, int] = min_bounding_box_size
        self.copy_scale_border: float = copy_scale_border

    @torch.no_grad()
    def forward(
        self,
        batch_source: List[Dict[str, Tensor | Instances]],
        batch_target: List[Dict[str, Tensor | Instances]],
    ) -> List[Dict[str, Tensor | Instances]]:
        # Get cropped instance masks, image crops, and classes
        instance_masks: List[Tensor] = []
        image_crops: List[Tensor] = []
        classes_list: List[Tensor] = []
        for sample in batch_source:
            if sample["instances"].gt_classes.shape[0] > 0:  # type: ignore
                classes_list.append(sample["instances"].gt_classes)  # type: ignore
                bounding_boxes = sample["instances"].gt_boxes.tensor.long()  # type: ignore
                for index, bounding_box in enumerate(bounding_boxes):
                    # Check that bounding box is large enough
                    if ((bounding_box[3] - bounding_box[1]) > self.min_bounding_box_size[0]) and (
                        (bounding_box[2] - bounding_box[0]) > self.min_bounding_box_size[1]
                    ):
                        image_crops.append(
                            sample["image"][:, bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]]
                        )
                        instance_masks.append(
                            sample["instances"].gt_masks.tensor[  # type: ignore
                                index,
                                bounding_box[1] : bounding_box[3],
                                bounding_box[0] : bounding_box[2],
                            ]
                        )
        # Catch case where we don't have any object proposals
        if len(instance_masks) == 0:
            return batch_target
        # Make copy of target batch
        batch_target_copy = deepcopy(batch_target)
        # Classes to tensor
        classes: Tensor = torch.cat(classes_list, dim=0)
        # Init output list
        output: List[Dict[str, Tensor | Instances]] = []
        # Paste objects into images
        for sample in batch_target:
            # Get data
            image_original: Tensor = sample["image"]
            semantic_segmentation_original: Tensor = sample["sem_seg"]
            instance_masks_original: Tensor = sample["instances"].gt_masks.tensor  # type: ignore
            bounding_boxes_original: Tensor = sample["instances"].gt_boxes.tensor  # type: ignore
            classes_original: Tensor = sample["instances"].gt_classes  # type: ignore
            # Put empty tensors to GPU
            if instance_masks_original.shape[0] == 0:
                instance_masks_original = instance_masks_original.to(image_original.device)
                bounding_boxes_original = bounding_boxes_original.to(image_original.device)
                classes_original = classes_original.to(image_original.device)
            # Get random number of pasted objects
            num_pasted_objects = torch.randint(low=1, high=self.max_num_pasted_objects + 1, size=(1,)).item()
            # Paste a single mask
            for _ in range(num_pasted_objects):  # type: ignore
                # Sample a random object
                random_index: int = int(torch.randint(0, len(instance_masks), size=(1,)).item())
                # Get instance mask and image crop
                instance_mask = instance_masks[random_index]
                image_crop = image_crops[random_index]
                # Augment object
                scale_factor: float = uniform(self.scale_range[0], self.scale_range[1])
                instance_mask = F.interpolate(
                    instance_mask[None, None].float(), scale_factor=scale_factor, mode="nearest"
                )[0, 0].bool()
                image_crop = F.interpolate(image_crop[None], scale_factor=scale_factor, mode="bilinear")[0]
                if self.use_random_horizontal_flipping and torch.rand(1).item() > 0.5:
                    instance_mask = instance_mask.flip(dims=(-1,))
                    image_crop = image_crop.flip(dims=(-1,))
                # In case the object proposal is too large we need to crop it
                original_shape = image_original.shape[1:]
                image_crop = image_crop[..., : original_shape[0], : original_shape[1]]
                instance_mask = instance_mask[: original_shape[0], : original_shape[1]]
                # Make copy instance mask
                instance_mask_copy = instance_mask.clone().float()
                instance_mask_copy[instance_mask_copy == 0] = self.copy_scale_border
                # Add class to original classes
                classes_original = torch.cat((classes_original, classes[random_index].view(1)), dim=0)
                # Randomly pad instance mask and image
                crop_shape = instance_mask.shape
                h_pad = original_shape[0] - crop_shape[0]
                w_pad = original_shape[1] - crop_shape[1]
                w_1_pad = int(uniform(0, w_pad))
                w_2_pad = w_pad - w_1_pad
                h_1_pad = int(uniform(0, h_pad))
                h_2_pad = h_pad - h_1_pad
                instance_padded: Tensor = F.pad(
                    instance_mask.float(), pad=(w_1_pad, w_2_pad, h_1_pad, h_2_pad), value=0
                ).bool()
                instance_copy_padded: Tensor = F.pad(
                    instance_mask_copy.float(), pad=(w_1_pad, w_2_pad, h_1_pad, h_2_pad), value=0
                )
                image_padded: Tensor = F.pad(image_crop, pad=(w_1_pad, w_2_pad, h_1_pad, h_2_pad), value=0)
                # Get bounding box of padded instance
                bounding_box = get_bounding_boxes(instance_padded)
                # Add to original bounding boxes
                bounding_boxes_original = torch.cat((bounding_boxes_original, bounding_box), dim=0)
                # Add object to original image
                image_original = instance_copy_padded * image_padded + (1.0 - instance_copy_padded) * image_original
                # Set existing masks to zero in the area of the pasted mask
                instance_masks_original[instance_padded[None].repeat(instance_masks_original.shape[0], 1, 1)] = False
                instance_masks_original = torch.cat((instance_masks_original, instance_padded[None]), dim=0)
                # Set semantic segmentation to thing class in the area of the object
                semantic_segmentation_original[instance_padded] = 0
            # We need to catch the case the pasted masks fully occlude another mask
            valid_objects = instance_masks_original.any(dim=-1).any(dim=-1)
            # If we have less bounding boxes than masks we just return the original target batch
            if valid_objects.shape[0] != bounding_boxes_original.shape[0]:
                return batch_target_copy
            instance_masks_original = instance_masks_original[valid_objects]
            bounding_boxes_original = bounding_boxes_original[valid_objects]
            classes_original = classes_original[valid_objects]
            # Make output
            output.append(
                {
                    "image": image_original,
                    "sem_seg": semantic_segmentation_original,
                    "instances": Instances(
                        image_size=tuple(image_original.shape[1:]),
                        gt_masks=BitMasks(instance_masks_original),
                        gt_boxes=Boxes(bounding_boxes_original),
                        gt_classes=classes_original,
                    ),
                }
            )
        return output


class RandomCrop(nn.Module):
    """This class implements random crop augmentation given a Detectron2 input batch."""

    def __init__(self, resolution_max: int = 1024, resolution_min: int = 512, long_side_scale: float = 2.0) -> None:
        # Call super constructor
        super(RandomCrop, self).__init__()
        # Save parameters
        self.resolution_max: int = resolution_max
        self.resolution_min: int = resolution_min
        self.long_side_scale: float = long_side_scale

    def forward(self, batch: List[Dict[str, Tensor | Instances]]) -> List[Dict[str, Tensor | Instances]]:
        """

        Args:
            batch (List[Dict[str, Tensor | Instances]]): Detectron2 panoptic input batch.

        Returns:
            batch_augmented (List[Dict[str, Tensor | Instances]]): Augmented input batch.
        """
        # Make copy of batch
        batch_augmented = deepcopy(batch)
        # Sample random resolution
        resolution = random.randint(self.resolution_min, self.resolution_max)
        # Init cropping module
        crop_module = kornia.augmentation.RandomCrop((resolution, round(self.long_side_scale * resolution)))
        # Iterate over all samples
        for index_batch in range(len(batch_augmented)):
            # Augment image
            batch_augmented[index_batch]["image"] = crop_module(batch_augmented[index_batch]["image"][None])[0]
            # Augment semantic segmentation
            batch_augmented[index_batch]["sem_seg"] = crop_module(
                batch_augmented[index_batch]["sem_seg"][None, None].float(), params=crop_module._params
            )[0, 0].long()
            # Augment instance maps
            if batch_augmented[index_batch]["instances"].gt_masks.tensor.shape[0] != 0:  # type: ignore
                instance_gt: Tensor = crop_module(
                    batch_augmented[index_batch]["instances"].gt_masks.tensor[None].float(),  # type: ignore
                    params=crop_module._params,  # type: ignore
                )[0]
                # Compute valid masks
                valid_masks = instance_gt.sum(dim=(1, 2)) > 4
                instance_gt = instance_gt[valid_masks]
                # Compute new bounding boxes
                if instance_gt.shape[0] > 0:
                    bounding_boxes_gt: Tensor = torch.cat([get_bounding_boxes(mask) for mask in instance_gt], dim=0)
                else:
                    bounding_boxes_gt = torch.zeros(0, 4, device=valid_masks.device)
                # Get gt classes
                classes_gt = batch_augmented[index_batch]["instances"].gt_classes[valid_masks]  # type: ignore
            else:
                instance_gt = torch.zeros(
                    0,
                    *batch_augmented[index_batch]["image"].shape[1:],
                    device=batch_augmented[index_batch]["image"].device,
                )
                bounding_boxes_gt = torch.zeros(0, 4, device=batch_augmented[index_batch]["image"].device)
                classes_gt = torch.zeros(0, device=batch_augmented[index_batch]["image"].device)
            # Save augmented instances
            batch_augmented[index_batch]["instances"] = Instances(
                image_size=tuple(instance_gt.shape[1:]),
                gt_masks=BitMasks(instance_gt.bool()),
                gt_boxes=Boxes(bounding_boxes_gt.long()),
                gt_classes=classes_gt.long(),
            )
        return batch_augmented


class ResolutionJitter(nn.Module):
    """This class implements resolution jitter augmentation giving a Detectron2 input batch."""

    def __init__(
        self,
        scales: Tuple[float, ...] | None = (0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1),
        resolutions: Tuple[Tuple[int, int], ...] | None = None,
    ) -> None:
        """Constructor method.

        Args:
            scales (Tuple[float, ...]): Scales to be randomly sampled for augmentation.
        """
        # Call super constructor
        super(ResolutionJitter, self).__init__()
        # Save parameters
        self.scales: Tuple[float, ...] | None = scales
        self.resolutions: Tuple[Tuple[int, int], ...] | None = resolutions

    def forward(self, batch: List[Dict[str, Tensor | Instances]]) -> List[Dict[str, Tensor | Instances]]:
        """

        Args:
            batch (List[Dict[str, Tensor | Instances]]): Detectron2 panoptic input batch.

        Returns:
            batch_augmented (List[Dict[str, Tensor | Instances]]): Augmented input batch.
        """
        # Make copy of batch
        batch_augmented = deepcopy(batch)
        # Get random scale
        if self.resolutions is None:
            scale: float | None = self.scales[torch.randint(high=len(self.scales), size=(1,)).item()]  # type: ignore
            resolution: Tuple[int, int] | None = None
        else:
            scale = None
            resolution = random.choice(self.resolutions)
        # Iterate over all samples
        for index_batch in range(len(batch_augmented)):
            # Augment image
            batch_augmented[index_batch]["image"] = F.interpolate(
                batch_augmented[index_batch]["image"][None], scale_factor=scale, size=resolution, mode="bilinear"
            )[0]
            # Augment semantic segmentation
            batch_augmented[index_batch]["sem_seg"] = F.interpolate(
                batch_augmented[index_batch]["sem_seg"][None, None].float(),
                scale_factor=scale,
                size=resolution,
                mode="nearest",
            )[0, 0].long()
            # Augment instance maps
            if batch_augmented[index_batch]["instances"].gt_masks.tensor.shape[0] != 0:  # type: ignore
                instance_gt: Tensor = F.interpolate(
                    batch_augmented[index_batch]["instances"].gt_masks.tensor[None].float(),  # type: ignore
                    scale_factor=scale,
                    size=resolution,
                    mode="nearest",
                )[0]
                # Compute valid masks
                valid_masks = instance_gt.sum(dim=(1, 2)) > 4
                instance_gt = instance_gt[valid_masks]
                # Compute new bounding boxes
                if instance_gt.shape[0] > 0:
                    bounding_boxes_gt: Tensor = torch.cat([get_bounding_boxes(mask) for mask in instance_gt], dim=0)
                else:
                    bounding_boxes_gt = torch.zeros(0, 4, device=valid_masks.device)
                # Get gt classes
                classes_gt = batch_augmented[index_batch]["instances"].gt_classes[valid_masks]  # type: ignore
            else:
                instance_gt = torch.zeros(
                    0,
                    *batch_augmented[index_batch]["image"].shape[1:],
                    device=batch_augmented[index_batch]["image"].device,
                )
                bounding_boxes_gt = torch.zeros(0, 4, device=batch_augmented[index_batch]["image"].device)
                classes_gt = torch.zeros(0, device=batch_augmented[index_batch]["image"].device)
            # Save augmented instances
            batch_augmented[index_batch]["instances"] = Instances(
                image_size=tuple(instance_gt.shape[1:]),
                gt_masks=BitMasks(instance_gt.bool()),
                gt_boxes=Boxes(bounding_boxes_gt.long()),
                gt_classes=classes_gt.long(),
            )
        return batch_augmented


class PhotometricAugmentations(nn.Module):
    """This class implements photometric augmentations."""

    def __init__(self) -> None:
        """Constructor method."""
        # Call super constructor
        super(PhotometricAugmentations, self).__init__()
        # Init augmentations
        self.augmentations: AugmentationSequential = AugmentationSequential(
            RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=1.0, keepdim=True),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5, keepdim=True),
            RandomGrayscale(p=0.2, keepdim=True),
            keepdim=True,
        )

    def forward(self, batch: List[Dict[str, Tensor | Instances]]) -> List[Dict[str, Tensor | Instances]]:
        """

        Args:
            batch (List[Dict[str, Tensor | Instances]]): Detectron2 panoptic input batch.

        Returns:
            batch_augmented (List[Dict[str, Tensor | Instances]]): Augmented input batch.
        """
        # Make copy of batch
        batch_augmented = deepcopy(batch)
        # Apply augmentations
        for index in range(len(batch_augmented)):
            batch_augmented[index]["image"] = self.augmentations(batch_augmented[index]["image"]).squeeze()
        return batch_augmented


def get_label_efficient_augmentations(resolution: Tuple[int, int] = (608, 1104)) -> AugmentationSequential:
    """Builds the augmentation pipeline used during label-efficient training.

    Args:
        resolution (Tuple[int, int]): Target resolution to be used in training.

    Returns:
        augmentations (AugmentationSequential): Augmentation pipeline.
    """
    augmentations: AugmentationSequential = AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(
            size=resolution,
            scale=(0.5, 2.0),
            ratio=(resolution[0] / resolution[1], resolution[0] / resolution[1]),
            p=1.0,
        ),
        data_keys=[DataKey.INPUT, DataKey.MASK, DataKey.MASK, DataKey.MASK],
    )
    return augmentations


def get_pseudo_label_augmentations(resolution: Tuple[int, int] = (608, 1104)) -> AugmentationSequential:
    """Builds the augmentation pipeline used during pseudo label training.

    Args:
        resolution (Tuple[int, int]): Target resolution to be used in training.

    Returns:
        augmentations (AugmentationSequential): Augmentation pipeline.
    """
    augmentations: AugmentationSequential = AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(
            size=resolution,
            scale=(0.7, 1.0),
            ratio=(resolution[0] / resolution[1], resolution[0] / resolution[1]),
            p=1.0,
        ),
        data_keys=[DataKey.INPUT, DataKey.MASK, DataKey.MASK, DataKey.MASK],
    )
    return augmentations
