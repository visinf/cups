import math
import random
from typing import Dict, Optional, Tuple

import kornia.enhance
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rtpt import RTPT
from torch import Tensor

__all__: Tuple[str, ...] = (
    "normalize",
    "normalize_min_max_m1_1",
    "resize",
    "RTPTCallback",
)


def normalize(
    images: Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tensor:
    """Normalize images with a given mean and std.

    Args:
        images (Tensor): Images (w/ pixel range [0, 1]) of the shape [B, 3, H, W].
        mean (Tuple[float, float, float]): Mean for each channel. Default ImageNet mean.
        std (Tuple[float, float, float]): Std for each channel. Default ImageNet std.

    Returns:
        images_normalized (Tensor): Normalized images (mean & std) of the shape [B, 3, H, W].
    """
    # Get dtype and device
    dtype: torch.dtype = images.dtype
    device: torch.device = images.device
    # Normalize images
    images_normalized: Tensor = kornia.enhance.normalize(
        images,
        mean=torch.tensor(mean, dtype=dtype, device=device),
        std=torch.tensor(std, dtype=dtype, device=device),
    )
    return images_normalized


def denormalize(
    images: Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tensor:
    """Denormalize images with a given mean and std.

    Args:
        images (Tensor): Images (w/ pixel range [0, 1]) of the shape [B, 3, H, W].
        mean (Tuple[float, float, float]): Mean for each channel. Default ImageNet mean.
        std (Tuple[float, float, float]): Std for each channel. Default ImageNet std.

    Returns:
        images_normalized (Tensor): Normalized images (mean & std) of the shape [B, 3, H, W].
    """
    # Get dtype and device
    dtype: torch.dtype = images.dtype
    device: torch.device = images.device
    # Normalize images
    images_normalized: Tensor = kornia.enhance.denormalize(
        images,
        mean=torch.tensor(mean, dtype=dtype, device=device),
        std=torch.tensor(std, dtype=dtype, device=device),
    )
    return images_normalized


def normalize_min_max_m1_1(images: Tensor) -> Tensor:
    """Normalizes images to the pixel range of [-1, 1] (used by RAFT).

    Args:
        images (Tensor): Images (w/ pixel range [0, 1]) of the shape [B, 3, H, W].

    Returns:
        images_normalized (Tensor): Normalized images [-1, 1] of the shape [B, 3, H, W].
    """
    # Normalize images
    images_normalized: Tensor = 2.0 * images - 1.0
    return images_normalized


def resize(tensor: Tensor, size: Optional[Tuple[int, int]] = None, scale_factor: Optional[float] = None) -> Tensor:
    """Resizes a 4D tensor using bicubic interpolation (w/ antialias if downsampling is performed)

    Notes:
        Either size or scale_factor must be given!

    Args:
        tensor (Tensor): Input tensor of the shape [B, C, H, W].
        size (Optional[Tuple[int, int]]): Size to be achieved as a tuple of two integer values.
        scale_factor (Optional[float]): Scale factor to be used.

    Returns:
        tensor_resized (Tensor): Resized tensor of the shape [B, C, H_new, W_new].
    """
    # Check parameters
    if (size is None) and (scale_factor is None):
        raise ValueError("Size or scale factor must be given.")
    # Perform interpolation
    tensor_resized: Tensor = F.interpolate(
        tensor, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False, antialias=True
    )
    return tensor_resized


class RTPTCallback(pl.Callback):
    """This warps the RTPT as a callback.

    Just shows infos about the training in process name.
    """

    def __init__(
        self,
        name_initials: str = "CR&OH",
        experiment_name: str = "UPS",
    ) -> None:
        """Constructor method.

        Args:
            name_initials (str): Name initials to show in process name.
            experiment_name (str): Experiment name to show in process name.
        """
        # Call super constructor
        super(RTPTCallback, self).__init__()
        # Save parameters
        self.name_initials: str = name_initials
        self.experiment_name: str = experiment_name

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Update data module at the end of an epoch.

        Args:
            trainer (pl.Trainer): Trainer module.
            pl_module (pl.LightningModule): Module.
        """
        # Init RTPT
        if trainer.current_epoch == 0:
            epochs: int = trainer.max_epochs  # type: ignore
            self.rtpt = RTPT(
                name_initials=self.name_initials,
                experiment_name=self.experiment_name,
                max_iterations=max(1, math.floor(epochs)),
            )
            self.rtpt.start()
            self.rtpt.step()
        # Update RTPT
        elif trainer.current_epoch > 0:
            self.rtpt.step()


def align_semantic_to_instance(
    semantic_prediction: Tensor, instance_prediction: Tensor, ignore_class: int = 255
) -> Dict[str, Tensor]:
    """Aligns semantic predictions to instance predictions."""

    # get dims from instance prediction
    num_ins, h, w = instance_prediction.max(), instance_prediction.shape[1], instance_prediction.shape[2]
    # check if instances in image
    if num_ins > 0:
        # map to consecutive ids
        mapping = torch.zeros(num_ins + 1, device=instance_prediction.device).long()  # type: ignore
        mapping[instance_prediction.unique()] = torch.arange(
            len(instance_prediction.unique()), device=instance_prediction.device
        ).long()  # type: ignore
        instance_prediction = mapping[instance_prediction]
        num_ins = instance_prediction.max()
        # instance prediction as one hot encoding
        one_hot_inst_preds = torch.zeros(num_ins + 1, h, w, device=instance_prediction.device)  # type: ignore
        one_hot_inst_preds = one_hot_inst_preds.scatter(0, instance_prediction, 1)[1:]
        sem_pred: Tensor = semantic_prediction
        # map class 0 to ignore class for
        sem_pred[sem_pred == 0] = ignore_class
        # repeat semantic prediction for each instance
        sem_pred_per_inst = torch.broadcast_to(sem_pred, one_hot_inst_preds.shape)
        # get majority semantic class for each instance
        sem_pred_per_inst = (one_hot_inst_preds * sem_pred_per_inst).view(num_ins, -1)
        majority_semcls_per_inst = torch.stack([torch.mode(i[i != 0])[0] for i in (sem_pred_per_inst)], dim=0)
        # map majority class to instance
        maped_ins = one_hot_inst_preds * majority_semcls_per_inst.view(num_ins, 1, 1)  # type: ignore
        # collapse instance predictions dimension from one hot
        maped_ins = torch.sum(maped_ins, dim=0)
        # fill non instance pixels with semantic prediction
        sem_pred = torch.where((maped_ins == 0), sem_pred.squeeze(), maped_ins)
        # map semantic class 0 back
        sem_pred[sem_pred == ignore_class] = 0
    else:
        # no instance -- nothing to do here
        sem_pred = semantic_prediction.squeeze()
        majority_semcls_per_inst = torch.Tensor([])

    return {"aligned_semantics": sem_pred, "majority_cls_per_inst": majority_semcls_per_inst}


def set_seed_everywhere(seed: int = 0):
    """Sets the seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def scale_intrinsics(
    intrinsics: Tensor,
    target_resolution: Tuple[int, int],
    original_resolution: Tuple[int, int] = (1280, 1920),
) -> Tensor:
    """Scales intrinsics to new resolution.

    Args:
        intrinsics (Tensor): Camera intrinsics of the shape [3, 3].
        target_resolution (Tuple[int, int]): Target resolution first height and then width.
        original_resolution (Tuple[int, int]): Original resolution first height and then width.

    Returns:
        intrinsics_scaled (Tensor): Scale camera intrinsics of the shape [3, 3].
    """
    # Clone tensor
    intrinsics_scaled: Tensor = intrinsics.clone()
    # Scale intrinsics in-place
    intrinsics_scaled[0] *= target_resolution[1] / original_resolution[1]
    intrinsics_scaled[1] *= target_resolution[0] / original_resolution[0]
    return intrinsics_scaled


def crop_object_proposal(object_proposal: Tensor, offset: int = 128) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
    """Crops an object proposal and returns the crop.

    Args:
        object_proposal (Tensor): Object proposal (soft value) of the shape [H, W].
        offset (int): Offset around the object proposal.

    Returns:
        object_proposal_cropped (Tensor): Cropped object proposal of the shape [H_new, W_new].
        bounding_box (Tensor): Bounding box of the object proposal [4] with offset.
        original_shape (Tuple[int, ...]): Original image size.
    """
    # Get original shape
    original_shape: Tuple[int, ...] = tuple(object_proposal.shape)
    # Threshold object proposal map
    object_proposal_hard: Tensor = object_proposal > 0.5
    # Get bounding box
    coordinates: Tensor = torch.argwhere(object_proposal_hard)
    # Compute bounding box parameters
    x_min = coordinates[:, 1].amin().item()
    y_min = coordinates[:, 0].amin().item()
    x_max = coordinates[:, 1].amax().item()
    y_max = coordinates[:, 0].amax().item()
    # Make bounding box tensor and save
    bounding_box: Tensor = torch.tensor(
        (
            max(x_min - offset, 0),
            max(y_min - offset, 0),
            min(x_max + offset, original_shape[1]),
            max(y_max + offset, original_shape[0]),
        ),
        dtype=torch.long,
        device=object_proposal.device,
    ).reshape(4)
    # Make crop
    object_proposal_cropped: Tensor = object_proposal[
        bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]
    ]
    return object_proposal_cropped, bounding_box, original_shape


def reverse_crop_object_proposal(
    object_proposal_cropped: Tensor,
    bounding_box: Tensor,
    original_shape: Tuple[int, int],
) -> Tensor:
    """Reverses the object proposal crop.

    Args:
        object_proposal_cropped (Tensor): Cropped object proposal of the shape [H, W].
        bounding_box (Tensor): Bounding box of crop with shape [4].
        original_shape (Tuple[int, int]): Original image size.

    Returns:
        object_proposal (Tensor): Object proposal at the original resolution [H_orig, W_orig].
    """
    # Make object proposal
    object_proposal: Tensor = torch.zeros(
        original_shape, dtype=object_proposal_cropped.dtype, device=object_proposal_cropped.device
    )
    # Cropped object proposal to full object proposal
    object_proposal[bounding_box[1] : bounding_box[3], bounding_box[0] : bounding_box[2]] = object_proposal_cropped
    return object_proposal
