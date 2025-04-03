from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Set, Tuple

import torch.nn as nn
import torch.optim
from detectron2.layers import FrozenBatchNorm2d
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.events import EventStorage
from torch import Tensor
from yacs.config import CfgNode

from cups.augmentation import RandomCrop
from cups.data.utils import get_bounding_boxes, instances_to_masks
from cups.model import panoptic_cascade_mask_r_cnn
from cups.pl_model_pseudo import UnsupervisedModel

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class SelfSupervisedModel(UnsupervisedModel):
    """This class implements the self-supervised model for training a Panoptic Cascade Mask R-CNN."""

    def __init__(
        self,
        model: nn.Module,
        num_thing_pseudo_classes: int,
        num_stuff_pseudo_classes: int,
        config: CfgNode,
        thing_classes: Set[int],
        stuff_classes: Set[int],
        copy_paste_augmentation: nn.Module = nn.Identity(),
        photometric_augmentation: nn.Module = nn.Identity(),
        resolution_jitter_augmentation: nn.Module = nn.Identity(),
        class_names: List[str] | None = None,
        classes_mask: List[bool] | None = None,
    ) -> None:
        """Constructor method.

        Args:
            model (nn.Module): Cascade Panoptic Mask R-CNN.
            num_thing_pseudo_classes (int): Number of estimated pseudo thing classes.
            num_thing_pseudo_classes (int): Number of estimated stuff thing classes.
            config (CfgNode): Config object.
            thing_classes (Set[int]): Set of thing classes.
            stuff_classes (Set[int]): Set of stuff classes.
            copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
            photometric_augmentation (nn.Module): Photometric augmentation module.
            resolution_jitter_augmentation (nn.Module): Resolution jitter augmentation module.
            class_to_name (List[str] | None): List containing the name of the semantic classes.
            classes_mask (List[bool] | None): Mask of valid classes in validation set.
        """
        # Call super constructor
        super(SelfSupervisedModel, self).__init__(
            model=model.model,  # type: ignore
            num_thing_pseudo_classes=num_thing_pseudo_classes,
            num_stuff_pseudo_classes=num_stuff_pseudo_classes,
            config=config,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            copy_paste_augmentation=copy_paste_augmentation,
            photometric_augmentation=photometric_augmentation,
            resolution_jitter_augmentation=resolution_jitter_augmentation,
            class_names=class_names,
            classes_mask=classes_mask,
        )
        # Init ema model
        self.teacher_model: nn.Module = copy.deepcopy(model)
        # Init crop module
        if config.DATA.DATASET == "kitti":
            self.crop_module: nn.Module = RandomCrop(resolution_max=368, resolution_min=288, long_side_scale=3.369)
        else:
            self.crop_module = RandomCrop()
        # Set self-training round
        self.round: int = 1

    def forward(self, input: List[Dict[str, Tensor]]) -> List[Dict[str, Any]]:
        """Just wraps the forward pass of the Cascade Panoptic Mask R-CNN.

        Args:
            input (List[Dict[str, Tensor]]): List of inputs (images during inference and images + labels for training)

        Returns:
            output (List[Dict[str, Any]]): Prediction of the model for training the loss.
        """
        output = self.model(input)
        return output  # type: ignore

    def training_step(self, batch: List[Dict[str, Any]], batch_index: int) -> Dict[str, Tensor]:
        """Training step.

        Args:
            batch (List[Dict[str, Any]])): Batch of training data.
            batch_index (int): Batch index.

        Returns:
            loss (Dict[str, Tensor]): Loss value in a dict.
        """
        # Make storage object
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
        # Make pseudo labels
        self.teacher_model.eval()
        with torch.no_grad():
            # Make prediction with TTA
            predictions_tta = self.teacher_model(batch)
            # Generate pseudo labels based on TTA prediction
            pseudo_labels = self.make_pseudo_labels(predictions_tta, batch, self.hparams.stuff_pseudo_classes)
            # Perform copy-paste augmentation
            if self.copy_paste_augmentation is not None:
                pseudo_labels = self.copy_paste_augmentation(pseudo_labels, pseudo_labels)
            # Apply photometric augmentations
            pseudo_labels = self.photometric_augmentation(pseudo_labels)
            # Crop data
            pseudo_labels = self.crop_module(pseudo_labels)
            # Perform resolution jitter
            pseudo_labels = self.resolution_jitter_augmentation(pseudo_labels)
        # Train using self pseudo labels
        loss_dict = self.model(pseudo_labels)
        # Compute sum of losses
        loss: Tensor = sum(loss_dict.values())
        # Log final loss
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        # Log all losses
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)
        # Log media
        if ((self.global_step) % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            # Make inference prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model([{"image": sample["image"]} for sample in pseudo_labels])
            self.model.train()
            self.log_visualizations(pseudo_labels, prediction)
        return {"loss": loss}

    def make_pseudo_labels(
        self,
        predictions: List[Dict[str, Tuple[Tensor, List[Dict[str, Any]]]]],
        images: List[Dict[str, Any]],
        stuff_classes: Tuple[int, ...],
    ) -> List[Dict[str, Any]]:
        """Function generates pseudo labels from TTA predictions.

        Args:
            predictions (List[Dict[str, Tuple[Tensor, List[Dict[str, Any]]]]]): TTA predictions.
            images (List[Tensor]): Corresponding original images.
            stuff_classes (Tuple[int, ...]): Semantic stuff classes.

        Returns:
            pseudo_labels (List[Dict[str, Any]]): Pseudo labels.
        """
        # Make output list
        pseudo_labels = []
        # Iterate over batch size
        for sample, image in zip(predictions, images):
            # Make weights for semantic and instance segmentation
            weight_semantic = (
                torch.ones(
                    sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
                    device=sample["panoptic_seg"][0].device,  # type: ignore
                    dtype=torch.long,
                )
                * 255.0
            )
            weight_instance = torch.zeros(  # type: ignore
                sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
                device=sample["panoptic_seg"][0].device,  # type: ignore
                dtype=torch.long,  # type: ignore
            )
            # Init object semantics
            object_semantics = []
            # Fill weights and get object semantics
            for object in sample["panoptic_seg"][1]:
                if object["isthing"]:
                    weight_semantic[object["id"]] = 0
                    weight_instance[object["id"]] = weight_instance.amax() + 1
                    object_semantics.append(object["category_id"])
                else:
                    weight_semantic[object["id"]] = object["category_id"]
            # Get instance map
            instance = torch.embedding(indices=sample["panoptic_seg"][0], weight=weight_instance.view(-1, 1)).squeeze()
            # Get raw semantic segmentation
            semantic_segmentation_raw = sample["sem_seg"]
            # Get max class scores
            max_class_scores = semantic_segmentation_raw.amax(dim=(1, 2), keepdim=True)  # type: ignore
            # Compute class threshold
            class_threshold = max_class_scores * self.hparams.config.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD
            # Make semantic pseudo label
            semantic_segmentation = torch.where(  # type: ignore
                semantic_segmentation_raw > class_threshold, semantic_segmentation_raw, 0.0  # type: ignore
            )
            semantic_segmentation_pseudo = semantic_segmentation.argmax(dim=0)
            semantic_segmentation_pseudo[semantic_segmentation.sum(dim=0) == 0] = 255
            # Construct output
            if instance.amax() > 0.0:
                # Make instance masks
                instance_masks = instances_to_masks(instance)
                # Make object semantics
                object_semantics_tensor: Tensor = torch.tensor(object_semantics, device=instance.device)
                pseudo_labels.append(
                    {
                        "image": image["image"].squeeze(),
                        "sem_seg": semantic_segmentation_pseudo.long(),
                        "instances": Instances(
                            image_size=tuple(image["image"].shape[1:]),
                            gt_masks=BitMasks(instance_masks),
                            gt_boxes=Boxes(get_bounding_boxes(instance)),
                            gt_classes=object_semantics_tensor,
                        ),
                    }
                )
            else:
                pseudo_labels.append(
                    {
                        "image": image["image"].squeeze(),
                        "sem_seg": semantic_segmentation_pseudo.long(),
                        "instances": Instances(
                            image_size=tuple(image["image"].shape),
                            gt_masks=BitMasks(torch.zeros(0, *image["image"].shape[1:]).bool()),
                            gt_boxes=Boxes(torch.zeros(0, 4).long()),
                            gt_classes=torch.zeros(0).long(),
                        ),
                    }
                )
        return pseudo_labels

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Updates teacher model.

        Args:
            outputs (Any): Unused.
            batch (Any): Unused.
            batch_idx (Any): Unused.
        """
        # Perform EMA update
        for train_parameter, teacher_parameter in zip(  # type: ignore
            self.model.parameters(), self.teacher_model.model.parameters()  # type: ignore
        ):  # type: ignore
            teacher_parameter.data.mul_(0.999).add_((1.0 - 0.999) * train_parameter.data)

    def on_train_epoch_end(self) -> None:
        """Stuff to perform at the end of the epoch."""
        # Just close the storage object
        self.storage.__exit__(None, None, None)  # type: ignore
        # Set storage to Nona
        self.storage = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Builds the models' optimizer.

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer of the model.
        """
        # Get only head parameters (without normalization layers)
        parameters = [
            parameter for name, parameter in self.model.named_parameters() if ("head" in name) and ("norm" not in name)
        ]
        # Init optimizer
        if self.hparams.config.TRAINING.OPTIMIZER == "sgd":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                params=parameters,
                lr=self.hparams.config.TRAINING.SGD.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.SGD.WEIGHT_DECAY,
                momentum=self.hparams.config.TRAINING.SGD.MOMENTUM,
            )
            log.info("SGD used.")
        else:
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY,
                betas=self.hparams.config.TRAINING.ADAMW.BETAS,
            )
            log.info("AdamW used.")
        return optimizer


def build_model_self(
    config: CfgNode,
    thing_pseudo_classes: Tuple[int, ...] | None,
    stuff_pseudo_classes: Tuple[int, ...] | None,
    thing_classes: Set[int],
    stuff_classes: Set[int],
    copy_paste_augmentation: nn.Module = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_weights: Tuple[float, ...] | None = None,
    class_names: List[str] | None = None,
    classes_mask: List[bool] | None = None,
    freeze_bn: bool = True,
) -> SelfSupervisedModel:
    """Function to build the model.

    Args:
        config (CfgNode): Config object.
        thing_pseudo_classes (int): Estimated pseudo thing classes.
        stuff_pseudo_classes (int): Estimated stuff thing classes.
        copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
        class_weights (Tuple[float, ...] | None): Semantic class weight. Default None.
        freeze_bn (bool): If true BN layers are frozen.

    Returns:
        model (UnsupervisedTrainer): Unsupervised trainer.
    """
    # Check parameters
    if thing_pseudo_classes is None or stuff_pseudo_classes is None:
        assert config.MODEL.CHECKPOINT is not None, "If thing stuff split is not given checkpoint needs the be given."
    # Load checkpoint if utilized
    if config.MODEL.CHECKPOINT is not None:
        checkpoint = torch.load(config.MODEL.CHECKPOINT)
        # Case if we have a lighting checkpoint
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            checkpoint = {key.replace("model.", ""): item for key, item in checkpoint.items()}
        # Case if we have a Detectron2 (U2Seg) checkpoint
        else:
            checkpoint = checkpoint["model"]
        # Get number of classes based on model weights
        num_clusters_stuffs: int = int(checkpoint["sem_seg_head.predictor.bias"].shape[0] - 1)
        num_clusters_things: int = int(checkpoint["roi_heads.mask_head.predictor.bias"].shape[0])
    else:
        num_clusters_things = len(thing_pseudo_classes)  # type: ignore
        num_clusters_stuffs = len(stuff_pseudo_classes)  # type: ignore
    # Init model
    model: nn.Module = panoptic_cascade_mask_r_cnn(
        load_dino=config.MODEL.USE_DINO,
        num_clusters_things=num_clusters_things,
        num_clusters_stuffs=num_clusters_stuffs + 1,  # Stuff classes plus single thing classes
        confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
        class_weights=class_weights,
        use_tta=True,
        tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
        use_drop_loss=config.SELF_TRAINING.USE_DROP_LOSS,
        tta_scales=config.MODEL.TTA_SCALES,
        default_size=config.DATA.CROP_RESOLUTION,
        drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
    )
    # Apply checkpoint
    if config.MODEL.CHECKPOINT is not None:
        log.info(f"Checkpoint loaded from {config.MODEL.CHECKPOINT}.")
        model.model.load_state_dict(checkpoint)  # type: ignore
    # Freeze BN layer
    if freeze_bn:
        log.info("Freeze batch norm layers")
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    # Init trainer
    model: SelfSupervisedModel = SelfSupervisedModel(
        model=model,
        num_thing_pseudo_classes=num_clusters_things,
        num_stuff_pseudo_classes=num_clusters_stuffs,
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        copy_paste_augmentation=copy_paste_augmentation,
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=class_names,
        classes_mask=classes_mask,
    )
    return model
