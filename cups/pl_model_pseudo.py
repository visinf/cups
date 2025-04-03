from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import torch.nn as nn
import torch.optim
import torchvision.transforms as tf
from detectron2.utils.events import EventStorage
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm, rank_zero_only
from torch import Tensor
from torchvision.ops._utils import split_normalization_params
from yacs.config import CfgNode

from cups.metrics import PanopticQualitySemanticMatching
from cups.model import (
    filter_predictions,
    panoptic_cascade_mask_r_cnn,
    prediction_to_label_format,
    prediction_to_standard_format,
)
from cups.visualization import (
    save_image,
    save_object_proposals,
    save_panoptic_segmentation,
    save_panoptic_segmentation_overlay,
    save_semantic_segmentation,
)

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class UnsupervisedModel(LightningModule):
    """This class implements the unsupervised model for training a Panoptic Cascade Mask R-CNN."""

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
        super(UnsupervisedModel, self).__init__()
        # Save model
        self.model: nn.Module = model
        # Save augmentation modules
        self.copy_paste_augmentation: nn.Module = copy_paste_augmentation
        self.photometric_augmentation: nn.Module = photometric_augmentation
        self.resolution_jitter_augmentation: nn.Module = resolution_jitter_augmentation
        # Make thing and stuff classes
        classes = range(num_stuff_pseudo_classes + num_thing_pseudo_classes)
        stuff_pseudo_classes = tuple(classes[:num_stuff_pseudo_classes])
        thing_pseudo_classes = tuple(classes[num_stuff_pseudo_classes:])
        # Omit class names if needed
        if classes_mask is not None:
            class_names = [name for index, name in enumerate(class_names) if classes_mask[index]]  # type: ignore
        # Save hyperparameters
        self.save_hyperparameters(
            {
                "config": config,
                "thing_pseudo_classes": thing_pseudo_classes,
                "stuff_pseudo_classes": stuff_pseudo_classes,
                "class_names": class_names,
                "classes_mask": classes_mask,
            }
        )
        # config
        self.config = config
        # Init storage object
        self.storage: EventStorage | None = None
        # Init metrics
        self.panoptic_quality: PanopticQualitySemanticMatching = PanopticQualitySemanticMatching(
            things=thing_classes,
            stuffs=stuff_classes,
            num_clusters=len(thing_pseudo_classes) + len(stuff_pseudo_classes),
            things_prototype=set(thing_pseudo_classes) if config.VALIDATION.ADHERE_THING_STUFF else None,
            stuffs_prototype=set(stuff_pseudo_classes) if config.VALIDATION.ADHERE_THING_STUFF else None,
            cache_device=config.VALIDATION.CACHE_DEVICE,
            classes_mask=classes_mask,
        )
        # Set parameters for second validation
        self.plot_validation_samples: bool = False
        self.assignments: Tensor | None = None
        # Set parameters for copy-paste augmentation
        self.prediction_temp: Dict | None = None
        # Get color template
        if config.DATA.DATASET == "cityscapes" and config.DATA.NUM_CLASSES == 27:
            self.color_template: str = "cityscapes"
        elif config.DATA.DATASET == "mots":
            self.color_template = "mots"
        else:
            self.color_template = "cityscapes_19"

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
        # Perform copy-paste augmentation
        if self.copy_paste_augmentation is not None:
            if self.global_step < self.hparams.config.AUGMENTATION.NUM_STEPS_STARTUP:
                batch_aug = self.copy_paste_augmentation(batch, deepcopy(batch))
            else:
                if self.prediction_temp is not None:
                    batch_aug = self.copy_paste_augmentation(self.prediction_temp, deepcopy(batch))
                else:
                    batch_aug = deepcopy(batch)
        else:
            batch_aug = deepcopy(batch)
        # Get losses
        loss_dict = self(self.photometric_augmentation(self.resolution_jitter_augmentation(batch_aug)))
        # Compute sum of losses
        loss: Tensor = sum(loss_dict.values())
        # Log final loss
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        # Log all losses
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)
        # Make inference prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([{"image": sample["image"]} for sample in batch])
            self.prediction_temp = prediction_to_label_format(  # type: ignore
                prediction,
                [sample["image"] for sample in batch],
                confidence_threshold=self.hparams.config.AUGMENTATION.CONFIDENCE,
            )
            self.prediction_temp = filter_predictions(self.prediction_temp, batch)  # type: ignore
        self.model.train()
        # Log media
        if ((self.global_step) % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            self.log_visualizations(batch_aug, prediction)
        return {"loss": loss}

    @rank_zero_only
    def log_visualizations(self, batch: List[Dict[str, Any]], prediction: List[Dict[str, Any]]) -> None:
        """Logs visualizations.

        Args:
            batch (List[Dict[str, Any]])): Batch of training data.
            prediction (List[Dict[str, Any]]): Inference predictions.
        """
        # Get object proposals
        object_proposal_pseudo: Tensor = (
            (
                batch[0]["instances"].gt_masks.tensor.cpu().float()
                * torch.arange(1, batch[0]["instances"].gt_masks.tensor.shape[0] + 1).view(-1, 1, 1)
            )
            .sum(dim=0)
            .long()
        )
        # Get raw semantic prediction
        semantic_prediction: Tensor = prediction[0]["sem_seg"].argmax(dim=0)
        # Convert panoptic prediction to standard format
        panoptic_prediction: Tensor = prediction_to_standard_format(
            prediction[0]["panoptic_seg"],
            stuff_classes=self.hparams.stuff_pseudo_classes,
            thing_classes=self.hparams.thing_pseudo_classes,
        )
        # Log image
        self.logger.log_image(key="training_image", images=[save_image(batch[0]["image"].cpu(), path=None)])
        # Log training panoptic prediction
        self.logger.log_image(
            key="training_panoptic_prediction",
            images=[
                save_panoptic_segmentation_overlay(
                    panoptic_prediction.cpu(),
                    batch[0]["image"].cpu(),
                    path=None,
                    dataset="pseudo",
                    bounding_boxes=True,
                )
            ],
        )
        # Log raw semantic prediction
        self.logger.log_image(
            key="training_raw_semantic_predictions",
            images=[
                save_semantic_segmentation(
                    semantic_prediction.cpu(),
                    path=None,
                    dataset="pseudo",
                )
            ],
        )
        # Log pseudo label
        self.logger.log_image(
            key="training_semantic_pseudo_label",
            images=[
                save_semantic_segmentation(
                    batch[0]["sem_seg"].cpu(),
                    path=None,
                    dataset="pseudo",
                )
            ],
        )
        # Log object proposals
        self.logger.log_image(
            key="training_object_proposal_pseudo_label",
            images=[
                save_object_proposals(
                    object_proposal_pseudo,
                    path=None,
                )
            ],
        )

    def validation_step(self, batch: Tuple[List[Dict[str, Tensor]], Tensor, List[str]], batch_index: int) -> None:
        """Validation step.

        Args:
            batch (Tuple[List[Dict[str, Tensor]], Tensor, List[str]]): Batch of training data.
            batch_index (int): Batch index.
        """
        # Get data
        images, panoptic_labels, image_names = batch
        # Semantic segmentation eval resize
        # # Make prediction
        prediction = self(images)
        # Convert to standard panoptic format
        panoptic_predictions: Tensor = torch.stack(
            [
                prediction_to_standard_format(
                    prediction[index]["panoptic_seg"],
                    stuff_classes=self.hparams.stuff_pseudo_classes,
                    thing_classes=self.hparams.thing_pseudo_classes,
                )
                for index in range(len(prediction))
            ],
            dim=0,
        )
        # We plot the samples if we have the assignments and plotting is enabled
        if self.assignments is not None and self.plot_validation_samples:
            # Remap prediction using assignments
            panoptic_predictions_remapped: Tensor = self.panoptic_quality.map_to_target(
                panoptic_predictions, self.assignments
            )
            for index, prediction in enumerate(panoptic_predictions_remapped):
                self.logger.log_image(
                    key="validation_prediction",
                    images=[save_panoptic_segmentation(prediction, path=None, dataset=self.color_template)],
                )
                save_panoptic_segmentation(
                    prediction,
                    path=os.path.join(self.logger.save_dir, image_names[index] + "_prediction.png"),
                    dataset=self.color_template,
                )
        # Update metric
        if self.config.VALIDATION.SEMSEG_CENTER_CROP_SIZE is not None:
            # Apply resize and center crop for semantic segmentation evaluation
            size = self.config.VALIDATION.SEMSEG_CENTER_CROP_SIZE
            resized_sem_pred = tf.CenterCrop(size)(
                tf.Resize(size, interpolation=tf.InterpolationMode.NEAREST)(panoptic_predictions[..., 0])
            )
            resized_sem_label = tf.CenterCrop(size)(
                tf.Resize(size, interpolation=tf.InterpolationMode.NEAREST)(panoptic_labels[..., 0])
            )
            zeros = torch.zeros_like(resized_sem_pred)
            resized_sem_pred = torch.stack((resized_sem_pred, zeros), dim=-1)
            resized_sem_label = torch.stack((resized_sem_label, zeros), dim=-1)
            self.panoptic_quality.update(resized_sem_pred, resized_sem_label)
        else:
            self.panoptic_quality.update(panoptic_predictions, panoptic_labels)

    def on_validation_epoch_end(self) -> None:
        """Accumulate metric after validation."""
        # Compute metrics
        (
            pq,
            sq,
            rq,
            pq_c,
            sq_c,
            rq_c,
            pq_t,
            sq_t,
            rq_t,
            pq_s,
            sq_s,
            rq_s,
            miou,
            acc,
            assignments,
            predictions,
        ) = self.panoptic_quality.compute()
        # Save assignments
        self.assignments = assignments
        # Log metrics
        self.log("pq_val", pq, sync_dist=True, rank_zero_only=True)
        self.log("sq_val", sq, sync_dist=True, rank_zero_only=True)
        self.log("rq_val", rq, sync_dist=True, rank_zero_only=True)
        self.log("pq_t_val", pq_t, sync_dist=True, rank_zero_only=True)
        self.log("sq_t_val", sq_t, sync_dist=True, rank_zero_only=True)
        self.log("rq_t_val", rq_t, sync_dist=True, rank_zero_only=True)
        self.log("pq_s_val", pq_s, sync_dist=True, rank_zero_only=True)
        self.log("sq_s_val", sq_s, sync_dist=True, rank_zero_only=True)
        self.log("rq_s_val", rq_s, sync_dist=True, rank_zero_only=True)
        self.log("miou_val", miou, sync_dist=True, rank_zero_only=True)
        self.log("acc_val", acc, sync_dist=True, rank_zero_only=True)
        # Log class-wise metrics
        for index in range(pq_c.shape[0]):
            if self.hparams.class_names is None:
                self.log(f"val/pq_{str(index).zfill(2)}_val", pq_c[index], sync_dist=True, rank_zero_only=True)
                self.log(f"val/sq_{str(index).zfill(2)}_val", sq_c[index], sync_dist=True, rank_zero_only=True)
                self.log(f"val/rq_{str(index).zfill(2)}_val", rq_c[index], sync_dist=True, rank_zero_only=True)
            else:
                self.log(
                    f"val/pq_{self.hparams.class_names[index]}_val", pq_c[index], sync_dist=True, rank_zero_only=True
                )
                self.log(
                    f"val/sq_{self.hparams.class_names[index]}_val", sq_c[index], sync_dist=True, rank_zero_only=True
                )
                self.log(
                    f"val/rq_{self.hparams.class_names[index]}_val", rq_c[index], sync_dist=True, rank_zero_only=True
                )
        # Log final prediction (only for training and if we not already plot the samples)
        if not self.plot_validation_samples:
            self.logger.log_image(
                key="validation_prediction",
                images=[
                    save_panoptic_segmentation(sample.cpu(), path=None, dataset=self.color_template)
                    for sample in predictions
                ],
            )
        # print results to copy to excel
        pq = pq.item()
        sq = sq.item()
        rq = rq.item()
        pq_t = pq_t.item()
        sq_t = sq_t.item()
        rq_t = rq_t.item()
        pq_s = pq_s.item()
        sq_s = sq_s.item()
        rq_s = rq_s.item()
        acc = acc.item()
        miou = miou.item()
        print("\nPQ, SQ, RQ, PQ_things, SQ_things, RQ_things, PQ_stuffs, SQ_stuffs, RQ_stuffs, Acc, mIoU")
        print("; ".join(map(str, [pq, sq, rq, pq_t, sq_t, rq_t, pq_s, sq_s, rq_s, acc, miou])))
        # Reset metric
        self.panoptic_quality.reset()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Things to do before the optimizer step.

        Args:
            optimizer (torch.optim.Optimizer): Unused.
        """
        # Compute gradient norms
        gradient_norms = grad_norm(self.model, norm_type=2)
        # Log gradient norms
        self.log_dict(gradient_norms)

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
        # Get parameter groups (other parameters and normalization parameters)
        parameter_groups = split_normalization_params(self.model)
        # Init optimizer
        if self.hparams.config.TRAINING.OPTIMIZER == "sgd":
            parameters = [
                {"params": parameter, "weight_decay": weight_decay}
                for parameter, weight_decay in zip(
                    parameter_groups, (0.0, self.hparams.config.TRAINING.SGD.WEIGHT_DECAY)
                )
                if parameter
            ]
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                params=parameters,
                lr=self.hparams.config.TRAINING.SGD.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.SGD.WEIGHT_DECAY,
                momentum=self.hparams.config.TRAINING.SGD.MOMENTUM,
            )
            log.info("SGD used.")
        else:
            parameters = [
                {"params": parameter, "weight_decay": weight_decay}
                for parameter, weight_decay in zip(
                    parameter_groups, (0.0, self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY)
                )
                if parameter
            ]
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.hparams.config.TRAINING.ADAMW.LEARNING_RATE,
                weight_decay=self.hparams.config.TRAINING.ADAMW.WEIGHT_DECAY,
                betas=self.hparams.config.TRAINING.ADAMW.BETAS,
            )
            log.info("AdamW used.")
        return optimizer


def build_model_pseudo(
    config: CfgNode,
    thing_pseudo_classes: Tuple[int, ...] | None,
    stuff_pseudo_classes: Tuple[int, ...] | None,
    thing_classes: Set[int],
    stuff_classes: Set[int],
    copy_paste_augmentation: Optional[nn.Module] = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_weights: Tuple[float, ...] | None = None,
    use_tta: bool = False,
    class_names: List[str] | None = None,
    classes_mask: List[bool] | None = None,
) -> UnsupervisedModel:
    """Function to build the model.

    Args:
        config (CfgNode): Config object.
        thing_pseudo_classes (int): Estimated pseudo thing classes.
        stuff_pseudo_classes (int): Estimated stuff thing classes.
        copy_paste_augmentation (nn.Module): Copy-paste augmentation module.
        class_weights (Tuple[float, ...] | None): Semantic class weight. Default None.
        use_tta (bool): If true TTA is used.
        class_to_name (List[str] | None): List containing the name of the semantic classes.
        classes_mask (List[bool] | None): Mask of valid classes in validation set.

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
            checkpoint = {
                key.replace("model.", ""): item for key, item in checkpoint.items() if "teacher_model." not in key
            }
        # Case if we have a Detectron2 (U2Seg) checkpoint
        else:
            checkpoint = checkpoint["model"]
        # Get number of classes based on model weights
        num_clusters_stuffs: int = int(checkpoint["sem_seg_head.predictor.bias"].shape[0] - 1)
        num_clusters_things: int = int(checkpoint["roi_heads.mask_head.predictor.bias"].shape[0])
        # Log info about checkpoint
        log.info(f"Checkpoint loaded from {config.MODEL.CHECKPOINT}.")
    else:
        num_clusters_things = len(thing_pseudo_classes)  # type: ignore
        num_clusters_stuffs = len(stuff_pseudo_classes)  # type: ignore
    # Init model
    model: nn.Module = panoptic_cascade_mask_r_cnn(
        load_dino=config.MODEL.USE_DINO,
        num_clusters_things=num_clusters_things,
        num_clusters_stuffs=num_clusters_stuffs + 1,  # Stuff classes plus single thing classes
        confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
        tta_detection_threshold=config.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD,
        class_weights=class_weights,
        use_tta=use_tta,
        tta_scales=config.MODEL.TTA_SCALES,
        default_size=config.DATA.CROP_RESOLUTION,
        drop_loss_iou_threshold=config.TRAINING.DROP_LOSS_IOU_THRESHOLD,
        use_drop_loss=config.TRAINING.DROP_LOSS,
    )
    # Apply checkpoint
    if config.MODEL.CHECKPOINT is not None:
        if use_tta:
            model.model.load_state_dict(checkpoint)  # type: ignore
        else:
            model.load_state_dict(checkpoint)
    # Init trainer
    model: UnsupervisedModel = UnsupervisedModel(
        model=model,
        num_thing_pseudo_classes=num_clusters_things,
        num_stuff_pseudo_classes=num_clusters_stuffs,
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        copy_paste_augmentation=copy_paste_augmentation,  # type: ignore
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=class_names,
        classes_mask=classes_mask,
    )
    return model
