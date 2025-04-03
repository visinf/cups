from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures import BitMasks, Boxes, Instances
from torch import Tensor

from cups.data.utils import get_bounding_boxes, instances_to_masks
from cups.model.modeling import PanopticFPNWithTTA
from cups.model.modeling.meta_arch import build_model

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

PARAMETERS_IGNORE: Tuple[str, ...] = (
    "roi_heads.box_predictor.0.cls_score.weight",
    "roi_heads.box_predictor.0.cls_score.bias",
    "roi_heads.box_predictor.1.cls_score.weight",
    "roi_heads.box_predictor.1.cls_score.bias",
    "roi_heads.box_predictor.2.cls_score.weight",
    "roi_heads.box_predictor.2.cls_score.bias",
    "roi_heads.mask_head.predictor.weight",
    "roi_heads.mask_head.predictor.bias",
    "sem_seg_head.predictor.weight",
    "sem_seg_head.predictor.bias",
)


def panoptic_cascade_mask_r_cnn_from_checkpoint(
    path: str,
    device: str | torch.device,
    confidence_threshold: float = 0.5,
) -> Tuple[nn.Module, int, int]:
    """Builds a Panoptic Cascade Mask R-CNN Detectron2 model with drop loss and DINO ResNet-50 from checkpoint.

    Args:
        path (str): Path to checkpoint.

    Returns:
        model (nn.Module): Panoptic Cascade Mask R-CNN Detectron2 model.
        num_clusters_things (int): Number of thing pseudo classes.
        num_clusters_stuffs (int): Number of stuff pseudo classes.
    """
    # Load checkpoint
    # Get checkpoint
    checkpoint = torch.load(path, map_location=device)
    checkpoint = checkpoint["state_dict"]
    checkpoint = {key.replace("model.", ""): item for key, item in checkpoint.items() if "teacher_model." not in key}
    # Get number of classes based on model weights
    num_clusters_stuffs: int = int(checkpoint["sem_seg_head.predictor.bias"].shape[0] - 1)
    num_clusters_things: int = int(checkpoint["roi_heads.mask_head.predictor.bias"].shape[0])
    # Load config
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("cups/model/Panoptic-Cascade-Mask-R-CNN.yaml")
    cfg.MODEL.DEVICE = device
    if cfg.MODEL.DEVICE == "cpu" and cfg.MODEL.RESNETS.NORM == "SyncBN":
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_clusters_things
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_clusters_stuffs + 1
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.freeze()
    # Log config
    print(cfg)
    # Init model
    model = build_model(cfg)
    # Load checkpoint
    model.load_state_dict(checkpoint)
    return model, num_clusters_things, num_clusters_stuffs


def panoptic_cascade_mask_r_cnn(
    load_dino: bool = True,
    num_clusters_things: int = 300,
    num_clusters_stuffs: int = 100,
    confidence_threshold: float = 0.4,
    class_weights: Tuple[float, ...] | None = None,
    use_tta: bool = False,
    tta_detection_threshold: float = 0.5,
    tta_scales: Tuple[float, ...] = (0.75, 1.0, 1.25, 1.5),
    default_size: Tuple[int, int] = (512, 1024),
    use_drop_loss: bool = True,
    drop_loss_iou_threshold: float = 0.2,
) -> nn.Module:
    """Builds a Panoptic Cascade Mask R-CNN Detectron2 model with drop loss and DINO ResNet-50.

    Notes:
        load_u2seg_checkpoint=True will overwrite DINO backbone weights.
        u2seg_training_dataset and u2seg_num_clusters_things only used if load_u2seg_checkpoint=True.

    Args:
        load_dino (bool): If true DINO ResNet-50 backbone will be used.
        num_clusters_things (int): Number of things clusters to be utilized.
        num_clusters_stuffs (int): Number of stuffs clusters to be utilized.
        confidence_threshold (float): Confidence threshold of object proposals. Default 0.0 (no threshold).
        class_weights (Tuple[float, ...] | None): Semantic class weight. Default None.
        use_tta (bool): If true TTA model is returned.
        tta_detection_threshold (float): Detection threshold used in TTA.
        use_drop_loss (bool): If true drop loss is used.
        drop_loss_iou_threshold (float): IoU threshold used in drop loss.

    Returns:
        model (nn.Module): Panoptic Cascade Mask R-CNN Detectron2 model.
    """
    # Load config
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("cups/model/Panoptic-Cascade-Mask-R-CNN.yaml")
    if cfg.MODEL.DEVICE == "cpu" and cfg.MODEL.RESNETS.NORM == "SyncBN":
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_clusters_things
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_clusters_stuffs
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT = class_weights
    cfg.TEST.INSTANCE_SCORE_THRESH = tta_detection_threshold
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = use_drop_loss
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = drop_loss_iou_threshold
    if use_tta:
        cfg.TEST.AUG.MIN_SIZES = tuple(int(default_size[0] * scale) for scale in tta_scales)
    cfg.freeze()
    # Log config
    print(cfg)
    # Init model
    model = build_model(cfg)
    # Load checkpoint
    if load_dino:
        log.info("DINO CutLer loaded")
        DetectionCheckpointer(model).load(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(), "backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
            )
        )
    # Make TTA model
    if use_tta:
        model = PanopticFPNWithTTA(cfg, model)
        log.info("TTA model is used.")
    return model  # type: ignore


def prediction_to_standard_format(
    prediction: Tuple[Tensor, List[Dict[str, Any]]],
    stuff_classes: Tuple[int, ...],
    thing_classes: Tuple[int, ...],
) -> Tensor:
    """Converts a Detectron2 panoptic prediction to the standard format.

    Args:
        prediction (Tuple[Tensor, List[Dict[str, Any]]]): Panoptic prediction as a Dict in the Detectron2 format.
        stuff_classes (Tuple[int, ...]): Original stuff classes
        thing_classes (Tuple[int, ...]): Original thing classes

    Returns:
        output (Tensor): Panoptic prediction with the shape [H, W, 2], including the semantic and instance prediction.
    """
    # Get device
    device: torch.device = prediction[0].device
    # Get instance ids
    instance_ids = torch.tensor(
        [pred["id"] for pred in prediction[1] if pred["isthing"]], dtype=torch.long, device=device
    )
    # Map instance map
    weight = torch.zeros(prediction[0].amax() + 1, device=device, dtype=torch.long)  # type: ignore
    weight[instance_ids] = torch.arange(instance_ids.shape[0], device=device) + 1
    instance_prediction: Tensor = torch.embedding(weight.reshape(-1, 1), prediction[0])[..., 0]
    # Get all ids
    ids = torch.tensor([pred["id"] for pred in prediction[1]], dtype=torch.long, device=device)
    # Get original classes
    original_classes = []
    for pred in prediction[1]:
        if not pred["isthing"]:
            original_classes.append(stuff_classes[pred["category_id"] - 1])
        else:
            original_classes.append(_remap_ignore(thing_classes[pred["category_id"]], max(thing_classes)))
    original_classes = torch.tensor(original_classes, device=device, dtype=torch.long).view(-1)
    # Semantic map
    weight = torch.ones(prediction[0].amax() + 1, device=device, dtype=torch.long) * 255  # type: ignore
    weight[ids] = original_classes
    semantic_prediction: Tensor = torch.embedding(weight.reshape(-1, 1), prediction[0])[..., 0]
    # Make final panoptic prediction
    panoptic_prediction: Tensor = torch.stack((semantic_prediction, instance_prediction), dim=-1)
    return panoptic_prediction


def _remap_ignore(class_id: int, max_class: int) -> int:
    """Circumvents 255 class id by mapping the 255 class to max class +1.

    Args:
        class_id (int): Class ID.
        max_class (int): Max class ID.

    Returns:
        corrected_id (int): Corrected ID.
    """
    if class_id == 255:
        return max_class + 1
    return class_id


def prediction_to_class_agnostic_detection(prediction: Tuple[Tensor, List[Dict[str, Any]]]) -> List[Dict[str, Tensor]]:
    """Converts a Detectron2 panoptic prediction to class agnostic object detection.

    Args:
        prediction (Tuple[Tensor, List[Dict[str, Any]]]): Panoptic prediction as a Dict in the Detectron2 format.

    Returns:
        output (Tensor): Class agnostic object detection including bounding box, score, and placeholder class.
    """
    # Init output list
    output = []
    # Iterate over all predictions
    for sample in prediction:
        # Get instances
        instances = sample["instances"]  # type: ignore
        # Build output
        output.append(
            dict(
                boxes=instances.pred_boxes.tensor,
                scores=instances.scores,
                labels=torch.zeros_like(instances.scores).long(),
            ),
        )
    return output


def filter_predictions(
    prediction: List[Dict[str, Any]],
    batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Omits object labels from prediction.

    Args:
        prediction (List[Dict[str, Any]]): Prediction in label format.
        batch (List[Dict[str, Any]]): Batch of images and labels.

    Returns:
        prediction (List[Dict[str, Any]]): Filtered predictions in label format.
    """
    # Iterate over batch dimension
    for batch_index in range(len(prediction)):
        # We only need to omit labels in case we have an object prediction or label at all
        if (len(prediction[batch_index]["instances"]) > 0) and (len(batch[batch_index]["instances"]) > 0):
            # Get bounding boxes
            bounding_boxes = prediction[batch_index]["instances"].gt_boxes.tensor
            # Get label bounding boxes
            bounding_boxes_label = batch[batch_index]["instances"].gt_boxes.tensor
            # Compute mIoU between bounding boxes
            iou = iou_bbox(bounding_boxes_label, bounding_boxes).amax(dim=0)
            # Compute no-overlapping objects
            keep_objects = iou < 0.5
            # Omit overlapping objects
            prediction[batch_index]["instances"] = Instances(
                image_size=prediction[batch_index]["instances"].image_size,
                gt_masks=prediction[batch_index]["instances"].gt_masks[keep_objects],
                gt_boxes=prediction[batch_index]["instances"].gt_boxes[keep_objects],
                gt_classes=prediction[batch_index]["instances"].gt_classes[keep_objects],
            )
    return prediction


def iou_bbox(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """Compute the IoU of the cartesian product of two sets of boxes.

    Each box in each set shall be (x1, y1, x2, y2).

    Args:
        boxes_1: a tensor of bounding boxes in :math:`(B1, 4)`.
        boxes_2: a tensor of bounding boxes in :math:`(B2, 4)`.

    Returns:
        a tensor in dimensions :math:`(B1, B2)`, representing the
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2.
    """
    # Find intersection
    lower_bounds = torch.max(boxes_1[:, :2].unsqueeze(1), boxes_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(boxes_1[:, 2:].unsqueeze(1), boxes_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)
    # Find areas of each box in both sets
    areas_set_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])  # (n1)
    areas_set_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])  # (n2)
    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)


def prediction_to_label_format(
    prediction: List[Dict[str, Tuple[Tensor, List[Dict[str, Any]]]]],
    images: List[Tensor],
    confidence_threshold: float = -1.0,
) -> List[Dict[str, Any]]:
    """Function converts a models' prediction into the label format.

    Args:
        prediction (Tuple[Tensor, List[Dict[str, Any]]]): Panoptic prediction as a Dict in the Detectron2 format.

    Returns:
        output (Dict[str, Any]): Detectron2 label format.
    """
    # Make output list
    output = []
    # Iterate over batch size
    for sample, image in zip(prediction, images):
        # Make weights for semantic and instance segmentation
        weight_semantic = (
            torch.ones(
                sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
                device=sample["panoptic_seg"][0].device,  # type: ignore
                dtype=torch.long,  # type: ignore
            )
            * 255.0
        )
        weight_instance = torch.zeros(
            sample["panoptic_seg"][0].amax().item() + 1,  # type: ignore
            device=sample["panoptic_seg"][0].device,  # type: ignore
            dtype=torch.long,  # type: ignore
        )
        # Init object semantics
        object_semantics = []
        # Fill weights and get object semantics
        for object in sample["panoptic_seg"][1]:
            if object["isthing"]:
                if object["score"] > confidence_threshold:
                    weight_semantic[object["id"]] = 0
                    weight_instance[object["id"]] = weight_instance.amax() + 1
                    object_semantics.append(object["category_id"])
            else:
                weight_semantic[object["id"]] = object["category_id"]
        # Get semantic segmentation map
        semantic = torch.embedding(indices=sample["panoptic_seg"][0], weight=weight_semantic.view(-1, 1)).squeeze()
        # Get instance map
        instance = torch.embedding(indices=sample["panoptic_seg"][0], weight=weight_instance.view(-1, 1)).squeeze()
        # Construct output
        if instance.amax() > 0.0:
            # Make instance masks
            instance_masks = instances_to_masks(instance)  # type: ignore
            # Make object semantics
            object_semantics = torch.tensor(object_semantics, device=instance.device)  # type: ignore
            output.append(
                {
                    "image": image.squeeze(),
                    "sem_seg": semantic,
                    "instances": Instances(
                        image_size=tuple(image.shape[1:]),
                        gt_masks=BitMasks(instance_masks),
                        gt_boxes=Boxes(get_bounding_boxes(instance)),
                        gt_classes=object_semantics,
                    ),
                }
            )
        else:
            output.append(
                {
                    "image": image.squeeze(),
                    "sem_seg": semantic,
                    "instances": Instances(
                        image_size=tuple(image.shape),
                        gt_masks=BitMasks(torch.zeros(0, *image.shape[1:]).bool()),
                        gt_boxes=Boxes(torch.zeros(0, 4).long()),
                        gt_classes=torch.zeros(0).long(),
                    ),
                }
            )
    return output
