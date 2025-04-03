from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics.detection import PanopticQuality as PanopticQualityTM

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class PanopticQualitySemanticMatching(PanopticQualityTM):
    """This class implements the Panoptic Quality metric using stego matching.

    Compute metric:
        1. In the update method we cache the matching cost matrix
        2. We also cache the predictions and labels at the end of the update method
        3. When calling compute matching is performed and cluster IDs are mapped to category IDs
        4. After matching we make a pass over all cached pred. and label pairs and compute PQ as well as the class mIoU
    """

    def __init__(
        self,
        things: Set[int],
        stuffs: Set[int],
        num_clusters: int,
        things_prototype: Set[int] | None = None,
        stuffs_prototype: Set[int] | None = None,
        classes_mask: List[bool] | None = None,
        allow_unknown_preds_category: bool = True,
        perform_one_to_one_matching: bool = False,
        cache_device: torch.device | str | None = None,
        disable_matching: bool = False,
    ) -> None:
        """Constructor method.

        Notes:
            This implementation sets void to 255.
            If cache_device the data is kept on its original device.


        Args:
            things (Set[int]): Class IDs of thing classes.
            stuffs (Set[int]): Class IDs of stuff classes.
            things_prototype (Set[int] | None): Prototype IDs of thing c. If None not considered in matching
            stuffs_prototype (Set[int] | None): Prototype IDs of stuff c. If None not considered in matching
            classes_mask (List[bool] | None): Optional class mask to ignore classes that are not present in the dataset.
            num_clusters (int): Number of semantic clusters. Should be equal or larger than the actual semantic classes.
            allow_unknown_preds_category (bool): Set to true if exception should be raised if unknown class is found.
            perform_one_to_one_matching (bool): If true the metric only performs one to one matching for overclust.
            cache_device (torch.device | str): Device to cache data on. You better use CPU here! Default: "cpu".
            disable_matching (bool): If true matching is disabled.
        """
        # Call super constructor
        super(PanopticQualitySemanticMatching, self).__init__(
            things=things,
            stuffs=stuffs,
            allow_unknown_preds_category=allow_unknown_preds_category,
            sync_on_compute=True,
        )
        # Check parameters
        assert num_clusters >= len(stuffs.union(things)), "Number of clusters must match or exceed number of classes."
        assert not (
            (things_prototype is None) ^ (stuffs_prototype is None)
        ), "Either both stuff and thing prototype classes must be None or both need to be given."
        # if things_prototype is not None:
        #     assert len(things_prototype) >= len(
        #         things
        #     ), "We need at least the same number of more things prototypes than things classes."
        #     assert len(stuffs_prototype) >= len(
        #         stuffs
        #     ), "We need at least the same number of more stuffs prototypes than stuffs classes."
        #     assert (
        #         len(things_prototype) + len(stuffs_prototype)
        #     ) == num_clusters, "Indexes of things and stuffs prototypes must count to number of cluster."
        # Save parameters
        self._things_prototype: Set[int] | None = things_prototype.copy() if things_prototype is not None else None
        self._stuffs_prototype: Set[int] | None = stuffs_prototype.copy() if stuffs_prototype is not None else None
        self.num_clusters: int = num_clusters
        self.num_classes: int = len(stuffs.union(things))
        self.perform_one_to_one_matching: bool = perform_one_to_one_matching
        self.cache_device: torch.device | str | None = cache_device
        self.assignments: None | Tensor = None
        self.classes_mask: List[bool] | None = classes_mask
        self.disable_matching: bool = disable_matching
        # Set void class to 255
        self.void_color: Tuple[int, int] = (255, 0)
        # Initialize state
        self.add_state("cost_matrix", default=torch.zeros(self.num_clusters, self.num_classes), dist_reduce_fx="sum")
        # Initialize lists to store predictions and labels
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def _cost_matrix_update(
        self,
        semantic_segmentation_pred: Tensor,
        semantic_segmentation_target: Tensor,
        num_classes: int,
        num_clusters: int,
    ) -> Tensor:
        """Computes the cost matrix update.

        Args:
            semantic_segmentation_pred (Tensor): Semantic segmentation prediction of the shape [N].
            semantic_segmentation_target (Tensor): Semantic segmentation target of the shape [N].
            num_classes (int): Number of classes to be used.
            num_clusters (int): Number of cluster to be used.

        Returns:
            cost_matrix_update (Tensor): Cost matrix update of the shape [num_clusters, num_classes]
        """
        # Estimate mask of valid pixels
        mask: Tensor = (
            (semantic_segmentation_target >= 0)
            & (semantic_segmentation_target < num_classes)
            & (semantic_segmentation_pred >= 0)
            & (semantic_segmentation_pred < num_clusters)
        )
        # Apply mask
        semantic_segmentation_pred = semantic_segmentation_pred[mask]
        semantic_segmentation_target = semantic_segmentation_target[mask]
        # Compute const matrix update
        cost_matrix_update: Tensor = (
            torch.bincount(
                (num_clusters) * semantic_segmentation_target + semantic_segmentation_pred,
                minlength=num_classes * (num_clusters),
            )
            .reshape(num_classes, num_clusters)
            .permute(1, 0)
        )
        return cost_matrix_update

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Updates the internal state of the metric. In particular, we track update the cost matrix for matching.

        Args:
            preds (Tensor): panoptic detection of shape ``[batch, *spatial_dims, 2]`` containing
                    the pair ``(category_id, instance_id)`` for each point.
                    If the ``category_id`` refer to a stuff, the instance_id is ignored.
            target (Tensor): ground truth of shape [batch, *spatial_dims, 2] containing
                the pair (category_id, instance_id) for each pixel of the image.
                If the category_id refer to a stuff, the instance_id is ignored.
        """
        # Track predictions and labels
        self.predictions.append(  # type: ignore
            preds.to(self.cache_device if self.cache_device is not None else preds.device).detach().clone()
        )  # type: ignore
        self.targets.append(  # type: ignore
            target.to(self.cache_device if self.cache_device is not None else preds.device).detach().clone()
        )  # type: ignore
        # Get semantic segmentation prediction and target and flatten
        semantic_segmentation_pred: Tensor = preds[..., 0].view(-1)
        semantic_segmentation_target: Tensor = target[..., 0].view(-1)
        # Compute const matrix update
        cost_matrix_update: Tensor = self._cost_matrix_update(
            semantic_segmentation_pred, semantic_segmentation_target, self.num_classes, self.num_clusters
        )
        # Update state
        self.cost_matrix += cost_matrix_update

    @staticmethod
    def map_to_target(panoptic_segmentation: Tensor, assignments: Tensor | None) -> Tensor:
        """Remaps the semantic classes to the target class using the latest assignments.

        Args:
            panoptic_segmentation (Tensor): Panoptic map of the shape [B, H, W, 2].
            assignments (Tensor | None): Class assignments as a tensor.

        Returns:
            panoptic_segmentation_mapped (Tensor): Mapped panoptic map of the shape [B, H, W, 2].
        """
        # Check if mapping is available
        assert assignments is not None, "Assignments not yet computed. Call compute before calling this method."
        # Assignments to same device
        assignments = assignments.to(panoptic_segmentation.device)
        # Get semantic segmentation prediction
        semantic_segmentation_prediction: Tensor = panoptic_segmentation[..., 0].clone()
        semantic_segmentation_prediction_no_void = semantic_segmentation_prediction[
            semantic_segmentation_prediction != 255
        ]
        semantic_segmentation_aligned: Tensor = torch.embedding(
            indices=semantic_segmentation_prediction_no_void, weight=assignments.view(-1, 1)
        ).squeeze(dim=-1)
        semantic_segmentation_prediction[semantic_segmentation_prediction != 255] = semantic_segmentation_aligned
        # Construct panoptic segmentation again
        panoptic_segmentation_mapped: Tensor = torch.stack(
            (semantic_segmentation_prediction, panoptic_segmentation[..., 1]), dim=-1
        )
        return panoptic_segmentation_mapped

    @property
    def things_prototype(self) -> Set[int]:
        """Getter method to access things prototypes.

        Returns:
            things_prototype (Set[int]): Things prototypes as a set of integer values.
        """
        return self._things_prototype  # type: ignore

    @property
    def stuffs_prototype(self) -> Set[int]:
        """Getter method to access stuffs prototypes.

        Returns:
            things_prototype (Set[int]): Stuffs prototypes as a set of integer values.
        """
        return self._stuffs_prototype  # type: ignore

    @things_prototype.setter  # type: ignore
    def things_prototype(self, value: Set[int]) -> None:
        """Setter method to access things prototypes.

        Args:
            value (Set[int] | None): Prototype IDs of thing c. If None not considered in matching
        """
        # Check value
        # assert len(value) >= len(
        #     self.things
        # ), "There must be at least the same or more things prototypes than things classes."
        self._things_prototype = value.copy()

    @stuffs_prototype.setter  # type: ignore
    def stuffs_prototype(self, value: Set[int]) -> None:
        """Setter method to access stuffs prototypes.

        Args:
            value (Set[int] | None): Prototype IDs of stuff c. If None not considered in matching
        """
        # Check value
        # assert len(value) >= len(
        #     self.stuffs
        # ), "There must be at least the same or more stuffs prototypes than stuffs classes."
        self._stuffs_prototype = value.copy()

    def _matching_core(self, num_classes: int, num_clusters: int, cost_matrix: Tensor) -> Tensor:
        """Core matching function.

        Returns:
            assignments (Tensor): Assignments as a tensor of the shape [num_clusters].
        """
        # Perform Hungarian matching in the case of one-to-one matching problem
        if num_classes == num_clusters:
            assignments: Tensor = torch.from_numpy(  # type: ignore
                linear_sum_assignment(cost_matrix.detach().cpu(), maximize=True)[-1]  # type: ignore
            )
        # Here we perform one-to-one matching in the overclustering setting
        elif self.perform_one_to_one_matching:
            indexes, assignments_main = linear_sum_assignment(  # type: ignore
                cost_matrix.detach().cpu(), maximize=True  # type: ignore
            )
            assignments_main = torch.from_numpy(assignments_main)
            indexes = torch.from_numpy(indexes)
            # Get missing clusters
            missing_clusters = set(range(num_clusters)) - set(indexes.tolist())
            missing_clusters = torch.tensor(list(missing_clusters))
            # Construct final assignments
            assignments = torch.empty(num_clusters, dtype=torch.long, device="cpu")
            assignments[indexes] = assignments_main
            assignments[missing_clusters] = 255
        # Perform Hungarian matching in the case of many-to-one matching problem
        else:
            assignments = -1 * torch.ones(num_clusters, dtype=torch.long, device="cpu")
            indices, values = linear_sum_assignment(cost_matrix.detach().cpu(), maximize=True)
            assignments[indices] = torch.Tensor(values).long()

            # Assign clusters using argmax
            max_assignments = cost_matrix.argmax(dim=-1).cpu()
            missing_indices = torch.Tensor([i for i in range(num_clusters) if i not in indices]).long()
            if len(missing_indices) > 0:
                assignments[missing_indices] = max_assignments[missing_indices]

        return assignments

    def _matching_no_separation(self) -> Tensor:
        """Performing without separating stuff and thing separation.

        Returns:
            assignments (Tensor): Assignments as a tensor of the shape [cluster IDs].
        """
        # Perform standard matching
        return self._matching_core(  # type: ignore
            num_classes=self.num_classes,  # type: ignore
            num_clusters=self.num_clusters,  # type: ignore
            cost_matrix=self.cost_matrix.clone(),  # type: ignore
        )

    def _matching_with_separation(self) -> Tensor:
        """Perform matching while adhering to thing-stuff split.

        Returns:
            assignments (Tensor): Assignments as a tensor of the shape [cluster IDs].
        """
        # Prototype and semantic classes to tensor
        things_prototype: Tensor = torch.as_tensor(
            list(self._things_prototype), device=self.cost_matrix.device  # type: ignore
        )
        stuffs_prototype: Tensor = torch.as_tensor(
            list(self._stuffs_prototype), device=self.cost_matrix.device  # type: ignore
        )  # type: ignore
        things: Tensor = torch.as_tensor(
            list(self.things),  # type: ignore
            device=self.cost_matrix.device,  # type: ignore
        )
        stuffs: Tensor = torch.as_tensor(
            list(self.stuffs),  # type: ignore
            device=self.cost_matrix.device,  # type: ignore
        )
        # Split cost matrix into stuff and thing
        cost_matrix_things: Tensor = self.cost_matrix[things_prototype][:, things]  # type: ignore
        cost_matrix_stuffs: Tensor = self.cost_matrix[stuffs_prototype][:, stuffs]  # type: ignore
        # Perform matching
        assignments_things: Tensor = self._matching_core(
            num_classes=len(self.things),  # type: ignore
            num_clusters=len(self._things_prototype),  # type: ignore
            cost_matrix=cost_matrix_things,
        )
        assignments_stuffs: Tensor = self._matching_core(
            num_classes=len(self.stuffs),  # type: ignore
            num_clusters=len(self._stuffs_prototype),  # type: ignore
            cost_matrix=cost_matrix_stuffs,
        )
        # Construct global assignment vector
        assignments = torch.empty(self.num_clusters, dtype=torch.long, device="cpu")
        assignments[things_prototype] = things.cpu()[assignments_things]
        assignments[stuffs_prototype] = stuffs.cpu()[assignments_stuffs]
        return assignments

    def matching(self) -> Tensor:
        """Perform matching between cluster IDs and class IDs.

        Notes:
            If thing and stuff prototype separation is given we perform separate matching for stuff and thing.

        Returns:
            assignments (Tensor): Assignments as a tensor of the shape [cluster IDs].
        """
        if self.stuffs_prototype is None:
            return self._matching_no_separation()
        return self._matching_with_separation()

    def compute(  # type: ignore
        self,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:  # type: ignore
        """Computes the final PQ alongside the SQ and RQ.

        Returns:
            pq (Tensor): Panoptic Quality averaged over all classes as a scalar tensor.
            sq (Tensor): Segmentation Quality averaged over all classes as a scalar tensor.
            rq (Tensor): Recognition Quality averaged over all classes as a scalar tensor.
            pq_per_class (Tensor): Panoptic Quality for each class shape is [C].
            sq_per_class (Tensor): Segmentation Quality for each class shape is [C].
            rq_per_class (Tensor): Recognition Quality for each class shape is [C].
            pq_things (Tensor): Panoptic Quality averaged over all things classes as a scalar tensor.
            sq_things (Tensor): Segmentation Quality averaged over all things classes as a scalar tensor.
            rq_things (Tensor): Recognition Quality averaged over all things classes as a scalar tensor.
            pq_stuffs (Tensor): Panoptic Quality averaged over all stuffs classes as a scalar tensor.
            sq_stuffs (Tensor): Segmentation Quality averaged over all stuffs classes as a scalar tensor.
            rq_stuffs (Tensor): Recognition Quality averaged over all stuffs classes as a scalar tensor.
            miou (Tensor): Class-wise mIoU as a scalar tensor.
            assignments (Tensor): The semantic class permutation from matching shape is [num clusters].
        """
        # Assignments to device
        self.assignments = self.matching().to(self.device)
        # Initialized cost matrix for class-wise mIoU
        cost_matrix_matched: Tensor = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        # Iterate over cached data
        for index, (prediction, target) in enumerate(zip(self.predictions, self.targets)):  # type: ignore
            # Data to device for speedup
            prediction = prediction.to(self.device)
            target = target.to(self.device)
            # Remap cluster IDs to classes
            if not self.disable_matching:
                prediction = self.map_to_target(prediction, self.assignments)
            # Save prediction to be returned
            if index == 0:
                prediction_output: Tensor = prediction.clone()
            # Update matched cost matrix
            cost_matrix_matched += self._cost_matrix_update(
                prediction[..., 0].view(-1),
                target[..., 0].view(-1),
                self.num_classes,
                self.num_classes,
            )
            # Add batch dimension if needed
            if prediction.ndim == 3:
                prediction = prediction[None]
            if target.ndim == 3:
                target = target[None]
            # Make PQ update
            super(PanopticQualitySemanticMatching, self).update(prediction, target)
        # Compute final metric
        (
            pq,
            sq,
            rq,
            pq_per_class,
            sq_per_class,
            rq_per_class,
            pq_things,
            sq_things,
            rq_things,
            pq_stuffs,
            sq_stuffs,
            rq_stuffs,
        ) = _panoptic_quality_compute(
            self.iou_sum,
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.cat_id_to_continuous_id,
            self.things,
            self.stuffs,
            self.classes_mask,
        )

        # match without separation for mIoU computation
        # Assignments to device
        miou_assignments = self._matching_no_separation().to(self.device)
        # Initialized cost matrix for class-wise mIoU
        cost_matrix_matched: Tensor = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        # Iterate over cached data
        for index, (prediction, target) in enumerate(zip(self.predictions, self.targets)):  # type: ignore
            # Data to device for speedup
            prediction = prediction.to(self.device)
            target = target.to(self.device)
            # Remap cluster IDs to classes
            if not self.disable_matching:
                prediction = self.map_to_target(prediction, miou_assignments)
            # Update matched cost matrix
            cost_matrix_matched += self._cost_matrix_update(
                prediction[..., 0].view(-1),
                target[..., 0].view(-1),
                self.num_classes,
                self.num_classes,
            )
        # Compute mIoU based on matched cost matrix
        miou, acc = _miou_compute(cost_matrix_matched)
        return (
            pq,
            sq,
            rq,
            pq_per_class,
            sq_per_class,
            rq_per_class,
            pq_things,
            sq_things,
            rq_things,
            pq_stuffs,
            sq_stuffs,
            rq_stuffs,
            miou,
            acc,
            self.assignments.clone(),
            prediction_output,
        )


def _miou_compute(cost_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes the final class-wise mIoU metric.

    Args:
        cost_matrix (Tensor): Cost matrix of the shape [num classes, num classes] (matrix needs to be aligned).

    Returns:
        miou (Tensor): Class-wise mean IoU as a scalar tensor.
        acc (Tensor): Semantic segmentation accuracy as a scalar tensor.
    """
    # Compute stats
    tp: Tensor = torch.diag(cost_matrix)
    fp: Tensor = torch.sum(cost_matrix, dim=0) - tp
    fn: Tensor = torch.sum(cost_matrix, dim=1) - tp
    # Compute IoU for each class
    iou = tp / (tp + fp + fn)
    # Average over all classes that are present
    miou: Tensor = iou[~torch.isnan(iou)].mean()
    # Compute accuracy
    acc = tp.sum() / cost_matrix.sum()
    return miou, acc


def _panoptic_quality_compute(
    iou_sum: Tensor,
    true_positives: Tensor,
    false_positives: Tensor,
    false_negatives: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    things: Set[int],
    stuffs: Set[int],
    classes_mask: List[bool] | None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the final panoptic quality from interim values.

    Args:
        iou_sum (Tensor): the iou sum from the update step.
        true_positives (Tensor): the TP value from the update step.
        false_positives (Tensor): the FP value from the update step.
        false_negatives (Tensor): the FN value from the update step.
        cat_id_to_continuous_id (Dict[int, int]): Mapping from class to internal class.
        things (Set[int]): Things classes.
        stuffs (Set[int]): Stuffs classes.
        classes_mask (List[bool] | None): Optional mask to ignore classes that are not present in the dataset.

    Returns:
        pq (Tensor): Panoptic Quality averaged over all classes as a scalar tensor.
        sq (Tensor): Segmentation Quality averaged over all classes as a scalar tensor.
        rq (Tensor): Recognition Quality averaged over all classes as a scalar tensor.
        pq_per_class (Tensor): Panoptic Quality for each class shape is [C].
        sq_per_class (Tensor): Segmentation Quality for each class shape is [C].
        rq_per_class (Tensor): Recognition Quality for each class shape is [C].
        pq_things (Tensor): Panoptic Quality averaged over all things classes as a scalar tensor.
        sq_things (Tensor): Segmentation Quality averaged over all things classes as a scalar tensor.
        rq_things (Tensor): Recognition Quality averaged over all things classes as a scalar tensor.
        pq_stuffs (Tensor): Panoptic Quality averaged over all stuffs classes as a scalar tensor.
        sq_stuffs (Tensor): Segmentation Quality averaged over all stuffs classes as a scalar tensor.
        rq_stuffs (Tensor): Recognition Quality averaged over all stuffs classes as a scalar tensor.
    """
    # Permute classes back to original order
    permutation = torch.tensor(
        [cat_id_to_continuous_id[key] for key in sorted(cat_id_to_continuous_id.keys())], device=iou_sum.device
    )
    iou_sum = iou_sum[permutation]
    true_positives = true_positives[permutation]
    false_positives = false_positives[permutation]
    false_negatives = false_negatives[permutation]
    # Things and suffs classes to tensor
    things_classes: Tensor = torch.tensor(list(things), device=iou_sum.device)
    stuffs_classes: Tensor = torch.tensor(list(stuffs), device=iou_sum.device)
    # If class mask is given we omit the respective classes
    if classes_mask is not None:
        # Class mask to tensor
        classes_mask = torch.tensor(classes_mask, device=iou_sum.device, dtype=torch.bool)  # type: ignore
        # We also need to omit the thing and stuff classes
        classes_mask_things: Tensor = classes_mask[things_classes]  # type: ignore
        classes_mask_stuffs: Tensor = classes_mask[stuffs_classes]  # type: ignore
        things_classes = things_classes[classes_mask_things]
        stuffs_classes = stuffs_classes[classes_mask_stuffs]
    else:
        classes_mask = torch.tensor(
            [True] * (len(things) + len(stuffs)), device=iou_sum.device, dtype=torch.bool  # type: ignore
        )
    # Compute segmentation and recognition quality (per-class)
    sq_per_class: Tensor = torch.where(true_positives > 0.0, iou_sum / true_positives, 0.0)
    denominator: Tensor = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    rq_per_class: Tensor = torch.where(denominator > 0.0, true_positives / denominator, 0.0)
    # Compute per-class panoptic quality
    pq_per_class: Tensor = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    # Compute things and stuff metrics
    pq_things: Tensor = torch.mean(pq_per_class[things_classes][denominator[things_classes] > 0])
    sq_things: Tensor = torch.mean(sq_per_class[things_classes][denominator[things_classes] > 0])
    rq_things: Tensor = torch.mean(rq_per_class[things_classes][denominator[things_classes] > 0])
    pq_stuffs: Tensor = torch.mean(pq_per_class[stuffs_classes][denominator[stuffs_classes] > 0])
    sq_stuffs: Tensor = torch.mean(sq_per_class[stuffs_classes][denominator[stuffs_classes] > 0])
    rq_stuffs: Tensor = torch.mean(rq_per_class[stuffs_classes][denominator[stuffs_classes] > 0])
    # Compute averages
    pq: Tensor = torch.mean(pq_per_class[classes_mask][denominator[classes_mask] > 0])
    sq: Tensor = torch.mean(sq_per_class[classes_mask][denominator[classes_mask] > 0])
    rq: Tensor = torch.mean(rq_per_class[classes_mask][denominator[classes_mask] > 0])
    return (
        pq,
        sq,
        rq,
        pq_per_class[classes_mask],
        sq_per_class[classes_mask],
        rq_per_class[classes_mask],
        pq_things,
        sq_things,
        rq_things,
        pq_stuffs,
        sq_stuffs,
        rq_stuffs,
    )


def print_metrics(metrics_out: Tuple[Tensor, ...], class_names: List[str]) -> None:
    """Displays the results of the PQ.

    Args:
        metrics_out (Tuple[Tensor, ...]): Output of the Panoptic Quality (semantic matching metric).
        class_names (List[str]): Class names as a list of strings.
    """
    log.info("--- CLASS PQ / SQ / RQ ---")
    for key, cpq, csq, crq in zip(
        [(c + ": ").ljust(20) for c in class_names], metrics_out[0], metrics_out[1], metrics_out[2]
    ):
        log.info(f"{key} {round(cpq.item(), 4)}, {round(csq.item(), 4)}, {round(crq.item(), 4)}")
    log.info("--- PQ / SQ / RQ ---")
    log.info(f"{round(metrics_out[3].item(), 4)}, {round(metrics_out[4].item(), 4)}, {round(metrics_out[5].item(), 4)}")
    log.info("--- mIoU ---")
    log.info(f"{round(metrics_out[6].item(), 4)}")
