import logging

import torch
import torchvision.io
from torch import Tensor

from cups.metrics import PanopticQualitySemanticMatching
from cups.model import (
    panoptic_cascade_mask_r_cnn_from_checkpoint,
    prediction_to_standard_format,
)
from cups.visualization import plot_panoptic_segmentation_overlay

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

IMAGE_PATH: str = "assets/stuttgart_02_000000_005445_leftImg8bit.png"
DEVICE: str = "cpu"
CHECKPOINT_PATH = "assets/cups.ckpt"


@torch.no_grad()
def main() -> None:
    # Init model
    model, num_clusters_things, num_clusters_stuffs = panoptic_cascade_mask_r_cnn_from_checkpoint(
        path=CHECKPOINT_PATH,
        device="cpu",
        confidence_threshold=0.5,
    )
    # Print model
    log.info(model)
    # Model to device
    model = model.to(DEVICE)
    # Model to eval mode
    model = model.eval()
    # Assignments of CUPS checkpoint
    assignments: Tensor = torch.tensor(
        [7, 4, 2, 4, 2, 6, 2, 0, 5, 8, 0, 2, 9, 10, 3, 8, 1, 2, 0, 0, 0, 0, 11, 13, 18, 15, 14], device=DEVICE
    )
    # Load image and map pixel-range to [0, 1]
    image: Tensor = torchvision.io.read_image(IMAGE_PATH).float() / 255.0
    # Image to device
    image = image.to(DEVICE)
    # Make panoptic prediction
    prediction = model([{"image": image.to(DEVICE)}])
    # Convert prediction into standard format
    prediction: Tensor = prediction_to_standard_format(
        prediction[0]["panoptic_seg"],
        stuff_classes=tuple(index for index in range(num_clusters_stuffs)),
        thing_classes=tuple(index + num_clusters_stuffs for index in range(num_clusters_things)),
    )
    # Remap semantics for visualization
    prediction = PanopticQualitySemanticMatching.map_to_target(
        panoptic_segmentation=prediction, assignments=assignments
    )
    # Plot panoptic prediction
    plot_panoptic_segmentation_overlay(
        panoptic_segmentation=prediction.cpu(), image=image.cpu(), dataset="cityscapes_19"
    )


if __name__ == "__main__":
    main()
