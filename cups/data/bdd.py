import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, PadTo
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

sys.path.append(os.getcwd())
from cups.data.utils import load_image
from cups.scene_flow_2_se3.utils import remap_ids


def get_bdd2cs_class_mapping(num_classes: int = 27, void_id: int = 255) -> Tensor:
    """FUnction returns the mapping from raw Cityscapes classes to the supported number of classes (27, 19, and 7).

    Args:
        num_classes (int): Number of classes to be utilized. Default is 27.
        void_id (int): Void id class to ignore parts.

    Returns:
        mapping_weights (Tensor): Mapping weights of the shape [34, 1]
    """
    # Check input
    assert num_classes in (27, 19, 7), f"{num_classes} classes is not supported."
    # 27 classes case
    weights: Tensor = torch.Tensor([c.to_cs27 for c in BDD_LABEL]).int()
    weights[weights == 255] = void_id
    # 19 classes case
    if num_classes == 19:
        for cs27_id, c in enumerate(Cityscapes.classes[7:-1]):
            if c.ignore_in_eval:
                weights[weights == cs27_id] = void_id
            else:
                weights[weights == cs27_id] = c.train_id
    # 7 parent classes
    if num_classes == 7:
        for cs27_id, c in enumerate(Cityscapes.classes[7:-1]):
            if c.ignore_in_eval:
                weights[weights == cs27_id] = void_id
            else:
                weights[weights == cs27_id] = c.category_id - 1

    # Reshape weights to [34, 1]
    weights = weights[..., None]
    return weights


# # """Label definition."""

from collections import namedtuple

# # # a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
        "to_cs27",  # mapping to 27 classes of cityscapes
    ],
)


# # # Our extended list of label types. Our train id is compatible with Cityscapes
BDD_LABEL = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color to_cs27
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0), 255),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81), 255),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160), 2),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140), 3),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128), 0),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232), 1),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100), 8),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70), 4),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153), 6),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180), 255),
    Label("guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180), 7),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90), 9),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156), 5),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100), 255),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250), 255),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0), 255),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60), 255),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153), 10),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153), 11),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100), 255),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0), 255),
    Label("traffic device", 24, 255, "object", 3, False, True, (220, 220, 220), 255),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30), 12),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0), 13),
    Label("traffic sign frame", 27, 255, "object", 3, False, True, (250, 170, 250), 255),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152), 15),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35), 14),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180), 16),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60), 17),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0), 18),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32), 26),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100), 21),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142), 19),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90), 22),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230), 25),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110), 23),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100), 24),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70), 20),
]


class BDD10kPanopticValidation(Dataset):
    """This class implements the KITTI panoptic validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (720, 1280),
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
        super(BDD10kPanopticValidation, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        self.pad_module_semantic: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=self.void_id)
        # Get class mapping
        self.class_mapping: Tensor = get_bdd2cs_class_mapping(num_classes=num_classes, void_id=self.void_id)
        # Init list to store paths
        self.images = []
        self.labels = []
        # Get image paths
        image_directory = os.path.join(root, "images", "10k", "val")
        label_directory_instance = os.path.join(root, "labels", "pan_seg", "bitmasks", "val")
        for file in sorted(os.listdir(image_directory)):
            if ".jpg" in file:
                self.images.append(os.path.join(image_directory, file))
                self.labels.append(os.path.join(label_directory_instance, file.replace(".jpg", ".png")))

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
        label_raw = torch.from_numpy(np.array(Image.open(self.labels[index])).astype(np.uint8)).long()
        category_map = label_raw[:, :, 0]
        # attributes_map = label_raw[:, :, 1]
        # load instance label set stuff areas in instance label to 0 and get consecutive instance ids
        instance_label = (label_raw[:, :, 2] << 8) + label_raw[:, :, 3]
        instance_label[category_map < 31] = 0
        # instance_label = torch.nn.functional.one_hot(instance_label)[..., instance_label.unique()].argmax(dim=-1)
        # map semantic label to cityscapes
        semantic_label = self.class_mapping[category_map].squeeze()
        # Resize labels
        semantic_label = F.interpolate(
            semantic_label[None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        instance_label = F.interpolate(
            instance_label[None].float(), scale_factor=self.resize_scale, mode="nearest"
        ).long()
        # Pad data to ensure min size
        image_0_l = self.pad_module(image_0_l)
        semantic_label = self.pad_module_semantic(semantic_label.float()).long()
        instance_label = self.pad_module(instance_label.float()).long()
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
            "image_name": self.images[index].split("/")[-1].replace(".jpg", ""),  # type: ignore
        }
        return output


if __name__ == "__main__":
    from cups.visualization import save_panoptic_segmentation_overlay

    dataset = BDD10kPanopticValidation(root="/path_to_dataset/", num_classes=7)
    output = dataset[0]

    save_panoptic_segmentation_overlay(
        torch.stack((output["semantic_gt"][0, 0], output["instance_gt"][0, 0]), dim=-1),
        output["image_0_l"][0],
        dataset="cityscapes_7",
        path="vis.png",
    )
    # for index, sample in enumerate(dataset):
    #     print(index)
