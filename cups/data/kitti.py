import os
import sys
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, PadTo
from kornia.geometry import scale_intrinsics
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from cups.data.cityscapes import get_class_mapping, load_panoptic_cityscapes_labels
from cups.data.utils import load_image
from cups.scene_flow_2_se3.utils import remap_ids

__all__: Tuple[str, ...] = (
    "KITTIStereoVideo",
    "KITTIRaw",
    "KITTIPanopticValidation",
    "KITTIInstanceSegmentation",
    "KITTI_INSTANCE_STUFF_CLASSES",
    "KITTI_INSTANCE_THING_CLASSES",
)

KITTI_INSTANCE_STUFF_CLASSES: Set[int] = {0}
KITTI_INSTANCE_THING_CLASSES: Set[int] = {1}

KITTI_VAL_RAW: Tuple[str, ...] = (
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000264.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000280.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000106.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000192.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000179.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000228.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000308.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000354.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000122.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000046.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000218.png",
    "2011_09_29/2011_09_29_drive_0071_sync/image_02/data/0000000059.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000313.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000299.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000147.png",
    "2011_09_26/2011_09_26_drive_0046_sync/image_02/data/0000000052.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000218.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000356.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000322.png",
    "2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000556.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000240.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000086.png",
    "2011_09_26/2011_09_26_drive_0104_sync/image_02/data/0000000035.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000150.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000070.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000278.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000286.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000339.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000167.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000125.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000191.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000374.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000340.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000284.png",
    "2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000054.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000282.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000071.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000258.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000140.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000111.png",
    "2011_09_26/2011_09_26_drive_0029_sync/image_02/data/0000000016.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000300.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000079.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000050.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000644.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000050.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000447.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000219.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000152.png",
    "2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000224.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000095.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000394.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000282.png",
    "2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000053.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000105.png",
    "2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000059.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000097.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000197.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000066.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000809.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000239.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000109.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000320.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000129.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000302.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000378.png",
    "2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000402.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000342.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000023.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000082.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000026.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000269.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000127.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000176.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000378.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000330.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000260.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000273.png",
    "2011_09_29/2011_09_29_drive_0071_sync/image_02/data/0000000943.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000084.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000172.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000133.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000364.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000132.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000207.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000157.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000319.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000303.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000350.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000096.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000238.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000213.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000381.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000040.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000137.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000312.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000379.png",
    "2011_09_26/2011_09_26_drive_0046_sync/image_02/data/0000000062.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000046.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000457.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000654.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000094.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000290.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000087.png",
    "2011_09_28/2011_09_28_drive_0002_sync/image_02/data/0000000343.png",
    "2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000069.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000141.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000209.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000285.png",
    "2011_09_26/2011_09_26_drive_0029_sync/image_02/data/0000000123.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000118.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000414.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000125.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000230.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000292.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000360.png",
    "2011_09_26/2011_09_26_drive_0104_sync/image_02/data/0000000015.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000076.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000036.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000634.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000060.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000310.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000229.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000175.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000040.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000187.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000384.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000201.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000114.png",
    "2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000010.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000162.png",
    "2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000103.png",
)

KITTI_IGNORE_CLIPS: Tuple[str, ...] = (
    # Calibration videos
    "2011_09_26_drive_0119",
    "2011_09_28_drive_0225",
    "2011_09_29_drive_0108",
    "2011_09_30_drive_0072",
    "2011_10_03_drive_0058",
    # Person videos
    # "2011_09_28_drive_0053",
    # "2011_09_28_drive_0054",
    "2011_09_28_drive_0057",
    "2011_09_28_drive_0065",
    "2011_09_28_drive_0066",
    "2011_09_28_drive_0068",
    "2011_09_28_drive_0070",
    "2011_09_28_drive_0071",
    "2011_09_28_drive_0075",
    "2011_09_28_drive_0077",
    "2011_09_28_drive_0078",
    "2011_09_28_drive_0080",
    "2011_09_28_drive_0082",
    "2011_09_28_drive_0086",
    "2011_09_28_drive_0087",
    "2011_09_28_drive_0089",
    "2011_09_28_drive_0090",
    "2011_09_28_drive_0094",
    "2011_09_28_drive_0095",
    "2011_09_28_drive_0096",
    "2011_09_28_drive_0098",
    "2011_09_28_drive_0100",
    "2011_09_28_drive_0102",
    "2011_09_28_drive_0103",
    "2011_09_28_drive_0104",
    "2011_09_28_drive_0106",
    "2011_09_28_drive_0108",
    "2011_09_28_drive_0110",
    "2011_09_28_drive_0113",
    "2011_09_28_drive_0117",
    "2011_09_28_drive_0119",
    "2011_09_28_drive_0121",
    "2011_09_28_drive_0122",
    "2011_09_28_drive_0125",
    "2011_09_28_drive_0126",
    "2011_09_28_drive_0128",
    "2011_09_28_drive_0132",
    "2011_09_28_drive_0134",
    "2011_09_28_drive_0135",
    "2011_09_28_drive_0136",
    "2011_09_28_drive_0138",
    "2011_09_28_drive_0141",
    "2011_09_28_drive_0143",
    "2011_09_28_drive_0145",
    "2011_09_28_drive_0146",
    "2011_09_28_drive_0149",
    "2011_09_28_drive_0153",
    "2011_09_28_drive_0154",
    "2011_09_28_drive_0155",
    "2011_09_28_drive_0156",
    "2011_09_28_drive_0160",
    "2011_09_28_drive_0161",
    "2011_09_28_drive_0162",
    "2011_09_28_drive_0165",
    "2011_09_28_drive_0166",
    "2011_09_28_drive_0167",
    "2011_09_28_drive_0168",
    "2011_09_28_drive_0171",
    "2011_09_28_drive_0174",
    "2011_09_28_drive_0177",
    "2011_09_28_drive_0179",
    "2011_09_28_drive_0183",
    "2011_09_28_drive_0184",
    "2011_09_28_drive_0185",
    "2011_09_28_drive_0186",
    "2011_09_28_drive_0187",
    "2011_09_28_drive_0191",
    "2011_09_28_drive_0192",
    "2011_09_28_drive_0195",
    "2011_09_28_drive_0198",
    "2011_09_28_drive_0199",
    "2011_09_28_drive_0201",
    "2011_09_28_drive_0204",
    "2011_09_28_drive_0205",
    "2011_09_28_drive_0208",
    "2011_09_28_drive_0209",
    "2011_09_28_drive_0214",
    "2011_09_28_drive_0216",
    "2011_09_28_drive_0220",
    "2011_09_28_drive_0222",
)


class KITTIStereoVideo(Dataset):
    """This class implements the unsupervised KITTI-2015 stereo video dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
        temporal_stride: int = 1,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to dataset folders (i.e. path to left/rightImg8bit_sequence)
            split (str): Split to be used.
            resize_scale (Tuple[int, int]): Scale to which the images are resized.
            crop_resolution (Tuple[int, int]): Resolution to which the images are cropped after resizing.
            temporal_stride (int): Temporal stride to be utilized (stride one is 17 FPS). Default: 2.
        """
        # Call super constructor
        super(KITTIStereoVideo, self).__init__()
        # Check parameters
        assert split in ["train", "test"]
        # Save parameters
        self.resize_scale: float = resize_scale
        self.temporal_stride: int = temporal_stride
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        # Get folder of split
        split_folder: str = "training" if split == "train" else "testing"
        # Init list to store path to data
        self.sample_path: List = []
        # Get clip ids
        clip_ids = {
            file.split("_")[0] for file in os.listdir(os.path.join(root, split_folder, "image_2")) if ".png" in file
        }
        # Get samples
        for clip_id in sorted(clip_ids):
            for frame_index in range(20 - temporal_stride):
                self.sample_path.append(
                    [
                        os.path.join(root, split_folder, "image_2", f"{clip_id}_{str(frame_index).zfill(2)}.png"),
                        os.path.join(
                            root,
                            split_folder,
                            "image_2",
                            f"{clip_id}_{str(frame_index + temporal_stride).zfill(2)}.png",
                        ),
                    ]
                )

    def __len__(self) -> int:
        """Returns the length of the dataset.

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
        image_0_l: Tensor = load_image(path=self.sample_path[index][0])[None]
        image_0_r: Tensor = load_image(path=self.sample_path[index][0].replace("image_2", "image_3"))[None]
        image_1_l: Tensor = load_image(path=self.sample_path[index][1])[None]
        image_1_r: Tensor = load_image(path=self.sample_path[index][1].replace("image_2", "image_3"))[None]
        # Make map of valid pixels (some pixels are not valid due to rectification)
        valid_pixels: Tensor = torch.ones(1, 1, *image_0_l.shape[-2:], device=image_0_l.device)
        valid_pixels[:, :, : round(valid_pixels.shape[2] * 0.2)] = 0.0
        # Load calibration file
        calibration_path = self.sample_path[index][0].replace("image_2", "calib_cam_to_cam")
        clip_id = self.sample_path[index][0].split("/")[-1].split("_")[0]
        calibration_path: str = os.path.join("/", *calibration_path.split("/")[:-1], f"{clip_id}.txt")
        baseline, intrinsics = read_calibration_file(path=calibration_path)
        # Resize images and scale intrinsics
        if self.resize_scale != 1.0:
            image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
            image_0_r = F.interpolate(image_0_r, scale_factor=self.resize_scale, mode="bilinear")
            image_1_l = F.interpolate(image_1_l, scale_factor=self.resize_scale, mode="bilinear")
            image_1_r = F.interpolate(image_1_r, scale_factor=self.resize_scale, mode="bilinear")
            valid_pixels = F.interpolate(valid_pixels, scale_factor=self.resize_scale, mode="nearest")
            intrinsics = scale_intrinsics(intrinsics, scale_factor=self.resize_scale)
        # Pad image
        image_0_l = self.pad_module(image_0_l)
        image_0_r = self.pad_module(image_0_r)
        image_1_l = self.pad_module(image_1_l)
        image_1_r = self.pad_module(image_1_r)
        valid_pixels = self.pad_module(valid_pixels)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": self.crop_module(image_0_l),
            "image_0_r": self.crop_module(image_0_r),
            "image_1_l": self.crop_module(image_1_l),
            "image_1_r": self.crop_module(image_1_r),
            "valid_pixels": self.crop_module(valid_pixels).bool(),
            "baseline": baseline.reshape(1),
            "intrinsics": intrinsics.reshape(1, 3, 3),
        }
        return output


class KITTIInstanceSegmentation(Dataset):
    """This class implements the KITTI instance segmentation validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
        void_id: int = 255,
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
        super(KITTIInstanceSegmentation, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.void_id: int = void_id
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        self.pad_module_semantic: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=self.void_id)
        # Init list to store paths
        self.images = []
        self.labels = []
        # Get image paths
        image_directory = os.path.join(root, "training", "image_2")
        for file in sorted(os.listdir(image_directory)):
            if ".png" in file:
                self.images.append(os.path.join(image_directory, file))
        # Get annotation paths
        label_directory_instance = os.path.join(root, "data_object_instance_2", "training", "instance_2")
        for file in sorted(os.listdir(label_directory_instance)):
            if ".png" in file:
                self.labels.append(os.path.join(label_directory_instance, file))

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
        label = torch.from_numpy(cv2.imread(self.labels[index], cv2.IMREAD_ANYDEPTH).astype(np.int64))
        # Make semantic segmentation
        semantic_label = (label != 0).long()[None]
        # Make instance label
        instance_label = remap_ids(label)[None]
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
            "image_name": self.images[index].split("/")[-1].replace(".png", ""),  # type: ignore
        }
        return output


class KITTIPanopticValidation(Dataset):
    """This class implements the KITTI panoptic validation dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
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
        super(KITTIPanopticValidation, self).__init__()
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
        image_directory = os.path.join(root, "validation", "images")
        for file in sorted(os.listdir(image_directory)):
            if ".png" in file:
                self.images.append(os.path.join(image_directory, file))
        # Get annotation paths
        label_directory_instance = os.path.join(root, "validation", "cityscapes_instance_format")
        for file in sorted(os.listdir(label_directory_instance)):
            if ".png" in file:
                self.labels.append(os.path.join(label_directory_instance, file))

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
            "image_name": self.images[index].split("/")[-1].replace(".png", ""),  # type: ignore
        }
        return output


class KITTIPanopticValidationStereo(KITTIPanopticValidation):
    """This class implements the KITTI panoptic stereo validation dataset."""

    def __init__(
        self,
        root: str,
        root_multi_view: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
        void_id: int = 255,
        num_classes: int = 27,
        temporal_stride: int = 1,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to the dataset.
            root_multi_view (str): Path to multi-view dataset
            resize_scale (float): Scale factor to be applied to the images and labels.
            crop_resolution (Tuple[int, int]): Crop resolution to be utilized after resizing. Default (368, 1240).
            void_id (int): Void ID to be utilized. Default 255.
            num_classes (int): Number of classes to be utilized. Default 27.
            temporal_stride (int): Temporal stride to be utilized. Default 1.
        """
        # Call super constructor
        super(KITTIPanopticValidationStereo, self).__init__(
            root=root,
            resize_scale=resize_scale,
            crop_resolution=crop_resolution,
            void_id=void_id,
            num_classes=num_classes,
        )
        # Save parameters
        self.root: str = root
        self.root_multi_view: str = root_multi_view
        self.temporal_stride: int = temporal_stride

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Method returns an instances of the dataset given its index.

        Args:
            index (int): Index of the sample.

        Returns:
            output (Dict[str, Tensor]): Sample containing the images, calibration, semantic, and instance segmentation.
        """
        # Get single view data
        output: Dict[str, Tensor] = super().__getitem__(index=index)  # type: ignore
        # Get path of multi-view images
        image_0_r_path = (
            self.images[index]
            .replace(self.root, self.root_multi_view)
            .replace("validation", "training")
            .replace("images", "image_3")
        )
        image_name_0_r = image_0_r_path.split("/")[-1].replace(".png", "")
        image_name_1_r = (
            image_name_0_r.split("_")[0] + "_" + str(int(image_name_0_r.split("_")[-1]) + self.temporal_stride).zfill(2)
        )
        image_1_r_path = image_0_r_path.replace(image_name_0_r, image_name_1_r)
        image_1_l_path = image_1_r_path.replace("image_3", "image_2")
        # Load multi view images
        image_0_r: Tensor = load_image(path=image_0_r_path)[None]
        image_1_l: Tensor = load_image(path=image_1_l_path)[None]
        image_1_r: Tensor = load_image(path=image_1_r_path)[None]
        # Resize images
        image_0_r = F.interpolate(image_0_r, scale_factor=self.resize_scale, mode="bilinear")
        image_1_l = F.interpolate(image_1_l, scale_factor=self.resize_scale, mode="bilinear")
        image_1_r = F.interpolate(image_1_r, scale_factor=self.resize_scale, mode="bilinear")
        # Pad data to ensure min size
        image_0_r = self.pad_module(image_0_r)
        image_1_l = self.pad_module(image_1_l)
        image_1_r = self.pad_module(image_1_r)
        # Crop data
        image_0_r = self.crop_module(image_0_r)
        image_1_l = self.crop_module(image_1_l)
        image_1_r = self.crop_module(image_1_r)
        # Load calibration file
        calibration_path = image_0_r_path.replace("image_3", "calib_cam_to_cam")
        clip_id = image_0_r_path.split("/")[-1].split("_")[0]
        calibration_path: str = os.path.join("/", *calibration_path.split("/")[:-1], f"{clip_id}.txt")
        baseline, intrinsics = read_calibration_file(path=calibration_path)
        # Add multi-view data to output dict
        output["image_0_r"] = image_0_r
        output["image_1_l"] = image_1_l
        output["image_1_r"] = image_1_r
        output["baseline"] = baseline.reshape(1)
        output["intrinsics"] = intrinsics.reshape(1, 3, 3)
        return output


class KITTIRaw(Dataset):
    """This dataset implements the KITTI raw stereo dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
        temporal_stride: int = 1,
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to the dataset.
            resize_scale (float): Scale factor to be applied to the images and labels.
            crop_resolution (Tuple[int, int]): Crop resolution to be utilized after resizing. Default (368, 1240).
            temporal_stride (int): Temporal stride to be utilized (stride one is 10 FPS). Default: 1.
        """
        # Call super constructor
        super(KITTIRaw, self).__init__()
        # Save parameters
        self.resize_scale: float = resize_scale
        self.temporal_stride: int = temporal_stride
        # Init crop module
        self.crop_module: nn.Module = CenterCrop(size=crop_resolution, keepdim=True)
        # Init padding module
        self.pad_module: nn.Module = PadTo(size=crop_resolution, pad_mode="constant", pad_value=0)
        # Load image paths
        self.images: List = []
        # Get all image paths
        for day in os.listdir(root):
            if os.path.isdir(os.path.join(root, day)):
                # Iterate over clips
                for clip in os.listdir(os.path.join(root, day)):
                    if ("_sync" in clip) and not (clip.replace("_sync", "") in KITTI_IGNORE_CLIPS):
                        # Iterate over left image
                        clip_path = os.path.join(root, day, clip, "image_02", "data")
                        # Get file names
                        file_names = [file_name for file_name in sorted(os.listdir(clip_path)) if ".png" in file_name]
                        # Get all frames to be ignored
                        files_to_be_ignored = [file for file in KITTI_VAL_RAW if clip in file]
                        # Omit labeled frame and neighbouring ones
                        file_names_split = []
                        if len(files_to_be_ignored) > 0:
                            indexes = [
                                file_names.index(file_to_be_ignored.split("/")[-1])
                                for file_to_be_ignored in sorted(files_to_be_ignored)
                            ]
                            previous_index = 0
                            for index in indexes:
                                start = max(previous_index, index - 5)
                                end = min(index + 5, len(file_names))
                                if start > previous_index:
                                    file_names_split.append(file_names[previous_index:start])
                                previous_index = end
                            if previous_index < (len(file_names) - 1):
                                file_names_split.append(file_names[previous_index:])
                        else:
                            file_names_split.append(file_names)
                        # Construct frame pairs
                        for file_names in file_names_split:
                            # Check that the clip part is larger than the temporal stride
                            if len(file_names) > self.temporal_stride:
                                # Make full file names
                                file_names_full = [
                                    os.path.join(root, day, clip, "image_02", "data", file) for file in file_names
                                ]
                                self.images.extend(
                                    zip(
                                        file_names_full[: -self.temporal_stride],
                                        file_names_full[self.temporal_stride :],
                                    )
                                )

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.images)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (images (0, 1, l, r), valid_pixels, baseline, intrinsics).
        """
        # Load data
        image_0_l: Tensor = load_image(path=self.images[index][0])[None]
        image_0_r: Tensor = load_image(path=self.images[index][0].replace("image_02", "image_03"))[None]
        image_1_l: Tensor = load_image(path=self.images[index][1])[None]
        image_1_r: Tensor = load_image(path=self.images[index][1].replace("image_02", "image_03"))[None]
        # Make map of valid pixels (some pixels are not valid due to rectification)
        valid_pixels: Tensor = torch.ones(1, 1, *image_0_l.shape[-2:], device=image_0_l.device)
        valid_pixels[:, :, : round(valid_pixels.shape[2] * 0.05)] = 0.0
        valid_pixels[:, :, -round(valid_pixels.shape[2] * 0.2) :] = 0.0
        # Load calibration file
        calibration_path = os.path.join("/", *self.images[index][0].split("/")[:-4], "calib_cam_to_cam.txt")
        baseline, intrinsics = read_calibration_file(path=calibration_path)
        # Resize images and scale intrinsics
        if self.resize_scale != 1.0:
            image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
            image_0_r = F.interpolate(image_0_r, scale_factor=self.resize_scale, mode="bilinear")
            image_1_l = F.interpolate(image_1_l, scale_factor=self.resize_scale, mode="bilinear")
            image_1_r = F.interpolate(image_1_r, scale_factor=self.resize_scale, mode="bilinear")
            valid_pixels = F.interpolate(valid_pixels, scale_factor=self.resize_scale, mode="nearest")
            intrinsics = scale_intrinsics(intrinsics, scale_factor=self.resize_scale)
        # Pad image
        image_0_l = self.pad_module(image_0_l)
        image_0_r = self.pad_module(image_0_r)
        image_1_l = self.pad_module(image_1_l)
        image_1_r = self.pad_module(image_1_r)
        valid_pixels = self.pad_module(valid_pixels)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image_0_l": self.crop_module(image_0_l),
            "image_0_r": self.crop_module(image_0_r),
            "image_1_l": self.crop_module(image_1_l),
            "image_1_r": self.crop_module(image_1_r),
            "valid_pixels": self.crop_module(valid_pixels).bool(),
            "baseline": baseline.reshape(1),
            "intrinsics": intrinsics.reshape(1, 3, 3),
            "image_0_l_path": self.images[index][0],
        }
        return output


class KITTISelfTraining(KITTIRaw):
    """This dataset implements the KITTI raw stereo dataset."""

    def __init__(
        self,
        root: str,
        resize_scale: float = 1.0,
        crop_resolution: Tuple[int, int] = (368, 1240),
    ) -> None:
        """Constructor method.

        Args:
            root (str): Path to the dataset.
            resize_scale (float): Scale factor to be applied to the images and labels.
            crop_resolution (Tuple[int, int]): Crop resolution to be utilized after resizing. Default (368, 1240).
        """
        # Call super constructor
        super(KITTISelfTraining, self).__init__(root=root, resize_scale=resize_scale, crop_resolution=crop_resolution)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        length: int = len(self.images)
        return length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns on instance of the dataset.

        Args:
            index (int): Index of the sample to be loaded.

        Returns:
            output (Dict[str, Tensor]): Dict of loaded data (images (0, 1, l, r), valid_pixels, baseline, intrinsics).
        """
        # Load data
        image_0_l: Tensor = load_image(path=self.images[index][0])[None]
        # Resize images and scale intrinsics
        if self.resize_scale != 1.0:
            image_0_l = F.interpolate(image_0_l, scale_factor=self.resize_scale, mode="bilinear")
        # Pad image
        image_0_l = self.pad_module(image_0_l)
        # Crop image
        image_0_l = self.crop_module(image_0_l)
        # Make output dict
        output: Dict[str, Tensor] = {
            "image": image_0_l.squeeze(),
            "height": image_0_l.shape[-2],
            "width": image_0_l.shape[-1],
        }
        return output


def read_calibration_file(path: str) -> Tuple[Tensor, Tensor]:
    """Function reads a KITTI calibration file and returns the baseline and intrinsics.

    Args:
        path (str): Path to calibration file.

    Returns:
        baseline (Tensor): Baseline tensor of the shape [1].
        intrinsics (Tensor): Intrinsics as a tensor of the shape [3, 3].
    """
    # Init dict to store values
    data = {}
    # Load file
    with open(path) as file:
        for line in file.readlines():
            key, value = line.split(":", 1)
            try:
                data[key] = torch.tensor([float(x) for x in value.split()])
            except ValueError:
                pass
    # Get baseline
    baseline = torch.tensor([data["P_rect_02"][3] / data["P_rect_02"][0] - data["P_rect_03"][3] / data["P_rect_03"][0]])
    # Get intrinsics
    fx = data["P_rect_02"][0]
    fy = data["P_rect_02"][5]
    cx = data["P_rect_02"][2]
    cy = data["P_rect_02"][6]
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return baseline, intrinsics
