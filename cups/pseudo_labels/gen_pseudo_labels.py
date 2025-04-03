import logging
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import pathlib

sys.path.append(os.getcwd())
import cups
from cups.crf import batched_crf
from cups.data import CityscapesStereoVideo, KITTIRaw
from cups.optical_flow import raft_smurf
from cups.scene_flow_2_se3 import sf2se3 as get_object_proposals
from cups.semantics.model import DepthG
from cups.thingstuff_split import ThingStuffSplitter
from cups.utils import align_semantic_to_instance, normalize, set_seed_everywhere


def configure() -> CfgNode:
    """Function loads default config, experiment config, and parses command line arguments.

    Returns:
        config (CfgNode): Config object.
    """
    # Manage command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--SYSTEM.EXPERIMENT_NAME",
        default=None,
        type=str,
        help="Experiment name.",
    )
    parser.add_argument(
        "--DATA.PSEUDO_ROOT",
        default=None,
        type=str,
        help="Root for pseudo labels.",
    )
    parser.add_argument(
        "--DATA.DATASET",
        default=None,
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "--SYSTEM.NUM_WORKERS",
        default=None,
        type=int,
        help="Number of workers to be used.",
    )
    parser.add_argument(
        "--DATA.NUM_PREPROCESSING_SUBSPLITS",
        default=None,
        type=int,
        help="Number of dataset subsplits.",
    )
    parser.add_argument(
        "--DATA.PREPROCESSING_SUBSPLIT",
        default=None,
        type=int,
        help="ID of scpecific subsplit.",
    )
    parser.add_argument(
        "--MODEL.CHECKPOINT",
        default="/path_to_checkpoint/depthg.ckpt",
        type=str,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        type=str,
        help="Sets the visible cuda devices.",
    )
    parser.add_argument(
        "--experiment_config_file",
        default="cups/pseudo_labels/config_pseudo_labels.yaml",
        type=str,
        help="Path to experiment config file.",
    )
    # Get arguments
    args = parser.parse_args()
    # Arguments to dict
    args_dict: Dict[str, Any] = vars(args)
    # Set cuda devices
    if (cuda_visible_devices := args_dict.pop("cuda_visible_devices")) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Get path to experiment config file
    experiment_config_file = args_dict.pop("experiment_config_file")
    # To list and remove all None entries from argument dict
    args_list: List[Union[str, Any]] = []
    for key, value in args_dict.items():
        if value is not None:
            args_list.extend((key, value))
    # Load config
    config: CfgNode = cups.get_default_config(
        experiment_config_file=experiment_config_file, command_line_arguments=args_list
    )
    return config


@torch.inference_mode()
def main() -> None:
    # get and log config
    config = configure()
    log.info(config)
    # set all seeds
    set_seed_everywhere(config.SYSTEM.SEED)
    # create directories
    pathlib.Path(config.DATA.PSEUDO_ROOT).mkdir(parents=True, exist_ok=True)

    # Init dataset
    if config.DATA.DATASET == "cityscapes":
        data = CityscapesStereoVideo(
            root=os.path.join(config.DATA.ROOT, "Cityscapes"),
            split="train",
        )
        img_shape = np.array([640, 1280])

    elif config.DATA.DATASET == "kitti":
        data = KITTIRaw(  # type: ignore
            root=os.path.join(config.DATA.ROOT, "KITTI-raw"),
            resize_scale=1.0,
            crop_resolution=(368, 1104),
        )
        img_shape = np.array([368, 1104])
    else:
        raise ValueError("Unknown dataset.")

    # split dataset to run generation in parallel
    splitsize = len(data) // config.DATA.NUM_PREPROCESSING_SUBSPLITS
    if len(data) % splitsize == 0:
        splitsize -= 1
    all_ranges = torch.arange(0, len(data), splitsize).int().tolist()
    all_ranges[-1] = len(data)
    data = Subset(  # type: ignore
        data,
        range(all_ranges[config.DATA.PREPROCESSING_SUBSPLIT - 1], all_ranges[config.DATA.PREPROCESSING_SUBSPLIT]),
    )
    print("Each subset has", len(data), "images.")

    data_loader = DataLoader(
        data,
        collate_fn=lambda x: x[0],
        num_workers=config.SYSTEM.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda:0",
        shuffle=False,
        batch_size=1,
    )

    # semantic segmentation model
    model = DepthG(
        device="cuda:0",
        checkpoint_root=config.MODEL.CHECKPOINT,
        img_shape=img_shape,  # type: ignore
        stride=(int(img_shape[0] // 4), int(img_shape[0] // 4)),
        crop=(int(img_shape[0] // 2), int(img_shape[0] // 2)),
    )

    # thing stuff splitter
    thingstuff_split = ThingStuffSplitter(num_classes_all=model.model.cluster_probe.n_classes)

    # optical flow model
    raft = raft_smurf()
    raft.to("cuda:0", torch.float32)  # type: ignore
    raft.eval()

    # generate path
    pathlib.Path(config.DATA.PSEUDO_ROOT).mkdir(parents=True, exist_ok=True)

    failed_images = []
    with Pool(config.SYSTEM.NUM_WORKERS * 2) as pool:
        for data in tqdm(data_loader):
            # generate name and save path
            img_name = os.path.split(data["image_0_l_path"])[-1][:-4]  # type: ignore
            if config.DATA.DATASET == "kitti":
                img_name = data["image_0_l_path"].split(os.path.sep)[-4] + "_" + img_name  # type: ignore
            semgt_save_path = os.path.join(config.DATA.PSEUDO_ROOT, img_name + "_" + "semantic" + ".png")
            instgt_save_path = os.path.join(config.DATA.PSEUDO_ROOT, img_name + "_" + "instance" + ".png")
            # check if peudo label already exists
            if os.path.isfile(semgt_save_path) and os.path.isfile(instgt_save_path):
                tqdm.write("Already processed: " + str(img_name))
                # skip pseudo label generation but update thingstuff splitter
                sem_pseudo = T.ToTensor()(Image.open(semgt_save_path)).squeeze()
                inst_pseudo = T.ToTensor()(Image.open(instgt_save_path)).squeeze()
                panoptic_pred = torch.stack([sem_pseudo, inst_pseudo], dim=-1).long()
                thingstuff_split.update(panoptic_pred)
                continue

            # Get instance data
            image_0_l = data["image_0_l"].to("cuda:0", torch.float32)  # type: ignore
            image_0_r = data["image_0_r"].to("cuda:0", torch.float32)  # type: ignore
            image_1_l = data["image_1_l"].to("cuda:0", torch.float32)  # type: ignore
            image_1_r = data["image_1_r"].to("cuda:0", torch.float32)  # type: ignore
            valid_pixels = data["valid_pixels"].to("cuda:0")  # type: ignore
            baseline = data["baseline"].to("cuda:0", torch.float32)  # type: ignore
            intrinsics = data["intrinsics"].to("cuda:0", torch.float32)  # type: ignore

            # Make forward passes
            optical_flow_l_forward = raft(image_0_l, image_1_l)
            optical_flow_l_backward = raft(image_1_l, image_0_l)
            disparity_1_forward = raft(image_0_l, image_0_r, disparity=True)
            disparity_2_forward = raft(image_1_l, image_1_r, disparity=True)
            disparity_1_backward = raft(image_0_r, image_0_l, disparity=True, forward=False)
            disparity_2_backward = raft(image_1_r, image_1_l, disparity=True, forward=False)

            try:
                # Get SE(3) object proposals
                object_proposals = get_object_proposals(
                    image_1_l=image_1_l,
                    optical_flow_l_forward=optical_flow_l_forward,
                    optical_flow_l_backward=optical_flow_l_backward,
                    disparity_1_forward=disparity_1_forward,
                    disparity_2_forward=disparity_2_forward,
                    disparity_1_backward=disparity_1_backward,
                    disparity_2_backward=disparity_2_backward,
                    intrinsics=intrinsics,
                    baseline=baseline,
                    valid_pixels=valid_pixels,
                )

            except Exception:
                print(Exception)
                failed_images.append(img_name)
                tqdm.write("Failed for: " + str(img_name))
                object_proposals = torch.zeros(image_0_l.shape[-2], image_0_l.shape[-1]).long()

            img = normalize(image_0_l)
            disp = disparity_1_forward
            fB = data["intrinsics"][0, 0, 0] * data["baseline"][0]  # type: ignore
            depth = fB / (disp.abs() + 1e-10) * disp.sign()
            depth_weight = 1 / (depth + 1)
            out = model.depth_guided_sliding_window(img, depth_weight)

            # apply CRF to semantic segmentation
            cluster_pred = batched_crf(pool, img, out).argmax(1).long()

            # skip if no object proposals
            if object_proposals.max().item() == 0:
                object_proposals = torch.zeros(image_0_l.shape[-2], image_0_l.shape[-1]).long()
            object_proposals[:, :32] = 0
            object_proposals[:, -32:] = 0

            # merge predictions
            panoptic_pred = torch.stack([cluster_pred.squeeze(), object_proposals.cpu()], dim=-1)
            # align semantic class to object proposal
            panoptic_pred[..., 0] = align_semantic_to_instance(
                panoptic_pred[..., 0], panoptic_pred[..., 1].unsqueeze(0)
            )["aligned_semantics"]
            # update thingstuff splitter
            thingstuff_split.update(panoptic_pred)

            # safe pseudo labels and images
            semantic_label = Image.fromarray(np.array(panoptic_pred[..., 0].cpu(), dtype=np.uint8))
            instance_label = Image.fromarray(np.array(panoptic_pred[..., 1].cpu(), dtype=np.uint8))

            # write image and label files
            semantic_label.save(semgt_save_path)
            instance_label.save(instgt_save_path)

    instances_distribution_pixel, instances_distribution_mask, pseudo_class_distribution = thingstuff_split.compute()
    save_data = {
        "distribution all pixels": pseudo_class_distribution,
        "distribution inside object proposals": instances_distribution_pixel,
        "distribution per object proposal": instances_distribution_mask,
    }
    torch.save(
        save_data,
        os.path.join(
            config.DATA.PSEUDO_ROOT, "pseudo_classes_split_" + str(config.DATA.PREPROCESSING_SUBSPLIT) + ".pt"
        ),
    )

    print("Failed images:", failed_images)


if __name__ == "__main__":
    main()
