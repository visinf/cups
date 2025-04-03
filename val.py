import logging
import os
from argparse import REMAINDER, ArgumentParser
from datetime import datetime
from typing import Any, Dict

import torch.nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode

import cups
from cups.data import (
    CITYSCAPES_CLASSNAMES,
    CITYSCAPES_CLASSNAMES_7,
    CITYSCAPES_CLASSNAMES_19,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_STUFF_CLASSES_7,
    CITYSCAPES_STUFF_CLASSES_19,
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_THING_CLASSES_7,
    CITYSCAPES_THING_CLASSES_19,
    KITTI_INSTANCE_STUFF_CLASSES,
    KITTI_INSTANCE_THING_CLASSES,
    MOTS,
    MOTS_STUFF_CLASSES,
    MOTS_THING_CLASSES,
    WAYMO_7_MISSING_CS_CLASSES,
    WAYMO_19_MISSING_CS_CLASSES,
    BDD10kPanopticValidation,
    CityscapesPanopticValidation,
    KITTIInstanceSegmentation,
    KITTIPanopticValidation,
    MUSESPanopticValidation,
    WaymoPanopticValidation,
    collate_function_validation,
)
from cups.utils import RTPTCallback

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

VISUALIZE_RESULTS = False


def configure() -> CfgNode:
    """Function loads default config, experiment config, and parses command line arguments.

    Returns:
        config (CfgNode): Config object.
    """
    # Manage command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        type=str,
        help="Sets the visible cuda devices.",
    )
    parser.add_argument(
        "config",
        help="Modify config options using the command-line",
        default=None,
        nargs=REMAINDER,
    )
    parser.add_argument(
        "--enable_wandb",
        default=False,
        action="store_true",
        help="Binary flag. If set run will be tracked with Weights and Biases.",
    )
    parser.add_argument(
        "--visualize_results",
        default=False,
        action="store_true",
        help="Binary flag. If set predictions and labels will be visualized (stored).",
    )
    parser.add_argument("--experiment_config_file", default=None, type=str, help="Path to experiment config file.")
    # Get arguments
    args = parser.parse_args()
    # Arguments to dict
    args_dict: Dict[str, Any] = vars(args)
    # Set cuda devices
    if (cuda_visible_devices := args_dict.pop("cuda_visible_devices")) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Disable Weights and Biases
    if not args_dict.pop("enable_wandb"):
        os.environ["WANDB_MODE"] = "disabled"
    # Get if results should be visualized
    global VISUALIZE_RESULTS
    VISUALIZE_RESULTS = args_dict.pop("visualize_results")
    # Get path to experiment config file
    experiment_config_file = args_dict.pop("experiment_config_file")
    # Load config
    config: CfgNode = cups.get_default_config(
        experiment_config_file=experiment_config_file, command_line_arguments=args.config
    )
    return config


def main() -> None:
    # Get config
    config: CfgNode = configure()
    # Print config
    log.info(config)
    # define evaluation batch size
    eval_batch_size = 4
    # Make dataset
    if config.DATA.DATASET == "cityscapes":
        log.info("Cityscapes dataset used.")
        validation_dataset = CityscapesPanopticValidation(
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        classes_mask = None
    elif config.DATA.DATASET == "bdd":
        log.info("BDD-10k dataset used.")
        validation_dataset = BDD10kPanopticValidation(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        classes_mask = None
    elif config.DATA.DATASET == "muses":
        log.info("MUSES dataset used.")
        validation_dataset = MUSESPanopticValidation(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        classes_mask = None
    elif config.DATA.DATASET == "waymo":
        log.info("WAYMO dataset used.")
        validation_dataset = WaymoPanopticValidation(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        classes_mask = WAYMO_19_MISSING_CS_CLASSES if config.DATA.NUM_CLASSES == 19 else WAYMO_7_MISSING_CS_CLASSES
    elif config.DATA.DATASET == "kitti_instance":
        log.info("KITTI instance dataset used.")
        validation_dataset = KITTIInstanceSegmentation(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
        )
        classes_mask = None
    elif config.DATA.DATASET == "mots":
        log.info("MOTSChallange dataset used.")
        validation_dataset = MOTS(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        eval_batch_size = 1
        classes_mask = None
    else:
        log.info("KITTI dataset used.")
        validation_dataset = KITTIPanopticValidation(  # type: ignore
            root=config.DATA.ROOT,
            resize_scale=config.DATA.VAL_SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            num_classes=config.DATA.NUM_CLASSES,
        )
        classes_mask = None
    # Print dataset length
    log.info(f"{len(validation_dataset)} validation samples detected.")
    # Make data loaders
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )
    # Get set of classes in dataset
    if config.DATA.DATASET == "kitti_instance":
        thing_classes = KITTI_INSTANCE_THING_CLASSES
        stuff_classes = KITTI_INSTANCE_STUFF_CLASSES
        class_names = ["background", "object"]
    elif config.DATA.DATASET == "mots":
        thing_classes = MOTS_THING_CLASSES
        stuff_classes = MOTS_STUFF_CLASSES
        class_names = CITYSCAPES_CLASSNAMES_7
    else:
        if config.DATA.NUM_CLASSES == 27:
            thing_classes = CITYSCAPES_THING_CLASSES
            stuff_classes = CITYSCAPES_STUFF_CLASSES
            class_names = CITYSCAPES_CLASSNAMES
        elif config.DATA.NUM_CLASSES == 19:
            thing_classes = CITYSCAPES_THING_CLASSES_19
            stuff_classes = CITYSCAPES_STUFF_CLASSES_19
            class_names = CITYSCAPES_CLASSNAMES_19
        else:
            thing_classes = CITYSCAPES_THING_CLASSES_7
            stuff_classes = CITYSCAPES_STUFF_CLASSES_7
            class_names = CITYSCAPES_CLASSNAMES_7
    # Init model
    model: LightningModule = cups.build_model_pseudo(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        copy_paste_augmentation=torch.nn.Identity(),
        resolution_jitter_augmentation=torch.nn.Identity(),
        photometric_augmentation=torch.nn.Identity(),
        use_tta=config.VALIDATION.USE_TTA,
        class_names=class_names,
        classes_mask=classes_mask,
    )
    # Print model
    log.info(model)
    # Init experiments folder since W&B otherwise warns and uses temp
    os.makedirs(
        os.path.join(os.getcwd() if config.SYSTEM.LOG_PATH is None else config.SYSTEM.LOG_PATH, "experiments"),
        exist_ok=True,
    )
    if config.SYSTEM.RUN_NAME is not None:
        run_name = config.SYSTEM.RUN_NAME
    else:
        run_name = "pseudo_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_path = os.path.join(
        os.getcwd() if config.SYSTEM.LOG_PATH is None else config.SYSTEM.LOG_PATH,
        "experiments",
        run_name,
    )
    os.makedirs(experiment_path)
    # Init logger
    logger = WandbLogger(
        name=run_name,
        save_dir=experiment_path,
        project="Unsupervised Panoptic Segmentation",
        log_model="all",
        entity="oliver_and_christoph",
    )
    # Init trainer
    trainer: Trainer = Trainer(
        default_root_dir=experiment_path,
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        num_nodes=config.SYSTEM.NUM_NODES,
        strategy=config.SYSTEM.DISTRIBUTED_BACKEND,
        precision=32,  # Validation is always done in full precision
        callbacks=[
            RTPTCallback(name_initials="CR&OH", experiment_name="UPS"),
            TQDMProgressBar(refresh_rate=1),
        ],
        logger=logger,
        log_every_n_steps=config.TRAINING.LOG_EVERT_N_STEPS,
        num_sanity_val_steps=0,
    )
    # Load checkpoint and validate
    trainer.validate(model=model, dataloaders=validation_data_loader)
    # We validate a second time using the previously computed matching and produce plots
    if VISUALIZE_RESULTS:
        log.info("Doing a second validation loop to visualize the results.")
        model.panoptic_quality.reset()
        model.plot_validation_samples = True
        trainer.validate(model=model, dataloaders=validation_data_loader)


if __name__ == "__main__":
    main()
