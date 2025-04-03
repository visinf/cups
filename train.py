import logging
import os
import resource
from argparse import REMAINDER, ArgumentParser
from datetime import datetime
from typing import Any, Dict

import torch.nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode

import cups
from cups.augmentation import (
    CopyPasteAugmentation,
    PhotometricAugmentations,
    ResolutionJitter,
    get_pseudo_label_augmentations,
)
from cups.data import (
    CITYSCAPES_CLASSNAMES,
    CITYSCAPES_CLASSNAMES_19,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_STUFF_CLASSES_19,
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_THING_CLASSES_19,
    CityscapesPanopticValidation,
    KITTIPanopticValidation,
    PseudoLabelDataset,
    StepDataset,
    collate_function_validation,
)
from cups.utils import RTPTCallback

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")


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
        "--disable_wandb",
        default=False,
        action="store_true",
        help="Binary flag. If set run will not be tracked with Weights and Biases.",
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
    if args_dict.pop("disable_wandb"):
        os.environ["WANDB_MODE"] = "disabled"
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
    # Set seed
    seed_everything(config.SYSTEM.SEED)
    # Make datasets
    training_dataset = PseudoLabelDataset(
        root=config.DATA.ROOT,
        root_pseudo=config.DATA.ROOT_PSEUDO,
        return_detectron2_format=True,
        ground_truth_scale=config.DATA.SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        thing_stuff_threshold=config.DATA.THING_STUFF_THRESHOLD,
        ignore_unknown_thing_regions=config.DATA.IGNORE_UNKNOWN_THING_REGIONS,
        augmentations=get_pseudo_label_augmentations(config.DATA.CROP_RESOLUTION),
        dataset=config.DATA.DATASET,
        only_use_non_empty_samples=True,
    )
    # Init validation set
    if config.DATA.DATASET == "cityscapes":
        validation_dataset = CityscapesPanopticValidation(
            root=config.DATA.ROOT_VAL,
            crop_resolution=(512, 1024),  # For efficiency, we validate on half of the original resolution
            num_classes=27,
            resize_scale=0.5,
        )
        # Get set of classes in dataset
        thing_classes = CITYSCAPES_THING_CLASSES
        stuff_classes = CITYSCAPES_STUFF_CLASSES
        class_names = CITYSCAPES_CLASSNAMES
    else:
        validation_dataset = KITTIPanopticValidation(  # type: ignore
            root=config.DATA.ROOT_VAL,
            crop_resolution=(368, 1240),
            num_classes=19,
            resize_scale=1.0,
        )
        # Get set of classes in dataset
        thing_classes = CITYSCAPES_THING_CLASSES_19
        stuff_classes = CITYSCAPES_STUFF_CLASSES_19
        class_names = CITYSCAPES_CLASSNAMES_19
    # Print dataset length
    log.info(f"{len(training_dataset)} training samples and {len(validation_dataset)} validation samples detected.")
    # Make data loaders
    training_data_loader = DataLoader(
        dataset=StepDataset(
            training_dataset, steps=config.TRAINING.STEPS * config.SYSTEM.NUM_GPUS * config.TRAINING.BATCH_SIZE
        ),
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=lambda x: x,
        drop_last=True,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=6,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=collate_function_validation,
        drop_last=False,
        pin_memory=False,
    )
    # Init model
    model: LightningModule = cups.build_model_pseudo(
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_pseudo_classes=training_dataset.things_classes,
        stuff_pseudo_classes=training_dataset.stuff_classes,
        class_weights=(
            tuple(
                (
                    1.0 / (torch.tensor(training_dataset.class_distribution) * len(training_dataset.class_distribution))
                ).tolist()
            )
            if config.TRAINING.CLASS_WEIGHTING
            else None
        ),
        copy_paste_augmentation=(
            CopyPasteAugmentation(
                thing_class=len(training_dataset.stuff_classes),
                max_num_pasted_objects=config.AUGMENTATION.MAX_NUM_PASTED_OBJECTS,
            )
            if config.AUGMENTATION.COPY_PASTE
            else None
        ),
        photometric_augmentation=PhotometricAugmentations(),
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
        class_names=class_names,
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
    os.makedirs(experiment_path, exist_ok=True)
    # Init logger
    logger = WandbLogger(
        name=run_name,
        log_model=False,
        save_dir=experiment_path,
        project="Unsupervised Panoptic Segmentation",
        entity="oliver_and_christoph",
    )
    # Init trainer
    trainer: Trainer = Trainer(
        default_root_dir=experiment_path,
        accelerator=config.SYSTEM.ACCELERATOR,
        devices=config.SYSTEM.NUM_GPUS,
        num_nodes=config.SYSTEM.NUM_NODES,
        strategy=config.SYSTEM.DISTRIBUTED_BACKEND,
        precision=config.TRAINING.PRECISION,
        max_steps=config.TRAINING.STEPS,
        min_steps=config.TRAINING.STEPS,
        callbacks=[
            RTPTCallback(name_initials="CR&OH", experiment_name="UPS"),
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                filename="ups_checkpoint_{step:06d}",
                every_n_train_steps=config.TRAINING.VAL_EVERY_N_STEPS,
                save_last=True,
                save_top_k=-1,
            ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
        ],
        logger=logger,
        log_every_n_steps=config.TRAINING.LOG_EVERT_N_STEPS,
        gradient_clip_algorithm=config.TRAINING.GRADIENT_CLIP_ALGORITHM,
        gradient_clip_val=config.TRAINING.GRADIENT_CLIP_VAL,
        check_val_every_n_epoch=None,
        val_check_interval=config.TRAINING.VAL_EVERY_N_STEPS,
        num_sanity_val_steps=0,
    )
    # Perform training
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )


if __name__ == "__main__":
    main()
