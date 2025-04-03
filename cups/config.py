import logging
import os
from typing import Any, List, Optional, Tuple

from yacs.config import CfgNode

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_C: CfgNode = CfgNode()

# General system configurations
_C.SYSTEM = CfgNode()
# Set accelerator
_C.SYSTEM.ACCELERATOR = "gpu"
# Number of GPUs to be utilized
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for data loading
_C.SYSTEM.NUM_WORKERS = 16
# Number of nodes to be utilized
_C.SYSTEM.NUM_NODES = 1
# Type of distributed backend to be used
_C.SYSTEM.DISTRIBUTED_BACKEND = "auto"
# Set logging path
_C.SYSTEM.LOG_PATH = "experiments"
# Set seed
_C.SYSTEM.SEED = 1996
# Set run name
_C.SYSTEM.RUN_NAME = None

# Model configurations
_C.MODEL = CfgNode()
# Set if DINO backbone should be used
_C.MODEL.USE_DINO = True
# Set model checkpoint
_C.MODEL.CHECKPOINT = None
# Set inference confidence threshold
_C.MODEL.INFERENCE_CONFIDENCE_THRESHOLD = 0.5
# Set TTA object detection threshold
_C.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD = 0.5
# Set TTA scales
_C.MODEL.TTA_SCALES = (0.5, 0.75, 1.0)

# Dataset configurations
_C.DATA = CfgNode()
# Dataset to be used (currently cityscapes or kitti)
_C.DATA.DATASET = "cityscapes"
# Dataset path
_C.DATA.ROOT = "datasets/Cityscapes"
# Dataset path validation set 1 (used in training for validation)
_C.DATA.ROOT_VAL = "datasets/Cityscapes"
# Pseudo label dataset
_C.DATA.ROOT_PSEUDO = "pseudo_labels"
# Number data splits
_C.DATA.NUM_PREPROCESSING_SUBSPLITS = 2
# Number data splits
_C.DATA.PREPROCESSING_SUBSPLIT = 1
# Dataset path
_C.DATA.PSEUDO_ROOT = "pseudo_labels"
# Number of semantic pseudo classes
_C.DATA.NUM_PSEUDO_CLASSES = 27
# Set crop resolution
_C.DATA.CROP_RESOLUTION = (640, 1280)
# Set thing stuff threshold
_C.DATA.THING_STUFF_THRESHOLD = 0.08
# Set number of classes to be used (27, 19, or 7)
_C.DATA.NUM_CLASSES = 27
# Set training scale
_C.DATA.SCALE = 0.625
# Set validation scale
_C.DATA.VAL_SCALE = 0.625
# Set if thing regions not occupied by an object proposal should be ignored
_C.DATA.IGNORE_UNKNOWN_THING_REGIONS = False

# Training specific config
_C.TRAINING = CfgNode()
# Set number of epochs to be performed
_C.TRAINING.STEPS = 4000
# Define batch size
_C.TRAINING.BATCH_SIZE = 16
# Set training precision to be used
_C.TRAINING.PRECISION = "bf16"
# Set gradient clipping approach
_C.TRAINING.GRADIENT_CLIP_ALGORITHM = "norm"
# Set gradient clipping value
_C.TRAINING.GRADIENT_CLIP_VAL = 1.0
# Set log frequency
_C.TRAINING.LOG_EVERT_N_STEPS = 1
# Set media log frequency
_C.TRAINING.LOG_MEDIA_N_STEPS = 100
# Set validation frequency
_C.TRAINING.VAL_EVERY_N_STEPS = 200
# Set if class weighting should be used
_C.TRAINING.CLASS_WEIGHTING = False
# Set type of optimizer to be used
_C.TRAINING.OPTIMIZER = "adamw"
# SGD specific config
_C.TRAINING.SGD = CfgNode()
# Set learning rate
_C.TRAINING.SGD.LEARNING_RATE = 0.005
# Set weight decay
_C.TRAINING.SGD.WEIGHT_DECAY = 0.00005
# Set Adam decays
_C.TRAINING.SGD.MOMENTUM = 0.9
# ADAMW specific config
_C.TRAINING.ADAMW = CfgNode()
# Set learning rate
_C.TRAINING.ADAMW.LEARNING_RATE = 0.0001
# Set weight decay
_C.TRAINING.ADAMW.WEIGHT_DECAY = 0.00001
# Set Adam decays
_C.TRAINING.ADAMW.BETAS = (0.9, 0.999)
# Set drop loss IoU threshold
_C.TRAINING.DROP_LOSS_IOU_THRESHOLD = 0.4
# Set binary flag if drop loss should be used
_C.TRAINING.DROP_LOSS = True

# Self-training specific config
_C.SELF_TRAINING = CfgNode()
# Round length in training steps
_C.SELF_TRAINING.ROUND_STEPS = 500
# Self-training rounds
_C.SELF_TRAINING.ROUNDS = 3
# Set if drop-loss should be used for self-training
_C.SELF_TRAINING.USE_DROP_LOSS = False
# Set semantic segmentation threshold
_C.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD = 0.5
# Set confidence step for each stage (base confidence + stage * confidence step)
_C.SELF_TRAINING.CONFIDENCE_STEP = 0.05

# Validation specific config
_C.VALIDATION = CfgNode()
# Set if PQ metric should adhere to thing stuff split
_C.VALIDATION.ADHERE_THING_STUFF = True
# Set device to cache labels and prediction on
_C.VALIDATION.CACHE_DEVICE = None
# Set if TTA should be used in validation script
_C.VALIDATION.USE_TTA = False
# Set if CRF and center crop should be used in validation script
_C.VALIDATION.USE_CRF = False
# Smaller image side for center crop
_C.VALIDATION.SEMSEG_CENTER_CROP_SIZE = None

# Augmentation specific config
_C.AUGMENTATION = CfgNode()
# Set if copy-paste augmentation should be used
_C.AUGMENTATION.COPY_PASTE = True
# Set number of pasted objects per image
_C.AUGMENTATION.MAX_NUM_PASTED_OBJECTS = 8
# Set number of epochs copy-paste augmentation should be performed using pseudo labels (else pred. will be used)
_C.AUGMENTATION.NUM_STEPS_STARTUP = 500
# Set confidence for copy-paste predictions
_C.AUGMENTATION.CONFIDENCE = 0.75
# Set resolutions for resolution jitter augmentation
_C.AUGMENTATION.RESOLUTIONS = (
    (384, 768),
    (416, 832),
    (448, 896),
    (480, 960),
    (512, 1024),
    (544, 1088),
    (576, 1152),
    (608, 1216),
    (640, 1280),
    (672, 1344),
    (704, 1408),
)

# Pseudo label generation specific config
_C.PSEUDOS = CfgNode()
# Which type of semantic pseudo label generation
_C.PSEUDOS.SEMANTIC_TYPE = ["vanilla", "dguided"][-1]
# Which type of instance pseudo label generation
_C.PSEUDOS.INSTANCE_TYPE = ["vanilla_se3", "ours_se3"][-1]
# Which type of motion for pseudo label generation
_C.PSEUDOS.MOTION_TYPE = ["raft", "smurf"][-1]
# Align semantic class to object proposal
_C.PSEUDOS.NOT_ALIGN_SEMANTIC_TO_INSTANCE_MASK = False

# Label efficient learning specific config
_C.SUPERVISED = CfgNode()
# Set dataset to be used
_C.SUPERVISED.DATASET = "cityscapes_5_0"
# Align semantic class to object proposal
_C.SUPERVISED.ONLY_TRAIN_HEADS = True

__all__: Tuple[str, ...] = ("get_default_config",)


def get_default_config(
    experiment_config_file: Optional[str] = None,
    command_line_arguments: Optional[List[Any]] = None,
) -> CfgNode:
    """Loads config object.

    Args:
        experiment_config_file (Optional[str]): If given config file is used to overwrite the default config.
        command_line_arguments Optional[List[Any]]: Optional command line arguments, overwrites experiment config.

    Returns:
        config (CfgNode): Config object.
    """
    # Get default config
    config = _C.clone()
    # Load and merge experiment config from file
    if experiment_config_file is not None:
        assert isinstance(
            experiment_config_file, str
        ), f"Experiment config file must be a string but {type(experiment_config_file)} given."
        assert os.path.exists(experiment_config_file), f"File {experiment_config_file} does not exists."
        config.merge_from_file(experiment_config_file)
        log.info(f"Experiment config file {experiment_config_file} loaded.")
    if command_line_arguments is not None:
        assert isinstance(
            command_line_arguments, list
        ), f"Command line arguments must be a list [keys, values, ...] but type {type(command_line_arguments)} given."
        config.merge_from_list(command_line_arguments)
        log.info("Command line arguments loaded.")
    # Freeze config
    config.freeze()
    return config
