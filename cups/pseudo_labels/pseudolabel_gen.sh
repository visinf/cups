#!/bin/bash
num_workers=6
num_subsplits=2

CUDA_VISIBLE_DEVICES=0 python ups/pseudo_labels/gen_pseudo_labels.py --DATA.DATASET "cityscapes" --MODEL.CHECKPOINT "/path_to_semantic_segmentation_model_checkpoint/depthg.ckpt" --DATA.NUM_PREPROCESSING_SUBSPLITS $num_subsplits --DATA.PREPROCESSING_SUBSPLIT 1 --DATA.PSEUDO_ROOT "/path_to_save_pseudo_labels/" --SYSTEM.NUM_WORKERS $num_workers &
CUDA_VISIBLE_DEVICES=0 python ups/pseudo_labels/gen_pseudo_labels.py --DATA.DATASET "cityscapes" --MODEL.CHECKPOINT "/path_to_semantic_segmentation_model_checkpoint/depthg.ckpt" --DATA.NUM_PREPROCESSING_SUBSPLITS $num_subsplits --DATA.PREPROCESSING_SUBSPLIT 2 --DATA.PSEUDO_ROOT "/path_to_save_pseudo_labels/" --SYSTEM.NUM_WORKERS $num_workers &
# Wait for all background processes to complete
wait

echo "ALL SCRIPTS FINISHED!"
