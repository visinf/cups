import os

import numpy as np
from PIL import Image


# preprocess waymo dataset for easy handling in pytorch
def preprocess():

    DATA_SPLIT = "validation"
    SAVE_TO_PATH = "/save_waymo_data_to"
    RAW_DATA_PATH = "/downloaded_waymo_data"
    SAVE_CALIBRATION = False

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import camera_segmentation_utils

    # generate folder for saving
    save_to_path_dataset = os.path.join(SAVE_TO_PATH, "waymo_preprocessed_new2", DATA_SPLIT)
    os.makedirs(save_to_path_dataset, exist_ok=True)

    camera_left_to_right_order = [
        open_dataset.CameraName.SIDE_LEFT,
        open_dataset.CameraName.FRONT_LEFT,
        open_dataset.CameraName.FRONT,
        open_dataset.CameraName.FRONT_RIGHT,
        open_dataset.CameraName.SIDE_RIGHT,
    ]

    if DATA_SPLIT == "notr":
        # read text file as list
        notr_txt_path = "external/emernerf/data/waymo_train_list.txt"
        with open(notr_txt_path, "r") as f:
            notr_list = f.readlines()
        notr_list = [x.strip() for x in notr_list]
        # paths to all waymo scenes
        waymo_train_paths = [
            os.path.join(RAW_DATA_PATH, "training", d)
            for d in os.listdir(os.path.join(RAW_DATA_PATH, "training"))
            if ".tfrecord" in d
        ]
        waymo_val_paths = [
            os.path.join(RAW_DATA_PATH, "validation", d)
            for d in os.listdir(os.path.join(RAW_DATA_PATH, "validation"))
            if ".tfrecord" in d
        ]
        waymo_all_paths = waymo_train_paths + waymo_val_paths
        # notr with waymo data
        scene_paths = []
        for scene in notr_list:
            for path in waymo_all_paths:
                if scene in path:
                    scene_paths.append(path)
    else:
        dataset_split_path = os.path.join(RAW_DATA_PATH, DATA_SPLIT)
        scene_paths = [os.path.join(dataset_split_path, d) for d in os.listdir(dataset_split_path) if ".tfrecord" in d]

    # Iterate through scenes and process frames
    for scene_idx, scene_path in enumerate(scene_paths):
        print("Scene: ", scene_idx, " of ", len(scene_paths) - 1)
        # Iter through frames
        with tf.device("/cpu:0"):
            scene = tf.data.TFRecordDataset(scene_path, compression_type="")
            for frame_idx, data in enumerate(scene):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if not frame.images[0].camera_segmentation_label.panoptic_label == b"":
                    # create folder for scene
                    save_to_path_scene = os.path.join(
                        save_to_path_dataset, os.path.basename(scene_path).replace(".tfrecord", "")
                    )
                    os.makedirs(save_to_path_scene, exist_ok=True)

                    segmentation_proto_dict = {image.name: image.camera_segmentation_label for image in frame.images}
                    segmentation_protos_flat = [segmentation_proto_dict[name] for name in camera_left_to_right_order]
                    (
                        panoptic_labels,
                        _,  # num_cameras_covered,
                        _,  # is_tracked_masks,
                        panoptic_label_divisor,
                    ) = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
                        segmentation_protos_flat, remap_to_global=True
                    )
                    # Separate the semantic and instance labels from the panoptic labels.
                    NUM_CAMERA_FRAMES = 5
                    semantic_labels_multiframe = []
                    instance_labels_multiframe = []
                    for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
                        semantic_labels = []
                        instance_labels = []
                        for j in range(NUM_CAMERA_FRAMES):
                            (
                                semantic_label,
                                instance_label,
                            ) = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                                panoptic_labels[i + j], panoptic_label_divisor
                            )
                            semantic_labels.append(semantic_label)
                            instance_labels.append(instance_label)
                        semantic_labels_multiframe.append(semantic_labels)
                        instance_labels_multiframe.append(instance_labels)
                    # Save image and labels
                    img = np.array(tf.image.decode_jpeg(frame.images[0].image), dtype=np.uint8)
                    semantic_label = np.array(semantic_labels_multiframe[0][2].squeeze(), dtype=np.uint16)
                    instance_label = np.array(instance_labels_multiframe[0][2].squeeze(), dtype=np.uint16)
                    Image.fromarray(img).save(os.path.join(save_to_path_scene, str(frame_idx) + "_" + "image" + ".png"))
                    Image.fromarray(semantic_label).save(
                        os.path.join(save_to_path_scene, str(frame_idx) + "_" + "semantic" + ".png")
                    )
                    Image.fromarray(instance_label).save(
                        os.path.join(save_to_path_scene, str(frame_idx) + "_" + "instance" + ".png")
                    )
                else:
                    # Skip frames without panoptic labels
                    pass

                if SAVE_CALIBRATION:
                    # save camera intrinsics and extrinsics
                    camera_calibrations = np.array(
                        list(frame.context.camera_calibrations[0].intrinsic)
                        + list(frame.context.camera_calibrations[0].extrinsic.transform)
                    )
                    np.save(
                        camera_calibrations,
                        os.path.join(save_to_path_scene, str(frame_idx) + "_" + "camera_calibration.npy"),
                    )


if __name__ == "__main__":
    preprocess()
