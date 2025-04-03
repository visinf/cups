import os
import time
import random
import warnings
from os.path import join
from pathlib import Path

import numpy as np
import torch.multiprocessing
import torchvision
from PIL import Image, ImageFile
import PIL
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from tqdm import tqdm
from torchvision.datasets.utils import download_url
import requests
import shutil
import tarfile
import zipfile
# import h5py
import sys
sys.path.append(os.getcwd())    
from cups.utils import normalize


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


class DirectoryDataset(Dataset):
    def __init__(self, root, path, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        # This was buggy
        # self.dir = join(root, path)
        self.dir = root
        self.img_dir = join(self.dir, "imgs", self.split)
        self.label_dir = join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert len(self.img_files) > 0
        if os.path.exists(join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_fn = self.img_files[index]
        img = Image.open(join(self.img_dir, image_fn))

        if self.label_files is not None:
            label_fn = self.label_files[index]
            label = Image.open(join(self.label_dir, label_fn))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64) - 1

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class Potsdam(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels, return_depth=False, depth_type="zoedepth"):
        super(Potsdam, self).__init__()
        print(f"Using Potsdam dataset with depth_type {depth_type} and split {image_set}. Return depth: {return_depth}")
        self.split = image_set
        self.root = os.path.join(root, "potsdam")
        self.transform = transform
        self.target_transform = target_transform
        self.return_depth = return_depth
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }

        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.filepaths = [join(self.root, "imgs", fn + ".png") for fn in self.files]
        self.depth_type = depth_type
        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        if self.return_depth:
            try:
                # Load depth png
                if self.depth_type == "zoedepth":
                    depth = Image.open(join(self.root, "zoe_depth", self.split, "imgs", image_id + "_zoedepth.png"))
                elif self.depth_type == "kbr":
                    depth = Image.open(join(self.root, "kbr_depth", self.split, image_id + ".png"))
                elif self.depth_type == "gt":
                    # TODO
                    depth = Image.open(join(self.root, "gt_depth", image_id + ".png"))
                else:
                    raise NotImplementedError("Depth type {} not implemented. Available depth types are zoedepth, kbr, midas.".format(self.depth_type))

                # Convert to Torch tensor
                depth = to_tensor(depth)

                if self.depth_type == "gt":
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                elif self.depth_type == "kbr":
                    assert depth.shape[0] == 3, "KBR depth map should have 3 channels"
                    # Input is depth with 3 channels, so we need to convert to 1 channel
                    depth = torch.mean(depth, dim=0).unsqueeze(0)

                    # Min-max normalize
                    depth = (depth - depth.min()) / (depth.max() - depth.min())


                # If depth not normalized, normalize it
                # if depth.max() > 1:
                #    depth = depth / 255
                # print("Found depth file for image {}".format(image_id))

            except FileNotFoundError:
                warnings.warn("Depth file not found for image {}".format(image_id))
                depth = torch.zeros(1, img.height, img.width)
            except PIL.UnidentifiedImageError:
                warnings.warn("Depth file not found for image {}".format(image_id))
                depth = torch.zeros(1, img.height, img.width)
        else:
            depth = torch.zeros(1, img.height, img.width)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        img = self.transform(img)

        depth = self.target_transform(depth).squeeze(0).float()

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask, depth

    def __len__(self):
        return len(self.files)


class PotsdamRaw(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(PotsdamRaw, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdamraw", "processed")
        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for im_num in range(38):
            for i_h in range(15):
                for i_w in range(15):
                    self.files.append("{}_{}_{}.mat".format(im_num, i_h, i_w))

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None, return_depth=False, depth_type="zoedepth"):
        super(Coco, self).__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things
        self.return_depth = return_depth

        print(f"self.return_depth {self.return_depth}")

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val", "demo"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"],
            "demo": ["demo"]
        }

        self.depth_type = depth_type

        self.image_files = []
        self.label_files = []
        if self.return_depth:
            self.depth_files = []

        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))
                    if self.return_depth:
                        if self.depth_type == "zoedepth":
                            self.depth_files.append(join(self.root, "depth", split_dir, img_id + "_zoedepth.png"))
                        elif self.depth_type == "kbr":
                            self.depth_files.append(join(self.root, "kbr_depth", split_dir, img_id + ".png"))
                        elif self.depth_type == "midas":
                            self.depth_files.append(join(self.root, "midas_depth", split_dir, img_id + "_midas.png"))
                        else:
                            raise NotImplementedError("Depth type {} not implemented. Available depth types are zoedepth, kbr, midas.".format(self.depth_type))

        self.filepaths = self.image_files

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
        # TODO: Here, continue to integrate the depth file loading
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        if self.return_depth:
            depth_path = self.depth_files[index]
            depth_map = Image.open(depth_path)

            # print("Depth map show")
            # depth_map.show()
            # print("Post Depth map show")

            # time.sleep(5)

            depth = to_tensor(depth_map)

            if self.depth_type == "kbr":
                # Input is depth with 3 channels, so we need to convert to 1 channel
                depth = torch.mean(depth, dim=0).unsqueeze(0)
                #depth = depth[0].unsqueeze(0)
                # Min-max normalize
                depth = (depth - depth.min()) / (depth.max() - depth.min())

            # print(f"Pre Transform: depth.min() {depth.min()}, depth.max() {depth.max()}")

            # depth = self.label_transform(depth).float().squeeze(0)

            # print(f"Post Transform Depth Show")
            # depth_show = to_pil_image(depth)
            # depth_show.show()

            # time.sleep(5)

            # print(f"Post Transform: depth.min() {depth.min()}, depth.max() {depth.max()}")

        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            if self.return_depth:
                return img, coarser_labels, coarser_labels >= 0, depth
            else:
                return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                if self.return_depth:
                    return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index), depth
                else:
                    return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                if self.return_depth:
                    return img, coarse_label, coarse_label >= 0, depth
                else:
                    return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform, return_depth: bool = False, depth_type="zoedepth", num_classes=27):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = root
        self.id_mapping_19 = torch.Tensor([c.train_id for c in Cityscapes.classes]).long()
        self.num_classes = num_classes
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)

        self.inner_loader.filepaths = [self.inner_loader.images[i] for i in range(len(self.inner_loader))]

        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7
        self.depth_folder_path = os.path.join("/path_to_datasets/depthg_processed_data/cityscapes_depth/leftImg8bit_sequence/", image_set)

        self.return_depth = return_depth
        self.depth_type = depth_type

    def __getitem__(self, index):
        depth = None
        if self.return_depth:
            path = Path(self.inner_loader.filepaths[index])
            filename = path.stem
            subfolder = path.parent.stem
            if self.depth_type == "zoedepth":
                depth_path = join(self.depth_folder_path, subfolder, filename + ".png") #"_zoedepth.png")
            else:
                raise NotImplementedError("Depth type {} not implemented. Available depth types are zoedepth, kbr, midas.".format(self.depth_type))

            depth = Image.open(depth_path)
            depth = to_tensor(depth)

        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)
            if self.num_classes == 19:
                target = self.id_mapping_19[target]
                target[target == 255] = -1
            else:
                target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1

            if self.return_depth:
                return image.squeeze(0), target.squeeze(0), mask.squeeze(0), depth.squeeze(0)
            else:
                return image.squeeze(0), target.squeeze(0), mask.squeeze(0)

        else:

            if self.return_depth:
                image, target = self.inner_loader[index]
                return image, target.squeeze(0), None, depth
            else:
                return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)

class KITTISeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform, return_depth: bool = False, depth_type="zoedepth"):
        super(KITTISeg, self).__init__()
        import sys
        sys.path.append(os.getcwd())
        from cups.data import KITTIRaw
        from cups.data.kitti import KITTIPanopticValidation
        self.image_set = image_set
        if image_set == "train":
            self.inner_loader = KITTIRaw(
                root = root,
                resize_scale = 1.0,
                crop_resolution = (368, 1240),
                temporal_stride = 1
            )
            # adjust dataset to have similar size to CS
            self.inner_loader.images = torch.load(os.path.join(root.replace('KITTI-raw', 'depthg_processed_data/KITTI-raw_depth'), 'kitti_raw_split.pth'))
            self.inner_loader.filepaths = self.inner_loader.images
        elif image_set == "val":
            self.inner_loader = KITTIPanopticValidation( 
                root = root.replace("KITTI-raw", "kitti_panoptic"),
                resize_scale = 0.8695652173913043,
                crop_resolution = (320, 320),
                num_classes=27,
            )

        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

        self.return_depth = return_depth
        self.depth_type = depth_type

    def __getitem__(self, index):
        depth = None
        if self.return_depth:
            depth_path = self.inner_loader.filepaths[index][0].replace("KITTI-raw", 'depthg_processed_data/KITTI-raw_depth')
            depth = Image.open(depth_path)
            depth = to_tensor(depth)

        if self.transform is not None:
            image = self.inner_loader[index]['image_0_l']
            image = normalize(image)
            if self.image_set == "train":
                target = torch.ones_like(image)[:, 0, :, :]*-1
            else:
                target = self.inner_loader[index]['semantic_gt']
                target[target==255] = -1
                
            mask = target == -1

            if self.return_depth:
                return image.squeeze(), target.squeeze(), mask.squeeze(), depth.squeeze()
            else:
                return image.squeeze(), target.squeeze(), mask.squeeze()

        else:

            if self.return_depth:
                image, target = self.inner_loader[index]
                return image.squeeze(), target.squeeze(), None, depth.squeeze()
            else:
                return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class NYUv2(Dataset):
    """
    From https://github.com/xapharius/pytorch-nyuv2/blob/master/nyuv2/nyuv2.py
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(
            self,
            root: str,
            image_set: str = "train",
            download: bool = False,
            transform=None,
            target_transform=None,
            sn_transform=None,
            depth_transform=None,
            return_depth: bool = False,
            depth_type: str = "gt"
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        """
        super().__init__()
        self.root = root

        self.rgb_transform = transform
        self.seg_transform = target_transform
        self.sn_transform = None
        self.depth_transform = depth_transform
        self.return_depth = return_depth

        self.train = image_set == "train"
        self._split = image_set
        if self._split == "val":
            self._split = "test"

        assert self._split in ["train", "test"], f"Invalid split {self._split}"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self._files = sorted(os.listdir(os.path.join(root, f"{self._split}_rgb")))
        self.filepaths = self._files
        self.depth_type = depth_type

    def __getitem__(self, index):
        folder = lambda name: os.path.join(self.root, f"{self._split}_{name}")

        depth = None

        if self.return_depth:
            if self.depth_type == "gt":
                depth = Image.open(os.path.join(folder("depth"), self._files[index]))
            elif self.depth_type == "zoedepth":
                depth = Image.open(os.path.join(folder(f"{self.depth_type}_depth"), self._files[index].replace(".png", f"_{self.depth_type}.png")))
            elif self.depth_type in ["kbr", "midas"]:
                depth = Image.open(os.path.join(folder(f"{self.depth_type}_depth"), self._files[index]))
            else:
                raise NotImplementedError("Depth type {} not implemented. Available depth types are None, zoedepth, kbr, midas.".format(self.depth_type))


            depth = to_tensor(depth)

            if isinstance(depth, torch.Tensor):
                # depth png is uint16
                depth = depth.float() / 1e4

            depth = (depth - depth.min()) / (depth.max() - depth.min())

        if self.rgb_transform is not None:
            image = Image.open(os.path.join(folder("rgb"), self._files[index]))
            target = Image.open(os.path.join(folder("seg13"), self._files[index]))

            if isinstance(target, torch.Tensor):
                # ToTensor scales to [0, 1] by default
                target = (target * 255).long()

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.rgb_transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.seg_transform(target)

            # target = target - self.first_nonvoid
            # target[target < 0] = -1
            # mask = target == -1

            if self.return_depth:
                return image, target.squeeze(0), 0, depth
            else:
                return image, target.squeeze(0), 0

        else:

            if self.return_depth:
                image = Image.open(os.path.join(folder("rgb"), self._files[index]))
                target = Image.open(os.path.join(folder("seg13"), self._files[index]))

                if isinstance(target, torch.Tensor):
                    # ToTensor scales to [0, 1] by default
                    target = (target * 255).long()

                return image, target.squeeze(0), 0, depth
            else:
                image = Image.open(os.path.join(folder("rgb"), self._files[index]))
                target = Image.open(os.path.join(folder("seg13"), self._files[index]))

                if isinstance(target, torch.Tensor):
                    # ToTensor scales to [0, 1] by default
                    target = (target * 255).long()

                return image, target

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                for part, transform in zip(
                        ["rgb", "seg13", "sn", "depth"],
                        [
                            self.rgb_transform,
                            self.seg_transform,
                            self.sn_transform,
                            self.depth_transform,
                        ],
                ):
                    if transform is None:
                        continue
                    path = os.path.join(self.root, f"{split}_{part}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        if self.rgb_transform is not None:
            download_rgb(self.root)
        if self.seg_transform is not None:
            download_seg(self.root)
        if self.sn_transform is not None:
            download_sn(self.root)
        if self.return_depth:
            download_depth(self.root)
        print("Done!")


class PascalVOC(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transforms, target_transforms, return_depth=False, depth_type="zoedepth"):
        super().__init__(root, year=year, image_set=image_set, download=download, transforms=transforms)
        self.target_transforms = target_transforms
        self.return_depth = return_depth
        self.depth_type = depth_type

        if self.return_depth:
            if self.depth_type == "zoedepth":
                depth_path = f"zoe_depth/{image_set}/JPEGImages"
                self.depth = [join(root, depth_path, os.path.basename(image_fn).split('/')[-1].replace(".jpg", "_zoedepth.png")) for image_fn in self.images]
            elif self.depth_type == "kbr":
                depth_path = f"kbr_depth/{image_set}/JPEGImages"
                self.depth = [join(root, depth_path, os.path.basename(image_fn).split('/')[-1].replace(".jpg", ".png")) for image_fn in self.images]
            elif self.depth_type == "midas":
                depth_path = f"midas_depth/{image_set}/JPEGImages"
                self.depth = [join(root, depth_path, os.path.basename(image_fn).split('/')[-1].replace(".jpg", "_midas.png")) for image_fn in self.images]
            else:
                raise NotImplementedError("Depth type {} not implemented. Available depth types are zoedepth, kbr, midas.".format(self.depth_type))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.masks[idx])

        if self.return_depth:
            if self.depth_type == "zoedepth":
                depth = Image.open(self.depth[idx])
            elif self.depth_type == "kbr":
                depth = Image.open(self.depth[idx])
            elif self.depth_type == "midas":
                depth = Image.open(self.depth[idx])
            else:
                raise NotImplementedError("Depth type {} not implemented. Available depth types are zoedepth, kbr, midas.".format(self.depth_type))

            depth = to_tensor(depth)

            # If Resize in transforms, resize to size
            for transform in self.transforms.transforms:
                if isinstance(transform, torchvision.transforms.Resize):
                    depth = resize(depth, (transform.size, transform.size), interpolation=Image.NEAREST)

            if isinstance(depth, torch.Tensor):
                # depth png is uint16
                depth = depth.float() / 1e4

            depth = (depth - depth.min()) / (depth.max() - depth.min())

        if self.transforms is not None:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed);
            image = self.transforms(image)
            self._set_seed(seed);
            label = self.target_transforms(label)
            label[label > 20] = -1

        if self.return_depth:
            return image, label.squeeze(0), label == -1, depth
        else:
            return image, label.squeeze(0), label == -1

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def PascalVOCGenerator(root, image_set, transform, target_transform, return_depth: bool = False, depth_type="zoedepth"):
        return PascalVOC(join(root, "pascalvoc"),
                         year='2012',
                         image_set=image_set,
                         download=False,
                         transforms=transform,
                         target_transforms=target_transform,
                         return_depth=return_depth,
                         depth_type=depth_type)


class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform,
                 return_depth: bool = False, depth_type="zoedepth"):
        super(CroppedDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = image_set

        print(f"Depth type: {depth_type}")
        print(f"Dataset Name: {dataset_name}")

        if depth_type == "gt":
            assert self.dataset_name in ["nyuv2", "potsdam"], "Depth type gt is only available for nyuv2 dataset."
            self.root = join(root, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        if depth_type.__contains__("zoedepth") and self.dataset_name != "nyuv2":
            self.root = join(root, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        elif depth_type == "zoedepth" and self.dataset_name == "nyuv2":
            self.root = join(root, "cropped", "{}_{}_crop_{}_{}".format(dataset_name, crop_type, crop_ratio, depth_type))
        else:
            self.root = join(root, "cropped", "{}_{}_crop_{}_{}".format(dataset_name, crop_type, crop_ratio, depth_type))
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = join(self.root, "img", self.split)
        self.label_dir = join(self.root, "label", self.split)
        self.depth_dir = join(self.root, "depth", self.split)
        self.return_label = True
        if not os.path.exists(self.label_dir):
            warnings.warn("No label directory found, returning only images")
            self.return_label = False


        self.plane_depth = depth_type.__contains__("plane")
        # Remove plane from string
        self.depth_type = depth_type.replace("_plane", "")
        self.num_images = len(os.listdir(self.img_dir))
        self.return_depth = return_depth

        print(f"Image Dir {self.img_dir}")
        print(f"Label Dir {self.label_dir}")

        print(f"Returning Depth in Dataloader: {self.return_depth}")

        print(f"self.num_images {self.num_images}")
        print(f"self.label_dir {len(os.listdir(self.label_dir))}")

        for i in range(self.num_images):
            img_exists = os.path.exists(join(self.img_dir, "{}.jpg".format(i)))
            label_exists = os.path.exists(join(self.label_dir, "{}.png".format(i)))

            if not img_exists or not label_exists:
                print(f"File with index {i} does not exists.")
                print(f"Label exists: {label_exists}")
                print(f"Image exists: {img_exists}")

        print(f"self.num_images {self.num_images}")
        print(f"len(os.listdir(self.label_dir) {len(os.listdir(self.label_dir))}")

        # assert self.num_images == len(os.listdir(self.label_dir))

    def __getitem__(self, index):
        # print("Img path", join(self.img_dir, "{}.jpg".format(index)))
        # print("Label path", join(self.label_dir, "{}.png".format(index)))

        image = Image.open(join(self.img_dir, "{}.jpg".format(index))).convert('RGB')

        if self.return_label:
            target = Image.open(join(self.label_dir, "{}.png".format(index)))
        else:
            target = np.random.randint(0, 255, size=image.size[::-1], dtype=np.uint8)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.target_transform(target)

        if self.return_depth:
            # depth_tensor = torch.load(join(self.label_dir, "{}.pt".format(index)))
            depth = Image.open(join(self.depth_dir, "{}_{}.png".format(index, self.depth_type)))
            depth = self.target_transform(depth).float()  # .squeeze(0) ??

            if self.plane_depth:
                depth = torch.ones_like(depth).float() * 255

        if self.return_label:
            target = target - 1
            mask = target == -1

        if self.return_depth:
            return image, target.squeeze(0), mask, depth
        elif self.return_label:
            return image, target.squeeze(0), mask
        else:
            return image, target, mask

    def __len__(self):
        return self.num_images


class MaterializedDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=12, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None,
                 return_depth: bool = False,
                 depth_type: str = "zoedepth"
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform
        self.return_depth = return_depth

        if dataset_name == "potsdam":
            self.n_classes = 3
            dataset_class = Potsdam
            extra_args = dict(coarse_labels=True, return_depth=return_depth, depth_type=depth_type)
        elif dataset_name == "potsdamraw":
            warnings.warn("Depth cannot be used with potsdamraw class. Ignoring depth.")
            self.n_classes = 3
            dataset_class = PotsdamRaw
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "directory":
            self.n_classes = cfg.dir_dataset_n_classes
            dataset_class = DirectoryDataset
            extra_args = dict(path=cfg.dir_dataset_name)
        elif dataset_name == "cityscapes" and crop_type is None:
            warnings.warn("Depth cannot be used with cityscapes dataset class when crop_type is None. Ignoring depth.")
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict(return_depth=return_depth)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")
        elif dataset_name == "cityscapes_19" and crop_type is None:
            self.n_classes = 19
            print('using 19 classes')
            dataset_class = CityscapesSeg
            extra_args = dict(return_depth=return_depth, num_classes=19)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")

        elif (dataset_name == "cityscapes" or dataset_name == "cityscapes_19") and crop_type is not None:
            if dataset_name == "cityscapes_19":
                self.n_classes = 19
            else:
                self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=cfg.crop_ratio,
                              return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")
        ### KITTI ###
        elif dataset_name == "kitti" and crop_type is None:
            self.n_classes = 27
            dataset_class = KITTISeg
            extra_args = dict(return_depth=return_depth)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")

        elif dataset_name == "kitti" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="kitti", crop_type=crop_type, crop_ratio=cfg.crop_ratio,
                              return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")


        ###




        elif dataset_name == "cocostuff3":
            self.n_classes = 3
            dataset_class = Coco
            extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        elif dataset_name == "cocostuff15":
            self.n_classes = 15
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff27", crop_type=cfg.crop_type, crop_ratio=cfg.crop_ratio,
                              return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")

        elif dataset_name == "cocostuff27" and crop_type is None:
            # Throw a warning that depth cannot be used with this dataset class
            self.n_classes = 27
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False, return_depth=return_depth, depth_type=depth_type)
            if image_set == "val":
                extra_args["subset"] = 7
        elif dataset_name == "nyuv2" and crop_type is not None:
            self.n_classes = 14
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="nyuv2", crop_type=cfg.crop_type, crop_ratio=cfg.crop_ratio,
                              return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")
        elif dataset_name == "nyuv2" and crop_type is None:
            self.n_classes = 14
            dataset_class = NYUv2
            extra_args = dict(return_depth=return_depth, depth_type=depth_type)
        elif dataset_name == "pascalvoc" and crop_type is None:
            self.n_classes = 21
            dataset_class = PascalVOC.PascalVOCGenerator
            extra_args = dict(return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")
        elif dataset_name == "pascalvoc" and crop_type is not None:
            self.n_classes = 21
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="pascalvoc", crop_type=cfg.crop_type, crop_ratio=cfg.crop_ratio,
                              return_depth=return_depth, depth_type=depth_type)
            print(f"DATASET DEBUG: Using {dataset_class} dataset class with crop type {crop_type} and return depth is "
                  f"set to {return_depth}.")
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        dataset_name = dataset_name.split('_')[0]
        nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
        feature_cache_file = join(data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            model_type, nice_dataset_name, image_set, crop_type, cfg.res))
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file) or compute_knns:
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind] 
        
        # pack[0] = pack[0].squeeze()
        # pack[1] = pack[1].squeeze()
        # if pack[2] is not None:
        #     pack[2] = pack[2].squeeze()
        # pack[3] = pack[3].squeeze()
        # Get the image file name

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        coord_entries = torch.meshgrid([torch.linspace(-1, 1, pack[0].shape[1]),
                                        torch.linspace(-1, 1, pack[0].shape[2])])
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        img = extra_trans(ind, pack[0])
        label = extra_trans(ind, pack[1])
        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1])
        }

        if self.return_depth:
            """print(len(pack))
            print(f"pack[3].shape {pack[3].shape}")
            print(f"pack[3].max() {pack[3].max()}")
            print(f"pack[3].mean() {pack[3].mean()}")"""

            # depth = to_pil_image(pack[3])

            # Display depth image with matplotlib
            # depth.show()

            # time.sleep(10)

            ret["depth"] = extra_trans(ind, pack[3])

            """print(f"ret.shape {extra_trans(ind, pack[3])[0].shape}")
            print(f"ret.max() {extra_trans(ind, pack[3])[0].max()}")"""

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

            if self.return_depth:
                ret["depth_pos"] = extra_trans(ind, pack_pos[3])

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        if self.aug_photometric_transform is not None:
            img_aug = self.aug_photometric_transform(self.aug_geometric_transform(pack[0]))

            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)

            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.dropbox.com/s/dn5sxhlgml78l03/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            req = requests.get(url + "?dl=1")  # dropbox
            with open(tar, 'wb') as f:
                f.write(req.content)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(root: str):
    url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 1e4).astype(np.uint16).T
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)
