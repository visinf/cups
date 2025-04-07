<div align="center">
<h1>Scene-Centric Unsupervised Panoptic Segmentation</h1>


[**Oliver Hahn**](https://olvrhhn.github.io)<sup>* 1</sup>    [**Christoph
Reich**](https://christophreich1996.github.io/)<sup>* 1,2,4,5</sup>    [**Nikita
Araslanov**](https://arnike.github.io/)<sup>2,4</sup>
[**Daniel Cremers**](https://cvg.cit.tum.de/members/cremers)<sup>2,4,5</sup>   [**Christian
Rupprecht**](https://chrirupp.github.io/)<sup>3</sup>  [**Stefan
Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>1,5,6</sup>


<sup>1</sup>TU Darmstadt  <sup>2</sup>TU Munich  <sup>3</sup>University of Oxford  <sup>4</sup>MCML  <sup>5</sup>
ELIZA  <sup>6</sup>hessian.AI  *equal contribution
<h3>CVPR 2025 Highlight</h3>


<a href="https://arxiv.org/abs/2504.01955"><img src='https://img.shields.io/badge/ArXiv-grey' alt='Paper PDF'></a>
<a href="https://visinf.github.io/cups/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'></a>
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


<center>
    <img src="./assets/cups_video.gif" width="512">
</center>
</div>

**TL;DR:** We present CUPS, a Scene-Centric Unsupervised Panoptic Segmentation method leveraging motion and depth from
stereo pairs to generate pseudo-labels. Using these labels, we train a monocular panoptic network, achieving
state-of-the-art results across multiple scene-centric benchmarks.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scene-centric-unsupervised-panoptic/unsupervised-panoptic-segmentation-on)](https://paperswithcode.com/sota/unsupervised-panoptic-segmentation-on?p=scene-centric-unsupervised-panoptic) 

## Abstract

Unsupervised panoptic segmentation aims to partition an image into semantically meaningful regions and distinct object
instances without training on manually annotated data. In contrast to prior work on unsupervised panoptic scene
understanding, we eliminate the need for object-centric training data, enabling the unsupervised understanding of
complex scenes. To that end, we present the first unsupervised panoptic method that directly trains on scene-centric
imagery. In particular, we propose an approach to obtain high-resolution panoptic pseudo labels on complex scene-centric
data combining visual representations, depth, and motion cues. Utilizing both pseudo-label training and a panoptic
self-training strategy yields a novel approach that accurately predicts panoptic segmentation of complex scenes without
requiring any human annotations. Our approach significantly improves panoptic quality, *e.g.*, surpassing the recent
state of the art in unsupervised panoptic segmentation on Cityscapes by 9.4% points in PQ.

## News

- `03/04/2025`: [ArXiv](https://arxiv.org/abs/2504.01955) preprint and code released. 🚀
- `26/02/2025`: CUPS has been accepted to [CVPR](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)! 🎉

## Installation

This project was originally developed with Python 3.9, PyTorch 2.1.2, and CUDA 11.8 on Linux. Create the conda
environment as follows:

```bash
# Clone the repository
git clone https://github.com/visinf/cups
# Move into repository
cd cups
# Create conda environment
conda create -n cups -c conda-forge -c nvidia -c pytorch3d -c pytorch  \
    python=3.9 cuda-version=11.8 pytorch3d==0.7.6 \
    pytorch==2.1.2 torchvision==0.16.1 pytorch-cuda=11.8 \
    nvidia/label/cuda-11.8.0::cuda nvidia/label/cuda-11.8.0::cuda-nvcc gxx_linux-64=11.2.0
# Activate conda env
conda activate cups
# Install additional required packages with pip
pip install -r requirements.txt
# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Install pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
# Install connected components CUDA
pip install git+https://github.com/zsef123/Connected_components_PyTorch.git
```

## Checkpoints

We demonstrate that CUPS outperforms the recent state of the art (U2Seg) across multiple scene-centric benchmarks in panoptic quality.
We provide our final checkpoint (after pseudo-label training and self-training) with 27 pseudo-classes below.
In case you would like to use different checkpoints, feel free to reach out to us or open an issue.

**Table 1.** Comparing CUPS to the previous SOTA in unsupervised panoptic segmentation using panoptic quality metric (PQ) in %.
<table><tbody>
<th valign="bottom">Method</th>
<th valign="bottom">Checkpoint</th>
<th valign="bottom">Cityscapes</th>
<th valign="bottom">KITTI</th>
<th valign="bottom">BDD</th>
<th valign="bottom">MUSES</th>
<th valign="bottom">Waymo</th>
<th valign="bottom">MOTS</th>
<tr><td align="center">Prev. SOTA</td>
<td align="center">-</td>
<td align="center">18.4</td>
<td align="center">20.6</td>
<td align="center">15.8</td>
<td align="center">20.3</td>
<td align="center">19.8</td>
<td align="center">50.7</td>
</tr>
<tr><td align="center"><b>CUPS</b></td>
<td align="center"><a href="https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4532/cups.ckpt">download</a></td>
<td align="center">27.8</td>
<td align="center">25.5</td>
<td align="center">19.9</td>
<td align="center">24.4</td>
<td align="center">26.4</td>
<td align="center">67.8</td>
</tr>
</tbody></table>

You can also use `wget` to download the CUPS checkpoint.

```bash
# Download checkpoint
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4532/cups.ckpt
```

All related files and checkpoints can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4532).

## Inference

You can use our model and checkpoint as follows:

```python
import torchvision
from cups.model import panoptic_cascade_mask_r_cnn_from_checkpoint

# Init model
model, num_clusters_things, num_clusters_stuffs = panoptic_cascade_mask_r_cnn_from_checkpoint(
    path="your_favorite_checkpoint_path",
    device="cpu",
    confidence_threshold=0.5,
)
# Model into evaluation mode
model.eval()
# Load an image
image = torchvision.io.read_image("assets/stuttgart_02_000000_005445_leftImg8bit.png").float() / 255.0
# Perform inference
prediction = model([{"image": image}])
```

Note our implementation takes in an image with pixel values within 0 to 1. Additionally, note that our CUPS model's raw
output semantics are not aligned with the ground truth semantic of Cityscapes.

**For a full inference example with alignment and visualization, please refer to our demo script ([demo.py](demo.py)).**

## Training

Training CUPS requires three stages: (1) pseudo-label generation, (2) training on the pseudo-labels, and (3)
self-training. We will describe how to execute every step individually. We performed training on four NVIDIA A100
(40GB) GPUs. You can train CUPS using fewer GPUs. Please set the respective parameters in the
[config file](cups/config.py).

### Prerequisites

For training CUPS, you need to download three checkpoints: (1) SMURF, (2) DepthG (w/ unsupervised depth), and (3)
ResNet-50 DINO backbone. These checkpoints are available in the TUdatalib project. You need to put the SMURF checkpoint
into [`cups/optical_flow/checkpoints`](cups/optical_flow/checkpoints) and the ResNet-50 DINO checkpoint into
[`cups/model/backbone_checkpoints`](cups/model/backbone_checkpoints). All required checkpoints can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4532).
You can download them manually or using wget.

```bash
# To SMURF checkpoint folder
cd cups/optical_flow/checkpoints
# Download SMURF checkpoint
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4532/raft_smurf.pt
# Back to repository root
cd ../../../
# To backbone checkpoint folder
cd cups/model/backbone_checkpoints
# Download ResNet-50 DINO backbone checkpoint
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4532/dino_RN50_pretrain_d2_format.pkl
```

The DepthG checkpoint does not need to be in a specific folder; you need to set the path to the checkpoint manually (see pseudo-label generation).

```bash
# To checkpoint folder
cd your_favorite_checkpoint_path
# Download DepthG checkpoint
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4532/depthg.ckpt
```

Note that the provided SMURF checkpoint is obtained from
the [official TensorFlow checkpoint](https://github.com/google-research/google-research/blob/master/smurf/README.md) and
converted using [this repository](https://github.com/ChristophReich1996/SMURF). The DepthG checkpoint was generated by
executing the original [DepthG](https://github.com/leonsick/depthg) original using unsupervised depth. Following U2Seg and
CutLER, we initialize our panoptic model using a DINO pre-trained backbone. The respective checkpoint is from the official
[CutLER](https://github.com/facebookresearch/CutLER/tree/main) repository and is published under
the [CC BY-NC-SA 4.0 license](https://github.com/facebookresearch/CutLER/blob/main/LICENSE).

Pseudo-label training and self-training uses Weights & Biases. If you want to disable logging in Weights & Biases, just
set the flag `--disable_wandb`.

### Pseudo-Label Generation

To generate pseudo labels for the Cityscapes or KITTI datasets, adjust the paths as needed and run:
```bash
cups/pseudo_labels/pseudolabel_gen.sh
```
Note this script requires the path to the DepthG checkpoint (`depthg.pt`).

Before running the script, specify your dataset path and the output path for pseudo labels in `cups/pseudo_labels/pseudolabel_gen.sh`. You also need to set the flag `--MODEL.CHECKPOINT` and provide the path to the DepthG checkpoint.

The script splits the dataset and runs multiple processes in parallel. We used 16 processes for parallelization.
Make sure to adjust the following parameters accordingly:

- `--DATA.NUM_PREPROCESSING_SUBSPLITS` (number of total runs)
- `--DATA.PREPROCESSING_SUBSPLIT` (id of specific run)

### Pseudo-label Training

After generating the pseudo-labels, you can train a monocular panoptic segmentation network by executing the following
script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train.py \
    --experiment_config_file configs/train_cityscapes.yaml \
    SYSTEM.NUM_GPUS 4 \
    DATA.ROOT "your_favorite_dataset_path/Cityscapes" \
    DATA.ROOT_VAL "your_favorite_dataset_path/Cityscapes" \
    DATA.ROOT_PSEUDO "your_favorite_pseudo_label_dataset_path/Cityscapes" \
    SYSTEM.LOG_PATH "experiments" \
    DATA.THING_STUFF_THRESHOLD 0.08
```

Add your dataset and pseudo-label path to the command. We trained using 4 GPUs; however, you can also train
using fewer GPUs by changing the respective parameters. Training takes around 6 to 7 hours on 4 GPUs.

### Self-Training

After the pseudo-label training, you can further enhance performance using our self-training. The self-training can be
executed using the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore train_self.py \
    --experiment_config_file configs/train_self_cityscapes.yaml \
    SYSTEM.NUM_GPUS 4 \
    DATA.ROOT "your_favorite_dataset_path/Cityscapes" \
    DATA.ROOT_VAL "your_favorite_dataset_path/Cityscapes" \
    MODEL.CHECKPOINT "your_favorite_checkpoint_path/ups_checkpoint_step=004000.ckpt" \
    SYSTEM.LOG_PATH "experiments"
```

Again, add your dataset and pseudo-label path to the command and make sure to also use your desired checkpoint
from the pseudo-label training for self-training. Checkpoints and other logs are stored in the path provided with
`SYSTEM.LOG_PATH`. Self-training on 4 GPUs takes about 4 hours.

## Validation

To run panoptic validation on Cityscapes, execute the following command:

```bash
# Run panoptic Cityscapes validation
CUDA_VISIBLE_DEVICES=0 python -W ignore val.py \
    --experiment_config_file "configs/val_cityscapes.yaml" \
    MODEL.CHECKPOINT "your_favorite_checkpoint_path/cups.ckpt" \
    DATA.ROOT "your_favorite_dataset_path/Cityscapes" \
    DATA.NUM_CLASSES 19 \
    MODEL.INFERENCE_CONFIDENCE_THRESHOLD 0.5
```

For validation on other datasets, just change the experiment config file (*e.g.*
, `--experiment_config_file "configs/val_kitti.yaml"` for KITTI Panoptic). All validation configs can be
found [here](configs). You can also change the number of semantic categories you want to evaluate on (7, 19, and 27
categories). If you want to visualize the predictions, you can add the flag `--visualize_results`.

<details>

<summary>Semantic Validation</summary>

### Semantic Validation

To run semantic validation on Cityscapes, execute the following command:

```bash
# Run semantic Cityscapes validation
CUDA_VISIBLE_DEVICES=0 python -W ignore val.py \
    --experiment_config_file "configs/val_cityscapes.yaml" \
    MODEL.CHECKPOINT "your_favorite_checkpoint_path/cups.ckpt" \
    DATA.ROOT "your_favorite_dataset_path/Cityscapes" \
    DATA.NUM_CLASSES 27 \
    MODEL.INFERENCE_CONFIDENCE_THRESHOLD 0.5 \
    VALIDATION.SEMSEG_CENTER_CROP_SIZE 320
```

We validate the semantic performance on 27 semantic categories and on a center crop with a resolution of 320x320,
following STEGO.

</details>

## Datasets

Here, we provide instructions on downloading and preparing the datasets for training and validation. For training CUPS only Cityscapes is required. Note: to download both the Cityscapes dataset, you must agree to their license terms by
opening an account on their websites.

<details>

<summary>Cityscapes (required for training)</summary>

### Cityscapes

To download the Cityscapes dataset, execute the following commands:

```bash
# Go to your dataset folder
cd your_favorite_dataset_path
# Make Cityscapes folder
mkdir Cityscapes
# Go to Cityscapes folder
cd Cityscapes
# Log into your Cityscapes account (credit to https://github.com/cemsaz/city-scapes-script)
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
# Download Cityscapes sequences (left and right images)
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=14
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=15
# Download Cityscapes labels
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
# Download camera calibration data
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=8
# Unzip all data
unzip leftImg8bit_sequence_trainvaltest.zip
unzip rightImg8bit_sequence_trainvaltest.zip
unzip gtFine_trainvaltest.zip
unzip camera_trainvaltest.zip
```

Note to add your username and password. You also need to set the path to which you would like to download the dataset.
If you do not have an account you can get your account on the [Cityscapes website](https://www.cityscapes-dataset.com).

These commands download the Cityscapes sequences (left and right frames), the ground panoptic labels (used only for
validation), and the camera parameters.

</details>

<details>

<summary>KITTI Panoptic</summary>

### KITTI Panoptic

To download the KITTI Panoptic dataset, execute the following commands:

```bash
# Go to your dataset folder
cd your_favorite_dataset_path
# Download the KITTI panoptic dataset
wget http://panoptic.cs.uni-freiburg.de/static/dataset/KITTI-panoptic-segmentation-dataset.zip?
# Unzip the dataset
unzip KITTI-panoptic-segmentation-dataset.zip
# Convert the instance mask to the Cityscapes format by using the provided script
cd kitti_panoptic
python convert_to_cityscapes_instance_format.py
```

</details>

<details>

<summary>BDD-10K</summary>

### BDD-10K

To download the BDD-10K panoptic dataset, execute the following commands:

```bash
# Go to your dataset folder
cd your_favorite_dataset_path
# Download images
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip
# Download panoptic labels
wget https://dl.cv.ethz.ch/bdd100k/data/bdd100k_pan_seg_labels_trainval.zip
# Unzip data
unzip 100k_images_val.zip
unzip bdd100k_pan_seg_labels_trainval.zip
```

</details>

<details>

<summary>MUSES</summary>

### MUSES

The MUSES dataset needs to be downloaded manually from the [official website](https://muses.vision.ee.ethz.ch/download).
It is required to download both the images (RGB_Frame_Camera_trainvaltest) and panoptic annotations
(Panoptic_Annotations_trainval) to your dataset folder. When downloaded both files use `unzip` to unpack the data.

</details>

<details>

<summary>Waymo</summary>

### Waymo

Download the Waymo Open Dataset via the [official website](https://waymo.com/open/).
We only use the **validation split**. Please ensure you download **version 1.4.0**, as corrupted annotations from previous versions have been fixed in this release.

```bash
# Install gsutils following https://cloud.google.com/storage/docs/gsutil_install
# Download validation split to /save/path/
gsutil -m cp -r  gs://waymo_open_dataset_v_1_4_0/individual_files/validation /your_save_path/
```
For convenience, we convert the Waymo dataset to image files for easier use in PyTorch.
Create the following environment and run the script:
```bash
# Create environment for waymo data processing
    conda create -n convert_waymo python==3.10
    pip install waymo-open-dataset-tf-2-12-0==1.6.4
    conda activate convert_waymo
```
Finally, adjust the data paths in the following script and run it as follows:
```bash
python -W ignore cups/data/download_preprocess_waymo.py
```

</details>

<details>

<summary>MOTS</summary>

### MOTS

To download the MOTS dataset, execute the following commands:

```bash
# Go to your dataset folder
cd your_favorite_dataset_path
# Download data
wget https://www.vision.rwth-aachen.de/media/resource_files/MOTSChallenge.zip
# Unzip MOTSChallenge dataset
unzip MOTSChallenge.zip
```

</details>

## Citation

If you find our work helpful, please consider citing the following paper and ⭐ the repo.

```
@inproceedings{Hahn:2025:UPS,
    title={Scene-Centric Unsupervised Panoptic Segmentation},
    author={Oliver Hahn and Christoph Reich and Nikita Araslanov and Daniel Cremers and Christian Rupprecht and Stefan Roth},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025},
}
```

## Contribute

If you want to contribute to this repository, please use [pre-commit](https://pre-commit.com/) and check your
code using [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html).

Just install the pre-commit hooks by
running `pre-commit install`. For running pre-commit manually on all files, simply run `pre-commit run --all-files`.
Sometimes it is required to run `pre-commit autoupdate` to get pre-commit to work.

Typing can be checked manually by using [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html). For example,
just run `mypy train.py --config-file mypy.ini --no-incremental --cache-dir=/dev/null` for type-checking the training
script and all subsequent code. For all files, just replace `mypy train.py` with `mypy .`. For Windows, use the
option `--cache-dir=nul` instead to not save any mypy cache.

## License

Our code is provided under the [Apache-2.0 license](LICENSE). Note some parts of the code (Panoptic Cascade Mask R-CNN)
is derived from the [U2Seg repository](https://github.com/u2seg/U2Seg). This code is also published under
the [Apache-2.0 license](https://github.com/u2seg/U2Seg?tab=Apache-2.0-1-ov-file#readme) and is heavily build
upon [Detectron2](https://github.com/facebookresearch/detectron2) (also released under
the [Apache-2.0 license](https://github.com/u2seg/U2Seg?tab=Apache-2.0-1-ov-file#readme)). The SMURF large checkpoint is
obtained from
the [original checkpoint from the authors](https://github.com/google-research/google-research/blob/master/smurf/README.md)
that is also published
under [Apache-2.0 license](https://github.com/google-research/google-research/blob/master/LICENSE). The ResNet-50 DINO
checkpoints from CutLER is published under
the [CC BY-NC-SA 4.0 license](https://github.com/facebookresearch/CutLER/blob/main/LICENSE).

## Acknowledgements

We thank [Leonhard Sommer](https://lmb.informatik.uni-freiburg.de/people/sommerl/) for open-sourcing
the [SF2SE3 code](https://github.com/lmb-freiburg/sf2se3). We acknowledge the authors of U2Seg, CutLER, STEGO, and
DepthG for open-sourcing their implementations. We also thank all contributors of libraries
like [PyTorch](https://github.com/pytorch/pytorch/graphs/contributors), [timm](https://github.com/huggingface/pytorch-image-models/graphs/contributors),
and [Kornia](https://github.com/kornia/kornia/graphs/contributors) without their efforts this research would not be
feasible.
