<div align="center">
<h1>Scene-Centric Unsupervised Panoptic Segmentation</h1>


[**Oliver Hahn**](https://olvrhhn.github.io)<sup>* 1</sup>    [**Christoph Reich**](https://christophreich1996.github.io/)<sup>* 1,2,4,5</sup>    [**Nikita Araslanov**](https://arnike.github.io/)<sup>2,4</sup>
[**Daniel Cremers**](https://cvg.cit.tum.de/members/cremers)<sup>2,4,5</sup>   [**Christian Rupprecht**](https://chrirupp.github.io/)<sup>3</sup>  [**Stefan Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>1,5,6</sup>
<h3>CVPR 2025</h3>
<sup>1</sup>TU Darmstadt  <sup>2</sup>TU Munich  <sup>3</sup>University of Oxford  <sup>4</sup>MCML  <sup>5</sup>ELIZA  <sup>6</sup>hessian.AI  *equal contribution

<a href="https://arxiv.org/"><img src='https://img.shields.io/badge/ArXiv-grey' alt='Paper PDF'></a>
<a href="https://arxiv.org/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'></a>
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<center>
    <img src="./assets/cups_video.gif" width="512">
</center>
</div>

**TL;DR:** We present CUPS, a Scene-Centric Unsupervised Panoptic Segmentation method leveraging motion and depth from stereo pairs to generate pseudo-labels. Using these labels, we train a monocular panoptic network, achieving state-of-the-art results across multiple scene-centric benchmarks.

## Abstract
Unsupervised panoptic segmentation aims to partition an image into semantically meaningful regions and distinct object instances without training on manually annotated data. In contrast to prior work on unsupervised panoptic scene understanding, we eliminate the need for object-centric training data, enabling the unsupervised understanding of complex scenes. To that end, we present the first unsupervised panoptic method that directly trains on scene-centric imagery. In particular, we propose an approach to obtain high-resolution panoptic pseudo labels on complex scene-centric data combining visual representations, depth, and motion cues. Utilizing both pseudo-label training and a panoptic self-training strategy yields a novel approach that accurately predicts panoptic segmentation of complex scenes without requiring any human annotations. Our approach significantly improves panoptic quality, e.g., surpassing the recent state of the art in unsupervised panoptic segmentation on Cityscapes by 9.4% points in PQ.

## News
- `28/03/2025`: [ArXiv](https://arxiv.org/) preprint and code released.
- `26/02/2025`: CUPS has been accepted to [CVPR](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)! 🎉 


## Installation
This project was originally developed with Python 3.9, PyTorch 2.1.2, and CUDA 11.8. We used a single NVIDIA A100 (40GB). Create the conda environment as follows:

```bash
# Create conda env
conda create -n cups -c rapidsai -c conda-forge -c nvidia -c pytorch3d -c pytorch  \
    cuml=24.04 python=3.9 cuda-version=11.8 pytorch3d==0.7.6 \
    pytorch==2.1.2  torchvision==0.16.1 pytorch-cuda=11.8 cuda
# Activate conda env
conda activate cups
# Install all additional required packages with pip
pip install -r requirements_new.txt
# Install Detectron2 (currently not needed, should also work on Mac)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Install pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
# Downgrade GNU-C++
conda install gxx_linux-64=11.2.0
# Install connected components CUDA
cd
git clone https://github.com/zsef123/Connected_components_PyTorch
cd Connected_components_PyTorch
python setup.py install
```

## Inference


## Training
### Baseline




## Citation
If you find our work helpful, please consider citing the following paper and ⭐ the repo.

```
@inproceedings{Hahn:2025:UPS,
    title = {Scene-Centric Unsupervised Panoptic Segmentation},
    author = {Oliver Hahn and Christoph Reich and Nikita Araslanov and Daniel Cremers and Christian Rupprecht and Stefan Roth},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025},
}
```
