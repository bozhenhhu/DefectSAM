# DefectSAM
Segment Anything in Defect Detection, this is the official repository for DefectSAM.

![img](./imgs/DefectSAM-model.png)



## Installation
1. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/), and torchvision.
2. Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```


## Get Started
Download the [model checkpoint](https://drive.google.com/file/d/1VX8O7R7UCUg8In9SShLxK1lVRi97luEf/view?usp=sharing) and place it at e.g., `weights/defect_vit_b`

## Dataset
We released more thermal data and artificial annotations in the Releases of this repository.

Contents of the released thermal defect detection database:

public

 ├── plane_0: One version of the thermal defect detection dataset that was publicly released. Each sample is uniquely identified by a name and is stored in mat format.
 
 ├── plane_1: One of the latest iterations of the thermal defect detection dataset is housed in this section. All samples in this release are categorized as flat-type specimens, each bearing a distinct name.
 
└── labels: Within this folder reside the labels corresponding to the samples found in the "plane" or "plane_0_public_history" directories. The labels are of two types: segmentation ground truth files, denoted by the .png extension, and box labels, indicated by the .json extension. These labels are associated with the sample names. For instance, for a sample named "0_20200615_1.mat" in the "plane" directory, its label can be found in either "labels/0_20200615_1.png" or "labels/0_20200615_1.json".


The generation of JSON labels is facilitated through the utilization of the Labelme tool. In laboratory experiments where equipment remains stationary, each mat file containing a series of frames is considered to represent a single ground truth. Annotating these files involves the collaboration of three experienced human annotators, who annotate the original thermal image sequences or images processed using PCA independently. Initial processing of the mat files through PCA enhances the depth of defect information. Subsequently, Labelme is employed to label the PCA-processed images, resulting in the creation of JSON files. The segmentation ground truth files are binary-valued.


### License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)


## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)
- We highly thank Jun Ma, etc., for making the source code of MedSAM [paper](https://arxiv.org/abs/2304.12306), [code](https://github.com/bowang-lab/MedSAM)

## Reference
```
@misc{hu2023segment,
      title={Segment Anything in Defect Detection}, 
      author={Bozhen Hu and Bin Gao and Cheng Tan and Tongle Wu and Stan Z. Li},
      year={2023},
      eprint={2311.10245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
