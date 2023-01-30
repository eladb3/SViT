# Bringing Image Scene Structure to Video via Frame-Clip Consistency of Object Tokens

This is an official pytorch implementation of the paper [Bringing Image Scene Structure to Video via Frame-Clip Consistency of Object Tokens](https://arxiv.org/abs/2206.06346). In this repository, we provide the PyTorch code we used to train and test our proposed SViT model.

If you find SViT useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@inproceedings{
avraham2022bringing,
title={Bringing Image Scene Structure to Video via Frame-Clip Consistency of Object Tokens},
author={Elad Ben Avraham and Roei Herzig and Karttikeya Mangalam and Amir Bar and Anna Rohrbach and Leonid Karlinsky and Trevor Darrell and Amir Globerson},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=0JV4VVBsK6a}
}
```

# Model Zoo

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url | config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SViT | SSv2 | 16 | 224 | 69.7 | 91.7 | [model](https://drive.google.com/file/d/1ZGHCkHwRfa1Nu36P_Y0RVFP631U6-s3j/view?usp=share_link) | `configs/ssv2.yaml` |

We start from K400 pretrain, provided [here](https://drive.google.com/file/d/1zfWLs78K55Dye2USUZ62SRgL3SA1xLO0/view?usp=share_link).


# Installation

First, create a conda virtual environment and activate it:
```
conda create -n svit python=3.8.5 -y
source activate svit
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- matplotlib: `pip install matplotlib`
- pandas: `pip install pandas`
- ffmeg: `pip install ffmpeg-python`

OR:

simply create conda environment with all packages just from yaml file:

`conda env create -f environment.yml`

Lastly, build the SViT codebase by running:
```
git clone https://github.com/eladb3/SViT.git
cd SViT
python setup.py build develop
```

# Usage

## Dataset Preparation

Instructions to achieve the SSv2 data (and boxes) can be found in [here](https://github.com/eladb3/ORViT/blob/master/slowfast/datasets/DATASET.md).

## Training the SViT

Training the default SViT that operates on 16-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg path_to_config \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset 
```


## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg path_to_config \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False 
```


# Acknowledgements

SViT is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast) and [MViT](https://github.com/facebookresearch/mvit). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```

```BibTeX
@inproceedings{fan2021multiscale,
  title={Multiscale vision transformers},
  author={Fan, Haoqi and Xiong, Bo and Mangalam, Karttikeya and Li, Yanghao and Yan, Zhicheng and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={ICCV},
  year={2021}
}
```

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
