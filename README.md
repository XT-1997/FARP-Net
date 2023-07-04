# TMM2023-FARP-Net: Local-Global Feature Aggregation and Relation-Aware Proposals for 3D Object Detection

This is a [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) implementation of the paper "FARP-Net: Local-Global Feature Aggregation and Relation-Aware Proposals for 3D Object Detection". 

## Prerequisites
The code is tested with Python3.7, PyTorch == 1.10, CUDA == 11.3, mmdet3d == 1.0.0rc2, mmcv_full == 1.5.0 and mmdet == 2.24.1. We recommend you to use anaconda to make sure that all dependencies are in place. Note that different versions of the library may cause changes in results.


**Step 1.** Create a conda environment and activate it.
```
conda create --name pt1.10.v1 python=3.7
conda activate pt1.10.v1
```

**Step 2.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) following the instruction [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

**Step 3.** Prepare SUN RGB-D Data following the procedure [here](https://github.com/open-mmlab/mmdetection3d/tree/master/data/sunrgbd).

## Getting Started

### for sunrgbd
```shell
sh tools/slurm_train.sh $PARTION $JOB_NAME configs/A2FRPG/A2FRPG_16x8_sunrgbd-3d-10class.py $WORK_DIR
```
### for scannet-1x-backbone
```shell
sh tools/slurm_train.sh $PARTION $JOB_NAME configs/configs/A2FRPG/A2FRPG_8x8_scannet-3d-18class.py $WORK_DIR
```

### for scannet-2x-backbone
```shell
sh tools/slurm_train.sh $PARTION $JOB_NAME configs/configs/A2FRPG/A2FRPG_8x8_scannet-3d-18class-2x.py $WORK_DIR
```
### for test the pretrained weight
```shell
sh tools/slurm_test.sh $PARTION $JOB_NAME configs/A2FRPG/A2FRPG_16x8_sunrgbd-3d-10class.py $PRETRAINED_CKPT --eval mAP --work-dir $WORK_DIR
```

## Main Results

### SUNRGB-D
| name      | Lr schd | mAP@0.25 | Download |
|-----------|---------|----------|----------|
| A2FRPGNet |   3x      | 64.1   | [model](https://drive.google.com/file/d/14SYx_D2YV0sWWjJpxmP-NdQ-82bQTFIp/view?usp=sharing) \| [log](https://drive.google.com/file/d/16DN1kH4llMfLtUpXuTb3DUIVPwnTnjaO/view?usp=sharing)     |

### ScanNet
| name      | Lr schd | backbone | mAP@0.25 | Download |
|-----------|---------|----------|---------|----------|
| A2FRPGNet | 3x      | 1x    | 69.1   | [model](https://drive.google.com/file/d/11Vj1Zq5jD-oUZQgesNnNRePdWR34WdvW/view?usp=sharing) \| [log](https://drive.google.com/file/d/1IuQx13g0XPT0Rytv5Gg725rL6jY2fio2/view?usp=sharing)     |
| A2FRPGNet | 3x      | 2x     | 70.9    | [model](https://drive.google.com/file/d/1rAXetejETb4Og3kggC55y78zYmjAGwdC/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ocrebuZ0kbV49DTFQiMVZjoP2DDYJV1E/view?usp=sharing)     |

## Bibtex
If this repo is helpful for you, please consider to cite it. Thank you! :)

```bibtex
@article{xie2023farp,
  title={FARP-Net: Local-Global Feature Aggregation and Relation-Aware Proposals for 3D Object Detection},
  author={Xie, Tao and Wang, Li and Wang, Ke and Li, Ruifeng and Zhang, Xinyu and Zhang, Haoming and Yang, Linqi and Liu, Huaping and Li, Jun},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```

