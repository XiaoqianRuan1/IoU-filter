# Fully Test-time Adaptation for Object Detection
Here is the code for Fully Test-time Adaptation for Object Detection. 

## Installation
### Install the dependencies by running
```bash
pip install -r requirements.txt
```
### Install other dependencies step by step
```bash
# Install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# compile and install maskrcnn-module
python setup.py build develop

```
## Data
| Task | Dataset | Link   |
|:----------:|:----------:|:-------------:|
| Training  | VOC2007&VOC2012 | http://host.robots.ox.ac.uk/pascal/VOC/ |
| Adaptation | Clipart1k&Comic2k&Watercolor2k | https://naoto0804.github.io/cross_domain_detection/ |

## Pretraining

To perform a *standard* TTA pretraing using Pascal VOC as source dataset:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain.yaml
```

## Testing

You can test a pretrained model based on the target datasets by referring to the correct config-file, for example, if testing on clipart dataset:

```bash
python tools/test_net.py --config-file configs/amd/tta_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

## Test-time Adaptation

To use TTA procedure and obtain results by referring to the config files. For example for clipart:

```bash
python tools/adapt_net.py --config-file configs/amd/tta_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```
## References
If you find our work helpful, please consider citing our paper.
```bash
@inproceedings{ruan2024fully,
  title={Fully Test-time Adaptation for Object Detection},
  author={Ruan, Xiaoqian and Tang, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1038--1047},
  year={2024}
}
```

## Acknowledge
This project is based on [OSHOT](https://github.com/VeloDC/oshot_detection). 
