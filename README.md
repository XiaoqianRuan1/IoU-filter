# Fully Test-time Adaptation for Object Detection
Here is the code for Fully Test-time Adaptation for Object Detection. The detailed example of training VOC dataset -> Clipart dataset.  

## Requirements
The detection framework is inherited from [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and uses Pytorch and CUDA.
Please refer to [INSTALL.md](INSTALL.md) for installation instructions. 

## Implementation details 

We build on top of Faster-RCNN with a ResNet-50 Backbone pre-trained on ImageNet, 300 top proposals after non-maximum-suppression, anchors at three scales (128, 256, 512) and three aspect ratios (1:1, 1:2, 2:1).

During pre-training phase, we train the base network for 70k iterations using SGD with momentum set at 0.9, the initial learning rate is 0.001 and decays after 50k iterations. During pre-training, the batch size is set as 1, batch normalization layers are fixed and the first two blocks are freezen. 

During adaptation phase, we adapt the model for five iterations based on one sample. We use SGD with momentum set at 0.9, and learning rate is set as 0.001. The batch normalization is fixed and the first two blocks are freezen. 

## Datasets
### VOC->Clipart, Comic, Watercolor
Create a folder named `datasets` and include VOC2007 and VOC2012 source datasets (download (http://host.robots.ox.ac.uk/pascal/VOC/)).

Download and extract clipart1k, comic2k and watercolor2k from (https://naoto0804.github.io/cross_domain_detection/).

### Cityscapes->Foggy, Rainy Cityscapes
Download Cityscapes, along with Foggy and Rainy Cityscapes datasets from (https://www.cityscapes-dataset.com/downloads/). 

## Performing pretraining 

To perform a *standard* TTA pretraing using Pascal VOC as source dataset:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain.yaml
```

## Testing pretrained model

You can test a pretrained model based on the target datasets by referring to the correct config-file, for example, if testing on clipart dataset:

```bash
python tools/test_net.py --config-file configs/amd/tta_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

## Performing the test-time adapation

To use TTA procedure and obtain results by referring to the config files. For example for clipart:

```bash
python tools/adapt_net.py --config-file configs/amd/tta_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

## Acknowledge
This project is based on [OSHOT](https://github.com/VeloDC/oshot_detection). 
