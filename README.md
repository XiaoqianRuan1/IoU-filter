# Demo implementation of *Fully Test-time Adaptation for Object Detection*

The detection framework is inherited from [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and uses Pytorch and CUDA.

This readme will guide you through a full run of our method for the Pascal VOC -> AMD benchmarks. 

## Implementation details 

We build on top of Faster-RCNN with a ResNet-50 Backbone pre-trained on ImageNet, 300 top proposals
after non-maximum-suppression, anchors at three scales (128, 256, 512) and three aspect ratios (1:1,
1:2, 2:1).

For OSHOT we train the base network for 70k iterations using SGD with momentum set at 0.9, the
initial learning rate is 0.001 and decays after 50k iterations. We use a batch size of 1, keep 
batch normalization layers fixed for both pretraining and adaptation phases and freeze the first 
2 blocks of ResNet50. The weight of the rotation task is set to λ=0.05.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Datasets

Create a folder named `datasets` and include VOC2007 and VOC2012 source datasets (download from
[Pascal VOC's website](http://host.robots.ox.ac.uk/pascal/VOC/)).

Download and extract clipart1k, comic2k and watercolor2k from [authors'
website](https://naoto0804.github.io/cross_domain_detection/).

## Performing pretraining 

To perform a *standard* OSHOT pretraing using Pascal VOC as source dataset:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain.yaml
```

Once you have performed a pretraining you can test the output model directly on the target domain or
perform the one-shot adaptation.

## Testing pretrained model

You can test a pretrained model on one of the AMD referring to the correct config-file. For example
for clipart:

```bash
python tools/test_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

## Performing the one-shot adaptation

To use OSHOT adaptation procedure and obtain results on one of the AMD please refer to one of the
config files. For example for clipart:

```bash
python tools/oshot_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```
