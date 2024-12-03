from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry

StageSpec = namedtuple(
    "StageSpec",
    [
        "index",
        "block_count",
        "return_features",
    ])

VGG16StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 2, False), (2, 3, False), (3, 3, False),(4, 3, True)))
    
VGG16StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 2, False), (2, 3, False), (3, 3, True)))

class VGG16(nn.Module):
    def __init__(self,cfg):
        super(VGG16,self).__init__()
        stem_module = _STEM_MODULES[cfg.MODEL.VGG16.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        self.stem = stem_module(cfg)
        in_channels = cfg.MODEL.VGG16.STEM_OUT_CHANNELS
        out_channels = in_channels * 2
        
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            module = _make_stage(in_channels, out_channels, stage_spec.block_count)
            in_channels = out_channels
            out_channels = out_channels*2
            self.add_module(name,module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
    
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)
    
    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return 
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem
            else:
                m = getattr(self, "layer"+str(stage_index))
        for p in m.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs

class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()
        self.conv1 = Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_func(64)
        self.conv2 = Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_func(64)
        
        for l in [self.conv1,self.conv2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        return x

class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
          cfg, norm_func=FrozenBatchNorm2d
        )

class Stage1Base(nn.Module):
    def __init__(self, in_channel, out_channel, norm_func):
        super(Stage1Base, self).__init__()
        self.conv1 = Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_func(out_channel)
        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight,a=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        return x

class Stage2Base(nn.Module):
    def __init__(self, channels, norm_func):
        super(Stage2Base, self).__init__()
        self.conv1 = Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_func(channels)
        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight,a=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        return x       

class Stage3Base(nn.Module):
    def __init__(self):
        super(Stage3Base, self).__init__()
    
    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        return x

class Stage1WithFixedBatchNorm(Stage1Base):
    def __init__(self, in_channels, out_channels):
        super(Stage1WithFixedBatchNorm, self).__init__(
            in_channels,
            out_channels,
            norm_func=FrozenBatchNorm2d
        )
        
class Stage2WithFixedBatchNorm(Stage2Base):
    def __init__(self, channels):
        super(Stage2WithFixedBatchNorm, self).__init__(
          channels,
          norm_func=FrozenBatchNorm2d
        )

def _make_stage(in_channels, out_channels, block_count):
    blocks = []
    stage1 = Stage1WithFixedBatchNorm(in_channels, out_channels)
    stage2 = Stage2WithFixedBatchNorm(out_channels)
    stage3 = Stage3Base()
    blocks.append(stage1)
    for _ in range(block_count-1):
        blocks.append(stage2)
    blocks.append(stage3)
    return nn.Sequential(*blocks)

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
})

_STAGE_SPECS = Registry({
    "VGG16-C4": VGG16StagesTo4,
    "VGG16-C5": VGG16StagesTo5,
})