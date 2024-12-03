# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .meta_generalized_rcnn import meta_generalizedRCNN
#from .pseudo_generalized_rcnn import Pseudo_generalizedRCNN



#_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "Meta_GeneralizedRCNN" : meta_generalizedRCNN,"Pseudo_GeneralizedRCNN":Pseudo_GeneralizedRCNN}
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "Meta_GeneralizedRCNN" : meta_generalizedRCNN}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
