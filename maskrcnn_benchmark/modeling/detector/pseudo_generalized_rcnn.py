import torch
import random
from maskrcnn_benchmark.structures.bounding_box import BoxList
from ..self_supervision_scramble import SelfSup_Scrambler
from maskrcnn_benchmark.structures.image_list import to_image_list
from .generalized_rcnn import GeneralizedRCNN

class Pseudo_GeneralizedRCNN(GeneralizedRCNN):
    def __init__(self,cfg):
        super(GeneralizedRCNN).__init__(cfg)
        self.student_model = GeneralizedRCNN
        self.teacher_model = GeneralizedRCNN

    def forward(self,images,targets=None,branch="semi_supervised"):
        if branch=="semi_supervised":
            images = to_image_list(images)
            features = self.teacher_model.backbone(images.tensor)
            results = self.teacher_model.obtain_pseudo_labels(images,features)
            return results
        elif branch=="supervised":
            images = to_image_list(images)
            features = self.teacher_model.backbone(images)
            straight_features = features
            rotated_img_features = {0:straight_features}
            rotated_images = {0:images}
            for rot_i in range(1,4):
                rot_images = []
                for img,img_size in zip(images.tensors,images.image_sizes):
                    rot_image,rot_index = SelfSup_Scrambler.rotate_single(img,rot_i)
                    rot_images.append(rot_image)
                stacked_tensor = torch.stack(rot_images)
                r_features = self.student_model.backbone(stacked_tensor)
                rotated_img_features[rot_i] = r_features
                rotated_images[rot_i] = to_image_list(rot_images)
            proposals,proposal_losses = self.student_model.rpn(images,features,targets)
            if self.roi_heads:
                x,result,detector_losses = self.student_model.roi_heads(features,proposals,targets)
            else:
                x = features
                result = proposals
                detector_losses = {}
            losses = {}
            if self.cfg.MODEL.SELF_SUPERVISOR.REGIONS=="detections" or ((self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "targets") and targets is None):
                test_result = self.teacher_model.obtain_pseudo_labels(images,features)
            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS=="targets":
                test_result = targets
            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS=="images":
                image_sizes = images.image_sizes
                test_result = []
                for height,width in image_sizes:
                    xmin = 0
                    ymin = 0
                    xmax = width
                    ymax = height
                    bbox = torch.tensor([[xmin,ymin,xmax,ymax]],dtype=torch.float)
                    boxlist = BoxList(bbox,(width,height))
                    boxlist = boxlist.to(images.tensors.device)
                    test_result.append(boxlist)
            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS=="crop":
                image_sizes = images.image_sizes
                test_result = []
                for height,width in image_sizes:
                    xmin,ymin,xmax,ymax = self.random_crop_image(width,height)
                    bbox = torch.tensor([[xmin,ymin,xmax,ymax]],dtype=torch.float)
                    boxlist = BoxList(bbox,(width,height))
                    boxlist = boxlist.to(images.tensors.device)
                    test_result.append(boxlist)
            rotated_regions = {0:test_result}
            for rot_i in range(1,4):
                r_result = [res[::] for res in test_result]
                for idx,box_list in enumerate(r_result):
                    rotated_boxes = box_list.transpose(rot_i+1)
                    r_result[idx] = rotated_boxes
                rotated_regions[rot_i] = r_result
            pooling_res = []
            rot_target_batch = []
            for idx_in_batch in range(len(test_result)):
                mul = 1
                rot_target = torch.ones((len(test_result[idx_in_batch])*mul),dtype=torch.long)
                for r in range(len(test_result[idx_in_batch])):
                    rot = random.randint(0,3)
                    features_r = rotated_img_features[rot]
                    regions_r = rotated_regions[rot][idx_in_batch][[r]]
                    l_regions_r = [regions_r]
                    pooled_features = self.student_model.region_feature_extractor(features_r,l_regions_r)
                    pooled_features = self.student_model.ss_adaptive_pooling(pooled_features)
                    pooled_features = pooled_features.view(pooled_features.size(0),-1)
                    class_preds = self.student_model.ss_classifier(self.student_model.ss_dropout(pooled_features))
                    pooling_res.append(class_preds)
                    rot_target[r] = rot 
                rot_target_batch.append(rot_target)
                
            if len(pooling_res)>0:
                pooling_res = torch.stack(pooling_res).squeeze(dim=1)
                rot_target_batch = torch.cat(rot_target_batch).to(pooling_res.device)
                aux_loss = self.student_model.ss_criterion(pooling_res,rot_target_batch)
                aux_loss = aux_loss.mean()
                losses["aux_loss"] = aux_loss
        
        if self.training:
            if targets is not None:
                losses.update(detector_losses)
                losses.update(proposal_losses)
            self.global_step+=1
            return losses