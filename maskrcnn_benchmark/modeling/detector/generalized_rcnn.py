# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..self_supervision_scramble import SelfSup_Scrambler
import random
from ..roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ..roi_heads.box_head.roi_box_predictors import make_roi_box_predictor
from ..roi_heads.box_head.inference import PostProcessor
import tensorboardX
from ...data.transforms.transforms import DeNormalize
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL.ImageDraw import Draw
from torchvision.transforms import ToPILImage

import cv2, copy

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.global_step = 0

        self.region_feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone.out_channels)
        self.region_roi_predictor = make_roi_box_predictor(cfg,self.region_feature_extractor.out_channels)
        self.region_post_processor = PostProcessor(score_thresh=-0.1,nms=0.0,detections_per_img=1000)

        self.cfg = cfg
        self.summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

        self.ss_criterion = nn.CrossEntropyLoss(reduction='none')
        self.ss_adaptive_pooling = nn.AdaptiveAvgPool2d(1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        if cfg.MODEL.SELF_SUPERVISED and cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
            self.ss_dropout = nn.Dropout(p=cfg.MODEL.SELF_SUPERVISOR.DROPOUT) 
            self.ss_classifier = nn.Linear(self.region_feature_extractor.out_channels, 4)

    def _scale_back_image(self, img):
        orig_image = img.numpy()
        t1 = np.transpose(orig_image, (1, 2, 0))
        transform1 = DeNormalize(self.cfg.INPUT.PIXEL_MEAN, self.cfg.INPUT.PIXEL_STD)
        orig_image = transform1(t1)
        orig_image = orig_image.astype(np.uint8)
        orig_image = np.transpose(orig_image, (2, 0, 1))

        return orig_image[::-1,:,:]

    def _log_image_tensorboard(self, image_list, targets, rotation):

        image_tensor = image_list.tensors[0].cpu()
        image_size = image_list.image_sizes[0]

        targets = targets[0]

        # fix size
        image = image_tensor[:image_size[1], :image_size[0]]

        image = self._scale_back_image(image)

        result = image.copy()

        result = self._overlay_boxes(result, targets)

        self.summary_writer.add_image('img_{}'.format(rotation), np.transpose(result,(2,0,1)), global_step=self.global_step)

    def _compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        if not labels.dtype == torch.int64:
            self.palette = self.palette.float()
        colors = labels[:, None] * self.palette.to(labels.device)
        colors = (colors % 255).cpu().numpy().astype("uint8")
        return colors

    def _overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        if predictions.has_field("labels"):
            labels = predictions.get_field("labels")
        else:
            labels = torch.ones(len(predictions.bbox), dtype=torch.float)
        boxes = predictions.bbox

        colors = self._compute_colors_for_labels(labels).tolist()

        pil_image = ToPILImage()(np.transpose(image, (1,2,0)))
        draw = Draw(pil_image)
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

            draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=tuple(color), width=2)

        del draw
        return np.array(pil_image)

    def random_crop(self, feature, crop_size):
        w, h = feature.size()[2:]
        th, tw = crop_size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        crop = feature[:,:,j:j+tw,i:i+th]
        return crop

    def random_crop_image(self, image_width, image_height, crop_size=237):

        crop_size = min(image_width, image_height, crop_size)

        xmin = random.randint(0, image_width - crop_size)
        ymin = random.randint(0, image_height -crop_size)

        xmax = xmin + crop_size
        ymax = ymin + crop_size
        return xmin, ymin, xmax, ymax        

    @torch.no_grad()
    def obtain_pseudo_labels(self, images, features):
        self.eval()

        test_proposals, _ = self.rpn(images, features)
        _, pseudo_targets, _ = self.roi_heads(features, test_proposals)

        #if len(pseudo_targets[0])==0:
        #    print("No pseudo targets!")
        self.train()
        return pseudo_targets
        #return test_proposals

    @torch.no_grad()
    def process_pseudo_label_uncertainty(self,proposals_rpn,device):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_box in proposals_rpn:
            proposal_box = self.bbox_with_uncertainty(proposal_box,device)
            num_proposal_output += len(proposal_box)
            list_instances.append(proposal_box)
        num_proposal_output = num_proposal_output/len(proposals_rpn)
        if len(list_instances[0]) == 0:
            return None,0
        return list_instances,num_proposal_output
    
    def generate_auged_boxes(self,proposal_rpn):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_box in proposal_rpn:
            #filtered_proposal = self.filter_proposals(proposal_box,threshold)
            #print(filtered_proposal)
            proposal_box = self.aug_roi(proposal_box)
            #print(proposal_box)
            num_proposal_output += len(proposal_box)
            list_instances.append(proposal_box)
            #filter_instances.append(filtered_proposal)
        num_proposal_output = num_proposal_output/len(proposal_rpn)
        if len(list_instances[0]) == 0:
            return None,0
        return list_instances,num_proposal_output
    
    def compute_with_uncertainty(self,auged_classes,auged_bboxes,results):
        labels = results[0].get_field("labels")
        scores = results[0].get_field("scores")
        boxes = results[0].bbox
        boxes = boxes.reshape(1,boxes.shape[0],boxes.shape[-1])
        auged_classes = auged_classes.reshape(1,auged_classes.shape[0],auged_classes.shape[-1])
        auged_bboxes = auged_bboxes.reshape(1,auged_bboxes.shape[0],auged_bboxes.shape[-1])
        reg_channel = max([bbox.shape[-1] for bbox in auged_bboxes])//4 
        bboxes = [bbox.reshape(10,-1,bbox.shape[-1]) 
        if bbox.numel()>0
        else bbox.new_zeros(10,0,4*reg_channel()).float()
        for bbox in auged_bboxes
        ]
        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        #print(bboxes[0].shape)
        #print("!!!!!!!!!!!!!!bboxes!!!!!!!!!!!")
        if reg_channel != 1:
            bboxes = [bbox.reshape(bbox.shape[0],reg_channel,4)[torch.arange(bbox.shape[0]),label]
            for bbox,label in zip(bboxes,labels)]
                
            box_unc = [unc.reshape(unc.shape[0],reg_channel,4)[torch.arange(unc.shape[0]),label]
            for unc,label in zip(box_unc,labels)]
        box_shape = [(bbox[:,2:4]-bbox[:,:2]).clamp(min=1.0) for bbox in bboxes]
        box_unc = [unc/wh[:,None,:].expand(-1,2,2).reshape(-1,4)
        if wh.numel()>0
        else unc
        for unc,wh in zip(box_unc,box_shape)]
        new_boxes = [torch.cat([box,unc],dim=1) for box,unc in zip(boxes,box_unc)]
        #print(new_boxes[0].shape)
        #print("!!!!!!!!!!uncertainty!!!!!!!!!!!!!!")
        return new_boxes,labels,scores
    
    def filter_invalid(self,bbox, label=None, mean=None, score=None, thr=0.0, min_size=0):
        bbox = bbox[0]
        mean = mean[0]
        if mean is not None:
            valid = mean> thr
            bbox = bbox[valid]
            if label is not None:
                label = label[valid]
            if score is not None:
                score = score[valid]
        if min_size is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            if label is not None:
                label = label[valid]
            if score is not None:
                score = score[valid]
        return bbox, label, score
    
    def generate_pseudo(self,bbox,label,score,proposal):
        image_shape = proposal.size
        new_proposals = BoxList(bbox,image_shape)
        new_proposals.add_field("labels",label)
        new_proposals.add_field("scores",score)
        return new_proposals
    
    def generate_pseudo_result(self,new_proposals):
        list_instances = []
        num_proposal_output = 0.0
        list_instances.append(new_proposals)
        num_proposal_output+=len(new_proposals)
        num_proposal_output = num_proposal_output/1
        if len(list_instances[0]) == 0:
            return None,0
        return list_instances,num_proposal_output
    
    def aug_box(self,proposal,times=10,frac=0.06):
        boxes = proposal.bbox
        scores = proposal.get_field("objectness")
        box_scale = boxes[:,2:4]-boxes[:,:2]
        box_scale = (box_scale.clamp(min=1)[:,None,:].expand(-1,2,2).reshape(-1,4))
        aug_scale = box_scale * frac
        
        offset = (torch.randn(times,boxes.shape[0],4,device=boxes.device)*aug_scale[None,...])
        new_box = boxes.clone()[None,...].expand(times,boxes.shape[0],-1)
        new_boxes =  torch.cat([new_box[:,:,:4].clone() + offset,new_box[:,:,4:]],dim=-1)
        new_scores = scores.clone()[None].expand(times,scores.shape[0])
        print(new_scores.shape)
        auged_bboxes = new_boxes.reshape(-1,new_boxes.shape[-1])
        auged_scores = new_scores.reshape(-1)
        
        image_shape = proposal.size
        #print(new_boxes.shape) #times,nums,4;
        new_proposals = BoxList(auged_bboxes,image_shape)
        #new_proposals.add_field("scores",new_scores)
        new_proposals.add_field("objectness",auged_scores)
        return new_proposals

    def aug_roi(self,proposal,times=10,frac=0.06):
        boxes = proposal.bbox
        scores = proposal.get_field("scores")
        labels = proposal.get_field("labels")
        box_scale = boxes[:,2:4]-boxes[:,:2]
        box_scale = (box_scale.clamp(min=1)[:,None,:].expand(-1,2,2).reshape(-1,4))
        aug_scale = box_scale * frac
        
        offset = (torch.randn(times,boxes.shape[0],4,device=boxes.device)*aug_scale[None,...])
        new_box = boxes.clone()[None,...].expand(times,boxes.shape[0],-1)
        new_boxes =  torch.cat([new_box[:,:,:4].clone() + offset,new_box[:,:,4:]],dim=-1)
        new_scores = scores.clone()[None].expand(times,scores.shape[0])
        new_labels = labels.clone()[None].expand(times,labels.shape[0])
        auged_bboxes = new_boxes.reshape(-1,new_boxes.shape[-1])
        auged_scores = new_scores.reshape(-1)
        auged_labels = new_labels.reshape(-1)
        
        image_shape = proposal.size
        #print(new_boxes.shape) #times,nums,4;
        new_proposals = BoxList(auged_bboxes,image_shape)
        #new_proposals.add_field("scores",new_scores)
        new_proposals.add_field("scores",auged_scores)
        new_proposals.add_field("labels",auged_labels)
        return new_proposals

    def bbox2point(self,box):
        min_x,min_y,max_x,max_y = torch.split(box[:,:4],[1,1,1,1],dim=1)
        return torch.cat([min_x,min_y,max_x,max_y,min_x,max_y],dim=1).reshape(-1,2)
    
    def points2bbox(self,point,max_w,max_h):
        point = point.reshape(-1, 1, 2)
        if point.size()[0] > 0:
            min_xy = point.min(dim=1)[0]
            max_xy = point.max(dim=1)[0]
            xmin = min_xy[:, 0].clamp(min=0, max=max_w)
            ymin = min_xy[:, 1].clamp(min=0, max=max_h)
            xmax = max_xy[:, 0].clamp(min=0, max=max_w)
            ymax = max_xy[:, 1].clamp(min=0, max=max_h)
            min_xy = torch.stack([xmin, ymin], dim=1)
            max_xy = torch.stack([xmax, ymax], dim=1)
            return torch.cat([min_xy, max_xy], dim=1)  # n,4
        else:
            return point.new_zeros(0, 4)

    def get_trans_mat(self,a,b):
        return [bt @ at.inverse() for bt,at in zip(b,a)]
    
    def transform_bboxes(self,bboxes,out_shape):
        bbox = bboxes[0]
        if bbox.shape[0] == 0:
            return bbox
        if bbox.shape[1] > 4:
            points = self.bbox2point(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            M = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])
            points = torch.matmul(torch.tensor(M).to(bbox.device).float(), points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = self.points2bbox(points, out_shape[1], out_shape[0])
            return bbox

    def filter_proposals(self,proposal_box,threshold):
        valid_map = proposal_box.get_field("objectness")>threshold
        image_shape = proposal_box.size
        new_box_loc = proposal_box.bbox[valid_map,:]
        new_boxes = BoxList(new_box_loc,image_shape)
        new_boxes.add_field("objectness",proposal_box.get_field("objectness")[valid_map])
        return new_boxes

    def forward(self, images, targets=None, auxiliary_task=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            auxiliary_task (Bool): if the auxiliary task is enabled during training

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training: 
            images = to_image_list(images)
            features = self.backbone(images.tensors)
            
            if auxiliary_task and self.cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
                straight_features = features
                rotated_img_features = {0: straight_features}
    
                rotated_images = {0: images}
    
                for rot_i in range(1, 4):
                    rot_images = []
                    for img, img_size in zip(images.tensors, images.image_sizes):
                        rot_image, rot_index = SelfSup_Scrambler.rotate_single(img, rot_i)
                        rot_images.append(rot_image)
    
                    stacked_tensor = torch.stack(rot_images)
                    r_features = self.backbone(stacked_tensor)
                    rotated_img_features[rot_i] = r_features
                    rotated_images[rot_i] = to_image_list(rot_images)
            if targets is not None:
                proposals, proposal_losses = self.rpn(images, features, targets)
                if self.roi_heads:
                    x, result, detector_losses = self.roi_heads(features, proposals, targets)
                    #result = self.get_pseudo_labels(images)
                else:
                    x = features
                    result = proposals
                    detector_loss = {}
    
            losses = {}
    
            pseudo_targets = None
    
            if auxiliary_task and self.cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
                if self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "detections" or ((self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "targets") and targets is None):
    
                    test_result = self.obtain_pseudo_labels(images, features)
    
                elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "targets":
    
                    test_result = targets
    
                elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "images":
    
                    image_sizes = images.image_sizes
                    test_result = []
                    
                    for height, width in image_sizes:
                        xmin = 0
                        ymin = 0
                        xmax = width
                        ymax = height
                        bbox = torch.tensor([[xmin,ymin,xmax,ymax]], dtype=torch.float)
                        boxlist = BoxList(bbox, (width, height))
    
                        boxlist = boxlist.to(images.tensors.device)            
                        test_result.append(boxlist)
    
                elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "crop":
                    image_sizes = images.image_sizes
                    test_result = []
                    for height, width in image_sizes:
                        xmin, ymin, xmax, ymax = self.random_crop_image(width, height)
                        bbox = torch.tensor([[xmin,ymin,xmax,ymax]], dtype=torch.float)
                        boxlist = BoxList(bbox, (width, height))
                        boxlist = boxlist.to(images.tensors.device)
                        test_result.append(boxlist)
    
                rotated_regions = {0: test_result}
                for rot_i in range(1, 4):
                    r_result = [res[::] for res in test_result]
    
                    for idx, box_list in enumerate(r_result):
                        rotated_boxes = box_list.transpose(rot_i + 1)
                        
                        r_result[idx] = rotated_boxes
    
                    rotated_regions[rot_i] = r_result
    
                pooling_res = []
                rot_target_batch = []
                for idx_in_batch in range(len(test_result)):
                    mul = 1
                    rot_target = torch.ones((len(test_result[idx_in_batch]) * mul), dtype=torch.long)
                    for r in range(len(test_result[idx_in_batch])):
                        rot = random.randint(0,3)
                        features_r = rotated_img_features[rot]
                        regions_r = rotated_regions[rot][idx_in_batch][[r]]
                        l_regions_r = [regions_r]
                        pooled_features = self.region_feature_extractor(features_r, l_regions_r)
                        pooled_features = self.ss_adaptive_pooling(pooled_features)
                        pooled_features = pooled_features.view(pooled_features.size(0), -1)
                        class_preds = self.ss_classifier(self.ss_dropout(pooled_features))
                        pooling_res.append(class_preds)
                        rot_target[r] = rot
                    rot_target_batch.append(rot_target)
    
                if len(pooling_res) > 0:
                    pooling_res = torch.stack(pooling_res).squeeze(dim=1)
                    rot_target_batch = torch.cat(rot_target_batch).to(pooling_res.device)
                    aux_loss = self.ss_criterion(pooling_res, rot_target_batch)
                    aux_loss = aux_loss.mean()
                    losses["aux_loss"] = aux_loss
            
            if targets is not None:
                if self.roi_heads:
                    losses.update(detector_losses)
                losses.update(proposal_losses)
                self.global_step += 1
            return losses
        else:
            images = to_image_list(images)
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, features, targets) 
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets) #14
            else:
                x = features
                result = proposals
                detector_losses = {}   
            return result
            
    def get_pseudo_labels(self,images):
          images = to_image_list(images)
          features = self.backbone(images.tensors)
          self.eval()
          proposals, proposal_losses = self.rpn(images, features, None) 
          x, result, detector_losses = self.roi_heads(features, proposals,None)
          if result==None:
              return result
          auged_proposals,_ = self.generate_auged_boxes(result)#140
          if auged_proposals==None:
              return result
          auged_x = self.region_feature_extractor(features,auged_proposals)
          auged_classes,auged_bboxes = self.region_roi_predictor(auged_x)# scores:140,21; bboxes: 140,84
          reg_unc,labels,scores = self.compute_with_uncertainty(auged_classes,auged_bboxes,result)
          gt_bboxes, gt_labels,gt_scores = self.filter_invalid(
              [bbox[:, :4] for bbox in reg_unc],
              labels,
              [-bbox[:, 4:].mean(dim=-1) for bbox in reg_unc],
              scores,
              thr=-0.0005
              #0.01
              #0.001
              #0.005
              #0.0001
              #0.0005
              #0.05
              #0.001
              #0.0001
              #0.00001
              #0.000001
              #0.0000001
              )
          results = self.generate_pseudo(gt_bboxes,gt_labels,gt_scores,proposals[0])
          if len(results) == 0:
              return result
          result,_ = self.generate_pseudo_result(results)
          return result