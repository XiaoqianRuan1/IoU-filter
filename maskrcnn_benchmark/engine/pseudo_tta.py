# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import copy
import random 
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from apex import amp
from collections import OrderedDict,defaultdict

from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..utils.log_image_bb import log_test_image
from .bbox_aug import im_detect_bbox_aug

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
   
def bbox_with_threshold(proposal_box,threshold):
	valid_map = proposal_box.get_field("scores")>threshold
	image_shape = proposal_box.size
	new_box_loc = proposal_box.bbox[valid_map,:]
	new_boxes = BoxList(new_box_loc,image_shape)
	new_boxes.add_field("scores",proposal_box.get_field("scores")[valid_map])
	new_boxes.add_field("labels",proposal_box.get_field("labels")[valid_map])
	return new_boxes

@torch.no_grad()
def bbox_with_max(proposal_box,device):
    if len(proposal_box)==0:
        return proposal_box
    score_index = np.argmax(proposal_box.get_field("scores").cpu().numpy())
    label = proposal_box.get_field("labels").cpu().numpy()[score_index]
    box = proposal_box.get_field("labels").cpu().numpy()
    valid_map = torch.tensor(np.isin(box,label)).to(device)
    image_shape = proposal_box.size
    new_box_loc = proposal_box.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",proposal_box.get_field("scores")[valid_map])
    new_boxes.add_field("labels",proposal_box.get_field("labels")[valid_map])
    return new_boxes

@torch.no_grad()
def get_unknown_bbox(proposal_box,prediction,threshold1,threshold2):
    objectness = proposal_box.get_field("objectness").cpu().numpy()>threshold1
    bbox = proposal_box.bbox.cpu().numpy()
    scores = prediction.get_field("scores").cpu().numpy()
    pre_bbox = prediction.bbox.cpu().numpy()
    iou = boxlist_iou(
        BoxList(bbox,proposal_box.size),
        BoxList(pre_bbox,proposal_box.size),
        ).numpy()  
    valid_map = np.all(iou<threshold2,axis=1)
    image_shape = prediction.size
    new_box_loc = prediction.bbox
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",proposal_box.get_field("objectness")[valid_map])
    new_boxes.add_field("labels","aeroplane")
    new_boxes.add_field("scores",prediction.get_field("scores"))
    new_boxes.add_field("labels",prediction.get_field("labels"))
    return new_boxes

@torch.no_grad()
def bbox_with_iou(proposal_box_used,proposal_box_now,thres1,thres2,device):
    proposal_box_used = proposal_box_used.to(device)
    proposal_box_now = proposal_box_now.to(device)
    if len(proposal_box_used)==0 or len(proposal_box_now)==0:
        return proposal_box_now
    scores_used = proposal_box_used.get_field("scores").cpu().numpy()
    scores_now = proposal_box_now.get_field("scores").cpu().numpy()
    bbox_used = proposal_box_used.bbox.cpu().numpy()
    bbox_now = proposal_box_now.bbox.cpu().numpy()
    label_now = proposal_box_now.get_field("labels").cpu().numpy()
    label_used = proposal_box_used.get_field("labels").cpu().numpy()
    valid_map = scores_now>thres1
    for l in np.unique(np.concatenate((label_used, label_now)).astype(int)):
        pred_mask_l = label_used == l
        pred_bbox_l = bbox_used[pred_mask_l]
        pred_score_l = scores_used[pred_mask_l]
        
        pred_mask_n = label_now == l
        pred_bbox_n = bbox_now[pred_mask_n]
        pred_score_n = scores_now[pred_mask_n]
        
        if len(pred_bbox_l)==0:
            continue
        if len(pred_bbox_n)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        pred_bbox_n = pred_bbox_n.copy()
        pred_bbox_n[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_n,proposal_box_now.size),
            BoxList(pred_bbox_l,proposal_box_now.size),
        ).numpy()
  
        valid_map[pred_mask_n] = np.logical_or(valid_map[pred_mask_n],iou.max(axis=1)>thres2)
    image_shape = proposal_box_now.size
    new_box_loc = proposal_box_now.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",proposal_box_now.get_field("scores")[valid_map])
    new_boxes.add_field("labels",proposal_box_now.get_field("labels")[valid_map])
    return new_boxes

def extract_area_prediction(pred_boxlist):
    image_shape = pred_boxlist.size
    width = image_shape[0]
    height = image_shape[1]
    pred_area = pred_boxlist.area()/(height*width)
    pred_area = pred_area.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    not_index = np.logical_and(pred_area>0.1,pred_score<0.5)
    index = np.logical_not(not_index)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    return new_boxes
    
def extract_small_prediction(pred_boxlist):
    pred_area = pred_boxlist.area()
    pred_area = pred_area.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    index = np.logical_or(pred_area<100,pred_score>0.85)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    return new_boxes
    
def extract_duplicate_prediction(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    
    valid_map = pred_score>0.0
    iou = boxlist_iou(
            BoxList(pred_bbox,pred_boxlist.size),
            BoxList(pred_bbox,pred_boxlist.size),
    ).numpy()
    row,col = np.diag_indices_from(iou)
    iou[row,col] = 0
    index = np.where(iou>0.9)
    if len(index[0])==0:
        return pred_boxlist
    for i in range(int(len(index[0])/2)):
        if pred_label[index[0][i]]!=pred_label[index[1][i]]:
            if pred_score[index[0][i]]>pred_score[index[1][i]]:
                valid_map[index[1][i]] = False
            else:
                valid_map[index[0][i]] = False
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    return new_boxes

def extract_crowded_prediction(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    
    valid_map = pred_score>1.0
    iou = boxlist_iou(
            BoxList(pred_bbox,pred_boxlist.size),
            BoxList(pred_bbox,pred_boxlist.size),
    ).numpy()
    row,col = np.diag_indices_from(iou)
    iou[row,col] = 0
    index = np.where(iou>0.2)
    if len(index[0])==0:
        return pred_boxlist
    else:
        for i in range(int(len(index[0])/2)):
            if pred_label[index[0][i]]==pred_label[index[1][i]]:
                valid_map[index[0][i]] = True
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    return new_boxes

@torch.no_grad()
def bbox_with_threshold(proposal_box,threshold):
    valid_map = proposal_box.get_field("scores")>threshold
    image_shape = proposal_box.size
    new_box_loc = proposal_box.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",proposal_box.get_field("scores")[valid_map])
    new_boxes.add_field("labels",proposal_box.get_field("labels")[valid_map])
    return new_boxes

@torch.no_grad()
def process_pseudo_label_threshold(proposals_rpn,threshold,device):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_box in proposals_rpn:
        proposal_box = bbox_with_threshold(proposal_box,threshold)
        num_proposal_output+=len(proposal_box)
        list_instances.append(proposal_box)
    num_proposal_output = num_proposal_output/len(proposals_rpn)
    if len(list_instances[0])==0:
        return None,0
    return list_instances,num_proposal_output

@torch.no_grad()
def process_unknown_labels(proposals_rpn,proposals_roi,threshold1,threshold2):
    list_instances = []
    num_proposal_output = 0.0
    if proposals_roi == None:
        return None,0
    for proposal_rpn,proposal_roi in zip(proposals_rpn,proposals_roi):
        new_proposal = get_unknown_bbox(proposal_rpn,proposal_roi,threshold1,threshold2)
        num_proposal_output += len(new_proposal)
        list_instances.append(new_proposal)
    num_proposal_output = num_proposal_output/len(proposals_rpn)
    if len(list_instances[0])==0:
        return None,0
    return list_instances,num_proposal_output

@torch.no_grad()
def process_pseudo_labels_steps(proposals_rpn,device):
    list_instances = []
    num_proposal_output = 0.0
    if proposals_rpn==None:
        return None,0
    for proposal_box in proposals_rpn:
        proposal_box = extract_area_prediction(proposal_box)
        #proposal_box = extract_small_prediction(proposal_box)
        #proposal_box = extract_crowded_prediction(proposal_box)
        proposal_box = extract_duplicate_prediction(proposal_box)
        num_proposal_output += len(proposal_box)
        list_instances.append(proposal_box)
    num_proposal_output = num_proposal_output/len(proposals_rpn)
    if len(list_instances[0])==0:
        return None,0
    return list_instances,num_proposal_output

@torch.no_grad()
def get_confidence(boxes,results):
    result = []
    scores = boxes[0].get_field("scores")
    labels = boxes[0].get_field("labels")
    for label, score in zip(labels,scores):
        result.append((label,score))
    results.append(result)
    return results

@torch.no_grad()
def process_pseudo_label_threshold(proposals_rpn,threshold,device):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_box in proposals_rpn:
        proposal_box = bbox_with_threshold(proposal_box,threshold)
        num_proposal_output+=len(proposal_box)
        list_instances.append(proposal_box)
    num_proposal_output = num_proposal_output/len(proposals_rpn)
    if len(list_instances[0])==0:
        return None,0
    return list_instances,num_proposal_output

@torch.no_grad()
def process_pseudo_label_confidence(proposals_rpn_used,proposals_rpn_now,thres1,thres2,device):
    list_instances = []
    num_proposal_output = 0.0
    if (proposals_rpn_used==None) or (proposals_rpn_now==None):
        return None,0
    for proposal_box_used,proposal_box_now in zip(proposals_rpn_used,proposals_rpn_now):
        proposal_box = bbox_with_iou(proposal_box_used,proposal_box_now,thres1,thres2,device)
        num_proposal_output+=len(proposal_box)
        list_instances.append(proposal_box)
    num_proposal_output = num_proposal_output/len(proposals_rpn_now)
    if len(list_instances[0])==0:
        return None,0
    return list_instances,num_proposal_output

@torch.no_grad()
def obtain_pseudo_labels(model,images,device):
    output = model(images.to(device))
    return output

def compute_on_dataset(model, data_loader, device, tta_breakpoints, timer=None, cfg=None):
    results = [{} for _ in range(len(tta_breakpoints) + 1)] # add one to store results of no adaptation (gamma=0)
    final_results = defaultdict(list)
    
    checkpoint = copy.deepcopy(model.state_dict())
    cpu_device = torch.device("cpu")
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    logging_enabled = False
    if logging_enabled:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets,image_ids = batch
        assert len(images) == 1, 'Must work with a batch size of 1 for TTA'
        
        model.load_state_dict(checkpoint)
        model.eval()
        with torch.no_grad():
            output = model(images.to(device))
            torch.cuda.synchronize()
            output = [o.to(cpu_device) for o in output]
            results[0].update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        output_used = output
        
        if logging_enabled:
            image_name = data_loader.dataset.get_img_name(image_ids[0])
            log_test_image(cfg, summary_writer, "detections_0_its".format(), image_ids[0], images, output, image_name=image_name)
        
        for tta_it in range(cfg.MODEL.SELF_SUPERVISOR.TTA_ITERATIONS):     
            model.train()
            optimizer.zero_grad()
            
            # random horizontal flip:
            model.eval()
            imgs = copy.deepcopy(images)
            #if random.random() < 0.5:
            imgs.tensors = torch.flip(imgs.tensors,dims=(3,))
            with torch.no_grad():
                output = model(imgs.to(device))
                pseudo_output,_ = process_pseudo_label_confidence(output_used,output,1.0,0.6,device)
                pseudo_output,_ = process_pseudo_labels_steps(pseudo_output,device) 
            output_used = output
            model.train()
            j_loss_dict = model(imgs.to(device),pseudo_output,auxiliary_task=False)
            for key in j_loss_dict.keys():
                if key == "loss_classifier":
                    j_loss_dict[key] = 0.0
            # compute total loss
            j_losses = sum(loss for loss in j_loss_dict.values())
            
            # apply self supervised weight
            losses_weights = [cfg.MODEL.SELF_SUPERVISOR.WEIGHT if 'aux' in k else 1.0 for k in j_loss_dict.keys()]
            j_losses = sum(loss * weight for loss, weight in zip(j_loss_dict.values(), losses_weights))

            if tta_it < cfg.MODEL.SELF_SUPERVISOR.TTA_WARMUP:
                j_losses = j_losses * (tta_it / cfg.MODEL.SELF_SUPERVISOR.TTA_WARMUP)
            try:
                j_losses.backward()

                optimizer.step()
            except AttributeError:
                #print("AttributeError, maybe no detections detected?")
                pass
                
            if (tta_it + 1) in tta_breakpoints:
                model.eval()
                with torch.no_grad():
                    if timer:
                        timer.tic()
                    if cfg.TEST.BBOX_AUG.ENABLED:
                        output = im_detect_bbox_aug(model, images, device)
                    else:
                        output = model(images.to(device))
                    if timer:
                        if not cfg.MODEL.DEVICE == 'cpu':
                            torch.cuda.synchronize()
                        timer.toc()
                    output = [o.to(cpu_device) for o in output]
                results[tta_breakpoints.index(tta_it+1) + 1].update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
                if logging_enabled:
                    log_test_image(cfg, summary_writer, "detections_{}_its".format(tta_it+1), image_ids[0], images, output, image_name=image_name)
    return results
    
def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    
    if len(image_ids) != image_ids[-1] + 1:
        import pdb; pdb.set_trace() 
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def tta_inference(
        source_model,
        model,
        data_loader,
        dataset_name,
        voc_evaluation,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        tta_breakpoints=(),
        cfg=None
):
    # add last breakpoint to tta 
    tta_breakpoints = (*tta_breakpoints,cfg.MODEL.SELF_SUPERVISOR.TTA_ITERATIONS,)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(source_model,data_loader, device, tta_breakpoints, inference_timer, cfg)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    for i, p in enumerate(predictions):
        predictions[i] = _accumulate_predictions_from_multiple_gpus(p)
    if not is_main_process():
        return

    if output_folder:
        # base6 is for foggy. 
        torch.save(predictions[0], os.path.join(output_folder, "prediction_0.pth"))
        for i in range(len(predictions) - 1):
            torch.save(predictions[i+1], os.path.join(output_folder, "prediction_%d.pth" % tta_breakpoints[i]))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    evaluations = []
    for i,p in enumerate(predictions):
        evaluations.append(evaluate(dataset=dataset,
                           predictions=p,
                           output_folder=output_folder,
                           voc_evaluate = voc_evaluation,
                           **extra_args))
    return evaluations
