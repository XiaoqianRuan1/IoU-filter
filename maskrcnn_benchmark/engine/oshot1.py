# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import copy
import random 

import torch
import torch.optim as optim
from tqdm import tqdm
from apex import amp
from collections import OrderedDict

from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..utils.log_image_bb import log_test_image
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(teacher_model, student_model, data_loader, device, oshot_breakpoints, timer=None, cfg=None):
    """
    Args:
        teacher_model: initialized with the source model and updated with EMA
        student_model: used to train with the pseudo labels generated with the teacher model
        data_loader: the val_dataset
        device: cpu
        oshot_breakpoints:
        timer:
        cfg:
    Returns: test-time adaptation;
    """
    results = [{} for _ in range(len(oshot_breakpoints) + 1)] # add one to store results of no adaptation (gamma=0)

    checkpoint = copy.deepcopy(teacher_model.state_dict())

    cpu_device = torch.device("cpu")

    optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    
    logging_enabled = False
    if logging_enabled:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    for index, batch in enumerate(tqdm(data_loader)):
        #_,_,images_w,targets_w,image_ids = batch
        images_q,target_q,images_w,targets_w,image_ids = batch
        assert len(images_w) == 1, 'Must work with a batch size of 1 for OSHOT'
        # Load checkpoint
        teacher_model.load_state_dict(checkpoint)
        student_model.load_state_dict(checkpoint)

        teacher_model.eval()
        with torch.no_grad():
            output = teacher_model(images_w.to(device))
            torch.cuda.synchronize()
            
            #output = [o.to(cpu_device) for o in output]
            results[0].update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )

        if logging_enabled:
            image_name = data_loader.dataset.get_img_name(image_ids[0])
            log_test_image(cfg, summary_writer, "detections_0_its".format(), image_ids[0], images_w, output, image_name=image_name)

        for oshot_it in range(cfg.MODEL.SEMISUP.OSHOT_ITERATIONS):
            teacher_model.eval()
            with torch.no_grad():
                pseudo_labels = teacher_model(images_w.to(device))
            
            if (oshot_it%cfg.MODEL.SEMISUP.TEACHER_UPDATE)==0:
                update_teacher_model(teacher_model,student_model,cfg.MODEL.SEMISUP.KEEP_RATE,device)
            
            student_model.train()
            optimizer.zero_grad()
            imgs_q = copy.deepcopy(images_q)
            if random.random() < 0.5:
                imgs_q.tensors = torch.flip(imgs_q.tensors,dims=(3,))
            #pseudo_loss = student_model(imgs.to(device),pseudo_labels,auxiliary_task=Falsesd)
            student_model.train()
            if len(pseudo_labels[0].bbox)!=0:
                j_loss_dict = student_model(imgs_q.to(device),pseudo_labels,auxiliary_task=True)
            
                losses_weights = [cfg.MODEL.SELF_SUPERVISOR.WEIGHT if 'aux' in k else 1 for k in j_loss_dict.keys()]
                j_losses = sum(loss * weight for loss, weight in zip(j_loss_dict.values(), losses_weights))

                if oshot_it < cfg.MODEL.SELF_SUPERVISOR.OSHOT_WARMUP:
                    j_losses = j_losses * (oshot_it / cfg.MODEL.SELF_SUPERVISOR.OSHOT_WARMUP)
                try:
                    j_losses.backward()

                    optimizer.step()
                except AttributeError:
                #print("AttributeError, maybe no detections detected?")
                    pass
            
            if (oshot_it + 1) in oshot_breakpoints:
                teacher_model.eval()
                with torch.no_grad():
                    if timer:
                        timer.tic()
                    if cfg.TEST.BBOX_AUG.ENABLED:
                        output = im_detect_bbox_aug(teacher_model, images_w, device)
                    else:
                        output = teacher_model(images_w.to(device))
                    if timer:
                        if not cfg.MODEL.DEVICE == 'cpu':
                            torch.cuda.synchronize()
                        timer.toc()
                    #output = [o.to(cpu_device) for o in output]
                results[oshot_breakpoints.index(oshot_it+1) + 1].update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
                if logging_enabled:
                    log_test_image(cfg, summary_writer, "detections_{}_its".format(oshot_it+1), image_ids[0], images_w, output, image_name=image_name)

    return results

@torch.no_grad()
def update_teacher_model(teacher_model,student_model,keep_rate,device):
    """
    Args:
        teacher_model: the original teacher model
        student_model: the original student model
    Returns: the updated teacher model
    """
    student_model_dict = student_model.to(device).state_dict()
    new_teacher_dict = OrderedDict()
    for key,value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key]*
                (1-keep_rate)+value*keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    return teacher_model.load_state_dict(new_teacher_dict)

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

def oshot_inference(
        teacher_model,
        student_model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        oshot_breakpoints=(),
        cfg=None
):
    # add last breakpoint to oshot 
    oshot_breakpoints = (*oshot_breakpoints, cfg.MODEL.SELF_SUPERVISOR.OSHOT_ITERATIONS)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    print(data_loader)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(teacher_model, student_model,data_loader, device, oshot_breakpoints, inference_timer, cfg)
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
        torch.save(predictions[0], os.path.join(output_folder, "oshot_predictions_0.pth"))
        for i in range(len(predictions) - 1):
            torch.save(predictions[i+1], os.path.join(output_folder, "oshot_predictions_%d.pth" % oshot_breakpoints[i]))

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
                           **extra_args))
    return evaluations
