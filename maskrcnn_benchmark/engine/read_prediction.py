import torch
from maskrcnn_benchmark.config import cfg
#from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.data.datasets.evaluation.voc.evaluate_voc import evaluate,plot_prediction_threshold,plot_correct_prediction,plot_prediction_iou,plot_rpn_target,print_area,do_voc_evaluation1,plot_ground_truth,calculate_percentage,do_voc_evaluation
from maskrcnn_benchmark.engine.pseudo_oshot import process_pseudo_label_threshold,process_pseudo_label_confidence,process_pseudo_labels_steps
import glob
import os
import numpy as np

def inference(
    data_loader,
    voc_evaluation,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    cfg=None,
):
    # convert to a torch.device for efficiency
    dataset = data_loader.dataset 
    #predictions = torch.load("/mnt/sde1/xiaoqianruan/OSHOT/outputs/VOC_baseline/inference/comic_test/predictions.pth")
    #return do_voc_evaluate(dataset,predictions)

    predictions = []
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/OSHOT_eval_VOC_to_clipart/inference/clipart_test/oshot_prediction_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/OSHOT_eval_VOC_to_comic/inference/comic_test/oshot_predictions_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/OSHOT_eval_VOC_to_watercolor/inference/watercolor_test/oshot_predictions_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT-meta-learning-master/outputs/META_OSHOT_eval_VOC_to_clipart/inference/clipart_test/oshot_predictions_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT-meta-learning-master/outputs/META_OSHOT_eval_VOC_to_comic/inference/comic_test/*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT-meta-learning-master/outputs/META_OSHOT_eval_VOC_to_watercolor/inference/watercolor_test/*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/OSHOT_eval_VOC_to_clipart/inference/clipart_test/base7_prediction_*.pth")
    paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/OSHOT_eval_VOC_to_comic/inference/comic_test/base7_prediction_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/VOC_baseline/inference/watercolor_test/base1_prediction_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT-meta-learning-master/outputs/rainycityscape_baseline/inference/cityscapes_detection_rainy_val/base6_predictions_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT-meta-learning-master/outputs/foggycityscape_baseline/inference/cityscapes_detection_foggy_val/base2_predictions_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/oshot_cityscapes_to_foggy/inference/cityscapes_detection_foggy_val/base6_prediction_*.pth")
    #paths = glob.glob("/mnt/sde1/xiaoqianruan/OSHOT/outputs/rainycityscape_baseline/inference/cityscapes_detection_rainy_val/base6_prediction_*.pth")
    #paths = "/mnt/sde1/xiaoqianruan/OSHOT/outputs/VOC_baseline/inference/clipart_test/base0_prediction_3.pth"
    #prediction = torch.load(paths)
    
    for path in paths:
        prediction = torch.load(path)
        predictions.append(prediction)
    """
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    voc_evaluate = voc_evaluation,
                    **extra_args)
    """
    #output_folder = "/mnt/sde1/xiaoqianruan/OSHOT/outputs/foggy_ours7/"
    #if not os.path.exists(output_folder):
    #    os.makedirs(output_folder)
    #return plot_correct_prediction(dataset,predictions[-2],output_folder)
    #return plot_prediction_iou(dataset, predictions[-1],output_folder)
    #return plot_ground_truth(dataset,prediction,output_folder)
    #return evaluate(dataset,predictions)
    prediction,_ = process_pseudo_label_confidence(predictions[0],predictions[1],1.0,0.7,device)
    prediction,_ = process_pseudo_labels_steps(prediction,device)
    prediction,_ = process_pseudo_label_threshold(prediction,0.2,device)
    results = do_voc_evaluation(dataset,prediction)
    #return calculate_percentage(dataset,predictions)
    #return print_area(dataset,predictions)
    #return do_voc_evaluation1(dataset,predictions[0])
    #calculate_percentage(dataset,predictions)
    #calculate_background_number(dataset,predictions)