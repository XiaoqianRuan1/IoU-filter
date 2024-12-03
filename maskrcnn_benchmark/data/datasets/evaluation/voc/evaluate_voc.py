# Perform TIDE evaluation on the detection data. We reuse the groundtruth and detection
# data from the previous code section. There's a bit more processing than in the TIDE
# GitHub sample because of the difference in data format.
from __future__ import division
from tidecv import TIDE, datasets, Data
import cv2
import numpy as np

import os
from collections import defaultdict
import math
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import torch

def map_to_classes():
    CLASSES = (
        "__background__",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    class_map = {}
    for i in range(len(CLASSES)):
        class_map[CLASSES[i]] = i
    return class_map

def map_to_index(i):
    CLASSES = (
        "__background__",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    #class_map = {}
    #for i in range(len(CLASSES)):
    #    class_map[CLASSES[i]] = i
    #return class_map
    return CLASSES[i]

def do_voc_evaluation1(dataset,predictions):
    gt_data = Data("gt_data")
    det_data = Data("det_data")
    tide = TIDE()

    for threshold in np.arange(0.0,-0.1,-0.1):
        for image_id,prediction in enumerate(predictions):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width,image_height))
            valid_map = prediction.get_field("scores")>threshold
            pred_bbox = prediction.bbox[valid_map,:].numpy()
            pred_label = prediction.get_field("labels")[valid_map].numpy()
            pred_score = prediction.get_field("scores")[valid_map].numpy()
            for i in range(len(pred_label)):
                label = pred_label[i]
                score = pred_score[i]
                bbox = pred_bbox[i]
                det_data.add_detection(image_id,label,score,bbox,None)
    
            gt_boxlist = dataset.get_groundtruth(image_id)
            gt_bbox = gt_boxlist.bbox.numpy()
            gt_label = gt_boxlist.get_field("labels").numpy()
            gt_difficult = gt_boxlist.get_field("difficult").numpy()
            for i in range(len(gt_difficult)):
                label = gt_label[i]
                bbox = gt_bbox[i]
                gt_data.add_ground_truth(image_id,label,bbox,None)
    
        tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
        tide.summarize()
        tide.plot()

def plot_ground_truth(dataset,predictions,output_folder):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        #data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/comic/JPEGImages/"
        #data_path = "/mnt/sde1/xiaoqianruan/SemiSeg-Contrastive-main/data/CityScapes/leftImg8bit_foggy/val/"
        data_path = "/mnt/sde1/xiaoqianruan/SemiSeg-Contrastive-main/data/CityScapes/leftImg8bit_rain/val/"
        files = image_name.split("_")[0]
        #image = cv2.imread(data_path +"/"+ str(files) +"/"+ str(image_name) + "_leftImg8bit_foggy_beta_0.005.png")
        #image = cv2.imread(data_path + str(image_name) + ".jpg")
        image = cv2.imread(data_path +"/"+ str(files) +"/"+ str(image_name) + "_leftImg8bit_rain_alpha_0.03_beta_0.015_dropsize_0.002_pattern_12.png")
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        for index, box in enumerate(gt_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 0), 4)
            #label = map_to_index(gt_label[index])
            #t_size = cv2.getTextSize(str(label), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            #textlbottom = a + np.array(list(t_size))
            #cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
            #a = list(a)
            #a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            #cv2.putText(image, str(label), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        if not os.path.exists(output_folder+"/"+ str(files) +"/"):
            print(output_folder+"/"+ str(files) +"/")
            os.makedirs(output_folder+"/"+ str(files) +"/")
        cv2.imwrite(output_folder+str(image_id)+".jpg",image)

def plot_correct_prediction(dataset,predictions,output_folder):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist = prediction.resize((image_width,image_height))
        #pred_boxlist = extract_duplicate_prediction(pred_boxlist)
        prediction, wrong_prediction = extract_correct_prediction(pred_boxlist,gt_boxlist)
        missing_pred = extract_missing_prediction(pred_boxlist,gt_boxlist)
        #prediction,_ = extract_target(pred_boxlist,gt_boxlist)
        
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()
        
        wrong_bbox = wrong_prediction.bbox.numpy()
        wrong_label = wrong_prediction.get_field("labels").numpy()
        wrong_score = wrong_prediction.get_field("scores").numpy()
        
        missing_bbox = missing_pred.bbox.numpy()
        missing_label = missing_pred.get_field("labels").numpy()
        data_path = "/mnt/sde1/xiaoqianruan/SemiSeg-Contrastive-main/data/CityScapes/leftImg8bit_foggy/val/"
        #data_path = "/mnt/sde1/xiaoqianruan/SemiSeg-Contrastive-main/data/CityScapes/leftImg8bit_rain/val/"
        #data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/comic/JPEGImages/"
        #image = cv2.imread(data_path + str(image_name) + ".jpg")
        files = image_name.split("_")[0]
        #image = cv2.imread(data_path +"/"+ str(files) +"/"+ str(image_name) + "_leftImg8bit_rain_alpha_0.03_beta_0.015_dropsize_0.002_pattern_12.png")
        image = cv2.imread(data_path +"/"+ str(files) +"/"+ str(image_name) + "_leftImg8bit_foggy_beta_0.005.png")
        #gt_boxlist = dataset.get_groundtruth(image_id)
        #gt_bbox = gt_boxlist.bbox.numpy()
        #gt_label = gt_boxlist.get_field("labels").numpy()
        #if not os.path.exists(output_folder+"/"+ str(files)):
        #    os.makedirs(output_folder+"/"+ str(files))
        for index, box in enumerate(pred_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 0), 4)
            #pred = map_to_index(pred_label[index])
            #t_size = cv2.getTextSize(str(pred), 1, cv2.FONT_HERSHEY_TRIPLEX,1)[0]
            #textlbottom = a + np.array(list(t_size))
            #cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
            #a = list(a)
            #a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            #cv2.putText(image, str(pred), tuple(a), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
            #score_size = cv2.getTextSize(str(round(pred_score[index],2)), 1, cv2.FONT_HERSHEY_TRIPLEX, 1)[0]
            #textl1 = (a[0],b[1]-np.array(list(score_size))[1]/2)
            #textl2 = (a[0]+np.array(list(score_size))[0]/2,b[1])
            #textrbottom = b + np.array(list(score_size))
            #cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 255, 0), -1)
            #cv2.rectangle(image,tuple(textl1),tuple(textl2),(0,255,0),-1)
            #b = list(b)
            #cv2.putText(image, str(round(pred_score[index],2)), (textl1[0], textl2[1]), cv2.FONT_HERSHEY_TRIPLEX, 1,
            #            (0, 0, 0), 1)
        if not os.path.exists(output_folder+"/"+ str(files) +"/"):
            os.makedirs(output_folder+"/"+ str(files) +"/")
        cv2.imwrite(output_folder+"/"+ str(files) +"/"+str(image_id)+".png",image)
        #print(output_folder+"/"+ str(files) +"/"+str(image_id)+".png")
        #cv2.imwrite(output_folder+str(image_id)+".jpg",image)
        
        for index, box in enumerate(wrong_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            #drawrect(image,a,b,(0,255,0),8,'dotted')
            if index==0:
                break
            cv2.rectangle(image, a, b, (0,255,0), 4)
                #pred = map_to_index(wrong_label[index])
                #t_size = cv2.getTextSize(str(pred), 1, cv2.FONT_HERSHEY_TRIPLEX, 1)[0]
                #textr1 = (b[0]-np.array(list(t_size)[0]/2),a[1])
                #textr2 = (b[0],a[1]+np.array(list(t_size)[1]/2))
                #textlbottom = a + np.array(list(t_size))
                #cv2.rectangle(image, tuple(textr1), tuple(textr2), (0, 0, 255), -1)
                #a = list(a)
                #a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
                #cv2.putText(image, str(pred), (textr1[0],textr2[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
                #score_size = cv2.getTextSize(str(round(wrong_score[index],2)), 1, cv2.FONT_HERSHEY_TRIPLEX, 1)[0]
                #textrbottom = b - np.array(list(score_size))
                #cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 0, 255), -1)
                #b = list(b)
                #cv2.putText(image, str(round(wrong_score[index],2)), (textrbottom[0], b[1]),cv2.FONT_HERSHEY_TRIPLEX, 1,
                #            (0, 0, 0), 1)
        """
        for index, box in enumerate(missing_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            drawrect(im,s,e,(0,255,255),1,'dotted')
            #cv2.rectangle(image, a, b, (255, 0, 0), 8)
        """
        cv2.imwrite(output_folder +"/"+ str(files) +"/" + str(image_id) + ".png", image)
        #cv2.imwrite(output_folder+str(image_id)+".jpg",image)

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

def plot_rpn_target(dataset,predictions,output_folder):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        #pred_boxlist = prediction.resize((image_width,image_height))
        #prediction,_ = extract_rpn_target(pred_boxlist,gt_boxlist)
        
        #pred_bbox = prediction.bbox.numpy()
        #pred_label = prediction.get_field("labels").numpy()
        #pred_score = prediction.get_field("objectness").numpy()
        data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/clipart/JPEGImages/"
        image = cv2.imread(data_path + str(image_name) + ".jpg")
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        for index, box in enumerate(gt_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 0), 2)
            label = map_to_index(gt_label[index])
            t_size = cv2.getTextSize(str(label), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textlbottom = a + np.array(list(t_size))
            cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
            a = list(a)
            a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            cv2.putText(image, str(label), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        cv2.imwrite(output_folder+str(image_id)+".jpg",image)
        
        #for index, box in enumerate(pred_bbox):
        #    xmin, ymin, xmax, ymax = box
        #    a = (int(xmin), int(ymin))
        #    b = (int(xmax), int(ymax))
        #    cv2.rectangle(image, a, b, (0, 0, 255), 2)
            #t_size = cv2.getTextSize(str(pred_label[index]), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            #textlbottom = a + np.array(list(t_size))
            #cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 0, 255), -1)
            #a = list(a)
            #a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            #cv2.putText(image, str(round(pred_label[index],2)), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
       #     score_size = cv2.getTextSize(str(round(pred_score[index],2)), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
       #     textrbottom = b - np.array(list(score_size))
       #     cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 0, 255), -1)
       #     b = list(b)
       #     cv2.putText(image, str(round(pred_score[index],2)), (textrbottom[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
       #                 (255, 255, 255), 1)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)

def calculate_percentage(dataset,predictions):
    num_correct,num_missing = 0,0
    for thresh in np.arange(0.0,1.1,0.1):
        thresh_num = 0
        correct_num = 0
        missing_num = 0
        for image_id in range(len(predictions)):
            gt_boxlist = dataset.get_groundtruth(image_id)
            image_name = dataset.get_img_name(image_id)
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            pred_boxlist0 = predictions[0][image_id].resize((image_width,image_height))
            pred_boxlist1 = predictions[1][image_id].resize((image_width,image_height))
            pred_boxlist2 = predictions[2][image_id].resize((image_width,image_height))
            pred_boxlist3 = predictions[3][image_id].resize((image_width,image_height))
            pred_boxlist4 = predictions[4][image_id].resize((image_width,image_height))
            pred_boxlist5 = predictions[5][image_id].resize((image_width,image_height))
            prediction, _ = extract_iou(pred_boxlist1,pred_boxlist2,thresh-0.1,thresh)
            #pred_boxlist1 = all_duplicate_prediction(pred_boxlist)
            #pred_boxlist2 = analysis_duplicate_prediction(pred_boxlist)
            prediction_correct,_ = extract_correct_prediction(prediction,gt_boxlist)
            missing_pred = extract_missing_prediction(prediction,gt_boxlist)
            all_correct ,_ = extract_correct_prediction(pred_boxlist1,gt_boxlist)
            correct_num += len(prediction_correct)
            missing_num += len(missing_pred)
        num_correct += correct_num
        num_missing += missing_num
        print(thresh)
        print(correct_num)
        print(missing_num)
    print(num_correct)
    print(num_missing)
    return num_correct, num_missing
        
def plot_all_predictions(dataset,predictions,output_folder):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist = prediction.resize((image_width,image_height))
        prediction, wrong_prediction = extract_correct_prediction(pred_boxlist,gt_boxlist)
        
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()
        
        wrong_bbox = wrong_prediction.bbox.numpy()
        wrong_label = wrong_prediction.get_field("labels").numpy()

def plot_prediction_iou(dataset, predictions,output_folder):
    prediction0 = predictions[-2]
    prediction1 = predictions[-1]
    for image_id,(prediction0,prediction1) in enumerate(zip(prediction0,prediction1)):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist0 = prediction0.resize((image_width,image_height))
        pred_boxlist1 = prediction1.resize((image_width,image_height))
        #pred_boxlist0,_,_ = extract_area_prediction(pred_boxlist0,image_width,image_height)
        #pred_boxlist1,_,_ = extract_area_prediction(pred_boxlist1,image_width,image_height)
        #pred_boxlist0 = extract_duplicate_prediction(pred_boxlist0)
        #pred_boxlist1 = extract_duplicate_prediction(pred_boxlist1)
        prediction,wrong_prediction = extract_iou(pred_boxlist0,pred_boxlist1,0.9,1.0)
        #prediction,_,_ = extract_area_prediction(prediction,image_width,image_height)
        #prediction = extract_duplicate_prediction(prediction)
        #prediction = generate_pseudo_labels(pred_boxlist0,pred_boxlist1,image_width,image_height)
        prediction_correct,prediction_wrong = extract_correct_prediction(prediction,gt_boxlist)
        wrong_prediction_correct,wrong_prediction_wrong = extract_correct_prediction(wrong_prediction,gt_boxlist)
        
        wrong_bbox_correct = wrong_prediction_correct.bbox.numpy()
        wrong_label_correct = wrong_prediction_correct.get_field("labels").numpy()
        wrong_score_correct = wrong_prediction_correct.get_field("scores").numpy()
        
        pred_bbox_correct = prediction_correct.bbox.numpy()
        pred_label_correct = prediction_correct.get_field("labels").numpy()
        pred_score_correct = prediction_correct.get_field("scores").numpy()
        
        wrong_bbox_wrong = wrong_prediction_wrong.bbox.numpy()
        wrong_label_wrong = wrong_prediction_wrong.get_field("labels").numpy()
        wrong_score_wrong = wrong_prediction_wrong.get_field("scores").numpy()
        
        pred_bbox_wrong = prediction_wrong.bbox.numpy()
        pred_label_wrong = prediction_wrong.get_field("labels").numpy()
        pred_score_wrong = prediction_wrong.get_field("scores").numpy()
        
        data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/clipart/JPEGImages/"
        image = cv2.imread(data_path + str(image_name) + ".jpg")
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        #for index, box in enumerate(gt_bbox):
        #    xmin, ymin, xmax, ymax = box
        #    a = (int(xmin), int(ymin))
        #    b = (int(xmax), int(ymax))
        #    cv2.rectangle(image, a, b, (0, 255, 0), 2)
        #    label = map_to_index(gt_label[index])
        #    t_size = cv2.getTextSize(str(label), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        #    textlbottom = a + np.array(list(t_size))
        #    cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
        #    a = list(a)
        #    a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            #label = gt_label[index]
        #    cv2.putText(image, str(label), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        #cv2.imwrite(output_folder+str(image_id)+".jpg",image)
        for index, box in enumerate(pred_bbox_correct):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 0), 2)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)
        
        for index, box in enumerate(pred_bbox_wrong):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 0, 255), 2)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)
        
        for index, box in enumerate(wrong_bbox_correct):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (255, 0, 0), 2)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)
        
        for index, box in enumerate(wrong_bbox_wrong):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 255), 2)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)
        
        #for index, box in enumerate(pred_bbox):
        #    xmin, ymin, xmax, ymax = box
        #    a = (int(xmin), int(ymin))
        #    b = (int(xmax), int(ymax))
        #    cv2.rectangle(image, a, b, (0, 0, 255), 2)
        #    pred = map_to_index(pred_label[index])
        #    t_size = cv2.getTextSize(str(pred), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        #    textlbottom = a + np.array(list(t_size))
        #    cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 0, 255), -1)
        #    a = list(a)
        #    a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
        #    cv2.putText(image, str(pred), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        #    score_size = cv2.getTextSize(str(round(pred_score[index],2)), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        #    textrbottom = b - np.array(list(score_size))
        #    cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 0, 255), -1)
        #    b = list(b)
        #    cv2.putText(image, str(round(pred_score[index],2)), (textrbottom[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
        #                (255, 255, 255), 1)
        #cv2.imwrite(output_folder + str(image_id) + ".jpg", image)

def plot_prediction_threshold(dataset,predictions,output_folder):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist = prediction.resize((image_width,image_height))
        prediction = extract_threshold(pred_boxlist)

        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()
        data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/clipart/JPEGImages/"
        image = cv2.imread(data_path + str(image_name) + ".jpg")
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        #for index, box in enumerate(gt_bbox):
        #    xmin, ymin, xmax, ymax = box
        #    a = (int(xmin), int(ymin))
        #    b = (int(xmax), int(ymax))
        #    cv2.rectangle(image, a, b, (0, 255, 0), 2)
        #    label = map_to_index(gt_label[index])
        #    t_size = cv2.getTextSize(str(label), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        #    textlbottom = a + np.array(list(t_size))
        #    cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
        #    a = list(a)
        #    a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
        #    cv2.putText(image, str(label), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        #cv2.imwrite(output_folder+str(image_id)+".jpg",image)
        
        for index, box in enumerate(pred_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 0, 255), 2)
            pred = map_to_index(pred_label[index])
            t_size = cv2.getTextSize(str(pred), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textlbottom = a + np.array(list(t_size))
            cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 0, 255), -1)
            a = list(a)
            a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            cv2.putText(image, str(pred), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            score_size = cv2.getTextSize(str(round(pred_score[index],2)), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textrbottom = b - np.array(list(score_size))
            cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 0, 255), -1)
            b = list(b)
            cv2.putText(image, str(round(pred_score[index],2)), (textrbottom[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)

def print_area(dataset,predictions):
    for image_id,prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        image_name = dataset.get_img_name(image_id)
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist = prediction.resize((image_width,image_height))
        calculate_correct_area(pred_boxlist,gt_boxlist,image_width,image_height)
        
def evaluate(dataset,predictions):
    correct_total = 0
    wrong_total = 0
    for thresh in np.arange(0.0,1.1,0.1):
      thresh_num = 0  
      correct_num = 0
      wrong_num = 0
      for image_id in range(len(predictions[0])):
          gt_boxlist = dataset.get_groundtruth(image_id)
          img_info = dataset.get_img_info(image_id)
          image_width = img_info["width"]
          image_height = img_info["height"]
          pred_boxlist0 = predictions[0][image_id].resize((image_width,image_height))
          pred_boxlist1 = predictions[1][image_id].resize((image_width,image_height))
          pred_boxlist2 = predictions[2][image_id].resize((image_width,image_height))
          pred_boxlist3 = predictions[3][image_id].resize((image_width,image_height))
          pred_boxlist4 = predictions[4][image_id].resize((image_width,image_height))
          pred_boxlist5 = predictions[5][image_id].resize((image_width,image_height))
          #correct_num,wrong_num = calculate_correct_number(pred_boxlist1,gt_boxlist,correct_num,wrong_num)
          #correct_num,wrong_num = analysis_confidence(pred_boxlist0,gt_boxlist,thresh-0.1,thresh,correct_num,wrong_num)
          #correct_num,wrong_num = analysis_iou(pred_boxlist3,pred_boxlist4,gt_boxlist,thresh-0.1,thresh,correct_num,wrong_num,image_width,image_height)
          #correct_num,wrong_num = calculate_iou_confidence_recall(dataset,pred_boxlist4,pred_boxlist5,gt_boxlist,0.9,1.0,correct_num,wrong_num,thresh-0.1,thresh,image_width,image_height)
          #correct_num,wrong_num = analysis_target(pred_boxlist0,gt_boxlist,thresh-0.1,thresh,correct_num,wrong_num)
          #test_gt_confidence(pred_boxlist0,gt_boxlist)
          #test_gt_confidence(pred_boxlist1,gt_boxlist)
          #test_gt_iou(pred_boxlist0,pred_boxlist1,gt_boxlist)
      correct_total += correct_num
      wrong_total += wrong_num
      print(thresh)
      print(correct_num)
      print(wrong_num)
      print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print(correct_total)
    print(wrong_total)   

def analysis_target(pred_boxlist,gt_boxlist,thresh1,thresh2,correct_num,wrong_num):
    correct_boxlist,wrong_boxlist = extract_target(pred_boxlist,gt_boxlist)
    correct_labels = correct_boxlist.get_field("labels").cpu().numpy()
    correct_scores = correct_boxlist.get_field("scores").cpu().numpy()
    wrong_labels = wrong_boxlist.get_field("labels").cpu().numpy()
    wrong_scores = wrong_boxlist.get_field("scores").cpu().numpy()
    valid_map = np.logical_and(correct_scores<=thresh2,correct_scores>thresh1)
    no_valid_map = np.logical_and(wrong_scores<=thresh2,wrong_scores>thresh1)
    correct_num += np.sum(valid_map!=0)
    wrong_num += np.sum(no_valid_map!=0)
    return correct_num,wrong_num

def analysis_confidence(pred_boxlist,gt_boxlist,thresh1,thresh2,correct_num,wrong_num):
    correct_boxlist,wrong_boxlist = extract_correct_prediction(pred_boxlist,gt_boxlist)
    correct_labels = correct_boxlist.get_field("labels").cpu().numpy()
    correct_scores = correct_boxlist.get_field("scores").cpu().numpy()
    wrong_labels = wrong_boxlist.get_field("labels").cpu().numpy()
    wrong_scores = wrong_boxlist.get_field("scores").cpu().numpy()
    valid_map = np.logical_and(correct_scores<=thresh2,correct_scores>thresh1)
    no_valid_map = np.logical_and(wrong_scores<=thresh2,wrong_scores>thresh1)
    correct_num += np.sum(valid_map!=0)
    wrong_num += np.sum(no_valid_map!=0)
    return correct_num,wrong_num    

def analysis_iou(pred_boxlist0,pred_boxlist1,gt_boxlist,thresh1,thresh2,correct_num,wrong_num,width,height):
    #pred_boxlist0, _ = extract_area_prediction(pred_boxlist0,width,height)
    #pred_boxlist1, _ = extract_area_prediction(pred_boxlist1,width,height)
    #pred_boxlist0 = extract_duplicate_prediction(pred_boxlist0)
    #pred_boxlist1 = extract_duplicate_prediction(pred_boxlist1)
    prediction = extract_iou(pred_boxlist0,pred_boxlist1,thresh1,thresh2)
    #prediction,_,_ = extract_area_prediction(prediction,width,height)
    #prediction = extract_small_prediction(prediction)
    #prediction = extract_crowded_prediction(prediction)
    #pred_boxlist = extract_duplicate_prediction(prediction)
    correct_boxlist,wrong_boxlist = extract_correct_prediction(prediction,gt_boxlist)
    correct_num+=len(correct_boxlist)
    wrong_num+=len(wrong_boxlist)
    #correct_boxlist1,wrong_boxlist1 = extract_correct_prediction(pred_boxlist1,gt_boxlist)
    
    #correct_num = get_iou_num(correct_boxlist0,correct_boxlist1,gt_boxlist,correct_num,thresh1,thresh2)
    #wrong_num = get_iou_num(wrong_boxlist0,wrong_boxlist1,gt_boxlist,wrong_num,thresh1,thresh2)
    return correct_num,wrong_num

def analysis_iou_confidence(pred_boxlist0,pred_boxlist1,gt_boxlist,thresh1,thresh2,correct_num,wrong_num,conf_thres1,conf_thres2,width,height):
    prediction,_ = extract_iou(pred_boxlist0,pred_boxlist1,thresh1,thresh2)
    #prediction,_,_ = extract_area_prediction(prediction,width,height)
    #prediction = extract_small_prediction(prediction)
    #prediction = extract_crowded_prediction(prediction)
    #pred_boxlist = extract_duplicate_prediction(prediction)
    correct_boxlist,wrong_boxlist = extract_correct_prediction(prediction,gt_boxlist)
    correct_num = iou_and_confidence(correct_boxlist,correct_num,conf_thres1,conf_thres2)
    wrong_num = iou_and_confidence(wrong_boxlist,wrong_num,conf_thres1,conf_thres2)
    return correct_num,wrong_num
    
def calculate_iou_confidence_recall(dataset,pred_boxlist0,pred_boxlist1,gt_boxlist,thresh1,thresh2,correct_num,wrong_num,conf_thres1,conf_thres2,width,height):
    prediction, _ = extract_iou(pred_boxlist0,pred_boxlist1,thresh1,thresh2)
    prediction = generate_prediction_by_threshold(prediction,conf_thres1,conf_thres2)
    correct_num, missing_num = calculate_percentage(dataset,prediction)
    return correct_num, missing_num

def generate_prediction_by_threshold(prediction,conf_thres1,conf_thres2): 
    pred_score = prediction.get_field("scores").cpu().numpy()
    valid_map = np.logical_and(pred_score<=conf_thres2,pred_score>conf_thres1) 
    image_shape = prediction.size
    new_box_loc = prediction.bbox[valid_map, :]
    new_boxes = BoxList(new_box_loc, image_shape)
    new_boxes.add_field("scores", prediction.get_field("scores")[valid_map])
    new_boxes.add_field("labels", prediction.get_field("labels")[valid_map])
    return new_boxes
    
def iou_and_confidence(pred_boxlist,correct_num,conf_thres1,conf_thres2):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    valid_map = np.logical_and(pred_score<=conf_thres2,pred_score>conf_thres1)
    correct_num += np.sum(valid_map!=0)
    return correct_num

def evaluate_recall(dataset,predictions):
    correct_total = 0
    missing_total = 0
    for thresh in np.arange(0.0,1.1,0.1):
      thresh_num = 0  
      correct_num = 0
      missing_num = 0
      for image_id in range(len(predictions[0])):
          gt_boxlist = dataset.get_groundtruth(image_id)
          img_info = dataset.get_img_info(image_id)
          image_width = img_info["width"]
          image_height = img_info["height"]
          pred_boxlist0 = predictions[0][image_id].resize((image_width,image_height))
          pred_boxlist1 = predictions[1][image_id].resize((image_width,image_height))
          pred_boxlist2 = predictions[2][image_id].resize((image_width,image_height))
          pred_boxlist3 = predictions[3][image_id].resize((image_width,image_height))
          pred_boxlist4 = predictions[4][image_id].resize((image_width,image_height))
          pred_boxlist5 = predictions[5][image_id].resize((image_width,image_height))
          correct_num,wrong_num = analysis_iou_confidence(pred_boxlist4,pred_boxlist5,gt_boxlist,0.9,1.0,correct_num,wrong_num,thresh-0.1,thresh,image_width,image_height)
      correct_total += correct_num
      wrong_total += wrong_num
      print(thresh)
      print(correct_num)
      print(missing_num)
    print(correct_total)
    print(missing_total)    

def get_iou_num(correct_boxlist0,correct_boxlist1,gt_boxlist,correct_num,thresh1,thresh2):
    correct_bbox0 = correct_boxlist0.bbox.cpu().numpy()
    correct_label0 = correct_boxlist0.get_field("labels").cpu().numpy()
    correct_score0 = correct_boxlist0.get_field("scores").cpu().numpy()
    
    correct_bbox1 = correct_boxlist1.bbox.cpu().numpy()
    correct_label1 = correct_boxlist1.get_field("labels").cpu().numpy()
    correct_score1 = correct_boxlist1.get_field("scores").cpu().numpy()
    
    for l in np.unique(np.concatenate((correct_label0,correct_label1)).astype(int)):
        pred_mask_l = correct_label0 == l
        pred_bbox_l = correct_bbox0[pred_mask_l]
        pred_score_l = correct_score0[pred_mask_l]
        order_l = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order_l]
        pred_score_l = pred_score_l[order_l]
        
        pred_mask_n = correct_label1 == l
        pred_bbox_n = correct_bbox1[pred_mask_n]
        pred_score_n = correct_score1[pred_mask_n]
        order_n = pred_score_n.argsort()[::-1]
        pred_bbox_n = pred_bbox_n[order_n]
        pred_score_n = pred_score_n[order_n]
        
        if len(pred_bbox_l)==0:
            continue
        if len(pred_bbox_n)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        pred_bbox_n = pred_bbox_n.copy()
        pred_bbox_n[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_l,gt_boxlist.size),
            BoxList(pred_bbox_n,gt_boxlist.size),
        ).numpy()
        gt_index = iou.argmax(axis=0)
        pred_index = iou.argmax(axis=1)
        index = np.logical_and(iou.max(axis=0)>=thresh1,iou.max(axis=0)<thresh2)
        index1 = np.logical_and(iou.max(axis=1)>=thresh1,iou.max(axis=1)<thresh2)
        correct_num += np.sum(index!=0)
    return correct_num

def generate_pseudo_labels(pred_boxlist0,pred_boxlist1,width,height):
    prediction0 = pred_boxlist0 #14
    prediction1 = pred_boxlist1
    _,_,index0 = extract_area_prediction(pred_boxlist0,width,height)
    _,_,index1 = extract_area_prediction(pred_boxlist1,width,height) 
    new_boxes1 = extract_iou(pred_boxlist0,pred_boxlist1,0.9,1.0)
    valid1 = np.logical_and(index1,boxes1)
    
    pred_boxlist0 = extract_small_prediction(pred_boxlist0)
    pred_boxlist1 = extract_small_prediction(pred_boxlist1)
    new_boxes2 = extract_iou(pred_boxlist0,pred_boxlist1,0.8,0.9)
    valid2 = new_boxes2.get_field("scores").cpu().numpy()>0.0
    
    pred_boxlist0 = extract_crowded_prediction(pred_boxlist0)
    pred_boxlist1 = extract_crowded_prediction(pred_boxlist1)
    new_boxes3 = extract_iou(pred_boxlist0,pred_boxlist1,0.0,0.8)
    valid3 = new_boxes3.get_field("scores").cpu().numpy()>0.0
    if len(valid2)==0 and len(valid3)==0:
        valid = valid1
    elif len(valid1)==0 and len(valid3) == 0:
        valid = valid2
    elif len(valid2) == 0 and len(valid1) == 0:
        valid = valid3
    elif len(valid1)==0:
        valid = np.logical_or(valid2,valid3)
    elif len(valid2) == 0:
        valid = np.logical_or(valid1,valid3)
    elif len(valid3) == 0:
        valid = np.logical_or(valid2,valid1)
    elif len(valid1==0) and len(valid2)==0 and len(valid3)==0:
        valid = valid1
    else:
        valid = np.logical_or(valid1,valid2,valid3)
    
    image_shape = prediction1.size
    new_box_loc = prediction1.bbox[valid,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",prediction1.get_field("scores")[valid])
    new_boxes.add_field("labels",prediction1.get_field("labels")[valid])
    return new_boxes 

def extract_threshold(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    valid_map = pred_score>0.0
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    return new_boxes

def extract_iou(pred_boxlist0,pred_boxlist1,threshold1,threshold2):
    pred_bbox0 = pred_boxlist0.bbox.cpu().numpy()
    pred_label0 = pred_boxlist0.get_field("labels").cpu().numpy()
    pred_score0 = pred_boxlist0.get_field("scores").cpu().numpy()
    
    pred_bbox1 = pred_boxlist1.bbox.cpu().numpy()
    pred_label1 = pred_boxlist1.get_field("labels").cpu().numpy()
    pred_score1 = pred_boxlist1.get_field("scores").cpu().numpy()
    
    valid_map = pred_score1>1.0
    for l in np.unique(np.concatenate((pred_label0,pred_label1)).astype(int)):
        pred_mask_l = pred_label0 == l
        pred_bbox_l = pred_bbox0[pred_mask_l]
        pred_score_l = pred_score0[pred_mask_l]
        order_l = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order_l]
        pred_score_l = pred_score_l[order_l]
        
        pred_mask_n = pred_label1 == l
        pred_bbox_n = pred_bbox1[pred_mask_n]
        pred_score_n = pred_score1[pred_mask_n]
        order_n = pred_score_n.argsort()[::-1]
        pred_bbox_n = pred_bbox_n[order_n]
        pred_score_n = pred_score_n[order_n]
        
        if len(pred_bbox_l)==0:
            continue
        if len(pred_bbox_n)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        pred_bbox_n = pred_bbox_n.copy()
        pred_bbox_n[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_l,pred_boxlist0.size),
            BoxList(pred_bbox_n,pred_boxlist0.size),
        ).numpy()
        gt_index = iou.argmax(axis=0)
        gt_index[iou.max(axis=0)<=threshold1] = -1
        gt_index[iou.max(axis=0)>threshold2] = -1
        del iou
        index = gt_index>=0
        valid_map[pred_mask_n] = index
    image_shape = pred_boxlist1.size
    new_box_loc = pred_boxlist1.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist1.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist1.get_field("labels")[valid_map])
    
    invalid_map = np.logical_not(valid_map)
    image_shape = pred_boxlist1.size
    wrong_box_loc = pred_boxlist1.bbox[invalid_map,:]
    wrong_boxes = BoxList(wrong_box_loc,image_shape)
    wrong_boxes.add_field("scores",pred_boxlist1.get_field("scores")[invalid_map])
    wrong_boxes.add_field("labels",pred_boxlist1.get_field("labels")[invalid_map]) 
    return new_boxes,wrong_boxes

def calculate_correct_area(pred_boxlist,gt_boxlist,width,height):
    correct_boxes, wrong_boxes = extract_correct_prediction(pred_boxlist,gt_boxlist)
    correct_area = correct_boxes.area()/(width*height)
    correct_score = correct_boxes.get_field("scores")
    
    wrong_area = wrong_boxes.area()/(width*height)
    wrong_score = wrong_boxes.get_field("scores")
    print("correct:",correct_area,correct_score)
    print("wrong:",wrong_area,wrong_score)

def extract_area_prediction(pred_boxlist,width,height):
    pred_area = pred_boxlist.area()/(height*width)
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    not_index = np.logical_and(pred_area>0.1,pred_score<0.51)
    index = np.logical_not(not_index)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    
    #not_index = np.logical_not(index)
    wrong_box_loc = pred_boxlist.bbox[not_index,:]
    wrong_boxes = BoxList(wrong_box_loc,image_shape)
    wrong_boxes.add_field("scores",pred_boxlist.get_field("scores")[not_index])
    wrong_boxes.add_field("labels",pred_boxlist.get_field("labels")[not_index])
    return new_boxes,wrong_boxes,index

def extract_small_prediction(pred_boxlist):
    pred_area = pred_boxlist.area()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    index = np.logical_or(pred_area<100,pred_score>0.85)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    return new_boxes
 
def analysis_duplicate_prediction(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores")
    pred_label = pred_boxlist.get_field("labels")
    
    valid_map = pred_score>1.0
    print(valid_map)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    iou = boxlist_iou(
    BoxList(pred_bbox,pred_boxlist.size),
            BoxList(pred_bbox,pred_boxlist.size),
    ).numpy()
    row,col = np.diag_indices_from(iou)
    iou[row,col] = 0
    index = np.where(iou>0.90)
    print(iou[index])
    if len(index[0])==0:
        image_shape = pred_boxlist.size
        new_box_loc = pred_boxlist.bbox[valid_map,:]
        new_boxes = BoxList(new_box_loc,image_shape)
        new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
        new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
        return new_boxes
    print(index)
    print(iou)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    for i in range(int(len(index[0])/2)):
        #print(index)
        #print(iou)
        if pred_label[index[0][i]]!=pred_label[index[1][i]]:
            if pred_score[index[0][i]]<pred_score[index[1][i]]:
                valid_map[index[1][i]] = True
                valid_map[index[0][i]] = False
            else:
                valid_map[index[0][i]] = True
                valid_map[index[1][i]] = False
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    print(valid_map)
    print("*****************************************88")
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    return new_boxes
    
def all_duplicate_prediction(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores")
    pred_label = pred_boxlist.get_field("labels")
    
    valid_map = pred_score>1.0
    iou = boxlist_iou(
    BoxList(pred_bbox,pred_boxlist.size),
            BoxList(pred_bbox,pred_boxlist.size),
    ).numpy()
    row,col = np.diag_indices_from(iou)
    iou[row,col] = 0
    index = np.where(iou>0.90)
    if len(index[0])==0:
        image_shape = pred_boxlist.size
        new_box_loc = pred_boxlist.bbox[valid_map,:]
        new_boxes = BoxList(new_box_loc,image_shape)
        new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
        new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
        return new_boxes
    for i in range(int(len(index[0])/2)):
        #print(index)
        #print(iou)
        if pred_label[index[0][i]]!=pred_label[index[1][i]]:
            if pred_score[index[0][i]]<pred_score[index[1][i]]:
                valid_map[index[1][i]] = True
                valid_map[index[0][i]] = True
            else:
                valid_map[index[0][i]] = True
                valid_map[index[1][i]] = True
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    print(valid_map)
    print("*****************************************88")
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    return new_boxes
   
def extract_duplicate_prediction(pred_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("scores")
    pred_label = pred_boxlist.get_field("labels")
    
    valid_map = pred_score>0.0
    iou = boxlist_iou(
            BoxList(pred_bbox,pred_boxlist.size),
            BoxList(pred_bbox,pred_boxlist.size),
    ).numpy()
    row,col = np.diag_indices_from(iou)
    iou[row,col] = 0
    index = np.where(iou>0.80)
    if len(index[0])==0:
        return pred_boxlist
    for i in range(int(len(index[0])/2)):
        #print(index)
        #print(iou)
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

def extract_rpn_target(pred_boxlist,gt_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_score = pred_boxlist.get_field("objectness").cpu().numpy()
    
    gt_bbox = gt_boxlist.bbox.cpu().numpy()
    gt_label = gt_boxlist.get_field("labels").cpu().numpy()
    gt_difficult = gt_boxlist.get_field("difficult").cpu().numpy()
    
    iou = boxlist_iou(
            BoxList(pred_bbox,gt_boxlist.size),
            BoxList(gt_bbox,gt_boxlist.size),
        ).numpy()
    gt_index = iou.argmax(axis=1)
    gt_index[iou.max(axis=1) < 0.5] = -1
    del iou
    index = np.logical_and(gt_index>=0, gt_difficult[gt_index] == 0)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    #new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    #new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    new_boxes.add_field("objectness",pred_boxlist.get_field("objectness")[index])
    
    not_valid_map = np.logical_not(index)
    wrong_box_loc = pred_boxlist.bbox[not_valid_map,:]
    wrong_boxes = BoxList(wrong_box_loc,image_shape)
    #wrong_boxes.add_field("scores",pred_boxlist.get_field("scores")[not_valid_map])
    #wrong_boxes.add_field("labels",pred_boxlist.get_field("labels")[not_valid_map])
    wrong_boxes.add_field("objectness",pred_boxlist.get_field("objectness")[not_valid_map])
    return new_boxes,wrong_boxes

def extract_target(pred_boxlist,gt_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    #pred_objectness = pred_boxlist.get_field("objectness").cpu().numpy()
    #print(pred_objectness)
    
    gt_bbox = gt_boxlist.bbox.cpu().numpy()
    gt_label = gt_boxlist.get_field("labels").cpu().numpy()
    gt_difficult = gt_boxlist.get_field("difficult").cpu().numpy()
    
    iou = boxlist_iou(
            BoxList(pred_bbox,gt_boxlist.size),
            BoxList(gt_bbox,gt_boxlist.size),
        ).numpy()
    gt_index = iou.argmax(axis=1)
    gt_index[iou.max(axis=1) < 0.5] = -1
    del iou
    index = np.logical_and(gt_index>=0, gt_difficult[gt_index] == 0)
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[index,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[index])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[index])
    
    not_valid_map = np.logical_not(index)
    wrong_box_loc = pred_boxlist.bbox[not_valid_map,:]
    wrong_boxes = BoxList(wrong_box_loc,image_shape)
    wrong_boxes.add_field("scores",pred_boxlist.get_field("scores")[not_valid_map])
    wrong_boxes.add_field("labels",pred_boxlist.get_field("labels")[not_valid_map])
    return new_boxes,wrong_boxes

def calculate_correct_number(prediction,gt_boxlist,correct_num,wrong_num):
    correct_prediction,wrong_prediction = extract_correct_prediction(prediction,gt_boxlist)
    correct_num+=len(correct_prediction)
    wrong_num+=len(wrong_prediction)
    return correct_num,wrong_num

def extract_correct_prediction(pred_boxlist,gt_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    
    gt_bbox = gt_boxlist.bbox.cpu().numpy()
    gt_label = gt_boxlist.get_field("labels").cpu().numpy()
    gt_difficult = gt_boxlist.get_field("difficult").cpu().numpy()
    
    valid_map = pred_score>1.0
    for l in np.unique(np.concatenate((pred_label,gt_label)).astype(int)):
        pred_mask_l = pred_label == l
        pred_bbox_l = pred_bbox[pred_mask_l]
        pred_score_l = pred_score[pred_mask_l]
        order_l = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order_l]
        pred_score_l = pred_score_l[order_l]
        
        gt_mask_l = gt_label == l
        gt_bbox_l = gt_bbox[gt_mask_l]
        gt_difficult_l = gt_difficult[gt_mask_l]
        
        if len(pred_bbox_l)==0:
            continue
        if len(gt_bbox_l)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        gt_bbox_l = gt_bbox_l.copy()
        gt_bbox_l[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_l,gt_boxlist.size),
            BoxList(gt_bbox_l,gt_boxlist.size),
        ).numpy()
        gt_index = iou.argmax(axis=1)
        gt_index[iou.max(axis=1) < 0.5] = -1
        del iou
        index = gt_index>=0
        #np.logical_and(gt_index>=0, gt_difficult_l[gt_index] == 0)
        valid_map[pred_mask_l] = index
    image_shape = pred_boxlist.size
    new_box_loc = pred_boxlist.bbox[valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    new_boxes.add_field("scores",pred_boxlist.get_field("scores")[valid_map])
    new_boxes.add_field("labels",pred_boxlist.get_field("labels")[valid_map])
    
    not_valid_map = np.logical_not(valid_map)
    wrong_box_loc = pred_boxlist.bbox[not_valid_map,:]
    wrong_boxes = BoxList(wrong_box_loc,image_shape)
    wrong_boxes.add_field("scores",pred_boxlist.get_field("scores")[not_valid_map])
    wrong_boxes.add_field("labels",pred_boxlist.get_field("labels")[not_valid_map])
    return new_boxes,wrong_boxes
        
def extract_missing_prediction(pred_boxlist,gt_boxlist):
    pred_bbox = pred_boxlist.bbox.cpu().numpy()
    pred_label = pred_boxlist.get_field("labels").cpu().numpy()
    pred_score = pred_boxlist.get_field("scores").cpu().numpy()
    
    gt_bbox = gt_boxlist.bbox.cpu().numpy()
    gt_label = gt_boxlist.get_field("labels").cpu().numpy()
    gt_difficult = gt_boxlist.get_field("difficult").cpu().numpy()
    
    valid_map = gt_label>21
    for l in np.unique(np.concatenate((pred_label,gt_label)).astype(int)):
        pred_mask_l = pred_label == l
        pred_bbox_l = pred_bbox[pred_mask_l]
        pred_score_l = pred_score[pred_mask_l]
        order_l = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order_l]
        pred_score_l = pred_score_l[order_l]
        
        gt_mask_l = gt_label == l
        gt_bbox_l = gt_bbox[gt_mask_l]
        gt_difficult_l = gt_difficult[gt_mask_l]
        
        if len(pred_bbox_l)==0:
            continue
        if len(gt_bbox_l)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        gt_bbox_l = gt_bbox_l.copy()
        gt_bbox_l[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_l,gt_boxlist.size),
            BoxList(gt_bbox_l,gt_boxlist.size),
        ).numpy()
        gt_index = iou.argmax(axis=0)
        gt_index[iou.max(axis=0) < 0.5] = -1
        del iou
        index = gt_index>=0
        valid_map[gt_mask_l] = index
    image_shape = gt_boxlist.size
    not_valid_map = np.logical_not(valid_map)
    new_box_loc = gt_boxlist.bbox[not_valid_map,:]
    new_boxes = BoxList(new_box_loc,image_shape)
    #new_boxes.add_field("scores",pred_boxlist.get_field("scores")[not_valid_map])
    new_boxes.add_field("labels",gt_boxlist.get_field("labels")[not_valid_map])
    return new_boxes
        
def test_gt_confidence(pred_boxlist,gt_boxlist):
    correct_boxlist,wrong_boxlist = extract_correct_prediction(pred_boxlist,gt_boxlist)
    correct_scores = correct_boxlist.get_field("scores").cpu().numpy()
    correct_labels = correct_boxlist.get_field("labels").cpu().numpy()
    
    wrong_scores = wrong_boxlist.get_field("scores").cpu().numpy()
    wrong_labels = wrong_boxlist.get_field("labels").cpu().numpy()

def test_gt_iou(pred_boxlist0,pred_boxlist1,gt_boxlist):
    correct_boxlist0,wrong_boxlist0 = extract_correct_prediction(pred_boxlist0,gt_boxlist)
    correct_boxlist1,wrong_boxlist1 = extract_correct_prediction(pred_boxlist1,gt_boxlist)
    
    pred_bbox0 = correct_boxlist0.bbox.cpu().numpy()
    pred_scores0 = correct_boxlist0.get_field("scores").cpu().numpy()
    pred_labels0 = correct_boxlist0.get_field("labels").cpu().numpy()
    
    pred_bbox1 = correct_boxlist1.bbox.cpu().numpy()
    pred_scores1 = correct_boxlist1.get_field("scores").cpu().numpy()
    pred_labels1 = correct_boxlist1.get_field("labels").cpu().numpy()
    
    for l in np.unique(np.concatenate((pred_labels0,pred_labels1)).astype(int)):
        pred_mask_l = pred_labels0 == l
        pred_bbox_l = pred_bbox0[pred_mask_l]
        pred_score_l = pred_scores0[pred_mask_l]
        order_l = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order_l]
        pred_score_l = pred_score_l[order_l]
        
        pred_mask_n = pred_labels1 == l
        pred_bbox_n = pred_bbox1[pred_mask_n]
        pred_score_n = pred_scores1[pred_mask_n]
        order_n = pred_score_n.argsort()[::-1]
        pred_bbox_n = pred_bbox_n[order_n]
        pred_score_n = pred_score_n[order_n]
        
        if len(pred_bbox_l)==0:
            continue
        if len(pred_bbox_n)==0:
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:,2:]+=1
        pred_bbox_n = pred_bbox_n.copy()
        pred_bbox_n[:,2:]+=1
        
        iou = boxlist_iou(
            BoxList(pred_bbox_l,pred_boxlist0.size),
            BoxList(pred_bbox_n,pred_boxlist0.size),
        ).numpy()
        print(iou)
        gt_index = iou.argmax(axis=1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)

def do_voc_evaluation(dataset, predictions):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        if not math.isnan(ap):
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i), ap
            )
    return result

def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    rec_results = 0
    prec_results = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.cpu().numpy()
        pred_label = pred_boxlist.get_field("labels").cpu().numpy()
        pred_score = pred_boxlist.get_field("scores").cpu().numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        prec_results += np.mean(prec[l])
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            rec_results += np.mean(rec[l])
    rec_results /= 6
    print(prec_results)
    prec_results /= 6
    print("!!!!!!!!!!!!!!!!!!")
    print(prec_results)
    print(rec_results)
    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap  