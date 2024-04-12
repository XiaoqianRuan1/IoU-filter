import numpy as np
import cv2

def do_voc_evaluation1(dataset, predictions):
    for image_id, prediction in enumerate(predictions):
        image_name = dataset.get_img_name(image_id)
        data_path = "/mnt/sde1/xiaoqianruan/H2FA_R-CNN-main/DETECTRON2_DATASETS/clipart/JPEGImages/"
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()
        image = cv2.imread(data_path + str(image_name) + ".jpg")
        output_folder = "./outputs/clipart_images/"
        for index, box in enumerate(gt_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 255, 0), 2)
            t_size = cv2.getTextSize(str(gt_label[index]), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textlbottom = a + np.array(list(t_size))
            cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 255, 0), -1)
            a = list(a)
            a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            label = gt_label[index]
            cv2.putText(image, str(label), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()
        for index, box in enumerate(pred_bbox):
            xmin, ymin, xmax, ymax = box
            a = (int(xmin), int(ymin))
            b = (int(xmax), int(ymax))
            cv2.rectangle(image, a, b, (0, 0, 255), 2)
            t_size = cv2.getTextSize(str(pred_label[index]), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textlbottom = a + np.array(list(t_size))
            cv2.rectangle(image, tuple(a), tuple(textlbottom), (0, 0, 255), -1)
            a = list(a)
            a[1] = int(a[1] + (list(t_size)[1] / 2 + 4))
            cv2.putText(image, str(pred_label[index]), tuple(a), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            score_size = cv2.getTextSize(str(pred_score[index]), 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            textrbottom = b - np.array(list(score_size))
            cv2.rectangle(image, tuple(textrbottom), tuple(b), (0, 0, 255), -1)
            b = list(b)
            cv2.putText(image, str(pred_score[index]), (textrbottom[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1)
        cv2.imwrite(output_folder + str(image_id) + ".jpg", image)