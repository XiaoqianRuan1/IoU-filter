import os
import torch
import torch.utils.data
from PIL import Image
import sys
from maskrcnn_benchmark.structures.bounding_box import BoxList

class BDD100KDetDataset(torch.utils.data.Dataset):  # type: ignore
    """BDD100K Dataset for detecion."""
    CLASSES = [
        "__background__",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign"
    ]
    
    def __init__(self, data_dir, anno_dir, split, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.anno_dir = anno_dir
        self.transforms = transforms
        
        self._annopath = os.path.join(self.anno_dir,"%s/%s.txt")
        self._imgpath = os.path.join(self.root, "%s")
        self._imgsetpath = os.path.join(self.anno_dir, "%s/data.txt")
    
        with open(self._imgsetpath%self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k:v for k,v in enumerate(self.ids)}
        
        cls = BDD100KDetDataset.CLASSES
        self.class_to_ind = dict(zip(cls,range(len(cls))))
        self.categories = dict(zip(range(len(cls)),cls))
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        images = Image.open(self._imgpath % img_id).convert("RGB")
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            images,target = self.transforms(images,target)
        return images,target,index
        
    def __len__(self):
        return len(self.ids)
    
    def get_groundtruth(self, index):
        img_id = self.ids[index]
        img_id = img_id.split(".")[0]
        file_path = os.path.join(self.anno_dir,self.image_set, img_id+".txt")
        f = open(os.path.join(file_path))
        anno = f.readlines()
        anno = self._preprocess_annotation(anno)
        height,width = anno["im_info"][0], anno["im_info"][1]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target
    
    def _preprocess_annotation(self, targets):
        boxes = []
        gt_classes = []
        difficult_box = []
        TO_REMOVE = 1
        width = targets[0].split("\n")[0].split(":")[-1]
        height = targets[1].split("\n")[0].split(":")[-1]
        im_info = tuple(map(int, (height, width)))
        for target in targets[2:]:
            target = target.split("\n")[0]
            target_list = target.split(",")
            category = target_list[0].split(":")[-1].split("'")[1]
            x1 = target_list[1].split(":")[-1].split(" ")[-1]
            y1 = target_list[2].split(":")[-1].split(" ")[-1]
            x2 = target_list[3].split(":")[-1].split(" ")[-1]
            y2 = target_list[4].split(":")[-1].split("}}")[0].split(" ")[-1]
            difficult = 0
            gt_classes.append(self.class_to_ind[category])
            box = [float(x1), float(y1), float(x2), float(y2)]
            bndbox = tuple(map(lambda x: x-TO_REMOVE, list(map(int,box))))
            boxes.append(bndbox)
            difficult_box.append(difficult)
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_box),
            "im_info": im_info,
        }
        return res
    
    def get_img_info(self,index):
        img_id = self.ids[index].split(".")[0]
        file_path = os.path.join(self.anno_dir,self.image_set, img_id+".txt")
        f = open(os.path.join(file_path))
        anno = f.readlines()
        width = anno[0].split("\n")[0].split(":")[-1]
        height = anno[1].split("\n")[0].split(":")[-1]
        return {"height": height, "width": width}
    
    def map_class_id_to_class_name(self,class_id):
        return BDD100KDetDataset.CLASSES[class_id]

    """
    def convert_format(
        self, results: List[List[np.ndarray]], out_dir: str  # type: ignore
    ) -> None:
        Format the results to the BDD100K prediction format.
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"Length of res and dset not equal: {len(results)} != {len(self)}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        frames = []
        ann_id = 0

        for img_idx in range(len(self)):
            img_name = self.data_infos[img_idx]["file_name"]
            frame = Frame(name=img_name, labels=[])
            frames.append(frame)

            result = results[img_idx]
            for cat_idx, bboxes in enumerate(result):
                for bbox in bboxes:
                    ann_id += 1
                    label = Label(
                        id=ann_id,
                        score=bbox[-1],
                        box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
                        category=self.CLASSES[cat_idx],
                    )
                    frame.labels.append(label)  # type: ignore

        out_path = osp.join(out_dir, "det.json")
        save(out_path, frames)
    """

if __name__=="__main__":
    data_dir = "/mnt/sde1/xiaoqianruan/OSHOT/datasets/bdd100k/"
    image_dir = "images/100k/val"
    ann_file =  "labels1"
    image_dir = os.path.join(data_dir,image_dir)
    ann_dir = os.path.join(data_dir,ann_file)
    split = "rainy"
    bdd_data = BDD100KDetDataset(image_dir,ann_dir,split)
    number = bdd_data.__len__()
    for index in range(number):
        #target = bdd_data.get_groundtruth(index)
        #print(target)
        print("!!!!!!!!!!!!!!!!!")
        results = bdd_data.__getitem__(index)
    