import numpy as np
import json

from .AutoDriveDataset_multi import AutoDriveDataset_multi
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = True       # just detect vehicle

class HeFeiDataset(AutoDriveDataset_multi):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            # label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            label_path = mask_path
            
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".txt", ".jpg")
            # lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            gt0, gt1, gt2 = self.parse_yolo_annotation_class(label_path)
            # 转成numpy 格式
            gt0 = np.array(gt0)
            gt1 = np.array(gt1)
            gt2 = np.array(gt2)
            # print("here is gt0: ", gt0)
            rec = [{
                'image': image_path,
                'label': gt0,
                'mask': gt1,
                'lane': gt2
            }]

            gt_db += rec
        print('database build finish')
        return gt_db
    
    
    # 解析YOLO标注文件
    def parse_yolo_annotation(self, annotation_file):
        with open(annotation_file, 'r') as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            annotations.append((class_id, x, y, width, height))
        return annotations
    
    
    # # 根据目标size的不同，划分成不同label组，用于不同的head
    # def parse_yolo_annotation_class(self, annotation_file):
        
    #     # 定义类别映射关系
    #     # classes = {
    #     #   
    #     #     "traffic-signal-system_good": 0,
    #     #     "traffic-signal-system_bad": 1,
    #     #     "traffic-guidance-system_good": 2,
    #     #     "traffic-guidance-system_bad": 3,
    #     #     "restricted-elevated_good": 4,
    #     #     "restricted-elevated_bad": 5,
    #     #     "cabinet_good": 6,
    #     #     "cabinet_bad": 7,
    #     #     "backpack-box_good": 8,
    #     #     "backpack-box_bad": 9,
    #     #     "off-site": 10,
    #     #     "Gun-type-Camera": 11,
    #     #     "Dome-Camera": 12,
    #     #     "Flashlight": 13,
    #     #     "b-Flashlight": 14
    #     # }
        
    #     with open(annotation_file, 'r') as file:
    #         lines = file.readlines()
            
    #     # 先假设有三组，分别是0.限高架、off-site
    #                     #   1.交通信号灯
    #                     #   2.交通诱导、机柜、背包箱、off-site上的小目标
    #     annotations_0 = []
    #     annotations_1 = []
    #     annotations_2 = []
    #     # 构建三类映射关系
    #     class_map_0 = {4:0, 5:1, 10:2}
    #     class_map_1 = {0:0, 1:1}
    #     class_map_2 = {2:0, 3:1, 6:2, 7:3, 8:4, 9:5, 11:6, 12:7, 13:8, 14:9}
    #     for line in lines:
    #         parts = line.strip().split()
    #         class_id = int(parts[0])
    #         x, y, width, height = map(float, parts[1:])
    #         if class_id in class_map_0.keys():
    #             new_class_id  = class_map_0[class_id]
    #             annotations_0.append([new_class_id , x, y, width, height])
    #         elif class_id in class_map_1.keys():
    #             new_class_id  = class_map_1[class_id]
    #             annotations_1.append([new_class_id , x, y, width, height])
    #         elif class_id in class_map_2:
    #             new_class_id  = class_map_2[class_id]
    #             annotations_2.append([new_class_id , x, y, width, height])
    #     return annotations_0, annotations_1, annotations_2

    # 根据目标size的不同，划分成不同label组，用于不同的head
    def parse_yolo_annotation_class(self, annotation_file):
        
        # 定义类别映射关系
        # classes = {
        #   
        #     "traffic-signal-system_good": 0,
        #     "traffic-signal-system_bad": 1,
        #     "traffic-guidance-system_good": 2,
        #     "traffic-guidance-system_bad": 3,
        #     "restricted-elevated_good": 4,
        #     "restricted-elevated_bad": 5,
        #     "cabinet_good": 6,
        #     "cabinet_bad": 7,
        #     "backpack-box_good": 8,
        #     "backpack-box_bad": 9,
        #     "off-site": 10,
        #     "Gun-type-Camera": 11,
        #     "Dome-Camera": 12,
        #     "Flashlight": 13,
        #     "b-Flashlight": 14
        # }
        
        with open(annotation_file, 'r') as file:
            lines = file.readlines()
            
        # 先假设有三组，分别是0.限高架、off-site
                        #   1.交通信号灯
                        #   2.交通诱导、机柜、背包箱、off-site上的小目标
        annotations_0 = []
        annotations_1 = []
        annotations_2 = []
        # 构建三类映射关系
        class_map_0 = {4:0, 5:1, 10:2}
        class_map_1 = {0:0, 1:1}
        class_map_2 = {2:0, 3:1, 6:2, 7:3, 8:4, 9:5, 11:6, 12:7, 13:8, 14:9}
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            if class_id in class_map_0.keys():
                # new_class_id  = class_map_0[class_id]
                new_class_id  = class_id
                annotations_0.append([new_class_id , x, y, width, height])
            elif class_id in class_map_1.keys():
                # new_class_id  = class_map_1[class_id]
                new_class_id = class_id
                annotations_1.append([new_class_id , x, y, width, height])
            elif class_id in class_map_2:
                # new_class_id  = class_map_2[class_id]
                new_class_id = class_id
                annotations_2.append([new_class_id , x, y, width, height])
        return annotations_0, annotations_1, annotations_2

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
