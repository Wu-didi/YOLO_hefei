import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, letterbox_for_img, augment_hsv, random_perspective, xyxy2xywh, cutout, random_perspective_for_img


class AutoDriveDataset_multi(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        
        img, ratio, pad = letterbox_for_img(img, resized_shape, auto=False, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label_0 = data["label"]
        det_label_1 = data["mask"]
        det_label_2 = data["lane"]
        labels_0 = []
        labels_1 = []
        labels_2 = []
                
        if det_label_0.size > 0:
            # Normalized xywh to pixel xyxy format
            labels_0 = det_label_0.copy()
            labels_0[:, 1] = ratio[0] * w * (det_label_0[:, 1] - det_label_0[:, 3] / 2) + pad[0]  # pad width
            labels_0[:, 2] = ratio[1] * h * (det_label_0[:, 2] - det_label_0[:, 4] / 2) + pad[1]  # pad height
            labels_0[:, 3] = ratio[0] * w * (det_label_0[:, 1] + det_label_0[:, 3] / 2) + pad[0]
            labels_0[:, 4] = ratio[1] * h * (det_label_0[:, 2] + det_label_0[:, 4] / 2) + pad[1]
        
        if det_label_1.size > 0:
            # Normalized xywh to pixel xyxy format
            labels_1 = det_label_1.copy()
            labels_1[:, 1] = ratio[0] * w * (det_label_1[:, 1] - det_label_1[:, 3] / 2) + pad[0]  # pad width
            labels_1[:, 2] = ratio[1] * h * (det_label_1[:, 2] - det_label_1[:, 4] / 2) + pad[1]  # pad height
            labels_1[:, 3] = ratio[0] * w * (det_label_1[:, 1] + det_label_1[:, 3] / 2) + pad[0]
            labels_1[:, 4] = ratio[1] * h * (det_label_1[:, 2] + det_label_1[:, 4] / 2) + pad[1]
         
        if det_label_2.size > 0:
            # Normalized xywh to pixel xyxy format
            labels_2 = det_label_2.copy()
            labels_2[:, 1] = ratio[0] * w * (det_label_2[:, 1] - det_label_2[:, 3] / 2) + pad[0]  # pad width
            labels_2[:, 2] = ratio[1] * h * (det_label_2[:, 2] - det_label_2[:, 4] / 2) + pad[1]  # pad height
            labels_2[:, 3] = ratio[0] * w * (det_label_2[:, 1] + det_label_2[:, 3] / 2) + pad[0]
            labels_2[:, 4] = ratio[1] * h * (det_label_2[:, 2] + det_label_2[:, 4] / 2) + pad[1]
         
        labels = [labels_0, labels_1, labels_2]    
        if self.is_train:
            # # combination = (img, seg_label, lane_label)
            # img, labels = random_perspective_for_img(
            #     combination=img,
            #     targets=labels,
            #     degrees=self.cfg.DATASET.ROT_FACTOR,
            #     translate=self.cfg.DATASET.TRANSLATE,
            #     scale=self.cfg.DATASET.SCALE_FACTOR,
            #     shear=self.cfg.DATASET.SHEAR
            # )
            # #print(labels.shape)
            # augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
            # # img, seg_label, labels = cutout(combination=combination, labels=labels)
            # labels_0, labels_1, labels_2 = labels[0], labels[1], labels[2]
            
            if len(labels_0):
                # convert xyxy to xywh
                labels_0[:, 1:5] = xyxy2xywh(labels_0[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_0[:, [2, 4]] /= img.shape[0]  # height
                labels_0[:, [1, 3]] /= img.shape[1]  # width
            
            if len(labels_1):
                # convert xyxy to xywh
                labels_1[:, 1:5] = xyxy2xywh(labels_1[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_1[:, [2, 4]] /= img.shape[0]  # height
                labels_1[:, [1, 3]] /= img.shape[1]  # width
                
            if len(labels_2):
                # convert xyxy to xywh
                labels_2[:, 1:5] = xyxy2xywh(labels_2[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_2[:, [2, 4]] /= img.shape[0]  # height
                labels_2[:, [1, 3]] /= img.shape[1]  # width

        
        else:
            if len(labels_0):
                # convert xyxy to xywh
                labels_0[:, 1:5] = xyxy2xywh(labels_0[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_0[:, [2, 4]] /= img.shape[0]  # height
                labels_0[:, [1, 3]] /= img.shape[1]  # width
            
            if len(labels_1):
                # convert xyxy to xywh
                labels_1[:, 1:5] = xyxy2xywh(labels_1[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_1[:, [2, 4]] /= img.shape[0]  # height
                labels_1[:, [1, 3]] /= img.shape[1]  # width
                
            if len(labels_2):
                # convert xyxy to xywh
                labels_2[:, 1:5] = xyxy2xywh(labels_2[:, 1:5])
                # Normalize coordinates 0 - 1
                labels_2[:, [2, 4]] /= img.shape[0]  # height
                labels_2[:, [1, 3]] /= img.shape[1]  # width


        labels_out_0 = torch.zeros((len(labels_0), 6))
        labels_out_1 = torch.zeros((len(labels_1), 6))
        labels_out_2 = torch.zeros((len(labels_2), 6))
        if len(labels_0):
            labels_out_0[:, 1:] = torch.from_numpy(labels_0)
            
        if len(labels_1):
            labels_out_1[:, 1:] = torch.from_numpy(labels_1)
            
        if len(labels_2):
            labels_out_2[:, 1:] = torch.from_numpy(labels_2)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        
        target = [labels_out_0, labels_out_1, labels_out_2]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            
            l_seg[:, 0] = i  # add target image index for build_targets()
            label_seg.append(l_seg)
            
            l_lane[:, 0] = i  # add target image index for build_targets()
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.cat(label_seg, 0), torch.cat(label_lane, 0)], paths, shapes

