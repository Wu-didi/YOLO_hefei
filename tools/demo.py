import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    det_1_names = model.module.det1_names if hasattr(model, 'module') else model.det1_names
    det_2_names = model.module.det2_names if hasattr(model, 'module') else model.det3_names
    det_3_names = model.module.det3_names if hasattr(model, 'module') else model.det3_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(det_1_names)+len(det_2_names)+len(det_3_names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, det_out_1, det_out_2 = model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_out_1, _ = det_out_1
        inf_out_2, _ = det_out_2
        
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        det_pred_1 = non_max_suppression(inf_out_1, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        det_pred_2 = non_max_suppression(inf_out_2, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()
        nms_time.update(t4-t3,img.size(0))
        
        det=det_pred[0]
        det_1=det_pred_1[0]
        det_2=det_pred_2[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # 第一个检测器
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{det_1_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            cv2.imwrite("img_det_1.jpg",img_det)
            
        # # 第二个检测器
        # if len(det_1):
        #     det_1[:,:4] = scale_coords(img.shape[2:],det_1[:,:4],img_det.shape).round()
        #     for *xyxy,conf,cls in reversed(det_1):
        #         label_det_pred = f'{det_2_names[int(cls)]} {conf:.2f}'
        #         plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        #     cv2.imwrite("img_det_2.jpg",img_det)
        # # 第三个检测器
        # if len(det_2):
        #     det_2[:,:4] = scale_coords(img.shape[2:],det_2[:,:4],img_det.shape).round()
        #     for *xyxy,conf,cls in reversed(det_2):
        #         label_det_pred = f'{det_3_names[int(cls)]} {conf:.2f}'
        #         plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        #     cv2.imwrite("img_det_3.jpg",img_det)
                

        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/wudi/code_v2/YOLO_hefei/runs/HeFeiDataset/_2024-06-17-14-46/epoch-100.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/home/wudi/code_v2/YOLO_hefei/datasets/hefei-dataset/hefei_multi/images/train', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
