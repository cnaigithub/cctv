import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tqdm import tqdm
import os
def calculate_iou(box1, box2):
    # box1 and box2 should be in the format (x1, y1, x2, y2)
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])
    
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area
    return iou

def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])

def get_vsize_lst():
    rts = ['/mnt/cn_hdd/video2', '/mnt/cn_hdd/video']
    vsize_dic = {}
    for rt in rts:    
        print('extracting vsize dic . . .')
        folders = [i for i in os.listdir(rt) if not i.startswith('.')]
        for folder in tqdm(folders):
            files = [i for i in os.listdir(os.path.join(rt,folder)) if not i.startswith('.')]
            for file in tqdm(files):        
                print(f'processing {file}')
                fname = file.split('/')[-1].split('.')[0]
                #vid2frame
                if '.mp4' in file:
                    vid = cv2.VideoCapture(os.path.join(rt,folder,file))
                    ret, frame = vid.read()
                    vsize_dic[fname] = frame.shape[:2] 
                    vid.release()
                    cv2.destroyAllWindows()
    return vsize_dic
            
def xywh2yolo(size,x,y,w,h):
    dh = 1./size[0]
    dw = 1./size[1]
    return (x*dw, y*dh, w*dw, h*dh)
def xywh2xyxy(lst):
    x,y,w,h = lst[0],lst[1],lst[2],lst[3]
    return [x, y, x+w, y+h]


def detect(save_img=False, source = '',bo = True):
    weights, view_img, save_txt, imgsz, trace = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    #LOAD GT LABELS
    source_fname = source.split('/')[-1].split('.')[0]

    # vsize_dic = get_vsize_lst()
    import pickle
    with open('vsize_dic.pkl', 'rb') as f:
        vsize_dic = pickle.load(f)

    label_gt = f'{source.split(source_fname)[0]}/{source_fname}.txt'
    gt_f = open(label_gt, 'r')
    lines = [l.strip() for l in gt_f.readlines()]
    gt_f.close()
    labels_per_frame = {}
    for line in lines:
        split_line = line.split(',')
        frame_num = split_line[0]
        ret = [int(x) for x in split_line[3:]]
        if frame_num not in labels_per_frame.keys():
            labels_per_frame[frame_num] = []
        labels_per_frame[frame_num].append(xywh2xyxy(ret))
    seen =0
    t0 = time.time()
    f1s = []
    tp, tn, fp, fn = 0,0,0,0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        
        no_target = False

        if str(seen) in labels_per_frame.keys():
            tbox = labels_per_frame[str(seen)] #lst
        else:
            tbox = [[]]
            no_target = True
    
        predbox = pred.copy() #lst of tensor
        correct = []

        skip_frame =False
        if bo:
            if not no_target:
                for p_box in predbox:
                    res = []
                    if len(p_box.tolist()) !=0:
                        p_box = [int(x) for x in p_box.tolist()[0][:4]]
                    else:
                        continue
                    for t_box in tbox:
                        res.append(calculate_iou(p_box,t_box))
                    if max(res)>0.5:
                        correct.append(max(res))
            if no_target:
                lt = 0
                if len(predbox) ==1:
                    if len(predbox[0].tolist()) !=0:
                        lp=0
                    else:
                        lp = len(predbox)
            else:
                lt = len(tbox)
                lp = len(correct)
            
            if lt == lp:
                tp +=1
            elif lt!=0 and lp==0:
                tn +=1
            elif lt<lp:
                fp +=1
            elif lt>lp:
                fn +=1
        
        else: #standard way of evaluating f1 score
            if not no_target:
                for p_box in predbox:
                    max_iou=0
                    max_iou_idx = -1
                    if len(p_box.tolist()) !=0:
                        p_box = [int(x) for x in p_box.tolist()[0][:4]]
                    else:
                        skip_frame = True
                        break

                    for idx, t_box in enumerate(tbox):
                        iou = (calculate_iou(p_box,t_box))
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_idx = idx

                    if max_iou>=0.1:
                        tp +=1
                        tbox.pop(max_iou_idx)
                    else:
                        fp+=1
                if skip_frame:
                    seen+=1
                    continue
                fn = len(tbox)
                tn = len(predbox) - tp
            else:
                seen+=1
                continue

            if tp == 0:
                pr = 0
                rc = 0
            else:
                rc = tp / (tp + fn)
                pr = tp /(tp+fp)
            if pr+rc ==0:
                f1 =0
            else:
                f1 = 2*((pr*rc)/(pr+rc))
            f1s.append(f1)
        seen +=1
        print(seen)
    if bo:
        if tp == 0:
            tp = 0.0001
        rc = tp / (tp + fn)
        pr = tp /(tp+fp)
        f1 = 2*((pr*rc)/(pr+rc))
        return rc, pr, f1 , tp, tn, fp, fn
    else:
        if len(f1s) ==0: 
            return 0
        else:
            return sum(f1s) / len(f1s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            f1s = []
            prs = []
            rcs = []

            with open(f'./f1_res/{opt.name}', 'w') as res_txt:
                rt = opt.source
                files = [i for i in os.listdir(rt) if not i.startswith('.')]
                for f_idx, file in enumerate(files):
                    print(f_idx, file)
                    if '.mp4' in file:
                        #regular
                        ret = detect(source = os.path.join(rt,file), bo=False)
                        f1s.append(ret)
                        print(f'FILE NAME : [{file}] f1 score: {ret}')
                        res_txt.write(f'FILE NAME : [{file}] f1 score: {ret}\n')
                        print(f'f1 score: {sum(f1s)/len(f1s)}')
                breakpoint()