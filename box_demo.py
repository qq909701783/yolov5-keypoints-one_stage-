import cv2
import numpy as np
import torch
from utils.general import *
from models.modules.experimental import *


conf_thres = 0.7
iou_thres = 0.15
device = torch.device('cuda')


def demo(model,img_path):
    img = cv2.imread(img_path)
    draw = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:img = img.unsqueeze(0)
    pred, stages_output = model(img)[0]
    print(pred)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,agnostic=False)

    for i, det in enumerate(pred):
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                x_min = xyxy[0].cpu()
                y_min = xyxy[1].cpu()
                x_max = xyxy[2].cpu()
                y_max = xyxy[3].cpu()
                score = conf.cpu()
                clas = cls.cpu()
                cv2.rectangle(draw,(x_min,y_min),(x_max,y_max),255,thickness=2)

    cv2.imwrite('test.jpg',draw)


if __name__ == '__main__':
    pt_path = r'runs/train/exp/weights/last.pt'
    img_path = r'train_datas/yoloimages/0.jpg'
    model = attempt_load(pt_path, map_location=device)  # load FP32 model
    demo(model,img_path)