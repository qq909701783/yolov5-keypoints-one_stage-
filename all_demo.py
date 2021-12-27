import cv2
import numpy as np
import torch
from utils.general import *
from models.modules.experimental import *
from utils.keypoints import extract_keypoints


stride = 8
upsample_ratio = 4
num_keypoints = 4
conf_thres = 0.7
iou_thres = 0.15
device = torch.device('cuda')


def getDist_P2P(Point0,PointA):
    distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance=math.sqrt(distance)
    return distance

def Get_newxy(box,points):
    mindis = 1000
    new_point = [0,0]
    for point in points:
        tmp_dis = getDist_P2P(box,point)
        if tmp_dis < mindis:
            mindis = tmp_dis
            new_point = point

    return new_point


def demo(model, img_path):
    img = cv2.imread(img_path)
    draw = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    out = model(img)
    pred,stages_output = out[0][0],out[1]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    ###key_points###
    stages_output = stages_output[-2]
    heatmaps = np.transpose(stages_output.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage_pafs = stages_output[-1]
    pafs = np.transpose(stage_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
    Points = []
    for keys in all_keypoints_by_type:
        for key in keys:
            x = (int(key[0]) * stride / upsample_ratio)
            y = (int(key[1]) * stride / upsample_ratio)
            Points.append((x,y))

    for i, det in enumerate(pred):
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                x_0 = xyxy[0].cpu()
                y_0 = xyxy[1].cpu()
                x_2 = xyxy[2].cpu()
                y_2 = xyxy[3].cpu()
                x_1 = x_2
                y_1 = y_0
                x_3 = x_0
                y_3 = y_2

                new_xy0 = Get_newxy((x_0,y_0),Points)
                new_xy1 = Get_newxy((x_1,y_1),Points)
                new_xy2 = Get_newxy((x_2,y_2),Points)
                new_xy3 = Get_newxy((x_3,y_3),Points)
                pp = [new_xy0,new_xy1,new_xy2,new_xy3]
                n_pp = np.array(pp,dtype=np.int32)
                cv2.drawContours(draw,[n_pp],-1,255,thickness=2)


    cv2.imwrite('end.jpg', draw)


if __name__ == '__main__':
    pt_path = r'runs/train/exp/weights/last.pt'
    img_path = r'train_datas/yoloimages/0.jpg'
    model = attempt_load(pt_path, map_location=device)  # load FP32 model
    demo(model, img_path)