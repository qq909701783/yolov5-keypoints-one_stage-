import cv2
import numpy as np
import torch
from utils.keypoints import group_keypoints,extract_keypoints
from models.modules.experimental import *
from models.modules.pose import *
from models import Model


stride = 8
upsample_ratio = 4
num_keypoints = 4
device = torch.device('cuda')


def demo(model,img_path):
    img = cv2.imread(img_path)
    draw = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    out = model(img)
    stages_output = out[1]
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

    id = 0
    for keys in all_keypoints_by_type:
        cls_id = 0
        if keys==[]:
            id +=1
            continue
        for key in keys:
            x = (int(key[0]) * stride / upsample_ratio)
            y = (int(key[1]) * stride / upsample_ratio)
            draw = cv2.circle(draw, (int(x), int(y)),radius=2, color=255, thickness=-1, lineType=cv2.LINE_AA)
            # draw = cv2.putText(draw, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 1)
            cls_id+=1
        id+=1

    cv2.imwrite('test.png',draw)

if __name__ == '__main__':
    pt_path = r'runs/train/exp/weights/last.pt'
    img_path = r'train_datas/yoloimages/0.jpg'
    cfg = r'configs/model_yolo.yaml'
    # model = Model(cfg).to(device)
    # model.load_state_dict(torch.load(pt_path)['model'])
    model = attempt_load(pt_path, map_location=device)  # load FP32 model
    demo(model,img_path)