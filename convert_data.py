import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2,json
import numpy as np

imspath = r'train_datas/images'
lbspath = r'train_datas/labels'
yoimgspath = r'train_datas/yoloimages'
yoloslabel = r'train_datas/yololabels'
key_json = r'train_datas/keypoints_label.json'
item_list = {"info": {"description": "2022 data",
                      "year": 2022,
                      "date_created": "2022"},
             "categories": [{"supercategory": "A",
                             "id": 1,
                             "name": "A",
                             "keypoints": ['0', '1', '2', '3', '4'],
                             "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]
                             }],
             "images": [],
             "annotations": [], }
input_size = (640,640)

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


id = 0
for imname in os.listdir(imspath):
    cls_id = 0
    impath = os.path.join(imspath,imname)
    saveimpath = os.path.join(yoimgspath,imname)
    img = cv2.imread(impath)
    h,w,_ = img.shape
    s_h,s_w = h / input_size[0],w / input_size[1]
    new_img = cv2.resize(img,input_size)

    lbname = imname.replace('jpg','txt')
    lbpath = os.path.join(lbspath,lbname)
    yolbpath = os.path.join(yoloslabel,lbname)
    txtlabel = open(lbpath,'r').readlines()
    txtyolb = open(yolbpath,'w')

    item_list['images'].append({"file_name": str(imname), "id": id, "height": input_size[0], "width": input_size[1], "crowdIndex": 0.1})
    for lb in txtlabel:
        lb = lb.strip()
        points = []
        is_y = False
        for i,key in enumerate(lb.split(',')[0:8]):
            key = key.replace('\ufeff','')
            if is_y==True:
                is_y=False
                key = float(key) / s_h
                points.append(key)
                points.append(2)
                continue
            elif i%2 == 0:
                key = float(key) / s_w
                points.append(key)
                is_y = True


        k_w,k_h = points[3]-points[0],points[7]-points[1]
        c_x, c_y = (points[3]+points[0])/2,(points[7]+points[1])/2
        box = [c_x,c_y,k_w,k_h]
        yo_box = ((c_x-k_w/2),(c_x+k_w/2),(c_y-k_h/2),(c_y+k_h/2))
        yo_bbox = convert(input_size,yo_box)
        txtyolb.write(str(cls_id) + " " + " ".join([str(a) for a in yo_bbox]) + '\n')
        area = (points[3]-points[0])*(points[7]-points[1])

        item_list['annotations'].append(
            {"num_keypoints": 8, "area": area, "iscrowd": 0, "keypoints": points, "image_id": id, "bbox": box,
             "category_id": 1, "id": 0})

    id+=1
    cv2.imwrite(saveimpath,new_img)

with open(key_json, 'w', encoding='utf-8') as f2:
    json.dump(item_list, f2, ensure_ascii=False,indent=2)
