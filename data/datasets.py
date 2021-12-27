import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import pickle
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, resample_segments, \
    clean_str
from utils.torch_utils import torch_distributed_zero_first
from data.convertKeypoints import ConvertKeypoints
from torchvision import transforms
from data.prepare_train_labels import Main as prep_label


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)
BODY_PARTS_KPT_IDS = [[0, 1], [1, 2], [2, 3], [3,0]]

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size,rank=-1, world_size=1, workers=8):
    # with torch_distributed_zero_first(rank):
    print(imgsz)
    dataset = LoadImagesAndLabels(path,imgsz)
    batch_size = min(batch_size, len(dataset))
    nw = workers
    loader = torch.utils.data.DataLoader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn)

    return dataloader, dataset



class LoadImagesAndLabels(Dataset):
    def __init__(self,path,input_size):
        self.path = path
        self.input_size = input_size
        self.keypoint_ann = prep_label(os.path.join(path,'keypoints_label.json'),input_size[0])
        self.images_folder = os.path.join(self.path,'yoloimages')
        self.label_folder = os.path.join(self.path,'yololabels')
        self._stride = 8
        self._sigma = 7
        self._paf_thickness = 1

    def __len__(self):
        return len(self.keypoint_ann)

    def __getitem__(self, index):
        label = copy.deepcopy(self.keypoint_ann[index])
        label_path = os.path.join(self.label_folder, label['img_paths'].replace('.jpg', '.txt'))
        img = cv2.imread(os.path.join(self.images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        sample = {'label': label,'image': img,'mask': mask}

        sample = transforms.Compose([ConvertKeypoints()])(sample)
        mask = cv2.resize(sample['mask'], dsize=None, fx=1 / self._stride, fy=1 / self._stride,interpolation=cv2.INTER_AREA)
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = torch.from_numpy(keypoint_maps)
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        mask = cv2.resize(mask, (keypoint_mask.shape[2], keypoint_mask.shape[1]), interpolation=cv2.INTER_AREA)
        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask
        sample['keypoint_mask'] = torch.from_numpy(keypoint_mask)

        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = torch.from_numpy(paf_maps)
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for idx in range(paf_mask.shape[0]): paf_mask[idx] = mask
        sample['paf_mask'] = torch.from_numpy(paf_mask)
        del sample['label']

        h,w = self.input_size
        with open(label_path,'r') as f:labels = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:],w,h)

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        sample['image'] = img

        return sample,labels_out



    def _generate_keypoint_maps(self, sample):
        n_keypoints = 4
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps


    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _generate_paf_maps(self, sample):
        n_pafs = len(BODY_PARTS_KPT_IDS)
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
            keypoint_b = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                              keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                              self._stride, self._paf_thickness)
            for another_annotation in label['processed_other_annotations']:
                keypoint_a = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
                keypoint_b = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
                if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                    self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                                  keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                                  self._stride, self._paf_thickness)
        return paf_maps


    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba
        y_ba /= norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba

    @staticmethod
    def collate_fn(batch):
        sample, label = zip(*batch)  # transposed
        images,keypoint_mask,paf_mask,keypoint_maps,paf_maps = [],[],[],[],[]
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
            images.append(sample[i]['image'])
            keypoint_mask.append(sample[i]['keypoint_mask'])
            paf_mask.append(sample[i]['paf_mask'])
            keypoint_maps.append(sample[i]['keypoint_maps'])
            paf_maps.append(sample[i]['paf_maps'])

        return (torch.stack(images,0),torch.stack(keypoint_mask,0),torch.stack(paf_mask,0),torch.stack(keypoint_maps,0), torch.stack(paf_maps,0)),torch.cat(label, 0)

if __name__ == '__main__':
    # Trainloader
    train_path = r'../train_datas'
    imgsz = [640,640]
    batch_size = 4
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size)
    for data,target in dataloader:
        print(data[0].shape)