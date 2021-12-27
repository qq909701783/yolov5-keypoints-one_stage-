import random
import cv2
import numpy as np

class ConvertKeypoints:
    def __call__(self, sample):
        label = sample['label']
        h, w, _ = sample['image'].shape
        keypoints = label['keypoints']
        for keypoint in keypoints:  # keypoint[2] == 0: occluded, == 1: visible, == 2: not in image
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2
            if (keypoint[0] < 0
                    or keypoint[0] >= w
                    or keypoint[1] < 0
                    or keypoint[1] >= h):
                keypoint[2] = 2
        for other_label in label['processed_other_annotations']:
            keypoints = other_label['keypoints']
            for keypoint in keypoints:
                if keypoint[0] == keypoint[1] == 0:
                    keypoint[2] = 2
                if (keypoint[0] < 0
                        or keypoint[0] >= w
                        or keypoint[1] < 0
                        or keypoint[1] >= h):
                    keypoint[2] = 2
        label['keypoints'] = self._convert(label['keypoints'], w, h)

        for other_label in label['processed_other_annotations']:
            other_label['keypoints'] = self._convert(other_label['keypoints'], w, h)
        return sample

    def _convert(self, keypoints, w, h):
        #### 因为与人体不同，所以忽略这一步 ####
        # Nose, Neck, R hand, L hand, R leg, L leg, Eyes, Ears
        reorder_map = [1, 2, 3, 4]
        converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
        # converted_keypoints.insert(1, [(keypoints[5][0] + keypoints[6][0]) / 2,
        #                                (keypoints[5][1] + keypoints[6][1]) / 2, 0])  # Add neck as a mean of shoulders
        # if keypoints[5][2] == 2 or keypoints[6][2] == 2:
        #     converted_keypoints[1][2] = 2
        # elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
        #     converted_keypoints[1][2] = 1
        # if (converted_keypoints[1][0] < 0
        #         or converted_keypoints[1][0] >= w
        #         or converted_keypoints[1][1] < 0
        #         or converted_keypoints[1][1] >= h):
        #     converted_keypoints[1][2] = 2
        return converted_keypoints
