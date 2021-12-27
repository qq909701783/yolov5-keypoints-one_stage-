# -*- coding: utf-8 -*-
from .yolov5 import YOLOv5
from .swin_transformer import SwinTransformer

__all__ = ['build_backbone']

support_backbone = ['resnet', 'shufflenetv2', 'mobilenetv3', 'YOLOv5', 'efficientnet', 'hrnet', 'SwinTransformer']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
