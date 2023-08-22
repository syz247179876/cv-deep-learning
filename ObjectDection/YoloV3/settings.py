"""
Global config
"""
# number of each residual block
DRK_53_RESIDUAL_BLOCK_NUMS = [1, 2, 8, 8, 4]
# output channels of each residual block
DRK_53_LAYER_OUT_CHANNELS = [64, 128, 256, 512, 1024]
# the number of anchors in each scale
ANCHORS_NUM = 3
# class of VOC dataset
VOC_CLASS_NUM = 20
# class of COCO dataset
COCO_CLASS_NUM = 80
# anchors obtained based on clustering algorithm using a distance of 1 - iou(anchors, gt_box)
ANCHORS = [[(116, 90), (156, 198), (327, 326)],
           [(30, 61), (64, 45), (59, 119)],
           [(10, 13), (16, 30), (33, 23)]]
ANCHORS_SORT = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
ANCHORS_MASK = [
    [6, 7, 8],
    [3, 4, 5],
    [0, 1, 2]
]
# VOC dataset category
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
# VOC dataset category num
VOC_CLASSES_LEN = 20
# xml dir
ANNOTATIONS_DIR = 'Annotations'
# pic dir
IMAGE_DIR = 'JPEGImages'
# self pic dir
SELF_IMAGE_DIR = 'SelfImages'
# Input Size
INPUT_SHAPE = 416, 416

from test import *
