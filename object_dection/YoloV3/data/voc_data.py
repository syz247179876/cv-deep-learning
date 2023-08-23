import os
import sys
import typing as t
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
if os.path.dirname(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.getcwd()))
from argument import args_train
from settings import *
from utils import ImageAugmentation, Normalization


class VOCDecodeTransform(object):
    """
    Decode the VOC dataset, extract information from it,
    and convert it into ndarray for bbox coordinates and class indexes
    """

    def __init__(self, class_idx: t.Optional[t.Dict] = None, keep_difficult: bool = False):
        self.class_idx = class_idx or dict(zip(VOC_CLASSES, range(VOC_CLASSES_LEN)))
        self.keep_difficult = keep_difficult
        self.coord = ['xmin', 'ymin', 'xmax', 'ymax']

    def __call__(self, root: ET.Element, width: int, height: int) -> np.ndarray:
        """
        Output:
            np.ndarray -> [[x_min, y_min, x_max, y_max, cls_id], [...],]
        """
        res = []

        for node in root.iter('object'):
            node: ET.Element
            difficult = int(node.find('difficult').text) == 1
            cls = node.find('name').text.lower().strip()
            bbox_node: ET.Element = node.find('bndbox')
            if not self.keep_difficult and difficult:
                continue

            x_min = int(float(bbox_node.find('xmin').text))
            y_min = int(float(bbox_node.find('ymin').text))
            x_max = int(float(bbox_node.find('xmax').text))
            y_max = int(float(bbox_node.find('ymax').text))
            bbox = [x_min, y_min, x_max, y_max, self.class_idx[cls]]
            res.append(bbox)
        return np.array(res)


class VOCDataset(Dataset):
    def __init__(
            self,
            mode: str = 'train',
            train_validate_ratio: float = 0.9,
            train_test_ratio: float = 0.8,
            anno_transform: t.Optional[t.Callable] = None,
            img_augmentation: t.Optional[t.Callable] = None,
            normalization: t.Optional[t.Callable] = None,
    ):
        self.opts = args_train.opts
        self.mode = mode
        self.anno_transform = anno_transform or VOCDecodeTransform()
        self.img_augmentation = img_augmentation or ImageAugmentation()
        self.normalization = normalization or Normalization()
        self.annotation_dir = os.path.join(self.opts.base_dir, ANNOTATIONS_DIR)
        self.img_dir = os.path.join(self.opts.base_dir, IMAGE_DIR)
        self.img_list: t.List = os.listdir(self.img_dir)
        self.use_img_path = []
        self.train_test_ratio = train_test_ratio
        self.train_validate_ratio = train_validate_ratio

        func = getattr(self, f'{mode}_data')
        func()

    def __len__(self):
        return len(self.use_img_path)

    def __getitem__(self, item: int) -> t.Tuple[torch.Tensor, torch.Tensor, str, t.Tuple[int, int]]:
        img, label, img_path, image_shape = self.pull_item(item)

        return img, label, img_path, image_shape

    def train_data(self):
        self.use_img_path = self.img_list[: int(len(self.img_list) * self.train_test_ratio)]

    def validate_data(self):
        pass

    def test_data(self):
        self.use_img_path = self.img_list[int(len(self.img_list) * self.train_test_ratio):]

    def self_data(self):
        self.img_dir = os.path.join(self.opts.base_dir, SELF_IMAGE_DIR)
        self.use_img_path = self.img_list = os.listdir(self.img_dir)

    def pull_item(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor, str, t.Tuple[int, int]]:
        img_id = self.use_img_path[index]
        img_path: str = os.path.join(self.img_dir, img_id)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        root = ET.parse(os.path.join(self.annotation_dir, f'{img_id.split(".")[0]}.xml')).getroot()
        target = self.anno_transform(root, width, height)
        img, target = self.img_augmentation(img_path, target)
        img, target = self.normalization(img, target)
        # the image channel read out using openCV is in BGR and needs to be converted back to RGB
        # img = img[:, :, (2, 1, 0)]
        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()  # (height, width, channels) -> (channels, height, width)
        target = torch.tensor(target)
        image_shape = (width, height)
        return img, target, img_path, image_shape
