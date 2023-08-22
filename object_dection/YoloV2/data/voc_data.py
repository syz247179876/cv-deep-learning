import os
import numpy as np
import torch
import typing as t
import xml.etree.ElementTree as ET
import cv2
from torch.utils.data import Dataset
from ..argument import args_train
from ..settings import *
from ..util import Augmentation


class VOCDecodeTransform(object):
    """
    直接解码VOC数据集, 提取其中信息, 转换成bbox坐标和标签索引的张量
    """

    def __init__(self, class_idx: t.Optional[t.Dict] = None, keep_difficult: bool = False):
        self.class_idx = class_idx or dict(zip(VOC_CLASSES, range(VOC_CLASSES_LEN)))
        self.keep_difficult = keep_difficult
        self.coord = ['xmin', 'ymin', 'xmax', 'ymax']

    def __call__(self, root: ET.Element, width: int, height: int) -> t.List[t.List]:
        res = []

        for node in root.iter('object'):
            node: ET.Element
            difficult = int(node.find('difficult').text) == 1
            cls = node.find('name').text.lower().strip()
            bbox_node: ET.Element = node.find('bndbox')
            if not self.keep_difficult and difficult:
                continue

            bbox = []
            for idx, info in enumerate(self.coord):
                cur_info = int(float(bbox_node.find(info).text))
                cur_info = cur_info / height if idx % 2 else cur_info / width
                bbox.append(cur_info)

            cls_id = self.class_idx[cls]
            bbox.append(cls_id)
            res.append(bbox)
        # [[xmin, ymin, xmax, ymax, cls_id], [...],]
        return res


class VOCDataset(Dataset):

    def __init__(
            self,
            mode: str = 'train',
            train_validate_ratio: float = 0.9,
            train_test_ratio: float = 0.8,
            anno_transform: t.Optional[t.Callable] = None,
            img_augmentation: t.Optional[t.Callable] = None,
            train_size: int = 416,
    ):
        self.opts = args_train.opts
        self.mode = mode
        self.anno_transform = anno_transform or VOCDecodeTransform()
        self.img_augmentation = img_augmentation or Augmentation(train_size)
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

    def __getitem__(self, item):
        img, label, _, _ = self.pull_item(item)
        return img, label

    def train_data(self):
        self.use_img_path = self.img_list[: int(len(self.img_list) * self.train_test_ratio)]

    def validate_data(self):
        pass

    def test_data(self):
        self.use_img_path = self.img_list[int(len(self.img_list) * self.train_test_ratio):]

    def pull_item(self, index: int):
        img_id = self.img_list[index]
        img_path: str = os.path.join(self.img_dir, img_id)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        root = ET.parse(os.path.join(self.annotation_dir, f'{img_id.split(".")[0]}.xml')).getroot()
        target = self.anno_transform(root, width, height)
        target = np.array(target)
        img = self.img_augmentation(img)
        # the image channel read out using openCV is in BGR and needs to be converted back to RGB

        img = img[:, :, (2, 1, 0)]
        img = torch.tensor(img)
        img = img.permute(2, 0, 1).float()  # (height, width, channels) -> (channels, height, width)

        return img, target, height, width


if __name__ == "__main__":
    dataset = VOCDataset(train_size=416)
    # for i in range(2, 100):
    #     img_, gt, h, w = dataset.pull_item(i)
    #     img_ = img_.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
    #     img_ = img_.copy()
    #     cls_id = gt[0][4]
    #     for box in gt:
    #         xmin, ymin, xmax, ymax, _ = box
    #         xmin *= train_size
    #         ymin *= train_size
    #         xmax *= train_size
    #         ymax *= train_size
    #         print((int(xmin), int(ymin)), (int(xmax), int(ymax)))
    #         cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)
    #     cv2.imshow('gt', img_)
    #     cv2.waitKey(0)
