from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from debug import DEBUG_OPEN
import os
import numpy as np
import torch
import typing as t


class YOLOV1Dataset(Dataset):

    def __init__(
            self,
            dataset_dir: str,
            mode: str = 'train',
            train_validate_ratio: float = 0.9,
            trans: t.Optional[t.Callable] = None,
    ):
        """
        mode可选: train, validate, test
        """
        func = getattr(self, f'{mode}_data')
        self.mode = mode
        self.use_img_path = []
        self.train_validate_ratio = train_validate_ratio
        self.trans = trans
        if mode == 'validate':
            mode = 'train'
        img_path = os.path.join(dataset_dir, f'{mode}.txt')
        label_path = os.path.join(dataset_dir, f'{mode}.csv')
        self.labels = np.loadtxt(label_path, dtype=np.float32)
        with open(img_path, 'r') as f:
            self.img_path_list = f.read().split('\n')
        self.len_all_data = len(self.img_path_list)
        func()

    def train_data(self):
        self.use_img_path = self.img_path_list[:int(self.len_all_data * self.train_validate_ratio)]

    def validate_data(self):
        self.use_img_path = self.img_path_list[int(self.len_all_data * self.train_validate_ratio):]

    def test_data(self):
        self.use_img_path = self.img_path_list

    def __getitem__(self, item):
        """
        传入指定的索引后，根据索引返回对应的单个样本及其对应的标签
        将图像数据转为torch.tensor数据结构，用于GPU加速训练
        """
        img_path = self.use_img_path[item]
        label = torch.tensor(self.labels[item, :])
        # 用于transform操作
        img = Image.open(img_path)
        trans = self.trans or transforms.Compose([
            transforms.ToTensor()
        ])
        img = trans(img)
        return img, label

    def __len__(self):
        return len(self.use_img_path)