import os
import typing as t

import numpy as np
import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from augment import args_train
from setting import *
from utils import shuffle


class FlowerTransform(object):
    """
    Flower Dataset
    """

    @classmethod
    def extract(cls, img_dir: str, use_shuffle: bool = False) -> t.List:
        files_path: t.List = []
        for root, dirs, files in os.walk(img_dir):
            if files:
                for file in files:
                    files_path.append(os.path.join(root, file))
        if use_shuffle:
            shuffle(files_path, len(files_path))
        return files_path


class VOCTransform(object):
    """
    Voc Dataset
    """

    @classmethod
    def extract(cls, img_dir: str, use_shuffle: bool = False) -> t.List:
        files_path = os.listdir(img_dir)
        if use_shuffle:
            shuffle(files_path, len(files_path))
        return files_path


class COCOTransform(object):
    """
    COCO Dataset
    """

    @classmethod
    def extract(cls, img_dir: str, use_shuffle: bool = False) -> t.List:
        files_path = os.listdir(img_dir)
        if use_shuffle:
            shuffle(files_path, len(files_path))
        return files_path


class ImageAugmentation(object):

    def __call__(
            self,
            img: JpegImageFile,
            mean: t.List,
            std: t.List,
            input_shape: t.Tuple = (224, 224),
    ) -> torch.Tensor:
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            # using z-score normalization can improve the latent representation learned by the model during training
            transforms.Normalize(mean, std),
        ])
        return transform(img)


class MAEDataset(Dataset):

    def __init__(
            self,
            mode: str = 'train',
            input_shape: t.Tuple[int, int] = (224, 224),
            train_validate_ratio: float = 0.9,
            train_test_ratio: float = 0.8,
            img_augmentation: t.Optional[t.Callable] = None,
    ):
        self.opts = args_train.opts
        self.mode = mode
        dataset_name = self.opts.dataset_name.upper()
        self.img_dir = globals().get(f'{dataset_name}_IMAGE_DIR')
        self.mean = globals().get(f'{dataset_name}_MEAN')
        self.std = globals().get(f'{dataset_name}_STD')
        self.anno_transform = globals().get(f'{dataset_name}Transform', None)
        assert self.anno_transform is not None and getattr(self.anno_transform, 'extract'), \
            f'Class {dataset_name}_Transform should be define and extract function should be implement!'
        self.use_img = []
        self.train_test_ratio = train_test_ratio
        self.train_validate_ratio = train_validate_ratio
        self.img_augmentation = img_augmentation and img_augmentation()
        self.img_list = self.anno_transform.extract(self.img_dir)
        self.input_shape = input_shape
        func = getattr(self, f'{mode}_data')
        func()

    def __len__(self):
        return len(self.use_img)

    def __getitem__(self, item) -> t.Tuple[torch.Tensor, str]:
        image_path = self.use_img[item]
        img = Image.open(os.path.join(self.img_dir, image_path))
        if self.img_augmentation:
            img = self.img_augmentation(img, self.mean, self.std, self.input_shape)
        else:
            img = torch.tensor(np.array(img)).float()
        return img, image_path

    def train_data(self):
        self.use_img = self.img_list[: int(len(self.img_list) * self.train_test_ratio)]

    def test_data(self):
        self.use_img = self.img_list[int(len(self.img_list) * self.train_test_ratio):]

    def validate_data(self):
        pass


# if __name__ == '__main__':
#     xxx = globals().get(f'COCOTransform', None)
#     if getattr(xxx, 'extract'):
#         print(222)
