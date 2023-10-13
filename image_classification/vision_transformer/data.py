import os.path
import random
import typing as t

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
from setting import *
from augment import args_train
from utils import shuffle, resize_img_box, normalize_factory


class FlowerTransform(object):
    """
    preprocess flower dataset to extract images and corresponding label
    """

    def __call__(self, img_dir: str, *args, **kwargs) -> t.Tuple[t.List[t.Tuple[str, int]], t.Dict]:
        files_path: t.List[t.Tuple] = []
        flower_class = {val: idx for idx, val in enumerate(sorted(os.listdir(img_dir)))}
        flower_class_idx = {idx: val for idx, val in enumerate(sorted(os.listdir(img_dir)))}
        for root, dirs, files in os.walk(img_dir):
            if files:
                for file in files:
                    files_path.append((os.path.join(root, file), flower_class[root.split('\\')[-1]]))
        shuffle(files_path, len(files_path))
        return files_path, flower_class_idx


class ImageAugmentation(object):

    def __call__(
            self,
            img: JpegImageFile,
            input_shape: t.Tuple = (224, 224),
            distort: bool = False,
            random_crop: bool = True,
    ) -> JpegImageFile:
        img.convert('RGB')
        # resize image and add grey on image
        image_data = resize_img_box(img, input_shape, distort, random_crop)
        return image_data


def get_mean_std(img_paths: t.List, input_shape: t.Tuple = (224, 224)):
    augmentation = ImageAugmentation()
    images = np.zeros((input_shape[0], input_shape[1], 3, 1))
    mean = []
    std = []
    for i in range(0, len(img_paths), 5):
        img_path = img_paths[i][0]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img.convert('RGB')
        img = augmentation(img, input_shape, distort=True)
        img = np.array(img)
        img = img[:, :, :, np.newaxis]
        images = np.concatenate((images, img), axis=3)
    images = images.astype(np.float32) / 255.
    for i in range(3):
        # flatten to one line
        pixels = images[:, :, i, :].ravel()
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))
    print(mean, std)


@normalize_factory(mode='z_score')
def z_score_(img: np.ndarray, man: t.List, std: t.List):
    """
    do nothing here, without some params needed, as it use z_score function implemented in normalize factory
    """
    pass


class ViTDataset(Dataset):
    def __init__(
            self,
            mode: str = 'train',
            train_validate_ratio: float = 0.9,
            train_test_ratio: float = 0.8,
            anno_transform: t.Optional[t.Callable] = None,
            img_augmentation: t.Optional[t.Callable] = None,
            normalization: t.Optional[t.Callable] = None,
            distort: float = False,
    ):
        self.opts = args_train().opts
        self.mode = mode
        self.img_dir = os.path.join(self.opts.base_dir, IMAGE_DIR)
        self.use_img = []
        self.train_test_ratio = train_test_ratio
        self.train_validate_ratio = train_validate_ratio
        self.anno_transform = anno_transform and anno_transform()
        self.img_augmentation = img_augmentation and img_augmentation()
        # self.normalization = z_score_

        self.img_list, self.classes = self.anno_transform(self.img_dir, opts=self.opts)
        self.distort = distort
        func = getattr(self, f'{mode}_data')
        func()

    def __len__(self):
        return len(self.use_img)

    def __getitem__(self, item) -> t.Union[t.Tuple[torch.Tensor, int], t.Tuple[torch.Tensor, int, str]]:
        img, label, img_path = self.pull_item(item)
        if self.mode == 'train':
            return img, label
        elif self.mode == 'test':
            return img, label, img_path

    def train_data(self):
        self.use_img = self.img_list[: int(len(self.img_list) * self.train_test_ratio)]

    def test_data(self):
        self.use_img = self.img_list[int(len(self.img_list) * self.train_test_ratio):]

    def validate_date(self):
        pass

    def pull_item(self, index: int) -> t.Tuple[torch.Tensor, int, str]:
        img_path, img_label = self.use_img[index]
        img = Image.open(img_path)
        if self.img_augmentation:
            img = self.img_augmentation(img, distort=self.distort)
        # if self.normalization:
        #     img3 = self.normalization(img, FLOWER_MEAN, FLOWER_STD)
        img = z_score_(img, FLOWER_MEAN, FLOWER_STD)
        return img, img_label, img_path


if __name__ == '__main__':
    dataset = ViTDataset(mode='train', anno_transform=FlowerTransform,
                         img_augmentation=ImageAugmentation, normalization=z_score_, distort=True)
    # get_mean_std(dataset.use_img)
    a = dataset[6]
