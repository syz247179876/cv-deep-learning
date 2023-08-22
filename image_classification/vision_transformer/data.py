import os.path
import typing as t

from PIL import Image
from torch.utils.data import Dataset
from setting import *
from augment import args_train
from utils import shuffle


class FlowerTransform(object):
    """
    preprocess flower dataset to decode images and corresponding label
    """

    def __call__(self, img_dir, *args, **kwargs) -> t.List[t.Tuple]:
        files_path: t.List[t.Tuple] = []
        for root, dirs, files in os.walk(img_dir):
            if files:
                for file in files:
                    files_path.append((os.path.join(root, file), root.split('\\')[-1]))
        shuffle(files_path, len(files_path))
        return files_path


class ViTDataset(Dataset):
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
        self.img_dir = os.path.join(self.opts.base_dir, IMAGE_DIR)
        self.use_img = []
        self.train_test_ratio = train_test_ratio
        self.train_validate_ratio = train_validate_ratio
        self.anno_transform = anno_transform()
        self.img_augmentation = img_augmentation()
        self.normalization = normalization()

        self.img_list = self.anno_transform(self.img_dir, opts=self.opts)

        func = getattr(self, f'{mode}_data')
        func()

    def __len__(self):
        return len(self.use_img)

    def __getitem__(self, item):
        pass

    def train_data(self):
        self.use_img = self.img_list[: int(len(self.img_list) * self.train_test_ratio)]

    def test_data(self):
        self.use_img = self.img_list[int(len(self.img_list) * self.train_test_ratio): ]

    def validate_date(self):
        pass

    def pull_item(self, index: int):
        img_path, img_label = self.img_list[index]



if __name__ == '__main__':
    dataset = ViTDataset(mode='train', anno_transform=FlowerTransform)
    FlowerTransform()(dataset.img_dir)
