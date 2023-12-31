import functools
import typing as t
import random

import cv2
import numpy as np
import torch
from PIL.JpegImagePlugin import JpegImageFile
from colorama import Fore
from torchvision import transforms

from setting import *
from PIL import Image


def shuffle(arr: t.List[t.Union[t.Tuple, str]], arr_len: int) -> None:
    """
    基于Fisher-Yates的洗牌算法打乱arr
    时间复杂度为O(N)
    """
    for i in range(arr_len - 1):
        idx = random.randint(i, arr_len - 1)
        arr[i], arr[idx] = arr[idx], arr[i]


def resize_img_box(
        img: JpegImageFile,
        new_size: t.Tuple[int, int],
        distort: bool,
        random_crop: bool,
) -> JpegImageFile:
    """
    resize img to new_size.
    use no deformed conversion or deformed conversion
    """
    o_w, o_h = img.size
    n_w, n_h = new_size

    if not distort:
        scale = min(n_w / o_w, n_h / o_h)
        n_w_ = int(o_w * scale)
        n_h_ = int(o_h * scale)
        dw = (n_w - n_w_) // 2
        dh = (n_h - n_h_) // 2
        img = img.resize((n_w_, n_h_), Image.BICUBIC)
        new_image = Image.new('RGB', new_size, (128, 128, 128))
        new_image.paste(img, (dw, dh))
    elif random_crop:
        new_image = transforms.RandomResizedCrop((n_w, n_h))(img)
    else:
        new_image = transforms.Resize((n_w, n_h))(img)
    return new_image


def normalize_factory(mode: str):
    def wrapper(func: t.Optional[t.Callable] = None):
        def simple(img: np.ndarray) -> torch.Tensor:
            return transforms.ToTensor()(img)

        def z_score(img: np.ndarray, mean: t.List, std: t.List) -> torch.Tensor:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            return transform(img)

        @functools.wraps(func)
        def min_max(img: np.ndarray):
            pass

        if mode in ('simple', 'min_max', 'z_score'):
            _func = locals().get(mode, None)
            if func is not None and _func.__name__ == func.__name__:
                return func
            return _func
        return func

    return wrapper


def classify_collate(batch: t.Iterable[t.Tuple]) -> t.Tuple[torch.Tensor, t.List]:
    labels = []
    images = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return torch.stack(images, dim=0), torch.tensor(labels)


def classify_collate_test(batch: t.Iterable[t.Tuple]) -> t.Tuple[torch.Tensor, torch.Tensor, t.List]:
    labels = []
    images = []
    img_paths = []
    for img, label, img_path in batch:
        images.append(img)
        labels.append(label)
        img_paths.append(img_path)
    return torch.stack(images, dim=0), torch.tensor(labels), img_paths


def print_log(txt: str, color: t.Any = Fore.GREEN):
    print(color, txt)


def print_detail(
        cur_epoch: int,
        end_epoch: int,
        batch: int,
        iter_total: int,
        loss_cls: float,
        avg_loss: float,
        log_f: t.TextIO,
        write: bool = False,
):
    info = f'Epoch: {cur_epoch}/{end_epoch} \tIter: {batch}/{iter_total}' \
           f'\tloss_cls: {round(loss_cls, 4)} \tavg_loss: {round(avg_loss, 6)}\n'

    print_log(info, color=Fore.RED)
    if write:
        log_f.write(info)
        log_f.flush()


def draw_image(
        img_path: str,
        loss: float,
        pred: torch.Tensor,
        classes: t.Dict[int, str],
        label_idx: int,
        single_display: bool = True,
) -> bool:
    """
    draw image in testing
    """
    img = cv2.imread(img_path)
    pred_classes_idx = torch.max(pred, dim=1)[1].item()
    print(f'prediction class: {classes[pred_classes_idx]} \tactual class: {classes[label_idx]} \tloss: {round(loss, 6)}')
    if single_display:
        cv2.imshow(f'class: {classes[pred_classes_idx]} loss: {round(loss, 6)}', img[:, :, ::-1])
        cv2.waitKey(0)
    return pred_classes_idx == label_idx
