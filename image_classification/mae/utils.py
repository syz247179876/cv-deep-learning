import random
import typing as t

import torch
from PIL.JpegImagePlugin import JpegImageFile
from colorama import Fore


def shuffle(arr: t.List[t.Union[t.Tuple, str]], arr_len: int) -> None:
    """
    基于Fisher-Yates的洗牌算法打乱arr
    时间复杂度为O(N)
    """
    for i in range(arr_len - 1):
        idx = random.randint(i, arr_len - 1)
        arr[i], arr[idx] = arr[idx], arr[i]


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
           f'\tcur_loss: {round(loss_cls, 4)} \tavg_loss: {round(avg_loss, 6)}\n'

    print_log(info, color=Fore.RED)
    if write:
        log_f.write(info)
        log_f.flush()


def mae_collate(batch: t.Iterable[t.Tuple]) -> t.Tuple[torch.Tensor, t.List]:
    images = []
    img_paths = []
    for img, path in batch:
        if img.size(0) != 3:
            continue
        images.append(img)
        img_paths.append(path)
    return torch.stack(images, dim=0), img_paths
