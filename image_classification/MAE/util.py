import random
import typing as t

import torch
from colorama import Fore
from torch import nn
from torchvision.transforms import transforms


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
           f'\tcur_loss: {round(loss_cls, 6)} \tavg_loss: {round(avg_loss, 6)}\n'

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


def reconstruction(
        pred: torch.Tensor,
        target: torch.Tensor,
        mean: t.List,
        std: t.List,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    b, _, _ = pred.size()
    # (b, mask_num, decoder_dim) -> (b, c, h, w), create a new tensor object
    recons_img = pred.view(
        b, 14, 14,
        16, 16, 3
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, 224, 224)

    patches_img = target.view(
        b, 14, 14,
        16, 16, 3
    ).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, 224, 224)
    recons_img = (recons_img * 255.).type(torch.uint8)
    patches_img = (patches_img * 255.).type(torch.uint8)
    # recons_img = z_score_denormalizer(recons_img, mean, std)
    # patches_img = z_score_denormalizer(recons_img, mean, std)
    return recons_img, patches_img


def z_score_denormalizer(
        img_t: torch.Tensor,
        mean: t.List,
        std: t.List
) -> torch.Tensor:
    """
    对z_score进行反归一化, 然后乘 255.
    """
    d_mean = [-m / d for m, d in zip(mean, std)]
    d_std = [1 / d for d in std]
    return (transforms.Normalize(d_mean, d_std)(img_t) * 255).type(torch.uint8)


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.zeros_(module.bias)
