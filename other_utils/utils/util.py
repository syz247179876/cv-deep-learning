import typing as t
from functools import partial

import torch
import torch.nn as nn


def auto_pad(k: t.Union[int, t.List], p: t.Optional[t.Union[int, t.List]] = None,
             d: int = 1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


"""
optimize memory overhead and time overhead, so, merge identity, 1x1, 3x3 into 3x3 conv
"""


def identity_trans_1x1(
        in_chans: int,
        out_chans: int,
        groups: int = 1,
        bias: bool = False
) -> nn.Module:
    """
    Convert identity to 1x1 convolution
    """

    conv = nn.Conv2d(in_chans, out_chans, kernel_size=1, groups=groups, bias=bias)
    input_dim = in_chans // groups
    conv_weight = torch.zeros((out_chans, input_dim, 1, 1), dtype=torch.float32)
    conv_bias = torch.zeros(out_chans, dtype=torch.float32)

    for idx in range(out_chans):
        conv_weight[idx, idx % input_dim, 0, 0] = 1.
    print(conv_bias)
    conv.weight.data = conv_weight
    if bias:
        conv.bias.data = conv_bias
    return conv


def identity_trans_3x3(
        in_chans: int,
        out_chans: int,
        groups: int = 1,
        bias: bool = False,
):
    """
    Convert identity to 3x3 convolution
    """
    assert in_chans == out_chans, 'input channel should equal to output channel'
    conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, groups=groups, bias=bias)
    input_dim = in_chans // groups
    conv_weight = torch.zeros((out_chans, input_dim, 3, 3))
    conv_bias = torch.zeros(out_chans)

    for idx in range(out_chans):
        conv_weight[idx, idx % input_dim, 1, 1] = 1.
    conv.weight.data = conv_weight.data
    if bias:
        conv.bias.data = conv_bias.data
    return conv


def conv1x1_trans_conv3x3(kernel1x1: torch.Tensor):
    """
    Convert 1x1 convolution to 3x3 convolution
    """
    return nn.functional.pad(kernel1x1, [1, 1, 1, 1])


REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hardswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}

REGISTERED_NORM_DICT: dict[str, type] = {
    'bn': nn.BatchNorm2d,
    'gn': partial(nn.GroupNorm, 32),
    'ln': nn.LayerNorm,
    'in': nn.InstanceNorm2d
}


def build_act(name: str) -> t.Union[nn.Module, t.Callable]:
    """
    build activation function based on param name
    """
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        return act_cls
    else:
        raise TypeError(f'no found act {name}')


def build_norm(name: str) -> t.Union[nn.Module, t.Callable]:
    """
    build normalization function based on param name
    """
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        return norm_cls
    else:
        raise TypeError(f'no found norm {name}')


if __name__ == '__main__':
    conv1x1 = identity_trans_1x1(3, 3)
    conv3x3 = identity_trans_3x3(3, 3)
    _x = torch.rand((2, 3, 224, 224))
    _y = conv3x3(_x)
    print(_y.size(), _x.size(), _y == _x)
