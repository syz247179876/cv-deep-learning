import torch
import torch.nn as nn
import typing as t


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    A residual block structure generated with a certain probability p based on the Bernoulli distribution.
    when p = 1, it is the original residual block; when p = 0, it drops the whole current network.

    the probability p is not const, while it related to the depth of the residual block.

    *note: stochastic depth actually is model fusion, usually used in residual block to skip some residual structures.
    to some extent, it fuses networks of different depths.
    """

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Stochastic depth level used in residual block.
    as the network become more deepens, training becomes complex,
    over-fitting and gradient disappearance occur, so, it uses stochastic depth
    to random drop partial network according to some samples in training, while keeps in testing.
    """

    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def auto_pad(k: t.Union[int, t.List], p: t.Optional[t.Union[int, t.List]] = None,
             d: int = 1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Basic Conv layer with conv, bn, activation
    the params group means group conv
    """
    default_act = nn.SiLU()

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: t.Union[t.List, int],
            stride: int,
            padding: t.Optional[t.Union[t.List, int]] = None,
            groups: int = 1,
            dilation: int = 1,
            act: t.Union[nn.Module, bool] = True
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, auto_pad(kernel_size, padding, dilation),
                              dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    """
    Standard BottleNeck layer, actually is residual block, add extra Regularization --- stochastic depth
    to drop some samples
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            drop_path_ratio: float = 0.0,
            groups: int = 1,
            expansion: float = 0.5,
            residual: bool = True
    ):
        super(BottleNeck, self).__init__()
        hidden_chans = int(in_chans * expansion)
        self.conv1 = Conv(in_chans, hidden_chans, 1, 1)
        self.conv3 = Conv(hidden_chans, out_chans, 3, 1, groups=groups)
        self.dropout = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.residual = residual and in_chans == out_chans

    def forward(self, x: torch.Tensor):
        return x + self.dropout(self.conv3(self.conv1(x))) if self.residual else self.dropout(self.conv3(self.conv1(x)))


class BottleNeckCSP(nn.Module):
    """
    CSPNet: https://arxiv.org/abs/1911.11929
    1.Enhance CNN's learning ability to maintain accuracy while reducing weight.
    2.Reduce computational bottlenecks and duplicate gradient information in DenseNet。
    3.Reduce memory costs.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            number: int = 1,
            drop_path_ratio: float = 0.0,
            groups: int = 1,
            expansion: int = 0.5,
    ):
        super(BottleNeckCSP, self).__init__()
        hidden_chans = int(in_chans * expansion)
        self.conv1 = Conv(in_chans, hidden_chans, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_chans, hidden_chans, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_chans, hidden_chans, kernel_size=1, stride=1, bias=False)
        self.conv4 = Conv(2 * hidden_chans, out_chans, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2 * hidden_chans)
        self.act = nn.SiLU()
        self.bottle_neck = nn.Sequential(*(BottleNeck(hidden_chans, hidden_chans, drop_path_ratio, groups, 1.0)
                                           for _ in range(number)))

    def forward(self, x: torch.Tensor):
        y1 = self.conv3(self.bottle_neck(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            number: int = 1,
            drop_path_ratio: float = 0.0,
            groups: int = 1,
            expansion: int = 0.5,
    ):
        super(C3, self).__init__()
        hidden_chans = int(in_chans * expansion)
        self.conv1 = Conv(in_chans, hidden_chans, kernel_size=1, stride=1)
        self.conv2 = Conv(in_chans, hidden_chans, kernel_size=1, stride=1)
        self.conv3 = Conv(2 * hidden_chans, out_chans, kernel_size=1, stride=1)
        self.bottle_neck = nn.Sequential(*(BottleNeck(hidden_chans, hidden_chans, drop_path_ratio, groups, 1.0)
                                           for _ in range(number)))

    def forward(self, x: torch.Tensor):
        return self.conv3(torch.cat((self.bottle_neck(self.conv1(x)), self.conv2(x)), dim=1))
