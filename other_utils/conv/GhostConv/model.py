import math

import torch
import torch.nn as nn
import typing as t
from other_utils.utils import auto_pad


class Conv(nn.Module):
    """
    Basic Conv includes Conv2d(1x1 or 3x3), BN, Relu(selectable)
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            need_act: bool = True,
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if need_act and act_layer is None:
            act_layer = nn.ReLU

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, dilation=dilation,
                              padding=auto_pad(kernel_size, d=dilation),
                              groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        if need_act:
            self.act = act_layer(inplace=True)
        self.need_act = need_act

    def forward(self, x: torch.Tensor):
        x = self.bn(self.conv(x))
        if self.need_act:
            x = self.act(x)
        return x


class GhostConv(nn.Module):
    """
    Ghost Module
    GhostNet proposes that redundant features can be generated through cheap transformation based on original intrinsic
    feature maps.

    the output consists of two parts.


        1.The number of convolution kernels in the first part is m, which is calculated by ordinary convolution filters,
        given a handful of intrinsic feature maps from the first part.

        2.The number of convolution kernels in the second part is s, calculated through a cheap liner operations based
        on intrinsic feature. In the second part of the implementation, it used DWC, which is used to reduce the amount of
        parameters and calculation, and generate some ghost(redundant) feature maps.

    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 1,
            ratio: int = 2,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            need_act: bool = True,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        """
        kernel_size: Convolutional kernel size for conventional convolution.

        ratio: The ratio parameter is used to partition m and s, in the paper, the default ratio parameter is 2,
        it also means the compression ratio, As the ratio increases, the theoretical training and inference speed
        will also increase by a factor of ratio. In general, the compression ratio is equal to the speed-up ratio

        dw_kernel_size: Convolutional kernel size for cheap transformation, such as DWC.
        """
        super(GhostConv, self).__init__()
        self.out_chans = out_chans
        self.m_channels = math.ceil(out_chans / ratio)
        self.s_channels = self.m_channels * (ratio - 1)

        self.primary_conv = Conv(in_chans, self.m_channels, kernel_size, stride, dilation=dilation,
                                 need_act=need_act, norm_layer=norm_layer, act_layer=act_layer)
        self.cheap_conv = Conv(self.m_channels, self.s_channels, dw_kernel_size, 1, groups=self.m_channels,
                               dilation=dilation, need_act=need_act, norm_layer=norm_layer, act_layer=act_layer)

    def forward(self, x: torch.Tensor):
        x1 = self.primary_conv(x)
        if self.s_channels == 0:
            return x1
        x2 = self.cheap_conv(x1)
        y = torch.cat((x1, x2), dim=1)
        return y[:, :self.out_chans, :, :]
