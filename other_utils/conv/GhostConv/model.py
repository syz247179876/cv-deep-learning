import math

import torch
import torch.nn as nn
import typing as t


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
    Basic Conv includes Conv2d(1x1 or 3x3), BN, Relu
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
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, dilation=dilation,
                              padding=auto_pad(kernel_size, d=dilation),
                              groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class GhostConv(nn.Module):
    """
    Ghost Module
    GhostNet proposes that redundant features can be generated through cheap transformation based on original intrinsic
    feature maps.

    the output consists of two parts.


        1.The number of convolution kernels in the first part is m, which is calculated by conventional convolution,
        given the intrinsic feature maps from the first part.

        2.The number of convolution kernels in the second part is s, calculated through a simple linear mapping,
        in the second part of the implementation, using DWC, which is used to reduce the amount of
        parameters and calculation.

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
    ):
        """
        kernel_size: Convolutional kernel size for conventional convolution
        ratio: The ratio parameter is used to partition m and s, in the paper, the default ratio parameter is 2
        dw_kernel_size: Convolutional kernel size for cheap transformation, such as DWC
        """
        super(GhostConv, self).__init__()
        self.out_chans = out_chans
        m_channels = math.ceil(out_chans / ratio)
        s_channels = m_channels * (ratio - 1)

        self.primary_conv = Conv(in_chans, m_channels, kernel_size, stride, dilation=dilation)
        self.cheap_conv = Conv(m_channels, s_channels, dw_kernel_size, 1, groups=m_channels, dilation=dilation)

    def forward(self, x: torch.Tensor):
        x1 = self.primary_conv(x)
        x2 = self.primary_conv(x1)
        y = torch.cat((x1, x2), dim=1)
        return y[:, :self.out_chans, :, :]
