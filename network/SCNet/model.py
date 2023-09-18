import typing as t

import torch
import torch.nn as nn
from torchsummary import summary

from network.ResNeXt import ResNeXt, BottleNeck
from other_utils.conv import SCConv
from torchvision.models import resnet50


class Conv(nn.Module):
    """
    replace standard 3x3 convolution by SCConv
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            **kwargs
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, groups=groups,
                                  bias=False)
        else:
            self.conv = SCConv(in_chans, stride=stride, **kwargs)
        self.bn = norm_layer(out_chans)
        self.act = act_layer()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


def resnet_50_sc(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def resnet_101_sc(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def resnet_152_sc(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def ResNext_50_SC_32x4d(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 2
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[128, 256, 512, 1024],
        num_classes=num_classes,
        groups=32,
        width_per_group=4,
        classifier=classifier,
        base_width=128,
    )


def ResNext_101_SC_32x4d(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 2
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[128, 256, 512, 1024],
        num_classes=num_classes,
        groups=32,
        width_per_group=4,
        classifier=classifier,
        base_width=128,
    )


def ResNext_152_SC_32x4d(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 2
    BottleNeck.set_conv(Conv)
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[128, 256, 512, 1024],
        num_classes=num_classes,
        groups=32,
        width_per_group=4,
        classifier=classifier,
        base_width=128,
    )


if __name__ == '__main__':
    _x = torch.randn((3, 3, 224, 224)).to(0)
    resnext50 = resnet_50_sc(num_classes=5, classifier=True).to(0)
    # res = resnext50(_x)
    # has fewer params, 14.5M
    summary(resnext50, (3, 224, 224), batch_size=4)
    # print(res.size())
