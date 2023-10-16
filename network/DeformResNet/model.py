import torch.nn as nn
import torch
import typing as t

from network.ResNeXt import BottleNeck, ResNeXt
from other_utils.conv import DeformConv
from other_utils.utils import auto_pad


class Conv(nn.Module):
    """
    Basic Conv includes DeformConv/conv2d, BN, Relu, routing weights in one Residual Block share the same
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
            ordinary_conv: bool = False
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        if ordinary_conv:
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride,
                                  padding=auto_pad(kernel_size), groups=groups, bias=False)
        else:
            self.conv = DeformConv(in_chans, out_chans, kernel_size, stride=stride,
                                   padding=auto_pad(kernel_size), groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class DeformBottleNeck(nn.Module):
    """
    DeformBottleNeck
    """
    expansion = 4

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int = 1,
            groups: int = 1,
            width_per_group: int = 64,
            down_sample: t.Optional[nn.Module] = None,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            base_width: int = 64,
            **kwargs
    ):
        super(DeformBottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = Conv(in_chans, width, 1, 1, 1, norm_layer, act_layer, ordinary_conv=True)
        # when stride > 1, need to down sample x in forward, use conv replace pooling
        self.conv2 = Conv(width, width, 3, stride, groups, norm_layer, act_layer)
        self.conv3 = nn.Conv2d(width, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    @classmethod
    def set_conv(cls, conv: t.Type[nn.Module]):
        cls._conv = conv

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)

        y = self.conv3(y)
        y = self.bn(y)
        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


def resnet_50_deform(num_classes: int = 1000, classifier: bool = True):
    DeformBottleNeck.expansion = 4
    return ResNeXt(
        block=[DeformBottleNeck, DeformBottleNeck, DeformBottleNeck, DeformBottleNeck],
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def resnet_101_deform(num_classes: int = 1000, classifier: bool = True):
    DeformBottleNeck.expansion = 4
    return ResNeXt(
        block=[DeformBottleNeck, DeformBottleNeck, DeformBottleNeck, DeformBottleNeck],
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


if __name__ == '__main__':
    _x = torch.rand((3, 3, 224, 224))
    model = resnet_50_deform()
    res = model(_x)
    print(res.size())
