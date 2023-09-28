import torch
import torch.nn as nn
import typing as t
from other_utils.conv import GhostConv
from network.ResNeXt import ResNeXt, resnet50
from torchsummary import summary


class GhostBottleneck(nn.Module):
    """
    Ghost bottleneck based on Residual
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
            ratio: int = 4,
            dw_kernel_size: int = 3,
    ):
        super(GhostBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = GhostConv(in_chans, width, 1, ratio, dw_kernel_size, norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = GhostConv(width, width, 3, ratio, dw_kernel_size, stride=stride, norm_layer=norm_layer,
                               act_layer=act_layer)
        self.conv3 = GhostConv(width, out_chans * self.expansion, 1, ratio, dw_kernel_size, need_act=False,
                               norm_layer=norm_layer, act_layer=act_layer)
        self.down_sample = down_sample
        self.stride = stride
        self.bn = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)

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


def resnet_50_ghost(num_classes: int = 1000, classifier: bool = True, ratio: int = 4, dw_kernel_size: int = 3):
    GhostBottleneck.expansion = 4
    return ResNeXt(
        block=GhostBottleneck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        ratio=ratio,
        dw_kernel_size=dw_kernel_size,
    )


def resnet_101_ghost(num_classes: int = 1000, classifier: bool = True, ratio: int = 4, dw_kernel_size: int = 3):
    GhostBottleneck.expansion = 4
    return ResNeXt(
        block=GhostBottleneck,
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        ratio=ratio,
        dw_kernel_size=dw_kernel_size,
    )


def resnet_152_ghost(num_classes: int = 1000, classifier: bool = True, ratio: int = 4, dw_kernel_size: int = 3):
    GhostBottleneck.expansion = 4
    return ResNeXt(
        block=GhostBottleneck,
        layers=[3, 4, 36, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        ratio=ratio,
        dw_kernel_size=dw_kernel_size,
    )


if __name__ == '__main__':
    ratio = 4
    _x = torch.rand((3, 3, 224, 224)).to(0)
    ghost = resnet_50_ghost(80, True, ratio).to(0)
    res = ghost(_x)
    print(res.size())
    resnet_50 = resnet50(80, True).to(0)

    # The parameter quantity of ghost-resnet-50 is one half of the ratio of resnet-50 parameters
    print('----resnet50_ghost--------')
    summary(ghost, (3, 224, 224))
    print('----resnet50--------')
    summary(resnet_50, (3, 224, 224))
