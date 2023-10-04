import torch
import torch.nn as nn
import typing as t

from network.ResNeXt import ResNeXt
from other_utils.conv import ODConv
from torchsummary import summary


class Conv(nn.Module):
    """
    Basic Conv includes ODConv/conv2d, BN, Relu, routing weights in one Residual Block share the same
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            expert_num: int = 4,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.conv = ODConv(in_chans, out_chans, kernel_size, stride, groups, dilation, reduction=reduction,
                           hidden_chans=hidden_chans, expert_num=expert_num, bias=False)
        self.norm = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    """
    Applied to reset18, resnet34, where replacing space convolution 3x3 with ODConv
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
            expert_num: int = 4,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            dilation: int = 1,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        self.conv1 = Conv(in_chans, out_chans, 3, stride, groups, dilation, reduction, hidden_chans, expert_num,
                          norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ODConv(out_chans, out_chans * self.expansion, 3, 1, groups, dilation, reduction, hidden_chans, expert_num,
                            norm_layer=norm_layer, act_layer=act_layer)
        self.norm2 = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: torch.Tensor):

        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.norm2(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


class Bottleneck(nn.Module):
    """
    Applied to reset50, resnet101, resnet152, where replacing space convolution 1x1 and 3x3 with ODConv
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
            expert_num: int = 4,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            dilation: int = 1,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        width = int(out_chans * (width_per_group / base_width) * groups)
        self.conv1 = Conv(in_chans, width, 1, 1, groups, dilation, reduction, hidden_chans, expert_num,
                          norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = Conv(width, width, 3, stride, groups, dilation, reduction, hidden_chans, expert_num,
                          norm_layer=norm_layer, act_layer=act_layer)
        self.conv3 = ODConv(width, out_chans * self.expansion, 1, 1, groups, dilation, reduction, hidden_chans,
                            expert_num, norm_layer=norm_layer, act_layer=act_layer)
        self.norm = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.norm(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


def resnet18_od(
        num_classes: int = 1000,
        classifier: bool = True,
        expert_num: int = 4,
        reduction: float = 0.0625,
        hidden_chans: int = 16,
):
    BasicBlock.expansion = 4
    return ResNeXt(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        expert_num=expert_num,
        reduction=reduction,
        hidden_chans=hidden_chans
    )


def resnet50_od(
        num_classes: int = 1000,
        classifier: bool = True,
        expert_num: int = 4,
        reduction: float = 0.0625,
        hidden_chans: int = 16,
):
    Bottleneck.expansion = 4
    return ResNeXt(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        expert_num=expert_num,
        reduction=reduction,
        hidden_chans=hidden_chans
    )


if __name__ == '__main__':
    model = resnet18_od().to(0)
    _x = torch.rand(4, 3, 224, 224).to(0)
    res = model(_x)
    print(res.size())
