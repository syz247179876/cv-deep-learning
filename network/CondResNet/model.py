import torch
import torch.nn as nn
import typing as t
from torchsummary import summary
from network.ResNeXt import ResNeXt, BottleNeck
from other_utils.conv import CondConv, RoutingFunc
from other_utils.utils import auto_pad


class Conv(nn.Module):
    """
    Basic Conv includes CondConv/conv2d, BN, Relu, routing weights in one Residual Block share the same
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            num_experts: int = 8,
            dilation: int = 1,
            drop_ratio: float = 0.6,
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
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding=auto_pad(kernel_size),
                                  groups=groups, bias=False)
        else:
            self.conv = CondConv(in_chans, out_chans, kernel_size, stride=stride, num_experts=num_experts,
                                 groups=groups, dilation=dilation, drop_ratio=drop_ratio, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)
        self.ordinary_conv = ordinary_conv

    def forward(self, x: torch.Tensor, routing_weights: t.Optional[torch.Tensor] = None):
        if self.ordinary_conv:
            x = self.conv(x)
        else:
            x = self.conv(x, routing_weights)
        return self.act(self.bn(x))


class CondBottleNeck(nn.Module):
    """
    bottleneck based on Residual and CondConv, the first 1x1 conv use ordinary convolution, the second 3x3
    conv and the third 1x1 conv use CondConv which share routing weights

    note:
    CondConv replaces the convolutional layers in the final several blocks, 3 for the ResNet Backbones, while
    6 for the MobileNetV2, the number of blocks applied in ResNet in less than MobileNetV2, because MobileNetV2 is
    more lightweight than ResNet model. As the model becomes larger, replacing CondConv results in more
    redundant information, and the benefits of improving accuracy are smaller than those of lightweight models.
    This was validated through experiments in the ODConv paper on ResNet50 and ResNet18 backbones.
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
            num_experts: int = 8,
            drop_ratio: float = 0.6,
            dilation: int = 1,
    ):

        super(CondBottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        width = int(out_chans * (width_per_group / base_width)) * groups

        self.conv1 = Conv(in_chans, width, 1, groups=groups, num_experts=num_experts,
                          drop_ratio=drop_ratio, norm_layer=norm_layer, act_layer=act_layer, ordinary_conv=True)
        self.conv2 = Conv(width, width, 3, stride, groups, num_experts, dilation, drop_ratio, norm_layer, act_layer)
        self.conv3 = CondConv(width, out_chans * self.expansion, 1, num_experts=num_experts, groups=groups,
                              drop_ratio=drop_ratio, bias=False)
        self.bn = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)
        self.stride = stride
        self.down_sample = down_sample
        self.routing = RoutingFunc(width, num_experts, drop_ratio)

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        routing_path = self.routing(y)
        y = self.conv2(y, routing_path)
        y = self.conv3(y, routing_path)
        y = self.bn(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


def resnet_50_cond(num_classes: int = 1000, classifier: bool = True, num_experts: int = 8, drop_ratio: float = 0.6):
    CondBottleNeck.expansion = 4
    BottleNeck.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, CondBottleNeck],
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        num_experts=num_experts,
        drop_ratio=drop_ratio
    )


def resnet_101_cond(num_classes: int = 1000, classifier: bool = True, num_experts: int = 8, drop_ratio: float = 0.6):
    CondBottleNeck.expansion = 4
    BottleNeck.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, CondBottleNeck],
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        num_experts=num_experts,
        drop_ratio=drop_ratio
    )


if __name__ == '__main__':
    model = resnet_50_cond().to(0)
    # model = resnet50(pretrained=False).to(0)
    _x = torch.rand((3, 3, 224, 224)).to(0)
    res = model(_x)
    print(res.size())
    summary(model, (3, 224, 224), batch_size=4)
