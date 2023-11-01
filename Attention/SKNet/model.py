import torch
import torch.nn as nn
import typing as t
from functools import reduce
from torchsummary import summary
from other_utils.conv import SKBlock


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
                              padding=1 + dilation - 1 if kernel_size == 3 else 0 + dilation - 1,
                              groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class SKBottleNeck(nn.Module):
    """
    ResNet or ResNeXt BottleNeck that includes SK Block
    """
    expansion = 2

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
            reduction: int = 16,
            base_width: int = 128,
    ):
        super(SKBottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = Conv(in_chans, width, 1, 1, norm_layer=norm_layer, act_layer=act_layer)
        self.sk = SKBlock(width, width, num=2, stride=stride, groups=groups, reduction=reduction, norm_layer=norm_layer,
                          act_layer=act_layer)
        self.conv3 = nn.Conv2d(width, out_chans * self.expansion, 1, bias=False)
        self.bn = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)
        self.down_sample = down_sample

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.sk(y)
        y = self.conv3(y)
        y = self.bn(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


class SKNet(nn.Module):
    """
    SK Block based on ResNet or ResNeXt
    """

    def __init__(
            self,
            block: t.Type[t.Union[SKBottleNeck]],
            layers: t.List[int],
            layers_output: t.List[int],  # each layer output by 3x3 kernel conv
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            classifier: bool = True,  # whether include GAP and fc layer to classify,
            base_width: int = 128,
    ):
        super(SKNet, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        if act_layer is None:
            self.act_layer = nn.ReLU

        self.in_chans = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.num_classes = num_classes

        # stage-conv1, 7x7 kernel, conv, s = 2, (224 - 7 + 2 * padding) / s + 1 == 112 => padding = 3
        self.conv1 = nn.Conv2d(3, self.in_chans, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.in_chans)
        self.act = self.act_layer(inplace=True)

        # stage-conv2, 3x3 kernel, max pool, s = 2, bottleneck
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers_output[0], layers[0], base_width=base_width)

        # stage-conv3
        self.layer2 = self._make_layer(block, layers_output[1], layers[1], 2, base_width=base_width)

        # stage-conv4
        self.layer3 = self._make_layer(block, layers_output[2], layers[2], 2, base_width=base_width)

        # stage-conv5
        self.layer4 = self._make_layer(block, layers_output[3], layers[3], 2, base_width=base_width)
        self.classifier = classifier
        if classifier:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(layers_output[-1] * block.expansion, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            # Initialize using normal distribution
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: t.Type[t.Union[SKBottleNeck, t.Callable]],
            out_chans: int,
            blocks: int,
            stride: int = 1,
            base_width: int = 128,
    ):
        down_sample = None
        # stride != 1 <==> in stage conv2~5, self.in_chans != out_chans * block.expansion <==> in stage conv1
        if stride != 1 or self.in_chans != out_chans * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_chans, out_chans * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(out_chans * block.expansion)
            )
        # each stage, the first block use to down sample, make sure that in_chans is equal to out_chans
        layers = [block(
            self.in_chans, out_chans, stride, self.groups, self.width_per_group, down_sample, self.norm_layer,
            self.act_layer, base_width=base_width
        )]
        self.in_chans = out_chans * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_chans,
                    out_chans,
                    1,
                    self.groups,
                    self.width_per_group,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    base_width=base_width
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # stage-conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # stage-conv2
        x = self.maxpool(x)
        x = self.layer1(x)

        # stage-conv3
        x = self.layer2(x)

        # stage-conv4
        x = self.layer3(x)

        # stage-conv5
        x = self.layer4(x)

        # classifier
        if self.classifier:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def sk_resnet_50(num_classes: int = 1000, classifier: bool = True):
    SKBottleNeck.expansion = 2
    return SKNet(
        SKBottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[128, 256, 512, 1024],
        groups=32,
        width_per_group=4,
        num_classes=num_classes,
        classifier=classifier
    )


def sk_resnet_101(num_classes: int = 1000, classifier: bool = True):
    SKBottleNeck.expansion = 2
    return SKNet(
        SKBottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[128, 256, 512, 1024],
        num_classes=num_classes,
        groups=32,
        width_per_group=4,
        classifier=classifier,
    )


def sk_resnet_152(num_classes: int = 1000, classifier: bool = True):
    SKBottleNeck.expansion = 2
    return SKNet(
        SKBottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[128, 256, 512, 1024],
        groups=32,
        width_per_group=4,
        num_classes=num_classes,
        classifier=classifier
    )


if __name__ == '__main__':
    _x = torch.rand((2, 3, 224, 224)).to(0)
    model = sk_resnet_50(1000, True).to(0)
    # res = model(_x)
    # print(res.size())
    summary(model, (3, 224, 224))
