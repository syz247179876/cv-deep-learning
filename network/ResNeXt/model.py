import torch
import torch.nn as nn
import typing as t
from torchsummary import summary
from torchvision.models import resnet50


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
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride,
                              padding=1 if kernel_size == 3 else 0, groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """
    BottleNeck for ResNet18 and ResNet34, with two number of 3x3 convolution kernel
    """

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
            base_width=64,
            **kwargs
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        # maybe use conv to down sample
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_chans)
        self.act = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(out_chans)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: torch.Tensor):

        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


class BottleNeck(nn.Module):
    """
    BottleNeck for ResNeXt50, ResNeXt101, ResNeXt152, with two 1x1 conv and one 3x3 conv

    add extra two params compared to ResNet BottleNeck:
        groups: the number of Group Convolution, group num.
        width_per_group: the number of Convolution kernel in each group
    """
    expansion = 2
    _conv = Conv

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
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = self._conv(in_chans, width, 1, 1, 1, norm_layer, act_layer)
        # when stride > 1, need to down sample x in forward, use conv replace pooling
        self.conv2 = self._conv(width, width, 3, stride, groups, norm_layer, act_layer)
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


class ResNeXt(nn.Module):
    """
    ResNeXt model integrates the ideas of VGG, ResNet, Inception, Group Convolution.
    ResNeXt proposed a new unit -- Cardinality, composed of Split, Transform, Merge, it can serve a hyperparameter.
    As the benefits of width and depth gradually decrease, increasing Cardinality can increase performance.

    ResNeXt has several features:

    1.VGG idea that stacking multiple modules of the same shape, while ResNeXt also stacking multiple modules
    horizontally, called 'Cardinality'.

    2.Inception idea that split-transform-merge, improving accuracy, decline error. However, It is somewhat different
    from traditional Inception.
        - Traditional Inception: Artificially designed convolutional combinations.
        - ResNeXt Inception idea: Each path(Cardinality) has the same topological structure, simple network design.

    3.Group Convolution is equivalent to ResNeXt, because the convolutional kernel weights between groups are
    independent of each other, it has the same effect as splitting into multiple path(a path is a Cardinality)
    convolution. Also, ResNeXt Block has fewer parameters quantities compared to ResNet Block, because multiple
    Cardinality or groups.
    """

    def __init__(
            self,
            block: t.Type[t.Union[BasicBlock, BottleNeck, t.Callable, t.List, t.Tuple]],
            layers: t.List[int],
            layers_output: t.List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            classifier: bool = True,  # whether include GAP and fc layer to classify
            base_width: int = 64,
            **kwargs
    ):
        super(ResNeXt, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        if act_layer is None:
            self.act_layer = nn.ReLU

        if not isinstance(block, (t.List, t.Tuple)):
            block = [block for _ in range(len(layers))]

        self.in_chans = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.num_classes = num_classes

        # stage-conv1, 7x7 kernel, conv, s = 2, (224 - 7 + 2 * padding) / s + 1 == 112 => padding = 3
        self.conv1 = nn.Conv2d(3, self.in_chans, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.in_chans)
        self.act = self.act_layer(inplace=True)

        # stage-conv2, 3x3 kernel, max pool, s = 2, bottleneck
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block[0], layers_output[0], layers[0], base_width=base_width, **kwargs)

        # stage-conv3
        self.layer2 = self.make_layer(block[1], layers_output[1], layers[1], 2, base_width=base_width, **kwargs)

        # stage-conv4
        self.layer3 = self.make_layer(block[2], layers_output[2], layers[2], 2, base_width=base_width, **kwargs)

        # stage-conv5
        self.layer4 = self.make_layer(block[3], layers_output[3], layers[3], 2, base_width=base_width, **kwargs)
        self.classifier = classifier
        if classifier:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(layers_output[-1] * block[-1].expansion, num_classes)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            # Initialize using normal distribution
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def make_layer(
            self,
            block: t.Type[t.Union[BasicBlock, BottleNeck, t.Callable]],
            out_chans: int,
            blocks: int,
            stride: int = 1,
            base_width=128,
            **kwargs
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
            self.act_layer, base_width=base_width, **kwargs
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
                    base_width=base_width,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # stage-conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # stage-conv2
        x = self.max_pool(x)
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


def resnet50(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def resnet101(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def resnet152(num_classes: int = 1000, classifier: bool = True):
    BottleNeck.expansion = 4
    return ResNeXt(
        block=BottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
    )


def ResNeXt50_32x4d(num_classes: int = 1000, classifier: bool = True):
    """
    ResNeXt-50 32 groups and 4 kernel per group
    """
    BottleNeck.expansion = 2
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


def ResNeXt101_32x4d(num_classes: int = 1000, classifier: bool = True):
    """
    ResNeXt-101 32 groups and 4 kernel per group
    """
    BottleNeck.expansion = 2
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


def ResNeXt152_32x4d(num_classes: int = 1000, classifier: bool = True):
    """
    ResNeXt-152 32 groups and 4 kernel per group
    """
    BottleNeck.expansion = 2
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
    resnext50 = ResNeXt50_32x4d(num_classes=1000, classifier=True).to(0)
    res = resnext50(_x)
    # has fewer params
    summary(resnext50, (3, 224, 224), batch_size=4)
