import torch
import torch.nn as nn
import typing as t
from torchsummary import summary


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
        self.act = act_layer()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """
    SE Block: A Channel Based Attention Mechanism.

        Traditional convolution in computation, it blends the feature relationships of the channel
    with the spatial relationships learned from the convolutional kernel, because a conv sum the
    operation results of each channel, so, using SE Block to pay attention to more important channels,
    suppress useless channels.

    SE Block Contains three parts:
    1.Squeeze: Global Information Embedding.
        Aggregate (H, W, C) dim to (1, 1, C) dim, use GAP to generate aggregation channel,
    encode the entire spatial feature on a channel to a global feature.

    2.Excitation: Adaptive Recalibration.
        It aims to fully capture channel-wise dependencies and improve the representation of image,
    by using two liner layer, one activation inside and sigmoid or softmax to normalize,
    to produce channel-wise weights.
        Maybe like using liner layer to extract feature map to classify, but this applies at channel
    level and pay attention to channels with a large number of information.

    3.Scale: feature recalibration.
        Multiply the learned weight with the original features to obtain new features.
        SE Block can be added to Residual Block.
    """

    def __init__(self, channel: int, reduction: int):
        super(SEBlock, self).__init__()
        # part 1:(H, W, C) -> (1, 1, C)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # part 2, compute weight of each channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),  # nn.Softmax is OK here
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

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
            base_width: int = 64,
    ):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.down_sample = down_sample
        self.conv1 = Conv(in_chans, out_chans, kernel_size=3, stride=stride,
                          norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)
        self.se = SEBlock(out_chans, reduction)

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.bn(y)
        y = self.se(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


class SEBottleNeck(nn.Module):
    """
    the bottleneck based on SE Block and Residual Block
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
            base_width: int = 64,
    ):
        super(SEBottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = Conv(in_chans, width, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = Conv(width, width, kernel_size=3, stride=stride, groups=groups,
                          norm_layer=norm_layer, act_layer=act_layer)

        self.conv3 = nn.Conv2d(width, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu3 = act_layer(inplace=True)

        self.se = SEBlock(out_chans * self.expansion, reduction)
        self.down_sample = down_sample

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.se(y)

        if self.down_sample is not None:
            identity = self.down_sample(identity)
        y += identity
        return self.relu3(y)


class SEResNet(nn.Module):
    """
    ResNet + SE Block
    """

    def __init__(
            self,
            block: t.Type[t.Union[SEBasicBlock, SEBottleNeck]],
            layers: t.List[int],
            layers_output: t.List[int],  # each layer output by 3x3 kernel conv
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            classifier: bool = True,  # whether include GAP and fc layer to classify,
            base_width: int = 64,
    ):
        super(SEResNet, self).__init__()
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
            block: t.Type[t.Union[SEBasicBlock, SEBottleNeck, t.Callable]],
            out_chans: int,
            blocks: int,
            stride: int = 1,
            base_width: int = 64,
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


def se_resnet_18(num_classes: int = 1000, classifier: bool = True):
    return SEResNet(
        SEBasicBlock,
        layers=[2, 2, 2, 2],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier
    )


def se_resnet_34(num_classes: int = 1000, classifier: bool = True):
    return SEResNet(
        SEBasicBlock,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier
    )


def se_resnet_50(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 4
    return SEResNet(
        SEBottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier
    )


def se_resnet_101(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 4
    return SEResNet(
        SEBottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
    )


def se_resnet_152(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 4
    return SEResNet(
        SEBottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier
    )


def se_resnext_50_32d4(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 2
    return SEResNet(
        SEBottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[128, 256, 512, 1024],
        groups=32,
        width_per_group=4,
        num_classes=num_classes,
        classifier=classifier,
        base_width=128
    )


def se_resnext_101_32d4(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 2
    return SEResNet(
        SEBottleNeck,
        layers=[3, 4, 23, 3],
        layers_output=[128, 256, 512, 1024],
        groups=32,
        width_per_group=4,
        num_classes=num_classes,
        classifier=classifier,
        base_width=128
    )


def se_resnext_152_32d4(num_classes: int = 1000, classifier: bool = True):
    SEBottleNeck.expansion = 2
    return SEResNet(
        SEBottleNeck,
        layers=[3, 8, 36, 3],
        layers_output=[128, 256, 512, 1024],
        groups=32,
        width_per_group=4,
        num_classes=num_classes,
        classifier=classifier,
        base_width=128
    )


if __name__ == '__main__':
    x_ = torch.randn(3, 3, 224, 224).to(0)
    model = se_resnet_50(20, classifier=True).to(0)
    model2 = se_resnext_50_32d4(20, classifier=True).to(0)
    model3 = se_resnet_101(20, classifier=True).to(0)
    model4 = se_resnet_18(20, classifier=True).to(0)
    # res = model(x_)
    # summary(model, (3, 224, 224))
    summary(model4, (3, 224, 224))
