import os.path
import torch.nn as nn
import torch
import typing as t
import yaml

from torchsummary import summary
from other_utils.utils import auto_pad

ROOT = os.path.dirname(__file__)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    # rounded up to a multiple of 8
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv(nn.Module):
    """
    Basic Conv includes Conv2d(1x1 or 3x3), BN, Relu6
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int = 1,
            groups: int = 1,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6

        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size, stride, auto_pad(kernel_size),
                               groups=groups, bias=False)
        self.bn = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv1(x)))


class InvertedResidual(nn.Module):
    """
    the bottleneck(InvertedResidual) of MobileNetV2.
    In MobileNet, there are fewer channels at both ends and more channels in the middle for residual, which exactly
    opposite to ResNet.

    From my personal, I think MobileNet V2 use first 1x1 conv to enhance dimension in order to prevent the drawback
    of DWC extracting less rich feature information on fewer feature dimensions, that is why it different from ResNet
    residual block.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int,
            expand_t: int = 1,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6
        super(InvertedResidual, self).__init__()
        # used for dimensionality enhancement
        hidden_chans = int(in_chans * expand_t)
        self.identity = stride == 1 and in_chans == out_chans
        if expand_t == 1:
            self.conv1 = Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, norm_layer=norm_layer,
                              act_layer=act_layer)

            self.conv2 = nn.Conv2d(hidden_chans, out_chans, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_chans)
        else:
            self.conv1 = Conv(in_chans, hidden_chans, 1, norm_layer=norm_layer, act_layer=act_layer)
            self.conv2 = Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, norm_layer=norm_layer,
                              act_layer=act_layer)
            self.conv3 = nn.Conv2d(hidden_chans, out_chans, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_chans)
        self.expand_t = expand_t
        self.out_chans = out_chans

    def forward(self, x: torch.Tensor):
        _identity = x
        if self.expand_t == 1:
            x = self.bn2(self.conv2(self.conv1(x)))
        else:
            x = self.bn3(self.conv3(self.conv2(self.conv1(x))))
        if self.identity:
            x = _identity + x
        return x


class MobileNetV2(nn.Module):
    """
    MobileNet V2 proposed inverted residual and linear bottleneck based on MobileNet V1
    """

    def __init__(
            self,
            block: t.Optional[t.Union[t.List, t.Tuple, t.Callable]],
            cfg_name: str,
            num_classes: int = 1000,
            classifier: bool = True,
            **ac_kwargs,
    ):
        super(MobileNetV2, self).__init__()
        self.classifier = classifier
        cfg_path = os.path.join(ROOT, cfg_name)
        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            self.width_m = width_m = cfg.get('width_m', 1.0)
            bottlenecks = cfg.get('bottlenecks', None)
            assert bottlenecks is not None, f'The bottlenecks structure should be assigned in the {cfg_path}'
        if not isinstance(block, t.List):
            blocks = [block for _ in range(len(bottlenecks))]
        else:
            blocks = block

        in_chans = _make_divisible(32 * width_m, 4 if width_m == 0.1 else 8)
        layers = [Conv(3, in_chans, 3, stride=2)]
        for i, (t_, c, n, s) in enumerate(bottlenecks):
            out_chans = _make_divisible(c * width_m, 4 if width_m == 0.1 else 8)
            for j in range(n):
                layers.append(blocks[j](in_chans, out_chans, s if j == 0 else 1, t_, **ac_kwargs))
                in_chans = out_chans
        out_chans = _make_divisible(1280 * width_m, 4 if width_m == 0.1 else 8) if width_m > 1. else 1280
        self.feature = nn.Sequential(*layers)
        self.conv1 = Conv(in_chans, out_chans, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if classifier:
            self.classifier_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(out_chans, num_classes)
            )

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        x = self.conv1(x)
        if self.classifier:
            x = self.gap(x).view(x.size(0), -1)
            x = self.classifier_head(x)
        return x


def mobilenet_v2_1(num_classes: int = 1000, classifier: bool = True):
    return MobileNetV2(
        block=InvertedResidual,
        cfg_name='mobilenetv2-1.0.yaml',
        num_classes=num_classes,
        classifier=classifier,
    )


def mobilenet_v2_075(num_classes: int = 1000, classifier: bool = True):
    return MobileNetV2(
        block=InvertedResidual,
        cfg_name='mobilenetv2-0.75.yaml',
        num_classes=num_classes,
        classifier=classifier,
    )


def mobilenet_v2_05(num_classes: int = 1000, classifier: bool = True):
    return MobileNetV2(
        block=InvertedResidual,
        cfg_name='mobilenetv2-0.5.yaml',
        num_classes=num_classes,
        classifier=classifier,
    )


if __name__ == '__main__':
    _x = torch.rand(2, 3, 224, 224).to(0)
    model = mobilenet_v2_1(1000).to(0)
    res = model(_x)
    print(res.size())
    summary(model, (3, 224, 224))
