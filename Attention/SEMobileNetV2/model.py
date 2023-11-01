import torch
import torch.nn as nn
import typing as t

from torchsummary import summary

from network.MobileNetV2 import Conv, MobileNetV2
from other_utils.conv import SEBlock


class SEInvertedResidual(nn.Module):
    """
    Inverted Residual Block of MobileNetV2 with SE module
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int,
            expand_t: int = 1,
            reduction: int = 32,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6
        super(SEInvertedResidual, self).__init__()
        # used for dimensionality enhancement
        hidden_chans = int(in_chans * expand_t)
        self.identity = (stride == 1 and in_chans == out_chans)
        if expand_t == 1:
            self.conv1 = Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, norm_layer=norm_layer,
                              act_layer=act_layer)
            self.conv2 = nn.Conv2d(hidden_chans, out_chans, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_chans)
        else:
            self.conv1 = Conv(in_chans, hidden_chans, 1, norm_layer=norm_layer, act_layer=act_layer)
            self.conv2 = Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, norm_layer=norm_layer,
                              act_layer=act_layer)
            self.attention = SEBlock(hidden_chans, reduction)
            self.conv3 = nn.Conv2d(hidden_chans, out_chans, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_chans)
        self.expand_t = expand_t
        self.out_chans = out_chans

    def forward(self, x: torch.Tensor):
        _identity = x
        if self.expand_t == 1:
            x = self.bn2(self.conv2(self.conv1(x)))
        else:
            x = self.bn3(self.conv3(self.attention(self.conv2(self.conv1(x)))))
        if self.identity:
            x = _identity + x
        return x


def se_mobilenet_v2_1(num_classes: int = 1000, classifier: bool = True, reduction: int = 32):
    return MobileNetV2(
        block=SEInvertedResidual,
        cfg_name='mobilenetv2-1.0.yaml',
        num_classes=num_classes,
        classifier=classifier,
        reduction=reduction,
    )


def se_mobilenet_v2_075(num_classes: int = 1000, classifier: bool = True, reduction: int = 32):
    return MobileNetV2(
        block=SEInvertedResidual,
        cfg_name='mobilenetv2-0.75.yaml',
        num_classes=num_classes,
        classifier=classifier,
        reduction=reduction,
    )


def se_mobilenet_v2_05(num_classes: int = 1000, classifier: bool = True, reduction: int = 32):
    return MobileNetV2(
        block=SEInvertedResidual,
        cfg_name='mobilenetv2-0.5.yaml',
        num_classes=num_classes,
        classifier=classifier,
        reduction=reduction,
    )


if __name__ == '__main__':
    _x = torch.rand((2, 3, 224, 224)).to(0)
    model = se_mobilenet_v2_1(1000).to(0)
    res = model(_x)
    print(res.size())
    summary(model, (3, 224, 224))
