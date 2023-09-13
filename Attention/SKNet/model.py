import torch
import torch.nn as nn
import typing as t
from functools import reduce


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
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
        # return self.act(self.bn(self.conv(x)))


class SKBlock(nn.Module):
    """
    SK Module combines the Inception and SE ideas, considering different channels and kernel block.
    It can split into three parts:

        1.Split: For any feature map, using different size kernel convolutions(3x3, 5x5)
        to extract new feature map. use dilation convolution (3x3, dilation=2) can
        increase regularization to avoid over-fitting caused by large convolutional kernels
        , add multi-scale information and increase receptive field.

        2.Fuse: Fusing different output(last layer feature map) of different branch and
        compute attention on channel-wise

        3.Select: 'chunk, scale, fuse'.  Focusing on different convolutional kernels for different target sizes.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            num: int = 2,
            stride: int = 1,
            groups: int = 1,
            reduction: int = 16,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            attention_mode: str = 'conv',
    ):
        """
            num: the number of different kernel, by the way, it means the number of different branch, using
                Inception ideas.
            reduction: Multiple of dimensionality reduction, used to reduce params quantities and improve nonlinear
                ability.
            attention_mode: use linear layer(linear) or 1x1 convolution(conv)
        """
        super(SKBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        self.num = num
        self.out_chans = out_chans
        self.conv = [Conv(in_chans, out_chans, 3, stride=stride, groups=groups, dilation=1 + i,
                          norm_layer=norm_layer, act_layer=act_layer) for i in range(num)]

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # fc can be implemented by 1x1 conv or linear
        assert attention_mode in ['conv', 'linear'], 'fc layer should be implemented by conv or linear'
        if attention_mode == 'conv':
            self.fc = nn.Sequential(
                # use relu act to improve nonlinear expression ability
                nn.Conv2d(in_chans, out_chans // reduction, 1, bias=False),
                act_layer(inplace=True),
                nn.Conv2d(out_chans // reduction, out_chans * self.num, 1, bias=False),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_chans, out_chans // reduction, bias=False),
                act_layer(inplace=True),
                nn.Linear(out_chans // reduction, out_chans * self.num, bias=False)
            )
        # compute channels weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        batch, c, _, _ = x.size()
        # use different convolutional kernel to conv
        temp_feature = [conv(x) for conv in self.conv]
        # fuse different output
        u = reduce(lambda a, b: a + b, temp_feature)
        # squeeze
        u = self.gap(u)
        # excitation
        z = self.fc(u)
        z = z.reshape(batch, self.num, self.out_chans, -1)
        z = self.softmax(z)
        # select
        a_b_weight: t.List[..., torch.Tensor] = torch.chunk(z, self.num, dim=1)
        a_b_weight = [c.reshape(batch, self.out_chans, 1, 1) for c in a_b_weight]
        v = map(lambda weight, feature: weight * feature, a_b_weight, temp_feature)
        v = reduce(lambda a, b: a + b, v)
        return v


class SKBottleNeck(nn.Module):
    """
    ResNet or ResNeXt BottleNeck that includes SK Block
    """

    def __init__(self):
        super(SKBottleNeck, self).__init__()


class SKNet(nn.Module):
    """
    SK Block based on ResNet or ResNeXt
    """

    def __init__(self):
        super(SKNet, self).__init__()


if __name__ == '__main__':
    _x = torch.rand((1, 128, 56, 56))
    block = SKBlock(128, 128, 2, groups=32)
    res = block(_x)
    print(res.size())
