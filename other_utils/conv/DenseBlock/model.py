import torch
import torch.nn as nn
import typing as t

from other_utils.utils import auto_pad


class Conv(nn.Module):
    """
    Basic Convolution include
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
            act_layer = nn.ReLU
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, auto_pad(kernel_size),
                              groups=groups, bias=False)
        self.norm = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(x)))


class DenseLayer(nn.Module):
    """
    channel number change:
    in_chans(l * growth_rate) -> growth_rate -> (l + 1) * growth_rate
    l means the number of layer in current dense block.
    """

    def __init__(
            self,
            in_chans: int,
            growth_rate: int,
            scale_factor: int = 4,
            drop_ratio: float = 0.1,
    ):
        super(DenseLayer, self).__init__()
        self.conv1 = Conv(in_chans, growth_rate * scale_factor, 1, 1)
        self.conv3 = Conv(growth_rate * scale_factor, growth_rate, 3, 1)
        self.drop = nn.Dropout(drop_ratio) if drop_ratio > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        new_feature = self.conv3(self.conv1(x))
        new_feature = self.drop(new_feature)
        return torch.cat((x, new_feature), dim=1)


class DenseBlock(nn.Module):

    def __init__(
            self,
            in_chans: int,
            growth_rate: int,
            layers_num: int,
            scale_factor: int = 4,
            drop_ratio: float = 0.1,
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(in_chans + i * growth_rate, growth_rate, scale_factor, drop_ratio)
                for i in range(layers_num)
            ])

    def forward(self, x: torch.Tensor):
        """
        the last x dim is (layer_num * growth_rate)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    reduce dimension,  dimension reduced from layer_num * growth_rate to num_feature * scale_factor
    the num_feature is the number of input channels for each dense block.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
    ):
        super(TransitionLayer, self).__init__()
        self.conv1 = Conv(in_chans, out_chans, 1, 1)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.pool(self.conv1(x))
