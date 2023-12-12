import torch
import torch.nn as nn
import typing as t

from other_utils.utils import auto_pad

__all__ = ['AFPNBlock', ]


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act: t.Union[bool, nn.Module] = True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class BasicBlock(nn.Module):
    """
    BasicBlock with residual
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int = 1,
            down_sample: t.Optional[nn.Module] = None,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
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


class ASFFTwo(nn.Module):
    """
    Adaptive attention mechanism based on spatial dimension

    Fusion two different layers into one new layers
    """

    def __init__(
            self,
            in_chans: int,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(ASFFTwo, self).__init__()

        if act_layer is None:
            act_layer = nn.ReLU
        compress_c = 8
        self.in_chas = in_chans
        self.weight_level_1 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=False))
        self.weight_level_2 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=False))

        # spatial attention
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.conv = Conv(in_chans, in_chans, 3, 1, act=act_layer(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        level_1 = self.weight_level_1(x1)
        level_2 = self.weight_level_2(x2)
        levels = torch.cat([level_1, level_2], dim=1)
        levels_weight = self.weight_levels(levels)
        levels_weight = self.softmax(levels_weight)
        # share for all channels
        fused_out = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :]
        out = self.conv(fused_out)
        return out


class ASFFThree(nn.Module):
    """
    Adaptive attention mechanism based on spatial dimension

    Fusion three different layers into one new layer
    """

    def __init__(self, in_chans: int, act_layer: t.Optional[t.Callable] = None):
        super(ASFFThree, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.in_chans = in_chans
        compress_c = 8
        self.weight_level_1 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=False))
        self.weight_level_2 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=False))
        self.weight_level_3 = Conv(in_chans, compress_c, 1, 1, act=act_layer(inplace=False))

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.conv = Conv(in_chans, in_chans, 3, 1, act=act_layer(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        level_1 = self.weight_level_1(x1)
        level_2 = self.weight_level_2(x2)
        level_3 = self.weight_level_3(x3)
        levels = torch.cat([level_1, level_2, level_3], dim=1)
        levels_weight = self.weight_levels(levels)
        levels_weight = self.softmax(levels_weight)
        # share for all channels
        fused_out = x1 * levels_weight[:, 0:1, :, :] + x2 * levels_weight[:, 1:2, :, :] + x3 * levels_weight[:, 2:3, :,
                                                                                               :]
        fused_out = self.conv(fused_out)
        return fused_out


class AFPNUpsample(nn.Module):
    """
    the upsample in AFPN
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            act_layer: t.Callable,
            scale_factor: int = 2,
            mode: str = 'bilinear',
    ):
        super(AFPNUpsample, self).__init__()
        self.upsample = nn.Sequential(
            Conv(in_chans, out_chans, act=act_layer(inplace=False)),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return x


class AFPNBlock(nn.Module):
    """
    Asymptotic Feature Pyramid Network (AFPN)
    AFPN Block, it adopts cross layer progressive fusion and adaptive space-wise attention mechanism (ASFF)
    """

    def __init__(
            self,
            channels: t.Union[t.List, t.Tuple],
            act_layer: t.Optional[t.Callable] = None
    ):
        super(AFPNBlock, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU

        # The semantic degree, from low to high, is 0, 1, and 2, corresponding to top, mid and bottom

        # fuse low and mid semantics layers, 0_0_1 means first fuse, from 0 layer to 1 layer
        self.down_sample_1_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=True))
        self.up_sample_1_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer)
        # 1_0 means first fuse, fused layer is 0 layer
        self.asff_top1_0 = ASFFTwo(in_chans=channels[0])
        self.asff_mid1_1 = ASFFTwo(in_chans=channels[1])

        # fuse low, mid, high semantics layers, 2_0_1 means second fuse, from 0 layer to 1 layer
        self.down_sample_2_0_1 = Conv(channels[0], channels[1], 2, 2, p=0, act=act_layer(inplace=False))
        self.down_sample_2_0_2 = Conv(channels[0], channels[2], 4, 4, p=0, act=act_layer(inplace=False))
        self.down_sample_2_1_2 = Conv(channels[1], channels[2], 2, 2, p=0, act=act_layer(inplace=False))
        self.up_sample_2_1_0 = AFPNUpsample(channels[1], channels[0], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')
        self.up_sample_2_2_0 = AFPNUpsample(channels[2], channels[0], act_layer=act_layer, scale_factor=4,
                                            mode='bilinear')
        self.up_sample_2_2_1 = AFPNUpsample(channels[2], channels[1], act_layer=act_layer, scale_factor=2,
                                            mode='bilinear')

        # 2_0 means second fuse, fused layer is 1 layer
        self.asff_top2_0 = ASFFThree(in_chans=channels[0])
        self.asff_mid2_1 = ASFFThree(in_chans=channels[1])
        self.asff_bottom2_2 = ASFFThree(in_chans=channels[2])

    def forward(self, x: t.Union[t.List[torch.Tensor], t.Tuple[torch.Tensor]]) -> t.Tuple[torch.Tensor, ...]:
        x0, x1, x2 = x

        x1_0 = self.asff_top1_0(x0, self.up_sample_1_1_0(x1))
        x1_1 = self.asff_mid1_1(self.down_sample_1_0_1(x0), x1)

        # TODO: Increase the connection between different scales in convolutional modeling, such as C3, C2f...
        x2_0 = self.asff_top2_0(x1_0, self.up_sample_2_1_0(x1_1), self.up_sample_2_2_0(x2))
        x2_1 = self.asff_mid2_1(self.down_sample_2_0_1(x1_0), x1_1, self.up_sample_2_2_1(x2))
        x2_2 = self.asff_bottom2_2(self.down_sample_2_0_2(x1_0), self.down_sample_2_1_2(x1_1), x2)

        return x2_0, x2_1, x2_2


if __name__ == '__main__':
    _x0 = torch.randn((1, 128, 64, 64))
    _x1 = torch.randn((1, 256, 32, 32))
    _x2 = torch.randn((1, 512, 16, 16))

    block = AFPNBlock([128, 256, 512], act_layer=nn.SiLU)

    _y0, _y1, _y2 = block([_x0, _x1, _x2])
    print(_y0, _y1, _y2)
    print(_y0.size(), _y1.size(), _y2.size())
