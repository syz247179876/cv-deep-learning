import math

import torch
import torch.nn as nn
import typing as t
from torchsummary import summary


class SpatialACT(nn.Module):
    """
    which is smoother than Relu
    """

    def __init__(self, inplace=True):
        super(SpatialACT, self).__init__()
        self.act = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor):
        return x * self.act(x + 3) / 6


class CABlock(nn.Module):
    """
    Coordinate Attention Block, which embeds positional information into channel attention.
    1.It considers spatial dimension attention and channel dimension attention, it helps model locate, identify and
    enhance more interesting objects.

    2.CA utilizes two 2-D GAP operation to respectively aggregate the input features along the vertical and horizontal
    directions into two separate direction aware feature maps. Then, encode these two feature maps separately into
    an attention tensor.

    3. Among these two feature maps(Cx1xW and CxHx1), one uses GAP to model the long-distance dependencies of
    the feature maps on a spatial dimension, while retaining position information int the other spatial dimension.
    In that case, the two feature maps, spatial information, and long-range dependencies complement each other.

    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            reduction: int = 32,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(CABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU6
        self.in_chans = in_chans
        self.out_chans = out_chans
        # (C, H, 1)
        self.gap_h = nn.AdaptiveAvgPool2d((None, 1))
        # (C, 1, W)
        self.gap_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_chans = max(8, in_chans // reduction)
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, 1)
        self.bn1 = norm_layer(hidden_chans)
        self.act = SpatialACT(inplace=True)

        self.attn_h = nn.Conv2d(hidden_chans, out_chans, 1)
        self.attn_w = nn.Conv2d(hidden_chans, out_chans, 1)
        self.sigmoid = nn.Sigmoid()

        # self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Obey uniform distribution during attention initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        identity = x
        b, c, h, w = x.size()
        x_h = self.gap_h(x)
        x_w = self.gap_w(x).permute(0, 1, 3, 2)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # compute attention
        a_h = self.sigmoid(self.attn_h(x_h))
        a_w = self.sigmoid(self.attn_w(x_w))

        return identity * a_w * a_h


if __name__ == '__main__':
    _x = torch.rand((3, 64, 7, 7)).to(0)
    ca = CABlock(64, 64).to(0)
    res = ca(_x)
    print(res.size())
    summary(ca, (64, 224, 224))
