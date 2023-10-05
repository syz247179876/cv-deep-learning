import torch
import torch.nn as nn
import typing as t
from network.ResNeXt import BaseConv, ResNeXt, BottleNeck


class Attention(nn.Module):
    """
    Basic self-attention layer with relative position embedding, but there are some difference from original
    transformer, difference as following:

    1.use relative position embedding
    2.use 1x1 convolution kernel to generate q, k, v instead of Linear layer.
    """

    def __init__(
            self,
            dim: int,
            height: int,
            width: int,
            head_num: int = 4,
            qkv_bias: bool = False,
    ):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num

        self.scale = qkv_bias or self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        # relative position embedding Rh and Rw
        self.rw = nn.Parameter(torch.randn((1, head_num, self.head_dim, 1, width)), requires_grad=True)
        self.rh = nn.Parameter(torch.randn((1, head_num, self.head_dim, height, 1)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        batch, c, h, w = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.head_num, c // self.head_num, -1).permute(1, 0, 2, 3, 4)
        # q, k, v dim is (b, head_num, head_dim, length)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute attention for content
        attn_content = (q.transpose(-2, -1) @ k) * self.scale

        # compute attention for position
        r = (self.rw + self.rh).view(1, self.head_num, self.head_dim, -1)
        attn_position = (q.transpose(-2, -1) @ r) * self.scale
        attn = self.softmax(attn_content + attn_position)

        attn = (attn @ v.transpose(-2, -1)).permute(0, 1, 3, 2).reshape(batch, c, h, w)
        return attn


class BoTBlock(nn.Module):
    """
    Bottleneck with Transformer(multi-head self-attention)

    Basic self-attention layer with relative position embedding
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
            head_num: int = 4,
            qkv_bias: bool = False,
    ):
        super(BoTBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        self.head_num = head_num
        self.qkv_bias = qkv_bias
        self.width = width = int(out_chans * (width_per_group / base_width)) * groups
        self.conv1 = BaseConv(in_chans, width, 1, 1, groups, norm_layer, act_layer)
        self.conv2 = nn.Conv2d(width, out_chans * self.expansion, 1, 1, 0, groups=groups, bias=False)
        self.norm = norm_layer(out_chans * self.expansion)
        self.act = act_layer(inplace=True)
        self.down_sample = down_sample
        self.stride = stride
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        identity = x
        y = self.conv1(x)
        attention = Attention(dim=self.width, height=h, width=w, head_num=self.head_num, qkv_bias=self.qkv_bias)
        y = attention(y)
        if self.stride != 1:
            y = self.pool(y)
        y = self.conv2(y)
        y = self.norm(y)

        if self.down_sample:
            identity = self.down_sample(identity)
        y += identity
        return self.act(y)


def resnet50_bot(num_classes: int = 1000, classifier: bool = False, head_num: int = 4, qkv_bias: bool = True):
    BoTBlock.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, BoTBlock],
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        head_num=head_num,
        qkv_bias=qkv_bias
    )


if __name__ == '__main__':
    _x = torch.rand((4, 3, 224, 224))
    model = resnet50_bot()

    res = model(_x)
    print(res.size())
