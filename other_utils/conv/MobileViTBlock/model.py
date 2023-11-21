import torch
import torch.nn as nn
import typing as t
from einops import rearrange

__all__ = ['MV2Block', 'MobileViTBlock', 'Conv']


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    A residual block structure generated with a certain probability p based on the Bernoulli distribution.
    when p = 1, it is the original residual block; when p = 0, it drops the whole current network.

    the probability p is not const, while it related to the depth of the residual block.

    *note: stochastic depth actually is model fusion, usually used in residual block to skip some residual structures.
    to some extent, it fuses networks of different depths.
    """

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Stochastic depth level used in residual block.
    as the network become more deepens, training becomes complex,
    over-fitting and gradient disappearance occur, so, it uses stochastic depth
    to random drop partial network according to some samples in training, while keeps in testing.
    """

    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
            norm_layer: t.Optional[t.Callable] = None,
            act_layer: t.Optional[t.Callable] = None,
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


class MLP(nn.Module):
    """
    FFN, two FC and one activation
    """

    def __init__(
            self,
            in_chans: int,
            hidden_chans: int,
            out_chans: int,
            dropout: float = 0.
    ):
        super(MLP, self).__init__()
        self.net = nn.Sequential(*(
            nn.Linear(in_chans, hidden_chans),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_chans, out_chans),
            nn.Dropout(dropout)
        ))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Attention(nn.Module):
    """
    standard MHSA
    """

    def __init__(
            self,
            dim: int,
            patch_size: int,
            num_heads,
            attn_drop_ratio,
            proj_drop_ratio,
            qkv_bias: bool,
            qk_scale: float,
    ):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        qkv = self.qkv(x).chunk(3, dim=-1)
        # 把每个patch的大小放在第二维度, 这样在做Q @ K时, 实际上是基于像素级细粒度, 每个patch中的各个像素点的值的注意力值不相同
        q, k, v = map(lambda t: rearrange(t, 'b p_s n (h d) -> b p_s h n d', h=self.num_heads), qkv)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        b = (attn @ v)
        b = rearrange(b, 'b p_s h n d -> b p_s n (h d)')

        b = self.proj(b)
        b = self.proj_drop(b)
        return b

    def forward2(self, x: torch.Tensor):
        x = rearrange(x, 'b p_s n c -> b n (p_s c)')
        batch, num, channel = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[1]
        # 将Q中每一个vector与K中每一个vector计算attention, 进行scale缩放, 避免点积带来的方差对梯度的影响，
        # 得到a(i,j), 然后经过softmax得到key的权重
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算k经过softmax得到的key的权重，然后乘上对应的value向量，得到self-attention score
        # 即k个key向量对应的value向量的加权平均值
        b = (attn @ v).transpose(1, 2).reshape(batch, num, channel)
        b = self.proj(b)
        b = self.proj_drop(b)
        return b

    # def forward(self, x):
    #     res1 = self.forward1(x1)


class Transformer(nn.Module):
    """
    Standard Transformer
    """

    def __init__(
            self,
            dim: int,
            patch_size: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            drop_path_ratio: float = 0.,
            mlp_ratio: float = 4.,
    ):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, patch_size, num_heads, attn_drop_ratio, proj_drop_ratio, qkv_bias, qk_scale)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MobileViTBlock(nn.Module):

    def __init__(
            self,
            channel: int,
            dim: int,
            kernel_size: int,
            depth: int,
            patch_size: int = 2,
            num_head: int = 8,
            mlp_ratio: float = 4.,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            drop_path_ratio: float = 0.,
            qk_bias: bool = False,
            qk_scale: t.Optional[float] = None,
    ):
        """
        depth: the number of MobileViT block
        """
        super(MobileViTBlock, self).__init__()
        # use to learn local representation
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = Conv(channel, channel, kernel_size=kernel_size, act_layer=nn.SiLU)
        self.conv2 = Conv(channel, dim, kernel_size=1, act_layer=nn.SiLU)
        self.transformers = nn.ModuleList([
            Transformer(dim, patch_size, num_head, qk_bias, qk_scale, attn_drop_ratio, proj_drop_ratio,
                        drop_path_ratio, mlp_ratio)
            for _ in range(depth)
        ])
        self.conv3 = Conv(dim, channel, kernel_size=1, act_layer=nn.SiLU)
        self.conv4 = Conv(2 * channel, channel, kernel_size=kernel_size, act_layer=nn.SiLU)

    def forward(self, x: torch.Tensor):
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        _, _, h, w = y.shape
        y = rearrange(y, 'b c (h ph) (w pw) -> b (ph pw) (h w) c', ph=self.ph, pw=self.pw)
        for transformer in self.transformers:
            y = transformer(y)
        y = rearrange(y, 'b (ph pw) (h w) c -> b c (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        y = self.conv3(y)
        y = torch.cat((identity, y), dim=1)
        y = self.conv4(y)
        return y


class MV2Block(nn.Module):
    """
    MobileNetV2
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            stride: int,
            expand_t: int = 6,
    ):
        super().__init__()
        hidden_chans = int(in_chans * expand_t)
        self.shortcut = stride == 1 and in_chans == out_chans

        if expand_t == 1:
            self.mv2 = nn.Sequential(*(
                Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, act_layer=nn.SiLU),
                nn.Conv2d(hidden_chans, out_chans, 1, bias=False),
                nn.BatchNorm2d(out_chans)
            ))
        else:
            self.mv2 = nn.Sequential(*(
                Conv(in_chans, hidden_chans, 1, act_layer=nn.SiLU),
                Conv(hidden_chans, hidden_chans, 3, stride, groups=hidden_chans, act_layer=nn.SiLU),
                nn.Conv2d(hidden_chans, out_chans, 1, bias=False),
                nn.BatchNorm2d(out_chans)
            ))

    def forward(self, x: torch.Tensor):
        if self.shortcut:
            return x + self.mv2(x)
        else:
            return self.mv2(x)


if __name__ == '__main__':
    _x = torch.randn((2, 64, 80, 80))
    model = MobileViTBlock(64, 64, 3, 1)
    model(_x)