import torch
import torch.nn as nn
import typing as t

import yaml
from einops import rearrange
from timm.models.layers import DropPath

from other_utils.utils import auto_pad

__all__ = ['cloformer', 'CloFormer']


class Conv(nn.Module):
    """
    default is conv + bn + relu
    """

    default_act = nn.ReLU()

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: t.Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            act: t.Union[t.Callable, nn.Module, bool] = True
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, auto_pad(kernel_size, padding, dilation),
                              dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class PatchEmbedding(nn.Module):
    """
    stem of CloFormer, the stem is comprised of four convolutions, each with strides of 2, 2, 1, and 1, respectively.
    use GN with group = 1, channel = out_chans to process 4D, its function is equivalent to LayerNorm(c, h, w)
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
    ):
        super(PatchEmbedding, self).__init__()
        self.stem = nn.Sequential(
            Conv(in_chans, out_chans // 2, kernel_size=3, stride=2),
            Conv(out_chans // 2, out_chans, kernel_size=3, stride=2),
            Conv(out_chans, out_chans, kernel_size=3, stride=1),
            Conv(out_chans, out_chans, kernel_size=3, stride=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1, stride=1),
            # using GN instead of LN to process 4D without passing in H, W
            nn.GroupNorm(1, out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class AttnMap(nn.Module):
    """
    The strong non-linearity leads to higher-quality context-aware weights.
    """

    def __init__(
            self,
            dim: int,
    ):
        super(AttnMap, self).__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Hardswish(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_block(x)


class CloAttention(nn.Module):
    """
    CloAttention with local branch and global branch,
    local branch uses shared and context-aware weights to extract low-frequency and high-frequency information,
    while global branch uses pooling to extract global information.

    note:
    1. kernel sizes used in AttnConv increased gradually and the number of channels occupied by global branches
    also increases with depth.
    2. the dimension of Q, K, V are equal.
    3. The attention here is not cascaded.
    4.Each head focuses on different aspects of features, such as some heads focusing on local branches and
    others on global branches
    5.sum(group_split) must be equal to num_heads
    6. The interaction between q and k is based on local windows, reducing computational complexity,
    but increasing receptive fields through pooling
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            group_split: t.List[int],
            kernel_sizes: t.List[int],
            window_size: int = 7,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            qkv_bias: bool = False,
    ):
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        super(CloAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaler = self.head_dim ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split

        convs = []
        act_blocks = []
        qkvs = []

        # local branch
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            qkvs.append(nn.Conv2d(dim, 3 * self.head_dim * group_head, 1, bias=qkv_bias))
            # DWC
            convs.append(nn.Conv2d(3 * self.head_dim * group_head, 3 * self.head_dim * group_head,
                                   kernel_size, 1, auto_pad(kernel_size), groups=3 * self.head_dim * group_head))
            # fc + act + fc
            act_blocks.append(AttnMap(self.head_dim * group_head))

        # global branch
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, self.head_dim * group_split[-1], 1, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, self.head_dim * group_split[-1] * 2, 1, bias=qkv_bias)
            # the interaction between q and k is based on local windows, reducing computational complexity,
            # but increasing receptive fields through pooling
            self.avg_pool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)

        self.proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def low_frequency_attention(self, x: torch.Tensor):
        """
        execute global branch, based on lcoal
        """

        # (b, c, h, w) -> (b, local_head_num, h*w, head_dim)
        b, c, h, w = x.shape
        q = self.global_q(x)
        q = rearrange(q, 'b (local_head_num head_dim) h w -> b local_head_num (h w) head_dim', head_dim=self.head_dim)
        kv = self.avg_pool(x)
        kv = self.global_kv(kv)
        kv = rearrange(kv, 'b (kv_num local_head_num head_dim) h w -> kv_num b local_head_num (h w) head_dim',
                       kv_num=2, head_dim=self.head_dim)
        # (b, head_num, h*w, head_dim)
        k, v = kv
        attn = q @ k.transpose(-1, -2) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v
        out = rearrange(out, 'b local_head_num (h w) head_dim -> b (local_head_num head_dim) h w',
                        h=h, w=w, head_dim=self.head_dim)
        return out

        # b, c, h, w = x.size()
        #
        # q = self.global_q(x).reshape(b, -1, self.head_dim, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        # kv = self.avg_pool(x)  # (b c h w)
        # kv = self.global_kv(kv).view(b, 2, -1, self.head_dim, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
        #                                                                                          3).contiguous()  # (2 b m (H W) d)
        # k, v = kv  # (b m (H W) d)
        # attn = self.scaler * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # res = attn @ v  # (b m (h w) d)
        # res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        # return res

    def high_frequency_attention(
            self,
            x: torch.Tensor,
            to_qkv: nn.Module,
            token_mixer: nn.Module,
            attn_block: nn.Module
    ) -> torch.Tensor:
        """
        execute local branch
        """
        b, c, h, w = x.size()
        qkv = to_qkv(x)
        # (3, b, dim, h, w), increase receptive fields
        qkv = token_mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        # (b, dim, h, w)
        q, k, v = qkv
        attn = attn_block(torch.mul(q, k).mul(self.scaler))
        attn = self.attn_drop(torch.tanh(attn))
        out = torch.mul(attn, v)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, c, h, w)
        # local_head_num + global_head_num = head_num = sum(groups_split)
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            # (b, local_head_num * head_dim, h, w)
            res.append(self.high_frequency_attention(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            # (b, global_head_num * head_dim, h, w)
            res.append(self.low_frequency_attention(x))
        # (b, c, h, w)
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class ConvFFN(nn.Module):
    """
    FFN with DWConv, without BN
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            hidden_chans: int,
            kernel_size: int,
            stride: int,
            act_layer=nn.GELU,
            mlp_drop=0.
    ):
        super(ConvFFN, self).__init__()
        self.fc1 = nn.Conv2d(in_chans, hidden_chans, 1, 1, 0)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_chans, hidden_chans, kernel_size, stride, auto_pad(kernel_size),
                             groups=hidden_chans)
        self.fc2 = nn.Conv2d(hidden_chans, out_chans, 1, 1, 0)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwc(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CloBlock(nn.Module):
    """
    CloBlock consists of Attention module and ConvFFN module, Attention module consists of a local branch
    and a global branch.

    1.A branch uses DWC with shared weights to aggregate local information for V.

    2.The other branch combines Q and K to generate context-aware weights, it is a different approach from local
    self-attention.
        - first, using two DWConv to aggregate local information for Q and K, respectively.
        - Second, calculate the Hadamard product of Q and K and introduce strong non-linearity to obtain higher quality
        context-aware weights.
        - Finally, fusion local branch and global branch to leverage the advantages of shared and context-aware weights
        in local perception.

    note:
    1.The CloBlock is entirely based on convolution, preserving the translation equivalence property of convolution.

    2.The module in second branch named AttnConv, it not only handles high-frequency(detail/texture/...) information
    better, but also better adapts to the input content during local perception.

    *3.Local branch and global branch have different learning conditions and requirements.
        - For local branch, it focuses on high-frequency details, local information, and low-level semantic information,
        it is more suitable for shallow layers.
        - For global branch, it focuses on low-frequency content, global information, and high-level semantic
        information, it is more suitable for deep layers.

    so, kernel sizes used in AttnConv increased gradually and the number of channels occupied by global branches
    also increases with depth.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            num_heads: int,
            group_split: t.List[int],
            kernel_sizes: t.List[int],
            window_size: int,
            mlp_kernel_size: int,
            mlp_ratio: int,
            stride: int,
            attn_drop: float = 0.,
            mlp_drop: float = 0.,
            drop_path: float = 0.,
            qkv_bias: bool = False,
    ):
        super(CloBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        # CloAttention
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = CloAttention(dim, num_heads, group_split, kernel_sizes, window_size, attn_drop, mlp_drop,
                                 qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ConvFFN
        self.norm2 = nn.GroupNorm(1, dim)
        hidden_dim = int(dim * mlp_ratio)
        self.stride = stride

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                Conv(dim, dim, mlp_kernel_size, stride=2, groups=dim, act=False),
                nn.Conv2d(dim, out_dim, 1, 1, 0),
            )
        self.mlp = ConvFFN(dim, out_dim, hidden_dim, mlp_kernel_size, stride, mlp_drop=mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CloLayer(nn.Module):
    """
    Stacking multiple CloBlocks
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            depth: int,
            num_heads: int,
            group_split: t.List[int],
            kernel_sizes: t.List[int],
            window_size: int,
            mlp_kernel_size: int,
            mlp_ratio: int,
            attn_drop: float = 0.,
            mlp_drop: float = 0.,
            drop_path: t.List[float] = [0., 0.],
            qkv_bias: bool = False,
            downsample: bool = True,
    ):
        super(CloLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.clo_blocks = nn.Sequential(
            *[
                CloBlock(dim, dim, num_heads, group_split, kernel_sizes, window_size, mlp_kernel_size, mlp_ratio,
                         1, attn_drop, mlp_drop, drop_path[i], qkv_bias=qkv_bias)
                for i in range(depth - 1)
            ]
        )
        if downsample:
            # need downsampling branch
            self.clo_blocks.append(
                CloBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size, mlp_kernel_size, mlp_ratio,
                         2, attn_drop, mlp_drop, drop_path[depth - 1], qkv_bias=qkv_bias)
            )
        else:
            self.clo_blocks.append(
                CloBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size, mlp_kernel_size, mlp_ratio,
                         1, attn_drop, mlp_drop, drop_path[depth - 1], qkv_bias=qkv_bias)
            )

    def forward(self, x: torch.Tensor):
        return self.clo_blocks(x)


class CloFormer(nn.Module):
    """
    CloFormer with four stages, each stages consist multiple CloBlocks.
    """

    def __init__(
            self,
            in_chans: int,
            embed_dims: t.List[int],
            depths: t.List[int],
            num_heads: t.List[int],
            group_splits: t.List[t.List[int]],
            kernel_sizes: t.List[t.List[int]],
            window_sizes: t.List[int],
            mlp_kernel_sizes: t.List[int],
            mlp_ratios: t.List[int],
            attn_drop: float = 0.,
            mlp_drop: float = 0.,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.1,
            num_classes: int = 1000,
            classifier: bool = False,
    ):
        super(CloFormer, self).__init__()
        self.num_classes = num_classes
        self.classifier = classifier
        self.num_layers = len(depths)

        # step1: PatchEmbedding
        self.stem = PatchEmbedding(in_chans, embed_dims[0])

        # step2: stack CloBlocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                layer = CloLayer(embed_dims[i], embed_dims[i + 1], depths[i], num_heads[i], group_splits[i],
                                 kernel_sizes[i], window_sizes[i], mlp_kernel_sizes[i], mlp_ratios[i], attn_drop,
                                 mlp_drop, dpr[sum(depths[:i]):sum(depths[:i + 1])], qkv_bias=qkv_bias, downsample=True)
            else:
                layer = CloLayer(embed_dims[i], embed_dims[i], depths[i], num_heads[i], group_splits[i],
                                 kernel_sizes[i], window_sizes[i], mlp_kernel_sizes[i], mlp_ratios[i], attn_drop,
                                 mlp_drop, dpr[sum(depths[:i]):sum(depths[:i + 1])], qkv_bias=qkv_bias,
                                 downsample=False)
            self.layers.append(layer)

        if classifier:
            self.classifier_head = nn.Sequential(
                nn.GroupNorm(1, embed_dims[-1]),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(embed_dims[-1], num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        if self.classifier:
            x = self.classifier_head(x)
        return x


def cloformer(num_classes: int = 1000, classifier: bool = False, cfg: str = 'CloFormer-XXS.yaml'):
    """
    CloFormer-XXS/XS/S
    """
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    return CloFormer(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


if __name__ == '__main__':
    x_ = torch.randn((1, 3, 640, 640)).to(0)
    model = cloformer().to(0)
    from torchsummary import summary

    summary(model, (3, 640, 640))
    res = model(x_)
    print(res.size())
