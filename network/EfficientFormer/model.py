import itertools

import torch
import torch.nn as nn
import typing as t

import yaml

from other_utils.utils import auto_pad
from timm.models.layers import DropPath

from other_utils.utils.util import build_norm, build_act


class Conv(nn.Module):
    """
    Conv + Normalization + Activation
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            use_bias: bool = False,
            dropout: float = 0.,
            norm_layer: nn.Module = nn.BatchNorm2d,
            act_layer: nn.Module = nn.ReLU,
    ):
        super(Conv, self).__init__()

        self.dropout = nn.Dropout2d(dropout, inplace=True) if dropout > 0. else None
        self.conv = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size, dilation),
            groups=groups,
            bias=use_bias
        )
        self.norm = norm_layer(out_chans)
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.norm(self.conv(x)))


class Conv2dBN(torch.nn.Sequential):
    """
    Conv + BN module, providing fuse function, in order to fuse Conv and BN
    """

    def __init__(self, in_chans: int, out_chans: int, ks=1, stride=1, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            in_chans, out_chans, ks, stride, auto_pad(ks, d=dilation), dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(out_chans))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(nn.Module):
    """
    an embedding operation to project embedding dimension and downsample token length.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 2,
    ):
        super(PatchMerging, self).__init__()
        self.down_padding = Conv2dBN(in_chans, out_chans, ks=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        return self.down_padding(x)


class Stem(nn.Module):
    """
    stem of EfficientFormer, including several hardware-efficient 3x3 convolutions
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
    ):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            Conv(in_chans, out_chans // 2, kernel_size=3, stride=2),
            Conv(out_chans // 2, out_chans, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor):
        return self.stem(x)


class FFN4D(nn.Module):
    """
    FFN with Conv1x1 + GeLU + BN
    """

    def __init__(
            self,
            in_chans: int,
            hidden_chans: int,
            drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
    ):
        super(FFN4D, self).__init__()
        self.conv_bn1 = Conv2dBN(in_chans, hidden_chans, ks=1)
        self.act1 = act_layer()
        self.conv_bn2 = Conv2dBN(hidden_chans, in_chans, ks=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.drop(self.act1(self.conv_bn1(x)))
        x = self.drop(self.conv_bn2(x))
        return x


class MB4D(nn.Module):
    """
    dimension is (B, C, H, W) 4D Tensor, it follows the TransformerBlock structure and MetaFormer(PoolFormer),
    using Pooling to replace MHSA to reduce reshape operation, using Conv1x1 + BN to replace LN + Linear,
    because Conv + BN can be fused into one Conv, while the accuracy drawback of Conv + BN is generally acceptable.
    """

    def __init__(
            self,
            dim: int,
            pool_size: int = 3,
            mlp_ratio: float = 4.,
            drop: float = 0.,
            drop_path: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            act_layer: nn.Module = nn.GELU,
    ):
        super(MB4D, self).__init__()
        hidden_chans = int(dim * mlp_ratio)
        self.token_mixer = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=auto_pad(k=pool_size),
                                        count_include_pad=False)
        self.mlp = FFN4D(dim, hidden_chans, drop, act_layer=act_layer)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        # use
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )

    def forward(self, x: torch.Tensor):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class Attention(nn.Module):
    """
    it follows conventional ViT structure, but has some improvements.
    - The dimension of Q and K is less than V, because Q and K may produce large redundancy when they have the
    same dimensions as V
    """

    def __init__(
            self,
            dim: int,
            q_k_dim: int = 32,
            num_head: int = 8,
            multi_v: float = 4.,
            qk_scale: t.Optional[float] = None,
            qkv_bias: bool = False,
            resolution: int = 25,
    ):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.scale = qk_scale or q_k_dim ** -0.5
        self.v_dim = int(multi_v * q_k_dim)
        self.multi_v = multi_v
        self.h_v_dim = self.v_dim * num_head
        self.q_k_dim = q_k_dim
        self.h_q_k_dim = self.q_k_dim * num_head

        self.qkv = nn.Linear(dim, self.h_v_dim + 2 * self.h_q_k_dim, bias=qkv_bias)
        self.proj = nn.Linear(self.h_v_dim, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # *** (h, h*w), attention_biases indicates the importance of each pixel in H*W area
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_head, len(attention_offsets)))
        # store in model.state_dict(), but no updates are made during backpropagation
        # *对称矩阵(symmetric matrix)
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """
        cache functional!

        when callback model.train(), because backpropagation involves updates, it is necessary to obtain the
        latest attention_biases.

        when callback model.eval(), self.ab_cache, it will not change and will save as cache in the instance,

        """
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab_cache = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor):
        """
        X: (b, length, c)
        """
        b, length, dim = x.shape
        training_ab = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab_cache
        qkv = self.qkv(x)
        # qkv: (b, length, num_head, c)
        qkv = qkv.reshape(b, length, self.num_head, -1).permute(0, 2, 1, 3)
        # q and k dim are (b, length, num_head, q_k_dim), v dim is (b, length, num_head, v_dim)
        q, k, v = qkv.split([self.q_k_dim, self.q_k_dim, self.v_dim], dim=3)

        # attn: (b, length, length)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_bias scale alignment
        training_ab = torch.nn.functional.interpolate(training_ab.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')

        attn = attn + training_ab
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, length, self.h_v_dim)
        x = self.proj(x)
        return x


class FFN3D(nn.Module):
    """
    FFN 3D use LN + Linear + GeLU
    """

    def __init__(
            self,
            in_chans: int,
            hidden_chans: int,
            drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super(FFN3D, self).__init__()
        self.fc1 = nn.Linear(in_chans, hidden_chans)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_chans, in_chans)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class MB3D(nn.Module):
    """
    dimension is (B, length, dim) / (B, length, head, dim) 3D/4D,  it follows conventional ViT structure.
    """

    def __init__(
            self,
            dim: int,
            num_head: int,
            q_k_dim: int,
            resolution: int,
            multi_v: float = 4.,
            mlp_ratio: float = 4.,
            qk_scale: t.Optional[float] = None,
            qkv_bias: bool = False,
            drop: float = 0.,
            drop_path: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        super(MB3D, self).__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim, q_k_dim, num_head, multi_v, qk_scale, qkv_bias)
        hidden_chans = int(mlp_ratio * dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FFN3D(dim, hidden_chans, drop, act_layer=act_layer, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )

    def forward(self, x: torch.Tensor):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Flat(nn.Module):
    """
    Flatten the MB4D --> MB3D
    from (B, C, H, W) to (B, H*W, C)
    """

    def __init__(self):
        super(Flat, self).__init__()
        self.flat = nn.Flatten(2)

    def forward(self, x: torch.Tensor):
        return self.flat(x).transpose(1, 2)


class MetaBlock(nn.Module):
    """
    Build MB3D or MB4D based on Index
    """

    def __init__(
            self,
            dim: int,
            index: int,
            depths: t.List[int],
            resolution: int,
            multi_v: float = 4.,
            num_head: int = 8,
            q_k_dim: int = 32,
            pool_size: int = 3,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            drop_ratio: float = 0.,
            drop_path_ratio: float = 0.,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
            vit_num: int = 1,
    ):
        super(MetaBlock, self).__init__()
        blocks = []
        if index == 3 and vit_num == depths[index]:
            blocks.append(Flat())
        for block_idx in range(depths[index]):
            block_dpr = drop_path_ratio * (
                    block_idx + sum(depths[:index])) / (sum(depths) - 1)
            if index == 3 and depths[index] - block_idx <= vit_num:
                blocks.append(MB3D(
                    dim=dim,
                    num_head=num_head,
                    q_k_dim=q_k_dim,
                    resolution=resolution,
                    multi_v=multi_v,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=drop_ratio,
                    drop_path=block_dpr,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value
                ))
            else:
                blocks.append(MB4D(
                    dim=dim,
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_ratio,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    act_layer=act_layer
                ))
                if index == 3 and depths[index] - block_idx - 1 == vit_num:
                    blocks.append(Flat())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class EfficientFormer(nn.Module):
    """

    """

    def __init__(
            self,
            in_chans: int,
            image_size: int,
            depths: t.List[int],
            embed_dims: t.List[int],
            attn_ratios: t.List[float],
            num_head: int,
            pool_size: int,
            q_k_dim: int,
            mlp_ratio: float,
            drop_ratio: float,
            drop_path_ratio: float,
            downsamples: t.List[bool],
            down_patch_size: int,
            down_stride: int,
            use_layer_scale: bool,
            layer_scale_init_value: float,
            vit_num: int,
            norm_layer: str = 'bn',
            act_layer: str = 'gelu',
            num_classes: int = 1000,
            classifier: bool = False,
            **kwargs
    ):
        super(EfficientFormer, self).__init__()

        norm_layer = build_norm(norm_layer)
        act_layer = build_act(act_layer)
        self.classifier = classifier

        resolution = image_size
        # step 1: stem (Pathch Embedding) use small kernel Conv
        self.stem = Stem(in_chans, embed_dims[0])
        resolution //= 4

        backbone = []
        # step 2: MB4D and MB3D stack
        for i in range(len(depths)):
            blocks = MetaBlock(
                dim=embed_dims[i],
                index=i,
                depths=depths,
                resolution=resolution,
                multi_v=attn_ratios[i],
                num_head=num_head,
                q_k_dim=q_k_dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                drop_ratio=drop_ratio,
                drop_path_ratio=drop_path_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            backbone.append(blocks)

            if i >= len(depths) - 1:
                break

            # add PatchMerging
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                backbone.append(PatchMerging(
                    in_chans=embed_dims[i],
                    out_chans=embed_dims[i + 1],
                    kernel_size=down_patch_size,
                    stride=down_stride
                ))
                resolution //= 2

        # step 3: downstream task
        if classifier:
            self.classifier_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(embed_dims[-1], num_classes)
            )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        if self.classifier:
            x = self.classifier_head(x)
        return x


def efficientformer_l1(num_classes: int = 1000, classifier: bool = False, cfg: str = 'EfficientFormer-L1.yaml'):
    """
    EfficientFormer-L1
    """
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['layer_scale_init_value'] = eval('1e-5')
    return EfficientFormer(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


def efficientformer_l3(num_classes: int = 1000, classifier: bool = False, cfg: str = 'EfficientFormer-L3.yaml'):
    """
    EfficientFormer-L1
    """
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['layer_scale_init_value'] = eval('1e-5')
    return EfficientFormer(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


def efficientformer_l7(num_classes: int = 1000, classifier: bool = False, cfg: str = 'EfficientFormer-L7.yaml'):
    """
    EfficientFormer-L1
    """
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['layer_scale_init_value'] = eval('1e-6')
    return EfficientFormer(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


if __name__ == '__main__':
    _x = torch.randn((1, 3, 640, 640)).to(0)
    model = efficientformer_l1(classifier=False).to(0)
    res = model(_x)
    print(res.size())

    from torchsummary import summary
    from thop import profile

    summary(model, (3, 640, 640), batch_size=1)
    flops, params = profile(model, (_x,))
    print(f"FLOPs={str(flops / 1e9)}G")
    print(f"params={str(params / 1e6)}M")

