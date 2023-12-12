import itertools
import torch
import torch.nn as nn
import typing as t
import yaml
from einops import rearrange
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SEModule
from other_utils.utils import auto_pad

# Official source code: https://github.com/microsoft/Cream/blob/main/EfficientViT/classification/model/efficientvit.py

__all__ = ['efficientvit', ]


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
    # binarize, get one or zero
    random_tensor = torch.floor(random_tensor)
    # enhance feature of samples that skip residual block
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


class Conv2dBN(torch.nn.Sequential):
    """
    Conv + BN module, providing fuse function, in order to fuse Conv and BN
    """

    def __init__(self, a, b, ks=1, stride=1, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, auto_pad(ks, d=dilation), dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
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


class BNLinear(torch.nn.Sequential):
    """
    BN + Linear module, providing fuse function, in order to fuse Conv and Linear
    """

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Conv(nn.Module):
    """
    Hardswish act + BN
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
            norm_layer: t.Optional[t.Callable] = None,
            act_layer: t.Optional[t.Callable] = None,
    ):
        super(Conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.Hardswish

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
        return self.act(self.bn(self.conv(x)))


class PatchEmbed(nn.Module):
    """
    Overlap PatchEmbedding, use small kernel instead of large kernel.
    """

    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            resolution: int,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(PatchEmbed, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.patch_embed = nn.Sequential(
            Conv2dBN(in_chans, embed_dim // 8, ks=3, stride=2, resolution=resolution),
            act_layer(inplace=True),
            Conv2dBN(embed_dim // 8, embed_dim // 4, ks=3, stride=2, resolution=resolution // 2),
            act_layer(inplace=True),
            Conv2dBN(embed_dim // 4, embed_dim // 2, ks=3, stride=2, resolution=resolution // 4),
            act_layer(inplace=True),
            Conv2dBN(embed_dim // 2, embed_dim, ks=3, stride=2, resolution=resolution // 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x


class Residual(nn.Module):
    """
    Residual Block with DropPath, if training OR dropout = 0., close DropPath
    """

    def __init__(self, module: t.Callable, dropout: float = 0.):
        super(Residual, self).__init__()
        self.module = module
        self.drop_path = DropPath(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.module(x))


class FFN(nn.Module):
    """
    Feed Forward Network
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            resolution: int,
            act_layer: t.Optional[t.Callable] = None,
    ):
        super(FFN, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.fc1 = Conv2dBN(embed_dim, hidden_dim, ks=1, resolution=resolution)
        self.act = act_layer(inplace=True)
        self.fc2 = Conv2dBN(hidden_dim, embed_dim, ks=1, resolution=resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(self.act(self.fc1(x)))
        return x


class CascadedGroupAttention(nn.Module):
    """
    CGA module(Cascaded Group Attention)

    Many heads learn similar projections of the same full feature and incur computation redundancy, so feeding
    each head with only a split of the full feature to effectively mitigate attention computation redundancy.

    Its cascading nature refers to the addition of the feature map calculated from the previous head to the input of
    the following head, In order to gradually learn feature maps, so that the last few layers can see all head
    levels and make comprehensive decisions.

    """

    def __init__(
            self,
            embed_dim: int,
            q_k_dim: int,
            num_head: int,
            multi_v: float,
            resolution: int,
            kernels: t.List,
            need_attn_offset: bool = True,
            scale: t.Optional[float] = None,
            act_layer: t.Optional[t.Callable] = None,
    ):
        """
        multi_v means multiplier for the query dim for value dimension.
        V prefers a relative large channels, being close to the input embedding dimension.

        need_attn_offset: Do we need to increase bias on attention weights?
        """
        super(CascadedGroupAttention, self).__init__()
        if act_layer is None:
            act_layer = nn.ReLU
        self.embed_dim = embed_dim
        self.q_k_dim = q_k_dim
        self.scale = q_k_dim ** -0.5 if scale is None else scale
        self.v_dim = int(multi_v * q_k_dim)
        self.num_head = num_head
        self.multi_v = multi_v
        self.kernels = kernels
        self.need_attn_offset = need_attn_offset

        qkv_list: t.List[nn.Module] = []
        dwc_list: t.List[nn.Module] = []

        # the projection and token interaction (DWC) for each head
        for i in range(num_head):
            qkv_list.append(Conv2dBN(embed_dim // num_head, q_k_dim * 2 + self.v_dim, resolution=resolution))
            dwc_list.append(Conv2dBN(q_k_dim, q_k_dim, kernels[i], groups=q_k_dim, resolution=resolution))
        self.qkv_list = nn.ModuleList(qkv_list)
        self.dwc_list = nn.ModuleList(dwc_list)

        # add learnable attention offset to enable the model to learn more flexibly how to allocate attention
        # Equivalent to adding attention operations to the attention matrix, Has it been done multiple times?
        # TODO: need to do Ablation

        # construct the offset of each point relative to all other points
        if need_attn_offset:
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
        self.proj = nn.Sequential(
            act_layer(inplace=True),
            Conv2dBN(self.v_dim * num_head, embed_dim, bn_weight_init=0, resolution=resolution)
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (B, C, H, W)
        """
        b, c, h, w = x.shape

        training_ab = self.attention_biases[:, self.attention_bias_idxs]

        # divide features into head_num overlapping blocks
        # (B, (q_k_dim * 2 + self.v_dim)/h, H, W)
        feature_overlap = x.chunk(len(self.qkv_list), dim=1)
        feature_outs = []

        feat = feature_overlap[0]
        # Cascaded ops
        for i, qkv in enumerate(self.qkv_list):
            # add last output into next input
            if i > 0:
                feat = feat + feature_overlap[i]
            # do projection
            feat = qkv(feat)
            # q: (B, q_k_dim/h, H, W), k: (B, q_k_dim/h, H, W), v: (B, v_dim/h, H, W)
            q, k, v = feat.reshape(b, -1, h, w).split([self.q_k_dim, self.q_k_dim, self.v_dim], dim=1)

            # make DWC to q, (B, q_k_dim/h, H, W)
            q = self.dwc_list[i](q)

            # q: (B, q_k_dim/h, N), k: (B, q_k_dim/h, N), v: (B, v_dim/h, N)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

            # compute attention score, (B, N, q_k_dim/h) @ (B, q_k_dim/h, N) ==> (B, N, N)
            if self.need_attn_offset:
                attn = (q.transpose(-2, -1) @ k) * self.scale + training_ab[i] if self.training else self.ab_cache[i]
            else:
                attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = attn.softmax(dim=-1)
            # (B, v_dim/h, N) @ (B, N, N) ==> (B, v_dim/h, N) ==> (B, v_dim/h, H, W)
            if isinstance(v, (torch.HalfTensor, torch.cuda.HalfTensor)):
                attn = attn.half()
            feat = (v @ attn.transpose(-2, -1)).reshape(b, self.v_dim, h, w)
            feature_outs.append(feat)
        # (B, embed_dim, H, W)
        x = self.proj(torch.cat(feature_outs, dim=1))
        return x


class LocalWindowAttention(nn.Module):
    """
    To reduce computational complexity, the entire feature map is divided into multiple small windows for interaction
    """

    def __init__(
            self,
            embed_dim: int,
            q_k_dim: int,
            num_head: int,
            multi_v: float,
            resolution: int,
            window_resolution: int,
            kernels: t.List,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(LocalWindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_head
        self.resolution = resolution
        assert window_resolution > 0, 'window size must be greater than 0'
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            embed_dim,
            q_k_dim,
            num_head,
            multi_v,
            window_resolution,
            kernels,
            act_layer=act_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = w = self.resolution
        # in detection model, in validation, maybe h_ != h, w_ != w
        b, c, h_, w_ = x.shape

        # if h_ != h, w_ != w, padding

        if h <= self.window_resolution and w <= self.window_resolution:
            x = self.attn(x)
        else:
            # (B, C, H, W) -> (B, H, W, C)
            x = x.permute(0, 2, 3, 1).contiguous()
            pad_b = (self.window_resolution - h_ %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - w_ %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                # fill reversed, from C -> W -> H
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = h_ + pad_b, w_ + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            #  window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = rearrange(x, 'b (nH h) (nW w) c -> (b nH nW) c h w', nH=nH, nW=nW)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = rearrange(x, '(b nH nW) c h w -> b (nH h) (nW w) c', nH=nH, nW=nW)

            if padding:
                x = x[:, :h_, :w_].contiguous()
            # (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class EfficientViTBlock(nn.Module):
    """
    The implementation of EfficientViTBlock
    note: The Q,K dimensions are largely trimmed, because the redundancy is Q, K is much larger than V when they have
    same dimensions.
    """

    def __init__(
            self,
            embed_dim: int,
            stage_mode: str,
            q_k_dim: int,
            num_head: int,
            window_resolution: int,
            resolution: int,
            kernels: t.List,
            multi_v: float,
            hidden_ratio: int = 2,
            act_layer: t.Optional[t.Callable] = None
    ):
        """
        multi_v means multiplier for the query dim for value dimension.
        V prefers a relative large channels, being close to the input embedding dimension.

        note:
        1.hidden_ration reduce from 4 to 2, it is different from other attention mechanisms here
        """
        super(EfficientViTBlock, self).__init__()

        hidden_dim = int(embed_dim * hidden_ratio)

        # (DWC + FFN)
        self.dwc0 = Residual(Conv2dBN(embed_dim, embed_dim, 3, 1, groups=embed_dim,
                                      bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(embed_dim, hidden_dim, resolution, act_layer=act_layer))

        # (CGA)
        assert stage_mode in ['window_attention', ], f'not support attention module {stage_mode}'
        if stage_mode == 'window_attention':
            self.mixer = Residual(LocalWindowAttention(embed_dim, q_k_dim, num_head, multi_v,
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels, act_layer=act_layer))

        # (DWC + FFN)
        self.dwc1 = Residual(Conv2dBN(embed_dim, embed_dim, 3, 1, groups=embed_dim,
                                      bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(embed_dim, hidden_dim, resolution, act_layer=act_layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn0(self.dwc0(x))
        x = self.mixer(x)
        x = self.ffn1(self.dwc1(x))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging operation

    PatchMerging use sandwich layout, and inverted residual block proposed in MobileNetV2, which reduce information
    loss during downsampling by increasing the number of channels
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            old_resolution: int,
            new_resolution: int,
            hidden_ratio: int = 2,
            act_layer: t.Optional[t.Callable] = None
    ):
        super(PatchMerging, self).__init__()

        ffn0_hidden_chans = int(hidden_ratio * in_chans)
        ffn1_hidden_chans = int(hidden_ratio * out_chans)
        hidden_chans = int(in_chans * 4)

        if act_layer is None:
            act_layer = nn.ReLU

        # (DWC + FFN)
        self.dwc0 = Residual(Conv2dBN(in_chans, in_chans, 3, 1, groups=in_chans, resolution=old_resolution))
        self.ffn0 = Residual(FFN(in_chans, ffn0_hidden_chans, resolution=old_resolution, act_layer=act_layer))

        # (down-sampling), inverted residual block
        self.down_sampling = nn.Sequential(
            Conv2dBN(in_chans, hidden_chans, resolution=old_resolution),
            act_layer(inplace=True),
            Conv2dBN(hidden_chans, hidden_chans, 3, 2, groups=hidden_chans, resolution=old_resolution),
            SEModule(hidden_chans, act_layer=act_layer),
            Conv2dBN(hidden_chans, out_chans, resolution=new_resolution),
        )

        # (DWC + FFN)
        self.dwc1 = Residual(Conv2dBN(out_chans, out_chans, 3, 1, groups=out_chans, resolution=new_resolution))
        self.ffn1 = Residual(FFN(out_chans, ffn1_hidden_chans, resolution=new_resolution, act_layer=act_layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn0(self.dwc0(x))
        x = self.down_sampling(x)
        x = self.ffn1(self.dwc1(x))
        return x


class EfficientViT(nn.Module):
    """

    EfficientViT: A Light-weight, General-purpose Vision Transformer, Proposed Cascaded Group Attention(CGA)

    From the perspective of redundancy, it analyzes three main factors that affect model inference speed.

    1.memory access
        - subnetworks with 20% ~ 40% MHSA layers tend to get better accuracy and speed, so reduce the number of
        layers in MHSA, use sandwich layout that employs less memory-bound self-attention layers and more
        memory-efficient FFN layers for channel communication.

    2.computation redundancy
        - many heads learn similar projections of the same full feature and incur computation redundancy, so feeding
        each head with only a split of the full feature to effectively mitigate attention computation redundancy.

    3.parameter usage
        - Because there is redundant information, which is mainly reflected by channels, so it need pruning, the
        pruning method removes unimportant channels under a certain resource constraint and keeps the most critical ones
        to best preserve the accuracy. It uses the multiplication of gradient and weight as channel importance.

    note:
    1.Usually, in many papers, which combined CNN and Transformer, such as BiFormer, MobileViT, EfficientViT,
    some of them use DWConv(BiFormer, EfficientViT) or regular convolution(MobileViT).

    2. In EfficientViT, it uses DWConv(called TokenInteraction in EfficientViT), which is added before MHSA to compensate
    for the lack of CNN inductive bias of the local structural information, DWC + FFN like depth-wise separable
    convolution to learn local information.


    EfficientViT has follow components:
    1. Overlap PatchEmbedding, consisting of four pair of Conv + BN, perform CNN first, reduce the resolution
    to a certain level, and then perform self attention operation.

    2. Three EfficientViT block with two EfficientViT Subsample sandwiched in the middle
        EfficientViT block is composed of memory-efficient sandwich layout, (DWC + FFN) -- (CGA) -- (DWC + FFN).

        - The first pair (DWC + FFN) introduced inductive bias of the local structural information to
        enhance model capability.

        - The middle module CGA(Cascaded Group Attention), it feeds each head with different splits of the full features.
        its cascading nature refers to the addition of the feature map calculated from the previous head to the input of
         the following head, In order to gradually learn feature maps, so that the last few layers can see all head
        levels and make comprehensive decisions.
            note: CGA used WindowAttention, in order to reduce computation

        - The last pair (DWC + FFN) used for inter channel interaction of feature maps that calculated in CGA.


    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            in_chans: int,
            stages_mode: t.List[str],
            embed_dims: t.List[int],
            key_dims: t.List[int],
            depths: t.List[int],
            num_heads: t.List[int],
            window_sizes: t.List[int],
            kernels: t.List[int],
            down_ops: t.List[t.Union[str, int]],
            hidden_ratios: t.List[int],
            num_classes: int = 1000,
            classifier: bool = False,
            **kwargs
    ):
        super(EfficientViT, self).__init__()

        resolution = image_size

        # step1: Patch Embedding
        self.patch_embed = PatchEmbed(in_chans, embed_dims[0], resolution)

        resolution = image_size // patch_size
        # attn_ratio means V is a multiple of Q and K on the channel
        attn_ratios = [embed_dims[i] / (key_dims[i] * num_heads[i]) for i in range(len(embed_dims))]
        # step2: Three EfficientViT and Two EfficientViT Subsample stack
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        for i, (stage_mode, embed_dim, depth, num_head, window_size, key_dim, hidden_ratio) in enumerate(
                zip(stages_mode, embed_dims, depths, num_heads, window_sizes, key_dims, hidden_ratios)
        ):
            for _ in range(depth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(
                    stage_mode=stage_mode,
                    embed_dim=embed_dim,
                    q_k_dim=key_dim,
                    num_head=num_head,
                    multi_v=attn_ratios[i],
                    resolution=resolution,
                    window_resolution=window_size,
                    kernels=kernels,
                    hidden_ratio=hidden_ratio
                ))

            if i < 2 and down_ops[0] == 'sandwich':
                block = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // down_ops[1] + 1
                block.append(PatchMerging(embed_dims[i], embed_dims[i + 1], resolution, resolution_, hidden_ratio))
                resolution = resolution_
        self.blocks1 = nn.Sequential(*self.blocks1)
        self.blocks2 = nn.Sequential(*self.blocks2)
        self.blocks3 = nn.Sequential(*self.blocks3)

        self.classifier = classifier

        if classifier:
            self.classifier_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(embed_dims[-1], num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        if self.classifier:
            x = self.classifier_head(x)
        return x


def efficientvit(num_classes: int = 1000, classifier: bool = False, cfg: str = 'EfficientViT-M0.yaml'):
    """
    EfficientViT-M0~M5
    """
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    return EfficientViT(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


if __name__ == '__main__':
    _x = torch.randn((1, 3, 224, 224)).to(0)
    model = efficientvit(classifier=False, cfg='EfficientViT-M4.yaml').to(0)
    # res = model(_x)
    # print(res.size())

    from torchsummary import summary

    summary(model, (3, 224, 224), batch_size=1)
