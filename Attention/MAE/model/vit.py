import typing as t
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from Attention.MAE.augment import args_train


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


class PatchEmbed(nn.Module):
    """
    Split the whole image into different patches.
    Here implement two function, conv and split
    """

    def __init__(
            self,
            image_size: t.Tuple[int, int] = (224, 224),
            patch_size: t.Tuple[int, int] = (16, 16),
            in_chans: int = 3,
            embed_dim: int = 768,
            mode: str = 'conv'
    ):
        super(PatchEmbed, self).__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        if mode == 'conv':
            self.conv_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patches_tuple = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_embed(x).flatten(2, -1).transpose(1, 2)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        patches = x.view(b, c, self.patches_tuple[0], self.patch_size[0], self.patches_tuple[1], self.patch_size[1])
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        func = getattr(self, f'forward_{self.mode}')
        return func(x)


class Attention(nn.Module):
    """
    Attention Module --- multi-head self-attention
    The attention in MAE based on standard ViT, while the encoder in MAE only
    operates on a small subset (about 25%) of full patches, called unmasked patches.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num, channel = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        b = (attn @ v).transpose(1, 2).reshape(batch, num, channel)
        b = self.proj(b)
        b = self.proj_drop(b)
        return b


class MLP(nn.Module):

    def __init__(
            self,
            in_chans: t.Optional[int] = None,
            hidden_chans: t.Optional[int] = None,
            out_chans: t.Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.
    ):
        super(MLP, self).__init__()
        hidden_layer = hidden_chans or in_chans
        out_channels = out_chans or in_chans
        self.fc1 = nn.Linear(in_chans, hidden_layer)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_layer, out_channels)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    ViT Encoder Block
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            drop_path_ratio: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.multi_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, act_layer, proj_drop_ratio)

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.multi_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            img_size: t.Union[t.List, t.Tuple] = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: t.Optional[int] = None,
            embed_dim: int = 768,
            block_depth: int = 12,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            drop_path_ratio: float = 0.,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            embed_layer: nn.Module = PatchEmbed,
            norm_layer: nn.Module = nn.LayerNorm,
            pos_embed_mode: str = 'cosine',
            representation_size: t.Optional[int] = None,
            classification: t.Optional[bool] = False,
            pool: str = 'cls',
    ):
        super(VisionTransformer, self).__init__()
        self.opts = args_train.opts
        self.nums_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        # cls_token
        self.cls_tokens = 1
        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.cls_tokens, embed_dim))
        if pos_embed_mode == 'cosine':
            # TODO: sin-cos embedding init
            assert 'need to implement sin-cos position embedding'
        else:
            # learnable position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_tokens, embed_dim))
        self.pos_drop = nn.Dropout(proj_drop_ratio)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, block_depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                             drop_path_ratio=dpr[i], mlp_ratio=mlp_ratio, act_layer=act_layer,
                             norm_layer=norm_layer)
            for i in range(block_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # if pre-training, such as ImageNet21k, use Linear + tanh activation + Linear
        # if train in custom dataset, use one Linear is ok

        if representation_size:
            self.has_logit = True
            self.num_features = representation_size
            self.pre_logit = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('activation', nn.Tanh())
            ]))
        else:
            self.has_logit = False
            self.pre_logit = nn.Identity()

        self.pool = pool
        self.classification = classification
        # classifier head
        if classification:
            self.class_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_vit_weights(self, module):
        """
        ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        generate patch embedding, concat class token and add position embedding,
        pass block_depth layer of transformer block and one layer norm.
        finally, extract class token
        """
        x = self.patch_embed(x)
        b, _, _ = x.size()
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.concat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor):
        """
        1.embedding, including patch embedding, cls token, position embedding
        2.Transformer Encoder:
            (1).multi-head self-attention(Encoder Block)
            (2).MLP Block
            (3).add residual to (1) and (2)
        3.head layer
        """
        # step 1 and 2
        x = self.forward_feature(x)
        # like GAP in CNN
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]
        else:
            raise ValueError('pool must be "mean" or "cls"')
        if self.classification:
            x = self.head(x)
        return x
