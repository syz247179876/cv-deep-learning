import contextlib
from collections import OrderedDict
import yaml
import torch
import torch.nn as nn
import typing as t

from augment import args_train


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


class PatchEmbed(nn.Module):
    """
    construct patch embedding, concat cls_token and add position embedding.
    Taking 224x224 size for example, split 224x224 into 14x14 = 196 patches,
    this dim is (B, C, H, W), we should transform it to (B, N, P X P X C),
    actually, it can use conv, its kernel size is 16x16, stride is 16, and obtained
    the patches embedding, then concat the learnable and sharable cls_token and position embedding.
    """

    def __init__(
            self,
            img_size: t.Union[t.List, t.Tuple] = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
    ):
        super(PatchEmbed, self).__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.image_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        # batch, channel, h, w = x.size()
        # assert h == self.image_size[0] and w == self.image_size[1]
        return self.embed(x).flatten(2, -1).transpose(1, 2)


class Attention(nn.Module):
    """
    Attention Module
    q,k,v are liner trans
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

        # use scale to avoid Q @ K.t() too larger so that
        # when passing softmax, the gradient computed in backward is too small
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        batch, num, channel = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[1]
        # 将Q中每一个vector与K中每一个vector计算attention, 进行scale缩放, 避免点积带来的方差影响，
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


class MLP(nn.Module):
    """
    MLP layer, include two FC and one activation
    """

    def __init__(
            self,
            in_channels: t.Optional[int] = None,
            hidden_channels: t.Optional[int] = None,
            out_channels: t.Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.
    ):
        super(MLP, self).__init__()
        hidden_layer = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_layer)
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
    Transformer Block, which include LayerNorm, Residual Block, MSA and MLP
    the dimension of hidden layer is 4x larger than in_channels and out_channels
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            drop_path_ratio: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_ratio = int(mlp_ratio * dim)
        self.mlp = MLP(in_channels=dim, hidden_channels=mlp_hidden_ratio,
                       out_channels=dim, proj_drop=proj_drop_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            img_size: t.Union[t.List, t.Tuple] = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
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
            representation_size: t.Optional[int] = None,
    ):
        super(VisionTransformer, self).__init__()
        self.opts = args_train.opts
        self.nums_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        # cls_token
        self.num_tokens = 1
        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim)).long()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)).long()
        self.pos_drop = nn.Dropout(proj_drop_ratio)

        if self.opts.use_gpu:
            self.cls_token = self.cls_token.to(self.opts.gpu_id)
            self.pos_embed = self.pos_embed.to(self.opts.gpu_id)

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

        # classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: weight init

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
        # extract cls token, according to the whole batch, the whole patches
        return self.pre_logit(x[:, 0])

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
        x = self.head(x)
        return x


class ModelFactory(object):
    """
    Model Factory
    """

    def __init__(self, model_name: str = 'base'):
        assert model_name in ('base', 'huge', 'large'), 'model name should be "base", "huge", "large"'
        with open(rf'vit-{model_name}.yaml', 'r') as m_f:
            config = yaml.safe_load(m_f)
        with contextlib.suppress(NameError):
            for key, value in config.items():
                if isinstance(value, str):
                    if value.startswith('nn'):
                        config[key] = getattr(nn, config.get(key).split('.')[-1])
                    else:
                        config[key] = globals().get(value)
        self.model = VisionTransformer(**config)


if __name__ == '__main__':
    _x = torch.randn(3, 3, 224, 224)
    f = PatchEmbed()
    f(_x)
    # r = f.main(_x)
    # print(r.size(), r)
