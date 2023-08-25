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
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
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


class Mask(object):
    """
    Different ratio of Mask strategy.
    Split patches into mask patches and unmask patches
    """

    def __init__(
            self,
            mask_ratio: float,
            num_patches: int,
    ):
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.opts = args_train.opts

    def __call__(self, patches: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        num_masked = int(self.num_patches * self.mask_ratio)
        shuffle_idx = torch.rand(patches, self.num_patches).argsort()
        mask_idx, unmask_idx = shuffle_idx[:, :num_masked], shuffle_idx[:, num_masked:]
        # in order to form a complete broadcasting domain
        # batch_id @ mask_idx or batch_id @ unmask_idx should be feasible
        batch_id = torch.arange(3).unsqueeze(dim=1)
        mask_p, unmask_p = patches[batch_id, mask_idx], patches[batch_id, unmask_idx]
        return mask_p, unmask_p


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
        qkv = self.qkv(x).reshape(batch, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
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


class EncoderBlock(nn.Module):
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
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.multi_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, act_layer, proj_drop_ratio)

    def forward(self, x: torch.Tensor):
        x += self.drop_path(self.multi_attn(self.norm1(x)))
        x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class Decode(nn.Module):
    pass


class MAE(nn.Module):

    def __init__(
            self,
            img_size: t.Tuple = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            block_depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            drop_path_ratio: float = 0.,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            mlp_ratio: float = 4.,
            embed_layer: t.Optional[t.Callable] = None,
            mask: t.Optional[t.Callable] = None,
            encoder_block: t.Optional[t.Callable] = None,
            act_layer: t.Optional[nn.Module] = None,
            norm_layer: t.Optional[nn.Module] = None,
            mode: str = 'split',

    ):
        super(MAE, self).__init__()
        self.opts = args_train.opts
        self.mask = mask or Mask(self.opts.mask_ratio, patch_size * patch_size)
        self.patch_embed = embed_layer or PatchEmbed(img_size, (patch_size, patch_size), in_chans, mode=mode)

        self.encoder_block = encoder_block or EncoderBlock(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio,
                                                           proj_drop_ratio, drop_path_ratio, mlp_ratio, act_layer,
                                                           norm_layer)
        self.num_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim)).long()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(proj_drop_ratio)

        if self.opts.use_gpu:
            self.cls_token = self.cls_token.to(self.opts.gpu_id)
            self.pos_embed = self.pos_embed.to(self.opts.gpu_id)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, block_depth)]
        self.blocks = nn.Sequential(*[
            EncoderBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                         drop_path_ratio=dpr[i], mlp_ratio=mlp_ratio, act_layer=act_layer,
                         norm_layer=norm_layer)
            for i in range(block_depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward_encoder(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        transformer encoder
        """
        patches = self.patch_embed(x)
        mask_patches, unmask_patches = self.mask(patches)
        b, _, _ = unmask_patches.size()
        cls_token = self.cls_token.expand(b, -1, -1)
        unmask_patches = torch.cat((cls_token, unmask_patches), dim=1)
        unmask_patches = self.pos_drop(unmask_patches + self.pos_embed)
        unmask_patches = self.blocks(unmask_patches)
        unmask_patches = self.norm(unmask_patches)
        return unmask_patches, mask_patches

    def forward_decoder(self, unmask_patches: torch.Tensor, mask_patches: torch.Tensor):
        """
        transformer decoder
        """
        pass

    def forward(self, x: torch.Tensor):
        unmask_patches, mask_patches = self.forward_encoder(x)


if __name__ == '__main__':
    _x = torch.randn((2, 3, 224, 224))
    patch = PatchEmbed(mode='split')
    res = patch(_x)
    print(res.size(), res)
