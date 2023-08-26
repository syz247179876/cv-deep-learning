import argparse

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
        shuffle_idx = torch.rand(len(patches[0]), self.num_patches).argsort()
        mask_idx, unmask_idx = shuffle_idx[:, :num_masked], shuffle_idx[:, num_masked:]
        # in order to form a complete broadcasting domain
        # batch_id:(b, 1), mask_idx:(b, n_patches)
        batch_id = torch.arange(len(patches[0])).unsqueeze(dim=1)
        mask_p, unmask_p = patches[batch_id, mask_idx], patches[batch_id, unmask_idx]

        # temp store, they will be used in Decoder
        setattr(self, 'mask_idx', mask_idx)
        setattr(self, 'num_masked', num_masked)
        setattr(self, 'batch_id', batch_id)
        setattr(self, 'shuffle_idx', shuffle_idx)
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
        x += self.drop_path(self.multi_attn(self.norm1(x)))
        x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            opts: argparse.Namespace,
            embed_dim: int,
            decoder_dim: int,
            mask_tokens: torch.Tensor,
            d_pos_embed: nn.Embedding,
            mask: Mask,
            num_heads: int,
            block_depth: int,
            mlp_ratio: float,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super(Decoder, self).__init__()
        self.mask_tokens = mask_tokens
        self.decoder_dim = decoder_dim
        self.dim_trans = nn.Linear(embed_dim, decoder_dim) if embed_dim != decoder_dim else nn.Identity()
        self.pos_embed = d_pos_embed
        self.mask_idx = getattr(mask, 'mask_idx')
        self.num_masked = getattr(mask, 'num_masked')
        self.batch_id = getattr(mask, 'batch_id')
        self.shuffle_idx = getattr(mask, 'shuffle_idx')
        self.use_gpu = opts.use_gpu
        self.gpu_id = opts.gpu_id
        self.decoder = nn.Sequential(*[
            TransformerBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=False, qk_scale=False,
                             attn_drop_ratio=0., proj_drop_ratio=0.,
                             drop_path_ratio=0., mlp_ratio=mlp_ratio, act_layer=act_layer,
                             norm_layer=norm_layer)
            for _ in range(block_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, embed_dim = x.size()
        # if encoder output dim is not equal to decoder input dim, perform dimension conversion
        x = self.dim_trans(x)
        # add pos embedding
        mask_tokens = self.mask_tokens.expand(b, self.num_masked, -1)
        # (b, mask_num, decoder_dim)
        mask_tokens += self.pos_embed(self.mask_idx)

        # concat shared, learnable mask_tokens with encoded unmask patches
        concat_tokens = torch.cat((mask_tokens, x), dim=1)
        # un-shuffle to ensure that the order corresponds
        dec_input_tokens = torch.empty_like(concat_tokens)
        if self.use_gpu:
            dec_input_tokens = dec_input_tokens.to(self.gpu_id)
        dec_input_tokens[self.batch_id, self.shuffle_idx] = concat_tokens

        # decoder is equivalent to stacking multiple transformer blocks
        decoder_tokens = self.decoder(dec_input_tokens)
        return decoder_tokens


class MAE(nn.Module):

    def __init__(
            self,
            img_size: t.Tuple = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            decoder_dim: int = 768,
            block_depth: int = 12,
            decoder_block_depth: int = 12,
            num_heads: int = 12,
            decoder_num_heads: int = 12,
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
            decoder: t.Optional[nn.Module] = None,
            mode: str = 'split',

    ):
        super(MAE, self).__init__()
        self.opts = args_train.opts
        self.mask = mask or Mask(self.opts.mask_ratio, patch_size * patch_size)
        self.patch_embed = embed_layer or PatchEmbed(img_size, (patch_size, patch_size), in_chans, mode=mode)

        self.encoder_block = encoder_block or TransformerBlock(embed_dim, num_heads, qkv_bias, qk_scale,
                                                               attn_drop_ratio,
                                                               proj_drop_ratio, drop_path_ratio, mlp_ratio, act_layer,
                                                               norm_layer)
        self.cls_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, self.cls_tokens, embed_dim)).long()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.cls_tokens, embed_dim))
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

        # Decoder

        # mask token that is shared and learned
        mask_tokens = nn.Parameter(torch.rand(1, 1, decoder_dim))
        d_pos_embed = nn.Embedding(self.patch_embed.num_patches, decoder_dim)
        self.decoder = decoder or Decoder(self.opts, embed_dim, decoder_dim, mask_tokens, d_pos_embed,
                                          self.mask, decoder_num_heads, decoder_block_depth, mlp_ratio,
                                          act_layer, norm_layer)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

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

    def forward_decoder(self, unmask_encoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        transformer decoder
        """
        # (b, num, dim)
        decoded_tokens = self.decoder(unmask_encoder_tokens)
        # retrieve masked tokens after decoder, (b, mask_num, decoder_dim)
        masked_tokens = decoded_tokens[getattr(self, 'batch_id'), getattr(self, 'mask_id'), :]
        return masked_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder:
            1.Using standard ViT to encoder, applied only on visible, shuffle randomly
            unmasked patches, about 75% unmasked patches.
            2.the output of my Encoder Module include two part.
                (1) encoded unmasked patches (25%)
                (2) uncoded masked patches (75%)
        Decoder:
            note: Downstream Tasks
            1.The input of MAE decoder is the full set of tokens consisting of encoded visible patches and mask tokens.
            2.Each mask token is a shared, learned vector that use to predict,
            mask token add position embedding, un-shuffle and feed to Decoder.
            3.The Decoder also consist of series of Transformer Block.
        """
        unmask_encoder_tokens, mask_encoder_tokens = self.forward_encoder(x)
        mask_decoder_tokens = self.forward_decoder(unmask_encoder_tokens)
        mask_decoder_tokens = self.head(mask_decoder_tokens)
        return mask_decoder_tokens


if __name__ == '__main__':
    _x = torch.randn((2, 3, 224, 224))
    patch = PatchEmbed(mode='split')
    res = patch(_x)
    print(res.size(), res)
