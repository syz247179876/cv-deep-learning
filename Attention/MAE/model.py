import argparse
import contextlib

import torch
import torch.nn as nn
import typing as t

import yaml

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
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patches_tuple = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2, -1).transpose(1, 2)

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
            num_masked: int,
            num_patches: int,
    ):
        self.mask_ratio = mask_ratio
        self.num_masked = num_masked
        self.num_patches = num_patches
        self.opts = args_train.opts

    def __call__(self, patches: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        shuffle_idx = torch.rand(len(patches), self.num_patches).argsort()
        if self.opts.use_gpu:
            shuffle_idx = shuffle_idx.to(self.opts.gpu_id)
        mask_idx, unmask_idx = shuffle_idx[:, :self.num_masked], shuffle_idx[:, self.num_masked:]
        # in order to form a complete broadcasting domain
        # batch_id:(b, 1), mask_idx:(b, n_patches)
        batch_id = torch.arange(len(patches)).unsqueeze(dim=1)
        if self.opts.use_gpu:
            batch_id = batch_id.to(self.opts.gpu_id)
        mask_p, unmask_p = patches[batch_id, mask_idx], patches[batch_id, unmask_idx]
        # temp store, they will be used in Decoder
        setattr(self, 'mask_idx', mask_idx)
        setattr(self, 'unmask_idx', unmask_idx)
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
        x = x + self.drop_path(self.multi_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
        self.mask = mask
        self.mask_tokens = mask_tokens
        self.decoder_dim = decoder_dim
        self.dim_trans = nn.Linear(embed_dim, decoder_dim, bias=True) if embed_dim != decoder_dim else nn.Identity()
        self.pos_embed = d_pos_embed
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=decoder_dim, num_heads=num_heads, qkv_bias=False, qk_scale=False,
                             attn_drop_ratio=0., proj_drop_ratio=0.,
                             drop_path_ratio=0., mlp_ratio=mlp_ratio, act_layer=act_layer,
                             norm_layer=norm_layer)
            for _ in range(block_depth)
        ])
        self.norm = norm_layer(decoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_masked = getattr(self.mask, 'num_masked')
        batch_id = getattr(self.mask, 'batch_id')
        shuffle_idx = getattr(self.mask, 'shuffle_idx')

        b, _, embed_dim = x.size()
        # if encoder output dim is not equal to decoder input dim, perform dimension conversion
        x = self.dim_trans(x)

        # add pos embedding
        # note: this operation can be change the memory layout of Tensor,
        # so use a = a + b, instead of a += b
        mask_tokens = self.mask_tokens.expand(b, num_masked, -1)

        # concat shared, learnable mask_tokens with encoded unmask patches without cls token
        concat_tokens = torch.cat((mask_tokens, x[:, 1:, :]), dim=1)
        # un-shuffle to ensure that the order corresponds
        dec_input_tokens = torch.empty_like(concat_tokens)
        dec_input_tokens[batch_id, shuffle_idx] = concat_tokens

        # (b, mask_num, decoder_dim), add position embedding to mask patches and add cls token
        # note: position embedding should correspond directly to each patch position
        x = torch.cat((x[:, :1, :], dec_input_tokens), dim=1)
        x = x + self.pos_embed

        # decoder is equivalent to stacking multiple transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x


class MAE(nn.Module):

    def __init__(
            self,
            img_size: t.Tuple = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            decoder_dim: int = 768,
            num_classes: int = 1000,
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        num_patches = int(img_size[0] * img_size[1] / (patch_size * patch_size))
        num_masked = int(num_patches * self.opts.mask_ratio)
        self.mask = mask or Mask(self.opts.mask_ratio, num_masked, num_patches)
        self.patch_embed = embed_layer or PatchEmbed(img_size, (patch_size, patch_size),
                                                     in_chans, embed_dim=embed_dim, mode=mode)

        # self.encoder_block = encoder_block or TransformerBlock(embed_dim, num_heads, qkv_bias, qk_scale,
        #                                                        attn_drop_ratio,
        #                                                        proj_drop_ratio, drop_path_ratio, mlp_ratio, act_layer,
        #                                                        norm_layer)
        self.cls_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, self.cls_tokens, embed_dim, device=self.opts.gpu_id))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_tokens, embed_dim,
                                                  device=self.opts.gpu_id), requires_grad=False)
        self.pos_drop = nn.Dropout(proj_drop_ratio)

        # if self.opts.use_gpu:
        #     self.cls_token = self.cls_token.to(self.opts.gpu_id)
        #     self.pos_embed = self.pos_embed.to(self.opts.gpu_id)

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
        # note: the type of mask_tokens should be int64, as needed to embed by nn.Embedding
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_dim, device=self.opts.gpu_id))
        # d_pos_embed = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim,
                                                          device=self.opts.gpu_id), requires_grad=False)
        # if self.opts.use_gpu:
        #     self.mask_tokens = self.mask_tokens.to(self.opts.gpu_id)
        self.decoder = decoder or Decoder(self.opts, embed_dim, decoder_dim, self.mask_tokens, self.decoder_pos_embed,
                                          self.mask, decoder_num_heads, decoder_block_depth, mlp_ratio,
                                          act_layer, norm_layer)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans, bias=True)

    def forward_encoder(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        transformer encoder
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        mask_patches, unmask_patches = self.mask(x)
        b, _, _ = unmask_patches.size()
        # extra add position embedding to cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(b, -1, -1)
        unmask_tokens = torch.cat((cls_token, unmask_patches), dim=1)
        unmask_tokens = self.pos_drop(unmask_tokens)
        unmask_tokens = self.blocks(unmask_tokens)
        unmask_tokens = self.norm(unmask_tokens)
        return unmask_tokens, mask_patches

    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        transformer decoder
        """
        # (b, num, dim)
        x = self.decoder(x)
        # predictor head
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        # retrieve masked tokens after decoder, (b, mask_num, decoder_dim)
        masked_tokens = x[getattr(self.mask, 'batch_id'), getattr(self.mask, 'mask_idx'), :]
        return masked_tokens

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, ...]:
        """
        *shuffle and un-shuffle
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
        unmask_encoder_tokens, mask_patches = self.forward_encoder(x)
        mask_decoder_tokens = self.forward_decoder(unmask_encoder_tokens)
        return mask_decoder_tokens, mask_patches

    def forward_test(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, t.Dict]:
        """
        Return additional data on top of forward
        """
        mask_decoder_tokens, mask_patches = self.forward(x)
        patch_embed = self.patch_embed(x)
        return mask_decoder_tokens, mask_patches, {
            'patches': patch_embed,
            'batch_id': getattr(self.mask, 'batch_id'),
            'mask_idx': getattr(self.mask, 'mask_idx'),
            'shuffle_idx': getattr(self.mask, 'shuffle_idx'),
            'patch_size': self.patch_size,
            'img_size': self.img_size,
            'in_chans': self.in_chans,
        }


class ModelFactory(object):
    """
    Model Factory
    """

    def __init__(self, model_name: str = 'base'):
        assert model_name in ('base', 'large', 'huge'), 'model name should be "base"'
        with open(rf'mae-{model_name}.yaml', 'r') as m_f:
            config = yaml.safe_load(m_f)
        with contextlib.suppress(NameError):
            for key, value in config.items():
                if isinstance(value, str):
                    if value.startswith('nn'):
                        config[key] = getattr(nn, config.get(key).split('.')[-1])
                    else:
                        config[key] = globals().get(value, value)
        self.model = MAE(**config)


if __name__ == '__main__':
    _x = torch.randn((2, 3, 224, 224))
    _x = _x.to(0)
    # f = PatchEmbed(mode='split')
    # res = f(_x)
    # print(res)
    # f2 = PatchEmbed(mode='split')
    # res2 = f2(_x)
    # print(res2)
    model = ModelFactory(model_name='large').model
    model = model.to(0)
    pred, label = model(_x)
    print(pred.size(), label.size(), nn.MSELoss(reduction='mean')(pred, label))
    ss = model.state_dict()
    s = 8
