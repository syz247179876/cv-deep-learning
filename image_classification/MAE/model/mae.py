import contextlib

import torch
import torch.nn as nn
import typing as t

import yaml

from image_classification.MAE.augment import args_train
from image_classification.MAE.util import init_weights
from image_classification.MAE.utils.pos_embed import get_2d_sincos_pos_embed
from .vit import PatchEmbed, TransformerBlock


class Mask(object):
    """
    Different ratio of Mask strategy.
    Split patches into mask patches and unmask patches
    """

    def __init__(
            self,
            mask_ratio: float,
    ):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [b, num, dim], sequence
        """
        b, num, dim = x.size()
        len_keep = int(num * self.mask_ratio)

        # rand noise
        noise = torch.rand(b, num, device=x.device)
        idx_shuffle = torch.argsort(noise, dim=1)
        idx_restore = torch.argsort(idx_shuffle, dim=1)

        # unmask patches
        idx_unmask = idx_shuffle[:, :len_keep]
        unmask_patches = torch.gather(x, dim=1, index=idx_unmask.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, num], device=x.device)
        mask[:, :len_keep] = 0
        # random again
        mask = torch.gather(mask, dim=1, index=idx_restore)
        return unmask_patches, mask, idx_restore


class MAE(nn.Module):

    def __init__(
            self,
            img_size: t.Tuple = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 1024,
            decoder_embed_dim: int = 512,
            num_heads: int = 16,
            decoder_num_heads: int = 16,
            block_depth: int = 24,
            decoder_block_depth: int = 8,
            qkv_bias: bool = False,
            qk_scale: t.Optional[float] = None,
            drop_path_ratio: float = 0.,
            attn_drop_ratio: float = 0.,
            proj_drop_ratio: float = 0.,
            mlp_ratio: float = 4.,
            mask_ratio: float = 0.75,
            embed_layer: t.Optional[t.Callable] = None,
            mask: t.Optional[t.Callable] = None,
            act_layer: t.Optional[nn.Module] = None,
            norm_layer: t.Optional[nn.Module] = None,
            mode: str = 'split',
    ):
        super(MAE, self).__init__()
        self.opts = args_train.opts
        self.patch_embed = embed_layer or PatchEmbed(img_size, (patch_size, patch_size), in_chans, mode=mode)
        num_patches = self.patch_embed.num_patches
        self.mask = mask or Mask(mask_ratio)
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.Sequential(*[
            TransformerBlock(dim=decoder_embed_dim, num_heads=decoder_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                             drop_path_ratio=0.0, mlp_ratio=mlp_ratio, act_layer=act_layer,
                             norm_layer=norm_layer)
            for _ in range(decoder_block_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """
        initialize (and freeze) pos_embed by sin-cos embedding
        """
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    def forward_encoder(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        b, num, dim = x.size()
        x = x + self.pos_embed[:, 1:, :]
        x, mask, idx_storage = self.mask(x)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, idx_storage

    def forward_decoder(self, x: torch.Tensor, idx_storage: torch.Tensor):
        x = self.decoder_embed(x)
        b, num, dim = x.size()
        setattr(self, '_target', x)
        # ignore cls token
        mask_tokens = self.mask_token.repeat(b, idx_storage.shape[1] - num + 1, 1)
        x_ = torch.cat((x[:, 1:, :], mask_tokens), dim=1)
        # un-shuffle
        x_ = torch.gather(x_, dim=1, index=idx_storage.unsqueeze(-1).repeat(1, 1, dim))
        # append cls_token
        x = torch.cat((x[:, :1, :], x_), dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_head(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unmask_patches, mask, idx_storage = self.forward_encoder(x)
        pred = self.forward_decoder(unmask_patches, idx_storage)
        return pred, mask, getattr(self, '_target')


def model_factory(
        model_name: str = 'base',
        load_file: bool = False,
        load_dict: t.Optional[t.Dict] = None,
):
    """
    Model Factory
    """
    assert model_name in ('base', 'huge', 'large'), 'model name should be "base", "huge", "large"'
    if load_dict:
        return MAE(**load_dict)
    elif load_file:
        with open(rf'mae-{model_name}.yaml', 'r') as m_f:
            config = yaml.safe_load(m_f)
        with contextlib.suppress(NameError):
            for key, value in config.items():
                if isinstance(value, str):
                    if value.startswith('nn'):
                        config[key] = getattr(nn, config.get(key).split('.')[-1])
                    else:
                        config[key] = globals().get(value, value)
        return MAE(**config)
    else:
        raise ValueError('can not build model')

# if __name__ == '__main__':
#     mask_ = Mask(0.75)
#     x_ = torch.randn(3, 100, 200)
#     # mask_, unmask_ = mask(x)
#     x_masked, mask_, ids_restore = mask(x_)
#     print(x_masked, mask_, ids_restore)
