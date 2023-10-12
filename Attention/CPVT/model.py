import torch
import torch.nn as nn
import typing as t

from torch.nn.init import trunc_normal_
from image_classification.vision_transformer import PatchEmbed, TransformerBlock
from other_utils.pos_embed import PEG


class CPVT(nn.Module):
    """
    Apply PEG after the first transformer block in the encoder of the transformer, Because after the
    first transformer block, the features have global information, and after PEG, better position information
    can be extracted, similar to using a large convolutional kernel.

    In ICLR 2020 Paper --- How much position information do convolutional neural networks encode?
    The author has demonstrated through experiments that encoding absolute position information is better
    when the number of layers is deeper or the convolutional kernel is larger
    """

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
            classification: t.Optional[bool] = False,
            pool: str = 'cls',
    ):
        super(CPVT, self).__init__()
        self.nums_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        # cls_token
        self.cls_tokens = 1
        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, self.cls_tokens, embed_dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, block_depth)]
        # define PEG block
        self.pos_block = PEG(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
                             drop_path_ratio=dpr[i], mlp_ratio=mlp_ratio, act_layer=act_layer,
                             norm_layer=norm_layer)
            for i in range(block_depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.pool = pool
        self.classification = classification
        # classifier head
        if classification:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_feature(self, x: torch.Tensor):
        b, c, h, w = x.size()
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.concat((cls_token, x), dim=1)
        p_h, p_w = h // self.patch_size, w // self.patch_size
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0:
                # add dynamic position embedding after first transformer block
                x = self.pos_block(x, p_h, p_w)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_feature(x)
        # like GAP
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]
        else:
            raise ValueError('pool must be "mean" or "cls"')

        if self.classification:
            x = self.head(x)
        return x


if __name__ == '__main__':
    _x = torch.rand((4, 3, 224, 224))

    model = CPVT(classification=True)
    res = model(_x)
    print(res.size())
