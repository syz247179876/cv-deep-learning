import torch
import torch.nn as nn
import typing as t

from augment import args_train


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
            mode: str = 'conv'
    ):
        super(PatchEmbed, self).__init__()
        self.mode = mode
        self.num_patches = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.conv_embed = nn.Conv2d(in_chans, self.in_chans * self.num_patches[0] * self.num_patches[1],
                                    kernel_size=patch_size, stride=patch_size)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_embed(x).flatten(2, -1).transpose(1, 2)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        patches = x.view(b, c, self.num_patches[0], self.patch_size[0], self.num_patches[1], self.patch_size[1])
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches[0] * self.num_patches[1], -1)
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


class MAE(nn.Module):

    def __init__(
            self,
            img_size: t.Tuple = (224, 224),
            patch_size: int = 16,
            in_chans: int = 3,
            mask: t.Optional[t.Callable] = None,
            patch_embed: t.Optional[t.Callable] = None,
            mode: str = 'split',
    ):
        super(MAE, self).__init__()
        self.opts = args_train.opts
        self.mask = mask or Mask(self.opts.mask_ratio, patch_size * patch_size)
        self.patch_embed = patch_embed or PatchEmbed(img_size, (patch_size, patch_size), in_chans, mode=mode)

    def forward(self, x: torch.Tensor):
        patches = self.patch_embed(x)
        mask_patches, unmask_patches = self.mask(patches)


if __name__ == '__main__':
    _x = torch.randn((2, 3, 224, 224))
    patch = PatchEmbed(mode='split')
    res = patch(_x)
    print(res.size(), res)