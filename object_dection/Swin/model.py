import torch
import torch.nn as nn
import typing as t


class PatchEmbed(nn.Module):
    """
    Patch Embedding
    dim trans procedure:
    (b, c, h, w) -> (b, embed_dim, patch_size, patch_size) across Conv
    (b, embed_dim, patch_size, patch_size) -> (b, sequence_length, embed_dim) across flatten and transpose
    """

    def __init__(
            self,
            img_size: t.Tuple[int, int] = (224, 224),
            patch_size: t.Tuple[int, int] = (4, 4),
            in_chans: int = 3,
            embed_dim: int = 96,
            norm_layer: t.Optional[nn.Module] = None,
    ):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.conv_embed = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size,
                                    stride=patch_size)

        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_embed(x).flatten(2, -1).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Its function is similar to down-sampling in CNN, in order to reduce resolution and
    increase channels, gradually inform a hierarchical feature map by merging patches in
    deeper layer

    The Patch Merging core is down-sampling for two times, and select four pixels at two intervals
    in row and col each time, then, concat them in last dim


    """

    def __init__(
            self,
            input_shape: t.Tuple[int, int],
            dim: int,
            norm_layer: nn.LayerNorm
    ):
        super(PatchMerging, self).__init__()
        self.input_shape = input_shape
        # self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
        self.reduction = nn.Conv2d(dim * 4, dim * 2)
        self.norm = norm_layer(dim * 4)

    def forward(self, x: torch.Tensor):
        """
        x dim: (b, length, dim)
        """
        h, w = self.input_shape
        b, length, dim = x.size()
        assert length == h * w, 'the length of sequence is not equal to h plus w'
        assert h % 2 == 0 and w % 2 == 0, 'the h or w is not even, can not do Patch Merging'

        x = x.view(b, h, w, dim)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat((x0, x1, x2, x3), dim=-1)
        x = x.view(b, -1, dim * 4)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer(nn.Module):
    pass
