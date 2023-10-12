import torch
import torch.nn as nn
import typing as t

from other_utils.utils import auto_pad


class PEG(nn.Module):
    """
    Dynamic position embedding generator, using 2-D convolution with kernel k(k >= 3) and (k - 1) / 2 zero paddings.
    this kxk convolution kernel can be conventional convolution or can be other convolutions, in CPVT paper, PEG
    relies on zero padding to extract absolute position encoding information.

    a successful positional encoding as follows mentioned in CPVT paper:
    1. it should be position sensitive and capable of translation-equivalence.
    2. it should handle the sequences longer than the ones during training.
    3. it should have the ability to provide the absolute position to a certain degree.
    """

    def __init__(
            self,
            dim: int,
            conv: t.Optional[nn.Module] = None,
            kernel_size: int = 3,
            **kwargs
    ):
        super(PEG, self).__init__()
        if conv is None:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size, 1, padding=auto_pad(kernel_size), groups=dim)
        else:
            self.pos_embed = conv(**kwargs)

    def forward(self, x: torch.Tensor, h: int, w: int):
        b, length, channel = x.size()
        cls_token, feat_tokens = x[:, 0], x[:, 1:]
        feat_tokens = feat_tokens.transpose(1, 2).view(b, channel, h, w)
        x = self.pos_embed(feat_tokens) + feat_tokens
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x



