import torch
import torch.nn as nn

__all__ = ['PartialConv', ]


class PartialConv(nn.Module):
    """
    PConv
    """

    def __init__(
            self,
            in_chans: int,
            n_div: int,
            forward: str,
    ):
        """
        in_chans is equal to out_chans
        """
        super(PartialConv, self).__init__()
        self.dim_partial = in_chans // n_div
        self.dim_untouched = in_chans - self.dim_partial
        self.partial_conv3 = nn.Conv2d(self.dim_partial, self.dim_partial, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor):
        # clone is used to some extent for identity, it returns an intermediate variable that can pass gradients.
        # It is equivalent to a deep copy and is suitable for scenarios where a variable is repeatedly used.
        x = x.clone()
        x[:, :self.dim_partial, :, :] = self.partial_conv(x[:, :self.dim_partial, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # split
        x1, x2 = torch.split(x, [self.dim_partial, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        return x
