import torch
import torch.nn as nn


class SRU(nn.Module):
    """
    Spatial Reconstruction Unit

    Reduce redundancy in spatial dimension

    main parts:
    1. Separate
        -Separate informative feature maps from less informative ones corresponding to the spatial content,
        so that, we can reconstruct low redundancy feature.

    2. Reconstruction
        -Interacting between different channels(informative channels and less informative channels)
        strengthen the information flow between these channels, so it may improve accuracy,
        reduce redundancy feature and improve feature representation of CNNs.

    """

    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: float = 0.5,
    ):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # split informative feature maps from less informative ones corresponding to the spatial content
        gn_x = self.gn(x)
        # use gn.weight to measure the variance of spatial pixels for each batch and channel
        w = (self.gn.weight / torch.sum(self.gn.weight)).view(1, -1, 1, 1)
        w = self.sigmoid(w * gn_x)
        infor_mask = w >= self.gate_threshold
        less_infor_maks = w < self.gate_threshold
        x1 = infor_mask * gn_x
        x2 = less_infor_maks * gn_x

        # reconstruct feature with informative feature and less informative feature
        x11, x12 = torch.split(x1, x1.size(1) // 2, dim=1)
        x21, x22 = torch.split(x2, x2.size(1) // 2, dim=1)
        out = torch.cat([x11 + x22, x12 + x21], dim=1)
        return out


class SCConv(nn.Module):

    def __init__(self):
        super(SCConv, self).__init__()


if __name__ == '__main__':
    _x = torch.rand((1, 64, 20, 20))
    s = SRU(64)
    s(_x)
