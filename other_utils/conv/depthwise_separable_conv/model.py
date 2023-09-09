import torch
import torch.nn as nn


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution, include Depthwise Convolution and Pointwise Convolution,
    which has fewer parameter quantities.
    """
    
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
    ):
        super(DSConv, self).__init__()
        self.depth_wise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.point_wise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        return self.point_wise(self.depth_wise(x))

