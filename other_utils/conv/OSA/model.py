"""
One-Shot Aggregation used in VoVNet to reduce inference time and memory compared DenseNet。
"""
import torch
import torch.nn as nn
import typing as t

from other_utils.utils import auto_pad


class OSA(nn.Module):
    """
    One-Shot Aggregation used in VoVNet to reduce inference time and memory,
    it makes the input and output channels of each layer the same except for the last one,
    minimizing the cost of MAC memory access.

    it also a lightweight model.

    *note:
    VoVNet: https://arxiv.org/abs/1904.09730
    VoVNet only aggregates features in the final stage of each module, which preserves the advantages of
    concatenate in DenseNet and optimizes the utilization of GPU parallelism (reduce the use of 1x1 conv).
    The input and output channels are the same and MAC is optimized.
    """

    def __init__(
            self,
            in_chans: int,
            growth_chans: int,
            concat_chans: int,
            block_num: int,
            kernel_size: int = 3,
            depth_wise: bool = False,
            use_identity: bool = True,
    ):
        super(OSA, self).__init__()
        self.in_chans = in_chans
        self.growth_chans = growth_chans
        self.concat_chans = concat_chans
        self.use_identity = use_identity

        if depth_wise and in_chans != growth_chans:
            # first trans in_chans to growth_rate in channel-wise
            self.conv_reduction = nn.Conv2d(in_chans, growth_chans, kernel_size=1, stride=1)
        in_channel = in_chans
        self.layers = nn.ModuleList()
        for i in range(block_num):
            if depth_wise:
                self.layers.append(nn.Conv2d(growth_chans, growth_chans,
                                             kernel_size=kernel_size, stride=1, padding=auto_pad(kernel_size)))
            else:
                self.layers.append(nn.Conv2d(in_channel, growth_chans,
                                             kernel_size=kernel_size, stride=1, padding=auto_pad(kernel_size)))
            in_channel = growth_chans

        last_chans = in_chans + growth_chans * block_num
        self.conv_last = nn.Conv2d(last_chans, concat_chans, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        if hasattr(self, 'conv_reduction'):
            x = self.conv_reduction(x)
        identity = None
        if self.use_identity:
            identity = x
        output = [x]
        # compute each layer and append output of each layer inorder to concat in the last layer
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        out = torch.cat(output, dim=1)
        out = self.conv_last(out)
        if self.use_identity:
            out = out + identity
        return out


if __name__ == '__main__':
    _x = torch.rand((2, 128, 224, 224))
    osa = OSA(128, 64, 128, 3, 3, False, True)
    res = osa(_x)
    print(res.size())
