import torch
import torch.nn as nn


class FrozenBatchNorm2D(nn.Module):
    """
    Freeze params of BN when use pre-trained file to load weight
    """

    def __init__(self, n: int, eps: float = 1e-5):
        super(FrozenBatchNorm2D, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Parameter recovery for this module
        """
        num_batches_tracked_name = prefix + 'num_batches_tracked'
        if num_batches_tracked_name in state_dict:
            del state_dict[num_batches_tracked_name]
        self._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                   unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor):
        """
        w and b can be used to restore the original distribution
        """
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
