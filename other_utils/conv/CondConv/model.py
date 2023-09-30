import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
from torchsummary import summary
from other_utils.utils import auto_pad


class RoutingFunc(nn.Module):
    """
    example-dependent routing weights r(x) = Sigmoid(GAP(x) * FC)
    A Shadow of Attention Mechanism, like SENet, SKNet, SCConv, but not exactly the same as them，

    note:
    1.In my implementation of CondConv, I used 1x1 conv instead of fc layer mentioned in the paper.
    2.uss Sigmoid instead of Softmax reflect the advantages of multiple experts, while use Softmax, it only
    one expert has an advantage.
    """

    def __init__(self, in_chans: int, num_experts: int, drop_ratio: float = 0.):
        super(RoutingFunc, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_ratio)
        self.fc = nn.Conv2d(in_chans, num_experts, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.avg_pool(x)
        # x = torch.flatten(x, start_dim=1)
        x = self.sigmoid(self.fc(self.dropout(x)))
        return x.view(x.size(0), x.size(1))


class CondConv(nn.Module):
    """
    CondConv, which also called Dynamic Convolution. plug-and-play module, which can replace convolution in each layer.

    CondConv increases the number of experts --- the number of convolutional kernels, which not only
    increase the capacity of the model but also maintains low latency in inference. CondConv Only add
    a small amount of additional computation compared to mixture of experts which using more depth/width/channel
    to improve capacity and performance of the model.

    The idea of ConvConv is first compute the weight coefficients for each expert --- a, then make decision and
    perform linear combination to generate new kernel, in last, make general convolution by using new kernel.

    note:
    1.the routing weight is sample-dependency, it means the routing weights are different in different samples.
    2.CondConv with the dynamic property through one dimension of kernel space,
    regarding the number of convolution kernel(experts)
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int = 1,
            num_experts: int = 8,
            groups: int = 1,
            dilation: int = 1,
            drop_ratio: float = 0.,
            bias: bool = False
    ):
        super(CondConv, self).__init__()
        self.num_experts = num_experts
        self.out_chans = out_chans
        self.in_chans = in_chans
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.routing = RoutingFunc(in_chans, num_experts, drop_ratio)
        # the standard dim of kernel is (out_chans, in_chans, kernel_size, kernel_size)
        # because new kernel is combined by the num_experts size kernels, so the first dim is num_experts
        self.kernel_weights = nn.Parameter(torch.Tensor(num_experts, out_chans, in_chans // groups,
                                                        kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_chans))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, routing_weights: t.Optional[torch.Tensor] = None):
        b, c, h, w = x.size()
        # compute weights of different experts in different samples
        if routing_weights is None:
            routing_weights = self.routing(x)
        x = x.view(1, b * c, h, w)
        kernel_weights = self.kernel_weights.view(self.num_experts, -1)
        # combine weights of all samples, so combined batch_size and out_chans,
        # then use group conv to split different samples
        combined_weights = torch.mm(routing_weights, kernel_weights).view(
            -1, self.in_chans // self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weights, self.bias).view(-1)
            # weight dim is (b * out_chans, in_chans, k, k), bias dim is (b * out_chans), groups is self.groups * b
            # x dim is (1, b * in_chans, h, w)
            y = F.conv2d(x, weight=combined_weights, bias=combined_bias, stride=self.stride, dilation=self.dilation,
                         padding=auto_pad(self.kernel_size, d=self.dilation), groups=self.groups * b)
        else:
            y = F.conv2d(x, weight=combined_weights, bias=None, stride=self.stride, dilation=self.dilation,
                         padding=auto_pad(self.kernel_size, d=self.dilation), groups=self.groups * b)
        y = y.view(b, self.out_chans, y.size(-2), y.size(-1))
        return y


if __name__ == '__main__':
    model = CondConv(64, 128, 3, bias=True).to(0)
    x_ = torch.randn((2, 64, 224, 224)).to(0)
    res = model(x_)
    print(res.size())
    summary(model, (64, 224, 224), batch_size=8)

