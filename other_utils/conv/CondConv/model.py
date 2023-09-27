import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from other_utils.utils import auto_pad


class RoutingFunc(nn.Module):
    """
    example-dependent routing weights r(x) = Sigmoid(GAP(x) * FC)
    A Shadow of Attention Mechanism, like SENet, SKNet, SCConv, but it's different from them
    """

    def __init__(self, in_chans: int, num_experts: int, drop_rate: float = 0.):
        super(RoutingFunc, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_chans, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.sigmoid(self.fc(self.dropout(x)))
        return x


class CondConv(nn.Module):
    """
    CondConv, which also called Dynamic Convolution.

    CondConv increases the number of experts --- the number of convolutional kernels, which not only
    increase the capacity of the model but also maintains low latency in inference. CondConv Only add
    a small amount of additional computation compared to mixture of experts which using more depth/width/channel
    to improve capacity and performance of the model.

    The idea of ConvConv is first compute the weight coefficients for each expert --- a, then make decision and
    perform linear combination to generate new kernel, in last, make general convolution by using new kernel.

    note: the routing weight is simple-dependency, it means the routing weights are different in different simples.
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
            drop_rate: float = 0.,
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

        self.routing = RoutingFunc(in_chans, num_experts, drop_rate)
        # the standard dim of kernel is (out_chans, in_chans, kernel_size, kernel_size)
        # because new kernel is combined by the num_experts size kernels, so the first dim is num_experts
        self.kernel_weights = nn.Parameter(torch.Tensor(num_experts, out_chans, in_chans // groups,
                                                        kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_chans))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # compute weights of different experts in different simples
        routing_weights = self.routing(x)
        x = x.view(1, b * c, h, w)
        kernel_weights = self.kernel_weights.view(self.num_experts, -1)
        # combine weights of all simples, so combined batch_size and out_chans,
        # then use group conv to split different simples
        combined_weights = torch.mm(routing_weights, kernel_weights).view(
            -1, self.in_chans // self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weights, self.bias).view(-1)
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

