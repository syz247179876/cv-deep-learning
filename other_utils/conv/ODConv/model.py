import torch
import torch.nn as nn
import typing as t
import torch.nn.functional as F

from other_utils.utils import auto_pad
from torchsummary import summary


class OmniAttention(nn.Module):
    """
    Attention, used to compute include four types attention of different dimensions in ODConv
    1.the spatial kernel size kxk, Cs
    2.the input channel number Cin
    3.the output channel number Cf
    4.the convolution kernel number Cw

    note:
    1.using multi-head attention to compute four dimension in a parallel manner.
    2.the weight of input channel, output channel, spatial kernel is shared to all convolution kernels.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            groups: int = 1,
            reduction: float = 0.0625,
            expert_num: int = 4,
            hidden_chans: int = 16,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        """
        reduction means the scaling factor of the first linear layer, default is 1 / 16 = 0.0625
        expert_num means the number of convolution kernels, default is 4
        hidden_chans means the scaling size after first liner layer, default is 16
        """
        super(OmniAttention, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        attention_chans = max(int(in_chans * reduction), hidden_chans)
        self.kernel_size = kernel_size
        self.expert_num = expert_num
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_chans, attention_chans, 1, bias=False)
        self.bn = norm_layer(attention_chans)
        self.act = act_layer(inplace=True)

        # in channel attention, like SE Attention Module
        self.in_chans_fc = nn.Conv2d(attention_chans, in_chans, 1, bias=True)
        self.channel_attention = self.get_channel_attention

        # out channel attention, like SE Attention Module
        # if current conv is DWC, ignore filter attention
        if in_chans == groups and in_chans == out_chans:
            self.filter_attention = self.ignore
        else:
            self.filter_fc = nn.Conv2d(attention_chans, out_chans, 1, bias=True)
            self.filter_attention = self.get_filter_attention

        # spatial channel attention
        # if current conv is PWD, ignore spatial channel attention
        if kernel_size == 1:
            self.spatial_attention = self.ignore
        else:
            self.spatial_fc = nn.Conv2d(attention_chans, kernel_size ** 2, 1, bias=True)
            self.spatial_attention = self.get_spatial_attention

        # kernel num attention, like SE Attention Module
        # if kernel num is one, ignore kernel num attention
        if expert_num == 1:
            self.expert_attention = self.ignore
        else:
            self.expert_fc = nn.Conv2d(attention_chans, expert_num, 1, bias=True)
            self.expert_attention = self.get_expert_attention

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def ignore(x: torch.Tensor) -> float:
        return 1.0

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, in_chans, 1, 1)
        """
        channel_attention = x.view(x.size(0), -1, 1, 1)
        channel_attention = self.in_chans_fc(channel_attention)
        channel_attention = torch.sigmoid(channel_attention)
        return channel_attention

    def get_filter_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, out_chans, 1, 1)
        """
        filter_attention = x.view(x.size(0), -1, 1, 1)
        filter_attention = self.filter_fc(filter_attention)
        filter_attention = torch.sigmoid(filter_attention)
        return filter_attention

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, 1, 1, 1, kernel_size, kernel_size)
        batch size, expert num, height, weight, kernel size, kernel size
        """
        # trans k(kernel size) x k x 1 to 1(channel) x 1(height) x 1(width) x k (kernel size) x k
        spatial_attention = self.spatial_fc(x)
        spatial_attention = spatial_attention.view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention)
        return spatial_attention

    def get_expert_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (b, in_chans, 1, 1)
        return dim is (b, expert_num, 1, 1, 1, 1)
        batch size, expert num, height, weight, kernel size, kernel size
        """
        expert_attention = self.expert_fc(x)
        # -1 means the number of convolution kernel
        expert_attention = expert_attention.view(x.size(0), -1, 1, 1, 1, 1)
        expert_attention = torch.softmax(expert_attention, dim=1)
        return expert_attention

    def forward(self, x: torch.Tensor):
        x = self.gap(x)
        x = self.fc(x)
        x = self.act(x)

        return self.channel_attention(x), self.filter_attention(x), self.spatial_attention(x), self.expert_attention(x)


class ODConv(nn.Module):
    """
    ODConv --- new dynamic convolution called Omni-Dimension Convolution, better than CondConv, DyConv...
    ODConv leverages a multi-dimensional attention mechanism to learn four types of attentions for
    convolutional kernels along four dimensions of the kernel space in a parallel manner.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            reduction: float = 0.0625,
            hidden_chans: int = 16,
            expert_num: int = 4,
            bias: bool = False,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        super(ODConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.reduction = reduction
        self.hidden_chans = hidden_chans
        self.expert_num = expert_num
        self.bias = bias

        self.attention = OmniAttention(in_chans, out_chans, kernel_size, groups, reduction, expert_num,
                                       hidden_chans, norm_layer=norm_layer, act_layer=act_layer)
        # expert weight and bias
        self.weight = nn.Parameter(torch.Tensor(expert_num, out_chans, in_chans // groups, kernel_size, kernel_size))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(expert_num, out_chans))
        else:
            self.bias = None

        # ODConv1x + PWC, not need spatial attention and expert attention
        if self.kernel_size == 1 and self.expert_num == 1:
            self._forward = self._forward_pw1x
        else:
            self._forward = self._forward_omni

        self._init_weights()

    def _init_weights(self):
        for i in range(self.expert_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def _forward_pw1x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Only learn two types of attentions for convolutional along two dimensions of input channel and output channel

        Broadcast point multiplication or matrix multiplication can be used, matrix multiplication reference
        CondConv implementation
        """
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        combined_bias = None

        # note: because expert_attention use softmax to normalize, when only have one expert and 1x1 size,
        # its attentions is 1. so, dynamic convolution in spatial size dimension and expert dimension will be
        # transformed into static convolution!
        if self.bias is not None:
            combined_bias = self.bias.squeeze(dim=0)
        y = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=combined_bias, stride=self.stride,
                     padding=auto_pad(self.kernel_size, d=self.dilation), groups=self.groups)
        y = y * filter_attention
        return y

    def _forward_omni(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learn four types of attentions for convolutional kernels along four dimensions of the kernel space

        Broadcast point multiplication or matrix multiplication can be used, matrix multiplication reference
        CondConv implementation
        """

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, h, w)
        combined_weights = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        combined_weights = torch.sum(combined_weights, dim=1).view(-1, self.in_chans // self.groups,
                                                                   self.kernel_size, self.kernel_size)
        combined_bias = None
        if self.bias is not None:
            combined_bias = torch.mm(kernel_attention.squeeze(), self.bias).view(-1)
        y = F.conv2d(x, weight=combined_weights, stride=self.stride, bias=combined_bias,
                     padding=auto_pad(self.kernel_size, d=self.dilation),
                     dilation=self.dilation, groups=self.groups * b)
        y = y.view(b, self.out_chans, y.size(-2), y.size(-1))
        y = y * filter_attention
        return y

    def forward(self, x: torch) -> torch.Tensor:
        return self._forward(x)


if __name__ == '__main__':
    model = ODConv(3, 64, 3, bias=True).to(0)
    _x = torch.rand(2, 3, 224, 224).to(0)
    res = model(_x)
    print(res.size())