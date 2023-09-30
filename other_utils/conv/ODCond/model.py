import torch
import torch.nn as nn
import typing as t


class OmniAttention(nn.Module):
    """
    Attention, used to compute include four types attention of different dimensions in ODConv
    1.the spatial kernel size kxk, Cis
    2.the input channel number Cin
    3.the output channel number Cfi
    4.the convolution kernel number Cwi

    using multi-head attention to compute four dimension in a parallel manner
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            groups: int = 1,
            reduction: float = 0.0625,
            experts_num: int = 4,
            hidden_chans: int = 16,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
    ):
        """
        reduction means the scaling factor of the first linear layer, default is 1 / 16 = 0.0625
        experts_num means the number of convolution kernels, default is 4
        hidden_chans means the scaling size after first liner layer, default is 16
        """
        super(OmniAttention, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        attention_chans = min(int(in_chans * reduction), hidden_chans)
        self.kernel_size = kernel_size
        self.experts_num = experts_num
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
        if experts_num == 1:
            self.expert_attention = self.ignore
        else:
            self.expert_fc = nn.Conv2d(attention_chans, experts_num, 1, bias=True)
            self.expert_attention = self.get_expert_attention

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def ignore():
        return 1.0

    def get_channel_attention(self, x: torch.Tensor):
        channel_attention = x.view(x.size(0), -1, 1, 1)
        channel_attention = self.in_chans_fc(channel_attention)
        channel_attention = torch.sigmoid(channel_attention)
        return channel_attention

    def get_filter_attention(self, x: torch.Tensor):
        filter_attention = x.view(x.size(0), -1, 1, 1)
        filter_attention = self.filter_fc(filter_attention)
        filter_attention = torch.sigmoid(filter_attention)
        return filter_attention

    def get_spatial_attention(self, x: torch.Tensor):
        # trans k(kernel size) x k x 1 to 1(channel) x 1(height) x 1(width) x k (kernel size) x k
        spatial_attention = self.spatial_fc(x)
        spatial_attention = spatial_attention.view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention)
        return spatial_attention

    def get_expert_attention(self, x: torch.Tensor):
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
