"""
Darknet-53 model
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import typing as t

from ..settings import *


class BasicBlock(nn.Module):
    """
    the basic block use conv, bn and relu
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: t.Union[int, t.Tuple[int, int]],
            stride: int = 1,
            norm_layer: t.Optional[t.Callable[..., nn.Module]] = None,
    ):
        super(BasicBlock, self).__init__()
        self.bn = norm_layer(out_channels) if norm_layer else nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        if isinstance(kernel_size, int) and kernel_size == 1 or isinstance(kernel_size, list) and kernel_size[0] == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        elif isinstance(kernel_size, int) and kernel_size == 3 or isinstance(kernel_size, list) and kernel_size[0] == 3:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            raise Exception('not support kernel size except for 1x1, 3x3')

    def forward(self, inputs: torch.Tensor):
        y = self.conv(inputs)
        y = self.bn(y)
        y = self.relu(y)
        return y


class ResidualBlock(nn.Module):
    """
    the residual block based on Darknet-53
    """

    def __init__(self, in_channel: int, channels: t.Tuple[int, int]):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channels[0], 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, inputs: torch.Tensor):
        """
        x means the features of the previous layer.
        y means the residual feature abstracted through convolution.
        """
        x = inputs
        y = self.conv1(inputs)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y += x
        return y


class BackBone(nn.Module):
    """
    Darknet-53
    """

    def __init__(self):
        """
        RESIDUAL_BLOCK_NUMS: store the number of each residual block
        in Darknet-53, the layers_num can be [1, 2, 8, 8, 4]
        """
        super(BackBone, self).__init__()
        # use to record input channel of conv layer, which between the mid of every two residual blocks
        self.channel = 32
        self.head = BasicBlock(3, 32, 3, 1)

        self.layer1 = self._create_layer((32, 64), DRK_53_RESIDUAL_BLOCK_NUMS[0])
        self.layer2 = self._create_layer((64, 128), DRK_53_RESIDUAL_BLOCK_NUMS[1])
        self.layer3 = self._create_layer((128, 256), DRK_53_RESIDUAL_BLOCK_NUMS[2])
        self.layer4 = self._create_layer((256, 512), DRK_53_RESIDUAL_BLOCK_NUMS[3])
        self.layer5 = self._create_layer((512, 1024), DRK_53_RESIDUAL_BLOCK_NUMS[4])

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1024, 1000),
            nn.Softmax()
        )

    def _create_layer(self, channels: t.Tuple[int, int], block_num: int) -> nn.Sequential:
        """
        channels: residual block channels, pls refer to the paper for the specific residual block channels
        1.down-sampling, stride=2, kernel_size=3x3
        2.residual block
        """
        layers = []
        down_sampling = BasicBlock(self.channel, channels[1], kernel_size=3, stride=2)
        layers.append(('down-sampling', down_sampling))
        self.channel = channels[1]
        for i in range(block_num):
            layers.append((f'residual-{i}', ResidualBlock(self.channel, channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        layer3, layer4, layer5 use to construct a feature pyramid for feature fusion
        their down-sampling multiple are 8, 16, 32 respectively
        their feature map are 52x52, 26x26, 13x13 based on Image size 416x416
        """
        x = self.head(inputs)
        x = self.layer1(x)
        x = self.layer2(x)

        out_3 = self.layer3(x)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_3, out_4, out_5


class Darknet53(nn.Module):

    def __init__(self, dataset_name: str = 'VOC'):
        """
        dataset can be 'VOC' or 'COCO',
        build the three FPN layer to enhance feature extraction
        """
        super(Darknet53, self).__init__()
        self.backbone = BackBone()
        feature_channel = ANCHORS_NUM * (1 + 4 + globals().get(f'{dataset_name}_CLASS_NUM'))
        self.final_filter_1 = self._last_layers((512, 1024), DRK_53_LAYER_OUT_CHANNELS[-1], feature_channel)
        # use to feature fusion
        self.final_filter_conv_1 = BasicBlock(512, 256, 1)
        # up-sampling
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_filter_2 = self._last_layers((256, 512), DRK_53_LAYER_OUT_CHANNELS[-2] + 256, feature_channel)
        self.final_filter_conv_2 = BasicBlock(256, 128, 1)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_filter_3 = self._last_layers((128, 256), DRK_53_LAYER_OUT_CHANNELS[-3] + 128, feature_channel)

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, ...]:
        """
        feature map size and down sample multiple
        x3: 52x52, 8
        x2: 26x26, 16
        x1: 13x13, 32

        Output:
            three scale features extracted after feature fusion

        note: three scale, two feature fusion
        """
        x3, x2, x1 = self.backbone(inputs)
        out_1_branch = self.final_filter_1[:5](x1)  # 13x13x512
        x1 = self.final_filter_1[5:](out_1_branch)  # 13x13x75 or 13x13x255
        # the first layer feature fusion
        x1_fusion = self.final_filter_conv_1(out_1_branch)
        x1_fusion = self.up_sample_1(x1_fusion)
        x1_fusion = torch.cat((x1_fusion, x2), dim=1)

        out_2_branch = self.final_filter_2[:5](x1_fusion)  # 26x26x256
        x2 = self.final_filter_2[5:](out_2_branch)  # 26x26x75 or 26x26x255
        # the second layer feature fusion
        x2_fusion = self.final_filter_conv_2(out_2_branch)
        x2_fusion = self.up_sample_2(x2_fusion)
        x2_fusion = torch.cat((x2_fusion, x3), dim=1)

        x3 = self.final_filter_3(x2_fusion)  # 52x52x75 or 52x52x255

        return x1, x2, x3

    @staticmethod
    def _last_layers(mid_channels: t.Tuple[int, int], in_channel: int, out_channel: int) -> nn.Sequential:
        """
        the last layer includes a total of 7 layers of convolutions,
        which are divided into two parts.
        1.the first five layers use to perform feature fusion with the previous layer
        2.the last two layers use to predict according to fusion features

        note:
        For example,
        if we use VOC dataset, the out_channels = anchor_num * (1 + 4 + class_num) = 3 * 25 = 75;
        if we use COCO dataset, the out_channels = 3 * (1 + 4 + 80) = 255.

        """
        return nn.Sequential(
            BasicBlock(in_channel, mid_channels[0], kernel_size=1),
            BasicBlock(mid_channels[0], mid_channels[1], kernel_size=3),
            BasicBlock(mid_channels[1], mid_channels[0], kernel_size=1),
            BasicBlock(mid_channels[0], mid_channels[1], kernel_size=3),
            BasicBlock(mid_channels[1], mid_channels[0], kernel_size=1),

            BasicBlock(mid_channels[0], mid_channels[1], kernel_size=3),
            nn.Conv2d(mid_channels[1], out_channel, kernel_size=1, bias=True)
        )


if __name__ == "__main__":
    model = BackBone()
    x_ = torch.randn([1, 3, 416, 416])
    res = model(x_)
    print(res[0].size(), res[1].size(), res[2].size())
    darknet_53 = Darknet53()
    res = darknet_53(x_)
    print(res[0].size(), res[1].size(), res[2].size())
