"""
根据YOLO V1论文构建网络结构
"""
import os
import random
import sys
import typing as t
import torch
import torch.nn as nn
import torchvision.models as tv_model

if os.path.dirname(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.getcwd()))

from argument import args_train
from settings import *


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def max_pool_2x2(stride: int = 2):
    """2x2 max pool"""
    return nn.MaxPool2d((2, 2), stride=stride)


class BasicBlock(nn.Module):
    """
    基于Darknet-19的BasicBlock ==> ConvBnLeakReLu
    带有BN层和leaky relu激活函数
    BN: 用于加快模型收敛速度, 减少过拟合, 使得后一层无需总是学习前一层的分布
    leaky relu:
    0.非线性激活函数, 能使模型具备更多的学习表达能力。
    1.用于解决梯度弥散问题, 因为其计算出的梯度值为0或1, 因此在反向传播导数连乘过程中, 梯度值不会像Sigmoid函数那样越来越小, 进而
    避免参数梯度更新慢的问题。
    2.单侧饱和区一定程度上可以减少噪声的干扰, 更具鲁棒性，比如负值(噪音)经过relu函数得到0, 避免了噪音的训练时的干扰
    3.使用leaky relu替代relu, 函数中在对负值区域由0 -> ax, a参数可学习, 解决当负值经过 leaky relu函数时, 也能计算出一个非0梯度值, 解决了
        出现太多死亡神经元问题。
    """

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: t.Tuple[int, int] = (1, 1),
            norm_layer: t.Optional[t.Callable[..., nn.Module]] = None,
    ):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer(out_planes) if norm_layer else nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

        if kernel_size[0] == 1:
            self.conv = conv1x1(in_planes, out_planes)
        elif kernel_size[0] == 3:
            self.conv = conv3x3(in_planes, out_planes)
        else:
            raise Exception('not support kernel size except for 1x1, 3x3')

    def forward(self, inputs: torch.Tensor):
        _x = self.conv(inputs)
        _x = self.norm_layer(_x)
        _x = self.relu(_x)
        return _x


class Darknet19(nn.Module):
    """
    用于分类的Darknet-19
    """

    def __init__(
            self,
            mode: str = 'classification',
    ):
        super(Darknet19, self).__init__()
        assert mode in ['classification', 'predication'], 'mode should be `classification` or `predication`'
        self.mode = mode
        self.layer_1 = nn.Sequential(
            BasicBlock(3, 32, kernel_size=(3, 3)),
            max_pool_2x2()
        )
        self.layer_2 = nn.Sequential(
            BasicBlock(32, 64, kernel_size=(3, 3)),
            max_pool_2x2()
        )
        self.layer_3 = nn.Sequential(
            BasicBlock(64, 128, kernel_size=(3, 3)),
            BasicBlock(128, 64, kernel_size=(1, 1)),
            BasicBlock(64, 128, kernel_size=(3, 3)),
            max_pool_2x2()
        )
        self.layer_4 = nn.Sequential(
            BasicBlock(128, 256, kernel_size=(3, 3)),
            BasicBlock(256, 128, kernel_size=(1, 1)),
            BasicBlock(128, 256, kernel_size=(3, 3)),
            max_pool_2x2()
        )
        self.layer_5 = nn.Sequential(
            BasicBlock(256, 512, kernel_size=(3, 3)),
            BasicBlock(512, 256, kernel_size=(1, 1)),
            BasicBlock(256, 512, kernel_size=(3, 3)),
            BasicBlock(512, 256, kernel_size=(1, 1)),
            BasicBlock(256, 512, kernel_size=(3, 3)),
        )
        self.layer_6 = nn.Sequential(
            BasicBlock(512, 1024, kernel_size=(3, 3)),
            BasicBlock(1024, 512, kernel_size=(1, 1)),
            BasicBlock(512, 1024, kernel_size=(3, 3)),
            BasicBlock(1024, 512, kernel_size=(1, 1)),
            BasicBlock(512, 1024, kernel_size=(3, 3)),
        )
        self.layer_7 = nn.Sequential(
            BasicBlock(1024, 1000, kernel_size=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Softmax(dim=0)  # 得到 1000x1x1 的张量
        )

    def forward(
            self,
            inputs: torch.Tensor
    ) -> t.Union[t.Tuple[torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]]:
        _x = self.layer_1(inputs)
        _x = self.layer_2(_x)
        _x = self.layer_3(_x)
        _x = self.layer_4(_x)
        shallow_x = self.layer_5(_x)
        _x = max_pool_2x2()(shallow_x)
        _x = self.layer_6(_x)
        if self.mode == 'classification':
            _x = self.layer_7(_x)
            return _x
        elif self.mode == 'predication':
            return shallow_x, _x


class YoloV2(nn.Module):
    """
    用于预测的网络结构
    用于预测时的Darknet, 输入图像大小为416x416, 输出13x13x125, 其中125 = 5(anchors num) * (5(pos) + 20(class))
    在backbone中 增加了一个passthrough层, 与骨干网络深层特征进行特征融合, 实现细粒度特征。
    """

    def __init__(self):
        super(YoloV2, self).__init__()
        self.opts = args_train.opts
        self.backbone = Darknet19('predication')
        self.deep_conv = nn.Sequential(
            BasicBlock(1024, 1024, kernel_size=(3, 3)),
            BasicBlock(1024, 1024, kernel_size=(3, 3))
        )
        self.shallow_conv = nn.Sequential(
            BasicBlock(512, 64, kernel_size=(1, 1)),
        )
        self.final_conv = nn.Sequential(
            BasicBlock(1280, 1024, kernel_size=(3, 3)),
            nn.Conv2d(1024, self.opts.anchors_num * (1 + 4 + VOC_CLASSES_LEN), 1),
        )

    @staticmethod
    def passthrough(inputs: torch.Tensor) -> torch.Tensor:
        """
        在通道数上将B x channels x 26 x 26 -> B x channels * 4 x 13 x 13
        融合高分辨率特征信息
        """
        return torch.cat(
            (inputs[:, :, ::2, ::2], inputs[:, :, 1::2, ::2], inputs[:, :, 1::2, ::2], inputs[:, :, 1::2, 1::2]), dim=1)

    @staticmethod
    def post_process(inputs: torch.Tensor):
        """
        tensor post-processing of network output

        note: use contiguous to open up new memory and the storage structure of '_x' is not the same as 'inputs', and
            then, we can use view to change the shape of _x
        """
        b, info, h, w = inputs.size()
        _x = inputs.permute(0, 2, 3, 1).contiguous().view(b, h * w, info)
        return _x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shallow_x, deep_x = self.backbone(inputs)
        shallow_x = self.shallow_conv(shallow_x)  # [B, 64, 26, 26]
        shallow_x = self.passthrough(shallow_x)  # [B, 256, 13, 13]
        deep_x = self.deep_conv(deep_x)  # [B, 1024, 13, 13]
        # feature fusion
        _x = torch.cat((deep_x, shallow_x), dim=1)
        _x = self.final_conv(_x)
        return self.post_process(_x)


if __name__ == '__main__':
    model_1 = Darknet19()
    image_size = random.randrange(320, 608 + 32, 32)
    test_inputs = torch.randn(5, 3, image_size, image_size)
    x_1 = model_1(test_inputs)

    model_2 = YoloV2()
    x_2 = model_2(test_inputs)
    print(x_2.size())
