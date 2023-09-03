"""
根据YOLO V1论文构建网络结构
"""

import torch
import torch.nn as nn
import torchvision.models as tv_model

import debug
from settings import *


class YoloV1Net(nn.Module):

    def __init__(self):
        super(YoloV1Net, self).__init__()
        self.output = None
        # 使用resnet34进行预训练
        resnet = tv_model.resnet34()
        # 记录resnet网络输出结果
        resnet_out_channel = resnet.fc.in_features
        # 省略resnet34倒数两层，使得resnet最后一层输出的通道数为512
        self.resnet = nn.Sequential(*list(resnet.children())[: -2])

        # YOLO v1的最后4个卷积层
        # padding = (f - 1) // 2 ==> padding = 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )

        # YOLO v1最后2个全连接层
        self.conn_layers = nn.Sequential(
            nn.Linear(GRID_NUM * GRID_NUM * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, GRID_NUM * GRID_NUM * (5 * BBOX_NUM + len(CLASSES))),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        计算前向传播, 输出网络3*3*7结果
        """
        _x = self.resnet(inputs)
        _x = self.conv_layers(_x)
        # x.view()转变x的维度, 将x转换为(x.size(0), x.size(1) * x.size(2) * x.size(3))
        _x = _x.view(_x.size(0), -1)
        _x = self.conn_layers(_x)
        self.output = _x.view(-1, GRID_NUM, GRID_NUM, (5 * BBOX_NUM + len(CLASSES)))
        return self.output






