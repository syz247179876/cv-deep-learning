import os

import torch.nn as nn
import torch
import typing as t

import yaml

from other_utils.conv import DenseBlock, TransitionLayer
from other_utils.utils import auto_pad
from torchsummary import summary

ROOT = os.path.dirname(__file__)


class DenseNet(nn.Module):

    def __init__(
            self,
            feature_num: int,
            growth_rate: int,
            block_nums: t.List[int],
            compress_ratio: float = 0.5,
            scale_factor: int = 4,
            drop_ratio: float = 0.1,
            classifier: bool = False,
            num_classes: int = 1000,
    ):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Sequential(*(
            nn.Conv2d(3, feature_num, 7, 2, padding=auto_pad(7), bias=False),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(inplace=True)
        ))
        self.layer2_5 = nn.ModuleList()

        for block_num in block_nums:
            self.layer2_5.append(nn.Sequential(*(
                DenseBlock(feature_num, growth_rate, block_num, scale_factor, drop_ratio),
                TransitionLayer(feature_num + block_num * growth_rate, int(feature_num * compress_ratio))
            )))
            feature_num = int(feature_num * compress_ratio)

        self.classifier = classifier
        if classifier:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier_head = nn.Linear(feature_num, num_classes)
            self.softmax = nn.Softmax(dim=1)

    def __init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: nn.Module):

        x = self.layer1(x)
        for layer2_5 in self.layer2_5:
            x = layer2_5(x)
        if self.classifier:
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.classifier_head(x)
            x = self.softmax(x)
        return x


def densenet_121(cfg_name: str = 'densenet-121.yaml', num_classes: int = 1000, classifier: bool = True):
    cfg_path = os.path.join(ROOT, cfg_name)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return DenseNet(
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


def densenet_169(cfg_name: str = 'densenet-169.yaml', num_classes: int = 1000, classifier: bool = True):
    cfg_path = os.path.join(ROOT, cfg_name)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return DenseNet(
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


def densenet_201(cfg_name: str = 'densenet-201.yaml', num_classes: int = 1000, classifier: bool = True):
    cfg_path = os.path.join(ROOT, cfg_name)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return DenseNet(
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


def densenet_264(cfg_name: str = 'densenet-264.yaml', num_classes: int = 1000, classifier: bool = True):
    cfg_path = os.path.join(ROOT, cfg_name)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return DenseNet(
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


if __name__ == '__main__':
    _x = torch.rand((2, 3, 224, 224)).to(0)
    model = densenet_264().to(0)

    res = model(_x)
    summary(model, input_size=(3, 224, 224))
