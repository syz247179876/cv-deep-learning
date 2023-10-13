import torch

from other_utils.conv import BoTBlock
from network.ResNeXt import ResNeXt, BottleNeck
from torchsummary import summary


def resnet50_bot(num_classes: int = 1000, classifier: bool = False, head_num: int = 4, qkv_bias: bool = True):
    BoTBlock.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, BoTBlock],
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        head_num=head_num,
        qkv_bias=qkv_bias
    )


def resnet101_bot(num_classes: int = 1000, classifier: bool = False, head_num: int = 4, qkv_bias: bool = True):
    BoTBlock.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, BoTBlock],
        layers=[3, 4, 23, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        head_num=head_num,
        qkv_bias=qkv_bias
    )


def resnet152_bot(num_classes: int = 1000, classifier: bool = False, head_num: int = 4, qkv_bias: bool = True):
    BoTBlock.expansion = 4
    return ResNeXt(
        block=[BottleNeck, BottleNeck, BottleNeck, BoTBlock],
        layers=[3, 4, 36, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        head_num=head_num,
        qkv_bias=qkv_bias
    )


if __name__ == '__main__':
    _x = torch.rand((4, 3, 1024, 1024)).to(0)
    model = resnet50_bot().to(0)

    res = model(_x)
    print(res.size())
