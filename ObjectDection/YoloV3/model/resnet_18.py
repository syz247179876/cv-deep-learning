import torch
import torch.nn as nn
import typing as t


class Residual(nn.Module):
    """
    residual widget
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            use_1x1_conv: bool = False,
            stride: int = 1
    ):
        # Same Convolution
        super(Residual, self).__init__()
        self.conv3x3 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)
        self.conv3x3_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1_conv:
            # when use_1x1_conv is True, we should ensure that x and residual block have the same channels, w, h
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        y = self.relu(self.bn1(self.conv3x3(inputs)))
        y = self.bn2(self.conv3x3_2(y))
        if self.conv1x1:
            inputs = self.conv1x1(inputs)
        y += inputs
        return self.relu(y)


def resnet_block(
        input_channels: int,
        output_channels: int,
        num_residuals: int,
        first_block: bool = False,
) -> t.List:
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use_1x1_conv=True, stride=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk


class ResNet18(nn.Module):
    """
    the implement of ResNet-18
    """

    def __init__(self):
        super(ResNet18, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_2 = nn.Sequential(
            # not need to halve length and width, as there is MaxPool level upper, so set first_block = True
            *resnet_block(64, 64, 2, first_block=True)
        )
        self.block_3 = nn.Sequential(
            *resnet_block(64, 128, 2)
        )
        self.block_4 = nn.Sequential(
            *resnet_block(128, 256, 2)
        )
        self.block_5 = nn.Sequential(
            *resnet_block(256, 512, 2)
        )
        self.net = nn.Sequential(self.block_1, self.block_2, self.block_3, self.block_4, self.block_5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),  # [B, 512, 1, 1] --> [B, 512]
                                 nn.Linear(512, 10)  # [B, 512] --> [B, 10]
                                 )

    def forward(self, inputs):
        return self.net(inputs)
        # for layer in self.model:
        #     inputs = layer(inputs)
        #     print(layer.__class__.__name__, 'output shape:\t', inputs.shape)
        # return inputs


if __name__ == "__main__":
    # residual_1 = Residual(3, 3, True)
    # x1 = torch.randn((4, 3, 6, 6))
    # y1 = residual_1(x1)
    # print(y1.size())  # [4, 3, 6, 6]
    #
    # # use 1x1 conv
    # residual = Residual(3, 6, True, stride=2)
    # x = torch.randn((4, 3, 6, 6))
    # y = residual(x)
    # print(y.size())  # [4, 6, 3, 3]

    X = torch.rand((4, 3, 224, 224))
    model = ResNet18()
    Y = model(X)
    print(model)
