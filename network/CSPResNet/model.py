import torch
import torch.nn as nn
import typing as t

from torchsummary import summary

from network.ResNeXt.model import BasicBlock, BottleNeck, ResNeXt, Conv
from torchvision.models import resnet50


class CSPResnet(ResNeXt):
    """
    the bottleneck of ResNet with CSP Block
    """

    def __init__(
            self,
            block: t.Type[t.Union[BasicBlock, BottleNeck, t.Callable, t.List, t.Tuple]],
            layers: t.List[int],
            layers_output: t.List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: t.Optional[nn.Module] = None,
            act_layer: t.Optional[nn.Module] = None,
            classifier: bool = True,  # whether include GAP and fc layer to classify
            base_width: int = 64,
            split_ratio: float = 0.5,
            **kwargs
    ):
        """
        split_group: Is it divided by the number of groups or by each group of channels,
            if True, divided by the number of groups, else, divided each group of channels.
        """
        self.device = kwargs.get('device')
        super(CSPResnet, self).__init__(
            block,
            layers,
            layers_output,
            num_classes,
            groups,
            width_per_group,
            norm_layer,
            act_layer,
            classifier,
            base_width,
            split_ratio=split_ratio,
            **kwargs
        )

    def make_layer(
            self,
            block: t.Type[t.Union[BasicBlock, BottleNeck, t.Callable]],
            out_chans: int,
            blocks: int,
            stride: int = 1,
            base_width=64,
            split_ratio: float = 0.5,
            **kwargs
    ) -> t.Tuple[nn.Module, nn.Module, nn.Module, int]:
        c0 = int(self.in_chans * split_ratio)
        c1 = self.in_chans - c0
        down_sample = None
        c1_out = int(out_chans * split_ratio) * block.expansion
        if stride != 1 or c1 != c1_out:
            down_sample = nn.Sequential(
                nn.Conv2d(c1, c1_out, 1, stride, bias=False),
                self.norm_layer(c1_out)
            )
        right_path_layers = [block(
            c1, int(out_chans * split_ratio), stride, self.groups, self.width_per_group, down_sample,
            self.norm_layer, self.act_layer, base_width=base_width, **kwargs
        ).to(self.device)]
        self.in_chans = c1_out
        for _ in range(1, blocks):
            right_path_layers.append(
                block(
                    self.in_chans,
                    int(out_chans * split_ratio),
                    1,
                    self.groups,
                    self.width_per_group,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    base_width=base_width,
                    **kwargs
                ).to(self.device)
            )
        # residual block path with 1x1
        right_path_layers.append(Conv(c1_out, c1_out, 1,
                                      norm_layer=self.norm_layer, act_layer=self.act_layer).to(self.device))

        # another path with 1x1
        left_path_layer = Conv(c0, c1_out, 1, stride=stride,
                               norm_layer=self.norm_layer, act_layer=self.act_layer).to(self.device)

        last_conv = Conv(c1_out * 2, c1_out * 2, 1,
                         norm_layer=self.norm_layer, act_layer=self.act_layer).to(self.device)
        self.in_chans *= 2
        return nn.Sequential(*right_path_layers), left_path_layer, last_conv, c0

    def forward(self, x: torch.Tensor):

        # stage-0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.max_pool(x)
        # stage-1
        right_path_layers, left_path_layer, last_conv, c0 = self.layer1
        x_left, x_right = x[:, :c0, :, :], x[:, c0:, :, :]
        x_right = right_path_layers(x_right)
        x_left = left_path_layer(x_left)
        x = last_conv(torch.cat((x_left, x_right), dim=1))

        # stage-2
        right_path_layers, left_path_layer, last_conv, c0 = self.layer2
        x_left, x_right = x[:, :c0, :, :], x[:, c0:, :, :]
        x_right = right_path_layers(x_right)
        x_left = left_path_layer(x_left)
        x = last_conv(torch.cat((x_left, x_right), dim=1))

        # stage-3
        right_path_layers, left_path_layer, last_conv, c0 = self.layer3
        x_left, x_right = x[:, :c0, :, :], x[:, c0:, :, :]
        x_right = right_path_layers(x_right)
        x_left = left_path_layer(x_left)
        x = last_conv(torch.cat((x_left, x_right), dim=1))

        # stage-4
        right_path_layers, left_path_layer, last_conv, c0 = self.layer4
        x_left, x_right = x[:, :c0, :, :], x[:, c0:, :, :]
        x_right = right_path_layers(x_right)
        x_left = left_path_layer(x_left)
        x = last_conv(torch.cat((x_left, x_right), dim=1))

        # head
        if self.classifier:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def csp_resnet_50(num_classes: int = 1000, classifier: bool = True, split_ratio: float = 0.5, device: str = 'cuda'):
    BottleNeck.expansion = 4
    return CSPResnet(
        block=BottleNeck,
        layers=[3, 4, 6, 3],
        layers_output=[64, 128, 256, 512],
        num_classes=num_classes,
        classifier=classifier,
        base_width=64,
        split_ratio=split_ratio,
        device=device
    )


if __name__ == '__main__':
    device = 'cuda'
    _x = torch.rand((2, 3, 224, 224)).to(device)
    model = csp_resnet_50().to(device)
    res = model(_x)
    print(res.size())
    summary(model, (3, 224, 224), device=device)
