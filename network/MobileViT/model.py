import torch
import torch.nn as nn
import typing as t

import yaml
from torchsummary import summary
from other_utils.conv import MobileViTBlock, MV2Block, MConv


class MobileViT(nn.Module):
    """
    MobileViT based on MobileNetV2 and MobileViT Block
    """
    default_act = nn.SiLU

    def __init__(
            self,
            in_chans: int,
            out_chans: t.List[int],
            dims: t.List[int],
            n_repeat_layer: t.List[int],
            n_repeat_transformer: t.List[int],
            kernel_size: int = 3,
            expand_t: int = 4,
            patch_size: int = 2,
            head_num: int = 8,
            mlp_ratio: float = 4.,
            classifier: bool = False,
            num_classes: int = 1000,
    ):
        super(MobileViT, self).__init__()
        self.net = nn.Sequential(
            MConv(in_chans, out_chans[0], 3, 2, act_layer=self.default_act),
            MV2Block(out_chans[0], out_chans[1], 1, expand_t),

            MV2Block(out_chans[1], out_chans[2], 2, expand_t),
            MV2Block(out_chans[2], out_chans[3], 1, expand_t),

            MV2Block(out_chans[3], out_chans[4], 2, expand_t),
            MobileViTBlock(out_chans[4], dims[0], kernel_size, n_repeat_transformer[0], patch_size, head_num,
                           mlp_ratio),

            MV2Block(out_chans[5], out_chans[6], 2, expand_t),
            MobileViTBlock(out_chans[6], dims[1], kernel_size, n_repeat_transformer[1], patch_size, head_num,
                           mlp_ratio),

            MV2Block(out_chans[7], out_chans[8], 2, expand_t),
            MobileViTBlock(out_chans[8], dims[2], kernel_size, n_repeat_transformer[2], patch_size, head_num,
                           mlp_ratio),

            MConv(out_chans[9], out_chans[10], 1, 1, act_layer=self.default_act)
        )

        self.classifier = classifier
        if classifier:
            self.classifier_head = nn.Sequential(*(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(out_chans[10], num_classes)
            ))

    def forward(self, x: torch.Tensor):

        out = self.net(x)
        if self.classifier:
            out = self.classifier_head(out)
        return out


def mobilevit(num_classes: int = 1000, classifier: bool = False, cfg: str = 'mobilevit-xxs.yaml'):
    with open(cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return MobileViT(
        in_chans=3,
        classifier=classifier,
        num_classes=num_classes,
        **cfg
    )


if __name__ == '__main__':
    _x = torch.randn((2, 3, 640, 640)).to(0)
    model = mobilevit(classifier=False).to(0)
    res = model(_x)
    print(res.size())
    summary(model, (3, 640, 640))


