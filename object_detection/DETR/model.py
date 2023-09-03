import numpy as np
import torch
import torch.nn as nn
import typing as t
from PIL import Image
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class DETR(nn.Module):

    def __init__(
            self,
            nums_head: int,
            hidden_dim: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_classes: int
    ):
        super(DETR, self).__init__()
        # CNN to extract abstract advanced semantic information, remove AAP and FC
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[: -2])
        self.conv1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # standard transformer encoder-decoder
        self.transformer = nn.Transformer(hidden_dim, nums_head, num_encoder_layers, num_decoder_layers)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # prediction heads, include null class
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = nn.Linear(hidden_dim, 4)

        # fixed position embedding, in ResNet50, down-sample multiple is 32, so the shape of feature map smaller than 50
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x: torch.Tensor):
        y = self.backbone(x)
        # writer.add_image(f'feature_map-dim-resnet50', make_grid(y[0].unsqueeze(1)), 2)
        y = self.conv1(y)
        # writer.add_image(f'feature_map-dim-proj', make_grid(y[0].unsqueeze(1)), 3)
        b, c, h, w = y.size()
        h_pos = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)
        w_pos = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)
        pos_embed = torch.cat((h_pos, w_pos), dim=-1).flatten(0, 1).unsqueeze(1)
        y = self.transformer(pos_embed + y.flatten(2, -1).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))
        return self.class_head(y), self.bbox_head(y).sigmoid()


if __name__ == '__main__':
    img_path = r'./imgs/img.png'
    img = Image.open(img_path)
    img = img.resize((640, 640))
    img_: torch.Tensor = torch.tensor(np.array(img)) / 255.
    img_ = img_.unsqueeze(0).permute(0, 3, 1, 2)
    writer_ = SummaryWriter('logs')
    origin_img = (img_.squeeze(0).permute(1, 2, 0) * 255.).type(torch.uint8)
    # writer_.add_images('origin_img', origin_img, 1, dataformats='HWC')

    detr = DETR(nums_head=8, hidden_dim=256, num_encoder_layers=6, num_decoder_layers=6, num_classes=91)
    res = detr(img_)
    print(res[0].size(), res[1].size())
    writer_.close()
