import torch
import typing as t
from torch import nn


class ViTLoss(nn.Module):

    def __init__(self):
        super(ViTLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> t.Tuple[torch.Tensor, int]:
        pred_classes_idx = torch.max(pred, dim=1)[1]
        accu_num = torch.eq(pred_classes_idx, labels).sum()
        return self.loss_function(pred, labels), accu_num
