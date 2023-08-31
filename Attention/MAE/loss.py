import torch
import torch.nn as nn


class MAELoss(nn.Module):
    """
    MAE Loss
    """

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()
