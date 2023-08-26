import torch.nn as nn
import torch


class MAELoss(nn.Module):
    """
    MAE Loss
    """
    
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss_function = nn.MSELoss(reduction='mean')

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_function(x, labels)

