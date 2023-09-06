import torch
import torch.nn as nn
import typing as t


class MAELoss(nn.Module):
    """
    MAE Loss
    """

    def __init__(self, patches_tuple: t.Tuple[int] = (14, 14), patch_size: t.Tuple[int] = (16, 16)):
        super(MAELoss, self).__init__()
        self.patches_tuple = patches_tuple
        self.patch_size = patch_size
        self.num_patches = self.patches_tuple[0] * self.patches_tuple[1]

    def patchify(self, images: torch.Tensor):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        b, c, h, w = images.size()
        patches = images.view(b, c, self.patches_tuple[0], self.patch_size[0], self.patches_tuple[1], self.patch_size[1])
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        return patches

    def forward(
            self,
            pred: torch.Tensor,
            images: torch.Tensor,
            mask: torch.Tensor,
            test: bool = False
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        targets = self.patchify(images)
        loss = (pred - targets) ** 2
        loss = loss.mean(dim=-1)
        if test:
            targets[mask == 0] = 0.1
            return (loss * mask).sum() / mask.sum(), targets
        return (loss * mask).sum() / mask.sum()
