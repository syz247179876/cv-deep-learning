import math

import torch.nn as nn
import torch
import typing as t
from alchemy.utils import Args
from Attention import se_resnet_18, se_resnet_34, se_resnet_50, se_resnet_101, se_resnet_152, se_resnext_50_32d4, \
    se_resnext_101_32d4, se_resnext_152_32d4
from alchemy.utils.common import TrainBase


class SEArgs(Args):
    """
    SENet Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(SEArgs, self).__init__()
        """
        super(SEArgs, self).__init__()
        self.set_train_args()


class SELoss(nn.Module):

    def __init__(self):
        super(SELoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        pred_classes_idx = torch.max(pred, dim=1)[1]
        acc_num = torch.eq(pred_classes_idx, labels).sum()
        return self.loss_func(pred, labels), acc_num


if __name__ == '__main__':
    args = SEArgs()
    model = se_resnet_50(num_classes=5)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_obj = SELoss()
    train = TrainBase(args, model, 'se_resnet_50', loss_obj=loss_obj)
    optimizer = torch.optim.SGD(params, lr=train.init_lr(), momentum=0.9,
                                weight_decay=args.opts.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.opts.decrease_interval + 1e-8),
                                0.5 * (math.cos(epoch / args.opts.end_epoch * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    train.scheduler_ = scheduler
    train.optimizer_ = optimizer

    train.main()

