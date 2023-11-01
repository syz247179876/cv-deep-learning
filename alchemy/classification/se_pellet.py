import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from Attention import se_resnet_18, se_resnet_34, se_resnet_50, se_resnet_101, se_resnet_152, se_resnext_50_32d4, \
    se_resnext_101_32d4, se_resnext_152_32d4, se_mobilenet_v2_1, se_mobilenet_v2_075, se_mobilenet_v2_05
from alchemy.utils import Args, TrainBase, ClassifyLoss, basic_run


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
        self.opts.pretrained = True
        self.batch_size = 1


class SELoss(ClassifyLoss):

    def __init__(self):
        super(SELoss, self).__init__()


class SETrain(TrainBase):

    def __init__(self, *args, **kwargs):
        super(SETrain, self).__init__(*args, **kwargs)

    def freeze_layers(self):
        """
        freeze layer except classifier head and se module
        """
        for name, params in self.model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                params.requires_grad = False
            if 'se' in name:
                params.requires_grad = True


def se_run(model_name: str, args: SEArgs, loss_obj: SELoss, **kwargs):
    """
    se-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj, train_class=SETrain, **kwargs)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = SEArgs()
    loss_obj_ = SELoss()
    args_.opts.model_name = 'se_resnet_50'
    se_run(args_.opts.model_name, args_, loss_obj_, ignore_layers=['fc', ])
