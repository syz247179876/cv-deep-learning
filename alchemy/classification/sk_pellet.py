import sys

if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from Attention import sk_resnet_50, sk_resnet_152, sk_resnet_101
from alchemy.utils import Args, ClassifyLoss, basic_run, TrainBase


class SKArgs(Args):
    """
    SKNet Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(SKArgs, self).__init__()
        """
        super(SKArgs, self).__init__()
        self.set_train_args()
        self.opts.pretrained = True


class SKLoss(ClassifyLoss):

    def __init__(self):
        super(SKLoss, self).__init__()


class SKTrain(TrainBase):

    def __init__(self, *args, **kwargs):
        super(SKTrain, self).__init__(*args, **kwargs)

    def freeze_layers(self):
        """
        freeze layer except classifier head and sk module
        """
        for name, params in self.model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                params.requires_grad = False
            if 'sk' in name:
                params.requires_grad = True

    def unfreeze_layers(self):
        """
        after some epoch, unfreeze previous layers
        """
        pass


def sk_run(model_name: str, args: SKArgs, loss_obj: SKLoss, **kwargs):
    """
    se-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj, train_class=SKTrain, **kwargs)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = SKArgs()
    loss_obj_ = SKLoss()
    args_.opts.model_name = 'sk_resnet_50'
    sk_run(args_.opts.model_name, args_, loss_obj_, ignore_layers=['fc', ])
