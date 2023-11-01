import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import resnet50, resnet101, resnet152
from alchemy.utils import Args, ClassifyLoss, basic_run


class ResNetArgs(Args):
    """
    ResNetArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(ResNetArgs, self).__init__()
        """
        super(ResNetArgs, self).__init__()
        self.set_train_args()


class ResNetLoss(ClassifyLoss):

    def __init__(self):
        super(ResNetLoss, self).__init__()


def resnet_run(model_name: str, args: ResNetArgs, loss_obj: ResNetLoss, **kwargs):
    """
    standard resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj, **kwargs)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = ResNetArgs()
    loss_obj_ = ResNetLoss()
    resnet_run(args_.opts.model_name, args_, loss_obj_, ignore_layers=['fc', ])
