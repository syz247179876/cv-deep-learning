import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import ResNeXt50_32x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
from alchemy.utils import Args, ClassifyLoss, basic_run


class ResNeXtArgs(Args):
    """
    ResNeXtArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(ResNeXtArgs, self).__init__()
        """
        super(ResNeXtArgs, self).__init__()
        self.set_train_args()


class ResNeXtLoss(ClassifyLoss):

    def __init__(self):
        super(ResNeXtLoss, self).__init__()


def resnext_run(model_name: str, args: ResNeXtArgs, loss_obj: ResNeXtLoss, **kwargs):
    """
    ResNeXt family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj, **kwargs)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = ResNeXtArgs()
    loss_obj_ = ResNeXtLoss()
    resnext_run(args_.opts.model_name, args_, loss_obj_, ignore_layers=['fc', ])
