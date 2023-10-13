import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import resnet50_bot, resnet101_bot, resnet152_bot
from alchemy.utils import Args, ClassifyLoss, basic_run


class BoTArgs(Args):
    """
    BoTArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(BoTArgs, self).__init__()
        """
        super(BoTArgs, self).__init__()
        self.set_train_args()


class BoTLoss(ClassifyLoss):

    def __init__(self):
        super(BoTLoss, self).__init__()


def bot_run(model_name: str, args: BoTArgs, loss_obj: BoTLoss):
    """
    bot-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = BoTArgs()
    loss_obj_ = BoTLoss()
    bot_run(args_.opts.model_name, args_, loss_obj_)
