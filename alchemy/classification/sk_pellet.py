import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from Attention import sk_resnet_50, sk_resnet_152, sk_resnet_101
from alchemy.utils import Args, ClassifyLoss, basic_run


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


class SKLoss(ClassifyLoss):

    def __init__(self):
        super(SKLoss, self).__init__()


def sk_run(model_name: str, args: SKArgs, loss_obj: SKLoss):
    """
    se-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = SKArgs()
    loss_obj_ = SKLoss()
    sk_run(args_.opts.model_name, args_, loss_obj_)
