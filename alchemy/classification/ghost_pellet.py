import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import resnet_50_ghost, resnet_101_ghost, resnet_152_ghost
from alchemy.utils import Args, ClassifyLoss, basic_run


class GhostArgs(Args):
    """
    GhostArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(GhostArgs, self).__init__()
        """
        super(GhostArgs, self).__init__()
        self.set_train_args()


class GhostLoss(ClassifyLoss):

    def __init__(self):
        super(GhostLoss, self).__init__()


def ghost_run(model_name: str, args: GhostArgs, loss_obj: GhostLoss):
    """
    ghost-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = GhostArgs()
    loss_obj_ = GhostLoss()
    ghost_run(args_.opts.model_name, args_, loss_obj_)
