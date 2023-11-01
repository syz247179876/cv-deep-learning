import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import ResNext_50_SC_32x4d, ResNext_101_SC_32x4d, ResNext_152_SC_32x4d
from alchemy.utils import Args, ClassifyLoss, basic_run, TrainBase


class SCArgs(Args):
    """
    SCArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(SCArgs, self).__init__()
        """
        super(SCArgs, self).__init__()
        self.set_train_args()


class SCLoss(ClassifyLoss):

    def __init__(self):
        super(SCLoss, self).__init__()


def sc_run(model_name: str, args: SCArgs, loss_obj: SCLoss, **kwargs):
    """
    sc-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj, **kwargs)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = SCArgs()
    loss_obj_ = SCLoss()
    sc_run(args_.opts.model_name, args_, loss_obj_, ignore_layers=['fc'])
