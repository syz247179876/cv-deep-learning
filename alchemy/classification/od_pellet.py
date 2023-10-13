import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import resnet18_od, resnet50_od
from alchemy.utils import Args, ClassifyLoss, basic_run


class ODArgs(Args):
    """
    CondArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(ODArgs, self).__init__()
        """
        super(ODArgs, self).__init__()
        self.set_train_args()


class ODLoss(ClassifyLoss):

    def __init__(self):
        super(ODLoss, self).__init__()


def od_run(model_name: str, args: ODArgs, loss_obj: ODLoss):
    """
    od-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = ODArgs()
    loss_obj_ = ODLoss()
    od_run(args_.opts.model_name, args_, loss_obj_)
