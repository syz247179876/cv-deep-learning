import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import csp_resnet_50
from alchemy.utils import Args, ClassifyLoss, basic_run


class CSPArgs(Args):
    """
    CSP Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(CSPArgs, self).__init__()
        """
        super(CSPArgs, self).__init__()
        self.set_train_args()


class CSPLoss(ClassifyLoss):

    def __init__(self):
        super(CSPLoss, self).__init__()


def deform_run(model_name: str, args: CSPArgs, loss_obj: CSPLoss):
    """
    deform-resnet family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = CSPArgs()
    loss_obj_ = CSPLoss()
    deform_run(args_.opts.model_name, args_, loss_obj_)
