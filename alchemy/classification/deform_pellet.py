import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import resnet_50_deform, resnet_101_deform
from alchemy.utils import Args, ClassifyLoss, basic_run


class DeformArgs(Args):
    """
    DeformArgs Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(DeformArgs, self).__init__()
        """
        super(DeformArgs, self).__init__()
        self.set_train_args()


class DeformLoss(ClassifyLoss):

    def __init__(self):
        super(DeformLoss, self).__init__()


def deform_run(model_name: str, args: DeformArgs, loss_obj: DeformLoss):
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
    args_ = DeformArgs()
    loss_obj_ = DeformLoss()
    deform_run(args_.opts.model_name, args_, loss_obj_)
