import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import mobilenet_v2_1, mobilenet_v2_075, mobilenet_v2_05
from alchemy.utils import Args, ClassifyLoss, basic_run


class MobileNetV2Args(Args):
    """
    MobileNet V2 Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(MobileNetV2Args, self).__init__()
        """
        super(MobileNetV2Args, self).__init__()
        self.set_train_args()


class MobileNetV2Loss(ClassifyLoss):

    def __init__(self):
        super(MobileNetV2Loss, self).__init__()


def mobilenet_run(model_name: str, args: MobileNetV2Args, loss_obj: MobileNetV2Loss):
    """
    mobilenet v2 family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = MobileNetV2Args()
    print(args_.opts)
    loss_obj_ = MobileNetV2Loss()
    mobilenet_run(args_.opts.model_name, args_, loss_obj_)
