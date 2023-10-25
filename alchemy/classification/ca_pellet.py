import sys
if r'D:\projects\cv-deep-learning' not in sys.path:
    sys.path.append(r'D:\projects\cv-deep-learning')
from network import ca_mobilenet_v2_1, ca_mobilenet_v2_075, ca_mobilenet_v2_05
from alchemy.utils import Args, ClassifyLoss, basic_run


class CAMobileNetV2Args(Args):
    """
    MobileNet V2 + CA Args
    """

    def __init__(self):
        """
        if add new args, plz callback self.parser.add_argument before super(CAMobileNetV2Args, self).__init__()
        """
        super(CAMobileNetV2Args, self).__init__()
        self.set_train_args()


class CAMobileNetV2Loss(ClassifyLoss):

    def __init__(self):
        super(CAMobileNetV2Loss, self).__init__()


def mobilenet_run(model_name: str, args: CAMobileNetV2Args, loss_obj: CAMobileNetV2Loss):
    """
    mobilenet v2 + CA family training
    """
    model = globals().get(model_name, None)(num_classes=5, classifier=True)
    assert model is not None, f'{model_name} not defined'
    basic_run(model, model_name, args, loss_obj)


if __name__ == '__main__':
    """
    note: 
    Starting from the command line requires adding a model_name parameter
    """
    args_ = CAMobileNetV2Args()
    print(args_.opts)
    loss_obj_ = CAMobileNetV2Loss()
    mobilenet_run(args_.opts.model_name, args_, loss_obj_)
