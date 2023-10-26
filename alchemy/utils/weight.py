"""
use to download pre-trained weight file of model and retrieve specified weight
"""
import typing as t

import torch

from alchemy.settings import PRETRAINED_MODEL
from torchvision.models import resnet18, resnet101


class PretrainedWeight(object):

    @staticmethod
    def get_weight(file_name: str, down_path: t.Optional[str] = None) -> str:
        if file_name in PRETRAINED_MODEL:
            return PRETRAINED_MODEL[file_name]
        else:
            # TODO: download weight through donw_path
            raise ValueError(f'please download weight: {file_name} ,then, put .pth into /alchmey/weights and '
                             f'add mapping of "{file_name}" and path in /alchmey/weights')

    def load_pretrained(self, model_name: str):
        """
        load pretrained weights, such as resnet18, resnet50,  resnet101 and others, then make adjustments and freezes.
        """
        weight_file = pretrained_weight.get_weight(model_name)
        weights = torch.load(weight_file)
        model = resnet101()
        ww = model.state_dict()
        ww['wss'] = torch.Tensor(0)
        model.load_state_dict(weights)
        s = 2


pretrained_weight = PretrainedWeight()

if __name__ == '__main__':
    pretrained_weight.load_pretrained('resnet101')


