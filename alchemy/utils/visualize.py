"""
This file is used to visualize feature maps
"""
import cv2
import numpy as np
import torch.nn as nn
import torch
import typing as t

from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from torchvision.models import resnet50

from Attention import se_resnet_50
from alchemy.settings import PROJECT_ROOT

ALCHEMY_ROOT = Path('../').resolve()


class Visualize(object):

    def __init__(
            self,
            model: nn.Module,
    ):
        self.model = model

    @staticmethod
    def read_img(img_name: str):
        img_path = str(ALCHEMY_ROOT / 'images' / img_name)
        origin_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    @staticmethod
    def preprocess() -> t.Tuple[transforms.Compose, transforms.Normalize]:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
        ])
        new_input = transforms.Normalize([0.476, 0.442, 0.349], [0.260, 0.237, 0.269])
        return trans, new_input

    def visualize_feature_map(self, img_name: str, layer_name: nn.Module):
        img = self.read_img(img_name)
        trans, new_input = self.preprocess()
        crop_img = trans(img)
        input_tensor = new_input(crop_img).unsqueeze(0)

        canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)
        canvas_img = np.float32(canvas_img) / 255
        with GradCAMPlusPlus(model=self.model, target_layers=[layer_name], use_cuda=True) as cam:
            garyscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization_img = show_cam_on_image(canvas_img, garyscale_cam, use_rgb=False)
        cv2.imshow('feature_map', visualization_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    model = se_resnet_50(num_classes=5)
    model_dict = torch.load(PROJECT_ROOT / r'checkpoints_dir/se_resnet_50-epoch63-0.7984-2023-11-01-pretrained.pth')
    model.load_state_dict(model_dict.get('model'))
    model.eval()
    v = Visualize(model)
    v.visualize_feature_map('1.jpg', model.layer4[-1].bn)
