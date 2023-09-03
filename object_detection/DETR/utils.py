import onnx
import onnx.utils
import torch
from torchvision.models import resnet18

from model import DETR


def export_model():
    # model = DETR(nums_head=8, hidden_dim=256, num_encoder_layers=6, num_decoder_layers=6, num_classes=91)
    model = resnet18()
    x = torch.randn(4, 3, 640, 640)
    model_file = 'model.onnx'
    # export model
    torch.onnx.export(
        model,
        x,
        model_file,
        export_params=True,
        opset_version=10,
    )

    # add dim info
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)


