import os.path
import sys

vit_module_path = os.path.join(os.path.dirname(__file__), 'vision_transformer')
if vit_module_path not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vision_transformer'))

# from .vision_transformer import PatchEmbed

""" Model Visualize """
# if __name__ == '__main__':
#     import torch
#     from vision_transformer.model import ModelFactory
#     from torchsummary import summary
#     model = ModelFactory(model_name='base').model
#     model = model.to(0)
#     x = torch.rand(3, 224, 224)
#     x = x.to(0)
#     summary(model, x.squeeze(0).shape)
