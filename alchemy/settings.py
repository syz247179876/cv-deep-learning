import os
from pathlib import Path
ROOT = os.path.dirname(__file__)
PROJECT_ROOT = Path('../../').resolve()

BAR_TRAIN_COLOR = 'MAGENTA'
BAR_VALIDATE_COLOR = 'YELLOW'
BAR_TEST_COLOR = 'GREEN'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

COCO_MEAN = [0.471, 0.448, 0.408]
COCO_STD = [0.234, 0.239, 0.242]

FLOWER_MEAN = [0.476, 0.442, 0.349]
FLOWER_STD = [0.260, 0.237, 0.269]

PRETRAINED_MODEL = {
    'resnet18': os.path.join(ROOT, 'weights/resnet18-f37072fd.pth'),
    'resnet50': os.path.join(ROOT, 'weights/resnet50-11ad3fa6.pth'),
    'resnet101': os.path.join(ROOT, 'weights/resnet101-cd907fc2.pth'),
    'resnet152': os.path.join(ROOT, 'weights/resnet152-f82ba261.pth'),
    'se_resnet_50': os.path.join(ROOT, 'weights/resnet50-11ad3fa6.pth'),
    'sk_resnet_50': os.path.join(ROOT, 'weights/resnext50_32x4d-1a0047aa.pth'),
    'ca_mobilenet_v2_1': os.path.join(ROOT, 'weights/mbv2_ca1.0.pth'),
    'mobilenet_v2_1': os.path.join(ROOT, 'weights/mobilenetv2_1.0-0c6065bc.pth'),
    'mobilenet_v2_075': os.path.join(ROOT, 'weights/mobilenetv2_0.75-dace9791.pth'),
    'mobilenet_v2_05': os.path.join(ROOT, 'weights/mobilenetv2_0.5-eaa6f9ad.pth'),
    'ResNext_50_SC_32x4d': os.path.join(ROOT, 'weights/resnext50_32x4d-1a0047aa.pth'),
    'ResNeXt50_32x4d': os.path.join(ROOT, 'weights/resnext50_32x4d-1a0047aa.pth'),
}

# network structure mapping
NET_MAP = {
    'resnet50': os.path.join(ROOT, 'net_maps/resnet50.json'),
    'resnet101': os.path.join(ROOT, 'net_maps/resnet101.json'),
    'resnet152': os.path.join(ROOT, 'net_maps/resnet152.json'),
    'se_resnet_50': os.path.join(ROOT, 'net_maps/resnet50.json'),
    'sk_resnet_50': os.path.join(ROOT, 'net_maps/resnext50.json'),
    'ResNeXt50_32x4d': os.path.join(ROOT, 'net_maps/resnext50.json'),
    'ResNext_50_SC_32x4d': os.path.join(ROOT, 'net_maps/resnext50.json'),
    'ca_mobilenet_v2_1': os.path.join(ROOT, 'net_maps/ca_mobilenet_v2_1.json'),
    'mobilenet_v2_1': os.path.join(ROOT, 'net_maps/mobilenet_v2_1.json'),
}
