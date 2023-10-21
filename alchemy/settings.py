import os
ROOT = os.path.dirname(__file__)

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
    'resnet50': os.path.join(ROOT, 'weights/resnet50-0676ba61.pth'),
    'resnet101': os.path.join(ROOT, 'weights/resnet101-63fe2227.pth'),
    'resnet152': os.path.join(ROOT, 'weights/resnet152-394f9c45.pth'),
}
