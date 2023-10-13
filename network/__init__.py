from .BoTNet import resnet50_bot, resnet101_bot, resnet152_bot
from .CondResNet import resnet_50_cond, resnet_101_cond
from .GhostNet import resnet_50_ghost, resnet_101_ghost, resnet_152_ghost
from .ODResNet import resnet18_od, resnet50_od
from .ResNeXt import resnet50, resnet101, resnet152, ResNeXt50_32x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
from .SCNet import ResNext_50_SC_32x4d, ResNext_101_SC_32x4d, ResNext_152_SC_32x4d

# model
__all__ = [
    # BoTNet
    'resnet50_bot',
    'resnet101_bot',
    'resnet152_bot',

    # CondNet
    'resnet_50_cond',
    'resnet_101_cond',

    # GhostNet
    'resnet_50_ghost',
    'resnet_101_ghost',
    'resnet_152_ghost',

    # ODNet
    'resnet18_od',
    'resnet50_od',

    # ResNet
    'resnet50',
    'resnet101',
    'resnet152',

    # ResNeXt
    'ResNeXt50_32x4d',
    'ResNeXt101_32x4d',
    'ResNeXt152_32x4d',
    
    # SCNet
    'ResNext_50_SC_32x4d',
    'ResNext_101_SC_32x4d',
    'ResNext_152_SC_32x4d'

]
