from .SENet import se_resnet_18, se_resnet_34, se_resnet_50, se_resnet_101, se_resnet_152, \
    se_resnext_50_32d4, se_resnext_101_32d4, se_resnext_152_32d4
from .SKNet import sk_resnet_50, sk_resnet_101, sk_resnet_152
from .CPVT import CPVT
# model
__all__ = [
    # SENet
    'se_resnet_18',
    'se_resnet_34',
    'se_resnet_50',
    'se_resnet_101',
    'se_resnet_152',
    'se_resnext_50_32d4',
    'se_resnext_101_32d4',
    'se_resnext_152_32d4',

    # SKNet
    'sk_resnet_50',
    'sk_resnet_101',
    'sk_resnet_152',

    # CPVT
    'CPVT'
]
