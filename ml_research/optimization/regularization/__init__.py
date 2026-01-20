"""
Regularization Methods Module

This module contains research indices for regularization techniques
used to prevent overfitting and improve generalization in neural networks.

Key Categories:
    - Weight Regularization: L1, L2, Elastic Net
    - Dropout: Standard, Spatial, DropConnect
    - Normalization: BatchNorm, LayerNorm, GroupNorm, RMSNorm
    - Data Augmentation: Mixup, CutMix, RandAugment, AutoAugment

Regularization Philosophy:
    All regularization methods constrain the model's capacity to prevent
    memorizing training data while maintaining the ability to generalize.
"""

from .l1_l2 import (
    L1_REGULARIZATION,
    L2_REGULARIZATION,
    ELASTIC_NET,
    l1_penalty,
    l2_penalty,
    elastic_net_penalty,
)
from .dropout import (
    DROPOUT,
    SPATIAL_DROPOUT,
    DROPCONNECT,
    dropout_forward,
    spatial_dropout_forward,
)
from .batch_norm import (
    BATCH_NORMALIZATION,
    LAYER_NORMALIZATION,
    GROUP_NORMALIZATION,
    RMS_NORMALIZATION,
    batch_norm_forward,
    layer_norm_forward,
    group_norm_forward,
    rms_norm_forward,
)
from .data_augmentation import (
    MIXUP,
    CUTMIX,
    RANDAUGMENT,
    AUTOAUGMENT,
    mixup_data,
    cutmix_data,
)

__all__ = [
    # Weight Regularization
    "L1_REGULARIZATION",
    "L2_REGULARIZATION",
    "ELASTIC_NET",
    "l1_penalty",
    "l2_penalty",
    "elastic_net_penalty",
    # Dropout
    "DROPOUT",
    "SPATIAL_DROPOUT",
    "DROPCONNECT",
    "dropout_forward",
    "spatial_dropout_forward",
    # Normalization
    "BATCH_NORMALIZATION",
    "LAYER_NORMALIZATION",
    "GROUP_NORMALIZATION",
    "RMS_NORMALIZATION",
    "batch_norm_forward",
    "layer_norm_forward",
    "group_norm_forward",
    "rms_norm_forward",
    # Data Augmentation
    "MIXUP",
    "CUTMIX",
    "RANDAUGMENT",
    "AUTOAUGMENT",
    "mixup_data",
    "cutmix_data",
]
