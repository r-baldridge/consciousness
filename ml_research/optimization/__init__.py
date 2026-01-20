"""
Optimization Methods Module

This module contains research indices for optimization algorithms used in
machine learning and deep learning. Covers gradient descent variants,
adaptive optimizers, regularization techniques, and learning rate scheduling.

Key Categories:
    - Gradient Descent: SGD, Momentum, Nesterov Accelerated Gradient
    - Adaptive Optimizers: AdaGrad, RMSProp, Adam, AdamW, LAMB, Lion
    - Regularization: L1/L2, Dropout, Normalization, Data Augmentation
    - Learning Rate: Schedulers, Warmup, LR Range Test
"""

# Gradient Descent Methods
from .gradient_descent import (
    SGD,
    MOMENTUM,
    NESTEROV_ACCELERATED_GRADIENT,
    sgd_update,
    momentum_update,
    nesterov_update,
)

# Adaptive Optimizers
from .adaptive import (
    ADAGRAD,
    RMSPROP,
    ADAM,
    ADAMW,
    LAMB,
    LION,
    adagrad_update,
    rmsprop_update,
    adam_update,
    adamw_update,
    lamb_update,
    lion_update,
)

# Regularization Methods
from .regularization import (
    L1_REGULARIZATION,
    L2_REGULARIZATION,
    ELASTIC_NET,
    DROPOUT,
    SPATIAL_DROPOUT,
    DROPCONNECT,
    BATCH_NORMALIZATION,
    LAYER_NORMALIZATION,
    GROUP_NORMALIZATION,
    RMS_NORMALIZATION,
    MIXUP,
    CUTMIX,
    RANDAUGMENT,
    AUTOAUGMENT,
    l1_penalty,
    l2_penalty,
    elastic_net_penalty,
    batch_norm_forward,
    layer_norm_forward,
)

# Learning Rate Methods
from .learning_rate import (
    STEP_DECAY,
    COSINE_ANNEALING,
    WARMUP,
    ONE_CYCLE_LR,
    LR_RANGE_TEST,
    step_decay_schedule,
    cosine_annealing_schedule,
    warmup_schedule,
    one_cycle_schedule,
    lr_range_test,
)

__all__ = [
    # Gradient Descent
    "SGD",
    "MOMENTUM",
    "NESTEROV_ACCELERATED_GRADIENT",
    "sgd_update",
    "momentum_update",
    "nesterov_update",
    # Adaptive Optimizers
    "ADAGRAD",
    "RMSPROP",
    "ADAM",
    "ADAMW",
    "LAMB",
    "LION",
    "adagrad_update",
    "rmsprop_update",
    "adam_update",
    "adamw_update",
    "lamb_update",
    "lion_update",
    # Regularization
    "L1_REGULARIZATION",
    "L2_REGULARIZATION",
    "ELASTIC_NET",
    "DROPOUT",
    "SPATIAL_DROPOUT",
    "DROPCONNECT",
    "BATCH_NORMALIZATION",
    "LAYER_NORMALIZATION",
    "GROUP_NORMALIZATION",
    "RMS_NORMALIZATION",
    "MIXUP",
    "CUTMIX",
    "RANDAUGMENT",
    "AUTOAUGMENT",
    "l1_penalty",
    "l2_penalty",
    "elastic_net_penalty",
    "batch_norm_forward",
    "layer_norm_forward",
    # Learning Rate
    "STEP_DECAY",
    "COSINE_ANNEALING",
    "WARMUP",
    "ONE_CYCLE_LR",
    "LR_RANGE_TEST",
    "step_decay_schedule",
    "cosine_annealing_schedule",
    "warmup_schedule",
    "one_cycle_schedule",
    "lr_range_test",
]
