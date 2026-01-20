"""
Adaptive Optimization Methods Module

This module contains research indices for adaptive learning rate optimizers
that automatically adjust per-parameter learning rates based on gradient history.

Key Methods:
    - AdaGrad (2011): Per-parameter learning rates based on gradient accumulation
    - RMSProp: Exponential moving average of squared gradients
    - Adam (2014): Combines momentum with adaptive learning rates
    - AdamW: Adam with decoupled weight decay
    - LAMB: Layer-wise Adaptive Moments for large batch training
    - Lion (2023): Sign-based optimizer discovered via symbolic search

Evolution:
    SGD -> AdaGrad -> RMSProp -> Adam -> AdamW/LAMB/Lion
"""

from .adagrad import (
    ADAGRAD,
    adagrad_update,
)
from .rmsprop import (
    RMSPROP,
    rmsprop_update,
)
from .adam import (
    ADAM,
    adam_update,
)
from .adamw import (
    ADAMW,
    adamw_update,
)
from .lamb import (
    LAMB,
    lamb_update,
)
from .lion import (
    LION,
    lion_update,
)

__all__ = [
    # AdaGrad
    "ADAGRAD",
    "adagrad_update",
    # RMSProp
    "RMSPROP",
    "rmsprop_update",
    # Adam
    "ADAM",
    "adam_update",
    # AdamW
    "ADAMW",
    "adamw_update",
    # LAMB
    "LAMB",
    "lamb_update",
    # Lion
    "LION",
    "lion_update",
]
