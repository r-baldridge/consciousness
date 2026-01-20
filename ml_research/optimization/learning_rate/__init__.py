"""
Learning Rate Methods Module

This module contains research indices for learning rate scheduling
techniques and learning rate finding methods.

Key Categories:
    - Schedulers: Step decay, cosine annealing, warmup, OneCycleLR
    - LR Finding: Learning rate range test for optimal LR discovery

Learning Rate Philosophy:
    The learning rate is often the most important hyperparameter in training.
    These methods help find good learning rates and adapt them during training.
"""

from .schedulers import (
    STEP_DECAY,
    COSINE_ANNEALING,
    WARMUP,
    ONE_CYCLE_LR,
    step_decay_schedule,
    cosine_annealing_schedule,
    warmup_schedule,
    one_cycle_schedule,
)
from .lr_finder import (
    LR_RANGE_TEST,
    lr_range_test,
)

__all__ = [
    # Schedulers
    "STEP_DECAY",
    "COSINE_ANNEALING",
    "WARMUP",
    "ONE_CYCLE_LR",
    "step_decay_schedule",
    "cosine_annealing_schedule",
    "warmup_schedule",
    "one_cycle_schedule",
    # LR Finding
    "LR_RANGE_TEST",
    "lr_range_test",
]
