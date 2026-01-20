"""
CTM Source Module

Core implementation of Continuous Thought Machine components.
"""

from .model import CTM, CTMConfig, CTMEmbedding, CTMOutputHead
from .layers import NeuronLevelModel, SynchronizationLayer, TemporalHistory, AdaptiveHaltingMechanism

__all__ = [
    "CTM",
    "CTMConfig",
    "CTMEmbedding",
    "CTMOutputHead",
    "NeuronLevelModel",
    "SynchronizationLayer",
    "TemporalHistory",
    "AdaptiveHaltingMechanism",
]
