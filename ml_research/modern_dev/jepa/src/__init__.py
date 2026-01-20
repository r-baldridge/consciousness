"""
JEPA Source Module

Core implementation of Joint Embedding Predictive Architecture components.
"""

from .model import JEPA, JEPAConfig, MultiBlockMasking, IJEPAForClassification
from .layers import (
    PatchEmbed,
    PositionalEncoding,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    ContextEncoder,
    TargetEncoder,
    Predictor,
    VICRegLoss,
)

__all__ = [
    # Model
    "JEPA",
    "JEPAConfig",
    "MultiBlockMasking",
    "IJEPAForClassification",
    # Layers
    "PatchEmbed",
    "PositionalEncoding",
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "ContextEncoder",
    "TargetEncoder",
    "Predictor",
    "VICRegLoss",
]
