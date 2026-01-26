"""TRM source modules."""
from .model import TRM, TRMConfig, CodeRepairTRM, CodeRepairConfig, CodeRepairDeepRecursion
from .layers import (
    TRMBlock,
    DeepRecursion,
    QHead,
    OutputHead,
    GridEmbedding,
    MLPSequence,
    # Code repair extensions
    GridPositionalEncoding,
    GridAttention,
    RecursiveBlock,
    FeedForward,
    IterationController,
    RMSNorm,
)
from .losses import (
    TokenCrossEntropyLoss,
    DiffWeightedLoss,
    IntermediateSupervisionLoss,
    CombinedCodeRepairLoss,
    compute_perplexity,
    compute_accuracy,
    compute_diff_accuracy,
    compute_edit_distance_metrics,
    compute_sequence_accuracy,
)

__all__ = [
    # Original TRM
    "TRM",
    "TRMConfig",
    "TRMBlock",
    "DeepRecursion",
    "QHead",
    "OutputHead",
    "GridEmbedding",
    "MLPSequence",
    # Code Repair TRM
    "CodeRepairTRM",
    "CodeRepairConfig",
    "CodeRepairDeepRecursion",
    "GridPositionalEncoding",
    "GridAttention",
    "RecursiveBlock",
    "FeedForward",
    "IterationController",
    "RMSNorm",
    # Loss Functions
    "TokenCrossEntropyLoss",
    "DiffWeightedLoss",
    "IntermediateSupervisionLoss",
    "CombinedCodeRepairLoss",
    # Utility Functions
    "compute_perplexity",
    "compute_accuracy",
    "compute_diff_accuracy",
    "compute_edit_distance_metrics",
    "compute_sequence_accuracy",
]
