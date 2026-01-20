"""
Efficient Attention Methods

This module contains research index entries for efficient attention mechanisms
that reduce the quadratic complexity of standard self-attention.

Methods included:
- Sparse Attention: Longformer, BigBird (2020)
- Linear Attention: Linformer, Performer (2020)
- Flash Attention (2022): IO-aware exact attention
- Mamba (2023): Selective state space models
"""

from .sparse_attention import (
    get_longformer_info,
    get_bigbird_info,
)
from .linear_attention import (
    get_linformer_info,
    get_performer_info,
)
from .flash_attention import get_method_info as get_flash_attention_info
from .mamba import get_method_info as get_mamba_info

__all__ = [
    "get_longformer_info",
    "get_bigbird_info",
    "get_linformer_info",
    "get_performer_info",
    "get_flash_attention_info",
    "get_mamba_info",
]
