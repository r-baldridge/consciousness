"""
State Space Methods Module

This module contains research indices for state space sequence models,
a paradigm that emerged as an efficient alternative to Transformers
for modeling long sequences.

Key Methods:
    - S4 (2021): Structured State Spaces with HiPPO initialization
    - Mamba (2023): Selective state spaces with input-dependent parameters
    - Jamba (2024): Hybrid Mamba-Transformer-MoE architecture

The state space approach models sequences as continuous-time dynamical systems:
    x'(t) = Ax(t) + Bu(t)    (state dynamics)
    y(t)  = Cx(t) + Du(t)    (observation)

Key advantages:
    - O(L) complexity vs O(L^2) for attention
    - O(1) inference memory vs O(L) for attention KV cache
    - Naturally handles very long sequences (100K+ tokens)
"""

from .s4 import (
    STRUCTURED_STATE_SPACE,
    state_space_continuous,
    discretize_bilinear,
    compute_ssm_kernel,
    hippo_matrix,
    dplr_representation,
    S4Architecture,
    S4_VARIANTS,
)
from .mamba import (
    MAMBA,
    selective_scan,
    selection_mechanism,
    parallel_associative_scan,
    mamba_block,
    MambaArchitecture,
    MAMBA_VARIANTS,
    MAMBA_INSIGHTS,
)
from .jamba import (
    JAMBA,
    jamba_layer_pattern,
    hybrid_kv_cache,
    jamba_moe_configuration,
    jamba_attention_layer,
    JambaArchitecture,
    JAMBA_COMPARISONS,
    JAMBA_DESIGN_RATIONALE,
)

__all__ = [
    # S4: Structured State Spaces
    "STRUCTURED_STATE_SPACE",
    "state_space_continuous",
    "discretize_bilinear",
    "compute_ssm_kernel",
    "hippo_matrix",
    "dplr_representation",
    "S4Architecture",
    "S4_VARIANTS",
    # Mamba: Selective State Spaces
    "MAMBA",
    "selective_scan",
    "selection_mechanism",
    "parallel_associative_scan",
    "mamba_block",
    "MambaArchitecture",
    "MAMBA_VARIANTS",
    "MAMBA_INSIGHTS",
    # Jamba: Hybrid Architecture
    "JAMBA",
    "jamba_layer_pattern",
    "hybrid_kv_cache",
    "jamba_moe_configuration",
    "jamba_attention_layer",
    "JambaArchitecture",
    "JAMBA_COMPARISONS",
    "JAMBA_DESIGN_RATIONALE",
]
