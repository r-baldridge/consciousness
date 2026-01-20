"""
Mixture of Experts (MoE) Methods Module

This module contains research indices for Mixture of Experts architectures,
a paradigm for conditional computation that scales model capacity without
proportionally increasing compute.

Key Methods:
    - MoE Basics (1991-): Core concepts, gating, load balancing
    - Switch Transformer (2021): Simplified top-1 routing, trillion-scale
    - Mixtral (2023): Production 8x7B model with top-2 routing

Core Concept:
    y = sum_{i=1}^{N} g(x)_i * E_i(x)

    Where g(x) is a gating function and E_i are expert networks.
    Sparse gating (top-k) enables scaling parameters without scaling compute.

Key Advantages:
    - Decouple parameters from compute (100B params, 10B active)
    - Specialized experts for different input types
    - Sub-linear scaling with respect to parameters
"""

from .moe_basics import (
    MIXTURE_OF_EXPERTS,
    dense_moe_forward,
    sparse_moe_forward,
    gating_function,
    load_balancing_loss,
    expert_capacity,
    MoEArchitecture,
    MOE_CONCEPTS,
    MOE_HISTORY,
)
from .switch_transformer import (
    SWITCH_TRANSFORMER,
    switch_routing,
    expert_capacity_computation,
    load_balancing_loss_switch,
    selective_precision,
    switch_layer_forward,
    SwitchTransformerArchitecture,
    SWITCH_SCALING,
    SWITCH_IMPLEMENTATION,
)
from .mixtral import (
    MIXTRAL,
    mixtral_routing,
    mixtral_expert_ffn,
    sliding_window_attention,
    mixtral_moe_layer,
    MixtralArchitecture,
    MIXTRAL_BENCHMARKS,
    MIXTRAL_VARIANTS,
    MIXTRAL_DEPLOYMENT,
)

__all__ = [
    # MoE Basics
    "MIXTURE_OF_EXPERTS",
    "dense_moe_forward",
    "sparse_moe_forward",
    "gating_function",
    "load_balancing_loss",
    "expert_capacity",
    "MoEArchitecture",
    "MOE_CONCEPTS",
    "MOE_HISTORY",
    # Switch Transformer
    "SWITCH_TRANSFORMER",
    "switch_routing",
    "expert_capacity_computation",
    "load_balancing_loss_switch",
    "selective_precision",
    "switch_layer_forward",
    "SwitchTransformerArchitecture",
    "SWITCH_SCALING",
    "SWITCH_IMPLEMENTATION",
    # Mixtral
    "MIXTRAL",
    "mixtral_routing",
    "mixtral_expert_ffn",
    "sliding_window_attention",
    "mixtral_moe_layer",
    "MixtralArchitecture",
    "MIXTRAL_BENCHMARKS",
    "MIXTRAL_VARIANTS",
    "MIXTRAL_DEPLOYMENT",
]
