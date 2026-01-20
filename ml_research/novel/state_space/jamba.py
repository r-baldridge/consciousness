"""
Jamba: Hybrid Mamba-Transformer Architecture - 2024

AI21 Labs' hybrid architecture combining Mamba state space layers with
Transformer attention layers and Mixture of Experts (MoE), achieving
strong performance with efficient long-context handling.

Paper: "Jamba: A Hybrid Transformer-Mamba Language Model"
arXiv: 2403.19887

Key Innovation:
    Hybrid architecture that combines:
    - Mamba layers: O(L) complexity for long-range modeling
    - Attention layers: Precise retrieval and in-context learning
    - MoE layers: Increased capacity with sparse activation

    Typical ratio: 7:1 Mamba to Attention layers
    MoE in some layers for capacity expansion without compute increase

Mathematical Formulation:
    Layer types alternate in a structured pattern:
    [Mamba] -> [Mamba] -> ... -> [Mamba] -> [Attention+MoE] -> repeat

    Mamba layers handle long-range dependencies efficiently
    Attention layers handle precise recall and complex reasoning
    MoE provides capacity for diverse knowledge storage
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

JAMBA = MLMethod(
    method_id="jamba_2024",
    name="Jamba",
    year=2024,

    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RNN_LINE],

    authors=["AI21 Labs"],
    paper_title="Jamba: A Hybrid Transformer-Mamba Language Model",
    paper_url="https://arxiv.org/abs/2403.19887",

    key_innovation=(
        "First production-scale hybrid combining Mamba, Attention, and MoE. "
        "Achieves 3x throughput improvement over Mixtral while supporting 256K context. "
        "Key insight: different layer types excel at different tasks - use each where "
        "it's strongest. Mamba for long-range, attention for recall, MoE for capacity."
    ),

    mathematical_formulation=r"""
Jamba Block Structure:
    Jamba Layer = [Mamba x a] + [Attention + MoE x b] repeated

    Where typically:
        a = 7 (Mamba layers per attention layer)
        b = 1 (attention layers with MoE)
        Ratio: 7:1 Mamba to Attention

Mamba Component (same as standard Mamba):
    x_t = A_bar(u_t) * x_{t-1} + B_bar(u_t) * u_t
    y_t = C(u_t) * x_t
    Complexity: O(L) per layer

Attention Component (standard multi-head attention):
    Q, K, V = Linear(x)
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    Complexity: O(L^2) per layer, but only 1 per 8 layers

MoE Component (sparse mixture):
    y = sum_{i in top-k} g_i(x) * Expert_i(x)
    Where g(x) = softmax(Linear(x))
    Top-k selection: typically k=2 out of 16 experts

Memory Efficiency:
    Mamba: O(d_state) per layer state (small constant)
    Attention: O(L * d_model) KV cache (scales with length)
    Hybrid: Mostly Mamba state + few attention KV caches

    Total KV cache: ~8x smaller than pure Transformer
""",

    predecessors=["mamba_2023", "mixtral_2023", "transformer_2017"],
    successors=[],

    tags=["hybrid", "mamba", "transformer", "moe", "long-context", "efficient"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def jamba_layer_pattern(num_layers, mamba_ratio=7, attention_moe_ratio=1):
    """
    Define Jamba's layer composition pattern.

    Jamba interleaves Mamba and Attention layers at a fixed ratio,
    with MoE applied to attention layers.

    Args:
        num_layers: Total number of layers
        mamba_ratio: Number of Mamba layers per attention layer
        attention_moe_ratio: Number of attention layers with MoE

    Returns:
        Dictionary describing layer pattern
    """
    return {
        "pattern": "[Mamba] * 7 + [Attention + MoE] * 1",
        "total_ratio": "7:1 Mamba to Attention",
        "moe_placement": "MoE applied to attention layers (not Mamba)",
        "reasoning": {
            "mamba_majority": "Efficient O(L) processing for most computation",
            "sparse_attention": "Precise recall capability where needed",
            "moe_scaling": "Increased capacity without proportional compute"
        },
        "example_32_layers": [
            "Layers 1-7: Mamba",
            "Layer 8: Attention + MoE",
            "Layers 9-15: Mamba",
            "Layer 16: Attention + MoE",
            "Layers 17-23: Mamba",
            "Layer 24: Attention + MoE",
            "Layers 25-31: Mamba",
            "Layer 32: Attention + MoE"
        ]
    }


def hybrid_kv_cache(seq_len, d_model, num_mamba_layers, num_attention_layers, d_state=16):
    """
    Compare KV cache requirements for hybrid vs pure Transformer.

    Jamba's hybrid design dramatically reduces KV cache by using
    Mamba (fixed state size) for most layers.

    Args:
        seq_len: Sequence length L
        d_model: Model dimension
        num_mamba_layers: Number of Mamba layers
        num_attention_layers: Number of attention layers
        d_state: Mamba state dimension

    Returns:
        Dictionary comparing memory requirements
    """
    return {
        "pure_transformer_kv_cache": f"O(num_layers * L * d_model) = O({32} * L * {d_model})",
        "jamba_attention_kv": f"O(num_attn_layers * L * d_model) = O({num_attention_layers} * L * {d_model})",
        "jamba_mamba_state": f"O(num_mamba_layers * d_state * d_model) = O({num_mamba_layers} * {d_state} * {d_model})",
        "reduction_factor": "~8x smaller (assuming 7:1 ratio)",
        "practical_benefit": "256K context on single 80GB GPU",
        "note": "Mamba state is O(1) in sequence length, attention KV is O(L)"
    }


def jamba_moe_configuration(d_model, num_experts=16, top_k=2, expert_capacity=None):
    """
    MoE configuration in Jamba's attention layers.

    Jamba uses sparse MoE in its attention layers to increase
    model capacity without proportional compute increase.

    Args:
        d_model: Model dimension
        num_experts: Total number of experts (typically 16)
        top_k: Number of active experts per token (typically 2)
        expert_capacity: Maximum tokens per expert

    Returns:
        Dictionary describing MoE configuration
    """
    return {
        "num_experts": num_experts,
        "active_experts": top_k,
        "routing": "Top-k softmax routing with load balancing",
        "capacity_factor": "1.25 (allows 25% buffer over uniform)",
        "expert_structure": f"FFN: Linear({d_model} -> 4*{d_model}) -> GELU -> Linear(4*{d_model} -> {d_model})",
        "total_vs_active_params": f"{num_experts}x params, {top_k}x compute",
        "placement": "Only in attention layers, not Mamba layers"
    }


def jamba_attention_layer(x, use_moe=True):
    """
    Attention layer with optional MoE in Jamba.

    Standard multi-head attention followed by MoE FFN.

    Args:
        x: Input [batch, seq_len, d_model]
        use_moe: Whether to use MoE for FFN

    Returns:
        Dictionary describing attention layer computation
    """
    return {
        "attention": {
            "qkv_proj": "Q, K, V = Linear(x)",
            "attention": "attn = softmax(QK^T / sqrt(d_k)) V",
            "output_proj": "out = Linear(attn)",
            "residual": "x = x + out"
        },
        "ffn_or_moe": {
            "if_moe": {
                "routing": "gates = softmax(Linear(x))",
                "selection": "experts = top_k(gates, k=2)",
                "compute": "out = sum(g_i * Expert_i(x)) for i in top_k",
                "residual": "x = x + out"
            },
            "if_dense": {
                "ffn": "out = GELU(Linear(x)) @ Linear",
                "residual": "x = x + out"
            }
        }
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class JambaArchitecture:
    """Reference architecture for Jamba model."""

    d_model: int = 4096
    num_layers: int = 32
    num_attention_layers: int = 4  # 32 / 8 = 4
    num_mamba_layers: int = 28  # 32 - 4 = 28
    num_experts: int = 16
    top_k: int = 2
    d_state: int = 16

    @staticmethod
    def layer_breakdown() -> str:
        """Detailed layer breakdown."""
        return """
Jamba-52B Architecture:
    Total Parameters: 52B (12B active)
    Context Length: 256K tokens
    Layers: 32 total
        - 28 Mamba layers (7/8 of layers)
        - 4 Attention + MoE layers (1/8 of layers)

    Mamba Layers:
        d_model: 4096
        d_state: 16
        d_conv: 4
        expand: 2

    Attention Layers:
        d_model: 4096
        num_heads: 32
        head_dim: 128
        KV cache: Only for 4 layers (not 32)

    MoE Configuration:
        num_experts: 16
        top_k: 2
        Expert: Standard FFN

    Memory at 256K context:
        Pure Transformer: ~512GB KV cache
        Jamba: ~64GB KV cache (8x reduction)
"""

    @staticmethod
    def forward_pass() -> str:
        """Forward pass pseudocode."""
        return """
Jamba Forward Pass:
    Input: tokens [batch, seq_len]

    x = Embedding(tokens)  # [batch, seq_len, d_model]

    mamba_count = 0
    for layer_idx in range(num_layers):
        x = RMSNorm(x)

        if is_attention_layer(layer_idx):  # Every 8th layer
            # Attention + MoE
            x = x + Attention(x)
            x = RMSNorm(x)
            x = x + MoE_FFN(x)  # Sparse expert routing
        else:
            # Mamba
            x = x + Mamba_Block(x)
            mamba_count += 1

    x = RMSNorm(x)
    logits = LM_Head(x)

    return logits
"""


# =============================================================================
# Comparison Reference
# =============================================================================

JAMBA_COMPARISONS = {
    "vs_transformer": {
        "throughput": "3x higher (due to linear Mamba layers)",
        "kv_cache": "8x smaller (only 4 attention layers vs 32)",
        "quality": "Comparable (attention handles precision tasks)",
        "max_context": "256K practical (vs 128K typical)"
    },
    "vs_mamba": {
        "recall_accuracy": "Higher (attention layers help)",
        "in_context_learning": "Better (attention provides ICL)",
        "throughput": "Slightly lower (attention layers)",
        "best_of_both": "Mamba efficiency + Attention precision"
    },
    "vs_mixtral": {
        "throughput": "3x higher",
        "active_params": "12B vs 12B (similar)",
        "total_params": "52B vs 47B (similar)",
        "context": "256K vs 32K (8x longer)"
    }
}


# =============================================================================
# Design Rationale
# =============================================================================

JAMBA_DESIGN_RATIONALE = {
    "why_hybrid": (
        "Pure Mamba struggles with tasks requiring precise recall (like retrieval). "
        "Pure Transformers scale quadratically with sequence length. "
        "Hybrid gets Mamba's efficiency for most computation while maintaining "
        "attention's precision for critical recall tasks."
    ),
    "why_7_to_1_ratio": (
        "Empirically found optimal balance. Too few Mamba layers loses efficiency. "
        "Too few attention layers hurts recall tasks. 7:1 gives near-linear scaling "
        "while maintaining Transformer-quality reasoning."
    ),
    "why_moe_on_attention_only": (
        "MoE adds routing overhead and memory. Placing it only on attention layers "
        "(which are sparse in the architecture) maximizes capacity benefit while "
        "minimizing overhead. Mamba layers are fast; don't slow them down."
    ),
    "scaling_properties": (
        "The hybrid design scales well: add more Mamba layers for efficiency, "
        "add more experts for capacity, adjust attention ratio for precision. "
        "Each component scales independently."
    )
}
