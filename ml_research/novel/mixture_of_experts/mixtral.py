"""
Mixtral of Experts - 2023

Mistral AI's production mixture of experts model achieving GPT-3.5 level
performance with 8x7B architecture. Uses top-2 routing from 8 experts,
resulting in 46.7B total parameters with 12.9B active per forward pass.

Paper: "Mixtral of Experts"
arXiv: 2401.04088

Key Features:
    - 8 expert networks, each 7B parameters
    - Top-2 routing: 2 experts active per token
    - Sliding window attention (4096 tokens)
    - 46.7B total params, 12.9B active params
    - Matches or exceeds GPT-3.5 / LLaMA 2 70B

Mathematical Formulation:
    y = sum_{i in top-2} softmax(router(x))_i * Expert_i(x)

    Each Expert is a standard FFN with ~7B parameters
    Routing is learned end-to-end with standard MoE auxiliary losses
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

MIXTRAL = MLMethod(
    method_id="mixtral_2023",
    name="Mixtral of Experts",
    year=2023,

    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],

    authors=["Mistral AI"],
    paper_title="Mixtral of Experts",
    paper_url="https://arxiv.org/abs/2401.04088",

    key_innovation=(
        "First open-weights production MoE model to match GPT-3.5 quality. "
        "Clean 8x7B architecture with top-2 routing achieves excellent "
        "quality/compute trade-off. Demonstrated MoE is viable for "
        "production deployment, not just research experiments."
    ),

    mathematical_formulation=r"""
Mixtral Architecture:
    Total Parameters: 46.7B
    Active Parameters: 12.9B (per forward pass)
    Architecture: 8 x 7B experts with top-2 routing

Top-2 Routing:
    router_logits = Linear(d_model -> 8)(x)         # [B, L, 8]
    router_probs = softmax(router_logits)           # [B, L, 8]
    top2_probs, top2_idx = top_k(router_probs, k=2) # [B, L, 2]
    top2_probs = top2_probs / sum(top2_probs)       # Renormalize

Expert Computation:
    y = sum_{i in top2_idx} top2_probs[i] * Expert_i(x)

    Expert_i(x) = SwiGLU FFN with ~7B parameters
        = W_down * (SiLU(W_gate * x) * (W_up * x))

Sliding Window Attention (SWA):
    Attention mask: token t attends to [t-W, t] where W=4096
    Effective context: layer * window_size = 32 * 4096 = 131K tokens

    Rolling KV Cache:
        Only store last W tokens in KV cache
        Memory: O(W) instead of O(L) for full attention

Model Configuration:
    d_model: 4096
    n_layers: 32
    n_heads: 32
    head_dim: 128
    n_experts: 8
    n_experts_per_tok: 2
    window_size: 4096
    vocab_size: 32000
""",

    predecessors=["mistral_7b_2023", "switch_transformer_2021", "llama_2023"],
    successors=["mixtral_8x22b_2024"],

    tags=["mixture-of-experts", "open-weights", "production", "llm", "efficient"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def mixtral_routing(x, router_weights, num_experts=8, top_k=2):
    """
    Mixtral's top-2 routing from 8 experts.

    Each token selects 2 experts, weighted by normalized softmax scores.

    Args:
        x: Input [batch, seq_len, d_model]
        router_weights: Routing weights [d_model, num_experts]
        num_experts: Number of experts (8 for Mixtral)
        top_k: Number of active experts (2 for Mixtral)

    Returns:
        Dictionary describing Mixtral routing
    """
    return {
        "step1": f"router_logits = x @ router_weights  # [B, L, {num_experts}]",
        "step2": "router_probs = softmax(router_logits, dim=-1)",
        "step3": f"top_k_probs, top_k_idx = topk(router_probs, k={top_k})",
        "step4": "top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)",
        "output": "(top_k_idx, top_k_probs)  # [B, L, 2] each",
        "interpretation": (
            "Each token gets routed to 2 experts. The routing weights are "
            "renormalized to sum to 1, so y = w1*E1(x) + w2*E2(x) where w1+w2=1"
        )
    }


def mixtral_expert_ffn(x, W_gate, W_up, W_down):
    """
    Mixtral's SwiGLU expert FFN architecture.

    Each expert uses gated linear unit with SiLU activation.

    Args:
        x: Input [num_tokens, d_model]
        W_gate: Gate projection [d_model, d_ff]
        W_up: Up projection [d_model, d_ff]
        W_down: Down projection [d_ff, d_model]

    Returns:
        Dictionary describing expert computation
    """
    return {
        "swiglu": "y = W_down @ (SiLU(W_gate @ x) * (W_up @ x))",
        "components": {
            "gate": "g = SiLU(x @ W_gate)  # [tokens, d_ff]",
            "up": "u = x @ W_up            # [tokens, d_ff]",
            "gated": "h = g * u            # Element-wise gating",
            "down": "y = h @ W_down        # [tokens, d_model]"
        },
        "dimensions": {
            "d_model": 4096,
            "d_ff": 14336,  # ~3.5x d_model for SwiGLU
            "params_per_expert": "4096*14336*3 = ~175M (gate+up+down)"
        },
        "silu": "SiLU(x) = x * sigmoid(x)  # Swish activation"
    }


def sliding_window_attention(Q, K, V, window_size=4096):
    """
    Mixtral's sliding window attention mechanism.

    Each token attends only to the previous W tokens, enabling
    efficient long-context processing with bounded memory.

    Args:
        Q: Query [batch, heads, seq_len, head_dim]
        K: Key [batch, heads, seq_len, head_dim]
        V: Value [batch, heads, seq_len, head_dim]
        window_size: Attention window size W

    Returns:
        Dictionary describing sliding window attention
    """
    return {
        "standard_attention": "Attention(Q,K,V) = softmax(QK^T/sqrt(d)) V",
        "sliding_window": {
            "mask": "token t attends to tokens [max(0, t-W+1), t]",
            "window_size": window_size,
            "implementation": (
                "Create band diagonal mask of width W. "
                "Can be implemented efficiently with custom kernels."
            )
        },
        "effective_context": {
            "per_layer": f"{window_size} tokens",
            "multi_layer": f"layer * W = 32 * {window_size} = 131K tokens",
            "explanation": (
                "Information propagates W tokens per layer. "
                "With 32 layers, token 0 can influence token 131K."
            )
        },
        "kv_cache_benefit": {
            "full_attention": "O(seq_len) per layer",
            "sliding_window": f"O({window_size}) per layer",
            "at_100k_tokens": f"~25x memory reduction"
        }
    }


def mixtral_moe_layer(x, experts, router):
    """
    Full Mixtral MoE layer computation.

    Args:
        x: Input [batch, seq_len, d_model]
        experts: List of 8 SwiGLU FFN experts
        router: Router network [d_model -> 8]

    Returns:
        Dictionary describing full MoE layer
    """
    return {
        "step1_attention": {
            "norm": "x_norm = RMSNorm(x)",
            "attn": "attn_out = SlidingWindowAttention(x_norm)",
            "residual": "x = x + attn_out"
        },
        "step2_moe": {
            "norm": "x_norm = RMSNorm(x)",
            "route": "top2_idx, top2_weights = router(x_norm)",
            "compute": """
                y = zeros_like(x_norm)
                for i in range(8):
                    mask = (top2_idx == i).any(dim=-1)
                    tokens = x_norm[mask]
                    expert_out = Expert_i(tokens)
                    weights = top2_weights[mask, top2_idx[mask] == i]
                    y[mask] += weights * expert_out
            """,
            "residual": "x = x + y"
        },
        "output": "x [batch, seq_len, d_model]"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class MixtralArchitecture:
    """Reference architecture for Mixtral 8x7B."""

    # Model dimensions
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 128
    d_ff: int = 14336

    # MoE configuration
    n_experts: int = 8
    n_experts_per_tok: int = 2

    # Attention configuration
    window_size: int = 4096

    # Vocabulary
    vocab_size: int = 32000

    @staticmethod
    def parameter_count() -> Dict:
        """Detailed parameter count breakdown."""
        return {
            "embedding": {
                "token_emb": "32000 * 4096 = 131M",
                "total": "131M"
            },
            "per_layer": {
                "attention": {
                    "q_proj": "4096 * 4096 = 16.7M",
                    "k_proj": "4096 * 1024 = 4.2M (GQA)",
                    "v_proj": "4096 * 1024 = 4.2M (GQA)",
                    "o_proj": "4096 * 4096 = 16.7M",
                    "total": "41.8M"
                },
                "moe": {
                    "router": "4096 * 8 = 33K",
                    "per_expert": "4096 * 14336 * 3 = 176M",
                    "all_experts": "176M * 8 = 1.4B",
                    "total": "1.4B"
                },
                "norms": "4096 * 2 = 8K",
                "layer_total": "1.44B"
            },
            "output_head": "4096 * 32000 = 131M",
            "total": {
                "all_params": "131M + 32*1.44B + 131M = 46.7B",
                "active_params": "131M + 32*(41.8M + 176M*2) + 131M = 12.9B"
            }
        }

    @staticmethod
    def model_card() -> str:
        """Mixtral model card summary."""
        return """
Mixtral 8x7B Model Card:
    Release: December 2023
    License: Apache 2.0

    Architecture:
        Type: Sparse Mixture of Experts
        Total Parameters: 46.7B
        Active Parameters: 12.9B
        Experts: 8 x ~7B each
        Active Experts: 2 per token

    Context:
        Window Size: 4096 (sliding window)
        Effective: 32K+ (stacked windows)

    Performance:
        - Matches/exceeds LLaMA 2 70B on most benchmarks
        - Matches GPT-3.5 on many tasks
        - 6x faster inference than 70B dense model

    Efficiency:
        - 12.9B FLOPs per token (vs 70B for comparable quality)
        - Sliding window reduces KV cache memory
        - Well-suited for production deployment
"""


# =============================================================================
# Performance Comparison
# =============================================================================

MIXTRAL_BENCHMARKS = {
    "vs_llama2_70b": {
        "mmlu": "70.6 (Mixtral) vs 69.8 (LLaMA 2 70B)",
        "gsm8k": "74.4 vs 56.8 (major improvement)",
        "humaneval": "40.2 vs 29.9",
        "arc_c": "65.3 vs 64.6",
        "inference_speed": "~6x faster"
    },
    "vs_gpt35": {
        "mmlu": "70.6 (Mixtral) vs ~70 (GPT-3.5)",
        "gsm8k": "74.4 vs ~57",
        "humaneval": "40.2 vs ~48",
        "note": "Competitive overall, GPT-3.5 better on code"
    },
    "efficiency": {
        "active_params": "12.9B",
        "comparable_dense": "~70B",
        "compute_ratio": "5.4x more efficient",
        "memory": "~47GB (all experts loaded)"
    }
}


# =============================================================================
# Mixtral Variants
# =============================================================================

MIXTRAL_VARIANTS = {
    "mixtral_8x7b": {
        "params_total": "46.7B",
        "params_active": "12.9B",
        "experts": 8,
        "expert_size": "~7B",
        "release": "December 2023"
    },
    "mixtral_8x7b_instruct": {
        "base": "mixtral_8x7b",
        "training": "Instruction tuning + RLHF",
        "note": "Optimized for chat/instruction following"
    },
    "mixtral_8x22b": {
        "params_total": "176B",
        "params_active": "39B",
        "experts": 8,
        "expert_size": "~22B",
        "release": "April 2024",
        "improvement": "Stronger reasoning, longer context (65K)"
    }
}


# =============================================================================
# Deployment Considerations
# =============================================================================

MIXTRAL_DEPLOYMENT = {
    "memory_requirements": {
        "fp16": "~94GB (all experts)",
        "int8": "~47GB",
        "int4": "~24GB",
        "single_gpu": "Possible with quantization on 80GB GPU"
    },
    "inference_optimization": {
        "expert_offloading": (
            "Can offload inactive experts to CPU/disk. "
            "Only 2 experts active, others can be loaded on demand."
        ),
        "speculative_decoding": "Works well with smaller draft model",
        "batching": "Group tokens by expert for efficiency"
    },
    "production_tips": {
        "quantization": "4-bit works well with minimal quality loss",
        "kv_cache": "Use sliding window for long contexts",
        "expert_caching": "Keep frequently-used experts in fast memory"
    }
}
