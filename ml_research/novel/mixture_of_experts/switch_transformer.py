"""
Switch Transformer - 2021

Fedus, Zoph & Shazeer's breakthrough work scaling MoE to trillion parameters
with simplified top-1 routing. Key insight: simpler routing can work as well
or better than complex schemes while being more efficient.

Paper: "Switch Transformers: Scaling to Trillion Parameter Models with Simple
        and Efficient Sparsity" (JMLR 2022)
arXiv: 2101.03961

Mathematical Formulation:
    Top-1 Routing (vs Top-k):
        y = g_i(x) * E_i(x)    where i = argmax(router(x))

    Only ONE expert is active per token, maximizing efficiency.

    Simplified Load Balancing:
        L_aux = alpha * N * sum_i (f_i * P_i)
        Same as before but works well with top-1

    Expert Capacity:
        capacity = (tokens_per_batch / num_experts) * capacity_factor
        Tokens exceeding capacity are passed through unchanged
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

SWITCH_TRANSFORMER = MLMethod(
    method_id="switch_transformer_2021",
    name="Switch Transformer",
    year=2021,

    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],

    authors=["William Fedus", "Barret Zoph", "Noam Shazeer"],
    paper_title="Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
    paper_url="https://arxiv.org/abs/2101.03961",

    key_innovation=(
        "Simplified MoE routing from top-k to top-1, reducing routing complexity "
        "and communication costs. Demonstrated stable training at 1.6T parameters. "
        "Showed that sparse models can be pre-trained faster than dense models "
        "and achieve strong downstream performance."
    ),

    mathematical_formulation=r"""
Switch Routing (Top-1):
    router_logits = W_r * x                    # [d_model] -> [num_experts]
    router_probs = softmax(router_logits)
    expert_idx = argmax(router_probs)          # Single expert selection
    gate = router_probs[expert_idx]            # Routing weight

    y = gate * Expert[expert_idx](x)           # Single expert output

Auxiliary Load Balancing Loss:
    L_aux = alpha * N * sum_{i=1}^{N} f_i * P_i

    f_i = (1/T) * sum_{t=1}^{T} 1[token t routed to expert i]
    P_i = (1/T) * sum_{t=1}^{T} router_probs_t[i]

    Optimal: f_i = P_i = 1/N (uniform distribution)

Expert Capacity:
    capacity = ceil((batch_size * seq_len / num_experts) * capacity_factor)

    capacity_factor = 1.0 to 2.0 (1.25 typical)

    If tokens_for_expert > capacity:
        overflow_tokens passed through unchanged (not processed by expert)

Token Dispatch (permutation-based):
    1. Compute routing for all tokens
    2. Group tokens by selected expert
    3. Pad each group to capacity
    4. Process in batched manner
    5. Scatter outputs back to original positions

Selective Precision (training stability):
    Router computation in float32
    Expert computation can be bfloat16/float16
""",

    predecessors=["transformer_2017", "sparsely_gated_moe_2017"],
    successors=["st_moe_2022", "mixtral_2023", "glam_2021"],

    tags=["mixture-of-experts", "sparse", "scaling", "transformer", "efficient"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def switch_routing(x, router_weights, num_experts):
    """
    Top-1 routing for Switch Transformer.

    Each token is routed to exactly one expert, maximizing efficiency.

    Args:
        x: Input [batch, seq_len, d_model]
        router_weights: Routing weight matrix [d_model, num_experts]
        num_experts: Number of experts N

    Returns:
        Dictionary describing Switch routing
    """
    return {
        "step1": "router_logits = x @ router_weights  # [B, L, N]",
        "step2": "router_probs = softmax(router_logits, dim=-1)  # [B, L, N]",
        "step3": "expert_idx = argmax(router_probs, dim=-1)  # [B, L]",
        "step4": "gate = gather(router_probs, expert_idx)  # [B, L]",
        "output": "(expert_idx, gate)",
        "key_difference": "Top-1 vs Top-k: only ONE expert per token",
        "benefit": "Reduced communication, simpler implementation"
    }


def expert_capacity_computation(batch_size, seq_len, num_experts, capacity_factor=1.25):
    """
    Compute expert capacity with buffer.

    Capacity limits prevent memory issues and enable efficient batching.
    Overflow tokens are handled by skip connection (not processed).

    Args:
        batch_size: Batch size B
        seq_len: Sequence length L
        num_experts: Number of experts N
        capacity_factor: Buffer factor (typically 1.0-2.0)

    Returns:
        Dictionary describing capacity computation
    """
    total_tokens = batch_size * seq_len
    tokens_per_expert = total_tokens / num_experts
    capacity = int(tokens_per_expert * capacity_factor)

    return {
        "formula": "capacity = ceil((B * L / N) * capacity_factor)",
        "total_tokens": total_tokens,
        "expected_per_expert": tokens_per_expert,
        "capacity": capacity,
        "capacity_factor": capacity_factor,
        "overflow_handling": (
            "Tokens exceeding capacity skip expert processing. "
            "Output = input (residual passthrough). "
            "This is a form of dropout that helps regularization."
        ),
        "trade_offs": {
            "low_capacity": "More dropped tokens, less memory, faster",
            "high_capacity": "Fewer dropped tokens, more memory, slower"
        }
    }


def load_balancing_loss_switch(router_probs, expert_mask, num_experts, alpha=0.01):
    """
    Compute Switch Transformer's load balancing loss.

    Same formulation as general MoE but tuned for top-1 routing.

    Args:
        router_probs: Routing probabilities [batch, seq, num_experts]
        expert_mask: Binary mask of expert selection [batch, seq, num_experts]
        num_experts: Number of experts N
        alpha: Loss coefficient (0.01 typical)

    Returns:
        Dictionary describing load balancing loss
    """
    return {
        "f_i": "f_i = mean(expert_mask[:, :, i])  # Fraction routed to i",
        "P_i": "P_i = mean(router_probs[:, :, i])  # Mean prob for i",
        "L_aux": f"L_aux = {alpha} * {num_experts} * sum(f_i * P_i)",
        "range": "0 (perfectly uniform) to N (all to one expert)",
        "optimal": f"f_i = P_i = 1/{num_experts} for all i",
        "interpretation": (
            "Minimizing f_i * P_i pushes the model to: "
            "(1) reduce probability mass on over-utilized experts, "
            "(2) route tokens more uniformly across experts"
        )
    }


def selective_precision(x, expert_fn, router_fn, use_float32_router=True):
    """
    Selective precision for training stability.

    Router computations in float32, expert computations can be lower precision.
    This prevents routing instability while maintaining efficiency.

    Args:
        x: Input tensor [batch, seq, d_model]
        expert_fn: Expert computation function
        router_fn: Router computation function
        use_float32_router: Whether to use float32 for routing

    Returns:
        Dictionary describing precision strategy
    """
    return {
        "router_precision": "float32 (prevents softmax instability)",
        "expert_precision": "bfloat16 or float16 (efficiency)",
        "rationale": (
            "Router softmax is sensitive to numerical precision. "
            "Small differences in logits can flip routing decisions. "
            "Experts are more robust to lower precision."
        ),
        "implementation": {
            "step1": "x_f32 = x.float()",
            "step2": "router_logits = router_fn(x_f32).float()",
            "step3": "probs = softmax(router_logits).cast(x.dtype)",
            "step4": "expert_output = expert_fn(x)  # original precision"
        }
    }


def switch_layer_forward(x, experts, router, capacity_factor=1.25):
    """
    Full Switch Transformer MoE layer forward pass.

    Args:
        x: Input [batch, seq_len, d_model]
        experts: List of N expert networks
        router: Router network [d_model -> num_experts]
        capacity_factor: Expert capacity buffer

    Returns:
        Dictionary describing full forward pass
    """
    return {
        "step1_route": {
            "logits": "router_logits = router(x.float()).float()",
            "probs": "router_probs = softmax(router_logits)",
            "select": "expert_idx = argmax(router_probs)",
            "gate": "gate = router_probs.gather(expert_idx)"
        },
        "step2_dispatch": {
            "compute_capacity": "capacity = (B*L/N) * capacity_factor",
            "group_tokens": "group tokens by expert_idx",
            "handle_overflow": "mask out tokens exceeding capacity"
        },
        "step3_compute": {
            "batch_experts": "process each expert's tokens in parallel",
            "expert_output": "out_i = expert_i(tokens_for_expert_i)"
        },
        "step4_combine": {
            "scatter": "scatter expert outputs to original positions",
            "gate_multiply": "y = gate * expert_output",
            "add_overflow": "y[overflow_mask] = x[overflow_mask]  # passthrough"
        },
        "step5_aux_loss": {
            "compute": "L_aux = alpha * N * sum(f_i * P_i)",
            "add_to_loss": "total_loss = main_loss + L_aux"
        }
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class SwitchTransformerArchitecture:
    """Reference architecture for Switch Transformer."""

    d_model: int = 768
    d_ff: int = 3072
    num_experts: int = 128
    num_layers: int = 12
    capacity_factor: float = 1.25

    @staticmethod
    def model_configurations() -> Dict:
        """Switch Transformer model configurations."""
        return {
            "switch_base_128": {
                "num_experts": 128,
                "d_model": 768,
                "d_ff": 3072,
                "layers": 12,
                "heads": 12,
                "total_params": "7.4B",
                "active_params": "~100M"
            },
            "switch_large_128": {
                "num_experts": 128,
                "d_model": 1024,
                "d_ff": 4096,
                "layers": 24,
                "heads": 16,
                "total_params": "26B",
                "active_params": "~300M"
            },
            "switch_xxl": {
                "num_experts": 2048,
                "d_model": 4096,
                "d_ff": 16384,
                "layers": 64,
                "heads": 64,
                "total_params": "1.6T",
                "active_params": "~1B"
            }
        }

    @staticmethod
    def switch_layer_structure() -> str:
        """Switch MoE layer structure."""
        return """
Switch Layer (replaces FFN in Transformer):
    Input: x [batch, seq_len, d_model]

    # Router (float32 for stability)
    router_logits = Linear(d_model -> num_experts)(x.float())
    router_probs = softmax(router_logits, dim=-1)

    # Top-1 Selection
    expert_idx = argmax(router_probs, dim=-1)  # [B, L]
    gate = gather(router_probs, expert_idx)     # [B, L]

    # Capacity Check
    capacity = (B * L / N) * capacity_factor
    tokens_per_expert = count per expert
    mask = tokens_per_expert <= capacity

    # Expert Computation (can be lower precision)
    for i in range(num_experts):
        tokens_i = x[expert_idx == i & mask]
        outputs_i = Expert_i(tokens_i)

    # Combine
    y = zeros_like(x)
    scatter(y, outputs, indices)
    y = y * gate.unsqueeze(-1)
    y[~mask] = x[~mask]  # Overflow passthrough

    # Auxiliary Loss
    f = one_hot(expert_idx).mean(dim=[0,1])
    P = router_probs.mean(dim=[0,1])
    L_aux = alpha * N * (f * P).sum()

    Output: y, L_aux
"""


# =============================================================================
# Scaling Properties
# =============================================================================

SWITCH_SCALING = {
    "parameter_efficiency": (
        "Switch Transformer scales parameters at ~10% compute cost. "
        "A 1.6T parameter model uses similar FLOPs to a 10B dense model. "
        "This enables exploring much larger parameter regimes."
    ),
    "scaling_laws": {
        "observation": "Sparse models follow different scaling laws",
        "finding": (
            "For a fixed compute budget, sparse models outperform dense models. "
            "Switch-Base (7.4B params) matches T5-Base in quality with 7x speedup."
        ),
        "diminishing_returns": (
            "Benefits decrease as we increase experts beyond ~128. "
            "Load balancing becomes harder with more experts."
        )
    },
    "training_efficiency": {
        "speedup": "4-7x faster pre-training vs dense baselines",
        "quality": "Matches or exceeds dense models on downstream tasks",
        "stability": "Requires careful precision handling and capacity tuning"
    },
    "inference_considerations": {
        "memory": "Must load all expert parameters",
        "latency": "Similar to dense for single sequences",
        "throughput": "Can batch across experts for efficiency"
    }
}


# =============================================================================
# Implementation Insights
# =============================================================================

SWITCH_IMPLEMENTATION = {
    "expert_parallelism": (
        "Experts distributed across devices. Each device hosts subset of experts. "
        "All-to-all communication routes tokens to appropriate devices. "
        "Key challenge: minimize communication overhead."
    ),
    "capacity_factor_tuning": {
        "1.0": "Minimal memory, maximum dropped tokens (~30% can be dropped)",
        "1.25": "Good balance (default), ~10% dropped",
        "2.0": "Almost no drops, 2x memory per expert",
        "dynamic": "Can adjust based on load imbalance"
    },
    "training_stability": {
        "selective_precision": "Router in float32, experts in bf16",
        "smaller_init": "Smaller initialization for router weights",
        "dropout": "Light dropout on routing for regularization",
        "gradient_clipping": "Clip gradients, especially early in training"
    },
    "load_balancing_tips": {
        "alpha": "0.01-0.1 typical, tune based on imbalance",
        "expert_dropout": "Randomly drop experts during training",
        "auxiliary_loss_annealing": "Start high, decrease over training"
    }
}
