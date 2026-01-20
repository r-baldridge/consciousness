"""
Mixture of Experts (MoE) - Fundamentals

Core concepts and mathematical foundations of Mixture of Experts,
a technique for conditional computation that enables scaling model
capacity without proportionally increasing compute.

Original Concept: Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"
Modern Revival: Shazeer et al. (2017) "Outrageously Large Neural Networks"

Mathematical Formulation:
    Output: y = sum_{i=1}^{N} g(x)_i * E_i(x)

    Where:
        - g(x) is the gating/routing function
        - E_i(x) is the i-th expert network
        - N is the number of experts

    Sparse Gating (Top-k):
        g(x) = TopK(softmax(W_g * x + noise), k)
        Only k experts are active per token

    Key Challenge: Load Balancing
        Without regularization, gating tends to collapse to few experts
        Auxiliary loss encourages balanced expert utilization
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

MIXTURE_OF_EXPERTS = MLMethod(
    method_id="moe_basics",
    name="Mixture of Experts",
    year=1991,

    era=MethodEra.CLASSICAL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["Robert A. Jacobs", "Michael I. Jordan", "Steven J. Nowlan", "Geoffrey E. Hinton"],
    paper_title="Adaptive Mixtures of Local Experts",
    paper_url="https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf",

    key_innovation=(
        "Introduced the concept of multiple specialized expert networks combined "
        "by a learned gating function. Each expert specializes in a region of input "
        "space, and the gating network learns to route inputs to appropriate experts. "
        "Foundation for modern conditional computation."
    ),

    mathematical_formulation=r"""
Basic MoE Formulation:
    y = sum_{i=1}^{N} g_i(x) * E_i(x)

    Where:
        g(x) = softmax(W_g * x)           # Gating weights [N]
        E_i(x) = Expert_i(x)              # Expert outputs [d_out]
        y = weighted sum of expert outputs # Final output [d_out]

Sparse MoE (Top-k Gating):
    router_logits = W_g * x + noise       # [N]
    gates = TopK(softmax(router_logits), k)
    y = sum_{i in top-k} gates_i * E_i(x)

    Noise: Gaussian noise for exploration/load balancing
    TopK: Zero out all but top-k values, renormalize

Expert Structure (typically FFN):
    E_i(x) = W_2^i * GELU(W_1^i * x)
    Where W_1^i: [d_model -> d_ff], W_2^i: [d_ff -> d_model]

Load Balancing Auxiliary Loss:
    L_aux = alpha * N * sum_{i=1}^{N} f_i * P_i

    Where:
        f_i = (1/T) * sum_{t=1}^{T} 1[expert i is selected for token t]
        P_i = (1/T) * sum_{t=1}^{T} g_i(x_t)
        T = total tokens in batch

    This encourages f_i ≈ P_i ≈ 1/N (uniform distribution)
""",

    predecessors=["mlp_1986"],
    successors=["sparsely_gated_moe_2017", "switch_transformer_2021", "mixtral_2023"],

    tags=["mixture-of-experts", "conditional-computation", "sparse", "gating", "scaling"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def dense_moe_forward(x, experts, gating_network):
    """
    Dense Mixture of Experts forward pass.

    All experts compute outputs, weighted by gating scores.

    Args:
        x: Input [batch, seq_len, d_model]
        experts: List of N expert networks
        gating_network: Network that produces expert weights

    Returns:
        Dictionary describing dense MoE computation

    Complexity: O(N * expert_cost) - computes all experts
    """
    return {
        "step1_gate": "gates = softmax(gating_network(x))  # [batch, seq, N]",
        "step2_experts": "expert_outputs = [E_i(x) for i in range(N)]  # N x [batch, seq, d_out]",
        "step3_combine": "y = sum(gates[:,:,i:i+1] * expert_outputs[i] for i in range(N))",
        "output_shape": "[batch, seq_len, d_model]",
        "complexity": "O(N * expert_cost)",
        "note": "Dense MoE computes ALL experts - expensive but simple"
    }


def sparse_moe_forward(x, experts, router, top_k=2):
    """
    Sparse Mixture of Experts forward pass with top-k routing.

    Only k experts are computed per token, drastically reducing compute.

    Args:
        x: Input [batch, seq_len, d_model]
        experts: List of N expert networks
        router: Network that produces routing logits
        top_k: Number of experts to use per token

    Returns:
        Dictionary describing sparse MoE computation
    """
    return {
        "step1_route": "logits = router(x)  # [batch, seq, N]",
        "step2_noise": "logits += gaussian_noise * training  # exploration",
        "step3_topk": "top_k_gates, top_k_indices = topk(softmax(logits), k)",
        "step4_normalize": "top_k_gates = top_k_gates / sum(top_k_gates)",
        "step5_dispatch": "route tokens to selected experts",
        "step6_compute": "compute only selected expert(s) per token",
        "step7_combine": "y = sum(gate_i * E_i(x)) for i in selected_experts",
        "complexity": f"O({top_k} * expert_cost) vs O(N * expert_cost)",
        "speedup": f"N/{top_k}x fewer expert computations"
    }


def gating_function(x, W_g, num_experts, noise_std=1.0, training=True):
    """
    Compute gating weights for expert routing.

    The gating function determines which experts process each input.
    Noise is added during training for exploration and load balancing.

    Args:
        x: Input [batch, seq_len, d_model]
        W_g: Gating weight matrix [d_model, num_experts]
        num_experts: Number of experts N
        noise_std: Standard deviation of Gaussian noise
        training: Whether in training mode

    Returns:
        Dictionary describing gating computation
    """
    return {
        "logits": "logits = x @ W_g  # [batch, seq, N]",
        "noise": f"if training: logits += Normal(0, {noise_std}) * softplus(W_noise @ x)",
        "gates": "gates = softmax(logits, dim=-1)",
        "interpretation": {
            "gates_i": "Probability of routing to expert i",
            "noise_purpose": "Encourages exploration, prevents collapse",
            "temperature": "Can add temperature scaling for sharper/softer routing"
        }
    }


def load_balancing_loss(gates, expert_assignments, num_experts, num_tokens):
    """
    Compute auxiliary loss for load balancing.

    Without load balancing, gating often collapses to using few experts.
    This auxiliary loss encourages uniform expert utilization.

    Args:
        gates: Gating probabilities [batch, seq, N]
        expert_assignments: Which experts were selected [batch, seq, k]
        num_experts: Number of experts N
        num_tokens: Total tokens in batch

    Returns:
        Dictionary describing load balancing loss

    Mathematical Form:
        L_aux = alpha * N * sum_i (f_i * P_i)

        Where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability to expert i
    """
    return {
        "f_i": "f_i = count(expert i selected) / num_tokens",
        "P_i": "P_i = mean(gates[:, :, i])",
        "loss": "L_aux = alpha * N * sum(f_i * P_i)",
        "interpretation": {
            "minimize": "When f_i and P_i are both high for one expert",
            "optimal": "f_i = P_i = 1/N for all experts (uniform)",
            "effect": "Pushes routing to be more balanced"
        },
        "alpha": "Typically 0.01 to 0.1 (trade-off with main loss)"
    }


def expert_capacity(batch_size, seq_len, num_experts, top_k, capacity_factor=1.25):
    """
    Compute expert capacity for buffer sizing.

    Each expert can only process a limited number of tokens to
    enable efficient batched computation. Overflow is dropped.

    Args:
        batch_size: Batch size B
        seq_len: Sequence length L
        num_experts: Number of experts N
        top_k: Experts per token k
        capacity_factor: Buffer factor (>1 for safety)

    Returns:
        Dictionary describing capacity calculation

    Formula:
        capacity = (total_tokens * top_k / num_experts) * capacity_factor
    """
    total_tokens = batch_size * seq_len
    expected_per_expert = total_tokens * top_k / num_experts
    capacity = int(expected_per_expert * capacity_factor)

    return {
        "formula": "capacity = (B * L * k / N) * capacity_factor",
        "total_tokens": total_tokens,
        "expected_per_expert": expected_per_expert,
        "capacity": capacity,
        "capacity_factor": capacity_factor,
        "overflow_handling": "Tokens exceeding capacity are dropped/passed through",
        "trade_off": "Higher capacity = more memory, less dropped tokens"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class MoEArchitecture:
    """Reference architecture for MoE layer."""

    d_model: int = 768
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25

    @staticmethod
    def moe_layer_structure() -> str:
        """MoE layer replacing FFN in Transformer."""
        return """
MoE Layer (replaces standard FFN):
    Input: x [batch_size, seq_len, d_model]

    # Routing
    router_logits = Linear(d_model -> num_experts)(x)  # [B, L, N]
    router_probs = softmax(router_logits, dim=-1)
    gates, indices = top_k(router_probs, k)            # [B, L, k], [B, L, k]

    # Dispatch tokens to experts
    # (Implementation varies: token choice vs expert choice)

    # Compute expert outputs
    for expert_idx in range(num_experts):
        tokens_for_expert = gather_tokens(x, indices == expert_idx)
        expert_output = Expert[expert_idx](tokens_for_expert)
        scatter_outputs(y, expert_output, indices == expert_idx)

    # Combine with gating weights
    y = sum over k: gates[:,:,i] * expert_outputs[:,:,i]

    Output: y [batch_size, seq_len, d_model]
"""

    @staticmethod
    def expert_structure() -> str:
        """Individual expert network (typically FFN)."""
        return """
Expert Network (standard FFN):
    Input: x [num_tokens, d_model]

    h = Linear(d_model -> d_ff)(x)      # Up projection
    h = GELU(h)                          # Activation
    y = Linear(d_ff -> d_model)(h)       # Down projection

    Output: y [num_tokens, d_model]

    Parameters per expert: 2 * d_model * d_ff
    Total expert params: num_experts * 2 * d_model * d_ff
    Active params per token: top_k * 2 * d_model * d_ff
"""


# =============================================================================
# MoE Concepts Reference
# =============================================================================

MOE_CONCEPTS = {
    "conditional_computation": (
        "The core idea: not all parameters need to be active for all inputs. "
        "By routing inputs to specialized experts, we can scale parameters "
        "without scaling compute. A 1T parameter model might use only 100B "
        "parameters per forward pass."
    ),
    "expert_specialization": (
        "Experts tend to specialize in different aspects of the data. "
        "Some experts might handle syntax, others semantics, others specific "
        "domains. This emerges naturally from training with load balancing."
    ),
    "routing_strategies": {
        "token_choice": "Each token picks its top-k experts",
        "expert_choice": "Each expert picks its top-c tokens",
        "hash_routing": "Deterministic routing based on token hash",
        "learned_routing": "Routing network learned end-to-end"
    },
    "challenges": {
        "load_imbalance": "Some experts may be over/under-utilized",
        "expert_collapse": "Gating may collapse to few experts",
        "communication": "In distributed setting, tokens must be sent to expert devices",
        "training_stability": "Sparse routing can cause training instability"
    },
    "solutions": {
        "auxiliary_loss": "Penalize imbalanced routing",
        "noise_injection": "Add noise to routing for exploration",
        "capacity_limits": "Limit tokens per expert, drop overflow",
        "expert_parallelism": "Distribute experts across devices efficiently"
    }
}


# =============================================================================
# Historical Evolution
# =============================================================================

MOE_HISTORY = {
    "1991_jacobs": {
        "contribution": "Original MoE paper",
        "key_idea": "Divide-and-conquer with specialized experts",
        "limitation": "Dense computation, small scale"
    },
    "2017_shazeer": {
        "paper": "Outrageously Large Neural Networks",
        "contribution": "Sparsely-gated MoE for large language models",
        "key_idea": "Top-k routing enables scaling to 137B parameters",
        "innovation": "Noisy top-k gating, load balancing"
    },
    "2021_switch": {
        "paper": "Switch Transformers",
        "contribution": "Simplified top-1 routing, trillion parameter scale",
        "key_idea": "Simpler routing can work as well or better"
    },
    "2022_stmoe": {
        "paper": "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
        "contribution": "Router z-loss for training stability"
    },
    "2023_mixtral": {
        "paper": "Mixtral of Experts",
        "contribution": "Production MoE LLM, open weights",
        "achievement": "47B params, 12B active, matches GPT-3.5"
    }
}
