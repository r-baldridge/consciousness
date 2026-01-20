"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces - 2023

Gu & Dao's breakthrough work introducing selection mechanisms to state space
models, making them content-aware while maintaining linear time complexity.
Key insight: make SSM parameters input-dependent (selective).

Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
arXiv: 2312.00752

Mathematical Formulation:
    Selective State Space Model:
        x_t = A_bar(u_t) * x_{t-1} + B_bar(u_t) * u_t
        y_t = C(u_t) * x_t

    Where the key difference from S4 is that B, C, and Delta are
    functions of the input u_t, making the model content-aware.

    Selection Mechanism:
        s_B(x) = Linear_B(x)     -> B depends on input
        s_C(x) = Linear_C(x)     -> C depends on input
        s_delta(x) = softplus(Linear_delta(x))  -> timescale depends on input

    Hardware-Aware Parallel Scan:
        Efficient GPU implementation using work-efficient parallel scan
        for the recurrence, avoiding sequential computation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

MAMBA = MLMethod(
    method_id="mamba_2023",
    name="Mamba",
    year=2023,

    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],

    authors=["Albert Gu", "Tri Dao"],
    paper_title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    paper_url="https://arxiv.org/abs/2312.00752",

    key_innovation=(
        "Introduced selection mechanism to state space models, making B, C, and Delta "
        "input-dependent. This allows the model to selectively propagate or forget "
        "information based on content. Combined with hardware-aware parallel scan, "
        "achieves 5x faster inference than Transformers at long sequences."
    ),

    mathematical_formulation=r"""
Selective State Space Model:
    Standard SSM (time-invariant):
        x_t = A_bar * x_{t-1} + B_bar * u_t
        y_t = C * x_t

    Selective SSM (input-dependent):
        B_t = s_B(u_t)          # B depends on input
        C_t = s_C(u_t)          # C depends on input
        Delta_t = s_Delta(u_t)   # Timescale depends on input

        A_bar_t = exp(Delta_t * A)
        B_bar_t = (A_bar_t - I) * A^(-1) * B_t

        x_t = A_bar_t * x_{t-1} + B_bar_t * u_t
        y_t = C_t * x_t

Selection Functions:
    s_B(x) = Linear_N(x)                    # [B, L, D] -> [B, L, N]
    s_C(x) = Linear_N(x)                    # [B, L, D] -> [B, L, N]
    s_Delta(x) = softplus(Linear_1(x))      # [B, L, D] -> [B, L, 1]

    Where N is state dimension, D is model dimension

Discretization (ZOH with input-dependent Delta):
    A_bar = exp(Delta * A)    # Delta varies per position
    B_bar = Delta * B         # Simplified when Delta is small

Hardware-Aware Parallel Scan:
    Given recurrence: x_t = a_t * x_{t-1} + b_t
    Parallel prefix sum formulation:
        (a_t, b_t) o (a_{t-1}, b_{t-1}) = (a_t * a_{t-1}, a_t * b_{t-1} + b_t)

    Enables O(L) work with O(log L) depth on parallel hardware
""",

    predecessors=["s4_2021", "s4d_2022", "h3_2022"],
    successors=["mamba2_2024", "jamba_2024", "vim_2024"],

    tags=["state-space", "selective", "linear-complexity", "hardware-aware", "language-model"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def selective_scan(u, delta, A, B, C):
    """
    Core selective scan operation in Mamba.

    Unlike S4 where A, B, C are fixed, in Mamba B, C, and delta
    vary with input, making the operation content-dependent.

    Args:
        u: Input sequence [batch, seq_len, d_inner]
        delta: Input-dependent timescale [batch, seq_len, d_inner]
        A: State matrix [d_inner, state_dim] (learned, shared)
        B: Input-dependent B [batch, seq_len, state_dim]
        C: Input-dependent C [batch, seq_len, state_dim]

    Returns:
        Dictionary describing the selective scan computation

    Key Insight:
        By making B, C, delta input-dependent:
        - Model can selectively remember (small delta -> slow decay)
        - Model can selectively forget (large delta -> fast decay)
        - Model can selectively update (B controls input injection)
    """
    return {
        "discretization": "A_bar = exp(delta * A), B_bar = delta * B",
        "recurrence": "x_t = A_bar_t * x_{t-1} + B_bar_t * u_t",
        "output": "y_t = C_t * x_t",
        "key_difference": "A_bar, B_bar, C vary per timestep",
        "selection": "Delta, B, C are functions of input u",
        "complexity": "O(B * L * D * N) but parallelizable"
    }


def selection_mechanism(x, dt_rank, d_state, d_inner):
    """
    Compute input-dependent parameters for selective SSM.

    The selection mechanism is what makes Mamba content-aware:
    - B controls what information enters the state
    - C controls what information is read from state
    - Delta controls the timescale (remember vs forget)

    Args:
        x: Input [batch, seq_len, d_inner]
        dt_rank: Rank for delta projection
        d_state: State dimension (N)
        d_inner: Inner dimension

    Returns:
        Dictionary describing selection projections
    """
    return {
        "B_projection": "B = Linear(x, d_inner -> d_state)",
        "C_projection": "C = Linear(x, d_inner -> d_state)",
        "delta_projection": "delta = softplus(Linear(Linear(x, d_inner -> dt_rank), dt_rank -> d_inner))",
        "softplus": "softplus(x) = log(1 + exp(x))",
        "interpretation": {
            "B": "Input selection - what to write to memory",
            "C": "Output selection - what to read from memory",
            "delta": "Timescale selection - how fast to update"
        }
    }


def parallel_associative_scan(a, b):
    """
    Hardware-efficient parallel scan for linear recurrence.

    For recurrence: x_t = a_t * x_{t-1} + b_t
    Can be parallelized using associative scan with operator:
        (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)

    Args:
        a: Multiplicative coefficients [batch, seq_len, dim]
        b: Additive coefficients [batch, seq_len, dim]

    Returns:
        Dictionary describing parallel scan algorithm

    Complexity:
        Sequential: O(L) depth
        Parallel: O(log L) depth with O(L) work

    Note:
        Mamba's hardware-aware implementation fuses operations
        and uses memory-efficient techniques for GPU execution.
    """
    return {
        "associative_op": "(a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)",
        "identity": "(1, 0)",
        "algorithm": "Binary tree reduction then broadcast",
        "depth": "O(log L)",
        "work": "O(L)",
        "gpu_optimization": "Kernel fusion, memory coalescing, work partitioning"
    }


def mamba_block(x, conv1d_kernel_size=4):
    """
    Full Mamba block computation.

    Architecture:
    x -> Linear -> [x_proj, z] -> Conv1D -> SiLU -> SSM -> * z -> Linear -> out

    Args:
        x: Input [batch, seq_len, d_model]
        conv1d_kernel_size: Size of causal convolution kernel

    Returns:
        Dictionary describing Mamba block
    """
    return {
        "step1": "x_proj, z = Linear(x).split([d_inner, d_inner])",
        "step2": "x_proj = SiLU(Conv1D(x_proj, kernel_size=4))",
        "step3": "B, C, delta = selection_mechanism(x_proj)",
        "step4": "y = selective_scan(x_proj, delta, A, B, C)",
        "step5": "y = y * SiLU(z)  # Gating",
        "step6": "out = Linear(y)",
        "d_inner": "Typically 2 * d_model (expansion factor)",
        "d_state": "Typically 16 (state dimension N)"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class MambaArchitecture:
    """Reference architecture for Mamba model."""

    d_model: int = 768
    d_state: int = 16  # N
    d_conv: int = 4
    expand: int = 2  # E
    num_layers: int = 24

    @staticmethod
    def mamba_block_structure() -> str:
        """Single Mamba block structure."""
        return """
Mamba Block:
    Input: x [batch_size, seq_len, d_model]

    # Input projections (GLU-style)
    x_proj = Linear(d_model -> 2 * d_inner)(x)
    x, z = x_proj.chunk(2, dim=-1)

    # Local convolution (short-range dependencies)
    x = SiLU(Conv1D(x, kernel=4, groups=d_inner))

    # Selection mechanism (input-dependent SSM params)
    B = Linear(d_inner -> d_state)(x)       # [B, L, N]
    C = Linear(d_inner -> d_state)(x)       # [B, L, N]
    delta = softplus(Linear(dt_rank)(Linear(d_inner -> dt_rank)(x)))

    # Selective SSM
    A_bar = exp(delta.unsqueeze(-1) * A)    # A is [d_inner, d_state], learned
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
    y = selective_scan(x, A_bar, B_bar, C)

    # Output gating and projection
    y = y * SiLU(z)
    out = Linear(d_inner -> d_model)(y)

    Output: out [batch_size, seq_len, d_model]
"""

    @staticmethod
    def full_model_structure() -> str:
        """Full Mamba language model."""
        return """
Mamba Language Model:
    Input: tokens [batch_size, seq_len]

    # Token embedding
    x = Embedding(vocab_size, d_model)(tokens)

    # Mamba layers with residual connections
    for layer in range(num_layers):
        x = x + Mamba_Block(RMSNorm(x))

    # Output head
    x = RMSNorm(x)
    logits = Linear(d_model -> vocab_size)(x)

    Output: logits [batch_size, seq_len, vocab_size]

Inference Mode (Autoregressive):
    # Cache: state x [batch, d_inner, d_state]
    # At each step, update state and output one token
    # Constant memory, O(1) per token (vs O(L) for attention)
"""


# =============================================================================
# Mamba Variants Reference
# =============================================================================

MAMBA_VARIANTS = {
    "mamba_130m": {
        "d_model": 768,
        "n_layers": 24,
        "d_state": 16,
        "parameters": "130M",
        "context": "Unlimited (linear scaling)"
    },
    "mamba_370m": {
        "d_model": 1024,
        "n_layers": 48,
        "d_state": 16,
        "parameters": "370M"
    },
    "mamba_790m": {
        "d_model": 1536,
        "n_layers": 48,
        "d_state": 16,
        "parameters": "790M"
    },
    "mamba_1.4b": {
        "d_model": 2048,
        "n_layers": 48,
        "d_state": 16,
        "parameters": "1.4B"
    },
    "mamba_2.8b": {
        "d_model": 2560,
        "n_layers": 64,
        "d_state": 16,
        "parameters": "2.8B"
    }
}


# =============================================================================
# Key Insights Reference
# =============================================================================

MAMBA_INSIGHTS = {
    "why_selection_matters": (
        "Fixed SSMs (like S4) apply the same transformation regardless of content. "
        "This limits their ability to perform content-based reasoning. Selection allows "
        "Mamba to: (1) selectively remember relevant information, (2) selectively forget "
        "irrelevant information, (3) adapt its behavior based on what it sees."
    ),
    "comparison_to_attention": {
        "attention": "O(L^2) time/space, but fully content-aware",
        "fixed_ssm": "O(L) time/space, but content-agnostic",
        "selective_ssm": "O(L) time/space AND content-aware (best of both)"
    },
    "hardware_efficiency": (
        "Mamba achieves 5x higher throughput than Transformers through: "
        "(1) Linear scaling with sequence length, (2) Hardware-aware kernel fusion, "
        "(3) Memory-efficient state caching during inference."
    ),
    "limitations": (
        "While Mamba excels at language modeling, some tasks requiring precise "
        "in-context recall (like retrieval) may benefit from explicit attention. "
        "This led to hybrid architectures like Jamba."
    )
}
