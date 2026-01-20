"""
S4: Structured State Spaces - 2021

Gu et al.'s foundational work on efficient sequence modeling using continuous-time
state space models with special initialization (HiPPO) and structure (diagonal plus
low-rank) for handling extremely long sequences.

Paper: "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022)
arXiv: 2111.00396

Mathematical Formulation:
    Continuous-time State Space Model:
        x'(t) = Ax(t) + Bu(t)   (state equation)
        y(t)  = Cx(t) + Du(t)   (output equation)

    Where:
        - x(t) in R^N is the hidden state
        - u(t) in R^1 is the input signal
        - y(t) in R^1 is the output signal
        - A in R^(NxN) is the state matrix
        - B in R^(Nx1) is the input matrix
        - C in R^(1xN) is the output matrix
        - D in R^1 is the feedthrough matrix

    Discretization (Zero-Order Hold):
        x_k = A_bar * x_{k-1} + B_bar * u_k
        y_k = C * x_k + D * u_k

        Where: A_bar = exp(Delta * A), B_bar = (A_bar - I) * A^(-1) * B

    HiPPO (High-order Polynomial Projection Operator):
        Initializes A to optimally compress historical information
        A_nk = -(2n+1)^(1/2) * (2k+1)^(1/2) if n > k
             = n+1 if n = k
             = 0 if n < k
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

STRUCTURED_STATE_SPACE = MLMethod(
    method_id="s4_2021",
    name="Structured State Spaces (S4)",
    year=2021,

    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],

    authors=["Albert Gu", "Karan Goel", "Christopher Re"],
    paper_title="Efficiently Modeling Long Sequences with Structured State Spaces",
    paper_url="https://arxiv.org/abs/2111.00396",

    key_innovation=(
        "Combined continuous-time state space formulation with HiPPO initialization "
        "and diagonal plus low-rank (DPLR) structure to achieve O(N log N) complexity "
        "for sequence modeling. First model to achieve strong performance on Long Range "
        "Arena benchmark including Path-X (16K tokens)."
    ),

    mathematical_formulation=r"""
Continuous-Time State Space Model:
    x'(t) = Ax(t) + Bu(t)    (state dynamics)
    y(t)  = Cx(t) + Du(t)    (observation)

    Parameters: A in C^(NxN), B in C^(Nx1), C in C^(1xN), D in C

Discretization (bilinear/ZOH):
    x_k = A_bar * x_{k-1} + B_bar * u_k
    y_k = C * x_k + D * u_k

    Bilinear: A_bar = (I - Delta/2 * A)^(-1) * (I + Delta/2 * A)
              B_bar = (I - Delta/2 * A)^(-1) * Delta * B

Convolutional View:
    y = K * u    (1D convolution)
    K = (C * B, C * A_bar * B, C * A_bar^2 * B, ..., C * A_bar^(L-1) * B)

    Kernel computation via Cauchy kernel: O(N * L) -> O(N + L) log(N + L)

HiPPO Initialization (Legendre polynomials):
    A_nk = -(2n+1)^(1/2) * (2k+1)^(1/2)  if n > k
         = n + 1                          if n = k
         = 0                              if n < k

DPLR Structure (Diagonal Plus Low-Rank):
    A = Lambda - P * P^H
    Where Lambda is diagonal, P is low-rank

    Enables efficient computation via Woodbury identity
""",

    predecessors=["rnn_1986", "hippo_2020"],
    successors=["s4d_2022", "h3_2022", "mamba_2023"],

    tags=["state-space", "sequence-modeling", "long-range", "hippo", "efficient"],
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def state_space_continuous(A, B, C, D):
    """
    Define continuous-time state space model.

    x'(t) = Ax(t) + Bu(t)
    y(t)  = Cx(t) + Du(t)

    Args:
        A: State matrix [N, N] - controls state dynamics
        B: Input matrix [N, 1] - controls input injection
        C: Output matrix [1, N] - controls observation
        D: Feedthrough [1] - direct input-output connection

    Returns:
        Dictionary describing the continuous system
    """
    return {
        "state_equation": "x'(t) = Ax(t) + Bu(t)",
        "output_equation": "y(t) = Cx(t) + Du(t)",
        "interpretation": "Linear time-invariant dynamical system",
        "state_dim": "N (hidden dimension)",
        "input_dim": "1 (per-channel)",
        "output_dim": "1 (per-channel)"
    }


def discretize_bilinear(A, B, C, D, delta):
    """
    Discretize continuous SSM using bilinear transform.

    The bilinear method (Tustin's method) preserves stability:
    A_bar = (I - Delta/2 * A)^(-1) * (I + Delta/2 * A)
    B_bar = (I - Delta/2 * A)^(-1) * Delta * B

    Args:
        A: Continuous state matrix [N, N]
        B: Continuous input matrix [N, 1]
        C: Output matrix [1, N]
        D: Feedthrough scalar
        delta: Discretization step size

    Returns:
        Dictionary with discretized parameters

    Properties:
        - Stable continuous system -> stable discrete system
        - Preserves frequency response characteristics
    """
    return {
        "A_bar": "(I - delta/2 * A)^(-1) * (I + delta/2 * A)",
        "B_bar": "(I - delta/2 * A)^(-1) * delta * B",
        "C_bar": "C (unchanged)",
        "D_bar": "D (unchanged)",
        "recurrence": "x_k = A_bar * x_{k-1} + B_bar * u_k",
        "output": "y_k = C * x_k + D * u_k"
    }


def compute_ssm_kernel(A_bar, B, C, length):
    """
    Compute convolutional kernel from discretized SSM.

    K = (C*B, C*A_bar*B, C*A_bar^2*B, ..., C*A_bar^(L-1)*B)

    This kernel allows SSM to be computed as convolution during training.

    Args:
        A_bar: Discretized state matrix [N, N]
        B: Input matrix [N, 1]
        C: Output matrix [1, N]
        length: Sequence length L

    Returns:
        Dictionary describing kernel computation

    Complexity:
        Naive: O(N^2 * L) via matrix powers
        S4 (DPLR): O((N + L) * log(N + L)) via Cauchy kernel + FFT
    """
    return {
        "kernel": "K_i = C * A_bar^i * B for i = 0, ..., L-1",
        "convolution": "y = K * u (1D convolution)",
        "naive_complexity": "O(N^2 * L)",
        "s4_complexity": "O((N + L) * log(N + L))",
        "dplr_trick": "Use diagonal + low-rank structure with Cauchy kernel"
    }


def hippo_matrix(N):
    """
    Construct HiPPO-LegS matrix for optimal polynomial projection.

    HiPPO (High-order Polynomial Projection Operator) initializes A
    to optimally compress the history of a function into coefficients
    of orthogonal polynomials.

    Args:
        N: State dimension (number of polynomial coefficients)

    Returns:
        Dictionary describing HiPPO initialization

    Mathematical Form:
        A_nk = -sqrt(2n+1) * sqrt(2k+1)  if n > k
             = n + 1                      if n = k
             = 0                          if n < k

    This gives optimal compression of history using Legendre polynomials.
    """
    return {
        "lower_triangular": "A_nk = -sqrt(2n+1) * sqrt(2k+1) for n > k",
        "diagonal": "A_nn = n + 1",
        "upper_triangular": "A_nk = 0 for n < k",
        "interpretation": "Project history onto Legendre polynomial basis",
        "property": "Optimal compression of sliding window of history"
    }


def dplr_representation(Lambda, P):
    """
    Diagonal Plus Low-Rank representation for efficient computation.

    A = Lambda - P * P^H

    Where Lambda is diagonal and P is low-rank. This structure enables
    efficient kernel computation via the Woodbury identity.

    Args:
        Lambda: Diagonal matrix [N] (complex diagonal entries)
        P: Low-rank factor [N, r] (typically r=1 or r=2)

    Returns:
        Dictionary describing DPLR structure

    Key Insight:
        HiPPO matrix can be decomposed into DPLR form, enabling
        O(N) computation of (zI - A)^(-1) via Woodbury identity.
    """
    return {
        "decomposition": "A = diag(Lambda) - P @ P.conj().T",
        "woodbury": "(zI - A)^(-1) = (zI - Lambda)^(-1) + correction",
        "cauchy_kernel": "Sum over Cauchy terms: 1/(z - lambda_i)",
        "complexity": "O(N) per frequency point",
        "fft_speedup": "FFT for frequency domain -> O(N log L)"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class S4Architecture:
    """Reference architecture for S4 layer."""

    input_dim: int = 256
    state_dim: int = 64  # N
    num_layers: int = 4

    @staticmethod
    def s4_layer_structure() -> str:
        """Single S4 layer structure."""
        return """
S4 Layer:
    Input: u [batch_size, seq_len, d_model]

    # Per-channel SSM (H independent SSMs)
    for h in range(d_model):
        # Option 1: Convolutional mode (training)
        K = compute_ssm_kernel(A_bar, B, C, seq_len)
        y_h = conv1d(u[:, :, h], K)

        # Option 2: Recurrent mode (inference)
        x = 0  # initial state
        for k in range(seq_len):
            x = A_bar @ x + B_bar * u[k, h]
            y_h[k] = C @ x + D * u[k, h]

    # Position-wise feedforward
    y = Linear(y) + u  # residual connection

    Output: y [batch_size, seq_len, d_model]
"""

    @staticmethod
    def full_model_structure() -> str:
        """Full S4 model stack."""
        return """
S4 Model:
    Input: x [batch_size, seq_len, input_dim]

    # Embedding
    h = Linear(x)  # [batch_size, seq_len, d_model]

    # S4 layers with residual connections
    for layer in S4_layers:
        h = LayerNorm(h)
        h = h + S4_Layer(h)
        h = h + FFN(h)

    # Output projection
    output = Linear(h)

    Output: [batch_size, seq_len, output_dim]
"""


# =============================================================================
# S4 Variants Reference
# =============================================================================

S4_VARIANTS = {
    "s4d": {
        "name": "S4D (Diagonal S4)",
        "year": 2022,
        "description": "Simplified S4 with purely diagonal state matrix",
        "key_change": "A = diag(Lambda), no low-rank component",
        "benefit": "Simpler implementation, similar performance"
    },
    "s4nd": {
        "name": "S4ND (S4 for N-Dimensional data)",
        "description": "Extension to 2D/3D data (images, video)",
        "key_change": "Multi-dimensional state space formulation",
        "application": "Vision tasks without attention"
    },
    "s5": {
        "name": "S5 (Simplified S4)",
        "year": 2022,
        "description": "Single MIMO SSM instead of H independent SISO SSMs",
        "key_change": "Multi-input multi-output state space",
        "benefit": "More parameter efficient, parallel scan implementation"
    },
    "h3": {
        "name": "H3 (Hungry Hungry Hippos)",
        "year": 2022,
        "description": "Hybrid SSM with multiplicative gating",
        "key_change": "SSM + linear attention hybrid",
        "benefit": "Better language modeling performance"
    }
}
