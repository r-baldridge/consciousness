"""
Mamba - Selective State Space Models

A family of efficient sequence models based on structured state space models (SSMs)
with selective/input-dependent mechanisms. Achieves linear-time complexity while
matching or exceeding Transformer performance on language modeling.

=============================================================================
MAMBA FAMILY TIMELINE
=============================================================================

S4 (Structured State Space Sequence) - 2021
    Paper: "Efficiently Modeling Long Sequences with Structured State Spaces"
    arXiv: https://arxiv.org/abs/2111.00396
    Authors: Albert Gu, Karan Goel, Christopher Ré (Stanford)
    Key: HiPPO initialization, diagonal plus low-rank structure

S4D (Diagonal S4) - 2022
    Paper: "On the Parameterization and Initialization of Diagonal State Space Models"
    arXiv: https://arxiv.org/abs/2206.11893
    Key: Simplified diagonal parameterization

Mamba (Selective SSM) - 2023
    Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    arXiv: https://arxiv.org/abs/2312.00752
    GitHub: https://github.com/state-spaces/mamba
    Authors: Albert Gu, Tri Dao (Carnegie Mellon, Princeton)
    Key: Input-dependent selection mechanism, hardware-aware algorithm

Mamba-2 - 2024
    Paper: "Transformers are SSMs: Generalized Models and Efficient Algorithms
            Through Structured State Space Duality"
    arXiv: https://arxiv.org/abs/2405.21060
    Key: SSM-Attention duality, 2-8x faster than Mamba-1

Jamba - 2024
    Paper: "Jamba: A Hybrid Transformer-Mamba Language Model"
    arXiv: https://arxiv.org/abs/2403.19887
    Organization: AI21 Labs
    Key: Hybrid Mamba-Attention-MoE architecture, 256K context

=============================================================================
CORE SSM CONCEPT
=============================================================================

State Space Models (SSMs) map input sequences to output sequences through
a hidden state, similar to RNNs but with specific structure enabling
efficient parallel computation.

Continuous-Time SSM:
    h'(t) = A h(t) + B x(t)     # State evolution
    y(t)  = C h(t) + D x(t)     # Output projection

Discrete-Time (after zero-order hold discretization):
    h_k = Ā h_{k-1} + B̄ x_k
    y_k = C h_k + D x_k

Where:
    h ∈ ℝ^N  = hidden state (N = state dimension, typically 16-64)
    x ∈ ℝ    = input (per channel)
    y ∈ ℝ    = output
    A ∈ ℝ^(N×N) = state matrix (structured/diagonal)
    B ∈ ℝ^(N×1) = input matrix
    C ∈ ℝ^(1×N) = output matrix
    D ∈ ℝ      = skip connection

=============================================================================
MAMBA'S KEY INNOVATION: SELECTION MECHANISM
=============================================================================

Traditional SSMs: A, B, C are fixed (input-independent)
    - Can be computed as convolution: y = x * k where k = C @ exp(A) @ B
    - Very efficient but limited expressiveness

Mamba: A, B, C are input-dependent (selective)
    - Δ, B, C = f(x)  where f is learned projections
    - Allows content-based reasoning (like attention)
    - But linear complexity O(L) instead of O(L²)

Selection enables:
    - Focus on relevant inputs
    - Filter out irrelevant context
    - Reset state at appropriate boundaries
    - Content-aware processing

=============================================================================
MAMBA ARCHITECTURE
=============================================================================

                     ┌─────────────────────────────────────┐
    Input x          │                                     │
        │            │    Linear Projection (expand)       │
        ▼            │              │                      │
    ┌───────┐        │         ┌────┴────┐                 │
    │ Norm  │        │         ▼         ▼                 │
    └───┬───┘        │    ┌─────────┐ ┌─────────┐          │
        │            │    │  Conv1D │ │ Linear  │          │
        ▼            │    └────┬────┘ │ (gate)  │          │
    ┌───────────┐    │         │      └────┬────┘          │
    │   Mamba   │ ◄──┤         ▼           │               │
    │   Block   │    │    ┌─────────┐      │               │
    └─────┬─────┘    │    │   SiLU  │      │               │
          │          │    └────┬────┘      │               │
          ▼          │         │           │               │
    ┌───────────┐    │    ┌────┴───────────┘               │
    │  Residual │    │    ▼                                │
    │     +     │    │ ┌───────────────────┐               │
    └───────────┘    │ │   Selective SSM   │               │
                     │ │  (Δ,B,C from x)   │               │
                     │ └─────────┬─────────┘               │
                     │           │                         │
                     │           ▼                         │
                     │    ┌─────────┐                      │
                     │    │  SiLU   │ × gate               │
                     │    └────┬────┘                      │
                     │         │                           │
                     │         ▼                           │
                     │   Linear (contract)                 │
                     │           │                         │
                     └───────────┼─────────────────────────┘
                                 ▼
                             Output

=============================================================================
SELECTIVE SSM COMPUTATION
=============================================================================

Input-Dependent Parameters:
    Δ = softplus(Linear_Δ(x))      # Discretization step (controls memory)
    B = Linear_B(x)                 # Input projection
    C = Linear_C(x)                 # Output projection

Discretization:
    Ā = exp(Δ ⊗ A)                  # Element-wise for diagonal A
    B̄ = Δ ⊗ B                       # Simplified discretization

Recurrent Computation (inference):
    h_k = Ā_k ⊙ h_{k-1} + B̄_k ⊙ x_k
    y_k = C_k ⊙ h_k

Parallel Scan (training):
    Use associative scan for parallel computation:
    (a₁, b₁) ⊕ (a₂, b₂) = (a₁ × a₂, a₂ × b₁ + b₂)

    Enables O(L) parallel training on GPU

=============================================================================
HARDWARE-AWARE ALGORITHM
=============================================================================

Challenge: Naive selective SSM requires materializing O(BLDN) state

Mamba's Solution:
    1. Fused kernel: Combine discretization + scan + output in single kernel
    2. Recomputation: Don't store intermediate states, recompute in backward
    3. Memory: O(BLD) instead of O(BLDN)
    4. Kernel fusion: Minimize HBM reads/writes

Implementation Strategy:
    - Load parameters from HBM to SRAM once
    - Compute discretization in SRAM
    - Perform selective scan in SRAM
    - Write only final output to HBM

    Result: 3-5x faster than naive implementation

=============================================================================
MAMBA-2 IMPROVEMENTS
=============================================================================

Key Insight: SSMs and Attention are duals under certain conditions

State Space Duality (SSD):
    Attention: y = softmax(QK^T/√d) V
    SSM:       y = M @ x  where M_ij depends on A, B, C

    When A is scalar (per head): SSM ≈ Linear Attention

Mamba-2 Architecture:
    - Multi-head structure (like attention)
    - Larger state dimension (64 → 128-256)
    - SSD algorithm: 2-8x faster than Mamba-1
    - Stronger theoretical foundation

=============================================================================
JAMBA: HYBRID ARCHITECTURE
=============================================================================

Combines three components:
    1. Mamba layers: Long-range, linear complexity
    2. Attention layers: Fine-grained pattern matching
    3. MoE layers: Capacity without compute cost

Architecture (52B parameters, 12B active):
    - Ratio: 7:1 Mamba to Attention layers
    - MoE on MLP layers (16 experts, 2 active)
    - 256K context window
    - Single 80GB GPU inference

Benefits:
    - Attention for precise retrieval
    - Mamba for long-range dependencies
    - MoE for capacity efficiency

=============================================================================
PERFORMANCE COMPARISON
=============================================================================

Language Modeling (perplexity):
    Model           Params    Context   Pile PPL
    ─────────────────────────────────────────────
    GPT-3           175B      2K        -
    LLaMA           7B        2K        6.97
    Mamba           2.8B      ∞*        6.22
    Mamba-2         2.7B      ∞*        ~6.0
    Jamba           12B*      256K      -

    * Mamba can handle arbitrary length at inference
    * Jamba: 12B active of 52B total

Inference Speed (tokens/sec on A100):
    Sequence Length    Transformer    Mamba
    ─────────────────────────────────────────
    2K                 ~5000          ~8000
    8K                 ~2000          ~8000
    32K                ~500           ~8000
    128K               OOM            ~7500

Training Efficiency:
    - 5x faster than FlashAttention-2 for 64K sequences
    - Linear scaling vs quadratic

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - Triton >= 2.0 (for custom kernels)
    - CUDA >= 11.6
    - einops
    - causal-conv1d (optional, for faster conv)

Core Components:
    1. Selective SSM layer (with input-dependent Δ, B, C)
    2. Causal Conv1D (short convolution before SSM)
    3. Parallel scan implementation (associative scan)
    4. Hardware-efficient fused kernel

Model Sizes:
    - Mamba-130M: d_model=768,  n_layer=24
    - Mamba-370M: d_model=1024, n_layer=48
    - Mamba-790M: d_model=1536, n_layer=48
    - Mamba-1.4B: d_model=2048, n_layer=48
    - Mamba-2.8B: d_model=2560, n_layer=64

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core SSM Implementation
    - [ ] S4D layer (diagonal SSM)
    - [ ] HiPPO initialization
    - [ ] Discretization methods
    - [ ] Parallel scan (associative scan)

Phase 2: Selective Mamba
    - [ ] Selection mechanism (Δ, B, C projection)
    - [ ] Fused selective scan kernel
    - [ ] Memory-efficient backward pass
    - [ ] Causal Conv1D integration

Phase 3: Full Architecture
    - [ ] Mamba block (with gating)
    - [ ] Mamba language model
    - [ ] Mamba-2 SSD algorithm
    - [ ] Multi-head selective SSM

Phase 4: Hybrid & Advanced
    - [ ] Jamba-style hybrid (Mamba + Attention)
    - [ ] MoE integration
    - [ ] Vision Mamba adaptation
    - [ ] Bidirectional Mamba

Phase 5: CLI Tools
    - [ ] mamba-train: Training script
    - [ ] mamba-generate: Text generation
    - [ ] mamba-convert: Convert from/to HF format
    - [ ] mamba-benchmark: Speed/memory benchmarks
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Model variants registry
VARIANTS = {
    "s4": {
        "name": "S4",
        "year": 2021,
        "paper_url": "https://arxiv.org/abs/2111.00396",
        "github_url": "https://github.com/state-spaces/s4",
        "organization": "Stanford",
        "key_innovation": "HiPPO initialization + diagonal plus low-rank structure",
    },
    "s4d": {
        "name": "S4D",
        "year": 2022,
        "paper_url": "https://arxiv.org/abs/2206.11893",
        "github_url": "https://github.com/state-spaces/s4",
        "organization": "Stanford",
        "key_innovation": "Simplified diagonal parameterization",
    },
    "mamba": {
        "name": "Mamba",
        "year": 2023,
        "paper_url": "https://arxiv.org/abs/2312.00752",
        "github_url": "https://github.com/state-spaces/mamba",
        "organization": "Carnegie Mellon / Princeton",
        "key_innovation": "Input-dependent selection mechanism + hardware-aware algorithm",
    },
    "mamba_2": {
        "name": "Mamba-2",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2405.21060",
        "github_url": "https://github.com/state-spaces/mamba",
        "organization": "Carnegie Mellon / Princeton",
        "key_innovation": "SSM-Attention duality, SSD algorithm, 2-8x speedup",
    },
    "jamba": {
        "name": "Jamba",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2403.19887",
        "github_url": None,
        "organization": "AI21 Labs",
        "key_innovation": "Hybrid Mamba-Attention-MoE, 256K context",
    },
}

# Mathematical formulation
FORMULATION = """
State Space Model (Continuous):
    h'(t) = A h(t) + B x(t)
    y(t)  = C h(t) + D x(t)

Discrete (Zero-Order Hold):
    Ā = exp(Δ A)
    B̄ = (Δ A)^{-1} (exp(Δ A) - I) Δ B  ≈ Δ B (for small Δ)

    h_k = Ā h_{k-1} + B̄ x_k
    y_k = C h_k

Mamba Selection Mechanism:
    Δ = softplus(W_Δ x + b_Δ)
    B = W_B x
    C = W_C x

    Note: A is learned but input-independent (diagonal)

Parallel Scan (Associative):
    Given pairs (a_i, b_i) representing h_i = a_i * h_{i-1} + b_i

    (a₁, b₁) ⊕ (a₂, b₂) = (a₁ a₂, a₂ b₁ + b₂)

    Enables O(log L) parallel computation
"""

# Default configurations
MAMBA_130M_CONFIG = {
    "d_model": 768,
    "n_layer": 24,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "vocab_size": 50280,
}

MAMBA_1_4B_CONFIG = {
    "d_model": 2048,
    "n_layer": 48,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "vocab_size": 50280,
}

MAMBA_2_CONFIG = {
    "d_model": 2048,
    "n_layer": 48,
    "d_state": 128,  # Larger state in Mamba-2
    "headdim": 64,   # Multi-head
    "ngroups": 8,
    "vocab_size": 50280,
}

JAMBA_CONFIG = {
    "d_model": 4096,
    "n_layer": 32,
    "n_mamba_layer": 28,  # 7:1 ratio
    "n_attention_layer": 4,
    "moe_num_experts": 16,
    "moe_num_active": 2,
    "context_length": 262144,  # 256K
}

# Placeholder imports
# from .src.ssm import S4D, SelectiveSSM
# from .src.mamba_block import MambaBlock
# from .src.parallel_scan import parallel_scan
# from .cli.train import main as train
