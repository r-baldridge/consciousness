"""
RWKV (Receptance Weighted Key Value) - BlinkDL / RWKV Foundation

A novel architecture combining the efficient parallelizable training of
Transformers with the efficient inference of RNNs, achieving O(n) time
and memory complexity while maintaining competitive performance.

Paper: "RWKV: Reinventing RNNs for the Transformer Era"
arXiv: https://arxiv.org/abs/2305.13048
GitHub: https://github.com/BlinkDL/RWKV-LM
Author: Bo Peng (BlinkDL) and RWKV Foundation
Year: 2023-2025 (ongoing development)

=============================================================================
VERSION HISTORY
=============================================================================

RWKV-4 (2023):
    - Original architecture from the paper
    - Linear attention mechanism with time-mixing
    - Proved viability of linear complexity language models

RWKV-5 "Eagle" (2024):
    - Multi-headed matrix-valued states
    - Improved expressiveness over RWKV-4
    - Better long-context performance

RWKV-6 "Finch" (2024):
    - Data-dependent linear recurrence
    - Dynamic decay based on input
    - Improved in-context learning

RWKV-7 "Goose" (2025):
    - Latest iteration with architectural refinements
    - Enhanced state capacity and expressiveness
    - State-of-the-art among linear complexity models

=============================================================================
KEY INNOVATIONS
=============================================================================

1. LINEAR COMPLEXITY
   - O(n) time complexity (vs O(n^2) for standard attention)
   - O(n) memory complexity
   - Enables processing of extremely long sequences
   - Constant memory during inference (RNN mode)

2. DUAL MODE OPERATION
   - Training: Parallel mode (like Transformers)
   - Inference: Recurrent mode (like RNNs)
   - Best of both worlds: fast training, efficient inference

3. TIME MIXING (Attention-like)
   - Linear attention without softmax
   - Exponential decay mechanism
   - Receptance gating for information flow

4. CHANNEL MIXING (FFN-like)
   - Replaces standard MLP/FFN
   - Similar gating mechanism as time mixing
   - Efficient parameter usage

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

Standard Transformer Block:
    Input → LayerNorm → Attention → Add → LayerNorm → FFN → Add → Output

RWKV Block:
    Input → LayerNorm → Time Mixing → Add → LayerNorm → Channel Mixing → Add → Output

                    ┌─────────────────────────────────────┐
    Input x_t →     │           RWKV Block                │
                    │                                     │
                    │  ┌─────────────┐                    │
                    │  │ Layer Norm  │                    │
                    │  └──────┬──────┘                    │
                    │         ▼                           │
                    │  ┌─────────────┐                    │
                    │  │ Time Mixing │ ← state_{t-1}      │
                    │  │  (WKV attn) │ → state_t          │
                    │  └──────┬──────┘                    │
                    │         ▼                           │
                    │      Add (residual)                 │
                    │         ▼                           │
                    │  ┌─────────────┐                    │
                    │  │ Layer Norm  │                    │
                    │  └──────┬──────┘                    │
                    │         ▼                           │
                    │  ┌─────────────────┐                │
                    │  │ Channel Mixing  │                │
                    │  │  (gated FFN)    │                │
                    │  └──────┬──────────┘                │
                    │         ▼                           │
                    │      Add (residual)                 │
                    │                                     │
                    └─────────────────────────────────────┘
                              ▼
                          Output x_t

=============================================================================
RWKV ATTENTION FORMULA (Time Mixing)
=============================================================================

The core WKV (Weighted Key Value) mechanism:

    wkv_t = (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} * v_i + e^u * k_t * v_t)
            / (sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^u * k_t)

Where:
    w = learnable decay vector (per channel)
    u = learnable bonus for current token
    k = key projection
    v = value projection
    t = current time step

Simplified Recurrence Form:
    a_t = e^{-w} * a_{t-1} + e^{k_t} * v_t
    b_t = e^{-w} * b_{t-1} + e^{k_t}
    wkv_t = a_t / b_t

Time Mixing Output:
    r_t = sigmoid(W_r * (mu_r * x_t + (1-mu_r) * x_{t-1}))  # receptance
    k_t = W_k * (mu_k * x_t + (1-mu_k) * x_{t-1})          # key
    v_t = W_v * (mu_v * x_t + (1-mu_v) * x_{t-1})          # value

    output = W_o * (r_t * wkv_t)

=============================================================================
CHANNEL MIXING (Gated FFN)
=============================================================================

    r_t = sigmoid(W_r * (mu_r * x_t + (1-mu_r) * x_{t-1}))  # receptance
    k_t = W_k * (mu_k * x_t + (1-mu_k) * x_{t-1})          # key (FFN input)

    output = r_t * (W_v * max(k_t, 0)^2)

Key Difference from Standard FFN:
    - Token-shifted mixing (uses previous token)
    - Receptance gating
    - Squared ReLU activation

=============================================================================
RWKV-6 DATA-DEPENDENT DECAY
=============================================================================

RWKV-6 introduced dynamic, input-dependent decay:

    w_t = w_base + tanh(x_t @ W_decay) * decay_scale

This allows the model to:
    - Adaptively control how quickly past information decays
    - Better handle varying context requirements
    - Improve in-context learning capabilities

=============================================================================
RWKV vs TRANSFORMERS vs RNNs
=============================================================================

                    RWKV        Transformer     RNN/LSTM
Time (train)        O(n)        O(n^2)          O(n)
Time (inference)    O(1)*       O(n)            O(1)
Memory (train)      O(n)        O(n^2)          O(n)
Memory (inference)  O(1)        O(n)            O(1)
Parallelizable      Yes         Yes             No
Long-range deps     Good        Excellent       Limited
Scaling             Proven      Proven          Limited

* O(1) per token in recurrent mode

=============================================================================
MODEL SIZES AND PERFORMANCE
=============================================================================

RWKV-4 Models:
    - 169M, 430M, 1.5B, 3B, 7B, 14B parameters
    - Trained on The Pile dataset

RWKV-5/6 Models:
    - 0.4B, 1.5B, 3B, 7B, 14B parameters
    - Various context lengths: 4K, 8K, 16K, 32K+

RWKV-7 "Goose" (2025):
    - Latest architecture iteration
    - Enhanced state expressiveness
    - Competitive with similar-sized Transformers

Performance Notes:
    - Competitive with GPT-3 class models at similar sizes
    - Particularly efficient for long-context tasks
    - Lower inference costs than Transformers

=============================================================================
TRAINING INFRASTRUCTURE
=============================================================================

Key Training Considerations:
    1. Custom CUDA kernels for WKV computation
    2. Gradient checkpointing for memory efficiency
    3. Careful initialization for stable training
    4. Learning rate scheduling (often cosine)

RWKV-Specific Optimizations:
    - Chunked computation for very long sequences
    - State caching for efficient inference
    - Quantization-friendly architecture

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - CUDA (for efficient training)
    - einops
    - numpy

Custom Components:
    - WKV CUDA kernel (for efficient computation)
    - Can use pure PyTorch (slower but functional)

Key Components to Implement:
    1. WKVKernel - Efficient WKV attention computation
    2. TimeMixing - Time mixing layer with token shift
    3. ChannelMixing - Gated FFN with token shift
    4. RWKVBlock - Complete RWKV block
    5. RWKVModel - Full model with embedding and head

=============================================================================
INFERENCE MODES
=============================================================================

Parallel Mode (Training/Batch):
    - Process entire sequence at once
    - Uses matrix operations
    - Efficient for training and batch inference

Recurrent Mode (Streaming):
    - Process one token at a time
    - Maintains hidden state
    - Constant memory usage
    - Ideal for real-time generation

State Management:
    - Each layer maintains (a, b) or matrix state
    - State carries context information
    - Can be cached and reused

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core Architecture (RWKV-4)
    - [ ] Implement WKV attention (pure PyTorch)
    - [ ] Time mixing layer
    - [ ] Channel mixing layer
    - [ ] Basic RWKV block

Phase 2: Optimizations
    - [ ] CUDA kernel for WKV
    - [ ] State caching utilities
    - [ ] Chunked processing for long sequences

Phase 3: Advanced Versions
    - [ ] RWKV-5 multi-head matrix states
    - [ ] RWKV-6 data-dependent decay
    - [ ] RWKV-7 enhancements

Phase 4: CLI Tools
    - [ ] rwkv-train: Training script
    - [ ] rwkv-infer: Streaming inference
    - [ ] rwkv-convert: Convert model formats
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Receptance Weighted Key Value",
    "abbreviation": "RWKV",
    "year": 2023,
    "organization": "BlinkDL / RWKV Foundation",
    "author": "Bo Peng (BlinkDL)",
    "paper_url": "https://arxiv.org/abs/2305.13048",
    "github_url": "https://github.com/BlinkDL/RWKV-LM",
    "complexity": {
        "training_time": "O(n)",
        "inference_time": "O(1) per token",
        "memory": "O(n) training, O(1) inference",
    },
}

# Version history
VERSIONS = {
    "rwkv_4": {
        "name": "RWKV-4",
        "year": 2023,
        "description": "Original architecture from paper",
    },
    "rwkv_5": {
        "name": "RWKV-5 Eagle",
        "year": 2024,
        "description": "Multi-headed matrix-valued states",
    },
    "rwkv_6": {
        "name": "RWKV-6 Finch",
        "year": 2024,
        "description": "Data-dependent linear recurrence",
    },
    "rwkv_7": {
        "name": "RWKV-7 Goose",
        "year": 2025,
        "description": "Latest iteration with enhanced expressiveness",
    },
}

# Mathematical formulation
FORMULATION = """
WKV Attention (Core Mechanism):

    Recurrence form:
        a_t = e^{-w} * a_{t-1} + e^{k_t} * v_t
        b_t = e^{-w} * b_{t-1} + e^{k_t}
        wkv_t = a_t / b_t

    where:
        w = learnable decay (per channel)
        k_t = key at time t
        v_t = value at time t
        (a, b) = running state

Time Mixing:
    x'_t = mu * x_t + (1-mu) * x_{t-1}     # token shift
    r_t = sigmoid(W_r @ x'_t)               # receptance (gate)
    k_t = W_k @ x'_t                        # key
    v_t = W_v @ x'_t                        # value
    output = W_o @ (r_t * wkv_t)

Channel Mixing:
    x'_t = mu * x_t + (1-mu) * x_{t-1}     # token shift
    r_t = sigmoid(W_r @ x'_t)               # receptance (gate)
    k_t = W_k @ x'_t                        # key (FFN input)
    output = r_t * (W_v @ squared_relu(k_t))

RWKV-6 Data-Dependent Decay:
    w_t = w_base + tanh(W_decay @ x_t) * scale
"""

# Default configuration
DEFAULT_CONFIG = {
    "vocab_size": 50277,
    "hidden_dim": 768,
    "num_layers": 12,
    "context_length": 4096,
    "head_size": 64,
    "decay_lora_dim": 64,  # for RWKV-6
    "use_data_dependent_decay": False,  # RWKV-6 feature
}

# Model size presets
MODEL_SIZES = {
    "small": {
        "hidden_dim": 768,
        "num_layers": 12,
        "params": "169M",
    },
    "medium": {
        "hidden_dim": 1024,
        "num_layers": 24,
        "params": "430M",
    },
    "large": {
        "hidden_dim": 2048,
        "num_layers": 24,
        "params": "1.5B",
    },
    "xl": {
        "hidden_dim": 2560,
        "num_layers": 32,
        "params": "3B",
    },
    "xxl": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "params": "7B",
    },
    "world": {
        "hidden_dim": 5120,
        "num_layers": 40,
        "params": "14B",
    },
}

# Placeholder imports - will be implemented
# from .src.model import RWKV, RWKVConfig
# from .src.wkv import WKVKernel, wkv_attention
# from .src.time_mixing import TimeMixing
# from .src.channel_mixing import ChannelMixing
# from .cli.train import main as train
# from .cli.infer import main as infer
