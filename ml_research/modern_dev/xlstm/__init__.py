"""
Extended Long Short-Term Memory (xLSTM) - NXAI Lab, 2024

A modernized LSTM architecture that reintroduces recurrent networks as
competitive alternatives to Transformers through exponential gating and
matrix-valued memory cells. Achieves transformer-level performance with
linear complexity.

Paper: "xLSTM: Extended Long Short-Term Memory"
arXiv: https://arxiv.org/abs/2405.04517
GitHub: https://github.com/NX-AI/xlstm
Authors: Maximilian Beck, Korbinian Poeppel, Markus Spanring, Andreas Auer,
         Oleksandra Prudnikova, Michael Kopp, Gunter Klambauer,
         Johannes Brandstetter, Sepp Hochreiter
Organization: NXAI Lab, Johannes Kepler University, Linz

=============================================================================
MOTIVATION: REVIVING LSTM
=============================================================================

LSTM's Original Limitations:
    1. Limited storage capacity (scalar cell state)
    2. Parallelization difficulties (sequential dependencies)
    3. Vanishing/exploding gradients at extreme lengths

xLSTM's Solutions:
    1. Exponential gating → stabilizes gradients, improves memory
    2. Matrix memory (mLSTM) → massively increased storage
    3. Parallelizable sLSTM → enables efficient GPU utilization

Key Insight:
    "What happens if we scale LSTMs to billions of parameters
     and equip them with modern techniques?"

    Answer: They match or exceed Transformers on many tasks

=============================================================================
ARCHITECTURE VARIANTS
=============================================================================

xLSTM has TWO complementary cell types:

┌─────────────────────────────────────────────────────────────────────────────┐
│                              sLSTM (scalar LSTM)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Purpose: Memory mixing with parallelizable structure                       │
│  Memory: Scalar cell state (like original LSTM)                            │
│  Gating: Exponential gates for stability                                   │
│  Parallelization: Yes, via associative scan                                │
│  Use case: Layers requiring memory state mixing                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             mLSTM (matrix LSTM)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Purpose: Massively increased storage capacity                             │
│  Memory: Matrix-valued cell state (C in R^{d x d})                        │
│  Gating: Exponential gates + covariance update rule                        │
│  Parallelization: Yes, fully parallelizable                                │
│  Use case: Main information storage backbone                               │
└─────────────────────────────────────────────────────────────────────────────┘

Typical xLSTM Block:
    [mLSTM] → [Conv] → [sLSTM] → [mLSTM] → ...

    mLSTM provides storage, sLSTM provides mixing

=============================================================================
sLSTM: SCALAR LSTM WITH EXPONENTIAL GATING
=============================================================================

Standard LSTM:
    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)    # input gate
    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)    # forget gate
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)    # output gate
    c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)      # candidate
    c_t = f_t * c_{t-1} + i_t * c̃_t              # cell update
    h_t = o_t * tanh(c_t)                         # hidden state

sLSTM (Exponential Gating):
    i_t = exp(W_i x_t + U_i h_{t-1} + b_i)        # EXPONENTIAL input gate
    f_t = exp(W_f x_t + U_f h_{t-1} + b_f)        # EXPONENTIAL forget gate
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)    # sigmoid output (unchanged)
    c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)      # candidate (unchanged)

    Normalizer state (prevents explosion):
        n_t = f_t * n_{t-1} + i_t
        c_t = f_t * c_{t-1} + i_t * c̃_t
        h_t = o_t * (c_t / n_t)                   # normalized output

Why Exponential Gating?
    - sigmoid: output in (0, 1) → gradients vanish over long sequences
    - exp: unbounded → maintains gradient flow
    - normalizer n_t: prevents values from exploding
    - Result: stable training at extreme sequence lengths

Parallelization via Associative Scan:
    c_t = sum_{s=1}^{t} (prod_{r=s+1}^{t} f_r) * i_s * c̃_s

    This can be computed in O(log T) parallel steps using:
        (a_1, b_1) ⊕ (a_2, b_2) = (a_1 * a_2, a_1 * b_2 + b_1)

=============================================================================
mLSTM: MATRIX MEMORY LSTM
=============================================================================

Core Innovation:
    Instead of scalar cell c_t in R, use matrix C_t in R^{d x d}

Key-Value Memory (inspired by attention):
    C_t = f_t * C_{t-1} + i_t * v_t k_t^T

    where:
        k_t = key vector (what to store under)
        v_t = value vector (what to store)
        C_t = matrix storing key-value associations

Retrieval (like attention):
    h_t = o_t * (C_t q_t) / (n_t^T q_t)

    where:
        q_t = query vector (what to retrieve)
        n_t = normalizer vector (prevents explosion)

Normalizer Update:
    n_t = f_t * n_{t-1} + i_t * k_t

Full mLSTM Update Equations:
    q_t = W_q x_t                    # query projection
    k_t = (1/sqrt(d)) * W_k x_t      # key projection (normalized)
    v_t = W_v x_t                    # value projection

    i_t = exp(w_i^T x_t + b_i)       # exponential input gate (scalar)
    f_t = exp(w_f^T x_t + b_f)       # exponential forget gate (scalar)
    o_t = sigmoid(W_o x_t + b_o)     # output gate (vector)

    C_t = f_t * C_{t-1} + i_t * v_t k_t^T   # matrix memory update
    n_t = f_t * n_{t-1} + i_t * k_t         # normalizer update

    h̃_t = C_t q_t                   # retrieve from memory
    h_t = o_t * h̃_t / max(|n_t^T q_t|, 1)  # gated, normalized output

Why Matrix Memory?
    - Scalar cell: O(d) storage → limited capacity
    - Matrix cell: O(d^2) storage → d times more capacity
    - Can store d separate key-value pairs
    - Retrieval is O(d^2) but highly parallelizable (matmul)

Connection to Linear Attention:
    If f_t = 1 and i_t = 1 for all t:
        C_t = sum_{s=1}^{t} v_s k_s^T
        h_t = (sum_{s=1}^{t} v_s k_s^T) q_t = sum_{s=1}^{t} (k_s^T q_t) v_s

    This IS linear attention! mLSTM adds:
        - Exponential gating (f_t, i_t) for memory control
        - Output gating (o_t) for selective output

=============================================================================
xLSTM BLOCK ARCHITECTURE
=============================================================================

Pre-LayerNorm xLSTM Block:

    x ─────┬──────────────────────────────────────────────────┐
           │                                                  │
           ▼                                                  │
    ┌─────────────┐                                          │
    │  LayerNorm  │                                          │
    └─────────────┘                                          │
           │                                                  │
           ▼                                                  │
    ┌─────────────┐     ┌─────────────┐                      │
    │   Up Proj   │ ──► │  Conv 1D    │ (optional)           │
    │   (2-4x)    │     │  (causal)   │                      │
    └─────────────┘     └─────────────┘                      │
           │                   │                              │
           ▼                   ▼                              │
    ┌───────────────────────────────┐                        │
    │     sLSTM or mLSTM Cell       │                        │
    └───────────────────────────────┘                        │
           │                                                  │
           ▼                                                  │
    ┌─────────────┐                                          │
    │  Down Proj  │                                          │
    └─────────────┘                                          │
           │                                                  │
           ▼                                                  │
    ┌─────────────┐                                          │
    │   + (Add)   │◄─────────────────────────────────────────┘
    └─────────────┘
           │
           ▼

Block Types:
    - sLSTM blocks: Include causal conv for local patterns
    - mLSTM blocks: No conv needed (attention-like global patterns)

=============================================================================
BENCHMARKS
=============================================================================

Language Modeling Perplexity (Pile, 15B training tokens):

    Model               Parameters    Perplexity
    ───────────────────────────────────────────────
    Transformer         125M          37.2
    RWKV-4             124M          38.1
    Mamba              130M          36.8
    xLSTM[1:0]         125M          36.3         # all mLSTM
    xLSTM[7:1]         125M          35.9         # mixed

    xLSTM[7:1] = 7 mLSTM blocks : 1 sLSTM block ratio

Scaling (up to 1.3B parameters):
    - xLSTM matches Transformer at all scales tested
    - Better than Mamba and RWKV at most scales
    - Favorable scaling trend suggests potential at larger scales

Long Context (4K to 16K tokens):
    - xLSTM maintains quality at longer contexts
    - mLSTM especially effective for retrieval tasks
    - Linear complexity enables practical long-context training

Associative Recall (synthetic benchmark):
    - mLSTM significantly outperforms Mamba
    - Competitive with Transformer attention
    - Matrix memory crucial for this capability

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - Triton >= 2.0 (for efficient kernels)
    - einops
    - flash-attn (optional, for hybrid models)

Core Components to Implement:
    1. sLSTMCell: Exponential gating with normalizer
    2. mLSTMCell: Matrix memory with key-value storage
    3. xLSTMBlock: Pre-norm block with up/down projection
    4. xLSTMModel: Full language model stack
    5. Triton kernels for parallel scan operations

Hardware Considerations:
    - mLSTM: O(d^2) memory per layer (matrix cell state)
    - Highly parallelizable (unlike traditional LSTM)
    - Benefits from tensor cores (matmul-heavy in mLSTM)
    - Efficient on both training and inference

=============================================================================
COMPARISON WITH OTHER ARCHITECTURES
=============================================================================

                    xLSTM       Transformer     Mamba       RWKV        TTT
Context Complexity  O(n)        O(n^2)          O(n)        O(n)        O(n)
Memory per Layer    O(d^2)      O(n*d)          O(d)        O(d)        O(d^2)
Parallelizable      Yes         Yes             Yes         Partial     Yes
Recurrent State     Yes         No*             Yes         Yes         Yes
Associative Recall  Strong      Strong          Weak        Weak        Strong
Training Stability  High        High            Medium      Medium      High

*Transformers can be made recurrent with KV-cache but lose global attention

xLSTM vs Mamba:
    - xLSTM: Explicit memory gating (LSTM heritage)
    - Mamba: Selection mechanism (S4 heritage)
    - xLSTM stronger on associative recall
    - Similar performance on language modeling

xLSTM vs TTT:
    - Both have O(d^2) per-layer state
    - TTT: State is learned via gradient descent
    - xLSTM: State updated via explicit gating rules
    - Different inductive biases

=============================================================================
MATHEMATICAL FORMULATION (COMPLETE)
=============================================================================

sLSTM Forward Pass:
    Input: x_t in R^d, h_{t-1} in R^d, c_{t-1} in R^d, n_{t-1} in R

    Gates (exponential):
        z_i = W_i x_t + R_i h_{t-1} + b_i
        z_f = W_f x_t + R_f h_{t-1} + b_f
        i_t = exp(z_i)
        f_t = exp(z_f)

    Candidate and output gate:
        z_c = W_c x_t + R_c h_{t-1} + b_c
        z_o = W_o x_t + R_o h_{t-1} + b_o
        c̃_t = tanh(z_c)
        o_t = sigmoid(z_o)

    State updates:
        c_t = f_t * c_{t-1} + i_t * c̃_t
        n_t = f_t * n_{t-1} + i_t
        h_t = o_t * (c_t / max(n_t, 1))

mLSTM Forward Pass:
    Input: x_t in R^d, C_{t-1} in R^{d x d}, n_{t-1} in R^d

    Projections:
        q_t = W_q x_t                           in R^d
        k_t = (1/sqrt(d)) * W_k x_t             in R^d
        v_t = W_v x_t                           in R^d

    Gates:
        i_t = exp(w_i^T x_t + b_i)              scalar
        f_t = exp(w_f^T x_t + b_f)              scalar
        o_t = sigmoid(W_o x_t + b_o)            in R^d

    State updates:
        C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)    in R^{d x d}
        n_t = f_t * n_{t-1} + i_t * k_t              in R^d

    Output:
        h̃_t = C_t @ q_t                              in R^d
        denom = max(|n_t^T @ q_t|, 1)                scalar
        h_t = o_t * h̃_t / denom                      in R^d

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core Cells
    - [ ] sLSTMCell with exponential gating
    - [ ] mLSTMCell with matrix memory
    - [ ] Normalizer state management
    - [ ] Unit tests for numerical stability

Phase 2: Parallel Scan
    - [ ] Associative scan for sLSTM
    - [ ] Triton kernel for mLSTM parallel computation
    - [ ] Benchmark against sequential implementation

Phase 3: Block Architecture
    - [ ] xLSTMBlock with up/down projections
    - [ ] Causal convolution integration
    - [ ] Pre-LayerNorm residual connections

Phase 4: Full Model
    - [ ] xLSTMLanguageModel
    - [ ] Configurable mLSTM:sLSTM ratio
    - [ ] Pre-trained checkpoint loading

Phase 5: CLI Tools
    - [ ] xlstm-train: Training script
    - [ ] xlstm-infer: Text generation
    - [ ] xlstm-analyze: Memory cell visualization
    - [ ] xlstm-benchmark: Comparison suite

=============================================================================
CONFIGURATION EXAMPLES
=============================================================================

Small Model (125M params):
    - embedding_dim: 768
    - num_layers: 12
    - mlstm_layers: [0, 1, 2, 4, 5, 6, 8, 9, 10]  # 9 mLSTM
    - slstm_layers: [3, 7, 11]                      # 3 sLSTM
    - up_proj_factor: 2
    - conv_kernel_size: 4

Medium Model (350M params):
    - embedding_dim: 1024
    - num_layers: 24
    - mlstm_layers: [0,1,2,3,4,5,6, 8,9,10,11,12,13,14, 16,17,18,19,20,21,22]
    - slstm_layers: [7, 15, 23]
    - up_proj_factor: 2.67

Large Model (1.3B params):
    - embedding_dim: 2048
    - num_layers: 36
    - Ratio: 7:1 mLSTM to sLSTM
    - up_proj_factor: 4

=============================================================================
CONNECTIONS TO CONSCIOUSNESS RESEARCH
=============================================================================

Relevance to Memory Systems:
    - mLSTM matrix memory = content-addressable memory
    - Gating mechanism = attention allocation
    - Normalizer = capacity management

Episodic vs Semantic Memory:
    - mLSTM could implement episodic storage (key-value pairs)
    - sLSTM could implement working memory mixing
    - Combination mirrors hippocampal-cortical interaction

Global Workspace Theory:
    - sLSTM mixing = broadcasting across workspace
    - mLSTM storage = peripheral memory systems
    - Output gating = access consciousness

Integration Points:
    - xLSTM as memory substrate in cognitive architecture
    - mLSTM for episodic memory in JEPA world models
    - Combine with CTM for adaptive temporal reasoning
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Extended Long Short-Term Memory",
    "abbreviation": "xLSTM",
    "year": 2024,
    "organization": "NXAI Lab, JKU Linz",
    "paper_url": "https://arxiv.org/abs/2405.04517",
    "github_url": "https://github.com/NX-AI/xlstm",
    "authors": [
        "Maximilian Beck", "Korbinian Poeppel", "Markus Spanring",
        "Andreas Auer", "Oleksandra Prudnikova", "Michael Kopp",
        "Gunter Klambauer", "Johannes Brandstetter", "Sepp Hochreiter"
    ],
    "key_innovation": "Exponential gating + matrix-valued memory cells",
}

# Cell type variants
VARIANTS = {
    "slstm": {
        "name": "sLSTM (scalar LSTM)",
        "memory_type": "Scalar cell state",
        "gating": "Exponential input/forget gates",
        "parallelization": "Via associative scan",
        "use_case": "Memory mixing layers",
    },
    "mlstm": {
        "name": "mLSTM (matrix LSTM)",
        "memory_type": "Matrix cell state (d x d)",
        "gating": "Exponential + key-value storage",
        "parallelization": "Fully parallel (matmul)",
        "use_case": "Main storage backbone",
    },
}

# Mathematical formulation summary
FORMULATION = """
sLSTM (Exponential Gating):
    i_t = exp(W_i x_t + R_i h_{t-1} + b_i)
    f_t = exp(W_f x_t + R_f h_{t-1} + b_f)
    c_t = f_t * c_{t-1} + i_t * tanh(W_c x_t + R_c h_{t-1})
    n_t = f_t * n_{t-1} + i_t                    # normalizer
    h_t = o_t * (c_t / max(n_t, 1))

mLSTM (Matrix Memory):
    k_t = (1/sqrt(d)) * W_k x_t                  # key
    v_t = W_v x_t                                # value
    q_t = W_q x_t                                # query

    C_t = f_t * C_{t-1} + i_t * v_t k_t^T       # matrix memory update
    n_t = f_t * n_{t-1} + i_t * k_t             # normalizer vector

    h_t = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)  # gated retrieval

Key Properties:
    - Exponential gates: exp() instead of sigmoid()
    - Normalizer state: prevents explosion
    - Matrix memory: O(d^2) storage capacity
    - Fully parallelizable: via associative scan / matmul
"""

# Default configurations
SLSTM_CONFIG = {
    "hidden_dim": 768,
    "num_heads": 4,
    "conv_kernel_size": 4,
    "use_conv": True,
    "dropout": 0.0,
}

MLSTM_CONFIG = {
    "hidden_dim": 768,
    "num_heads": 8,
    "up_proj_factor": 2,
    "dropout": 0.0,
}

XLSTM_MODEL_CONFIG = {
    "vocab_size": 50257,
    "embedding_dim": 768,
    "num_layers": 12,
    "layer_types": ["m", "m", "m", "s", "m", "m", "m", "s", "m", "m", "m", "s"],
    "up_proj_factor": 2,
    "conv_kernel_size": 4,
    "dropout": 0.0,
    "tie_weights": True,
}

# Benchmark results
BENCHMARKS = {
    "pile_perplexity_125M": {
        "transformer": 37.2,
        "rwkv4": 38.1,
        "mamba": 36.8,
        "xlstm_all_mlstm": 36.3,
        "xlstm_7_1_ratio": 35.9,
    },
    "associative_recall": "Strong (competitive with Transformer)",
    "long_context": "Maintains quality to 16K+",
    "scaling_trend": "Favorable to 1.3B parameters",
}

# Placeholder imports - will be implemented
# from .src.slstm import sLSTMCell, sLSTM
# from .src.mlstm import mLSTMCell, mLSTM
# from .src.block import xLSTMBlock
# from .src.model import xLSTMLanguageModel, xLSTMConfig
# from .cli.train import main as train
# from .cli.infer import main as infer
