# Extended Long Short-Term Memory (xLSTM)

> A modernized LSTM architecture that reintroduces recurrent networks as competitive alternatives to Transformers through exponential gating and matrix-valued memory cells.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
**Year:** 2024
**Organization:** NXAI Lab, Johannes Kepler University, Linz

## Overview

xLSTM answers a provocative question: "What happens if we scale LSTMs to billions of parameters and equip them with modern techniques?" The answer is that they match or exceed Transformers on many tasks while maintaining linear complexity. This architecture addresses the original limitations of LSTMs-limited storage capacity, parallelization difficulties, and vanishing gradients-through two complementary innovations.

The first innovation is exponential gating, which replaces the sigmoid gates of traditional LSTMs with exponential functions. This stabilizes gradients and enables training at extreme sequence lengths. A normalizer state prevents value explosion while maintaining gradient flow. The second innovation is matrix-valued memory cells (mLSTM), which expand the scalar cell state to a full matrix C in R^{d x d}, providing d times more storage capacity than traditional LSTMs.

xLSTM combines two cell types: sLSTM (scalar LSTM) provides parallelizable memory mixing with exponential gates, while mLSTM (matrix LSTM) serves as the main storage backbone with its massively increased capacity. The typical architecture uses a 7:1 ratio of mLSTM to sLSTM blocks, achieving competitive performance with Transformers while maintaining the efficiency benefits of recurrent architectures.

## Key Innovations

- **Exponential Gating**: Replaces sigmoid gates with exp() functions for stable gradient flow at extreme sequence lengths, with normalizer states preventing explosion.

- **Matrix Memory (mLSTM)**: Matrix-valued cell state (C in R^{d x d}) provides O(d^2) storage capacity, enabling d separate key-value pair associations.

- **Parallelizable Training**: Both sLSTM (via associative scan) and mLSTM (via matmul) support efficient parallel GPU computation.

## Architecture Diagram

```
xLSTM Block Types:

+-----------------------------------------------------------------------------+
|                              sLSTM (scalar LSTM)                            |
+-----------------------------------------------------------------------------+
|  Purpose: Memory mixing with parallelizable structure                       |
|  Memory: Scalar cell state (like original LSTM)                             |
|  Gating: Exponential gates for stability                                    |
|  Parallelization: Yes, via associative scan                                 |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
|                             mLSTM (matrix LSTM)                             |
+-----------------------------------------------------------------------------+
|  Purpose: Massively increased storage capacity                              |
|  Memory: Matrix-valued cell state (C in R^{d x d})                          |
|  Gating: Exponential gates + covariance update rule                         |
|  Parallelization: Yes, fully parallelizable                                 |
+-----------------------------------------------------------------------------+

Pre-LayerNorm xLSTM Block:

    x ------+----------------------------------------------+
            |                                              |
            v                                              |
    +-------------+                                        |
    |  LayerNorm  |                                        |
    +------+------+                                        |
           v                                               |
    +-------------+     +----------+                       |
    |   Up Proj   | --> | Conv 1D  | (optional)            |
    |   (2-4x)    |     | (causal) |                       |
    +------+------+     +----+-----+                       |
           |                 |                             |
           v                 v                             |
    +-----------------------------+                        |
    |     sLSTM or mLSTM Cell     |                        |
    +-------------+---------------+                        |
                  |                                        |
                  v                                        |
    +-------------+                                        |
    |  Down Proj  |                                        |
    +------+------+                                        |
           |                                               |
           v                                               |
    +------+------+<---------------------------------------+
    |   + (Add)   |
    +------+------+
           |
           v
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| sLSTMCell | Stub | Exponential gating + normalizer |
| mLSTMCell | Stub | Matrix memory + key-value storage |
| xLSTMBlock | Stub | With up/down projections |
| Associative Scan | Stub | For parallel sLSTM |
| Triton Kernels | Stub | For efficient mLSTM |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | 125M to 1.3B configs |

## Cell Type Comparison

| Feature | sLSTM | mLSTM |
|---------|-------|-------|
| Memory Type | Scalar cell state | Matrix cell state (d x d) |
| Gating | Exponential input/forget | Exponential + key-value |
| Parallelization | Via associative scan | Fully parallel (matmul) |
| Use Case | Memory mixing layers | Main storage backbone |
| Capacity | O(d) | O(d^2) |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- Triton >= 2.0 (for efficient kernels)
- einops
- flash-attn (optional, for hybrid models)

### Hardware
- GPU recommended (highly parallelizable)
- mLSTM: O(d^2) memory per layer (matrix cell state)
- Benefits from tensor cores (matmul-heavy in mLSTM)

### External Resources
- [ ] Official implementation: [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [ ] Training data: The Pile for language modeling benchmarks
- [ ] Pretrained weights from official release

## Quick Start

```python
from consciousness.ml_research.modern_dev.xlstm import XLSTM_MODEL_CONFIG

# xLSTM Model Configuration
config = {
    "vocab_size": 50257,
    "embedding_dim": 768,
    "num_layers": 12,
    "layer_types": ["m", "m", "m", "s", "m", "m", "m", "s", "m", "m", "m", "s"],
    "up_proj_factor": 2,
    "conv_kernel_size": 4,
    "dropout": 0.0,
    "tie_weights": True,
}

# Layer types: "m" = mLSTM, "s" = sLSTM
# Typical ratio: 7:1 mLSTM to sLSTM

# When implemented:
# from consciousness.ml_research.modern_dev.xlstm.src.model import xLSTMLanguageModel
# model = xLSTMLanguageModel(**config)
```

## File Structure

```
xlstm/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── slstm.py      # sLSTMCell and sLSTM layer
│   ├── mlstm.py      # mLSTMCell and mLSTM layer
│   ├── block.py      # xLSTMBlock with projections
│   └── model.py      # xLSTMLanguageModel
├── configs/
│   ├── xlstm_125m.yaml  # 125M parameter config
│   ├── xlstm_350m.yaml  # 350M parameter config
│   └── xlstm_1_3b.yaml  # 1.3B parameter config
├── cli/
│   ├── train.py      # Training script (xlstm-train)
│   ├── infer.py      # Text generation (xlstm-infer)
│   └── analyze.py    # Memory cell visualization
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**sLSTM (Exponential Gating):**
```
Gates (exponential):
    i_t = exp(W_i @ x_t + R_i @ h_{t-1} + b_i)    # input gate
    f_t = exp(W_f @ x_t + R_f @ h_{t-1} + b_f)    # forget gate
    o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o) # output gate

State updates:
    c_tilde = tanh(W_c @ x_t + R_c @ h_{t-1} + b_c)
    c_t = f_t * c_{t-1} + i_t * c_tilde
    n_t = f_t * n_{t-1} + i_t                      # normalizer
    h_t = o_t * (c_t / max(n_t, 1))
```

**mLSTM (Matrix Memory):**
```
Projections:
    q_t = W_q @ x_t                    # query (d-dim)
    k_t = (1/sqrt(d)) * W_k @ x_t      # key (d-dim)
    v_t = W_v @ x_t                    # value (d-dim)

Gates:
    i_t = exp(w_i^T @ x_t + b_i)       # scalar input gate
    f_t = exp(w_f^T @ x_t + b_f)       # scalar forget gate
    o_t = sigmoid(W_o @ x_t + b_o)     # vector output gate

State updates:
    C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)    # matrix memory (d x d)
    n_t = f_t * n_{t-1} + i_t * k_t              # normalizer vector

Output:
    h_tilde = C_t @ q_t                          # retrieve from memory
    h_t = o_t * h_tilde / max(|n_t^T @ q_t|, 1)  # gated, normalized
```

## Benchmarks

**Language Modeling Perplexity (Pile, 15B training tokens):**

| Model | Parameters | Perplexity |
|-------|-----------|------------|
| Transformer | 125M | 37.2 |
| RWKV-4 | 124M | 38.1 |
| Mamba | 130M | 36.8 |
| xLSTM[1:0] (all mLSTM) | 125M | 36.3 |
| xLSTM[7:1] (mixed) | 125M | 35.9 |

**Key Results:**
- xLSTM matches Transformer at all scales tested (up to 1.3B)
- Better than Mamba and RWKV at most scales
- Strong on associative recall (competitive with Transformer attention)

## Comparison with Other Architectures

|                    | xLSTM | Transformer | Mamba | RWKV | TTT |
|--------------------|-------|-------------|-------|------|-----|
| Context Complexity | O(n) | O(n^2) | O(n) | O(n) | O(n) |
| Memory per Layer | O(d^2) | O(n*d) | O(d) | O(d) | O(d^2) |
| Parallelizable | Yes | Yes | Yes | Partial | Yes |
| Recurrent State | Yes | No* | Yes | Yes | Yes |
| Associative Recall | Strong | Strong | Weak | Weak | Strong |
| Training Stability | High | High | Medium | Medium | High |

## References

- Beck, M., et al. "xLSTM: Extended Long Short-Term Memory" (2024). arXiv:2405.04517
- [Official xLSTM Implementation](https://github.com/NX-AI/xlstm)
- Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory" (1997)
- Related: LSTM, Linear Attention, Fast Weight Programmers

## Connections to Consciousness Research

- **Content-Addressable Memory**: mLSTM matrix memory implements key-value associations, analogous to episodic memory storage
- **Memory Gating**: Exponential gates model attention allocation and capacity management
- **Global Workspace**: sLSTM mixing broadcasts information across the workspace, while mLSTM stores peripheral memories

## Contributing

To complete this implementation:

1. **Phase 1: Core Cells**
   - Implement `sLSTMCell` with exponential gating and normalizer
   - Implement `mLSTMCell` with matrix memory and key-value storage
   - Add unit tests for numerical stability

2. **Phase 2: Parallel Scan**
   - Implement associative scan for sLSTM training
   - Create Triton kernel for mLSTM parallel computation
   - Benchmark against sequential implementation

3. **Phase 3: Full Model**
   - Build xLSTMBlock with up/down projections
   - Create xLSTMLanguageModel with configurable mLSTM:sLSTM ratio
   - Enable loading of official pretrained checkpoints
