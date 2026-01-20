# RWKV (Receptance Weighted Key Value)

> A novel architecture combining the efficient parallelizable training of Transformers with the efficient inference of RNNs, achieving O(n) time and memory complexity while maintaining competitive performance.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
**Year:** 2023
**Organization:** BlinkDL / RWKV Foundation
**Author:** Bo Peng (BlinkDL)

## Overview

RWKV (Receptance Weighted Key Value) represents a unique approach to efficient sequence modeling that achieves the best of both worlds: the parallelizable training of Transformers and the efficient O(1) inference of RNNs. Unlike attention mechanisms that require O(n^2) computation, RWKV uses a linear attention variant with exponential decay and token-shifted mixing to capture temporal dependencies.

The architecture introduces two key components: Time Mixing (an attention-like mechanism using the WKV formulation) and Channel Mixing (a gated FFN replacement). Both use a distinctive token-shifting mechanism where the current representation is mixed with the previous token's representation, enabling temporal flow without explicit recurrence during training.

RWKV has evolved through multiple versions: RWKV-4 (2023) proved the viability of linear complexity language models, RWKV-5 "Eagle" (2024) added multi-headed matrix-valued states, RWKV-6 "Finch" (2024) introduced data-dependent decay, and RWKV-7 "Goose" (2025) represents the latest iteration with enhanced expressiveness competitive with similar-sized Transformers.

## Key Innovations

- **Linear Complexity**: O(n) time and memory for both training and inference, with O(1) memory during streaming inference in recurrent mode.

- **Dual Mode Operation**: Parallel mode for efficient training (like Transformers) and recurrent mode for efficient streaming inference (like RNNs).

- **WKV Attention Mechanism**: Linear attention without softmax using exponential decay and receptance gating, enabling efficient computation while maintaining expressiveness.

## Architecture Diagram

```
Standard Transformer Block:
    Input --> LayerNorm --> Attention --> Add --> LayerNorm --> FFN --> Add --> Output

RWKV Block:
    Input --> LayerNorm --> Time Mixing --> Add --> LayerNorm --> Channel Mixing --> Add --> Output

                    +-------------------------------------+
    Input x_t -->   |           RWKV Block                |
                    |                                     |
                    |  +-----------+                      |
                    |  | LayerNorm |                      |
                    |  +-----+-----+                      |
                    |        v                            |
                    |  +-----------+                      |
                    |  |Time Mixing| <-- state_{t-1}      |
                    |  | (WKV attn)| --> state_t          |
                    |  +-----+-----+                      |
                    |        v                            |
                    |     Add (residual)                  |
                    |        v                            |
                    |  +-----------+                      |
                    |  | LayerNorm |                      |
                    |  +-----+-----+                      |
                    |        v                            |
                    |  +---------------+                  |
                    |  |Channel Mixing |                  |
                    |  | (gated FFN)   |                  |
                    |  +-------+-------+                  |
                    |          v                          |
                    |       Add (residual)                |
                    |                                     |
                    +-------------------------------------+
                              v
                          Output x_t
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| WKVKernel | Stub | Efficient attention computation |
| TimeMixing | Stub | With token shift |
| ChannelMixing | Stub | Gated FFN |
| RWKVBlock | Stub | Complete block |
| RWKVModel | Stub | Full model |
| CUDA Kernel | Stub | For WKV computation |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | 169M to 14B configs |

## RWKV Version History

| Version | Year | Codename | Key Feature |
|---------|------|----------|-------------|
| RWKV-4 | 2023 | - | Original architecture |
| RWKV-5 | 2024 | Eagle | Multi-headed matrix-valued states |
| RWKV-6 | 2024 | Finch | Data-dependent linear recurrence |
| RWKV-7 | 2025 | Goose | Enhanced expressiveness |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- CUDA (for efficient training)
- einops
- numpy

### Hardware
- GPU recommended for training
- Can run inference on CPU (slower but functional)
- Memory scales linearly with sequence length during training

### External Resources
- [ ] Official implementation: [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [ ] Pretrained weights from official releases
- [ ] Training data: The Pile and various multilingual corpora

## Quick Start

```python
from consciousness.ml_research.modern_dev.rwkv import DEFAULT_CONFIG, MODEL_SIZES

# Default Configuration
config = {
    "vocab_size": 50277,
    "hidden_dim": 768,
    "num_layers": 12,
    "context_length": 4096,
    "head_size": 64,
    "decay_lora_dim": 64,
    "use_data_dependent_decay": False,  # RWKV-6 feature
}

# Model size presets
# MODEL_SIZES["small"]  -> 169M params
# MODEL_SIZES["large"]  -> 1.5B params
# MODEL_SIZES["world"]  -> 14B params

# When implemented:
# from consciousness.ml_research.modern_dev.rwkv.src.model import RWKV, RWKVConfig
# model = RWKV(RWKVConfig(**config))
```

## File Structure

```
rwkv/
├── __init__.py       # Module documentation and version metadata
├── README.md         # This file
├── src/
│   ├── model.py      # RWKV and RWKVConfig
│   ├── wkv.py        # WKVKernel and wkv_attention
│   ├── time_mixing.py    # TimeMixing layer
│   └── channel_mixing.py # ChannelMixing layer
├── configs/
│   ├── rwkv4.yaml    # RWKV-4 configuration
│   ├── rwkv6.yaml    # RWKV-6 with data-dependent decay
│   └── sizes/        # Size-specific configs
├── cli/
│   ├── train.py      # Training script (rwkv-train)
│   ├── infer.py      # Streaming inference (rwkv-infer)
│   └── convert.py    # Model format conversion
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**WKV Attention (Core Mechanism):**
```
Recurrence form:
    a_t = e^{-w} * a_{t-1} + e^{k_t} * v_t
    b_t = e^{-w} * b_{t-1} + e^{k_t}
    wkv_t = a_t / b_t

where:
    w = learnable decay (per channel)
    k_t = key at time t
    v_t = value at time t
    (a, b) = running state
```

**Time Mixing:**
```
x'_t = mu * x_t + (1-mu) * x_{t-1}     # token shift
r_t = sigmoid(W_r @ x'_t)               # receptance (gate)
k_t = W_k @ x'_t                        # key
v_t = W_v @ x'_t                        # value
output = W_o @ (r_t * wkv_t)
```

**Channel Mixing:**
```
x'_t = mu * x_t + (1-mu) * x_{t-1}     # token shift
r_t = sigmoid(W_r @ x'_t)               # receptance (gate)
k_t = W_k @ x'_t                        # key (FFN input)
output = r_t * (W_v @ squared_relu(k_t))
```

**RWKV-6 Data-Dependent Decay:**
```
w_t = w_base + tanh(W_decay @ x_t) * scale
```

## Model Sizes

| Size | hidden_dim | num_layers | Parameters |
|------|------------|------------|------------|
| small | 768 | 12 | 169M |
| medium | 1024 | 24 | 430M |
| large | 2048 | 24 | 1.5B |
| xl | 2560 | 32 | 3B |
| xxl | 4096 | 32 | 7B |
| world | 5120 | 40 | 14B |

## Benchmarks

**Complexity Comparison:**

|                    | RWKV | Transformer | RNN/LSTM |
|--------------------|------|-------------|----------|
| Time (train) | O(n) | O(n^2) | O(n) |
| Time (inference) | O(1)* | O(n) | O(1) |
| Memory (train) | O(n) | O(n^2) | O(n) |
| Memory (inference) | O(1) | O(n) | O(1) |
| Parallelizable | Yes | Yes | No |
| Long-range deps | Good | Excellent | Limited |

*O(1) per token in recurrent mode

## Inference Modes

**Parallel Mode (Training/Batch):**
- Process entire sequence at once
- Uses matrix operations
- Efficient for training and batch inference

**Recurrent Mode (Streaming):**
- Process one token at a time
- Maintains hidden state
- Constant memory usage
- Ideal for real-time generation

## References

- Peng, B. et al. "RWKV: Reinventing RNNs for the Transformer Era" (2023). arXiv:2305.13048
- [Official RWKV Implementation](https://github.com/BlinkDL/RWKV-LM)
- [RWKV Foundation](https://www.rwkv.com/)
- Related: Linear Attention, AFT (Attention Free Transformer)

## Contributing

To complete this implementation:

1. **Phase 1: Core Architecture (RWKV-4)**
   - Implement WKV attention (pure PyTorch version)
   - Create Time Mixing layer with token shift
   - Build Channel Mixing layer with squared ReLU
   - Assemble basic RWKV block

2. **Phase 2: Optimizations**
   - Create CUDA kernel for WKV computation
   - Implement state caching utilities
   - Add chunked processing for long sequences

3. **Phase 3: Advanced Versions**
   - Add RWKV-5 multi-head matrix states
   - Implement RWKV-6 data-dependent decay
   - Integrate RWKV-7 enhancements
