# Griffin - Hybrid Gated Linear Recurrences with Local Attention

> A hybrid architecture combining gated linear recurrences with local sliding window attention, designed for efficient long-context modeling and serving as the foundation for RecurrentGemma.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)
**Year:** 2024
**Organization:** Google DeepMind

## Overview

Griffin represents Google DeepMind's approach to efficient long-context language modeling through a hybrid architecture that combines the strengths of recurrent and attention-based approaches. Rather than choosing between pure attention (excellent quality, poor efficiency) or pure recurrence (excellent efficiency, sometimes lower quality), Griffin uses both: gated linear recurrences provide efficient long-range context modeling with O(1) memory during inference, while local sliding window attention handles fine-grained local pattern matching.

The core innovation is the Real-Gated Linear Recurrence Unit (RG-LRU), a simplified recurrent cell that uses real-valued (not complex) gated recurrence for stability and efficiency. The input-dependent gating allows adaptive decay based on content, and the sqrt(1-a^2) scaling preserves norm through the recurrence. This design enables efficient parallel training via associative scan while maintaining constant memory during generation.

Griffin's hybrid architecture typically alternates two recurrent blocks with one local attention block, using a fixed sliding window (typically 2048 tokens) for attention. This pattern is used in Google's production RecurrentGemma models (2B and 9B parameters), achieving competitive performance with Gemma models while offering significantly better inference efficiency, especially at long contexts.

## Key Innovations

- **Hybrid Architecture**: Combines recurrent layers (O(1) inference memory) with local attention (strong local patterns), achieving efficiency without sacrificing quality.

- **Real-Gated Linear Recurrence (RG-LRU)**: Real-valued gated recurrence with input-dependent decay and norm-preserving scaling for stable, efficient computation.

- **Local Sliding Window Attention**: Fixed-size attention window (2048 tokens) complements global recurrent context, providing fine-grained local modeling.

## Architecture Diagram

```
Griffin uses a repeating pattern of recurrent and attention blocks:

    Pattern: [Recurrent] --> [Recurrent] --> [Local Attention] --> repeat

                    +-------------------------------------+
                    |         Griffin Model               |
                    |                                     |
    Input -->       |  +---------------------------+      |
                    |  |   Embedding Layer         |      |
                    |  +-------------+-------------+      |
                    |                v                    |
                    |  +---------------------------+      |
                    |  |   Recurrent Block (RG-LRU)|      | x 2
                    |  +-------------+-------------+      |
                    |                v                    |
                    |  +---------------------------+      |
                    |  |  Local Attention Block    |      | x 1
                    |  |  (sliding window)         |      |
                    |  +-------------+-------------+      |
                    |                v                    |
                    |            [repeat pattern]        |
                    |                v                    |
                    |  +---------------------------+      |
                    |  |   Output Head             |      |
                    |  +---------------------------+      |
                    +-------------------------------------+

Recurrent Block Structure:
    +----------------------------------------------+
    |                Recurrent Block               |
    |                                              |
    |  Input x                                     |
    |     |                                        |
    |     +--------+--------+--------+             |
    |     v        v        v        v             |
    |  [Linear] [Linear] [Linear] [Linear]         |
    |     |        |        |        |             |
    |     v        v        v        v             |
    |   gate     input     key    recurrence       |
    |     a        x        k       (h)            |
    |     |        |        |        |             |
    |     +--------+--------+--------+             |
    |                    v                         |
    |              [RG-LRU Cell]                   |
    |                    |                         |
    |                    v                         |
    |               [Gated MLP]                    |
    |                    |                         |
    |                    v                         |
    |                 Output                       |
    +----------------------------------------------+
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| RGLRUCell | Stub | Real-gated linear recurrence |
| RecurrentBlock | Stub | Full recurrent layer with MLP |
| LocalAttention | Stub | Sliding window attention |
| GriffinBlock | Stub | Combines recurrent + attention |
| GriffinModel | Stub | Full model |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | 2B and 9B configs |

## RecurrentGemma Models

| Model | Parameters | Hidden | Layers | Heads | Window | Context |
|-------|-----------|--------|--------|-------|--------|---------|
| RecurrentGemma-2B | 2B | 2048 | 26 | 8 | 2048 | 8192 |
| RecurrentGemma-9B | 9B | 4096 | 42 | 16 | 2048 | 8192 |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0 (for efficient scans)
- JAX/Flax (for official implementation compatibility)
- einops
- numpy

### Hardware
- Single GPU sufficient for inference
- Multi-GPU recommended for training
- O(1) memory during generation (significant savings at long contexts)

### External Resources
- [ ] Official implementation: [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma)
- [ ] Pretrained weights from Google
- [ ] Training data: Similar to Gemma models

## Quick Start

```python
from consciousness.ml_research.modern_dev.griffin import DEFAULT_CONFIG, VARIANTS

# Default Configuration
config = {
    "vocab_size": 256000,
    "hidden_dim": 2048,
    "num_layers": 26,
    "num_heads": 8,
    "head_dim": 256,
    "mlp_expansion": 8,
    "window_size": 2048,
    "context_length": 8192,
    "recurrent_state_dim": 2048,
    "block_pattern": ["recurrent", "recurrent", "attention"],
}

# Model variants
# VARIANTS["recurrent_gemma_2b"] -> 2B parameter config
# VARIANTS["recurrent_gemma_9b"] -> 9B parameter config

# When implemented:
# from consciousness.ml_research.modern_dev.griffin.src.model import Griffin, GriffinConfig
# model = Griffin(GriffinConfig(**config))
```

## File Structure

```
griffin/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── model.py      # Griffin and GriffinConfig
│   ├── rglru.py      # RGLRUCell and RecurrentBlock
│   └── local_attention.py # LocalSlidingWindowAttention
├── configs/
│   ├── griffin_2b.yaml   # RecurrentGemma-2B config
│   └── griffin_9b.yaml   # RecurrentGemma-9B config
├── cli/
│   ├── train.py      # Training script (griffin-train)
│   ├── infer.py      # Efficient generation (griffin-infer)
│   └── convert.py    # Convert from Gemma format
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**RG-LRU (Real-Gated Linear Recurrence):**
```
Gate computation:
    r_t = sigmoid(W_r @ x_t)                    # recurrence gate
    i_t = sigmoid(W_i @ x_t)                    # input gate
    a_t = sigmoid(-8 * softplus(lambda) * r_t)  # decay (data-dependent)

Recurrence:
    h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (B @ x_t)

where:
    h_t = hidden state (dimension d_state)
    a_t = element-wise decay gate
    B = input projection
    lambda = learnable decay parameter

Output:
    y_t = C @ h_t  # output projection
```

**Local Sliding Window Attention:**
```
Q, K, V = W_q @ x, W_k @ x, W_v @ x

Mask[i,j] = 0      if |i-j| <= w/2
            -inf   otherwise

Attention = softmax(Q @ K^T / sqrt(d) + Mask) @ V
```

**Griffin Block Pattern:**
```
[Recurrent] -> [Recurrent] -> [Local Attention] -> repeat
```

## Efficiency Analysis

**Memory During Generation:**

| Context Length | Transformer | Griffin | Savings |
|----------------|-------------|---------|---------|
| 2K | ~200MB | ~50MB | 4x |
| 8K | ~800MB | ~50MB | 16x |
| 32K | ~3.2GB | ~50MB | 64x |
| 128K | ~12.8GB | ~50MB | 256x |

**Inference Speed (tokens/second):**

| Context | Transformer | Griffin | Speedup |
|---------|-------------|---------|---------|
| 2K | 500 | 520 | 1.0x |
| 8K | 400 | 510 | 1.3x |
| 32K | 200 | 500 | 2.5x |
| 128K | 50 | 490 | 9.8x |

## Comparison with Other Architectures

|                    | Griffin | Mamba | Transformer |
|--------------------|---------|-------|-------------|
| Architecture | Hybrid | Pure recurrent | Pure attention |
| Long-range | Good (hybrid) | Good | Excellent |
| Local patterns | Excellent | Good | Excellent |
| Training parallel | Yes | Yes | Yes |
| Inference memory | O(1) | O(1) | O(n) |
| Inference speed | Fast | Fast | Slower |
| Complexity class | O(n * w) | O(n) | O(n^2) |

## Block Pattern Options

| Pattern | Composition | Use Case |
|---------|-------------|----------|
| standard | [rec, rec, attn] | Balanced efficiency/quality |
| attention_heavy | [rec, attn, attn] | Quality-focused |
| recurrent_heavy | [rec, rec, rec, attn] | Efficiency-focused |

## References

- De, S., Smith, S.L., et al. "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models" (2024). arXiv:2402.19427
- [RecurrentGemma Repository](https://github.com/google-deepmind/recurrentgemma)
- Related: Mamba, RWKV, Gemma

## Contributing

To complete this implementation:

1. **Phase 1: Core Components**
   - Implement RG-LRU cell with data-dependent gating
   - Create parallel scan implementation for training
   - Build local sliding window attention
   - Implement gated MLP

2. **Phase 2: Model Assembly**
   - Create RecurrentBlock with RG-LRU and MLP
   - Build attention block with sliding window
   - Assemble full Griffin model with block pattern
   - Enable weight loading from RecurrentGemma checkpoints

3. **Phase 3: Optimization**
   - Create efficient CUDA kernels for RG-LRU
   - Implement state management for generation
   - Add chunked processing for very long contexts
