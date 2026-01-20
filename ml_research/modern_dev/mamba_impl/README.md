# Mamba - Selective State Space Models

> A family of efficient sequence models based on structured state space models (SSMs) with selective/input-dependent mechanisms, achieving linear-time complexity while matching or exceeding Transformer performance.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
**Year:** 2023
**Organization:** Carnegie Mellon University / Princeton

## Overview

Mamba represents a breakthrough in efficient sequence modeling by introducing a selection mechanism into structured state space models (SSMs). Traditional SSMs achieve linear complexity through fixed state transitions, but this limits their ability to perform content-based reasoning. Mamba solves this by making the state transition parameters (delta, B, C) input-dependent, allowing the model to selectively focus on or filter out information based on the current input.

The architecture combines this selection mechanism with a hardware-aware algorithm that fuses operations and minimizes memory transfers, achieving 3-5x speedup over naive implementations. By avoiding materialization of the full state tensor, Mamba maintains linear memory complexity while enabling efficient parallel training through associative scan operations.

Mamba has spawned a family of related architectures: Mamba-2 (2024) introduces the State Space Duality (SSD) framework showing theoretical connections to attention, while Jamba (AI21 Labs, 2024) demonstrates the power of hybrid Mamba-Attention-MoE architectures for 256K context production models.

## Key Innovations

- **Selective State Spaces**: Input-dependent parameters (delta, B, C) enable content-based reasoning while maintaining linear complexity, allowing the model to adaptively focus on relevant inputs.

- **Hardware-Aware Algorithm**: Fused CUDA kernels minimize HBM transfers by computing discretization, scan, and output in single kernels, achieving 3-5x speedup.

- **Parallel Scan Training**: Uses associative scan operations for O(log L) parallel computation, enabling efficient GPU utilization during training.

## Architecture Diagram

```
                     +-------------------------------------+
    Input x          |                                     |
        |            |    Linear Projection (expand)       |
        v            |              |                      |
    +-------+        |         +----+----+                 |
    | Norm  |        |         v         v                 |
    +---+---+        |    +---------+ +---------+          |
        |            |    |  Conv1D | | Linear  |          |
        v            |    +----+----+ | (gate)  |          |
    +-----------+    |         |      +----+----+          |
    |   Mamba   | <--|         v           |               |
    |   Block   |    |    +---------+      |               |
    +-----+-----+    |    |   SiLU  |      |               |
          |          |    +----+----+      |               |
          v          |         |           |               |
    +-----------+    |    +----+-----------+               |
    |  Residual |    |    v                                |
    |     +     |    | +-------------------+               |
    +-----------+    | |   Selective SSM   |               |
                     | |  (Delta,B,C from x)|              |
                     | +--------+----------+               |
                     |          |                          |
                     |          v                          |
                     |    +---------+                      |
                     |    |  SiLU   | x gate               |
                     |    +----+----+                      |
                     |         |                           |
                     |         v                           |
                     |   Linear (contract)                 |
                     |           |                         |
                     +-----------+--------------------------+
                                 v
                             Output
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| S4D Layer | Stub | Diagonal SSM base |
| SelectiveSSM | Stub | Core selection mechanism |
| MambaBlock | Stub | Full block with gating |
| Parallel Scan | Stub | Associative scan ops |
| CUDA Kernels | Stub | Hardware-aware algorithm |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | 130M to 2.8B configs |

## Mamba Family Timeline

| Model | Year | Key Innovation | Paper |
|-------|------|----------------|-------|
| S4 | 2021 | HiPPO initialization | [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) |
| S4D | 2022 | Diagonal parameterization | [arXiv:2206.11893](https://arxiv.org/abs/2206.11893) |
| Mamba | 2023 | Selective state spaces | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Mamba-2 | 2024 | SSD algorithm, 2-8x speedup | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| Jamba | 2024 | Hybrid Mamba-Attention-MoE | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- Triton >= 2.0 (for custom kernels)
- CUDA >= 11.6
- einops
- causal-conv1d (optional, for faster conv)

### Hardware
- GPU required for efficient training
- Single A100 sufficient for inference
- Memory scales linearly with sequence length

### External Resources
- [ ] Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- [ ] Pretrained weights from official release
- [ ] Training data: The Pile for language modeling

## Quick Start

```python
from consciousness.ml_research.modern_dev.mamba_impl import MAMBA_1_4B_CONFIG

# Mamba-1.4B Configuration
config = {
    "d_model": 2048,
    "n_layer": 48,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "vocab_size": 50280,
}

# When implemented:
# from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_block import MambaBlock
# block = MambaBlock(**config)
```

## File Structure

```
mamba_impl/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── ssm.py        # S4D and SelectiveSSM layers
│   ├── mamba_block.py # MambaBlock implementation
│   └── parallel_scan.py # Associative scan operations
├── configs/
│   ├── mamba_130m.yaml  # 130M parameter config
│   ├── mamba_1_4b.yaml  # 1.4B parameter config
│   └── mamba_2.yaml     # Mamba-2 config
├── cli/
│   ├── train.py      # Training script (mamba-train)
│   ├── generate.py   # Text generation (mamba-generate)
│   └── benchmark.py  # Speed/memory benchmarks
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**State Space Model (Continuous):**
```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

**Discrete (Zero-Order Hold):**
```
A_bar = exp(Delta * A)
B_bar = (Delta * A)^{-1} (exp(Delta * A) - I) * Delta * B  approx Delta * B

h_k = A_bar * h_{k-1} + B_bar * x_k
y_k = C * h_k
```

**Mamba Selection Mechanism:**
```
Delta = softplus(W_Delta * x + b_Delta)   # Discretization step
B = W_B * x                                # Input projection
C = W_C * x                                # Output projection

Note: A is learned but input-independent (diagonal)
```

**Parallel Scan (Associative):**
```
Given pairs (a_i, b_i) representing h_i = a_i * h_{i-1} + b_i

(a_1, b_1) + (a_2, b_2) = (a_1 * a_2, a_2 * b_1 + b_2)

Enables O(log L) parallel computation
```

## Model Sizes

| Model | d_model | n_layer | Parameters |
|-------|---------|---------|------------|
| Mamba-130M | 768 | 24 | 130M |
| Mamba-370M | 1024 | 48 | 370M |
| Mamba-790M | 1536 | 48 | 790M |
| Mamba-1.4B | 2048 | 48 | 1.4B |
| Mamba-2.8B | 2560 | 64 | 2.8B |

## Benchmarks

**Language Modeling (Perplexity):**

| Model | Params | Context | Pile PPL |
|-------|--------|---------|----------|
| GPT-3 | 175B | 2K | - |
| LLaMA | 7B | 2K | 6.97 |
| Mamba | 2.8B | Unlimited | 6.22 |
| Mamba-2 | 2.7B | Unlimited | ~6.0 |

**Inference Speed (tokens/sec on A100):**

| Sequence Length | Transformer | Mamba |
|-----------------|-------------|-------|
| 2K | ~5000 | ~8000 |
| 8K | ~2000 | ~8000 |
| 32K | ~500 | ~8000 |
| 128K | OOM | ~7500 |

## References

- Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023). arXiv:2312.00752
- Dao, T. & Gu, A. "Transformers are SSMs" (2024). arXiv:2405.21060
- [Official Mamba Implementation](https://github.com/state-spaces/mamba)
- Related: S4, HiPPO, Linear Attention

## Contributing

To complete this implementation:

1. **Phase 1: Core SSM Implementation**
   - Implement S4D layer with diagonal parameterization
   - Add HiPPO initialization for the A matrix
   - Create discretization methods (zero-order hold)
   - Implement parallel scan (associative scan)

2. **Phase 2: Selective Mamba**
   - Implement selection mechanism (Delta, B, C projections)
   - Create fused selective scan kernel (or Triton equivalent)
   - Add memory-efficient backward pass with recomputation
   - Integrate causal Conv1D

3. **Phase 3: Full Architecture**
   - Build complete MambaBlock with gating
   - Create Mamba language model wrapper
   - Add Mamba-2 SSD algorithm
   - Enable multi-head selective SSM
