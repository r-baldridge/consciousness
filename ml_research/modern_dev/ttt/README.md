# Test-Time Training (TTT) Layers

> A revolutionary sequence modeling architecture that replaces attention with learnable hidden states that update via gradient descent during inference, achieving linear complexity with true long-context capability.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
**Year:** 2024
**Organization:** Stanford University / NVIDIA

## Overview

Test-Time Training (TTT) represents a paradigm shift in sequence modeling by reconceptualizing what a hidden state can be. While traditional RNNs use fixed-size vectors as hidden states (creating an information bottleneck), TTT layers use entire neural networks with trainable weights as their hidden state. During inference, these weights are updated via gradient descent on a self-supervised objective, effectively performing "learning" at every timestep.

This approach achieves linear complexity O(n) while maintaining expressiveness that approaches attention mechanisms. The key insight is that gradient descent induces an implicit kernel between inputs, theoretically connecting TTT to attention while avoiding the quadratic cost. The architecture can process arbitrarily long sequences without the context window limitations of transformers.

TTT comes in two main variants: TTT-Linear uses a single linear layer as the hidden state for maximum speed, while TTT-MLP uses a two-layer MLP for greater expressiveness. Both variants support mini-batch processing for GPU parallelism, enabling efficient training while maintaining the benefits of online learning at inference time.

## Key Innovations

- **Neural Network as Hidden State**: The hidden state is a model's weights that are trained on each input token via gradient descent during the forward pass.

- **Self-Supervised Learning at Test Time**: Each token's representation is improved through real-time self-supervised learning, with the hidden state specializing to the current context.

- **Linear Complexity with Unbounded Context**: Achieves O(n) complexity while supporting unlimited context length, unlike attention's O(n^2) with fixed windows.

## Architecture Diagram

```
Standard Transformer:
    Token_1 --+-- Attention --+-- Output_1
    Token_2 --|    (O(n^2))   |-- Output_2
    Token_n --+---------------+-- Output_n

Traditional RNN:
    Token_1 --> [h_1] --> Token_2 --> [h_2] --> ... --> Token_n --> [h_n]
                  |                     |                            |
              Output_1              Output_2                    Output_n
    Hidden state: fixed-size vector (compression bottleneck)

TTT Layer:
    Token_1 --> [W_1] --> Token_2 --> [W_2] --> ... --> Token_n --> [W_n]
                  |                     |                            |
              Output_1              Output_2                    Output_n

    Hidden state W_t = neural network weights
    W_t updated by gradient descent: W_{t+1} = W_t - eta * grad_L(x_t; W_t)
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| TTTLinear Layer | Stub | API defined |
| TTT_MLP Layer | Stub | API defined |
| Mini-batch Gradient | Stub | Parallel computation ready |
| Associative Scan | Stub | For cumulative updates |
| Triton Kernels | Stub | Custom ops planned |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | TTT-Linear and TTT-MLP configs |

## TTT Variants

| Variant | Hidden State | Complexity | Speed | Expressivity |
|---------|-------------|------------|-------|--------------|
| TTT-Linear | Linear layer (W in R^{d x d}) | O(d^2) per token | Fast | Good baseline |
| TTT-MLP | Two-layer MLP | O(d * d_hidden) per token | Moderate | Higher |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- Triton >= 2.0 (for custom kernels)
- einops
- flash-attn (optional, for hybrid models)

### Hardware
- GPU strongly recommended (parallel gradient computation)
- VRAM scales with d^2 (hidden state size)
- Benefits from tensor cores (matmul-heavy operations)

### External Resources
- [ ] Official implementation: [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)
- [ ] Training data: The Pile for language modeling benchmarks
- [ ] Pretrained checkpoints from official release

## Quick Start

```python
from consciousness.ml_research.modern_dev.ttt import TTT_LINEAR_CONFIG, TTT_MLP_CONFIG

# TTT-Linear Configuration
linear_config = {
    "hidden_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "learning_rate": 1.0,  # TTT learning rate (often 1.0 works well)
    "mini_batch_size": 16,
    "rope_base": 10000,
    "layer_norm_eps": 1e-5,
}

# TTT-MLP Configuration
mlp_config = {
    "hidden_dim": 768,
    "mlp_hidden_dim": 2048,
    "num_layers": 12,
    "num_heads": 12,
    "learning_rate": 1.0,
    "mini_batch_size": 16,
    "activation": "gelu",
}

# When implemented:
# from consciousness.ml_research.modern_dev.ttt.src.ttt_linear import TTTLinear
# layer = TTTLinear(linear_config)
```

## File Structure

```
ttt/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── ttt_linear.py # TTT-Linear layer implementation
│   ├── ttt_mlp.py    # TTT-MLP layer implementation
│   └── model.py      # TTTLanguageModel wrapper
├── configs/
│   ├── ttt_linear.yaml  # TTT-Linear configuration
│   └── ttt_mlp.yaml     # TTT-MLP configuration
├── cli/
│   ├── train.py      # Training script (ttt-train)
│   ├── infer.py      # Inference with streaming (ttt-infer)
│   └── benchmark.py  # Speed/quality benchmarks
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**Self-Supervised Learning Objective:**
```
L(x; W) = ||f(x * W_K; W) - x * W_V||^2

where:
    x = input token embedding (1 x d)
    W_K = key projection (d x d_k), fixed after training
    W_V = value projection (d x d_v), fixed after training
    W = hidden state weights, updated at test time
    f(.; W) = TTT model (linear: f(z; W) = z*W, or MLP)
```

**Hidden State Update (Online Learning):**
```
W_{t+1} = W_t - eta * nabla_W L(x_t; W_t)

Gradient for TTT-Linear:
    nabla_W L = (x_t W_K W_t - x_t W_V)^T * (x_t W_K)
```

**Output Computation:**
```
z_t = f(x_t * W_K; W_t) + x_t     # with residual
output_t = z_t * W_O              # output projection
```

**Mini-Batch TTT (for parallelism):**
```
For batch [x_1, ..., x_B]:
    Cumulative W_t = W_0 - eta * sum_{i<t} nabla_W L(x_i; W_{i-1})

Computed efficiently via associative scan
```

## Benchmarks

**Language Modeling (Perplexity, lower is better):**

| Model | Pile (val) | Books | ArXiv | Code |
|-------|-----------|-------|-------|------|
| Mamba 1.4B | 8.85 | 12.20 | 8.16 | 9.48 |
| TTT-Linear 1.3B | 8.71 | 11.23 | 7.91 | 9.33 |
| TTT-MLP 1.3B | 8.52 | 10.87 | 7.72 | 9.08 |
| Transformer 1.3B | 8.44 | 10.71 | 7.58 | 8.95 |

**Long Context:** TTT maintains quality at 128K+ tokens with linear scaling.

**Throughput:** TTT-Linear achieves ~90% of Mamba throughput, faster than Transformer at long sequences.

## Comparison with Other Architectures

|                    | TTT | Transformer | Mamba | RWKV |
|--------------------|-----|-------------|-------|------|
| Context Length | Unlimited | Limited* | Unlimited | Unlimited |
| Complexity | O(n) | O(n^2) | O(n) | O(n) |
| Parallel Training | Yes | Yes | Yes | Yes |
| Parallel Inference | Mini-batch | Chunked | Sequential | Sequential |
| Expressivity | High | Highest | Medium | Medium |
| Memory (inference) | O(d^2) | O(n*d) | O(d) | O(d) |

## References

- Sun, Y., et al. "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (2024). arXiv:2407.04620
- [Official PyTorch Implementation](https://github.com/test-time-training/ttt-lm-pytorch)
- Related: Meta-learning, Online Learning, Linear Attention

## Connections to Consciousness Research

- **Working Memory**: TTT hidden states can model adaptive working memory that updates in real-time
- **Metacognition**: Hidden state can be inspected mid-inference, revealing the "thinking process"
- **Integration**: TTT as memory substrate for cognitive architectures, combinable with CTM for adaptive temporal reasoning

## Contributing

To complete this implementation:

1. **Phase 1: Core Implementation**
   - Implement `TTTLinear` layer with online gradient computation
   - Implement `TTT_MLP` layer with two-layer hidden state
   - Create basic forward pass with weight accumulation

2. **Phase 2: Optimization**
   - Implement mini-batch TTT with associative scan
   - Create Triton kernels for efficient gradient computation
   - Add memory-efficient weight accumulation

3. **Phase 3: Model Integration**
   - Build TTT language model (GPT-style architecture)
   - Add hybrid TTT-Attention model support
   - Enable loading of official pretrained checkpoints
