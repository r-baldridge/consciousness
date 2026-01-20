# Titans - Learning to Memorize at Test Time

> A neural architecture that introduces learnable memory modules capable of memorizing at test time through gradient descent, enabling dynamic and adaptive long-term memory without static KV caches.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
**Year:** 2025
**Organization:** Google Research

## Overview

Titans introduces a fundamentally new approach to memory in neural networks: instead of storing context in fixed key-value caches (like Transformers) or static state vectors (like RNNs), Titans uses a small neural network as its memory module that is trained via gradient descent during inference. The model learns not just what to store, but how to memorize, with the memory module's parameters adapting to the specific context at hand.

The key innovation is surprise-based memory gating: the model computes a "surprise" signal (prediction error on the current input), and high surprise triggers memory writes while low surprise allows fast-path processing with cached representations. This makes Titans both adaptive and efficient-it focuses computational resources on novel, important information while handling predictable sequences quickly.

Titans offers three architectural variants for integrating the memory module: MAC (Memory as Context) concatenates memory reads with attention context, MAG (Memory as Gate) uses memory to modulate attention output, and MAL (Memory as Layer) treats memory as a separate processing step. All variants maintain linear complexity while achieving near-perfect retrieval at contexts exceeding 2M tokens.

## Key Innovations

- **Test-Time Learning for Memory**: Memory module parameters update via gradient descent during inference, learning what to store based on incoming data dynamically.

- **Surprise-Based Gating**: Prediction error triggers memory writes, focusing resources on novel information while fast-tracking predictable sequences.

- **Neural Long-Term Memory Module**: Small MLP that stores compressed representations via weight updates, providing unlimited capacity that grows with context.

## Architecture Diagram

```
Traditional Transformer:
    Input -> [Attention (KV cache)] -> Output
             |
             KV cache grows linearly with context
             Fixed storage mechanism

Titans Architecture:
    Input -> [Core Module + Memory Module] -> Output
             |
             Core: Short-term via attention
             Memory: Long-term via learnable module

    Memory Module:
        +-------------------------------------+
        |  Memory Network M (small MLP)       |
        |                                     |
        |  Read:  m_t = M(query)              |
        |  Write: M = M - lr * grad(loss)     |
        |                                     |
        |  Loss = surprise(x_t, prediction)   |
        +-------------------------------------+

    Three Variants:
    1. MAC (Memory as Context): Memory read added to attention
    2. MAG (Memory as Gate): Memory gates attention output
    3. MAL (Memory as Layer): Memory as separate processing layer

Full Titans Layer:
    +--------------------------------------------------+
    |                  Titans Layer                     |
    |                                                   |
    |  h = LayerNorm(x)                                |
    |          |                                        |
    |     +----+----+                                   |
    |     v         v                                   |
    | [Attention] [Memory]                              |
    |     |  (short-term)  |  (long-term + TTL update) |
    |     +----+----+                                   |
    |          |                                        |
    |          v                                        |
    |       [Gate]  (combine based on variant)          |
    |          |                                        |
    |          v                                        |
    |     x + FFN(output)                              |
    +--------------------------------------------------+
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| MemoryModule | Stub | Learnable MLP |
| SurpriseComputer | Stub | Prediction error |
| MemoryGate | Stub | Write decision |
| TestTimeLearner | Stub | Gradient updates |
| TitansLayer | Stub | Integrated layer |
| MAC/MAG/MAL | Stub | Integration variants |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | Default and variant configs |

## Architectural Variants

| Variant | Integration Method | Best For |
|---------|-------------------|----------|
| MAC (Memory as Context) | Memory read concatenated with attention KV | Explicit memory retrieval tasks |
| MAG (Memory as Gate) | Memory modulates attention output | Adaptive short/long-term blending |
| MAL (Memory as Layer) | Memory as separate processing step | Deep memory integration |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- einops
- Flash Attention (optional, for core attention)

### Hardware
- Single GPU sufficient (no distributed requirement)
- Memory: Standard transformer requirements
- Compute: Slight overhead for memory gradients (~5%)

### External Resources
- [ ] Paper implementation details (no official repo yet)
- [ ] Training data: Standard language modeling datasets
- [ ] Evaluation: Needle-in-haystack benchmarks

## Quick Start

```python
from consciousness.ml_research.modern_dev.titans import DEFAULT_CONFIG, VARIANT_CONFIGS

# Default Configuration
config = {
    "hidden_dim": 2048,
    "num_layers": 24,
    "num_heads": 16,
    "head_dim": 128,

    # Memory module config
    "memory_dim": 256,
    "memory_layers": 2,
    "memory_lr": 0.01,  # Learning rate for test-time updates
    "surprise_threshold": 0.1,

    # Integration variant
    "variant": "MAG",  # MAC, MAG, or MAL
}

# Variant-specific configs
# VARIANT_CONFIGS["MAC"] -> Memory as Context settings
# VARIANT_CONFIGS["MAG"] -> Memory as Gate settings
# VARIANT_CONFIGS["MAL"] -> Memory as Layer settings

# When implemented:
# from consciousness.ml_research.modern_dev.titans.src.model import TitansModel
# model = TitansModel(**config)
```

## File Structure

```
titans/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── memory_module.py  # MemoryModule MLP
│   ├── surprise.py       # SurpriseComputer
│   ├── titans_layer.py   # TitansLayer, MAC, MAG, MAL
│   └── model.py          # TitansModel
├── configs/
│   ├── titans.yaml       # Default configuration
│   ├── mac.yaml          # MAC variant config
│   ├── mag.yaml          # MAG variant config
│   └── mal.yaml          # MAL variant config
├── cli/
│   ├── train.py      # Training script (titans-train)
│   ├── infer.py      # Long-context inference (titans-infer)
│   └── analyze.py    # Memory visualization (titans-analyze)
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**Memory Module M with parameters theta_M:**
```
m_t = M_theta(h_t)  # Memory read (forward pass)
```

**Surprise Signal:**
```
s_t = ||f(h_t) - embed(x_{t+1})||^2

where f is a learned linear predictor
```

**Test-Time Memory Update:**
```
If s_t > tau (surprise threshold):
    L_memory = s_t  # or reconstruction loss
    theta_M = theta_M - alpha * nabla_{theta_M} L_memory

This is ONE gradient step, not full training
```

**Memory Gating (MAG variant):**
```
g_t = sigmoid(W_g * [h_t; m_t] + b_g)
o_t = g_t * attention_out + (1 - g_t) * m_t
```

**Full Titans Layer:**
```
h = LayerNorm(x)
a = Attention(h)  # Short-term
m = Memory(h)     # Long-term (with test-time update)
o = Gate(a, m)    # Combine based on variant
out = x + FFN(o)
```

## Test-Time Learning Process

```
For each token x_t:
    1. Query memory: m_t = Memory(h_t)
    2. Compute surprise: s_t = ||x_t - prediction||^2
    3. If s_t > threshold:
        # Update memory with gradient descent
        grad = nabla_Memory(s_t)
        Memory = Memory - lr * grad
    4. Use m_t in attention/output

Key Properties:
    - Memory learns associations online
    - Recent surprising events stored
    - Gradual forgetting of old info
    - No explicit memory capacity limit
```

## Benchmarks

**Long Context Performance:**
- Near-perfect retrieval at 2M+ context
- Attention: degrades after 128K
- Mamba: degrades after 64K

**Efficiency:**
- 1.5-2x faster than full attention at long contexts
- Memory module: <5% compute overhead
- Constant memory usage regardless of history length

## Comparison with Related Approaches

|                | Titans | TTT-Linear | Mamba | Attention |
|----------------|--------|------------|-------|-----------|
| Memory Type | Learned | Learned | Fixed SSM | KV Cache |
| Test-Time Learning | Gradient descent | Gradient descent | None | None |
| Context | Unlimited | Unlimited | Unlimited | Limited |
| Complexity | O(n) | O(n) | O(n) | O(n^2) |
| Interpretable | Yes | Partial | No | Yes |

**Key Differences from TTT:**
- TTT: Hidden state IS the model being trained
- Titans: Separate memory module trained alongside main model
- Titans: Surprise-based gating for efficiency
- Titans: Multiple integration variants (MAC/MAG/MAL)

## Training Considerations

**Two-Stage Training:**
```
Stage 1: Train core model (attention + FFN)
Stage 2: Train memory module integration
```

**Meta-Learning Aspect:**
- Model learns HOW to learn at test time
- Training teaches memory module what to store
- Test-time learning applies learned strategy

**Loss Function:**
```
L = L_main + lambda * L_memory

where L_memory encourages:
    - Good memory retrieval
    - Appropriate surprise detection
    - Stable test-time updates
```

## References

- Behrouz, A., Zhong, P., Mirrokni, V. "Titans: Learning to Memorize at Test Time" (2025). arXiv:2501.00663
- Related: TTT, Meta-Learning, Memory-Augmented Neural Networks

## Contributing

To complete this implementation:

1. **Phase 1: Core Memory Module**
   - Implement MemoryModule as small MLP
   - Create surprise computation mechanism
   - Add test-time gradient update infrastructure
   - Build basic forward/backward pass

2. **Phase 2: Integration Variants**
   - Implement MAC (Memory as Context)
   - Implement MAG (Memory as Gate)
   - Implement MAL (Memory as Layer)
   - Create variant comparison benchmarks

3. **Phase 3: Training Infrastructure**
   - Build two-stage training pipeline
   - Add memory initialization strategies
   - Implement hyperparameter tuning
   - Create long-context data loading utilities
