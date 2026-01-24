# Continuous Thought Machine (CTM) - Architecture Specification

## Overview

The Continuous Thought Machine (CTM) by Sakana AI is a neural architecture that models cognition as a temporal process. Unlike traditional neural networks where computation happens instantaneously, CTM unfolds neural activity over an **internal temporal axis** decoupled from input data.

**Core Philosophy**: "Thought takes time and reasoning is a process."

## Key Innovations

### 1. Internal Temporal Axis
- Computation proceeds through T internal "ticks" independent of input sequence length
- Enables iterative refinement of representations even for static inputs
- Allows the model to "think" for varying amounts of time based on problem complexity

### 2. Neuron-Level Models (NLMs)
- Each neuron has **private MLP parameters** θ_d
- Processes a history of M incoming pre-activations
- Transforms pre-activations → post-activations: `z_d^(t+1) = g_θd(A_d^t)`
- Enables fine-grained temporal dynamics per neuron

### 3. Neural Synchronization
- Direct encoding of information in the **timing** of neural activity
- Computed as dot products between neuron time-series
- Used for both output predictions and attention queries
- Efficient recurrence: `sync = (r × prev_sync + pairwise_product) / √(decay_count)`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Continuous Thought Machine                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ─────                                                                      │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   Image/    │────▶│    Backbone     │────▶│   Positional    │           │
│  │   Text/     │     │  (ResNet/Conv)  │     │   Embedding     │           │
│  │   Code      │     └─────────────────┘     └────────┬────────┘           │
│  └─────────────┘                                      │                     │
│                                                       ▼                     │
│                                              ┌─────────────────┐           │
│                                              │  KV Projection  │           │
│                                              │  for Attention  │           │
│                                              └────────┬────────┘           │
│                                                       │                     │
│  INTERNAL RECURRENCE (T iterations)                   │                     │
│  ══════════════════════════════════                   │                     │
│  ┌────────────────────────────────────────────────────┼────────────────┐   │
│  │                                                    ▼                │   │
│  │  ┌──────────────────┐    ┌─────────────┐    ┌───────────────┐      │   │
│  │  │ Action Synch     │───▶│  Attention  │◀───│  Key/Value    │      │   │
│  │  │ (query source)   │    │  Mechanism  │    │  from Input   │      │   │
│  │  └──────────────────┘    └──────┬──────┘    └───────────────┘      │   │
│  │           ▲                     │                                   │   │
│  │           │                     ▼                                   │   │
│  │  ┌────────┴─────────────────────────────────────────────────┐      │   │
│  │  │                                                          │      │   │
│  │  │   ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │      │   │
│  │  │   │  Concat     │───▶│   Synapse   │───▶│   State    │  │      │   │
│  │  │   │  [attn +    │    │   (U-NET)   │    │   Update   │  │      │   │
│  │  │   │   state]    │    └─────────────┘    └─────┬──────┘  │      │   │
│  │  │   └─────────────┘                             │         │      │   │
│  │  │                                               ▼         │      │   │
│  │  │   ┌──────────────────────────────────────────────────┐  │      │   │
│  │  │   │              State Trace Buffer                  │  │      │   │
│  │  │   │         (M steps of pre-activations)             │  │      │   │
│  │  │   │              [B, D, M] tensor                    │  │      │   │
│  │  │   └────────────────────┬─────────────────────────────┘  │      │   │
│  │  │                        │                                │      │   │
│  │  │                        ▼                                │      │   │
│  │  │   ┌──────────────────────────────────────────────────┐  │      │   │
│  │  │   │           Neuron-Level Models (NLMs)             │  │      │   │
│  │  │   │   Each neuron d has private MLP parameters θ_d   │  │      │   │
│  │  │   │   z_d = g_θd(trace_d)  for d ∈ {1,...,D}        │  │      │   │
│  │  │   └────────────────────┬─────────────────────────────┘  │      │   │
│  │  │                        │                                │      │   │
│  │  │                        ▼                                │      │   │
│  │  │   ┌─────────────────────────────────────┐               │      │   │
│  │  │   │        Activated State              │───────────────┘      │   │
│  │  │   │        [B, D] tensor                │                      │   │
│  │  │   └─────────────────┬───────────────────┘                      │   │
│  │  │                     │                                          │   │
│  │  └─────────────────────┼──────────────────────────────────────────┘   │
│  │                        │                                              │
│  │                        ▼                                              │
│  │  ┌──────────────────────────────────────────────────────────────┐    │
│  │  │                 Output Synchronization                        │    │
│  │  │   S^t = dot_product(z_i^t, z_j^t) for selected (i,j) pairs   │    │
│  │  └────────────────────────┬─────────────────────────────────────┘    │
│  │                           │                                          │
│  └───────────────────────────┼──────────────────────────────────────────┘
│                              │                                           │
│                              ▼                                           │
│  OUTPUT                                                                  │
│  ──────                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │  Linear Projection  │───▶│    Predictions      │                     │
│  │  (synch → output)   │    │  (per iteration)    │                     │
│  └─────────────────────┘    └─────────────────────┘                     │
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │ Certainty Estimate  │───▶│  Select Most        │                     │
│  │ (entropy-based)     │    │  Certain Output     │                     │
│  └─────────────────────┘    └─────────────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Backbone Networks

| Type | Description | Use Case |
|------|-------------|----------|
| `resnet18-2` | Modified ResNet18 (channels ÷ 2) | Image classification |
| `resnet34-2` | Modified ResNet34 (channels ÷ 2) | Mazes, complex images |
| `shallow-wide` | Simple conv net | Lightweight tasks |
| `none` | Identity | Pre-processed inputs |

### Positional Embeddings

| Type | Description |
|------|-------------|
| `learnable-fourier` | Fourier features with learned projections |
| `custom-rotational` | Rotation-based 2D embeddings |
| `none` | No positional encoding |

### Synapse Network

**Single Layer** (depth=1):
```
Dropout → LazyLinear → GLU → LayerNorm
```

**U-NET** (depth>1):
```
Down blocks → Skip connections → Up blocks → LayerNorm
```
- Minimum width: 16
- Processes: [attention_output ⊕ current_state] → pre-activations

### Neuron-Level Models (NLMs)

**Deep Variant** (deep_nlms=True):
```
SuperLinear → GLU → SuperLinear → GLU → Squeeze
```

**Linear Variant** (deep_nlms=False):
```
SuperLinear → GLU → Squeeze
```

Each neuron has independent parameters processed via grouped linear operations.

### Synchronization Computation

```python
# Efficient recurrence for synchronization
decay = exp(-learnable_decay_param)
sync_numerator = decay * prev_numerator + z_i * z_j
sync_denominator = decay * prev_denominator + 1
synchronization = sync_numerator / sqrt(sync_denominator)
```

**Neuron Selection Strategies**:
- `random-pairing`: Sample n_synch random (i,j) pairs
- `first-last`: Pair neurons [0:n] with neurons [-n:]
- `random`: Random subset of upper triangular matrix

---

## Hyperparameter Reference

### Core Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Internal latent dimension |
| `iterations` | 75 | Internal thought ticks (T) |
| `memory_length` | 25 | NLM history window (M) |
| `heads` | 8 | Attention heads |
| `n_synch_out` | 32 | Output sync neurons |
| `n_synch_action` | 32 | Action sync neurons |
| `synapse_depth` | 8 | Synapse U-NET depth |
| `deep_nlms` | True | Use deep NLM variant |
| `memory_hidden_dims` | 32 | NLM hidden size |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Initial LR |
| `weight_decay` | 0.0 | AdamW decay |
| `batch_size` | 16-32 | Training batch |
| `warmup_steps` | 5000 | LR warmup |
| `scheduler` | cosine | LR schedule type |

---

## Memory & Compute Profile

### Memory Usage (Batch=16, D=512, T=75, M=25)

| Component | Size | Notes |
|-----------|------|-------|
| Model parameters | ~50 MB | Backbone + CTM |
| State trace | 800 KB | [B, D, M] |
| Activated state | 32 KB | [B, D] |
| Sync accumulators | 64 KB | Decay numerator/denominator |
| Attention cache | Variable | Depends on input size |
| **Per-iteration** | ~5 MB | Activations + gradients |
| **Total (T=75)** | ~400 MB | Full unroll |

### Compute Characteristics

- **Sequential**: T iterations cannot be parallelized
- **Memory-bound**: Large state traces dominate
- **Attention**: O(D × input_tokens) per iteration
- **NLM**: O(D × M × hidden) per iteration

---

## Comparison: CTM vs TRM

| Aspect | CTM | TRM |
|--------|-----|-----|
| **Recursion** | External loop with state | Recursive weight sharing |
| **Temporal** | Neuron-level history traces | Dual semantic states (y, z) |
| **Synchronization** | Neural pairwise dot products | Q-head halting |
| **Memory** | Explicit trace buffer | Implicit in recursion |
| **Parallelism** | Sequential iterations | Can checkpoint |
| **Certainty** | Entropy-based | Learned Q-head |
| **Focus** | Temporal neural dynamics | Depth through recursion |

---

## Performance Benchmarks (Original Paper)

| Task | CTM Performance | Notes |
|------|-----------------|-------|
| ImageNet | Strong top-5 accuracy | Excellent calibration |
| CIFAR-10 | Best calibration | Better than humans |
| 2D Mazes (39×39) | ~100% accuracy | Path length 100 |
| 2D Mazes (99×99) | Generalizes well | 6× longer paths |
| Parity | 100% (75+ ticks) | LSTMs fail >10 ticks |
| RL (CartPole) | Competitive | Stable with more ticks |

---

## Code Repair Adaptation

To adapt CTM for Python code repair:

### Input Representation
- **Backbone**: Code-specific encoder (not ResNet)
- **Tokenization**: BPE vocabulary (32K tokens)
- **Positional**: 2D grid positions (line, column)

### Architecture Modifications
- **d_model**: 256-512 (code has less visual complexity)
- **iterations**: 50-100 (reasoning about code)
- **memory_length**: 32 (longer context for code patterns)
- **Output**: Vocabulary-sized logits per grid position

### Training Objectives
- **Primary**: Cross-entropy on fixed code tokens
- **Auxiliary**: Bug location prediction (synchronization)
- **Certainty**: When to stop iterating

### Expected Advantages
1. **Iterative refinement**: Complex bugs may need multiple passes
2. **Synchronization**: May capture code structure patterns
3. **Adaptive compute**: Simple fixes need fewer iterations
