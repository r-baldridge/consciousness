# Continuous Thought Machine (CTM)

> A neural architecture leveraging neural dynamics as its core representation, departing from conventional feed-forward models through decoupled internal time and neural synchronization.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Continuous Thought Machines](https://arxiv.org/abs/2505.05522)
**Year:** 2025
**Organization:** Sakana AI
**Recognition:** NeurIPS 2025 Spotlight

## Overview

The Continuous Thought Machine (CTM) represents a fundamental departure from traditional neural network architectures by treating neural dynamics itself as the core computational representation. Unlike standard transformers that process information through fixed-depth feed-forward layers, CTM introduces an internal temporal axis that is decoupled from input data, allowing neurons to unfold their activity over this internal time dimension.

At the heart of CTM are Neuron-Level Models (NLMs), where each neuron maintains its own internal weights and processes the history of incoming signals to determine its next activation. This creates a mid-level abstraction between single weights and full layers, enabling adaptive computation depth that naturally emerges from the architecture rather than being explicitly programmed.

The architecture's most distinctive feature is neural synchronization, where information is encoded not just in activation values but in the timing of neural activity. Synchronization patterns form latent representations directly, leading to emergent behaviors such as adaptive computation for complex inputs and learned algorithms for tasks like maze solving that arise from architectural constraints rather than explicit training objectives.

## Key Innovations

- **Decoupled Internal Time Dimension**: Neuron activity unfolds over an internal temporal axis independent from input data, enabling variable computation depth per input based on complexity.

- **Neuron-Level Models (NLMs)**: Each neuron has its own internal weights and processes its activation history, calculating next activations based on temporal context.

- **Neural Synchronization**: Information is encoded in the timing and coordination of neural activity, with synchronization patterns forming latent representations for task solutions.

## Architecture Diagram

```
Standard Transformer:
    Input --> [Layer 1] --> [Layer 2] --> ... --> [Layer N] --> Output
    (Fixed depth, parallel processing within layers)

Continuous Thought Machine:
    Input --> [NLM neurons with internal time] --> Output
                       |
                 Each neuron:
                 - Maintains activation history h_i(t)
                 - Has unique temporal weights W_i
                 - Computes: a_i(t+1) = f(W_i * h_i(t))
                 - Synchronizes with other neurons
                       |
                 Output derived from synchronization patterns:
                       y = g(sync_matrix)
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Model | Stub | NeuronLevelModel API defined |
| TemporalHistory | Stub | Buffer structure defined |
| SynchronizationLayer | Stub | API defined |
| Training Loop | Stub | CLI ready |
| Inference | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | Default YAML |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- einops
- numpy

### Hardware
- GPU recommended for training
- Adaptive compute makes inference timing variable
- VRAM requirements scale with num_neurons * history_length

### External Resources
- [ ] Official implementation reference: [SakanaAI/continuous-thought-machines](https://github.com/SakanaAI/continuous-thought-machines)
- [ ] Training data: ImageNet for classification benchmarks
- [ ] Maze datasets for algorithmic emergence studies

## Quick Start

```python
from consciousness.ml_research.modern_dev.ctm import DEFAULT_CONFIG

# Configuration
config = {
    "hidden_dim": 512,
    "num_neurons": 1024,
    "history_length": 8,
    "max_internal_steps": 32,
    "sync_window": 4,
    "halt_threshold": 0.01,
    "neuron_activation": "gelu",
}

# When implemented:
# from consciousness.ml_research.modern_dev.ctm.src.model import CTM, CTMConfig
# model = CTM(CTMConfig(**config))
```

## File Structure

```
ctm/
├── __init__.py       # Module documentation and metadata
├── README.md         # This file
├── src/
│   ├── model.py      # Main CTM implementation
│   ├── nlm.py        # NeuronLevelModel class
│   └── sync.py       # SynchronizationLayer
├── configs/
│   └── default.yaml  # Default configuration
├── cli/
│   ├── train.py      # Training script (ctm-train)
│   └── infer.py      # Inference with visualization (ctm-infer)
├── models/           # Pretrained model checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**Neuron-Level Model (NLM):**
```
a_i(t+1) = f(W_i * h_i(t))

where:
    a_i(t) = activation of neuron i at internal time t
    h_i(t) = [a_i(t), a_i(t-1), ..., a_i(t-k)] = history
    W_i = unique weights for neuron i
    f = activation function (GELU)
```

**Neural Synchronization:**
```
sync(i, j) = correlation(a_i(t), a_j(t)) over time window

Output derived from synchronization patterns:
    y = g(sync_matrix)
```

**Adaptive Computation:**
```
Halt when synchronization stabilizes:
    ||sync(t) - sync(t-1)|| < threshold
```

## Benchmarks

| Task | Performance |
|------|-------------|
| ImageNet-1K Top-1 | 72.47% |
| ImageNet-1K Top-5 | 89.89% |
| 2D Maze Solving | Emergent algorithms |
| Sorting | Competitive |
| Parity Computation | Strong |

## References

- Song, L., Jones, L., Ha, D., et al. "Continuous Thought Machines" (2025). arXiv:2505.05522
- [Official GitHub Implementation](https://github.com/SakanaAI/continuous-thought-machines)
- Related: Adaptive Computation Time (Graves, 2016)

## Contributing

To complete this implementation:

1. **Phase 1: Core Architecture**
   - Implement `NeuronLevelModel` class with temporal history tracking
   - Create `TemporalHistory` buffer with efficient circular storage
   - Build basic forward pass with internal time loop

2. **Phase 2: Training Infrastructure**
   - Implement loss functions for synchronization learning
   - Add gradient handling for temporal unrolling
   - Create training loop with variable compute handling

3. **Phase 3: Tasks and Evaluation**
   - Implement ImageNet classification pipeline
   - Create maze solving visualization tools
   - Add attention/synchronization analysis utilities
