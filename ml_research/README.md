# ML Research Module

A comprehensive index and implementation hub for machine learning methods spanning from 1943 to present day.

## Overview

The `ml_research` module serves as both a **research encyclopedia** and an **active development workspace** for machine learning architectures. It provides:

- **Historical Index**: 200+ methods organized by era, category, and lineage
- **Method Registry**: Central database with papers, benchmarks, and relationships
- **Modern Implementations**: Working implementations of cutting-edge 2023-2025 architectures
- **Application Techniques**: Composable patterns for deploying ML models effectively

This module enables the system to understand the evolution of ML methods, select appropriate architectures for tasks, and apply them using proven techniques.

---

## Module Structure

```
ml_research/
├── foundations/          # Era 1: 1943-1980 - Birth of neural networks
│   ├── mcculloch_pitts.py      # First computational neuron model (1943)
│   ├── perceptron.py           # Rosenblatt's perceptron (1958)
│   ├── adaline_madaline.py     # Adaptive linear elements (1960)
│   ├── hebbian_learning.py     # Hebb's learning rule
│   ├── hopfield_network.py     # Associative memory networks (1982)
│   └── backpropagation_early.py # Early backprop concepts
│
├── classical/            # Era 2: 1980-2006 - Foundations solidify
│   ├── mlp.py                  # Multi-layer perceptrons
│   ├── backprop_rumelhart.py   # Modern backpropagation (1986)
│   ├── cnn_lecun.py            # LeNet and early CNNs
│   ├── svm.py                  # Support Vector Machines
│   ├── rbf_networks.py         # Radial Basis Function networks
│   ├── autoencoders.py         # Auto-associative networks
│   ├── recurrent/              # RNN, LSTM, GRU
│   ├── ensemble/               # Random Forest, Boosting, XGBoost
│   └── probabilistic/          # Boltzmann machines, RBM, DBN
│
├── deep_learning/        # Era 3: 2006-2017 - Deep learning revolution
│   ├── alexnet.py              # ImageNet breakthrough (2012)
│   ├── vgg.py                  # Very deep networks
│   ├── resnet.py               # Residual connections
│   ├── inception.py            # Inception modules
│   ├── densenet.py             # Dense connections
│   ├── batch_norm.py           # Batch normalization
│   ├── dropout.py              # Regularization technique
│   ├── seq2seq.py              # Sequence-to-sequence models
│   ├── unet.py                 # U-Net for segmentation
│   ├── generative/             # GAN, VAE, diffusion models
│   └── object_detection/       # R-CNN, YOLO, SSD
│
├── attention/            # Era 4: 2017-present - Transformer era
│   ├── attention_mechanism.py  # Core attention concepts
│   ├── self_attention.py       # Self-attention implementation
│   ├── transformer.py          # "Attention Is All You Need"
│   ├── positional_encoding.py  # Position representations
│   ├── language_models/        # GPT, BERT, T5, LLaMA, Mistral, Claude
│   ├── vision_transformers/    # ViT, DeiT, Swin, SAM
│   ├── multimodal/             # CLIP, Flamingo, LLaVA, GPT-4V
│   └── efficient_attention/    # Flash, Sparse, Linear, Mamba
│
├── novel/                # Era 5: 2023+ - Emerging methods
│   ├── state_space/            # S4, Mamba, SSM variants
│   ├── mixture_of_experts/     # MoE architectures
│   ├── world_models/           # World modeling approaches
│   ├── emerging/               # KAN, Liquid Networks
│   ├── neural_architecture_search/  # NAS, ENAS, DARTS
│   └── neuro_symbolic/         # Neural Turing, differentiable programming
│
├── reinforcement/        # Reinforcement Learning methods
│   ├── classical/              # Q-learning, SARSA, TD, Policy Gradient
│   ├── deep_rl/                # DQN, A3C, PPO, SAC, DDPG, Rainbow
│   └── rlhf.py                 # RL from Human Feedback
│
├── optimization/         # Training methods
│   ├── gradient_descent.py     # Core optimization
│   ├── adaptive/               # Adam, AdamW, Lion, LAMB, etc.
│   └── learning_rate/          # Schedulers, LR finder
│
├── core/                 # Registry and taxonomy system
│   ├── method_registry.py      # Central method index
│   ├── taxonomy.py             # Classification system
│   ├── timeline.py             # Historical progression
│   ├── lineage_tracker.py      # Evolution and influence
│   ├── paper_index.py          # Landmark papers database
│   └── benchmark_tracker.py    # SOTA tracking
│
├── modern_dev/           # ACTIVE DEVELOPMENT - 2023-2025 architectures
│   ├── orchestrator/           # Dynamic architecture selection
│   ├── ctm/                    # Continuous Thought Machine
│   ├── jepa/                   # Joint Embedding Predictive Architecture
│   ├── ttt/                    # Test-Time Training
│   ├── mamba_impl/             # Mamba SSM implementation
│   ├── rwkv/                   # RWKV-7 "Goose"
│   ├── xlstm/                  # Extended LSTM
│   ├── hyena/                  # Long convolutions
│   ├── griffin/                # Gated linear recurrence
│   ├── titans/                 # Meta in-context memory
│   ├── flow_matching/          # Optimal transport generation
│   ├── consistency_models/     # Fast diffusion sampling
│   └── ring_attention/         # Infinite context attention
│
├── ml_techniques/        # Application patterns and techniques
│   ├── decomposition/          # Task breakdown methods
│   ├── prompting/              # CoT, ToT, GoT, few-shot
│   ├── agentic/                # ReAct, tool calling, multi-agent
│   ├── memory/                 # RAG, episodic memory, compression
│   ├── code_synthesis/         # Program synthesis, RLM, self-debugging
│   ├── orchestration/          # Routing, ensembles, pipelines
│   ├── verification/           # Self-eval, CoVe, constitutional
│   └── optimization/           # DSPy, automatic prompt engineering
│
└── config/               # Configuration files
    ├── methods.yaml            # Method definitions
    ├── papers.yaml             # Paper references
    ├── benchmarks.yaml         # Benchmark data
    └── lineages.yaml           # Method relationships
```

---

## Modern Dev Section

The `modern_dev/` directory contains **active development implementations** of cutting-edge ML architectures from 2023-2025. Unlike the research index modules, these are intended to become working implementations with command-line interfaces.

### Purpose

- Provide working implementations of the latest sequence modeling and generative architectures
- Enable the system to dynamically select and employ appropriate architectures for tasks
- Support research and experimentation with emerging methods
- Create modular, swappable implementations for different use cases

### Indexed Architectures

**Tier 1 - Production Ready** (open source, well-documented):

| Architecture | Organization | Key Innovation |
|-------------|--------------|----------------|
| **CTM** | Sakana AI | Neural dynamics with decoupled internal time, emergent adaptive computation |
| **JEPA** | Meta AI | Latent space prediction without pixel reconstruction |
| **xLSTM** | NXAI Lab | Exponential gating + matrix memory for transformer-level performance |
| **RWKV** | RWKV Foundation | RNN efficiency with transformer parallelization, O(n) complexity |
| **Griffin** | Google DeepMind | Gated linear recurrence + local attention hybrid |
| **Mamba** | CMU/Princeton | Selective state spaces with hardware-aware parallel scan |

**Tier 2 - Research Ready** (papers available, some code):

| Architecture | Organization | Key Innovation |
|-------------|--------------|----------------|
| **TTT** | Stanford/NVIDIA | Learnable hidden states that adapt during inference |
| **Hyena** | Hazy Research | Implicit long convolutions for 100x speedup at 64K |
| **Consistency Models** | OpenAI | One-step generation with diffusion quality |
| **Flow Matching** | Meta/DeepMind | Optimal transport paths for fast sampling |
| **Ring Attention** | UC Berkeley | Distributed attention for near-infinite context |
| **Titans** | Google | Meta in-context memory learning at test-time |

### Orchestrator System

The orchestrator provides dynamic architecture selection based on task requirements:

```python
from consciousness.ml_research.modern_dev import Orchestrator, TaskType

# Initialize orchestrator
orch = Orchestrator(device="cuda", max_loaded_models=2)

# Automatic architecture selection
result = orch.run(
    task_type=TaskType.LONG_CONTEXT,
    input_data={"text": "...very long document..."},
    constraints={"context_length": 100000, "max_memory_gb": 16}
)

# Or specify architecture directly
result = orch.run(
    task_type=TaskType.TEXT_GENERATION,
    input_data={"prompt": "Hello"},
    architecture="mamba"
)
```

### Current Status

| Component | Status |
|-----------|--------|
| Architecture index | Complete |
| Orchestrator base | Complete |
| Architecture scaffolding | Complete |
| CTM implementation | In progress |
| JEPA implementation | Indexed |
| Other implementations | Indexed (scaffolding ready) |

---

## ML Techniques Section

The `ml_techniques/` module provides composable **application patterns** for using ML models effectively. These are not architectures but ways to structure model usage for complex tasks.

### Categories

| Category | Techniques | Description |
|----------|------------|-------------|
| **Decomposition** | 3 | Breaking complex tasks into manageable parts (recursive, least-to-most, hierarchical) |
| **Prompting** | 5 | Structured input formulation (CoT, ToT, GoT, self-consistency, few-shot) |
| **Agentic** | 5 | Autonomous execution patterns (ReAct, tool calling, multi-agent, reflexion, planning) |
| **Memory** | 3 | Context and knowledge management (RAG, episodic memory, compression) |
| **Code Synthesis** | 4 | Code generation patterns (RLM, program synthesis, self-debugging, code-as-policy) |
| **Orchestration** | 3 | Multi-component coordination (routing, ensembles, hooks) |
| **Verification** | 4 | Output validation (self-eval, CoVe, constitutional, debate) |
| **Optimization** | 2 | Technique improvement (DSPy, automatic prompt engineering) |

**Total: 29 indexed techniques**

### Composable Pipeline System

Techniques can be composed using intuitive operators:

```python
from consciousness.ml_research.ml_techniques import Pipeline, compose
from consciousness.ml_research.ml_techniques import TechniqueBase

# Sequential composition with >>
pipeline = technique_a >> technique_b >> technique_c

# Parallel composition with |
parallel = technique_a | technique_b | technique_c

# Or explicit composition
pipeline = compose([
    RecursiveDecomposition(max_depth=3),
    ChainOfThought(cot_trigger="Let's think step by step."),
    ToolCalling(tools=[calculator, search]),
    SelfConsistency(samples=5),
])

result = pipeline.run(complex_task)
```

### Configuration-Driven Usage

```python
from consciousness.ml_research.ml_techniques import load_config

# Load from YAML configuration
config = load_config("configs/research_pipeline.yaml")
```

---

## Goals

### Research Index Goals
- Document the complete evolution of ML methods from 1943 to present
- Track method lineages and influence relationships
- Maintain paper references and benchmark results
- Enable historical understanding and method comparison

### Modern Dev Goals
- **Dynamic Selection**: Enable the system to automatically select appropriate architectures based on task requirements, constraints, and available resources
- **Modular Implementations**: Provide swappable implementations that follow a consistent interface
- **Research Support**: Support experimentation with architecture modifications and hybrid approaches
- **Production Path**: Create a clear path from research prototype to production deployment

### ML Techniques Goals
- **Composability**: Allow techniques to be combined freely for complex pipelines
- **Configurability**: Support YAML/JSON configuration for reproducibility
- **Pluggability**: Work with any compatible model backend
- **Traceability**: Provide execution traces for debugging and analysis

---

## Getting Started

### Basic Usage - Research Index

```python
from consciousness.ml_research import (
    MethodRegistry,
    Timeline,
    LineageTracker,
    MethodEra,
)

# Get all indexed methods
all_methods = MethodRegistry.get_all()
print(f"Total methods indexed: {len(all_methods)}")

# Get methods by era
attention_methods = MethodRegistry.get_by_era(MethodEra.ATTENTION)

# Show method lineage
LineageTracker.show_lineage('transformer')

# View timeline for an era
Timeline.show_era('deep_learning')

# Search for methods
results = MethodRegistry.search('attention')
```

### Using Modern Architectures

```python
from consciousness.ml_research.modern_dev import (
    Orchestrator,
    TaskType,
    get_architecture_info,
    list_architectures,
    ImplementationTier,
)

# Get information about an architecture
ctm_info = get_architecture_info('ctm')
print(f"CTM: {ctm_info.key_innovation}")

# List production-ready architectures
tier1 = list_architectures(tier=ImplementationTier.TIER_1)

# Use orchestrator for task execution
orch = Orchestrator()

# Run with automatic architecture selection
result = orch.run(
    task_type=TaskType.REASONING,
    input_data={"problem": "Solve this puzzle..."},
)

# Check what architectures are available for a task
available = orch.list_available(task_type=TaskType.LONG_CONTEXT)
```

### Applying ML Techniques

```python
from consciousness.ml_research.ml_techniques import (
    get_technique_info,
    list_techniques,
    TechniqueCategory,
    compose,
    Pipeline,
)

# Get technique information
cot_info = get_technique_info('chain_of_thought')
print(f"CoT composable with: {cot_info.composable_with}")

# List techniques by category
prompting_techniques = list_techniques(category=TechniqueCategory.PROMPTING)

# Find composable techniques
from consciousness.ml_research.ml_techniques import get_composable_with
compatible = get_composable_with('react')

# Build a custom pipeline (when implementations are complete)
# pipeline = compose([
#     ChainOfThought(),
#     ToolCalling(tools=[...]),
#     SelfConsistency(samples=5),
# ])
```

---

## Roadmap

### Complete
- [x] Research index structure (foundations through novel)
- [x] Core registry system (method_registry, taxonomy, lineage, timeline)
- [x] Modern dev architecture index (12 architectures catalogued)
- [x] Orchestrator base implementation
- [x] ML techniques index (29 techniques across 8 categories)
- [x] Technique base classes and composition system
- [x] Architecture scaffolding for all modern_dev modules

### In Progress
- [ ] CTM (Continuous Thought Machine) implementation
- [ ] JEPA family implementation
- [ ] Technique implementations (ChainOfThought, RAG, etc.)

### Planned
- [ ] Mamba implementation with hardware-aware kernels
- [ ] xLSTM with parallelizable training
- [ ] Ring Attention for distributed inference
- [ ] Full technique implementations with model backends
- [ ] Configuration-driven pipeline loading
- [ ] Benchmark integration for architecture comparison
- [ ] CLI tools for each architecture

---

## Architecture

### Design Principles

1. **Separation of Concerns**: Index/registry is separate from implementations
2. **Consistent Interfaces**: All architectures inherit from `ArchitectureBase`
3. **Lazy Loading**: Models load only when needed
4. **Resource Management**: Orchestrator manages memory with LRU eviction
5. **Composability**: Techniques combine freely via operators

### Key Classes

```python
# Core taxonomy
from consciousness.ml_research.core import (
    MethodEra,      # FOUNDATIONS, CLASSICAL, DEEP_LEARNING, ATTENTION, NOVEL
    MethodCategory, # SUPERVISED, UNSUPERVISED, GENERATIVE, etc.
    MLMethod,       # Method metadata dataclass
)

# Modern dev
from consciousness.ml_research.modern_dev import (
    ArchitectureBase,    # Base class for implementations
    ArchitectureIndex,   # Architecture metadata
    Orchestrator,        # Task routing and execution
    TaskType,            # Task categorization
    TaskSpec,            # Task specification
    TaskResult,          # Execution result
)

# ML techniques
from consciousness.ml_research.ml_techniques import (
    TechniqueBase,       # Base class for techniques
    TechniqueCategory,   # PROMPTING, AGENTIC, MEMORY, etc.
    TechniqueResult,     # Execution result
    Pipeline,            # Sequential composition
    ParallelComposition, # Parallel execution
)
```

---

## Contributing

When adding new methods or implementations:

1. **Research Index**: Add to appropriate era module, include paper reference and lineage
2. **Modern Dev**:
   - Create subdirectory with `__init__.py`, `src/`, `configs/`, `tests/`
   - Implement `ArchitectureBase` subclass
   - Register in `ARCHITECTURE_CAPABILITIES`
3. **ML Techniques**:
   - Add `TechniqueIndex` entry to registry
   - Implement `TechniqueBase` subclass
   - Document composability with other techniques

---

## Version

- **Module Version**: 1.0.0
- **Modern Dev Version**: 0.1.0 (development)
- **ML Techniques Version**: 0.1.0 (indexed)
