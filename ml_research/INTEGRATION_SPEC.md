# ML Research Module - Unified Integration Specification

## Overview

This document defines the integration layer that unifies the three major components of the `ml_research` module:

1. **Historical Research Index** (`foundations/`, `classical/`, `deep_learning/`, `attention/`, `novel/`, `reinforcement/`, `optimization/`) - 200+ methods from 1943-present
2. **Modern Architectures** (`modern_dev/`) - 12 state-of-the-art implementations (2023-2025+)
3. **Application Techniques** (`ml_techniques/`) - 50+ composable patterns for applying models
4. **Shared Infrastructure** (`shared/`) - Unified data pipeline and resources

---

## Component Summary

### Historical Research Index (core/)
- **Registry**: `MethodRegistry` - 200+ methods
- **Taxonomy**: `MLMethod`, `MethodEra`, `MethodCategory`, `MethodLineage`
- **Timeline**: Historical progression tracking
- **Lineage**: Evolution and influence relationships

### Modern Architectures (modern_dev/)
- **Registry**: `ARCHITECTURES` list with `ArchitectureIndex` entries
- **Orchestrator**: `Orchestrator` for task routing across architectures
- **Capabilities**: `ARCHITECTURE_CAPABILITIES` mapping
- **Status**: `DevelopmentStatus`, `ImplementationTier`

### Application Techniques (ml_techniques/)
- **Registry**: `TECHNIQUE_INDEX` with `TechniqueIndex` entries
- **Categories**: 8 technique categories (Decomposition, Prompting, Agentic, etc.)
- **Composition**: `Pipeline`, `compose()` for technique chaining
- **Execution**: `TechniqueResult`, `TechniqueConfig`

### Shared Infrastructure (shared/)
- **Data Pipeline**: Canonical Parquet format for training data
- **Loaders**: Architecture-specific data loaders (TRM, CTM, etc.)
- **Taxonomy**: Bug types and categories for code repair

---

## Unified Index Structure

```python
class UnifiedIndex:
    """
    Central access point for all ml_research components.

    Cross-references:
    - Historical methods → Modern implementations
    - Architectures → Compatible techniques
    - Techniques → Required capabilities
    - Data loaders → Architecture requirements
    """

    # Historical → Modern mapping
    LINEAGE_TO_ARCHITECTURE = {
        'transformer': ['ctm', 'titans', 'ring_attention'],
        'lstm': ['xlstm', 'rwkv', 'griffin'],
        'state_space': ['mamba_impl', 'hyena'],
        'vae': ['flow_matching', 'consistency_models'],
        'attention': ['ctm', 'flash_attention', 'ring_attention'],
    }

    # Architecture → Technique compatibility
    ARCHITECTURE_TECHNIQUES = {
        'ctm': ['chain_of_thought', 'recursive_decomposition', 'self_reflection'],
        'mamba_impl': ['streaming', 'long_context', 'efficient_inference'],
        'xlstm': ['sequential_reasoning', 'time_series', 'streaming'],
        'jepa': ['self_supervised', 'representation_learning', 'multimodal'],
    }

    # Technique → Required capabilities
    TECHNIQUE_REQUIREMENTS = {
        'chain_of_thought': {'reasoning': True, 'min_context': 2048},
        'tool_calling': {'function_calling': True, 'structured_output': True},
        'rag': {'embeddings': True, 'retrieval': True},
    }
```

---

## Cross-Reference Mappings

### 1. Historical Method → Modern Architecture

Maps foundational methods to their modern implementations:

| Historical Method | Era | Modern Architecture | Relationship |
|-------------------|-----|---------------------|--------------|
| Transformer (2017) | Attention | CTM, Titans, Ring Attention | Evolution |
| LSTM (1997) | Classical | xLSTM, RWKV, Griffin | Modernization |
| S4/SSM (2021) | Novel | Mamba, Hyena | Implementation |
| VAE (2013) | Deep Learning | Flow Matching, Consistency | Alternative |
| Attention (2014) | Attention | Flash Attention, Ring Attention | Optimization |
| Diffusion (2020) | Novel | Flow Matching, Consistency | Successor |

### 2. Architecture → Technique Compatibility

| Architecture | Compatible Techniques | Incompatible | Notes |
|--------------|----------------------|--------------|-------|
| CTM | CoT, ToT, Reflection, Iterative | Streaming | Requires iterative processing |
| Mamba | Streaming, Long Context | High-memory RAG | O(1) inference per token |
| xLSTM | Sequential, Time Series | Heavy parallel | Linear complexity |
| JEPA | Self-supervised, Multimodal | Text-only | Vision-first architecture |
| RWKV | Streaming, Long Context | Dense attention | RNN-style inference |

### 3. Technique → Data Requirements

| Technique | Data Format | Loader | Special Fields |
|-----------|-------------|--------|----------------|
| Code Repair | Canonical Parquet | TRM/CTM Loader | bug_location, diff_mask |
| Chain-of-Thought | JSON with steps | Generic | reasoning_trace |
| RAG | Documents + Embeddings | RAG Loader | chunk_embeddings |
| Self-Reflection | Multi-turn dialogue | Dialogue Loader | reflection_turns |

---

## API Design

### Unified Import Interface

```python
# Single entry point for all components
from ml_research import (
    # Core taxonomy
    MethodRegistry, MLMethod, MethodEra, MethodCategory,

    # Modern architectures
    Orchestrator, ARCHITECTURES, TaskType, run_task,

    # Techniques
    TECHNIQUE_INDEX, Pipeline, compose,

    # Data
    CanonicalCodeSample, TRMDataLoader, CTMDataLoader,

    # Unified index
    UnifiedIndex, get_architecture_for_method, get_techniques_for_architecture,
)
```

### Cross-Reference Queries

```python
# Find modern implementations of a historical method
modern_archs = UnifiedIndex.get_modern_implementations('transformer')
# Returns: ['ctm', 'titans', 'ring_attention']

# Find compatible techniques for an architecture
techniques = UnifiedIndex.get_compatible_techniques('ctm')
# Returns: ['chain_of_thought', 'recursive_decomposition', 'self_reflection']

# Find architectures that support a technique
archs = UnifiedIndex.get_architectures_for_technique('streaming')
# Returns: ['mamba_impl', 'rwkv', 'griffin']

# Get full lineage from historical to modern
lineage = UnifiedIndex.trace_lineage('perceptron', 'ctm')
# Returns: ['perceptron', 'mlp', 'transformer', 'ctm']

# Get data loader for architecture
loader_class = UnifiedIndex.get_data_loader('ctm')
# Returns: CTMDataLoader
```

---

## Integration Points

### 1. Orchestrator Integration

The `Orchestrator` in `modern_dev/` routes tasks to architectures. Integration adds:

```python
class EnhancedOrchestrator(Orchestrator):
    """Orchestrator with technique and lineage awareness."""

    def select_architecture(self, task: TaskSpec) -> str:
        # Base selection from capabilities
        candidates = super().select_architecture(task)

        # Filter by technique compatibility
        if task.required_techniques:
            candidates = [
                arch for arch in candidates
                if UnifiedIndex.supports_techniques(arch, task.required_techniques)
            ]

        # Prefer architectures with historical lineage to task domain
        if task.domain_hint:
            candidates = sorted(
                candidates,
                key=lambda a: UnifiedIndex.lineage_relevance(a, task.domain_hint),
                reverse=True
            )

        return candidates[0] if candidates else None
```

### 2. Technique Pipeline Integration

Techniques can reference architecture capabilities:

```python
class ArchitectureAwarePipeline(Pipeline):
    """Pipeline that validates technique-architecture compatibility."""

    def __init__(self, techniques: List[Technique], architecture: str = None):
        self.architecture = architecture

        if architecture:
            # Validate compatibility
            for technique in techniques:
                if not UnifiedIndex.is_compatible(architecture, technique.id):
                    raise IncompatibleTechniqueError(
                        f"{technique.id} not compatible with {architecture}"
                    )

        super().__init__(techniques)
```

### 3. Data Pipeline Integration

Data loaders are registered with the unified index:

```python
# In shared/data/loaders/__init__.py
from ..unified_index import UnifiedIndex

UnifiedIndex.register_loader('trm', TRMDataLoader)
UnifiedIndex.register_loader('ctm', CTMDataLoader)
# Future: register loaders for other architectures

# Architecture-specific requirements are documented
LOADER_REQUIREMENTS = {
    'trm': {
        'grid_dimensions': (64, 48),
        'special_fields': ['buggy_grid', 'fixed_grid', 'diff_mask'],
    },
    'ctm': {
        'grid_dimensions': (64, 48),
        'special_fields': ['positions', 'bug_location', 'bug_location_mask'],
    },
}
```

---

## Directory Structure

```
ml_research/
├── __init__.py                    # Updated with unified exports
├── INTEGRATION_SPEC.md            # This document
├── core/
│   ├── __init__.py
│   ├── unified_index.py          # NEW: Cross-reference index
│   ├── method_registry.py
│   ├── taxonomy.py
│   ├── timeline.py
│   └── lineage_tracker.py
├── modern_dev/
│   ├── __init__.py               # Architecture registry
│   ├── orchestrator/             # Task routing
│   ├── trm/                      # TRM implementation
│   ├── ctm/                      # CTM implementation
│   └── ...                       # Other architectures
├── ml_techniques/
│   ├── __init__.py               # Technique registry
│   ├── prompting/
│   ├── agentic/
│   └── ...                       # Technique categories
├── shared/
│   ├── __init__.py
│   ├── data/                     # Unified data pipeline
│   │   ├── schema.py             # CanonicalCodeSample
│   │   └── loaders/              # Architecture-specific loaders
│   └── taxonomy/                 # Bug types
├── foundations/                   # Historical: 1943-1980
├── classical/                     # Historical: 1980-2006
├── deep_learning/                 # Historical: 2006-2017
├── attention/                     # Historical: 2017-present
├── novel/                         # Historical: 2023+
├── reinforcement/                 # RL methods
└── optimization/                  # Optimization methods
```

---

## Migration Path

### Phase 1: Create Unified Index (Current)
- [x] Analyze existing components
- [ ] Create `core/unified_index.py`
- [ ] Add cross-reference mappings
- [ ] Update root `__init__.py`

### Phase 2: Enhance Orchestrator
- [ ] Add technique compatibility to task selection
- [ ] Add lineage-aware routing
- [ ] Integrate data loader discovery

### Phase 3: Connect Techniques
- [ ] Add architecture validation to pipelines
- [ ] Link technique requirements to capabilities
- [ ] Enable auto-configuration based on architecture

### Phase 4: Documentation
- [ ] Generate unified API reference
- [ ] Create usage examples
- [ ] Build interactive lineage browser

---

## Usage Examples

### Example 1: Find Best Architecture for Code Repair

```python
from ml_research import UnifiedIndex, Orchestrator, TaskSpec, TaskType

# Define task
task = TaskSpec(
    task_type=TaskType.CODE_REPAIR,
    complexity='hard',
    required_techniques=['chain_of_thought', 'self_reflection'],
)

# Get compatible architectures
archs = UnifiedIndex.get_architectures_for_task(task)
# Returns: ['ctm', 'trm'] - both support iterative reasoning

# Select best based on capabilities
selected = Orchestrator.select_best(task)
# Returns: 'ctm' - higher capacity for hard problems
```

### Example 2: Trace Method Evolution

```python
from ml_research import UnifiedIndex, Timeline

# Trace from foundational to modern
lineage = UnifiedIndex.trace_full_lineage('perceptron')
# Returns: {
#     'perceptron': ['mlp', 'deep_networks'],
#     'mlp': ['backprop', 'cnn', 'transformer'],
#     'transformer': ['bert', 'gpt', 'ctm', 'titans'],
# }

# Get timeline with modern implementations
timeline = Timeline.get_extended(include_modern=True)
# Includes both historical methods and modern architectures
```

### Example 3: Build Compatible Pipeline

```python
from ml_research import UnifiedIndex, compose
from ml_techniques import ChainOfThought, SelfReflection, ToolCalling

# Check architecture compatibility
arch = 'ctm'
techniques = [ChainOfThought(), SelfReflection(), ToolCalling()]

compatible = UnifiedIndex.filter_compatible(arch, techniques)
# Returns: [ChainOfThought(), SelfReflection()]
# ToolCalling filtered out if CTM doesn't support function calling

# Build valid pipeline
pipeline = compose(compatible, architecture=arch)
```

---

## Versioning

- Integration Spec Version: 1.0.0
- ml_research Version: 1.0.0
- modern_dev Version: 0.1.0
- ml_techniques Version: 0.1.0
- shared Version: 0.1.0

---

## Appendix: Complete Component Counts

| Component | Count | Status |
|-----------|-------|--------|
| Historical Methods | 200+ | Indexed |
| Modern Architectures | 12 | 6 Tier 1, 6 Tier 2 |
| Technique Categories | 8 | All implemented |
| Individual Techniques | 50+ | Indexed |
| Data Loaders | 2 | TRM, CTM |
| Research Eras | 5 | Complete |
| Curriculum Stages | 4 | Defined |
| Bug Categories | 5 | Syntax, Logic, Style, Security, Framework |
| Bug Types | 100+ | Full taxonomy |
