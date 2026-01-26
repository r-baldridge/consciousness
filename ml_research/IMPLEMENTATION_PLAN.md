# ML Research Implementation Plan

## Executive Summary

This plan orchestrates parallel development of four critical components:
1. **TRM (Tiny Recursive Model)** - 7M parameter recursive reasoning for code repair
2. **RLM (Recursive Language Model)** - Code synthesis through decomposition
3. **Mamba (Selective State Spaces)** - Linear-time sequence modeling for long context
4. **Integration Layer** - Unified pipeline combining all three

**Timeline**: 10 weeks with parallel subagent execution
**Team Size**: 6 parallel subagents + 1 coordinator
**Goal**: End-to-end code repair system with Mamba context → RLM decomposition → TRM refinement

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MAMBA-RLM-TRM PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   MAMBA      │    │    RLM       │    │    TRM       │              │
│  │   Encoder    │───▶│  Decomposer  │───▶│   Refiner    │──▶ Fixed    │
│  │              │    │              │    │              │     Code     │
│  │ Long Context │    │ Fix Strategy │    │ 8 Iterations │              │
│  │ O(N) time    │    │ Recursive    │    │ 42 eff depth │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         └───────────────────┴───────────────────┘                       │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │  Shared Data    │                                  │
│                    │    Pipeline     │                                  │
│                    │  (Canonical)    │                                  │
│                    └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Structure

```
PHASE 1 (Weeks 1-2): Foundation & Quick Wins
├── Agent-A: TRM Forward Pass
├── Agent-B: RLM Variable Extractor
├── Agent-C: Mamba Discrete SSM
└── Agent-D: Shared Backbone Library

PHASE 2 (Weeks 3-4): Core Completion
├── Agent-A: TRM Training Loop
├── Agent-B: RLM Code Generator
├── Agent-C: Mamba Selective Mechanism
└── Agent-E: Data Pipeline Validation

PHASE 3 (Weeks 5-7): Integration
├── Agent-A: TRM + RLM Composition
├── Agent-C: Mamba + Parallel Scan
├── Agent-F: Orchestrator Integration
└── Agent-D: Testing Framework

PHASE 4 (Weeks 8-10): Full Pipeline
├── Agent-F: End-to-End Pipeline
├── All: Benchmarking & Optimization
└── All: Documentation & Release
```

---

## Subagent Specifications

### Agent-A: TRM Specialist

**Responsibility**: Complete TRM architecture to trainable state

**Phase 1 Tasks** (Week 1-2):
```yaml
task_id: TRM-001
name: "Implement TRM Forward Pass"
files:
  - modern_dev/trm/src/model.py
  - modern_dev/trm/src/layers.py
deliverables:
  - DeepRecursion.forward() with full computation
  - RecursiveBlock with residual connections
  - Early stopping based on q_threshold
  - Gradient checkpointing for memory efficiency
acceptance_criteria:
  - Forward pass runs without error on sample input
  - Output shape matches expected [B, H, W, vocab_size]
  - Memory usage < 8GB for batch_size=32
dependencies: []
estimated_hours: 20
```

```yaml
task_id: TRM-002
name: "Implement Iteration Control"
files:
  - modern_dev/trm/src/model.py
deliverables:
  - IterationController class
  - Confidence-based early stopping
  - Iteration history tracking
  - Visualization hooks for analysis
acceptance_criteria:
  - Model can stop between 1-8 iterations based on confidence
  - Iteration decisions logged for debugging
dependencies: [TRM-001]
estimated_hours: 12
```

**Phase 2 Tasks** (Week 3-4):
```yaml
task_id: TRM-003
name: "Implement TRM Training Loop"
files:
  - modern_dev/trm/src/training.py
  - modern_dev/trm/cli/train.py
deliverables:
  - Full training loop with gradient accumulation
  - Mixed precision (FP16/BF16) support
  - Curriculum learning integration
  - Checkpoint saving/loading
  - Wandb/TensorBoard logging
acceptance_criteria:
  - Training runs for 100 steps without crash
  - Loss decreases over training
  - Checkpoints save and load correctly
dependencies: [TRM-001, TRM-002]
estimated_hours: 24
```

```yaml
task_id: TRM-004
name: "Implement TRM Loss Functions"
files:
  - modern_dev/trm/src/losses.py
deliverables:
  - Token-level cross-entropy loss
  - Diff-weighted loss (focus on changed tokens)
  - Intermediate supervision loss
  - Combined loss with configurable weights
acceptance_criteria:
  - All loss functions produce valid gradients
  - Diff weighting improves focus on bug locations
dependencies: [TRM-001]
estimated_hours: 8
```

**Phase 3 Tasks** (Week 5-7):
```yaml
task_id: TRM-005
name: "TRM + RLM Composition"
files:
  - modern_dev/trm/src/composition.py
  - modern_dev/shared/pipelines/trm_rlm.py
deliverables:
  - TRMWithRLM wrapper class
  - RLM strategy → TRM input conversion
  - Multi-step refinement with RLM re-decomposition
  - Error feedback loop
acceptance_criteria:
  - End-to-end pipeline processes bug → fix
  - RLM decomposition improves TRM accuracy
  - Feedback loop corrects initial failures
dependencies: [TRM-003, RLM-003]
estimated_hours: 20
```

---

### Agent-B: RLM Specialist

**Responsibility**: Implement Recursive Language Model for code synthesis

**Phase 1 Tasks** (Week 1-2):
```yaml
task_id: RLM-001
name: "Implement Variable Extractor"
files:
  - ml_techniques/code_synthesis/rlm.py
  - ml_techniques/code_synthesis/variable_extractor.py
deliverables:
  - parse() function for specification analysis
  - Entity extraction (functions, classes, variables)
  - Type inference from context
  - Relationship mapping between entities
acceptance_criteria:
  - Correctly extracts entities from 10 test cases
  - Type inference accuracy > 80%
  - Handles nested structures
dependencies: []
estimated_hours: 20
```

```yaml
task_id: RLM-002
name: "Implement Constraint Analyzer"
files:
  - ml_techniques/code_synthesis/constraints.py
deliverables:
  - Constraint extraction from specifications
  - Dependency graph construction
  - Validation rule generation
  - Constraint satisfaction checking
acceptance_criteria:
  - Extracts constraints from natural language
  - Builds accurate dependency graphs
  - Validates generated code against constraints
dependencies: [RLM-001]
estimated_hours: 16
```

**Phase 2 Tasks** (Week 3-4):
```yaml
task_id: RLM-003
name: "Implement Code Generator"
files:
  - ml_techniques/code_synthesis/generator.py
deliverables:
  - Template-based code generation
  - LLM-powered code completion
  - Multi-step generation with verification
  - Syntax validation
acceptance_criteria:
  - Generates syntactically valid Python
  - Passes type checking (mypy)
  - Handles common patterns (loops, conditionals, functions)
dependencies: [RLM-001, RLM-002]
estimated_hours: 24
```

```yaml
task_id: RLM-004
name: "Implement Self-Debugging Loop"
files:
  - ml_techniques/code_synthesis/debugger.py
deliverables:
  - Execution sandbox for generated code
  - Error analysis and classification
  - Fix strategy generation
  - Recursive repair attempts
acceptance_criteria:
  - Safely executes generated code
  - Classifies errors correctly
  - Fixes > 50% of simple errors in 3 attempts
dependencies: [RLM-003]
estimated_hours: 16
```

**Phase 3 Tasks** (Week 5-7):
```yaml
task_id: RLM-005
name: "RLM Pipeline Integration"
files:
  - ml_techniques/code_synthesis/pipeline.py
  - ml_techniques/__init__.py
deliverables:
  - RLMPipeline class with full workflow
  - Configuration via YAML
  - Hooks for external models (TRM, Mamba)
  - Streaming output support
acceptance_criteria:
  - Full pipeline runs end-to-end
  - Configurable via external config
  - Integrates with TRM seamlessly
dependencies: [RLM-003, RLM-004]
estimated_hours: 16
```

---

### Agent-C: Mamba Specialist

**Responsibility**: Implement Mamba architecture with selective state spaces

**Phase 1 Tasks** (Week 1-2):
```yaml
task_id: MAMBA-001
name: "Implement Discrete SSM Core"
files:
  - modern_dev/mamba_impl/src/ssm.py
  - modern_dev/mamba_impl/src/layers.py
deliverables:
  - S4D layer (diagonal state space)
  - Zero-order hold discretization
  - Recurrent mode for inference
  - Convolutional mode for training
acceptance_criteria:
  - SSM produces correct output shapes
  - Recurrent and conv modes produce same output
  - Memory efficient for long sequences
dependencies: []
estimated_hours: 24
```

```yaml
task_id: MAMBA-002
name: "Implement State Space Parameterization"
files:
  - modern_dev/mamba_impl/src/parameterization.py
deliverables:
  - HiPPO initialization for A matrix
  - Diagonal approximation
  - Normalization schemes
  - Numerical stability fixes
acceptance_criteria:
  - A matrix properly initialized
  - Eigenvalues in stable region
  - No numerical overflow/underflow
dependencies: [MAMBA-001]
estimated_hours: 12
```

**Phase 2 Tasks** (Week 3-4):
```yaml
task_id: MAMBA-003
name: "Implement Selective Mechanism"
files:
  - modern_dev/mamba_impl/src/selective.py
deliverables:
  - Input-dependent delta projection
  - Input-dependent B, C projections
  - Selective scan function
  - Gate computation
acceptance_criteria:
  - Selection parameters vary with input
  - Gate properly modulates state updates
  - Selectivity improves over static SSM
dependencies: [MAMBA-001, MAMBA-002]
estimated_hours: 20
```

```yaml
task_id: MAMBA-004
name: "Implement Mamba Block"
files:
  - modern_dev/mamba_impl/src/block.py
deliverables:
  - Full Mamba block architecture
  - Conv1D for local context
  - SiLU activation
  - Residual connections
  - Layer normalization
acceptance_criteria:
  - Block matches paper architecture
  - Forward pass is memory efficient
  - Backward pass computes gradients
dependencies: [MAMBA-003]
estimated_hours: 16
```

**Phase 3 Tasks** (Week 5-7):
```yaml
task_id: MAMBA-005
name: "Implement Parallel Scan"
files:
  - modern_dev/mamba_impl/src/scan.py
  - modern_dev/mamba_impl/src/kernels.py
deliverables:
  - Associative scan implementation
  - Work-efficient parallel algorithm
  - Triton kernel (or pure PyTorch fallback)
  - Memory-efficient chunked processing
acceptance_criteria:
  - O(log L) parallel complexity
  - Matches sequential scan output exactly
  - 3-5x speedup over sequential
dependencies: [MAMBA-003, MAMBA-004]
estimated_hours: 32
```

```yaml
task_id: MAMBA-006
name: "Implement Mamba Model"
files:
  - modern_dev/mamba_impl/src/model.py
deliverables:
  - Full Mamba model architecture
  - Token embedding layer
  - Stacked Mamba blocks
  - Output projection and loss
  - Generation/inference mode
acceptance_criteria:
  - Model trains on sample data
  - Generation produces coherent output
  - Inference is O(1) per token
dependencies: [MAMBA-004, MAMBA-005]
estimated_hours: 20
```

---

### Agent-D: Infrastructure Specialist

**Responsibility**: Shared components, testing, and backbone library

**Phase 1 Tasks** (Week 1-2):
```yaml
task_id: INFRA-001
name: "Create Shared Backbone Library"
files:
  - modern_dev/shared/blocks/__init__.py
  - modern_dev/shared/blocks/attention.py
  - modern_dev/shared/blocks/normalization.py
  - modern_dev/shared/blocks/activations.py
  - modern_dev/shared/blocks/embeddings.py
deliverables:
  - MultiHeadAttention (standard, flash, sparse)
  - RMSNorm, LayerNorm, GroupNorm
  - SwiGLU, GeGLU, SiLU activations
  - RotaryEmbedding, ALiBi
  - Conv1D with various initializations
acceptance_criteria:
  - All blocks have consistent API
  - Unit tests pass
  - Can be imported by all architectures
dependencies: []
estimated_hours: 24
```

```yaml
task_id: INFRA-002
name: "Create Testing Framework"
files:
  - modern_dev/tests/__init__.py
  - modern_dev/tests/conftest.py
  - modern_dev/tests/fixtures.py
deliverables:
  - Pytest configuration
  - Shared fixtures (sample data, models)
  - Performance benchmarking utilities
  - Memory profiling helpers
acceptance_criteria:
  - Tests run with `pytest modern_dev/tests/`
  - Fixtures work across all architectures
  - Benchmark results are reproducible
dependencies: []
estimated_hours: 12
```

**Phase 2 Tasks** (Week 3-4):
```yaml
task_id: INFRA-003
name: "Implement Shared Loss Functions"
files:
  - modern_dev/shared/losses/__init__.py
  - modern_dev/shared/losses/sequence.py
  - modern_dev/shared/losses/code_repair.py
deliverables:
  - Cross-entropy with label smoothing
  - Focal loss for imbalanced classes
  - Diff-weighted loss for code repair
  - Perplexity computation
acceptance_criteria:
  - All losses compute valid gradients
  - Losses work with any sequence model
dependencies: [INFRA-001]
estimated_hours: 12
```

**Phase 3 Tasks** (Week 5-7):
```yaml
task_id: INFRA-004
name: "Comprehensive Test Suite"
files:
  - modern_dev/tests/test_trm.py
  - modern_dev/tests/test_mamba.py
  - modern_dev/tests/test_integration.py
deliverables:
  - Unit tests for TRM components
  - Unit tests for Mamba components
  - Integration tests for full pipeline
  - Performance regression tests
acceptance_criteria:
  - > 80% code coverage
  - All critical paths tested
  - CI/CD ready
dependencies: [TRM-003, MAMBA-006, INFRA-002]
estimated_hours: 24
```

---

### Agent-E: Data Pipeline Specialist

**Responsibility**: Validate and extend data pipeline

**Phase 2 Tasks** (Week 3-4):
```yaml
task_id: DATA-001
name: "Validate Data Loaders"
files:
  - modern_dev/shared/data/loaders/trm.py
  - modern_dev/shared/data/loaders/ctm.py
  - modern_dev/shared/data/loaders/mamba.py
deliverables:
  - Unit tests for all loaders
  - Sample data generation scripts
  - Loader performance benchmarks
  - Documentation with examples
acceptance_criteria:
  - Loaders work with real Parquet files
  - Performance is adequate for training
  - All edge cases handled
dependencies: []
estimated_hours: 16
```

```yaml
task_id: DATA-002
name: "Create Mamba Data Loader"
files:
  - modern_dev/shared/data/loaders/mamba.py
deliverables:
  - MambaDataLoader class
  - Long sequence handling (chunking)
  - Streaming data support
  - Integration with curriculum learning
acceptance_criteria:
  - Handles sequences up to 100K tokens
  - Memory-efficient streaming
  - Works with existing bug taxonomy
dependencies: [DATA-001]
estimated_hours: 16
```

```yaml
task_id: DATA-003
name: "Build Sample Dataset"
files:
  - modern_dev/shared/data/scripts/build_dataset.py
  - modern_dev/shared/data/sample/
deliverables:
  - Script to convert GitHub diffs
  - Sample dataset (1000 examples)
  - Train/val/test splits
  - Quality tier assignments
acceptance_criteria:
  - Dataset follows CanonicalCodeSample schema
  - All bug categories represented
  - Ready for immediate training
dependencies: [DATA-001]
estimated_hours: 20
```

---

### Agent-F: Integration Specialist

**Responsibility**: Orchestrator integration and end-to-end pipeline

**Phase 3 Tasks** (Week 5-7):
```yaml
task_id: INTEG-001
name: "Enhance Orchestrator"
files:
  - modern_dev/orchestrator/__init__.py
  - modern_dev/orchestrator/router.py
deliverables:
  - Technique-aware routing
  - Architecture capability matching
  - Unified index integration
  - Fallback handling
acceptance_criteria:
  - Routes tasks to optimal architecture
  - Respects technique compatibility
  - Handles architecture failures gracefully
dependencies: [TRM-003, MAMBA-006]
estimated_hours: 16
```

```yaml
task_id: INTEG-002
name: "Build End-to-End Pipeline"
files:
  - modern_dev/pipelines/__init__.py
  - modern_dev/pipelines/code_repair.py
  - modern_dev/pipelines/mamba_trm_rlm.py
deliverables:
  - CodeRepairPipeline class
  - MambaTRMRLMPipeline class
  - Configuration loading
  - CLI interface
acceptance_criteria:
  - Pipeline runs bug → fix end-to-end
  - All three components integrated
  - Configurable via YAML
dependencies: [TRM-005, RLM-005, MAMBA-006, INTEG-001]
estimated_hours: 24
```

**Phase 4 Tasks** (Week 8-10):
```yaml
task_id: INTEG-003
name: "Benchmarking Suite"
files:
  - modern_dev/benchmarks/__init__.py
  - modern_dev/benchmarks/code_repair.py
  - modern_dev/benchmarks/speed.py
deliverables:
  - Code repair accuracy benchmarks
  - Speed/latency benchmarks
  - Memory usage profiling
  - Comparison with baselines
acceptance_criteria:
  - Benchmarks run reproducibly
  - Results exportable to JSON/CSV
  - Comparison charts generated
dependencies: [INTEG-002]
estimated_hours: 20
```

```yaml
task_id: INTEG-004
name: "Documentation and Examples"
files:
  - modern_dev/docs/QUICKSTART.md
  - modern_dev/docs/API.md
  - modern_dev/examples/
deliverables:
  - Quick start guide
  - API documentation
  - Jupyter notebook examples
  - Training tutorials
acceptance_criteria:
  - New user can run pipeline in < 30 min
  - All public APIs documented
  - Examples work out of the box
dependencies: [INTEG-002, INTEG-003]
estimated_hours: 16
```

---

## Dependency Graph

```
Week 1-2 (Phase 1 - Foundation)
┌─────────────────────────────────────────────────────────────┐
│  TRM-001 ──┐                                                │
│            ├──▶ TRM-002                                     │
│  RLM-001 ──┼──▶ RLM-002                                     │
│            │                                                │
│  MAMBA-001 ┼──▶ MAMBA-002                                   │
│            │                                                │
│  INFRA-001 ┴──▶ INFRA-002                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
Week 3-4 (Phase 2 - Core Completion)
┌─────────────────────────────────────────────────────────────┐
│  TRM-002 ───▶ TRM-003 ───▶ TRM-004                          │
│                                                             │
│  RLM-002 ───▶ RLM-003 ───▶ RLM-004                          │
│                                                             │
│  MAMBA-002 ─▶ MAMBA-003 ─▶ MAMBA-004                        │
│                                                             │
│  DATA-001 ──▶ DATA-002 ──▶ DATA-003                         │
│                                                             │
│  INFRA-001 ─▶ INFRA-003                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
Week 5-7 (Phase 3 - Integration)
┌─────────────────────────────────────────────────────────────┐
│  TRM-003 ───┐                                               │
│  RLM-003 ───┼──▶ TRM-005 (TRM + RLM Composition)            │
│             │                                               │
│  RLM-004 ───┴──▶ RLM-005 (RLM Pipeline)                     │
│                                                             │
│  MAMBA-004 ─┬──▶ MAMBA-005 (Parallel Scan)                  │
│             └──▶ MAMBA-006 (Full Model)                     │
│                                                             │
│  TRM-003 ───┐                                               │
│  MAMBA-006 ─┴──▶ INTEG-001 (Orchestrator)                   │
│                                                             │
│  INFRA-002 ─┬──▶ INFRA-004 (Test Suite)                     │
│  TRM-003 ───┤                                               │
│  MAMBA-006 ─┘                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
Week 8-10 (Phase 4 - Full Pipeline)
┌─────────────────────────────────────────────────────────────┐
│  TRM-005 ───┐                                               │
│  RLM-005 ───┼──▶ INTEG-002 (End-to-End Pipeline)            │
│  MAMBA-006 ─┤                                               │
│  INTEG-001 ─┘                                               │
│                                                             │
│  INTEG-002 ─┬──▶ INTEG-003 (Benchmarks)                     │
│             └──▶ INTEG-004 (Documentation)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Parallel Execution Schedule

### Week 1-2: Maximum Parallelism (4 agents)

| Agent | Tasks | Hours | Blocking |
|-------|-------|-------|----------|
| Agent-A | TRM-001, TRM-002 | 32 | None |
| Agent-B | RLM-001, RLM-002 | 36 | None |
| Agent-C | MAMBA-001, MAMBA-002 | 36 | None |
| Agent-D | INFRA-001, INFRA-002 | 36 | None |

### Week 3-4: Core Development (5 agents)

| Agent | Tasks | Hours | Blocking |
|-------|-------|-------|----------|
| Agent-A | TRM-003, TRM-004 | 32 | TRM-001/002 |
| Agent-B | RLM-003, RLM-004 | 40 | RLM-001/002 |
| Agent-C | MAMBA-003, MAMBA-004 | 36 | MAMBA-001/002 |
| Agent-D | INFRA-003 | 12 | INFRA-001 |
| Agent-E | DATA-001, DATA-002, DATA-003 | 52 | None |

### Week 5-7: Integration (6 agents)

| Agent | Tasks | Hours | Blocking |
|-------|-------|-------|----------|
| Agent-A | TRM-005 | 20 | TRM-003, RLM-003 |
| Agent-B | RLM-005 | 16 | RLM-003/004 |
| Agent-C | MAMBA-005, MAMBA-006 | 52 | MAMBA-003/004 |
| Agent-D | INFRA-004 | 24 | TRM-003, MAMBA-006 |
| Agent-F | INTEG-001 | 16 | TRM-003, MAMBA-006 |

### Week 8-10: Finalization (All agents)

| Agent | Tasks | Hours | Blocking |
|-------|-------|-------|----------|
| Agent-F | INTEG-002, INTEG-003, INTEG-004 | 60 | All Phase 3 |
| All | Bug fixes, optimization, polish | Variable | INTEG-002 |

---

## Subagent Prompts

### Agent-A Spawn Prompt (TRM)
```
You are Agent-A, the TRM Specialist. Your mission is to implement the
Tiny Recursive Model (TRM) - a 7M parameter recursive reasoning architecture
for code repair.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- TRM uses 8 recursive iterations with 42 effective layers
- Grid-based input: 64 rows × 48 tokens
- Key innovation: early stopping based on confidence threshold
- Must integrate with shared data pipeline

Implementation guidelines:
1. Read existing code in modern_dev/trm/src/ first
2. Follow patterns in modern_dev/shared/blocks/
3. Use PyTorch with mixed precision support
4. Add comprehensive docstrings
5. Write unit tests for each component

Report completion with:
- Files created/modified
- Test results
- Any blockers or decisions needed
```

### Agent-B Spawn Prompt (RLM)
```
You are Agent-B, the RLM Specialist. Your mission is to implement the
Recursive Language Model technique for code synthesis and decomposition.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- RLM decomposes problems into variables, constraints, and relationships
- Generates code through recursive specification refinement
- Includes self-debugging loop for error correction
- Must integrate with TRM for iterative refinement

Implementation guidelines:
1. Read existing technique patterns in ml_techniques/
2. Use TechniqueBase as parent class
3. Support both local execution and LLM backends
4. Handle Python code generation initially
5. Create sandboxed execution environment

Report completion with:
- Files created/modified
- Test results on sample specifications
- Any blockers or decisions needed
```

### Agent-C Spawn Prompt (Mamba)
```
You are Agent-C, the Mamba Specialist. Your mission is to implement
Mamba - a selective state space model with linear-time sequence modeling.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- Mamba uses selective state spaces with input-dependent parameters
- O(N) complexity vs O(N²) for attention
- Key components: discretization, selection, parallel scan
- Target: process 100K+ token sequences efficiently

Implementation guidelines:
1. Read the Mamba paper: arxiv.org/abs/2312.00752
2. Start with S4D (diagonal state spaces)
3. Implement both recurrent and convolutional modes
4. Use Triton for parallel scan (or pure PyTorch fallback)
5. Memory-efficient chunked processing for long sequences

Report completion with:
- Files created/modified
- Benchmark results (speed, memory)
- Comparison with sequential baseline
```

### Agent-D Spawn Prompt (Infrastructure)
```
You are Agent-D, the Infrastructure Specialist. Your mission is to create
shared components, testing framework, and ensure consistency across architectures.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- Multiple architectures need shared building blocks
- Testing must be comprehensive and reproducible
- Performance benchmarking is critical
- Code reuse reduces bugs and development time

Implementation guidelines:
1. Create modular, composable components
2. Use consistent APIs across all blocks
3. Write extensive unit tests
4. Document all public interfaces
5. Benchmark performance of each component

Report completion with:
- Files created/modified
- Test coverage report
- Performance benchmarks
```

### Agent-E Spawn Prompt (Data)
```
You are Agent-E, the Data Pipeline Specialist. Your mission is to validate
and extend the data pipeline for training code repair models.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- Canonical format: CanonicalCodeSample in Parquet
- Grid representation: 64×48 tokens
- Bug taxonomy: 5 categories, 100+ types
- Curriculum learning with 4 stages

Implementation guidelines:
1. Test all existing loaders thoroughly
2. Create Mamba-specific loader for long sequences
3. Build sample dataset from real code diffs
4. Ensure quality tier assignments are correct
5. Document data format specifications

Report completion with:
- Files created/modified
- Sample dataset statistics
- Loader performance benchmarks
```

### Agent-F Spawn Prompt (Integration)
```
You are Agent-F, the Integration Specialist. Your mission is to combine
TRM, RLM, and Mamba into a unified code repair pipeline.

Current task: [TASK_ID]
Files to modify: [FILES]
Deliverables: [DELIVERABLES]
Acceptance criteria: [CRITERIA]

Context:
- Pipeline: Mamba (context) → RLM (decompose) → TRM (refine)
- Orchestrator routes tasks to optimal architecture
- Must handle failures gracefully with fallbacks
- End-to-end from bug report to fixed code

Implementation guidelines:
1. Build on existing orchestrator infrastructure
2. Use unified index for capability matching
3. Create comprehensive integration tests
4. Benchmark against baseline approaches
5. Document the full pipeline architecture

Report completion with:
- Files created/modified
- End-to-end test results
- Performance comparisons
- User-facing documentation
```

---

## Success Metrics

### Phase 1 (Week 2 End)
- [ ] TRM forward pass produces valid output
- [ ] RLM extracts variables from 10 test specs
- [ ] Mamba SSM runs without errors
- [ ] Shared blocks have >90% test coverage

### Phase 2 (Week 4 End)
- [ ] TRM training loop runs for 1000 steps
- [ ] RLM generates syntactically valid Python
- [ ] Mamba selective mechanism varies with input
- [ ] Sample dataset with 1000 examples ready

### Phase 3 (Week 7 End)
- [ ] TRM + RLM composition improves accuracy
- [ ] Mamba parallel scan achieves 3x speedup
- [ ] Orchestrator routes tasks correctly
- [ ] Integration tests pass

### Phase 4 (Week 10 End)
- [ ] End-to-end pipeline: bug → fix in <5 seconds
- [ ] Code repair accuracy: >50% on test set
- [ ] Documentation enables new user onboarding
- [ ] Benchmarks published with reproducible results

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Mamba parallel scan complexity | Start with sequential fallback, optimize later |
| RLM code generation quality | Use established LLM backend (Claude API) |
| Integration complexity | Define clear interfaces early, test incrementally |
| Memory constraints | Implement gradient checkpointing, mixed precision |
| Dataset quality | Manual review of sample dataset, quality tiers |

---

## Execution Commands

### Start Phase 1 (4 parallel agents)
```bash
# Spawn all Phase 1 agents simultaneously
claude --agent TRM-001 --prompt "agents/trm_phase1.md"
claude --agent RLM-001 --prompt "agents/rlm_phase1.md"
claude --agent MAMBA-001 --prompt "agents/mamba_phase1.md"
claude --agent INFRA-001 --prompt "agents/infra_phase1.md"
```

### Monitor Progress
```bash
# Check agent status
claude --status all

# View agent outputs
claude --output TRM-001
claude --output RLM-001
```

### Validate Phase Completion
```bash
# Run phase validation tests
pytest modern_dev/tests/phase1/ -v
```

---

## File Organization

```
modern_dev/
├── trm/
│   └── src/
│       ├── model.py          # Agent-A: TRM-001, TRM-002
│       ├── layers.py         # Agent-A: TRM-001
│       ├── training.py       # Agent-A: TRM-003
│       ├── losses.py         # Agent-A: TRM-004
│       └── composition.py    # Agent-A: TRM-005
├── mamba_impl/
│   └── src/
│       ├── ssm.py            # Agent-C: MAMBA-001
│       ├── parameterization.py # Agent-C: MAMBA-002
│       ├── selective.py      # Agent-C: MAMBA-003
│       ├── block.py          # Agent-C: MAMBA-004
│       ├── scan.py           # Agent-C: MAMBA-005
│       └── model.py          # Agent-C: MAMBA-006
├── shared/
│   ├── blocks/               # Agent-D: INFRA-001
│   ├── losses/               # Agent-D: INFRA-003
│   ├── data/
│   │   └── loaders/
│   │       └── mamba.py      # Agent-E: DATA-002
│   └── pipelines/            # Agent-A: TRM-005, Agent-F: INTEG-002
├── orchestrator/
│   └── router.py             # Agent-F: INTEG-001
├── pipelines/                # Agent-F: INTEG-002
├── benchmarks/               # Agent-F: INTEG-003
├── tests/                    # Agent-D: INFRA-002, INFRA-004
└── docs/                     # Agent-F: INTEG-004

ml_techniques/
└── code_synthesis/
    ├── rlm.py                # Agent-B: RLM-001
    ├── variable_extractor.py # Agent-B: RLM-001
    ├── constraints.py        # Agent-B: RLM-002
    ├── generator.py          # Agent-B: RLM-003
    ├── debugger.py           # Agent-B: RLM-004
    └── pipeline.py           # Agent-B: RLM-005
```
