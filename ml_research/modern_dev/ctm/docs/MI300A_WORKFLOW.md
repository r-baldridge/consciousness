# CTM Code Repair - MI300A Workflow Specification

## System Configuration

### AMD Strix Halo (Max+395) Specifications
| Component | Specification |
|-----------|---------------|
| CPU | 24 Zen 5 cores (16P + 8E) |
| GPU | Radeon 8060S (40 RDNA 3.5 CUs) |
| Unified Memory | 128GB (96GB allocated to VRAM) |
| Memory Bandwidth | 256 GB/s |
| TDP | 120W configurable |

### Software Stack
```yaml
os: Ubuntu 24.04 LTS
python: 3.11
pytorch: 2.4+ (ROCm 6.2)
rocm: 6.2
compiler: hipcc
monitoring: rocm-smi, htop, nvtop-rocm
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CTM Code Repair - MI300A Workflow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA COLLECTION        PHASE 2: CTM IMPLEMENTATION                │
│  ══════════════════════          ══════════════════════════                 │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │ GitHub Collector    │         │ Core CTM Modules    │                   │
│  │ [CPU: 8, RAM: 16GB] │         │ [CPU: 4, RAM: 8GB]  │                   │
│  │ Duration: 24h       │         │ Duration: 8h        │                   │
│  └─────────┬───────────┘         └─────────┬───────────┘                   │
│            │                               │                                │
│  ┌─────────▼───────────┐         ┌─────────▼───────────┐                   │
│  │ Synthetic Generator │         │ Validation Tests    │                   │
│  │ [CPU: 12, RAM: 24GB]│         │ [GPU: 40%, VRAM:8GB]│                   │
│  │ Duration: 16h       │         │ Duration: 4h        │                   │
│  └─────────┬───────────┘         └─────────────────────┘                   │
│            │                                                                │
│  ┌─────────▼───────────┐                                                   │
│  │ Grid Encoding       │                                                   │
│  │ [CPU: 16, RAM: 32GB]│                                                   │
│  │ Duration: 8h        │                                                   │
│  └─────────────────────┘                                                   │
│                                                                             │
│  PHASE 3: CURRICULUM TRAINING    PHASE 4: EVALUATION                       │
│  ════════════════════════        ══════════════════                        │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │ Stage 1: Syntax     │         │ Benchmark Suite     │                   │
│  │ [GPU: 85%, VRAM:64GB│         │ [GPU: 80%, VRAM:48GB│                   │
│  │ Duration: 12h       │         │ Duration: 8h        │                   │
│  └─────────┬───────────┘         └─────────┬───────────┘                   │
│            │                               │                                │
│  ┌─────────▼───────────┐         ┌─────────▼───────────┐                   │
│  │ Stage 2: Logic      │         │ Sync Analysis       │                   │
│  │ [GPU: 90%, VRAM:72GB│         │ [GPU: 60%, VRAM:32GB│                   │
│  │ Duration: 18h       │         │ Duration: 6h        │                   │
│  └─────────┬───────────┘         └─────────────────────┘                   │
│            │                                                                │
│  ┌─────────▼───────────┐                                                   │
│  │ Stage 3: Complex    │                                                   │
│  │ [GPU: 95%, VRAM:80GB│                                                   │
│  │ Duration: 24h       │                                                   │
│  └─────────┬───────────┘                                                   │
│            │                                                                │
│  ┌─────────▼───────────┐                                                   │
│  │ Stage 4: Full       │                                                   │
│  │ [GPU: 95%, VRAM:88GB│                                                   │
│  │ Duration: 36h       │                                                   │
│  └─────────────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Collection & Processing

### Task 1.1: GitHub Data Collection
```yaml
task: github_collection
description: Collect Python bug fixes from GitHub commits
resources:
  cpu_cores: 8
  ram_gb: 16
  gpu_percent: 0
  vram_gb: 0
  disk_io: High (write)
  network: 50 Mbps sustained

duration: 24 hours
output: 500,000 commit pairs (~80GB raw)

checkpoints:
  - 6h: 125,000 commits collected
  - 12h: 250,000 commits collected
  - 18h: 375,000 commits collected
  - 24h: 500,000 commits collected

metrics:
  commits_per_hour: 20,833
  api_rate: GitHub API limits (5000/hr authenticated)
  retry_policy: exponential backoff
```

### Task 1.2: Synthetic Bug Generation
```yaml
task: synthetic_generation
description: Generate synthetic bug/fix pairs from clean code
resources:
  cpu_cores: 12
  ram_gb: 24
  gpu_percent: 0
  vram_gb: 0
  disk_io: Medium

duration: 16 hours
output: 400,000 synthetic pairs (~30GB)

checkpoints:
  - 4h: 100,000 pairs generated
  - 8h: 200,000 pairs generated
  - 12h: 300,000 pairs generated
  - 16h: 400,000 pairs generated

metrics:
  pairs_per_second: ~7
  mutation_success_rate: >60%
  validation_pass_rate: >80%
```

### Task 1.3: Stack Overflow Collection
```yaml
task: stackoverflow_collection
description: Collect code corrections from Stack Overflow
resources:
  cpu_cores: 4
  ram_gb: 8
  gpu_percent: 0
  vram_gb: 0
  network: 20 Mbps sustained

duration: 8 hours
output: 100,000 Q&A pairs (~15GB)

checkpoints:
  - 2h: 25,000 pairs
  - 4h: 50,000 pairs
  - 6h: 75,000 pairs
  - 8h: 100,000 pairs
```

### Task 1.4: Grid Encoding
```yaml
task: grid_encoding
description: Convert all pairs to 64x48 CTM grid format
resources:
  cpu_cores: 16
  ram_gb: 32
  gpu_percent: 0
  vram_gb: 0
  disk_io: Very High

duration: 8 hours
input: 1,000,000 text pairs
output: 1,000,000 grid pairs (~120GB numpy)

checkpoints:
  - 2h: 250,000 encoded
  - 4h: 500,000 encoded
  - 6h: 750,000 encoded
  - 8h: 1,000,000 encoded

metrics:
  encoding_rate: 35 pairs/second
  compression_ratio: ~0.4x (text → grid)
```

### Phase 1 Summary
| Task | Duration | CPU | RAM | GPU | VRAM | Output |
|------|----------|-----|-----|-----|------|--------|
| GitHub Collection | 24h | 8 | 16GB | 0% | 0 | 500K pairs |
| Synthetic Gen | 16h | 12 | 24GB | 0% | 0 | 400K pairs |
| Stack Overflow | 8h | 4 | 8GB | 0% | 0 | 100K pairs |
| Grid Encoding | 8h | 16 | 32GB | 0% | 0 | 1M grids |
| **Total** | **56h** | - | - | - | - | **1M samples** |

---

## Phase 2: CTM Implementation & Validation

### Task 2.1: Core Module Implementation
```yaml
task: ctm_implementation
description: Implement CTM architecture modules
resources:
  cpu_cores: 4
  ram_gb: 8
  gpu_percent: 0
  vram_gb: 0

duration: 8 hours (development time)

modules:
  - synapse.py: U-NET synapse network
  - nlm.py: Neuron-level models
  - synchronization.py: Sync computation
  - attention.py: Multi-head attention
  - backbone.py: Code encoder
  - model.py: Main CTM class

checkpoints:
  - 2h: Synapse + NLM modules
  - 4h: Synchronization + Attention
  - 6h: Backbone + Integration
  - 8h: Full model assembled
```

### Task 2.2: Model Validation
```yaml
task: model_validation
description: Validate CTM implementation correctness
resources:
  cpu_cores: 8
  ram_gb: 16
  gpu_percent: 40
  vram_gb: 8

duration: 4 hours

tests:
  - shape_validation: All tensor shapes correct
  - gradient_flow: Gradients flow through all paths
  - memory_profile: VRAM usage within bounds
  - iteration_test: Output improves with more iterations
  - synchronization_test: Sync values are meaningful

checkpoints:
  - 1h: Unit tests pass
  - 2h: Integration tests pass
  - 3h: Gradient flow verified
  - 4h: Memory profile validated
```

### Task 2.3: Baseline Benchmark
```yaml
task: baseline_benchmark
description: Establish baseline metrics before training
resources:
  cpu_cores: 8
  ram_gb: 16
  gpu_percent: 60
  vram_gb: 16

duration: 2 hours

metrics_to_capture:
  - random_init_accuracy: Expected ~0.001%
  - forward_pass_time: Target <500ms per sample
  - memory_per_sample: Target <200MB at batch=1
  - iteration_scaling: Time vs iterations
```

### Phase 2 Summary
| Task | Duration | CPU | RAM | GPU | VRAM |
|------|----------|-----|-----|-----|------|
| Implementation | 8h | 4 | 8GB | 0% | 0 |
| Validation | 4h | 8 | 16GB | 40% | 8GB |
| Baseline | 2h | 8 | 16GB | 60% | 16GB |
| **Total** | **14h** | - | - | - | - |

---

## Phase 3: Curriculum Training

### CTM-Specific Training Considerations

**Key Differences from TRM:**
1. **Sequential Iterations**: T=64 iterations cannot be parallelized
2. **State Traces**: Memory grows with iteration count (M=32 history)
3. **NLM Overhead**: Per-neuron MLPs add compute per iteration
4. **Synchronization**: Pairwise computations scale O(n²)

**Memory Formula:**
```
VRAM = batch_size × (
    model_params +                    # ~15M × 4 bytes = 60MB
    state_trace × iterations +        # B × D × M × T × 4 bytes
    activated_state +                 # B × D × 4 bytes
    sync_accumulators +               # B × n_sync × 4 bytes
    attention_cache +                 # Variable
    gradients                         # ~2× model params
)
```

### Stage 1: Syntax Bugs (16 iterations)
```yaml
task: stage_1_training
description: Train on syntax errors with reduced iterations
config:
  iterations: 16
  memory_length: 16
  batch_size: 64
  gradient_accumulation: 2
  effective_batch: 128
  learning_rate: 3e-4
  epochs: 2

resources:
  cpu_cores: 12
  ram_gb: 32
  gpu_percent: 85
  vram_gb: 64

duration: 12 hours
samples: 200,000

checkpoints:
  - 3h: 50K samples, loss < 4.0
  - 6h: 100K samples, loss < 3.0
  - 9h: 150K samples, loss < 2.5
  - 12h: 200K samples, syntax accuracy > 80%

expected_metrics:
  throughput: 4.6 samples/second
  loss_curve: Steep initial descent
  syntax_accuracy: >85%
```

### Stage 2: Logic Bugs (32 iterations)
```yaml
task: stage_2_training
description: Add simple logic bugs, increase iterations
config:
  iterations: 32
  memory_length: 24
  batch_size: 48
  gradient_accumulation: 3
  effective_batch: 144
  learning_rate: 2e-4
  epochs: 2

resources:
  cpu_cores: 12
  ram_gb: 48
  gpu_percent: 90
  vram_gb: 72

duration: 18 hours
samples: 300,000

checkpoints:
  - 4.5h: 75K samples, syntax accuracy maintained
  - 9h: 150K samples, logic accuracy > 40%
  - 13.5h: 225K samples, logic accuracy > 55%
  - 18h: 300K samples, logic accuracy > 65%

expected_metrics:
  throughput: 2.8 samples/second
  iteration_efficiency: Accuracy improves with more iterations
  certainty_calibration: Certainty correlates with accuracy
```

### Stage 3: Complex Bugs (48 iterations)
```yaml
task: stage_3_training
description: Full bug taxonomy, near-full iterations
config:
  iterations: 48
  memory_length: 28
  batch_size: 32
  gradient_accumulation: 4
  effective_batch: 128
  learning_rate: 1e-4
  epochs: 2

resources:
  cpu_cores: 16
  ram_gb: 64
  gpu_percent: 95
  vram_gb: 80

duration: 24 hours
samples: 500,000

checkpoints:
  - 6h: 125K samples, overall accuracy > 50%
  - 12h: 250K samples, overall accuracy > 60%
  - 18h: 375K samples, overall accuracy > 68%
  - 24h: 500K samples, overall accuracy > 72%

expected_metrics:
  throughput: 1.8 samples/second
  sync_patterns: Distinct patterns for bug types
  early_stopping_potential: Can stop at T=32 for simple bugs
```

### Stage 4: Full Training (64 iterations)
```yaml
task: stage_4_training
description: Full dataset, full iterations, fine-tuning
config:
  iterations: 64
  memory_length: 32
  batch_size: 24
  gradient_accumulation: 6
  effective_batch: 144
  learning_rate: 5e-5
  epochs: 3

resources:
  cpu_cores: 16
  ram_gb: 80
  gpu_percent: 95
  vram_gb: 88

duration: 36 hours
samples: 1,000,000

checkpoints:
  - 9h: 250K samples, overall accuracy > 70%
  - 18h: 500K samples, overall accuracy > 73%
  - 27h: 750K samples, overall accuracy > 75%
  - 36h: 1M samples, overall accuracy > 76%

expected_metrics:
  throughput: 1.2 samples/second
  final_syntax_accuracy: >93%
  final_logic_accuracy: >72%
  final_security_accuracy: >78%
  certainty_ece: <0.05 (well calibrated)
```

### Phase 3 Summary
| Stage | Iterations | Batch | Duration | VRAM | Target Accuracy |
|-------|------------|-------|----------|------|-----------------|
| Stage 1 | 16 | 64 | 12h | 64GB | >85% syntax |
| Stage 2 | 32 | 48 | 18h | 72GB | >65% logic |
| Stage 3 | 48 | 32 | 24h | 80GB | >72% overall |
| Stage 4 | 64 | 24 | 36h | 88GB | >76% overall |
| **Total** | - | - | **90h** | - | - |

---

## Phase 4: Evaluation & Analysis

### Task 4.1: Benchmark Suite
```yaml
task: benchmark_evaluation
description: Run full benchmark suite on trained model
resources:
  cpu_cores: 8
  ram_gb: 32
  gpu_percent: 80
  vram_gb: 48

duration: 8 hours

benchmarks:
  syntax_benchmark:
    samples: 5,000
    expected_accuracy: >93%

  logic_benchmark:
    samples: 5,000
    expected_accuracy: >72%

  security_benchmark:
    samples: 2,000
    expected_accuracy: >78%

  iteration_scaling:
    samples: 1,000
    iterations: [8, 16, 32, 48, 64]
    measure: accuracy_vs_compute

  trm_comparison:
    samples: 5,000
    compare: accuracy, calibration, speed

checkpoints:
  - 2h: Syntax benchmark complete
  - 4h: Logic benchmark complete
  - 6h: Security + scaling complete
  - 8h: All benchmarks + comparisons complete
```

### Task 4.2: Synchronization Analysis
```yaml
task: sync_analysis
description: Analyze neural synchronization patterns
resources:
  cpu_cores: 8
  ram_gb: 24
  gpu_percent: 60
  vram_gb: 32

duration: 6 hours

analyses:
  - sync_vs_bug_type: Distinct patterns per bug category
  - sync_over_time: How sync evolves across iterations
  - sync_vs_accuracy: Correlation with prediction quality
  - neuron_pair_importance: Which pairs are most informative
  - interpretability: Can sync identify bug location?

outputs:
  - sync_heatmaps.png
  - iteration_progression.gif
  - neuron_importance.csv
  - sync_interpretation_report.md
```

### Task 4.3: Iteration Efficiency Analysis
```yaml
task: iteration_analysis
description: Analyze adaptive computation potential
resources:
  cpu_cores: 8
  ram_gb: 16
  gpu_percent: 60
  vram_gb: 24

duration: 4 hours

analyses:
  - accuracy_by_iteration: Per-iteration accuracy curves
  - certainty_by_iteration: When model becomes confident
  - early_exit_potential: Could stop early for easy bugs
  - compute_vs_difficulty: Harder bugs need more iterations

expected_findings:
  - Syntax bugs: Solved by iteration 16
  - Logic bugs: Need 32-48 iterations
  - Complex bugs: Benefit from full 64 iterations
```

### Task 4.4: Model Export
```yaml
task: model_export
description: Export final model for deployment
resources:
  cpu_cores: 4
  ram_gb: 16
  gpu_percent: 40
  vram_gb: 16

duration: 2 hours

exports:
  - pytorch_checkpoint: ctm_code_repair_final.pt
  - onnx_export: ctm_code_repair.onnx
  - config: final_config.yaml
  - tokenizer: tokenizer.json
  - metrics: final_metrics.json
```

### Phase 4 Summary
| Task | Duration | CPU | RAM | GPU | VRAM |
|------|----------|-----|-----|-----|------|
| Benchmarks | 8h | 8 | 32GB | 80% | 48GB |
| Sync Analysis | 6h | 8 | 24GB | 60% | 32GB |
| Iteration Analysis | 4h | 8 | 16GB | 60% | 24GB |
| Model Export | 2h | 4 | 16GB | 40% | 16GB |
| **Total** | **20h** | - | - | - | - |

---

## Complete Timeline

```
Day 1-2:   Phase 1A - GitHub Collection (24h)
           ├── Concurrent: SO Collection (8h)
           └── Concurrent: Synthetic Gen starts

Day 3:     Phase 1B - Synthetic Generation completes (16h)
           └── Grid Encoding starts

Day 4:     Phase 1C - Grid Encoding completes (8h)
           Phase 2 - Implementation & Validation (14h)

Day 5:     Phase 3A - Stage 1 Training (12h)

Day 6:     Phase 3B - Stage 2 Training (18h)

Day 7-8:   Phase 3C - Stage 3 Training (24h)

Day 9-10:  Phase 3D - Stage 4 Training (36h)

Day 11:    Phase 4 - Evaluation & Analysis (20h)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~180 hours = 7.5 days continuous
       With breaks/monitoring: ~10-11 days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Progress Tracking Dashboard

### Daily Metrics Template

```markdown
## CTM Training Progress - Day [X]

### Current Phase: [Phase X - Task Name]

#### Resource Utilization
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| GPU Utilization | __% | __% | ✓/✗ |
| VRAM Usage | __GB | __GB | ✓/✗ |
| CPU Utilization | __% | __% | ✓/✗ |
| RAM Usage | __GB | __GB | ✓/✗ |
| Disk I/O | __MB/s | __MB/s | ✓/✗ |

#### Training Metrics (if applicable)
| Metric | Current | Checkpoint Target | Status |
|--------|---------|-------------------|--------|
| Samples Processed | __ | __ | ✓/✗ |
| Training Loss | __ | <__ | ✓/✗ |
| Validation Accuracy | __% | >__% | ✓/✗ |
| Throughput | __/s | __/s | ✓/✗ |
| Iterations/sample | __ | __ | ✓/✗ |

#### CTM-Specific Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Sync Variance | __ | Higher = more differentiated |
| Certainty Mean | __ | Should increase over training |
| Iteration Efficiency | __ | Accuracy gain per iteration |
| NLM Activation Stats | __ | Distribution check |

#### Issues/Notes
- [Any issues encountered]
- [Adjustments made]
- [Observations]

#### Next Checkpoint
- Target: [description]
- ETA: [time]
```

---

## CTM vs TRM: Training Comparison

| Aspect | CTM | TRM |
|--------|-----|-----|
| **Model Size** | ~15M params | ~7M params |
| **Iterations** | 64 (sequential) | 8 (recursive) |
| **Memory Pattern** | State traces grow | Fixed recursion |
| **Max Batch Size** | 24 @ 64 iters | 64 |
| **Training Time** | ~90h | ~50h |
| **Total Workflow** | ~180h | ~200h |
| **VRAM Peak** | 88GB | 72GB |
| **Throughput** | 1.2 samples/s | 3.5 samples/s |
| **Expected Accuracy** | >76% | >75% |

---

## Risk Mitigation

### Memory Overflow
- **Risk**: VRAM exhaustion at high iterations
- **Mitigation**: Gradient checkpointing, reduce batch dynamically
- **Fallback**: Reduce iterations to 48 for full training

### Training Instability
- **Risk**: CTM's complex dynamics may be unstable
- **Mitigation**: Lower learning rate, gradient clipping at 0.5
- **Fallback**: Simpler NLM architecture (linear variant)

### Slow Convergence
- **Risk**: Sequential iterations slow training
- **Mitigation**: Curriculum learning (fewer iterations early)
- **Fallback**: Accept lower accuracy, optimize for speed

### Sync Collapse
- **Risk**: All sync values become uniform
- **Mitigation**: Sync diversity loss term
- **Fallback**: Reduce n_synch, use different pairing strategy

---

## Cost Estimate

### Power Consumption
| Phase | Duration | Avg Power | Energy |
|-------|----------|-----------|--------|
| Phase 1 | 56h | 80W | 4.5 kWh |
| Phase 2 | 14h | 90W | 1.3 kWh |
| Phase 3 | 90h | 115W | 10.4 kWh |
| Phase 4 | 20h | 100W | 2.0 kWh |
| **Total** | **180h** | - | **18.2 kWh** |

### Cost at $0.15/kWh: **~$3**

### Storage Requirements
| Component | Size |
|-----------|------|
| Raw data | 125 GB |
| Encoded grids | 120 GB |
| Checkpoints (4 stages × 5 each) | 30 GB |
| Final model | 60 MB |
| Logs/metrics | 5 GB |
| **Total** | **~280 GB** |

---

## Success Criteria

### Minimum Viable
- [ ] CTM trains without memory errors
- [ ] Accuracy improves with more iterations
- [ ] Final accuracy matches TRM baseline (>75%)

### Target
- [ ] Exceeds TRM on complex logic bugs (>72%)
- [ ] Good certainty calibration (ECE < 0.05)
- [ ] Meaningful sync patterns by bug type

### Stretch
- [ ] State-of-the-art code repair accuracy
- [ ] Interpretable sync → bug location mapping
- [ ] Efficient early stopping (50% compute savings on easy bugs)
