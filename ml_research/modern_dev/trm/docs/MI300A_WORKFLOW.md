# TRM Code Repair - AMD MI300A Single-System Workflow

## System Specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AMD Instinct MI300A (Max+ 395)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU Compute                          GPU Compute                           │
│  ┌─────────────────────────┐          ┌─────────────────────────┐          │
│  │ 24× Zen 4 cores         │          │ 228 Compute Units       │          │
│  │ Up to 3.7 GHz           │          │ CDNA 3 Architecture     │          │
│  │ 96 threads (SMT)        │          │ 14,592 stream processors│          │
│  └─────────────────────────┘          └─────────────────────────┘          │
│                                                                             │
│  Unified Memory: 96 GB HBM3                                                 │
│  Memory Bandwidth: 5.3 TB/s                                                 │
│  Peak FP16: ~1.3 PFLOPS                                                     │
│  Peak FP32: ~653 TFLOPS                                                     │
│  TDP: 760W                                                                  │
│                                                                             │
│  Software Stack:                                                            │
│  ├── ROCm 6.x                                                               │
│  ├── PyTorch 2.x (ROCm build)                                               │
│  ├── Python 3.11+                                                           │
│  └── Docker (for sandboxed validation)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  PHASE 1: DATA COLLECTION                         Days 1-5            ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  [1.1] GitHub Mining ─────────────────────────────────── 100 hrs     ║ │
│  ║        └─ CPU: 8 cores │ RAM: 16GB │ GPU: 0% │ Net: 2.5GB/hr        ║ │
│  ║                                                                       ║ │
│  ║  [1.2] Stack Overflow ─────────────────────────────────── 7 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 64GB │ GPU: 0% │ Net: 18GB download   ║ │
│  ║                                                                       ║ │
│  ║  [1.3] Synthetic Generation ───────────────────────────── 8 hrs      ║ │
│  ║        └─ CPU: 24 cores │ RAM: 32GB │ GPU: 0% │ Net: 0              ║ │
│  ║                                                                       ║ │
│  ║  [1.4] Linter Execution ──────────────────────────────── 24 hrs      ║ │
│  ║        └─ CPU: 24 cores │ RAM: 48GB │ GPU: 0% │ Net: NAS I/O        ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              ↓                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  PHASE 2: DATA PROCESSING                         Days 5-7            ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  [2.1] Vocabulary Building ────────────────────────────── 4 hrs      ║ │
│  ║        └─ CPU: 16 cores │ RAM: 64GB │ GPU: 0%                        ║ │
│  ║                                                                       ║ │
│  ║  [2.2] Tokenization ───────────────────────────────────── 6 hrs      ║ │
│  ║        └─ CPU: 24 cores │ RAM: 32GB │ GPU: 0%                        ║ │
│  ║                                                                       ║ │
│  ║  [2.3] Validation ─────────────────────────────────────── 18 hrs     ║ │
│  ║        └─ CPU: 24 cores │ RAM: 64GB │ GPU: 0% │ Docker active       ║ │
│  ║                                                                       ║ │
│  ║  [2.4] Grid Encoding ──────────────────────────────────── 1 hr       ║ │
│  ║        └─ CPU: 4 cores │ RAM: 16GB │ GPU: 50% │ 32GB VRAM           ║ │
│  ║                                                                       ║ │
│  ║  [2.5] Dataset Finalization ───────────────────────────── 2 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 96GB │ GPU: 0%                         ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              ↓                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  PHASE 3: MODEL TRAINING                          Days 7-9            ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  [3.1] Stage 1: Syntax ────────────────────────────────── 4 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90% │ 72GB VRAM           ║ │
│  ║                                                                       ║ │
│  ║  [3.2] Stage 2: Simple Logic ──────────────────────────── 5 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90% │ 72GB VRAM           ║ │
│  ║                                                                       ║ │
│  ║  [3.3] Stage 3: Complex Logic ─────────────────────────── 6 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90% │ 72GB VRAM           ║ │
│  ║                                                                       ║ │
│  ║  [3.4] Stage 4: Advanced ──────────────────────────────── 7 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90% │ 72GB VRAM           ║ │
│  ║                                                                       ║ │
│  ║  [3.5] Stage 5: Full Dataset ──────────────────────────── 10 hrs     ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90% │ 72GB VRAM           ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              ↓                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  PHASE 4: EVALUATION & DEPLOYMENT                 Days 9-10           ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  [4.1] Benchmark Evaluation ───────────────────────────── 2 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 16GB │ GPU: 40%                        ║ │
│  ║                                                                       ║ │
│  ║  [4.2] Error Analysis ─────────────────────────────────── 4 hrs      ║ │
│  ║        └─ CPU: 16 cores │ RAM: 32GB │ GPU: 10%                       ║ │
│  ║                                                                       ║ │
│  ║  [4.3] Fine-tuning (if needed) ────────────────────────── 4 hrs      ║ │
│  ║        └─ CPU: 8 cores │ RAM: 32GB │ GPU: 90%                        ║ │
│  ║                                                                       ║ │
│  ║  [4.4] Final Export ───────────────────────────────────── 1 hr       ║ │
│  ║        └─ CPU: 4 cores │ RAM: 8GB │ GPU: 0%                          ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase Specifications

### PHASE 1: Data Collection (Days 1-5)

#### Task 1.1: GitHub Commit Mining

```yaml
task_id: P1.1
name: GitHub Bug-Fix Commit Mining
target_output: 500,000 bug-fix pairs

schedule:
  start: Day 1, Hour 0
  end: Day 5, Hour 4
  duration: 100 hours
  runs_parallel_with: [P1.2, P1.3] # After initial setup

resource_allocation:
  cpu_cores: 8 (of 24)
  cpu_utilization: 30-40%
  ram_allocated: 16 GB
  ram_utilization: 15%
  gpu_utilization: 0%
  vram_used: 0 GB

network_io:
  github_api_requests: 5,000/hour (with token)
  download_rate: 2.5 GB/hour average
  total_download: ~250 GB
  upload: negligible

storage_io:
  write_rate: 2.5 GB/hour
  total_written: 250 GB raw data
  destination: NAS or local NVMe

bottleneck: GitHub API rate limits (not hardware)

progress_checkpoints:
  hour_10:
    repos_discovered: 5,000
    commits_analyzed: 50,000
    pairs_extracted: 25,000
  hour_25:
    repos_discovered: 10,000
    commits_analyzed: 200,000
    pairs_extracted: 100,000
  hour_50:
    commits_analyzed: 500,000
    pairs_extracted: 250,000
  hour_75:
    commits_analyzed: 800,000
    pairs_extracted: 400,000
  hour_100:
    commits_analyzed: 1,000,000
    pairs_extracted: 500,000

success_criteria:
  - 500,000+ raw pairs collected
  - <5% API errors
  - All priority repos processed
```

#### Task 1.2: Stack Overflow Processing

```yaml
task_id: P1.2
name: Stack Overflow Data Extraction
target_output: 100,000 Q&A pairs

schedule:
  start: Day 1, Hour 1
  end: Day 1, Hour 8
  duration: 7 hours
  runs_parallel_with: [P1.1]

resource_allocation:
  cpu_cores: 8 (of 24)
  cpu_utilization: 60-80%
  ram_allocated: 64 GB (XML parsing needs memory)
  ram_utilization: 65%
  gpu_utilization: 0%
  vram_used: 0 GB

network_io:
  download: 18 GB (compressed dump)
  download_time: 3 minutes @ 100 MB/s
  upload: 0

storage_io:
  temp_space: 100 GB (decompressed XML)
  output_size: 15 GB (extracted pairs)

subtasks:
  download:
    duration: 5 minutes
    progress: bytes downloaded / 18 GB

  extraction:
    duration: 30 minutes
    progress: bytes extracted / 90 GB

  xml_parsing:
    duration: 4 hours
    progress: posts parsed / 50M total
    rate: ~3,500 posts/second

  filtering:
    duration: 2 hours
    progress: python posts found
    expected: ~2M python posts

  pair_matching:
    duration: 30 minutes
    progress: pairs matched / target
    expected: 100,000 pairs

progress_checkpoints:
  hour_1: Download complete, extraction started
  hour_2: Extraction complete, parsing at 25%
  hour_4: Parsing complete, filtering started
  hour_6: Filtering complete, matching started
  hour_7: Complete, 100K pairs extracted

success_criteria:
  - 100,000+ Q&A pairs
  - Code blocks present in both Q and A
  - Answer score >= 5
```

#### Task 1.3: Synthetic Bug Generation

```yaml
task_id: P1.3
name: Synthetic Bug Generation
target_output: 500,000 synthetic pairs

schedule:
  start: Day 1, Hour 2
  end: Day 1, Hour 10
  duration: 8 hours
  runs_parallel_with: [P1.1, P1.2]

resource_allocation:
  cpu_cores: 24 (all cores)
  cpu_utilization: 95%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 0%
  vram_used: 0 GB

network_io:
  download: 0 (uses local code corpus)
  upload: 0

storage_io:
  input: 10 GB (clean code corpus)
  output: 50 GB (buggy/fixed pairs)
  write_rate: 6 GB/hour

processing_rate:
  pairs_per_minute: 1,000
  pairs_per_hour: 60,000

subtasks:
  corpus_loading:
    duration: 15 minutes
    files_loaded: 100,000 Python files

  mutation_generation:
    duration: 7 hours
    progress: pairs generated / 500,000

  serialization:
    duration: 45 minutes
    progress: pairs written to disk

progress_checkpoints:
  hour_1: 60,000 pairs generated
  hour_2: 120,000 pairs generated
  hour_4: 240,000 pairs generated
  hour_6: 360,000 pairs generated
  hour_8: 500,000 pairs generated

bug_type_distribution:
  syntax_errors: 150,000 (30%)
  logic_errors: 200,000 (40%)
  style_issues: 100,000 (20%)
  security_bugs: 50,000 (10%)

success_criteria:
  - 500,000 pairs generated
  - All 12+ mutation types represented
  - <1% generation failures
```

#### Task 1.4: Linter Execution

```yaml
task_id: P1.4
name: Static Analysis & Linter Execution
target_output: 300,000 linter-derived pairs

schedule:
  start: Day 2, Hour 0
  end: Day 3, Hour 0
  duration: 24 hours
  depends_on: [P1.1 partial] # Needs some repos cloned

resource_allocation:
  cpu_cores: 24 (all cores)
  cpu_utilization: 90%
  ram_allocated: 48 GB
  ram_utilization: 50%
  gpu_utilization: 0%
  vram_used: 0 GB

network_io:
  from_nas: 200 GB (source files)
  to_nas: 30 GB (linter outputs)

tools_executed:
  pylint:
    files_per_second: 0.5
    coverage: all files
  mypy:
    files_per_second: 1.0
    coverage: typed files
  ruff:
    files_per_second: 10.0
    coverage: all files
  bandit:
    files_per_second: 2.0
    coverage: all files

subtasks:
  file_discovery:
    duration: 1 hour
    files_found: 500,000

  pylint_pass:
    duration: 12 hours
    progress: files_processed / 500,000

  mypy_pass:
    duration: 6 hours
    progress: files_processed / 500,000

  ruff_pass:
    duration: 1 hour
    progress: files_processed / 500,000

  bandit_pass:
    duration: 3 hours
    progress: files_processed / 500,000

  issue_to_pair_mapping:
    duration: 1 hour
    progress: pairs_created / 300,000

progress_checkpoints:
  hour_6: pylint 50%, mypy started
  hour_12: pylint complete, mypy 50%
  hour_18: mypy complete, ruff/bandit running
  hour_24: all linters complete, pairs mapped

success_criteria:
  - 300,000 linter-derived pairs
  - Coverage of 50+ linter rules
  - Autofix applied where available
```

---

### PHASE 2: Data Processing (Days 5-7)

#### Task 2.1: Vocabulary Building

```yaml
task_id: P2.1
name: BPE Vocabulary Training
target_output: 32K token vocabulary

schedule:
  start: Day 5, Hour 4
  end: Day 5, Hour 8
  duration: 4 hours
  depends_on: [P1.1, P1.2, P1.3, P1.4]

resource_allocation:
  cpu_cores: 16
  cpu_utilization: 85%
  ram_allocated: 64 GB
  ram_utilization: 60%
  gpu_utilization: 0%
  vram_used: 0 GB

input:
  total_samples: 1,400,000 raw pairs
  total_code_size: ~50 GB text

processing:
  algorithm: BPE (Byte Pair Encoding)
  vocab_size: 32,768
  min_frequency: 100

subtasks:
  corpus_preparation:
    duration: 30 minutes
    progress: samples loaded

  frequency_counting:
    duration: 1 hour
    progress: tokens counted

  merge_learning:
    duration: 2 hours
    progress: merges learned / 32,000

  vocab_serialization:
    duration: 30 minutes
    output_files:
      - vocab.json (32K entries)
      - merges.txt (32K merges)
      - token_frequencies.json

progress_checkpoints:
  hour_1: Corpus loaded, counting complete
  hour_2: 10,000 merges learned
  hour_3: 25,000 merges learned
  hour_4: Complete, vocab saved

success_criteria:
  - 32,768 tokens in vocabulary
  - All Python keywords included
  - OOV rate < 5% on test set
```

#### Task 2.2: Tokenization

```yaml
task_id: P2.2
name: Full Corpus Tokenization
target_output: 1.4M tokenized samples

schedule:
  start: Day 5, Hour 8
  end: Day 5, Hour 14
  duration: 6 hours
  depends_on: [P2.1]

resource_allocation:
  cpu_cores: 24
  cpu_utilization: 90%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 0%
  vram_used: 0 GB

processing:
  samples_per_second: 65
  samples_per_hour: 234,000

input:
  raw_pairs: 1,400,000
  avg_tokens_per_sample: 500

output:
  tokenized_shards: 140 files
  shard_size: 10,000 samples each
  total_size: ~80 GB

subtasks:
  shard_0_to_50:
    duration: 2.1 hours
    samples: 500,000

  shard_51_to_100:
    duration: 2.1 hours
    samples: 500,000

  shard_101_to_140:
    duration: 1.8 hours
    samples: 400,000

progress_checkpoints:
  hour_1: 234,000 samples (17%)
  hour_2: 468,000 samples (33%)
  hour_3: 702,000 samples (50%)
  hour_4: 936,000 samples (67%)
  hour_5: 1,170,000 samples (84%)
  hour_6: 1,400,000 samples (100%)

success_criteria:
  - All samples tokenized
  - No tokenization errors
  - Shards balanced by size
```

#### Task 2.3: Validation Pipeline

```yaml
task_id: P2.3
name: Quality Validation
target_output: 1.2M validated samples

schedule:
  start: Day 5, Hour 14
  end: Day 6, Hour 8
  duration: 18 hours
  depends_on: [P2.2]

resource_allocation:
  cpu_cores: 24
  cpu_utilization: 85%
  ram_allocated: 64 GB
  ram_utilization: 60%
  gpu_utilization: 0%
  vram_used: 0 GB
  docker_containers: 50 concurrent

validation_stages:
  stage_1_syntax:
    check: Both buggy and fixed parse correctly
    duration: 2 hours
    rate: 700,000 samples/hour
    expected_pass: 98%

  stage_2_diff_check:
    check: Meaningful difference exists
    duration: 3 hours
    rate: 460,000 samples/hour
    expected_pass: 95%

  stage_3_deduplication:
    check: Not a near-duplicate
    duration: 4 hours
    rate: 350,000 samples/hour
    expected_pass: 90%

  stage_4_test_execution:
    check: Fixed code passes tests (subset)
    samples_tested: 200,000
    duration: 8 hours
    rate: 25,000 samples/hour
    expected_pass: 85%

  stage_5_final_filter:
    check: Meets all quality thresholds
    duration: 1 hour
    rate: 1,200,000 samples/hour

expected_attrition:
  input: 1,400,000
  after_syntax: 1,372,000 (-2%)
  after_diff: 1,303,000 (-5%)
  after_dedup: 1,173,000 (-10%)
  after_tests: ~1,200,000 (extrapolated)
  final: 1,200,000

progress_checkpoints:
  hour_2: Syntax validation complete
  hour_5: Diff check complete
  hour_9: Deduplication complete
  hour_17: Test execution complete
  hour_18: Final filtering complete

success_criteria:
  - 1,200,000+ validated samples
  - Validation rate > 85%
  - All bug types represented
```

#### Task 2.4: Grid Encoding

```yaml
task_id: P2.4
name: Code-to-Grid Encoding
target_output: 1.2M encoded grids

schedule:
  start: Day 6, Hour 8
  end: Day 6, Hour 9
  duration: 1 hour
  depends_on: [P2.3]

resource_allocation:
  cpu_cores: 4 (data loading)
  cpu_utilization: 15%
  ram_allocated: 16 GB
  ram_utilization: 15%
  gpu_utilization: 50%
  vram_used: 32 GB

processing:
  batch_size: 4096
  batches_total: 293
  time_per_batch: 12 seconds
  samples_per_second: 340

gpu_operations:
  - Token ID lookup (vectorized)
  - Grid padding/masking
  - Compression preparation

output:
  grid_shape: (64, 48) per sample
  dtype: int16
  total_size: 14 GB compressed

progress_checkpoints:
  minute_15: 300,000 samples (25%)
  minute_30: 600,000 samples (50%)
  minute_45: 900,000 samples (75%)
  minute_60: 1,200,000 samples (100%)

success_criteria:
  - All samples encoded
  - Grid shape consistent
  - No encoding errors
```

#### Task 2.5: Dataset Finalization

```yaml
task_id: P2.5
name: Dataset Split & Shuffle
target_output: Train/Val/Test splits

schedule:
  start: Day 6, Hour 9
  end: Day 6, Hour 11
  duration: 2 hours
  depends_on: [P2.4]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 40%
  ram_allocated: 96 GB (full dataset in memory)
  ram_utilization: 95%
  gpu_utilization: 0%
  vram_used: 0 GB

operations:
  load_all_shards:
    duration: 20 minutes
    memory_peak: 90 GB

  global_shuffle:
    duration: 10 minutes
    algorithm: Fisher-Yates

  stratified_split:
    duration: 10 minutes
    train_ratio: 0.90
    val_ratio: 0.05
    test_ratio: 0.05

  difficulty_scoring:
    duration: 30 minutes
    scores_computed: 1,200,000

  curriculum_bucketing:
    duration: 20 minutes
    buckets: 5 difficulty levels

  write_final_shards:
    duration: 30 minutes
    output_format: Parquet

output:
  train_samples: 1,080,000
  val_samples: 60,000
  test_samples: 60,000

  curriculum_distribution:
    level_1_easy: 216,000 (20%)
    level_2: 270,000 (25%)
    level_3: 270,000 (25%)
    level_4: 216,000 (20%)
    level_5_hard: 108,000 (10%)

progress_checkpoints:
  minute_20: Data loaded
  minute_30: Shuffle complete
  minute_40: Split complete
  minute_70: Difficulty scored
  minute_90: Curriculum ready
  minute_120: All files written

success_criteria:
  - 90/5/5 split maintained
  - Bug types balanced in each split
  - Curriculum levels populated
```

---

### PHASE 3: Model Training (Days 7-9)

#### Training Configuration

```yaml
model_config:
  name: TRM-CodeRepair-7M
  architecture:
    d_model: 256
    n_heads: 8
    d_ff: 1024
    n_layers: 2
    n_recursive_steps: 21
    grid_height: 64
    grid_width: 48
    vocab_size: 32768

  total_parameters: 7,340,032

mi300a_training_config:
  precision: bfloat16
  batch_size: 768
  gradient_accumulation: 2
  effective_batch_size: 1536

  optimizer:
    name: AdamW
    learning_rate: 3e-4 (stage 1)
    weight_decay: 0.01
    betas: [0.9, 0.95]

  scheduler:
    name: CosineAnnealingWarmRestarts
    warmup_steps: 500

memory_usage:
  model_params: 14 MB (BF16)
  gradients: 14 MB
  optimizer_states: 56 MB
  activations: 55 GB (batch=768, recursive)
  data_buffers: 4 GB
  rocm_overhead: 6 GB
  total: ~80 GB of 96 GB (83%)
```

#### Task 3.1: Stage 1 - Syntax Errors

```yaml
task_id: P3.1
name: Curriculum Stage 1 - Syntax
target: Learn basic syntax error patterns

schedule:
  start: Day 7, Hour 0
  end: Day 7, Hour 4
  duration: 4 hours
  depends_on: [P2.5]

resource_allocation:
  cpu_cores: 8 (data loading)
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

training_params:
  samples: 150,000 (difficulty 0.0-0.2)
  epochs: 2
  steps_per_epoch: 98
  total_steps: 196
  learning_rate: 3e-4

bug_types_included:
  - missing_colon
  - missing_parenthesis
  - indentation_error
  - missing_comma
  - missing_bracket

expected_metrics:
  initial_loss: ~8.0
  final_loss: ~2.5
  syntax_accuracy: 75%+

progress_checkpoints:
  step_50: loss < 5.0
  step_100: loss < 3.5
  step_150: loss < 3.0
  step_196: loss < 2.5, accuracy > 75%

checkpoint_saves:
  - step_100
  - step_196 (stage complete)

success_criteria:
  - Loss < 2.5
  - Syntax bug accuracy > 75%
  - No training instability
```

#### Task 3.2: Stage 2 - Simple Logic

```yaml
task_id: P3.2
name: Curriculum Stage 2 - Simple Logic
target: Learn basic logic error patterns

schedule:
  start: Day 7, Hour 4
  end: Day 7, Hour 9
  duration: 5 hours
  depends_on: [P3.1]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

training_params:
  samples: 250,000 (difficulty 0.0-0.4)
  epochs: 2
  steps_per_epoch: 163
  total_steps: 326
  learning_rate: 2e-4

bug_types_included:
  - [all syntax from stage 1]
  - off_by_one
  - wrong_operator
  - wrong_comparison
  - missing_return

expected_metrics:
  initial_loss: ~3.0 (warm start)
  final_loss: ~1.8
  syntax_accuracy: 85%+
  simple_logic_accuracy: 65%+

progress_checkpoints:
  step_80: loss < 2.5
  step_160: loss < 2.2
  step_240: loss < 2.0
  step_326: loss < 1.8

success_criteria:
  - Loss < 1.8
  - Combined accuracy > 70%
```

#### Task 3.3: Stage 3 - Complex Logic

```yaml
task_id: P3.3
name: Curriculum Stage 3 - Complex Logic
target: Learn complex logic patterns

schedule:
  start: Day 7, Hour 9
  end: Day 7, Hour 15
  duration: 6 hours
  depends_on: [P3.2]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

training_params:
  samples: 350,000 (difficulty 0.0-0.6)
  epochs: 2
  steps_per_epoch: 228
  total_steps: 456
  learning_rate: 1e-4

bug_types_included:
  - [all from stages 1-2]
  - mutable_default
  - shallow_copy
  - none_check_missing
  - wrong_exception
  - bare_except

expected_metrics:
  initial_loss: ~2.0
  final_loss: ~1.4
  overall_accuracy: 72%+

progress_checkpoints:
  step_100: loss < 1.8
  step_200: loss < 1.6
  step_350: loss < 1.5
  step_456: loss < 1.4

success_criteria:
  - Loss < 1.4
  - No catastrophic forgetting of earlier stages
```

#### Task 3.4: Stage 4 - Advanced

```yaml
task_id: P3.4
name: Curriculum Stage 4 - Advanced
target: Learn advanced patterns

schedule:
  start: Day 7, Hour 15
  end: Day 7, Hour 22
  duration: 7 hours
  depends_on: [P3.3]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

training_params:
  samples: 500,000 (difficulty 0.0-0.8)
  epochs: 2
  steps_per_epoch: 326
  total_steps: 652
  learning_rate: 5e-5

bug_types_included:
  - [all from stages 1-3]
  - async_not_awaited
  - sql_injection
  - pandas_settingwithcopy
  - framework_specific

expected_metrics:
  initial_loss: ~1.5
  final_loss: ~1.1
  overall_accuracy: 75%+

progress_checkpoints:
  step_150: loss < 1.4
  step_300: loss < 1.3
  step_450: loss < 1.2
  step_652: loss < 1.1

success_criteria:
  - Loss < 1.1
  - Security bug detection > 70%
```

#### Task 3.5: Stage 5 - Full Dataset

```yaml
task_id: P3.5
name: Curriculum Stage 5 - Full Training
target: Final training on all data

schedule:
  start: Day 7, Hour 22
  end: Day 8, Hour 8
  duration: 10 hours
  depends_on: [P3.4]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

training_params:
  samples: 1,080,000 (all difficulties)
  epochs: 2
  steps_per_epoch: 703
  total_steps: 1406
  learning_rate: 2e-5

expected_metrics:
  initial_loss: ~1.2
  final_loss: ~0.8
  overall_accuracy: 78%+

progress_checkpoints:
  step_350: loss < 1.0
  step_700: loss < 0.9
  step_1050: loss < 0.85
  step_1406: loss < 0.8

checkpoint_saves:
  - Every 200 steps
  - Best model (by validation loss)
  - Final model

success_criteria:
  - Loss < 0.8
  - Validation loss < 0.9
  - No overfitting (val_loss stable)
```

---

### PHASE 4: Evaluation & Export (Days 9-10)

#### Task 4.1: Benchmark Evaluation

```yaml
task_id: P4.1
name: Comprehensive Benchmarking
target: Full performance metrics

schedule:
  start: Day 8, Hour 8
  end: Day 8, Hour 10
  duration: 2 hours
  depends_on: [P3.5]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 40%
  ram_allocated: 16 GB
  ram_utilization: 15%
  gpu_utilization: 40%
  vram_used: 20 GB

benchmarks:
  syntax_benchmark:
    samples: 5,000
    duration: 15 minutes
    metrics: [accuracy, precision, recall, f1]

  logic_benchmark:
    samples: 5,000
    duration: 20 minutes
    metrics: [accuracy, exact_match, partial_match]

  security_benchmark:
    samples: 2,000
    duration: 15 minutes
    metrics: [detection_rate, false_positive_rate]

  real_world_benchmark:
    samples: 1,000
    duration: 30 minutes
    includes_test_execution: true

  speed_benchmark:
    samples: 10,000
    duration: 10 minutes
    metrics: [throughput, latency_p50, latency_p99]

expected_results:
  syntax_accuracy: >90%
  logic_accuracy: >70%
  security_detection: >80%
  real_world_fix_rate: >60%
  throughput: >100 samples/sec
  latency_p99: <100ms

output:
  benchmark_report.json
  confusion_matrices/
  per_bug_type_metrics.csv

success_criteria:
  - All benchmarks complete
  - Syntax accuracy > 85%
  - Overall accuracy > 70%
```

#### Task 4.2: Error Analysis

```yaml
task_id: P4.2
name: Failure Analysis
target: Identify improvement areas

schedule:
  start: Day 8, Hour 10
  end: Day 8, Hour 14
  duration: 4 hours
  depends_on: [P4.1]

resource_allocation:
  cpu_cores: 16
  cpu_utilization: 60%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 10%
  vram_used: 8 GB

analysis_tasks:
  failure_collection:
    duration: 30 minutes
    collect: all incorrect predictions from benchmarks

  failure_clustering:
    duration: 1 hour
    method: embedding-based clustering
    clusters_expected: 20-50

  attention_visualization:
    duration: 1 hour
    samples_visualized: 100
    output: attention_maps/

  difficulty_correlation:
    duration: 30 minutes
    analyze: accuracy vs difficulty score

  bug_type_breakdown:
    duration: 30 minutes
    output: per_type_analysis.json

  recommendation_generation:
    duration: 30 minutes
    output: improvement_recommendations.md

output:
  error_analysis_report.md
  failure_clusters.json
  attention_visualizations/
  improvement_recommendations.md

success_criteria:
  - Top 5 failure modes identified
  - Actionable recommendations generated
```

#### Task 4.3: Fine-tuning (Conditional)

```yaml
task_id: P4.3
name: Targeted Fine-tuning
target: Address identified weaknesses
condition: Only if accuracy < 75% or specific weakness found

schedule:
  start: Day 8, Hour 14
  end: Day 8, Hour 18
  duration: 4 hours (if needed)
  depends_on: [P4.2]

resource_allocation:
  cpu_cores: 8
  cpu_utilization: 30%
  ram_allocated: 32 GB
  ram_utilization: 30%
  gpu_utilization: 90%
  vram_used: 80 GB

fine_tuning_options:
  weak_bug_types:
    samples: 50,000 (curated hard cases)
    epochs: 3
    learning_rate: 1e-5

  security_focus:
    samples: 30,000 (security-only)
    epochs: 2
    learning_rate: 5e-6

  framework_focus:
    samples: 30,000 (framework-specific)
    epochs: 2
    learning_rate: 5e-6

success_criteria:
  - Improvement on weak areas > 5%
  - No regression on other areas
```

#### Task 4.4: Final Export

```yaml
task_id: P4.4
name: Model Export & Packaging
target: Production-ready artifacts

schedule:
  start: Day 8, Hour 18
  end: Day 8, Hour 19
  duration: 1 hour
  depends_on: [P4.1, P4.3 if run]

resource_allocation:
  cpu_cores: 4
  cpu_utilization: 20%
  ram_allocated: 8 GB
  ram_utilization: 10%
  gpu_utilization: 0%
  vram_used: 0 GB

export_artifacts:
  model_weights:
    format: safetensors
    size: ~30 MB
    path: models/trm-coderepair-7m/model.safetensors

  config:
    format: json
    path: models/trm-coderepair-7m/config.json

  tokenizer:
    files: [vocab.json, merges.txt]
    path: models/trm-coderepair-7m/

  model_card:
    format: markdown
    path: models/trm-coderepair-7m/README.md

  onnx_export:
    format: onnx
    size: ~35 MB
    path: models/trm-coderepair-7m/model.onnx

  quantized_version:
    format: int8
    size: ~10 MB
    path: models/trm-coderepair-7m/model_int8.safetensors

verification:
  - Load model and run inference
  - Compare outputs to training checkpoint
  - Verify ONNX consistency

success_criteria:
  - All artifacts exported
  - Inference verified working
  - Model card complete
```

---

## Progress Tracking Dashboard

### Overall Progress Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TRM Training Progress Dashboard                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA COLLECTION                                                   │
│  ════════════════════════════════════════════════════════════════════════  │
│  □ P1.1 GitHub Mining      [░░░░░░░░░░░░░░░░░░░░] 0%     0/500K pairs      │
│  □ P1.2 Stack Overflow     [░░░░░░░░░░░░░░░░░░░░] 0%     0/100K pairs      │
│  □ P1.3 Synthetic Gen      [░░░░░░░░░░░░░░░░░░░░] 0%     0/500K pairs      │
│  □ P1.4 Linter Execution   [░░░░░░░░░░░░░░░░░░░░] 0%     0/300K pairs      │
│                                                                             │
│  PHASE 2: DATA PROCESSING                                                   │
│  ════════════════════════════════════════════════════════════════════════  │
│  □ P2.1 Vocabulary         [░░░░░░░░░░░░░░░░░░░░] 0%     0/32K tokens      │
│  □ P2.2 Tokenization       [░░░░░░░░░░░░░░░░░░░░] 0%     0/1.4M samples    │
│  □ P2.3 Validation         [░░░░░░░░░░░░░░░░░░░░] 0%     0/1.4M samples    │
│  □ P2.4 Grid Encoding      [░░░░░░░░░░░░░░░░░░░░] 0%     0/1.2M grids      │
│  □ P2.5 Finalization       [░░░░░░░░░░░░░░░░░░░░] 0%     Not started       │
│                                                                             │
│  PHASE 3: MODEL TRAINING                                                    │
│  ════════════════════════════════════════════════════════════════════════  │
│  □ P3.1 Stage 1 (Syntax)   [░░░░░░░░░░░░░░░░░░░░] 0%     Loss: --          │
│  □ P3.2 Stage 2 (Logic)    [░░░░░░░░░░░░░░░░░░░░] 0%     Loss: --          │
│  □ P3.3 Stage 3 (Complex)  [░░░░░░░░░░░░░░░░░░░░] 0%     Loss: --          │
│  □ P3.4 Stage 4 (Advanced) [░░░░░░░░░░░░░░░░░░░░] 0%     Loss: --          │
│  □ P3.5 Stage 5 (Full)     [░░░░░░░░░░░░░░░░░░░░] 0%     Loss: --          │
│                                                                             │
│  PHASE 4: EVALUATION                                                        │
│  ════════════════════════════════════════════════════════════════════════  │
│  □ P4.1 Benchmarking       [░░░░░░░░░░░░░░░░░░░░] 0%     Accuracy: --      │
│  □ P4.2 Error Analysis     [░░░░░░░░░░░░░░░░░░░░] 0%     Not started       │
│  □ P4.3 Fine-tuning        [░░░░░░░░░░░░░░░░░░░░] 0%     Conditional       │
│  □ P4.4 Export             [░░░░░░░░░░░░░░░░░░░░] 0%     Not started       │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│  OVERALL: 0% Complete      Time: Day 0/10      ETA: ~200 hours             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Real-Time Monitoring Commands

```bash
# System resource monitoring
watch -n 1 'rocm-smi && echo "---" && htop -n 1 | head -20'

# Training progress
tail -f logs/training.log | grep -E "(step|loss|accuracy)"

# Data collection progress
watch -n 60 'wc -l data/raw/github/*.jsonl | tail -1'

# Disk usage
watch -n 300 'du -sh data/*'

# Network I/O
iftop -i eth0

# Combined dashboard (tmux)
tmux new-session -d -s monitor
tmux split-window -h
tmux send-keys -t 0 'watch rocm-smi' Enter
tmux send-keys -t 1 'tail -f logs/training.log' Enter
tmux attach -t monitor
```

### Log File Locations

```
logs/
├── collection/
│   ├── github.log           # GitHub API calls, rate limits
│   ├── stackoverflow.log    # SO processing
│   └── synthetic.log        # Generation progress
├── processing/
│   ├── tokenization.log     # Tokenization progress
│   ├── validation.log       # Validation results
│   └── encoding.log         # Grid encoding
├── training/
│   ├── stage_1.log          # Stage 1 training
│   ├── stage_2.log          # Stage 2 training
│   ├── stage_3.log          # Stage 3 training
│   ├── stage_4.log          # Stage 4 training
│   ├── stage_5.log          # Stage 5 training
│   └── tensorboard/         # TensorBoard logs
└── evaluation/
    ├── benchmark.log        # Benchmark results
    └── analysis.log         # Error analysis
```

---

## Summary Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Day 1  │████████│ P1.1 GitHub (starts), P1.2 SO (completes), P1.3 Synth   │
│  Day 2  │████████│ P1.1 GitHub (continues), P1.4 Linter                    │
│  Day 3  │████████│ P1.1 GitHub (continues), P1.4 Linter (completes)        │
│  Day 4  │████████│ P1.1 GitHub (continues)                                 │
│  Day 5  │████████│ P1.1 GitHub (completes), P2.1-P2.2 Processing           │
│  Day 6  │████████│ P2.3 Validation, P2.4-P2.5 Finalization                 │
│  Day 7  │████████│ P3.1-P3.4 Training Stages 1-4                           │
│  Day 8  │████████│ P3.5 Training Stage 5, P4.1-P4.4 Evaluation             │
│  Day 9  │████    │ Buffer / Fine-tuning if needed                          │
│  Day 10 │████    │ Buffer / Documentation                                  │
│                                                                             │
│  Total Wall Clock: ~200 hours (8-9 days)                                   │
│  Total CPU Hours: ~450 hours                                                │
│  Total GPU Hours: ~40 hours                                                 │
│  Electricity: ~$15                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria Summary

| Milestone | Metric | Target | Checkpoint |
|-----------|--------|--------|------------|
| Data Collection | Total pairs | 1.4M | Day 5 |
| Validation | Pass rate | >85% | Day 6 |
| Training Stage 1 | Syntax accuracy | >75% | Day 7 |
| Training Stage 5 | Overall loss | <0.8 | Day 8 |
| Final Evaluation | Syntax accuracy | >90% | Day 8 |
| Final Evaluation | Logic accuracy | >70% | Day 8 |
| Final Evaluation | Overall accuracy | >75% | Day 8 |
| Export | Model size | <50MB | Day 8 |
