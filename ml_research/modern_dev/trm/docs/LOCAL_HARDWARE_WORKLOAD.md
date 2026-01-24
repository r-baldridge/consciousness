# TRM Code Repair - Local Hardware Workload Specification

## Hardware Configuration

### Primary Workstation
- **CPU/GPU**: AMD Instinct MI300A (Max+ 395)
  - 24 Zen 4 CPU cores
  - 228 CUs (CDNA 3 architecture)
  - 96GB HBM3 unified memory
  - 5.3 TB/s memory bandwidth
  - ~1.3 PFLOPS FP16 / ~2.6 PFLOPS FP8

### AI Accelerator
- **NVIDIA DGX Spark** (Grace Blackwell)
  - NVIDIA GB10 Grace Blackwell Superchip
  - 128GB unified memory
  - Up to 1 PFLOPS AI performance
  - NVLink-C2C interconnect

### Network Storage
- **NAS**: Assumed 10GbE or higher connection
- **Servers**: Additional compute available on demand

---

## Process Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRM Training Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: Data Collection          PHASE 2: Processing                      │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │ GitHub API Mining   │──────────▶│ Tokenization        │                  │
│  │ Stack Overflow Dump │──────────▶│ Validation          │                  │
│  │ Synthetic Gen       │──────────▶│ Deduplication       │                  │
│  │ Linter Execution    │──────────▶│ Grid Encoding       │                  │
│  └─────────────────────┘           └──────────┬──────────┘                  │
│         │                                     │                              │
│         │ Network I/O                         │ CPU Intensive                │
│         │ ~500GB download                     │ ~200 CPU-hours               │
│         ▼                                     ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │     NAS Storage     │◀──────────│   Processed Data    │                  │
│  │     (Raw Data)      │           │   (Parquet Shards)  │                  │
│  └─────────────────────┘           └──────────┬──────────┘                  │
│                                               │                              │
│                                               │ Data Loading                 │
│                                               ▼                              │
│  PHASE 3: Training                 PHASE 4: Evaluation                      │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │ Curriculum Stage 1  │──────────▶│ Benchmark Suite     │                  │
│  │ Curriculum Stage 2  │──────────▶│ Error Analysis      │                  │
│  │ Curriculum Stage 3  │──────────▶│ Human Evaluation    │                  │
│  │ Fine-tuning         │──────────▶│ Production Tests    │                  │
│  └─────────────────────┘           └─────────────────────┘                  │
│         │                                     │                              │
│         │ GPU Intensive                       │ Mixed CPU/GPU                │
│         │ ~150 GPU-hours                      │ ~20 GPU-hours                │
│         ▼                                     ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                  │
│  │  Model Checkpoints  │──────────▶│   Final Model       │                  │
│  │  (NAS Storage)      │           │   (Deployment)      │                  │
│  └─────────────────────┘           └─────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Collection

### 1.1 GitHub Commit Mining

```yaml
Task: Mine bug-fix commits from 500+ repositories
Target: 500,000 bug-fix pairs

Hardware Allocation:
  Primary: AMD MI300A (CPU cores only)
  GPU: Idle (not needed)

Resource Usage:
  CPU Cores: 8-12 cores (rate-limited by API)
  RAM: 16-32 GB
  GPU: 0%
  Network: Variable (API throttled)

Network Traffic:
  API Requests: ~2M requests total
  Data Download: ~50GB metadata + ~200GB repo clones
  Rate: Limited to 5,000 req/hr (GitHub API)

Duration Estimate:
  With 1 token: ~400 hours (17 days)
  With 4 tokens: ~100 hours (4 days)
  With GitHub App: ~50 hours (2 days)

Bottleneck: GitHub API rate limits (not hardware)
```

**Network Traffic Breakdown:**
| Stage | Download | Upload | Duration |
|-------|----------|--------|----------|
| Repo discovery | 2 GB | 50 MB | 4 hrs |
| Commit history | 15 GB | 100 MB | 20 hrs |
| Diff extraction | 30 GB | 200 MB | 40 hrs |
| File content | 200 GB | 500 MB | 36 hrs |
| **Total** | **~250 GB** | **~1 GB** | **~100 hrs** |

### 1.2 Stack Overflow Data Dump

```yaml
Task: Download and process SO data dump
Target: 100,000 Q&A pairs

Hardware Allocation:
  Primary: AMD MI300A (CPU + NVMe)
  Storage: NAS for archive

Resource Usage:
  CPU Cores: 4-8 cores (decompression/parsing)
  RAM: 64 GB (XML parsing)
  GPU: 0%
  Disk I/O: High (sequential read)

Network Traffic:
  Download: 18 GB compressed (archive.org)
  Rate: ~100 MB/s typical

Duration Estimate:
  Download: 3 minutes (on fast connection)
  Extraction: 30 minutes (18GB → 90GB)
  Parsing: 4 hours
  Filtering: 2 hours
  Total: ~7 hours

Bottleneck: XML parsing (CPU-bound)
```

### 1.3 Synthetic Bug Generation

```yaml
Task: Generate synthetic buggy/fixed pairs
Target: 500,000 pairs

Hardware Allocation:
  Primary: AMD MI300A (CPU cores)
  Storage: Local NVMe

Resource Usage:
  CPU Cores: 20-24 cores (parallel generation)
  RAM: 32 GB
  GPU: 0% (AST parsing is CPU-only)
  Disk I/O: Moderate write

Network Traffic:
  None (local processing)

Duration Estimate:
  Generation rate: ~1,000 pairs/minute
  Total: ~8 hours

Process Breakdown:
  - Load source code corpus: 30 min
  - AST parsing: 2 hours
  - Mutation application: 3 hours
  - Validation: 2 hours
  - Serialization: 30 min

Bottleneck: Python AST parsing (CPU-bound)
```

### 1.4 Linter Execution

```yaml
Task: Run static analyzers on code corpus
Target: 300,000 linter-derived pairs

Hardware Allocation:
  Primary: AMD MI300A (CPU cores)
  Secondary: Can distribute to network servers

Resource Usage:
  CPU Cores: 24 cores (fully parallel)
  RAM: 48 GB
  GPU: 0%
  Disk I/O: High (many small files)

Tools Run:
  - pylint: ~2 sec/file
  - mypy: ~1 sec/file
  - ruff: ~0.1 sec/file
  - bandit: ~0.5 sec/file

Network Traffic:
  To NAS: ~100 GB (results storage)
  From NAS: ~200 GB (source files)

Duration Estimate:
  Files to process: ~500,000
  Per-file time: ~4 seconds average
  With 24 cores: ~23 hours

Bottleneck: Disk I/O for many small files
```

---

## Phase 2: Data Processing

### 2.1 Tokenization & Vocabulary Building

```yaml
Task: Build 32K BPE vocabulary and tokenize all data
Target: 1.5M samples tokenized

Hardware Allocation:
  Primary: AMD MI300A

  Vocabulary Training:
    CPU: 16 cores
    RAM: 64 GB (full corpus in memory)
    GPU: Optional (can accelerate BPE)

  Tokenization:
    CPU: 24 cores
    RAM: 32 GB
    GPU: 0%

Network Traffic:
  From NAS: ~500 GB (raw data)
  To NAS: ~200 GB (tokenized data)

Duration Estimate:
  Vocabulary training: 4 hours
  Tokenization (1.5M samples): 6 hours
  Total: ~10 hours

CPU Load Profile:
  ┌────────────────────────────────────────┐
  │ 100%│████████████████                  │ Vocab training
  │  75%│                ████████████████  │ Tokenization
  │  50%│                                  │
  │  25%│                                  │
  │   0%│────────────────────────────────  │
  │     0h      2h      4h      6h     10h │
  └────────────────────────────────────────┘
```

### 2.2 Validation Pipeline

```yaml
Task: Validate all bug-fix pairs
Target: 1.5M samples validated, ~1.2M passing

Hardware Allocation:
  Primary: AMD MI300A (CPU)
  Sandbox: Docker containers (for test execution)

Resource Usage:
  CPU Cores: 24 cores
  RAM: 64 GB (Docker overhead)
  GPU: 0%
  Docker: 100+ containers cycling

Validation Steps:
  1. Syntax validation (fast): 0.01s/sample
  2. Semantic diff check: 0.05s/sample
  3. Test execution (subset): 2s/sample
  4. Deduplication (LSH): 0.1s/sample

Network Traffic:
  Docker images: 5 GB initial pull
  Inter-container: minimal
  To NAS: 50 GB validation results

Duration Estimate:
  Syntax + semantic (1.5M): 2 hours
  Test execution (200K subset): 12 hours
  Deduplication: 4 hours
  Total: ~18 hours

Bottleneck: Test execution in sandbox
```

### 2.3 Grid Encoding

```yaml
Task: Convert tokenized code to 64x48 grids
Target: 1.2M validated samples

Hardware Allocation:
  Primary: AMD MI300A
  Optional: DGX Spark for acceleration

Resource Usage:
  CPU-only mode:
    CPU: 24 cores
    RAM: 32 GB
    Throughput: ~5,000 samples/sec

  GPU-accelerated mode:
    GPU: MI300A (partial allocation)
    VRAM: 16 GB
    Throughput: ~50,000 samples/sec

Network Traffic:
  From NAS: 200 GB tokenized data
  To NAS: 150 GB grid data (compressed parquet)

Duration Estimate:
  CPU-only: 4 hours
  GPU-accelerated: 25 minutes

Output Format:
  - Sharded parquet files (1GB each)
  - ~150 shards total
  - Metadata JSON per shard
```

### 2.4 Dataset Finalization

```yaml
Task: Split, shuffle, and prepare final dataset
Target: Train/Val/Test splits

Hardware Allocation:
  Primary: AMD MI300A (CPU + high RAM)

Resource Usage:
  CPU Cores: 8 cores
  RAM: 96 GB (full dataset in memory for shuffle)
  GPU: 0%

Operations:
  1. Global shuffle (RAM-intensive)
  2. Stratified split by bug type
  3. Curriculum difficulty scoring
  4. Final validation pass

Network Traffic:
  From NAS: 150 GB (all shards)
  To NAS: 150 GB (reorganized)

Duration Estimate:
  Load all shards: 30 min
  Shuffle: 15 min
  Split: 10 min
  Write: 45 min
  Total: ~2 hours

Final Dataset Structure:
  train/: 1.08M samples (90%)
  val/:   60K samples (5%)
  test/:  60K samples (5%)
```

---

## Phase 3: Model Training

### 3.1 Hardware Strategy

Given the available hardware, we have two powerful options:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Hardware Options                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option A: AMD MI300A Primary                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  96GB HBM3 Unified Memory                                    │   │
│  │  - Can train larger batch sizes                              │   │
│  │  - ROCm + PyTorch 2.x                                        │   │
│  │  - ~1.3 PFLOPS FP16                                          │   │
│  │  - Best for: Large batch training, memory-bound models       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Option B: DGX Spark Primary                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  128GB Unified Memory (Grace Blackwell)                      │   │
│  │  - Blackwell architecture optimizations                       │   │
│  │  - CUDA + latest optimizations                               │   │
│  │  - ~1 PFLOPS AI performance                                  │   │
│  │  - Best for: Transformer training, mixed precision           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Option C: Distributed (Both)                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Combined: 224GB unified memory                              │   │
│  │  - Data parallel training                                    │   │
│  │  - Requires network synchronization                          │   │
│  │  - ~2x throughput potential                                  │   │
│  │  - Best for: Hyperparameter search, ensemble training        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Recommendation**: Use DGX Spark for primary training (better Transformer optimization), MI300A for data preprocessing and parallel experiments.

### 3.2 TRM Model Configuration

```yaml
# Training configuration for 7M parameter TRM

model:
  name: TRM-CodeRepair-7M
  parameters: 7,340,032

  architecture:
    d_model: 256
    n_heads: 8
    d_ff: 1024
    n_actual_layers: 2
    n_recursive_steps: 21  # 2 * 21 = 42 effective layers
    grid_height: 64
    grid_width: 48
    vocab_size: 32768

  memory_footprint:
    parameters: 28 MB (FP32) / 14 MB (FP16)
    gradients: 28 MB (FP32)
    optimizer_states: 56 MB (Adam)
    activations_per_sample: ~50 MB (with recursion)

training:
  # Batch size calculations for different hardware

  dgx_spark_128gb:
    max_batch_size: 2048
    recommended_batch_size: 1024
    gradient_accumulation: 1
    effective_batch_size: 1024

  mi300a_96gb:
    max_batch_size: 1536
    recommended_batch_size: 768
    gradient_accumulation: 2
    effective_batch_size: 1536

  distributed_both:
    batch_per_device: 768
    devices: 2
    gradient_accumulation: 1
    effective_batch_size: 1536
```

### 3.3 Curriculum Training Schedule

```yaml
# 5-stage curriculum training

curriculum:
  total_samples: 1,080,000
  total_epochs: 10

  stage_1_syntax:
    samples: 150,000
    epochs: 2
    bug_types: [syntax_errors]
    difficulty: 0.0-0.2
    learning_rate: 3e-4

  stage_2_simple_logic:
    samples: 250,000
    epochs: 2
    bug_types: [syntax_errors, simple_logic]
    difficulty: 0.0-0.4
    learning_rate: 2e-4

  stage_3_complex_logic:
    samples: 350,000
    epochs: 2
    bug_types: [all_logic]
    difficulty: 0.0-0.6
    learning_rate: 1e-4

  stage_4_advanced:
    samples: 500,000
    epochs: 2
    bug_types: [all_except_expert]
    difficulty: 0.0-0.8
    learning_rate: 5e-5

  stage_5_full:
    samples: 1,080,000
    epochs: 2
    bug_types: [all]
    difficulty: 0.0-1.0
    learning_rate: 2e-5
```

### 3.4 Training Duration Estimates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Training Time Estimates                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Configuration: TRM-7M, Batch=1024, 10 epochs over 1.08M samples            │
│  Total training steps: ~10,500 steps                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Hardware        │ Steps/sec │ Time/Epoch │ Total Time │ GPU Util   │   │
│  ├─────────────────┼───────────┼────────────┼────────────┼────────────┤   │
│  │ DGX Spark       │ 8-12      │ 1.5-2 hrs  │ 15-20 hrs  │ 85-95%     │   │
│  │ MI300A          │ 6-10      │ 2-2.5 hrs  │ 20-25 hrs  │ 80-90%     │   │
│  │ Both (parallel) │ 12-18     │ 1-1.5 hrs  │ 10-15 hrs  │ 80-90%     │   │
│  └─────────────────┴───────────┴────────────┴────────────┴────────────┘   │
│                                                                             │
│  Per-Stage Breakdown (using DGX Spark):                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 1 (Syntax)      : 2 epochs ×  150K = 300K samples →  3 hrs   │   │
│  │ Stage 2 (Simple Logic): 2 epochs ×  250K = 500K samples →  4 hrs   │   │
│  │ Stage 3 (Complex)     : 2 epochs ×  350K = 700K samples →  5 hrs   │   │
│  │ Stage 4 (Advanced)    : 2 epochs ×  500K = 1M samples   →  5 hrs   │   │
│  │ Stage 5 (Full)        : 2 epochs × 1.08M = 2.16M samples → 8 hrs   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ TOTAL CURRICULUM TRAINING TIME: ~25 hours                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 GPU Memory Usage During Training

```
DGX Spark (128GB) Memory Profile:
┌────────────────────────────────────────────────────────────────┐
│ Component                    │ Memory    │ % of 128GB          │
├──────────────────────────────┼───────────┼─────────────────────┤
│ Model Parameters (FP16)      │ 14 MB     │ 0.01%               │
│ Gradients (FP16)             │ 14 MB     │ 0.01%               │
│ Optimizer States (FP32)      │ 56 MB     │ 0.04%               │
│ Activations (batch=1024)     │ 51 GB     │ 40%                 │
│ Attention Cache              │ 8 GB      │ 6%                  │
│ Data Loader Buffers          │ 4 GB      │ 3%                  │
│ CUDA/ROCm Overhead           │ 8 GB      │ 6%                  │
├──────────────────────────────┼───────────┼─────────────────────┤
│ TOTAL USED                   │ ~72 GB    │ 56%                 │
│ HEADROOM                     │ ~56 GB    │ 44%                 │
└──────────────────────────────┴───────────┴─────────────────────┘

Note: Large headroom allows for:
- Gradient checkpointing disabled (faster)
- Larger batch sizes if needed
- Mixed precision training buffers
- Debugging/profiling overhead
```

### 3.6 Network Traffic During Training

```yaml
training_network_io:

  data_loading:
    source: NAS
    pattern: Sequential shard reads
    throughput_needed: 500 MB/s sustained
    actual_with_10gbe: 1.2 GB/s max
    bottleneck: No (network sufficient)

  checkpoint_saves:
    frequency: Every 1000 steps
    checkpoint_size: 150 MB (full state)
    destination: NAS
    time_per_save: <1 second

  wandb_logging:
    frequency: Every 10 steps
    payload: ~10 KB metrics
    destination: Internet (wandb.ai)
    bandwidth: Negligible

  tensorboard:
    frequency: Every 100 steps
    payload: ~1 MB histograms
    destination: Local or NAS
    bandwidth: Negligible

total_network_io_training:
  download_from_nas: ~500 GB (dataset, read multiple times)
  upload_to_nas: ~5 GB (checkpoints)
  internet: ~100 MB (logging)
```

---

## Phase 4: Evaluation & Fine-tuning

### 4.1 Benchmark Evaluation

```yaml
Task: Evaluate model on benchmark suite
Target: Comprehensive performance metrics

Hardware Allocation:
  Primary: DGX Spark (inference)
  Secondary: MI300A (parallel benchmarks)

Benchmarks:
  syntax_benchmark:
    samples: 5,000
    inference_time: 10 min

  logic_benchmark:
    samples: 5,000
    inference_time: 15 min

  security_benchmark:
    samples: 2,000
    inference_time: 8 min

  real_world_benchmark:
    samples: 1,000
    inference_time: 20 min (includes test execution)

Resource Usage:
  GPU: 30-50% (inference is memory-bound)
  CPU: 20% (test execution)
  RAM: 16 GB

Duration: ~1 hour total
```

### 4.2 Error Analysis

```yaml
Task: Analyze model failures and edge cases
Target: Identify improvement areas

Hardware Allocation:
  Primary: MI300A (analysis scripts)
  Notebooks: Jupyter on either system

Operations:
  - Confusion matrix computation
  - Failure case clustering
  - Attention visualization
  - Difficulty correlation analysis

Resource Usage:
  CPU: 8-16 cores
  GPU: Minimal (visualization)
  RAM: 32 GB

Duration: 4-8 hours (semi-manual)
```

### 4.3 Targeted Fine-tuning

```yaml
Task: Fine-tune on failure cases
Target: Improve weak areas

Hardware Allocation:
  Primary: DGX Spark

Fine-tuning Configs:
  weak_area_finetune:
    samples: 50,000 (curated hard cases)
    epochs: 5
    learning_rate: 1e-5

  security_specialization:
    samples: 30,000 (security-focused)
    epochs: 3
    learning_rate: 5e-6

Duration: 3-5 hours per fine-tune run
```

---

## Complete Timeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Complete Processing Timeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA COLLECTION (Running in parallel where possible)              │
│  ─────────────────────────────────────────────────────────────              │
│  │                                                                          │
│  │  GitHub Mining ════════════════════════════════════════  100 hrs        │
│  │  (API rate-limited, can run 24/7 in background)                         │
│  │                                                                          │
│  │  Stack Overflow ████  7 hrs                                              │
│  │                                                                          │
│  │  Synthetic Gen  ████████  8 hrs                                          │
│  │                                                                          │
│  │  Linter Execution ██████████████████████████  23 hrs                    │
│  │                                                                          │
│  │  Phase 1 Wall Clock: ~100 hours (GitHub is bottleneck)                  │
│  │  Phase 1 CPU-hours: ~200 hours                                          │
│  │                                                                          │
│  PHASE 2: DATA PROCESSING                                                   │
│  ─────────────────────────────────────────────────────────                  │
│  │                                                                          │
│  │  Tokenization    ██████████  10 hrs                                      │
│  │  Validation      ██████████████████  18 hrs                              │
│  │  Grid Encoding   █  0.5 hrs (GPU accelerated)                            │
│  │  Finalization    ██  2 hrs                                               │
│  │                                                                          │
│  │  Phase 2 Wall Clock: ~30 hours                                          │
│  │  Phase 2 CPU-hours: ~150 hours                                          │
│  │                                                                          │
│  PHASE 3: MODEL TRAINING                                                    │
│  ─────────────────────────────────────────────────────────                  │
│  │                                                                          │
│  │  Curriculum Training ████████████████████████████  25 hrs               │
│  │  (5 stages, 10 total epochs)                                            │
│  │                                                                          │
│  │  Phase 3 Wall Clock: ~25 hours                                          │
│  │  Phase 3 GPU-hours: ~25 hours                                           │
│  │                                                                          │
│  PHASE 4: EVALUATION & FINE-TUNING                                         │
│  ─────────────────────────────────────────────────────────                  │
│  │                                                                          │
│  │  Benchmarking    █  1 hr                                                 │
│  │  Error Analysis  ████████  8 hrs                                         │
│  │  Fine-tuning     █████  5 hrs                                            │
│  │  Final Eval      █  1 hr                                                 │
│  │                                                                          │
│  │  Phase 4 Wall Clock: ~15 hours                                          │
│  │                                                                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│  TOTAL WALL CLOCK TIME: ~170 hours (~7 days)                               │
│  TOTAL CPU-HOURS: ~400 hours                                                │
│  TOTAL GPU-HOURS: ~45 hours                                                 │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Resource Utilization Summary

### CPU Utilization Over Time

```
100%│     ▄▄▄▄          ▄▄▄▄▄▄▄▄
 75%│   ▄█████▄       ▄██████████▄
 50%│  ▄███████▄     ▄████████████▄      ▄▄
 25%│▄▄█████████▄▄▄▄▄██████████████▄▄▄▄▄████▄▄
  0%│─────────────────────────────────────────
    │ Day1  Day2  Day3  Day4  Day5  Day6  Day7
    │ Collection  │ Processing │Training│Eval│
```

### GPU Utilization Over Time

```
100%│                              ▄▄▄▄▄▄▄▄▄▄
 75%│                             █████████████
 50%│                            ██████████████
 25%│            ▄               ██████████████
  0%│▄▄▄▄▄▄▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄███████████████
    │ Day1  Day2  Day3  Day4  Day5  Day6  Day7
    │ Collection  │ Processing │Training│Eval│
```

### Network I/O Over Time

```
1GB/s│  ▄      ▄▄                ▄▄▄▄▄▄
     │ ██▄    ███▄             ██████████
500MB│████▄  █████            ████████████
     │██████▄█████▄▄▄▄▄▄▄▄▄▄▄▄█████████████▄▄▄
  0  │─────────────────────────────────────────
     │ Day1  Day2  Day3  Day4  Day5  Day6  Day7
     │ Download  │ Process   │ Training Data │
```

---

## Power & Cost Estimates

### Power Consumption

```yaml
hardware_power:
  amd_mi300a:
    tdp: 760W
    typical_load: 500-600W
    idle: 100W

  dgx_spark:
    tdp: 500W (estimated for GB10)
    typical_load: 350-450W
    idle: 75W

  nas_and_network:
    typical: 100W

  total_system:
    peak: 1360W
    typical_training: 1000W
    typical_processing: 700W
    idle: 275W
```

### Electricity Cost

```yaml
electricity_calculation:
  # Assuming $0.12/kWh (US average)

  phase_1_collection:
    hours: 100
    avg_power: 400W  # CPU only
    kwh: 40
    cost: $4.80

  phase_2_processing:
    hours: 30
    avg_power: 600W
    kwh: 18
    cost: $2.16

  phase_3_training:
    hours: 25
    avg_power: 900W  # GPU active
    kwh: 22.5
    cost: $2.70

  phase_4_evaluation:
    hours: 15
    avg_power: 500W
    kwh: 7.5
    cost: $0.90

  total_electricity:
    kwh: 88
    cost: $10.56
```

### Total Local Processing Cost

```
┌─────────────────────────────────────────────────────────────┐
│              Local Hardware Processing Costs                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fixed Costs (already owned):                               │
│    AMD MI300A system          : $0 (sunk cost)              │
│    DGX Spark                  : $0 (sunk cost)              │
│    NAS storage                : $0 (sunk cost)              │
│                                                             │
│  Variable Costs:                                            │
│    Electricity (88 kWh)       : $10.56                      │
│    Internet bandwidth         : ~$0 (existing connection)   │
│    NAS wear (negligible)      : ~$5                         │
│                                                             │
│  API Costs:                                                 │
│    GitHub API                 : $0 (free tier)              │
│    Stack Exchange API         : $0 (free tier)              │
│    Weights & Biases           : $0 (free tier)              │
│                                                             │
│  Optional Services:                                         │
│    Cloud backup (S3, 500GB)   : ~$12/month                  │
│    Domain for demo            : ~$12/year                   │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  TOTAL VARIABLE COST: ~$30 per full training run            │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Optimization Recommendations

### 1. Parallel Processing Strategy

```yaml
optimal_workflow:

  # Start GitHub collection immediately (runs for days)
  day_1_start:
    - github_collection: background, 24/7

  # While GitHub runs, do local generation
  day_1_parallel:
    - synthetic_generation: MI300A CPU
    - stackoverflow_download: background

  # Process as data arrives
  day_2_onwards:
    - process_stackoverflow: when download complete
    - process_synthetic: when generation complete
    - linter_execution: as repos are cloned

  # Start training on partial data
  day_4:
    - begin_stage_1_training: on synthetic data
    - continue_collection: background

  # Full training when all data ready
  day_7:
    - full_curriculum_training: all data
```

### 2. Memory Optimization

```yaml
memory_tips:

  during_collection:
    - Stream data to NAS, don't buffer in RAM
    - Use memory-mapped files for large datasets

  during_processing:
    - Process in shards (1GB each)
    - Use Arrow/Parquet for zero-copy reads

  during_training:
    - Enable gradient checkpointing if needed
    - Use mixed precision (FP16/BF16)
    - Pin memory for data loaders
```

### 3. Network Optimization

```yaml
network_tips:

  nas_access:
    - Use NFS v4.1 with parallel NFS for multi-stream
    - Enable jumbo frames (MTU 9000) if supported
    - Place hot data on SSD tier of NAS

  data_loading:
    - Pre-fetch next batch during GPU compute
    - Use multiple data loader workers (4-8)
    - Cache frequently accessed data locally
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRM Training Quick Reference                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TOTAL TIME: ~7 days wall clock                                             │
│  TOTAL COST: ~$30 electricity + API fees                                    │
│                                                                             │
│  Hardware Assignment:                                                       │
│  ├── DGX Spark    → Primary training (better Transformer perf)              │
│  ├── MI300A       → Data processing, parallel experiments                   │
│  └── NAS          → All data storage, checkpoints                           │
│                                                                             │
│  Key Commands:                                                              │
│  ├── make collect-all      # Start data collection                          │
│  ├── make process-data     # Process when collection done                   │
│  ├── make train-curriculum # Full curriculum training                       │
│  └── make evaluate         # Final evaluation                               │
│                                                                             │
│  Monitoring:                                                                │
│  ├── wandb.ai              # Training metrics                               │
│  ├── nvidia-smi / rocm-smi # GPU utilization                                │
│  └── htop                  # CPU/memory                                     │
│                                                                             │
│  Expected Output:                                                           │
│  ├── 1.2M validated training samples                                        │
│  ├── TRM-7M model (~50MB)                                                   │
│  ├── >80% accuracy on syntax bugs                                           │
│  └── >60% accuracy on logic bugs                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
