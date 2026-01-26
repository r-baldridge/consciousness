# Architecture Overview

This document describes the system architecture for the ML Research Code Repair Pipeline.

---

## Table of Contents

1. [Pipeline Flow](#pipeline-flow)
2. [Component Architecture](#component-architecture)
3. [TRM Architecture](#trm-architecture)
4. [Mamba Architecture](#mamba-architecture)
5. [Orchestrator Design](#orchestrator-design)
6. [Data Flow](#data-flow)
7. [Design Decisions](#design-decisions)

---

## Pipeline Flow

```
Input (buggy code)
       |
       v
+------------------+
|   MLOrchestrator |  <-- Entry point, coordinates all components
+------------------+
       |
       v
+------------------+
|   TaskRouter     |  <-- Analyzes task, selects optimal architecture
+------------------+
       |
       +------------------------------------------+
       |                    |                     |
       v                    v                     v
+------------+       +------------+       +------------+
|    TRM     |       |   Mamba    |       |    RLM     |
| (Refiner)  |       | (Encoder)  |       | (Synth)    |
+------------+       +------------+       +------------+
       |                    |                     |
       +------------------------------------------+
       |
       v
+------------------+
| FallbackHandler  |  <-- Handles failures, retries with fallbacks
+------------------+
       |
       v
Output (fixed code)
```

### High-Level Flow

1. **Request Reception**: Client sends `Request` to `MLOrchestrator`
2. **Task Analysis**: `TaskRouter` analyzes input characteristics
3. **Architecture Selection**: Best architecture chosen based on task type
4. **Execution**: Selected architecture processes the task
5. **Fallback Handling**: On failure, try alternative architectures
6. **Response**: Return `Response` with results and metadata

---

## Component Architecture

### Orchestrator Components

```
+--------------------------------------------------------------------------------+
|                              MLOrchestrator                                     |
|--------------------------------------------------------------------------------|
|  +-------------------+  +----------------+  +------------------+               |
|  | ArchitectureReg.  |  |  TaskRouter    |  | FallbackHandler  |               |
|  |-------------------|  |----------------|  |------------------|               |
|  | - architectures   |  | - registry     |  | - router         |               |
|  | - factories       |  | - strategy     |  | - max_attempts   |               |
|  | - instances       |  | - affinity_map |  | - retry_delay    |               |
|  | - perf_history    |  | - history      |  | - failure_hist   |               |
|  +-------------------+  +----------------+  +------------------+               |
|                                                                                 |
|  +-------------------+                                                          |
|  |   UnifiedIndex    |  <- Technique-to-architecture mapping                   |
|  |-------------------|                                                          |
|  | - technique_map   |                                                          |
|  | - arch_configs    |                                                          |
|  | - metadata        |                                                          |
|  +-------------------+                                                          |
+--------------------------------------------------------------------------------+
```

### Architecture Registry

The registry maintains:

```python
{
    "trm": {
        "capability": ArchitectureCapability(...),
        "factory": lambda: load_trm(),
        "instance": None,  # Lazy loaded
        "performance": [...]  # Historical metrics
    },
    "mamba": {...},
    "rlm": {...}
}
```

---

## TRM Architecture

TRM (Tiny Recursive Model) uses iterative refinement for code repair.

### Code Repair TRM Flow

```
Input Token IDs (batch, 64, 48)
         |
         v
+-------------------+
| Token Embedding   |  <- Embed tokens to continuous space
| (vocab -> d_model)|
+-------------------+
         |
         v
+-------------------+
| Positional Enc.   |  <- 2D grid-aware positional encoding
| (height x width)  |
+-------------------+
         |
         v
+------------------------+
| Deep Recursion Module  |<---------+
|------------------------|          |
| Block 1: Attn + FFN   |          |
| Block 2: Attn + FFN   |          | 8 iterations
| Block 3: Attn + FFN   |          | (weight shared)
| Block 4: Attn + FFN   |          |
| Block 5: Attn + FFN   |          |
| Block 6: Attn + FFN   |          |
+------------------------+----------+
         |
    [Iteration Controller]
    - Compute confidence (q_hat)
    - Early stop if q > 0.95
         |
         v
+-------------------+
| Output Head       |  <- Project back to vocab
| (d_model -> vocab)|
+-------------------+
         |
         v
Output Logits (batch, 64, 48, vocab)
```

### Key TRM Innovations

1. **Weight Sharing**: 6 blocks x 8 iterations = 48 effective layers with only 6 layers of parameters
2. **Early Stopping**: Halt when confidence exceeds threshold
3. **Grid Representation**: Code as 2D grid (64 lines x 48 tokens)
4. **Iterative Refinement**: Each iteration improves the solution

### RecursiveBlock Structure

```
     x
     |
     +---> RMSNorm ---> Attention ---> +
     |                                 |
     +---------------------------------+
                                       |
     +---> RMSNorm ---> FFN ---------->+
     |                                 |
     +---------------------------------+
                                       |
                                       v
                                     output
```

---

## Mamba Architecture

Mamba provides efficient long-context processing with O(N) complexity.

### Mamba Language Model Flow

```
Input Token IDs (batch, seq_len)
         |
         v
+-------------------+
| Token Embedding   |
| (vocab -> d_model)|
+-------------------+
         |
         v
+---------------------------------------------+
|           MambaBlock x N                     |
|---------------------------------------------|
|  +----------------+                          |
|  |   RMSNorm      |                          |
|  +----------------+                          |
|          |                                   |
|          v                                   |
|  +----------------+                          |
|  | Input Project  | -> x, z branches         |
|  | (d -> 2*d_in)  |                          |
|  +----------------+                          |
|       |      |                               |
|       v      v                               |
|  +-------+  +--------+                       |
|  | Conv  |  |  Gate  |                       |
|  +-------+  +--------+                       |
|       |          |                           |
|       v          |                           |
|  +----------+    |                           |
|  |   SSM    |    |  <- Selective State Space |
|  +----------+    |                           |
|       |          |                           |
|       +----*-----+  <- Gating                |
|            |                                 |
|            v                                 |
|  +----------------+                          |
|  | Output Project |                          |
|  | (d_in -> d)    |                          |
|  +----------------+                          |
|            |                                 |
|       + Residual                             |
+---------------------------------------------+
         |
         v
+-------------------+
|   Final RMSNorm   |
+-------------------+
         |
         v
+-------------------+
|    LM Head        |  <- Tied with embedding
| (d_model -> vocab)|
+-------------------+
         |
         v
Output Logits (batch, seq_len, vocab)
```

### SSM (Selective State Space) Detail

```
x(t) --> [dt, B, C projection] --> Discretization --> State Update --> Output
                |                        |                |
                v                        v                |
         Input-dependent           A_bar = exp(dt*A)      |
         parameters                B_bar = dt * B         |
                                        |                 |
                                        v                 v
                              h(t) = A_bar * h(t-1) + B_bar * x(t)
                                        |
                                        v
                              y(t) = C(t) @ h(t) + D * x(t)
```

### Mamba Key Innovations

1. **Selective SSM**: Input-dependent state transitions
2. **O(1) Inference**: Constant time per token with caching
3. **Linear Training**: O(N) complexity for training
4. **Hardware-Aware**: Optimized for modern GPUs

---

## Orchestrator Design

### Routing Decision Flow

```
Task
  |
  v
+-------------------+
| Analyze Task      |
|-------------------|
| - complexity      |
| - context_length  |
| - requires_iter   |
| - requires_stream |
+-------------------+
  |
  v
+-------------------+
| Match Capabilities|
|-------------------|
| For each arch:    |
| - task_match?     |
| - constraints_ok? |
| - compute_score   |
+-------------------+
  |
  v
+-------------------+
| Select Best       |
|-------------------|
| - primary         |
| - fallback        |
| - confidence      |
+-------------------+
  |
  v
RoutingDecision
```

### Task-to-Architecture Affinity

```python
TASK_AFFINITY = {
    "code_repair":         ["trm", "rlm"],
    "iterative_refinement": ["trm"],
    "code_generation":     ["rlm", "mamba"],
    "code_analysis":       ["mamba", "trm"],
    "long_context":        ["mamba"],
    "streaming":           ["mamba"],
    "reasoning":           ["trm"],
    "puzzle_solving":      ["trm"],
    "text_generation":     ["mamba", "rlm"],
}
```

### Capability Scoring

Score is computed from multiple factors:

```
score = 0.4 * task_match
      + 0.2 * strength_match
      + 0.2 * resource_efficiency
      + 0.2 * context_headroom
```

---

## Data Flow

### Request Processing

```
Client                 Orchestrator              Architecture
  |                         |                         |
  |  Request(task, data)    |                         |
  |------------------------>|                         |
  |                         |                         |
  |                   [Create Task]                   |
  |                   [Route Task]                    |
  |                         |                         |
  |                         |  execute(arch, task)    |
  |                         |------------------------>|
  |                         |                         |
  |                         |        result           |
  |                         |<------------------------|
  |                         |                         |
  |  Response(output)       |                         |
  |<------------------------|                         |
```

### Batch Processing Flow

```
Files[1..N]
     |
     v
+------------------+
| Create Tasks     |  <- One task per file
+------------------+
     |
     v
+------------------+
| Group by Arch    |  <- Batch compatible tasks
+------------------+
     |
     v
+------------------+
| Parallel Exec    |  <- Process batches
+------------------+
     |
     v
+------------------+
| Merge Results    |
+------------------+
     |
     v
Results[1..N]
```

---

## Design Decisions

### Why TRM for Code Repair?

| Feature | TRM | Transformer | Mamba |
|---------|-----|-------------|-------|
| Iterative refinement | Native | No | No |
| Parameter efficiency | High (weight sharing) | Low | Medium |
| Early stopping | Yes | No | No |
| Grid representation | Native | Adapted | Sequence |
| Interpretability | High | Medium | Low |

TRM's recursive nature matches the iterative process of debugging code.

### Why Mamba for Long Context?

| Feature | Mamba | Transformer | TRM |
|---------|-------|-------------|-----|
| Training complexity | O(N) | O(N^2) | O(N*iter) |
| Inference per token | O(1) | O(N) | O(iter) |
| Max context | 100K+ | ~8K (practical) | ~3K |
| Memory usage | Low | High | Low |

Mamba's linear complexity enables processing much longer sequences.

### Orchestration Benefits

1. **Optimal Routing**: Each task goes to the best architecture
2. **Graceful Fallback**: Failures don't break the system
3. **Resource Efficiency**: Use smallest capable model
4. **Extensibility**: Easy to add new architectures

### Hybrid Architecture Advantages

```
       Long Context Input
             |
             v
       [Mamba Encoder]  <- O(N) encoding
             |
             v
       [RLM Decomposer] <- Break into subtasks
             |
             v
       [TRM Refiner]    <- Iterative fix per subtask
             |
             v
       Repaired Output
```

This hybrid approach combines:
- Mamba's efficient context encoding
- RLM's structured decomposition
- TRM's iterative refinement

---

## Memory Layout

### TRM Memory (Code Repair)

```
Component          | Shape                    | Memory (fp32)
-------------------|--------------------------|---------------
Embedding          | [32768, 256]             | 32MB
Pos Encoding       | [64, 48, 256]            | 3MB
Blocks (x6)        |                          |
  - Attention      | [256, 256*3]             | 0.8MB
  - FFN            | [256, 1024] + [1024,256] | 2MB
Output Head        | [256, 32768]             | 32MB
-------------------|--------------------------|---------------
Total              |                          | ~70MB
```

### Mamba Memory

```
Component          | Shape                    | Memory (fp32)
-------------------|--------------------------|---------------
Embedding          | [50257, 768]             | 154MB
Layers (x24)       |                          |
  - in_proj        | [768, 768*2]             | 4.5MB
  - conv1d         | [768, 1, 4]              | 12KB
  - x_proj         | [768, 48+32]             | 245KB
  - dt_proj        | [48, 768]                | 147KB
  - A_log          | [768, 16]                | 49KB
  - out_proj       | [768, 768]               | 2.4MB
-------------------|--------------------------|---------------
Total              |                          | ~350MB
```

---

## Extension Points

### Adding a New Architecture

1. Define capability:
```python
capability = ArchitectureCapability(
    name="NewArch",
    supported_tasks=["..."],
    max_context_length=...,
    ...
)
```

2. Implement factory:
```python
def load_new_arch():
    return NewArchModel(config)
```

3. Register:
```python
registry.register("new_arch", capability, load_new_arch)
```

### Adding a New Technique

1. Define technique:
```python
index.register_technique(
    "new_technique",
    architectures=["trm", "mamba"],
    configs={
        "trm": {...},
        "mamba": {...},
    }
)
```

2. Implement execution logic in `_execute_{arch}` methods

---

## Performance Characteristics

### TRM Performance

| Input Size | Iterations | Latency (GPU) | Memory |
|------------|-----------|---------------|--------|
| 64x48      | 8         | ~50ms         | 200MB  |
| 64x48      | 4 (early) | ~25ms         | 200MB  |

### Mamba Performance

| Context | Training | Inference (cached) | Memory |
|---------|----------|-------------------|--------|
| 1K      | 10ms     | 1ms/token         | 500MB  |
| 8K      | 80ms     | 1ms/token         | 600MB  |
| 32K     | 320ms    | 1ms/token         | 800MB  |

### Comparative Summary

```
             Latency        Memory        Context
TRM:         Medium         Low           Short
Mamba:       Fast           Medium        Very Long
RLM:         Slow           Medium        Medium
Orchestrator: +5ms overhead
```
