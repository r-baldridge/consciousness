# ML Research System Overview

How every component of `ml_research/` fits together — from the 1943 McCulloch-Pitts neuron to 2025 agent frameworks.

---

## 1. The Three Layers

The system is organized into three conceptual layers that build on each other:

```
Layer 3: AGENT INFRASTRUCTURE        (agent_frameworks/)
         How to build and coordinate autonomous agents
              │
              ▼ provides execution substrate for
Layer 2: APPLICATION PATTERNS         (ml_techniques/)
         How to use models effectively (49 composable techniques)
              │
              ▼ applies patterns to
Layer 1: ARCHITECTURES & MODELS       (modern_dev/ + historical eras)
         What the models are (200+ methods, 12 modern architectures)
```

Each layer depends on the one below it, and the **core/** registry ties them together through cross-references.

---

## 2. Component Map

### 2.1 Historical Research Index (200+ methods, 5 eras)

These modules provide the intellectual genealogy of every modern architecture:

| Module | Era | Years | Key Methods |
|--------|-----|-------|-------------|
| `foundations/` | 1 | 1943-1980 | McCulloch-Pitts, Perceptron, Hebbian, Hopfield |
| `classical/` | 2 | 1980-2006 | MLP, CNN, SVM, RBF, LSTM, GRU, Random Forest |
| `deep_learning/` | 3 | 2006-2017 | AlexNet, ResNet, GAN, VAE, U-Net, YOLO |
| `attention/` | 4 | 2017-present | Transformer, BERT, GPT, ViT, CLIP, Flash Attention |
| `novel/` | 5 | 2023+ | Mamba, MoE, KAN, Liquid Networks, NAS |
| `reinforcement/` | Cross-era | 1989-present | Q-Learning, DQN, PPO, SAC, RLHF |
| `optimization/` | Cross-era | 1847-present | SGD, Adam, AdamW, Lion, LR schedulers |

Each method is a Python module with:
- Mathematical formulation and implementation
- Paper reference and historical context
- Predecessors and successors (lineage)
- Benchmark results

### 2.2 Core Registry (`core/`)

The core module is the **central nervous system** connecting everything:

```
core/
├── taxonomy.py          # Enums: MethodEra, MethodCategory, MethodLineage
│                        # Dataclasses: MLMethod, Paper, Benchmark
├── method_registry.py   # MethodRegistry: register, search, filter by era/category
├── unified_index.py     # UnifiedIndex: cross-references across all components
├── timeline.py          # Historical progression tracking
├── lineage_tracker.py   # Graph of method evolution and influence
├── paper_index.py       # Landmark papers database
└── benchmark_tracker.py # SOTA tracking across datasets
```

**UnifiedIndex** is the key integration point. It maintains three critical mappings:

```python
# Which modern architectures evolved from which historical methods
LINEAGE_TO_ARCHITECTURE = {
    'transformer': ['ctm', 'titans', 'ring_attention'],
    'lstm':        ['xlstm', 'rwkv', 'griffin'],
    'state_space': ['mamba_impl', 'hyena'],
    'vae':         ['flow_matching', 'consistency_models'],
}

# Which techniques work well with which architectures
ARCHITECTURE_TECHNIQUES = {
    'ctm':        ['chain_of_thought', 'recursive_decomposition', 'self_reflection'],
    'mamba_impl': ['streaming', 'long_context', 'efficient_inference'],
    'xlstm':      ['sequential_reasoning', 'time_series', 'streaming'],
}

# What capabilities each technique requires
TECHNIQUE_REQUIREMENTS = {
    'chain_of_thought': {'reasoning': True, 'min_context': 2048},
    'tool_calling':     {'function_calling': True, 'structured_output': True},
}
```

### 2.3 Modern Architectures (`modern_dev/`)

Twelve state-of-the-art architectures (2023-2025) organized by readiness:

**Tier 1 — Production Ready:**
| Architecture | Origin | Innovation |
|-------------|--------|------------|
| CTM | Sakana AI 2025 | Neural dynamics with decoupled internal time |
| JEPA | Meta 2023 | Self-supervised vision via latent prediction |
| xLSTM | NXAI Lab 2024 | Exponential gating + matrix memory |
| RWKV | BlinkDL 2023 | RNN-Transformer hybrid, O(n) complexity |
| Griffin | DeepMind 2024 | Gated linear recurrence + local attention |
| Mamba | CMU/Princeton 2023 | Selective state spaces, hardware-aware scan |

**Tier 2 — Research Ready:**
| Architecture | Origin | Innovation |
|-------------|--------|------------|
| TTT | Stanford/NVIDIA 2024 | Learnable hidden states that adapt at inference |
| Hyena | Hazy Research 2023 | Long convolutions, subquadratic attention |
| Consistency Models | OpenAI 2023 | Fast generative models via learned consistency |
| Flow Matching | Meta/DeepMind 2024 | Optimal transport for generation |
| Ring Attention | UC Berkeley 2023 | Distributed attention for infinite context |
| Titans | Google 2025 | Meta in-context memory learning at test time |

Each architecture lives in its own directory with `src/`, `configs/`, `tests/`, `docs/`.

**The Orchestrator** (`modern_dev/orchestrator/`) selects and executes architectures:

```
User Task
    ↓
Orchestrator.run(task_type, input_data, constraints)
    ↓
select_architecture()           ← Scores all architectures on task fit
    ↓
ArchitectureLoader.instantiate()  ← Dynamic module loading
    ↓
Execute (dispatch by task type)
    ↓
TaskResult (output, timing, memory)
```

The orchestrator uses LRU memory management (max N loaded models) and supports 15 task types: image classification, text generation, reasoning, video understanding, etc.

### 2.4 Application Techniques (`ml_techniques/`)

49 composable patterns across 8 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **Decomposition** | 3 | Recursive decomposition, least-to-most |
| **Prompting** | 10 | Chain-of-Thought, Tree-of-Thought, few-shot |
| **Agentic** | 10 | ReAct, tool calling, multi-agent, planning |
| **Memory** | 3 | RAG, episodic memory, context compression |
| **Code Synthesis** | 7 | RLM, program synthesis, self-debugging |
| **Orchestration** | 3 | Task routing, ensemble, hooks |
| **Verification** | 6 | Self-eval, chain-of-verification, constitutional |
| **Optimization** | 7 | DSPy, OPRO, prompt breeding |

All techniques inherit from `TechniqueBase` and support composition:

```python
# Sequential: output of A feeds into B
pipeline = chain_of_thought >> self_consistency >> verification

# Parallel: all run on same input, results aggregated
ensemble = react_agent | planning_agent | reflexion_agent
```

**Integration layer** (`ml_techniques/integration/`) bridges techniques to modern architectures:
- `trm_techniques.py` — TRM-backed decomposition and code repair
- `ctm_techniques.py` — CTM-backed temporal reasoning and verification
- `mamba_techniques.py` — Mamba-backed long-context RAG and streaming

### 2.5 Agent Frameworks (`agent_frameworks/`)

Complete infrastructure for building AI coding assistants, synthesizing patterns from Aider, OpenCode, HumanLayer, and Clawdbot:

```
agent_frameworks/
├── core/           # AgentBase, state machine, composition operators
├── backends/       # 7 LLM backends (Anthropic, OpenAI, Ollama, vLLM, LiteLLM, Router)
├── tools/          # 18 built-in tools + permission system
├── context/        # AST-based repo mapping, semantic index, file selection
├── execution/      # Architect-editor pattern, session management, sandboxing
├── human_loop/     # Approval workflows, escalation chains, feedback collection
├── orchestration/  # Gateway, agent routing, event bus, workspaces
├── memory/         # Context window, episodic, semantic, checkpoints
├── auditor/        # Framework analysis and pattern extraction
└── integrations/   # LangChain, CrewAI, AutoGen, Claude SDK adapters
```

### 2.6 Shared Data Infrastructure (`modern_dev/shared/`)

Provides common schemas and data loaders:
- `CanonicalCodeSample` — Superset format for code repair training data
- `BUG_TAXONOMY` — 100+ bug types organized by category and difficulty
- Architecture-specific loaders (TRM, CTM) with curriculum learning stages
- Quality tiers (GOLD, SILVER, BRONZE) for training data

### 2.7 Configuration (`config/`)

YAML files providing indexed metadata:
- `methods.yaml` — 200+ method definitions
- `papers.yaml` — Landmark paper references
- `benchmarks.yaml` — SOTA benchmark data
- `lineages.yaml` — Method evolution relationships

---

## 3. How Components Connect

### 3.1 The Full Data Flow

Starting from a user task and tracing through all layers:

```
USER TASK: "Fix this buggy Python function"
    │
    ▼
┌─────────────────────────────────────────┐
│  AGENT FRAMEWORKS (Layer 3)             │
│                                         │
│  AgentBase receives task                │
│  StateMachine: IDLE → PLANNING          │
│  ArchitectEditor: plan without side     │
│    effects (architect mode)             │
│  ApprovalGate: human reviews plan       │
│  StateMachine: PLANNING → EXECUTING     │
│  ToolExecutor: sandboxed execution      │
│                                         │
│  Uses LLM backends for intelligence     │
│  Records in ConversationHistory         │
│  Checkpoints state for recovery         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  ML TECHNIQUES (Layer 2)                │
│                                         │
│  Pipeline: decompose >> react >> verify │
│                                         │
│  1. RecursiveDecomposition:             │
│     Break repair into subproblems       │
│  2. ReAct:                              │
│     Thought-action-observation loop     │
│     Tool calls: read file, grep, edit   │
│  3. SelfConsistency:                    │
│     Multiple samples, vote on best      │
│  4. ChainOfVerification:               │
│     Verify the fix is correct           │
│                                         │
│  Returns TechniqueResult with trace     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  ARCHITECTURES (Layer 1)                │
│                                         │
│  Orchestrator selects best model:       │
│  • Mamba: encode long file context      │
│  • TRM: recursive code repair           │
│  • CTM: verify via neural dynamics      │
│                                         │
│  Or external LLM backend:              │
│  • Claude (Anthropic) for planning      │
│  • GPT for alternative perspective      │
│  • Ollama for local inference           │
│                                         │
│  RouterBackend: intelligent routing     │
│  with fallback, cost-awareness          │
└─────────────────────────────────────────┘
```

### 3.2 Registry Connections

Every component registers itself and can be discovered:

```
MethodRegistry (core/)
    ├── 200+ MLMethod entries
    ├── Searchable by era, category, lineage, keyword
    └── Cross-referenced via UnifiedIndex
            │
            ├─→ ARCHITECTURES list (modern_dev/)
            │       16 ArchitectureIndex entries
            │       Discoverable by TaskType, tier, status
            │
            ├─→ TECHNIQUE_INDEX (ml_techniques/)
            │       49 TechniqueIndex entries
            │       Composable via >> and | operators
            │
            └─→ AgentRegistry (agent_frameworks/)
                    Agent classes registered by ID
                    Discoverable by framework, tags
```

### 3.3 Shared Design Patterns

All modules consistently use:

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Singleton Registry** | MethodRegistry, ToolRegistry, AgentRegistry | Global discovery without coupling |
| **Dataclass + Enum** | MLMethod, ArchitectureIndex, TechniqueIndex | Type-safe metadata with validation |
| **Composition Operators** | `>>` (sequential), `\|` (parallel) | Build complex workflows from simple parts |
| **Abstract Base Classes** | ArchitectureBase, TechniqueBase, AgentBase | Consistent interfaces for extension |
| **Factory Methods** | `Task.create()`, `ToolResult.ok()` | Clean object construction |
| **Builder Pattern** | StateMachineBuilder, LLMConfig.with_*() | Fluent configuration |
| **Lazy Loading** | Orchestrator model loading, technique imports | Memory efficiency |
| **Hook System** | TechniqueBase pre/post hooks, agent callbacks | Extensibility without inheritance |

---

## 4. The Pipelines

### 4.1 Code Repair Pipeline (modern_dev/pipelines/)

The flagship end-to-end pipeline chains three architectures:

```
Input: Buggy code + error message + surrounding context
    ↓
1. Mamba Encoder
   Encodes surrounding file context with O(n) complexity
   Handles 200K+ tokens of codebase context
    ↓
2. RLM Decomposer
   Breaks repair task into sub-problems
   Identifies variable constraints and dependencies
    ↓
3. TRM Refiner
   Recursive refinement (8 cycles, 42 effective depth)
   Only 2 actual network layers, 7M parameters
   Early stopping when confidence threshold met
    ↓
Output: Repaired code + confidence score + explanation
```

### 4.2 Technique Integration Pipeline

The `ml_techniques/integration/` module connects techniques directly to architectures:

```python
# TRM-backed decomposition
trm_decomposer = TRMDecomposer(model=trm_model)
result = trm_decomposer.run("complex reasoning task")

# Mamba-backed long-context RAG
mamba_rag = MambaRAG(model=mamba_model, context_limit=100000)
result = mamba_rag.run("query", documents=large_corpus)

# CTM-backed verification
ctm_verifier = CTMVerification(model=ctm_model)
result = ctm_verifier.run(candidate_answer, original_question)
```

---

## 5. The Backends

### 5.1 LLM Backend Hierarchy

Two backend systems exist (for different purposes):

**`backends/` (top-level)** — Simple adapters for the historical research modules:
- MockBackend for testing
- LocalModelBackend for PyTorch models

**`agent_frameworks/backends/`** — Full-featured production backends:
- AnthropicBackend (Claude, with tool use, vision, streaming)
- OpenAIBackend (GPT, with function calling, embeddings)
- OllamaBackend (local models, model management)
- VLLMBackend (high-throughput local inference)
- LiteLLMBackend (100+ providers via unified interface)
- RouterBackend (intelligent multi-backend routing)

The RouterBackend supports 7 strategies:
- **Primary/Fallback**: Try primary, fall back on error
- **Round-Robin**: Rotate through backends
- **Least-Latency**: Route to fastest measured backend
- **Cost-Aware**: Route by task complexity + pricing
- **Weighted**: User-assigned preference weights

### 5.2 Backend Selection Logic

```
Agent needs LLM call
    ↓
RouterBackend.complete(messages, config)
    ↓
_select_backends() by strategy
    ├── Check health (circuit breaker: 3 consecutive failures = unhealthy)
    ├── Score by strategy (latency, cost, weight, etc.)
    └── Return ordered list
    ↓
Try first backend
    ├── Success → update stats, return response
    └── RateLimitError → try next backend
    ↓
Fallback chain until success or exhaustion
```

---

## 6. Configuration & YAML Files

The `config/` directory provides static metadata that modules load at initialization:

```yaml
# methods.yaml — Method definitions
transformer:
  name: "Transformer"
  year: 2017
  era: ATTENTION
  category: ARCHITECTURE
  authors: ["Vaswani et al."]
  paper: "Attention Is All You Need"
  key_innovation: "Self-attention replacing recurrence entirely"

# lineages.yaml — Evolution relationships
attention_line:
  - bahdanau_attention_2014
  - self_attention_2016
  - transformer_2017
  - gpt_2018
  - bert_2018

# benchmarks.yaml — SOTA tracking
imagenet:
  dataset: "ImageNet-1K"
  metric: "Top-1 Accuracy"
  current_sota: {model: "EVA-02", value: 90.0}
```

---

## 7. Test Infrastructure

```
tests/
├── conftest.py           # MockTensor, MockBackend, MockOrchestrator fixtures
├── test_integration.py   # Full pipeline integration tests
└── (per-module tests in each subdirectory)
```

Key fixtures:
- `MockTensor`: Simulates PyTorch tensors without GPU dependency
- `MockBackend`: Deterministic LLM responses for reproducible testing
- `MockOrchestrator`: Architecture selection without model loading

Custom markers: `@pytest.mark.slow`, `@pytest.mark.pytorch`, `@pytest.mark.gpu`

---

## 8. Summary: What Goes Where

| "I want to..." | Start here |
|----------------|-----------|
| Look up a historical method | `core/method_registry.py` → `MethodRegistry.search()` |
| See what evolved from transformers | `core/unified_index.py` → `LINEAGE_TO_ARCHITECTURE` |
| Run a modern architecture on a task | `modern_dev/orchestrator/` → `Orchestrator.run()` |
| Compose reasoning techniques | `ml_techniques/` → `Pipeline` or `>>` operator |
| Build an autonomous agent | `agent_frameworks/core/base_agent.py` → `AgentBase` |
| Add human approval to a workflow | `agent_frameworks/human_loop/approval.py` → `@require_approval` |
| Route between LLM providers | `agent_frameworks/backends/router_backend.py` → `RouterBackend` |
| Analyze a new framework | `agent_frameworks/auditor/auditor_agent.py` → `AuditorAgent` |
| Repair buggy code | `modern_dev/pipelines/code_repair.py` → `CodeRepairPipeline` |
| Train on code repair data | `modern_dev/shared/data/` → `TRMDataLoader` or `CTMDataLoader` |
