# Agent and Subagent Interaction Guide

How agents, subagents, and orchestration systems interact across the ml_research codebase.

---

## 1. Agent Systems at a Glance

There are **three distinct agent-like systems** in the codebase, each at a different abstraction level:

```
┌────────────────────────────────────────────────────────────┐
│  1. AGENT FRAMEWORKS  (agent_frameworks/)                  │
│     Full autonomous agents with state, memory, tools,      │
│     human approval, and multi-agent orchestration          │
│                                                            │
│     Key class: AgentBase                                   │
│     Execution: async run(task) → TaskResult                │
│     Composition: agent_a >> agent_b | agent_c              │
└────────────────────────┬───────────────────────────────────┘
                         │ agents can use techniques as
                         │ reasoning strategies
                         ▼
┌────────────────────────────────────────────────────────────┐
│  2. AGENTIC TECHNIQUES  (ml_techniques/agentic/)           │
│     Reasoning patterns: ReAct, Reflexion, MultiAgent,      │
│     LATS, Planning, CRITIC, ReWOO                          │
│                                                            │
│     Key class: TechniqueBase                               │
│     Execution: run(input_data, context) → TechniqueResult  │
│     Composition: technique_a >> technique_b                 │
└────────────────────────┬───────────────────────────────────┘
                         │ techniques invoke architectures
                         │ via backends
                         ▼
┌────────────────────────────────────────────────────────────┐
│  3. ARCHITECTURE ORCHESTRATOR  (modern_dev/orchestrator/)   │
│     Dynamic model selection and execution                   │
│     Routes tasks to best architecture (Mamba, TRM, CTM...) │
│                                                            │
│     Key class: Orchestrator                                │
│     Execution: run(task_type, input_data) → TaskResult     │
│     Selection: score architectures, pick best fit          │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Frameworks: Full Agent Lifecycle

### 2.1 Agent State Machine

Every agent in `agent_frameworks/` follows an 8-state lifecycle:

```
         ┌──────────┐
         │   IDLE   │ ◄─── initial state
         └────┬─────┘
              │ start
              ▼
         ┌──────────┐
    ┌───▶│ PLANNING │
    │    └────┬─────┘
    │         │ plan ready
    │         ▼
    │    ┌──────────────────┐        ┌───────────┐
    │    │    EXECUTING     │───────▶│ COMPLETED │ (terminal)
    │    └───┬──────┬───────┘        └───────────┘
    │        │      │
    │        │      │ needs approval
    │        │      ▼
    │        │  ┌────────────────┐
    │        │  │WAITING_APPROVAL│──── approved ───▶ back to EXECUTING
    │        │  └────────────────┘
    │        │
    │        │ pause
    │        ▼
    │    ┌──────────┐
    │    │  PAUSED  │──── resume ───▶ back to EXECUTING
    │    └──────────┘
    │
    │    On error from any active state:
    │    ┌──────────┐
    └────│  FAILED  │ (terminal)
         └──────────┘
```

State transitions are guarded (async guard functions can block transitions), trigger callbacks on entry/exit, and are recorded in a history log with timestamps.

### 2.2 The Architect-Editor Pattern

The central execution pattern separates **planning** from **doing**:

```
Task: "Add authentication to the API"
    │
    ▼
ARCHITECT MODE (no side effects)
    │
    │  LLM generates ExecutionPlan:
    │  ├── goal: "Add JWT authentication"
    │  ├── steps: [create middleware, add routes, update config]
    │  ├── files_to_modify: ["app.py", "routes.py"]
    │  ├── files_to_create: ["auth.py", "middleware.py"]
    │  ├── risk_assessment: "medium — modifies request pipeline"
    │  └── rollback_strategy: "revert modified files from git"
    │
    ▼
APPROVAL GATE (optional)
    │
    │  Human reviews plan via channel (Slack, email, console)
    │  Approves or denies with feedback
    │
    ▼
EDITOR MODE (executes changes)
    │
    │  For each step (topologically sorted by dependencies):
    │  1. Resolve tool (read_file, write_file, bash, etc.)
    │  2. Execute via ToolExecutor (sandboxed)
    │  3. Record ExecutionResult
    │  4. If failure → attempt rollback
    │
    ▼
TaskResult (all execution results, conversation history, timing)
```

### 2.3 Agent Composition

Agents compose via operators:

**Sequential Pipeline** (`>>`):
```python
# Output of architect feeds as input to editor
pipeline = architect_agent >> approval_gate >> editor_agent

# Internally:
# 1. architect_agent.run(task) → TaskResult
# 2. output_transformer(result) → new Task
# 3. approval_gate.run(new_task) → approved/denied
# 4. editor_agent.run(approved_task) → final TaskResult
```

**Parallel Pipeline** (`|`):
```python
# All agents receive same input, run concurrently
specialists = code_agent | test_agent | doc_agent

# Internally:
# asyncio.gather(
#     code_agent.run(task),
#     test_agent.run(task),
#     doc_agent.run(task),
# )
# Results aggregated (list or custom aggregator)
```

**Advanced Composition:**
```python
# Retry with exponential backoff
safe_agent = Retry(flaky_agent, max_attempts=3, backoff_factor=2.0)

# Conditional branching
smart_pipeline = ConditionalPipeline(
    condition=lambda task: "urgent" in task.description,
    if_true=fast_agent,
    if_false=thorough_agent,
)

# Fan-out: split task into subtasks, run in parallel
fan_out = FanOut(
    splitter=lambda task: [Task.create(sub) for sub in decompose(task)],
    agents=[worker_a, worker_b, worker_c],  # round-robin assignment
    aggregator=merge_results,
)
```

### 2.4 Multi-Agent Orchestration

The `orchestration/` module provides infrastructure for coordinating multiple agents:

```
Gateway (Central Control Plane)
    │
    ├── AgentRouter: Routes tasks to appropriate agents
    │   ├── Rule-based routing (regex patterns + conditions)
    │   ├── Task-type routing (agent declares supported types)
    │   └── Capability matching (weighted scoring)
    │
    ├── EventBus: Pub/sub inter-agent communication
    │   ├── Wildcard subscriptions ("task.*" matches "task.complete")
    │   ├── Priority-based handler ordering
    │   ├── Request-reply pattern (publish_and_wait)
    │   └── One-time subscriptions (auto-unsubscribe)
    │
    ├── Workspace: Per-agent isolated environments
    │   ├── Path security (escape prevention)
    │   ├── Resource limits (max size, max files)
    │   ├── Snapshot/restore for rollback
    │   └── Extension whitelist
    │
    └── ChannelAdapter: Multi-channel message handling
        ├── Slack, Discord, Telegram, Webhook, WebSocket
        └── MultiChannelManager: unified receive across all
```

**Dispatch patterns:**
```python
# Single task to specific agent
result = await gateway.dispatch(task, agent_id="code_agent")

# Single task, auto-select agent
result = await gateway.dispatch(task)  # router selects best

# Batch with concurrency control
results = await gateway.dispatch_batch(tasks, max_concurrent=5)

# Async (non-blocking)
await gateway.dispatch(task, wait=False)  # returns immediately
```

### 2.5 Human-in-the-Loop

The `human_loop/` module provides structured human interaction:

**Approval Workflow:**
```python
@require_approval(channel="slack", timeout=300)
async def deploy_to_production(config):
    """Blocks until human approves via Slack."""
    ...

# Auto-approve conditions skip the workflow:
@require_approval(
    channel="slack",
    auto_approve_if=lambda config: config.environment == "staging"
)
async def deploy(config):
    ...
```

**Human-as-Tool** (agents query humans mid-execution):
```python
human_tool = HumanAsTool()

# Agent can ask questions during execution:
response = await human_tool.ask_question(
    "The test suite has 3 failures. Should I fix them or skip?",
    query_type=QueryType.MULTIPLE_CHOICE,
    options=["Fix all", "Fix critical only", "Skip"]
)
```

**Escalation Chains:**
```
Level 1: Developer (5 min timeout)
    │ timeout
    ▼
Level 2: Tech Lead (10 min timeout)
    │ timeout
    ▼
Level 3: Manager (15 min timeout, broadcast to all channels)
    │ timeout
    ▼
Final Action: AUTO_DENY (or AUTO_APPROVE for lenient chains)
```

**Feedback Collection:**
When humans deny requests, the system collects structured feedback:
- Denial reason and severity
- Suggested alternatives
- Pattern detection (action denied 3+ times → generate guardrail recommendation)
- Training data export (DPO/RLHF format for model improvement)

---

## 3. Agentic Techniques: Reasoning Patterns

### 3.1 The 10 Agentic Patterns

These are **reasoning strategies** that any agent can use:

**1. ReAct** — Interleaved thinking and acting:
```
Thought: I need to find the bug in the function
Action: read_file(path="utils.py", start_line=42, end_line=60)
Observation: [file contents]
Thought: The issue is an off-by-one error on line 47
Action: edit_file(path="utils.py", old_text="i < len", new_text="i <= len")
Observation: File edited successfully
Thought: I should verify the fix works
Action: bash(command="python -m pytest tests/test_utils.py")
Observation: All tests passed
Final Answer: Fixed off-by-one error on line 47
```

**2. Planning (Plan-and-Execute):**
```
Plan: [
    Step 1: Read the failing test (no deps)
    Step 2: Read the implementation (no deps)
    Step 3: Identify the root cause (deps: 1, 2)
    Step 4: Implement fix (deps: 3)
    Step 5: Run tests (deps: 4)
]
Execute: Steps in topological order, dynamic replanning on failure
```

**3. Reflexion** — Learn from failures:
```
Attempt 1: Try fix → Tests fail
Reflection: "I only fixed the symptom, not the root cause.
             The variable is shared across threads."
Attempt 2: Try fix with reflection context → Tests pass
```

**4. MultiAgent** — 4 coordination patterns:
```
Debate:     Proposer argues → Critic challenges → Judge decides
Division:   Coordinator splits → Workers execute → Results merge
Hierarchical: Manager directs → Workers report → Manager adjusts
Peer Review: Worker A produces → Worker B reviews → Iterate
```

**5. LATS** — Tree search with UCB1 scoring (Monte Carlo Tree Search inspired)

**6. ReWOO** — Plan with tool slots, execute all at once

**7. InnerMonologue** — Self-narration during reasoning

**8. Toolformer** — Learn when/how to use tools

**9. CRITIC** — Generate → Critique → Refine cycle

**10. Tool Calling** — Structured function invocation with retry

### 3.2 Technique Composition

Techniques compose the same way agents do:

```python
from ml_techniques import compose
from ml_techniques.decomposition import RecursiveDecomposition
from ml_techniques.agentic import ReAct
from ml_techniques.verification import SelfConsistency

# Build a reasoning pipeline
pipeline = compose([
    RecursiveDecomposition(max_depth=3),   # Break into subproblems
    ReAct(backend=backend, tools=[...]),    # Solve each with tools
    SelfConsistency(samples=5),            # Verify via majority vote
])

result = pipeline.run("Complex multi-step reasoning task")
# Returns TechniqueResult with full execution trace
```

### 3.3 How Techniques Use Backends

Techniques access LLMs through a backend parameter:

```python
react = ReAct(
    backend="anthropic",    # String name resolved to backend instance
    tools=[search, edit],
    max_iterations=10,
    temperature=0.7,
)

# Internally:
# 1. Resolve "anthropic" → AnthropicBackend instance
# 2. Build prompt with conversation history
# 3. Call backend.generate(prompt) → LLM response
# 4. Parse tool calls from response
# 5. Execute tools, feed results back
# 6. Loop until final answer or max iterations
```

---

## 4. Architecture Orchestrator: Model Selection

### 4.1 Task Routing

The orchestrator selects the best architecture for each task:

```python
orchestrator = Orchestrator(device="cuda", max_loaded_models=2)

result = orchestrator.run(
    task_type="text_generation",
    input_data={"prompt": "Explain quantum computing"},
    constraints={"max_latency_ms": 1000, "max_memory_gb": 8},
)
```

**Selection scoring:**
```
For each architecture:
    score = 0
    if supports_task(task_type):        score += 1.0
    if context_length >= required:       score += 0.5
    if memory <= constraint:             score += 0.3
    if "strength" keyword matches task:  score += 0.2

Select highest score, load if not cached, execute
```

**Architecture capabilities map:**
```python
ARCHITECTURE_CAPABILITIES = {
    'mamba_impl': {
        'tasks': [TEXT_GENERATION, SEQUENCE_MODELING, LONG_CONTEXT],
        'strengths': ['efficiency', 'long_context', 'streaming'],
        'context_length': 200000,
        'memory_gb': 4.0,
    },
    'ctm': {
        'tasks': [REASONING, CONTINUOUS_DYNAMICS, PLANNING],
        'strengths': ['adaptive_computation', 'reasoning'],
        'context_length': 8192,
        'memory_gb': 2.0,
    },
    'trm': {
        'tasks': [REASONING, PLANNING],
        'strengths': ['recursive_depth', 'parameter_efficiency'],
        'context_length': 4096,
        'memory_gb': 0.5,
    },
}
```

### 4.2 The Router (Advanced Routing)

The `orchestrator/router.py` provides more sophisticated routing:

```
TaskRouter
    ├── Capability-based scoring (default)
    ├── Performance-based (historical latency data)
    ├── Resource-aware (available GPU memory)
    ├── Latency-optimized (deadline-driven)
    └── Quality-optimized (accuracy over speed)

FallbackHandler
    ├── RETRY: Same architecture, different config
    ├── FALLBACK: Next-best architecture
    └── ABORT: Return error with diagnosis
```

---

## 5. Cross-System Interactions

### 5.1 Agent → Technique → Architecture

The intended full-stack interaction:

```python
# Agent uses a technique pipeline for reasoning
class CodeRepairAgent(AgentBase):
    AGENT_ID = "code_repair"
    SUPPORTED_MODES = [AgentMode.CODE]

    async def run(self, task):
        # Use ml_technique for reasoning
        pipeline = RecursiveDecomposition() >> ReAct(
            backend=self._backend,
            tools=self._tools,
        )
        technique_result = pipeline.run(task.description)

        # Or use architecture orchestrator directly
        arch_result = orchestrator.run(
            task_type="reasoning",
            input_data={"code": task.context},
        )

        return TaskResult.success(technique_result.output)
```

### 5.2 Event-Driven Agent Communication

Agents communicate through the EventBus:

```python
bus = EventBus()

# Agent A publishes work completion
await bus.publish(Event(
    type="task.code_review.complete",
    source="code_agent",
    data={"files": ["auth.py"], "changes": 5},
))

# Agent B subscribes to code changes
@bus.on("task.code_review.*")
async def handle_review(event):
    # Run tests on changed files
    await test_agent.run(Task.create(
        f"Run tests for {event.data['files']}"
    ))

# Request-reply pattern
replies = await bus.publish_and_wait(
    Event(type="agent.capability.query", source="gateway"),
    timeout=5.0,
    expected_replies=3,
)
```

### 5.3 Tool Sharing

Both agent_frameworks and ml_techniques use tools, but through different interfaces:

```
agent_frameworks tools:
    Tool ABC → ToolSchema → ToolRegistry → ToolExecutor
    18 built-in tools with permission system

ml_techniques tools:
    ToolSpec dataclass → tool registry dict → direct execution
    Simpler interface, no permission layer

Bridge: agent_frameworks tools can wrap ml_techniques tools
    via FunctionTool or the @tool decorator
```

### 5.4 Memory Systems

```
agent_frameworks/memory/:
    ContextWindow     → Token-aware sliding window with auto-summarization
    EpisodicMemory    → Episode storage with search and disk persistence
    SemanticMemory    → Vector-based retrieval with cosine similarity
    CheckpointManager → Full state serialization for recovery

ml_techniques/memory/:
    RAG               → Retrieval Augmented Generation
    EpisodicMemory    → Episode-based recall (technique-level)
    ContextCompression → Token-efficient memory reduction
```

The agent_frameworks memory is **infrastructure** (persistent, shareable). The ml_techniques memory is a **reasoning pattern** (how to use retrieval effectively).

---

## 6. Auditor Agent: Self-Analysis

The auditor agent (`agent_frameworks/auditor/`) can analyze other frameworks:

```python
auditor = AuditorAgent(backend=claude_backend)

# Analyze a new framework from GitHub
analysis = await auditor.analyze_framework(
    FrameworkSource(url="https://github.com/new-framework/agent")
)
# Returns: components, patterns, strengths, weaknesses

# Extract reusable patterns
patterns = await auditor.extract_patterns(analysis)
# Each pattern: category, description, reusability score (0-1)

# Generate integration code
for pattern in patterns:
    code = await auditor.generate_integration(pattern)
    # Returns: production code, tests, usage example, dependencies

# Compare frameworks
report = await auditor.compare_frameworks(["langchain", "crewai", "autogen"])
# Returns: capability matrix, best-for-use-case recommendations

# Track industry evolution
evolution = await auditor.track_evolution()
# Returns: current trends, new frameworks, deprecated patterns
```

The auditor uses:
- **LLM-driven analysis** as primary (recursive decomposition of source code into ~4KB chunks)
- **Rule-based AST matching** as secondary (15 built-in patterns for common agent constructs)
- **Caching** to avoid re-analyzing the same framework

---

## 7. Integration Adapters

The `agent_frameworks/integrations/` module provides bidirectional bridges:

| Adapter | Direction | Purpose |
|---------|-----------|---------|
| `langchain_adapter.py` | Bidirectional | Our agents/tools ↔ LangChain agents/tools |
| `crewai_adapter.py` | Our → CrewAI | Create CrewAI teams from our agents |
| `autogen_adapter.py` | Our → AutoGen | Create AutoGen group chats from our agents |
| `claude_sdk_adapter.py` | Our → Claude SDK | Native Anthropic tool use integration |
| `aider_compat.py` | Drop-in | Replace Aider with our framework (/add, /drop, /ask, /code) |

```python
# Convert our agent to LangChain
from agent_frameworks.integrations import langchain_adapter
lc_agent = langchain_adapter.to_langchain_agent(our_agent)
lc_tools = [langchain_adapter.to_langchain_tool(t) for t in our_tools]

# Convert LangChain agent to ours
our_agent = langchain_adapter.from_langchain_agent(lc_agent)
```

---

## 8. Concurrency Model

All agent systems use Python's `asyncio`:

| Mechanism | Used For |
|-----------|---------|
| `asyncio.Lock` | State machine transitions, approval dict, workspace init |
| `asyncio.Event` | Signaling approval decisions, query responses |
| `asyncio.Queue` | Channel message passing, reply collection |
| `asyncio.Semaphore` | Batch dispatch concurrency limits |
| `asyncio.gather` | Parallel agent/technique execution |
| `run_in_executor` | Blocking I/O (file ops, SMTP, subprocess) |

Key design rule: all heavy operations are async. Sync code runs in executors to avoid blocking the event loop.
