# Agent Frameworks Integration Library: Comprehensive Architecture Document

**Version**: 0.1.0
**Scope**: Complete analysis of all modules, processes, and framework comparisons
**Source frameworks**: Aider, OpenCode, HumanLayer, Clawdbot

---

## Table of Contents

1. [Common Agent Processes](#1-common-agent-processes)
2. [Core Abstractions](#2-core-abstractions)
3. [LLM Backend Layer](#3-llm-backend-layer)
4. [Tool System](#4-tool-system)
5. [Context Management](#5-context-management)
6. [Execution Engine](#6-execution-engine)
7. [Human-in-the-Loop](#7-human-in-the-loop)
8. [Multi-Agent Orchestration](#8-multi-agent-orchestration)
9. [Memory Systems](#9-memory-systems)
10. [Auditor Agent](#10-auditor-agent)
11. [Framework Integrations](#11-framework-integrations)
12. [Cross-Framework Comparison](#12-cross-framework-comparison)
13. [Unique Capabilities by Framework](#13-unique-capabilities-by-framework)

---

## 1. Common Agent Processes

All four source frameworks (Aider, OpenCode, HumanLayer, Clawdbot) share a set of fundamental processes. This library abstracts these into composable modules:

### 1.1 User Input Handling

Every agent framework receives user input and routes it through a processing pipeline:

```
User Input (text/voice/API)
    -> Input Parsing (intent detection, command extraction)
    -> Context Assembly (gather relevant files, history, memory)
    -> LLM Processing (generate plan or response)
    -> Action Execution (tools, code changes, queries)
    -> Output Formatting (response to user)
```

**Implementation**: `core/base_agent.py` defines the `AgentBase.run(task)` abstract method as the universal entry point. The `Task` dataclass wraps user input with `description`, `context`, `constraints`, and `metadata`. All frameworks normalize user input to a `Task` before processing.

**How each framework handles it**:
- **Aider**: Commands like `/add`, `/code`, `/ask` parsed into task types; text passed as coding instructions
- **OpenCode**: Client/server split -- UI (client) captures input, agent (server) processes it
- **HumanLayer**: Input may originate from agents (approval requests) rather than users
- **Clawdbot**: Multi-channel input (Slack, Discord, Telegram) normalized to a common `Message` format

### 1.2 Memory and State

All frameworks maintain some form of state across interactions:

| Process | Implementation | Module |
|---------|---------------|--------|
| Conversation history | `ConversationHistory` with rolling window | `core/message_types.py` |
| Session persistence | JSON/pickle serialization to disk | `execution/session_manager.py` |
| Context window management | Token-aware sliding window with auto-summarization | `memory/context_window.py` |
| Episodic memory | Conversation episodes with search | `memory/episodic.py` |
| Semantic memory | Vector-based retrieval with embeddings | `memory/semantic.py` |
| State checkpoints | Snapshot and restore agent state | `memory/checkpoint.py` |
| Agent state machine | FSM with 8 states and guarded transitions | `core/state_machine.py` |

### 1.3 Context Management

Selecting what information to include in LLM prompts is critical across all frameworks:

```
Repository Structure (AST-based map)
    + Relevant Files (scored by query relevance)
    + Recent Changes (diffs and modifications)
    + Semantic Matches (embedding-based code search)
    = Assembled Context (token-budgeted)
```

**Implementation**: The `context/` module provides four complementary systems:
- `RepositoryMap`: Builds a compressed map of all symbols (functions, classes) via AST parsing
- `FileSelector`: Scores and ranks files using 5 signals (path, content, semantic, structure, recency)
- `DiffTracker`: Tracks file changes in-memory and via git
- `SemanticIndex`: Chunks code and enables embedding-based search

### 1.4 Tool Calling

Every agent framework provides tools for interacting with the environment:

```
LLM Response with tool_calls
    -> Tool Registry lookup
    -> Permission check (policy evaluation)
    -> Argument validation (JSON Schema)
    -> Sandboxed execution
    -> Result formatting
    -> Feed back to LLM
```

**Implementation**: `tools/` provides `Tool` base class, `ToolRegistry` singleton, and 18 built-in tools across 5 categories (file, git, shell, search, web). Permission policies (AllowAll, DenyDangerous, RequireApproval, CompositePolicy) gate execution.

### 1.5 Task Planning and Reasoning

All frameworks separate thinking from doing to varying degrees:

```
Task Description
    -> Architect Mode: Generate ExecutionPlan (no side effects)
    -> Dependency ordering (topological sort of steps)
    -> Editor Mode: Execute steps with tools
    -> Result aggregation and verification
```

**Implementation**: `execution/architect_editor.py` implements the Architect/Editor pattern. The `ArchitectEditor` class uses lower temperature (0.3) for planning and produces structured `ExecutionPlan` objects with dependency graphs.

### 1.6 Completion Checking

Determining when a task is complete:

- **State Machine**: `AgentState.COMPLETED` or `AgentState.FAILED` as terminal states
- **Plan Execution**: All steps in `ExecutionPlan` have `StepResult` objects
- **Iteration Limits**: `max_iterations` configuration prevents infinite loops
- **Success Criteria**: `TaskResult.success` boolean with aggregated execution results

### 1.7 Error Handling and Recovery

```
Error Detection
    -> Classify (rate limit, auth, context length, content filter, tool error)
    -> Recovery Strategy:
        - Rate limit: exponential backoff with jitter
        - Context overflow: auto-summarize and retry
        - Tool error: retry with modified args or skip
        - Auth error: surface to user
    -> Retry Logic (configurable attempts, backoff, pattern matching)
```

**Implementation**: `core/composition.py` provides `Retry` wrapper with `RetryConfig` (max_attempts, delay, backoff_factor, max_delay, retry_on patterns). Backend-specific errors mapped to `BackendError` hierarchy.

---

## 2. Core Abstractions

### 2.1 AgentBase (`core/base_agent.py`)

The foundational abstract class every agent inherits from.

**Class Attributes**:
- `AGENT_ID: str` -- Unique identifier for registry lookup
- `SUPPORTED_MODES: List[AgentMode]` -- ARCHITECT, EDITOR, ASK, CODE
- `REQUIRES_APPROVAL: List[str]` -- Tool names needing human approval

**Abstract Methods** (subclasses must implement):
- `async run(task: Task) -> TaskResult` -- Main execution loop
- `async plan(task: Task) -> Plan` -- Generate execution plan
- `async execute(plan: Plan) -> List[ExecutionResult]` -- Execute plan steps

**Concrete Methods**:
- `with_approval(channel)` -- Attach approval channel (returns self for chaining)
- `with_memory(memory)` -- Attach memory system
- `with_tools(tools)` -- Attach tool provider
- `with_mode(mode)` -- Set operating mode
- `store_memory(key, value)` -- Persist to memory backend
- `retrieve_memory(key)` -- Retrieve from memory
- `search_memory(query, limit)` -- Semantic search in memory
- `invoke_tool(name, args)` -- Execute a tool
- `list_tools()` -- Get available tool schemas

**Operator Overloading**:
- `agent1 >> agent2` creates `SequentialPipeline` (output feeds forward)
- `agent1 | agent2` creates `ParallelPipeline` (concurrent execution)

**SimpleAgent** provides a concrete implementation for testing with a pluggable `handler` callable.

### 2.2 Message Protocol (`core/message_types.py`)

**AgentMessage** is the universal message format:

```python
AgentMessage:
    role: MessageRole       # USER, ASSISTANT, SYSTEM, TOOL, HUMAN
    content: str            # Message body
    tool_calls: List[ToolCall]       # Requested tool invocations
    tool_results: List[ToolResult]   # Tool execution outcomes
    metadata: Dict[str, Any]         # Extensible metadata
    approval_status: ApprovalStatus  # PENDING, APPROVED, DENIED, TIMEOUT
    timestamp: datetime
```

**ConversationHistory** manages message sequences with optional rolling window (`max_messages`). Supports filtering by role, retrieving last N messages, and serialization.

### 2.3 State Machine (`core/state_machine.py`)

Eight-state FSM governing agent lifecycle:

```
IDLE -> PLANNING -> EXECUTING -> COMPLETED
                 \-> WAITING_APPROVAL -/
                 \-> PAUSED -> EXECUTING
                 \-> FAILED
                 \-> CANCELLED
```

**Features**:
- Guarded transitions: async guard functions must return True before transition
- Entry/exit callbacks: run async hooks on state changes
- History tracking: records all transitions with timestamps and metadata
- Builder pattern: `StateMachineBuilder` for fluent FSM configuration
- Thread-safe: `asyncio.Lock` protects all state changes

### 2.4 Composition (`core/composition.py`)

Six composition operators for building agent workflows:

| Operator | Class | Behavior |
|----------|-------|----------|
| `>>` | `SequentialPipeline` | Chain agents, output transforms to next input |
| `\|` | `ParallelPipeline` | Run concurrently via `asyncio.gather()`, aggregate results |
| gate | `ApprovalGate` | Insert human approval checkpoint |
| retry | `Retry` | Wrap with exponential backoff retry |
| if/else | `ConditionalPipeline` | Route to different agents based on condition |
| fan-out | `FanOut` | Split task into sub-tasks, run in parallel |

**SequentialPipeline** uses a configurable `output_transformer` to convert one agent's `TaskResult` into the next agent's `Task`. Default: use output string as next task description.

**ParallelPipeline** supports `fail_fast` mode (abort on first failure) and custom `result_aggregator` functions. Partial success is tracked in metadata.

### 2.5 Agent Registry (`core/agent_registry.py`)

Singleton registry with thread-safe double-check locking:

- `register_agent(agent_class, agent_id, framework, tags)` -- Register with metadata
- `get_agent(agent_id, instantiate=True)` -- Retrieve and optionally instantiate
- `list_agents(framework, tags)` -- Query with filters
- `@register` decorator -- Auto-register at import time

Frameworks can also be registered with `FrameworkConfig` including version, description, default agent, and settings.

---

## 3. LLM Backend Layer

### 3.1 Unified Interface (`backends/backend_base.py`)

**LLMBackend** abstract class defines the provider-agnostic interface:

```python
class LLMBackend(ABC):
    async complete(messages, config) -> LLMResponse    # Single response
    async stream(messages, config) -> AsyncIterator     # Streaming
    async embed(text) -> List[float]                    # Embeddings
    supports_tools: bool                                # Tool calling
    supports_vision: bool                               # Multimodal
    supports_streaming: bool
    async health_check() -> bool
    async count_tokens(text) -> int
    async close()
```

**LLMConfig** unifies configuration: model, temperature, max_tokens, top_p, stop_sequences, tools, tool_choice, system_prompt.

**LLMResponse** standardizes output: content, tool_calls, usage (input/output tokens), model, finish_reason (STOP, LENGTH, TOOL_USE, CONTENT_FILTER, ERROR).

**Exception hierarchy**: BackendError -> AuthenticationError, RateLimitError (retry_after), ModelNotFoundError, ContextLengthError, ContentFilterError.

### 3.2 Backend Implementations

| Backend | Provider | Key Features |
|---------|----------|--------------|
| `AnthropicBackend` | Claude (cloud) | Tool use blocks, streaming, vision, rate limit retry with jitter |
| `OpenAIBackend` | GPT (cloud) | Function calling, tiktoken counting, batch embeddings |
| `OllamaBackend` | Local models | Model management (pull/delete), streaming, multimodal (llava) |
| `VLLMBackend` | Local vLLM | OpenAI-compatible API, high throughput, health checks |
| `LiteLLMBackend` | 100+ providers | Auto-detection from model name, cost tracking |
| `RouterBackend` | Multi-backend | Intelligent routing with 7 strategies |

### 3.3 Router Backend (Intelligent Routing)

**7 Routing Strategies**:
- `PRIMARY_FALLBACK`: Try primary, fall back on error
- `ROUND_ROBIN`: Rotate through backends
- `LEAST_LATENCY`: Select by measured response time
- `LEAST_COST`: Cheapest backend first
- `COST_AWARE`: Route by task complexity (simple->cheap, complex->expensive)
- `RANDOM`: Random selection
- `WEIGHTED`: User-assigned weights

**Health Management**:
- Circuit breaker: 3 consecutive failures marks backend unhealthy
- Background health check loop (60s default interval)
- Per-backend statistics: success/fail counts, latency, token usage
- Auto-recovery when backend returns to health

**Cost Model**: Built-in pricing for major models (GPT-4o, Claude Sonnet, etc.), local models priced at 0.

---

## 4. Tool System

### 4.1 Tool Base (`tools/tool_base.py`)

**Tool** abstract class:
- `schema: ToolSchema` -- Name, description, JSON Schema parameters, permissions
- `async execute(**kwargs) -> ToolResult` -- Execute the tool
- `validate_args(args)` -- JSON Schema validation
- Permissions: READ, WRITE, EXECUTE, NETWORK, DANGEROUS

**ToolResult** with factory methods: `ToolResult.ok(output, **metadata)`, `ToolResult.fail(error, **metadata)`.

Type-to-schema conversion: Python type hints automatically mapped to JSON Schema (str->string, int->integer, Optional[X]->nullable, List[X]->array, Dict->object).

### 4.2 Tool Registry (`tools/tool_registry.py`)

Singleton `ToolRegistry`:
- `register(tool, override=False)` -- Add tool
- `get(name)` -- Retrieve by name
- `get_by_permission(perm)` -- Filter by permission
- `get_safe_tools()` -- All without DANGEROUS
- `get_schemas()` -- Export all schemas for LLM

**@tool decorator**: Convert async functions to tools with auto-inferred schemas:
```python
@tool(name="greet", permissions=[ToolPermission.READ])
async def greet(name: str, greeting: str = "Hello") -> ToolResult:
    return ToolResult.ok(f"{greeting}, {name}!")
```

### 4.3 Permission System (`tools/permissions.py`)

**Multi-layer access control**:

1. `PermissionDecision`: ALLOW, DENY, REQUIRE_APPROVAL
2. `UserContext`: user_id, roles, permissions (factory: `admin()`, `anonymous()`)
3. `PermissionPolicy` (5 implementations):
   - `AllowAllPolicy` -- Allow everything
   - `DenyDangerousPolicy` -- Block DANGEROUS unless admin
   - `RequireApprovalPolicy` -- Approval for WRITE/EXECUTE/DANGEROUS
   - `PermissionMatchingPolicy` -- User perms must contain tool perms
   - `CompositePolicy` -- Combine policies (any DENY wins)
4. `PermissionGrant`: Temporary or permanent grants with expiration
5. `PermissionManager`: Check, grant, revoke, audit log

### 4.4 Built-in Tools (18 total)

**File Tools** (5): ReadFileTool (line ranges, encoding), WriteFileTool (append/overwrite, auto-mkdir), EditFileTool (regex/literal search-replace), GlobTool (pattern matching, sorted by mtime), ListDirectoryTool

**Git Tools** (6): GitStatusTool (porcelain parsing), GitDiffTool (staged/unstaged/commit), GitCommitTool (staging + commit), GitBranchTool (list/create/delete/checkout), GitLogTool (structured commit parsing), GitAddTool

**Shell Tools** (3): BashTool (sandboxed execution with blocked commands/patterns, timeout, output truncation), BackgroundProcessTool (non-blocking), WorkingDirectoryTool

**Search Tools** (3): GrepTool (regex with context lines), RipgrepTool (rg CLI with JSON output, fallback to GrepTool), SemanticSearchTool (embedding-based with chunking)

**Web Tools** (3): WebFetchTool (domain allow/block, HTML text extraction, SSL), WebSearchTool (Google Custom Search API), URLParserTool (component extraction, URL joining)

**Shell Sandbox** blocks: rm, dd, shutdown, reboot, mkfs, fdisk, chmod 777, curl|bash, eval, rm -rf /, redirects to /dev/.

---

## 5. Context Management

### 5.1 Repository Map (`context/repository_map.py`)

**AST-based codebase indexing** (Aider-inspired):

**Language Support**:
- Python: Native `ast` module (zero dependencies)
- JavaScript/TypeScript: tree-sitter (optional), regex fallback for classes/functions/methods
- Go: struct/interface patterns, receiver methods
- Rust: struct/enum/trait, impl blocks

**Data Structures**:
```
FunctionSignature: name, line, params, return_type, docstring, is_async, decorators
ClassDefinition: name, line, methods[], base_classes[], class_variables[], decorators
ModuleInfo: file_path, language, imports[], classes[], functions[], global_vars[]
```

**Context Generation**:
- Token budgeting via configurable `chars_per_token` ratio (default 4.0)
- Prioritized file ordering
- Truncated docstrings (60 chars) for compression
- Symbol relationship finding via similarity scoring

**Indexing**: Concurrent parsing with semaphore (50 files), triple-indexed by file path, qualified name, and method path.

### 5.2 File Selector (`context/file_selector.py`)

**5-signal scoring algorithm**:
```
Total = (path_weight * path_score)
      + (content_weight * content_score)
      + (semantic_weight * semantic_score)
      + (structure_weight * structure_score)
      + (recency_weight * recency_score)
```

| Signal | Method | Score Range |
|--------|--------|-------------|
| Path | Filename match (1.0), contains (0.7), path contains (0.4) | 0-1 |
| Content | Logarithmic keyword frequency: min(1.0, log1p(count)/5.0) | 0-1 |
| Structure | Class name (1.0), function (0.8), method (0.5), import (0.3) | 0-1 |
| Semantic | Cosine similarity of query vs file embeddings | 0-1 |
| Recency | Linear decay over configurable window (30 days default) | 0-1 |

**6 Selection Strategies**: BALANCED, PATH_FOCUSED, CONTENT_FOCUSED, SEMANTIC, RECENT, STRUCTURE -- each with different weight distributions.

### 5.3 Diff Tracker (`context/diff_tracker.py`)

**Dual tracking**:
- In-memory: SHA256 snapshots, unified diffs via `difflib`
- Git: `git status --porcelain`, `git diff`, `git log` integration

**Change Events**: file_path, change_type (ADDED/MODIFIED/DELETED/RENAMED), diff text, lines_added/removed, parsed hunks with line numbers.

**Watch mechanism**: Async generator polling at configurable interval, automatic snapshot refresh on detection.

### 5.4 Semantic Index (`context/semantic_index.py`)

**5 chunking strategies**:
1. FIXED_SIZE: Character boundaries (1500 chars default)
2. LINE_BASED: Fixed lines per chunk (50 default)
3. SEMANTIC: AST-based for Python, regex for JS/TS, line fallback
4. SLIDING_WINDOW: Overlapping windows (200 char overlap)
5. HYBRID: Semantic units first, then sliding window for large chunks

**Embedding providers**: Abstract interface with `SimpleEmbeddingProvider` fallback (TF-IDF via feature hashing, character n-grams, L2 normalization). No external dependencies required.

**Search**: Cosine similarity with per-file deduplication (max 2 results per file), min_score threshold, top-K results.

---

## 6. Execution Engine

### 6.1 Architect/Editor Pattern (`execution/architect_editor.py`)

Separates planning from execution (OpenCode-inspired):

**Planning (Architect Mode)**:
- Temperature 0.3 for deterministic planning
- Generates structured JSON `ExecutionPlan` via LLM
- Plan includes: goal, steps, files_to_modify, files_to_create, requires_approval, reasoning, risk_assessment, rollback_strategy

**Execution (Editor Mode)**:
- Topological sort of steps by dependency graph
- Cycle detection via DFS
- Per-step tool invocation with argument passing
- Duration tracking and result aggregation

**Full workflow**: `architect_then_edit()` = plan -> optional approval callback -> execute.

### 6.2 Session Manager (`execution/session_manager.py`)

**Session lifecycle**: Created -> Active -> [Paused] -> Completed/Terminated/Error

**Features**:
- JSON or pickle persistence (per-session files)
- Auto-save background loop (60s default)
- Session forking (UUID-based copy with parent link)
- Hierarchical sessions (parent/child relationships)
- Project-based grouping and tag-based organization
- Cleanup: remove sessions older than max_age_days in terminal states

### 6.3 Tool Executor (`execution/tool_executor.py`)

**3 sandbox modes**:
- `NONE`: Direct in-process (fastest, no isolation)
- `SUBPROCESS`: Isolated process with `ulimit` resource limits (Unix)
- `DOCKER`: Full container isolation with configurable image

**Resource limits**: max_memory_mb, max_cpu_percent, max_file_size_mb, max_open_files, max_processes, network_enabled, filesystem_readonly.

**Security**: Path allowlist/blocklist, sensitive data redaction (password, secret, token, api_key), JSONL audit log with every execution.

### 6.4 Client/Server Architecture (`execution/client_server.py`)

Decoupled architecture (OpenCode-inspired):

**Protocol** (WebSocket primary, HTTP fallback):
- AUTH: Token-based authentication with constant-time comparison
- TASK: Submit task, receive single response or streaming chunks
- CANCEL: Cancel running task
- PING/PONG: Keepalive
- SUBSCRIBE/UNSUBSCRIBE: Event channel pub/sub
- Rate limiting per client (requests/minute)

**Server**: WebSocket + aiohttp HTTP, SSL/TLS, CORS, heartbeat (30s), max 100 clients.
**Client**: Auto-reconnection with exponential backoff, future-based response routing, stream queues.

---

## 7. Human-in-the-Loop

### 7.1 Approval Workflow (`human_loop/approval.py`)

**@require_approval decorator** wraps async functions with approval gates:
```python
@require_approval(channel="slack", timeout=300, auto_approve_if=is_safe)
async def deploy(config): ...
```

**Lifecycle**:
1. Auto-approve check (callable on function args)
2. Create `ApprovalRequest` with serialized arguments
3. Route to channel via `ChannelRouter`
4. Wait with timeout (`asyncio.wait_for` on Event)
5. On approval: execute function
6. On denial: raise `ApprovalDeniedError` with reason
7. On timeout: raise `ApprovalTimeoutError`

### 7.2 Human as Tool (`human_loop/human_as_tool.py`)

Implements `Tool` interface so agents can query humans mid-execution:

**Query types**: FREE_FORM, MULTIPLE_CHOICE, CONFIRMATION, NUMERIC, DATE

**Validation**: Multiple choice validates against option indices/text; confirmation validates yes/no/true/false; numeric checks min/max bounds.

**Convenience methods**: `ask_question()`, `ask_multiple_choice()`, `ask_confirmation()`, `ask_number()`.

### 7.3 Channel Router (`human_loop/channel_router.py`)

**5 channel implementations**:
- `ConsoleChannel`: stdin/stdout via asyncio executor (testing/CLI)
- `SlackChannel`: Web API with Block Kit formatting, polling for responses
- `EmailChannel`: SMTP/TLS with Markdown-to-HTML conversion
- `DiscordChannel`: Webhook with embeds and priority-based colors
- `WebhookChannel`: HTTP POST outgoing, queue-based incoming (callback pattern)

**Routing**: Custom rules (regex + conditions) evaluated by priority, fallback to default channel.

### 7.4 Escalation (`human_loop/escalation.py`)

**Configurable escalation chains**:
```
Level 1: Console (5 min) -> Level 2: Slack+Email (10 min) -> Level 3: Broadcast (15 min) -> Final: AUTO_DENY
```

**Triggers**: TIMEOUT, MANUAL, CONDITION, SCHEDULE
**Actions**: NOTIFY, TRANSFER, BROADCAST, AUTO_APPROVE, AUTO_DENY

**Predefined chains**: `create_standard_chain()` (3-level), `create_urgent_chain()` (2-level, 60s timeouts), `create_lenient_chain()` (auto-approve on exhaust).

### 7.5 Feedback Collector (`human_loop/feedback_collector.py`)

Collects denial reasons for agent improvement:

**Categories**: SAFETY, PRIVACY, ACCURACY, SCOPE, TIMING, PERMISSION, QUALITY, POLICY

**Pattern detection**: Identifies clusters of denials by action (>=3 occurrences), category, or keyword in reason.

**Training data export**: DPO/RLHF preference format, instruction format, raw format -- with confidence scoring (severity * reason quality * alternative availability).

---

## 8. Multi-Agent Orchestration

### 8.1 Gateway (`orchestration/gateway.py`)

Central control plane for managing multiple agents:

**Agent lifecycle**: Register (INITIALIZING->IDLE) -> Dispatch (IDLE->RUNNING) -> Complete (->IDLE) or Error (->ERROR) -> Unregister (cleanup)

**Dispatch modes**:
- Single synchronous: `dispatch(task, agent_id, wait=True)`
- Async queued: `dispatch(task, wait=False)` returns immediately
- Batch: `dispatch_batch(tasks, max_concurrent=5)` with semaphore

**Agent selection**: Prioritizes IDLE agents that `can_handle(task)`, falls back to any non-error agent.

**Health checks**: Background loop (60s) calling `agent.health_check()`, marks unhealthy agents as ERROR.

### 8.2 Agent Router (`orchestration/agent_router.py`)

**3-tier routing**:
1. Priority rules: Regex pattern + optional conditions, evaluated in priority order
2. Task type matching: Agent registered for specific task types (case-insensitive substring)
3. Capability matching: Agent capabilities vs task requirements, weighted by preference

**ConditionalRouter** extension: Async condition evaluation at runtime (load, health, state-dependent).

### 8.3 Multi-Channel Adapters (`orchestration/channel_adapter.py`)

**5 adapters**: Slack, Discord, Telegram, Webhook, WebSocket
**MultiChannelManager**: Unified receive from any channel, per-channel queues, broadcast.

### 8.4 Workspaces (`orchestration/workspace.py`)

**Per-agent isolated directories**:
- Path traversal prevention via `.resolve()` validation
- Size limits (max_size_mb), file count limits, extension whitelists
- Snapshot mechanism: Copy workspace state, SHA256 checksums, auto-cleanup of old snapshots
- Restore from any snapshot

### 8.5 Event Bus (`orchestration/event_bus.py`)

**Pub/sub with advanced features**:
- Wildcard patterns: `task.*` matches `task.complete`, `task.fail`; `*` matches all
- Priority-ordered handler invocation (higher priority first)
- One-time subscriptions (auto-unsubscribe after first event)
- **Request-reply pattern**: `publish_and_wait(event, timeout)` collects replies
- Event history ring buffer (1000 events)
- Decorator support: `@bus.on("task.complete")`
- `EventEmitter` mixin for objects that emit events

---

## 9. Memory Systems

### 9.1 Context Window (`memory/context_window.py`)

**Token-aware sliding window**:
- Tracks cumulative token count across all messages
- Summarization triggers at 80% capacity (configurable)
- Smart pruning: keeps recent N + all system messages, summarizes older
- LLM summarization when backend available, fallback to last-10-message concatenation
- Token estimation: backend `count_tokens()` or `len(text)/4` heuristic

### 9.2 Episodic Memory (`memory/episodic.py`)

**Conversation episode storage**:
- Episodes: UUID-based, message list, auto-summary, metadata/tags
- Disk persistence: `episodes/` directory + `index.json` (ordered by recency)
- Search scoring: +3.0 for query in summary, +0.5 per word match, +1.0 per message match
- LLM summarization: main topic, decisions, unresolved items, key facts

### 9.3 Semantic Memory (`memory/semantic.py`)

**Vector-based retrieval**:
- Entries: content, embedding, metadata, timestamp
- Search: cosine similarity with threshold filtering, top-K results
- **Fallback text search** when no embeddings available: full phrase match (0.8) + word overlap ratio
- Persistence: JSON with embedded vectors
- O(1) entry lookup via id-to-position index

### 9.4 Checkpoints (`memory/checkpoint.py`)

**State serialization**:
- Captures agent state via `get_state()` / `to_dict()` / `__dict__`
- Captures messages via `get_messages()` / `messages` / `context.messages`
- Auto-cleanup: keeps N most recent per agent (default 10)
- Named checkpoints for human-readable identification

---

## 10. Auditor Agent

### 10.1 Architecture Analysis (`auditor/auditor_agent.py`)

**LLM-centric framework analysis pipeline**:
```
FrameworkSource (GitHub URL / local path / docs URL)
    -> framework_fetcher.fetch_github() -- clone/cache
    -> recursive_decomposer.decompose() -- break into ~4KB chunks
    -> LLM: analyze each chunk (parallel)
    -> LLM: synthesize unified analysis
    -> rule_engine.extract() -- AST validation
    -> Merged FrameworkAnalysis
```

**Outputs**: Architecture patterns, components, strengths, weaknesses, integration points.

### 10.2 Pattern Extraction (`auditor/pattern_extractor.py`)

**10 pattern categories**: orchestration, memory, tools, execution, human_loop, context, planning, reflection, retrieval, multi_agent

**Dual extraction**:
- LLM-based: Semantic understanding of code and documentation
- Rule-based: 15 AST rules for class hierarchies, decorators, naming conventions

**Reusability scoring**: LLM evaluates generality, independence, clarity, testability, modularity (0-1 scale). Fallback heuristic adds 0.1 for each quality indicator.

### 10.3 Code Generation (`auditor/integration_generator.py`)

Generates production-ready integration code from extracted patterns:
- Main integration code
- Usage examples
- Pytest tests
- Requirements.txt with dependency extraction from imports

### 10.4 Benchmarking (`auditor/benchmark_runner.py`)

**10 capabilities benchmarked**: tool_execution, multi_agent, memory, human_loop, context_management, streaming, error_handling, extensibility, documentation, type_safety

Generates comparison matrices with best-for-use-case recommendations.

### 10.5 Evolution Tracking (`auditor/evolution_tracker.py`)

Monitors industry trends via LLM analysis:
- Trend scanning (architecture patterns, capabilities, deployment)
- New framework discovery with maturity assessment
- Deprecated pattern identification
- Strategic recommendations
- 24-hour report caching
- Reference set of 14 known frameworks

---

## 11. Framework Integrations

### 11.1 Aider Compatibility (`integrations/aider_compat.py`)

Drop-in replacement for Aider workflows:

| Aider Command | Method | Behavior |
|---------------|--------|----------|
| `/add file` | `add_file(path)` | Load file into context |
| `/drop file` | `drop_file(path)` | Remove from context |
| `/ask query` | `ask(question)` | Read-only query |
| `/code instruction` | `code(instruction)` | Modify files |
| `/run command` | Shell execution | Subprocess |

**Key features**: Repository map generation, file change detection from LLM output, auto-commit with instruction summary, system prompt with current files and repo map.

### 11.2 LangChain Adapter (`integrations/langchain_adapter.py`)

**Bidirectional conversion**:
- Our Agent -> `_LangChainAgentWrapper` (implements LangChain's `plan()` interface)
- LangChain Agent -> `_OurAgentFromLangChain` (wraps in our `AgentBase`)
- Our Tool -> LangChain `BaseTool` (wraps schema, `_run`/`_arun`)
- LangChain Tool -> Our `Tool` (extracts schema, wraps execution)

### 11.3 CrewAI Adapter (`integrations/crewai_adapter.py`)

- `to_crew_agent()`: Wraps our agent with role/goal/backstory
- `create_crew()`: Multi-agent team with task assignment
- `run_crew()`: Async wrapper around crew.kickoff()
- Supports sequential and hierarchical processes

### 11.4 Claude SDK Adapter (`integrations/claude_sdk_adapter.py`)

**Native Claude integration**:
- `ClaudeAgentWrapper`: Maintains message history, handles tool_use blocks in agentic loop
- Tool wrapping: Our `ToolSchema` -> Claude `input_schema`
- Full conversation loop: message -> tool_use -> tool_result -> continue

### 11.5 AutoGen Adapter (`integrations/autogen_adapter.py`)

- `to_autogen_agent()`: AssistantAgent or UserProxyAgent wrapper
- `create_group_chat()`: Multi-agent conversations
- `run_group_chat()`: Sequential or hierarchical speaker selection

---

## 12. Cross-Framework Comparison

### Processes Common to All Four Source Frameworks

| Process | Aider | OpenCode | HumanLayer | Clawdbot | Our Library |
|---------|-------|----------|------------|----------|-------------|
| **User Input** | CLI commands | Terminal UI | Agent-initiated | Multi-channel | `AgentBase.run(Task)` |
| **LLM Calling** | Direct API | Model-agnostic | Provider-agnostic | Multi-model | `LLMBackend.complete()` |
| **Context Assembly** | Repo map + file context | LSP + file tree | N/A | Channel history | `RepositoryMap` + `FileSelector` |
| **Tool Execution** | Built-in (edit, git) | Shell + LSP | Approval gates | Tool registry | `ToolRegistry` + 18 built-ins |
| **State Tracking** | Git history | Session state | Request lifecycle | Per-channel state | `StateMachine` + `SessionManager` |
| **Error Recovery** | Lint/test loop | Auto-retry | Escalation | Channel retry | `Retry` + backend backoff |
| **Output** | Code diffs + commits | Terminal stream | Approval decisions | Multi-channel | `TaskResult` + streaming |

### Capability Matrix

| Capability | Aider | OpenCode | HumanLayer | Clawdbot | Our Library |
|------------|-------|----------|------------|----------|-------------|
| Repository map (AST) | **Primary** | Via LSP | - | - | Full (4 languages) |
| Multi-file editing | Yes | Yes | - | - | Via tools |
| Git integration | **Auto-commit** | Basic | - | - | 6 git tools |
| Architect/Editor split | Yes | Yes | - | - | `ArchitectEditor` |
| Multi-session | - | **Primary** | - | - | `SessionManager` |
| Client/server | - | **Primary** | - | Partial | `AgentServer/Client` |
| Privacy/local-first | - | **Primary** | - | Yes | Ollama + vLLM backends |
| Model agnostic | Partial | **75+ providers** | Partial | Yes | 7 backends + router |
| Approval workflows | - | - | **Primary** | - | `@require_approval` |
| Human as tool | - | - | **Primary** | - | `HumanAsTool` |
| Channel routing | - | - | **Primary** | Partial | `ChannelRouter` |
| Escalation chains | - | - | **Primary** | - | `EscalationManager` |
| Feedback/learning | - | - | **Primary** | - | `FeedbackCollector` |
| Multi-agent routing | - | - | - | **Primary** | `Gateway` + `AgentRouter` |
| Multi-channel inbox | - | - | Partial | **Primary** | 5 channel adapters |
| Agent isolation | - | - | - | **Primary** | `Workspace` |
| Inter-agent messaging | - | - | - | **Primary** | `EventBus` |
| Voice/multimodal | - | - | - | **Primary** | Vision-capable backends |
| Semantic memory | - | - | - | - | **New** |
| Episodic memory | - | - | - | - | **New** |
| Framework auditing | - | - | - | - | **New** |

---

## 13. Unique Capabilities by Framework

### Aider: Repository Intelligence

**Unique strengths our library captures**:
- **Repository Map**: AST-based codebase context with function signatures. Our `RepositoryMap` supports Python (native ast), JS/TS, Go, Rust (tree-sitter + regex fallback). Compressed context strings with token budgeting enable including entire codebase structure in prompts.
- **Lint/Test Loop**: Aider auto-runs linters and tests after every edit, feeding failures back for correction. Our library models this as `SequentialPipeline(edit_agent >> test_agent)` with `Retry` wrapper.
- **Git Auto-Commit**: Semantic commit messages generated automatically. Our `GitCommitTool` supports this, and `AiderCompatAgent` implements the full workflow.
- **Multi-file Context**: Intelligent selection of which files to include. Our `FileSelector` uses 5-signal scoring with 6 strategies.

**What Aider lacks** that others provide:
- No human approval gates
- No multi-agent orchestration
- No multi-channel communication
- Single-session only

### OpenCode: Architectural Separation

**Unique strengths our library captures**:
- **Client/Server Decoupling**: UI completely separated from agent logic. Our `AgentServer`/`AgentClient` implement WebSocket + HTTP with auth, streaming, and pub/sub events.
- **Multi-Session**: Parallel agents working on the same project. Our `SessionManager` supports forking, hierarchical sessions, and concurrent access.
- **LSP Integration**: Language server for semantic understanding (go-to-definition, find-references). Our `SemanticIndex` has an `LSPClient` protocol for pluggable LSP backends.
- **Privacy-First**: Local-only option. Our Ollama and vLLM backends provide fully local LLM inference.
- **Model Agnostic**: 75+ providers. Our `LiteLLMBackend` + `RouterBackend` support 100+ models with intelligent routing.

**What OpenCode lacks** that others provide:
- No human approval workflows
- No multi-agent coordination
- No escalation or feedback collection
- Limited multi-channel support

### HumanLayer: Approval Infrastructure

**Unique strengths our library captures**:
- **@require_approval Decorator**: Simple annotation for functions needing human sign-off. Supports auto-approve conditions, custom channels, configurable timeouts. No other framework has this pattern.
- **Human as Tool**: Agents can query humans mid-execution using the standard Tool interface. Supports free-form, multiple choice, confirmation, and numeric queries with validation.
- **Omnichannel Routing**: Route approval requests to Slack, email, Discord, webhooks, or console based on rules. Priority-based routing with fallbacks.
- **Escalation Chains**: Multi-level escalation with configurable timeouts and auto-actions. Predefined chains (standard, urgent, lenient) for common patterns.
- **Feedback Loop**: Collects denial reasons, identifies patterns, exports RLHF training data, generates policy recommendations.

**What HumanLayer lacks** that others provide:
- No code editing or repository awareness
- No multi-agent orchestration
- No git integration
- No session management
- No codebase context

### Clawdbot: Multi-Agent Gateway

**Unique strengths our library captures**:
- **Multi-Agent Routing**: Central gateway dispatches tasks to specialized agents based on capabilities. Rule-based, capability-based, and weight-based routing strategies.
- **Multi-Channel Inbox**: Unified message handling across Slack, Discord, Telegram, webhooks, WebSockets. `MultiChannelManager` provides receive-from-any and broadcast.
- **Agent Isolation**: Per-agent workspaces with path security, size limits, snapshots, and restore. Prevents cross-contamination between agents.
- **Inter-Agent Communication**: Event bus with wildcard subscriptions, priority ordering, request-reply pattern. Enables agents to coordinate without direct coupling.
- **Voice/Multimodal**: Wake words, speech, visual canvas. Our backends support vision (Anthropic, OpenAI, Ollama with llava).

**What Clawdbot lacks** that others provide:
- No deep repository understanding (AST, repo maps)
- No sophisticated approval workflows
- No feedback collection for learning
- No architect/editor separation
- Limited context management

### Our Library: Novel Additions

**Capabilities not found in any source framework**:

1. **Architecture Auditor Agent**: LLM-powered analysis of new frameworks with recursive decomposition, AST pattern matching, integration code generation, benchmarking, and evolution tracking. Enables the library to analyze and integrate new frameworks as they emerge.

2. **Unified Memory Stack**: Three complementary memory systems (context window, episodic, semantic) plus checkpoints. No single source framework has all four.

3. **Agent Composition Operators**: `>>` (sequential), `|` (parallel), approval gates, retry, conditional, fan-out. Enables building complex workflows from simple agents using operator syntax.

4. **Intelligent Backend Routing**: 7 routing strategies including cost-aware routing that classifies task complexity and routes accordingly. Circuit breaker pattern for health management.

5. **Composable Permission Policies**: Multiple policies can be combined (CompositePolicy). Grant/revoke system with temporal expiration. Full audit logging.

6. **Cross-Framework Adapters**: Bidirectional integration with LangChain, CrewAI, AutoGen, and Claude Agent SDK. Our agents can participate in any of these ecosystems.

---

## Appendix: Data Flow Diagrams

### Complete Agent Execution Flow
```
User Input
    |
    v
[AgentBase.run(Task)]
    |
    +-- State: IDLE -> PLANNING
    |
    v
[AgentBase.plan(Task)] -> ExecutionPlan
    |
    +-- State: PLANNING -> WAITING_APPROVAL (if requires_approval)
    |
    v
[ApprovalGate] -> APPROVED/DENIED
    |
    +-- State: WAITING_APPROVAL -> EXECUTING
    |
    v
[AgentBase.execute(Plan)]
    |-- For each PlanStep (topological order):
    |     +-- PermissionManager.check_permission(tool, user)
    |     +-- ToolExecutor.execute(tool, args, sandbox)
    |     +-- AuditLog.record(entry)
    |     +-- ExecutionResult
    |
    +-- State: EXECUTING -> COMPLETED/FAILED
    |
    v
TaskResult (success, output, execution_results[], messages[], duration_ms)
```

### Memory Integration Flow
```
Agent Message
    |
    v
[ContextWindow.add()] -- token tracking + auto-summarization at 80%
    |
    v
[EpisodicMemory.save_episode()] -- on session end, with auto-summary
    |
    v
[SemanticMemory.add()] -- embed content for future retrieval
    |
    v
[CheckpointManager.save()] -- periodic state snapshots
    |
    v
[Disk Persistence] -- JSON/pickle files
```

### Approval Workflow
```
@require_approval(channel="slack", timeout=300)
    |
    v
Auto-approve check -> True: execute directly
    |                  False: continue
    v
ApprovalRequest created
    |
    v
ChannelRouter.route() -> SlackChannel.send()
    |
    v
Wait (asyncio.wait_for with timeout)
    |
    +-- APPROVED -> Execute function -> Return result
    |-- DENIED -> FeedbackCollector.collect() -> Raise ApprovalDeniedError
    +-- TIMEOUT -> EscalationManager.escalate() -> Next level or Raise ApprovalTimeoutError
```
