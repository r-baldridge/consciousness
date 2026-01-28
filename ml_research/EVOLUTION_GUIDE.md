# Evolution Guide: Extending ML Research as Technology Advances

How to add new architectures, techniques, agents, and integrations — and how to use the existing tools to stay current.

---

## 1. What the System Already Provides for Evolution

Before building anything new, leverage what exists:

### 1.1 The Auditor Agent (Automated Framework Analysis)

The auditor agent at `agent_frameworks/auditor/` can **automatically analyze new frameworks** and generate integration code:

```python
from agent_frameworks.auditor import AuditorAgent
from agent_frameworks.backends import AnthropicBackend

auditor = AuditorAgent(backend=AnthropicBackend())

# Point it at a new framework
analysis = await auditor.analyze_framework(
    FrameworkSource(url="https://github.com/new-framework/agent")
)

# Get reusable patterns
patterns = await auditor.extract_patterns(analysis)

# Auto-generate integration code with tests
for pattern in patterns:
    code = await auditor.generate_integration(pattern)
    await code.save("integrations/")
```

### 1.2 The Evolution Tracker (Industry Monitoring)

```python
# Track what's happening in the field
evolution = await auditor.track_evolution()

# Returns:
# - Current trends (10-15 identified)
# - New frameworks discovered
# - Deprecated patterns to phase out
# - Strategic recommendations
```

The tracker knows about 14 major frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, Semantic Kernel, Haystack, Guidance, DSPy, Agency Swarm, LangGraph, Pydantic AI, Marvin, Instructor, Outlines) and uses web search + LLM analysis to find new ones.

### 1.3 The Unified Index (Cross-Reference System)

The `core/unified_index.py` maintains a live map connecting historical methods to modern implementations to compatible techniques. When adding anything new, register it here to make it discoverable.

---

## 2. Adding a New Architecture

### 2.1 When to Add

Add a new architecture when:
- A paper shows significant improvement over existing approaches
- A new paradigm emerges (like SSMs replacing attention for some tasks)
- An existing indexed architecture becomes implementable (checkpoints released)

### 2.2 Step-by-Step

**Step 1: Create the directory structure**

```
modern_dev/new_architecture/
├── __init__.py          # Metadata, configs, exports
├── README.md            # Paper summary, benchmarks, usage
├── src/
│   ├── __init__.py
│   ├── model.py         # Main model class
│   ├── layers.py        # Custom layers
│   └── config.py        # Configuration dataclass
├── configs/
│   ├── default.yaml
│   └── presets.yaml     # tiny, small, medium, large
├── tests/
│   ├── __init__.py
│   └── test_model.py
└── docs/
    └── ARCHITECTURE.md
```

**Step 2: Implement `ArchitectureBase`**

```python
# new_architecture/src/model.py
from modern_dev.orchestrator import ArchitectureBase, TaskType

class NewArchitecture(ArchitectureBase):
    ARCHITECTURE_ID = "new_arch"
    SUPPORTED_TASKS = [TaskType.TEXT_GENERATION, TaskType.REASONING]
    MAX_CONTEXT_LENGTH = 32768
    MEMORY_REQUIREMENT_GB = 2.0

    def load(self, config=None):
        """Load model weights."""
        ...

    def unload(self):
        """Free GPU memory."""
        ...

    def run(self, task_spec):
        """Execute a task."""
        ...

    def is_loaded(self) -> bool:
        ...
```

**Step 3: Register with the orchestrator**

```python
# modern_dev/__init__.py — add to ARCHITECTURES list
ARCHITECTURES.append(ArchitectureIndex(
    id="new_arch",
    name="New Architecture",
    year=2025,
    authors=["Author et al."],
    organization="Lab Name",
    paper_title="Paper Title",
    paper_url="https://arxiv.org/abs/...",
    key_innovation="What makes it novel",
    tier=ImplementationTier.TIER_2,
    status=DevelopmentStatus.ALPHA,
    use_cases=["reasoning", "long-context"],
    related_architectures=["mamba_impl", "ctm"],
))
```

**Step 4: Add capability mapping**

```python
# modern_dev/orchestrator/__init__.py — add to ARCHITECTURE_CAPABILITIES
ARCHITECTURE_CAPABILITIES["new_arch"] = {
    "tasks": [TaskType.TEXT_GENERATION, TaskType.REASONING],
    "strengths": ["efficiency", "reasoning"],
    "context_length": 32768,
    "memory_gb": 2.0,
}
```

**Step 5: Update cross-references**

```python
# core/unified_index.py — add lineage and technique mappings
LINEAGE_TO_ARCHITECTURE["transformer"].append("new_arch")  # if transformer-derived
ARCHITECTURE_TECHNIQUES["new_arch"] = [
    "chain_of_thought", "recursive_decomposition"
]
```

**Step 6: Add data loader (if training)**

```python
# modern_dev/shared/data/loaders/new_arch.py
class NewArchDataLoader(BaseDataLoader):
    """Architecture-specific data loading."""
    ...
```

### 2.3 Indexing vs. Implementing

Not every architecture needs a full implementation. The codebase supports progressive depth:

| Level | What to Create | When |
|-------|---------------|------|
| **Indexed** | `__init__.py` with metadata, README | Paper is interesting, no implementation yet |
| **Stubbed** | + model.py with class structure | Want to test orchestrator routing |
| **Alpha** | + forward pass works | Core innovation implemented |
| **Beta** | + training loop, data loading | Can train from scratch |
| **Production** | + optimized, benchmarked, documented | Ready for real workloads |

---

## 3. Adding a New Technique

### 3.1 When to Add

Add a new technique when:
- A new prompting or reasoning strategy shows consistent improvement
- A new agent coordination pattern emerges
- A specialized pattern is needed for a new task domain

### 3.2 Step-by-Step

**Step 1: Identify the category**

Pick from the 8 existing categories, or create a new one if none fits:
- `decomposition/`, `prompting/`, `agentic/`, `memory/`
- `code_synthesis/`, `orchestration/`, `verification/`, `optimization/`

**Step 2: Create the technique**

```python
# ml_techniques/agentic/new_pattern.py (or add to __init__.py)
from ml_techniques import TechniqueBase, TechniqueCategory, TechniqueResult

class NewPattern(TechniqueBase):
    TECHNIQUE_ID = "new_pattern"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(self, backend=None, max_iterations=5, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.max_iterations = max_iterations

    def run(self, input_data, context=None):
        steps = []
        # Implement the reasoning pattern
        for i in range(self.max_iterations):
            # ... reasoning logic ...
            steps.append({"iteration": i, "output": output})

        return TechniqueResult(
            success=True,
            output=final_output,
            technique_id=self.TECHNIQUE_ID,
            intermediate_steps=steps,
        )
```

**Step 3: Register in the category index**

```python
# ml_techniques/agentic/__init__.py — add to exports and registry
from .new_pattern import NewPattern

AGENTIC_TECHNIQUES["new_pattern"] = {
    "class": NewPattern,
    "description": "What it does",
    "paper": "https://arxiv.org/abs/...",
}
```

**Step 4: Add architecture integration (optional)**

If the technique benefits from a specific architecture:

```python
# ml_techniques/integration/new_arch_techniques.py
class NewArchPatternIntegration(NewPattern):
    """NewPattern optimized for NewArchitecture."""

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def run(self, input_data, context=None):
        # Use architecture-specific capabilities
        ...
```

Register the mapping:
```python
# ml_techniques/integration/__init__.py
ARCHITECTURE_TECHNIQUE_MAP["new_arch"].append("new_pattern")
```

**Step 5: Verify composition works**

```python
# The technique should compose with existing ones:
pipeline = new_pattern >> self_consistency >> verification
result = pipeline.run("test input")
assert result.success
```

---

## 4. Adding a New Agent

### 4.1 When to Add

Add a new agent when:
- A new execution pattern doesn't fit existing agents
- A specialized domain requires custom tool sets or approval flows
- A new multi-agent coordination pattern is needed

### 4.2 Step-by-Step

**Step 1: Define the agent**

```python
# agent_frameworks/core/custom_agent.py (or in a new module)
from agent_frameworks.core import AgentBase, AgentMode, Task, TaskResult

class SecurityAuditAgent(AgentBase):
    AGENT_ID = "security_audit"
    SUPPORTED_MODES = [AgentMode.ASK, AgentMode.CODE]
    REQUIRES_APPROVAL = ["bash", "write_file"]  # Tools needing approval

    async def run(self, task: Task) -> TaskResult:
        plan = await self.plan(task)
        results = await self.execute(plan)
        return TaskResult.success(
            output=results,
            execution_results=results,
        )

    async def plan(self, task: Task) -> Plan:
        # Use architect mode to plan the audit
        ...

    async def execute(self, plan: Plan) -> list:
        # Execute audit steps
        ...
```

**Step 2: Register the agent**

```python
from agent_frameworks.core import register

@register(framework="security", tags=["audit", "analysis"])
class SecurityAuditAgent(AgentBase):
    ...
```

**Step 3: Wire up tools and permissions**

```python
# Configure with appropriate tools and approval
agent = SecurityAuditAgent()
agent.with_tools(security_tools)
agent.with_approval(slack_channel)
agent.with_memory(episodic_memory)
```

**Step 4: Add to composition pipelines**

```python
# Use in a pipeline
audit_pipeline = (
    code_review_agent
    >> SecurityAuditAgent()
    >> ApprovalGate(message="Deploy after security review?")
    >> deploy_agent
)
```

---

## 5. Adding a New Backend

### 5.1 When to Add

Add when a new LLM provider emerges, a local inference engine gains features, or you need a specialized routing strategy.

### 5.2 Step-by-Step

```python
# agent_frameworks/backends/new_backend.py
from agent_frameworks.backends.backend_base import (
    LLMBackend, LLMResponse, LLMConfig, ToolCall
)

class NewProviderBackend(LLMBackend):
    """Backend for NewProvider API."""

    def __init__(self, api_key=None, model="default-model"):
        self.api_key = api_key or os.environ.get("NEW_PROVIDER_API_KEY")
        self.model = model

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(self, messages, config=None):
        # Convert messages to provider format
        # Make API call
        # Parse response into LLMResponse
        ...

    async def stream(self, messages, config=None):
        # Yield text chunks
        ...

    async def embed(self, text):
        # Return embedding vector
        ...
```

Add to the RouterBackend's available backends:
```python
router = RouterBackend(
    backends=[
        ("new_provider", NewProviderBackend()),
        ("anthropic", AnthropicBackend()),
    ],
    strategy=RoutingStrategy.PRIMARY_FALLBACK,
)
```

---

## 6. Adding New Tools

### 6.1 Quick Way (Decorator)

```python
from agent_frameworks.tools import tool, ToolPermission, ToolResult

@tool(
    name="kubernetes_deploy",
    description="Deploy a container to Kubernetes",
    permissions=[ToolPermission.EXECUTE, ToolPermission.NETWORK],
)
async def kubernetes_deploy(
    image: str,
    namespace: str = "default",
    replicas: int = 1,
) -> ToolResult:
    """Deploy container image to k8s cluster."""
    # Implementation
    return ToolResult.ok({"status": "deployed", "replicas": replicas})
```

### 6.2 Full Way (Class)

```python
from agent_frameworks.tools import Tool, ToolSchema, ToolResult, ToolPermission

class DatabaseQueryTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="database_query",
            description="Execute a read-only SQL query",
            parameters={
                "query": {"type": "string", "description": "SQL query"},
                "database": {"type": "string", "description": "Database name"},
            },
            required=["query", "database"],
            permissions=[ToolPermission.READ, ToolPermission.NETWORK],
        )

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs["query"]
        if not query.strip().upper().startswith("SELECT"):
            return ToolResult.fail("Only SELECT queries allowed")
        # Execute query
        return ToolResult.ok(results)
```

---

## 7. Adding New Integrations

### 7.1 Adapter Pattern

When a new third-party framework appears:

```python
# agent_frameworks/integrations/new_framework_adapter.py

def to_new_framework_agent(our_agent):
    """Convert our agent to NewFramework format."""
    # Wrap our agent's interface
    ...

def from_new_framework_agent(their_agent):
    """Convert NewFramework agent to our format."""
    # Wrap their agent's interface
    ...

def to_new_framework_tool(our_tool):
    """Convert our tool to NewFramework format."""
    ...

def from_new_framework_tool(their_tool):
    """Convert NewFramework tool to our format."""
    ...
```

### 7.2 Use the Auditor for Automatic Integration

```python
# Let the auditor analyze and generate the integration:
analysis = await auditor.analyze_framework(
    FrameworkSource(url="https://github.com/new-framework/repo")
)
patterns = await auditor.extract_patterns(analysis)
adapter_code = await auditor.generate_adapter(
    source_framework="agent_frameworks",
    target_framework="new_framework",
    patterns=patterns,
)
```

---

## 8. Updating Historical Methods

When a new paper reinterprets or extends a historical method:

**Step 1: Update the method entry**
```python
# In the appropriate era module (e.g., attention/language_models/)
# Add new variant or update benchmarks
```

**Step 2: Update lineage**
```yaml
# config/lineages.yaml
attention_line:
  - transformer_2017
  - gpt_2018
  - new_variant_2025  # ← add here
```

**Step 3: Update cross-references**
```python
# core/unified_index.py
LINEAGE_TO_ARCHITECTURE["attention_line"].append("new_variant")
```

---

## 9. Upgrading the Pipeline System

### 9.1 Adding a New Pipeline

```python
# modern_dev/pipelines/new_pipeline.py
class NewPipeline:
    def __init__(self, config):
        self.config = config

    def process(self, request):
        # Chain architectures
        context = mamba.encode(request.context)
        plan = trm.decompose(request.task, context)
        result = new_arch.execute(plan)
        return PipelineResponse(output=result)
```

### 9.2 Extending the Code Repair Pipeline

The existing Mamba → RLM → TRM pipeline can be extended:

```python
# Add a verification step
pipeline = (
    MambaEncoder()      # O(n) context encoding
    >> RLMDecomposer()  # Break into sub-problems
    >> TRMRefiner()     # Recursive refinement
    >> CTMVerifier()    # Neural dynamics verification  ← NEW
)
```

---

## 10. Staying Current: Recommended Workflows

### 10.1 Monthly Framework Scan

```python
# Run the evolution tracker monthly
evolution = await auditor.track_evolution()

for framework in evolution.new_frameworks:
    if framework.maturity >= "alpha":
        analysis = await auditor.analyze_framework(framework.source)
        patterns = await auditor.extract_patterns(analysis)
        # Review patterns, add promising ones
```

### 10.2 Paper-to-Implementation Pipeline

When a significant paper appears:

1. **Index it**: Add metadata to `config/papers.yaml` and the appropriate era module
2. **Map lineage**: Update `config/lineages.yaml` with predecessor/successor relationships
3. **Stub architecture**: Create directory in `modern_dev/` with metadata only
4. **Evaluate fit**: Check `ARCHITECTURE_TECHNIQUES` compatibility
5. **Implement incrementally**: Follow the indexed → stubbed → alpha → beta → production progression

### 10.3 Technique Adoption Checklist

When a new reasoning pattern shows promise:

- [ ] Read the paper and identify the core loop (thought-action-observation, plan-execute, etc.)
- [ ] Check if it maps to an existing technique (some "new" patterns are variants)
- [ ] Implement as `TechniqueBase` subclass with `run()` returning `TechniqueResult`
- [ ] Verify it composes with `>>` and `|` operators
- [ ] Add architecture integration if relevant
- [ ] Write tests with `MockBackend` for deterministic verification
- [ ] Register in the appropriate category

### 10.4 Backend Monitoring

As LLM providers evolve:

- **New providers**: Implement `LLMBackend` subclass, add to `RouterBackend`
- **New capabilities**: Update `supports_tools`, `supports_vision`, `supports_streaming`
- **Pricing changes**: Update cost model in `router_backend.py`
- **New models**: Update model lists in existing backends
- **Deprecations**: Mark old models, add migration warnings

---

## 11. Architecture Decisions for the Future

### 11.1 Separation Principle

Keep the three layers independent:

```
DO:   Add a new technique that works with any backend
DO:   Add a new architecture the orchestrator can route to
DO:   Add a new agent that uses existing techniques

DON'T: Make a technique that only works with one architecture
       (use integration/ layer instead)
DON'T: Hard-code model names in agent logic
       (use backend abstraction)
DON'T: Put agent infrastructure in technique code
       (techniques are stateless reasoning patterns)
```

### 11.2 Extension Points

The system is designed to be extended at these points:

| Extension Point | Mechanism | Where |
|----------------|-----------|-------|
| New method | Register in MethodRegistry | `core/method_registry.py` |
| New architecture | Implement ArchitectureBase | `modern_dev/` |
| New technique | Implement TechniqueBase | `ml_techniques/` |
| New agent | Implement AgentBase | `agent_frameworks/core/` |
| New tool | @tool decorator or Tool subclass | `agent_frameworks/tools/` |
| New backend | Implement LLMBackend | `agent_frameworks/backends/` |
| New channel | Implement Channel | `agent_frameworks/human_loop/` |
| New data loader | Extend BaseDataLoader | `modern_dev/shared/data/` |
| New integration | Adapter functions | `agent_frameworks/integrations/` |

### 11.3 What to Watch

Key trends that will require system evolution:

| Trend | Impact | Where to Extend |
|-------|--------|----------------|
| Longer context windows | More architectures compete with Mamba on long-context | `ARCHITECTURE_CAPABILITIES` context limits |
| Multi-modal models | New task types, new data loaders | `TaskType` enum, `shared/data/` |
| Inference-time compute (o1-style) | New technique category for test-time scaling | `ml_techniques/` new category |
| Tool-use models | Native function calling reduces ReAct overhead | `agentic/` technique updates |
| Open-weight models | More local backends, fine-tuning integration | `backends/` new providers |
| Agent protocols (MCP) | Standardized tool/resource sharing | `integrations/` new adapter |
| Multi-agent standards | Emerging coordination protocols | `orchestration/` updates |
