"""
Agent Frameworks Integration Library

A composable library for building AI coding assistants,
synthesizing patterns from Aider, OpenCode, HumanLayer, and Clawdbot.

This library provides a unified framework for building AI agents that can:
- Understand and navigate codebases (context management)
- Execute tasks with human oversight (human-in-the-loop)
- Coordinate multiple agents (orchestration)
- Learn from past interactions (memory)
- Integrate with various LLM backends (backends)

Example:
    from agent_frameworks import (
        ArchitectEditor,
        AnthropicBackend,
        ToolRegistry,
        require_approval,
    )

    # Set up backend
    backend = AnthropicBackend(api_key="...")

    # Create architect-editor for planning and execution
    agent = ArchitectEditor(backend)

    # Plan a task
    plan = await agent.plan(
        task="Add user authentication",
        context=repository_context
    )

    # Execute with tool access
    result = await agent.execute(plan, tools)

Modules:
    - core: Base agent classes, state machines, and composition
    - backends: LLM provider integrations (Anthropic, OpenAI, Ollama)
    - tools: Tool definitions, registry, and permissions
    - context: Repository mapping and file selection
    - execution: Architect-editor pattern and session management
    - human_loop: Approval workflows and escalation
    - orchestration: Multi-agent coordination
    - memory: Context windows and checkpointing
    - auditor: Framework analysis and pattern extraction
"""

__version__ = "0.1.0"
__author__ = "Agent Frameworks Contributors"

# ---------------------------------------------------------------------------
# Core Module Imports
# ---------------------------------------------------------------------------
# Core provides base agent abstractions, message types, state machines,
# and composition operators for building agent pipelines.

try:
    from .core import (
        AgentBase,
        AgentMode,
        AgentRegistry,
        register,
        AgentMessage,
        MessageRole,
        ToolCall,
        ToolResult as CoreToolResult,
        StateMachine,
        AgentState,
        SequentialPipeline,
        ParallelPipeline,
        ApprovalGate,
        Task,
        TaskResult,
        Plan,
        ExecutionPlan as CoreExecutionPlan,
    )
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    # Provide placeholder types for documentation
    AgentBase = None
    AgentMode = None
    AgentRegistry = None
    register = None

# ---------------------------------------------------------------------------
# Backend Module Imports
# ---------------------------------------------------------------------------
# Backends provide async access to various LLM providers with unified interface.

try:
    from .backends.backend_base import (
        LLMBackend,
        LLMConfig,
        LLMResponse,
        ToolCall as BackendToolCall,
        BackendError,
        AuthenticationError,
        RateLimitError,
        ModelNotFoundError,
        ContextLengthError,
        ContentFilterError,
    )
    _BACKEND_BASE_AVAILABLE = True
except ImportError:
    _BACKEND_BASE_AVAILABLE = False
    LLMBackend = None
    LLMConfig = None
    LLMResponse = None

try:
    from .backends.anthropic_backend import AnthropicBackend
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    AnthropicBackend = None

try:
    from .backends.openai_backend import OpenAIBackend
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAIBackend = None

try:
    from .backends.ollama_backend import OllamaBackend
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    OllamaBackend = None

try:
    from .backends.router_backend import RouterBackend
    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False
    RouterBackend = None

# ---------------------------------------------------------------------------
# Tools Module Imports
# ---------------------------------------------------------------------------
# Tools provides the base classes and registry for agent tools.

try:
    from .tools.tool_base import (
        Tool,
        ToolSchema,
        ToolResult,
        ToolPermission,
    )
    from .tools.tool_registry import (
        ToolRegistry,
        tool,
        FunctionTool,
        get_registry,
        ToolNotFoundError,
        ToolAlreadyRegisteredError,
    )
    from .tools.permissions import (
        PermissionManager,
        PermissionPolicy,
        PermissionDecision,
        UserContext,
        PermissionRequest,
        PermissionGrant,
        AllowAllPolicy,
        DenyDangerousPolicy,
        RequireApprovalPolicy,
    )
    _TOOLS_AVAILABLE = True
except ImportError:
    _TOOLS_AVAILABLE = False
    Tool = None
    ToolSchema = None
    ToolResult = None
    ToolRegistry = None
    tool = None

# ---------------------------------------------------------------------------
# Context Module Imports
# ---------------------------------------------------------------------------
# Context provides intelligent context management for coding agents.

try:
    from .context import (
        RepositoryMap,
        FunctionSignature,
        ClassDefinition,
        ModuleInfo,
        FileSelector,
        FileScore,
        SelectionStrategy,
        DiffTracker,
        ChangeEvent,
        ChangeType,
        SemanticIndex,
        SearchResult,
        ChunkingStrategy,
    )
    _CONTEXT_AVAILABLE = True
except ImportError:
    _CONTEXT_AVAILABLE = False
    RepositoryMap = None
    FileSelector = None
    DiffTracker = None
    SemanticIndex = None

# ---------------------------------------------------------------------------
# Execution Module Imports
# ---------------------------------------------------------------------------
# Execution provides the architect-editor pattern and session management.

try:
    from .execution import (
        ArchitectEditor,
        ExecutionMode,
        ExecutionPlan,
        ExecutionResult,
        PlanStep,
        SessionManager,
        Session,
        SessionState,
        ToolExecutor,
        SandboxMode,
        ResourceLimits,
        ExecutionContext,
        AuditLogEntry,
        AgentServer,
        AgentClient,
        ServerConfig,
        TaskResult as ExecutionTaskResult,
    )
    _EXECUTION_AVAILABLE = True
except ImportError:
    _EXECUTION_AVAILABLE = False
    ArchitectEditor = None
    ExecutionMode = None
    ExecutionPlan = None
    SessionManager = None
    ToolExecutor = None
    AgentServer = None
    AgentClient = None

# ---------------------------------------------------------------------------
# Human Loop Module Imports
# ---------------------------------------------------------------------------
# Human loop provides approval workflows and escalation management.

try:
    from .human_loop import (
        require_approval,
        ApprovalWorkflow,
        ApprovalRequest,
        ApprovalStatus,
        HumanAsTool,
        ChannelRouter,
        Channel,
        ChannelType,
        EscalationManager,
        EscalationLevel,
    )
    _HUMAN_LOOP_AVAILABLE = True
except ImportError:
    _HUMAN_LOOP_AVAILABLE = False
    require_approval = None
    ApprovalWorkflow = None
    ApprovalRequest = None
    HumanAsTool = None
    ChannelRouter = None
    EscalationManager = None

# ---------------------------------------------------------------------------
# Orchestration Module Imports
# ---------------------------------------------------------------------------
# Orchestration provides multi-agent coordination.

try:
    from .orchestration import (
        Gateway,
        AgentRouter,
        RoutingStrategy,
        Workspace,
        WorkspaceConfig,
        EventBus,
    )
    _ORCHESTRATION_AVAILABLE = True
except ImportError:
    _ORCHESTRATION_AVAILABLE = False
    Gateway = None
    AgentRouter = None
    Workspace = None
    EventBus = None

# ---------------------------------------------------------------------------
# Memory Module Imports
# ---------------------------------------------------------------------------
# Memory provides context management and checkpointing.

try:
    from .memory import (
        ContextWindow,
        EpisodicMemory,
        Episode,
        SemanticMemory,
        MemoryItem,
        CheckpointManager,
        Checkpoint,
    )
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False
    ContextWindow = None
    EpisodicMemory = None
    SemanticMemory = None
    CheckpointManager = None

# ---------------------------------------------------------------------------
# Auditor Module Imports
# ---------------------------------------------------------------------------
# Auditor provides framework analysis and pattern extraction.

try:
    from .auditor import (
        AuditorAgent,
        PatternExtractor,
        IntegrationGenerator,
        FrameworkSource,
        FrameworkAnalysis,
        Pattern,
        PatternCategory,
        IntegrationSpec,
    )
    _AUDITOR_AVAILABLE = True
except ImportError:
    _AUDITOR_AVAILABLE = False
    AuditorAgent = None
    PatternExtractor = None
    IntegrationGenerator = None
    FrameworkSource = None
    FrameworkAnalysis = None
    Pattern = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",

    # Core (when available)
    "AgentBase",
    "AgentMode",
    "AgentRegistry",
    "register",
    "AgentMessage",
    "MessageRole",
    "StateMachine",
    "AgentState",
    "SequentialPipeline",
    "ParallelPipeline",
    "ApprovalGate",
    "Task",
    "TaskResult",
    "Plan",

    # Backends
    "LLMBackend",
    "LLMConfig",
    "LLMResponse",
    "AnthropicBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "RouterBackend",
    "BackendError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContextLengthError",
    "ContentFilterError",

    # Tools
    "Tool",
    "ToolSchema",
    "ToolResult",
    "ToolPermission",
    "ToolRegistry",
    "tool",
    "FunctionTool",
    "get_registry",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "PermissionManager",
    "PermissionPolicy",
    "PermissionDecision",
    "UserContext",
    "AllowAllPolicy",
    "DenyDangerousPolicy",
    "RequireApprovalPolicy",

    # Context
    "RepositoryMap",
    "FunctionSignature",
    "ClassDefinition",
    "ModuleInfo",
    "FileSelector",
    "FileScore",
    "SelectionStrategy",
    "DiffTracker",
    "ChangeEvent",
    "ChangeType",
    "SemanticIndex",
    "SearchResult",
    "ChunkingStrategy",

    # Execution
    "ArchitectEditor",
    "ExecutionMode",
    "ExecutionPlan",
    "ExecutionResult",
    "PlanStep",
    "SessionManager",
    "Session",
    "SessionState",
    "ToolExecutor",
    "SandboxMode",
    "ResourceLimits",
    "ExecutionContext",
    "AgentServer",
    "AgentClient",
    "ServerConfig",

    # Human Loop
    "require_approval",
    "ApprovalWorkflow",
    "ApprovalRequest",
    "ApprovalStatus",
    "HumanAsTool",
    "ChannelRouter",
    "Channel",
    "ChannelType",
    "EscalationManager",
    "EscalationLevel",

    # Orchestration
    "Gateway",
    "AgentRouter",
    "RoutingStrategy",
    "Workspace",
    "WorkspaceConfig",
    "EventBus",

    # Memory
    "ContextWindow",
    "EpisodicMemory",
    "Episode",
    "SemanticMemory",
    "MemoryItem",
    "CheckpointManager",
    "Checkpoint",

    # Auditor
    "AuditorAgent",
    "PatternExtractor",
    "IntegrationGenerator",
    "FrameworkSource",
    "FrameworkAnalysis",
    "Pattern",
    "PatternCategory",
    "IntegrationSpec",
]


# ---------------------------------------------------------------------------
# Module Availability Check
# ---------------------------------------------------------------------------

def get_available_modules() -> dict:
    """
    Check which modules are available.

    Returns:
        Dictionary mapping module names to availability status.

    Example:
        >>> available = get_available_modules()
        >>> if available['backends']:
        ...     from agent_frameworks import AnthropicBackend
    """
    return {
        "core": _CORE_AVAILABLE if '_CORE_AVAILABLE' in dir() else False,
        "backends": _BACKEND_BASE_AVAILABLE,
        "backends.anthropic": _ANTHROPIC_AVAILABLE,
        "backends.openai": _OPENAI_AVAILABLE if '_OPENAI_AVAILABLE' in dir() else False,
        "backends.ollama": _OLLAMA_AVAILABLE if '_OLLAMA_AVAILABLE' in dir() else False,
        "tools": _TOOLS_AVAILABLE,
        "context": _CONTEXT_AVAILABLE,
        "execution": _EXECUTION_AVAILABLE,
        "human_loop": _HUMAN_LOOP_AVAILABLE if '_HUMAN_LOOP_AVAILABLE' in dir() else False,
        "orchestration": _ORCHESTRATION_AVAILABLE if '_ORCHESTRATION_AVAILABLE' in dir() else False,
        "memory": _MEMORY_AVAILABLE if '_MEMORY_AVAILABLE' in dir() else False,
        "auditor": _AUDITOR_AVAILABLE if '_AUDITOR_AVAILABLE' in dir() else False,
    }


def check_dependencies() -> dict:
    """
    Check external dependency availability.

    Returns:
        Dictionary mapping package names to availability status.
    """
    dependencies = {}

    try:
        import anthropic
        dependencies["anthropic"] = anthropic.__version__
    except ImportError:
        dependencies["anthropic"] = None

    try:
        import openai
        dependencies["openai"] = openai.__version__
    except ImportError:
        dependencies["openai"] = None

    try:
        import httpx
        dependencies["httpx"] = httpx.__version__
    except ImportError:
        dependencies["httpx"] = None

    try:
        import pydantic
        dependencies["pydantic"] = pydantic.__version__
    except ImportError:
        dependencies["pydantic"] = None

    try:
        import yaml
        dependencies["pyyaml"] = True
    except ImportError:
        dependencies["pyyaml"] = None

    return dependencies
