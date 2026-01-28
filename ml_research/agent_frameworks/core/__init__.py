"""
Core module for the agent_frameworks library.

This module provides the foundational building blocks for creating and
composing AI agents, including:

- Message types for standardized communication
- Base agent class with planning and execution abstractions
- Agent and framework registry for discovery and management
- State machine for managing agent execution lifecycle
- Composition operators for building complex agent workflows

Example usage:

    from agent_frameworks.core import (
        AgentBase, AgentMode, Task, TaskResult,
        AgentRegistry, register,
        StateMachine, AgentState,
        SequentialPipeline, ParallelPipeline, Retry
    )

    # Create a simple agent
    @register(framework="my_framework")
    class MyAgent(AgentBase):
        AGENT_ID = "my_agent"
        SUPPORTED_MODES = [AgentMode.CODE, AgentMode.ASK]

        async def run(self, task):
            # Implementation
            pass

        async def plan(self, task):
            # Implementation
            pass

        async def execute(self, plan):
            # Implementation
            pass

    # Compose agents into pipelines
    pipeline = agent1 >> agent2  # Sequential
    parallel = agent1 | agent2   # Parallel
    robust = Retry(agent1, max_attempts=3)
"""

# Message types
from .message_types import (
    MessageRole,
    ApprovalStatus,
    ToolCall,
    ToolResult,
    AgentMessage,
    ConversationHistory,
)

# Base agent and related types
from .base_agent import (
    AgentMode,
    Task,
    PlanStep,
    Plan,
    ExecutionResult,
    TaskResult,
    ApprovalChannel,
    MemorySystem,
    ToolProvider,
    AgentBase,
    SimpleAgent,
)

# Registry
from .agent_registry import (
    FrameworkConfig,
    AgentRegistration,
    AgentRegistry,
    get_registry,
    register,
    register_agent,
)

# State machine
from .state_machine import (
    AgentState,
    StateTransition,
    TransitionRule,
    StateMachine,
    StateMachineBuilder,
    StateCallback,
    TransitionGuard,
)

# Composition
from .composition import (
    ComposableAgent,
    SequentialPipeline,
    ParallelPipeline,
    ApprovalRequest,
    ApprovalGate,
    RetryConfig,
    Retry,
    ConditionalPipeline,
    FanOut,
)

__all__ = [
    # Message types
    "MessageRole",
    "ApprovalStatus",
    "ToolCall",
    "ToolResult",
    "AgentMessage",
    "ConversationHistory",
    # Base agent
    "AgentMode",
    "Task",
    "PlanStep",
    "Plan",
    "ExecutionResult",
    "TaskResult",
    "ApprovalChannel",
    "MemorySystem",
    "ToolProvider",
    "AgentBase",
    "SimpleAgent",
    # Registry
    "FrameworkConfig",
    "AgentRegistration",
    "AgentRegistry",
    "get_registry",
    "register",
    "register_agent",
    # State machine
    "AgentState",
    "StateTransition",
    "TransitionRule",
    "StateMachine",
    "StateMachineBuilder",
    "StateCallback",
    "TransitionGuard",
    # Composition
    "ComposableAgent",
    "SequentialPipeline",
    "ParallelPipeline",
    "ApprovalRequest",
    "ApprovalGate",
    "RetryConfig",
    "Retry",
    "ConditionalPipeline",
    "FanOut",
]

__version__ = "0.1.0"
