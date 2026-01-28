"""
Base agent abstract class for all agent implementations.

This module provides the foundational AgentBase class that all agents
should inherit from, along with supporting dataclasses for tasks, plans,
and execution results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    List, Optional, Dict, Any, TypeVar, Generic,
    Callable, Awaitable, TYPE_CHECKING, Union
)
from datetime import datetime
import uuid
import asyncio

from .message_types import AgentMessage, ApprovalStatus, ConversationHistory

if TYPE_CHECKING:
    from .composition import SequentialPipeline, ParallelPipeline


class AgentMode(Enum):
    """Operating modes for agents."""
    ARCHITECT = auto()  # High-level planning and design
    EDITOR = auto()     # Code/content editing
    ASK = auto()        # Question answering and information retrieval
    CODE = auto()       # Code generation and implementation

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Task:
    """Represents a task to be executed by an agent.

    Attributes:
        id: Unique identifier for this task
        description: Human-readable description of what needs to be done
        context: Additional context or background information
        constraints: Any constraints or requirements for the task
        metadata: Additional task metadata
        created_at: When the task was created
    """
    id: str
    description: str
    context: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, description: str, **kwargs) -> "Task":
        """Factory method to create a Task with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "context": self.context,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PlanStep:
    """A single step within a plan.

    Attributes:
        id: Unique identifier for this step
        action: Description of the action to take
        dependencies: IDs of steps that must complete before this one
        estimated_effort: Optional effort estimate
        metadata: Additional step metadata
    """
    id: str
    action: str
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, action: str, **kwargs) -> "PlanStep":
        """Factory method to create a PlanStep with auto-generated ID."""
        return cls(id=str(uuid.uuid4()), action=action, **kwargs)


@dataclass
class Plan:
    """A plan for executing a task.

    Attributes:
        id: Unique identifier for this plan
        task_id: ID of the task this plan addresses
        steps: Ordered list of plan steps
        rationale: Explanation of the planning approach
        metadata: Additional plan metadata
        created_at: When the plan was created
    """
    id: str
    task_id: str
    steps: List[PlanStep]
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, task_id: str, steps: List[PlanStep], **kwargs) -> "Plan":
        """Factory method to create a Plan with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            steps=steps,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "steps": [
                {
                    "id": s.id,
                    "action": s.action,
                    "dependencies": s.dependencies,
                    "estimated_effort": s.estimated_effort,
                    "metadata": s.metadata
                }
                for s in self.steps
            ],
            "rationale": self.rationale,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ExecutionResult:
    """Result of executing a single plan step.

    Attributes:
        step_id: ID of the executed step
        success: Whether execution succeeded
        output: The output/result of execution
        error: Error message if execution failed
        duration_ms: Execution time in milliseconds
        metadata: Additional result metadata
    """
    step_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, step_id: str, output: Any, **kwargs) -> "ExecutionResult":
        """Factory method for successful execution results."""
        return cls(step_id=step_id, success=True, output=output, **kwargs)

    @classmethod
    def error_result(cls, step_id: str, error: str, **kwargs) -> "ExecutionResult":
        """Factory method for failed execution results."""
        return cls(step_id=step_id, success=False, error=error, **kwargs)


@dataclass
class TaskResult:
    """Final result of a task execution.

    Attributes:
        task_id: ID of the completed task
        success: Whether the task completed successfully
        output: The final output/result
        execution_results: Results from individual plan steps
        messages: Conversation history during execution
        error: Error message if the task failed
        duration_ms: Total execution time in milliseconds
        metadata: Additional result metadata
    """
    task_id: str
    success: bool
    output: Any = None
    execution_results: List[ExecutionResult] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, task_id: str, output: Any, **kwargs) -> "TaskResult":
        """Factory method for successful task results."""
        return cls(task_id=task_id, success=True, output=output, **kwargs)

    @classmethod
    def error_result(cls, task_id: str, error: str, **kwargs) -> "TaskResult":
        """Factory method for failed task results."""
        return cls(task_id=task_id, success=False, error=error, **kwargs)


# Protocol for approval channels
class ApprovalChannel(ABC):
    """Abstract base class for human-in-the-loop approval channels."""

    @abstractmethod
    async def request_approval(
        self,
        message: str,
        context: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ApprovalStatus:
        """Request approval from a human.

        Args:
            message: Description of what needs approval
            context: Additional context for the approver
            timeout: Optional timeout in seconds

        Returns:
            The approval status
        """
        pass


# Protocol for memory systems
class MemorySystem(ABC):
    """Abstract base class for agent memory systems."""

    @abstractmethod
    async def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory for relevant items."""
        pass


# Protocol for tool providers
class ToolProvider(ABC):
    """Abstract base class for tool providers."""

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with their schemas."""
        pass

    @abstractmethod
    async def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a tool with the given arguments."""
        pass


T = TypeVar('T', bound='AgentBase')


class AgentBase(ABC):
    """Abstract base class for all agents.

    All agents must inherit from this class and implement the abstract methods.
    The class provides common functionality for approval workflows, memory,
    and tool integration.

    Class Attributes:
        AGENT_ID: Unique identifier for the agent type
        SUPPORTED_MODES: List of AgentMode values this agent supports
        REQUIRES_APPROVAL: Whether this agent requires human approval by default
    """

    AGENT_ID: str = "base"
    SUPPORTED_MODES: List[AgentMode] = [AgentMode.ASK]
    REQUIRES_APPROVAL: bool = False

    def __init__(self):
        """Initialize the agent."""
        self._approval_channel: Optional[ApprovalChannel] = None
        self._memory: Optional[MemorySystem] = None
        self._tools: Optional[ToolProvider] = None
        self._mode: AgentMode = self.SUPPORTED_MODES[0] if self.SUPPORTED_MODES else AgentMode.ASK
        self._conversation: ConversationHistory = ConversationHistory()
        self._metadata: Dict[str, Any] = {}

    @property
    def mode(self) -> AgentMode:
        """Get the current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        """Set the operating mode."""
        if value not in self.SUPPORTED_MODES:
            raise ValueError(f"Mode {value} not supported by {self.AGENT_ID}")
        self._mode = value

    @property
    def conversation(self) -> ConversationHistory:
        """Get the conversation history."""
        return self._conversation

    def with_approval(self: T, channel: ApprovalChannel) -> T:
        """Configure the agent with an approval channel.

        Args:
            channel: The approval channel to use

        Returns:
            self for method chaining
        """
        self._approval_channel = channel
        return self

    def with_memory(self: T, memory: MemorySystem) -> T:
        """Configure the agent with a memory system.

        Args:
            memory: The memory system to use

        Returns:
            self for method chaining
        """
        self._memory = memory
        return self

    def with_tools(self: T, tools: ToolProvider) -> T:
        """Configure the agent with a tool provider.

        Args:
            tools: The tool provider to use

        Returns:
            self for method chaining
        """
        self._tools = tools
        return self

    def with_mode(self: T, mode: AgentMode) -> T:
        """Configure the agent's operating mode.

        Args:
            mode: The operating mode to use

        Returns:
            self for method chaining
        """
        self.mode = mode
        return self

    async def request_approval(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> ApprovalStatus:
        """Request human approval if an approval channel is configured.

        Args:
            message: Description of what needs approval
            context: Additional context for the approver
            timeout: Optional timeout in seconds

        Returns:
            ApprovalStatus.APPROVED if no channel configured, otherwise the result
        """
        if self._approval_channel is None:
            return ApprovalStatus.APPROVED

        return await self._approval_channel.request_approval(
            message=message,
            context=context or {},
            timeout=timeout
        )

    async def store_memory(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory if a memory system is configured.

        Args:
            key: The key to store under
            value: The value to store
            metadata: Optional metadata
        """
        if self._memory is not None:
            await self._memory.store(key, value, metadata)

    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory if a memory system is configured.

        Args:
            key: The key to retrieve

        Returns:
            The stored value or None
        """
        if self._memory is not None:
            return await self._memory.retrieve(key)
        return None

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory if a memory system is configured.

        Args:
            query: The search query
            limit: Maximum number of results

        Returns:
            List of matching items or empty list
        """
        if self._memory is not None:
            return await self._memory.search(query, limit)
        return []

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a tool if a tool provider is configured.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool

        Returns:
            The tool result

        Raises:
            RuntimeError: If no tool provider is configured
        """
        if self._tools is None:
            raise RuntimeError("No tool provider configured")
        return await self._tools.invoke(tool_name, arguments)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools.

        Returns:
            List of tool schemas or empty list if no provider configured
        """
        if self._tools is not None:
            return self._tools.list_tools()
        return []

    @abstractmethod
    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute a task end-to-end.

        This is the main entry point for running an agent. It should handle
        the complete lifecycle of planning and executing the task.

        Args:
            task: The task to execute (string or Task object)

        Returns:
            The result of the task execution
        """
        pass

    @abstractmethod
    async def plan(self, task: Task) -> Plan:
        """Create a plan for executing the given task.

        Args:
            task: The task to plan for

        Returns:
            A plan for executing the task
        """
        pass

    @abstractmethod
    async def execute(self, plan: Plan) -> List[ExecutionResult]:
        """Execute a plan step by step.

        Args:
            plan: The plan to execute

        Returns:
            Results from each step of execution
        """
        pass

    def __rshift__(self, other: "AgentBase") -> "SequentialPipeline":
        """Create a sequential pipeline: self >> other.

        The output of self feeds into other as input.

        Args:
            other: The agent to chain after this one

        Returns:
            A SequentialPipeline containing both agents
        """
        from .composition import SequentialPipeline
        return SequentialPipeline([self, other])

    def __or__(self, other: "AgentBase") -> "ParallelPipeline":
        """Create a parallel pipeline: self | other.

        Both agents run in parallel on the same input.

        Args:
            other: The agent to run in parallel with this one

        Returns:
            A ParallelPipeline containing both agents
        """
        from .composition import ParallelPipeline
        return ParallelPipeline([self, other])

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.AGENT_ID} mode={self._mode}>"


class SimpleAgent(AgentBase):
    """A simple concrete agent implementation for testing and basic use cases.

    This agent creates single-step plans and executes them directly.
    """

    AGENT_ID: str = "simple"
    SUPPORTED_MODES: List[AgentMode] = list(AgentMode)
    REQUIRES_APPROVAL: bool = False

    def __init__(self, handler: Optional[Callable[[Task], Awaitable[Any]]] = None):
        """Initialize the simple agent.

        Args:
            handler: Optional async function to handle task execution
        """
        super().__init__()
        self._handler = handler

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute a task end-to-end."""
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()

        try:
            # Create and execute plan
            plan = await self.plan(task)
            results = await self.execute(plan)

            # Collect results
            success = all(r.success for r in results)
            output = [r.output for r in results if r.success]
            errors = [r.error for r in results if not r.success]

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            if success:
                return TaskResult.success_result(
                    task_id=task.id,
                    output=output[0] if len(output) == 1 else output,
                    execution_results=results,
                    messages=list(self._conversation),
                    duration_ms=duration
                )
            else:
                return TaskResult.error_result(
                    task_id=task.id,
                    error="; ".join(e for e in errors if e),
                    execution_results=results,
                    messages=list(self._conversation),
                    duration_ms=duration
                )

        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            return TaskResult.error_result(
                task_id=task.id,
                error=str(e),
                duration_ms=duration
            )

    async def plan(self, task: Task) -> Plan:
        """Create a simple single-step plan."""
        step = PlanStep.create(action=task.description)
        return Plan.create(
            task_id=task.id,
            steps=[step],
            rationale="Simple direct execution"
        )

    async def execute(self, plan: Plan) -> List[ExecutionResult]:
        """Execute the plan steps."""
        results = []

        for step in plan.steps:
            start_time = datetime.now()

            try:
                if self._handler:
                    # Use custom handler
                    output = await self._handler(
                        Task.create(step.action)
                    )
                else:
                    # Default: just return the action as output
                    output = f"Executed: {step.action}"

                duration = int((datetime.now() - start_time).total_seconds() * 1000)
                results.append(ExecutionResult.success_result(
                    step_id=step.id,
                    output=output,
                    duration_ms=duration
                ))

            except Exception as e:
                duration = int((datetime.now() - start_time).total_seconds() * 1000)
                results.append(ExecutionResult.error_result(
                    step_id=step.id,
                    error=str(e),
                    duration_ms=duration
                ))

        return results
