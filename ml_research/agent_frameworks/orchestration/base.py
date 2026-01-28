"""
Base types and interfaces for the orchestration module.

This module defines the foundational data classes and abstract base classes
used throughout the orchestration system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import uuid


class TaskStatus(Enum):
    """Status of a task in the orchestration system."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class Task:
    """
    Represents a task to be executed by an agent.

    Attributes:
        id: Unique identifier for the task
        type: Type/category of the task (e.g., "code_review", "generate")
        description: Human-readable description of what to do
        input_data: Input data/parameters for the task
        priority: Priority level for scheduling
        metadata: Additional task metadata
        created_at: When the task was created
        timeout: Maximum execution time in seconds
        dependencies: IDs of tasks that must complete first
        tags: Tags for categorization and routing
    """
    type: str
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "input_data": self.input_data,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout,
            "dependencies": self.dependencies,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create a Task from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data["type"],
            description=data["description"],
            input_data=data.get("input_data", {}),
            priority=TaskPriority(data.get("priority", 1)),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            timeout=data.get("timeout"),
            dependencies=data.get("dependencies", []),
            tags=set(data.get("tags", [])),
        )


@dataclass
class TaskResult:
    """
    Result of executing a task.

    Attributes:
        task_id: ID of the completed task
        status: Final status of the task
        output: Output data from execution
        error: Error message if failed
        execution_time: Time taken to execute in seconds
        agent_id: ID of the agent that executed the task
        metadata: Additional result metadata
        completed_at: When the task completed
    """
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def success_result(
        cls,
        task_id: str,
        output: Any,
        agent_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        **metadata
    ) -> "TaskResult":
        """Create a successful result."""
        return cls(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            output=output,
            agent_id=agent_id,
            execution_time=execution_time,
            metadata=metadata,
        )

    @classmethod
    def failure_result(
        cls,
        task_id: str,
        error: str,
        agent_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        **metadata
    ) -> "TaskResult":
        """Create a failure result."""
        return cls(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=error,
            agent_id=agent_id,
            execution_time=execution_time,
            metadata=metadata,
        )


class AgentCapability(Enum):
    """Capabilities that an agent can have."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    EXECUTION = "execution"


@dataclass
class Session:
    """
    Represents a session with an agent.

    Attributes:
        id: Unique session identifier
        agent_id: ID of the agent this session belongs to
        created_at: When the session started
        last_activity: Last activity timestamp
        metadata: Session metadata
        history: Message history for the session
        state: Arbitrary session state data
    """
    id: str
    agent_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to the session history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        })
        self.last_activity = datetime.now()

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get message history, optionally limited to last N messages."""
        if limit is None:
            return self.history.copy()
        return self.history[-limit:]

    def clear_history(self) -> None:
        """Clear the session history."""
        self.history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary representation."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata,
            "history": self.history,
            "state": self.state,
        }


class AgentBase(ABC):
    """
    Abstract base class for agents in the orchestration system.

    All agents must implement this interface to be managed by the Gateway.
    """

    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for the agent (auto-generated if not provided)
        """
        self._id = agent_id or str(uuid.uuid4())
        self._capabilities: Set[AgentCapability] = set()
        self._metadata: Dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Get the agent's unique identifier."""
        return self._id

    @property
    def capabilities(self) -> Set[AgentCapability]:
        """Get the agent's capabilities."""
        return self._capabilities

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the agent's metadata."""
        return self._metadata

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task.

        Args:
            task: The task to execute

        Returns:
            TaskResult containing the execution outcome
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent.

        Called when the agent is registered with a Gateway.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the agent.

        Called when the agent is unregistered or the Gateway shuts down.
        """
        ...

    def can_handle(self, task: Task) -> bool:
        """
        Check if this agent can handle a given task.

        Default implementation checks task type against capabilities.
        Override for custom matching logic.

        Args:
            task: The task to check

        Returns:
            True if the agent can handle the task
        """
        # Default: check if any capability matches task type
        task_type_lower = task.type.lower()
        for cap in self._capabilities:
            if cap.value.lower() in task_type_lower or task_type_lower in cap.value.lower():
                return True
        return False

    async def health_check(self) -> bool:
        """
        Check if the agent is healthy.

        Returns:
            True if the agent is operational
        """
        return True

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to this agent."""
        self._capabilities.add(capability)

    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove a capability from this agent."""
        self._capabilities.discard(capability)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self._metadata[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, capabilities={len(self._capabilities)})"
