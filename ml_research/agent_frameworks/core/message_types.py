"""
Standardized message formats for agent communication.

This module provides dataclasses and enums for consistent message handling
across different agent frameworks and implementations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class MessageRole(Enum):
    """Enumeration of possible message roles in agent conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    HUMAN = "human"

    def __str__(self) -> str:
        return self.value


class ApprovalStatus(Enum):
    """Status of human-in-the-loop approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this status represents a terminal state."""
        return self in (ApprovalStatus.APPROVED, ApprovalStatus.DENIED, ApprovalStatus.TIMEOUT)

    @property
    def allows_continuation(self) -> bool:
        """Check if this status allows the agent to continue execution."""
        return self == ApprovalStatus.APPROVED


@dataclass
class ToolCall:
    """Represents a tool invocation request from an agent.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Dictionary of arguments to pass to the tool
    """
    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def create(cls, name: str, arguments: Dict[str, Any]) -> "ToolCall":
        """Factory method to create a ToolCall with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            arguments=arguments
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {})
        )


@dataclass
class ToolResult:
    """Represents the result of a tool invocation.

    Attributes:
        tool_call_id: ID of the corresponding ToolCall
        output: The result returned by the tool
        error: Error message if the tool invocation failed
    """
    tool_call_id: str
    output: Any
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the tool invocation was successful."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary representation."""
        return cls(
            tool_call_id=data["tool_call_id"],
            output=data.get("output"),
            error=data.get("error")
        )

    @classmethod
    def success_result(cls, tool_call_id: str, output: Any) -> "ToolResult":
        """Factory method for successful tool results."""
        return cls(tool_call_id=tool_call_id, output=output, error=None)

    @classmethod
    def error_result(cls, tool_call_id: str, error: str) -> "ToolResult":
        """Factory method for failed tool results."""
        return cls(tool_call_id=tool_call_id, output=None, error=error)


@dataclass
class AgentMessage:
    """A message in an agent conversation.

    Attributes:
        role: The role of the message sender
        content: Text content of the message
        tool_calls: Optional list of tool invocation requests
        tool_results: Optional list of tool invocation results
        metadata: Additional metadata associated with the message
        approval_status: Status if this message requires approval
        timestamp: When the message was created
    """
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    approval_status: Optional[ApprovalStatus] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate message after initialization."""
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
        if isinstance(self.approval_status, str):
            self.approval_status = ApprovalStatus(self.approval_status)

    @property
    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def has_tool_results(self) -> bool:
        """Check if this message contains tool results."""
        return self.tool_results is not None and len(self.tool_results) > 0

    @property
    def requires_approval(self) -> bool:
        """Check if this message is pending approval."""
        return self.approval_status == ApprovalStatus.PENDING

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to this message."""
        if self.tool_calls is None:
            self.tool_calls = []
        self.tool_calls.append(tool_call)

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Add a tool result to this message."""
        if self.tool_results is None:
            self.tool_results = []
        self.tool_results.append(tool_result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_results:
            result["tool_results"] = [tr.to_dict() for tr in self.tool_results]
        if self.approval_status:
            result["approval_status"] = self.approval_status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary representation."""
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        tool_results = None
        if "tool_results" in data:
            tool_results = [ToolResult.from_dict(tr) for tr in data["tool_results"]]

        approval_status = None
        if "approval_status" in data:
            approval_status = ApprovalStatus(data["approval_status"])

        timestamp = datetime.now()
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata=data.get("metadata", {}),
            approval_status=approval_status,
            timestamp=timestamp
        )

    @classmethod
    def user(cls, content: str, **metadata) -> "AgentMessage":
        """Factory method for user messages."""
        return cls(role=MessageRole.USER, content=content, metadata=metadata)

    @classmethod
    def assistant(cls, content: str, **metadata) -> "AgentMessage":
        """Factory method for assistant messages."""
        return cls(role=MessageRole.ASSISTANT, content=content, metadata=metadata)

    @classmethod
    def system(cls, content: str, **metadata) -> "AgentMessage":
        """Factory method for system messages."""
        return cls(role=MessageRole.SYSTEM, content=content, metadata=metadata)

    @classmethod
    def tool(cls, content: str, tool_results: List[ToolResult], **metadata) -> "AgentMessage":
        """Factory method for tool result messages."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_results=tool_results,
            metadata=metadata
        )


@dataclass
class ConversationHistory:
    """Manages a sequence of agent messages.

    Attributes:
        messages: List of messages in the conversation
        max_messages: Optional limit on message history
    """
    messages: List[AgentMessage] = field(default_factory=list)
    max_messages: Optional[int] = None

    def add(self, message: AgentMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self, role: Optional[MessageRole] = None) -> List[AgentMessage]:
        """Get messages, optionally filtered by role."""
        if role is None:
            return self.messages.copy()
        return [m for m in self.messages if m.role == role]

    def get_last(self, n: int = 1) -> List[AgentMessage]:
        """Get the last n messages."""
        return self.messages[-n:]

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all messages to dictionary representation."""
        return [m.to_dict() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)
