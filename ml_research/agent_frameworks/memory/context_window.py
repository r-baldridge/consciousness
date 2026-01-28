"""
Sliding context window management with automatic summarization.

This module provides intelligent context window management that automatically
handles token limits, summarization, and message prioritization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
import json
import logging

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Represents a message in the agent conversation.

    Attributes:
        role: The role of the message sender (user, assistant, system, tool)
        content: The text content of the message
        timestamp: When the message was created
        metadata: Additional data about the message (tool calls, etc.)
        token_count: Cached token count for this message
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM APIs."""
        result = {
            "role": self.role,
            "content": self.content,
        }
        # Include tool-related fields if present
        if "tool_call_id" in self.metadata:
            result["tool_call_id"] = self.metadata["tool_call_id"]
        if "tool_calls" in self.metadata:
            result["tool_calls"] = self.metadata["tool_calls"]
        if "name" in self.metadata:
            result["name"] = self.metadata["name"]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        metadata = {}
        for key in ["tool_call_id", "tool_calls", "name"]:
            if key in data:
                metadata[key] = data[key]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            metadata=metadata
        )


@dataclass
class ContextConfig:
    """Configuration for context window management.

    Attributes:
        max_tokens: Maximum tokens allowed in context
        reserve_tokens: Tokens to reserve for response generation
        summarization_threshold: Fraction at which to trigger summarization (0.0-1.0)
        min_messages_to_keep: Minimum recent messages to always keep
        summarization_prompt: Custom prompt for summarization
        preserve_system_messages: Whether to always keep system messages
    """
    max_tokens: int = 100000
    reserve_tokens: int = 4000
    summarization_threshold: float = 0.8
    min_messages_to_keep: int = 4
    summarization_prompt: Optional[str] = None
    preserve_system_messages: bool = True

    @property
    def available_tokens(self) -> int:
        """Tokens available for context (excluding reserve)."""
        return self.max_tokens - self.reserve_tokens

    @property
    def summarization_trigger(self) -> int:
        """Token count at which to trigger summarization."""
        return int(self.available_tokens * self.summarization_threshold)


class ContextWindow:
    """Manages sliding context window with automatic summarization.

    This class maintains a list of messages and automatically handles
    context window limits through summarization and pruning.

    Example:
        config = ContextConfig(max_tokens=100000)
        context = ContextWindow(config, backend)

        await context.add(AgentMessage(role="user", content="Hello"))
        messages = await context.get_context()
    """

    DEFAULT_SUMMARIZATION_PROMPT = """Summarize the following conversation concisely,
preserving key information, decisions made, and any important context that would
be needed to continue the conversation effectively. Focus on:
1. Main topics discussed
2. Decisions or conclusions reached
3. Pending tasks or questions
4. Important facts or preferences mentioned

Conversation to summarize:
{conversation}

Provide a concise summary:"""

    def __init__(
        self,
        config: ContextConfig,
        backend: Optional["LLMBackend"] = None
    ):
        """Initialize the context window.

        Args:
            config: Configuration for context management
            backend: Optional LLM backend for token counting and summarization
        """
        self.config = config
        self.backend = backend
        self.messages: List[AgentMessage] = []
        self.token_count: int = 0
        self._summaries: List[str] = []  # Store past summaries

    async def add(self, message: AgentMessage) -> None:
        """Add a message to the context window.

        Args:
            message: The message to add
        """
        # Estimate tokens if not cached
        if message.token_count is None:
            message.token_count = await self.estimate_tokens(message.content)

        self.messages.append(message)
        self.token_count += message.token_count

        # Check if we need to summarize
        if self.token_count >= self.config.summarization_trigger:
            await self.summarize_if_needed()

    async def add_many(self, messages: List[AgentMessage]) -> None:
        """Add multiple messages to the context window.

        Args:
            messages: List of messages to add
        """
        for message in messages:
            await self.add(message)

    async def get_context(self) -> List[AgentMessage]:
        """Get the current context as a list of messages.

        Returns:
            List of messages suitable for sending to an LLM
        """
        context_messages = []

        # Include summary as system message if we have one
        if self._summaries:
            summary_content = "Previous conversation summary:\n" + "\n\n".join(self._summaries)
            context_messages.append(AgentMessage(
                role="system",
                content=summary_content,
                metadata={"is_summary": True}
            ))

        # Add current messages
        context_messages.extend(self.messages)

        return context_messages

    async def get_context_dicts(self) -> List[Dict[str, Any]]:
        """Get context as list of dictionaries for LLM APIs.

        Returns:
            List of message dictionaries
        """
        messages = await self.get_context()
        return [msg.to_dict() for msg in messages]

    async def summarize_if_needed(self) -> bool:
        """Summarize older messages if context is too large.

        Returns:
            True if summarization was performed
        """
        if self.token_count < self.config.summarization_trigger:
            return False

        if len(self.messages) <= self.config.min_messages_to_keep:
            return False

        # Determine messages to summarize (keep recent ones)
        messages_to_keep = self.config.min_messages_to_keep
        messages_to_summarize = self.messages[:-messages_to_keep]

        # Keep system messages if configured
        if self.config.preserve_system_messages:
            system_messages = [m for m in messages_to_summarize if m.role == "system"]
            messages_to_summarize = [m for m in messages_to_summarize if m.role != "system"]
        else:
            system_messages = []

        if not messages_to_summarize:
            return False

        # Generate summary
        summary = await self._generate_summary(messages_to_summarize)
        if summary:
            self._summaries.append(summary)

        # Update messages list
        kept_messages = self.messages[-messages_to_keep:]
        self.messages = system_messages + kept_messages

        # Recalculate token count
        self.token_count = sum(
            m.token_count or await self.estimate_tokens(m.content)
            for m in self.messages
        )

        logger.info(
            f"Summarized {len(messages_to_summarize)} messages, "
            f"new token count: {self.token_count}"
        )

        return True

    async def _generate_summary(self, messages: List[AgentMessage]) -> Optional[str]:
        """Generate a summary of the given messages.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text or None if no backend available
        """
        if not self.backend:
            # Fallback: create a simple concatenation
            return self._simple_summary(messages)

        # Format conversation for summarization
        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )

        prompt = (self.config.summarization_prompt or self.DEFAULT_SUMMARIZATION_PROMPT)
        prompt = prompt.format(conversation=conversation)

        try:
            from ..backends.backend_base import LLMConfig

            response = await self.backend.complete(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.backend.default_model,
                    max_tokens=1000,
                    temperature=0.3
                )
            )
            return response.content
        except Exception as e:
            logger.warning(f"Summarization failed: {e}, using fallback")
            return self._simple_summary(messages)

    def _simple_summary(self, messages: List[AgentMessage]) -> str:
        """Create a simple summary without LLM.

        Args:
            messages: Messages to summarize

        Returns:
            Simple concatenated summary
        """
        # Extract key information
        topics = []
        for msg in messages:
            # Take first 100 chars of each message
            snippet = msg.content[:100].replace("\n", " ")
            if len(msg.content) > 100:
                snippet += "..."
            topics.append(f"[{msg.role}]: {snippet}")

        return "Previous context (condensed):\n" + "\n".join(topics[-10:])

    async def estimate_tokens(self, text: str) -> int:
        """Estimate the token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self.backend:
            try:
                return await self.backend.count_tokens(text)
            except Exception:
                pass

        # Fallback: rough estimate (~4 chars per token)
        return len(text) // 4

    def clear(self) -> None:
        """Clear all messages and reset the context window."""
        self.messages.clear()
        self._summaries.clear()
        self.token_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context.

        Returns:
            Dictionary with context statistics
        """
        return {
            "message_count": len(self.messages),
            "token_count": self.token_count,
            "summary_count": len(self._summaries),
            "available_tokens": self.config.available_tokens,
            "utilization": self.token_count / self.config.available_tokens if self.config.available_tokens > 0 else 0,
            "messages_by_role": {
                role: sum(1 for m in self.messages if m.role == role)
                for role in set(m.role for m in self.messages)
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context window state.

        Returns:
            Dictionary representation of state
        """
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                    "token_count": m.token_count
                }
                for m in self.messages
            ],
            "summaries": self._summaries,
            "token_count": self.token_count,
            "config": {
                "max_tokens": self.config.max_tokens,
                "reserve_tokens": self.config.reserve_tokens,
                "summarization_threshold": self.config.summarization_threshold,
                "min_messages_to_keep": self.config.min_messages_to_keep,
                "preserve_system_messages": self.config.preserve_system_messages
            }
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        backend: Optional["LLMBackend"] = None
    ) -> "ContextWindow":
        """Restore context window from serialized state.

        Args:
            data: Serialized state dictionary
            backend: Optional LLM backend

        Returns:
            Restored ContextWindow instance
        """
        config_data = data.get("config", {})
        config = ContextConfig(
            max_tokens=config_data.get("max_tokens", 100000),
            reserve_tokens=config_data.get("reserve_tokens", 4000),
            summarization_threshold=config_data.get("summarization_threshold", 0.8),
            min_messages_to_keep=config_data.get("min_messages_to_keep", 4),
            preserve_system_messages=config_data.get("preserve_system_messages", True)
        )

        instance = cls(config, backend)

        for msg_data in data.get("messages", []):
            msg = AgentMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]) if "timestamp" in msg_data else datetime.now(),
                metadata=msg_data.get("metadata", {}),
                token_count=msg_data.get("token_count")
            )
            instance.messages.append(msg)

        instance._summaries = data.get("summaries", [])
        instance.token_count = data.get("token_count", 0)

        return instance
