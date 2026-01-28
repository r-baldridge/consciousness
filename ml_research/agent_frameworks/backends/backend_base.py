"""
Abstract base class for LLM backends.

This module defines the interface that all LLM backends must implement,
along with common data classes for responses and configuration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum


class FinishReason(Enum):
    """Reasons why the model stopped generating."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a tool/function call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM backend.

    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls if the model requested any
        usage: Token usage statistics
        model: The model that generated the response
        finish_reason: Why the model stopped generating
        raw_response: The original response from the provider (for debugging)
    """
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0) if self.usage else 0

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0) if self.usage else 0

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class LLMConfig:
    """
    Configuration for LLM requests.

    Attributes:
        model: The model identifier to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        stop_sequences: Sequences that stop generation
        tools: Tool definitions for function calling
        tool_choice: How to handle tool selection ("auto", "required", "none", or specific tool)
        system: System message/prompt
        metadata: Additional provider-specific options
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    system: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_model(self, model: str) -> "LLMConfig":
        """Create a copy with a different model."""
        return LLMConfig(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences,
            tools=self.tools,
            tool_choice=self.tool_choice,
            system=self.system,
            metadata=self.metadata.copy()
        )

    def with_temperature(self, temperature: float) -> "LLMConfig":
        """Create a copy with a different temperature."""
        return LLMConfig(
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences,
            tools=self.tools,
            tool_choice=self.tool_choice,
            system=self.system,
            metadata=self.metadata.copy()
        )


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All backends must implement the core methods for completion,
    streaming, and embedding. Properties indicate capability support.
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this backend supports tool/function calling."""
        ...

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this backend supports image inputs."""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming responses."""
        ...

    @property
    def name(self) -> str:
        """Return the backend name."""
        return self.__class__.__name__

    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and accessible.

        Returns:
            True if the backend is operational
        """
        try:
            # Simple test completion
            config = LLMConfig(model=self.default_model, max_tokens=5)
            await self.complete(
                [{"role": "user", "content": "Hi"}],
                config
            )
            return True
        except Exception:
            return False

    @property
    def default_model(self) -> str:
        """Return the default model for this backend."""
        return "unknown"

    async def list_models(self) -> List[str]:
        """
        List available models for this backend.

        Returns:
            List of model identifiers
        """
        return [self.default_model]

    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Default implementation uses a rough estimate.
        Backends should override with accurate counting.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    async def close(self) -> None:
        """Clean up any resources held by the backend."""
        pass

    async def __aenter__(self) -> "LLMBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class BackendError(Exception):
    """Base exception for backend errors."""
    pass


class AuthenticationError(BackendError):
    """Raised when authentication fails."""
    pass


class RateLimitError(BackendError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelNotFoundError(BackendError):
    """Raised when the requested model is not available."""
    pass


class ContextLengthError(BackendError):
    """Raised when input exceeds context window."""
    pass


class ContentFilterError(BackendError):
    """Raised when content is filtered/blocked."""
    pass
