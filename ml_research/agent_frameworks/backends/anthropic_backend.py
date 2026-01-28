"""
Anthropic Claude backend implementation.

Provides async access to Claude models with full tool use support,
streaming, and automatic rate limit handling.
"""

import asyncio
import os
import json
import uuid
from typing import List, Dict, Any, Optional, AsyncIterator

from .backend_base import (
    LLMBackend,
    LLMResponse,
    LLMConfig,
    ToolCall,
    BackendError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthError,
    ContentFilterError,
)

# Try to import anthropic
try:
    import anthropic
    from anthropic import AsyncAnthropic, APIError, AuthenticationError as AnthropicAuthError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    AsyncAnthropic = None


class AnthropicBackend(LLMBackend):
    """
    Anthropic Claude backend with full async support.

    Features:
    - Complete tool/function calling support
    - Streaming responses
    - Vision/image input support
    - Automatic rate limit handling with exponential backoff
    - Proper Anthropic API message format conversion
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    SUPPORTED_MODELS = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the Anthropic backend.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_model: Default model to use
            max_retries: Maximum retries for rate limit errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._default_model = default_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

        self._client = AsyncAnthropic(api_key=self._api_key)

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate a completion using Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        request_params = self._build_request(messages, config)

        response = await self._make_request_with_retry(
            self._client.messages.create,
            **request_params
        )

        return self._parse_response(response)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion from Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        request_params = self._build_request(messages, config)

        async with self._client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Note: Anthropic doesn't have a native embedding API.
        This raises NotImplementedError.

        For embeddings with Anthropic models, consider using
        a third-party embedding service or the voyageai package.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Consider using VoyageAI (voyage-ai package) for embeddings "
            "that work well with Claude, or use OpenAI's embedding API."
        )

    async def list_models(self) -> List[str]:
        """Return list of available Claude models."""
        return self.SUPPORTED_MODELS.copy()

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Anthropic's tokenizer.

        Args:
            text: The text to count tokens for

        Returns:
            Token count
        """
        try:
            # Anthropic provides a token counting endpoint
            response = await self._client.messages.count_tokens(
                model=self._default_model,
                messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens
        except Exception:
            # Fallback to rough estimate
            return len(text) // 4

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Build Anthropic API request parameters."""
        # Convert messages to Anthropic format
        anthropic_messages, system = self._convert_messages(messages)

        params = {
            "model": config.model or self._default_model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        # Add system message
        if config.system:
            params["system"] = config.system
        elif system:
            params["system"] = system

        # Add stop sequences
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences

        # Add tools if provided
        if config.tools:
            params["tools"] = self._convert_tools(config.tools)

            if config.tool_choice:
                if config.tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif config.tool_choice == "required":
                    params["tool_choice"] = {"type": "any"}
                elif config.tool_choice == "none":
                    # Don't include tool_choice to disable
                    pass
                else:
                    # Specific tool name
                    params["tool_choice"] = {
                        "type": "tool",
                        "name": config.tool_choice
                    }

        # Add any metadata/extra params
        for key, value in config.metadata.items():
            if key not in params:
                params[key] = value

        return params

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert generic messages to Anthropic format.

        Returns:
            Tuple of (converted messages, system message if extracted)
        """
        anthropic_messages = []
        system_message = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Extract system message
                system_message = content
                continue

            if role == "assistant":
                # Handle assistant messages with tool calls
                if "tool_calls" in msg:
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", str(uuid.uuid4())),
                            "name": tc["name"],
                            "input": tc.get("arguments", tc.get("input", {}))
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content
                    })

            elif role == "tool":
                # Tool result message
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content
                    }]
                })

            elif role == "user":
                # Handle multimodal content
                if isinstance(content, list):
                    # Already in content block format
                    converted_content = []
                    for block in content:
                        if block.get("type") == "image_url":
                            # Convert from OpenAI format
                            url = block["image_url"]["url"]
                            if url.startswith("data:"):
                                # Base64 encoded
                                media_type, data = url.split(";base64,")
                                media_type = media_type.replace("data:", "")
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                })
                            else:
                                # URL (Anthropic supports this too)
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                })
                        elif block.get("type") == "image":
                            # Already Anthropic format
                            converted_content.append(block)
                        elif block.get("type") == "text":
                            converted_content.append(block)
                        else:
                            # Unknown block type, try to use as-is
                            converted_content.append(block)
                    anthropic_messages.append({
                        "role": "user",
                        "content": converted_content
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": content
                    })
            else:
                # Unknown role, try as user
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })

        return anthropic_messages, system_message

    def _convert_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert generic tool format to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            if "function" in tool:
                # OpenAI format
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {}
                    })
                })
            elif "name" in tool:
                # Already Anthropic-like format
                anthropic_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", tool.get("parameters", {
                        "type": "object",
                        "properties": {}
                    }))
                })

        return anthropic_tools

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic API response to LLMResponse."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))

        # Map finish reason
        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_use",
        }
        finish_reason = finish_reason_map.get(
            response.stop_reason,
            response.stop_reason
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            finish_reason=finish_reason,
            raw_response=response
        )

    async def _make_request_with_retry(
        self,
        func,
        **kwargs
    ) -> Any:
        """Make API request with exponential backoff on rate limits."""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                return await func(**kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < self._max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self._base_delay * (2 ** attempt),
                        self._max_delay
                    )
                    # Add jitter
                    delay *= (0.5 + 0.5 * (hash(str(attempt)) % 100) / 100)
                    await asyncio.sleep(delay)
                else:
                    raise RateLimitError(
                        f"Rate limit exceeded after {self._max_retries} retries: {e}",
                        retry_after=None
                    )
            except anthropic.AuthenticationError as e:
                raise AuthenticationError(f"Authentication failed: {e}")
            except anthropic.NotFoundError as e:
                raise ModelNotFoundError(f"Model not found: {e}")
            except anthropic.BadRequestError as e:
                if "context_length" in str(e).lower():
                    raise ContextLengthError(f"Context length exceeded: {e}")
                raise BackendError(f"Bad request: {e}")
            except anthropic.APIError as e:
                # Check for content filter
                if "content" in str(e).lower() and "filter" in str(e).lower():
                    raise ContentFilterError(f"Content filtered: {e}")
                raise BackendError(f"API error: {e}")

        if last_error:
            raise last_error

    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()
