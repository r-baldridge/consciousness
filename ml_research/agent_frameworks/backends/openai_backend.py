"""
OpenAI GPT backend implementation.

Provides async access to GPT models with full tool/function calling support,
streaming, embeddings, and automatic rate limit handling.
"""

import asyncio
import os
import json
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

# Try to import openai
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    AsyncOpenAI = None


class OpenAIBackend(LLMBackend):
    """
    OpenAI GPT backend with full async support.

    Features:
    - Complete tool/function calling support
    - Streaming responses
    - Vision/image input support (GPT-4V)
    - Native embedding support
    - Automatic rate limit handling with exponential backoff
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the OpenAI backend.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            default_model: Default model to use
            embedding_model: Model to use for embeddings
            organization: Optional organization ID
            base_url: Optional base URL for API (for Azure or proxies)
            max_retries: Maximum retries for rate limit errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._default_model = default_model
        self._embedding_model = embedding_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

        client_kwargs = {"api_key": self._api_key}
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

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
        Generate a completion using GPT.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        request_params = self._build_request(messages, config)

        response = await self._make_request_with_retry(
            self._client.chat.completions.create,
            **request_params
        )

        return self._parse_response(response)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion from GPT.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = True

        response = await self._make_request_with_retry(
            self._client.chat.completions.create,
            **request_params
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = await self._make_request_with_retry(
            self._client.embeddings.create,
            model=self._embedding_model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = await self._make_request_with_retry(
            self._client.embeddings.create,
            model=self._embedding_model,
            input=texts
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def list_models(self) -> List[str]:
        """
        List available models from OpenAI.

        Returns:
            List of model identifiers
        """
        try:
            response = await self._client.models.list()
            return [model.id for model in response.data]
        except Exception:
            # Fall back to known models
            return self.SUPPORTED_MODELS.copy()

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: The text to count tokens for

        Returns:
            Token count
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self._default_model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimate
            return len(text) // 4
        except Exception:
            return len(text) // 4

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Build OpenAI API request parameters."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, config.system)

        params = {
            "model": config.model or self._default_model,
            "messages": openai_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        # Add stop sequences
        if config.stop_sequences:
            params["stop"] = config.stop_sequences

        # Add tools if provided
        if config.tools:
            params["tools"] = self._convert_tools(config.tools)

            if config.tool_choice:
                if config.tool_choice in ("auto", "required", "none"):
                    params["tool_choice"] = config.tool_choice
                else:
                    # Specific tool name
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": config.tool_choice}
                    }

        # Add any metadata/extra params
        for key, value in config.metadata.items():
            if key not in params:
                params[key] = value

        return params

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert generic messages to OpenAI format."""
        openai_messages = []

        # Add system message if provided
        if system:
            openai_messages.append({
                "role": "system",
                "content": system
            })

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                openai_messages.append({
                    "role": "system",
                    "content": content
                })

            elif role == "assistant":
                assistant_msg = {"role": "assistant"}

                # Handle tool calls
                if "tool_calls" in msg:
                    assistant_msg["content"] = content if content else None
                    assistant_msg["tool_calls"] = []
                    for tc in msg["tool_calls"]:
                        tool_call = {
                            "id": tc.get("id", f"call_{id(tc)}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", tc.get("input", {})))
                            }
                        }
                        assistant_msg["tool_calls"].append(tool_call)
                else:
                    assistant_msg["content"] = content

                openai_messages.append(assistant_msg)

            elif role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content
                })

            elif role == "user":
                # Handle multimodal content
                if isinstance(content, list):
                    converted_content = []
                    for block in content:
                        if block.get("type") == "image":
                            # Convert from Anthropic format
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                data_url = f"data:{source['media_type']};base64,{source['data']}"
                                converted_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": data_url}
                                })
                            elif source.get("type") == "url":
                                converted_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": source["url"]}
                                })
                        elif block.get("type") == "image_url":
                            # Already OpenAI format
                            converted_content.append(block)
                        elif block.get("type") == "text":
                            converted_content.append(block)
                        else:
                            converted_content.append(block)
                    openai_messages.append({
                        "role": "user",
                        "content": converted_content
                    })
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": content
                    })
            else:
                # Unknown role, try as user
                openai_messages.append({
                    "role": "user",
                    "content": content
                })

        return openai_messages

    def _convert_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert generic tool format to OpenAI format."""
        openai_tools = []

        for tool in tools:
            if "function" in tool:
                # Already OpenAI format
                openai_tools.append(tool)
            elif "name" in tool:
                # Anthropic-like format
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {
                            "type": "object",
                            "properties": {}
                        }))
                    }
                })

        return openai_tools

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI API response to LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments
                ))

        # Map finish reason
        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }
        finish_reason = finish_reason_map.get(
            choice.finish_reason,
            choice.finish_reason
        )

        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
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
            except openai.RateLimitError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = min(
                        self._base_delay * (2 ** attempt),
                        self._max_delay
                    )
                    delay *= (0.5 + 0.5 * (hash(str(attempt)) % 100) / 100)
                    await asyncio.sleep(delay)
                else:
                    raise RateLimitError(
                        f"Rate limit exceeded after {self._max_retries} retries: {e}",
                        retry_after=None
                    )
            except openai.AuthenticationError as e:
                raise AuthenticationError(f"Authentication failed: {e}")
            except openai.NotFoundError as e:
                raise ModelNotFoundError(f"Model not found: {e}")
            except openai.BadRequestError as e:
                if "context_length" in str(e).lower() or "maximum context" in str(e).lower():
                    raise ContextLengthError(f"Context length exceeded: {e}")
                raise BackendError(f"Bad request: {e}")
            except openai.APIError as e:
                if hasattr(e, 'code') and e.code == 'content_filter':
                    raise ContentFilterError(f"Content filtered: {e}")
                raise BackendError(f"API error: {e}")

        if last_error:
            raise last_error

    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()
