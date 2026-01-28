"""
LiteLLM unified backend implementation.

Provides async access to 100+ LLM providers through a single
interface using the litellm library.
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

# Try to import litellm
try:
    import litellm
    from litellm import acompletion, aembedding
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    acompletion = None
    aembedding = None


class LiteLLMBackend(LLMBackend):
    """
    LiteLLM unified backend supporting 100+ providers.

    Features:
    - Unified interface for OpenAI, Anthropic, Azure, Cohere, Replicate,
      Hugging Face, Together AI, Bedrock, Vertex AI, and many more
    - Automatic provider detection from model name
    - Streaming support for all providers
    - Unified tool calling format
    - Cost tracking

    Model naming convention:
    - OpenAI: gpt-4, gpt-3.5-turbo
    - Anthropic: claude-3-opus-20240229, claude-sonnet-4-20250514
    - Azure: azure/gpt-4-deployment-name
    - Bedrock: bedrock/anthropic.claude-v2
    - Vertex: vertex_ai/gemini-pro
    - Together: together_ai/togethercomputer/llama-2-70b-chat
    - Replicate: replicate/llama-2-70b-chat
    - Cohere: command, command-nightly
    - Hugging Face: huggingface/bigcode/starcoder
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 300.0,
        drop_params: bool = True,
        **kwargs,
    ):
        """
        Initialize the LiteLLM backend.

        Args:
            default_model: Default model to use (provider/model format)
            api_key: API key (or set provider-specific env vars)
            api_base: Custom API base URL
            max_retries: Maximum retries on failure
            timeout: Request timeout in seconds
            drop_params: Drop unsupported params instead of erroring
            **kwargs: Additional litellm configuration options
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm package is not installed. "
                "Install it with: pip install litellm"
            )

        self._default_model = default_model
        self._api_key = api_key
        self._api_base = api_base
        self._max_retries = max_retries
        self._timeout = timeout
        self._extra_kwargs = kwargs

        # Configure litellm
        litellm.drop_params = drop_params
        litellm.num_retries = max_retries

        if api_key:
            # Set as environment variable for litellm to pick up
            # LiteLLM auto-detects which provider based on model name
            pass

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        # Most modern models support tools
        return True

    @property
    def supports_vision(self) -> bool:
        # Depends on the model
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
        Generate a completion using LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        request_params = self._build_request(messages, config)

        try:
            response = await acompletion(**request_params)
            return self._parse_response(response)

        except Exception as e:
            self._handle_exception(e)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion using LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = True

        try:
            response = await acompletion(**request_params)

            async for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        except Exception as e:
            self._handle_exception(e)

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Uses the appropriate embedding model based on provider.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Determine embedding model based on default model provider
        model = self._default_model
        if model.startswith("gpt") or model.startswith("openai"):
            embedding_model = "text-embedding-3-small"
        elif model.startswith("claude") or model.startswith("anthropic"):
            # Anthropic doesn't have embeddings, use OpenAI
            embedding_model = "text-embedding-3-small"
        elif model.startswith("bedrock"):
            embedding_model = "bedrock/amazon.titan-embed-text-v1"
        elif model.startswith("vertex"):
            embedding_model = "vertex_ai/textembedding-gecko"
        elif model.startswith("cohere"):
            embedding_model = "embed-english-v3.0"
        else:
            embedding_model = "text-embedding-3-small"

        try:
            response = await aembedding(
                model=embedding_model,
                input=[text],
                api_key=self._api_key,
            )
            return response.data[0]["embedding"]

        except Exception as e:
            self._handle_exception(e)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Same logic as embed for model selection
        model = self._default_model
        if model.startswith("gpt") or model.startswith("openai"):
            embedding_model = "text-embedding-3-small"
        else:
            embedding_model = "text-embedding-3-small"

        try:
            response = await aembedding(
                model=embedding_model,
                input=texts,
                api_key=self._api_key,
            )
            # Sort by index
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

        except Exception as e:
            self._handle_exception(e)

    async def list_models(self) -> List[str]:
        """
        List available models.

        Returns a curated list of popular models across providers.
        """
        return [
            # OpenAI
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            # Anthropic
            "claude-3-opus-20240229",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            # Google
            "gemini/gemini-1.5-pro",
            "gemini/gemini-1.5-flash",
            "vertex_ai/gemini-pro",
            # Azure
            "azure/gpt-4",
            "azure/gpt-35-turbo",
            # AWS Bedrock
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-v2",
            "bedrock/amazon.titan-text-express-v1",
            # Together AI
            "together_ai/meta-llama/Llama-3-70b-chat-hf",
            "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            # Cohere
            "command",
            "command-nightly",
            # Replicate
            "replicate/meta/llama-2-70b-chat",
            # Groq
            "groq/llama3-70b-8192",
            "groq/mixtral-8x7b-32768",
        ]

    async def get_model_cost(
        self,
        model: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get cost per token for a model.

        Args:
            model: Model name (uses default if not provided)

        Returns:
            Dict with 'input' and 'output' cost per 1K tokens
        """
        model = model or self._default_model

        try:
            cost_info = litellm.get_model_cost_map(model)
            return {
                "input": cost_info.get("input_cost_per_token", 0) * 1000,
                "output": cost_info.get("output_cost_per_token", 0) * 1000,
            }
        except Exception:
            return {"input": 0.0, "output": 0.0}

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count tokens for

        Returns:
            Token count
        """
        try:
            return litellm.token_counter(
                model=self._default_model,
                text=text
            )
        except Exception:
            # Fallback
            return len(text) // 4

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Build LiteLLM request parameters."""
        # LiteLLM accepts OpenAI format
        litellm_messages = self._convert_messages(messages, config.system)

        params = {
            "model": config.model or self._default_model,
            "messages": litellm_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "timeout": self._timeout,
        }

        if self._api_key:
            params["api_key"] = self._api_key

        if self._api_base:
            params["api_base"] = self._api_base

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
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": config.tool_choice}
                    }

        # Add extra kwargs
        params.update(self._extra_kwargs)

        # Add metadata options
        for key, value in config.metadata.items():
            if key not in params:
                params[key] = value

        return params

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format (used by LiteLLM)."""
        litellm_messages = []

        # Add system message
        if system:
            litellm_messages.append({
                "role": "system",
                "content": system
            })

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                litellm_messages.append({
                    "role": "system",
                    "content": content
                })

            elif role == "assistant":
                assistant_msg = {"role": "assistant"}

                if "tool_calls" in msg:
                    assistant_msg["content"] = content if content else None
                    assistant_msg["tool_calls"] = []
                    for tc in msg["tool_calls"]:
                        args = tc.get("arguments", tc.get("input", {}))
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        tool_call = {
                            "id": tc.get("id", f"call_{id(tc)}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": args
                            }
                        }
                        assistant_msg["tool_calls"].append(tool_call)
                else:
                    assistant_msg["content"] = content

                litellm_messages.append(assistant_msg)

            elif role == "tool":
                litellm_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content
                })

            else:  # user
                if isinstance(content, list):
                    # Multimodal content - pass through
                    litellm_messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    litellm_messages.append({
                        "role": "user",
                        "content": content
                    })

        return litellm_messages

    def _convert_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        litellm_tools = []

        for tool in tools:
            if "function" in tool:
                litellm_tools.append(tool)
            elif "name" in tool:
                litellm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {}))
                    }
                })

        return litellm_tools

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response to LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = []

        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    arguments = {"raw": str(tc.function.arguments)}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments
                ))

        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
            "end_turn": "stop",
        }
        finish_reason = finish_reason_map.get(
            choice.finish_reason,
            choice.finish_reason
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model if hasattr(response, 'model') else None,
            finish_reason=finish_reason,
            raw_response=response
        )

    def _handle_exception(self, e: Exception) -> None:
        """Convert litellm exceptions to our exceptions."""
        error_str = str(e).lower()

        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(f"Authentication failed: {e}")
        elif "rate limit" in error_str:
            raise RateLimitError(f"Rate limit exceeded: {e}")
        elif "not found" in error_str or "does not exist" in error_str:
            raise ModelNotFoundError(f"Model not found: {e}")
        elif "context" in error_str and "length" in error_str:
            raise ContextLengthError(f"Context length exceeded: {e}")
        elif "content" in error_str and "filter" in error_str:
            raise ContentFilterError(f"Content filtered: {e}")
        else:
            raise BackendError(f"LiteLLM error: {e}")

    async def close(self) -> None:
        """Clean up resources."""
        pass  # LiteLLM manages its own connections
