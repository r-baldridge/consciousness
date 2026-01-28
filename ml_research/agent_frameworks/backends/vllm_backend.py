"""
vLLM local LLM backend implementation.

Provides async access to locally running vLLM server via its
OpenAI-compatible API, with streaming and high-throughput support.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncIterator

from .backend_base import (
    LLMBackend,
    LLMResponse,
    LLMConfig,
    ToolCall,
    BackendError,
    ModelNotFoundError,
)

# Try to import aiohttp for async HTTP
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


class VLLMBackend(LLMBackend):
    """
    vLLM backend with OpenAI-compatible API.

    Features:
    - High-throughput inference
    - Streaming responses
    - OpenAI-compatible API format
    - Continuous batching for efficiency
    - Works with any vLLM-supported model

    vLLM server should be started with:
        python -m vllm.entrypoints.openai.api_server --model <model_name>
    """

    DEFAULT_HOST = "http://localhost:8000"

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the vLLM backend.

        Args:
            host: vLLM server URL (default: http://localhost:8000)
            model: Model being served by vLLM
            api_key: Optional API key if vLLM is configured with auth
            timeout: Request timeout in seconds
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp package is not installed. "
                "Install it with: pip install aiohttp"
            )

        self._host = host.rstrip("/")
        self._default_model = model
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=headers
            )
        return self._session

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        # vLLM supports function calling for compatible models
        return True

    @property
    def supports_vision(self) -> bool:
        # Depends on the model
        return False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate a completion using vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = False

        session = await self._get_session()
        url = f"{self._host}/v1/chat/completions"

        try:
            async with session.post(url, json=request_params) as response:
                if response.status == 404:
                    raise ModelNotFoundError(
                        f"Model '{config.model or self._default_model}' not found on vLLM server."
                    )
                if response.status == 401:
                    raise BackendError("Authentication failed. Check your API key.")

                response.raise_for_status()
                data = await response.json()

                return self._parse_response(data)

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to vLLM at {self._host}. "
                f"Make sure vLLM server is running."
            ) from e
        except aiohttp.ClientResponseError as e:
            raise BackendError(f"vLLM API error: {e}")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion from vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = True

        session = await self._get_session()
        url = f"{self._host}/v1/chat/completions"

        try:
            async with session.post(url, json=request_params) as response:
                if response.status == 404:
                    raise ModelNotFoundError(
                        f"Model '{config.model or self._default_model}' not found."
                    )
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to vLLM at {self._host}."
            ) from e

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Note: vLLM embedding support depends on the model and server configuration.
        The server must be started with an embedding model.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        session = await self._get_session()
        url = f"{self._host}/v1/embeddings"

        request_data = {
            "model": self._default_model,
            "input": text
        }

        try:
            async with session.post(url, json=request_data) as response:
                if response.status == 404:
                    raise NotImplementedError(
                        "Embeddings not available. Ensure vLLM is serving an embedding model."
                    )
                response.raise_for_status()
                data = await response.json()
                return data["data"][0]["embedding"]

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to vLLM at {self._host}."
            ) from e

    async def list_models(self) -> List[str]:
        """
        List available models on the vLLM server.

        Returns:
            List of model names
        """
        session = await self._get_session()
        url = f"{self._host}/v1/models"

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                models = data.get("data", [])
                return [m.get("id", "") for m in models]

        except Exception:
            return [self._default_model]

    async def health_check(self) -> bool:
        """
        Check if vLLM server is running and accessible.

        Returns:
            True if vLLM is healthy
        """
        try:
            session = await self._get_session()
            # vLLM provides a health endpoint
            url = f"{self._host}/health"

            async with session.get(url) as response:
                return response.status == 200
        except Exception:
            # Try models endpoint as fallback
            try:
                url = f"{self._host}/v1/models"
                async with session.get(url) as response:
                    return response.status == 200
            except Exception:
                return False

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently served model.

        Returns:
            Model information dict
        """
        session = await self._get_session()
        url = f"{self._host}/v1/models"

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                models = data.get("data", [])
                if models:
                    return models[0]
                return {}
        except Exception as e:
            raise BackendError(f"Could not get model info: {e}")

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Build vLLM (OpenAI-compatible) API request parameters."""
        # Convert messages - vLLM uses OpenAI format
        vllm_messages = self._convert_messages(messages, config.system)

        params = {
            "model": config.model or self._default_model,
            "messages": vllm_messages,
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
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": config.tool_choice}
                    }

        # Add any extra parameters from metadata
        for key, value in config.metadata.items():
            if key not in params:
                params[key] = value

        return params

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert generic messages to OpenAI/vLLM format."""
        vllm_messages = []

        # Add system message if provided
        if system:
            vllm_messages.append({
                "role": "system",
                "content": system
            })

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                vllm_messages.append({
                    "role": "system",
                    "content": content
                })

            elif role == "assistant":
                assistant_msg = {"role": "assistant"}

                if "tool_calls" in msg:
                    assistant_msg["content"] = content if content else None
                    assistant_msg["tool_calls"] = []
                    for tc in msg["tool_calls"]:
                        tool_call = {
                            "id": tc.get("id", f"call_{id(tc)}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {}))
                            }
                        }
                        assistant_msg["tool_calls"].append(tool_call)
                else:
                    assistant_msg["content"] = content

                vllm_messages.append(assistant_msg)

            elif role == "tool":
                vllm_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content
                })

            else:  # user or other
                vllm_messages.append({
                    "role": "user",
                    "content": content
                })

        return vllm_messages

    def _convert_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert generic tool format to OpenAI/vLLM format."""
        vllm_tools = []

        for tool in tools:
            if "function" in tool:
                # Already OpenAI format
                vllm_tools.append(tool)
            elif "name" in tool:
                # Anthropic-like format
                vllm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {}))
                    }
                })

        return vllm_tools

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse vLLM (OpenAI-compatible) API response to LLMResponse."""
        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content", "") or ""
        tool_calls = []

        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {"raw": func.get("arguments", "")}

                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{id(tc)}"),
                    name=func.get("name", ""),
                    arguments=arguments
                ))

        usage = None
        if "usage" in data:
            usage = {
                "input_tokens": data["usage"].get("prompt_tokens", 0),
                "output_tokens": data["usage"].get("completion_tokens", 0),
            }

        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_use",
        }
        finish_reason = finish_reason_map.get(
            choice.get("finish_reason", "stop"),
            choice.get("finish_reason", "stop")
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=data.get("model"),
            finish_reason=finish_reason,
            raw_response=data
        )

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
