"""
Ollama local LLM backend implementation.

Provides async access to locally running Ollama models with
streaming support and model management.
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


class OllamaBackend(LLMBackend):
    """
    Ollama local LLM backend with full async support.

    Features:
    - Streaming responses
    - Model listing and management
    - Embedding generation
    - Pull models on demand
    - Works with any Ollama-compatible model
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
    ):
        """
        Initialize the Ollama backend.

        Args:
            host: Ollama server URL (default: http://localhost:11434)
            model: Default model to use
            timeout: Request timeout in seconds
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp package is not installed. "
                "Install it with: pip install aiohttp"
            )

        self._host = host.rstrip("/")
        self._default_model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        # Ollama has experimental tool support in newer versions
        return True

    @property
    def supports_vision(self) -> bool:
        # Depends on the model (llava, bakllava, etc.)
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
        Generate a completion using Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = False

        session = await self._get_session()
        url = f"{self._host}/api/chat"

        try:
            async with session.post(url, json=request_params) as response:
                if response.status == 404:
                    raise ModelNotFoundError(
                        f"Model '{config.model or self._default_model}' not found. "
                        f"Try: ollama pull {config.model or self._default_model}"
                    )
                response.raise_for_status()
                data = await response.json()

                return self._parse_response(data)

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to Ollama at {self._host}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e
        except aiohttp.ClientResponseError as e:
            raise BackendError(f"Ollama API error: {e}")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion from Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        request_params = self._build_request(messages, config)
        request_params["stream"] = True

        session = await self._get_session()
        url = f"{self._host}/api/chat"

        try:
            async with session.post(url, json=request_params) as response:
                if response.status == 404:
                    raise ModelNotFoundError(
                        f"Model '{config.model or self._default_model}' not found. "
                        f"Try: ollama pull {config.model or self._default_model}"
                    )
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to Ollama at {self._host}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        session = await self._get_session()
        url = f"{self._host}/api/embeddings"

        request_data = {
            "model": self._default_model,
            "prompt": text
        }

        try:
            async with session.post(url, json=request_data) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("embedding", [])

        except aiohttp.ClientConnectorError as e:
            raise BackendError(
                f"Could not connect to Ollama at {self._host}."
            ) from e

    async def list_models(self) -> List[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        session = await self._get_session()
        url = f"{self._host}/api/tags"

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models]

        except aiohttp.ClientConnectorError:
            return []
        except Exception:
            return []

    async def pull_model(self, model: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Pull/download a model from Ollama library.

        Args:
            model: Model name to pull

        Yields:
            Progress updates as dicts
        """
        session = await self._get_session()
        url = f"{self._host}/api/pull"

        request_data = {
            "name": model,
            "stream": True
        }

        async with session.post(url, json=request_data) as response:
            response.raise_for_status()

            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def delete_model(self, model: str) -> bool:
        """
        Delete a model from Ollama.

        Args:
            model: Model name to delete

        Returns:
            True if successful
        """
        session = await self._get_session()
        url = f"{self._host}/api/delete"

        request_data = {"name": model}

        try:
            async with session.delete(url, json=request_data) as response:
                return response.status == 200
        except Exception:
            return False

    async def model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Model name

        Returns:
            Model information dict
        """
        session = await self._get_session()
        url = f"{self._host}/api/show"

        request_data = {"name": model}

        try:
            async with session.post(url, json=request_data) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            raise BackendError(f"Could not get model info: {e}")

    async def health_check(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama is healthy
        """
        try:
            session = await self._get_session()
            url = f"{self._host}/api/tags"

            async with session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """Build Ollama API request parameters."""
        # Convert messages
        ollama_messages = self._convert_messages(messages)

        params = {
            "model": config.model or self._default_model,
            "messages": ollama_messages,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            }
        }

        # Add stop sequences
        if config.stop_sequences:
            params["options"]["stop"] = config.stop_sequences

        # Add tools if provided (experimental in Ollama)
        if config.tools:
            params["tools"] = self._convert_tools(config.tools)

        # Add system message
        if config.system:
            # Ollama supports system in options
            params["system"] = config.system

        return params

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert generic messages to Ollama format."""
        ollama_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles
            if role == "tool":
                role = "assistant"  # Ollama doesn't have a tool role

            ollama_msg = {
                "role": role,
                "content": content if isinstance(content, str) else ""
            }

            # Handle images for multimodal models
            if isinstance(content, list):
                images = []
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            images.append(source.get("data", ""))
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Extract base64 data
                            _, data = url.split(";base64,", 1)
                            images.append(data)

                ollama_msg["content"] = "\n".join(text_parts)
                if images:
                    ollama_msg["images"] = images

            ollama_messages.append(ollama_msg)

        return ollama_messages

    def _convert_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert generic tool format to Ollama format."""
        ollama_tools = []

        for tool in tools:
            if "function" in tool:
                # OpenAI format
                func = tool["function"]
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    }
                })
            elif "name" in tool:
                # Anthropic format
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {}))
                    }
                })

        return ollama_tools

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse Ollama API response to LLMResponse."""
        message = data.get("message", {})
        content = message.get("content", "")

        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{id(tc)}"),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {})
                ))

        # Ollama provides various metrics
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            }

        finish_reason = "stop"
        if data.get("done_reason"):
            finish_reason = data["done_reason"]

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
