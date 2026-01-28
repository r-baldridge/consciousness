"""
LLM Backends for Agent Frameworks.

This module provides unified interfaces to various LLM providers,
enabling easy switching between cloud and local models.

Available backends:
- AnthropicBackend: Claude models via Anthropic API
- OpenAIBackend: GPT models via OpenAI API
- OllamaBackend: Local models via Ollama
- VLLMBackend: High-throughput local inference via vLLM
- LiteLLMBackend: Unified interface to 100+ providers
- RouterBackend: Intelligent routing between multiple backends

Usage:
    from agent_frameworks.backends import AnthropicBackend, LLMConfig

    async with AnthropicBackend() as backend:
        response = await backend.complete(
            messages=[{"role": "user", "content": "Hello!"}],
            config=LLMConfig(model="claude-sonnet-4-20250514")
        )
        print(response.content)
"""

# Base classes and types - always available
from .backend_base import (
    LLMBackend,
    LLMResponse,
    LLMConfig,
    ToolCall,
    FinishReason,
    BackendError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthError,
    ContentFilterError,
)

# Import backends with graceful degradation for missing dependencies
_AVAILABLE_BACKENDS = []

# Anthropic backend
try:
    from .anthropic_backend import AnthropicBackend
    _AVAILABLE_BACKENDS.append("AnthropicBackend")
except ImportError:
    AnthropicBackend = None  # type: ignore

# OpenAI backend
try:
    from .openai_backend import OpenAIBackend
    _AVAILABLE_BACKENDS.append("OpenAIBackend")
except ImportError:
    OpenAIBackend = None  # type: ignore

# Ollama backend
try:
    from .ollama_backend import OllamaBackend
    _AVAILABLE_BACKENDS.append("OllamaBackend")
except ImportError:
    OllamaBackend = None  # type: ignore

# vLLM backend
try:
    from .vllm_backend import VLLMBackend
    _AVAILABLE_BACKENDS.append("VLLMBackend")
except ImportError:
    VLLMBackend = None  # type: ignore

# LiteLLM backend
try:
    from .litellm_backend import LiteLLMBackend
    _AVAILABLE_BACKENDS.append("LiteLLMBackend")
except ImportError:
    LiteLLMBackend = None  # type: ignore

# Router backend (only needs aiohttp for health checks)
try:
    from .router_backend import (
        RouterBackend,
        RouterConfig,
        RoutingStrategy,
        BackendStats,
        ModelCost,
        create_fallback_router,
        create_cost_aware_router,
    )
    _AVAILABLE_BACKENDS.append("RouterBackend")
except ImportError:
    RouterBackend = None  # type: ignore
    RouterConfig = None  # type: ignore
    RoutingStrategy = None  # type: ignore
    BackendStats = None  # type: ignore
    ModelCost = None  # type: ignore
    create_fallback_router = None  # type: ignore
    create_cost_aware_router = None  # type: ignore


def get_available_backends() -> list:
    """Return list of available backend class names."""
    return _AVAILABLE_BACKENDS.copy()


def get_backend(
    provider: str,
    **kwargs
) -> LLMBackend:
    """
    Factory function to get a backend by provider name.

    Args:
        provider: One of "anthropic", "openai", "ollama", "vllm", "litellm"
        **kwargs: Backend-specific configuration

    Returns:
        Configured LLMBackend instance

    Raises:
        ValueError: If provider is unknown
        ImportError: If required package is not installed
    """
    provider = provider.lower()

    if provider == "anthropic":
        if AnthropicBackend is None:
            raise ImportError(
                "anthropic package is required for AnthropicBackend. "
                "Install with: pip install anthropic"
            )
        return AnthropicBackend(**kwargs)

    elif provider == "openai":
        if OpenAIBackend is None:
            raise ImportError(
                "openai package is required for OpenAIBackend. "
                "Install with: pip install openai"
            )
        return OpenAIBackend(**kwargs)

    elif provider == "ollama":
        if OllamaBackend is None:
            raise ImportError(
                "aiohttp package is required for OllamaBackend. "
                "Install with: pip install aiohttp"
            )
        return OllamaBackend(**kwargs)

    elif provider == "vllm":
        if VLLMBackend is None:
            raise ImportError(
                "aiohttp package is required for VLLMBackend. "
                "Install with: pip install aiohttp"
            )
        return VLLMBackend(**kwargs)

    elif provider == "litellm":
        if LiteLLMBackend is None:
            raise ImportError(
                "litellm package is required for LiteLLMBackend. "
                "Install with: pip install litellm"
            )
        return LiteLLMBackend(**kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: anthropic, openai, ollama, vllm, litellm"
        )


# Define exports
__all__ = [
    # Base classes
    "LLMBackend",
    "LLMResponse",
    "LLMConfig",
    "ToolCall",
    "FinishReason",
    # Exceptions
    "BackendError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContextLengthError",
    "ContentFilterError",
    # Backends
    "AnthropicBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "VLLMBackend",
    "LiteLLMBackend",
    "RouterBackend",
    # Router extras
    "RouterConfig",
    "RoutingStrategy",
    "BackendStats",
    "ModelCost",
    "create_fallback_router",
    "create_cost_aware_router",
    # Utilities
    "get_available_backends",
    "get_backend",
]
