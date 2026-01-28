"""
Router backend for intelligent model selection.

Provides automatic routing between multiple backends based on
cost, latency, availability, and task complexity.
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from .backend_base import (
    LLMBackend,
    LLMResponse,
    LLMConfig,
    BackendError,
    RateLimitError,
)


class RoutingStrategy(Enum):
    """Strategies for selecting which backend to use."""
    PRIMARY_FALLBACK = "primary_fallback"  # Use primary, fall back on error
    ROUND_ROBIN = "round_robin"  # Rotate through backends
    LEAST_LATENCY = "least_latency"  # Use fastest backend
    LEAST_COST = "least_cost"  # Use cheapest backend
    COST_AWARE = "cost_aware"  # Route by task complexity
    RANDOM = "random"  # Random selection
    WEIGHTED = "weighted"  # Weighted random selection


@dataclass
class BackendStats:
    """Statistics for a backend."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    consecutive_failures: int = 0

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        if self.successful_requests == 0:
            return float('inf')
        return self.total_latency / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def is_healthy(self) -> bool:
        """Check if backend is considered healthy."""
        # Unhealthy if more than 3 consecutive failures
        if self.consecutive_failures >= 3:
            return False
        # Unhealthy if recent error (within 60 seconds)
        if self.last_error_time and time.time() - self.last_error_time < 60:
            if self.consecutive_failures >= 2:
                return False
        return True


@dataclass
class RouterConfig:
    """Configuration for the router backend."""
    strategy: RoutingStrategy = RoutingStrategy.PRIMARY_FALLBACK
    health_check_interval: float = 60.0  # seconds
    circuit_breaker_threshold: int = 3  # failures before circuit opens
    circuit_breaker_timeout: float = 60.0  # seconds before retry
    cost_threshold: float = 0.01  # cost threshold for routing decisions
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelCost:
    """Cost information for a model."""
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return (
            (input_tokens / 1000) * self.input_cost_per_1k +
            (output_tokens / 1000) * self.output_cost_per_1k
        )


# Default cost estimates for common models
DEFAULT_COSTS: Dict[str, ModelCost] = {
    # OpenAI
    "gpt-4o": ModelCost(0.005, 0.015),
    "gpt-4o-mini": ModelCost(0.00015, 0.0006),
    "gpt-4-turbo": ModelCost(0.01, 0.03),
    "gpt-4": ModelCost(0.03, 0.06),
    "gpt-3.5-turbo": ModelCost(0.0005, 0.0015),
    # Anthropic
    "claude-3-opus-20240229": ModelCost(0.015, 0.075),
    "claude-sonnet-4-20250514": ModelCost(0.003, 0.015),
    "claude-3-5-sonnet-20241022": ModelCost(0.003, 0.015),
    "claude-3-haiku-20240307": ModelCost(0.00025, 0.00125),
    # Local (free)
    "llama3": ModelCost(0.0, 0.0),
    "llama3.1": ModelCost(0.0, 0.0),
    "mistral": ModelCost(0.0, 0.0),
}


class RouterBackend(LLMBackend):
    """
    Intelligent routing backend that selects between multiple backends.

    Features:
    - Primary/fallback model routing
    - Cost-aware routing (use cheap models for simple tasks)
    - Latency-aware routing (use fastest available)
    - Automatic fallback on errors
    - Health checks and circuit breaker
    - Request statistics and monitoring
    """

    def __init__(
        self,
        backends: Dict[str, LLMBackend],
        primary: Optional[str] = None,
        config: Optional[RouterConfig] = None,
        cost_classifier: Optional[Callable[[List[Dict]], Awaitable[str]]] = None,
    ):
        """
        Initialize the router backend.

        Args:
            backends: Dict mapping names to backend instances
            primary: Name of the primary backend (first if not specified)
            config: Router configuration
            cost_classifier: Optional async function to classify task complexity
                            Returns "simple", "medium", or "complex"
        """
        if not backends:
            raise ValueError("At least one backend must be provided")

        self._backends = backends
        self._primary = primary or list(backends.keys())[0]
        self._config = config or RouterConfig()
        self._cost_classifier = cost_classifier

        # Initialize stats for each backend
        self._stats: Dict[str, BackendStats] = {
            name: BackendStats() for name in backends
        }

        # Backend order for iteration
        self._backend_order = list(backends.keys())
        if self._primary in self._backend_order:
            # Move primary to front
            self._backend_order.remove(self._primary)
            self._backend_order.insert(0, self._primary)

        # Round-robin index
        self._rr_index = 0

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def default_model(self) -> str:
        return self._backends[self._primary].default_model

    @property
    def supports_tools(self) -> bool:
        return any(b.supports_tools for b in self._backends.values())

    @property
    def supports_vision(self) -> bool:
        return any(b.supports_vision for b in self._backends.values())

    @property
    def supports_streaming(self) -> bool:
        return any(b.supports_streaming for b in self._backends.values())

    async def start(self) -> None:
        """Start the router (begins health checks)."""
        self._running = True
        if self._config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

    async def stop(self) -> None:
        """Stop the router."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        Generate a completion by routing to the appropriate backend.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Returns:
            LLMResponse with the generated content
        """
        # Select backend(s) based on strategy
        backends_to_try = await self._select_backends(messages, config)

        last_error = None
        for backend_name in backends_to_try:
            backend = self._backends[backend_name]
            stats = self._stats[backend_name]

            # Skip unhealthy backends
            if not stats.is_healthy:
                continue

            start_time = time.time()
            try:
                # Use backend's default model if not specified
                effective_config = config
                if not config.model:
                    effective_config = config.with_model(backend.default_model)

                response = await backend.complete(messages, effective_config)

                # Update stats
                latency = time.time() - start_time
                stats.total_requests += 1
                stats.successful_requests += 1
                stats.total_latency += latency
                stats.consecutive_failures = 0
                if response.usage:
                    stats.total_input_tokens += response.usage.get("input_tokens", 0)
                    stats.total_output_tokens += response.usage.get("output_tokens", 0)

                return response

            except Exception as e:
                # Update failure stats
                stats.total_requests += 1
                stats.failed_requests += 1
                stats.consecutive_failures += 1
                stats.last_error = str(e)
                stats.last_error_time = time.time()
                last_error = e

                # Don't continue to fallback for non-retryable errors
                if isinstance(e, RateLimitError):
                    continue  # Try next backend

        # All backends failed
        if last_error:
            raise last_error
        raise BackendError("All backends are unavailable")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Stream a completion by routing to the appropriate backend.

        Args:
            messages: List of message dicts with 'role' and 'content'
            config: LLM configuration options

        Yields:
            String chunks as they are generated
        """
        # Select backends
        backends_to_try = await self._select_backends(messages, config)

        for backend_name in backends_to_try:
            backend = self._backends[backend_name]
            stats = self._stats[backend_name]

            if not stats.is_healthy:
                continue

            if not backend.supports_streaming:
                continue

            try:
                effective_config = config
                if not config.model:
                    effective_config = config.with_model(backend.default_model)

                async for chunk in backend.stream(messages, effective_config):
                    yield chunk

                # Success
                stats.successful_requests += 1
                stats.consecutive_failures = 0
                return

            except Exception as e:
                stats.failed_requests += 1
                stats.consecutive_failures += 1
                stats.last_error = str(e)
                stats.last_error_time = time.time()

        raise BackendError("No streaming backend available")

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using the first available backend.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        for backend_name in self._backend_order:
            backend = self._backends[backend_name]
            stats = self._stats[backend_name]

            if not stats.is_healthy:
                continue

            try:
                return await backend.embed(text)
            except NotImplementedError:
                continue
            except Exception:
                continue

        raise NotImplementedError("No backend supports embeddings")

    async def _select_backends(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> List[str]:
        """Select backends based on routing strategy."""
        strategy = self._config.strategy

        if strategy == RoutingStrategy.PRIMARY_FALLBACK:
            return self._backend_order

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate through healthy backends
            result = []
            for i in range(len(self._backend_order)):
                idx = (self._rr_index + i) % len(self._backend_order)
                name = self._backend_order[idx]
                if self._stats[name].is_healthy:
                    result.append(name)
            self._rr_index = (self._rr_index + 1) % len(self._backend_order)
            return result if result else self._backend_order

        elif strategy == RoutingStrategy.LEAST_LATENCY:
            # Sort by average latency
            sorted_backends = sorted(
                self._backend_order,
                key=lambda n: self._stats[n].avg_latency
            )
            return sorted_backends

        elif strategy == RoutingStrategy.LEAST_COST:
            # Sort by cost
            sorted_backends = sorted(
                self._backend_order,
                key=lambda n: self._get_backend_cost(n)
            )
            return sorted_backends

        elif strategy == RoutingStrategy.COST_AWARE:
            # Use classifier to determine task complexity
            complexity = "medium"
            if self._cost_classifier:
                try:
                    complexity = await self._cost_classifier(messages)
                except Exception:
                    pass

            return self._select_by_complexity(complexity)

        elif strategy == RoutingStrategy.RANDOM:
            shuffled = self._backend_order.copy()
            random.shuffle(shuffled)
            return shuffled

        elif strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            weights = self._config.weights
            if not weights:
                return self._backend_order

            # Create weighted list
            weighted = []
            for name in self._backend_order:
                weight = weights.get(name, 1.0)
                if self._stats[name].is_healthy and weight > 0:
                    weighted.append((name, weight))

            if not weighted:
                return self._backend_order

            # Sort by weight (highest first)
            weighted.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in weighted]

        return self._backend_order

    def _select_by_complexity(self, complexity: str) -> List[str]:
        """Select backends based on task complexity."""
        # Group backends by cost tier
        cheap = []
        medium = []
        expensive = []

        for name in self._backend_order:
            cost = self._get_backend_cost(name)
            if cost < 0.001:  # < $0.001 per 1K tokens
                cheap.append(name)
            elif cost < 0.01:  # < $0.01 per 1K tokens
                medium.append(name)
            else:
                expensive.append(name)

        if complexity == "simple":
            # Prefer cheap, then medium, then expensive
            return cheap + medium + expensive
        elif complexity == "complex":
            # Prefer expensive (powerful), then medium, then cheap
            return expensive + medium + cheap
        else:  # medium
            # Prefer medium, then cheap, then expensive
            return medium + cheap + expensive

    def _get_backend_cost(self, name: str) -> float:
        """Get estimated cost for a backend."""
        model = self._backends[name].default_model
        if model in DEFAULT_COSTS:
            cost = DEFAULT_COSTS[model]
            # Use average of input and output cost
            return (cost.input_cost_per_1k + cost.output_cost_per_1k) / 2
        return 0.01  # Default medium cost

    async def _health_check_loop(self) -> None:
        """Periodically check backend health."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _run_health_checks(self) -> None:
        """Run health checks on all backends."""
        for name, backend in self._backends.items():
            stats = self._stats[name]

            # Skip if recently checked and healthy
            if stats.is_healthy and stats.consecutive_failures == 0:
                continue

            try:
                is_healthy = await backend.health_check()
                if is_healthy:
                    stats.consecutive_failures = 0
            except Exception:
                stats.consecutive_failures += 1

    def get_stats(self) -> Dict[str, BackendStats]:
        """Get statistics for all backends."""
        return self._stats.copy()

    def get_backend(self, name: str) -> Optional[LLMBackend]:
        """Get a specific backend by name."""
        return self._backends.get(name)

    async def list_models(self) -> List[str]:
        """List models from all backends."""
        models = []
        for backend in self._backends.values():
            try:
                backend_models = await backend.list_models()
                models.extend(backend_models)
            except Exception:
                pass
        return list(set(models))

    async def close(self) -> None:
        """Close all backends and stop the router."""
        await self.stop()
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception:
                pass

    async def __aenter__(self) -> "RouterBackend":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# Convenience function to create a simple primary/fallback router
def create_fallback_router(
    primary: LLMBackend,
    fallback: LLMBackend,
    primary_name: str = "primary",
    fallback_name: str = "fallback",
) -> RouterBackend:
    """
    Create a simple primary/fallback router.

    Args:
        primary: Primary backend to use
        fallback: Fallback backend if primary fails
        primary_name: Name for primary backend
        fallback_name: Name for fallback backend

    Returns:
        RouterBackend configured for primary/fallback
    """
    return RouterBackend(
        backends={primary_name: primary, fallback_name: fallback},
        primary=primary_name,
        config=RouterConfig(strategy=RoutingStrategy.PRIMARY_FALLBACK)
    )


# Convenience function for cost-aware routing
def create_cost_aware_router(
    cheap: LLMBackend,
    expensive: LLMBackend,
    classifier: Optional[Callable[[List[Dict]], Awaitable[str]]] = None,
) -> RouterBackend:
    """
    Create a cost-aware router that uses cheap models for simple tasks.

    Args:
        cheap: Low-cost backend for simple tasks
        expensive: High-cost backend for complex tasks
        classifier: Function to classify task complexity

    Returns:
        RouterBackend configured for cost-aware routing
    """
    return RouterBackend(
        backends={"cheap": cheap, "expensive": expensive},
        primary="cheap",
        config=RouterConfig(strategy=RoutingStrategy.COST_AWARE),
        cost_classifier=classifier
    )
