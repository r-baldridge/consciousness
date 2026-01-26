"""
Intelligent Task Router for ML Research Orchestrator

Provides intelligent routing between TRM, RLM, Mamba, and other architectures
based on task characteristics, resource constraints, and historical performance.

This module implements INTEG-001: Orchestrator Enhancement with:
- ArchitectureRegistry: Central registry of architecture capabilities
- TaskRouter: Intelligent routing logic
- UnifiedIndex: Technique-to-architecture mapping
- FallbackHandler: Graceful failure handling
- MLOrchestrator: Main coordination class

Example:
    from modern_dev.orchestrator.router import MLOrchestrator

    orch = MLOrchestrator()
    result = orch.process(Request(
        task_type="code_repair",
        input_data={"buggy_code": "..."},
    ))
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class TaskCategory(Enum):
    """Categories of tasks for routing decisions."""
    CODE_REPAIR = "code_repair"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    LONG_CONTEXT = "long_context"
    STREAMING = "streaming"
    REASONING = "reasoning"
    PUZZLE_SOLVING = "puzzle_solving"
    TEXT_GENERATION = "text_generation"
    MEMORY_INTENSIVE = "memory_intensive"


class InferenceSpeed(Enum):
    """Inference speed categories."""
    FAST = "fast"       # < 50ms per token
    MEDIUM = "medium"   # 50-200ms per token
    SLOW = "slow"       # > 200ms per token


class MemoryRequirement(Enum):
    """Memory requirement categories."""
    LOW = "low"         # < 2GB
    MEDIUM = "medium"   # 2-8GB
    HIGH = "high"       # > 8GB


class RoutingStrategy(Enum):
    """Strategies for task routing."""
    CAPABILITY_MATCH = "capability_match"   # Match based on capabilities
    PERFORMANCE_BASED = "performance_based"  # Use historical performance
    RESOURCE_AWARE = "resource_aware"       # Optimize for resources
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality


class FailureAction(Enum):
    """Actions to take on failure."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ABORT = "abort"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ArchitectureCapability:
    """
    Describes what an architecture can do.

    Captures the strengths, weaknesses, and operational characteristics
    of a model architecture to enable intelligent routing decisions.

    Attributes:
        name: Human-readable name of the architecture
        supported_tasks: List of task categories this architecture handles
        max_context_length: Maximum input context in tokens
        inference_speed: Speed category for inference
        memory_requirement: Memory usage category
        strengths: List of specific strengths
        weaknesses: List of known limitations
        optimal_input_size: Ideal input size range (min, max) in tokens
        batch_size_range: Supported batch size range (min, max)
        supports_streaming: Whether streaming inference is supported
        supports_early_stopping: Whether early stopping is supported
    """
    name: str
    supported_tasks: List[str]
    max_context_length: int
    inference_speed: str
    memory_requirement: str
    strengths: List[str]
    weaknesses: List[str] = field(default_factory=list)
    optimal_input_size: Tuple[int, int] = (1, 4096)
    batch_size_range: Tuple[int, int] = (1, 32)
    supports_streaming: bool = False
    supports_early_stopping: bool = False

    def matches_task(self, task: str) -> bool:
        """Check if this architecture supports a given task."""
        return task in self.supported_tasks

    def matches_constraints(self, constraints: Dict[str, Any]) -> bool:
        """Check if architecture meets the given constraints."""
        # Check context length
        required_context = constraints.get("context_length", 0)
        if required_context > self.max_context_length:
            return False

        # Check memory
        max_memory = constraints.get("max_memory", None)
        if max_memory is not None:
            mem_map = {"low": 2, "medium": 8, "high": 16}
            if mem_map.get(self.memory_requirement, 16) > max_memory:
                return False

        # Check latency requirements
        max_latency = constraints.get("max_latency_ms", None)
        if max_latency is not None:
            speed_map = {"fast": 50, "medium": 200, "slow": 500}
            if speed_map.get(self.inference_speed, 500) > max_latency:
                return False

        return True

    def compute_score(
        self,
        task: str,
        constraints: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute a capability score for this architecture given task and constraints.

        Args:
            task: The task type
            constraints: Resource/performance constraints
            weights: Optional weights for different scoring factors

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.matches_task(task):
            return 0.0

        if not self.matches_constraints(constraints):
            return 0.0

        weights = weights or {
            "task_match": 0.4,
            "strength_match": 0.2,
            "resource_efficiency": 0.2,
            "context_headroom": 0.2,
        }

        score = 0.0

        # Base score for task match
        score += weights.get("task_match", 0.4)

        # Bonus for strengths that match the task
        task_keywords = task.lower().replace("_", " ").split()
        strength_matches = sum(
            1 for s in self.strengths
            if any(kw in s.lower() for kw in task_keywords)
        )
        score += weights.get("strength_match", 0.2) * min(strength_matches / 3, 1.0)

        # Resource efficiency (prefer lower memory for same capability)
        mem_efficiency = {"low": 1.0, "medium": 0.7, "high": 0.4}
        score += weights.get("resource_efficiency", 0.2) * mem_efficiency.get(
            self.memory_requirement, 0.4
        )

        # Context headroom
        required_context = constraints.get("context_length", 0)
        if self.max_context_length > 0 and required_context > 0:
            headroom = min(self.max_context_length / required_context, 2.0) / 2.0
            score += weights.get("context_headroom", 0.2) * headroom
        else:
            score += weights.get("context_headroom", 0.2)

        return min(score, 1.0)


@dataclass
class TaskAnalysis:
    """
    Analysis of a task to determine routing requirements.

    Attributes:
        task_type: Identified task category
        estimated_complexity: Complexity score (0.0 to 1.0)
        estimated_context_length: Approximate context length needed
        requires_iteration: Whether iterative refinement is needed
        requires_long_context: Whether long context processing is needed
        requires_streaming: Whether streaming output is needed
        priority: Task priority (higher = more important)
        metadata: Additional analysis metadata
    """
    task_type: str
    estimated_complexity: float
    estimated_context_length: int
    requires_iteration: bool = False
    requires_long_context: bool = False
    requires_streaming: bool = False
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureMatch:
    """
    A match between task requirements and an architecture.

    Attributes:
        architecture_name: Name of the matched architecture
        score: Match score (0.0 to 1.0)
        reasons: List of reasons for this match
        warnings: Any warnings about this choice
    """
    architecture_name: str
    score: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """
    Result of a routing decision.

    Attributes:
        primary: Primary architecture to use
        fallback: Backup architecture if primary fails
        reasoning: Explanation of why this routing was chosen
        confidence: Confidence in this decision (0.0 to 1.0)
        estimated_latency_ms: Estimated execution latency
        estimated_memory_mb: Estimated memory usage
    """
    primary: str
    fallback: Optional[str]
    reasoning: str
    confidence: float
    estimated_latency_ms: float = 0.0
    estimated_memory_mb: float = 0.0


@dataclass
class Task:
    """
    A task to be processed by the orchestrator.

    Attributes:
        task_id: Unique identifier
        task_type: Type/category of the task
        input_data: Input data for the task
        constraints: Resource/performance constraints
        metadata: Additional task metadata
    """
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        task_type: str,
        input_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> "Task":
        """Create a task with auto-generated ID."""
        import uuid
        return cls(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            input_data=input_data,
            constraints=constraints or {},
        )


@dataclass
class ExecutionResult:
    """
    Result from executing a task.

    Attributes:
        success: Whether execution succeeded
        output: Output from the execution
        architecture_used: Architecture that processed the task
        execution_time_ms: Time taken for execution
        attempts: Number of attempts made
        error: Error message if failed
        trace: Execution trace for debugging
    """
    success: bool
    output: Any
    architecture_used: str
    execution_time_ms: float
    attempts: int = 1
    error: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    """
    A request to the MLOrchestrator.

    Attributes:
        task_type: Type of task to perform
        input_data: Input data for processing
        constraints: Optional resource constraints
        technique: Optional specific technique to use
        preferred_architecture: Optional architecture preference
        config: Optional configuration overrides
    """
    task_type: str
    input_data: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    technique: Optional[str] = None
    preferred_architecture: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    """
    Response from the MLOrchestrator.

    Attributes:
        success: Whether processing succeeded
        output: Processing output
        architecture_used: Architecture that handled the request
        technique_used: Technique applied (if any)
        execution_time_ms: Total execution time
        routing_decision: How the request was routed
        metadata: Additional response metadata
    """
    success: bool
    output: Any
    architecture_used: str
    technique_used: Optional[str] = None
    execution_time_ms: float = 0.0
    routing_decision: Optional[RoutingDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ARCHITECTURE REGISTRY
# =============================================================================


class ArchitectureRegistry:
    """
    Registry of available architectures and their capabilities.

    Provides a central location for registering, querying, and managing
    architecture implementations and their capability metadata.

    Example:
        registry = ArchitectureRegistry()

        registry.register("trm", ArchitectureCapability(
            name="TRM",
            supported_tasks=["code_repair", "iterative_refinement"],
            max_context_length=3072,
            inference_speed="medium",
            memory_requirement="low",
            strengths=["recursive reasoning", "code repair"],
        ), lambda: load_trm())

        capable = registry.get_capable("code_repair")
    """

    def __init__(self):
        """Initialize the architecture registry."""
        self._architectures: Dict[str, ArchitectureCapability] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}

    def register(
        self,
        name: str,
        capability: ArchitectureCapability,
        model_factory: Callable[[], Any],
    ) -> None:
        """
        Register an architecture with its capabilities and factory.

        Args:
            name: Unique identifier for the architecture
            capability: ArchitectureCapability describing what it can do
            model_factory: Callable that creates/loads the model instance
        """
        self._architectures[name] = capability
        self._factories[name] = model_factory
        self._performance_history[name] = []
        logger.info(f"Registered architecture: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an architecture.

        Args:
            name: Name of architecture to unregister

        Returns:
            True if architecture was registered and removed
        """
        if name in self._architectures:
            del self._architectures[name]
            del self._factories[name]
            self._instances.pop(name, None)
            self._performance_history.pop(name, None)
            logger.info(f"Unregistered architecture: {name}")
            return True
        return False

    def get_capability(self, name: str) -> Optional[ArchitectureCapability]:
        """Get capability info for an architecture."""
        return self._architectures.get(name)

    def get_instance(self, name: str) -> Optional[Any]:
        """
        Get or create a model instance.

        Args:
            name: Architecture name

        Returns:
            Model instance or None if not registered
        """
        if name not in self._factories:
            return None

        if name not in self._instances:
            try:
                self._instances[name] = self._factories[name]()
                logger.info(f"Created instance for: {name}")
            except Exception as e:
                logger.error(f"Failed to create instance for {name}: {e}")
                return None

        return self._instances[name]

    def get_capable(
        self,
        task: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get architectures capable of handling a task.

        Args:
            task: Task type to check
            constraints: Optional constraints to filter by

        Returns:
            List of architecture names that can handle the task
        """
        constraints = constraints or {}
        capable = []

        for name, capability in self._architectures.items():
            if capability.matches_task(task) and capability.matches_constraints(constraints):
                capable.append(name)

        return capable

    def get_ranked(
        self,
        task: str,
        constraints: Optional[Dict[str, Any]] = None,
        strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
    ) -> List[Tuple[str, float]]:
        """
        Get architectures ranked by suitability for a task.

        Args:
            task: Task type
            constraints: Optional constraints
            strategy: Ranking strategy to use

        Returns:
            List of (architecture_name, score) tuples, sorted by score descending
        """
        constraints = constraints or {}
        scores = []

        for name, capability in self._architectures.items():
            if strategy == RoutingStrategy.PERFORMANCE_BASED:
                # Use historical performance
                score = self._compute_performance_score(name, task)
            else:
                # Use capability matching
                score = capability.compute_score(task, constraints)

            if score > 0:
                scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def record_performance(
        self,
        name: str,
        task: str,
        success: bool,
        latency_ms: float,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Record performance data for future routing decisions.

        Args:
            name: Architecture name
            task: Task type that was performed
            success: Whether execution succeeded
            latency_ms: Execution latency
            quality_score: Optional quality metric
        """
        if name in self._performance_history:
            self._performance_history[name].append({
                "task": task,
                "success": success,
                "latency_ms": latency_ms,
                "quality_score": quality_score,
                "timestamp": time.time(),
            })

            # Keep only recent history
            max_history = 100
            if len(self._performance_history[name]) > max_history:
                self._performance_history[name] = \
                    self._performance_history[name][-max_history:]

    def _compute_performance_score(self, name: str, task: str) -> float:
        """Compute score based on historical performance."""
        history = self._performance_history.get(name, [])

        # Filter to relevant task history
        relevant = [h for h in history if h["task"] == task]

        if not relevant:
            # No history, fall back to capability score
            cap = self._architectures.get(name)
            if cap:
                return cap.compute_score(task, {})
            return 0.0

        # Compute success rate
        success_rate = sum(1 for h in relevant if h["success"]) / len(relevant)

        # Compute average latency (normalized)
        avg_latency = sum(h["latency_ms"] for h in relevant) / len(relevant)
        latency_score = max(0, 1 - (avg_latency / 1000))  # Normalize to 1 second

        # Compute average quality if available
        quality_scores = [h["quality_score"] for h in relevant if h["quality_score"]]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

        # Weighted combination
        return 0.4 * success_rate + 0.3 * latency_score + 0.3 * avg_quality

    def list_architectures(self) -> List[str]:
        """List all registered architecture names."""
        return list(self._architectures.keys())

    def get_all_capabilities(self) -> Dict[str, ArchitectureCapability]:
        """Get all registered capabilities."""
        return dict(self._architectures)

    def clear_instances(self) -> None:
        """Clear all cached model instances."""
        self._instances.clear()
        logger.info("Cleared all architecture instances")


# =============================================================================
# TASK ROUTER
# =============================================================================


class TaskRouter:
    """
    Routes tasks to optimal architecture based on multiple factors.

    Considers:
    - Task type (code repair, generation, analysis)
    - Input characteristics (length, complexity)
    - Resource constraints (memory, latency)
    - Historical performance

    Example:
        router = TaskRouter(registry)

        task = Task.create("code_repair", {"buggy_code": "..."})
        decision = router.route(task)

        print(f"Use {decision.primary}, fallback: {decision.fallback}")
        print(f"Reasoning: {decision.reasoning}")
    """

    # Task type to architecture affinity mapping
    TASK_AFFINITY = {
        "code_repair": ["trm", "rlm"],
        "iterative_refinement": ["trm"],
        "code_generation": ["rlm", "mamba"],
        "code_analysis": ["mamba", "trm"],
        "long_context": ["mamba"],
        "streaming": ["mamba"],
        "reasoning": ["trm"],
        "puzzle_solving": ["trm"],
        "text_generation": ["mamba", "rlm"],
        "generation": ["mamba", "rlm"],
        "analysis": ["mamba", "trm"],
    }

    def __init__(
        self,
        registry: ArchitectureRegistry,
        default_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
    ):
        """
        Initialize the task router.

        Args:
            registry: ArchitectureRegistry to query for capabilities
            default_strategy: Default routing strategy to use
        """
        self.registry = registry
        self.default_strategy = default_strategy
        self._routing_history: List[Dict[str, Any]] = []

    def route(
        self,
        task: Task,
        strategy: Optional[RoutingStrategy] = None,
    ) -> RoutingDecision:
        """
        Determine best architecture for a task.

        Args:
            task: Task to route
            strategy: Optional override for routing strategy

        Returns:
            RoutingDecision with primary, fallback, and reasoning
        """
        strategy = strategy or self.default_strategy

        # Analyze the task
        analysis = self._analyze_task(task)

        # Match capabilities
        matches = self._match_capabilities(analysis, task.constraints, strategy)

        if not matches:
            return RoutingDecision(
                primary="none",
                fallback=None,
                reasoning="No suitable architecture found for task requirements",
                confidence=0.0,
            )

        # Select primary and fallback
        primary = matches[0]
        fallback = matches[1] if len(matches) > 1 else None

        # Build reasoning
        reasoning = self._build_reasoning(analysis, primary, fallback)

        # Estimate performance
        primary_cap = self.registry.get_capability(primary.architecture_name)
        latency_map = {"fast": 50, "medium": 200, "slow": 500}
        memory_map = {"low": 500, "medium": 4000, "high": 12000}

        decision = RoutingDecision(
            primary=primary.architecture_name,
            fallback=fallback.architecture_name if fallback else None,
            reasoning=reasoning,
            confidence=primary.score,
            estimated_latency_ms=latency_map.get(
                primary_cap.inference_speed if primary_cap else "medium",
                200
            ),
            estimated_memory_mb=memory_map.get(
                primary_cap.memory_requirement if primary_cap else "medium",
                4000
            ),
        )

        # Record for analytics
        self._routing_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "decision": decision.primary,
            "confidence": decision.confidence,
            "timestamp": time.time(),
        })

        return decision

    def _analyze_task(self, task: Task) -> TaskAnalysis:
        """
        Analyze task to determine requirements.

        Args:
            task: Task to analyze

        Returns:
            TaskAnalysis with identified requirements
        """
        input_data = task.input_data

        # Estimate complexity
        complexity = self._estimate_complexity(input_data)

        # Estimate context length
        context_length = self._estimate_context_length(input_data)

        # Determine requirements
        requires_iteration = task.task_type in [
            "code_repair",
            "iterative_refinement",
            "puzzle_solving",
        ]

        requires_long_context = context_length > 4096

        requires_streaming = task.constraints.get("streaming", False)

        return TaskAnalysis(
            task_type=task.task_type,
            estimated_complexity=complexity,
            estimated_context_length=context_length,
            requires_iteration=requires_iteration,
            requires_long_context=requires_long_context,
            requires_streaming=requires_streaming,
            priority=task.constraints.get("priority", 1),
            metadata={
                "original_task_type": task.task_type,
                "has_constraints": bool(task.constraints),
            },
        )

    def _estimate_complexity(self, input_data: Dict[str, Any]) -> float:
        """Estimate task complexity from input data."""
        complexity = 0.5  # Base complexity

        # Check for code inputs
        code = input_data.get("buggy_code", input_data.get("code", ""))
        if code:
            # More lines = more complex
            lines = code.count("\n") + 1
            complexity += min(0.3, lines / 100 * 0.3)

            # Nested structures indicate complexity
            nesting = code.count("{") + code.count("(")
            complexity += min(0.2, nesting / 50 * 0.2)

        # Check for explicit complexity hint
        if "complexity" in input_data:
            complexity = input_data["complexity"]

        return min(complexity, 1.0)

    def _estimate_context_length(self, input_data: Dict[str, Any]) -> int:
        """Estimate required context length from input data."""
        total_length = 0

        # Count characters and estimate tokens (rough: 4 chars per token)
        for key, value in input_data.items():
            if isinstance(value, str):
                total_length += len(value) // 4
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total_length += len(item) // 4

        return max(total_length, 100)  # Minimum 100 tokens

    def _match_capabilities(
        self,
        requirements: TaskAnalysis,
        constraints: Dict[str, Any],
        strategy: RoutingStrategy,
    ) -> List[ArchitectureMatch]:
        """
        Match requirements to architecture capabilities.

        Args:
            requirements: Analyzed task requirements
            constraints: Resource constraints
            strategy: Routing strategy

        Returns:
            List of ArchitectureMatch objects, sorted by score
        """
        matches = []

        # Get ranked architectures
        ranked = self.registry.get_ranked(
            requirements.task_type,
            constraints,
            strategy,
        )

        # Build match objects with reasons
        for arch_name, score in ranked:
            capability = self.registry.get_capability(arch_name)
            if not capability:
                continue

            reasons = []
            warnings = []

            # Check task affinity
            if arch_name in self.TASK_AFFINITY.get(requirements.task_type, []):
                reasons.append(f"High affinity for {requirements.task_type}")
                score += 0.1

            # Check strengths
            for strength in capability.strengths:
                if requirements.task_type.replace("_", " ") in strength.lower():
                    reasons.append(f"Strength: {strength}")

            # Check for iteration support
            if requirements.requires_iteration:
                if capability.supports_early_stopping:
                    reasons.append("Supports early stopping for iteration")
                else:
                    warnings.append("Task benefits from iteration support")

            # Check context requirements
            if requirements.requires_long_context:
                if capability.max_context_length >= requirements.estimated_context_length:
                    reasons.append(f"Sufficient context length ({capability.max_context_length})")
                else:
                    warnings.append("Context length may be insufficient")
                    score *= 0.8

            # Check streaming
            if requirements.requires_streaming:
                if capability.supports_streaming:
                    reasons.append("Supports streaming output")
                else:
                    warnings.append("Streaming not supported")
                    score *= 0.9

            if not reasons:
                reasons.append("Meets basic task requirements")

            matches.append(ArchitectureMatch(
                architecture_name=arch_name,
                score=min(score, 1.0),
                reasons=reasons,
                warnings=warnings,
            ))

        # Sort by score
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _build_reasoning(
        self,
        analysis: TaskAnalysis,
        primary: ArchitectureMatch,
        fallback: Optional[ArchitectureMatch],
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = [
            f"Task type: {analysis.task_type}",
            f"Complexity: {analysis.estimated_complexity:.2f}",
            f"Context length: {analysis.estimated_context_length} tokens",
        ]

        parts.append(f"\nPrimary: {primary.architecture_name} (score: {primary.score:.2f})")
        for reason in primary.reasons:
            parts.append(f"  - {reason}")
        for warning in primary.warnings:
            parts.append(f"  ! {warning}")

        if fallback:
            parts.append(f"\nFallback: {fallback.architecture_name} (score: {fallback.score:.2f})")
            for reason in fallback.reasons[:2]:  # Limit fallback reasons
                parts.append(f"  - {reason}")

        return "\n".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_routings": 0}

        # Count by architecture
        arch_counts: Dict[str, int] = {}
        for entry in self._routing_history:
            arch = entry["decision"]
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

        # Average confidence
        avg_confidence = sum(
            e["confidence"] for e in self._routing_history
        ) / len(self._routing_history)

        return {
            "total_routings": len(self._routing_history),
            "architecture_distribution": arch_counts,
            "average_confidence": avg_confidence,
        }


# =============================================================================
# UNIFIED INDEX
# =============================================================================


class UnifiedIndex:
    """
    Index connecting techniques to architectures.

    Maps: Technique -> Compatible Architectures -> Optimal Configuration

    This enables the orchestrator to determine which architecture to use
    for a specific technique, and what configuration to apply.

    Example:
        index = UnifiedIndex()

        index.register_technique(
            "chain_of_thought",
            architectures=["trm", "mamba"],
            configs={
                "trm": {"reasoning_steps": 10},
                "mamba": {"temperature": 0.7},
            }
        )

        config = index.get_config("chain_of_thought", "trm")
    """

    def __init__(self):
        """Initialize the unified index."""
        self._technique_map: Dict[str, List[str]] = {}
        self._architecture_configs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._technique_metadata: Dict[str, Dict[str, Any]] = {}

    def register_technique(
        self,
        technique: str,
        architectures: List[str],
        configs: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a technique with its compatible architectures and configs.

        Args:
            technique: Technique identifier
            architectures: List of compatible architecture names
            configs: Dict mapping architecture name to configuration
            metadata: Optional additional metadata
        """
        self._technique_map[technique] = architectures
        self._architecture_configs[technique] = configs

        if metadata:
            self._technique_metadata[technique] = metadata

        logger.info(f"Registered technique: {technique} -> {architectures}")

    def get_architectures(self, technique: str) -> List[str]:
        """Get architectures compatible with a technique."""
        return self._technique_map.get(technique, [])

    def get_config(
        self,
        technique: str,
        architecture: str,
    ) -> Dict[str, Any]:
        """
        Get configuration for a technique-architecture pair.

        Args:
            technique: Technique identifier
            architecture: Architecture name

        Returns:
            Configuration dict, or empty dict if not found
        """
        technique_configs = self._architecture_configs.get(technique, {})
        return technique_configs.get(architecture, {})

    def get_best_architecture(
        self,
        technique: str,
        available: List[str],
    ) -> Optional[str]:
        """
        Get the best available architecture for a technique.

        Args:
            technique: Technique identifier
            available: List of currently available architectures

        Returns:
            Best architecture name, or None if no match
        """
        compatible = self._technique_map.get(technique, [])

        for arch in compatible:
            if arch in available:
                return arch

        return None

    def get_techniques_for_architecture(self, architecture: str) -> List[str]:
        """Get all techniques that can use a specific architecture."""
        techniques = []

        for technique, archs in self._technique_map.items():
            if architecture in archs:
                techniques.append(technique)

        return techniques

    def get_metadata(self, technique: str) -> Dict[str, Any]:
        """Get metadata for a technique."""
        return self._technique_metadata.get(technique, {})

    def list_techniques(self) -> List[str]:
        """List all registered techniques."""
        return list(self._technique_map.keys())


# =============================================================================
# FALLBACK HANDLER
# =============================================================================


class FallbackHandler:
    """
    Handles failures and routes to backup architectures.

    Provides robust execution with automatic retry and fallback mechanisms.

    Example:
        handler = FallbackHandler(router)

        result = handler.execute_with_fallback(task, max_attempts=3)
        if result.success:
            print(f"Succeeded with {result.architecture_used}")
        else:
            print(f"Failed after {result.attempts} attempts: {result.error}")
    """

    def __init__(
        self,
        router: TaskRouter,
        default_max_attempts: int = 3,
        retry_delay_ms: float = 100,
    ):
        """
        Initialize the fallback handler.

        Args:
            router: TaskRouter for routing decisions
            default_max_attempts: Default maximum attempts
            retry_delay_ms: Delay between retries
        """
        self.router = router
        self.default_max_attempts = default_max_attempts
        self.retry_delay_ms = retry_delay_ms
        self._failure_history: List[Dict[str, Any]] = []

    def execute_with_fallback(
        self,
        task: Task,
        executor: Callable[[str, Task], ExecutionResult],
        max_attempts: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute task with automatic fallback on failure.

        Args:
            task: Task to execute
            executor: Function that executes task with given architecture
            max_attempts: Maximum attempts (uses default if not specified)

        Returns:
            ExecutionResult with final outcome
        """
        max_attempts = max_attempts or self.default_max_attempts

        # Get routing decision
        decision = self.router.route(task)

        if decision.primary == "none":
            return ExecutionResult(
                success=False,
                output=None,
                architecture_used="none",
                execution_time_ms=0,
                error="No suitable architecture found",
            )

        architectures_to_try = [decision.primary]
        if decision.fallback:
            architectures_to_try.append(decision.fallback)

        trace = []
        total_time = 0
        last_error = None

        for attempt in range(max_attempts):
            # Select architecture for this attempt
            arch_index = min(attempt, len(architectures_to_try) - 1)
            architecture = architectures_to_try[arch_index]

            trace.append({
                "attempt": attempt + 1,
                "architecture": architecture,
                "timestamp": time.time(),
            })

            try:
                start = time.time()
                result = executor(architecture, task)
                elapsed = (time.time() - start) * 1000
                total_time += elapsed

                trace[-1]["elapsed_ms"] = elapsed
                trace[-1]["success"] = result.success

                if result.success:
                    # Record success for future routing
                    self.router.registry.record_performance(
                        architecture,
                        task.task_type,
                        success=True,
                        latency_ms=elapsed,
                    )

                    result.trace = trace
                    result.attempts = attempt + 1
                    return result

                # Execution returned but indicated failure
                last_error = result.error or "Execution returned failure"
                trace[-1]["error"] = last_error

            except Exception as e:
                last_error = str(e)
                trace[-1]["error"] = last_error
                trace[-1]["exception"] = type(e).__name__

                logger.warning(
                    f"Attempt {attempt + 1} failed on {architecture}: {e}"
                )

            # Determine action
            action = self._handle_failure(task, last_error, attempt, max_attempts)
            trace[-1]["action"] = action.value

            if action == FailureAction.ABORT:
                break

            if action == FailureAction.RETRY:
                time.sleep(self.retry_delay_ms / 1000)

        # Record failure
        self._failure_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "attempts": len(trace),
            "architectures_tried": [t["architecture"] for t in trace],
            "final_error": last_error,
            "timestamp": time.time(),
        })

        return ExecutionResult(
            success=False,
            output=None,
            architecture_used=trace[-1]["architecture"] if trace else "none",
            execution_time_ms=total_time,
            attempts=len(trace),
            error=last_error,
            trace=trace,
        )

    def _handle_failure(
        self,
        task: Task,
        error: str,
        attempt: int,
        max_attempts: int,
    ) -> FailureAction:
        """
        Determine action on failure.

        Args:
            task: The task that failed
            error: Error message
            attempt: Current attempt number (0-indexed)
            max_attempts: Maximum attempts allowed

        Returns:
            FailureAction indicating what to do next
        """
        # Last attempt - abort
        if attempt >= max_attempts - 1:
            return FailureAction.ABORT

        # Check for fatal errors that shouldn't retry
        fatal_patterns = [
            "not found",
            "not registered",
            "invalid input",
            "permission denied",
        ]

        error_lower = error.lower()
        for pattern in fatal_patterns:
            if pattern in error_lower:
                return FailureAction.ABORT

        # Check for transient errors that should retry same architecture
        transient_patterns = [
            "timeout",
            "connection",
            "temporary",
            "busy",
        ]

        for pattern in transient_patterns:
            if pattern in error_lower:
                return FailureAction.RETRY

        # Default: try fallback (which happens naturally with next attempt)
        return FailureAction.FALLBACK

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics."""
        if not self._failure_history:
            return {"total_failures": 0}

        # Count by error type
        error_counts: Dict[str, int] = {}
        for entry in self._failure_history:
            error = entry.get("final_error", "unknown")[:50]
            error_counts[error] = error_counts.get(error, 0) + 1

        # Average attempts before failure
        avg_attempts = sum(
            e["attempts"] for e in self._failure_history
        ) / len(self._failure_history)

        return {
            "total_failures": len(self._failure_history),
            "error_distribution": error_counts,
            "average_attempts": avg_attempts,
        }


# =============================================================================
# ML ORCHESTRATOR
# =============================================================================


class MLOrchestrator:
    """
    Main orchestrator for ML research pipeline.

    Coordinates:
    - TRM for recursive refinement and code repair
    - RLM for code synthesis
    - Mamba for long-context processing

    Provides a high-level interface for processing requests through the
    optimal architecture pipeline.

    Example:
        orch = MLOrchestrator()

        response = orch.process(Request(
            task_type="code_repair",
            input_data={"buggy_code": "def foo(): retrun 1"},
        ))

        if response.success:
            print(f"Repaired code: {response.output}")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ML orchestrator.

        Args:
            config_path: Optional path to configuration file
        """
        self.registry = ArchitectureRegistry()
        self.router = TaskRouter(self.registry)
        self.fallback = FallbackHandler(self.router)
        self.index = UnifiedIndex()

        self._config_path = config_path
        self._initialized = False

        # Setup defaults
        self._setup_defaults()
        self._initialized = True

    def _setup_defaults(self) -> None:
        """Register default architectures and techniques."""
        # Register TRM
        self.registry.register(
            "trm",
            ArchitectureCapability(
                name="TRM",
                supported_tasks=[
                    "code_repair",
                    "iterative_refinement",
                    "reasoning",
                    "puzzle_solving",
                ],
                max_context_length=3072,  # 64x48 grid
                inference_speed="medium",
                memory_requirement="low",
                strengths=[
                    "recursive reasoning",
                    "code repair",
                    "iterative refinement",
                    "puzzle solving",
                    "ARC-AGI tasks",
                ],
                weaknesses=[
                    "limited context length",
                    "not suitable for long sequences",
                ],
                supports_early_stopping=True,
            ),
            self._load_trm,
        )

        # Register Mamba
        self.registry.register(
            "mamba",
            ArchitectureCapability(
                name="Mamba",
                supported_tasks=[
                    "generation",
                    "long_context",
                    "analysis",
                    "text_generation",
                    "streaming",
                ],
                max_context_length=100000,
                inference_speed="fast",
                memory_requirement="medium",
                strengths=[
                    "long sequences",
                    "efficient inference",
                    "linear complexity",
                    "streaming output",
                ],
                weaknesses=[
                    "less effective at iterative refinement",
                    "no built-in recursion",
                ],
                supports_streaming=True,
            ),
            self._load_mamba,
        )

        # Register RLM
        self.registry.register(
            "rlm",
            ArchitectureCapability(
                name="RLM",
                supported_tasks=[
                    "code_generation",
                    "code_synthesis",
                    "text_generation",
                    "generation",
                ],
                max_context_length=8192,
                inference_speed="medium",
                memory_requirement="medium",
                strengths=[
                    "code generation",
                    "variable extraction",
                    "constraint satisfaction",
                    "structured output",
                ],
                weaknesses=[
                    "requires well-formed specifications",
                    "less flexible for freeform generation",
                ],
            ),
            self._load_rlm,
        )

        # Register techniques
        self._setup_default_techniques()

    def _setup_default_techniques(self) -> None:
        """Setup default technique mappings."""
        # Chain of thought
        self.index.register_technique(
            "chain_of_thought",
            architectures=["trm", "mamba"],
            configs={
                "trm": {"reasoning_steps": 10, "use_halting": True},
                "mamba": {"temperature": 0.7, "max_tokens": 500},
            },
        )

        # Code repair
        self.index.register_technique(
            "trm_code_repair",
            architectures=["trm"],
            configs={
                "trm": {"max_repair_iterations": 16, "target_language": "python"},
            },
        )

        # RAG
        self.index.register_technique(
            "mamba_rag",
            architectures=["mamba"],
            configs={
                "mamba": {"max_context_tokens": 32000, "top_k": 10},
            },
        )

        # Streaming
        self.index.register_technique(
            "mamba_streaming",
            architectures=["mamba"],
            configs={
                "mamba": {"buffer_size": 32, "latency_target_ms": 50},
            },
        )

        # Code synthesis
        self.index.register_technique(
            "rlm_synthesis",
            architectures=["rlm", "mamba"],
            configs={
                "rlm": {"max_recursion": 5, "variable_extraction": "hybrid"},
                "mamba": {"temperature": 0.3, "max_tokens": 1000},
            },
        )

        # Recursive decomposition
        self.index.register_technique(
            "recursive_decomposition",
            architectures=["trm", "rlm"],
            configs={
                "trm": {"max_depth": 5, "use_halting": True},
                "rlm": {"max_depth": 5, "infer_intermediates": True},
            },
        )

    def _load_trm(self) -> Any:
        """Load TRM model."""
        try:
            from ..trm.src.model import TRM, TRMConfig
            return TRM(TRMConfig())
        except ImportError:
            logger.warning("TRM not available, using placeholder")
            return None

    def _load_mamba(self) -> Any:
        """Load Mamba model."""
        try:
            from ..mamba_impl.src.model import Mamba, MambaConfig
            return Mamba(MambaConfig())
        except ImportError:
            logger.warning("Mamba not available, using placeholder")
            return None

    def _load_rlm(self) -> Any:
        """Load RLM technique."""
        try:
            from ...ml_techniques.code_synthesis.rlm import RLMExtractor
            return RLMExtractor()
        except ImportError:
            logger.warning("RLM not available, using placeholder")
            return None

    def process(self, request: Request) -> Response:
        """
        Process a request through the optimal pipeline.

        Args:
            request: Request to process

        Returns:
            Response with results
        """
        start_time = time.time()

        # Create task from request
        task = Task.create(
            task_type=request.task_type,
            input_data=request.input_data,
            constraints=request.constraints or {},
        )

        # Add preferred architecture if specified
        if request.preferred_architecture:
            task.constraints["preferred_architecture"] = request.preferred_architecture

        # Get routing decision
        decision = self.router.route(task)

        # Check for technique-specific routing
        if request.technique:
            technique_arch = self.index.get_best_architecture(
                request.technique,
                self.registry.list_architectures(),
            )
            if technique_arch:
                decision = RoutingDecision(
                    primary=technique_arch,
                    fallback=decision.primary if decision.primary != technique_arch else decision.fallback,
                    reasoning=f"Using {technique_arch} for technique {request.technique}",
                    confidence=decision.confidence,
                )

        # Execute with fallback
        result = self.fallback.execute_with_fallback(
            task,
            self._execute_on_architecture,
        )

        execution_time = (time.time() - start_time) * 1000

        return Response(
            success=result.success,
            output=result.output,
            architecture_used=result.architecture_used,
            technique_used=request.technique,
            execution_time_ms=execution_time,
            routing_decision=decision,
            metadata={
                "attempts": result.attempts,
                "trace": result.trace,
            },
        )

    def _execute_on_architecture(
        self,
        architecture: str,
        task: Task,
    ) -> ExecutionResult:
        """
        Execute a task on a specific architecture.

        Args:
            architecture: Architecture to use
            task: Task to execute

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        # Get model instance
        model = self.registry.get_instance(architecture)

        if model is None:
            return ExecutionResult(
                success=False,
                output=None,
                architecture_used=architecture,
                execution_time_ms=0,
                error=f"Architecture {architecture} not available",
            )

        try:
            # Execute based on architecture type
            if architecture == "trm":
                output = self._execute_trm(model, task)
            elif architecture == "mamba":
                output = self._execute_mamba(model, task)
            elif architecture == "rlm":
                output = self._execute_rlm(model, task)
            else:
                output = {"note": f"Generic execution on {architecture}"}

            return ExecutionResult(
                success=True,
                output=output,
                architecture_used=architecture,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Execution error on {architecture}: {e}")
            return ExecutionResult(
                success=False,
                output=None,
                architecture_used=architecture,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _execute_trm(self, model: Any, task: Task) -> Dict[str, Any]:
        """Execute task using TRM."""
        if model is None:
            return {"error": "TRM model not loaded"}

        # TRM-specific execution
        input_data = task.input_data

        if task.task_type == "code_repair":
            return {
                "status": "processed",
                "model": "TRM",
                "task": task.task_type,
                "input_preview": str(input_data)[:100],
            }

        return {
            "status": "processed",
            "model": "TRM",
            "task": task.task_type,
        }

    def _execute_mamba(self, model: Any, task: Task) -> Dict[str, Any]:
        """Execute task using Mamba."""
        if model is None:
            return {"error": "Mamba model not loaded"}

        # Mamba-specific execution
        return {
            "status": "processed",
            "model": "Mamba",
            "task": task.task_type,
        }

    def _execute_rlm(self, model: Any, task: Task) -> Dict[str, Any]:
        """Execute task using RLM."""
        if model is None:
            return {"error": "RLM not loaded"}

        # RLM-specific execution
        if hasattr(model, "run"):
            spec = task.input_data.get("specification", str(task.input_data))
            result = model.run(spec)
            return {
                "status": "processed",
                "model": "RLM",
                "success": result.success,
                "output": result.output,
            }

        return {
            "status": "processed",
            "model": "RLM",
            "task": task.task_type,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "initialized": self._initialized,
            "architectures": self.registry.list_architectures(),
            "techniques": self.index.list_techniques(),
            "routing_stats": self.router.get_statistics(),
            "failure_stats": self.fallback.get_failure_statistics(),
        }

    def get_architecture_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an architecture."""
        cap = self.registry.get_capability(name)
        if cap is None:
            return None

        return {
            "name": cap.name,
            "supported_tasks": cap.supported_tasks,
            "max_context_length": cap.max_context_length,
            "inference_speed": cap.inference_speed,
            "memory_requirement": cap.memory_requirement,
            "strengths": cap.strengths,
            "weaknesses": cap.weaknesses,
            "techniques": self.index.get_techniques_for_architecture(name),
        }

    def list_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """List all architecture capabilities."""
        result = {}
        for name in self.registry.list_architectures():
            info = self.get_architecture_info(name)
            if info:
                result[name] = info
        return result


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "TaskCategory",
    "InferenceSpeed",
    "MemoryRequirement",
    "RoutingStrategy",
    "FailureAction",
    # Data classes
    "ArchitectureCapability",
    "TaskAnalysis",
    "ArchitectureMatch",
    "RoutingDecision",
    "Task",
    "ExecutionResult",
    "Request",
    "Response",
    # Core classes
    "ArchitectureRegistry",
    "TaskRouter",
    "UnifiedIndex",
    "FallbackHandler",
    "MLOrchestrator",
]
