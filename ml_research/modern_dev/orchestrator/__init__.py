"""
Architecture Orchestrator

A modular orchestration system that can dynamically select and launch
appropriate architectures based on task requirements.

Usage:
    from consciousness.ml_research.modern_dev.orchestrator import Orchestrator

    orch = Orchestrator()
    result = orch.run(task="Generate image from text", input_data={"prompt": "..."})

    # Or specify architecture directly
    result = orch.run(task="...", architecture="jepa", input_data={...})
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import importlib


class TaskType(Enum):
    """Types of tasks architectures can handle."""
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    IMAGE_SEGMENTATION = "image_segmentation"
    VIDEO_UNDERSTANDING = "video_understanding"
    VIDEO_GENERATION = "video_generation"

    # Language
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_MODELING = "sequence_modeling"
    LONG_CONTEXT = "long_context"

    # Multimodal
    VISION_LANGUAGE = "vision_language"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_TEXT = "image_to_text"

    # Reasoning
    REASONING = "reasoning"
    PLANNING = "planning"
    WORLD_MODELING = "world_modeling"

    # Specialized
    ROBOTICS = "robotics"
    CONTINUOUS_DYNAMICS = "continuous_dynamics"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class TaskSpec:
    """Specification for a task to be executed."""
    task_type: TaskType
    input_data: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    # Constraints can include:
    #   - max_latency_ms: int
    #   - max_memory_gb: float
    #   - context_length: int
    #   - quality_priority: float (0-1, higher = prioritize quality over speed)
    #   - preferred_architecture: Optional[str]


@dataclass
class TaskResult:
    """Result from executing a task."""
    success: bool
    output: Any
    architecture_used: str
    execution_time_ms: float
    memory_used_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ArchitectureBase(ABC):
    """
    Abstract base class that all architecture implementations must inherit.

    This ensures a consistent interface for the orchestrator to interact with
    any architecture regardless of its internal implementation.
    """

    # Class-level metadata (override in subclasses)
    ARCHITECTURE_ID: str = "base"
    SUPPORTED_TASKS: List[TaskType] = []
    MAX_CONTEXT_LENGTH: int = 0
    MEMORY_REQUIREMENT_GB: float = 0.0

    @abstractmethod
    def load(self, checkpoint_path: Optional[str] = None, device: str = "cuda") -> None:
        """Load model weights and prepare for inference."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def run(self, task_spec: TaskSpec) -> TaskResult:
        """Execute a task and return results."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass

    @classmethod
    def supports_task(cls, task_type: TaskType) -> bool:
        """Check if this architecture supports a given task type."""
        return task_type in cls.SUPPORTED_TASKS

    @classmethod
    def get_capability_score(cls, task_spec: TaskSpec) -> float:
        """
        Score how well this architecture fits the task (0-1).
        Higher = better fit.

        Override in subclasses for architecture-specific scoring.
        """
        if not cls.supports_task(task_spec.task_type):
            return 0.0

        score = 0.5  # Base score for supported tasks

        # Adjust for context length requirements
        required_context = task_spec.constraints.get("context_length", 0)
        if required_context > 0:
            if cls.MAX_CONTEXT_LENGTH >= required_context:
                score += 0.2
            elif cls.MAX_CONTEXT_LENGTH < required_context:
                return 0.0  # Can't handle required context

        # Adjust for memory constraints
        max_memory = task_spec.constraints.get("max_memory_gb", float("inf"))
        if cls.MEMORY_REQUIREMENT_GB <= max_memory:
            score += 0.1
        else:
            return 0.0  # Exceeds memory constraint

        return min(score, 1.0)


# Architecture capability mapping
ARCHITECTURE_CAPABILITIES = {
    "ctm": {
        "tasks": [
            TaskType.CONTINUOUS_DYNAMICS,
            TaskType.REASONING,
            TaskType.PLANNING,
            TaskType.SEQUENCE_MODELING,
        ],
        "strengths": ["adaptive computation", "neural synchronization", "maze solving"],
        "context_length": 8192,
        "memory_gb": 8.0,
    },
    "jepa": {
        "tasks": [
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.VIDEO_UNDERSTANDING,
            TaskType.VISION_LANGUAGE,
            TaskType.WORLD_MODELING,
        ],
        "strengths": ["self-supervised", "latent prediction", "video understanding"],
        "context_length": 0,  # Not sequence-based
        "memory_gb": 16.0,
    },
    "mamba": {
        "tasks": [
            TaskType.TEXT_GENERATION,
            TaskType.SEQUENCE_MODELING,
            TaskType.LONG_CONTEXT,
        ],
        "strengths": ["linear complexity", "infinite context", "fast inference"],
        "context_length": 1000000,  # Effectively unlimited
        "memory_gb": 8.0,
    },
    "xlstm": {
        "tasks": [
            TaskType.TEXT_GENERATION,
            TaskType.SEQUENCE_MODELING,
            TaskType.LONG_CONTEXT,
        ],
        "strengths": ["exponential gating", "matrix memory", "parallelizable"],
        "context_length": 100000,
        "memory_gb": 12.0,
    },
    "rwkv": {
        "tasks": [
            TaskType.TEXT_GENERATION,
            TaskType.SEQUENCE_MODELING,
            TaskType.LONG_CONTEXT,
        ],
        "strengths": ["RNN efficiency", "unlimited context", "low memory"],
        "context_length": 1000000,
        "memory_gb": 4.0,
    },
    "griffin": {
        "tasks": [
            TaskType.TEXT_GENERATION,
            TaskType.SEQUENCE_MODELING,
            TaskType.LONG_CONTEXT,
        ],
        "strengths": ["hybrid attention", "linear recurrence", "local patterns"],
        "context_length": 8192,
        "memory_gb": 10.0,
    },
    "ttt": {
        "tasks": [
            TaskType.TEXT_GENERATION,
            TaskType.REASONING,
            TaskType.MEMORY_INTENSIVE,
        ],
        "strengths": ["test-time adaptation", "learnable states", "context compression"],
        "context_length": 32000,
        "memory_gb": 16.0,
    },
    "hyena": {
        "tasks": [
            TaskType.SEQUENCE_MODELING,
            TaskType.LONG_CONTEXT,
            TaskType.TEXT_GENERATION,
        ],
        "strengths": ["implicit convolutions", "subquadratic", "very long sequences"],
        "context_length": 65536,
        "memory_gb": 8.0,
    },
    "consistency_models": {
        "tasks": [
            TaskType.IMAGE_GENERATION,
            TaskType.VIDEO_GENERATION,
        ],
        "strengths": ["one-step generation", "fast inference", "high quality"],
        "context_length": 0,
        "memory_gb": 12.0,
    },
    "ring_attention": {
        "tasks": [
            TaskType.LONG_CONTEXT,
            TaskType.TEXT_GENERATION,
            TaskType.REASONING,
        ],
        "strengths": ["infinite context", "distributed", "memory efficient"],
        "context_length": 10000000,  # Millions of tokens
        "memory_gb": 80.0,  # Distributed across GPUs
    },
    "flow_matching": {
        "tasks": [
            TaskType.IMAGE_GENERATION,
            TaskType.VIDEO_GENERATION,
        ],
        "strengths": ["optimal transport", "fast sampling", "simulation-free"],
        "context_length": 0,
        "memory_gb": 10.0,
    },
    "titans": {
        "tasks": [
            TaskType.LONG_CONTEXT,
            TaskType.MEMORY_INTENSIVE,
            TaskType.REASONING,
        ],
        "strengths": ["meta-learning", "test-time memory", "in-context learning"],
        "context_length": 2000000,
        "memory_gb": 24.0,
    },
}


class Orchestrator:
    """
    Central orchestrator that routes tasks to appropriate architectures.

    The orchestrator can:
    1. Automatically select the best architecture for a task
    2. Load/unload models dynamically to manage memory
    3. Run tasks with fallback options
    4. Track execution statistics
    """

    def __init__(
        self,
        device: str = "cuda",
        max_loaded_models: int = 2,
        auto_unload: bool = True,
    ):
        self.device = device
        self.max_loaded_models = max_loaded_models
        self.auto_unload = auto_unload

        self._loaded_architectures: Dict[str, ArchitectureBase] = {}
        self._architecture_registry: Dict[str, Type[ArchitectureBase]] = {}
        self._load_order: List[str] = []  # For LRU eviction

    def register_architecture(
        self,
        arch_id: str,
        arch_class: Type[ArchitectureBase],
    ) -> None:
        """Register an architecture implementation."""
        self._architecture_registry[arch_id] = arch_class

    def discover_architectures(self) -> List[str]:
        """
        Discover available architecture implementations.

        Looks for ArchitectureBase subclasses in each architecture's src/ directory.
        """
        discovered = []

        for arch_id in ARCHITECTURE_CAPABILITIES.keys():
            try:
                # Try to import the architecture module
                module = importlib.import_module(
                    f"consciousness.ml_research.modern_dev.{arch_id}.src.model"
                )

                # Look for ArchitectureBase subclass
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, ArchitectureBase)
                        and obj is not ArchitectureBase
                    ):
                        self.register_architecture(arch_id, obj)
                        discovered.append(arch_id)
                        break

            except ImportError:
                # Architecture not yet implemented
                pass

        return discovered

    def select_architecture(self, task_spec: TaskSpec) -> Optional[str]:
        """
        Select the best architecture for a given task.

        Returns architecture ID or None if no suitable architecture found.
        """
        # Check for explicit preference
        preferred = task_spec.constraints.get("preferred_architecture")
        if preferred:
            # If preferred is registered (implemented), use it directly
            if preferred in self._architecture_registry:
                return preferred
            # If preferred is in capabilities, validate task support
            if preferred in ARCHITECTURE_CAPABILITIES:
                caps = ARCHITECTURE_CAPABILITIES[preferred]
                if task_spec.task_type in [TaskType(t) if isinstance(t, str) else t for t in caps["tasks"]]:
                    return preferred

        # Score all architectures
        scores: List[tuple] = []

        for arch_id, caps in ARCHITECTURE_CAPABILITIES.items():
            task_types = caps["tasks"]

            # Check if architecture supports this task type
            if task_spec.task_type not in task_types:
                continue

            # Check context length requirement
            required_context = task_spec.constraints.get("context_length", 0)
            if required_context > caps["context_length"] > 0:
                continue

            # Check memory constraint
            max_memory = task_spec.constraints.get("max_memory_gb", float("inf"))
            if caps["memory_gb"] > max_memory:
                continue

            # Calculate score
            score = 0.5  # Base score

            # Bonus for context capability
            if caps["context_length"] > 0:
                context_headroom = caps["context_length"] / max(required_context, 1)
                score += min(0.2, 0.1 * context_headroom)

            # Bonus for memory efficiency
            if max_memory < float("inf"):
                memory_efficiency = 1 - (caps["memory_gb"] / max_memory)
                score += 0.1 * memory_efficiency

            # Check if architecture is already loaded (faster to use)
            if arch_id in self._loaded_architectures:
                score += 0.15

            scores.append((arch_id, score))

        if not scores:
            return None

        # Return highest scoring architecture
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def load_architecture(self, arch_id: str, checkpoint: Optional[str] = None) -> bool:
        """Load an architecture into memory."""
        if arch_id in self._loaded_architectures:
            return True  # Already loaded

        if arch_id not in self._architecture_registry:
            return False  # Not registered

        # Evict if necessary
        if self.auto_unload and len(self._loaded_architectures) >= self.max_loaded_models:
            self._evict_lru()

        # Instantiate and load
        arch_class = self._architecture_registry[arch_id]
        arch_instance = arch_class()
        arch_instance.load(checkpoint_path=checkpoint, device=self.device)

        self._loaded_architectures[arch_id] = arch_instance
        self._load_order.append(arch_id)

        return True

    def unload_architecture(self, arch_id: str) -> bool:
        """Unload an architecture from memory."""
        if arch_id not in self._loaded_architectures:
            return False

        self._loaded_architectures[arch_id].unload()
        del self._loaded_architectures[arch_id]

        if arch_id in self._load_order:
            self._load_order.remove(arch_id)

        return True

    def _evict_lru(self) -> None:
        """Evict least recently used architecture."""
        if self._load_order:
            lru_arch = self._load_order[0]
            self.unload_architecture(lru_arch)

    def run(
        self,
        task_type: Union[TaskType, str],
        input_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        architecture: Optional[str] = None,
    ) -> TaskResult:
        """
        Execute a task using the appropriate architecture.

        Args:
            task_type: Type of task to execute
            input_data: Input data for the task
            constraints: Optional constraints (memory, latency, etc.)
            architecture: Optional specific architecture to use

        Returns:
            TaskResult with output and metadata
        """
        import time

        # Convert string to TaskType if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        # Create task spec
        task_spec = TaskSpec(
            task_type=task_type,
            input_data=input_data,
            constraints=constraints or {},
        )

        # Add preferred architecture to constraints if specified
        if architecture:
            task_spec.constraints["preferred_architecture"] = architecture

        # Select architecture
        arch_id = self.select_architecture(task_spec)

        if arch_id is None:
            return TaskResult(
                success=False,
                output=None,
                architecture_used="none",
                execution_time_ms=0,
                memory_used_mb=0,
                error="No suitable architecture found for task",
            )

        # Check if architecture is implemented
        if arch_id not in self._architecture_registry:
            return TaskResult(
                success=False,
                output=None,
                architecture_used=arch_id,
                execution_time_ms=0,
                memory_used_mb=0,
                error=f"Architecture '{arch_id}' is indexed but not yet implemented. "
                      f"Implement in: modern_dev/{arch_id}/src/model.py",
            )

        # Load architecture if needed
        if arch_id not in self._loaded_architectures:
            if not self.load_architecture(arch_id):
                return TaskResult(
                    success=False,
                    output=None,
                    architecture_used=arch_id,
                    execution_time_ms=0,
                    memory_used_mb=0,
                    error=f"Failed to load architecture '{arch_id}'",
                )

        # Update LRU order
        if arch_id in self._load_order:
            self._load_order.remove(arch_id)
        self._load_order.append(arch_id)

        # Execute task
        start_time = time.time()

        try:
            result = self._loaded_architectures[arch_id].run(task_spec)
            result.architecture_used = arch_id
            return result

        except Exception as e:
            return TaskResult(
                success=False,
                output=None,
                architecture_used=arch_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_used_mb=0,
                error=str(e),
            )

    def list_available(self, task_type: Optional[TaskType] = None) -> Dict[str, Dict]:
        """List available architectures, optionally filtered by task type."""
        result = {}

        for arch_id, caps in ARCHITECTURE_CAPABILITIES.items():
            if task_type and task_type not in caps["tasks"]:
                continue

            result[arch_id] = {
                **caps,
                "implemented": arch_id in self._architecture_registry,
                "loaded": arch_id in self._loaded_architectures,
            }

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "device": self.device,
            "max_loaded_models": self.max_loaded_models,
            "loaded_architectures": list(self._loaded_architectures.keys()),
            "registered_architectures": list(self._architecture_registry.keys()),
            "available_architectures": list(ARCHITECTURE_CAPABILITIES.keys()),
        }


# Convenience function for quick task execution
def run_task(
    task_type: Union[TaskType, str],
    input_data: Dict[str, Any],
    **kwargs,
) -> TaskResult:
    """
    Quick function to run a task without manually managing the orchestrator.

    Usage:
        result = run_task("text_generation", {"prompt": "Hello, world!"})
    """
    orch = Orchestrator()
    orch.discover_architectures()
    return orch.run(task_type, input_data, **kwargs)
