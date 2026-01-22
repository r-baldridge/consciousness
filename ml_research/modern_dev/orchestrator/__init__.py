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

    # With actual model execution:
    orch = Orchestrator()
    result = orch.run(
        task_type="text_generation",
        input_data={"prompt": "Hello"},
        architecture="mamba"
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable
import importlib
import logging
import time

logger = logging.getLogger(__name__)


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


# Architecture module path registry - maps architecture IDs to their module paths
ARCHITECTURE_MODULE_PATHS = {
    "mamba": "consciousness.ml_research.modern_dev.mamba_impl.src.model",
    "ctm": "consciousness.ml_research.modern_dev.ctm.src.model",
    "jepa": "consciousness.ml_research.modern_dev.jepa.src.model",
    "rwkv": "consciousness.ml_research.modern_dev.rwkv.src.model",
    "xlstm": "consciousness.ml_research.modern_dev.xlstm.src.model",
    "griffin": "consciousness.ml_research.modern_dev.griffin.src.model",
    "hyena": "consciousness.ml_research.modern_dev.hyena.src.model",
    "ttt": "consciousness.ml_research.modern_dev.ttt.src.model",
    "ring_attention": "consciousness.ml_research.modern_dev.ring_attention.src.model",
    "flow_matching": "consciousness.ml_research.modern_dev.flow_matching.src.model",
    "consistency_models": "consciousness.ml_research.modern_dev.consistency_models.src.model",
    "titans": "consciousness.ml_research.modern_dev.titans.src.model",
}

# Architecture class name registry - maps architecture IDs to their main model class names
ARCHITECTURE_CLASS_NAMES = {
    "mamba": "Mamba",
    "ctm": "CTM",
    "jepa": "JEPA",
    "rwkv": "RWKV",
    "xlstm": "xLSTM",
    "griffin": "Griffin",
    "hyena": "Hyena",
    "ttt": "TTT",
    "ring_attention": "RingAttention",
    "flow_matching": "FlowMatching",
    "consistency_models": "ConsistencyModel",
    "titans": "Titans",
}

# Config class name registry - maps architecture IDs to their config class names
ARCHITECTURE_CONFIG_NAMES = {
    "mamba": "MambaConfig",
    "ctm": "CTMConfig",
    "jepa": "JEPAConfig",
    "rwkv": "RWKVConfig",
    "xlstm": "xLSTMConfig",
    "griffin": "GriffinConfig",
    "hyena": "HyenaConfig",
    "ttt": "TTTConfig",
    "ring_attention": "RingAttentionConfig",
    "flow_matching": "FlowMatchingConfig",
    "consistency_models": "ConsistencyModelConfig",
    "titans": "TitansConfig",
}


@dataclass
class ModelPreset:
    """Preset configuration for model sizes."""
    name: str
    description: str
    config_overrides: Dict[str, Any]


# Common presets for different model sizes
MODEL_PRESETS = {
    "tiny": ModelPreset(
        name="tiny",
        description="Very small model for quick testing",
        config_overrides={
            # Mamba/RWKV/xLSTM style
            "d_model": 128,
            "n_layer": 2,
            "num_layers": 2,
            "vocab_size": 1000,
            # CTM style
            "hidden_dim": 128,
            "num_neurons": 256,
            "max_internal_steps": 4,
            # JEPA style
            "embed_dim": 128,
            "encoder_depth": 2,
            "predictor_depth": 2,
            "image_size": 64,
            "patch_size": 8,
            # General
            "embedding_dim": 128,
        }
    ),
    "small": ModelPreset(
        name="small",
        description="Small model suitable for development",
        config_overrides={
            # Mamba/RWKV/xLSTM style
            "d_model": 256,
            "n_layer": 6,
            "num_layers": 6,
            "vocab_size": 8000,
            # CTM style
            "hidden_dim": 256,
            "num_neurons": 512,
            "max_internal_steps": 8,
            # JEPA style
            "embed_dim": 256,
            "encoder_depth": 6,
            "predictor_depth": 6,
            "image_size": 112,
            "patch_size": 16,
            # General
            "embedding_dim": 256,
        }
    ),
    "medium": ModelPreset(
        name="medium",
        description="Medium model for balanced performance",
        config_overrides={
            # Mamba/RWKV/xLSTM style
            "d_model": 512,
            "n_layer": 12,
            "num_layers": 12,
            "vocab_size": 32000,
            # CTM style
            "hidden_dim": 512,
            "num_neurons": 1024,
            "max_internal_steps": 16,
            # JEPA style
            "embed_dim": 512,
            "encoder_depth": 12,
            "predictor_depth": 8,
            "image_size": 224,
            "patch_size": 16,
            # General
            "embedding_dim": 512,
        }
    ),
    "large": ModelPreset(
        name="large",
        description="Large model for production use",
        config_overrides={
            # Mamba/RWKV/xLSTM style
            "d_model": 768,
            "n_layer": 24,
            "num_layers": 24,
            "vocab_size": 50280,
            # CTM style
            "hidden_dim": 768,
            "num_neurons": 2048,
            "max_internal_steps": 32,
            # JEPA style
            "embed_dim": 768,
            "encoder_depth": 24,
            "predictor_depth": 12,
            "image_size": 224,
            "patch_size": 14,
            # General
            "embedding_dim": 768,
        }
    ),
}


class ArchitectureLoader:
    """
    Dynamic loader for architecture implementations.

    Handles importing, instantiating, and caching of architecture models.
    """

    def __init__(self):
        self._module_cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        self._config_cache: Dict[str, Any] = {}
        self._custom_registrations: Dict[str, Dict[str, str]] = {}

    def register_architecture(
        self,
        arch_id: str,
        module_path: str,
        class_name: str,
        config_class_name: Optional[str] = None,
    ) -> None:
        """
        Register a custom architecture module path.

        Args:
            arch_id: Unique identifier for the architecture
            module_path: Full Python module path (e.g., 'mypackage.models.mymodel')
            class_name: Name of the model class in the module
            config_class_name: Name of the config class (optional)
        """
        self._custom_registrations[arch_id] = {
            "module_path": module_path,
            "class_name": class_name,
            "config_class_name": config_class_name,
        }
        logger.info(f"Registered custom architecture: {arch_id} -> {module_path}.{class_name}")

    def get_module_path(self, arch_id: str) -> Optional[str]:
        """Get the module path for an architecture."""
        if arch_id in self._custom_registrations:
            return self._custom_registrations[arch_id]["module_path"]
        return ARCHITECTURE_MODULE_PATHS.get(arch_id)

    def get_class_name(self, arch_id: str) -> Optional[str]:
        """Get the model class name for an architecture."""
        if arch_id in self._custom_registrations:
            return self._custom_registrations[arch_id]["class_name"]
        return ARCHITECTURE_CLASS_NAMES.get(arch_id)

    def get_config_class_name(self, arch_id: str) -> Optional[str]:
        """Get the config class name for an architecture."""
        if arch_id in self._custom_registrations:
            return self._custom_registrations[arch_id].get("config_class_name")
        return ARCHITECTURE_CONFIG_NAMES.get(arch_id)

    def load_module(self, arch_id: str) -> Optional[Any]:
        """
        Load and cache the module for an architecture.

        Args:
            arch_id: Architecture identifier

        Returns:
            The loaded module or None if not found
        """
        if arch_id in self._module_cache:
            return self._module_cache[arch_id]

        module_path = self.get_module_path(arch_id)
        if module_path is None:
            logger.warning(f"No module path registered for architecture: {arch_id}")
            return None

        try:
            module = importlib.import_module(module_path)
            self._module_cache[arch_id] = module
            logger.info(f"Loaded module for {arch_id}: {module_path}")
            return module
        except ImportError as e:
            logger.error(f"Failed to import module for {arch_id}: {e}")
            return None

    def get_model_class(self, arch_id: str) -> Optional[Type]:
        """
        Get the model class for an architecture.

        Args:
            arch_id: Architecture identifier

        Returns:
            The model class or None if not found
        """
        module = self.load_module(arch_id)
        if module is None:
            return None

        class_name = self.get_class_name(arch_id)
        if class_name is None:
            logger.warning(f"No class name registered for architecture: {arch_id}")
            return None

        if not hasattr(module, class_name):
            logger.error(f"Class {class_name} not found in module for {arch_id}")
            return None

        return getattr(module, class_name)

    def get_config_class(self, arch_id: str) -> Optional[Type]:
        """
        Get the config class for an architecture.

        Args:
            arch_id: Architecture identifier

        Returns:
            The config class or None if not found
        """
        module = self.load_module(arch_id)
        if module is None:
            return None

        config_class_name = self.get_config_class_name(arch_id)
        if config_class_name is None:
            return None

        if not hasattr(module, config_class_name):
            logger.warning(f"Config class {config_class_name} not found in module for {arch_id}")
            return None

        return getattr(module, config_class_name)

    def create_config(
        self,
        arch_id: str,
        preset: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Create a configuration object for an architecture.

        Args:
            arch_id: Architecture identifier
            preset: Optional preset name ('tiny', 'small', 'medium', 'large')
            **kwargs: Additional config overrides

        Returns:
            Config instance or None if not found
        """
        config_class = self.get_config_class(arch_id)
        if config_class is None:
            logger.warning(f"No config class available for {arch_id}, using defaults")
            return None

        # Start with preset overrides if specified
        config_kwargs = {}
        if preset and preset in MODEL_PRESETS:
            preset_overrides = MODEL_PRESETS[preset].config_overrides
            # Only use relevant keys for this config class
            for key, value in preset_overrides.items():
                # Try to see if this key is valid for the config
                config_kwargs[key] = value

        # Apply user overrides
        config_kwargs.update(kwargs)

        # Filter to only valid keys for this config class
        import inspect
        if hasattr(config_class, "__dataclass_fields__"):
            valid_keys = set(config_class.__dataclass_fields__.keys())
        else:
            sig = inspect.signature(config_class.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self"}

        filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_keys}

        try:
            return config_class(**filtered_kwargs)
        except Exception as e:
            logger.error(f"Failed to create config for {arch_id}: {e}")
            # Try with no kwargs
            try:
                return config_class()
            except Exception:
                return None

    def instantiate_model(
        self,
        arch_id: str,
        config: Optional[Any] = None,
        preset: Optional[str] = None,
        cache: bool = True,
        **config_kwargs,
    ) -> Optional[Any]:
        """
        Instantiate a model for an architecture.

        Args:
            arch_id: Architecture identifier
            config: Pre-built config object (optional)
            preset: Preset name for config ('tiny', 'small', 'medium', 'large')
            cache: Whether to cache the model instance
            **config_kwargs: Additional config parameters

        Returns:
            Model instance or None if instantiation fails
        """
        # Check cache first
        cache_key = f"{arch_id}_{preset or 'default'}"
        if cache and cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_class = self.get_model_class(arch_id)
        if model_class is None:
            return None

        # Create config if not provided
        if config is None:
            config = self.create_config(arch_id, preset=preset, **config_kwargs)

        try:
            if config is not None:
                model = model_class(config)
            else:
                # Try without config for models that don't require it
                model = model_class()

            if cache:
                self._model_cache[cache_key] = model

            logger.info(f"Instantiated model for {arch_id} (preset={preset})")
            return model

        except Exception as e:
            logger.error(f"Failed to instantiate model for {arch_id}: {e}")
            return None

    def clear_cache(self, arch_id: Optional[str] = None) -> None:
        """
        Clear cached modules and models.

        Args:
            arch_id: Specific architecture to clear, or None for all
        """
        if arch_id:
            self._module_cache.pop(arch_id, None)
            # Clear all model cache entries for this arch
            keys_to_remove = [k for k in self._model_cache if k.startswith(arch_id)]
            for k in keys_to_remove:
                del self._model_cache[k]
        else:
            self._module_cache.clear()
            self._model_cache.clear()

        logger.info(f"Cleared cache for: {arch_id or 'all architectures'}")

    def list_available(self) -> List[str]:
        """List all registered architecture IDs."""
        all_archs = set(ARCHITECTURE_MODULE_PATHS.keys())
        all_archs.update(self._custom_registrations.keys())
        return sorted(all_archs)


class Orchestrator:
    """
    Central orchestrator that routes tasks to appropriate architectures.

    The orchestrator can:
    1. Automatically select the best architecture for a task
    2. Load/unload models dynamically to manage memory
    3. Run tasks with fallback options
    4. Track execution statistics
    5. Dynamically load and instantiate real model implementations

    Usage:
        # Basic usage with automatic architecture selection
        orch = Orchestrator()
        result = orch.run(
            task_type="text_generation",
            input_data={"prompt": "Hello, world!"}
        )

        # Specify architecture directly
        result = orch.run(
            task_type="text_generation",
            input_data={"prompt": "Hello"},
            architecture="mamba",
            preset="small"  # Use small model preset
        )

        # Register custom architecture
        orch.register_architecture(
            "my_arch",
            "mypackage.models.mymodel",
            "MyModelClass",
            "MyModelConfig"
        )
    """

    def __init__(
        self,
        device: str = "cuda",
        max_loaded_models: int = 2,
        auto_unload: bool = True,
        default_preset: Optional[str] = "small",
    ):
        self.device = device
        self.max_loaded_models = max_loaded_models
        self.auto_unload = auto_unload
        self.default_preset = default_preset

        self._loaded_architectures: Dict[str, ArchitectureBase] = {}
        self._architecture_registry: Dict[str, Type[ArchitectureBase]] = {}
        self._load_order: List[str] = []  # For LRU eviction

        # Architecture loader for dynamic model loading
        self._loader = ArchitectureLoader()

        # Cache for instantiated models (separate from ArchitectureBase wrappers)
        self._model_instances: Dict[str, Any] = {}
        self._model_configs: Dict[str, Any] = {}

    def register_architecture(
        self,
        arch_id: str,
        module_path_or_class: Union[str, Type[ArchitectureBase]],
        class_name: Optional[str] = None,
        config_class_name: Optional[str] = None,
    ) -> None:
        """
        Register an architecture implementation.

        Can be called in two ways:
        1. With module path (dynamic loading):
           register_architecture("mamba", "mypackage.models", "MambaModel", "MambaConfig")

        2. With ArchitectureBase class (direct registration):
           register_architecture("mamba", MambaArchitectureClass)

        Args:
            arch_id: Unique identifier for the architecture
            module_path_or_class: Either a module path string or an ArchitectureBase class
            class_name: Model class name (required if module_path is a string)
            config_class_name: Config class name (optional)
        """
        if isinstance(module_path_or_class, str):
            # Dynamic module registration
            if class_name is None:
                raise ValueError("class_name is required when registering with module path")
            self._loader.register_architecture(
                arch_id, module_path_or_class, class_name, config_class_name
            )
        else:
            # Direct class registration (legacy ArchitectureBase style)
            self._architecture_registry[arch_id] = module_path_or_class

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
        preset: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Execute a task using the appropriate architecture.

        This method will:
        1. Select the best architecture (or use the specified one)
        2. Dynamically load and instantiate the model
        3. Execute the task using the model's forward/generate methods
        4. Return real results from the model

        Args:
            task_type: Type of task to execute
            input_data: Input data for the task
            constraints: Optional constraints (memory, latency, etc.)
            architecture: Optional specific architecture to use
            preset: Model size preset ('tiny', 'small', 'medium', 'large')
            config_overrides: Additional config parameters to override

        Returns:
            TaskResult with output and metadata
        """
        start_time = time.time()

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

        # Use default preset if none specified
        effective_preset = preset or self.default_preset

        # Try to load and run using dynamic loader first
        result = self._run_with_dynamic_model(
            arch_id=arch_id,
            task_spec=task_spec,
            preset=effective_preset,
            config_overrides=config_overrides,
            start_time=start_time,
        )

        if result is not None:
            return result

        # Fall back to ArchitectureBase registry if dynamic loading failed
        if arch_id in self._architecture_registry:
            return self._run_with_architecture_base(
                arch_id=arch_id,
                task_spec=task_spec,
                start_time=start_time,
            )

        # Neither method worked
        return TaskResult(
            success=False,
            output=None,
            architecture_used=arch_id,
            execution_time_ms=(time.time() - start_time) * 1000,
            memory_used_mb=0,
            error=f"Architecture '{arch_id}' could not be loaded. "
                  f"Check implementation at: modern_dev/{arch_id}/src/model.py",
        )

    def _run_with_dynamic_model(
        self,
        arch_id: str,
        task_spec: TaskSpec,
        preset: Optional[str],
        config_overrides: Optional[Dict[str, Any]],
        start_time: float,
    ) -> Optional[TaskResult]:
        """
        Run a task using dynamically loaded model.

        Returns TaskResult on success, None if dynamic loading not available.
        """
        try:
            # Get or create model instance
            cache_key = f"{arch_id}_{preset or 'default'}"

            if cache_key not in self._model_instances:
                model = self._loader.instantiate_model(
                    arch_id=arch_id,
                    preset=preset,
                    cache=True,
                    **(config_overrides or {}),
                )
                if model is None:
                    return None

                # Move to device if possible
                if hasattr(model, "to"):
                    try:
                        model = model.to(self.device)
                    except Exception as e:
                        logger.warning(f"Could not move model to {self.device}: {e}")
                        # Try CPU
                        try:
                            model = model.to("cpu")
                        except Exception:
                            pass

                # Set to eval mode
                if hasattr(model, "eval"):
                    model.eval()

                self._model_instances[cache_key] = model

                # Manage cache size
                self._manage_model_cache()

            model = self._model_instances[cache_key]

            # Execute based on task type
            output = self._execute_model_task(model, arch_id, task_spec)

            execution_time = (time.time() - start_time) * 1000

            # Estimate memory usage
            memory_mb = self._estimate_memory_usage(model)

            return TaskResult(
                success=True,
                output=output,
                architecture_used=arch_id,
                execution_time_ms=execution_time,
                memory_used_mb=memory_mb,
                metadata={
                    "preset": preset,
                    "device": str(self.device),
                    "model_type": type(model).__name__,
                },
            )

        except Exception as e:
            logger.error(f"Error running dynamic model {arch_id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                architecture_used=arch_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_used_mb=0,
                error=str(e),
            )

    def _execute_model_task(
        self,
        model: Any,
        arch_id: str,
        task_spec: TaskSpec,
    ) -> Any:
        """
        Execute a task on the model based on task type.

        Dispatches to appropriate model method based on task type and
        architecture capabilities.
        """
        import torch

        task_type = task_spec.task_type
        input_data = task_spec.input_data

        # Text generation tasks
        if task_type in [TaskType.TEXT_GENERATION, TaskType.LONG_CONTEXT, TaskType.SEQUENCE_MODELING]:
            return self._execute_text_generation(model, arch_id, input_data)

        # Reasoning tasks (similar to text generation for most models)
        elif task_type in [TaskType.REASONING, TaskType.PLANNING]:
            return self._execute_reasoning(model, arch_id, input_data)

        # Vision tasks
        elif task_type in [TaskType.IMAGE_CLASSIFICATION, TaskType.VISION_LANGUAGE]:
            return self._execute_vision_task(model, arch_id, input_data, task_type)

        # Video tasks
        elif task_type in [TaskType.VIDEO_UNDERSTANDING, TaskType.VIDEO_GENERATION]:
            return self._execute_video_task(model, arch_id, input_data, task_type)

        # Image generation tasks
        elif task_type in [TaskType.IMAGE_GENERATION, TaskType.TEXT_TO_IMAGE]:
            return self._execute_image_generation(model, arch_id, input_data)

        # World modeling (JEPA-style)
        elif task_type == TaskType.WORLD_MODELING:
            return self._execute_world_modeling(model, arch_id, input_data)

        # Continuous dynamics (CTM-style)
        elif task_type == TaskType.CONTINUOUS_DYNAMICS:
            return self._execute_continuous_dynamics(model, arch_id, input_data)

        # Memory-intensive tasks
        elif task_type == TaskType.MEMORY_INTENSIVE:
            return self._execute_memory_task(model, arch_id, input_data)

        # Default: try forward pass
        else:
            return self._execute_generic_forward(model, arch_id, input_data)

    def _execute_text_generation(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute text generation task."""
        import torch

        prompt = input_data.get("prompt", "")
        max_new_tokens = input_data.get("max_new_tokens", input_data.get("max_length", 50))
        temperature = input_data.get("temperature", 1.0)
        top_k = input_data.get("top_k", None)
        top_p = input_data.get("top_p", None)

        # For models like Mamba, RWKV, xLSTM that expect input_ids
        if hasattr(model, "generate"):
            # Create dummy input_ids from prompt (simple encoding)
            # In practice, you'd use a proper tokenizer
            input_ids = self._simple_tokenize(prompt, model)

            with torch.no_grad():
                if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
                    # Models with generate method
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

            # Simple decode
            generated_text = self._simple_decode(output_ids, input_ids.shape[1])

            return {
                "generated_text": generated_text,
                "input_ids": input_ids.tolist(),
                "output_ids": output_ids.tolist(),
                "num_generated_tokens": output_ids.shape[1] - input_ids.shape[1],
            }

        # For models with only forward method
        elif hasattr(model, "forward"):
            input_ids = self._simple_tokenize(prompt, model)

            with torch.no_grad():
                output = model(input_ids)

            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, dict):
                logits = output.get("logits", output.get("output", output))
            else:
                logits = output[0] if isinstance(output, tuple) else output

            return {
                "logits_shape": list(logits.shape) if hasattr(logits, "shape") else None,
                "output_type": type(output).__name__,
                "note": "Raw forward pass output (no autoregressive generation)",
            }

        else:
            return {"error": f"Model {arch_id} does not support text generation"}

    def _execute_reasoning(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute reasoning task (CTM-style or general)."""
        import torch

        # CTM has a specific interface
        if arch_id == "ctm" and hasattr(model, "forward"):
            # CTM expects input tensor, not tokens
            input_tensor = input_data.get("input_tensor")
            if input_tensor is None:
                # Create from prompt or default
                prompt = input_data.get("prompt", "")
                batch_size = input_data.get("batch_size", 1)
                input_dim = getattr(model.config, "input_dim", 768)
                input_tensor = torch.randn(batch_size, input_dim)

            if isinstance(input_tensor, list):
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)

            num_steps = input_data.get("num_steps", None)
            return_intermediates = input_data.get("return_intermediates", False)

            with torch.no_grad():
                result = model(
                    input_tensor,
                    num_steps=num_steps,
                    return_intermediates=return_intermediates,
                )

            # Convert tensors to lists for JSON serialization
            output = {
                "output": result["output"].tolist(),
                "num_steps_used": result.get("num_steps_used", 0),
            }

            if "activations" in result:
                output["final_activation_norm"] = result["activations"].norm().item()

            return output

        # For other models, fall back to text generation style
        return self._execute_text_generation(model, arch_id, input_data)

    def _execute_vision_task(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
        task_type: TaskType,
    ) -> Dict[str, Any]:
        """Execute vision task (JEPA-style)."""
        import torch

        # JEPA expects image tensors
        images = input_data.get("images", input_data.get("image"))

        if images is None:
            # Create dummy image tensor for testing
            batch_size = input_data.get("batch_size", 1)
            if hasattr(model, "config"):
                img_size = getattr(model.config, "image_size", 224)
                channels = getattr(model.config, "in_channels", 3)
            else:
                img_size = 224
                channels = 3

            images = torch.randn(batch_size, channels, img_size, img_size)

        if isinstance(images, list):
            images = torch.tensor(images, dtype=torch.float32)

        device = next(model.parameters()).device
        images = images.to(device)

        with torch.no_grad():
            if hasattr(model, "encode"):
                # JEPA inference mode
                embeddings = model.encode(images)
                return {
                    "embeddings_shape": list(embeddings.shape),
                    "embedding_norm": embeddings.norm(dim=-1).mean().item(),
                }
            elif hasattr(model, "forward"):
                # Training mode (returns loss, etc.)
                result = model(images, return_embeddings=True)
                if isinstance(result, dict):
                    return {
                        k: v.tolist() if isinstance(v, torch.Tensor) else v
                        for k, v in result.items()
                        if k != "intermediates"  # Skip large intermediate data
                    }
                return {"output": result}

        return {"error": f"Model {arch_id} does not support vision tasks"}

    def _execute_video_task(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
        task_type: TaskType,
    ) -> Dict[str, Any]:
        """Execute video understanding/generation task."""
        # Similar to vision but with temporal dimension
        return {"note": f"Video task on {arch_id}", "task_type": task_type.value}

    def _execute_image_generation(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute image generation task (consistency models, flow matching)."""
        import torch

        prompt = input_data.get("prompt", "")
        num_steps = input_data.get("num_steps", 1)
        batch_size = input_data.get("batch_size", 1)

        # These models typically have a sample() method
        if hasattr(model, "sample"):
            with torch.no_grad():
                samples = model.sample(batch_size=batch_size, num_steps=num_steps)
            return {
                "samples_shape": list(samples.shape),
                "samples_min": samples.min().item(),
                "samples_max": samples.max().item(),
            }

        return {"error": f"Model {arch_id} does not support image generation"}

    def _execute_world_modeling(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute world modeling task."""
        return self._execute_vision_task(model, arch_id, input_data, TaskType.WORLD_MODELING)

    def _execute_continuous_dynamics(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute continuous dynamics task (CTM)."""
        return self._execute_reasoning(model, arch_id, input_data)

    def _execute_memory_task(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute memory-intensive task (Titans, TTT)."""
        # Similar to text generation but may use different interfaces
        return self._execute_text_generation(model, arch_id, input_data)

    def _execute_generic_forward(
        self,
        model: Any,
        arch_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute generic forward pass."""
        import torch

        if hasattr(model, "forward"):
            # Try to determine input format
            if "input_ids" in input_data:
                input_tensor = torch.tensor(input_data["input_ids"])
            elif "input" in input_data:
                input_tensor = torch.tensor(input_data["input"])
            else:
                return {"error": "No recognizable input format"}

            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)

            if isinstance(output, torch.Tensor):
                return {"output_shape": list(output.shape)}
            elif isinstance(output, dict):
                return {k: str(type(v)) for k, v in output.items()}
            else:
                return {"output_type": str(type(output))}

        return {"error": f"Model {arch_id} has no forward method"}

    def _simple_tokenize(self, text: str, model: Any) -> "torch.Tensor":
        """Simple tokenization for testing (not for production)."""
        import torch

        # Get vocab size from model config
        vocab_size = 1000  # Default
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", 1000)

        # Simple character-level tokenization clamped to vocab size
        tokens = [min(ord(c) % vocab_size, vocab_size - 1) for c in text]
        if not tokens:
            tokens = [0]  # Empty input gets a padding token

        device = next(model.parameters()).device
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def _simple_decode(self, output_ids: "torch.Tensor", prompt_len: int) -> str:
        """Simple decoding for testing (not for production)."""
        # Get new tokens only
        new_tokens = output_ids[0, prompt_len:].tolist()
        # Simple decode: treat as character codes (clamped)
        chars = [chr(min(max(t, 32), 126)) for t in new_tokens]
        return "".join(chars)

    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            else:
                # Estimate from parameter count
                if hasattr(model, "parameters"):
                    param_count = sum(p.numel() for p in model.parameters())
                    return param_count * 4 / 1024 / 1024  # Assume float32
        except Exception:
            pass
        return 0.0

    def _manage_model_cache(self) -> None:
        """Manage model instance cache size."""
        if len(self._model_instances) > self.max_loaded_models:
            # Remove oldest entry
            oldest_key = next(iter(self._model_instances))
            del self._model_instances[oldest_key]
            logger.info(f"Evicted model from cache: {oldest_key}")

    def _run_with_architecture_base(
        self,
        arch_id: str,
        task_spec: TaskSpec,
        start_time: float,
    ) -> TaskResult:
        """Run a task using ArchitectureBase registry (legacy support)."""
        # Load architecture if needed
        if arch_id not in self._loaded_architectures:
            if not self.load_architecture(arch_id):
                return TaskResult(
                    success=False,
                    output=None,
                    architecture_used=arch_id,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    memory_used_mb=0,
                    error=f"Failed to load architecture '{arch_id}'",
                )

        # Update LRU order
        if arch_id in self._load_order:
            self._load_order.remove(arch_id)
        self._load_order.append(arch_id)

        # Execute task
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

            # Check if dynamically loadable
            is_loadable = self._loader.get_module_path(arch_id) is not None

            result[arch_id] = {
                **caps,
                "implemented": arch_id in self._architecture_registry or is_loadable,
                "loaded": arch_id in self._loaded_architectures or arch_id in self._model_instances,
                "module_path": self._loader.get_module_path(arch_id),
            }

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "device": self.device,
            "max_loaded_models": self.max_loaded_models,
            "default_preset": self.default_preset,
            "loaded_architectures": list(self._loaded_architectures.keys()),
            "registered_architectures": list(self._architecture_registry.keys()),
            "cached_models": list(self._model_instances.keys()),
            "available_architectures": list(ARCHITECTURE_CAPABILITIES.keys()),
            "loadable_architectures": self._loader.list_available(),
        }

    def get_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available model presets."""
        return {
            name: {
                "name": preset.name,
                "description": preset.description,
                "config_keys": list(preset.config_overrides.keys()),
            }
            for name, preset in MODEL_PRESETS.items()
        }

    def clear_model_cache(self, arch_id: Optional[str] = None) -> None:
        """
        Clear cached model instances.

        Args:
            arch_id: Specific architecture to clear, or None for all
        """
        if arch_id:
            keys_to_remove = [k for k in self._model_instances if k.startswith(arch_id)]
            for k in keys_to_remove:
                del self._model_instances[k]
        else:
            self._model_instances.clear()

        self._loader.clear_cache(arch_id)
        logger.info(f"Cleared model cache for: {arch_id or 'all'}")


# Convenience function for quick task execution
def run_task(
    task_type: Union[TaskType, str],
    input_data: Dict[str, Any],
    architecture: Optional[str] = None,
    preset: Optional[str] = "small",
    device: str = "cpu",
    **kwargs,
) -> TaskResult:
    """
    Quick function to run a task without manually managing the orchestrator.

    Usage:
        # Basic usage
        result = run_task("text_generation", {"prompt": "Hello, world!"})

        # With specific architecture
        result = run_task(
            "text_generation",
            {"prompt": "Hello"},
            architecture="mamba",
            preset="tiny"
        )

    Args:
        task_type: Type of task to execute
        input_data: Input data for the task
        architecture: Optional specific architecture to use
        preset: Model size preset ('tiny', 'small', 'medium', 'large')
        device: Device to run on ('cpu', 'cuda', 'mps')
        **kwargs: Additional arguments passed to orchestrator.run()

    Returns:
        TaskResult with output and metadata
    """
    orch = Orchestrator(device=device, default_preset=preset)
    orch.discover_architectures()
    return orch.run(
        task_type,
        input_data,
        architecture=architecture,
        preset=preset,
        **kwargs,
    )


# Additional convenience functions
def create_model(
    arch_id: str,
    preset: str = "small",
    device: str = "cpu",
    **config_overrides,
) -> Any:
    """
    Create a model instance directly without the orchestrator.

    Usage:
        model = create_model("mamba", preset="tiny")
        output = model(input_ids)

    Args:
        arch_id: Architecture identifier
        preset: Model size preset
        device: Device to load model on
        **config_overrides: Additional config parameters

    Returns:
        Model instance
    """
    loader = ArchitectureLoader()
    model = loader.instantiate_model(
        arch_id=arch_id,
        preset=preset,
        cache=False,
        **config_overrides,
    )

    if model is not None and hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass

    if model is not None and hasattr(model, "eval"):
        model.eval()

    return model


def list_architectures(task_type: Optional[str] = None) -> Dict[str, Dict]:
    """
    List available architectures.

    Args:
        task_type: Optional task type to filter by

    Returns:
        Dictionary of architecture information
    """
    orch = Orchestrator()

    if task_type:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        return orch.list_available(task_type)

    return orch.list_available()


# Export key classes and functions
__all__ = [
    # Core classes
    "Orchestrator",
    "ArchitectureLoader",
    "ArchitectureBase",
    # Data classes
    "TaskType",
    "TaskSpec",
    "TaskResult",
    "ModelPreset",
    # Registries
    "ARCHITECTURE_CAPABILITIES",
    "ARCHITECTURE_MODULE_PATHS",
    "ARCHITECTURE_CLASS_NAMES",
    "MODEL_PRESETS",
    # Convenience functions
    "run_task",
    "create_model",
    "list_architectures",
]
