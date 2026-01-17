"""
Model Registry - Manages model lifecycle: loading, unloading, versioning
Part of the Neural Network module for the Consciousness system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Possible states for a loaded model."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"
    WARMING_UP = "warming_up"


class Priority(Enum):
    """Priority levels for model loading and resource allocation."""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

    def __lt__(self, other: "Priority") -> bool:
        return self.value < other.value

    def __gt__(self, other: "Priority") -> bool:
        return self.value > other.value


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    form_id: str
    name: str
    model_type: str
    model_name: str
    quantization: str
    size_mb: int
    vram_mb: int
    max_batch_size: int
    timeout_ms: int
    priority: Priority
    preemptible: bool
    critical: bool = False
    always_loaded: bool = False
    fallback_model: Optional[str] = None
    update_frequency_hz: Optional[float] = None
    input_spec: Dict[str, Any] = field(default_factory=dict)
    output_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """Represents a model that is loaded in memory."""
    form_id: str
    config: ModelConfig
    state: ModelState
    model_instance: Optional[Any] = None
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    inference_count: int = 0
    total_inference_time_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    vram_allocated_mb: int = 0

    @property
    def avg_inference_time_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time_ms / self.inference_count

    def record_inference(self, duration_ms: float) -> None:
        """Record an inference execution."""
        self.inference_count += 1
        self.total_inference_time_ms += duration_ms
        self.last_used = datetime.utcnow()

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error


class ModelRegistry:
    """
    Manages model lifecycle: loading, unloading, versioning.

    The registry maintains a catalog of all available models and tracks
    which models are currently loaded in memory.
    """

    def __init__(self, config_path: str):
        """
        Initialize the model registry.

        Args:
            config_path: Path to the model_configs.yaml file
        """
        self.config_path = Path(config_path)
        self.configs: Dict[str, ModelConfig] = {}
        self.models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._load_callbacks: List[Callable] = []
        self._unload_callbacks: List[Callable] = []
        self._model_loader: Optional[Any] = None  # Set by NervousSystem

        self._load_config()

    def _load_config(self) -> None:
        """Load model configurations from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        forms = config_data.get('forms', {})
        for form_id, form_config in forms.items():
            priority_str = form_config.get('priority', 'normal').upper()
            priority = Priority[priority_str]

            self.configs[form_id] = ModelConfig(
                form_id=form_id,
                name=form_config.get('name', form_id),
                model_type=form_config.get('model_type', ''),
                model_name=form_config.get('model_name', ''),
                quantization=form_config.get('quantization', 'fp16'),
                size_mb=form_config.get('size_mb', 0),
                vram_mb=form_config.get('vram_mb', 0),
                max_batch_size=form_config.get('max_batch_size', 4),
                timeout_ms=form_config.get('timeout_ms', 100),
                priority=priority,
                preemptible=form_config.get('preemptible', True),
                critical=form_config.get('critical', False),
                always_loaded=form_config.get('always_loaded', False),
                fallback_model=form_config.get('fallback_model'),
                update_frequency_hz=form_config.get('update_frequency_hz'),
                input_spec=form_config.get('input_spec', {}),
                output_spec=form_config.get('output_spec', {}),
            )

        logger.info(f"Loaded {len(self.configs)} model configurations")

    def set_model_loader(self, loader: Any) -> None:
        """Set the model loader instance."""
        self._model_loader = loader

    def on_model_load(self, callback: Callable) -> None:
        """Register a callback for when models are loaded."""
        self._load_callbacks.append(callback)

    def on_model_unload(self, callback: Callable) -> None:
        """Register a callback for when models are unloaded."""
        self._unload_callbacks.append(callback)

    def get_config(self, form_id: str) -> Optional[ModelConfig]:
        """Get the configuration for a form."""
        return self.configs.get(form_id)

    def get_all_configs(self) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        return dict(self.configs)

    def get_critical_forms(self) -> List[str]:
        """Get list of critical form IDs that should always be loaded."""
        return [
            form_id for form_id, config in self.configs.items()
            if config.critical or config.always_loaded
        ]

    async def load_model(
        self,
        form_id: str,
        priority: Optional[Priority] = None,
        force: bool = False
    ) -> Optional[LoadedModel]:
        """
        Load a model with resource allocation.

        Args:
            form_id: The form ID to load
            priority: Override priority for loading
            force: Force load even if already loaded

        Returns:
            LoadedModel if successful, None otherwise
        """
        async with self._lock:
            # Check if already loaded
            if form_id in self.models and not force:
                existing = self.models[form_id]
                if existing.state == ModelState.LOADED:
                    logger.debug(f"Model {form_id} already loaded")
                    return existing

            # Get configuration
            config = self.configs.get(form_id)
            if not config:
                logger.error(f"No configuration found for form {form_id}")
                return None

            # Use provided priority or config priority
            load_priority = priority or config.priority

            # Create LoadedModel entry
            loaded_model = LoadedModel(
                form_id=form_id,
                config=config,
                state=ModelState.LOADING,
            )
            self.models[form_id] = loaded_model

        try:
            # Perform actual model loading
            if self._model_loader:
                model_instance = await self._model_loader.load(
                    form_id=form_id,
                    model_name=config.model_name,
                    model_type=config.model_type,
                    quantization=config.quantization,
                )
                loaded_model.model_instance = model_instance
                loaded_model.vram_allocated_mb = config.vram_mb

            loaded_model.state = ModelState.LOADED
            loaded_model.loaded_at = datetime.utcnow()
            loaded_model.last_used = datetime.utcnow()

            logger.info(f"Loaded model {form_id} ({config.name})")

            # Notify callbacks
            for callback in self._load_callbacks:
                try:
                    await callback(form_id, loaded_model)
                except Exception as e:
                    logger.error(f"Load callback error: {e}")

            return loaded_model

        except Exception as e:
            logger.error(f"Failed to load model {form_id}: {e}")
            loaded_model.state = ModelState.ERROR
            loaded_model.record_error(str(e))

            # Try fallback model if available
            if config.fallback_model:
                logger.info(f"Attempting fallback model for {form_id}")
                # Could implement fallback loading here

            return None

    async def unload_model(self, form_id: str, force: bool = False) -> bool:
        """
        Release model resources.

        Args:
            form_id: The form ID to unload
            force: Force unload even for critical models

        Returns:
            True if unloaded successfully, False otherwise
        """
        async with self._lock:
            if form_id not in self.models:
                logger.debug(f"Model {form_id} not loaded")
                return True

            loaded_model = self.models[form_id]

            # Check if model can be unloaded
            if not force and loaded_model.config.critical:
                logger.warning(f"Cannot unload critical model {form_id}")
                return False

            loaded_model.state = ModelState.UNLOADING

        try:
            # Perform actual model unloading
            if self._model_loader and loaded_model.model_instance:
                await self._model_loader.unload(
                    form_id=form_id,
                    model_instance=loaded_model.model_instance
                )

            # Notify callbacks before removal
            for callback in self._unload_callbacks:
                try:
                    await callback(form_id, loaded_model)
                except Exception as e:
                    logger.error(f"Unload callback error: {e}")

            async with self._lock:
                del self.models[form_id]

            logger.info(f"Unloaded model {form_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model {form_id}: {e}")
            loaded_model.state = ModelState.ERROR
            loaded_model.record_error(str(e))
            return False

    async def get_model(self, form_id: str) -> Optional[LoadedModel]:
        """
        Get a loaded model or None if not loaded.

        Args:
            form_id: The form ID to get

        Returns:
            LoadedModel if loaded, None otherwise
        """
        return self.models.get(form_id)

    async def is_loaded(self, form_id: str) -> bool:
        """Check if a model is loaded."""
        model = self.models.get(form_id)
        return model is not None and model.state == ModelState.LOADED

    async def preempt_for(
        self,
        form_id: str,
        required_memory_mb: int,
        resource_manager: Any
    ) -> bool:
        """
        Preempt lower-priority models to free memory for a higher-priority model.

        Args:
            form_id: The form ID that needs resources
            required_memory_mb: Amount of VRAM needed
            resource_manager: ResourceManager instance

        Returns:
            True if enough memory was freed, False otherwise
        """
        config = self.configs.get(form_id)
        if not config:
            return False

        source_priority = config.priority

        # Find preemptible models with lower priority
        preemptible = []
        for model_id, loaded in self.models.items():
            if loaded.config.preemptible and loaded.config.priority < source_priority:
                preemptible.append((model_id, loaded))

        # Sort by priority (lowest first) then by last used (oldest first)
        preemptible.sort(
            key=lambda x: (x[1].config.priority.value, x[1].last_used or datetime.min)
        )

        freed_memory = 0
        preempted = []

        for model_id, loaded in preemptible:
            if freed_memory >= required_memory_mb:
                break

            freed_memory += loaded.vram_allocated_mb
            preempted.append(model_id)

        if freed_memory < required_memory_mb:
            logger.warning(
                f"Cannot free enough memory for {form_id}: "
                f"need {required_memory_mb}MB, can free {freed_memory}MB"
            )
            return False

        # Actually preempt the models
        for model_id in preempted:
            logger.info(f"Preempting model {model_id} for {form_id}")
            await self.unload_model(model_id)

        return True

    @property
    def loaded_count(self) -> int:
        """Get count of loaded models."""
        return len([m for m in self.models.values() if m.state == ModelState.LOADED])

    @property
    def total_vram_used_mb(self) -> int:
        """Get total VRAM used by loaded models."""
        return sum(
            m.vram_allocated_mb
            for m in self.models.values()
            if m.state == ModelState.LOADED
        )

    def get_status(self) -> Dict[str, Any]:
        """Get registry status information."""
        loaded_forms = []
        for form_id, model in self.models.items():
            if model.state == ModelState.LOADED:
                loaded_forms.append({
                    'form_id': form_id,
                    'name': model.config.name,
                    'priority': model.config.priority.name,
                    'vram_mb': model.vram_allocated_mb,
                    'inference_count': model.inference_count,
                    'avg_inference_ms': round(model.avg_inference_time_ms, 2),
                    'loaded_at': model.loaded_at.isoformat() if model.loaded_at else None,
                    'last_used': model.last_used.isoformat() if model.last_used else None,
                })

        return {
            'total_forms': len(self.configs),
            'loaded_forms': self.loaded_count,
            'total_vram_used_mb': self.total_vram_used_mb,
            'forms': loaded_forms,
        }
