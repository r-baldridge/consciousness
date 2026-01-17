"""
Base Adapter - Abstract interface for all form adapters
Part of the Neural Network module for the Consciousness system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.model_registry import LoadedModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for an adapter."""
    form_id: str
    name: str
    model_config: "ModelConfig"
    custom_settings: Dict[str, Any]


@dataclass
class InferenceMetrics:
    """Metrics from an inference call."""
    latency_ms: float
    preprocessing_ms: float
    model_inference_ms: float
    postprocessing_ms: float
    batch_size: int
    timestamp: datetime


class FormAdapter(ABC):
    """
    Abstract base class for form adapters.

    Each consciousness form has an adapter that:
    - Manages the form's AI model
    - Handles input preprocessing
    - Performs inference
    - Handles output postprocessing
    - Provides form-specific functionality
    """

    def __init__(self, form_id: str, name: str):
        """
        Initialize the adapter.

        Args:
            form_id: The form identifier (e.g., '01-visual')
            name: Human-readable form name
        """
        self.form_id = form_id
        self.name = name
        self.model: Optional["LoadedModel"] = None
        self._initialized = False
        self._last_inference: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0

    @property
    def is_initialized(self) -> bool:
        """Check if adapter is initialized."""
        return self._initialized

    @property
    def has_model(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    async def initialize(self, model: "LoadedModel") -> None:
        """
        Initialize the adapter with a loaded model.

        Args:
            model: The loaded model instance
        """
        self.model = model
        self._initialized = True
        logger.info(f"Adapter {self.form_id} initialized")

    async def shutdown(self) -> None:
        """Shutdown the adapter and release resources."""
        self.model = None
        self._initialized = False
        logger.info(f"Adapter {self.form_id} shutdown")

    @abstractmethod
    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data for the model.

        Args:
            input_data: Raw input data

        Returns:
            Preprocessed data ready for inference
        """
        pass

    @abstractmethod
    async def postprocess(self, model_output: Any) -> Any:
        """
        Postprocess model output.

        Args:
            model_output: Raw model output

        Returns:
            Processed output in the expected format
        """
        pass

    @abstractmethod
    async def inference(self, input_data: Any) -> Any:
        """
        Perform inference with the model.

        Args:
            input_data: Input data (will be preprocessed)

        Returns:
            Processed output
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_input_spec(self) -> Dict[str, Any]:
        """
        Get the input specification for this form.

        Returns:
            Dictionary describing expected input format
        """
        pass

    @abstractmethod
    def get_output_spec(self) -> Dict[str, Any]:
        """
        Get the output specification for this form.

        Returns:
            Dictionary describing output format
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the adapter.

        Returns:
            Health status information
        """
        return {
            'form_id': self.form_id,
            'name': self.name,
            'initialized': self._initialized,
            'has_model': self.has_model,
            'inference_count': self._inference_count,
            'error_count': self._error_count,
            'last_inference': (
                self._last_inference.isoformat()
                if self._last_inference else None
            ),
            'status': 'healthy' if self._initialized and self.has_model else 'degraded',
        }

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            'form_id': self.form_id,
            'name': self.name,
            'initialized': self._initialized,
            'has_model': self.has_model,
            'inference_count': self._inference_count,
            'error_count': self._error_count,
        }

    def _record_inference(self) -> None:
        """Record an inference execution."""
        self._inference_count += 1
        self._last_inference = datetime.now(timezone.utc)

    def _record_error(self) -> None:
        """Record an error."""
        self._error_count += 1


class SensoryAdapter(FormAdapter):
    """
    Base adapter for sensory forms (01-06).

    Sensory adapters handle:
    - Raw sensory input processing
    - Feature extraction
    - Salience computation
    - Cross-modal binding
    """

    def __init__(self, form_id: str, name: str, modality: str):
        super().__init__(form_id, name)
        self.modality = modality

    def get_salience(self, features: Any) -> float:
        """
        Compute salience score for extracted features.

        Args:
            features: Extracted features

        Returns:
            Salience score 0.0-1.0
        """
        return 0.5  # Default

    def get_binding_info(self) -> Dict[str, Any]:
        """
        Get cross-modal binding information.

        Returns:
            Binding information for integration
        """
        return {
            'modality': self.modality,
            'form_id': self.form_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }


class CognitiveAdapter(FormAdapter):
    """
    Base adapter for cognitive forms (07-12).

    Cognitive adapters handle:
    - Higher-level processing
    - State management
    - Memory interaction
    - Executive functions
    """

    def __init__(self, form_id: str, name: str):
        super().__init__(form_id, name)
        self._state: Dict[str, Any] = {}

    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        return dict(self._state)

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update cognitive state."""
        self._state.update(updates)


class TheoreticalAdapter(FormAdapter):
    """
    Base adapter for theoretical forms (13-17).

    Theoretical adapters implement:
    - Consciousness theory computations
    - Integration measures
    - Global workspace operations
    - Meta-cognitive functions
    """

    def __init__(self, form_id: str, name: str, theory: str):
        super().__init__(form_id, name)
        self.theory = theory

    def get_theory_metrics(self) -> Dict[str, Any]:
        """
        Get theory-specific metrics.

        Returns:
            Metrics relevant to the consciousness theory
        """
        return {
            'theory': self.theory,
            'form_id': self.form_id,
        }


class SpecializedAdapter(FormAdapter):
    """
    Base adapter for specialized forms (18-27).

    Specialized adapters handle:
    - Altered states of consciousness
    - Non-standard processing modes
    - Rare but important experiences
    """

    def __init__(self, form_id: str, name: str, specialization: str):
        super().__init__(form_id, name)
        self.specialization = specialization

    def is_state_active(self) -> bool:
        """Check if the specialized state is currently active."""
        return False

    def get_state_indicators(self) -> Dict[str, Any]:
        """Get indicators of the specialized state."""
        return {
            'specialization': self.specialization,
            'active': self.is_state_active(),
        }


# ============================================
# AROUSAL ADAPTER INTERFACE (Form 08)
# ============================================

class ArousalAdapterInterface(ABC):
    """Interface for the Arousal adapter (Form 08)."""

    @abstractmethod
    async def get_arousal_level(self) -> float:
        """
        Get current arousal level.

        Returns:
            Arousal level 0.0-1.0
        """
        pass

    @abstractmethod
    async def get_arousal_state(self) -> str:
        """
        Get current arousal state category.

        Returns:
            State: 'sleep', 'drowsy', 'relaxed', 'alert', 'focused', 'hyperaroused'
        """
        pass

    @abstractmethod
    async def get_gating_signals(self) -> Dict[str, float]:
        """
        Get gating signals for other forms.

        Returns:
            Dictionary of form_id -> gate_value (0.0-1.0)
        """
        pass


# ============================================
# IIT ADAPTER INTERFACE (Form 13)
# ============================================

class IITAdapterInterface(ABC):
    """Interface for the Integrated Information Theory adapter (Form 13)."""

    @abstractmethod
    async def compute_phi(self) -> Dict[str, Any]:
        """
        Compute phi value.

        Returns:
            Dict with phi_value, major_complex, integration_structure
        """
        pass

    @abstractmethod
    async def get_integration_structure(self) -> Dict[str, Any]:
        """
        Get the current integration structure.

        Returns:
            Integration structure information
        """
        pass


# ============================================
# GLOBAL WORKSPACE ADAPTER INTERFACE (Form 14)
# ============================================

class GlobalWorkspaceAdapterInterface(ABC):
    """Interface for the Global Workspace adapter (Form 14)."""

    @abstractmethod
    async def get_workspace_state(self) -> Dict[str, Any]:
        """
        Get current workspace state.

        Returns:
            Workspace state with contents and slots
        """
        pass

    @abstractmethod
    async def submit_content(self, content: Dict[str, Any]) -> bool:
        """
        Submit content for workspace competition.

        Args:
            content: Content to submit

        Returns:
            True if accepted for competition
        """
        pass

    @abstractmethod
    async def broadcast(self, content: Dict[str, Any]) -> None:
        """
        Broadcast content to all forms.

        Args:
            content: Content to broadcast
        """
        pass
