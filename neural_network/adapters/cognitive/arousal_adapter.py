"""
Arousal Adapter - Form 08: Arousal/Vigilance Consciousness
Critical form that gates all other forms.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import CognitiveAdapter, ArousalAdapterInterface

logger = logging.getLogger(__name__)


class ArousalAdapter(CognitiveAdapter, ArousalAdapterInterface):
    """
    Adapter for Form 08: Arousal/Vigilance Consciousness.

    CRITICAL FORM - Must always be loaded.

    Manages arousal levels that gate resource allocation
    to all other consciousness forms.
    """

    FORM_ID = "08-arousal"
    NAME = "Arousal Consciousness"

    # Arousal state categories
    STATES = ['sleep', 'drowsy', 'relaxed', 'alert', 'focused', 'hyperaroused']

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME)

        # Arousal state
        self._arousal_level: float = 0.5
        self._arousal_state: str = 'alert'
        self._arousal_trend: float = 0.0  # -1 to 1
        self._gating_signals: Dict[str, float] = {}

        # Input tracking
        self._sensory_arousal: float = 0.5
        self._emotional_arousal: float = 0.5
        self._circadian_factor: float = 0.5

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess arousal inputs."""
        if isinstance(input_data, dict):
            # Extract arousal signals
            self._sensory_arousal = input_data.get('sensory_arousal', self._sensory_arousal)
            self._emotional_arousal = input_data.get('emotional_arousal', self._emotional_arousal)
            self._circadian_factor = input_data.get('circadian_factor', self._circadian_factor)
        return input_data

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess arousal model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            level = model_output.get('arousal_level', self._arousal_level)
            state = model_output.get('arousal_state', self._arousal_state)
            trend = model_output.get('arousal_trend', 0.0)
            gating = model_output.get('gating_signals', {})
        else:
            level = float(model_output) if model_output else self._arousal_level
            state = self._arousal_state
            trend = 0.0
            gating = {}

        self._arousal_level = max(0.0, min(1.0, level))
        self._arousal_state = state
        self._arousal_trend = trend
        self._gating_signals = gating

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'arousal_level': self._arousal_level,
            'arousal_state': self._arousal_state,
            'arousal_trend': self._arousal_trend,
            'gating_signals': self._gating_signals,
            'state': self.get_state(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform arousal inference."""
        processed = await self.preprocess(input_data)

        if not self.has_model:
            # Compute arousal from inputs
            level = self._compute_arousal()
            model_output = {
                'arousal_level': level,
                'arousal_state': self._level_to_state(level),
                'arousal_trend': 0.0,
                'gating_signals': self._compute_gating(level),
            }
        else:
            try:
                if self.model and self.model.model_instance:
                    model_output = self.model.model_instance(processed)
                else:
                    model_output = self._mock_model_output(processed)
            except Exception as e:
                self._record_error()
                logger.error(f"Arousal inference error: {e}")
                raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate arousal input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get arousal input specification."""
        return {
            'type': 'arousal_signals',
            'includes_sensory': True,
            'includes_emotional': True,
            'includes_circadian': True,
            'includes_task_demands': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get arousal output specification."""
        return {
            'type': 'arousal_level',
            'range': [0.0, 1.0],
            'states': self.STATES,
            'includes_gating_signals': True,
        }

    # ArousalAdapterInterface implementation

    async def get_arousal_level(self) -> float:
        """Get current arousal level (0.0-1.0)."""
        return self._arousal_level

    async def get_arousal_state(self) -> str:
        """Get current arousal state category."""
        return self._arousal_state

    async def get_gating_signals(self) -> Dict[str, float]:
        """Get gating signals for other forms."""
        return dict(self._gating_signals)

    def _compute_arousal(self) -> float:
        """Compute arousal level from inputs."""
        # Weighted combination of inputs
        level = (
            0.4 * self._sensory_arousal +
            0.3 * self._emotional_arousal +
            0.3 * self._circadian_factor
        )
        return max(0.0, min(1.0, level))

    def _level_to_state(self, level: float) -> str:
        """Convert arousal level to state category."""
        if level < 0.1:
            return 'sleep'
        elif level < 0.3:
            return 'drowsy'
        elif level < 0.5:
            return 'relaxed'
        elif level < 0.7:
            return 'alert'
        elif level < 0.9:
            return 'focused'
        else:
            return 'hyperaroused'

    def _compute_gating(self, level: float) -> Dict[str, float]:
        """Compute gating signals based on arousal level."""
        # Default gating based on arousal
        base_gate = level

        return {
            'sensory_gate': min(1.0, base_gate * 1.2),
            'cognitive_gate': base_gate,
            'emotional_gate': min(1.0, base_gate * 1.1),
            'memory_gate': base_gate * 0.9,
        }

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        level = self._compute_arousal()
        return {
            'arousal_level': level,
            'arousal_state': self._level_to_state(level),
            'arousal_trend': 0.0,
            'gating_signals': self._compute_gating(level),
        }
