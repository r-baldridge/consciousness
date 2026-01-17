"""
Recurrent Adapter - Form 17: Recurrent Processing Consciousness
Implements recurrent/feedback processing for conscious perception.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import TheoreticalAdapter

logger = logging.getLogger(__name__)


class RecurrentAdapter(TheoreticalAdapter):
    """
    Adapter for Form 17: Recurrent Processing Consciousness.

    Implements recurrent processing theory where consciousness requires
    feedback loops between higher and lower processing levels.
    """

    FORM_ID = "17-recurrent-processing"
    NAME = "Recurrent Processing Consciousness"
    THEORY = "Recurrent Processing Theory"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.THEORY)

        # Recurrent state
        self._feedforward_state: Dict[str, Any] = {}
        self._feedback_state: Dict[str, Any] = {}
        self._recurrent_loops: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess feedforward activations."""
        if isinstance(input_data, dict):
            return input_data
        return {'activations': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess recurrent processing output."""
        self._record_inference()

        if isinstance(model_output, dict):
            feedback = model_output.get('feedback_signals', {})
            loops = model_output.get('recurrent_loops', [])
            recurrence_strength = model_output.get('recurrence_strength', 0.0)
        else:
            feedback = {}
            loops = []
            recurrence_strength = 0.0

        self._feedback_state = feedback
        self._recurrent_loops = loops

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback_signals': feedback,
            'recurrent_loops': loops,
            'recurrence_strength': recurrence_strength,
            'loop_count': len(loops),
            'theory_metrics': self.get_theory_metrics(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform recurrent processing inference."""
        processed = await self.preprocess(input_data)

        if isinstance(processed, dict) and 'activations' in processed:
            self._feedforward_state = processed

        if not self.has_model:
            return self._mock_inference(processed)

        try:
            if self.model and self.model.model_instance:
                model_output = self.model.model_instance(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"Recurrent inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate recurrent input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get recurrent input specification."""
        return {
            'type': 'feedforward_activations',
            'layers': ['early', 'mid', 'late'],
            'includes_timing': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get recurrent output specification."""
        return {
            'type': 'recurrent_state',
            'includes_feedback_signals': True,
            'includes_loop_info': True,
        }

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'feedback_signals': {'early': 0.5, 'mid': 0.6, 'late': 0.7},
            'recurrent_loops': [{'from': 'late', 'to': 'early', 'strength': 0.5}],
            'recurrence_strength': 0.5,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback_signals': {},
            'recurrent_loops': [],
            'recurrence_strength': 0.0,
            'loop_count': 0,
            'theory_metrics': self.get_theory_metrics(),
            'mock': True,
        }
