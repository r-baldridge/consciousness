"""
HOT Adapter - Form 15: Higher-Order Thought
Implements meta-cognitive awareness of mental states.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import TheoreticalAdapter

logger = logging.getLogger(__name__)


class HOTAdapter(TheoreticalAdapter):
    """
    Adapter for Form 15: Higher-Order Thought (HOT).

    Implements higher-order representations that make first-order
    mental states conscious through meta-cognitive awareness.
    """

    FORM_ID = "15-higher-order-thought"
    NAME = "Higher-Order Thought Consciousness"
    THEORY = "Higher-Order Thought Theory"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.THEORY)

        # HOT state
        self._first_order_states: List[Dict[str, Any]] = []
        self._higher_order_representations: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess first-order states for HOT processing."""
        if isinstance(input_data, dict):
            return input_data
        return {'states': [input_data] if input_data else []}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess HOT output."""
        self._record_inference()

        if isinstance(model_output, dict):
            hot_reps = model_output.get('higher_order_representations', [])
            meta_awareness = model_output.get('meta_awareness', 0.5)
        else:
            hot_reps = []
            meta_awareness = 0.5

        self._higher_order_representations = hot_reps

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'higher_order_representations': hot_reps,
            'meta_awareness_level': meta_awareness,
            'first_order_count': len(self._first_order_states),
            'theory_metrics': self.get_theory_metrics(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform HOT inference."""
        processed = await self.preprocess(input_data)

        if isinstance(processed, dict) and 'states' in processed:
            self._first_order_states = processed['states']

        if not self.has_model:
            return self._mock_inference(processed)

        try:
            if self.model and self.model.model_instance:
                model_output = self.model.model_instance(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"HOT inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate HOT input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get HOT input specification."""
        return {
            'type': 'first_order_states',
            'max_states': 50,
            'includes_state_content': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get HOT output specification."""
        return {
            'type': 'higher_order_representations',
            'includes_meta_awareness': True,
            'includes_introspective_content': True,
        }

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'higher_order_representations': [],
            'meta_awareness': 0.5,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'higher_order_representations': [],
            'meta_awareness_level': 0.5,
            'first_order_count': len(self._first_order_states),
            'theory_metrics': self.get_theory_metrics(),
            'mock': True,
        }
