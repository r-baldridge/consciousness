"""
Attention Adapter - Form 07: Emotional/Attention Consciousness
Manages attention allocation across forms.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import CognitiveAdapter

logger = logging.getLogger(__name__)


class AttentionAdapter(CognitiveAdapter):
    """
    Adapter for Form 07: Emotional/Attention Consciousness.

    Manages attention allocation, focus, and filtering
    across all consciousness forms.
    """

    FORM_ID = "07-emotional"
    NAME = "Attention Consciousness"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME)

        # Attention state
        self._attention_weights: Dict[str, float] = {}
        self._focus_target: Optional[str] = None
        self._attention_breadth: float = 0.5  # 0=narrow, 1=broad

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess attention request."""
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess attention model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            weights = model_output.get('attention_weights', {})
            focus = model_output.get('focus_target', None)
            breadth = model_output.get('breadth', 0.5)
        else:
            weights = {}
            focus = None
            breadth = 0.5

        self._attention_weights = weights
        self._focus_target = focus
        self._attention_breadth = breadth

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'attention_weights': weights,
            'focus_target': focus,
            'breadth': breadth,
            'state': self.get_state(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform attention inference."""
        if not self.has_model:
            return self._mock_inference(input_data)

        processed = await self.preprocess(input_data)

        try:
            if self.model and self.model.model_instance:
                model_output = self.model.model_instance(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"Attention inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate attention input."""
        return True  # Flexible input

    def get_input_spec(self) -> Dict[str, Any]:
        """Get attention input specification."""
        return {
            'type': 'attention_request',
            'supports_multimodal': True,
            'fields': ['targets', 'priorities', 'context'],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get attention output specification."""
        return {
            'type': 'attention_weights',
            'dimensions': 512,
            'includes_focus_target': True,
            'includes_breadth': True,
        }

    def get_attention_weights(self) -> Dict[str, float]:
        """Get current attention weights."""
        return dict(self._attention_weights)

    def set_attention(self, target: str, weight: float) -> None:
        """Set attention weight for a target."""
        self._attention_weights[target] = max(0.0, min(1.0, weight))

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'attention_weights': {},
            'focus_target': None,
            'breadth': 0.5,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'attention_weights': {},
            'focus_target': None,
            'breadth': 0.5,
            'state': {},
            'mock': True,
        }
