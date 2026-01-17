"""
Gustatory Adapter - Form 05: Gustatory Consciousness
Processes taste sensor data.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class GustatoryAdapter(SensoryAdapter):
    """
    Adapter for Form 05: Gustatory Consciousness.

    Processes taste inputs using MLP ensembles
    for taste classification and intensity analysis.
    """

    FORM_ID = "05-gustatory"
    NAME = "Gustatory Consciousness"
    MODALITY = "gustatory"

    # Five basic taste categories
    TASTE_CATEGORIES = ['sweet', 'sour', 'salty', 'bitter', 'umami']

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)
        self._taste_profile: Dict[str, float] = {t: 0.0 for t in self.TASTE_CATEGORIES}

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess taste sensor data."""
        if isinstance(input_data, dict):
            return input_data.get('taste_vector', input_data)
        return input_data

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess gustatory model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            taste_profile = model_output.get('taste_profile', {})
            intensity = model_output.get('intensity', 0.0)
        else:
            embeddings = model_output
            taste_profile = {}
            intensity = 0.0

        self._taste_profile = taste_profile

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'taste_profile': taste_profile,
            'intensity': intensity,
            'dominant_taste': max(taste_profile, key=taste_profile.get) if taste_profile else None,
            'salience': self.get_salience(embeddings),
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform gustatory inference."""
        if not self.has_model:
            return self._mock_inference(input_data)

        processed = await self.preprocess(input_data)

        try:
            if self.model and self.model.model_instance:
                model_output = await self._run_model(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"Gustatory inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate gustatory input."""
        if isinstance(input_data, dict):
            return 'taste_vector' in input_data
        if hasattr(input_data, 'shape'):
            return input_data.shape[-1] == 64
        return isinstance(input_data, (list, tuple)) and len(input_data) == 64

    def get_input_spec(self) -> Dict[str, Any]:
        """Get gustatory input specification."""
        return {
            'type': 'taste_vector',
            'dimensions': 64,
            'taste_categories': self.TASTE_CATEGORIES,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get gustatory output specification."""
        return {
            'type': 'gustatory_perception',
            'embeddings': {'dimensions': 64},
            'taste_profile': {'categories': self.TASTE_CATEGORIES},
            'intensity': {'range': [0.0, 1.0]},
        }

    async def _run_model(self, processed_input: Any) -> Any:
        """Run model inference."""
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        taste_profile = {t: np.random.rand() for t in self.TASTE_CATEGORIES}
        return {
            'embeddings': np.random.randn(64).astype(np.float32),
            'taste_profile': taste_profile,
            'intensity': 0.5,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 64,
            'taste_profile': {t: 0.0 for t in self.TASTE_CATEGORIES},
            'intensity': 0.0,
            'dominant_taste': None,
            'salience': 0.3,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
