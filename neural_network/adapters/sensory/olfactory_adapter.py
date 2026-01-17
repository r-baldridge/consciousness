"""
Olfactory Adapter - Form 04: Olfactory Consciousness
Processes smell/chemical sensor data.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class OlfactoryAdapter(SensoryAdapter):
    """
    Adapter for Form 04: Olfactory Consciousness.

    Processes olfactory inputs using MLP ensembles
    for odor classification and chemical analysis.
    """

    FORM_ID = "04-olfactory"
    NAME = "Olfactory Consciousness"
    MODALITY = "olfactory"

    # Common odor categories
    ODOR_CATEGORIES = [
        'floral', 'fruity', 'woody', 'chemical',
        'minty', 'sweet', 'toasted', 'decayed'
    ]

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)
        self._detected_odors: List[str] = []
        self._concentration: float = 0.0

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess chemical sensor data."""
        if isinstance(input_data, dict):
            return input_data.get('chemical_vector', input_data)
        return input_data

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess olfactory model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            odors = model_output.get('odors', [])
            concentration = model_output.get('concentration', 0.0)
        else:
            embeddings = model_output
            odors = []
            concentration = 0.0

        self._detected_odors = odors
        self._concentration = concentration

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'detected_odors': odors,
            'concentration': concentration,
            'salience': self.get_salience(embeddings),
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform olfactory inference."""
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
            logger.error(f"Olfactory inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate olfactory input."""
        if isinstance(input_data, dict):
            return 'chemical_vector' in input_data
        if hasattr(input_data, 'shape'):
            return input_data.shape[-1] == 128
        return isinstance(input_data, (list, tuple)) and len(input_data) == 128

    def get_input_spec(self) -> Dict[str, Any]:
        """Get olfactory input specification."""
        return {
            'type': 'chemical_vector',
            'dimensions': 128,
            'value_range': [0.0, 1.0],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get olfactory output specification."""
        return {
            'type': 'olfactory_perception',
            'embeddings': {'dimensions': 64},
            'odor_categories': self.ODOR_CATEGORIES,
            'concentration': {'range': [0.0, 1.0]},
        }

    async def _run_model(self, processed_input: Any) -> Any:
        """Run model inference."""
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'embeddings': np.random.randn(64).astype(np.float32),
            'odors': ['neutral'],
            'concentration': 0.3,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 64,
            'detected_odors': [],
            'concentration': 0.0,
            'salience': 0.3,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
