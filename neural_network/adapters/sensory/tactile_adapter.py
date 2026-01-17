"""
Tactile Adapter - Form 03: Somatosensory/Tactile Consciousness
Processes touch and pressure sensor data.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class TactileAdapter(SensoryAdapter):
    """
    Adapter for Form 03: Somatosensory/Tactile Consciousness.

    Processes tactile inputs from sensor arrays using CNNs
    for texture recognition and pressure mapping.
    """

    FORM_ID = "03-somatosensory"
    NAME = "Somatosensory Consciousness"
    MODALITY = "tactile"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)
        self._pressure_map: Optional[np.ndarray] = None
        self._texture_features: Optional[np.ndarray] = None

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess tactile sensor data."""
        if isinstance(input_data, dict):
            return self._process_sensor_array(input_data)
        return self._normalize_tactile(input_data)

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess tactile model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            pressure = model_output.get('pressure_map', None)
            texture = model_output.get('texture', None)
        else:
            embeddings = model_output
            pressure = None
            texture = None

        self._pressure_map = pressure
        self._texture_features = embeddings

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'pressure_map': pressure.tolist() if pressure is not None and hasattr(pressure, 'tolist') else None,
            'texture_classification': texture,
            'salience': self.get_salience(embeddings),
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform tactile inference."""
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
            logger.error(f"Tactile inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate tactile input."""
        if isinstance(input_data, dict):
            return 'sensors' in input_data or 'pressure' in input_data
        if hasattr(input_data, 'shape'):
            return len(input_data.shape) >= 2
        return False

    def get_input_spec(self) -> Dict[str, Any]:
        """Get tactile input specification."""
        return {
            'type': 'sensor_array',
            'channels': 256,
            'spatial_dims': [32, 32],
            'value_range': [0.0, 1.0],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get tactile output specification."""
        return {
            'type': 'tactile_perception',
            'embeddings': {'dimensions': 256},
            'pressure_map': {'shape': [32, 32]},
            'texture_classification': {'classes': ['smooth', 'rough', 'soft', 'hard']},
        }

    def _process_sensor_array(self, input_data: Dict[str, Any]) -> Any:
        """Process structured sensor array input."""
        return input_data.get('sensors', input_data)

    def _normalize_tactile(self, data: Any) -> Any:
        """Normalize tactile data."""
        return data

    async def _run_model(self, processed_input: Any) -> Any:
        """Run model inference."""
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'embeddings': np.random.randn(256).astype(np.float32),
            'pressure_map': np.random.rand(32, 32).astype(np.float32),
            'texture': 'smooth',
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 256,
            'pressure_map': None,
            'texture_classification': None,
            'salience': 0.5,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
