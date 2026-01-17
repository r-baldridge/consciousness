"""
Proprioceptive Adapter - Form 06: Interoceptive/Proprioceptive Consciousness
Processes body state and position awareness data.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class ProprioceptiveAdapter(SensoryAdapter):
    """
    Adapter for Form 06: Interoceptive/Proprioceptive Consciousness.

    Processes body state inputs including joint positions,
    muscle tension, and internal body state awareness.
    """

    FORM_ID = "06-interoceptive"
    NAME = "Interoceptive Consciousness"
    MODALITY = "proprioceptive"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)
        self._body_state: Dict[str, Any] = {}
        self._joint_positions: Optional[np.ndarray] = None
        self._muscle_tension: Optional[np.ndarray] = None

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess body state data."""
        if isinstance(input_data, dict):
            return self._process_body_state(input_data)
        return input_data

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess proprioceptive model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            body_schema = model_output.get('body_schema', {})
            balance = model_output.get('balance', 0.5)
        else:
            embeddings = model_output
            body_schema = {}
            balance = 0.5

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'body_schema': body_schema,
            'balance': balance,
            'salience': self.get_salience(embeddings),
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform proprioceptive inference."""
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
            logger.error(f"Proprioceptive inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate proprioceptive input."""
        if isinstance(input_data, dict):
            return any(k in input_data for k in ['body_state', 'joints', 'muscles'])
        if hasattr(input_data, 'shape'):
            return input_data.shape[-1] == 512
        return False

    def get_input_spec(self) -> Dict[str, Any]:
        """Get proprioceptive input specification."""
        return {
            'type': 'body_state_vector',
            'dimensions': 512,
            'includes_joint_positions': True,
            'includes_muscle_tension': True,
            'includes_balance': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get proprioceptive output specification."""
        return {
            'type': 'proprioceptive_perception',
            'embeddings': {'dimensions': 256},
            'body_schema': {'format': 'hierarchical'},
            'balance': {'range': [0.0, 1.0]},
        }

    def _process_body_state(self, input_data: Dict[str, Any]) -> Any:
        """Process structured body state input."""
        self._body_state = input_data
        return input_data

    async def _run_model(self, processed_input: Any) -> Any:
        """Run model inference."""
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'embeddings': np.random.randn(256).astype(np.float32),
            'body_schema': {
                'head': {'position': [0, 0, 1.7]},
                'torso': {'position': [0, 0, 1.0]},
                'limbs': {'arms': 'relaxed', 'legs': 'standing'},
            },
            'balance': 0.9,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 256,
            'body_schema': {},
            'balance': 0.5,
            'salience': 0.4,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
