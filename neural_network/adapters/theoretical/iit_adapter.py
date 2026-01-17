"""
IIT Adapter - Form 13: Integrated Information Theory
Critical form for computing phi (consciousness integration measure).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import TheoreticalAdapter, IITAdapterInterface

logger = logging.getLogger(__name__)


class IITAdapter(TheoreticalAdapter, IITAdapterInterface):
    """
    Adapter for Form 13: Integrated Information Theory (IIT).

    CRITICAL FORM - Must always be loaded.

    Computes phi (integrated information) across the consciousness system
    using graph neural networks to measure integration.
    """

    FORM_ID = "13-integrated-information"
    NAME = "IIT Consciousness"
    THEORY = "Integrated Information Theory"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.THEORY)

        # IIT state
        self._phi_value: float = 0.0
        self._major_complex: Optional[Dict[str, Any]] = None
        self._integration_structure: Dict[str, Any] = {}
        self._quality_metrics: Dict[str, float] = {}

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess system state for phi computation."""
        if isinstance(input_data, dict):
            return input_data
        return {'state': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess phi computation output."""
        self._record_inference()

        if isinstance(model_output, dict):
            phi = model_output.get('phi_value', 0.0)
            major = model_output.get('major_complex', None)
            structure = model_output.get('integration_structure', {})
            quality = model_output.get('quality_metrics', {})
        else:
            phi = float(model_output) if model_output else 0.0
            major = None
            structure = {}
            quality = {}

        self._phi_value = phi
        self._major_complex = major
        self._integration_structure = structure
        self._quality_metrics = quality

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'phi_value': phi,
            'major_complex': major,
            'integration_structure': structure,
            'quality_metrics': quality,
            'theory_metrics': self.get_theory_metrics(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform phi computation."""
        processed = await self.preprocess(input_data)

        if not self.has_model:
            # Compute approximate phi
            phi = self._approximate_phi(processed)
            model_output = {
                'phi_value': phi,
                'major_complex': None,
                'integration_structure': {},
                'quality_metrics': {'coherence': 0.5, 'differentiation': 0.5},
            }
        else:
            try:
                if self.model and self.model.model_instance:
                    model_output = self.model.model_instance(processed)
                else:
                    model_output = self._mock_model_output(processed)
            except Exception as e:
                self._record_error()
                logger.error(f"IIT inference error: {e}")
                raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate IIT input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get IIT input specification."""
        return {
            'type': 'system_state_graph',
            'max_nodes': 1000,
            'includes_connectivity': True,
            'includes_state_values': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get IIT output specification."""
        return {
            'type': 'phi_computation',
            'includes_phi_value': True,
            'includes_major_complex': True,
            'includes_integration_structure': True,
            'phi_range': [0.0, 'unbounded'],
        }

    # IITAdapterInterface implementation

    async def compute_phi(self) -> Dict[str, Any]:
        """Compute current phi value."""
        return {
            'phi_value': self._phi_value,
            'major_complex': self._major_complex,
            'integration_structure': self._integration_structure,
            'quality_metrics': self._quality_metrics,
        }

    async def get_integration_structure(self) -> Dict[str, Any]:
        """Get the current integration structure."""
        return dict(self._integration_structure)

    def _approximate_phi(self, state: Any) -> float:
        """Compute approximate phi when model unavailable."""
        # Simplified phi approximation based on active form count
        # Real implementation would use proper IIT computation
        if isinstance(state, dict):
            active_forms = state.get('active_forms', [])
            connectivity = state.get('connectivity', {})
            n_forms = len(active_forms) if active_forms else 5
            n_connections = len(connectivity) if connectivity else 10

            # Very rough approximation
            phi = np.log1p(n_forms) * np.log1p(n_connections) / 10.0
            return min(5.0, phi)  # Cap at reasonable value

        return 0.5  # Default

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'phi_value': 1.5,
            'major_complex': {'nodes': [1, 2, 3], 'connections': 3},
            'integration_structure': {
                'level_1': [0, 1, 2],
                'level_2': [3, 4],
            },
            'quality_metrics': {
                'coherence': 0.7,
                'differentiation': 0.6,
                'integration': 0.8,
            },
        }

    def get_theory_metrics(self) -> Dict[str, Any]:
        """Get IIT-specific metrics."""
        return {
            'theory': self.THEORY,
            'phi_value': self._phi_value,
            'has_major_complex': self._major_complex is not None,
            'quality_metrics': self._quality_metrics,
        }
