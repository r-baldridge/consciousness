"""
Visual Adapter - Form 01: Visual Consciousness
Processes visual input using ViT/CLIP models.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class VisualAdapter(SensoryAdapter):
    """
    Adapter for Form 01: Visual Consciousness.

    Processes visual inputs using Vision Transformer (ViT) and CLIP models
    for feature extraction, object recognition, and semantic understanding.
    """

    FORM_ID = "01-visual"
    NAME = "Visual Consciousness"
    MODALITY = "visual"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)

        # Visual processing state
        self._last_features: Optional[np.ndarray] = None
        self._attention_map: Optional[np.ndarray] = None
        self._detected_objects: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess visual input for the model.

        Args:
            input_data: Image data (numpy array, PIL Image, or path)

        Returns:
            Preprocessed tensor ready for inference
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Load image from path
            processed = await self._load_image(input_data)
        elif isinstance(input_data, dict):
            # Structured input with metadata
            processed = await self._process_structured_input(input_data)
        else:
            # Assume numpy array or tensor
            processed = self._normalize_image(input_data)

        return processed

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess model output.

        Args:
            model_output: Raw model output (embeddings, attention maps)

        Returns:
            Structured visual consciousness output
        """
        self._record_inference()

        # Extract components from model output
        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            attention = model_output.get('attention', None)
            objects = model_output.get('objects', [])
        else:
            embeddings = model_output
            attention = None
            objects = []

        # Store for later use
        self._last_features = embeddings
        if attention is not None:
            self._attention_map = attention

        # Compute salience
        salience = self.get_salience(embeddings)

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'attention_map': attention.tolist() if attention is not None and hasattr(attention, 'tolist') else None,
            'detected_objects': objects,
            'salience': salience,
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Perform visual inference.

        Args:
            input_data: Visual input

        Returns:
            Visual consciousness output
        """
        if not self.has_model:
            # Return mock output for testing
            return self._mock_inference(input_data)

        # Preprocess
        processed = await self.preprocess(input_data)

        # Run model inference
        try:
            if self.model and self.model.model_instance:
                model_output = await self._run_model(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"Visual inference error: {e}")
            raise

        # Postprocess
        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate visual input."""
        if isinstance(input_data, str):
            return True  # Will validate path during load
        if isinstance(input_data, dict):
            return 'image' in input_data or 'path' in input_data
        if hasattr(input_data, 'shape'):
            shape = input_data.shape
            # Should be (H, W, C) or (C, H, W) or (B, C, H, W)
            return len(shape) >= 2
        return False

    def get_input_spec(self) -> Dict[str, Any]:
        """Get visual input specification."""
        return {
            'type': 'image',
            'formats': ['RGB', 'RGBA', 'grayscale'],
            'max_resolution': [1024, 1024],
            'supported_inputs': [
                {'type': 'numpy_array', 'shape': '(H, W, C) or (C, H, W)'},
                {'type': 'path', 'extensions': ['.jpg', '.png', '.bmp']},
                {'type': 'dict', 'required_keys': ['image']},
            ],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get visual output specification."""
        return {
            'type': 'visual_perception',
            'embeddings': {'dimensions': 768},
            'attention_map': {'shape': 'variable'},
            'objects': {'format': 'list of detected objects'},
            'salience': {'range': [0.0, 1.0]},
        }

    def get_salience(self, features: Any) -> float:
        """
        Compute visual salience from features.

        Higher salience for:
        - Motion (change from previous frame)
        - High contrast
        - Novel patterns
        """
        if features is None:
            return 0.5

        salience = 0.5  # Base salience

        # Check for change from previous
        if self._last_features is not None and hasattr(features, 'shape'):
            try:
                # Compute difference
                diff = np.abs(features - self._last_features).mean()
                salience += min(0.3, diff * 0.5)  # Motion bonus
            except Exception:
                pass

        return min(1.0, salience)

    async def _load_image(self, path: str) -> Any:
        """Load image from path."""
        # Placeholder - would use PIL or cv2
        return {'path': path, 'loaded': True}

    async def _process_structured_input(self, input_data: Dict[str, Any]) -> Any:
        """Process structured input with metadata."""
        image = input_data.get('image')
        if image is not None:
            return self._normalize_image(image)
        path = input_data.get('path')
        if path:
            return await self._load_image(path)
        return input_data

    def _normalize_image(self, image: Any) -> Any:
        """Normalize image to standard format."""
        # Placeholder for normalization
        return image

    async def _run_model(self, processed_input: Any) -> Any:
        """Run the actual model inference."""
        # This would call the actual model
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output for testing."""
        return {
            'embeddings': np.random.randn(768).astype(np.float32),
            'attention': np.random.randn(14, 14).astype(np.float32),
            'objects': [
                {'class': 'object', 'confidence': 0.8, 'bbox': [0, 0, 100, 100]},
            ],
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 768,
            'attention_map': None,
            'detected_objects': [],
            'salience': 0.5,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
