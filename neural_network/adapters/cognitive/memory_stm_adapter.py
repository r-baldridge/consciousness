"""
Memory STM Adapter - Form 09: Perceptual/Short-Term Memory Consciousness
Manages short-term/working memory.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import CognitiveAdapter

logger = logging.getLogger(__name__)


class MemorySTMAdapter(CognitiveAdapter):
    """
    Adapter for Form 09: Perceptual/Short-Term Memory Consciousness.

    Manages working memory using LSTM-based models
    with capacity limits and decay.
    """

    FORM_ID = "09-perceptual"
    NAME = "Short-Term Memory Consciousness"

    # Working memory capacity (classic 7+/-2)
    CAPACITY = 7

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME)

        # Memory state
        self._memory_buffer: List[Dict[str, Any]] = []
        self._attention_focus: Optional[int] = None

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess memory input."""
        if isinstance(input_data, dict):
            return input_data
        return {'item': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess memory model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            decay = model_output.get('decay_prediction', [])
            retrieval = model_output.get('retrieval', None)
        else:
            embeddings = model_output
            decay = []
            retrieval = None

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'decay_prediction': decay,
            'retrieval': retrieval,
            'buffer_size': len(self._memory_buffer),
            'capacity': self.CAPACITY,
            'state': self.get_state(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform memory inference."""
        processed = await self.preprocess(input_data)

        if not self.has_model:
            return self._mock_inference(processed)

        try:
            if self.model and self.model.model_instance:
                model_output = self.model.model_instance(processed)
            else:
                model_output = self._mock_model_output(processed)
        except Exception as e:
            self._record_error()
            logger.error(f"Memory STM inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate memory input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get memory input specification."""
        return {
            'type': 'memory_item',
            'max_sequence_length': 128,
            'operations': ['store', 'retrieve', 'update', 'forget'],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get memory output specification."""
        return {
            'type': 'memory_embedding',
            'dimensions': 512,
            'includes_decay_prediction': True,
            'capacity': self.CAPACITY,
        }

    async def store(self, item: Dict[str, Any]) -> bool:
        """Store an item in working memory."""
        if len(self._memory_buffer) >= self.CAPACITY:
            # Remove oldest item (FIFO)
            self._memory_buffer.pop(0)

        item['stored_at'] = datetime.now(timezone.utc).isoformat()
        self._memory_buffer.append(item)
        return True

    async def retrieve(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieve an item from working memory."""
        if 0 <= index < len(self._memory_buffer):
            return self._memory_buffer[index]
        return None

    async def clear(self) -> None:
        """Clear working memory."""
        self._memory_buffer.clear()

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'embeddings': np.random.randn(512).astype(np.float32),
            'decay_prediction': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            'retrieval': None,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 512,
            'decay_prediction': [],
            'retrieval': None,
            'buffer_size': len(self._memory_buffer),
            'capacity': self.CAPACITY,
            'state': {},
            'mock': True,
        }
