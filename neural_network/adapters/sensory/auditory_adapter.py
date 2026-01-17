"""
Auditory Adapter - Form 02: Auditory Consciousness
Processes audio input using Whisper and audio transformers.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import SensoryAdapter

logger = logging.getLogger(__name__)


class AuditoryAdapter(SensoryAdapter):
    """
    Adapter for Form 02: Auditory Consciousness.

    Processes audio inputs using Whisper for transcription and
    audio transformers for feature extraction.
    """

    FORM_ID = "02-auditory"
    NAME = "Auditory Consciousness"
    MODALITY = "auditory"
    SAMPLE_RATE = 16000

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.MODALITY)

        # Audio processing state
        self._last_features: Optional[np.ndarray] = None
        self._transcription: str = ""
        self._audio_events: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess audio input for the model.

        Args:
            input_data: Audio data (numpy array, path, or bytes)

        Returns:
            Preprocessed audio ready for inference
        """
        if isinstance(input_data, str):
            # Load audio from path
            processed = await self._load_audio(input_data)
        elif isinstance(input_data, bytes):
            # Decode audio bytes
            processed = self._decode_audio(input_data)
        elif isinstance(input_data, dict):
            processed = await self._process_structured_input(input_data)
        else:
            # Assume numpy array
            processed = self._normalize_audio(input_data)

        return processed

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            embeddings = model_output.get('embeddings', [])
            transcription = model_output.get('transcription', '')
            events = model_output.get('events', [])
        else:
            embeddings = model_output
            transcription = ""
            events = []

        self._last_features = embeddings
        self._transcription = transcription

        salience = self.get_salience(embeddings)

        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'transcription': transcription,
            'audio_events': events,
            'salience': salience,
            'binding_info': self.get_binding_info(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform auditory inference."""
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
            logger.error(f"Auditory inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate audio input."""
        if isinstance(input_data, str):
            return True
        if isinstance(input_data, bytes):
            return len(input_data) > 0
        if isinstance(input_data, dict):
            return 'audio' in input_data or 'path' in input_data
        if hasattr(input_data, 'shape'):
            return len(input_data.shape) in [1, 2]
        return False

    def get_input_spec(self) -> Dict[str, Any]:
        """Get audio input specification."""
        return {
            'type': 'audio',
            'sample_rate': self.SAMPLE_RATE,
            'max_duration_seconds': 30,
            'formats': ['wav', 'mp3', 'flac'],
            'channels': [1, 2],
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get audio output specification."""
        return {
            'type': 'auditory_perception',
            'embeddings': {'dimensions': 1024},
            'transcription': {'type': 'string'},
            'audio_events': {'format': 'list of events'},
            'salience': {'range': [0.0, 1.0]},
        }

    def get_salience(self, features: Any) -> float:
        """Compute auditory salience."""
        if features is None:
            return 0.5

        salience = 0.5

        # Salience based on audio energy/loudness
        if hasattr(features, 'mean'):
            energy = np.abs(features).mean()
            salience += min(0.3, energy * 0.1)

        return min(1.0, salience)

    async def _load_audio(self, path: str) -> Any:
        """Load audio from path."""
        return {'path': path, 'sample_rate': self.SAMPLE_RATE}

    def _decode_audio(self, audio_bytes: bytes) -> Any:
        """Decode audio from bytes."""
        return {'bytes': len(audio_bytes), 'sample_rate': self.SAMPLE_RATE}

    async def _process_structured_input(self, input_data: Dict[str, Any]) -> Any:
        """Process structured audio input."""
        if 'audio' in input_data:
            return self._normalize_audio(input_data['audio'])
        if 'path' in input_data:
            return await self._load_audio(input_data['path'])
        return input_data

    def _normalize_audio(self, audio: Any) -> Any:
        """Normalize audio to standard format."""
        return audio

    async def _run_model(self, processed_input: Any) -> Any:
        """Run model inference."""
        if self.model and self.model.model_instance:
            return self.model.model_instance(processed_input)
        return self._mock_model_output(processed_input)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'embeddings': np.random.randn(1024).astype(np.float32),
            'transcription': '',
            'events': [],
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'modality': self.MODALITY,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'embeddings': [0.0] * 1024,
            'transcription': '',
            'audio_events': [],
            'salience': 0.5,
            'binding_info': self.get_binding_info(),
            'mock': True,
        }
