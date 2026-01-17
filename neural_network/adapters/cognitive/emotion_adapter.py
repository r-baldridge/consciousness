"""
Emotion Adapter - Form 11: Meta-Consciousness/Emotion Processing
Manages emotion recognition and processing.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import CognitiveAdapter

logger = logging.getLogger(__name__)


class EmotionAdapter(CognitiveAdapter):
    """
    Adapter for Form 11: Meta-Consciousness/Emotion Processing.

    Processes emotional inputs using transformer-based models
    for emotion classification and valence-arousal mapping.
    """

    FORM_ID = "11-meta-consciousness"
    NAME = "Emotion Processing Consciousness"

    # Emotion categories (GoEmotions-style)
    EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME)

        # Emotion state
        self._current_emotions: Dict[str, float] = {}
        self._valence: float = 0.0  # -1 to 1
        self._emotional_arousal: float = 0.5  # 0 to 1

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess emotional input."""
        if isinstance(input_data, dict):
            return input_data
        if isinstance(input_data, str):
            return {'text': input_data}
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess emotion model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            emotions = model_output.get('emotions', {})
            valence = model_output.get('valence', 0.0)
            arousal = model_output.get('arousal', 0.5)
        else:
            emotions = {}
            valence = 0.0
            arousal = 0.5

        self._current_emotions = emotions
        self._valence = valence
        self._emotional_arousal = arousal

        # Find dominant emotion
        dominant = max(emotions.items(), key=lambda x: x[1]) if emotions else ('neutral', 1.0)

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'emotions': emotions,
            'dominant_emotion': dominant[0],
            'dominant_score': dominant[1],
            'valence': valence,
            'arousal': arousal,
            'state': self.get_state(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform emotion inference."""
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
            logger.error(f"Emotion inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate emotion input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get emotion input specification."""
        return {
            'type': 'emotional_input',
            'modalities': ['text', 'audio_features', 'visual_features'],
            'max_length': 512,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get emotion output specification."""
        return {
            'type': 'emotion_classification',
            'categories': len(self.EMOTIONS),
            'includes_valence_arousal': True,
            'valence_range': [-1.0, 1.0],
            'arousal_range': [0.0, 1.0],
        }

    def get_current_emotions(self) -> Dict[str, float]:
        """Get current emotion state."""
        return dict(self._current_emotions)

    def get_valence_arousal(self) -> tuple:
        """Get current valence and arousal."""
        return (self._valence, self._emotional_arousal)

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'emotions': {'neutral': 1.0},
            'valence': 0.0,
            'arousal': 0.5,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'emotions': {'neutral': 1.0},
            'dominant_emotion': 'neutral',
            'dominant_score': 1.0,
            'valence': 0.0,
            'arousal': 0.5,
            'state': {},
            'mock': True,
        }
