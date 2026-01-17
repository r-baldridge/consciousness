"""
Specialized Adapters - Forms 18-27
Consolidated implementation for specialized consciousness forms.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import SpecializedAdapter

logger = logging.getLogger(__name__)


class PrimaryAdapter(SpecializedAdapter):
    """
    Adapter for Form 18: Primary/Unified Sensory Consciousness.
    Integrates all sensory modalities into unified percept.
    """

    FORM_ID = "18-primary-consciousness"
    NAME = "Primary Consciousness"
    SPECIALIZATION = "unified_sensory"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._unified_percept: Dict[str, Any] = {}

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        if isinstance(model_output, dict):
            self._unified_percept = model_output.get('unified', {})
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unified_percept': self._unified_percept,
            'binding_complete': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess({'unified': processed})

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'multimodal_sensory'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'unified_percept'}


class ReflectiveAdapter(SpecializedAdapter):
    """
    Adapter for Form 19: Reflective/Self-Aware Consciousness.
    Enables self-reflection and meta-awareness.
    """

    FORM_ID = "19-reflective-consciousness"
    NAME = "Reflective Consciousness"
    SPECIALIZATION = "self_reflection"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._self_model: Dict[str, Any] = {}

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'self_model': self._self_model,
            'reflection': model_output if isinstance(model_output, dict) else {},
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'self_model'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'reflective_judgment'}


class SocialAdapter(SpecializedAdapter):
    """
    Adapter for Form 20: Collective/Social Consciousness.
    Models shared attention and social coordination.
    """

    FORM_ID = "20-collective-consciousness"
    NAME = "Social Consciousness"
    SPECIALIZATION = "collective"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._social_state: Dict[str, Any] = {}

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'social_state': model_output if isinstance(model_output, dict) else {},
            'shared_attention': None,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'social_graph'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'collective_state'}


class ArtificialAdapter(SpecializedAdapter):
    """
    Adapter for Form 21: Artificial/Meta-Controller Consciousness.
    System-level meta-controller.
    """

    FORM_ID = "21-artificial-consciousness"
    NAME = "Artificial Consciousness"
    SPECIALIZATION = "meta_control"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'control_signals': model_output if isinstance(model_output, dict) else {},
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'system_metrics'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'system_control'}


class DreamAdapter(SpecializedAdapter):
    """
    Adapter for Form 22: Dream State Consciousness.
    Generates dream content from memory traces.
    """

    FORM_ID = "22-dream-consciousness"
    NAME = "Dream Consciousness"
    SPECIALIZATION = "dream_state"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._dream_active = False

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'dream_content': model_output if isinstance(model_output, dict) else {},
            'dream_active': self._dream_active,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'memory_traces'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'dream_narrative'}

    def is_state_active(self) -> bool:
        return self._dream_active


class MeditationAdapter(SpecializedAdapter):
    """
    Adapter for Form 23: Meditative State Consciousness.
    Classifies meditation states.
    """

    FORM_ID = "23-lucid-dream"
    NAME = "Meditation Consciousness"
    SPECIALIZATION = "meditation"

    STATES = ['focused', 'open_awareness', 'non_dual', 'baseline']

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._meditation_state = 'baseline'

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        state = 'baseline'
        if isinstance(model_output, dict):
            state = model_output.get('state', 'baseline')
        self._meditation_state = state
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'meditation_state': state,
            'depth': model_output.get('depth', 0.0) if isinstance(model_output, dict) else 0.0,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'neural_oscillations'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'meditation_state', 'states': self.STATES}


class FlowAdapter(SpecializedAdapter):
    """
    Adapter for Form 24: Flow State Consciousness.
    Detects optimal performance states.
    """

    FORM_ID = "24-locked-in"
    NAME = "Flow State Consciousness"
    SPECIALIZATION = "flow"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._flow_active = False
        self._flow_level = 0.0

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        flow_level = 0.0
        if isinstance(model_output, dict):
            flow_level = model_output.get('flow_level', 0.0)
        self._flow_level = flow_level
        self._flow_active = flow_level > 0.7
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'flow_active': self._flow_active,
            'flow_level': flow_level,
            'challenge_skill_balance': model_output.get('balance', 0.5) if isinstance(model_output, dict) else 0.5,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'performance_metrics'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'flow_state'}

    def is_state_active(self) -> bool:
        return self._flow_active


class MysticalAdapter(SpecializedAdapter):
    """
    Adapter for Form 25: Mystical/Peak Experience Consciousness.
    Detects mystical/transcendent experiences.
    """

    FORM_ID = "25-blindsight"
    NAME = "Mystical Experience Consciousness"
    SPECIALIZATION = "mystical"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._unity_experience = False

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'mystical_indicators': model_output if isinstance(model_output, dict) else {},
            'unity_experience': self._unity_experience,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'unified_experience'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'mystical_state'}


class ThresholdAdapter(SpecializedAdapter):
    """
    Adapter for Form 26: Near-Death/Threshold Consciousness.
    Manages threshold/minimal consciousness states.
    """

    FORM_ID = "26-split-brain"
    NAME = "Threshold Consciousness"
    SPECIALIZATION = "threshold"

    STATES = ['normal', 'transition', 'minimal', 'restored']

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._threshold_state = 'normal'

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        state = 'normal'
        if isinstance(model_output, dict):
            state = model_output.get('state', 'normal')
        self._threshold_state = state
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'threshold_state': state,
            'resource_level': model_output.get('resources', 1.0) if isinstance(model_output, dict) else 1.0,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'vital_signals'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'threshold_state', 'states': self.STATES}


class AlteredAdapter(SpecializedAdapter):
    """
    Adapter for Form 27: Altered State Consciousness.
    Handles non-ordinary states of consciousness.
    """

    FORM_ID = "27-altered-state"
    NAME = "Altered State Consciousness"
    SPECIALIZATION = "altered"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._altered_state_active = False
        self._state_type = 'normal'

    async def preprocess(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data
        return {'input': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        self._record_inference()
        state_type = 'normal'
        if isinstance(model_output, dict):
            state_type = model_output.get('state_type', 'normal')
        self._state_type = state_type
        self._altered_state_active = state_type != 'normal'
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'altered_state_active': self._altered_state_active,
            'state_type': state_type,
            'transition_dynamics': model_output.get('dynamics', {}) if isinstance(model_output, dict) else {},
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        processed = await self.preprocess(input_data)
        return await self.postprocess(processed)

    def validate_input(self, input_data: Any) -> bool:
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        return {'type': 'consciousness_markers'}

    def get_output_spec(self) -> Dict[str, Any]:
        return {'type': 'altered_state'}

    def is_state_active(self) -> bool:
        return self._altered_state_active
