"""
Executive Adapter - Form 12: Narrative/Executive Consciousness
Manages executive functions and decision-making.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import CognitiveAdapter

logger = logging.getLogger(__name__)


class ExecutiveAdapter(CognitiveAdapter):
    """
    Adapter for Form 12: Narrative/Executive Consciousness.

    Manages executive functions including planning, decision-making,
    and goal management using small language models.
    """

    FORM_ID = "12-narrative-consciousness"
    NAME = "Executive Function Consciousness"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME)

        # Executive state
        self._current_goals: List[Dict[str, Any]] = []
        self._active_plan: Optional[Dict[str, Any]] = None
        self._decision_history: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess executive input."""
        if isinstance(input_data, dict):
            return input_data
        return {'task': str(input_data)}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess executive model output."""
        self._record_inference()

        if isinstance(model_output, dict):
            decision = model_output.get('decision', None)
            reasoning = model_output.get('reasoning_trace', [])
            plan = model_output.get('plan', None)
        else:
            decision = str(model_output) if model_output else None
            reasoning = []
            plan = None

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': decision,
            'reasoning_trace': reasoning,
            'plan': plan,
            'current_goals': self._current_goals,
            'state': self.get_state(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform executive inference."""
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
            logger.error(f"Executive inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate executive input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get executive input specification."""
        return {
            'type': 'task_context',
            'max_tokens': 4096,
            'supports_goals': True,
            'supports_constraints': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get executive output specification."""
        return {
            'type': 'executive_decision',
            'includes_reasoning_trace': True,
            'includes_plan': True,
            'includes_confidence': True,
        }

    async def set_goal(self, goal: Dict[str, Any]) -> None:
        """Set a new goal."""
        goal['created_at'] = datetime.now(timezone.utc).isoformat()
        self._current_goals.append(goal)

    async def clear_goals(self) -> None:
        """Clear all goals."""
        self._current_goals.clear()

    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision given context."""
        result = await self.inference(context)
        self._decision_history.append({
            'context': context,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        return result

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'decision': 'continue',
            'reasoning_trace': ['Analyzed input', 'No action required'],
            'plan': None,
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': 'continue',
            'reasoning_trace': [],
            'plan': None,
            'current_goals': [],
            'state': {},
            'mock': True,
        }
