"""
Global Workspace Adapter - Form 14: Global Workspace Theory
Critical form for conscious content broadcasting.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from ..base_adapter import TheoreticalAdapter, GlobalWorkspaceAdapterInterface

logger = logging.getLogger(__name__)


class GlobalWorkspaceAdapter(TheoreticalAdapter, GlobalWorkspaceAdapterInterface):
    """
    Adapter for Form 14: Global Workspace Theory (GWT).

    CRITICAL FORM - Must always be loaded.

    Manages the global workspace with 7 slots for conscious content,
    implementing competition and broadcasting mechanisms.
    """

    FORM_ID = "14-global-workspace"
    NAME = "Global Workspace Consciousness"
    THEORY = "Global Workspace Theory"

    # Classic 7+/-2 capacity
    WORKSPACE_CAPACITY = 7

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.THEORY)

        # Workspace state
        self._workspace_slots: List[Dict[str, Any]] = []
        self._competition_queue: List[Dict[str, Any]] = []
        self._broadcast_history: List[Dict[str, Any]] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess content for workspace competition."""
        if isinstance(input_data, dict):
            return input_data
        return {'content': input_data}

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess workspace state."""
        self._record_inference()

        if isinstance(model_output, dict):
            contents = model_output.get('contents', [])
            broadcast = model_output.get('broadcast_signals', {})
            competition = model_output.get('competition_results', [])
        else:
            contents = self._workspace_slots
            broadcast = {}
            competition = []

        self._workspace_slots = contents[:self.WORKSPACE_CAPACITY]

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'workspace_contents': self._workspace_slots,
            'slot_count': len(self._workspace_slots),
            'capacity': self.WORKSPACE_CAPACITY,
            'broadcast_signals': broadcast,
            'competition_results': competition,
            'theory_metrics': self.get_theory_metrics(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Perform workspace processing."""
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
            logger.error(f"Global Workspace inference error: {e}")
            raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate workspace input."""
        return True

    def get_input_spec(self) -> Dict[str, Any]:
        """Get workspace input specification."""
        return {
            'type': 'consciousness_candidates',
            'max_candidates': 100,
            'from_all_modules': True,
            'includes_salience': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get workspace output specification."""
        return {
            'type': 'workspace_state',
            'slots': self.WORKSPACE_CAPACITY,
            'includes_broadcast_signals': True,
            'includes_competition_results': True,
        }

    # GlobalWorkspaceAdapterInterface implementation

    async def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state."""
        return {
            'contents': self._workspace_slots,
            'slot_count': len(self._workspace_slots),
            'capacity': self.WORKSPACE_CAPACITY,
            'queue_size': len(self._competition_queue),
        }

    async def submit_content(self, content: Dict[str, Any]) -> bool:
        """Submit content for workspace competition."""
        content['submitted_at'] = datetime.now(timezone.utc).isoformat()
        content['salience'] = content.get('salience', 0.5)

        self._competition_queue.append(content)

        # Run competition if queue is full
        if len(self._competition_queue) >= 5:
            await self._run_competition()

        return True

    async def broadcast(self, content: Dict[str, Any]) -> None:
        """Broadcast content to all forms."""
        broadcast_record = {
            'content': content,
            'broadcast_at': datetime.now(timezone.utc).isoformat(),
            'reach': 26,  # All other forms
        }
        self._broadcast_history.append(broadcast_record)

        # Keep limited history
        if len(self._broadcast_history) > 100:
            self._broadcast_history = self._broadcast_history[-100:]

    async def _run_competition(self) -> None:
        """Run workspace competition among queued content."""
        if not self._competition_queue:
            return

        # Sort by salience
        self._competition_queue.sort(key=lambda x: x.get('salience', 0), reverse=True)

        # Take winners (up to capacity)
        available_slots = self.WORKSPACE_CAPACITY - len(self._workspace_slots)
        if available_slots > 0:
            winners = self._competition_queue[:available_slots]
            for winner in winners:
                winner['entered_workspace'] = datetime.now(timezone.utc).isoformat()
                self._workspace_slots.append(winner)

        # Clear losers from queue
        self._competition_queue.clear()

        # Expire old workspace contents
        await self._expire_old_contents()

    async def _expire_old_contents(self) -> None:
        """Remove old content from workspace."""
        # Keep only recent content (simplified expiration)
        if len(self._workspace_slots) > self.WORKSPACE_CAPACITY:
            self._workspace_slots = self._workspace_slots[-self.WORKSPACE_CAPACITY:]

    def _mock_model_output(self, processed_input: Any) -> Dict[str, Any]:
        """Generate mock output."""
        return {
            'contents': self._workspace_slots,
            'broadcast_signals': {'global': True},
            'competition_results': [],
        }

    def _mock_inference(self, input_data: Any) -> Dict[str, Any]:
        """Generate mock inference result."""
        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'workspace_contents': self._workspace_slots,
            'slot_count': len(self._workspace_slots),
            'capacity': self.WORKSPACE_CAPACITY,
            'broadcast_signals': {},
            'competition_results': [],
            'theory_metrics': self.get_theory_metrics(),
            'mock': True,
        }

    def get_theory_metrics(self) -> Dict[str, Any]:
        """Get GWT-specific metrics."""
        return {
            'theory': self.THEORY,
            'slot_utilization': len(self._workspace_slots) / self.WORKSPACE_CAPACITY,
            'queue_size': len(self._competition_queue),
            'recent_broadcasts': len(self._broadcast_history),
        }
