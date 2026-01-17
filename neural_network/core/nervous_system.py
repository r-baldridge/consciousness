"""
Nervous System Coordinator - The central 'nervous system' connecting all forms
Part of the Neural Network module for the Consciousness system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .model_registry import ModelRegistry, Priority, LoadedModel, ModelState
from .resource_manager import ResourceManager, ResourceRequest, ArousalState
from .message_bus import MessageBus, FormMessage, MessageType

if TYPE_CHECKING:
    from ..adapters.base_adapter import FormAdapter

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessState:
    """Current state of the consciousness system."""
    arousal_level: float
    arousal_state: ArousalState
    phi_value: float
    global_workspace_contents: List[str]
    active_forms: List[str]
    processing_rate: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'arousal_level': round(self.arousal_level, 3),
            'arousal_state': self.arousal_state.value,
            'phi_value': round(self.phi_value, 4),
            'global_workspace_contents': self.global_workspace_contents,
            'active_forms': self.active_forms,
            'processing_rate': round(self.processing_rate, 2),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class InferenceRequest:
    """Request for model inference."""
    form_id: str
    input_data: Any
    priority: Priority = Priority.NORMAL
    timeout_ms: int = 100
    correlation_id: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of model inference."""
    form_id: str
    output: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None
    correlation_id: Optional[str] = None


class NervousSystem:
    """
    Central coordinator - the 'nervous system' connecting all forms.

    The NervousSystem orchestrates:
    - Model loading and resource allocation
    - Inter-form message routing
    - Arousal-gated processing
    - Global workspace updates
    - Phi (IIT) computation coordination
    """

    # Critical forms that must always be loaded
    CRITICAL_FORMS = ['08-arousal', '13-integrated-information', '14-global-workspace']

    # Processing cycle rate
    CYCLE_RATE_HZ = 20  # 20 cycles per second = 50ms per cycle

    def __init__(
        self,
        model_config_path: str = "config/model_configs.yaml",
        resource_config_path: str = "config/resource_budgets.yaml",
    ):
        """
        Initialize the Nervous System.

        Args:
            model_config_path: Path to model configurations
            resource_config_path: Path to resource budget configurations
        """
        # Resolve paths relative to module directory
        base_path = Path(__file__).parent.parent
        model_path = base_path / model_config_path
        resource_path = base_path / resource_config_path

        # Core components
        self.registry = ModelRegistry(str(model_path))
        self.resources = ResourceManager(str(resource_path))
        self.bus = MessageBus()

        # Form adapters: form_id -> FormAdapter
        self.adapters: Dict[str, "FormAdapter"] = {}

        # State tracking
        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None
        self._current_state = ConsciousnessState(
            arousal_level=0.5,
            arousal_state=ArousalState.ALERT,
            phi_value=0.0,
            global_workspace_contents=[],
            active_forms=[],
            processing_rate=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        # Phi tracking (from Form 13)
        self._phi_value = 0.0

        # Global workspace slots (from Form 14)
        self._workspace_slots: List[Dict[str, Any]] = []
        self._workspace_capacity = 7  # Classic 7+/-2 slots

        # Metrics
        self._cycle_count = 0
        self._inference_count = 0
        self._total_cycle_time_ms = 0.0

        # Callbacks
        self._state_callbacks: List[callable] = []

    def register_adapter(self, form_id: str, adapter: "FormAdapter") -> None:
        """
        Register a form adapter.

        Args:
            form_id: The form ID
            adapter: The adapter instance
        """
        self.adapters[form_id] = adapter
        logger.info(f"Registered adapter for {form_id}")

    def on_state_change(self, callback: callable) -> None:
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    async def initialize(self) -> None:
        """
        Initialize the nervous system and load critical forms.

        This loads the always-on forms:
        - Form 08: Arousal (gates all other forms)
        - Form 13: IIT (consciousness integration metric)
        - Form 14: Global Workspace (central broadcast)
        """
        logger.info("Initializing Nervous System...")

        # Start message bus processing
        await self.bus.start_processing()

        # Load critical forms
        for form_id in self.CRITICAL_FORMS:
            config = self.registry.get_config(form_id)
            if config:
                # Allocate resources
                request = ResourceRequest(
                    form_id=form_id,
                    vram_mb=config.vram_mb,
                    priority=Priority.CRITICAL,
                    preemptible=False,
                )
                allocation = await self.resources.allocate(request)

                if allocation:
                    # Load the model
                    await self.registry.load_model(form_id, Priority.CRITICAL)
                    logger.info(f"Loaded critical form: {form_id}")
                else:
                    logger.error(f"Failed to allocate resources for critical form: {form_id}")
            else:
                logger.warning(f"No config found for critical form: {form_id}")

        # Subscribe critical forms to channels
        await self.bus.subscribe_channel('08-arousal', 'arousal')
        await self.bus.subscribe_channel('14-global-workspace', 'global_workspace')

        logger.info("Nervous System initialized")

    async def start(self) -> None:
        """Start the main processing loop."""
        if self._running:
            return

        self._running = True
        self._cycle_task = asyncio.create_task(self._main_loop())
        logger.info("Nervous System started")

    async def stop(self) -> None:
        """Stop the nervous system."""
        self._running = False

        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        await self.bus.stop_processing()
        logger.info("Nervous System stopped")

    async def _main_loop(self) -> None:
        """Main processing loop - arousal-gated resource allocation."""
        cycle_interval = 1.0 / self.CYCLE_RATE_HZ

        while self._running:
            cycle_start = datetime.now(timezone.utc)

            try:
                await self.process_cycle()
            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")

            # Calculate timing
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            self._total_cycle_time_ms += cycle_duration * 1000
            self._cycle_count += 1

            # Sleep for remaining time
            sleep_time = max(0, cycle_interval - cycle_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def process_cycle(self) -> None:
        """
        Single processing cycle.

        Steps:
        1. Get arousal level from Form 08
        2. Gate resources based on arousal
        3. Process messages
        4. Update global workspace (Form 14)
        5. Compute phi value (Form 13)
        """
        # Step 1: Get arousal level
        arousal_level = await self._get_arousal_level()
        self.resources.set_arousal_level(arousal_level)

        # Step 2: Gate resources based on arousal
        await self._gate_resources(arousal_level)

        # Step 3: Process queued messages
        await self._process_messages()

        # Step 4: Update global workspace
        await self._update_global_workspace()

        # Step 5: Compute phi
        phi_value = await self._compute_phi()
        self._phi_value = phi_value

        # Update current state
        self._current_state = ConsciousnessState(
            arousal_level=arousal_level,
            arousal_state=self.resources.arousal_state,
            phi_value=phi_value,
            global_workspace_contents=[
                slot.get('content_id', '')
                for slot in self._workspace_slots
            ],
            active_forms=list(self.registry.models.keys()),
            processing_rate=self.resources.get_processing_rate(),
            timestamp=datetime.now(timezone.utc),
        )

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                await callback(self._current_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    async def _get_arousal_level(self) -> float:
        """Get current arousal level from Form 08."""
        adapter = self.adapters.get('08-arousal')
        if adapter:
            try:
                result = await adapter.get_arousal_level()
                return result
            except Exception as e:
                logger.error(f"Error getting arousal level: {e}")

        # Default to moderate arousal
        return 0.5

    async def _gate_resources(self, arousal_level: float) -> None:
        """Apply arousal-based gating per Form 08 spec."""
        # Forms to potentially unload based on arousal
        if arousal_level < 0.3:
            # Low arousal - unload non-critical forms
            for form_id, model in list(self.registry.models.items()):
                if form_id not in self.CRITICAL_FORMS:
                    if not self.resources.is_form_allowed(form_id):
                        await self.registry.unload_model(form_id)
                        await self.resources.release(form_id)

        elif arousal_level > 0.9:
            # Hyperaroused - focus on high priority only
            for form_id, model in list(self.registry.models.items()):
                config = self.registry.get_config(form_id)
                if config and config.priority not in [Priority.CRITICAL, Priority.HIGH]:
                    if not self.resources.is_form_allowed(form_id):
                        await self.registry.unload_model(form_id)
                        await self.resources.release(form_id)

    async def _process_messages(self) -> None:
        """Process messages from the bus."""
        # Messages are processed by the bus's internal loop
        # Here we can inject any system-level message handling
        pass

    async def _update_global_workspace(self) -> None:
        """Update global workspace state via Form 14."""
        adapter = self.adapters.get('14-global-workspace')
        if adapter:
            try:
                workspace_state = await adapter.get_workspace_state()
                self._workspace_slots = workspace_state.get('contents', [])

                # Broadcast workspace update
                await self.bus.broadcast_to_channel(
                    source_form='14-global-workspace',
                    channel='global_workspace',
                    message_type=MessageType.WORKSPACE_BROADCAST,
                    body={
                        'workspace_state': workspace_state,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    },
                    priority=Priority.HIGH,
                )
            except Exception as e:
                logger.error(f"Error updating global workspace: {e}")

    async def _compute_phi(self) -> float:
        """Compute phi value via Form 13."""
        adapter = self.adapters.get('13-integrated-information')
        if adapter:
            try:
                phi_result = await adapter.compute_phi()
                return phi_result.get('phi_value', 0.0)
            except Exception as e:
                logger.error(f"Error computing phi: {e}")

        return 0.0

    async def inference(
        self,
        request: InferenceRequest
    ) -> InferenceResult:
        """
        Route inference through appropriate adapter.

        Args:
            request: The inference request

        Returns:
            InferenceResult with output or error
        """
        start_time = datetime.now(timezone.utc)
        form_id = request.form_id

        # Check if form is allowed by arousal gating
        if not self.resources.is_form_allowed(form_id):
            return InferenceResult(
                form_id=form_id,
                output=None,
                latency_ms=0,
                success=False,
                error=f"Form {form_id} gated by arousal state",
                correlation_id=request.correlation_id,
            )

        # Ensure model is loaded
        model = await self.registry.get_model(form_id)
        if not model or model.state != ModelState.LOADED:
            # Try to load the model
            config = self.registry.get_config(form_id)
            if config:
                alloc_request = ResourceRequest(
                    form_id=form_id,
                    vram_mb=config.vram_mb,
                    priority=request.priority,
                    preemptible=config.preemptible,
                )
                allocation = await self.resources.allocate(alloc_request)
                if allocation:
                    model = await self.registry.load_model(form_id, request.priority)

            if not model:
                return InferenceResult(
                    form_id=form_id,
                    output=None,
                    latency_ms=0,
                    success=False,
                    error=f"Failed to load model for {form_id}",
                    correlation_id=request.correlation_id,
                )

        # Get adapter and run inference
        adapter = self.adapters.get(form_id)
        if not adapter:
            return InferenceResult(
                form_id=form_id,
                output=None,
                latency_ms=0,
                success=False,
                error=f"No adapter registered for {form_id}",
                correlation_id=request.correlation_id,
            )

        try:
            # Run inference with timeout
            output = await asyncio.wait_for(
                adapter.inference(request.input_data),
                timeout=request.timeout_ms / 1000,
            )

            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update model stats
            model.record_inference(latency)
            await self.resources.touch(form_id)

            self._inference_count += 1

            return InferenceResult(
                form_id=form_id,
                output=output,
                latency_ms=latency,
                success=True,
                correlation_id=request.correlation_id,
            )

        except asyncio.TimeoutError:
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return InferenceResult(
                form_id=form_id,
                output=None,
                latency_ms=latency,
                success=False,
                error=f"Inference timeout after {request.timeout_ms}ms",
                correlation_id=request.correlation_id,
            )

        except Exception as e:
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            model.record_error(str(e))
            return InferenceResult(
                form_id=form_id,
                output=None,
                latency_ms=latency,
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
            )

    async def load_form(self, form_id: str) -> bool:
        """
        Load a form's model.

        Args:
            form_id: The form to load

        Returns:
            True if successful
        """
        config = self.registry.get_config(form_id)
        if not config:
            return False

        request = ResourceRequest(
            form_id=form_id,
            vram_mb=config.vram_mb,
            priority=config.priority,
            preemptible=config.preemptible,
        )

        allocation = await self.resources.allocate(request)
        if not allocation:
            return False

        model = await self.registry.load_model(form_id)
        return model is not None

    async def unload_form(self, form_id: str) -> bool:
        """
        Unload a form's model.

        Args:
            form_id: The form to unload

        Returns:
            True if successful
        """
        if form_id in self.CRITICAL_FORMS:
            logger.warning(f"Cannot unload critical form {form_id}")
            return False

        await self.resources.release(form_id)
        return await self.registry.unload_model(form_id)

    def get_consciousness_state(self) -> ConsciousnessState:
        """Get the current consciousness state."""
        return self._current_state

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        avg_cycle_time = (
            self._total_cycle_time_ms / self._cycle_count
            if self._cycle_count > 0 else 0
        )

        return {
            'running': self._running,
            'consciousness_state': self._current_state.to_dict(),
            'registry': self.registry.get_status(),
            'resources': self.resources.get_status(),
            'message_bus': self.bus.get_status(),
            'adapters': list(self.adapters.keys()),
            'metrics': {
                'cycle_count': self._cycle_count,
                'inference_count': self._inference_count,
                'avg_cycle_time_ms': round(avg_cycle_time, 2),
                'cycle_rate_hz': self.CYCLE_RATE_HZ,
            },
        }
