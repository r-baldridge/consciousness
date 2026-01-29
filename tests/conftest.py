"""
Shared Test Fixtures for Cross-Form Integration Tests

Provides unittest-compatible fixtures, helper functions, and setup utilities
for testing cross-form interactions in the consciousness architecture.

Usage:
    Import fixtures and helpers directly:
        from conftest import create_all_form_interfaces, ArousalGatedSystem, ...
"""

import asyncio
import logging
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure the project root is on sys.path so form packages can be imported
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import registry utilities
from tests.form_registry_utils import (
    discover_forms,
    load_form_interface,
    extract_form_metadata,
    instantiate_form_interface,
    get_all_form_ids,
    get_form_directory,
    FORM_DIRECTORY_MAP,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: Run async code from synchronous unittest methods
# ============================================================================

def run_async(coro):
    """
    Run an async coroutine from synchronous test code.

    Creates a new event loop, runs the coroutine, and cleans up.
    Use this in unittest methods that need to await async form interfaces.

    Args:
        coro: An awaitable coroutine.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# Form Interface Cache
# ============================================================================

_loaded_modules: Dict[str, Any] = {}
_loaded_instances: Dict[str, Any] = {}


def get_loaded_module(form_id: str) -> Optional[Any]:
    """
    Get a cached loaded module for a form, or load it if not cached.

    Args:
        form_id: The form identifier.

    Returns:
        The loaded module, or None if loading fails.
    """
    if form_id in _loaded_modules:
        return _loaded_modules[form_id]

    discovered = discover_forms()
    if form_id not in discovered:
        return None

    try:
        module = load_form_interface(discovered[form_id])
        _loaded_modules[form_id] = module
        return module
    except Exception as exc:
        logger.warning("Could not load module for %s: %s", form_id, exc)
        return None


def get_loaded_instance(form_id: str) -> Optional[Any]:
    """
    Get a cached interface instance for a form, or create one if not cached.

    Args:
        form_id: The form identifier.

    Returns:
        An interface instance, or None if instantiation fails.
    """
    if form_id in _loaded_instances:
        return _loaded_instances[form_id]

    module = get_loaded_module(form_id)
    if module is None:
        return None

    instance = instantiate_form_interface(module)
    if instance is not None:
        _loaded_instances[form_id] = instance
    return instance


# ============================================================================
# Fixture: Create All Form Interfaces
# ============================================================================

def create_all_form_interfaces() -> Dict[str, Any]:
    """
    Discover and instantiate all available form interfaces.

    Returns:
        Dictionary mapping form_id to interface instance for every form
        that has a loadable Python interface.
    """
    interfaces: Dict[str, Any] = {}
    discovered = discover_forms()

    for form_id, file_path in discovered.items():
        try:
            module = load_form_interface(file_path)
            instance = instantiate_form_interface(module)
            if instance is not None:
                interfaces[form_id] = instance
        except Exception as exc:
            logger.warning("Skipping %s: %s", form_id, exc)

    return interfaces


# ============================================================================
# Fixture: Arousal-Gated System
# ============================================================================

class ArousalGatedSystem:
    """
    A test harness that sets up an arousal interface with registered forms,
    simulating the gating mechanism that controls all other forms.

    This fixture creates the arousal interface (Form 08) and registers
    a configurable set of other forms with their resource demands.

    Attributes:
        arousal: The ArousalConsciousnessInterface instance.
        registered_forms: Set of form IDs registered with the arousal system.
        form_demands: Dict mapping form IDs to their resource demands.
    """

    def __init__(
        self,
        forms_to_register: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the arousal-gated system.

        Args:
            forms_to_register: Optional dict of form_id -> demand_level.
                Defaults to a standard set of sensory + cognitive forms.
        """
        self.arousal = None
        self.registered_forms: Set[str] = set()
        self.form_demands: Dict[str, float] = {}

        # Default form registration if none provided
        if forms_to_register is None:
            forms_to_register = {
                "01-visual": 0.20,
                "02-auditory": 0.15,
                "03-somatosensory": 0.10,
                "07-emotional": 0.18,
                "09-perceptual": 0.15,
                "11-meta-consciousness": 0.12,
                "13-integrated-information": 0.10,
                "14-global-workspace": 0.10,
            }

        self._forms_to_register = forms_to_register

    def setup(self) -> "ArousalGatedSystem":
        """
        Create and configure the arousal system.

        Returns:
            self, for method chaining.
        """
        # Add arousal interface path
        arousal_dir = get_form_directory("08-arousal")
        interface_dir = str(arousal_dir / "interface")
        arousal_root = str(arousal_dir)

        if interface_dir not in sys.path:
            sys.path.insert(0, interface_dir)
        if arousal_root not in sys.path:
            sys.path.insert(0, arousal_root)

        from interface import (
            ArousalConsciousnessInterface,
            create_arousal_interface,
        )

        self.arousal = create_arousal_interface()

        # Register forms
        for form_id, demand in self._forms_to_register.items():
            self.arousal.register_form(form_id, demand)
            self.registered_forms.add(form_id)
            self.form_demands[form_id] = demand

        return self

    def get_arousal_interface(self):
        """Return the arousal interface instance."""
        return self.arousal

    def is_form_allowed(self, form_id: str) -> bool:
        """Check if a form is allowed under current arousal gating."""
        if self.arousal is None:
            return False
        return self.arousal.is_form_allowed(form_id)

    def get_gate_for_form(self, form_id: str) -> float:
        """Get the gating value for a specific form."""
        if self.arousal is None:
            return 0.0
        return self.arousal.get_gate_for_form(form_id)


# ============================================================================
# Test Data Helpers
# ============================================================================

def create_sensory_stimulus(
    modality: str = "visual",
    intensity: float = 0.5,
    salience: float = 0.5,
) -> Dict[str, Any]:
    """
    Create a test sensory stimulus dictionary.

    Args:
        modality: Sensory modality name.
        intensity: Stimulus intensity (0-1).
        salience: Stimulus salience (0-1).

    Returns:
        Dictionary representing a sensory stimulus.
    """
    return {
        "modality": modality,
        "intensity": max(0.0, min(1.0, intensity)),
        "salience": max(0.0, min(1.0, salience)),
        "timestamp": 0.0,
        "source": "test",
        "features": {
            "complexity": intensity * 0.8,
            "novelty": salience * 0.6,
            "change_rate": 0.3,
        },
    }


def create_emotional_state(
    valence: float = 0.0,
    arousal_component: float = 0.5,
    dominant_emotion: str = "neutral",
) -> Dict[str, Any]:
    """
    Create a test emotional state dictionary.

    Args:
        valence: Emotional valence (-1 to 1).
        arousal_component: Arousal contribution (0-1).
        dominant_emotion: Name of dominant emotion.

    Returns:
        Dictionary representing an emotional state.
    """
    return {
        "valence": max(-1.0, min(1.0, valence)),
        "arousal_component": max(0.0, min(1.0, arousal_component)),
        "dominant_emotion": dominant_emotion,
        "intensity": abs(valence) * 0.8 + arousal_component * 0.2,
        "stability": 0.7,
    }


def create_meta_cognitive_report(
    target_form: str = "09-perceptual",
    confidence: float = 0.7,
    monitoring_quality: float = 0.8,
) -> Dict[str, Any]:
    """
    Create a test meta-cognitive monitoring report.

    Args:
        target_form: The form being monitored.
        confidence: Confidence in the monitoring assessment.
        monitoring_quality: Quality of the meta-cognitive monitoring.

    Returns:
        Dictionary representing a meta-cognitive report.
    """
    return {
        "target_form": target_form,
        "confidence": max(0.0, min(1.0, confidence)),
        "monitoring_quality": max(0.0, min(1.0, monitoring_quality)),
        "assessment": "nominal" if confidence > 0.5 else "uncertain",
        "introspective_accuracy": confidence * 0.9,
        "error_detection_active": confidence > 0.3,
    }


def create_iit_measurement(
    phi_value: float = 0.5,
    num_elements: int = 10,
    integration_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Create a test IIT (Integrated Information Theory) measurement.

    Args:
        phi_value: The phi value (0-1).
        num_elements: Number of elements in the measured system.
        integration_level: Qualitative integration level.

    Returns:
        Dictionary representing an IIT measurement.
    """
    return {
        "phi": max(0.0, min(1.0, phi_value)),
        "num_elements": num_elements,
        "integration_level": integration_level,
        "main_complex": list(range(min(num_elements, 5))),
        "partition_info": {
            "minimum_information_partition": 0.3,
            "conceptual_structure": phi_value * 0.8,
        },
    }


def create_gwt_broadcast(
    content: str = "test_stimulus",
    access_strength: float = 0.6,
    broadcasting: bool = True,
) -> Dict[str, Any]:
    """
    Create a test GWT (Global Workspace Theory) broadcast.

    Args:
        content: Content being broadcast.
        access_strength: Strength of conscious access.
        broadcasting: Whether the workspace is actively broadcasting.

    Returns:
        Dictionary representing a GWT broadcast.
    """
    return {
        "content": content,
        "access_strength": max(0.0, min(1.0, access_strength)),
        "broadcasting": broadcasting,
        "workspace_capacity": 0.8,
        "competing_coalitions": 3,
        "winning_coalition": content if broadcasting else None,
        "broadcast_reach": ["01-visual", "02-auditory", "07-emotional"]
        if broadcasting
        else [],
    }


def create_message(
    source_form: str,
    target_form: str,
    message_type: str = "data",
    payload: Optional[Dict[str, Any]] = None,
    priority: float = 0.5,
) -> Dict[str, Any]:
    """
    Create a test inter-form message.

    Args:
        source_form: Sending form ID.
        target_form: Receiving form ID.
        message_type: Type of message (data, query, control, broadcast).
        payload: Message payload.
        priority: Message priority (0-1).

    Returns:
        Dictionary representing an inter-form message.
    """
    return {
        "source": source_form,
        "target": target_form,
        "type": message_type,
        "payload": payload or {},
        "priority": max(0.0, min(1.0, priority)),
        "timestamp": 0.0,
        "requires_ack": message_type in ("query", "control"),
    }
