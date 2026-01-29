"""
Form 03: Somatosensory Consciousness Interface

Body sensation processing system for consciousness.
"""

from .somatosensory_consciousness_interface import (
    # Enums
    TouchType,
    PainType,
    BodyRegion,
    ProprioceptiveChannel,
    BodySchemaState,
    # Input dataclasses
    TouchInput,
    PainInput,
    ProprioceptiveInput,
    SomatosensoryInput,
    # Output dataclasses
    TouchClassification,
    PainAssessment,
    BodySchema,
    SomatosensoryOutput,
    # Main interface
    SomatosensoryConsciousnessInterface,
    # Convenience functions
    create_somatosensory_interface,
    create_simple_touch_input,
)

__all__ = [
    # Enums
    "TouchType",
    "PainType",
    "BodyRegion",
    "ProprioceptiveChannel",
    "BodySchemaState",
    # Input dataclasses
    "TouchInput",
    "PainInput",
    "ProprioceptiveInput",
    "SomatosensoryInput",
    # Output dataclasses
    "TouchClassification",
    "PainAssessment",
    "BodySchema",
    "SomatosensoryOutput",
    # Main interface
    "SomatosensoryConsciousnessInterface",
    # Convenience functions
    "create_somatosensory_interface",
    "create_simple_touch_input",
]
