"""
Form 06: Interoceptive Consciousness Interface

Internal body state processing system for consciousness.
"""

from .interoceptive_consciousness_interface import (
    # Enums
    InteroceptiveChannel,
    BodySystem,
    HomeostaticNeed,
    BodyStateCategory,
    InteroceptiveAccuracy,
    # Input dataclasses
    OrganSignal,
    HomeostaticData,
    InteroceptiveInput,
    # Output dataclasses
    BodyStateAssessment,
    HomeostaticNeedsReport,
    EmotionalGrounding,
    InteroceptiveOutput,
    # Main interface
    InteroceptiveConsciousnessInterface,
    # Convenience functions
    create_interoceptive_interface,
    create_simple_interoceptive_input,
)

__all__ = [
    # Enums
    "InteroceptiveChannel",
    "BodySystem",
    "HomeostaticNeed",
    "BodyStateCategory",
    "InteroceptiveAccuracy",
    # Input dataclasses
    "OrganSignal",
    "HomeostaticData",
    "InteroceptiveInput",
    # Output dataclasses
    "BodyStateAssessment",
    "HomeostaticNeedsReport",
    "EmotionalGrounding",
    "InteroceptiveOutput",
    # Main interface
    "InteroceptiveConsciousnessInterface",
    # Convenience functions
    "create_interoceptive_interface",
    "create_simple_interoceptive_input",
]
