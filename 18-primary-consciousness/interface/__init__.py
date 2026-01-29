"""
Form 18: Primary Consciousness (Edelman) Interface

Sensory-driven awareness: perceptual categorization, value tagging,
scene construction, and the remembered present.
"""

from .primary_consciousness_interface import (
    # Enums
    PrimaryAwarenessLevel,
    SensoryBoundState,
    PerceptualCategory,
    ValueAssignment,
    SceneCoherence,
    # Input dataclasses
    SensoryChannelData,
    PrimarySensoryInput,
    # Output dataclasses
    CategorizedPercept,
    RememberedPresent,
    PrimaryConsciousnessOutput,
    # Interface
    PrimaryConsciousnessInterface,
    # Convenience
    create_primary_consciousness_interface,
)

__all__ = [
    "PrimaryAwarenessLevel",
    "SensoryBoundState",
    "PerceptualCategory",
    "ValueAssignment",
    "SceneCoherence",
    "SensoryChannelData",
    "PrimarySensoryInput",
    "CategorizedPercept",
    "RememberedPresent",
    "PrimaryConsciousnessOutput",
    "PrimaryConsciousnessInterface",
    "create_primary_consciousness_interface",
]
