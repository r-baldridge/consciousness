"""
Form 09: Perceptual Consciousness Interface

Binding of sensory features into unified percepts and scene representation.
"""

from .perceptual_consciousness_interface import (
    # Enums
    PerceptualBindingType,
    GestaltPrinciple,
    AttentionalMode,
    SensoryChannel,
    PerceptualQuality,
    # Input dataclasses
    SensoryFeature,
    PerceptualInput,
    # Output dataclasses
    BoundPercept,
    SceneRepresentation,
    PerceptualOutput,
    PerceptualSystemStatus,
    # Engines
    FeatureBindingEngine,
    PerceptualOrganizationEngine,
    # Main interface
    PerceptualConsciousnessInterface,
    # Convenience
    create_perceptual_interface,
)

__all__ = [
    "PerceptualBindingType",
    "GestaltPrinciple",
    "AttentionalMode",
    "SensoryChannel",
    "PerceptualQuality",
    "SensoryFeature",
    "PerceptualInput",
    "BoundPercept",
    "SceneRepresentation",
    "PerceptualOutput",
    "PerceptualSystemStatus",
    "FeatureBindingEngine",
    "PerceptualOrganizationEngine",
    "PerceptualConsciousnessInterface",
    "create_perceptual_interface",
]
