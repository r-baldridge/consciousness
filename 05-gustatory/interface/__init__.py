"""
Form 05: Gustatory Consciousness Interface

Taste processing system for consciousness.
"""

from .gustatory_consciousness_interface import (
    # Enums
    TasteModality,
    FlavorProfile,
    TextureQuality,
    PalatabilityLevel,
    AppetiteState,
    # Input dataclasses
    TasteReceptorData,
    GustatoryInput,
    # Output dataclasses
    TasteIdentification,
    FlavorIntegration,
    PalatabilityAssessment,
    GustatoryOutput,
    # Main interface
    GustatoryConsciousnessInterface,
    # Convenience functions
    create_gustatory_interface,
    create_simple_gustatory_input,
)

__all__ = [
    # Enums
    "TasteModality",
    "FlavorProfile",
    "TextureQuality",
    "PalatabilityLevel",
    "AppetiteState",
    # Input dataclasses
    "TasteReceptorData",
    "GustatoryInput",
    # Output dataclasses
    "TasteIdentification",
    "FlavorIntegration",
    "PalatabilityAssessment",
    "GustatoryOutput",
    # Main interface
    "GustatoryConsciousnessInterface",
    # Convenience functions
    "create_gustatory_interface",
    "create_simple_gustatory_input",
]
