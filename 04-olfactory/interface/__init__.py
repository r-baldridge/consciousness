"""
Form 04: Olfactory Consciousness Interface

Smell processing system for consciousness.
"""

from .olfactory_consciousness_interface import (
    # Enums
    OdorCategory,
    OlfactoryQuality,
    OdorIntensityLevel,
    HedonicValence,
    OlfactoryAdaptationState,
    # Input dataclasses
    ChemicalFeatures,
    OlfactoryInput,
    # Output dataclasses
    OdorIdentification,
    HedonicEvaluation,
    MemoryAssociation,
    OlfactoryOutput,
    # Main interface
    OlfactoryConsciousnessInterface,
    # Convenience functions
    create_olfactory_interface,
    create_simple_olfactory_input,
)

__all__ = [
    # Enums
    "OdorCategory",
    "OlfactoryQuality",
    "OdorIntensityLevel",
    "HedonicValence",
    "OlfactoryAdaptationState",
    # Input dataclasses
    "ChemicalFeatures",
    "OlfactoryInput",
    # Output dataclasses
    "OdorIdentification",
    "HedonicEvaluation",
    "MemoryAssociation",
    "OlfactoryOutput",
    # Main interface
    "OlfactoryConsciousnessInterface",
    # Convenience functions
    "create_olfactory_interface",
    "create_simple_olfactory_input",
]
