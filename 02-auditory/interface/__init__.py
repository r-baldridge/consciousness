"""
Form 02: Auditory Consciousness Interface

Primary auditory processing system for consciousness.
"""

from .auditory_consciousness_interface import (
    # Enums
    SoundCategory,
    FrequencyBand,
    AuditoryScene,
    SpeechContent,
    SpatialDirection,
    # Input dataclasses
    SpectralData,
    AuditoryInput,
    # Output dataclasses
    SoundIdentification,
    SpatialLocation,
    SpeechAnalysis,
    AuditorySceneAnalysis,
    AuditoryOutput,
    # Main interface
    AuditoryConsciousnessInterface,
    # Convenience functions
    create_auditory_interface,
    create_simple_auditory_input,
)

__all__ = [
    # Enums
    "SoundCategory",
    "FrequencyBand",
    "AuditoryScene",
    "SpeechContent",
    "SpatialDirection",
    # Input dataclasses
    "SpectralData",
    "AuditoryInput",
    # Output dataclasses
    "SoundIdentification",
    "SpatialLocation",
    "SpeechAnalysis",
    "AuditorySceneAnalysis",
    "AuditoryOutput",
    # Main interface
    "AuditoryConsciousnessInterface",
    # Convenience functions
    "create_auditory_interface",
    "create_simple_auditory_input",
]
