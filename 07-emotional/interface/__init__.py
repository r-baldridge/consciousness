"""
Form 07: Emotional Consciousness Interface

Processes emotions, affect, mood, and emotional regulation.
"""

from .emotional_consciousness_interface import (
    # Enums
    EmotionCategory,
    EmotionalValence,
    AffectiveState,
    MoodState,
    EmotionalRegulationStrategy,
    # Input dataclasses
    EmotionalStimulus,
    AppraisalInput,
    BodilySignalInput,
    EmotionalInput,
    # Output dataclasses
    EmotionIdentification,
    EmotionalOutput,
    MoodReport,
    EmotionalSystemStatus,
    # Engines
    EmotionProcessingEngine,
    MoodTrackingEngine,
    EmotionRegulationEngine,
    # Main interface
    EmotionalConsciousnessInterface,
    # Convenience
    create_emotional_interface,
)

__all__ = [
    "EmotionCategory",
    "EmotionalValence",
    "AffectiveState",
    "MoodState",
    "EmotionalRegulationStrategy",
    "EmotionalStimulus",
    "AppraisalInput",
    "BodilySignalInput",
    "EmotionalInput",
    "EmotionIdentification",
    "EmotionalOutput",
    "MoodReport",
    "EmotionalSystemStatus",
    "EmotionProcessingEngine",
    "MoodTrackingEngine",
    "EmotionRegulationEngine",
    "EmotionalConsciousnessInterface",
    "create_emotional_interface",
]
