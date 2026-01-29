"""
Form 22: Dream Consciousness Interface

Dream states, sleep stages, dream generation, content analysis,
and bizarreness computation.
"""

from .dream_consciousness_interface import (
    # Enums
    SleepStage,
    DreamType,
    DreamEmotion,
    BizarrenessSource,
    DreamGenerationModel,
    # Input dataclasses
    SleepStateInput,
    RecentMemory,
    DreamInput,
    # Output dataclasses
    DreamElement,
    DreamOutput,
    # Interface
    DreamConsciousnessInterface,
    # Convenience
    create_dream_consciousness_interface,
)

__all__ = [
    "SleepStage",
    "DreamType",
    "DreamEmotion",
    "BizarrenessSource",
    "DreamGenerationModel",
    "SleepStateInput",
    "RecentMemory",
    "DreamInput",
    "DreamElement",
    "DreamOutput",
    "DreamConsciousnessInterface",
    "create_dream_consciousness_interface",
]
