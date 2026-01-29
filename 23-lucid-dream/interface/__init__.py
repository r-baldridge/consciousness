"""
Form 23: Lucid Dream Consciousness Interface

Models awareness within the dream state, including lucidity detection,
dream control mechanisms, reality checking, and dream stabilization.
"""

from .lucid_dream_interface import (
    # Enums
    LucidityLevel,
    DreamControl,
    LucidTrigger,
    DreamStability,
    DreamPhase,
    # Input dataclasses
    LucidDreamInput,
    DreamControlInput,
    # Output dataclasses
    LucidDreamOutput,
    DreamControlOutput,
    RealityCheckResult,
    DreamStateSnapshot,
    # Interface
    LucidDreamInterface,
    # Convenience
    create_lucid_dream_interface,
)

__all__ = [
    # Enums
    "LucidityLevel",
    "DreamControl",
    "LucidTrigger",
    "DreamStability",
    "DreamPhase",
    # Input dataclasses
    "LucidDreamInput",
    "DreamControlInput",
    # Output dataclasses
    "LucidDreamOutput",
    "DreamControlOutput",
    "RealityCheckResult",
    "DreamStateSnapshot",
    # Interface
    "LucidDreamInterface",
    # Convenience
    "create_lucid_dream_interface",
]
