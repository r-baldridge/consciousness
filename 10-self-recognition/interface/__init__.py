"""
Form 10: Self-Recognition Consciousness Interface

Self-awareness, self-model, body ownership, and sense of agency.
"""

from .self_recognition_interface import (
    # Enums
    SelfAspect,
    AgencyLevel,
    OwnershipState,
    SelfBoundaryType,
    SelfRecognitionMode,
    # Input dataclasses
    BodySignalInput,
    SocialContextInput,
    ActionFeedback,
    SelfInput,
    # Output dataclasses
    SelfModelOutput,
    AgencyAssessment,
    SelfOutput,
    SelfSystemStatus,
    # Engines
    SelfModelEngine,
    AgencyEngine,
    # Main interface
    SelfRecognitionInterface,
    # Convenience
    create_self_recognition_interface,
)

__all__ = [
    "SelfAspect",
    "AgencyLevel",
    "OwnershipState",
    "SelfBoundaryType",
    "SelfRecognitionMode",
    "BodySignalInput",
    "SocialContextInput",
    "ActionFeedback",
    "SelfInput",
    "SelfModelOutput",
    "AgencyAssessment",
    "SelfOutput",
    "SelfSystemStatus",
    "SelfModelEngine",
    "AgencyEngine",
    "SelfRecognitionInterface",
    "create_self_recognition_interface",
]
