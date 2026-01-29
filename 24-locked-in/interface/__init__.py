"""
Form 24: Locked-In Consciousness Interface

Models full awareness with minimal motor output, including awareness
detection, intention decoding, and communication channel establishment
for locked-in syndrome states.
"""

from .locked_in_interface import (
    # Enums
    LockedInType,
    CommunicationChannel,
    AwarenessState,
    SignalQuality,
    CognitiveFunction,
    # Input dataclasses
    LockedInInput,
    CommunicationAttemptInput,
    # Output dataclasses
    DecodedIntention,
    AwarenessAssessment,
    CommunicationResult,
    ConsciousnessMonitorReading,
    # Interface
    LockedInInterface,
    # Convenience
    create_locked_in_interface,
)

__all__ = [
    # Enums
    "LockedInType",
    "CommunicationChannel",
    "AwarenessState",
    "SignalQuality",
    "CognitiveFunction",
    # Input dataclasses
    "LockedInInput",
    "CommunicationAttemptInput",
    # Output dataclasses
    "DecodedIntention",
    "AwarenessAssessment",
    "CommunicationResult",
    "ConsciousnessMonitorReading",
    # Interface
    "LockedInInterface",
    # Convenience
    "create_locked_in_interface",
]
