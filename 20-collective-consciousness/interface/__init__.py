"""
Form 20: Collective Consciousness (Durkheim) Interface

Shared beliefs, social cohesion, group mind states, meme propagation,
and emergent collective properties.
"""

from .collective_consciousness_interface import (
    # Enums
    CollectiveType,
    SocialCohesion,
    GroupMindState,
    BeliefStrength,
    PropagationMode,
    # Input dataclasses
    IndividualBelief,
    SocialSignal,
    CollectiveInput,
    # Output dataclasses
    SharedRepresentation,
    EmergentProperty,
    CollectiveOutput,
    # Interface
    CollectiveConsciousnessInterface,
    # Convenience
    create_collective_consciousness_interface,
)

__all__ = [
    "CollectiveType",
    "SocialCohesion",
    "GroupMindState",
    "BeliefStrength",
    "PropagationMode",
    "IndividualBelief",
    "SocialSignal",
    "CollectiveInput",
    "SharedRepresentation",
    "EmergentProperty",
    "CollectiveOutput",
    "CollectiveConsciousnessInterface",
    "create_collective_consciousness_interface",
]
