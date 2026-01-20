"""
Form 08: Arousal/Vigilance Consciousness Interface

CRITICAL FORM - Always loaded, gates all other forms.
"""

from .arousal_consciousness_interface import (
    # Enums
    ArousalState,
    ArousalSource,
    GateCategory,
    SensoryModality,
    StateTransitionType,
    # Input dataclasses
    SensoryArousalInput,
    ThreatInput,
    NoveltyInput,
    CircadianInput,
    EmotionalInput,
    TaskDemandInput,
    ResourceInput,
    ArousalInputBundle,
    # Output dataclasses
    ArousalLevelOutput,
    GatingSignal,
    ConsciousnessGatingOutput,
    ResourceAllocationOutput,
    StateTransition,
    ArousalSystemStatus,
    # Engines
    ArousalComputationEngine,
    ConsciousnessGatingEngine,
    ResourceAllocationEngine,
    # Main interface
    ArousalConsciousnessInterface,
    # Convenience functions
    create_arousal_interface,
    create_simple_input,
)

__all__ = [
    # Enums
    "ArousalState",
    "ArousalSource",
    "GateCategory",
    "SensoryModality",
    "StateTransitionType",
    # Input dataclasses
    "SensoryArousalInput",
    "ThreatInput",
    "NoveltyInput",
    "CircadianInput",
    "EmotionalInput",
    "TaskDemandInput",
    "ResourceInput",
    "ArousalInputBundle",
    # Output dataclasses
    "ArousalLevelOutput",
    "GatingSignal",
    "ConsciousnessGatingOutput",
    "ResourceAllocationOutput",
    "StateTransition",
    "ArousalSystemStatus",
    # Engines
    "ArousalComputationEngine",
    "ConsciousnessGatingEngine",
    "ResourceAllocationEngine",
    # Main interface
    "ArousalConsciousnessInterface",
    # Convenience functions
    "create_arousal_interface",
    "create_simple_input",
]
