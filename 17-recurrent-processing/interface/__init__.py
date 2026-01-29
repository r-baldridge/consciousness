"""
Form 17: Recurrent Processing Theory (RPT) Consciousness Interface

Implements Lamme's RPT for consciousness through recurrent feedback processing.
"""

from .recurrent_processing_interface import (
    # Enums
    ProcessingPhase,
    RecurrenceType,
    ProcessingLevel,
    ConsciousnessState,
    MaskingEffect,
    # Input dataclasses
    FeedforwardSweep,
    RecurrentSignal,
    MaskingInput,
    RecurrentProcessingInput,
    # Output dataclasses
    RecurrentState,
    ConsciousnessThresholdResult,
    RecurrentProcessingOutput,
    RPTSystemStatus,
    # Engines
    FeedforwardEngine,
    RecurrenceEngine,
    # Main interface
    RecurrentProcessingInterface,
    # Convenience functions
    create_recurrent_processing_interface,
    create_feedforward_sweep,
    create_masking_input,
)

__all__ = [
    # Enums
    "ProcessingPhase",
    "RecurrenceType",
    "ProcessingLevel",
    "ConsciousnessState",
    "MaskingEffect",
    # Input dataclasses
    "FeedforwardSweep",
    "RecurrentSignal",
    "MaskingInput",
    "RecurrentProcessingInput",
    # Output dataclasses
    "RecurrentState",
    "ConsciousnessThresholdResult",
    "RecurrentProcessingOutput",
    "RPTSystemStatus",
    # Engines
    "FeedforwardEngine",
    "RecurrenceEngine",
    # Main interface
    "RecurrentProcessingInterface",
    # Convenience functions
    "create_recurrent_processing_interface",
    "create_feedforward_sweep",
    "create_masking_input",
]
