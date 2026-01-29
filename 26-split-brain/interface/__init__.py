"""
Form 26: Split-Brain Consciousness Interface

Models dual consciousness after corpus callosum severing, including
hemispheric specialization, lateralized processing, inter-hemispheric
conflict, and left-hemisphere confabulation.
"""

from .split_brain_interface import (
    # Enums
    Hemisphere,
    ProcessingDomain,
    InterhemisphericState,
    ConflictType,
    ConfabulationType,
    LateralizedField,
    # Input dataclasses
    SplitBrainInput,
    BilateralInput,
    # Output dataclasses
    HemisphereResponse,
    SplitBrainOutput,
    ConflictAnalysis,
    ConfabulationOutput,
    # Interface
    SplitBrainInterface,
    # Convenience
    create_split_brain_interface,
)

__all__ = [
    # Enums
    "Hemisphere",
    "ProcessingDomain",
    "InterhemisphericState",
    "ConflictType",
    "ConfabulationType",
    "LateralizedField",
    # Input dataclasses
    "SplitBrainInput",
    "BilateralInput",
    # Output dataclasses
    "HemisphereResponse",
    "SplitBrainOutput",
    "ConflictAnalysis",
    "ConfabulationOutput",
    # Interface
    "SplitBrainInterface",
    # Convenience
    "create_split_brain_interface",
]
