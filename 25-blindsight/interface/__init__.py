"""
Form 25: Blindsight Consciousness Interface

Models the dissociation between conscious and unconscious visual processing,
including blind-field processing, forced-choice paradigms, implicit detection
assessment, and visual pathway analysis.
"""

from .blindsight_interface import (
    # Enums
    BlindsightType,
    VisualFieldRegion,
    ProcessingPathway,
    StimulusProperty,
    DetectionConfidence,
    # Input dataclasses
    BlindsightInput,
    ForcedChoiceTrialInput,
    # Output dataclasses
    BlindsightOutput,
    PathwayAnalysis,
    ImplicitDetectionResult,
    BlindsightProfile,
    # Interface
    BlindsightInterface,
    # Convenience
    create_blindsight_interface,
)

__all__ = [
    # Enums
    "BlindsightType",
    "VisualFieldRegion",
    "ProcessingPathway",
    "StimulusProperty",
    "DetectionConfidence",
    # Input dataclasses
    "BlindsightInput",
    "ForcedChoiceTrialInput",
    # Output dataclasses
    "BlindsightOutput",
    "PathwayAnalysis",
    "ImplicitDetectionResult",
    "BlindsightProfile",
    # Interface
    "BlindsightInterface",
    # Convenience
    "create_blindsight_interface",
]
