"""
Form 16: Predictive Coding / Free Energy Principle Consciousness Interface

Implements Friston/Clark's predictive processing framework for consciousness.
"""

from .predictive_coding_interface import (
    # Enums
    PredictionLevel,
    ErrorType,
    PrecisionLevel,
    UpdateStrategy,
    FreeEnergyComponent,
    # Input dataclasses
    Prediction,
    SensoryEvidence,
    PredictiveCodingInput,
    # Output dataclasses
    PredictionError,
    ModelUpdate,
    FreeEnergyState,
    PredictiveCodingOutput,
    PredictiveCodingSystemStatus,
    # Engines
    PredictionErrorEngine,
    GenerativeModelEngine,
    FreeEnergyEngine,
    # Main interface
    PredictiveCodingInterface,
    # Convenience functions
    create_predictive_coding_interface,
    create_prediction,
    create_evidence,
)

__all__ = [
    # Enums
    "PredictionLevel",
    "ErrorType",
    "PrecisionLevel",
    "UpdateStrategy",
    "FreeEnergyComponent",
    # Input dataclasses
    "Prediction",
    "SensoryEvidence",
    "PredictiveCodingInput",
    # Output dataclasses
    "PredictionError",
    "ModelUpdate",
    "FreeEnergyState",
    "PredictiveCodingOutput",
    "PredictiveCodingSystemStatus",
    # Engines
    "PredictionErrorEngine",
    "GenerativeModelEngine",
    "FreeEnergyEngine",
    # Main interface
    "PredictiveCodingInterface",
    # Convenience functions
    "create_predictive_coding_interface",
    "create_prediction",
    "create_evidence",
]
