"""
Form 15: Higher-Order Thought (HOT) Theory Consciousness Interface

Implements Rosenthal/Lau's HOT theory for meta-representational consciousness.
"""

from .higher_order_thought_interface import (
    # Enums
    RepresentationOrder,
    ConsciousnessType,
    RepresentationModality,
    HOTQuality,
    AssessmentCriterion,
    # Input dataclasses
    FirstOrderState,
    HOTRequest,
    # Output dataclasses
    HigherOrderRepresentation,
    ConsciousnessAssessment,
    RepresentationHierarchy,
    HOTOutput,
    HOTSystemStatus,
    # Engines
    HOTGenerationEngine,
    ConsciousnessAssessmentEngine,
    # Main interface
    HigherOrderThoughtInterface,
    # Convenience functions
    create_higher_order_thought_interface,
    create_first_order_state,
)

__all__ = [
    # Enums
    "RepresentationOrder",
    "ConsciousnessType",
    "RepresentationModality",
    "HOTQuality",
    "AssessmentCriterion",
    # Input dataclasses
    "FirstOrderState",
    "HOTRequest",
    # Output dataclasses
    "HigherOrderRepresentation",
    "ConsciousnessAssessment",
    "RepresentationHierarchy",
    "HOTOutput",
    "HOTSystemStatus",
    # Engines
    "HOTGenerationEngine",
    "ConsciousnessAssessmentEngine",
    # Main interface
    "HigherOrderThoughtInterface",
    # Convenience functions
    "create_higher_order_thought_interface",
    "create_first_order_state",
]
