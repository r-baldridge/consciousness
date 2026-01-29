"""
Form 01: Visual Consciousness Interface

Primary visual processing system for consciousness.
"""

from .visual_consciousness_interface import (
    # Enums
    VisualFeatureType,
    ColorSpace,
    SceneCategory,
    ObjectCategory,
    AttentionMode,
    # Input dataclasses
    VisualFeatureVector,
    VisualInput,
    ObjectDetection,
    # Output dataclasses
    FeatureExtractionResult,
    SceneInterpretation,
    SalienceMap,
    VisualOutput,
    # Main interface
    VisualConsciousnessInterface,
    # Convenience functions
    create_visual_interface,
    create_simple_visual_input,
)

__all__ = [
    # Enums
    "VisualFeatureType",
    "ColorSpace",
    "SceneCategory",
    "ObjectCategory",
    "AttentionMode",
    # Input dataclasses
    "VisualFeatureVector",
    "VisualInput",
    "ObjectDetection",
    # Output dataclasses
    "FeatureExtractionResult",
    "SceneInterpretation",
    "SalienceMap",
    "VisualOutput",
    # Main interface
    "VisualConsciousnessInterface",
    # Convenience functions
    "create_visual_interface",
    "create_simple_visual_input",
]
