#!/usr/bin/env python3
"""
Visual Consciousness Interface

Form 01: The primary visual processing system for consciousness.
Visual consciousness processes raw visual input through feature extraction,
object recognition, scene understanding, face detection, and salience mapping
to construct a coherent visual experience.

This form handles the transformation of visual stimuli into conscious
visual perception, including color, shape, motion, depth, and spatial
relationships.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class VisualFeatureType(Enum):
    """Types of visual features that can be extracted."""
    EDGE = "edge"
    COLOR = "color"
    TEXTURE = "texture"
    SHAPE = "shape"
    MOTION = "motion"
    DEPTH = "depth"
    LUMINANCE = "luminance"
    ORIENTATION = "orientation"
    SPATIAL_FREQUENCY = "spatial_frequency"
    CONTRAST = "contrast"


class ColorSpace(Enum):
    """Color spaces for visual processing."""
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    GRAYSCALE = "grayscale"
    OPPONENT = "opponent"  # Red-green, blue-yellow opponent channels


class SceneCategory(Enum):
    """High-level scene categories for scene understanding."""
    NATURAL = "natural"
    URBAN = "urban"
    INDOOR = "indoor"
    ABSTRACT = "abstract"
    SOCIAL = "social"
    THREATENING = "threatening"
    FAMILIAR = "familiar"
    NOVEL = "novel"


class ObjectCategory(Enum):
    """Categories of recognized objects."""
    FACE = "face"
    BODY = "body"
    ANIMAL = "animal"
    TOOL = "tool"
    VEHICLE = "vehicle"
    FOOD = "food"
    NATURAL_OBJECT = "natural_object"
    TEXT = "text"
    UNKNOWN = "unknown"


class AttentionMode(Enum):
    """Visual attention processing modes."""
    FOCAL = "focal"           # Focused on specific location/object
    AMBIENT = "ambient"       # Broad environmental scanning
    SACCADIC = "saccadic"     # Rapid eye movement scanning
    SUSTAINED = "sustained"   # Prolonged focus on target
    DIVIDED = "divided"       # Attending to multiple targets


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class VisualFeatureVector:
    """A vector of extracted visual features."""
    feature_type: VisualFeatureType
    values: List[float]
    spatial_location: Tuple[float, float] = (0.5, 0.5)  # Normalized x, y
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type": self.feature_type.value,
            "values": self.values,
            "spatial_location": list(self.spatial_location),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VisualInput:
    """Input to the visual consciousness system."""
    feature_vectors: List[VisualFeatureVector] = field(default_factory=list)
    color_space: ColorSpace = ColorSpace.RGB
    image_dimensions: Tuple[int, int] = (0, 0)  # width, height
    luminance_level: float = 0.5  # 0.0 (dark) to 1.0 (bright)
    contrast_level: float = 0.5   # 0.0 (low) to 1.0 (high)
    motion_detected: bool = False
    motion_velocity: float = 0.0   # Speed of detected motion
    motion_direction: float = 0.0  # Direction in degrees
    depth_available: bool = False
    raw_salience_map: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_features": len(self.feature_vectors),
            "color_space": self.color_space.value,
            "image_dimensions": list(self.image_dimensions),
            "luminance_level": self.luminance_level,
            "contrast_level": self.contrast_level,
            "motion_detected": self.motion_detected,
            "motion_velocity": self.motion_velocity,
            "depth_available": self.depth_available,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ObjectDetection:
    """A detected object in the visual field."""
    category: ObjectCategory
    label: str
    confidence: float
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x, y, w, h
    is_face: bool = False
    emotional_valence: float = 0.0  # -1.0 to 1.0 for faces
    familiarity: float = 0.0  # 0.0 (novel) to 1.0 (very familiar)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bounding_box": list(self.bounding_box),
            "is_face": self.is_face,
            "emotional_valence": self.emotional_valence,
            "familiarity": self.familiarity,
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class FeatureExtractionResult:
    """Result of visual feature extraction."""
    features_extracted: Dict[str, float]  # feature_name -> strength
    dominant_feature: VisualFeatureType
    feature_coherence: float  # 0.0-1.0 how well features form unified percept
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features_extracted": {k: round(v, 4) for k, v in self.features_extracted.items()},
            "dominant_feature": self.dominant_feature.value,
            "feature_coherence": round(self.feature_coherence, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SceneInterpretation:
    """High-level interpretation of the visual scene."""
    scene_category: SceneCategory
    scene_description: str
    scene_confidence: float
    spatial_layout: Dict[str, float]  # region -> importance
    gist_features: List[str]  # Quick scene gist descriptors
    emotional_tone: float  # -1.0 to 1.0
    novelty_score: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_category": self.scene_category.value,
            "scene_description": self.scene_description,
            "scene_confidence": round(self.scene_confidence, 4),
            "spatial_layout": {k: round(v, 4) for k, v in self.spatial_layout.items()},
            "gist_features": self.gist_features,
            "emotional_tone": round(self.emotional_tone, 4),
            "novelty_score": round(self.novelty_score, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SalienceMap:
    """Visual salience map indicating attention-grabbing regions."""
    salience_values: List[float]  # Flattened salience grid
    peak_location: Tuple[float, float]  # Most salient point (x, y)
    peak_salience: float
    num_hotspots: int
    attention_recommendation: AttentionMode
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_salience_points": len(self.salience_values),
            "peak_location": list(self.peak_location),
            "peak_salience": round(self.peak_salience, 4),
            "num_hotspots": self.num_hotspots,
            "attention_recommendation": self.attention_recommendation.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VisualOutput:
    """Complete output of visual consciousness processing."""
    feature_result: FeatureExtractionResult
    objects_detected: List[ObjectDetection]
    scene_interpretation: SceneInterpretation
    salience_map: SalienceMap
    overall_confidence: float
    visual_clarity: float  # 0.0-1.0 clarity of conscious visual experience
    requires_attention_shift: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_result": self.feature_result.to_dict(),
            "num_objects": len(self.objects_detected),
            "objects": [obj.to_dict() for obj in self.objects_detected],
            "scene_interpretation": self.scene_interpretation.to_dict(),
            "salience_map": self.salience_map.to_dict(),
            "overall_confidence": round(self.overall_confidence, 4),
            "visual_clarity": round(self.visual_clarity, 4),
            "requires_attention_shift": self.requires_attention_shift,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class VisualConsciousnessInterface:
    """
    Main interface for Form 01: Visual Consciousness.

    Processes visual stimuli through a pipeline of feature extraction,
    object recognition, scene understanding, and salience computation
    to produce conscious visual experience.
    """

    FORM_ID = "01-visual"
    FORM_NAME = "Visual Consciousness"

    def __init__(self):
        """Initialize the visual consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[VisualOutput] = None
        self._attention_mode = AttentionMode.AMBIENT
        self._feature_history: List[FeatureExtractionResult] = []
        self._object_memory: Dict[str, float] = {}  # label -> familiarity
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the visual processing pipeline."""
        self._initialized = True
        self._attention_mode = AttentionMode.AMBIENT
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_visual_input(self, visual_input: VisualInput) -> VisualOutput:
        """
        Process visual input through the full consciousness pipeline.

        Pipeline stages:
        1. Feature extraction
        2. Object recognition
        3. Scene interpretation
        4. Salience computation
        5. Conscious integration
        """
        self._processing_count += 1

        # Stage 1: Feature extraction
        feature_result = await self._extract_features(visual_input)

        # Stage 2: Object recognition
        objects = await self._recognize_objects(visual_input, feature_result)

        # Stage 3: Scene interpretation
        scene = await self._interpret_scene(visual_input, feature_result, objects)

        # Stage 4: Salience computation
        salience = await self._compute_salience(visual_input, objects, scene)

        # Stage 5: Integration
        clarity = self._compute_visual_clarity(visual_input, feature_result)
        confidence = self._compute_overall_confidence(feature_result, objects, scene)
        needs_shift = salience.peak_salience > 0.8 and self._attention_mode != AttentionMode.FOCAL

        output = VisualOutput(
            feature_result=feature_result,
            objects_detected=objects,
            scene_interpretation=scene,
            salience_map=salience,
            overall_confidence=confidence,
            visual_clarity=clarity,
            requires_attention_shift=needs_shift,
        )

        self._current_output = output
        self._update_history(feature_result, objects)

        return output

    async def _extract_features(self, visual_input: VisualInput) -> FeatureExtractionResult:
        """Extract low-level visual features."""
        features = {}

        # Process each feature vector
        for fv in visual_input.feature_vectors:
            feature_name = fv.feature_type.value
            feature_strength = sum(fv.values) / max(1, len(fv.values)) if fv.values else 0.0
            features[feature_name] = min(1.0, feature_strength * fv.confidence)

        # Add basic features from input properties
        features.setdefault("luminance", visual_input.luminance_level)
        features.setdefault("contrast", visual_input.contrast_level)
        if visual_input.motion_detected:
            features["motion"] = min(1.0, visual_input.motion_velocity)

        # Determine dominant feature
        dominant = VisualFeatureType.LUMINANCE
        if features:
            dominant_name = max(features, key=features.get)
            for ft in VisualFeatureType:
                if ft.value == dominant_name:
                    dominant = ft
                    break

        # Compute coherence
        values = list(features.values()) if features else [0.5]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        coherence = max(0.0, 1.0 - variance * 4)

        return FeatureExtractionResult(
            features_extracted=features,
            dominant_feature=dominant,
            feature_coherence=coherence,
        )

    async def _recognize_objects(
        self, visual_input: VisualInput, features: FeatureExtractionResult
    ) -> List[ObjectDetection]:
        """Recognize objects in the visual field."""
        objects = []

        # Simple heuristic object detection based on features
        feature_strength = sum(features.features_extracted.values()) / max(1, len(features.features_extracted))

        if feature_strength > 0.3:
            # Detect a generic object based on feature patterns
            obj = ObjectDetection(
                category=ObjectCategory.UNKNOWN,
                label="detected_object",
                confidence=feature_strength,
                bounding_box=(0.2, 0.2, 0.6, 0.6),
                familiarity=self._object_memory.get("detected_object", 0.0),
            )
            objects.append(obj)

        # Check for face-like patterns (high contrast + shape features)
        has_contrast = features.features_extracted.get("contrast", 0) > 0.5
        has_shape = features.features_extracted.get("shape", 0) > 0.3
        if has_contrast and has_shape:
            face = ObjectDetection(
                category=ObjectCategory.FACE,
                label="face",
                confidence=0.6,
                bounding_box=(0.3, 0.1, 0.4, 0.5),
                is_face=True,
                emotional_valence=0.0,
                familiarity=self._object_memory.get("face", 0.0),
            )
            objects.append(face)

        return objects

    async def _interpret_scene(
        self,
        visual_input: VisualInput,
        features: FeatureExtractionResult,
        objects: List[ObjectDetection],
    ) -> SceneInterpretation:
        """Interpret the overall visual scene."""
        # Determine scene category based on features and objects
        has_faces = any(obj.is_face for obj in objects)
        has_motion = visual_input.motion_detected
        luminance = visual_input.luminance_level

        if has_faces:
            category = SceneCategory.SOCIAL
        elif has_motion and luminance > 0.6:
            category = SceneCategory.URBAN
        elif luminance > 0.4:
            category = SceneCategory.NATURAL
        else:
            category = SceneCategory.ABSTRACT

        # Build gist features
        gist = []
        if luminance > 0.7:
            gist.append("bright")
        elif luminance < 0.3:
            gist.append("dark")
        if has_motion:
            gist.append("dynamic")
        else:
            gist.append("static")
        if visual_input.contrast_level > 0.6:
            gist.append("high_contrast")
        if has_faces:
            gist.append("faces_present")

        # Compute novelty
        novelty = 1.0 - features.feature_coherence * 0.5
        if len(self._feature_history) > 3:
            novelty = min(1.0, novelty + 0.2)

        return SceneInterpretation(
            scene_category=category,
            scene_description=f"{category.value} scene with {len(objects)} objects",
            scene_confidence=features.feature_coherence * 0.8,
            spatial_layout={"center": 0.6, "periphery": 0.4},
            gist_features=gist,
            emotional_tone=0.0 if not has_faces else 0.2,
            novelty_score=novelty,
        )

    async def _compute_salience(
        self,
        visual_input: VisualInput,
        objects: List[ObjectDetection],
        scene: SceneInterpretation,
    ) -> SalienceMap:
        """Compute visual salience map."""
        # Build simple salience values
        salience_values = [0.3] * 9  # 3x3 grid

        # Center gets baseline salience
        salience_values[4] = 0.5

        # Objects increase salience at their locations
        for obj in objects:
            bx, by = obj.bounding_box[0], obj.bounding_box[1]
            grid_idx = min(8, int(by * 3) * 3 + int(bx * 3))
            salience_values[grid_idx] = min(1.0, salience_values[grid_idx] + obj.confidence * 0.4)

        # Motion increases salience
        if visual_input.motion_detected:
            for i in range(len(salience_values)):
                salience_values[i] = min(1.0, salience_values[i] + 0.2)

        peak_val = max(salience_values)
        peak_idx = salience_values.index(peak_val)
        peak_x = (peak_idx % 3) / 2.0
        peak_y = (peak_idx // 3) / 2.0

        hotspots = sum(1 for v in salience_values if v > 0.6)

        # Determine attention recommendation
        if hotspots == 0:
            attention = AttentionMode.AMBIENT
        elif hotspots == 1:
            attention = AttentionMode.FOCAL
        elif hotspots > 3:
            attention = AttentionMode.DIVIDED
        else:
            attention = AttentionMode.SACCADIC

        return SalienceMap(
            salience_values=salience_values,
            peak_location=(peak_x, peak_y),
            peak_salience=peak_val,
            num_hotspots=hotspots,
            attention_recommendation=attention,
        )

    def _compute_visual_clarity(
        self, visual_input: VisualInput, features: FeatureExtractionResult
    ) -> float:
        """Compute clarity of the visual conscious experience."""
        clarity = (
            visual_input.luminance_level * 0.3 +
            visual_input.contrast_level * 0.3 +
            features.feature_coherence * 0.4
        )
        return max(0.0, min(1.0, clarity))

    def _compute_overall_confidence(
        self,
        features: FeatureExtractionResult,
        objects: List[ObjectDetection],
        scene: SceneInterpretation,
    ) -> float:
        """Compute overall processing confidence."""
        feature_conf = features.feature_coherence
        object_conf = (
            sum(obj.confidence for obj in objects) / max(1, len(objects))
            if objects else 0.5
        )
        scene_conf = scene.scene_confidence
        return (feature_conf + object_conf + scene_conf) / 3.0

    def _update_history(
        self, features: FeatureExtractionResult, objects: List[ObjectDetection]
    ) -> None:
        """Update processing history and object memory."""
        self._feature_history.append(features)
        if len(self._feature_history) > self._max_history:
            self._feature_history.pop(0)

        for obj in objects:
            current = self._object_memory.get(obj.label, 0.0)
            self._object_memory[obj.label] = min(1.0, current + 0.1)

    def set_attention_mode(self, mode: AttentionMode) -> None:
        """Set the current visual attention mode."""
        self._attention_mode = mode

    def get_attention_mode(self) -> AttentionMode:
        """Get the current visual attention mode."""
        return self._attention_mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "attention_mode": self._attention_mode.value,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "known_objects": len(self._object_memory),
            "history_length": len(self._feature_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "attention_mode": self._attention_mode.value,
            "visual_clarity": (
                self._current_output.visual_clarity if self._current_output else 0.0
            ),
            "objects_tracked": len(self._object_memory),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_visual_interface() -> VisualConsciousnessInterface:
    """Create and return a visual consciousness interface."""
    return VisualConsciousnessInterface()


def create_simple_visual_input(
    luminance: float = 0.5,
    contrast: float = 0.5,
    motion: bool = False,
    motion_speed: float = 0.0,
) -> VisualInput:
    """Create a simple visual input for testing."""
    return VisualInput(
        feature_vectors=[
            VisualFeatureVector(
                feature_type=VisualFeatureType.LUMINANCE,
                values=[luminance],
            ),
            VisualFeatureVector(
                feature_type=VisualFeatureType.CONTRAST,
                values=[contrast],
            ),
        ],
        luminance_level=luminance,
        contrast_level=contrast,
        motion_detected=motion,
        motion_velocity=motion_speed,
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
