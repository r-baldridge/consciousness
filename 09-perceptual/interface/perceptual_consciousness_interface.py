#!/usr/bin/env python3
"""
Perceptual Consciousness Interface

Form 09: Perceptual Consciousness handles the binding of sensory features
into unified percepts. It implements feature binding, perceptual organization,
cross-modal integration, and scene representation. This form is responsible
for the Gestalt principles that organize raw sensory data into meaningful
perceptual wholes, resolving the binding problem of consciousness.

This form receives processed features from Forms 01-06 (sensory forms)
and produces unified perceptual representations for higher-level processing.
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

class PerceptualBindingType(Enum):
    """Types of perceptual feature binding."""
    SPATIAL = "spatial"                  # Binding by shared location
    TEMPORAL = "temporal"                # Binding by simultaneity
    FEATURE = "feature"                  # Binding by shared features
    OBJECT = "object"                    # Binding into coherent objects
    CROSS_MODAL = "cross_modal"          # Binding across sensory modalities
    SEMANTIC = "semantic"                # Binding by shared meaning


class GestaltPrinciple(Enum):
    """Gestalt principles of perceptual organization."""
    PROXIMITY = "proximity"              # Nearby elements grouped together
    SIMILARITY = "similarity"            # Similar elements grouped together
    CONTINUITY = "continuity"            # Smooth continuation preferred
    CLOSURE = "closure"                  # Incomplete figures completed
    COMMON_FATE = "common_fate"          # Elements moving together grouped
    FIGURE_GROUND = "figure_ground"      # Separation of figure from background
    PRAGNANZ = "pragnanz"                # Simplest interpretation preferred
    COMMON_REGION = "common_region"      # Elements in shared region grouped


class AttentionalMode(Enum):
    """Modes of attentional selection in perception."""
    FOCAL = "focal"                      # Narrow, focused attention
    DIFFUSE = "diffuse"                  # Broad, distributed attention
    FEATURE_BASED = "feature_based"      # Attention to specific features
    OBJECT_BASED = "object_based"        # Attention to whole objects
    SPATIAL = "spatial"                  # Attention to spatial locations
    EXOGENOUS = "exogenous"              # Stimulus-driven capture
    ENDOGENOUS = "endogenous"            # Goal-driven direction


class SensoryChannel(Enum):
    """Sensory channels providing features for binding."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    SOMATOSENSORY = "somatosensory"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"
    INTEROCEPTIVE = "interoceptive"


class PerceptualQuality(Enum):
    """Quality assessment of perceptual binding."""
    VIVID = "vivid"
    CLEAR = "clear"
    FUZZY = "fuzzy"
    FRAGMENTARY = "fragmentary"
    ILLUSORY = "illusory"


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SensoryFeature:
    """A single sensory feature to be bound into a percept."""
    feature_id: str
    channel: SensoryChannel
    feature_type: str            # "color", "shape", "pitch", "texture", etc.
    feature_value: Any           # The actual feature value
    intensity: float             # 0.0-1.0
    spatial_location: Optional[Tuple[float, float, float]] = None  # x, y, z
    temporal_onset: float = 0.0  # milliseconds from reference
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "channel": self.channel.value,
            "feature_type": self.feature_type,
            "feature_value": str(self.feature_value),
            "intensity": self.intensity,
            "spatial_location": list(self.spatial_location) if self.spatial_location else None,
            "temporal_onset": self.temporal_onset,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerceptualInput:
    """Complete multi-sensory input for perceptual binding."""
    features: List[SensoryFeature] = field(default_factory=list)
    attentional_mode: AttentionalMode = AttentionalMode.DIFFUSE
    attentional_focus: Optional[str] = None    # Feature or location to focus on
    prior_context: Optional[str] = None         # What was perceived before
    expectations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class BoundPercept:
    """A unified percept created from bound features."""
    percept_id: str
    binding_type: PerceptualBindingType
    bound_features: List[str]        # Feature IDs that were bound
    channels_involved: List[SensoryChannel]
    gestalt_principles: List[GestaltPrinciple]
    coherence: float                 # 0.0-1.0 how well features bind
    salience: float                  # 0.0-1.0 attentional priority
    label: Optional[str] = None      # Identified object/event label
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percept_id": self.percept_id,
            "binding_type": self.binding_type.value,
            "bound_features": self.bound_features,
            "channels_involved": [c.value for c in self.channels_involved],
            "gestalt_principles": [g.value for g in self.gestalt_principles],
            "coherence": round(self.coherence, 4),
            "salience": round(self.salience, 4),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SceneRepresentation:
    """Representation of the complete perceptual scene."""
    percepts: List[BoundPercept]
    figure: Optional[str] = None          # Primary figure percept ID
    ground: Optional[str] = None          # Background percept ID
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    scene_coherence: float = 0.5          # Overall scene unity
    complexity: float = 0.5               # Scene complexity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percepts": [p.to_dict() for p in self.percepts],
            "figure": self.figure,
            "ground": self.ground,
            "scene_coherence": round(self.scene_coherence, 4),
            "complexity": round(self.complexity, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerceptualOutput:
    """Complete output from perceptual binding."""
    bound_percepts: List[BoundPercept]
    scene: SceneRepresentation
    dominant_channel: SensoryChannel
    attentional_mode_used: AttentionalMode
    perceptual_quality: PerceptualQuality
    binding_success_rate: float          # 0.0-1.0
    cross_modal_coherence: float         # 0.0-1.0
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound_percepts": [p.to_dict() for p in self.bound_percepts],
            "scene": self.scene.to_dict(),
            "dominant_channel": self.dominant_channel.value,
            "attentional_mode_used": self.attentional_mode_used.value,
            "perceptual_quality": self.perceptual_quality.value,
            "binding_success_rate": round(self.binding_success_rate, 4),
            "cross_modal_coherence": round(self.cross_modal_coherence, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerceptualSystemStatus:
    """Complete perceptual system status."""
    active_percepts: int
    binding_load: float                  # 0.0-1.0 current processing load
    dominant_channel: SensoryChannel
    attentional_mode: AttentionalMode
    perceptual_quality: PerceptualQuality
    system_health: float                 # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# FEATURE BINDING ENGINE
# ============================================================================

class FeatureBindingEngine:
    """
    Core engine for binding sensory features into unified percepts.

    Implements spatial, temporal, and cross-modal binding with
    Gestalt-based organization principles.
    """

    # Temporal binding window in milliseconds
    TEMPORAL_BINDING_WINDOW = 50.0
    # Spatial binding radius (normalized units)
    SPATIAL_BINDING_RADIUS = 0.3

    def __init__(self):
        self._binding_history: List[BoundPercept] = []
        self._max_history = 50
        self._next_percept_id = 0

    def bind_features(self, features: List[SensoryFeature]) -> List[BoundPercept]:
        """Bind a list of features into coherent percepts."""
        if not features:
            return []

        percepts = []

        # Group by spatial proximity
        spatial_groups = self._group_by_spatial(features)
        for group in spatial_groups:
            if len(group) > 1:
                percept = self._create_percept(
                    group, PerceptualBindingType.SPATIAL,
                    [GestaltPrinciple.PROXIMITY]
                )
                percepts.append(percept)

        # Group by temporal co-occurrence
        temporal_groups = self._group_by_temporal(features)
        for group in temporal_groups:
            if len(group) > 1 and not self._already_bound(group, percepts):
                percept = self._create_percept(
                    group, PerceptualBindingType.TEMPORAL,
                    [GestaltPrinciple.COMMON_FATE]
                )
                percepts.append(percept)

        # Cross-modal binding (always created if detected, even if features
        # are already bound within-modality, since cross-modal is a distinct binding)
        cross_modal = self._bind_cross_modal(features)
        for group in cross_modal:
            percept = self._create_percept(
                group, PerceptualBindingType.CROSS_MODAL,
                [GestaltPrinciple.COMMON_FATE, GestaltPrinciple.PROXIMITY]
            )
            percepts.append(percept)

        # If some features remain unbound, create single-feature percepts
        bound_ids = set()
        for p in percepts:
            bound_ids.update(p.bound_features)
        for f in features:
            if f.feature_id not in bound_ids:
                percept = self._create_percept(
                    [f], PerceptualBindingType.FEATURE,
                    [GestaltPrinciple.PRAGNANZ]
                )
                percepts.append(percept)

        self._binding_history.extend(percepts)
        if len(self._binding_history) > self._max_history:
            self._binding_history = self._binding_history[-self._max_history:]

        return percepts

    def compute_binding_success(self, features: List[SensoryFeature], percepts: List[BoundPercept]) -> float:
        """Compute the overall binding success rate."""
        if not features:
            return 1.0
        bound_count = sum(len(p.bound_features) for p in percepts)
        multi_bound = sum(1 for p in percepts if len(p.bound_features) > 1)
        total = len(features)
        return min(1.0, (bound_count / max(1, total)) * 0.5 + (multi_bound / max(1, len(percepts))) * 0.5)

    def _group_by_spatial(self, features: List[SensoryFeature]) -> List[List[SensoryFeature]]:
        """Group features by spatial proximity."""
        located = [f for f in features if f.spatial_location is not None]
        if not located:
            return []

        groups = []
        used = set()
        for i, f1 in enumerate(located):
            if i in used:
                continue
            group = [f1]
            used.add(i)
            for j, f2 in enumerate(located):
                if j in used:
                    continue
                dist = self._spatial_distance(f1.spatial_location, f2.spatial_location)
                if dist < self.SPATIAL_BINDING_RADIUS:
                    group.append(f2)
                    used.add(j)
            if len(group) > 1:
                groups.append(group)

        return groups

    def _group_by_temporal(self, features: List[SensoryFeature]) -> List[List[SensoryFeature]]:
        """Group features by temporal co-occurrence."""
        if len(features) < 2:
            return []

        groups = []
        used = set()
        for i, f1 in enumerate(features):
            if i in used:
                continue
            group = [f1]
            used.add(i)
            for j, f2 in enumerate(features):
                if j in used:
                    continue
                if abs(f1.temporal_onset - f2.temporal_onset) < self.TEMPORAL_BINDING_WINDOW:
                    group.append(f2)
                    used.add(j)
            if len(group) > 1:
                groups.append(group)

        return groups

    def _bind_cross_modal(self, features: List[SensoryFeature]) -> List[List[SensoryFeature]]:
        """Bind features across different sensory modalities."""
        channels = {}
        for f in features:
            channels.setdefault(f.channel, []).append(f)

        if len(channels) < 2:
            return []

        groups = []
        channel_list = list(channels.values())
        # Simple pairwise cross-modal binding by temporal proximity
        for i in range(len(channel_list)):
            for j in range(i + 1, len(channel_list)):
                for f1 in channel_list[i]:
                    for f2 in channel_list[j]:
                        if abs(f1.temporal_onset - f2.temporal_onset) < self.TEMPORAL_BINDING_WINDOW * 2:
                            groups.append([f1, f2])

        return groups

    def _create_percept(
        self,
        features: List[SensoryFeature],
        binding_type: PerceptualBindingType,
        gestalt_principles: List[GestaltPrinciple],
    ) -> BoundPercept:
        """Create a bound percept from features."""
        self._next_percept_id += 1
        channels = list(set(f.channel for f in features))
        coherence = sum(f.confidence for f in features) / len(features)
        salience = max(f.intensity for f in features)

        return BoundPercept(
            percept_id=f"percept_{self._next_percept_id:04d}",
            binding_type=binding_type,
            bound_features=[f.feature_id for f in features],
            channels_involved=channels,
            gestalt_principles=gestalt_principles,
            coherence=coherence,
            salience=salience,
        )

    def _already_bound(self, features: List[SensoryFeature], existing: List[BoundPercept]) -> bool:
        """Check if features are already in an existing percept."""
        feature_ids = set(f.feature_id for f in features)
        for p in existing:
            if feature_ids.issubset(set(p.bound_features)):
                return True
        return False

    def _spatial_distance(
        self, loc1: Tuple[float, float, float], loc2: Tuple[float, float, float]
    ) -> float:
        """Compute Euclidean distance between two spatial locations."""
        return sum((a - b) ** 2 for a, b in zip(loc1, loc2)) ** 0.5


# ============================================================================
# PERCEPTUAL ORGANIZATION ENGINE
# ============================================================================

class PerceptualOrganizationEngine:
    """
    Organizes bound percepts into a coherent scene representation.

    Handles figure-ground segregation, scene layout, and
    perceptual quality assessment.
    """

    def organize_scene(self, percepts: List[BoundPercept]) -> SceneRepresentation:
        """Organize percepts into a scene representation."""
        if not percepts:
            return SceneRepresentation(percepts=[])

        # Determine figure (most salient percept) and ground
        sorted_by_salience = sorted(percepts, key=lambda p: p.salience, reverse=True)
        figure_id = sorted_by_salience[0].percept_id
        ground_id = sorted_by_salience[-1].percept_id if len(sorted_by_salience) > 1 else None

        # Compute scene coherence
        coherence = sum(p.coherence for p in percepts) / len(percepts)
        complexity = min(1.0, len(percepts) / 10.0)

        return SceneRepresentation(
            percepts=percepts,
            figure=figure_id,
            ground=ground_id,
            scene_coherence=coherence,
            complexity=complexity,
        )

    def assess_quality(self, percepts: List[BoundPercept]) -> PerceptualQuality:
        """Assess the overall perceptual quality."""
        if not percepts:
            return PerceptualQuality.FRAGMENTARY

        avg_coherence = sum(p.coherence for p in percepts) / len(percepts)
        avg_confidence = sum(p.confidence for p in percepts) / len(percepts)
        combined = (avg_coherence + avg_confidence) / 2

        if combined > 0.8:
            return PerceptualQuality.VIVID
        elif combined > 0.6:
            return PerceptualQuality.CLEAR
        elif combined > 0.4:
            return PerceptualQuality.FUZZY
        else:
            return PerceptualQuality.FRAGMENTARY

    def compute_cross_modal_coherence(self, percepts: List[BoundPercept]) -> float:
        """Compute coherence across modalities."""
        cross_modal = [p for p in percepts if p.binding_type == PerceptualBindingType.CROSS_MODAL]
        if not cross_modal:
            return 0.5
        return sum(p.coherence for p in cross_modal) / len(cross_modal)


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class PerceptualConsciousnessInterface:
    """
    Main interface for Form 09: Perceptual Consciousness.

    Binds sensory features into unified percepts, organizes perceptual
    scenes, and manages cross-modal integration.
    """

    FORM_ID = "09-perceptual"
    FORM_NAME = "Perceptual Consciousness"

    def __init__(self):
        """Initialize the perceptual consciousness interface."""
        self.binding_engine = FeatureBindingEngine()
        self.organization_engine = PerceptualOrganizationEngine()

        self._current_output: Optional[PerceptualOutput] = None
        self._current_scene: Optional[SceneRepresentation] = None
        self._active_percepts: List[BoundPercept] = []
        self._attentional_mode: AttentionalMode = AttentionalMode.DIFFUSE
        self._initialized: bool = False

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the perceptual consciousness system."""
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized")

    async def process_perception(self, perceptual_input: PerceptualInput) -> PerceptualOutput:
        """
        Process multi-sensory features into unified perception.

        This is the main entry point for perceptual processing.
        """
        self._attentional_mode = perceptual_input.attentional_mode

        # Bind features
        percepts = self.binding_engine.bind_features(perceptual_input.features)
        self._active_percepts = percepts

        # Organize into scene
        scene = self.organization_engine.organize_scene(percepts)
        self._current_scene = scene

        # Assess quality
        quality = self.organization_engine.assess_quality(percepts)

        # Compute metrics
        binding_success = self.binding_engine.compute_binding_success(
            perceptual_input.features, percepts
        )
        cross_modal_coherence = self.organization_engine.compute_cross_modal_coherence(percepts)

        # Determine dominant channel
        dominant = self._determine_dominant_channel(perceptual_input.features)

        output = PerceptualOutput(
            bound_percepts=percepts,
            scene=scene,
            dominant_channel=dominant,
            attentional_mode_used=perceptual_input.attentional_mode,
            perceptual_quality=quality,
            binding_success_rate=binding_success,
            cross_modal_coherence=cross_modal_coherence,
        )
        self._current_output = output
        return output

    async def focus_attention(
        self, mode: AttentionalMode, target: Optional[str] = None
    ) -> None:
        """Change attentional mode and focus."""
        self._attentional_mode = mode
        logger.info(f"Attention shifted to {mode.value}, target: {target}")

    def get_active_percepts(self) -> List[BoundPercept]:
        """Get currently active percepts."""
        return self._active_percepts

    def get_current_scene(self) -> Optional[SceneRepresentation]:
        """Get current scene representation."""
        return self._current_scene

    def get_status(self) -> PerceptualSystemStatus:
        """Get complete perceptual system status."""
        quality = PerceptualQuality.FRAGMENTARY
        dominant = SensoryChannel.VISUAL
        if self._active_percepts:
            quality = self.organization_engine.assess_quality(self._active_percepts)
        if self._current_output:
            dominant = self._current_output.dominant_channel

        return PerceptualSystemStatus(
            active_percepts=len(self._active_percepts),
            binding_load=min(1.0, len(self._active_percepts) / 20.0),
            dominant_channel=dominant,
            attentional_mode=self._attentional_mode,
            perceptual_quality=quality,
            system_health=self._compute_health(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "active_percepts": len(self._active_percepts),
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "current_scene": self._current_scene.to_dict() if self._current_scene else None,
            "attentional_mode": self._attentional_mode.value,
            "initialized": self._initialized,
        }

    def _determine_dominant_channel(self, features: List[SensoryFeature]) -> SensoryChannel:
        """Determine which sensory channel dominates input."""
        if not features:
            return SensoryChannel.VISUAL
        channel_counts: Dict[SensoryChannel, float] = {}
        for f in features:
            channel_counts[f.channel] = channel_counts.get(f.channel, 0.0) + f.intensity
        return max(channel_counts, key=channel_counts.get)

    def _compute_health(self) -> float:
        """Compute perceptual system health."""
        if not self._active_percepts:
            return 1.0
        avg_coherence = sum(p.coherence for p in self._active_percepts) / len(self._active_percepts)
        return avg_coherence


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_perceptual_interface() -> PerceptualConsciousnessInterface:
    """Create and return a perceptual consciousness interface."""
    return PerceptualConsciousnessInterface()


__all__ = [
    # Enums
    "PerceptualBindingType",
    "GestaltPrinciple",
    "AttentionalMode",
    "SensoryChannel",
    "PerceptualQuality",
    # Input dataclasses
    "SensoryFeature",
    "PerceptualInput",
    # Output dataclasses
    "BoundPercept",
    "SceneRepresentation",
    "PerceptualOutput",
    "PerceptualSystemStatus",
    # Engines
    "FeatureBindingEngine",
    "PerceptualOrganizationEngine",
    # Main interface
    "PerceptualConsciousnessInterface",
    # Convenience
    "create_perceptual_interface",
]
