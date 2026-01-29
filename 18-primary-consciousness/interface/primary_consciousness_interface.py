#!/usr/bin/env python3
"""
Primary Consciousness Interface

Form 18: Primary consciousness as described by Gerald Edelman -- the ability
to construct a unified scene from ongoing sensory data, categorized through
value-laden memory. Primary consciousness is present-oriented awareness
without self-reflection, language, or autobiographical sense.

It involves:
- Perceptual categorization of sensory inputs
- Value-category memory linking percepts to hedonic values
- The "remembered present" -- current perception shaped by past experience
- Scene construction binding multimodal percepts into a coherent whole

Primary consciousness is shared by many mammals and birds. It is distinct
from higher-order consciousness (Form 19) which adds self-reflection.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PrimaryAwarenessLevel(Enum):
    """
    Levels of primary awareness intensity.

    Represents the degree to which an organism is forming a coherent
    perceptual scene from sensory data.
    """
    ABSENT = "absent"              # No primary awareness (deep sleep, coma)
    MINIMAL = "minimal"            # Fragmentary awareness, pre-scene
    PARTIAL = "partial"            # Some scene elements bound, gaps remain
    COHERENT = "coherent"          # Full scene constructed, normal waking
    HEIGHTENED = "heightened"       # Enhanced perceptual vividness (novelty, threat)


class SensoryBoundState(Enum):
    """
    State of multimodal sensory binding in a perceptual scene.

    Edelman emphasizes that primary consciousness depends on reentrant
    signaling that binds distributed cortical maps into a unified percept.
    """
    UNBOUND = "unbound"            # Sensory streams separate, no integration
    LOOSELY_BOUND = "loosely_bound" # Partial binding, some cross-modal links
    BOUND = "bound"                # Normal binding, coherent scene
    TIGHTLY_BOUND = "tightly_bound" # Strong binding, vivid unified percept
    FRAGMENTED = "fragmented"       # Previously bound scene breaking apart


class PerceptualCategory(Enum):
    """
    Categories into which sensory inputs are classified.

    Perceptual categorization is the foundation of primary consciousness --
    the ability to discriminate and classify sensory signals without
    requiring language or symbolic thought.
    """
    OBJECT = "object"              # Discrete bounded entities
    SURFACE = "surface"            # Textures, colors, materials
    MOTION = "motion"              # Movement patterns
    SPATIAL = "spatial"            # Locations, distances, layouts
    AUDITORY_PATTERN = "auditory_pattern"  # Sound sequences, tones
    OLFACTORY_SIGNATURE = "olfactory_signature"  # Smell patterns
    TACTILE_QUALITY = "tactile_quality"  # Touch, temperature, pressure
    SOCIAL_SIGNAL = "social_signal"  # Faces, gestures, vocalizations
    THREAT_CUE = "threat_cue"      # Danger-related patterns
    REWARD_CUE = "reward_cue"      # Benefit-related patterns


class ValueAssignment(Enum):
    """
    Hedonic value assignments linked to perceptual categories.

    In Edelman's theory, the value system (based on brainstem nuclei
    and limbic structures) assigns salience and hedonic tone to percepts,
    shaping which elements enter primary consciousness.
    """
    STRONGLY_AVERSIVE = "strongly_aversive"  # Pain, danger
    MILDLY_AVERSIVE = "mildly_aversive"      # Discomfort, mild threat
    NEUTRAL = "neutral"                       # No hedonic charge
    MILDLY_APPETITIVE = "mildly_appetitive"   # Mild interest, curiosity
    STRONGLY_APPETITIVE = "strongly_appetitive"  # Pleasure, strong reward
    NOVEL = "novel"                           # New, unclassified stimulus


class SceneCoherence(Enum):
    """
    Degree of coherence of the constructed perceptual scene.
    """
    INCOHERENT = "incoherent"      # No meaningful scene
    FRAGMENTARY = "fragmentary"     # Isolated percepts, no scene unity
    PARTIAL = "partial"            # Some scene structure, gaps present
    COHERENT = "coherent"          # Normal unified scene
    VIVID = "vivid"                # Exceptionally clear, detailed scene


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SensoryChannelData:
    """Raw data from a single sensory channel."""
    channel: str                   # e.g., "visual", "auditory", "tactile"
    raw_intensity: float           # 0.0-1.0
    pattern_complexity: float      # 0.0-1.0
    novelty_score: float           # 0.0-1.0 relative to recent history
    temporal_rate: float           # Rate of change in the channel
    reliability: float = 1.0      # Signal quality / confidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "raw_intensity": round(self.raw_intensity, 4),
            "pattern_complexity": round(self.pattern_complexity, 4),
            "novelty_score": round(self.novelty_score, 4),
            "temporal_rate": round(self.temporal_rate, 4),
            "reliability": round(self.reliability, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PrimarySensoryInput:
    """
    Bundled sensory input for primary consciousness processing.

    Represents the raw multimodal sensory data that primary consciousness
    must categorize, value-tag, and bind into a scene.
    """
    channels: List[SensoryChannelData] = field(default_factory=list)
    ambient_arousal: float = 0.5   # Overall arousal level from Form 08
    context_id: str = ""           # Identifier for the current environmental context
    elapsed_ms: float = 0.0        # Time since last processing cycle
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": [ch.to_dict() for ch in self.channels],
            "ambient_arousal": round(self.ambient_arousal, 4),
            "context_id": self.context_id,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class CategorizedPercept:
    """A single percept that has been categorized and value-tagged."""
    percept_id: str
    category: PerceptualCategory
    value: ValueAssignment
    salience: float                # 0.0-1.0, how strongly it enters awareness
    source_channels: List[str]     # Which sensory channels contribute
    confidence: float = 0.8       # Categorization confidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percept_id": self.percept_id,
            "category": self.category.value,
            "value": self.value.value,
            "salience": round(self.salience, 4),
            "source_channels": self.source_channels,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RememberedPresent:
    """
    The remembered present -- Edelman's central concept.

    The current perceptual scene is not a raw snapshot but is shaped by
    value-category memory, producing a present moment informed by the past.
    """
    scene_id: str
    percepts: List[CategorizedPercept]
    scene_coherence: SceneCoherence
    binding_state: SensoryBoundState
    dominant_value: ValueAssignment
    memory_influence: float        # 0.0-1.0, how much memory shapes scene
    novelty_fraction: float        # Proportion of scene that is novel
    temporal_continuity: float     # 0.0-1.0, continuity with previous scene
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "percept_count": len(self.percepts),
            "scene_coherence": self.scene_coherence.value,
            "binding_state": self.binding_state.value,
            "dominant_value": self.dominant_value.value,
            "memory_influence": round(self.memory_influence, 4),
            "novelty_fraction": round(self.novelty_fraction, 4),
            "temporal_continuity": round(self.temporal_continuity, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PrimaryConsciousnessOutput:
    """
    Complete output of a primary consciousness processing cycle.
    """
    awareness_level: PrimaryAwarenessLevel
    remembered_present: RememberedPresent
    categorized_percepts: List[CategorizedPercept]
    total_salience: float          # Sum of salience across percepts
    value_summary: Dict[str, float]  # Value -> aggregate salience
    processing_latency_ms: float
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "awareness_level": self.awareness_level.value,
            "remembered_present": self.remembered_present.to_dict(),
            "percept_count": len(self.categorized_percepts),
            "total_salience": round(self.total_salience, 4),
            "value_summary": {k: round(v, 4) for k, v in self.value_summary.items()},
            "processing_latency_ms": round(self.processing_latency_ms, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class PrimaryConsciousnessInterface:
    """
    Main interface for Form 18: Primary Consciousness (Edelman).

    Implements the core mechanisms of primary consciousness:
    1. Perceptual categorization -- classifying sensory input
    2. Value assignment -- tagging percepts with hedonic value
    3. Scene construction -- binding percepts into a unified scene
    4. The remembered present -- memory-shaped current awareness

    Primary consciousness is present-oriented, non-reflective, and
    does not require language or a concept of self.
    """

    FORM_ID = "18-primary-consciousness"
    FORM_NAME = "Primary Consciousness (Edelman)"

    def __init__(self):
        """Initialize the primary consciousness interface."""
        # Categorization state
        self._category_history: Dict[str, List[CategorizedPercept]] = {}
        self._value_memory: Dict[str, ValueAssignment] = {}

        # Scene state
        self._current_scene: Optional[RememberedPresent] = None
        self._scene_counter: int = 0
        self._percept_counter: int = 0

        # Processing parameters
        self._binding_threshold: float = 0.3
        self._salience_threshold: float = 0.2
        self._memory_decay: float = 0.95
        self._novelty_boost: float = 0.3

        # History
        self._scene_history: List[RememberedPresent] = []
        self._max_history: int = 50

        self._initialized = False
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the interface and prepare processing pipelines."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize category history for each perceptual category
        for cat in PerceptualCategory:
            self._category_history[cat.value] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def categorize_input(
        self, sensory_input: PrimarySensoryInput
    ) -> List[CategorizedPercept]:
        """
        Categorize raw sensory input into perceptual categories.

        This is the first step of primary consciousness: discriminating
        and classifying incoming sensory signals based on learned patterns.
        Each channel is analyzed for pattern type, complexity, and novelty,
        then assigned a perceptual category and hedonic value.

        Args:
            sensory_input: Bundled multimodal sensory data.

        Returns:
            List of categorized percepts extracted from the input.
        """
        percepts: List[CategorizedPercept] = []

        for channel_data in sensory_input.channels:
            # Determine perceptual category based on channel and pattern
            category = self._infer_category(channel_data)

            # Assign hedonic value from memory or novelty
            value = self._assign_value(channel_data, category)

            # Compute salience
            salience = self._compute_salience(
                channel_data, value, sensory_input.ambient_arousal
            )

            if salience >= self._salience_threshold:
                self._percept_counter += 1
                percept = CategorizedPercept(
                    percept_id=f"percept_{self._percept_counter:06d}",
                    category=category,
                    value=value,
                    salience=salience,
                    source_channels=[channel_data.channel],
                    confidence=channel_data.reliability * 0.9,
                )
                percepts.append(percept)

                # Update category history
                if category.value in self._category_history:
                    history = self._category_history[category.value]
                    history.append(percept)
                    if len(history) > self._max_history:
                        history.pop(0)

                # Update value memory
                self._value_memory[f"{category.value}_{channel_data.channel}"] = value

        return percepts

    async def construct_scene(
        self, percepts: List[CategorizedPercept]
    ) -> RememberedPresent:
        """
        Construct a unified perceptual scene from categorized percepts.

        Scene construction binds distributed percepts into a coherent
        whole through simulated reentrant signaling. The result is the
        "remembered present" -- the current scene shaped by memory.

        Args:
            percepts: Categorized percepts to bind into a scene.

        Returns:
            A RememberedPresent representing the constructed scene.
        """
        self._scene_counter += 1
        scene_id = f"scene_{self._scene_counter:06d}"

        # Determine binding state
        binding_state = self._evaluate_binding(percepts)

        # Determine scene coherence
        coherence = self._evaluate_coherence(percepts, binding_state)

        # Compute dominant value across percepts
        dominant_value = self._compute_dominant_value(percepts)

        # Compute memory influence
        memory_influence = self._compute_memory_influence(percepts)

        # Compute novelty fraction
        novelty_fraction = self._compute_novelty_fraction(percepts)

        # Compute temporal continuity with previous scene
        temporal_continuity = self._compute_temporal_continuity(percepts)

        scene = RememberedPresent(
            scene_id=scene_id,
            percepts=percepts,
            scene_coherence=coherence,
            binding_state=binding_state,
            dominant_value=dominant_value,
            memory_influence=memory_influence,
            novelty_fraction=novelty_fraction,
            temporal_continuity=temporal_continuity,
        )

        # Store in history
        self._current_scene = scene
        self._scene_history.append(scene)
        if len(self._scene_history) > self._max_history:
            self._scene_history.pop(0)

        return scene

    async def get_remembered_present(self) -> Optional[RememberedPresent]:
        """
        Return the current remembered present.

        The remembered present is the ongoing, memory-informed perceptual
        scene that constitutes primary conscious experience.

        Returns:
            The current RememberedPresent, or None if no scene has been
            constructed yet.
        """
        return self._current_scene

    async def assess_primary_awareness(self) -> PrimaryAwarenessLevel:
        """
        Assess the current level of primary awareness.

        Based on the coherence and binding of the current scene,
        determine the overall level of primary consciousness.

        Returns:
            The current PrimaryAwarenessLevel.
        """
        if self._current_scene is None:
            return PrimaryAwarenessLevel.ABSENT

        coherence = self._current_scene.scene_coherence
        binding = self._current_scene.binding_state

        if coherence == SceneCoherence.INCOHERENT:
            return PrimaryAwarenessLevel.ABSENT
        elif coherence == SceneCoherence.FRAGMENTARY:
            return PrimaryAwarenessLevel.MINIMAL
        elif coherence == SceneCoherence.PARTIAL:
            return PrimaryAwarenessLevel.PARTIAL
        elif coherence == SceneCoherence.VIVID:
            return PrimaryAwarenessLevel.HEIGHTENED
        else:
            # Coherent scene
            if binding == SensoryBoundState.TIGHTLY_BOUND:
                return PrimaryAwarenessLevel.HEIGHTENED
            return PrimaryAwarenessLevel.COHERENT

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "scene_count": self._scene_counter,
            "percept_count": self._percept_counter,
            "current_scene": (
                self._current_scene.to_dict() if self._current_scene else None
            ),
            "value_memory_size": len(self._value_memory),
            "history_length": len(self._scene_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current operational status of the interface."""
        awareness = PrimaryAwarenessLevel.ABSENT
        if self._current_scene:
            # Synchronous approximation
            coherence = self._current_scene.scene_coherence
            if coherence == SceneCoherence.COHERENT:
                awareness = PrimaryAwarenessLevel.COHERENT
            elif coherence == SceneCoherence.VIVID:
                awareness = PrimaryAwarenessLevel.HEIGHTENED
            elif coherence == SceneCoherence.PARTIAL:
                awareness = PrimaryAwarenessLevel.PARTIAL
            elif coherence == SceneCoherence.FRAGMENTARY:
                awareness = PrimaryAwarenessLevel.MINIMAL

        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "awareness_level": awareness.value,
            "scenes_constructed": self._scene_counter,
            "percepts_categorized": self._percept_counter,
            "binding_state": (
                self._current_scene.binding_state.value
                if self._current_scene else "none"
            ),
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _infer_category(self, channel_data: SensoryChannelData) -> PerceptualCategory:
        """Infer perceptual category from channel data."""
        channel = channel_data.channel.lower()

        # Map channels to likely categories
        channel_category_map = {
            "visual": PerceptualCategory.OBJECT,
            "auditory": PerceptualCategory.AUDITORY_PATTERN,
            "tactile": PerceptualCategory.TACTILE_QUALITY,
            "olfactory": PerceptualCategory.OLFACTORY_SIGNATURE,
            "motion": PerceptualCategory.MOTION,
            "spatial": PerceptualCategory.SPATIAL,
            "social": PerceptualCategory.SOCIAL_SIGNAL,
        }

        category = channel_category_map.get(channel, PerceptualCategory.OBJECT)

        # High novelty with high intensity may indicate threat
        if channel_data.novelty_score > 0.8 and channel_data.raw_intensity > 0.7:
            category = PerceptualCategory.THREAT_CUE

        # Low novelty with moderate positive pattern may indicate reward
        if (channel_data.novelty_score < 0.3 and
                channel_data.pattern_complexity > 0.5 and
                channel_data.raw_intensity > 0.4):
            category = PerceptualCategory.REWARD_CUE

        return category

    def _assign_value(
        self, channel_data: SensoryChannelData, category: PerceptualCategory
    ) -> ValueAssignment:
        """Assign hedonic value based on memory and novelty."""
        key = f"{category.value}_{channel_data.channel}"

        # Check value memory first
        if key in self._value_memory:
            return self._value_memory[key]

        # Novel stimuli get the NOVEL tag
        if channel_data.novelty_score > 0.7:
            return ValueAssignment.NOVEL

        # Threat cues are aversive
        if category == PerceptualCategory.THREAT_CUE:
            if channel_data.raw_intensity > 0.7:
                return ValueAssignment.STRONGLY_AVERSIVE
            return ValueAssignment.MILDLY_AVERSIVE

        # Reward cues are appetitive
        if category == PerceptualCategory.REWARD_CUE:
            if channel_data.raw_intensity > 0.7:
                return ValueAssignment.STRONGLY_APPETITIVE
            return ValueAssignment.MILDLY_APPETITIVE

        return ValueAssignment.NEUTRAL

    def _compute_salience(
        self,
        channel_data: SensoryChannelData,
        value: ValueAssignment,
        ambient_arousal: float,
    ) -> float:
        """Compute salience of a percept."""
        # Base salience from intensity and novelty
        base = (channel_data.raw_intensity * 0.4 +
                channel_data.novelty_score * 0.3 +
                channel_data.pattern_complexity * 0.3)

        # Value modulation
        value_boost = {
            ValueAssignment.STRONGLY_AVERSIVE: 0.4,
            ValueAssignment.STRONGLY_APPETITIVE: 0.3,
            ValueAssignment.NOVEL: 0.25,
            ValueAssignment.MILDLY_AVERSIVE: 0.15,
            ValueAssignment.MILDLY_APPETITIVE: 0.1,
            ValueAssignment.NEUTRAL: 0.0,
        }
        base += value_boost.get(value, 0.0)

        # Arousal modulation
        base *= (0.5 + 0.5 * ambient_arousal)

        return max(0.0, min(1.0, base))

    def _evaluate_binding(self, percepts: List[CategorizedPercept]) -> SensoryBoundState:
        """Evaluate the binding state of percepts."""
        if not percepts:
            return SensoryBoundState.UNBOUND

        # Count unique source channels
        all_channels = set()
        for p in percepts:
            all_channels.update(p.source_channels)

        channel_count = len(all_channels)
        avg_salience = sum(p.salience for p in percepts) / len(percepts)

        if channel_count <= 1:
            return SensoryBoundState.LOOSELY_BOUND
        elif avg_salience > 0.7 and channel_count >= 3:
            return SensoryBoundState.TIGHTLY_BOUND
        elif avg_salience > 0.4 and channel_count >= 2:
            return SensoryBoundState.BOUND
        else:
            return SensoryBoundState.LOOSELY_BOUND

    def _evaluate_coherence(
        self,
        percepts: List[CategorizedPercept],
        binding: SensoryBoundState,
    ) -> SceneCoherence:
        """Evaluate scene coherence."""
        if not percepts:
            return SceneCoherence.INCOHERENT

        count = len(percepts)
        avg_confidence = sum(p.confidence for p in percepts) / count

        if binding == SensoryBoundState.UNBOUND:
            return SceneCoherence.INCOHERENT
        elif binding == SensoryBoundState.FRAGMENTED:
            return SceneCoherence.FRAGMENTARY
        elif binding == SensoryBoundState.LOOSELY_BOUND:
            if avg_confidence > 0.6:
                return SceneCoherence.PARTIAL
            return SceneCoherence.FRAGMENTARY
        elif binding == SensoryBoundState.TIGHTLY_BOUND:
            if avg_confidence > 0.8:
                return SceneCoherence.VIVID
            return SceneCoherence.COHERENT
        else:
            # BOUND
            return SceneCoherence.COHERENT

    def _compute_dominant_value(
        self, percepts: List[CategorizedPercept]
    ) -> ValueAssignment:
        """Find the dominant value among percepts weighted by salience."""
        if not percepts:
            return ValueAssignment.NEUTRAL

        value_salience: Dict[ValueAssignment, float] = {}
        for p in percepts:
            value_salience[p.value] = value_salience.get(p.value, 0.0) + p.salience

        return max(value_salience, key=value_salience.get)

    def _compute_memory_influence(self, percepts: List[CategorizedPercept]) -> float:
        """Compute how much memory shapes the current scene."""
        if not percepts:
            return 0.0

        # Percepts categorized with high confidence suggest strong memory influence
        avg_confidence = sum(p.confidence for p in percepts) / len(percepts)

        # Non-novel percepts indicate memory-driven categorization
        novel_count = sum(1 for p in percepts if p.value == ValueAssignment.NOVEL)
        non_novel_ratio = 1.0 - (novel_count / len(percepts))

        return (avg_confidence * 0.5 + non_novel_ratio * 0.5)

    def _compute_novelty_fraction(self, percepts: List[CategorizedPercept]) -> float:
        """Compute the fraction of percepts that are novel."""
        if not percepts:
            return 0.0

        novel_count = sum(1 for p in percepts if p.value == ValueAssignment.NOVEL)
        return novel_count / len(percepts)

    def _compute_temporal_continuity(
        self, percepts: List[CategorizedPercept]
    ) -> float:
        """Compute continuity with the previous scene."""
        if not self._current_scene or not self._current_scene.percepts:
            return 0.0

        prev_categories = set(p.category for p in self._current_scene.percepts)
        curr_categories = set(p.category for p in percepts)

        if not prev_categories and not curr_categories:
            return 1.0

        overlap = len(prev_categories & curr_categories)
        union = len(prev_categories | curr_categories)
        return overlap / union if union > 0 else 0.0


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_primary_consciousness_interface() -> PrimaryConsciousnessInterface:
    """Create and return a primary consciousness interface instance."""
    return PrimaryConsciousnessInterface()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "PrimaryAwarenessLevel",
    "SensoryBoundState",
    "PerceptualCategory",
    "ValueAssignment",
    "SceneCoherence",
    # Input dataclasses
    "SensoryChannelData",
    "PrimarySensoryInput",
    # Output dataclasses
    "CategorizedPercept",
    "RememberedPresent",
    "PrimaryConsciousnessOutput",
    # Interface
    "PrimaryConsciousnessInterface",
    # Convenience
    "create_primary_consciousness_interface",
]
