#!/usr/bin/env python3
"""
Dream Consciousness Interface

Form 22: Dream consciousness -- the unique form of awareness present
during sleep, particularly during REM sleep. Dreams involve vivid
subjective experience despite reduced external sensory input, altered
self-awareness, and often bizarre or illogical content.

Core concepts modeled:
- Sleep stages: Wake, N1, N2, N3 (slow-wave), REM
- Dream generation: activation-synthesis, threat simulation, memory consolidation
- Dream content: narrative structure, emotional tone, bizarreness
- Lucid dreaming: awareness within the dream state
- Sleep-stage transitions: ultradian cycling through stages

Theoretical influences include Hobson (activation-synthesis), Revonsuo
(threat simulation theory), and Walker (memory consolidation hypothesis).
"""

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SleepStage(Enum):
    """
    Sleep stages based on polysomnographic classification.
    """
    WAKE = "wake"                  # Alert wakefulness
    N1 = "N1"                      # Light sleep, hypnagogic imagery
    N2 = "N2"                      # Intermediate sleep, sleep spindles
    N3 = "N3"                      # Deep slow-wave sleep, minimal dreaming
    REM = "REM"                    # Rapid eye movement, vivid dreaming


class DreamType(Enum):
    """
    Types of dream experiences.
    """
    NARRATIVE = "narrative"        # Story-like, sequential events
    BIZARRE = "bizarre"            # Impossible / illogical content
    NIGHTMARE = "nightmare"        # Threat-focused, intense negative emotion
    PROPHETIC = "prophetic"        # Seemingly predictive (pattern recognition)
    LUCID = "lucid"                # Dreamer is aware they are dreaming
    RECURRING = "recurring"        # Repeated dream theme or scenario
    HYPNAGOGIC = "hypnagogic"      # Imagery during sleep onset (N1)
    FRAGMENTARY = "fragmentary"    # Brief, disconnected images or feelings


class DreamEmotion(Enum):
    """
    Primary emotions experienced in dreams.
    """
    FEAR = "fear"                  # Most common dream emotion
    ANXIETY = "anxiety"            # Diffuse worry / dread
    JOY = "joy"                    # Happiness, elation
    SADNESS = "sadness"            # Grief, loss
    ANGER = "anger"                # Frustration, rage
    SURPRISE = "surprise"          # Unexpected events
    CONFUSION = "confusion"        # Disorientation, puzzlement
    AWE = "awe"                    # Wonder, transcendence
    NEUTRAL = "neutral"            # No strong emotional tone


class BizarrenessSource(Enum):
    """Sources of bizarreness in dream content."""
    SPATIAL = "spatial"            # Impossible spaces, teleportation
    TEMPORAL = "temporal"          # Time distortion, anachronism
    IDENTITY = "identity"          # Shape-shifting, merged identities
    PHYSICAL = "physical"          # Impossible physics (flying, etc.)
    NARRATIVE = "narrative"        # Plot discontinuities
    LOGICAL = "logical"            # Contradictions accepted as normal
    EMOTIONAL = "emotional"        # Inappropriate emotional reactions


class DreamGenerationModel(Enum):
    """Theoretical models for dream generation."""
    ACTIVATION_SYNTHESIS = "activation_synthesis"  # Hobson: random brainstem + cortical interpretation
    THREAT_SIMULATION = "threat_simulation"        # Revonsuo: rehearsal of threats
    MEMORY_CONSOLIDATION = "memory_consolidation"  # Walker: memory processing
    WISH_FULFILLMENT = "wish_fulfillment"          # Freud: disguised desires
    CONTINUITY = "continuity"                      # Domhoff: extension of waking concerns
    DEFAULT_NETWORK = "default_network"            # DMN activity during sleep


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SleepStateInput:
    """Current physiological sleep state."""
    stage: SleepStage
    time_in_stage_minutes: float   # Minutes in current stage
    cycle_number: int              # Which ultradian cycle (1-6 typical)
    cortisol_level: float          # 0.0-1.0, rises toward morning
    melatonin_level: float         # 0.0-1.0, high at night
    eeg_power_delta: float         # 0.0-1.0, slow-wave power
    eeg_power_theta: float         # 0.0-1.0, theta power
    eye_movement_density: float    # 0.0-1.0, REM density
    muscle_atonia: bool = False    # True during REM
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "time_in_stage_minutes": round(self.time_in_stage_minutes, 2),
            "cycle_number": self.cycle_number,
            "cortisol_level": round(self.cortisol_level, 4),
            "melatonin_level": round(self.melatonin_level, 4),
            "eeg_power_delta": round(self.eeg_power_delta, 4),
            "eeg_power_theta": round(self.eeg_power_theta, 4),
            "eye_movement_density": round(self.eye_movement_density, 4),
            "muscle_atonia": self.muscle_atonia,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecentMemory:
    """A recent memory available for dream incorporation."""
    memory_id: str
    content: str
    emotional_charge: float        # -1.0 to 1.0
    recency_hours: float           # Hours since encoding
    importance: float              # 0.0-1.0, personal significance
    is_threat: bool = False
    is_unresolved: bool = False    # Unresolved concerns (continuity hypothesis)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "emotional_charge": round(self.emotional_charge, 4),
            "recency_hours": round(self.recency_hours, 2),
            "importance": round(self.importance, 4),
            "is_threat": self.is_threat,
            "is_unresolved": self.is_unresolved,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamInput:
    """
    Complete input for dream generation.
    """
    sleep_state: SleepStateInput
    recent_memories: List[RecentMemory] = field(default_factory=list)
    generation_model: DreamGenerationModel = DreamGenerationModel.ACTIVATION_SYNTHESIS
    stress_level: float = 0.3      # 0.0-1.0, current stress
    creativity_factor: float = 0.5  # 0.0-1.0, tendency toward bizarre content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sleep_stage": self.sleep_state.stage.value,
            "memory_count": len(self.recent_memories),
            "generation_model": self.generation_model.value,
            "stress_level": round(self.stress_level, 4),
            "creativity_factor": round(self.creativity_factor, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class DreamElement:
    """A single element or scene within a dream."""
    element_id: str
    description: str
    source_memory: Optional[str]   # memory_id if traceable
    bizarreness: float             # 0.0-1.0
    bizarreness_sources: List[BizarrenessSource] = field(default_factory=list)
    emotional_tone: DreamEmotion = DreamEmotion.NEUTRAL
    vividness: float = 0.5        # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "description": self.description,
            "source_memory": self.source_memory,
            "bizarreness": round(self.bizarreness, 4),
            "bizarreness_sources": [s.value for s in self.bizarreness_sources],
            "emotional_tone": self.emotional_tone.value,
            "vividness": round(self.vividness, 4),
        }


@dataclass
class DreamOutput:
    """
    Complete output of a dream generation cycle.
    """
    dream_id: str
    dream_type: DreamType
    sleep_stage: SleepStage
    elements: List[DreamElement]
    primary_emotion: DreamEmotion
    emotional_intensity: float     # 0.0-1.0
    bizarreness_score: float       # 0.0-1.0, overall bizarreness
    narrative_coherence: float     # 0.0-1.0
    vividness: float               # 0.0-1.0
    lucidity: float                # 0.0-1.0, dreamer's self-awareness
    memory_sources_used: int       # How many memories contributed
    generation_model: DreamGenerationModel = DreamGenerationModel.ACTIVATION_SYNTHESIS
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "dream_type": self.dream_type.value,
            "sleep_stage": self.sleep_stage.value,
            "element_count": len(self.elements),
            "primary_emotion": self.primary_emotion.value,
            "emotional_intensity": round(self.emotional_intensity, 4),
            "bizarreness_score": round(self.bizarreness_score, 4),
            "narrative_coherence": round(self.narrative_coherence, 4),
            "vividness": round(self.vividness, 4),
            "lucidity": round(self.lucidity, 4),
            "memory_sources_used": self.memory_sources_used,
            "generation_model": self.generation_model.value,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class DreamConsciousnessInterface:
    """
    Main interface for Form 22: Dream Consciousness.

    Models dream generation, sleep-stage transitions, dream content
    analysis, and bizarreness computation. Supports multiple theoretical
    models of dream generation.
    """

    FORM_ID = "22-dream-consciousness"
    FORM_NAME = "Dream Consciousness"

    # Typical stage durations in minutes
    STAGE_DURATIONS = {
        SleepStage.WAKE: 0,
        SleepStage.N1: 5,
        SleepStage.N2: 25,
        SleepStage.N3: 40,
        SleepStage.REM: 20,
    }

    # Typical ultradian cycle order
    CYCLE_ORDER = [
        SleepStage.WAKE, SleepStage.N1, SleepStage.N2, SleepStage.N3,
        SleepStage.N2, SleepStage.REM,
    ]

    def __init__(self):
        """Initialize the dream consciousness interface."""
        # Dream history
        self._dream_history: List[DreamOutput] = []
        self._dream_counter: int = 0
        self._element_counter: int = 0

        # Current sleep state
        self._current_stage: SleepStage = SleepStage.WAKE
        self._current_cycle: int = 0
        self._time_in_stage: float = 0.0

        # Configuration
        self._max_history: int = 100
        self._rem_vividness_base: float = 0.7
        self._n1_vividness_base: float = 0.3
        self._n2_vividness_base: float = 0.2
        self._n3_vividness_base: float = 0.1

        self._initialized = False
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the dream consciousness interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def generate_dream(self, dream_input: DreamInput) -> DreamOutput:
        """
        Generate dream content based on sleep state and available memories.

        Uses the specified theoretical model to determine dream content,
        emotional tone, and narrative structure. REM stages produce the
        most vivid and narratively complex dreams.

        Args:
            dream_input: Sleep state, memories, and generation parameters.

        Returns:
            DreamOutput with dream content and analysis.
        """
        self._dream_counter += 1
        dream_id = f"dream_{self._dream_counter:06d}"
        stage = dream_input.sleep_state.stage

        # Generate dream elements
        elements = self._generate_elements(dream_input)

        # Determine dream type
        dream_type = self._classify_dream_type(elements, dream_input)

        # Compute metrics
        primary_emotion = self._determine_primary_emotion(elements, dream_input)
        emotional_intensity = self._compute_emotional_intensity(
            elements, dream_input
        )
        bizarreness = self._compute_overall_bizarreness(elements)
        coherence = self._compute_narrative_coherence(elements, stage)
        vividness = self._compute_vividness(stage, dream_input)
        lucidity = self._compute_lucidity(stage, dream_input)

        memory_count = sum(
            1 for e in elements if e.source_memory is not None
        )

        output = DreamOutput(
            dream_id=dream_id,
            dream_type=dream_type,
            sleep_stage=stage,
            elements=elements,
            primary_emotion=primary_emotion,
            emotional_intensity=emotional_intensity,
            bizarreness_score=bizarreness,
            narrative_coherence=coherence,
            vividness=vividness,
            lucidity=lucidity,
            memory_sources_used=memory_count,
            generation_model=dream_input.generation_model,
        )

        self._dream_history.append(output)
        if len(self._dream_history) > self._max_history:
            self._dream_history.pop(0)

        return output

    async def analyze_content(self, dream: DreamOutput) -> Dict[str, Any]:
        """
        Analyze the content of a generated or reported dream.

        Examines the dream's elements, emotional profile, bizarreness
        patterns, and narrative structure.

        Args:
            dream: The dream to analyze.

        Returns:
            Dictionary with content analysis results.
        """
        # Emotion distribution
        emotion_counts: Dict[str, int] = {}
        for elem in dream.elements:
            key = elem.emotional_tone.value
            emotion_counts[key] = emotion_counts.get(key, 0) + 1

        # Bizarreness source distribution
        biz_sources: Dict[str, int] = {}
        for elem in dream.elements:
            for src in elem.bizarreness_sources:
                biz_sources[src.value] = biz_sources.get(src.value, 0) + 1

        # Memory incorporation rate
        memory_elements = sum(1 for e in dream.elements if e.source_memory)
        incorporation_rate = (
            memory_elements / len(dream.elements) if dream.elements else 0.0
        )

        return {
            "dream_id": dream.dream_id,
            "dream_type": dream.dream_type.value,
            "element_count": len(dream.elements),
            "emotion_distribution": emotion_counts,
            "bizarreness_source_distribution": biz_sources,
            "memory_incorporation_rate": round(incorporation_rate, 4),
            "average_vividness": round(
                sum(e.vividness for e in dream.elements) / max(1, len(dream.elements)), 4
            ),
            "narrative_coherence": round(dream.narrative_coherence, 4),
            "lucidity": round(dream.lucidity, 4),
        }

    async def get_sleep_stage(self) -> SleepStage:
        """
        Return the current sleep stage.

        Returns:
            The current SleepStage.
        """
        return self._current_stage

    async def compute_bizarreness(self, dream: DreamOutput) -> Dict[str, Any]:
        """
        Compute a detailed bizarreness profile for a dream.

        Analyzes each element for sources of bizarreness and provides
        an overall bizarreness breakdown.

        Args:
            dream: The dream to analyze.

        Returns:
            Dictionary with detailed bizarreness metrics.
        """
        if not dream.elements:
            return {
                "overall_score": 0.0,
                "element_scores": [],
                "source_breakdown": {},
                "max_bizarreness": 0.0,
                "min_bizarreness": 0.0,
            }

        element_scores = [
            {"element_id": e.element_id, "score": e.bizarreness}
            for e in dream.elements
        ]

        source_counts: Dict[str, int] = {}
        for elem in dream.elements:
            for src in elem.bizarreness_sources:
                source_counts[src.value] = source_counts.get(src.value, 0) + 1

        scores = [e.bizarreness for e in dream.elements]
        return {
            "overall_score": round(dream.bizarreness_score, 4),
            "element_scores": element_scores,
            "source_breakdown": source_counts,
            "max_bizarreness": round(max(scores), 4),
            "min_bizarreness": round(min(scores), 4),
            "mean_bizarreness": round(sum(scores) / len(scores), 4),
        }

    # ========================================================================
    # SLEEP STAGE TRANSITION
    # ========================================================================

    def transition_stage(self, new_stage: SleepStage) -> Dict[str, Any]:
        """
        Transition to a new sleep stage.

        Args:
            new_stage: The stage to transition to.

        Returns:
            Dictionary describing the transition.
        """
        old_stage = self._current_stage
        self._current_stage = new_stage
        self._time_in_stage = 0.0

        if new_stage == SleepStage.REM:
            self._current_cycle += 1

        return {
            "from_stage": old_stage.value,
            "to_stage": new_stage.value,
            "cycle_number": self._current_cycle,
        }

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "current_stage": self._current_stage.value,
            "current_cycle": self._current_cycle,
            "dreams_generated": self._dream_counter,
            "history_length": len(self._dream_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current operational status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "sleep_stage": self._current_stage.value,
            "cycle_number": self._current_cycle,
            "dreams_generated": self._dream_counter,
            "elements_generated": self._element_counter,
        }

    # ========================================================================
    # PRIVATE METHODS - DREAM GENERATION
    # ========================================================================

    def _generate_elements(self, dream_input: DreamInput) -> List[DreamElement]:
        """Generate dream elements based on input."""
        stage = dream_input.sleep_state.stage
        memories = dream_input.recent_memories

        # Number of elements depends on stage
        element_counts = {
            SleepStage.WAKE: 0,
            SleepStage.N1: 2,
            SleepStage.N2: 3,
            SleepStage.N3: 1,
            SleepStage.REM: 5,
        }
        count = element_counts.get(stage, 0)
        if count == 0:
            return []

        elements: List[DreamElement] = []
        for i in range(count):
            self._element_counter += 1
            elem_id = f"elem_{self._element_counter:06d}"

            # Select source memory if available
            source_memory = None
            description = "Abstract dream imagery"
            if memories and i < len(memories):
                mem = memories[i]
                source_memory = mem.memory_id
                description = f"Dream element derived from: {mem.content[:50]}"

            # Compute bizarreness
            biz_score = self._compute_element_bizarreness(
                stage, dream_input.creativity_factor
            )
            biz_sources = self._select_bizarreness_sources(biz_score)

            # Determine emotional tone
            emotion = self._select_element_emotion(dream_input, i)

            # Vividness
            vividness = self._stage_vividness(stage) + dream_input.creativity_factor * 0.2
            vividness = max(0.0, min(1.0, vividness))

            elements.append(DreamElement(
                element_id=elem_id,
                description=description,
                source_memory=source_memory,
                bizarreness=biz_score,
                bizarreness_sources=biz_sources,
                emotional_tone=emotion,
                vividness=vividness,
            ))

        return elements

    def _compute_element_bizarreness(
        self, stage: SleepStage, creativity: float
    ) -> float:
        """Compute bizarreness for a single dream element."""
        base = {
            SleepStage.WAKE: 0.0,
            SleepStage.N1: 0.2,
            SleepStage.N2: 0.3,
            SleepStage.N3: 0.1,
            SleepStage.REM: 0.5,
        }.get(stage, 0.0)

        return max(0.0, min(1.0, base + creativity * 0.3))

    def _select_bizarreness_sources(
        self, score: float
    ) -> List[BizarrenessSource]:
        """Select bizarreness sources based on score."""
        if score < 0.2:
            return []

        # More sources for higher bizarreness
        all_sources = list(BizarrenessSource)
        count = max(1, int(score * len(all_sources)))
        count = min(count, len(all_sources))

        # Deterministic selection based on score level
        selected = all_sources[:count]
        return selected

    def _select_element_emotion(
        self, dream_input: DreamInput, element_index: int
    ) -> DreamEmotion:
        """Select emotion for a dream element."""
        stress = dream_input.stress_level
        memories = dream_input.recent_memories

        # Threat memories push toward fear/anxiety
        if memories and element_index < len(memories):
            mem = memories[element_index]
            if mem.is_threat:
                return DreamEmotion.FEAR
            if mem.emotional_charge < -0.5:
                return DreamEmotion.ANXIETY
            if mem.emotional_charge > 0.5:
                return DreamEmotion.JOY

        # High stress biases toward negative emotions
        if stress > 0.7:
            return DreamEmotion.ANXIETY
        elif stress > 0.5:
            return DreamEmotion.CONFUSION

        return DreamEmotion.NEUTRAL

    def _stage_vividness(self, stage: SleepStage) -> float:
        """Get base vividness for a sleep stage."""
        return {
            SleepStage.WAKE: 0.0,
            SleepStage.N1: self._n1_vividness_base,
            SleepStage.N2: self._n2_vividness_base,
            SleepStage.N3: self._n3_vividness_base,
            SleepStage.REM: self._rem_vividness_base,
        }.get(stage, 0.0)

    def _classify_dream_type(
        self, elements: List[DreamElement], dream_input: DreamInput
    ) -> DreamType:
        """Classify the dream type based on content."""
        if not elements:
            return DreamType.FRAGMENTARY

        stage = dream_input.sleep_state.stage

        # Hypnagogic for N1
        if stage == SleepStage.N1:
            return DreamType.HYPNAGOGIC

        # Check for nightmare (high fear/anxiety + threat memories)
        fear_count = sum(1 for e in elements
                        if e.emotional_tone in [DreamEmotion.FEAR, DreamEmotion.ANXIETY])
        if fear_count > len(elements) / 2:
            return DreamType.NIGHTMARE

        # Check for bizarre content
        avg_biz = sum(e.bizarreness for e in elements) / len(elements)
        if avg_biz > 0.6:
            return DreamType.BIZARRE

        # Check for lucidity
        if dream_input.sleep_state.stage == SleepStage.REM:
            lucidity = self._compute_lucidity(stage, dream_input)
            if lucidity > 0.6:
                return DreamType.LUCID

        # Default to narrative for REM, fragmentary for other stages
        if stage == SleepStage.REM:
            return DreamType.NARRATIVE
        return DreamType.FRAGMENTARY

    def _determine_primary_emotion(
        self, elements: List[DreamElement], dream_input: DreamInput
    ) -> DreamEmotion:
        """Determine the primary emotion of the dream."""
        if not elements:
            return DreamEmotion.NEUTRAL

        emotion_counts: Dict[DreamEmotion, int] = {}
        for e in elements:
            emotion_counts[e.emotional_tone] = emotion_counts.get(e.emotional_tone, 0) + 1

        return max(emotion_counts, key=emotion_counts.get)

    def _compute_emotional_intensity(
        self, elements: List[DreamElement], dream_input: DreamInput
    ) -> float:
        """Compute overall emotional intensity."""
        if not elements:
            return 0.0

        # Base from stress and stage
        base = dream_input.stress_level * 0.3

        # Boost from non-neutral emotions
        non_neutral = sum(
            1 for e in elements if e.emotional_tone != DreamEmotion.NEUTRAL
        )
        emotion_factor = non_neutral / len(elements)

        # REM has higher emotional intensity
        if dream_input.sleep_state.stage == SleepStage.REM:
            base += 0.3

        return max(0.0, min(1.0, base + emotion_factor * 0.5))

    def _compute_overall_bizarreness(self, elements: List[DreamElement]) -> float:
        """Compute overall bizarreness score."""
        if not elements:
            return 0.0
        return sum(e.bizarreness for e in elements) / len(elements)

    def _compute_narrative_coherence(
        self, elements: List[DreamElement], stage: SleepStage
    ) -> float:
        """Compute narrative coherence."""
        if not elements:
            return 0.0

        # REM tends to have more coherent narratives
        stage_coherence = {
            SleepStage.WAKE: 1.0,
            SleepStage.N1: 0.2,
            SleepStage.N2: 0.3,
            SleepStage.N3: 0.1,
            SleepStage.REM: 0.6,
        }.get(stage, 0.0)

        # Bizarreness reduces coherence
        avg_biz = sum(e.bizarreness for e in elements) / len(elements)
        coherence = stage_coherence * (1.0 - avg_biz * 0.5)

        return max(0.0, min(1.0, coherence))

    def _compute_vividness(
        self, stage: SleepStage, dream_input: DreamInput
    ) -> float:
        """Compute overall dream vividness."""
        base = self._stage_vividness(stage)
        # Later cycles tend to have more vivid REM
        cycle_boost = min(0.3, dream_input.sleep_state.cycle_number * 0.05)
        return max(0.0, min(1.0, base + cycle_boost))

    def _compute_lucidity(
        self, stage: SleepStage, dream_input: DreamInput
    ) -> float:
        """Compute lucidity level."""
        if stage != SleepStage.REM:
            return 0.0

        # Lucidity is rare, slightly more likely with higher creativity
        base = 0.05
        base += dream_input.creativity_factor * 0.15
        # Later cycles slightly increase lucidity potential
        base += dream_input.sleep_state.cycle_number * 0.02

        return max(0.0, min(1.0, base))


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_dream_consciousness_interface() -> DreamConsciousnessInterface:
    """Create and return a dream consciousness interface instance."""
    return DreamConsciousnessInterface()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "SleepStage",
    "DreamType",
    "DreamEmotion",
    "BizarrenessSource",
    "DreamGenerationModel",
    # Input dataclasses
    "SleepStateInput",
    "RecentMemory",
    "DreamInput",
    # Output dataclasses
    "DreamElement",
    "DreamOutput",
    # Interface
    "DreamConsciousnessInterface",
    # Convenience
    "create_dream_consciousness_interface",
]
