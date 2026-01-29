#!/usr/bin/env python3
"""
Narrative Consciousness Interface

Form 12: Narrative Consciousness manages self-narrative construction,
autobiographical memory, and temporal consciousness. It weaves discrete
events and experiences into a coherent life story, maintains personal
identity over time, and provides the temporal framework that gives
meaning to present experience through connection to past and future.

This form draws on Form 10 (Self-Recognition) for self-model and
Form 11 (Meta-Consciousness) for reflective awareness of the narrative
process itself.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class NarrativeMode(Enum):
    """Modes of narrative consciousness processing."""
    AUTOBIOGRAPHICAL = "autobiographical"    # Personal life story
    EPISODIC = "episodic"                    # Specific episode recall
    SEMANTIC = "semantic"                    # General self-knowledge
    PROSPECTIVE = "prospective"              # Future-oriented narrative
    CONFABULATION = "confabulation"          # Gap-filling narrative
    REFLECTIVE = "reflective"               # Self-examining narrative


class TemporalOrientation(Enum):
    """Temporal orientation of narrative consciousness."""
    PAST_DISTANT = "past_distant"            # Far past
    PAST_RECENT = "past_recent"              # Recent past
    PRESENT = "present"                      # Current moment
    FUTURE_NEAR = "future_near"              # Near future
    FUTURE_DISTANT = "future_distant"        # Far future
    TIMELESS = "timeless"                    # Non-temporal (semantic knowledge)


class MemoryType(Enum):
    """Types of memory in narrative consciousness."""
    EPISODIC = "episodic"                    # Specific event memories
    SEMANTIC = "semantic"                    # General knowledge
    PROCEDURAL = "procedural"               # How-to knowledge
    AUTOBIOGRAPHICAL = "autobiographical"    # Personal life events
    PROSPECTIVE = "prospective"             # Future intentions
    FLASHBULB = "flashbulb"                 # Vivid, emotional memories
    WORKING = "working"                      # Currently active memory


class NarrativeCoherence(Enum):
    """Levels of narrative coherence."""
    HIGHLY_COHERENT = "highly_coherent"      # Strong, unified narrative
    COHERENT = "coherent"                    # Reasonably connected
    FRAGMENTED = "fragmented"               # Partially connected
    DISJOINTED = "disjointed"               # Poorly connected
    INCOHERENT = "incoherent"               # No apparent connection


class IdentityTheme(Enum):
    """Core themes in narrative identity."""
    AGENCY = "agency"                        # Self as actor
    COMMUNION = "communion"                  # Self in relationship
    GROWTH = "growth"                        # Personal development
    RESILIENCE = "resilience"               # Overcoming adversity
    DISCOVERY = "discovery"                 # Learning and exploration
    CONTINUITY = "continuity"               # Stable self over time
    TRANSFORMATION = "transformation"        # Fundamental change


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class EventInput:
    """An event to be incorporated into the narrative."""
    event_id: str
    description: str
    significance: float              # 0.0-1.0 personal significance
    emotional_valence: float         # -1.0 to 1.0
    emotional_intensity: float       # 0.0-1.0
    temporal_location: TemporalOrientation
    participants: List[str] = field(default_factory=list)
    causal_links: List[str] = field(default_factory=list)  # IDs of related events
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "description": self.description,
            "significance": round(self.significance, 4),
            "emotional_valence": round(self.emotional_valence, 4),
            "emotional_intensity": round(self.emotional_intensity, 4),
            "temporal_location": self.temporal_location.value,
            "participants": self.participants,
            "causal_links": self.causal_links,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MemoryInput:
    """A memory retrieval input for narrative processing."""
    memory_id: str
    memory_type: MemoryType
    content_summary: str
    vividness: float                 # 0.0-1.0
    confidence: float                # 0.0-1.0 belief in accuracy
    emotional_charge: float          # 0.0-1.0
    temporal_distance: float         # 0.0-1.0 (recent to distant)
    retrieval_cue: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content_summary": self.content_summary,
            "vividness": round(self.vividness, 4),
            "confidence": round(self.confidence, 4),
            "emotional_charge": round(self.emotional_charge, 4),
            "temporal_distance": round(self.temporal_distance, 4),
            "retrieval_cue": self.retrieval_cue,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NarrativeInput:
    """Complete input for narrative processing."""
    events: List[EventInput] = field(default_factory=list)
    memories: List[MemoryInput] = field(default_factory=list)
    narrative_mode: NarrativeMode = NarrativeMode.AUTOBIOGRAPHICAL
    temporal_focus: TemporalOrientation = TemporalOrientation.PRESENT
    identity_question: Optional[str] = None  # "Who am I?" type queries
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class NarrativeSegment:
    """A segment of the constructed narrative."""
    segment_id: str
    content: str
    temporal_position: TemporalOrientation
    emotional_tone: float            # -1.0 to 1.0
    significance: float              # 0.0-1.0
    related_events: List[str]
    related_memories: List[str]
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "content": self.content,
            "temporal_position": self.temporal_position.value,
            "emotional_tone": round(self.emotional_tone, 4),
            "significance": round(self.significance, 4),
            "related_events": self.related_events,
            "related_memories": self.related_memories,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NarrativeStructure:
    """Overall structure of the constructed narrative."""
    segments: List[NarrativeSegment]
    coherence: NarrativeCoherence
    coherence_score: float           # 0.0-1.0
    dominant_theme: Optional[IdentityTheme]
    temporal_span: List[TemporalOrientation]
    narrative_arc: str               # "ascending", "descending", "stable", "complex"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "coherence": self.coherence.value,
            "coherence_score": round(self.coherence_score, 4),
            "dominant_theme": self.dominant_theme.value if self.dominant_theme else None,
            "temporal_span": [t.value for t in self.temporal_span],
            "narrative_arc": self.narrative_arc,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TemporalFrame:
    """Temporal frame of consciousness."""
    orientation: TemporalOrientation
    subjective_present_duration: float   # Perceived "now" duration in seconds
    temporal_coherence: float            # 0.0-1.0 how connected past-present-future
    future_orientation: float            # 0.0-1.0 how much attention to future
    past_orientation: float              # 0.0-1.0 how much attention to past
    present_orientation: float           # 0.0-1.0 how much attention to present
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orientation": self.orientation.value,
            "subjective_present_duration": round(self.subjective_present_duration, 4),
            "temporal_coherence": round(self.temporal_coherence, 4),
            "future_orientation": round(self.future_orientation, 4),
            "past_orientation": round(self.past_orientation, 4),
            "present_orientation": round(self.present_orientation, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IdentityCoherenceReport:
    """Report on narrative identity coherence."""
    overall_coherence: float             # 0.0-1.0
    dominant_themes: List[IdentityTheme]
    self_continuity: float               # 0.0-1.0 sense of being same person
    self_change: float                   # 0.0-1.0 sense of growth/change
    narrative_complexity: float          # 0.0-1.0
    unresolved_conflicts: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_coherence": round(self.overall_coherence, 4),
            "dominant_themes": [t.value for t in self.dominant_themes],
            "self_continuity": round(self.self_continuity, 4),
            "self_change": round(self.self_change, 4),
            "narrative_complexity": round(self.narrative_complexity, 4),
            "unresolved_conflicts": self.unresolved_conflicts,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NarrativeOutput:
    """Complete output from narrative processing."""
    narrative: NarrativeStructure
    temporal_frame: TemporalFrame
    identity_coherence: IdentityCoherenceReport
    active_mode: NarrativeMode
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative.to_dict(),
            "temporal_frame": self.temporal_frame.to_dict(),
            "identity_coherence": self.identity_coherence.to_dict(),
            "active_mode": self.active_mode.value,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NarrativeSystemStatus:
    """Complete narrative system status."""
    active_mode: NarrativeMode
    temporal_orientation: TemporalOrientation
    narrative_coherence: float           # 0.0-1.0
    memory_bank_size: int
    identity_stability: float            # 0.0-1.0
    system_health: float                 # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# NARRATIVE CONSTRUCTION ENGINE
# ============================================================================

class NarrativeConstructionEngine:
    """
    Engine for constructing narratives from events and memories.

    Weaves discrete events into coherent narrative structures
    with causal and thematic connections.
    """

    def __init__(self):
        self._narrative_history: List[NarrativeSegment] = []
        self._max_history = 100
        self._next_segment_id = 0

    def construct_narrative(
        self,
        events: List[EventInput],
        memories: List[MemoryInput],
        mode: NarrativeMode,
    ) -> NarrativeStructure:
        """Construct a narrative from events and memories."""
        segments = []

        # Create segments from events
        for event in events:
            segment = self._event_to_segment(event)
            segments.append(segment)

        # Create segments from memories
        for memory in memories:
            segment = self._memory_to_segment(memory)
            segments.append(segment)

        # Sort by temporal position
        temporal_order = {
            TemporalOrientation.PAST_DISTANT: 0,
            TemporalOrientation.PAST_RECENT: 1,
            TemporalOrientation.PRESENT: 2,
            TemporalOrientation.FUTURE_NEAR: 3,
            TemporalOrientation.FUTURE_DISTANT: 4,
            TemporalOrientation.TIMELESS: 5,
        }
        segments.sort(key=lambda s: temporal_order.get(s.temporal_position, 2))

        # Compute coherence
        coherence_score = self._compute_coherence(segments)
        coherence = self._classify_coherence(coherence_score)

        # Identify theme
        theme = self._identify_theme(events, memories)

        # Determine temporal span
        temporal_span = list(set(s.temporal_position for s in segments))

        # Determine narrative arc
        arc = self._determine_arc(segments)

        # Store segments
        self._narrative_history.extend(segments)
        if len(self._narrative_history) > self._max_history:
            self._narrative_history = self._narrative_history[-self._max_history:]

        return NarrativeStructure(
            segments=segments,
            coherence=coherence,
            coherence_score=coherence_score,
            dominant_theme=theme,
            temporal_span=temporal_span,
            narrative_arc=arc,
        )

    def _event_to_segment(self, event: EventInput) -> NarrativeSegment:
        """Convert an event to a narrative segment."""
        self._next_segment_id += 1
        return NarrativeSegment(
            segment_id=f"seg_{self._next_segment_id:04d}",
            content=event.description,
            temporal_position=event.temporal_location,
            emotional_tone=event.emotional_valence,
            significance=event.significance,
            related_events=[event.event_id] + event.causal_links,
            related_memories=[],
        )

    def _memory_to_segment(self, memory: MemoryInput) -> NarrativeSegment:
        """Convert a memory to a narrative segment."""
        self._next_segment_id += 1

        # Map temporal distance to orientation
        if memory.temporal_distance > 0.7:
            temporal = TemporalOrientation.PAST_DISTANT
        elif memory.temporal_distance > 0.3:
            temporal = TemporalOrientation.PAST_RECENT
        else:
            temporal = TemporalOrientation.PRESENT

        return NarrativeSegment(
            segment_id=f"seg_{self._next_segment_id:04d}",
            content=memory.content_summary,
            temporal_position=temporal,
            emotional_tone=0.0,  # Memories may not carry explicit valence
            significance=memory.emotional_charge,
            related_events=[],
            related_memories=[memory.memory_id],
            confidence=memory.confidence,
        )

    def _compute_coherence(self, segments: List[NarrativeSegment]) -> float:
        """Compute narrative coherence from segments."""
        if not segments:
            return 0.0
        if len(segments) == 1:
            return 0.8

        # Coherence based on: thematic consistency, causal connections, temporal ordering
        # Simplified: average significance weighted by number of connections
        total_connections = sum(
            len(s.related_events) + len(s.related_memories) for s in segments
        )
        avg_significance = sum(s.significance for s in segments) / len(segments)
        connection_density = min(1.0, total_connections / max(1, len(segments) * 3))

        return (avg_significance * 0.4 + connection_density * 0.3 + 0.3)

    def _classify_coherence(self, score: float) -> NarrativeCoherence:
        """Classify coherence score into category."""
        if score > 0.8:
            return NarrativeCoherence.HIGHLY_COHERENT
        elif score > 0.6:
            return NarrativeCoherence.COHERENT
        elif score > 0.4:
            return NarrativeCoherence.FRAGMENTED
        elif score > 0.2:
            return NarrativeCoherence.DISJOINTED
        else:
            return NarrativeCoherence.INCOHERENT

    def _identify_theme(
        self, events: List[EventInput], memories: List[MemoryInput]
    ) -> Optional[IdentityTheme]:
        """Identify the dominant narrative theme."""
        if not events and not memories:
            return None

        # Simple heuristic based on event properties
        avg_valence = 0.0
        avg_significance = 0.0
        if events:
            avg_valence = sum(e.emotional_valence for e in events) / len(events)
            avg_significance = sum(e.significance for e in events) / len(events)

        if avg_valence > 0.3 and avg_significance > 0.6:
            return IdentityTheme.GROWTH
        elif avg_valence < -0.3 and avg_significance > 0.6:
            return IdentityTheme.RESILIENCE
        elif avg_significance > 0.7:
            return IdentityTheme.AGENCY
        elif avg_valence > 0.2:
            return IdentityTheme.DISCOVERY
        else:
            return IdentityTheme.CONTINUITY

    def _determine_arc(self, segments: List[NarrativeSegment]) -> str:
        """Determine the narrative arc from emotional trajectory."""
        if not segments:
            return "stable"
        if len(segments) == 1:
            return "stable"

        tones = [s.emotional_tone for s in segments]
        first_half = sum(tones[:len(tones)//2]) / max(1, len(tones)//2)
        second_half = sum(tones[len(tones)//2:]) / max(1, len(tones) - len(tones)//2)

        diff = second_half - first_half
        if diff > 0.3:
            return "ascending"
        elif diff < -0.3:
            return "descending"
        elif abs(diff) < 0.1:
            return "stable"
        else:
            return "complex"


# ============================================================================
# TEMPORAL CONSCIOUSNESS ENGINE
# ============================================================================

class TemporalConsciousnessEngine:
    """
    Engine for temporal aspects of narrative consciousness.

    Manages the subjective sense of time, temporal orientation,
    and the binding of past, present, and future.
    """

    def __init__(self):
        self._temporal_history: List[TemporalOrientation] = []
        self._max_history = 50

    def compute_temporal_frame(
        self,
        events: List[EventInput],
        memories: List[MemoryInput],
        focus: TemporalOrientation,
    ) -> TemporalFrame:
        """Compute the current temporal frame of consciousness."""
        # Compute orientation weights
        past_weight, present_weight, future_weight = self._compute_orientation_weights(
            events, memories, focus
        )

        # Temporal coherence
        coherence = self._compute_temporal_coherence(events, memories)

        # Subjective present duration (more engagement = shorter present)
        total_events = len(events) + len(memories)
        present_duration = max(0.5, 3.0 - total_events * 0.3)

        self._temporal_history.append(focus)
        if len(self._temporal_history) > self._max_history:
            self._temporal_history.pop(0)

        return TemporalFrame(
            orientation=focus,
            subjective_present_duration=present_duration,
            temporal_coherence=coherence,
            future_orientation=future_weight,
            past_orientation=past_weight,
            present_orientation=present_weight,
        )

    def _compute_orientation_weights(
        self,
        events: List[EventInput],
        memories: List[MemoryInput],
        focus: TemporalOrientation,
    ) -> tuple:
        """Compute past, present, future orientation weights."""
        past_events = sum(
            1 for e in events
            if e.temporal_location in (TemporalOrientation.PAST_DISTANT, TemporalOrientation.PAST_RECENT)
        )
        present_events = sum(
            1 for e in events
            if e.temporal_location == TemporalOrientation.PRESENT
        )
        future_events = sum(
            1 for e in events
            if e.temporal_location in (TemporalOrientation.FUTURE_NEAR, TemporalOrientation.FUTURE_DISTANT)
        )
        past_memories = len(memories)  # Memories are inherently past-oriented

        total = past_events + present_events + future_events + past_memories + 1

        past_weight = (past_events + past_memories) / total
        present_weight = (present_events + 1) / total  # Base present awareness
        future_weight = future_events / total

        # Adjust by focus
        if focus in (TemporalOrientation.PAST_DISTANT, TemporalOrientation.PAST_RECENT):
            past_weight = min(1.0, past_weight + 0.2)
        elif focus == TemporalOrientation.PRESENT:
            present_weight = min(1.0, present_weight + 0.2)
        elif focus in (TemporalOrientation.FUTURE_NEAR, TemporalOrientation.FUTURE_DISTANT):
            future_weight = min(1.0, future_weight + 0.2)

        # Normalize
        total_w = past_weight + present_weight + future_weight
        if total_w > 0:
            past_weight /= total_w
            present_weight /= total_w
            future_weight /= total_w

        return past_weight, present_weight, future_weight

    def _compute_temporal_coherence(
        self, events: List[EventInput], memories: List[MemoryInput]
    ) -> float:
        """Compute how coherently past-present-future are connected."""
        if not events and not memories:
            return 0.5

        # Causal connections indicate temporal coherence
        total_links = sum(len(e.causal_links) for e in events)
        total_items = len(events) + len(memories)
        link_density = min(1.0, total_links / max(1, total_items))

        # Memory confidence contributes
        avg_confidence = 0.5
        if memories:
            avg_confidence = sum(m.confidence for m in memories) / len(memories)

        return (link_density * 0.5 + avg_confidence * 0.5)


# ============================================================================
# AUTOBIOGRAPHICAL MEMORY ENGINE
# ============================================================================

class AutobiographicalMemoryEngine:
    """
    Engine for managing autobiographical memory in narrative consciousness.

    Stores, retrieves, and organizes memories that form the
    basis of narrative identity.
    """

    def __init__(self):
        self._memory_bank: Dict[str, MemoryInput] = {}
        self._event_bank: Dict[str, EventInput] = {}

    def store_memory(self, memory: MemoryInput) -> None:
        """Store a memory in the autobiographical memory bank."""
        self._memory_bank[memory.memory_id] = memory

    def store_event(self, event: EventInput) -> None:
        """Store an event in the autobiographical memory bank."""
        self._event_bank[event.event_id] = event

    def retrieve_by_cue(self, cue: str) -> List[MemoryInput]:
        """Retrieve memories matching a retrieval cue."""
        matches = []
        for memory in self._memory_bank.values():
            if memory.retrieval_cue and cue.lower() in memory.retrieval_cue.lower():
                matches.append(memory)
            elif cue.lower() in memory.content_summary.lower():
                matches.append(memory)
        return matches

    def retrieve_by_type(self, memory_type: MemoryType) -> List[MemoryInput]:
        """Retrieve memories of a specific type."""
        return [m for m in self._memory_bank.values() if m.memory_type == memory_type]

    def compute_identity_coherence(self) -> IdentityCoherenceReport:
        """Compute overall identity coherence from stored memories."""
        memories = list(self._memory_bank.values())
        events = list(self._event_bank.values())

        # Overall coherence
        if not memories and not events:
            return IdentityCoherenceReport(
                overall_coherence=0.5,
                dominant_themes=[IdentityTheme.CONTINUITY],
                self_continuity=0.5,
                self_change=0.0,
                narrative_complexity=0.0,
                unresolved_conflicts=0,
            )

        # Compute metrics
        avg_confidence = 0.5
        if memories:
            avg_confidence = sum(m.confidence for m in memories) / len(memories)

        # Count themes
        themes = self._identify_themes(events)

        # Self-continuity based on memory confidence and consistency
        continuity = avg_confidence

        # Self-change based on emotional range
        valences = [e.emotional_valence for e in events] if events else [0.0]
        change = max(valences) - min(valences) if len(valences) > 1 else 0.0

        # Complexity
        complexity = min(1.0, (len(memories) + len(events)) / 20.0)

        return IdentityCoherenceReport(
            overall_coherence=avg_confidence,
            dominant_themes=themes,
            self_continuity=continuity,
            self_change=change,
            narrative_complexity=complexity,
            unresolved_conflicts=0,
        )

    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return len(self._memory_bank)

    def get_event_count(self) -> int:
        """Get total number of stored events."""
        return len(self._event_bank)

    def _identify_themes(self, events: List[EventInput]) -> List[IdentityTheme]:
        """Identify dominant narrative themes."""
        if not events:
            return [IdentityTheme.CONTINUITY]

        themes = []
        avg_valence = sum(e.emotional_valence for e in events) / len(events)
        avg_sig = sum(e.significance for e in events) / len(events)

        if avg_valence > 0.3:
            themes.append(IdentityTheme.GROWTH)
        if avg_sig > 0.6:
            themes.append(IdentityTheme.AGENCY)
        if avg_valence < -0.2 and avg_sig > 0.5:
            themes.append(IdentityTheme.RESILIENCE)

        if not themes:
            themes.append(IdentityTheme.CONTINUITY)

        return themes


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class NarrativeConsciousnessInterface:
    """
    Main interface for Form 12: Narrative Consciousness.

    Constructs self-narratives, manages temporal consciousness,
    and maintains autobiographical memory for identity coherence.
    """

    FORM_ID = "12-narrative-consciousness"
    FORM_NAME = "Narrative Consciousness"

    def __init__(self):
        """Initialize the narrative consciousness interface."""
        self.narrative_engine = NarrativeConstructionEngine()
        self.temporal_engine = TemporalConsciousnessEngine()
        self.memory_engine = AutobiographicalMemoryEngine()

        self._current_output: Optional[NarrativeOutput] = None
        self._current_narrative: Optional[NarrativeStructure] = None
        self._current_mode: NarrativeMode = NarrativeMode.AUTOBIOGRAPHICAL
        self._initialized: bool = False

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the narrative consciousness system."""
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized")

    async def process_narrative(self, narrative_input: NarrativeInput) -> NarrativeOutput:
        """
        Process narrative input and generate narrative output.

        This is the main entry point for narrative processing.
        """
        self._current_mode = narrative_input.narrative_mode

        # Store events and memories
        for event in narrative_input.events:
            self.memory_engine.store_event(event)
        for memory in narrative_input.memories:
            self.memory_engine.store_memory(memory)

        # Construct narrative
        narrative = self.narrative_engine.construct_narrative(
            narrative_input.events,
            narrative_input.memories,
            narrative_input.narrative_mode,
        )
        self._current_narrative = narrative

        # Compute temporal frame
        temporal_frame = self.temporal_engine.compute_temporal_frame(
            narrative_input.events,
            narrative_input.memories,
            narrative_input.temporal_focus,
        )

        # Compute identity coherence
        identity = self.memory_engine.compute_identity_coherence()

        output = NarrativeOutput(
            narrative=narrative,
            temporal_frame=temporal_frame,
            identity_coherence=identity,
            active_mode=narrative_input.narrative_mode,
        )
        self._current_output = output
        return output

    async def recall_memories(self, cue: str) -> List[MemoryInput]:
        """Recall memories matching a cue."""
        return self.memory_engine.retrieve_by_cue(cue)

    async def get_identity_report(self) -> IdentityCoherenceReport:
        """Get current identity coherence report."""
        return self.memory_engine.compute_identity_coherence()

    def get_current_narrative(self) -> Optional[NarrativeStructure]:
        """Get current narrative structure."""
        return self._current_narrative

    def get_status(self) -> NarrativeSystemStatus:
        """Get complete narrative system status."""
        coherence = 0.5
        if self._current_narrative:
            coherence = self._current_narrative.coherence_score

        identity = self.memory_engine.compute_identity_coherence()

        return NarrativeSystemStatus(
            active_mode=self._current_mode,
            temporal_orientation=TemporalOrientation.PRESENT,
            narrative_coherence=coherence,
            memory_bank_size=self.memory_engine.get_memory_count() + self.memory_engine.get_event_count(),
            identity_stability=identity.self_continuity,
            system_health=self._compute_health(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "active_mode": self._current_mode.value,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "memory_bank_size": self.memory_engine.get_memory_count() + self.memory_engine.get_event_count(),
            "initialized": self._initialized,
        }

    def _compute_health(self) -> float:
        """Compute narrative system health."""
        if not self._current_narrative:
            return 1.0
        return self._current_narrative.coherence_score


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_narrative_consciousness_interface() -> NarrativeConsciousnessInterface:
    """Create and return a narrative consciousness interface."""
    return NarrativeConsciousnessInterface()


__all__ = [
    # Enums
    "NarrativeMode",
    "TemporalOrientation",
    "MemoryType",
    "NarrativeCoherence",
    "IdentityTheme",
    # Input dataclasses
    "EventInput",
    "MemoryInput",
    "NarrativeInput",
    # Output dataclasses
    "NarrativeSegment",
    "NarrativeStructure",
    "TemporalFrame",
    "IdentityCoherenceReport",
    "NarrativeOutput",
    "NarrativeSystemStatus",
    # Engines
    "NarrativeConstructionEngine",
    "TemporalConsciousnessEngine",
    "AutobiographicalMemoryEngine",
    # Main interface
    "NarrativeConsciousnessInterface",
    # Convenience
    "create_narrative_consciousness_interface",
]
