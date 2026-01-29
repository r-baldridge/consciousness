#!/usr/bin/env python3
"""
Global Workspace Theory (GWT) Consciousness Interface

Form 14: Implements Global Workspace Theory as proposed by Bernard Baars
and further developed by Stanislas Dehaene and Jean-Pierre Changeux.
GWT proposes that consciousness arises when information wins a competition
for access to a global workspace, where it is then broadcast to a wide
network of specialized processors. The workspace acts as a central hub
that enables information sharing across otherwise independent modules.

This module manages workspace competition, content broadcasting,
ignition dynamics, and the broadcast history that characterizes
conscious access according to GWT.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class WorkspaceState(Enum):
    """
    States of the global workspace.

    The workspace cycles between these states as content
    competes for and gains conscious access.
    """
    IDLE = "idle"                    # No content currently in workspace
    COMPETITION = "competition"      # Multiple contents competing for access
    BROADCASTING = "broadcasting"    # Content is being broadcast globally
    REFRACTORY = "refractory"        # Brief pause after broadcast
    CONSOLIDATING = "consolidating"  # Integrating broadcast results


class ContentType(Enum):
    """Types of content that can enter the workspace."""
    PERCEPTUAL = "perceptual"            # Sensory/perceptual representations
    COGNITIVE = "cognitive"              # Thoughts, concepts, ideas
    EMOTIONAL = "emotional"              # Emotional content
    MOTOR = "motor"                      # Motor plans and intentions
    MEMORY = "memory"                    # Retrieved memories
    LINGUISTIC = "linguistic"            # Language-related content
    EXECUTIVE = "executive"              # Executive control signals
    METACOGNITIVE = "metacognitive"      # Meta-cognitive assessments


class ProcessorType(Enum):
    """Types of specialized processors in the global workspace architecture."""
    SENSORY = "sensory"                  # Sensory processing modules
    MOTOR = "motor"                      # Motor control modules
    MEMORY = "memory"                    # Long-term memory systems
    ATTENTION = "attention"              # Attentional control
    LANGUAGE = "language"                # Language processing
    EMOTION = "emotion"                  # Emotional evaluation
    EXECUTIVE = "executive"              # Executive functions
    EVALUATION = "evaluation"            # Value/relevance assessment


class BroadcastStrength(Enum):
    """Strength of workspace broadcast."""
    WEAK = "weak"                        # Limited broadcast reach
    MODERATE = "moderate"                # Normal broadcast
    STRONG = "strong"                    # Enhanced broadcast
    IGNITION = "ignition"                # Full ignition - maximum broadcast


class CompetitionOutcome(Enum):
    """Outcome of workspace competition."""
    WON = "won"                          # Content won access
    LOST = "lost"                        # Content lost competition
    PREEMPTED = "preempted"              # Content was displaced by higher priority
    EXPIRED = "expired"                  # Content timed out before access
    MERGED = "merged"                    # Content merged with winning content


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class WorkspaceContent:
    """Content submitted for workspace access."""
    content_id: str
    content_type: ContentType
    content_data: Dict[str, Any]
    salience: float                     # 0.0-1.0: How attention-grabbing
    source_processor: ProcessorType
    relevance: float = 0.5             # 0.0-1.0: Task relevance
    urgency: float = 0.0              # 0.0-1.0: Time pressure
    coalition_strength: float = 0.0    # 0.0-1.0: Support from allied processors
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def competition_score(self) -> float:
        """Compute the competition score for workspace access."""
        return (
            self.salience * 0.35 +
            self.relevance * 0.30 +
            self.urgency * 0.20 +
            self.coalition_strength * 0.15
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_type": self.content_type.value,
            "salience": round(self.salience, 4),
            "source_processor": self.source_processor.value,
            "relevance": round(self.relevance, 4),
            "urgency": round(self.urgency, 4),
            "coalition_strength": round(self.coalition_strength, 4),
            "competition_score": round(self.competition_score, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProcessorRegistration:
    """Registration of a specialized processor."""
    processor_id: str
    processor_type: ProcessorType
    receptive_content_types: List[ContentType]
    processing_capacity: float = 1.0
    is_active: bool = True


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class BroadcastEvent:
    """Record of a broadcast event."""
    broadcast_id: str
    content: WorkspaceContent
    broadcast_strength: BroadcastStrength
    receiving_processors: List[str]
    num_receivers: int
    duration_ms: float
    processor_responses: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "broadcast_id": self.broadcast_id,
            "content_id": self.content.content_id,
            "content_type": self.content.content_type.value,
            "broadcast_strength": self.broadcast_strength.value,
            "num_receivers": self.num_receivers,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CompetitionResult:
    """Result of a workspace competition cycle."""
    winner: Optional[WorkspaceContent]
    competitors: List[WorkspaceContent]
    outcome: CompetitionOutcome
    winning_score: float
    margin: float                       # Score margin over second place
    cycle_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner_id": self.winner.content_id if self.winner else None,
            "num_competitors": len(self.competitors),
            "outcome": self.outcome.value,
            "winning_score": round(self.winning_score, 4),
            "margin": round(self.margin, 4),
            "cycle_duration_ms": round(self.cycle_duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkspaceSnapshot:
    """Snapshot of the current workspace state."""
    state: WorkspaceState
    current_content: Optional[WorkspaceContent]
    pending_contents: List[WorkspaceContent]
    active_processors: List[str]
    broadcast_count: int
    competition_count: int
    avg_broadcast_strength: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "current_content": self.current_content.to_dict() if self.current_content else None,
            "num_pending": len(self.pending_contents),
            "num_active_processors": len(self.active_processors),
            "broadcast_count": self.broadcast_count,
            "competition_count": self.competition_count,
            "avg_broadcast_strength": round(self.avg_broadcast_strength, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GWTSystemStatus:
    """Complete status of the Global Workspace system."""
    is_initialized: bool
    workspace_state: WorkspaceState
    current_content: Optional[WorkspaceContent]
    num_registered_processors: int
    broadcast_count: int
    competition_count: int
    system_health: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# COMPETITION ENGINE
# ============================================================================

class WorkspaceCompetitionEngine:
    """
    Engine that manages competition for workspace access.

    Content from different processors competes based on salience,
    relevance, urgency, and coalition support. The winning content
    gains exclusive access to the global workspace for broadcasting.
    """

    def __init__(self, competition_threshold: float = 0.3):
        self.competition_threshold = competition_threshold
        self._competition_history: List[CompetitionResult] = []
        self._max_history = 50

    def run_competition(
        self, candidates: List[WorkspaceContent]
    ) -> CompetitionResult:
        """
        Run a competition cycle among candidate contents.

        The content with the highest competition score wins access
        to the global workspace.
        """
        if not candidates:
            return CompetitionResult(
                winner=None,
                competitors=[],
                outcome=CompetitionOutcome.EXPIRED,
                winning_score=0.0,
                margin=0.0,
                cycle_duration_ms=0.0,
            )

        # Sort by competition score (descending)
        sorted_candidates = sorted(
            candidates, key=lambda c: c.competition_score, reverse=True
        )

        winner = sorted_candidates[0]
        winning_score = winner.competition_score

        # Check if winner meets threshold
        if winning_score < self.competition_threshold:
            result = CompetitionResult(
                winner=None,
                competitors=candidates,
                outcome=CompetitionOutcome.EXPIRED,
                winning_score=winning_score,
                margin=0.0,
                cycle_duration_ms=1.0,
            )
        else:
            # Compute margin
            second_score = sorted_candidates[1].competition_score if len(sorted_candidates) > 1 else 0.0
            margin = winning_score - second_score

            result = CompetitionResult(
                winner=winner,
                competitors=candidates,
                outcome=CompetitionOutcome.WON,
                winning_score=winning_score,
                margin=margin,
                cycle_duration_ms=1.0 + len(candidates) * 0.5,
            )

        self._competition_history.append(result)
        if len(self._competition_history) > self._max_history:
            self._competition_history.pop(0)

        return result


# ============================================================================
# BROADCAST ENGINE
# ============================================================================

class BroadcastEngine:
    """
    Engine that manages broadcasting of workspace content.

    Once content wins the competition, it is broadcast to all
    registered processors, enabling global information sharing.
    """

    def __init__(self):
        self._broadcast_counter = 0
        self._broadcast_history: List[BroadcastEvent] = []
        self._max_history = 100

    def broadcast(
        self,
        content: WorkspaceContent,
        processors: List[ProcessorRegistration]
    ) -> BroadcastEvent:
        """
        Broadcast content to all compatible processors.

        Returns the broadcast event with information about
        which processors received the content.
        """
        self._broadcast_counter += 1

        # Determine which processors receive the broadcast
        receiving = []
        for proc in processors:
            if proc.is_active and content.content_type in proc.receptive_content_types:
                receiving.append(proc.processor_id)
            elif proc.is_active and proc.processor_type in [
                ProcessorType.ATTENTION, ProcessorType.EXECUTIVE, ProcessorType.EVALUATION
            ]:
                # These processors receive all broadcasts
                receiving.append(proc.processor_id)

        # Determine broadcast strength
        strength = self._determine_strength(content, len(receiving), len(processors))

        event = BroadcastEvent(
            broadcast_id=f"broadcast_{self._broadcast_counter}",
            content=content,
            broadcast_strength=strength,
            receiving_processors=receiving,
            num_receivers=len(receiving),
            duration_ms=2.0 + len(receiving) * 0.3,
        )

        self._broadcast_history.append(event)
        if len(self._broadcast_history) > self._max_history:
            self._broadcast_history.pop(0)

        return event

    def _determine_strength(
        self, content: WorkspaceContent, num_receivers: int, total_processors: int
    ) -> BroadcastStrength:
        """Determine the broadcast strength based on content and reach."""
        score = content.competition_score
        reach_ratio = num_receivers / max(1, total_processors)

        combined = score * 0.6 + reach_ratio * 0.4

        if combined >= 0.8:
            return BroadcastStrength.IGNITION
        elif combined >= 0.6:
            return BroadcastStrength.STRONG
        elif combined >= 0.4:
            return BroadcastStrength.MODERATE
        else:
            return BroadcastStrength.WEAK

    def get_history(self, limit: int = 10) -> List[BroadcastEvent]:
        """Get recent broadcast history."""
        return self._broadcast_history[-limit:]


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class GlobalWorkspaceInterface:
    """
    Main interface for Form 14: Global Workspace Theory.

    Implements GWT's model of consciousness as a global information
    broadcasting mechanism. Content from specialized processors
    competes for workspace access; the winner is broadcast globally,
    making the information available to all other processors.
    """

    FORM_ID = "14-global-workspace"
    FORM_NAME = "Global Workspace Theory (GWT)"

    def __init__(self):
        """Initialize the Global Workspace interface."""
        self._competition_engine = WorkspaceCompetitionEngine()
        self._broadcast_engine = BroadcastEngine()

        # Workspace state
        self._state = WorkspaceState.IDLE
        self._current_content: Optional[WorkspaceContent] = None
        self._pending_contents: List[WorkspaceContent] = []
        self._max_pending = 20

        # Registered processors
        self._processors: Dict[str, ProcessorRegistration] = {}

        # Tracking
        self._is_initialized = False
        self._broadcast_count = 0
        self._competition_count = 0
        self._broadcast_callbacks: List[Callable] = []

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the global workspace system."""
        self._is_initialized = True
        self._state = WorkspaceState.IDLE
        self._register_default_processors()
        logger.info(f"{self.FORM_NAME} initialized with {len(self._processors)} processors")

    def _register_default_processors(self) -> None:
        """Register default set of processors."""
        defaults = [
            ProcessorRegistration(
                "proc_sensory", ProcessorType.SENSORY,
                [ContentType.PERCEPTUAL], 1.0, True
            ),
            ProcessorRegistration(
                "proc_motor", ProcessorType.MOTOR,
                [ContentType.MOTOR, ContentType.EXECUTIVE], 1.0, True
            ),
            ProcessorRegistration(
                "proc_memory", ProcessorType.MEMORY,
                [ContentType.MEMORY, ContentType.PERCEPTUAL, ContentType.COGNITIVE], 1.0, True
            ),
            ProcessorRegistration(
                "proc_attention", ProcessorType.ATTENTION,
                list(ContentType), 1.0, True
            ),
            ProcessorRegistration(
                "proc_language", ProcessorType.LANGUAGE,
                [ContentType.LINGUISTIC, ContentType.COGNITIVE], 1.0, True
            ),
            ProcessorRegistration(
                "proc_emotion", ProcessorType.EMOTION,
                [ContentType.EMOTIONAL, ContentType.PERCEPTUAL], 1.0, True
            ),
            ProcessorRegistration(
                "proc_executive", ProcessorType.EXECUTIVE,
                list(ContentType), 1.0, True
            ),
            ProcessorRegistration(
                "proc_evaluation", ProcessorType.EVALUATION,
                list(ContentType), 1.0, True
            ),
        ]
        for proc in defaults:
            self._processors[proc.processor_id] = proc

    async def submit_content(self, content: WorkspaceContent) -> CompetitionOutcome:
        """
        Submit content for workspace competition.

        Content is added to the pending queue and a competition
        cycle is triggered if the workspace is available.
        """
        self._pending_contents.append(content)
        if len(self._pending_contents) > self._max_pending:
            # Remove oldest low-salience content
            self._pending_contents.sort(key=lambda c: c.competition_score, reverse=True)
            self._pending_contents = self._pending_contents[:self._max_pending]

        # Trigger competition if workspace is idle
        if self._state == WorkspaceState.IDLE:
            result = await self._run_competition_cycle()
            if result.winner and result.winner.content_id == content.content_id:
                return CompetitionOutcome.WON
            elif result.winner:
                return CompetitionOutcome.LOST
            return CompetitionOutcome.EXPIRED

        return CompetitionOutcome.LOST

    async def broadcast(self) -> Optional[BroadcastEvent]:
        """
        Broadcast the current workspace content to all processors.

        Returns the broadcast event or None if nothing to broadcast.
        """
        if self._current_content is None:
            return None

        self._state = WorkspaceState.BROADCASTING

        processors = list(self._processors.values())
        event = self._broadcast_engine.broadcast(self._current_content, processors)
        self._broadcast_count += 1

        # Notify callbacks
        for callback in self._broadcast_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Broadcast callback error: {e}")

        # Move to consolidating then idle
        self._state = WorkspaceState.CONSOLIDATING
        self._current_content = None
        self._state = WorkspaceState.IDLE

        return event

    async def get_workspace_state(self) -> WorkspaceSnapshot:
        """Get current workspace state snapshot."""
        broadcast_history = self._broadcast_engine.get_history()
        avg_strength = 0.0
        if broadcast_history:
            strength_values = {
                BroadcastStrength.WEAK: 0.25,
                BroadcastStrength.MODERATE: 0.5,
                BroadcastStrength.STRONG: 0.75,
                BroadcastStrength.IGNITION: 1.0,
            }
            avg_strength = sum(
                strength_values.get(e.broadcast_strength, 0.5)
                for e in broadcast_history
            ) / len(broadcast_history)

        return WorkspaceSnapshot(
            state=self._state,
            current_content=self._current_content,
            pending_contents=self._pending_contents.copy(),
            active_processors=[
                pid for pid, p in self._processors.items() if p.is_active
            ],
            broadcast_count=self._broadcast_count,
            competition_count=self._competition_count,
            avg_broadcast_strength=avg_strength,
        )

    async def get_broadcast_history(self, limit: int = 10) -> List[BroadcastEvent]:
        """Get recent broadcast history."""
        return self._broadcast_engine.get_history(limit)

    def register_processor(self, registration: ProcessorRegistration) -> None:
        """Register a new processor."""
        self._processors[registration.processor_id] = registration
        logger.info(f"Registered processor: {registration.processor_id}")

    def unregister_processor(self, processor_id: str) -> None:
        """Unregister a processor."""
        self._processors.pop(processor_id, None)

    def on_broadcast(self, callback: Callable) -> None:
        """Register a callback for broadcast events."""
        self._broadcast_callbacks.append(callback)

    async def _run_competition_cycle(self) -> CompetitionResult:
        """Run a competition cycle with pending contents."""
        self._state = WorkspaceState.COMPETITION
        self._competition_count += 1

        result = self._competition_engine.run_competition(self._pending_contents)

        if result.winner:
            self._current_content = result.winner
            # Remove winner from pending
            self._pending_contents = [
                c for c in self._pending_contents
                if c.content_id != result.winner.content_id
            ]
            # Auto-broadcast
            await self.broadcast()
        else:
            self._state = WorkspaceState.IDLE

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "is_initialized": self._is_initialized,
            "workspace_state": self._state.value,
            "current_content": self._current_content.to_dict() if self._current_content else None,
            "num_pending": len(self._pending_contents),
            "num_processors": len(self._processors),
            "broadcast_count": self._broadcast_count,
            "competition_count": self._competition_count,
        }

    def get_status(self) -> GWTSystemStatus:
        """Get current system status."""
        return GWTSystemStatus(
            is_initialized=self._is_initialized,
            workspace_state=self._state,
            current_content=self._current_content,
            num_registered_processors=len(self._processors),
            broadcast_count=self._broadcast_count,
            competition_count=self._competition_count,
            system_health=1.0 if self._is_initialized else 0.5,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_global_workspace_interface() -> GlobalWorkspaceInterface:
    """Create and return a Global Workspace interface."""
    return GlobalWorkspaceInterface()


def create_workspace_content(
    content_id: str,
    content_type: ContentType = ContentType.PERCEPTUAL,
    salience: float = 0.5,
    source: ProcessorType = ProcessorType.SENSORY,
    data: Optional[Dict[str, Any]] = None,
) -> WorkspaceContent:
    """Create a workspace content item for testing."""
    return WorkspaceContent(
        content_id=content_id,
        content_type=content_type,
        content_data=data or {"description": f"Content {content_id}"},
        salience=salience,
        source_processor=source,
    )


__all__ = [
    # Enums
    "WorkspaceState",
    "ContentType",
    "ProcessorType",
    "BroadcastStrength",
    "CompetitionOutcome",
    # Input dataclasses
    "WorkspaceContent",
    "ProcessorRegistration",
    # Output dataclasses
    "BroadcastEvent",
    "CompetitionResult",
    "WorkspaceSnapshot",
    "GWTSystemStatus",
    # Engines
    "WorkspaceCompetitionEngine",
    "BroadcastEngine",
    # Main interface
    "GlobalWorkspaceInterface",
    # Convenience functions
    "create_global_workspace_interface",
    "create_workspace_content",
]
