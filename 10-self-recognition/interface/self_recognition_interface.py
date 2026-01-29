#!/usr/bin/env python3
"""
Self-Recognition Consciousness Interface

Form 10: Self-Recognition Consciousness manages self-awareness, self-model
construction, body ownership, and sense of agency. It maintains the
fundamental distinction between self and other, tracks bodily boundaries,
and supports the minimal phenomenal self that underlies all conscious
experience.

This form integrates signals from Form 06 (Interoceptive) for body
awareness, Form 07 (Emotional) for affective self-states, and Form 11
(Meta-Consciousness) for reflective self-awareness.
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

class SelfAspect(Enum):
    """Aspects of self-awareness, from minimal to narrative."""
    BODILY = "bodily"                # Pre-reflective body ownership
    MINIMAL = "minimal"              # Basic sense of being a subject
    NARRATIVE = "narrative"          # Autobiographical self-identity
    SOCIAL = "social"                # Self as perceived by others
    EXPERIENTIAL = "experiential"    # Self as the locus of experience


class AgencyLevel(Enum):
    """Levels of experienced agency over actions."""
    FULL = "full"                    # Complete voluntary control
    PARTIAL = "partial"              # Some control, some automatic
    DIMINISHED = "diminished"        # Limited sense of agency
    ABSENT = "absent"                # No sense of agency (alien hand, etc.)
    INVOLUNTARY = "involuntary"      # Action perceived as not self-initiated


class OwnershipState(Enum):
    """States of body/experience ownership."""
    OWNED = "owned"                  # Full ownership ("this is my body")
    UNCERTAIN = "uncertain"          # Ambiguous ownership
    DISOWNED = "disowned"            # Feeling of disownership
    EXTENDED = "extended"            # Ownership extended to tools/prosthetics
    VIRTUAL = "virtual"              # Ownership of virtual/avatar body


class SelfBoundaryType(Enum):
    """Types of self-other boundaries."""
    PHYSICAL = "physical"            # Body boundary
    PSYCHOLOGICAL = "psychological"  # Mental/emotional boundary
    SOCIAL = "social"                # Interpersonal boundary
    TEMPORAL = "temporal"            # Past-future self continuity


class SelfRecognitionMode(Enum):
    """Modes of self-recognition processing."""
    MIRROR = "mirror"                # Visual self-recognition
    PROPRIOCEPTIVE = "proprioceptive"  # Body position awareness
    INTEROCEPTIVE = "interoceptive"  # Internal state awareness
    SOCIAL = "social"                # Self through others' perspectives
    REFLECTIVE = "reflective"        # Metacognitive self-awareness


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class BodySignalInput:
    """Bodily signals contributing to self-model."""
    proprioceptive_coherence: float    # 0.0-1.0 body position consistency
    interoceptive_intensity: float     # 0.0-1.0 internal sensation strength
    vestibular_stability: float        # 0.0-1.0 balance/orientation
    pain_level: float                  # 0.0-1.0
    body_temperature: float            # Normalized 0.0-1.0
    heartbeat_awareness: float         # 0.0-1.0 interoceptive accuracy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proprioceptive_coherence": self.proprioceptive_coherence,
            "interoceptive_intensity": self.interoceptive_intensity,
            "vestibular_stability": self.vestibular_stability,
            "pain_level": self.pain_level,
            "body_temperature": self.body_temperature,
            "heartbeat_awareness": self.heartbeat_awareness,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SocialContextInput:
    """Social signals contributing to self-awareness."""
    social_presence: bool              # Others present?
    being_observed: bool               # Feeling of being watched
    social_role: str                   # "leader", "follower", "peer", "stranger"
    empathy_activation: float          # 0.0-1.0 mirror neuron activity
    social_evaluation_threat: float    # 0.0-1.0 fear of judgment
    perspective_taking: float          # 0.0-1.0 theory of mind engagement
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActionFeedback:
    """Feedback about actions for agency assessment."""
    action_id: str
    intended: bool                     # Was the action intended?
    predicted_outcome: str
    actual_outcome: str
    outcome_match: float               # 0.0-1.0 how well outcome matched prediction
    effort_level: float                # 0.0-1.0
    timing_accuracy: float             # 0.0-1.0 temporal prediction accuracy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SelfInput:
    """Complete input for self-recognition processing."""
    body_signals: Optional[BodySignalInput] = None
    social_context: Optional[SocialContextInput] = None
    action_feedback: Optional[ActionFeedback] = None
    visual_self_input: Optional[bool] = None    # Mirror/photo self-recognition
    name_recognition: Optional[bool] = None     # Hearing own name
    memory_continuity: float = 0.8              # Sense of personal continuity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class SelfModelOutput:
    """Current self-model representation."""
    active_aspects: List[SelfAspect]
    body_ownership: OwnershipState
    self_coherence: float              # 0.0-1.0 overall self-model coherence
    self_distinctness: float           # 0.0-1.0 clarity of self-other boundary
    embodiment_level: float            # 0.0-1.0 sense of being embodied
    self_continuity: float             # 0.0-1.0 temporal self-continuity
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_aspects": [a.value for a in self.active_aspects],
            "body_ownership": self.body_ownership.value,
            "self_coherence": round(self.self_coherence, 4),
            "self_distinctness": round(self.self_distinctness, 4),
            "embodiment_level": round(self.embodiment_level, 4),
            "self_continuity": round(self.self_continuity, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgencyAssessment:
    """Assessment of sense of agency."""
    agency_level: AgencyLevel
    agency_score: float                # 0.0-1.0
    prediction_accuracy: float         # 0.0-1.0 how well predictions matched
    control_feeling: float             # 0.0-1.0 subjective control
    authorship_confidence: float       # 0.0-1.0 "I did this" confidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agency_level": self.agency_level.value,
            "agency_score": round(self.agency_score, 4),
            "prediction_accuracy": round(self.prediction_accuracy, 4),
            "control_feeling": round(self.control_feeling, 4),
            "authorship_confidence": round(self.authorship_confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SelfOutput:
    """Complete output from self-recognition processing."""
    self_model: SelfModelOutput
    agency: AgencyAssessment
    mirror_recognition: bool           # Can recognize self
    self_other_distinction: float      # 0.0-1.0
    boundary_integrity: float          # 0.0-1.0
    recognition_mode: SelfRecognitionMode
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_model": self.self_model.to_dict(),
            "agency": self.agency.to_dict(),
            "mirror_recognition": self.mirror_recognition,
            "self_other_distinction": round(self.self_other_distinction, 4),
            "boundary_integrity": round(self.boundary_integrity, 4),
            "recognition_mode": self.recognition_mode.value,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SelfSystemStatus:
    """Complete self-recognition system status."""
    current_self_model: Optional[SelfModelOutput]
    current_agency: Optional[AgencyAssessment]
    active_recognition_mode: SelfRecognitionMode
    self_integrity: float              # 0.0-1.0 overall self-system health
    system_health: float               # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# SELF-MODEL ENGINE
# ============================================================================

class SelfModelEngine:
    """
    Engine for constructing and maintaining the self-model.

    Integrates bodily, social, and cognitive signals to maintain
    a coherent representation of self.
    """

    def __init__(self):
        self._body_history: List[float] = []
        self._social_history: List[float] = []
        self._max_history = 50

    def update_self_model(self, self_input: SelfInput) -> SelfModelOutput:
        """Update the self-model based on new inputs."""
        active_aspects = self._determine_active_aspects(self_input)
        body_ownership = self._assess_body_ownership(self_input)
        coherence = self._compute_coherence(self_input)
        distinctness = self._compute_distinctness(self_input)
        embodiment = self._compute_embodiment(self_input)
        continuity = self_input.memory_continuity

        # Track history
        if self_input.body_signals:
            self._body_history.append(self_input.body_signals.proprioceptive_coherence)
        if len(self._body_history) > self._max_history:
            self._body_history.pop(0)

        confidence = (coherence + distinctness) / 2

        return SelfModelOutput(
            active_aspects=active_aspects,
            body_ownership=body_ownership,
            self_coherence=coherence,
            self_distinctness=distinctness,
            embodiment_level=embodiment,
            self_continuity=continuity,
            confidence=confidence,
        )

    def _determine_active_aspects(self, self_input: SelfInput) -> List[SelfAspect]:
        """Determine which self-aspects are currently active."""
        aspects = [SelfAspect.MINIMAL]  # Always active

        if self_input.body_signals:
            if self_input.body_signals.proprioceptive_coherence > 0.3:
                aspects.append(SelfAspect.BODILY)

        if self_input.social_context:
            if self_input.social_context.social_presence:
                aspects.append(SelfAspect.SOCIAL)

        if self_input.memory_continuity > 0.5:
            aspects.append(SelfAspect.NARRATIVE)

        aspects.append(SelfAspect.EXPERIENTIAL)
        return aspects

    def _assess_body_ownership(self, self_input: SelfInput) -> OwnershipState:
        """Assess current body ownership state."""
        if not self_input.body_signals:
            return OwnershipState.UNCERTAIN

        coherence = self_input.body_signals.proprioceptive_coherence
        stability = self_input.body_signals.vestibular_stability

        combined = (coherence + stability) / 2
        if combined > 0.7:
            return OwnershipState.OWNED
        elif combined > 0.4:
            return OwnershipState.UNCERTAIN
        else:
            return OwnershipState.DISOWNED

    def _compute_coherence(self, self_input: SelfInput) -> float:
        """Compute overall self-model coherence."""
        scores = [self_input.memory_continuity]

        if self_input.body_signals:
            scores.append(self_input.body_signals.proprioceptive_coherence)
            scores.append(self_input.body_signals.vestibular_stability)

        if self_input.action_feedback:
            scores.append(self_input.action_feedback.outcome_match)

        return sum(scores) / len(scores) if scores else 0.5

    def _compute_distinctness(self, self_input: SelfInput) -> float:
        """Compute self-other boundary distinctness."""
        base = 0.7

        if self_input.social_context:
            # High empathy can blur boundaries
            base -= self_input.social_context.empathy_activation * 0.2
            # Perspective-taking also affects boundaries
            base -= self_input.social_context.perspective_taking * 0.1
            # Being observed heightens self-awareness
            if self_input.social_context.being_observed:
                base += 0.1

        if self_input.body_signals:
            base += self_input.body_signals.proprioceptive_coherence * 0.2

        return max(0.0, min(1.0, base))

    def _compute_embodiment(self, self_input: SelfInput) -> float:
        """Compute sense of embodiment."""
        if not self_input.body_signals:
            return 0.5

        bs = self_input.body_signals
        return (
            bs.proprioceptive_coherence * 0.3 +
            bs.interoceptive_intensity * 0.2 +
            bs.vestibular_stability * 0.2 +
            bs.heartbeat_awareness * 0.2 +
            (1.0 - bs.pain_level) * 0.1
        )


# ============================================================================
# AGENCY ENGINE
# ============================================================================

class AgencyEngine:
    """
    Engine for assessing sense of agency.

    Uses the comparator model: agency arises when predicted outcomes
    match actual outcomes of actions.
    """

    def __init__(self):
        self._agency_history: List[float] = []
        self._max_history = 30

    def assess_agency(self, action_feedback: Optional[ActionFeedback]) -> AgencyAssessment:
        """Assess sense of agency from action feedback."""
        if not action_feedback:
            return AgencyAssessment(
                agency_level=AgencyLevel.PARTIAL,
                agency_score=0.5,
                prediction_accuracy=0.5,
                control_feeling=0.5,
                authorship_confidence=0.5,
            )

        # Comparator model: prediction vs outcome
        prediction_accuracy = action_feedback.outcome_match
        timing = action_feedback.timing_accuracy
        intended = 1.0 if action_feedback.intended else 0.0

        # Agency score combines multiple signals
        agency_score = (
            prediction_accuracy * 0.35 +
            timing * 0.25 +
            intended * 0.25 +
            action_feedback.effort_level * 0.15
        )
        agency_score = max(0.0, min(1.0, agency_score))

        # Classify agency level
        agency_level = self._classify_agency(agency_score, action_feedback.intended)

        # Control feeling
        control_feeling = (agency_score + intended) / 2

        # Authorship confidence
        authorship = agency_score * (0.8 if action_feedback.intended else 0.3)

        self._agency_history.append(agency_score)
        if len(self._agency_history) > self._max_history:
            self._agency_history.pop(0)

        return AgencyAssessment(
            agency_level=agency_level,
            agency_score=agency_score,
            prediction_accuracy=prediction_accuracy,
            control_feeling=control_feeling,
            authorship_confidence=authorship,
        )

    def _classify_agency(self, score: float, intended: bool) -> AgencyLevel:
        """Classify agency level from score."""
        if not intended:
            return AgencyLevel.INVOLUNTARY

        if score > 0.8:
            return AgencyLevel.FULL
        elif score > 0.5:
            return AgencyLevel.PARTIAL
        elif score > 0.2:
            return AgencyLevel.DIMINISHED
        else:
            return AgencyLevel.ABSENT


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class SelfRecognitionInterface:
    """
    Main interface for Form 10: Self-Recognition Consciousness.

    Maintains self-model, assesses body ownership and agency,
    and supports mirror-test-level self-recognition.
    """

    FORM_ID = "10-self-recognition"
    FORM_NAME = "Self-Recognition Consciousness"

    def __init__(self):
        """Initialize the self-recognition interface."""
        self.self_model_engine = SelfModelEngine()
        self.agency_engine = AgencyEngine()

        self._current_self_model: Optional[SelfModelOutput] = None
        self._current_agency: Optional[AgencyAssessment] = None
        self._current_output: Optional[SelfOutput] = None
        self._recognition_mode: SelfRecognitionMode = SelfRecognitionMode.PROPRIOCEPTIVE
        self._mirror_capable: bool = True
        self._initialized: bool = False

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the self-recognition system."""
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized")

    async def process_self_recognition(self, self_input: SelfInput) -> SelfOutput:
        """
        Process self-recognition from input signals.

        This is the main entry point for self-recognition processing.
        """
        # Update self-model
        self_model = self.self_model_engine.update_self_model(self_input)
        self._current_self_model = self_model

        # Assess agency
        agency = self.agency_engine.assess_agency(self_input.action_feedback)
        self._current_agency = agency

        # Determine recognition mode
        mode = self._determine_mode(self_input)
        self._recognition_mode = mode

        # Mirror recognition check
        mirror = self._check_mirror_recognition(self_input)

        # Self-other distinction
        self_other = self_model.self_distinctness

        # Boundary integrity
        boundary = self._compute_boundary_integrity(self_model, agency)

        output = SelfOutput(
            self_model=self_model,
            agency=agency,
            mirror_recognition=mirror,
            self_other_distinction=self_other,
            boundary_integrity=boundary,
            recognition_mode=mode,
            confidence=self_model.confidence,
        )
        self._current_output = output
        return output

    async def perform_mirror_test(self) -> bool:
        """Perform a mirror self-recognition test."""
        if not self._mirror_capable:
            return False
        if self._current_self_model:
            return self._current_self_model.self_coherence > 0.5
        return True

    def get_self_model(self) -> Optional[SelfModelOutput]:
        """Get current self-model."""
        return self._current_self_model

    def get_agency(self) -> Optional[AgencyAssessment]:
        """Get current agency assessment."""
        return self._current_agency

    def get_status(self) -> SelfSystemStatus:
        """Get complete self-recognition system status."""
        integrity = 0.5
        if self._current_self_model and self._current_agency:
            integrity = (
                self._current_self_model.self_coherence * 0.5 +
                self._current_agency.agency_score * 0.3 +
                self._current_self_model.self_distinctness * 0.2
            )

        return SelfSystemStatus(
            current_self_model=self._current_self_model,
            current_agency=self._current_agency,
            active_recognition_mode=self._recognition_mode,
            self_integrity=integrity,
            system_health=self._compute_health(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "self_model": self._current_self_model.to_dict() if self._current_self_model else None,
            "agency": self._current_agency.to_dict() if self._current_agency else None,
            "recognition_mode": self._recognition_mode.value,
            "mirror_capable": self._mirror_capable,
            "initialized": self._initialized,
        }

    def _determine_mode(self, self_input: SelfInput) -> SelfRecognitionMode:
        """Determine the active recognition mode."""
        if self_input.visual_self_input:
            return SelfRecognitionMode.MIRROR
        elif self_input.social_context and self_input.social_context.social_presence:
            return SelfRecognitionMode.SOCIAL
        elif self_input.body_signals and self_input.body_signals.interoceptive_intensity > 0.6:
            return SelfRecognitionMode.INTEROCEPTIVE
        elif self_input.body_signals:
            return SelfRecognitionMode.PROPRIOCEPTIVE
        else:
            return SelfRecognitionMode.REFLECTIVE

    def _check_mirror_recognition(self, self_input: SelfInput) -> bool:
        """Check if self-recognition occurs in mirror test."""
        if not self._mirror_capable:
            return False
        if self_input.visual_self_input is not None:
            return self_input.visual_self_input
        return True

    def _compute_boundary_integrity(
        self, self_model: SelfModelOutput, agency: AgencyAssessment
    ) -> float:
        """Compute self-other boundary integrity."""
        return (
            self_model.self_distinctness * 0.4 +
            self_model.self_coherence * 0.3 +
            agency.agency_score * 0.3
        )

    def _compute_health(self) -> float:
        """Compute self-recognition system health."""
        if not self._current_self_model:
            return 1.0
        return (
            self._current_self_model.self_coherence * 0.5 +
            self._current_self_model.embodiment_level * 0.3 +
            self._current_self_model.self_continuity * 0.2
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_self_recognition_interface() -> SelfRecognitionInterface:
    """Create and return a self-recognition interface."""
    return SelfRecognitionInterface()


__all__ = [
    # Enums
    "SelfAspect",
    "AgencyLevel",
    "OwnershipState",
    "SelfBoundaryType",
    "SelfRecognitionMode",
    # Input dataclasses
    "BodySignalInput",
    "SocialContextInput",
    "ActionFeedback",
    "SelfInput",
    # Output dataclasses
    "SelfModelOutput",
    "AgencyAssessment",
    "SelfOutput",
    "SelfSystemStatus",
    # Engines
    "SelfModelEngine",
    "AgencyEngine",
    # Main interface
    "SelfRecognitionInterface",
    # Convenience
    "create_self_recognition_interface",
]
