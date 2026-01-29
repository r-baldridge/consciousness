#!/usr/bin/env python3
"""
Lucid Dream Consciousness Interface

Form 23: Models awareness within the dream state, including the detection
of lucidity, mechanisms of dream control, reality checking, and dream
stabilization. Lucid dreaming represents a unique form of consciousness
where meta-awareness arises within the dream state, enabling the dreamer
to recognize the dream as a dream and potentially exert volitional control
over dream content.

This form explores the spectrum from non-lucid dreaming through pre-lucid
awareness to full lucidity with dream control, modeling the neural and
cognitive signatures associated with each level.
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

class LucidityLevel(Enum):
    """
    Levels of lucidity within the dream state.

    Ranges from completely non-lucid (no awareness of dreaming) through
    pre-lucid states (vague suspicion) to full lucidity with complete
    meta-awareness and reflective capacity.
    """
    NON_LUCID = "non_lucid"          # No awareness of dream state
    PRE_LUCID = "pre_lucid"          # Vague suspicion something is off
    SEMI_LUCID = "semi_lucid"        # Partial awareness, may fade quickly
    FULLY_LUCID = "fully_lucid"      # Complete awareness of dreaming


class DreamControl(Enum):
    """
    Degree of volitional control over dream content.

    Dream control is independent of lucidity level; a dreamer can be
    fully lucid but have no control, or have partial control without
    complete meta-awareness.
    """
    NONE = "none"                    # No ability to influence dream content
    PARTIAL = "partial"              # Can influence some elements
    FULL = "full"                    # Complete control over dream environment


class LucidTrigger(Enum):
    """
    Methods and triggers that induce lucid awareness in dreams.

    Each trigger corresponds to a different induction technique or
    spontaneous recognition mechanism.
    """
    REALITY_CHECK = "reality_check"          # Habitual reality testing carries into dream
    ANOMALY_RECOGNITION = "anomaly_recognition"  # Noticing impossible/inconsistent elements
    WILD = "wild"                            # Wake-Initiated Lucid Dream (direct entry)
    MILD = "mild"                            # Mnemonic Induction of Lucid Dreams


class DreamStability(Enum):
    """
    Stability of the dream state once lucidity is achieved.

    Many lucid dreamers experience rapid destabilization upon becoming
    lucid, requiring stabilization techniques to maintain the dream.
    """
    COLLAPSING = "collapsing"        # Dream rapidly dissolving
    UNSTABLE = "unstable"            # Dream flickering, may collapse
    STABLE = "stable"                # Dream holding steady
    VIVID = "vivid"                  # Exceptionally clear and stable


class DreamPhase(Enum):
    """Sleep and dream cycle phases relevant to lucid dreaming."""
    NREM_LIGHT = "nrem_light"        # Light non-REM sleep (stages 1-2)
    NREM_DEEP = "nrem_deep"          # Deep non-REM sleep (stages 3-4)
    REM_EARLY = "rem_early"          # Early REM periods (shorter, less vivid)
    REM_LATE = "rem_late"            # Late REM periods (longer, more vivid)
    REM_EXTENDED = "rem_extended"    # Extended morning REM (most conducive)


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class LucidDreamInput:
    """
    Input data representing the current dream state and awareness signals.

    Encapsulates all signals relevant to determining the dreamer's level
    of awareness within the dream, including physiological markers and
    cognitive state indicators.
    """
    dream_phase: DreamPhase
    awareness_signals: float              # 0.0-1.0: strength of meta-awareness signals
    dream_vividness: float                # 0.0-1.0: how vivid the dream content is
    anomaly_count: int = 0                # Number of recognized anomalies
    reality_check_performed: bool = False # Whether a reality check was attempted
    emotional_intensity: float = 0.5      # 0.0-1.0: emotional content intensity
    narrative_coherence: float = 0.5      # 0.0-1.0: how coherent the dream narrative is
    sensory_detail_level: float = 0.5     # 0.0-1.0: richness of sensory detail
    time_in_rem: float = 0.0             # Minutes spent in current REM period
    prior_lucid_count: int = 0           # Number of prior lucid episodes this session
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_phase": self.dream_phase.value,
            "awareness_signals": self.awareness_signals,
            "dream_vividness": self.dream_vividness,
            "anomaly_count": self.anomaly_count,
            "reality_check_performed": self.reality_check_performed,
            "emotional_intensity": self.emotional_intensity,
            "narrative_coherence": self.narrative_coherence,
            "sensory_detail_level": self.sensory_detail_level,
            "time_in_rem": self.time_in_rem,
            "prior_lucid_count": self.prior_lucid_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamControlInput:
    """Input representing an attempt to exert control over the dream."""
    target_element: str                   # What the dreamer is trying to control
    control_intention: str                # The intended change
    effort_level: float = 0.5            # 0.0-1.0: how much effort applied
    technique_used: str = "direct_will"  # Technique: direct_will, expectation, spinning
    confidence: float = 0.5              # 0.0-1.0: dreamer's confidence in success
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_element": self.target_element,
            "control_intention": self.control_intention,
            "effort_level": self.effort_level,
            "technique_used": self.technique_used,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class LucidDreamOutput:
    """
    Output from lucidity assessment processing.

    Contains the determined lucidity level, control degree, dream stability,
    and associated confidence metrics.
    """
    lucidity_level: LucidityLevel
    control_degree: DreamControl
    dream_stability: DreamStability
    lucidity_score: float                 # 0.0-1.0: continuous lucidity measure
    control_score: float                  # 0.0-1.0: continuous control measure
    stability_score: float                # 0.0-1.0: continuous stability measure
    trigger_detected: Optional[LucidTrigger] = None
    meta_awareness_index: float = 0.0     # 0.0-1.0: depth of meta-awareness
    dream_clarity: float = 0.5           # 0.0-1.0: perceptual clarity
    confidence: float = 0.5              # 0.0-1.0: assessment confidence
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lucidity_level": self.lucidity_level.value,
            "control_degree": self.control_degree.value,
            "dream_stability": self.dream_stability.value,
            "lucidity_score": self.lucidity_score,
            "control_score": self.control_score,
            "stability_score": self.stability_score,
            "trigger_detected": self.trigger_detected.value if self.trigger_detected else None,
            "meta_awareness_index": self.meta_awareness_index,
            "dream_clarity": self.dream_clarity,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamControlOutput:
    """Output from a dream control attempt."""
    success: bool
    control_degree_achieved: DreamControl
    stability_impact: float               # -1.0 to 1.0: impact on dream stability
    element_modified: str                 # What was actually changed
    side_effects: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "control_degree_achieved": self.control_degree_achieved.value,
            "stability_impact": self.stability_impact,
            "element_modified": self.element_modified,
            "side_effects": self.side_effects,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RealityCheckResult:
    """Result of a reality check performed within the dream."""
    check_type: str                       # Type: hand_check, text_check, light_switch, etc.
    result_anomalous: bool                # Whether the result indicates dreaming
    anomaly_description: str = ""         # What was anomalous
    lucidity_boost: float = 0.0          # 0.0-1.0: how much this boosted lucidity
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_type": self.check_type,
            "result_anomalous": self.result_anomalous,
            "anomaly_description": self.anomaly_description,
            "lucidity_boost": self.lucidity_boost,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamStateSnapshot:
    """Complete snapshot of the current dream state."""
    lucidity: LucidDreamOutput
    phase: DreamPhase
    elapsed_time: float                   # Minutes in current dream
    stability_trend: float                # -1.0 to 1.0: stability trajectory
    control_history: List[DreamControlOutput] = field(default_factory=list)
    reality_checks: List[RealityCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lucidity": self.lucidity.to_dict(),
            "phase": self.phase.value,
            "elapsed_time": self.elapsed_time,
            "stability_trend": self.stability_trend,
            "control_history": [c.to_dict() for c in self.control_history],
            "reality_checks": [r.to_dict() for r in self.reality_checks],
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class LucidDreamInterface:
    """
    Main interface for Form 23: Lucid Dream Consciousness.

    Provides methods for detecting lucidity within dream states, assessing
    dream control capabilities, performing reality checks, and stabilizing
    the dream environment. Models the full spectrum of lucid dream
    experiences from non-lucid through fully lucid with dream control.
    """

    FORM_ID = "23-lucid-dream"
    FORM_NAME = "Lucid Dream Consciousness"

    def __init__(self):
        """Initialize the Lucid Dream Consciousness Interface."""
        self._initialized = False
        self._dream_history: List[DreamStateSnapshot] = []
        self._lucidity_baseline: float = 0.0
        self._control_baseline: float = 0.0
        self._stability_history: List[float] = []
        self._current_phase: Optional[DreamPhase] = None
        self._current_lucidity: Optional[LucidDreamOutput] = None

    async def initialize(self) -> None:
        """Initialize the interface and set baseline parameters."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        self._lucidity_baseline = 0.1
        self._control_baseline = 0.0
        self._stability_history = []
        self._current_phase = DreamPhase.NREM_LIGHT

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def assess_lucidity(self, dream_input: LucidDreamInput) -> LucidDreamOutput:
        """
        Assess the current level of lucidity based on dream state signals.

        Analyzes awareness signals, anomaly detection, reality check results,
        and dream phase to determine the lucidity level and associated metrics.

        Args:
            dream_input: Current dream state and awareness signals.

        Returns:
            LucidDreamOutput with lucidity assessment.
        """
        if not self._initialized:
            await self.initialize()

        # Compute lucidity score from multiple factors
        lucidity_score = self._compute_lucidity_score(dream_input)

        # Determine discrete lucidity level
        lucidity_level = self._score_to_lucidity_level(lucidity_score)

        # Assess control capability based on lucidity
        control_score = self._compute_control_score(lucidity_score, dream_input)
        control_degree = self._score_to_control_degree(control_score)

        # Assess dream stability
        stability_score = self._compute_stability_score(dream_input, lucidity_score)
        dream_stability = self._score_to_stability(stability_score)

        # Detect what triggered lucidity (if any)
        trigger = self._detect_trigger(dream_input)

        # Meta-awareness index
        meta_awareness = min(1.0, lucidity_score * 1.2) if lucidity_score > 0.3 else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            lucidity_level, dream_stability, control_degree
        )

        output = LucidDreamOutput(
            lucidity_level=lucidity_level,
            control_degree=control_degree,
            dream_stability=dream_stability,
            lucidity_score=lucidity_score,
            control_score=control_score,
            stability_score=stability_score,
            trigger_detected=trigger,
            meta_awareness_index=meta_awareness,
            dream_clarity=dream_input.dream_vividness,
            confidence=self._compute_confidence(dream_input),
            recommendations=recommendations,
        )

        self._current_lucidity = output
        self._stability_history.append(stability_score)
        if len(self._stability_history) > 50:
            self._stability_history.pop(0)

        return output

    async def attempt_control(
        self, control_input: DreamControlInput
    ) -> DreamControlOutput:
        """
        Attempt to exert control over dream content.

        Success depends on current lucidity level, technique used,
        confidence, and dream stability.

        Args:
            control_input: The control attempt parameters.

        Returns:
            DreamControlOutput indicating success and impact.
        """
        if not self._initialized:
            await self.initialize()

        current_lucidity = self._current_lucidity
        if current_lucidity is None:
            return DreamControlOutput(
                success=False,
                control_degree_achieved=DreamControl.NONE,
                stability_impact=-0.1,
                element_modified="none",
                side_effects=["No lucidity state established"],
                confidence=0.2,
            )

        # Control success probability
        base_probability = current_lucidity.control_score
        technique_modifier = self._technique_modifier(control_input.technique_used)
        confidence_modifier = control_input.confidence * 0.3
        effort_penalty = max(0.0, (control_input.effort_level - 0.7) * 0.5)

        success_probability = min(1.0, max(0.0,
            base_probability * technique_modifier + confidence_modifier - effort_penalty
        ))

        success = success_probability > 0.4

        # Stability impact - more forceful control destabilizes
        stability_impact = -control_input.effort_level * 0.3 if success else -0.2
        if control_input.technique_used == "expectation":
            stability_impact += 0.1  # Gentle technique

        # Side effects
        side_effects = []
        if control_input.effort_level > 0.8:
            side_effects.append("High effort may destabilize dream")
        if not success:
            side_effects.append("Failed control attempt may reduce lucidity")

        control_degree = DreamControl.FULL if success_probability > 0.7 else (
            DreamControl.PARTIAL if success else DreamControl.NONE
        )

        return DreamControlOutput(
            success=success,
            control_degree_achieved=control_degree,
            stability_impact=stability_impact,
            element_modified=control_input.target_element if success else "none",
            side_effects=side_effects,
            confidence=min(1.0, success_probability + 0.1),
        )

    async def reality_check(self, check_type: str = "hand_check") -> RealityCheckResult:
        """
        Perform a reality check within the dream state.

        Reality checks are tests that produce different results in waking
        vs. dreaming states, helping to trigger or confirm lucidity.

        Args:
            check_type: Type of reality check (hand_check, text_check,
                       light_switch, nose_pinch, clock_check).

        Returns:
            RealityCheckResult with check outcome.
        """
        if not self._initialized:
            await self.initialize()

        # In a dream state, reality checks should show anomalies
        check_anomaly_rates = {
            "hand_check": 0.85,      # Hands often look wrong in dreams
            "text_check": 0.90,      # Text changes on re-reading
            "light_switch": 0.80,    # Light switches often don't work
            "nose_pinch": 0.95,      # Can still breathe with pinched nose in dreams
            "clock_check": 0.88,     # Clocks show inconsistent time
        }

        anomaly_rate = check_anomaly_rates.get(check_type, 0.75)
        result_anomalous = True  # In dream context, checks show anomalies

        anomaly_descriptions = {
            "hand_check": "Fingers appear distorted, wrong number of digits",
            "text_check": "Text morphs and changes upon re-reading",
            "light_switch": "Light switch has no effect on illumination",
            "nose_pinch": "Breathing continues despite pinched nostrils",
            "clock_check": "Clock displays impossible or shifting time",
        }

        lucidity_boost = anomaly_rate * 0.4 if result_anomalous else 0.0

        return RealityCheckResult(
            check_type=check_type,
            result_anomalous=result_anomalous,
            anomaly_description=anomaly_descriptions.get(check_type, "Anomaly detected"),
            lucidity_boost=lucidity_boost,
            confidence=anomaly_rate,
        )

    async def stabilize_dream(self) -> Dict[str, Any]:
        """
        Attempt to stabilize the dream when it begins to collapse.

        Uses common stabilization techniques: spinning, rubbing hands,
        engaging senses, and verbal commands.

        Returns:
            Dictionary with stabilization result and new stability score.
        """
        if not self._initialized:
            await self.initialize()

        current_stability = (
            self._stability_history[-1] if self._stability_history else 0.5
        )

        # Stabilization techniques and their effectiveness
        techniques = {
            "hand_rubbing": 0.15,
            "spinning": 0.20,
            "sensory_engagement": 0.25,
            "verbal_command": 0.10,
            "grounding": 0.18,
        }

        total_boost = sum(techniques.values()) * 0.5  # Apply partial effect
        new_stability = min(1.0, current_stability + total_boost)

        stability_enum = self._score_to_stability(new_stability)

        self._stability_history.append(new_stability)

        return {
            "previous_stability": current_stability,
            "new_stability": new_stability,
            "stability_state": stability_enum.value,
            "techniques_applied": list(techniques.keys()),
            "boost_achieved": total_boost,
            "success": new_stability > current_stability,
        }

    async def get_dream_state(self) -> DreamStateSnapshot:
        """
        Get a complete snapshot of the current dream state.

        Returns:
            DreamStateSnapshot with all current state information.
        """
        if not self._initialized:
            await self.initialize()

        current_lucidity = self._current_lucidity
        if current_lucidity is None:
            current_lucidity = LucidDreamOutput(
                lucidity_level=LucidityLevel.NON_LUCID,
                control_degree=DreamControl.NONE,
                dream_stability=DreamStability.STABLE,
                lucidity_score=0.0,
                control_score=0.0,
                stability_score=0.5,
            )

        stability_trend = self._compute_stability_trend()

        return DreamStateSnapshot(
            lucidity=current_lucidity,
            phase=self._current_phase or DreamPhase.REM_LATE,
            elapsed_time=len(self._dream_history) * 5.0,
            stability_trend=stability_trend,
        )

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the interface state to a dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "current_phase": self._current_phase.value if self._current_phase else None,
            "lucidity_baseline": self._lucidity_baseline,
            "control_baseline": self._control_baseline,
            "stability_history_length": len(self._stability_history),
            "dream_history_length": len(self._dream_history),
            "current_lucidity": (
                self._current_lucidity.to_dict() if self._current_lucidity else None
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lucid dream interface."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "current_phase": self._current_phase.value if self._current_phase else None,
            "current_lucidity_level": (
                self._current_lucidity.lucidity_level.value
                if self._current_lucidity else "unknown"
            ),
            "stability_readings": len(self._stability_history),
            "dream_sessions": len(self._dream_history),
        }

    # ========================================================================
    # PRIVATE COMPUTATION METHODS
    # ========================================================================

    def _compute_lucidity_score(self, dream_input: LucidDreamInput) -> float:
        """Compute continuous lucidity score from input signals."""
        score = 0.0

        # Awareness signals are the primary indicator
        score += dream_input.awareness_signals * 0.40

        # Anomaly recognition contributes
        anomaly_factor = min(1.0, dream_input.anomaly_count * 0.25)
        score += anomaly_factor * 0.20

        # Reality check is a strong trigger
        if dream_input.reality_check_performed:
            score += 0.20

        # REM phase affects lucidity potential
        phase_modifiers = {
            DreamPhase.NREM_LIGHT: 0.0,
            DreamPhase.NREM_DEEP: 0.0,
            DreamPhase.REM_EARLY: 0.05,
            DreamPhase.REM_LATE: 0.10,
            DreamPhase.REM_EXTENDED: 0.15,
        }
        score += phase_modifiers.get(dream_input.dream_phase, 0.0)

        # Vividness contributes to awareness
        score += dream_input.dream_vividness * 0.05

        return max(0.0, min(1.0, score))

    def _score_to_lucidity_level(self, score: float) -> LucidityLevel:
        """Convert continuous score to discrete lucidity level."""
        if score < 0.2:
            return LucidityLevel.NON_LUCID
        elif score < 0.4:
            return LucidityLevel.PRE_LUCID
        elif score < 0.7:
            return LucidityLevel.SEMI_LUCID
        else:
            return LucidityLevel.FULLY_LUCID

    def _compute_control_score(
        self, lucidity_score: float, dream_input: LucidDreamInput
    ) -> float:
        """Compute dream control capability score."""
        if lucidity_score < 0.3:
            return 0.0

        # Control requires lucidity as a foundation
        base_control = (lucidity_score - 0.3) * 1.4

        # Vividness supports control
        base_control += dream_input.dream_vividness * 0.1

        # Prior experience helps
        experience_bonus = min(0.2, dream_input.prior_lucid_count * 0.05)
        base_control += experience_bonus

        return max(0.0, min(1.0, base_control))

    def _score_to_control_degree(self, score: float) -> DreamControl:
        """Convert control score to discrete control degree."""
        if score < 0.2:
            return DreamControl.NONE
        elif score < 0.6:
            return DreamControl.PARTIAL
        else:
            return DreamControl.FULL

    def _compute_stability_score(
        self, dream_input: LucidDreamInput, lucidity_score: float
    ) -> float:
        """Compute dream stability score."""
        # Base stability from dream vividness and coherence
        stability = (
            dream_input.dream_vividness * 0.3 +
            dream_input.narrative_coherence * 0.3 +
            dream_input.sensory_detail_level * 0.2
        )

        # High lucidity can initially destabilize
        if lucidity_score > 0.5:
            excitement_penalty = (lucidity_score - 0.5) * 0.3
            stability -= excitement_penalty

        # High emotion destabilizes
        if dream_input.emotional_intensity > 0.7:
            stability -= (dream_input.emotional_intensity - 0.7) * 0.4

        # Time in REM helps stability
        rem_bonus = min(0.15, dream_input.time_in_rem * 0.005)
        stability += rem_bonus

        return max(0.0, min(1.0, stability))

    def _score_to_stability(self, score: float) -> DreamStability:
        """Convert stability score to discrete stability state."""
        if score < 0.2:
            return DreamStability.COLLAPSING
        elif score < 0.4:
            return DreamStability.UNSTABLE
        elif score < 0.7:
            return DreamStability.STABLE
        else:
            return DreamStability.VIVID

    def _detect_trigger(self, dream_input: LucidDreamInput) -> Optional[LucidTrigger]:
        """Detect what triggered lucidity, if anything."""
        if dream_input.reality_check_performed:
            return LucidTrigger.REALITY_CHECK
        if dream_input.anomaly_count >= 2:
            return LucidTrigger.ANOMALY_RECOGNITION
        if (dream_input.awareness_signals > 0.7 and
                dream_input.dream_phase in (DreamPhase.NREM_LIGHT, DreamPhase.REM_EARLY)):
            return LucidTrigger.WILD
        if dream_input.awareness_signals > 0.5 and dream_input.prior_lucid_count > 0:
            return LucidTrigger.MILD
        return None

    def _technique_modifier(self, technique: str) -> float:
        """Get effectiveness modifier for a control technique."""
        modifiers = {
            "direct_will": 0.8,
            "expectation": 1.2,
            "spinning": 0.9,
            "verbal_command": 1.0,
            "visualization": 1.1,
        }
        return modifiers.get(technique, 0.8)

    def _compute_confidence(self, dream_input: LucidDreamInput) -> float:
        """Compute confidence in the assessment."""
        # Higher confidence with stronger signals
        confidence = 0.5
        confidence += dream_input.awareness_signals * 0.2
        confidence += dream_input.dream_vividness * 0.1
        if dream_input.reality_check_performed:
            confidence += 0.15
        return min(1.0, confidence)

    def _generate_recommendations(
        self,
        lucidity_level: LucidityLevel,
        stability: DreamStability,
        control: DreamControl,
    ) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []

        if lucidity_level == LucidityLevel.NON_LUCID:
            recommendations.append("Perform reality checks to trigger awareness")
        elif lucidity_level == LucidityLevel.PRE_LUCID:
            recommendations.append("Look for dream signs and anomalies")
            recommendations.append("Attempt a reality check now")
        elif lucidity_level == LucidityLevel.SEMI_LUCID:
            recommendations.append("Stabilize awareness before attempting control")
            if stability == DreamStability.UNSTABLE:
                recommendations.append("Use grounding techniques: rub hands, engage senses")

        if stability == DreamStability.COLLAPSING:
            recommendations.append("Spin in place or rub hands vigorously")
            recommendations.append("Engage tactile senses to anchor the dream")
        elif stability == DreamStability.UNSTABLE:
            recommendations.append("Avoid sudden movements, stay calm")

        if control == DreamControl.NONE and lucidity_level == LucidityLevel.FULLY_LUCID:
            recommendations.append("Start with small changes using expectation technique")

        return recommendations

    def _compute_stability_trend(self) -> float:
        """Compute the trend in dream stability over time."""
        if len(self._stability_history) < 3:
            return 0.0

        recent = self._stability_history[-5:]
        if len(recent) < 2:
            return 0.0

        trend = (recent[-1] - recent[0]) / len(recent)
        return max(-1.0, min(1.0, trend * 5))


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "LucidityLevel",
    "DreamControl",
    "LucidTrigger",
    "DreamStability",
    "DreamPhase",
    # Input dataclasses
    "LucidDreamInput",
    "DreamControlInput",
    # Output dataclasses
    "LucidDreamOutput",
    "DreamControlOutput",
    "RealityCheckResult",
    "DreamStateSnapshot",
    # Interface
    "LucidDreamInterface",
    # Convenience
    "create_lucid_dream_interface",
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_lucid_dream_interface() -> LucidDreamInterface:
    """
    Create and return a new LucidDreamInterface instance.

    Note: Call await interface.initialize() before use.

    Returns:
        A new LucidDreamInterface instance.
    """
    return LucidDreamInterface()
