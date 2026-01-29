#!/usr/bin/env python3
"""
Locked-In Consciousness Interface

Form 24: Models the locked-in syndrome state of consciousness, where
full cognitive awareness persists with minimal or absent motor output.
This form addresses one of the most challenging scenarios in consciousness
studies: detecting and communicating with awareness that is trapped within
an unresponsive body.

Locked-in syndrome can result from brainstem lesions (typically pontine),
and ranges from classic locked-in (preserved vertical eye movements) to
total locked-in (complete motor paralysis with preserved consciousness).
This interface models awareness detection, intention decoding, and
communication channel establishment for locked-in states.
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

class LockedInType(Enum):
    """
    Classification of locked-in syndrome types.

    Based on the degree of motor output preservation, which determines
    available communication channels.
    """
    CLASSIC = "classic"              # Vertical eye movements and blinking preserved
    INCOMPLETE = "incomplete"        # Some voluntary movements beyond eyes preserved
    TOTAL = "total"                  # Complete motor paralysis, no voluntary movement


class CommunicationChannel(Enum):
    """
    Available channels for communication with locked-in patients.

    Each channel has different bandwidth, reliability, and
    cognitive demands for the patient.
    """
    EYE_MOVEMENT = "eye_movement"                    # Vertical eye movements, blinking
    BRAIN_COMPUTER_INTERFACE = "brain_computer_interface"  # EEG/fMRI-based BCI
    MUSCLE_TWITCH = "muscle_twitch"                  # Minimal residual muscle activity
    PUPIL_DILATION = "pupil_dilation"                # Involuntary but modifiable pupil response
    RESPIRATORY = "respiratory"                       # Breathing pattern modification


class AwarenessState(Enum):
    """
    Assessed state of awareness in a locked-in patient.

    Ranges from unresponsive (possibly no awareness) through minimally
    conscious to full awareness. Careful assessment is critical to avoid
    misdiagnosis of awareness states.
    """
    UNRESPONSIVE = "unresponsive"          # No detectable awareness signals
    POSSIBLE_AWARENESS = "possible_awareness"  # Ambiguous signals detected
    MINIMAL_CONSCIOUSNESS = "minimal_consciousness"  # Inconsistent but present responses
    FULL_AWARENESS = "full_awareness"      # Clear evidence of intact cognition


class SignalQuality(Enum):
    """Quality rating for decoded neural or physiological signals."""
    NOISE = "noise"                  # Indistinguishable from background noise
    POOR = "poor"                    # Weak signal, low confidence
    FAIR = "fair"                    # Detectable but ambiguous
    GOOD = "good"                    # Clear signal with moderate confidence
    EXCELLENT = "excellent"          # Strong, reliable signal


class CognitiveFunction(Enum):
    """Cognitive functions assessed in locked-in patients."""
    LANGUAGE_COMPREHENSION = "language_comprehension"
    SPATIAL_REASONING = "spatial_reasoning"
    EMOTIONAL_PROCESSING = "emotional_processing"
    MEMORY_RETRIEVAL = "memory_retrieval"
    ATTENTION = "attention"
    EXECUTIVE_FUNCTION = "executive_function"


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class LockedInInput:
    """
    Input data representing neural and physiological signals from
    a locked-in patient.

    Aggregates all available signal sources for consciousness
    assessment and intention decoding.
    """
    locked_in_type: LockedInType
    neural_signal_strength: float         # 0.0-1.0: EEG/fMRI signal strength
    physiological_signals: Dict[str, float] = field(default_factory=dict)
    eye_tracking_data: Optional[Dict[str, Any]] = None
    bci_signal: Optional[Dict[str, float]] = None
    muscle_emg_data: Optional[Dict[str, float]] = None
    stimulus_presented: Optional[str] = None
    stimulus_type: str = "auditory"       # auditory, visual, tactile, verbal
    time_since_onset: float = 0.0         # Days since locked-in state onset
    medication_state: str = "stable"      # stable, sedated, alert
    signal_quality: SignalQuality = SignalQuality.FAIR
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locked_in_type": self.locked_in_type.value,
            "neural_signal_strength": self.neural_signal_strength,
            "physiological_signals": self.physiological_signals,
            "eye_tracking_data": self.eye_tracking_data,
            "bci_signal": self.bci_signal,
            "muscle_emg_data": self.muscle_emg_data,
            "stimulus_presented": self.stimulus_presented,
            "stimulus_type": self.stimulus_type,
            "time_since_onset": self.time_since_onset,
            "medication_state": self.medication_state,
            "signal_quality": self.signal_quality.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CommunicationAttemptInput:
    """Input for a communication attempt with a locked-in patient."""
    channel: CommunicationChannel
    message_to_patient: str               # What is being asked or conveyed
    expected_response_type: str = "yes_no"  # yes_no, letter_selection, free
    response_window_seconds: float = 10.0   # Time allowed for response
    repetitions: int = 1                    # Number of times to repeat
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "message_to_patient": self.message_to_patient,
            "expected_response_type": self.expected_response_type,
            "response_window_seconds": self.response_window_seconds,
            "repetitions": self.repetitions,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class DecodedIntention:
    """
    Output representing a decoded intention from locked-in patient signals.
    """
    intention_detected: bool
    decoded_content: str                  # Decoded intention or response
    confidence: float                     # 0.0-1.0: confidence in decoding
    channel_used: CommunicationChannel
    signal_quality: SignalQuality
    alternative_interpretations: List[str] = field(default_factory=list)
    latency_ms: float = 0.0             # Response latency in milliseconds
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intention_detected": self.intention_detected,
            "decoded_content": self.decoded_content,
            "confidence": self.confidence,
            "channel_used": self.channel_used.value,
            "signal_quality": self.signal_quality.value,
            "alternative_interpretations": self.alternative_interpretations,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AwarenessAssessment:
    """
    Output from a comprehensive awareness assessment.

    This assessment must be performed with extreme care to avoid
    false negatives (concluding no awareness when awareness exists).
    """
    awareness_state: AwarenessState
    confidence: float                     # 0.0-1.0: confidence in assessment
    cognitive_profile: Dict[str, float] = field(default_factory=dict)
    responsive_channels: List[CommunicationChannel] = field(default_factory=list)
    recommended_channels: List[CommunicationChannel] = field(default_factory=list)
    assessment_notes: List[str] = field(default_factory=list)
    false_negative_risk: float = 0.0     # 0.0-1.0: risk of missed awareness
    reassessment_recommended: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "awareness_state": self.awareness_state.value,
            "confidence": self.confidence,
            "cognitive_profile": self.cognitive_profile,
            "responsive_channels": [c.value for c in self.responsive_channels],
            "recommended_channels": [c.value for c in self.recommended_channels],
            "assessment_notes": self.assessment_notes,
            "false_negative_risk": self.false_negative_risk,
            "reassessment_recommended": self.reassessment_recommended,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CommunicationResult:
    """Output from a communication attempt with a locked-in patient."""
    success: bool
    decoded_response: str
    confidence: float
    channel_used: CommunicationChannel
    response_latency_ms: float = 0.0
    signal_quality: SignalQuality = SignalQuality.FAIR
    needs_confirmation: bool = True
    notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "decoded_response": self.decoded_response,
            "confidence": self.confidence,
            "channel_used": self.channel_used.value,
            "response_latency_ms": self.response_latency_ms,
            "signal_quality": self.signal_quality.value,
            "needs_confirmation": self.needs_confirmation,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsciousnessMonitorReading:
    """Continuous monitoring reading of consciousness state."""
    awareness_level: float                # 0.0-1.0: estimated awareness
    awareness_state: AwarenessState
    signal_stability: float               # 0.0-1.0: signal consistency
    active_channels: List[CommunicationChannel] = field(default_factory=list)
    cognitive_load_estimate: float = 0.0  # 0.0-1.0: estimated cognitive effort
    fatigue_indicator: float = 0.0        # 0.0-1.0: estimated fatigue
    alertness_trend: float = 0.0          # -1.0 to 1.0: trend direction
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "awareness_level": self.awareness_level,
            "awareness_state": self.awareness_state.value,
            "signal_stability": self.signal_stability,
            "active_channels": [c.value for c in self.active_channels],
            "cognitive_load_estimate": self.cognitive_load_estimate,
            "fatigue_indicator": self.fatigue_indicator,
            "alertness_trend": self.alertness_trend,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class LockedInInterface:
    """
    Main interface for Form 24: Locked-In Consciousness.

    Provides methods for detecting awareness, decoding intentions, and
    establishing communication channels with locked-in patients. Designed
    with extreme sensitivity to the ethical imperative of correctly
    identifying consciousness in apparently unresponsive patients.
    """

    FORM_ID = "24-locked-in"
    FORM_NAME = "Locked-In Consciousness"

    def __init__(self):
        """Initialize the Locked-In Consciousness Interface."""
        self._initialized = False
        self._assessment_history: List[AwarenessAssessment] = []
        self._communication_log: List[CommunicationResult] = []
        self._monitor_readings: List[ConsciousnessMonitorReading] = []
        self._active_channels: List[CommunicationChannel] = []
        self._current_awareness: Optional[AwarenessAssessment] = None
        self._patient_profile: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        self._active_channels = []
        self._patient_profile = {
            "locked_in_type": None,
            "time_since_onset": 0.0,
            "best_channel": None,
            "assessed_awareness": None,
        }

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def decode_intention(
        self, locked_in_input: LockedInInput
    ) -> DecodedIntention:
        """
        Attempt to decode an intention from locked-in patient signals.

        Analyzes neural, physiological, and available motor signals to
        decode the patient's intended communication or response.

        Args:
            locked_in_input: Neural and physiological signal data.

        Returns:
            DecodedIntention with decoded content and confidence.
        """
        if not self._initialized:
            await self.initialize()

        # Determine best available channel based on locked-in type
        channel = self._determine_channel(locked_in_input)

        # Decode signal based on channel
        decoded_content, confidence = self._decode_signal(locked_in_input, channel)

        # Assess signal quality
        quality = self._assess_signal_quality(locked_in_input)

        # Generate alternative interpretations for safety
        alternatives = self._generate_alternatives(decoded_content, confidence)

        intention_detected = confidence > 0.3

        return DecodedIntention(
            intention_detected=intention_detected,
            decoded_content=decoded_content,
            confidence=confidence,
            channel_used=channel,
            signal_quality=quality,
            alternative_interpretations=alternatives,
            latency_ms=self._estimate_latency(locked_in_input),
        )

    async def assess_awareness(
        self, locked_in_input: LockedInInput
    ) -> AwarenessAssessment:
        """
        Perform a comprehensive awareness assessment.

        This is the most critical function: determining whether
        consciousness is present in a patient who cannot move. Errs
        strongly on the side of detecting awareness to minimize
        false negative risk.

        Args:
            locked_in_input: All available signal data.

        Returns:
            AwarenessAssessment with state determination and recommendations.
        """
        if not self._initialized:
            await self.initialize()

        # Multi-modal awareness detection
        awareness_score = self._compute_awareness_score(locked_in_input)

        # Map to awareness state
        awareness_state = self._score_to_awareness_state(awareness_score)

        # Build cognitive profile
        cognitive_profile = self._build_cognitive_profile(locked_in_input)

        # Determine responsive channels
        responsive = self._find_responsive_channels(locked_in_input)

        # Recommend optimal channels
        recommended = self._recommend_channels(locked_in_input, responsive)

        # Compute false negative risk
        fn_risk = self._compute_false_negative_risk(
            locked_in_input, awareness_score
        )

        # Assessment notes
        notes = self._generate_assessment_notes(
            awareness_state, locked_in_input, fn_risk
        )

        assessment = AwarenessAssessment(
            awareness_state=awareness_state,
            confidence=min(1.0, awareness_score + 0.1),
            cognitive_profile=cognitive_profile,
            responsive_channels=responsive,
            recommended_channels=recommended,
            assessment_notes=notes,
            false_negative_risk=fn_risk,
            reassessment_recommended=awareness_state != AwarenessState.FULL_AWARENESS,
        )

        self._current_awareness = assessment
        self._assessment_history.append(assessment)

        # Update patient profile
        self._patient_profile["locked_in_type"] = locked_in_input.locked_in_type.value
        self._patient_profile["assessed_awareness"] = awareness_state.value

        return assessment

    async def attempt_communication(
        self, comm_input: CommunicationAttemptInput
    ) -> CommunicationResult:
        """
        Attempt to communicate with a locked-in patient through a
        specified channel.

        Args:
            comm_input: Communication attempt parameters.

        Returns:
            CommunicationResult with decoded response.
        """
        if not self._initialized:
            await self.initialize()

        # Check if channel is in active list
        channel_available = comm_input.channel in self._active_channels

        if not channel_available and self._current_awareness:
            # Try using responsive channels from last assessment
            channel_available = (
                comm_input.channel in self._current_awareness.responsive_channels
            )

        # Simulate communication attempt
        if comm_input.expected_response_type == "yes_no":
            response, confidence = self._decode_yes_no(comm_input)
        elif comm_input.expected_response_type == "letter_selection":
            response, confidence = self._decode_letter_selection(comm_input)
        else:
            response, confidence = self._decode_free_response(comm_input)

        success = confidence > 0.3

        notes = []
        if not channel_available:
            notes.append("Channel not previously verified as responsive")
            confidence *= 0.7
        if comm_input.repetitions > 1:
            notes.append(f"Response averaged over {comm_input.repetitions} repetitions")

        result = CommunicationResult(
            success=success,
            decoded_response=response,
            confidence=confidence,
            channel_used=comm_input.channel,
            response_latency_ms=comm_input.response_window_seconds * 200,
            signal_quality=SignalQuality.GOOD if confidence > 0.5 else SignalQuality.FAIR,
            needs_confirmation=confidence < 0.7,
            notes=notes,
        )

        self._communication_log.append(result)
        return result

    async def monitor_consciousness(
        self, locked_in_input: LockedInInput
    ) -> ConsciousnessMonitorReading:
        """
        Take a continuous monitoring reading of consciousness state.

        Designed for ongoing monitoring rather than one-time assessment.

        Args:
            locked_in_input: Current signal data.

        Returns:
            ConsciousnessMonitorReading with current state.
        """
        if not self._initialized:
            await self.initialize()

        awareness_level = self._compute_awareness_score(locked_in_input)
        awareness_state = self._score_to_awareness_state(awareness_level)

        # Signal stability from recent readings
        stability = self._compute_signal_stability()

        # Active channels
        active = self._find_responsive_channels(locked_in_input)

        # Fatigue estimation
        fatigue = self._estimate_fatigue()

        # Alertness trend
        trend = self._compute_alertness_trend()

        reading = ConsciousnessMonitorReading(
            awareness_level=awareness_level,
            awareness_state=awareness_state,
            signal_stability=stability,
            active_channels=active,
            cognitive_load_estimate=locked_in_input.neural_signal_strength * 0.6,
            fatigue_indicator=fatigue,
            alertness_trend=trend,
        )

        self._monitor_readings.append(reading)
        if len(self._monitor_readings) > 100:
            self._monitor_readings.pop(0)

        return reading

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the interface state to a dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "assessment_count": len(self._assessment_history),
            "communication_attempts": len(self._communication_log),
            "monitor_readings": len(self._monitor_readings),
            "active_channels": [c.value for c in self._active_channels],
            "patient_profile": self._patient_profile,
            "current_awareness": (
                self._current_awareness.to_dict() if self._current_awareness else None
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the locked-in interface."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "current_awareness": (
                self._current_awareness.awareness_state.value
                if self._current_awareness else "unknown"
            ),
            "active_channels": [c.value for c in self._active_channels],
            "total_assessments": len(self._assessment_history),
            "total_communications": len(self._communication_log),
        }

    # ========================================================================
    # PRIVATE COMPUTATION METHODS
    # ========================================================================

    def _determine_channel(self, inp: LockedInInput) -> CommunicationChannel:
        """Determine best communication channel based on locked-in type."""
        if inp.locked_in_type == LockedInType.CLASSIC:
            if inp.eye_tracking_data:
                return CommunicationChannel.EYE_MOVEMENT
        elif inp.locked_in_type == LockedInType.INCOMPLETE:
            if inp.muscle_emg_data:
                return CommunicationChannel.MUSCLE_TWITCH
            if inp.eye_tracking_data:
                return CommunicationChannel.EYE_MOVEMENT

        # Total locked-in or no specific signals: use BCI
        if inp.bci_signal:
            return CommunicationChannel.BRAIN_COMPUTER_INTERFACE

        return CommunicationChannel.BRAIN_COMPUTER_INTERFACE

    def _decode_signal(
        self, inp: LockedInInput, channel: CommunicationChannel
    ) -> Tuple[str, float]:
        """Decode signal through the specified channel."""
        base_confidence = inp.neural_signal_strength

        # Channel-specific decoding
        if channel == CommunicationChannel.EYE_MOVEMENT:
            if inp.eye_tracking_data:
                base_confidence *= 1.2
                return "affirmative_response", min(1.0, base_confidence)
            return "no_signal", 0.1

        elif channel == CommunicationChannel.BRAIN_COMPUTER_INTERFACE:
            if inp.bci_signal:
                max_signal = max(inp.bci_signal.values()) if inp.bci_signal else 0.0
                return "bci_decoded_response", min(1.0, max_signal * base_confidence)
            return "weak_bci_signal", max(0.1, base_confidence * 0.5)

        elif channel == CommunicationChannel.MUSCLE_TWITCH:
            if inp.muscle_emg_data:
                max_emg = max(inp.muscle_emg_data.values()) if inp.muscle_emg_data else 0.0
                return "muscle_response", min(1.0, max_emg * base_confidence)
            return "no_muscle_signal", 0.1

        return "undetermined", base_confidence * 0.3

    def _assess_signal_quality(self, inp: LockedInInput) -> SignalQuality:
        """Assess overall signal quality."""
        strength = inp.neural_signal_strength

        if strength < 0.1:
            return SignalQuality.NOISE
        elif strength < 0.3:
            return SignalQuality.POOR
        elif strength < 0.5:
            return SignalQuality.FAIR
        elif strength < 0.8:
            return SignalQuality.GOOD
        else:
            return SignalQuality.EXCELLENT

    def _generate_alternatives(
        self, primary: str, confidence: float
    ) -> List[str]:
        """Generate alternative interpretations for safety."""
        alternatives = []
        if confidence < 0.7:
            alternatives.append("noise_artifact")
        if confidence < 0.5:
            alternatives.append("involuntary_response")
            alternatives.append("ambiguous_signal")
        return alternatives

    def _estimate_latency(self, inp: LockedInInput) -> float:
        """Estimate response latency in milliseconds."""
        base_latency = 500.0  # Base processing time
        if inp.locked_in_type == LockedInType.TOTAL:
            base_latency += 1000.0  # Additional BCI processing
        quality_modifier = {
            SignalQuality.EXCELLENT: 0.5,
            SignalQuality.GOOD: 0.7,
            SignalQuality.FAIR: 1.0,
            SignalQuality.POOR: 1.5,
            SignalQuality.NOISE: 2.0,
        }
        return base_latency * quality_modifier.get(inp.signal_quality, 1.0)

    def _compute_awareness_score(self, inp: LockedInInput) -> float:
        """Compute awareness score from all available signals."""
        score = 0.0

        # Neural signal strength is primary indicator
        score += inp.neural_signal_strength * 0.40

        # Signal quality modifies base score
        quality_weights = {
            SignalQuality.EXCELLENT: 1.0,
            SignalQuality.GOOD: 0.8,
            SignalQuality.FAIR: 0.6,
            SignalQuality.POOR: 0.3,
            SignalQuality.NOISE: 0.1,
        }
        quality_mod = quality_weights.get(inp.signal_quality, 0.5)
        score *= quality_mod

        # Eye tracking data strongly suggests awareness
        if inp.eye_tracking_data:
            score += 0.25

        # BCI signals suggest cognitive activity
        if inp.bci_signal:
            bci_strength = max(inp.bci_signal.values()) if inp.bci_signal else 0.0
            score += bci_strength * 0.20

        # Physiological responses
        if inp.physiological_signals:
            physio_avg = (
                sum(inp.physiological_signals.values()) /
                len(inp.physiological_signals)
            )
            score += physio_avg * 0.15

        # Medication state affects assessment
        if inp.medication_state == "sedated":
            score *= 0.5
        elif inp.medication_state == "alert":
            score *= 1.1

        return max(0.0, min(1.0, score))

    def _score_to_awareness_state(self, score: float) -> AwarenessState:
        """Convert awareness score to discrete state."""
        if score < 0.15:
            return AwarenessState.UNRESPONSIVE
        elif score < 0.35:
            return AwarenessState.POSSIBLE_AWARENESS
        elif score < 0.6:
            return AwarenessState.MINIMAL_CONSCIOUSNESS
        else:
            return AwarenessState.FULL_AWARENESS

    def _build_cognitive_profile(self, inp: LockedInInput) -> Dict[str, float]:
        """Build cognitive function profile from signals."""
        base = inp.neural_signal_strength
        return {
            CognitiveFunction.LANGUAGE_COMPREHENSION.value: base * 0.9,
            CognitiveFunction.SPATIAL_REASONING.value: base * 0.7,
            CognitiveFunction.EMOTIONAL_PROCESSING.value: base * 0.95,
            CognitiveFunction.MEMORY_RETRIEVAL.value: base * 0.8,
            CognitiveFunction.ATTENTION.value: base * 0.85,
            CognitiveFunction.EXECUTIVE_FUNCTION.value: base * 0.6,
        }

    def _find_responsive_channels(
        self, inp: LockedInInput
    ) -> List[CommunicationChannel]:
        """Find channels that show responsiveness."""
        responsive = []

        if inp.locked_in_type in (LockedInType.CLASSIC, LockedInType.INCOMPLETE):
            if inp.eye_tracking_data:
                responsive.append(CommunicationChannel.EYE_MOVEMENT)

        if inp.locked_in_type == LockedInType.INCOMPLETE:
            if inp.muscle_emg_data:
                responsive.append(CommunicationChannel.MUSCLE_TWITCH)

        if inp.bci_signal and inp.neural_signal_strength > 0.3:
            responsive.append(CommunicationChannel.BRAIN_COMPUTER_INTERFACE)

        return responsive

    def _recommend_channels(
        self,
        inp: LockedInInput,
        responsive: List[CommunicationChannel],
    ) -> List[CommunicationChannel]:
        """Recommend optimal communication channels."""
        recommended = list(responsive)

        # Always recommend BCI as it has highest bandwidth potential
        if CommunicationChannel.BRAIN_COMPUTER_INTERFACE not in recommended:
            recommended.append(CommunicationChannel.BRAIN_COMPUTER_INTERFACE)

        # For total locked-in, BCI is the primary option
        if inp.locked_in_type == LockedInType.TOTAL:
            if CommunicationChannel.PUPIL_DILATION not in recommended:
                recommended.append(CommunicationChannel.PUPIL_DILATION)

        return recommended

    def _compute_false_negative_risk(
        self, inp: LockedInInput, awareness_score: float
    ) -> float:
        """Compute risk of incorrectly concluding no awareness."""
        risk = 0.0

        # Low signal quality increases false negative risk
        if inp.signal_quality in (SignalQuality.NOISE, SignalQuality.POOR):
            risk += 0.3

        # Sedation increases risk
        if inp.medication_state == "sedated":
            risk += 0.25

        # Total locked-in has highest false negative risk
        if inp.locked_in_type == LockedInType.TOTAL:
            risk += 0.2

        # Low awareness score with any positive signal is risky
        if awareness_score < 0.3 and inp.neural_signal_strength > 0.2:
            risk += 0.15

        return min(1.0, risk)

    def _generate_assessment_notes(
        self,
        state: AwarenessState,
        inp: LockedInInput,
        fn_risk: float,
    ) -> List[str]:
        """Generate clinical assessment notes."""
        notes = []

        if fn_risk > 0.3:
            notes.append("CAUTION: Elevated false negative risk - reassess with improved signals")

        if state == AwarenessState.UNRESPONSIVE:
            notes.append("No awareness detected, but reassessment strongly recommended")
            notes.append("Consider repeat assessment with multiple stimulus modalities")

        if inp.medication_state == "sedated":
            notes.append("Patient is sedated - awareness may be suppressed by medication")

        if inp.locked_in_type == LockedInType.TOTAL:
            notes.append("Total locked-in state: BCI is primary assessment channel")

        if state == AwarenessState.FULL_AWARENESS:
            notes.append("Full awareness detected - prioritize communication channel establishment")

        return notes

    def _decode_yes_no(
        self, comm: CommunicationAttemptInput
    ) -> Tuple[str, float]:
        """Decode a yes/no response."""
        # Simplified model - in reality would use actual signal processing
        return "yes", 0.6

    def _decode_letter_selection(
        self, comm: CommunicationAttemptInput
    ) -> Tuple[str, float]:
        """Decode a letter selection response."""
        return "A", 0.4

    def _decode_free_response(
        self, comm: CommunicationAttemptInput
    ) -> Tuple[str, float]:
        """Decode a free-form response."""
        return "complex_response", 0.3

    def _compute_signal_stability(self) -> float:
        """Compute stability of signal readings over time."""
        if len(self._monitor_readings) < 3:
            return 0.5

        recent = [r.awareness_level for r in self._monitor_readings[-10:]]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return max(0.0, 1.0 - variance * 5)

    def _estimate_fatigue(self) -> float:
        """Estimate patient fatigue from communication history."""
        if len(self._communication_log) == 0:
            return 0.0

        # More communication attempts increase fatigue
        fatigue = min(1.0, len(self._communication_log) * 0.1)
        return fatigue

    def _compute_alertness_trend(self) -> float:
        """Compute trend in alertness from monitoring history."""
        if len(self._monitor_readings) < 3:
            return 0.0

        recent = [r.awareness_level for r in self._monitor_readings[-5:]]
        if len(recent) < 2:
            return 0.0

        trend = (recent[-1] - recent[0]) / len(recent)
        return max(-1.0, min(1.0, trend * 5))


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "LockedInType",
    "CommunicationChannel",
    "AwarenessState",
    "SignalQuality",
    "CognitiveFunction",
    # Input dataclasses
    "LockedInInput",
    "CommunicationAttemptInput",
    # Output dataclasses
    "DecodedIntention",
    "AwarenessAssessment",
    "CommunicationResult",
    "ConsciousnessMonitorReading",
    # Interface
    "LockedInInterface",
    # Convenience
    "create_locked_in_interface",
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_locked_in_interface() -> LockedInInterface:
    """
    Create and return a new LockedInInterface instance.

    Note: Call await interface.initialize() before use.

    Returns:
        A new LockedInInterface instance.
    """
    return LockedInInterface()
