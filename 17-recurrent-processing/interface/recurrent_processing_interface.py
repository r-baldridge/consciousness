#!/usr/bin/env python3
"""
Recurrent Processing Theory (RPT) Consciousness Interface

Form 17: Implements Recurrent Processing Theory as proposed by Victor Lamme.
RPT posits that consciousness requires recurrent (feedback) processing,
not just feedforward activation. The initial feedforward sweep through
the cortical hierarchy is unconscious; consciousness emerges only when
recurrent signals feed back from higher to lower areas, creating
sustained loops of neural activity. Local recurrence supports
phenomenal consciousness, while global recurrence enables reportable
(access) consciousness.

This module models feedforward vs recurrent processing phases and
their role in the emergence of consciousness.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ProcessingPhase(Enum):
    """
    Phases of neural processing in RPT.

    Processing progresses from feedforward through local and global
    recurrence, with consciousness emerging at recurrent stages.
    """
    FEEDFORWARD = "feedforward"            # Initial bottom-up sweep (unconscious)
    LOCAL_RECURRENCE = "local_recurrence"  # Local feedback loops (phenomenal)
    GLOBAL_RECURRENCE = "global_recurrence"  # Global feedback (access consciousness)
    SUSTAINED = "sustained"                # Sustained recurrent activity
    DECAYING = "decaying"                  # Recurrence fading


class RecurrenceType(Enum):
    """Types of recurrent connections."""
    LATERAL = "lateral"                    # Within same processing level
    FEEDBACK = "feedback"                  # From higher to lower level
    FEEDFORWARD = "feedforward"            # Standard bottom-up (not recurrent)
    RE_ENTRANT = "re_entrant"              # Full re-entrant loop
    TOP_DOWN = "top_down"                  # Top-down modulatory


class ProcessingLevel(Enum):
    """Hierarchical processing levels in the visual/cortical system."""
    PRIMARY = "primary"                    # V1 / primary sensory cortex
    SECONDARY = "secondary"                # V2/V3 / secondary areas
    ASSOCIATION = "association"            # Association cortex
    PREFRONTAL = "prefrontal"             # Prefrontal cortex
    PARIETAL = "parietal"                  # Parietal cortex


class ConsciousnessState(Enum):
    """Consciousness states according to RPT."""
    UNCONSCIOUS = "unconscious"            # Feedforward only
    PHENOMENAL = "phenomenal"              # Local recurrence present
    ACCESS = "access"                      # Global recurrence achieved
    FULL = "full"                          # Sustained global recurrence
    FADING = "fading"                      # Recurrence decaying


class MaskingEffect(Enum):
    """Types of masking that can disrupt recurrence."""
    NONE = "none"                          # No masking
    FORWARD_MASK = "forward_mask"          # Mask precedes target
    BACKWARD_MASK = "backward_mask"        # Mask follows target (disrupts recurrence)
    METACONTRAST = "metacontrast"          # Contour masking
    OBJECT_SUBSTITUTION = "object_substitution"  # Object substitution masking


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class FeedforwardSweep:
    """
    Initial feedforward activation through the processing hierarchy.

    The feedforward sweep carries stimulus information bottom-up
    through cortical areas but does not by itself produce consciousness.
    """
    sweep_id: str
    stimulus_content: Dict[str, Any]    # What was presented
    stimulus_intensity: float           # 0.0-1.0: Stimulus strength
    onset_time_ms: float                # Stimulus onset time
    processing_levels_reached: List[ProcessingLevel]  # How far the sweep went
    activation_strengths: Dict[str, float]  # Level -> activation strength
    duration_ms: float = 50.0          # Typical feedforward sweep duration
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "stimulus_content": self.stimulus_content,
            "stimulus_intensity": round(self.stimulus_intensity, 4),
            "onset_time_ms": self.onset_time_ms,
            "processing_levels": [l.value for l in self.processing_levels_reached],
            "activation_strengths": {k: round(v, 4) for k, v in self.activation_strengths.items()},
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecurrentSignal:
    """
    A recurrent (feedback) signal from a higher to lower processing level.

    These feedback signals are the key ingredient for consciousness
    according to RPT.
    """
    signal_id: str
    source_level: ProcessingLevel       # Where the signal originates
    target_level: ProcessingLevel       # Where the signal feeds back to
    recurrence_type: RecurrenceType
    signal_strength: float              # 0.0-1.0
    latency_ms: float                   # Time for signal to arrive
    content_modulation: Dict[str, float]  # How the signal modulates content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "source_level": self.source_level.value,
            "target_level": self.target_level.value,
            "recurrence_type": self.recurrence_type.value,
            "signal_strength": round(self.signal_strength, 4),
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MaskingInput:
    """Input that may mask (disrupt) recurrent processing."""
    mask_type: MaskingEffect
    mask_strength: float                # 0.0-1.0
    mask_onset_ms: float                # When the mask appears
    stimulus_onset_asynchrony: float    # SOA between target and mask
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_type": self.mask_type.value,
            "mask_strength": round(self.mask_strength, 4),
            "mask_onset_ms": round(self.mask_onset_ms, 2),
            "soa": round(self.stimulus_onset_asynchrony, 2),
        }


@dataclass
class RecurrentProcessingInput:
    """Complete input for recurrent processing."""
    feedforward_sweep: FeedforwardSweep
    masking: Optional[MaskingInput] = None
    attention_modulation: float = 1.0   # 0.0-1.0: Attentional boost
    report_required: bool = False       # Whether verbal report is needed
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class RecurrentState:
    """Current state of recurrent processing."""
    processing_phase: ProcessingPhase
    consciousness_state: ConsciousnessState
    active_recurrent_loops: List[Tuple[str, str]]  # (source, target) level pairs
    recurrence_strength: float          # 0.0-1.0: Overall recurrence strength
    loop_duration_ms: float             # How long recurrence has been active
    is_globally_recurrent: bool         # Whether global recurrence achieved
    is_locally_recurrent: bool          # Whether local recurrence present
    masking_disruption: float           # 0.0-1.0: How much masking disrupted
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_phase": self.processing_phase.value,
            "consciousness_state": self.consciousness_state.value,
            "active_loops": [(s, t) for s, t in self.active_recurrent_loops],
            "recurrence_strength": round(self.recurrence_strength, 4),
            "loop_duration_ms": round(self.loop_duration_ms, 2),
            "is_globally_recurrent": self.is_globally_recurrent,
            "is_locally_recurrent": self.is_locally_recurrent,
            "masking_disruption": round(self.masking_disruption, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsciousnessThresholdResult:
    """Result of checking if consciousness threshold is reached."""
    threshold_reached: bool
    consciousness_state: ConsciousnessState
    recurrence_strength: float
    required_threshold: float
    margin: float                       # How far above/below threshold
    explanation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold_reached": self.threshold_reached,
            "consciousness_state": self.consciousness_state.value,
            "recurrence_strength": round(self.recurrence_strength, 4),
            "required_threshold": round(self.required_threshold, 4),
            "margin": round(self.margin, 4),
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecurrentProcessingOutput:
    """Complete output from recurrent processing."""
    feedforward_result: Dict[str, Any]  # What the feedforward sweep extracted
    recurrent_state: RecurrentState
    consciousness_threshold: ConsciousnessThresholdResult
    recurrent_signals: List[RecurrentSignal]
    total_processing_time_ms: float
    stimulus_percept: Dict[str, Any]    # Final perceived content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedforward_result": self.feedforward_result,
            "recurrent_state": self.recurrent_state.to_dict(),
            "consciousness_threshold": self.consciousness_threshold.to_dict(),
            "num_recurrent_signals": len(self.recurrent_signals),
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "stimulus_percept": self.stimulus_percept,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RPTSystemStatus:
    """Status of the Recurrent Processing Theory system."""
    is_initialized: bool
    total_sweeps_processed: int
    total_recurrent_loops: int
    conscious_events: int
    unconscious_events: int
    average_recurrence_strength: float
    system_health: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# FEEDFORWARD PROCESSING ENGINE
# ============================================================================

class FeedforwardEngine:
    """
    Engine for modeling the initial feedforward sweep.

    The feedforward sweep is rapid (~50ms) and processes stimulus
    information in a strictly bottom-up fashion through the cortical
    hierarchy. It extracts features but does not produce consciousness.
    """

    # Typical latencies for each processing level (ms)
    LEVEL_LATENCIES = {
        ProcessingLevel.PRIMARY: 40.0,
        ProcessingLevel.SECONDARY: 60.0,
        ProcessingLevel.ASSOCIATION: 80.0,
        ProcessingLevel.PREFRONTAL: 120.0,
        ProcessingLevel.PARIETAL: 100.0,
    }

    # Activation decay per level
    LEVEL_DECAY = {
        ProcessingLevel.PRIMARY: 0.95,
        ProcessingLevel.SECONDARY: 0.85,
        ProcessingLevel.ASSOCIATION: 0.75,
        ProcessingLevel.PREFRONTAL: 0.65,
        ProcessingLevel.PARIETAL: 0.70,
    }

    def process_feedforward(
        self, sweep: FeedforwardSweep
    ) -> Dict[str, Any]:
        """
        Process the feedforward sweep and compute activations.

        Returns extracted features and activation levels at each
        hierarchical level.
        """
        result = {
            "sweep_id": sweep.sweep_id,
            "features_extracted": {},
            "activation_profile": {},
            "max_level_reached": None,
            "total_latency_ms": 0.0,
        }

        current_activation = sweep.stimulus_intensity

        for level in sweep.processing_levels_reached:
            decay = self.LEVEL_DECAY.get(level, 0.8)
            current_activation *= decay

            # Add any pre-computed activation
            if level.value in sweep.activation_strengths:
                current_activation = max(
                    current_activation,
                    sweep.activation_strengths[level.value]
                )

            result["activation_profile"][level.value] = current_activation
            result["max_level_reached"] = level.value

            latency = self.LEVEL_LATENCIES.get(level, 80.0)
            result["total_latency_ms"] += latency

            # Extract level-appropriate features
            if level == ProcessingLevel.PRIMARY:
                result["features_extracted"]["edges"] = current_activation * 0.9
                result["features_extracted"]["orientation"] = current_activation * 0.8
            elif level == ProcessingLevel.SECONDARY:
                result["features_extracted"]["contours"] = current_activation * 0.7
                result["features_extracted"]["texture"] = current_activation * 0.6
            elif level == ProcessingLevel.ASSOCIATION:
                result["features_extracted"]["objects"] = current_activation * 0.5
                result["features_extracted"]["categories"] = current_activation * 0.4
            elif level == ProcessingLevel.PREFRONTAL:
                result["features_extracted"]["meaning"] = current_activation * 0.3
                result["features_extracted"]["context"] = current_activation * 0.3

        return result


# ============================================================================
# RECURRENCE ENGINE
# ============================================================================

class RecurrenceEngine:
    """
    Engine for modeling recurrent (feedback) processing.

    Recurrent processing is the key mechanism for consciousness in RPT.
    Local recurrence produces phenomenal consciousness; global recurrence
    enables access consciousness and reportability.
    """

    SIGNAL_COUNTER = 0

    # Thresholds for consciousness
    LOCAL_RECURRENCE_THRESHOLD = 0.3   # Minimum for phenomenal consciousness
    GLOBAL_RECURRENCE_THRESHOLD = 0.5  # Minimum for access consciousness
    FULL_CONSCIOUSNESS_THRESHOLD = 0.7 # Minimum for full consciousness

    def __init__(self):
        self._recurrent_signals: List[RecurrentSignal] = []

    def initiate_recurrence(
        self,
        feedforward_result: Dict[str, Any],
        stimulus_intensity: float,
        attention_modulation: float = 1.0,
        masking: Optional[MaskingInput] = None
    ) -> Tuple[List[RecurrentSignal], RecurrentState]:
        """
        Initiate recurrent processing based on the feedforward sweep result.

        Generates feedback signals from higher to lower levels,
        creating the recurrent loops that support consciousness.
        """
        signals = []
        activation_profile = feedforward_result.get("activation_profile", {})

        # Compute masking disruption
        masking_disruption = self._compute_masking_disruption(masking)

        # Generate recurrent signals between levels
        levels_reached = [
            ProcessingLevel(k) for k in activation_profile.keys()
        ]

        # Sort levels by hierarchy (lower first)
        level_order = [
            ProcessingLevel.PRIMARY,
            ProcessingLevel.SECONDARY,
            ProcessingLevel.ASSOCIATION,
            ProcessingLevel.PARIETAL,
            ProcessingLevel.PREFRONTAL,
        ]
        sorted_levels = [l for l in level_order if l in levels_reached]

        local_recurrence_strength = 0.0
        global_recurrence_strength = 0.0
        active_loops = []

        # Generate local recurrent signals (between adjacent levels)
        for i in range(len(sorted_levels) - 1):
            higher = sorted_levels[i + 1]
            lower = sorted_levels[i]

            higher_activation = activation_profile.get(higher.value, 0.0)
            lower_activation = activation_profile.get(lower.value, 0.0)

            signal_strength = (
                higher_activation * 0.6 +
                lower_activation * 0.4
            ) * attention_modulation * (1.0 - masking_disruption * 0.7)

            signal_strength = max(0.0, min(1.0, signal_strength))

            RecurrenceEngine.SIGNAL_COUNTER += 1
            signal = RecurrentSignal(
                signal_id=f"rec_{RecurrenceEngine.SIGNAL_COUNTER}",
                source_level=higher,
                target_level=lower,
                recurrence_type=RecurrenceType.FEEDBACK,
                signal_strength=signal_strength,
                latency_ms=20.0 + (i * 10.0),
                content_modulation={"enhancement": signal_strength * 0.5},
            )
            signals.append(signal)
            self._recurrent_signals.append(signal)

            local_recurrence_strength = max(local_recurrence_strength, signal_strength)
            active_loops.append((higher.value, lower.value))

        # Generate global recurrent signals (prefrontal/parietal -> primary/secondary)
        if ProcessingLevel.PREFRONTAL in sorted_levels and ProcessingLevel.PRIMARY in sorted_levels:
            pfc_activation = activation_profile.get(ProcessingLevel.PREFRONTAL.value, 0.0)
            primary_activation = activation_profile.get(ProcessingLevel.PRIMARY.value, 0.0)

            global_strength = (
                pfc_activation * 0.5 +
                primary_activation * 0.3 +
                stimulus_intensity * 0.2
            ) * attention_modulation * (1.0 - masking_disruption)

            global_strength = max(0.0, min(1.0, global_strength))

            RecurrenceEngine.SIGNAL_COUNTER += 1
            global_signal = RecurrentSignal(
                signal_id=f"rec_{RecurrenceEngine.SIGNAL_COUNTER}",
                source_level=ProcessingLevel.PREFRONTAL,
                target_level=ProcessingLevel.PRIMARY,
                recurrence_type=RecurrenceType.RE_ENTRANT,
                signal_strength=global_strength,
                latency_ms=80.0,
                content_modulation={"global_broadcast": global_strength * 0.7},
            )
            signals.append(global_signal)
            self._recurrent_signals.append(global_signal)
            global_recurrence_strength = global_strength
            active_loops.append((ProcessingLevel.PREFRONTAL.value, ProcessingLevel.PRIMARY.value))

        # Determine overall recurrence strength
        overall_strength = max(local_recurrence_strength, global_recurrence_strength)

        # Determine processing phase and consciousness state
        phase = self._determine_phase(local_recurrence_strength, global_recurrence_strength)
        consciousness = self._determine_consciousness(
            local_recurrence_strength, global_recurrence_strength
        )

        # Compute loop duration
        loop_duration = sum(s.latency_ms for s in signals)

        state = RecurrentState(
            processing_phase=phase,
            consciousness_state=consciousness,
            active_recurrent_loops=active_loops,
            recurrence_strength=overall_strength,
            loop_duration_ms=loop_duration,
            is_globally_recurrent=global_recurrence_strength >= self.GLOBAL_RECURRENCE_THRESHOLD,
            is_locally_recurrent=local_recurrence_strength >= self.LOCAL_RECURRENCE_THRESHOLD,
            masking_disruption=masking_disruption,
        )

        return signals, state

    def check_conscious_threshold(
        self, recurrent_state: RecurrentState
    ) -> ConsciousnessThresholdResult:
        """
        Check if the recurrent processing has reached the consciousness threshold.

        Different thresholds apply for phenomenal vs access consciousness.
        """
        strength = recurrent_state.recurrence_strength

        if recurrent_state.is_globally_recurrent:
            if strength >= self.FULL_CONSCIOUSNESS_THRESHOLD:
                return ConsciousnessThresholdResult(
                    threshold_reached=True,
                    consciousness_state=ConsciousnessState.FULL,
                    recurrence_strength=strength,
                    required_threshold=self.FULL_CONSCIOUSNESS_THRESHOLD,
                    margin=strength - self.FULL_CONSCIOUSNESS_THRESHOLD,
                    explanation="Full consciousness: strong global recurrence established.",
                )
            else:
                return ConsciousnessThresholdResult(
                    threshold_reached=True,
                    consciousness_state=ConsciousnessState.ACCESS,
                    recurrence_strength=strength,
                    required_threshold=self.GLOBAL_RECURRENCE_THRESHOLD,
                    margin=strength - self.GLOBAL_RECURRENCE_THRESHOLD,
                    explanation="Access consciousness: global recurrence achieved.",
                )
        elif recurrent_state.is_locally_recurrent:
            return ConsciousnessThresholdResult(
                threshold_reached=True,
                consciousness_state=ConsciousnessState.PHENOMENAL,
                recurrence_strength=strength,
                required_threshold=self.LOCAL_RECURRENCE_THRESHOLD,
                margin=strength - self.LOCAL_RECURRENCE_THRESHOLD,
                explanation="Phenomenal consciousness: local recurrence present but no global recurrence.",
            )
        else:
            return ConsciousnessThresholdResult(
                threshold_reached=False,
                consciousness_state=ConsciousnessState.UNCONSCIOUS,
                recurrence_strength=strength,
                required_threshold=self.LOCAL_RECURRENCE_THRESHOLD,
                margin=strength - self.LOCAL_RECURRENCE_THRESHOLD,
                explanation="Unconscious: insufficient recurrent processing.",
            )

    def _compute_masking_disruption(
        self, masking: Optional[MaskingInput]
    ) -> float:
        """Compute how much masking disrupts recurrent processing."""
        if masking is None or masking.mask_type == MaskingEffect.NONE:
            return 0.0

        # Backward masks are most effective at disrupting recurrence
        effectiveness = {
            MaskingEffect.BACKWARD_MASK: 1.0,
            MaskingEffect.OBJECT_SUBSTITUTION: 0.9,
            MaskingEffect.METACONTRAST: 0.8,
            MaskingEffect.FORWARD_MASK: 0.3,
        }.get(masking.mask_type, 0.5)

        # SOA affects masking: short SOA = more disruption
        soa_factor = max(0.0, 1.0 - masking.stimulus_onset_asynchrony / 200.0)

        disruption = masking.mask_strength * effectiveness * soa_factor
        return max(0.0, min(1.0, disruption))

    def _determine_phase(
        self, local: float, global_val: float
    ) -> ProcessingPhase:
        """Determine current processing phase."""
        if global_val >= self.FULL_CONSCIOUSNESS_THRESHOLD:
            return ProcessingPhase.SUSTAINED
        elif global_val >= self.GLOBAL_RECURRENCE_THRESHOLD:
            return ProcessingPhase.GLOBAL_RECURRENCE
        elif local >= self.LOCAL_RECURRENCE_THRESHOLD:
            return ProcessingPhase.LOCAL_RECURRENCE
        else:
            return ProcessingPhase.FEEDFORWARD

    def _determine_consciousness(
        self, local: float, global_val: float
    ) -> ConsciousnessState:
        """Determine consciousness state from recurrence levels."""
        if global_val >= self.FULL_CONSCIOUSNESS_THRESHOLD:
            return ConsciousnessState.FULL
        elif global_val >= self.GLOBAL_RECURRENCE_THRESHOLD:
            return ConsciousnessState.ACCESS
        elif local >= self.LOCAL_RECURRENCE_THRESHOLD:
            return ConsciousnessState.PHENOMENAL
        else:
            return ConsciousnessState.UNCONSCIOUS


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class RecurrentProcessingInterface:
    """
    Main interface for Form 17: Recurrent Processing Theory.

    Implements RPT's model where consciousness requires recurrent
    feedback processing. The feedforward sweep is fast but unconscious;
    recurrent loops from higher to lower areas create the sustained
    activity necessary for conscious experience.
    """

    FORM_ID = "17-recurrent-processing"
    FORM_NAME = "Recurrent Processing Theory (RPT)"

    def __init__(self):
        """Initialize the Recurrent Processing interface."""
        self._feedforward_engine = FeedforwardEngine()
        self._recurrence_engine = RecurrenceEngine()

        # State tracking
        self._is_initialized = False
        self._sweeps_processed = 0
        self._recurrent_loops_total = 0
        self._conscious_events = 0
        self._unconscious_events = 0
        self._recurrence_history: List[float] = []
        self._max_history = 100

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the recurrent processing system."""
        self._is_initialized = True
        logger.info(f"{self.FORM_NAME} initialized and ready")

    async def feedforward_sweep(
        self, sweep: FeedforwardSweep
    ) -> Dict[str, Any]:
        """
        Process the initial feedforward sweep.

        This is the first stage: rapid bottom-up processing that extracts
        features but does not produce consciousness.
        """
        result = self._feedforward_engine.process_feedforward(sweep)
        self._sweeps_processed += 1
        return result

    async def initiate_recurrence(
        self,
        feedforward_result: Dict[str, Any],
        stimulus_intensity: float,
        attention_modulation: float = 1.0,
        masking: Optional[MaskingInput] = None
    ) -> Tuple[List[RecurrentSignal], RecurrentState]:
        """
        Initiate recurrent processing based on the feedforward result.

        This is the critical stage where consciousness may emerge
        through feedback processing loops.
        """
        signals, state = self._recurrence_engine.initiate_recurrence(
            feedforward_result, stimulus_intensity,
            attention_modulation, masking
        )

        self._recurrent_loops_total += len(state.active_recurrent_loops)
        self._recurrence_history.append(state.recurrence_strength)
        if len(self._recurrence_history) > self._max_history:
            self._recurrence_history.pop(0)

        return signals, state

    async def get_processing_phase(
        self, recurrent_state: RecurrentState
    ) -> ProcessingPhase:
        """
        Get the current processing phase.

        Returns whether the system is in feedforward, local recurrence,
        or global recurrence phase.
        """
        return recurrent_state.processing_phase

    async def check_conscious_threshold(
        self, recurrent_state: RecurrentState
    ) -> ConsciousnessThresholdResult:
        """
        Check if the consciousness threshold has been reached.

        Evaluates whether recurrent processing is strong enough
        for phenomenal or access consciousness.
        """
        result = self._recurrence_engine.check_conscious_threshold(recurrent_state)

        if result.consciousness_state in [
            ConsciousnessState.PHENOMENAL,
            ConsciousnessState.ACCESS,
            ConsciousnessState.FULL
        ]:
            self._conscious_events += 1
        else:
            self._unconscious_events += 1

        return result

    async def process_stimulus(
        self,
        rp_input: RecurrentProcessingInput
    ) -> RecurrentProcessingOutput:
        """
        Full processing pipeline: feedforward sweep then recurrent processing.

        This is the main entry point that models the complete
        sequence from stimulus onset to conscious percept (or not).
        """
        # Stage 1: Feedforward sweep
        ff_result = await self.feedforward_sweep(rp_input.feedforward_sweep)

        # Stage 2: Recurrent processing
        signals, rec_state = await self.initiate_recurrence(
            ff_result,
            rp_input.feedforward_sweep.stimulus_intensity,
            rp_input.attention_modulation,
            rp_input.masking,
        )

        # Stage 3: Check consciousness threshold
        threshold_result = await self.check_conscious_threshold(rec_state)

        # Build final percept
        stimulus_percept = self._build_percept(
            ff_result, rec_state, rp_input.feedforward_sweep.stimulus_content
        )

        # Total processing time
        ff_time = ff_result.get("total_latency_ms", 50.0)
        rec_time = rec_state.loop_duration_ms
        total_time = ff_time + rec_time

        return RecurrentProcessingOutput(
            feedforward_result=ff_result,
            recurrent_state=rec_state,
            consciousness_threshold=threshold_result,
            recurrent_signals=signals,
            total_processing_time_ms=total_time,
            stimulus_percept=stimulus_percept,
        )

    def _build_percept(
        self,
        ff_result: Dict[str, Any],
        rec_state: RecurrentState,
        original_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final percept based on processing results."""
        percept = dict(original_content)
        features = ff_result.get("features_extracted", {})

        # Recurrence enhances features
        enhancement = rec_state.recurrence_strength
        for key, value in features.items():
            if isinstance(value, (int, float)):
                percept[f"processed_{key}"] = value * (1.0 + enhancement * 0.5)

        percept["consciousness_state"] = rec_state.consciousness_state.value
        percept["processing_phase"] = rec_state.processing_phase.value
        percept["recurrence_strength"] = rec_state.recurrence_strength

        return percept

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        avg_recurrence = (
            sum(self._recurrence_history) / len(self._recurrence_history)
            if self._recurrence_history else 0.0
        )
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "is_initialized": self._is_initialized,
            "sweeps_processed": self._sweeps_processed,
            "recurrent_loops_total": self._recurrent_loops_total,
            "conscious_events": self._conscious_events,
            "unconscious_events": self._unconscious_events,
            "average_recurrence_strength": round(avg_recurrence, 4),
        }

    def get_status(self) -> RPTSystemStatus:
        """Get current system status."""
        avg_recurrence = (
            sum(self._recurrence_history) / len(self._recurrence_history)
            if self._recurrence_history else 0.0
        )
        return RPTSystemStatus(
            is_initialized=self._is_initialized,
            total_sweeps_processed=self._sweeps_processed,
            total_recurrent_loops=self._recurrent_loops_total,
            conscious_events=self._conscious_events,
            unconscious_events=self._unconscious_events,
            average_recurrence_strength=avg_recurrence,
            system_health=1.0 if self._is_initialized else 0.5,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_recurrent_processing_interface() -> RecurrentProcessingInterface:
    """Create and return a Recurrent Processing interface."""
    return RecurrentProcessingInterface()


def create_feedforward_sweep(
    sweep_id: str,
    stimulus_intensity: float = 0.7,
    content: Optional[Dict[str, Any]] = None,
    levels: Optional[List[ProcessingLevel]] = None,
) -> FeedforwardSweep:
    """Create a feedforward sweep for testing."""
    if levels is None:
        levels = [
            ProcessingLevel.PRIMARY,
            ProcessingLevel.SECONDARY,
            ProcessingLevel.ASSOCIATION,
            ProcessingLevel.PREFRONTAL,
        ]

    activations = {}
    current = stimulus_intensity
    for level in levels:
        activations[level.value] = current
        current *= 0.85

    return FeedforwardSweep(
        sweep_id=sweep_id,
        stimulus_content=content or {"stimulus": "test"},
        stimulus_intensity=stimulus_intensity,
        onset_time_ms=0.0,
        processing_levels_reached=levels,
        activation_strengths=activations,
    )


def create_masking_input(
    mask_type: MaskingEffect = MaskingEffect.BACKWARD_MASK,
    mask_strength: float = 0.8,
    soa: float = 50.0,
) -> MaskingInput:
    """Create a masking input for testing."""
    return MaskingInput(
        mask_type=mask_type,
        mask_strength=mask_strength,
        mask_onset_ms=soa,
        stimulus_onset_asynchrony=soa,
    )


__all__ = [
    # Enums
    "ProcessingPhase",
    "RecurrenceType",
    "ProcessingLevel",
    "ConsciousnessState",
    "MaskingEffect",
    # Input dataclasses
    "FeedforwardSweep",
    "RecurrentSignal",
    "MaskingInput",
    "RecurrentProcessingInput",
    # Output dataclasses
    "RecurrentState",
    "ConsciousnessThresholdResult",
    "RecurrentProcessingOutput",
    "RPTSystemStatus",
    # Engines
    "FeedforwardEngine",
    "RecurrenceEngine",
    # Main interface
    "RecurrentProcessingInterface",
    # Convenience functions
    "create_recurrent_processing_interface",
    "create_feedforward_sweep",
    "create_masking_input",
]
