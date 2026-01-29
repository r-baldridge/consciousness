#!/usr/bin/env python3
"""
Interoceptive Consciousness Interface

Form 06: The interoceptive processing system for consciousness.
Interoceptive consciousness processes internal body signals including
heartbeat awareness, breathing, gut feelings, hunger, thirst, temperature
regulation, and visceral sensations to construct conscious awareness
of bodily state.

This form is foundational for emotional experience, self-awareness,
and homeostatic regulation. Interoception provides the "felt sense"
of being alive and grounds consciousness in embodied experience.
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

class InteroceptiveChannel(Enum):
    """Channels of interoceptive information."""
    CARDIAC = "cardiac"              # Heart rate, rhythm, force
    RESPIRATORY = "respiratory"      # Breathing rate, depth, effort
    GASTROINTESTINAL = "gastrointestinal"  # Gut activity, hunger, fullness
    THERMOREGULATORY = "thermoregulatory"  # Body temperature regulation
    NOCICEPTIVE = "nociceptive"      # Internal pain signals
    BLADDER = "bladder"              # Urinary urgency
    IMMUNE = "immune"                # Inflammation, sickness signals
    METABOLIC = "metabolic"          # Blood sugar, energy state
    MUSCULAR = "muscular"            # Fatigue, tension, energy
    OSMOTIC = "osmotic"              # Thirst, hydration


class BodySystem(Enum):
    """Major body systems for interoceptive monitoring."""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY_SYSTEM = "respiratory_system"
    DIGESTIVE = "digestive"
    NERVOUS = "nervous"
    ENDOCRINE = "endocrine"
    IMMUNE_SYSTEM = "immune_system"
    MUSCULOSKELETAL = "musculoskeletal"
    URINARY = "urinary"


class HomeostaticNeed(Enum):
    """Homeostatic needs detected through interoception."""
    HUNGER = "hunger"
    THIRST = "thirst"
    OXYGEN = "oxygen"
    WARMTH = "warmth"
    COOLING = "cooling"
    REST = "rest"
    ELIMINATION = "elimination"
    MOVEMENT = "movement"
    SAFETY = "safety"
    NONE = "none"


class BodyStateCategory(Enum):
    """Overall body state categories."""
    OPTIMAL = "optimal"           # All systems balanced
    STRESSED = "stressed"         # Elevated sympathetic activity
    FATIGUED = "fatigued"         # Energy depletion
    ILL = "ill"                   # Immune activation / sickness
    ENERGIZED = "energized"       # High energy, ready state
    RELAXED = "relaxed"           # Parasympathetic dominant
    ANXIOUS = "anxious"           # Sympathetic activation without threat
    DEPLETED = "depleted"         # Resource depletion
    RECOVERING = "recovering"     # Post-exertion recovery


class InteroceptiveAccuracy(Enum):
    """Levels of interoceptive accuracy (body awareness)."""
    HIGH = "high"                  # Accurate body signal detection
    MODERATE = "moderate"
    LOW = "low"                    # Poor body signal detection
    HYPERSENSITIVE = "hypersensitive"  # Excessive body signal awareness
    DISSOCIATED = "dissociated"    # Disconnected from body signals


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class OrganSignal:
    """Signal from a specific organ or body system."""
    channel: InteroceptiveChannel
    body_system: BodySystem
    activation_level: float    # 0.0-1.0
    deviation_from_baseline: float  # -1.0 to 1.0 (how far from normal)
    urgency: float             # 0.0-1.0
    signal_quality: float = 0.8  # Reliability of signal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "body_system": self.body_system.value,
            "activation_level": round(self.activation_level, 4),
            "deviation_from_baseline": round(self.deviation_from_baseline, 4),
            "urgency": round(self.urgency, 4),
            "signal_quality": round(self.signal_quality, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HomeostaticData:
    """Homeostatic regulation data."""
    body_temperature: float = 0.5    # 0.0 (cold) - 0.5 (normal) - 1.0 (hot)
    blood_glucose: float = 0.5       # 0.0 (low) - 0.5 (normal) - 1.0 (high)
    hydration_level: float = 0.7     # 0.0 (dehydrated) - 1.0 (overhydrated)
    oxygen_saturation: float = 0.95  # 0.0-1.0
    heart_rate_normalized: float = 0.5  # 0.0 (bradycardia) - 0.5 (normal) - 1.0 (tachycardia)
    respiratory_rate_normalized: float = 0.5  # Similarly normalized
    energy_reserves: float = 0.6     # 0.0-1.0
    stress_hormones: float = 0.3     # 0.0-1.0 (cortisol, adrenaline)
    immune_activation: float = 0.1   # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_temperature": round(self.body_temperature, 4),
            "blood_glucose": round(self.blood_glucose, 4),
            "hydration_level": round(self.hydration_level, 4),
            "oxygen_saturation": round(self.oxygen_saturation, 4),
            "heart_rate_normalized": round(self.heart_rate_normalized, 4),
            "respiratory_rate_normalized": round(self.respiratory_rate_normalized, 4),
            "energy_reserves": round(self.energy_reserves, 4),
            "stress_hormones": round(self.stress_hormones, 4),
            "immune_activation": round(self.immune_activation, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InteroceptiveInput:
    """Complete input to the interoceptive consciousness system."""
    organ_signals: List[OrganSignal] = field(default_factory=list)
    homeostatic_data: Optional[HomeostaticData] = None
    emotional_body_state: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    gut_feeling: float = 0.0           # -1.0 (bad) to 1.0 (good)
    subjective_energy: float = 0.5     # 0.0-1.0
    body_awareness_focus: float = 0.3  # 0.0-1.0 how much attention on body
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_organ_signals": len(self.organ_signals),
            "has_homeostatic_data": self.homeostatic_data is not None,
            "emotional_body_state": round(self.emotional_body_state, 4),
            "gut_feeling": round(self.gut_feeling, 4),
            "subjective_energy": round(self.subjective_energy, 4),
            "body_awareness_focus": round(self.body_awareness_focus, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class BodyStateAssessment:
    """Assessment of overall body state."""
    state_category: BodyStateCategory
    overall_wellbeing: float     # -1.0 to 1.0
    sympathetic_activation: float  # 0.0-1.0
    parasympathetic_activation: float  # 0.0-1.0
    autonomic_balance: float     # -1.0 (sympathetic) to 1.0 (parasympathetic)
    body_coherence: float        # 0.0-1.0 (systems working in harmony)
    vitality: float              # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_category": self.state_category.value,
            "overall_wellbeing": round(self.overall_wellbeing, 4),
            "sympathetic_activation": round(self.sympathetic_activation, 4),
            "parasympathetic_activation": round(self.parasympathetic_activation, 4),
            "autonomic_balance": round(self.autonomic_balance, 4),
            "body_coherence": round(self.body_coherence, 4),
            "vitality": round(self.vitality, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HomeostaticNeedsReport:
    """Report of detected homeostatic needs."""
    active_needs: List[HomeostaticNeed]
    primary_need: HomeostaticNeed
    primary_urgency: float       # 0.0-1.0
    need_intensities: Dict[str, float]  # need -> intensity
    homeostatic_deviation: float  # 0.0-1.0 overall deviation from optimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_needs": [n.value for n in self.active_needs],
            "primary_need": self.primary_need.value,
            "primary_urgency": round(self.primary_urgency, 4),
            "need_intensities": {k: round(v, 4) for k, v in self.need_intensities.items()},
            "homeostatic_deviation": round(self.homeostatic_deviation, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmotionalGrounding:
    """Interoceptive grounding for emotional experience."""
    body_valence: float          # -1.0 to 1.0 (body-based feeling)
    body_arousal: float          # 0.0-1.0
    gut_intuition: float         # -1.0 to 1.0
    felt_sense_clarity: float    # 0.0-1.0 how clear the body feeling is
    emotional_readiness: float   # 0.0-1.0 readiness for emotional processing
    somatic_markers: List[str]   # Active somatic markers
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_valence": round(self.body_valence, 4),
            "body_arousal": round(self.body_arousal, 4),
            "gut_intuition": round(self.gut_intuition, 4),
            "felt_sense_clarity": round(self.felt_sense_clarity, 4),
            "emotional_readiness": round(self.emotional_readiness, 4),
            "somatic_markers": self.somatic_markers,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InteroceptiveOutput:
    """Complete output of interoceptive consciousness processing."""
    body_state: BodyStateAssessment
    homeostatic_needs: HomeostaticNeedsReport
    emotional_grounding: EmotionalGrounding
    interoceptive_accuracy: InteroceptiveAccuracy
    body_awareness_level: float   # 0.0-1.0
    requires_action: bool
    action_priority: float = 0.0  # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_state": self.body_state.to_dict(),
            "homeostatic_needs": self.homeostatic_needs.to_dict(),
            "emotional_grounding": self.emotional_grounding.to_dict(),
            "interoceptive_accuracy": self.interoceptive_accuracy.value,
            "body_awareness_level": round(self.body_awareness_level, 4),
            "requires_action": self.requires_action,
            "action_priority": round(self.action_priority, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class InteroceptiveConsciousnessInterface:
    """
    Main interface for Form 06: Interoceptive Consciousness.

    Processes internal body signals through organ signal analysis,
    homeostatic monitoring, need detection, and emotional grounding
    to produce conscious body state awareness. This form provides
    the embodied foundation for emotional experience and self-awareness.
    """

    FORM_ID = "06-interoceptive"
    FORM_NAME = "Interoceptive Consciousness"

    def __init__(self):
        """Initialize the interoceptive consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[InteroceptiveOutput] = None
        self._body_state_history: List[BodyStateCategory] = []
        self._need_history: List[HomeostaticNeed] = []
        self._baseline_homeostatic = HomeostaticData()  # Normal baseline
        self._interoceptive_sensitivity = 0.5  # 0.0-1.0
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the interoceptive processing pipeline."""
        self._initialized = True
        self._interoceptive_sensitivity = 0.5
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_interoceptive_input(
        self, interoceptive_input: InteroceptiveInput
    ) -> InteroceptiveOutput:
        """
        Process interoceptive input through the consciousness pipeline.

        Pipeline stages:
        1. Body state assessment
        2. Homeostatic need detection
        3. Emotional grounding
        4. Interoceptive accuracy evaluation
        5. Action requirement assessment
        """
        self._processing_count += 1

        # Stage 1: Assess body state
        body_state = await self._assess_body_state(interoceptive_input)

        # Stage 2: Detect homeostatic needs
        needs = await self._detect_homeostatic_needs(interoceptive_input)

        # Stage 3: Compute emotional grounding
        emotional = await self._compute_emotional_grounding(
            interoceptive_input, body_state, needs
        )

        # Stage 4: Evaluate interoceptive accuracy
        accuracy = self._evaluate_accuracy(interoceptive_input)

        # Stage 5: Determine action requirements
        awareness = self._compute_body_awareness(interoceptive_input, body_state)
        requires_action = needs.primary_urgency > 0.6 or body_state.overall_wellbeing < -0.5
        action_priority = max(needs.primary_urgency, abs(min(0, body_state.overall_wellbeing)))

        output = InteroceptiveOutput(
            body_state=body_state,
            homeostatic_needs=needs,
            emotional_grounding=emotional,
            interoceptive_accuracy=accuracy,
            body_awareness_level=awareness,
            requires_action=requires_action,
            action_priority=action_priority,
        )

        self._current_output = output
        self._update_history(body_state.state_category, needs.primary_need)

        return output

    async def _assess_body_state(
        self, interoceptive_input: InteroceptiveInput
    ) -> BodyStateAssessment:
        """Assess overall body state from interoceptive signals."""
        homeostatic = interoceptive_input.homeostatic_data or self._baseline_homeostatic

        # Compute sympathetic/parasympathetic balance
        sympathetic = self._compute_sympathetic(homeostatic, interoceptive_input)
        parasympathetic = self._compute_parasympathetic(homeostatic, interoceptive_input)
        balance = parasympathetic - sympathetic  # Positive = parasympathetic dominant

        # Compute overall wellbeing
        wellbeing = self._compute_wellbeing(homeostatic, interoceptive_input)

        # Compute vitality
        vitality = (
            homeostatic.energy_reserves * 0.4 +
            interoceptive_input.subjective_energy * 0.3 +
            homeostatic.oxygen_saturation * 0.3
        )

        # Compute body coherence
        deviations = [
            abs(homeostatic.body_temperature - 0.5),
            abs(homeostatic.blood_glucose - 0.5),
            abs(homeostatic.heart_rate_normalized - 0.5),
            abs(homeostatic.respiratory_rate_normalized - 0.5),
        ]
        coherence = max(0.0, 1.0 - sum(deviations) / len(deviations) * 2)

        # Determine state category
        state = self._determine_body_state(
            wellbeing, sympathetic, parasympathetic, homeostatic, interoceptive_input
        )

        return BodyStateAssessment(
            state_category=state,
            overall_wellbeing=wellbeing,
            sympathetic_activation=sympathetic,
            parasympathetic_activation=parasympathetic,
            autonomic_balance=max(-1.0, min(1.0, balance)),
            body_coherence=coherence,
            vitality=min(1.0, vitality),
        )

    async def _detect_homeostatic_needs(
        self, interoceptive_input: InteroceptiveInput
    ) -> HomeostaticNeedsReport:
        """Detect homeostatic needs from body signals."""
        homeostatic = interoceptive_input.homeostatic_data or self._baseline_homeostatic
        needs = []
        intensities = {}

        # Check hunger
        if homeostatic.blood_glucose < 0.3:
            needs.append(HomeostaticNeed.HUNGER)
            intensities["hunger"] = (0.3 - homeostatic.blood_glucose) / 0.3

        # Check thirst
        if homeostatic.hydration_level < 0.4:
            needs.append(HomeostaticNeed.THIRST)
            intensities["thirst"] = (0.4 - homeostatic.hydration_level) / 0.4

        # Check oxygen
        if homeostatic.oxygen_saturation < 0.9:
            needs.append(HomeostaticNeed.OXYGEN)
            intensities["oxygen"] = (0.9 - homeostatic.oxygen_saturation) / 0.9

        # Check warmth/cooling
        if homeostatic.body_temperature < 0.3:
            needs.append(HomeostaticNeed.WARMTH)
            intensities["warmth"] = (0.3 - homeostatic.body_temperature) / 0.3
        elif homeostatic.body_temperature > 0.7:
            needs.append(HomeostaticNeed.COOLING)
            intensities["cooling"] = (homeostatic.body_temperature - 0.7) / 0.3

        # Check rest
        if homeostatic.energy_reserves < 0.2:
            needs.append(HomeostaticNeed.REST)
            intensities["rest"] = (0.2 - homeostatic.energy_reserves) / 0.2

        # Check movement (prolonged stillness)
        if interoceptive_input.subjective_energy > 0.7 and homeostatic.energy_reserves > 0.6:
            organ_fatigue = any(
                s.channel == InteroceptiveChannel.MUSCULAR and s.activation_level < 0.2
                for s in interoceptive_input.organ_signals
            )
            if organ_fatigue:
                needs.append(HomeostaticNeed.MOVEMENT)
                intensities["movement"] = 0.4

        if not needs:
            needs.append(HomeostaticNeed.NONE)
            primary = HomeostaticNeed.NONE
            urgency = 0.0
        else:
            # Primary need is most urgent
            primary = needs[0]
            urgency = max(intensities.values()) if intensities else 0.0
            for need in needs:
                need_key = need.value
                if intensities.get(need_key, 0) > intensities.get(primary.value, 0):
                    primary = need

        deviation = sum(intensities.values()) / max(1, len(intensities)) if intensities else 0.0

        return HomeostaticNeedsReport(
            active_needs=needs,
            primary_need=primary,
            primary_urgency=min(1.0, urgency),
            need_intensities=intensities,
            homeostatic_deviation=min(1.0, deviation),
        )

    async def _compute_emotional_grounding(
        self,
        interoceptive_input: InteroceptiveInput,
        body_state: BodyStateAssessment,
        needs: HomeostaticNeedsReport,
    ) -> EmotionalGrounding:
        """Compute interoceptive grounding for emotional experience."""
        # Body valence from wellbeing and gut feeling
        body_valence = (
            body_state.overall_wellbeing * 0.5 +
            interoceptive_input.gut_feeling * 0.3 +
            interoceptive_input.emotional_body_state * 0.2
        )

        # Body arousal from sympathetic activation
        body_arousal = body_state.sympathetic_activation

        # Gut intuition
        gut_intuition = interoceptive_input.gut_feeling

        # Felt sense clarity depends on body awareness and signal quality
        signals = interoceptive_input.organ_signals
        avg_quality = (
            sum(s.signal_quality for s in signals) / max(1, len(signals))
            if signals else 0.5
        )
        felt_sense_clarity = (
            interoceptive_input.body_awareness_focus * 0.5 +
            avg_quality * 0.3 +
            self._interoceptive_sensitivity * 0.2
        )

        # Emotional readiness
        readiness = body_state.body_coherence * 0.5 + body_state.vitality * 0.5

        # Somatic markers
        markers = []
        if body_state.sympathetic_activation > 0.7:
            markers.append("elevated_arousal")
        if body_state.overall_wellbeing < -0.3:
            markers.append("distress")
        if interoceptive_input.gut_feeling < -0.3:
            markers.append("unease")
        if interoceptive_input.gut_feeling > 0.3:
            markers.append("confidence")
        if needs.primary_urgency > 0.5:
            markers.append("need_state")
        if body_state.vitality > 0.7:
            markers.append("vigor")

        return EmotionalGrounding(
            body_valence=max(-1.0, min(1.0, body_valence)),
            body_arousal=body_arousal,
            gut_intuition=gut_intuition,
            felt_sense_clarity=min(1.0, felt_sense_clarity),
            emotional_readiness=min(1.0, readiness),
            somatic_markers=markers,
        )

    def _compute_sympathetic(
        self, homeostatic: HomeostaticData, interoceptive_input: InteroceptiveInput
    ) -> float:
        """Compute sympathetic nervous system activation."""
        sympathetic = (
            homeostatic.heart_rate_normalized * 0.3 +
            homeostatic.stress_hormones * 0.3 +
            homeostatic.respiratory_rate_normalized * 0.2 +
            max(0, -interoceptive_input.gut_feeling) * 0.2
        )
        return max(0.0, min(1.0, sympathetic))

    def _compute_parasympathetic(
        self, homeostatic: HomeostaticData, interoceptive_input: InteroceptiveInput
    ) -> float:
        """Compute parasympathetic nervous system activation."""
        parasympathetic = (
            (1.0 - homeostatic.heart_rate_normalized) * 0.3 +
            (1.0 - homeostatic.stress_hormones) * 0.3 +
            homeostatic.energy_reserves * 0.2 +
            max(0, interoceptive_input.gut_feeling) * 0.2
        )
        return max(0.0, min(1.0, parasympathetic))

    def _compute_wellbeing(
        self, homeostatic: HomeostaticData, interoceptive_input: InteroceptiveInput
    ) -> float:
        """Compute overall wellbeing score."""
        # Deviations from normal reduce wellbeing
        deviations = [
            abs(homeostatic.body_temperature - 0.5),
            abs(homeostatic.blood_glucose - 0.5),
            abs(homeostatic.heart_rate_normalized - 0.5),
        ]
        avg_deviation = sum(deviations) / len(deviations)

        wellbeing = (
            interoceptive_input.subjective_energy * 0.3 +
            (1.0 - avg_deviation) * 0.3 +
            interoceptive_input.gut_feeling * 0.2 +
            (1.0 - homeostatic.stress_hormones) * 0.2
        )
        return max(-1.0, min(1.0, wellbeing * 2 - 1.0))

    def _determine_body_state(
        self,
        wellbeing: float,
        sympathetic: float,
        parasympathetic: float,
        homeostatic: HomeostaticData,
        interoceptive_input: InteroceptiveInput,
    ) -> BodyStateCategory:
        """Determine the overall body state category."""
        if homeostatic.immune_activation > 0.5:
            return BodyStateCategory.ILL
        if homeostatic.energy_reserves < 0.2:
            return BodyStateCategory.DEPLETED
        if wellbeing < -0.5 and sympathetic > 0.7:
            return BodyStateCategory.ANXIOUS
        if sympathetic > 0.7:
            return BodyStateCategory.STRESSED
        if homeostatic.energy_reserves < 0.3 and interoceptive_input.subjective_energy < 0.3:
            return BodyStateCategory.FATIGUED
        if interoceptive_input.subjective_energy > 0.7 and wellbeing > 0.3:
            return BodyStateCategory.ENERGIZED
        if parasympathetic > 0.6 and wellbeing > 0.0:
            return BodyStateCategory.RELAXED
        if wellbeing > 0.2:
            return BodyStateCategory.OPTIMAL
        return BodyStateCategory.RECOVERING

    def _evaluate_accuracy(self, interoceptive_input: InteroceptiveInput) -> InteroceptiveAccuracy:
        """Evaluate interoceptive accuracy."""
        focus = interoceptive_input.body_awareness_focus
        sensitivity = self._interoceptive_sensitivity

        if focus > 0.8 and sensitivity > 0.8:
            return InteroceptiveAccuracy.HYPERSENSITIVE
        if focus > 0.5 and sensitivity > 0.5:
            return InteroceptiveAccuracy.HIGH
        if focus > 0.3 or sensitivity > 0.3:
            return InteroceptiveAccuracy.MODERATE
        if focus < 0.1 and sensitivity < 0.1:
            return InteroceptiveAccuracy.DISSOCIATED
        return InteroceptiveAccuracy.LOW

    def _compute_body_awareness(
        self, interoceptive_input: InteroceptiveInput, body_state: BodyStateAssessment
    ) -> float:
        """Compute level of body awareness."""
        awareness = (
            interoceptive_input.body_awareness_focus * 0.4 +
            self._interoceptive_sensitivity * 0.3 +
            len(interoceptive_input.organ_signals) * 0.05 +
            (1.0 - body_state.body_coherence) * 0.2  # Disruption increases awareness
        )
        return max(0.0, min(1.0, awareness))

    def _update_history(self, state: BodyStateCategory, need: HomeostaticNeed) -> None:
        """Update processing history."""
        self._body_state_history.append(state)
        self._need_history.append(need)
        if len(self._body_state_history) > self._max_history:
            self._body_state_history.pop(0)
        if len(self._need_history) > self._max_history:
            self._need_history.pop(0)

    def set_interoceptive_sensitivity(self, sensitivity: float) -> None:
        """Set interoceptive sensitivity level."""
        self._interoceptive_sensitivity = max(0.0, min(1.0, sensitivity))

    def get_interoceptive_sensitivity(self) -> float:
        """Get current interoceptive sensitivity."""
        return self._interoceptive_sensitivity

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "interoceptive_sensitivity": round(self._interoceptive_sensitivity, 4),
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "state_history_length": len(self._body_state_history),
            "need_history_length": len(self._need_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "interoceptive_sensitivity": round(self._interoceptive_sensitivity, 4),
            "body_awareness": (
                self._current_output.body_awareness_level if self._current_output else 0.0
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_interoceptive_interface() -> InteroceptiveConsciousnessInterface:
    """Create and return an interoceptive consciousness interface."""
    return InteroceptiveConsciousnessInterface()


def create_simple_interoceptive_input(
    energy: float = 0.5,
    stress: float = 0.3,
    gut_feeling: float = 0.0,
    body_temperature: float = 0.5,
    blood_glucose: float = 0.5,
    hydration: float = 0.7,
) -> InteroceptiveInput:
    """Create a simple interoceptive input for testing."""
    return InteroceptiveInput(
        homeostatic_data=HomeostaticData(
            body_temperature=body_temperature,
            blood_glucose=blood_glucose,
            hydration_level=hydration,
            energy_reserves=energy,
            stress_hormones=stress,
        ),
        subjective_energy=energy,
        gut_feeling=gut_feeling,
        body_awareness_focus=0.3,
    )


__all__ = [
    # Enums
    "InteroceptiveChannel",
    "BodySystem",
    "HomeostaticNeed",
    "BodyStateCategory",
    "InteroceptiveAccuracy",
    # Input dataclasses
    "OrganSignal",
    "HomeostaticData",
    "InteroceptiveInput",
    # Output dataclasses
    "BodyStateAssessment",
    "HomeostaticNeedsReport",
    "EmotionalGrounding",
    "InteroceptiveOutput",
    # Main interface
    "InteroceptiveConsciousnessInterface",
    # Convenience functions
    "create_interoceptive_interface",
    "create_simple_interoceptive_input",
]
