#!/usr/bin/env python3
"""
Somatosensory Consciousness Interface

Form 03: The somatosensory processing system for consciousness.
Somatosensory consciousness processes touch, pressure, temperature,
pain, proprioception, and kinesthesia to construct a coherent body
experience and body schema.

This form handles the transformation of bodily stimuli into conscious
awareness of touch, pain, body position, and physical boundaries.
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

class TouchType(Enum):
    """Types of tactile stimulation."""
    LIGHT_TOUCH = "light_touch"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    TEXTURE = "texture"
    TEMPERATURE_WARM = "temperature_warm"
    TEMPERATURE_COLD = "temperature_cold"
    ITCH = "itch"
    TICKLE = "tickle"
    NONE = "none"


class PainType(Enum):
    """Types of pain signals."""
    SHARP = "sharp"          # Fast, acute pain (A-delta fibers)
    DULL = "dull"            # Slow, diffuse pain (C fibers)
    BURNING = "burning"
    THROBBING = "throbbing"
    ACHING = "aching"
    TINGLING = "tingling"
    NEUROPATHIC = "neuropathic"
    REFERRED = "referred"
    NONE = "none"


class BodyRegion(Enum):
    """Body regions for somatosensory mapping."""
    HEAD = "head"
    FACE = "face"
    NECK = "neck"
    SHOULDER_LEFT = "shoulder_left"
    SHOULDER_RIGHT = "shoulder_right"
    ARM_LEFT = "arm_left"
    ARM_RIGHT = "arm_right"
    HAND_LEFT = "hand_left"
    HAND_RIGHT = "hand_right"
    TORSO_FRONT = "torso_front"
    TORSO_BACK = "torso_back"
    HIP_LEFT = "hip_left"
    HIP_RIGHT = "hip_right"
    LEG_LEFT = "leg_left"
    LEG_RIGHT = "leg_right"
    FOOT_LEFT = "foot_left"
    FOOT_RIGHT = "foot_right"
    INTERNAL = "internal"


class ProprioceptiveChannel(Enum):
    """Proprioceptive information channels."""
    JOINT_POSITION = "joint_position"
    MUSCLE_TENSION = "muscle_tension"
    BALANCE = "balance"
    MOVEMENT_SENSE = "movement_sense"    # Kinesthesia
    FORCE_SENSE = "force_sense"
    BODY_POSITION = "body_position"


class BodySchemaState(Enum):
    """States of body schema awareness."""
    NORMAL = "normal"
    HEIGHTENED = "heightened"       # Increased body awareness
    DIMINISHED = "diminished"       # Reduced body awareness
    DISTORTED = "distorted"         # Altered body perception
    PHANTOM = "phantom"             # Phantom sensations
    DISSOCIATED = "dissociated"     # Feeling disconnected from body


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class TouchInput:
    """A tactile stimulus input."""
    touch_type: TouchType
    body_region: BodyRegion
    intensity: float          # 0.0-1.0
    area_size: float          # Relative size of touched area 0.0-1.0
    duration_ms: float        # Duration of touch
    temperature: float = 0.5  # 0.0 (cold) - 0.5 (neutral) - 1.0 (hot)
    is_self_generated: bool = False  # Self-touch vs external
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "touch_type": self.touch_type.value,
            "body_region": self.body_region.value,
            "intensity": round(self.intensity, 4),
            "area_size": round(self.area_size, 4),
            "duration_ms": round(self.duration_ms, 2),
            "temperature": round(self.temperature, 4),
            "is_self_generated": self.is_self_generated,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PainInput:
    """A pain signal input."""
    pain_type: PainType
    body_region: BodyRegion
    intensity: float          # 0.0-1.0 (subjective pain level)
    sharpness: float          # 0.0 (dull) to 1.0 (sharp)
    duration_ms: float
    is_chronic: bool = False  # Acute vs chronic
    tissue_damage: float = 0.0  # Estimated tissue damage level
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pain_type": self.pain_type.value,
            "body_region": self.body_region.value,
            "intensity": round(self.intensity, 4),
            "sharpness": round(self.sharpness, 4),
            "duration_ms": round(self.duration_ms, 2),
            "is_chronic": self.is_chronic,
            "tissue_damage": round(self.tissue_damage, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProprioceptiveInput:
    """Proprioceptive body position data."""
    channel: ProprioceptiveChannel
    body_region: BodyRegion
    value: float              # Channel-specific value (0.0-1.0)
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "body_region": self.body_region.value,
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SomatosensoryInput:
    """Complete input to the somatosensory system."""
    touch_inputs: List[TouchInput] = field(default_factory=list)
    pain_inputs: List[PainInput] = field(default_factory=list)
    proprioceptive_inputs: List[ProprioceptiveInput] = field(default_factory=list)
    overall_body_temperature: float = 0.5  # 0.0-1.0 (normalized)
    muscle_tension_global: float = 0.3     # 0.0-1.0 (relaxed to tense)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_touch_inputs": len(self.touch_inputs),
            "num_pain_inputs": len(self.pain_inputs),
            "num_proprioceptive_inputs": len(self.proprioceptive_inputs),
            "body_temperature": round(self.overall_body_temperature, 4),
            "muscle_tension": round(self.muscle_tension_global, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class TouchClassification:
    """Classification result for a touch stimulus."""
    touch_type: TouchType
    body_region: BodyRegion
    perceived_intensity: float
    pleasantness: float       # -1.0 (unpleasant) to 1.0 (pleasant)
    novelty: float            # 0.0-1.0
    threat_level: float       # 0.0-1.0
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "touch_type": self.touch_type.value,
            "body_region": self.body_region.value,
            "perceived_intensity": round(self.perceived_intensity, 4),
            "pleasantness": round(self.pleasantness, 4),
            "novelty": round(self.novelty, 4),
            "threat_level": round(self.threat_level, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PainAssessment:
    """Assessment of a pain signal."""
    pain_type: PainType
    body_region: BodyRegion
    subjective_intensity: float  # 0.0-1.0
    emotional_distress: float    # 0.0-1.0
    action_urgency: float        # 0.0-1.0 urgency of protective action
    coping_capacity: float       # 0.0-1.0 ability to cope
    requires_attention: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pain_type": self.pain_type.value,
            "body_region": self.body_region.value,
            "subjective_intensity": round(self.subjective_intensity, 4),
            "emotional_distress": round(self.emotional_distress, 4),
            "action_urgency": round(self.action_urgency, 4),
            "coping_capacity": round(self.coping_capacity, 4),
            "requires_attention": self.requires_attention,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BodySchema:
    """Current body schema representation."""
    schema_state: BodySchemaState
    body_boundary_clarity: float   # 0.0-1.0
    postural_stability: float      # 0.0-1.0
    body_ownership: float          # 0.0-1.0 sense of body belonging to self
    region_activation: Dict[str, float]  # region -> activation level
    balance_state: float           # -1.0 (unstable) to 1.0 (stable)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_state": self.schema_state.value,
            "body_boundary_clarity": round(self.body_boundary_clarity, 4),
            "postural_stability": round(self.postural_stability, 4),
            "body_ownership": round(self.body_ownership, 4),
            "region_activation": {k: round(v, 4) for k, v in self.region_activation.items()},
            "balance_state": round(self.balance_state, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SomatosensoryOutput:
    """Complete output of somatosensory consciousness processing."""
    touch_classifications: List[TouchClassification]
    pain_assessments: List[PainAssessment]
    body_schema: BodySchema
    overall_comfort: float        # -1.0 (distress) to 1.0 (comfort)
    body_awareness_level: float   # 0.0-1.0
    requires_protective_action: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_touch": len(self.touch_classifications),
            "touch_classifications": [t.to_dict() for t in self.touch_classifications],
            "num_pain": len(self.pain_assessments),
            "pain_assessments": [p.to_dict() for p in self.pain_assessments],
            "body_schema": self.body_schema.to_dict(),
            "overall_comfort": round(self.overall_comfort, 4),
            "body_awareness_level": round(self.body_awareness_level, 4),
            "requires_protective_action": self.requires_protective_action,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class SomatosensoryConsciousnessInterface:
    """
    Main interface for Form 03: Somatosensory Consciousness.

    Processes tactile stimuli, pain signals, proprioceptive data,
    and temperature to produce conscious body awareness, touch
    perception, and body schema maintenance.
    """

    FORM_ID = "03-somatosensory"
    FORM_NAME = "Somatosensory Consciousness"

    def __init__(self):
        """Initialize the somatosensory consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[SomatosensoryOutput] = None
        self._body_schema_state = BodySchemaState.NORMAL
        self._region_history: Dict[str, List[float]] = {}
        self._pain_history: List[PainAssessment] = []
        self._comfort_baseline = 0.5
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the somatosensory processing pipeline."""
        self._initialized = True
        self._body_schema_state = BodySchemaState.NORMAL
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_somatosensory_input(
        self, somatosensory_input: SomatosensoryInput
    ) -> SomatosensoryOutput:
        """
        Process somatosensory input through the consciousness pipeline.

        Pipeline stages:
        1. Touch classification
        2. Pain assessment
        3. Body schema update
        4. Integration and comfort evaluation
        """
        self._processing_count += 1

        # Stage 1: Classify touch inputs
        touch_results = await self._classify_touches(somatosensory_input.touch_inputs)

        # Stage 2: Assess pain signals
        pain_results = await self._assess_pain(somatosensory_input.pain_inputs)

        # Stage 3: Update body schema
        body_schema = await self._update_body_schema(
            somatosensory_input, touch_results, pain_results
        )

        # Stage 4: Integration
        comfort = self._compute_comfort(touch_results, pain_results, somatosensory_input)
        awareness = self._compute_body_awareness(
            somatosensory_input, touch_results, pain_results
        )
        protective = any(p.requires_attention and p.action_urgency > 0.7 for p in pain_results)

        output = SomatosensoryOutput(
            touch_classifications=touch_results,
            pain_assessments=pain_results,
            body_schema=body_schema,
            overall_comfort=comfort,
            body_awareness_level=awareness,
            requires_protective_action=protective,
        )

        self._current_output = output
        self._update_history(pain_results)

        return output

    async def _classify_touches(self, touches: List[TouchInput]) -> List[TouchClassification]:
        """Classify all touch inputs."""
        results = []
        for touch in touches:
            pleasantness = self._compute_pleasantness(touch)
            novelty = self._compute_touch_novelty(touch)
            threat = self._compute_touch_threat(touch)

            results.append(TouchClassification(
                touch_type=touch.touch_type,
                body_region=touch.body_region,
                perceived_intensity=touch.intensity,
                pleasantness=pleasantness,
                novelty=novelty,
                threat_level=threat,
                confidence=0.8,
            ))
        return results

    async def _assess_pain(self, pains: List[PainInput]) -> List[PainAssessment]:
        """Assess all pain signals."""
        results = []
        for pain in pains:
            distress = pain.intensity * (1.0 + pain.sharpness) / 2.0
            urgency = self._compute_pain_urgency(pain)
            coping = max(0.1, 1.0 - pain.intensity * 0.8)

            results.append(PainAssessment(
                pain_type=pain.pain_type,
                body_region=pain.body_region,
                subjective_intensity=pain.intensity,
                emotional_distress=min(1.0, distress),
                action_urgency=urgency,
                coping_capacity=coping,
                requires_attention=pain.intensity > 0.5,
            ))
        return results

    async def _update_body_schema(
        self,
        somatosensory_input: SomatosensoryInput,
        touches: List[TouchClassification],
        pains: List[PainAssessment],
    ) -> BodySchema:
        """Update the body schema based on current inputs."""
        # Build region activation map
        region_activation: Dict[str, float] = {}
        for touch in touches:
            region = touch.body_region.value
            current = region_activation.get(region, 0.0)
            region_activation[region] = min(1.0, current + touch.perceived_intensity)

        for pain in pains:
            region = pain.body_region.value
            current = region_activation.get(region, 0.0)
            region_activation[region] = min(1.0, current + pain.subjective_intensity)

        for prop in somatosensory_input.proprioceptive_inputs:
            region = prop.body_region.value
            current = region_activation.get(region, 0.0)
            region_activation[region] = min(1.0, current + prop.value * 0.5)

        # Compute body schema properties
        has_pain = len(pains) > 0 and any(p.subjective_intensity > 0.5 for p in pains)
        total_activation = sum(region_activation.values()) if region_activation else 0.0
        num_active = len(region_activation)

        if has_pain and total_activation > 2.0:
            schema_state = BodySchemaState.HEIGHTENED
        elif total_activation < 0.2:
            schema_state = BodySchemaState.DIMINISHED
        else:
            schema_state = BodySchemaState.NORMAL

        self._body_schema_state = schema_state

        # Compute postural stability from proprioception
        balance_inputs = [
            p.value for p in somatosensory_input.proprioceptive_inputs
            if p.channel == ProprioceptiveChannel.BALANCE
        ]
        balance = sum(balance_inputs) / max(1, len(balance_inputs)) if balance_inputs else 0.5

        boundary_clarity = min(1.0, 0.5 + total_activation * 0.1)
        stability = 0.5 + balance * 0.5 - somatosensory_input.muscle_tension_global * 0.2

        return BodySchema(
            schema_state=schema_state,
            body_boundary_clarity=max(0.0, min(1.0, boundary_clarity)),
            postural_stability=max(0.0, min(1.0, stability)),
            body_ownership=0.9,
            region_activation=region_activation,
            balance_state=balance * 2.0 - 1.0,  # Scale to -1 to 1
        )

    def _compute_pleasantness(self, touch: TouchInput) -> float:
        """Compute pleasantness of a touch stimulus."""
        base = 0.0
        if touch.touch_type == TouchType.LIGHT_TOUCH:
            base = 0.3
        elif touch.touch_type == TouchType.TEXTURE:
            base = 0.2
        elif touch.touch_type == TouchType.VIBRATION:
            base = 0.1
        elif touch.touch_type == TouchType.PRESSURE:
            base = -0.1 if touch.intensity > 0.7 else 0.1
        elif touch.touch_type == TouchType.TEMPERATURE_WARM:
            base = 0.2 if touch.temperature < 0.8 else -0.3
        elif touch.touch_type == TouchType.TEMPERATURE_COLD:
            base = -0.2
        elif touch.touch_type == TouchType.ITCH:
            base = -0.4
        elif touch.touch_type == TouchType.TICKLE:
            base = 0.1

        # Intensity modulation
        if touch.intensity > 0.8:
            base -= 0.3
        return max(-1.0, min(1.0, base))

    def _compute_touch_novelty(self, touch: TouchInput) -> float:
        """Compute novelty of a touch stimulus."""
        region_key = touch.body_region.value
        history = self._region_history.get(region_key, [])
        if len(history) < 2:
            return 0.7
        return max(0.0, 0.5 - len(history) * 0.05)

    def _compute_touch_threat(self, touch: TouchInput) -> float:
        """Compute threat level of a touch stimulus."""
        threat = 0.0
        if touch.intensity > 0.8:
            threat += 0.3
        if touch.touch_type in [TouchType.TEMPERATURE_WARM, TouchType.TEMPERATURE_COLD]:
            if touch.temperature > 0.9 or touch.temperature < 0.1:
                threat += 0.5
        if not touch.is_self_generated and touch.intensity > 0.6:
            threat += 0.2
        return min(1.0, threat)

    def _compute_pain_urgency(self, pain: PainInput) -> float:
        """Compute urgency of pain response."""
        urgency = pain.intensity * 0.5
        if pain.pain_type == PainType.SHARP:
            urgency += 0.3
        if pain.tissue_damage > 0.5:
            urgency += 0.2
        if not pain.is_chronic:
            urgency += 0.1  # Acute pain is more urgent
        return min(1.0, urgency)

    def _compute_comfort(
        self,
        touches: List[TouchClassification],
        pains: List[PainAssessment],
        somatosensory_input: SomatosensoryInput,
    ) -> float:
        """Compute overall comfort level."""
        comfort = self._comfort_baseline

        # Pleasant touches increase comfort
        for touch in touches:
            comfort += touch.pleasantness * 0.2

        # Pain decreases comfort
        for pain in pains:
            comfort -= pain.subjective_intensity * 0.4

        # Temperature extremes decrease comfort
        temp = somatosensory_input.overall_body_temperature
        if temp < 0.3 or temp > 0.7:
            comfort -= abs(temp - 0.5) * 0.3

        # High muscle tension decreases comfort
        comfort -= somatosensory_input.muscle_tension_global * 0.2

        return max(-1.0, min(1.0, comfort))

    def _compute_body_awareness(
        self,
        somatosensory_input: SomatosensoryInput,
        touches: List[TouchClassification],
        pains: List[PainAssessment],
    ) -> float:
        """Compute level of body awareness."""
        awareness = 0.3  # Baseline
        awareness += len(touches) * 0.1
        awareness += len(pains) * 0.15
        awareness += len(somatosensory_input.proprioceptive_inputs) * 0.05
        return min(1.0, awareness)

    def _update_history(self, pains: List[PainAssessment]) -> None:
        """Update processing history."""
        self._pain_history.extend(pains)
        if len(self._pain_history) > self._max_history:
            self._pain_history = self._pain_history[-self._max_history:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "body_schema_state": self._body_schema_state.value,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "pain_history_length": len(self._pain_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "body_schema_state": self._body_schema_state.value,
            "overall_comfort": (
                self._current_output.overall_comfort if self._current_output else 0.0
            ),
            "body_awareness": (
                self._current_output.body_awareness_level if self._current_output else 0.0
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_somatosensory_interface() -> SomatosensoryConsciousnessInterface:
    """Create and return a somatosensory consciousness interface."""
    return SomatosensoryConsciousnessInterface()


def create_simple_touch_input(
    touch_type: TouchType = TouchType.LIGHT_TOUCH,
    body_region: BodyRegion = BodyRegion.HAND_RIGHT,
    intensity: float = 0.5,
    temperature: float = 0.5,
) -> SomatosensoryInput:
    """Create a simple somatosensory input for testing."""
    return SomatosensoryInput(
        touch_inputs=[
            TouchInput(
                touch_type=touch_type,
                body_region=body_region,
                intensity=intensity,
                area_size=0.1,
                duration_ms=100.0,
                temperature=temperature,
            )
        ],
        overall_body_temperature=temperature,
    )


__all__ = [
    # Enums
    "TouchType",
    "PainType",
    "BodyRegion",
    "ProprioceptiveChannel",
    "BodySchemaState",
    # Input dataclasses
    "TouchInput",
    "PainInput",
    "ProprioceptiveInput",
    "SomatosensoryInput",
    # Output dataclasses
    "TouchClassification",
    "PainAssessment",
    "BodySchema",
    "SomatosensoryOutput",
    # Main interface
    "SomatosensoryConsciousnessInterface",
    # Convenience functions
    "create_somatosensory_interface",
    "create_simple_touch_input",
]
