#!/usr/bin/env python3
"""
Arousal Consciousness Interface

Form 08: The critical gating mechanism for the entire consciousness system.
Arousal/Vigilance Consciousness controls resource allocation, information
gating, and processing capacity across all other consciousness forms.

This module is CRITICAL - it must always be loaded and operational as it
gates all other forms based on current arousal state.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ArousalState(Enum):
    """
    Discrete arousal state categories based on arousal level.

    These states correspond to different processing modes and
    resource allocation strategies.
    """
    SLEEP = "sleep"  # 0.0 - 0.1: Minimal consciousness, recovery mode
    DROWSY = "drowsy"  # 0.1 - 0.3: Reduced awareness, limited processing
    RELAXED = "relaxed"  # 0.3 - 0.5: Calm wakefulness, baseline processing
    ALERT = "alert"  # 0.5 - 0.7: Active engagement, normal processing
    FOCUSED = "focused"  # 0.7 - 0.9: High engagement, enhanced processing
    HYPERAROUSED = "hyperaroused"  # 0.9 - 1.0: Maximum activation, emergency mode


class ArousalSource(Enum):
    """Sources contributing to arousal level."""
    ENVIRONMENTAL = "environmental"  # External sensory stimulation
    EMOTIONAL = "emotional"  # Emotional activation
    CIRCADIAN = "circadian"  # Biological rhythm
    TASK_DEMAND = "task_demand"  # Cognitive task requirements
    RESOURCE_STATE = "resource_state"  # System resource availability
    THREAT = "threat"  # Detected threats
    NOVELTY = "novelty"  # Novel stimuli
    SOCIAL = "social"  # Social interaction signals
    INTERNAL = "internal"  # Internal state changes


class GateCategory(Enum):
    """Categories of consciousness gates."""
    SENSORY = "sensory"  # Sensory input gates
    COGNITIVE = "cognitive"  # Cognitive processing gates
    EMOTIONAL = "emotional"  # Emotional processing gates
    MEMORY = "memory"  # Memory access gates
    EXECUTIVE = "executive"  # Executive control gates
    META = "meta"  # Meta-cognitive gates


class SensoryModality(Enum):
    """Sensory modalities for gating."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    SOMATOSENSORY = "somatosensory"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"
    INTEROCEPTIVE = "interoceptive"


class StateTransitionType(Enum):
    """Types of arousal state transitions."""
    GRADUAL = "gradual"  # Slow, natural transition
    RAPID = "rapid"  # Fast transition (e.g., startle)
    FORCED = "forced"  # Externally imposed transition
    RECOVERY = "recovery"  # Return to baseline


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SensoryArousalInput:
    """Sensory stimulation contribution to arousal."""
    source_modality: SensoryModality
    stimulus_type: str  # "novel", "threat", "familiar", "neutral"
    intensity: float  # 0.0-1.0
    salience: float  # 0.0-1.0
    change_rate: float  # Rate of stimulus change
    processing_confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_modality": self.source_modality.value,
            "stimulus_type": self.stimulus_type,
            "intensity": self.intensity,
            "salience": self.salience,
            "change_rate": self.change_rate,
            "processing_confidence": self.processing_confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThreatInput:
    """Threat detection contribution to arousal."""
    threat_level: float  # 0.0-1.0
    threat_type: str  # "physical", "social", "cognitive", "unknown"
    proximity: float  # 0.0-1.0 (closer = higher)
    certainty: float  # Confidence in threat assessment
    response_urgency: float  # How quickly response is needed
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NoveltyInput:
    """Novelty detection contribution to arousal."""
    novelty_level: float  # 0.0-1.0
    novelty_type: str  # "stimulus", "pattern", "context", "semantic"
    learning_opportunity: float  # Potential learning value
    exploration_value: float  # Benefit of increased arousal
    memory_mismatch: float  # Degree of mismatch with stored patterns
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CircadianInput:
    """Circadian rhythm contribution to arousal."""
    circadian_phase: float  # 0.0-24.0 hours
    melatonin_level: float  # 0.0-1.0 (normalized)
    cortisol_level: float  # 0.0-1.0 (normalized)
    sleep_pressure: float  # 0.0-1.0 (homeostatic sleep drive)
    light_exposure: float  # Recent light exposure
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmotionalInput:
    """Emotional state contribution to arousal."""
    valence: float  # -1.0 to 1.0 (negative to positive)
    arousal_component: float  # 0.0-1.0 (emotional activation)
    fear: float = 0.0
    excitement: float = 0.0
    anxiety: float = 0.0
    curiosity: float = 0.0
    stress: float = 0.0
    calm: float = 0.0
    stability: float = 0.5  # Emotional stability
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskDemandInput:
    """Task demand contribution to arousal."""
    complexity: float  # 0.0-1.0
    importance: float  # 0.0-1.0
    time_pressure: float  # 0.0-1.0
    performance_requirements: float  # Precision demands
    sustained_attention_needs: float  # Duration of attention required
    decision_complexity: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResourceInput:
    """System resource state contribution to arousal."""
    computational_capacity: float  # 0.0-1.0 available
    energy_level: float  # 0.0-1.0 available
    memory_load: float  # 0.0-1.0 current load
    attention_capacity: float  # 0.0-1.0 available
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArousalInputBundle:
    """Complete bundle of all arousal inputs."""
    sensory_inputs: List[SensoryArousalInput] = field(default_factory=list)
    threat_input: Optional[ThreatInput] = None
    novelty_input: Optional[NoveltyInput] = None
    circadian_input: Optional[CircadianInput] = None
    emotional_input: Optional[EmotionalInput] = None
    task_demand_input: Optional[TaskDemandInput] = None
    resource_input: Optional[ResourceInput] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class ArousalLevelOutput:
    """Current arousal level assessment."""
    arousal_level: float  # 0.0-1.0
    arousal_state: ArousalState
    arousal_trend: float  # -1.0 to 1.0 (decreasing to increasing)
    arousal_stability: float  # 0.0-1.0 (consistency)
    primary_source: ArousalSource
    confidence: float  # 0.0-1.0
    components: Dict[ArousalSource, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arousal_level": round(self.arousal_level, 4),
            "arousal_state": self.arousal_state.value,
            "arousal_trend": round(self.arousal_trend, 4),
            "arousal_stability": round(self.arousal_stability, 4),
            "primary_source": self.primary_source.value,
            "confidence": round(self.confidence, 4),
            "components": {k.value: round(v, 4) for k, v in self.components.items()},
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GatingSignal:
    """Gating signal for a specific gate."""
    gate_id: str
    category: GateCategory
    openness: float  # 0.0-1.0 (closed to open)
    modulation_factor: float  # Adjustment factor
    priority_boost: float  # Priority adjustment
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConsciousnessGatingOutput:
    """Complete consciousness gating configuration."""
    sensory_gates: Dict[str, float]  # modality -> openness
    cognitive_gates: Dict[str, float]  # gate_name -> openness
    global_threshold: float  # Minimum activation for consciousness access
    gate_adaptation_rate: float  # Speed of gate adjustment
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensory_gates": {k: round(v, 4) for k, v in self.sensory_gates.items()},
            "cognitive_gates": {k: round(v, 4) for k, v in self.cognitive_gates.items()},
            "global_threshold": round(self.global_threshold, 4),
            "gate_adaptation_rate": round(self.gate_adaptation_rate, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ResourceAllocationOutput:
    """Resource allocation across consciousness forms."""
    total_available: float  # Total available capacity
    allocations: Dict[str, float]  # form_id -> allocation
    reserve_capacity: float  # Held for urgent needs
    allocation_confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_available": round(self.total_available, 4),
            "allocations": {k: round(v, 4) for k, v in self.allocations.items()},
            "reserve_capacity": round(self.reserve_capacity, 4),
            "allocation_confidence": round(self.allocation_confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StateTransition:
    """Record of an arousal state transition."""
    from_state: ArousalState
    to_state: ArousalState
    transition_type: StateTransitionType
    trigger: ArousalSource
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArousalSystemStatus:
    """Complete arousal system status."""
    current_level: ArousalLevelOutput
    current_gating: ConsciousnessGatingOutput
    current_allocation: ResourceAllocationOutput
    recent_transitions: List[StateTransition]
    system_health: float  # 0.0-1.0
    responsiveness: float  # Speed of adjustments
    stability: float  # Consistency of regulation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# AROUSAL COMPUTATION ENGINE
# ============================================================================

class ArousalComputationEngine:
    """
    Core computation engine for arousal level calculation.

    Implements multi-input integration with non-linear dynamics
    for realistic arousal behavior.
    """

    # Weight factors for different input sources
    DEFAULT_WEIGHTS = {
        ArousalSource.ENVIRONMENTAL: 0.25,
        ArousalSource.EMOTIONAL: 0.20,
        ArousalSource.CIRCADIAN: 0.15,
        ArousalSource.TASK_DEMAND: 0.15,
        ArousalSource.RESOURCE_STATE: 0.10,
        ArousalSource.THREAT: 0.10,  # Can override others
        ArousalSource.NOVELTY: 0.05,
    }

    # State thresholds
    STATE_THRESHOLDS = {
        ArousalState.SLEEP: (0.0, 0.1),
        ArousalState.DROWSY: (0.1, 0.3),
        ArousalState.RELAXED: (0.3, 0.5),
        ArousalState.ALERT: (0.5, 0.7),
        ArousalState.FOCUSED: (0.7, 0.9),
        ArousalState.HYPERAROUSED: (0.9, 1.0),
    }

    def __init__(self, weights: Optional[Dict[ArousalSource, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._arousal_history: List[float] = []
        self._max_history = 100
        self._last_arousal = 0.5
        self._inertia_factor = 0.3  # Resistance to change

    def compute_arousal(self, inputs: ArousalInputBundle) -> ArousalLevelOutput:
        """
        Compute arousal level from all input sources.

        Uses weighted integration with non-linear dynamics,
        threat override, and temporal smoothing.
        """
        components = {}

        # Compute environmental arousal from sensory inputs
        env_arousal = self._compute_environmental_arousal(inputs.sensory_inputs)
        components[ArousalSource.ENVIRONMENTAL] = env_arousal

        # Compute emotional arousal
        emotional_arousal = self._compute_emotional_arousal(inputs.emotional_input)
        components[ArousalSource.EMOTIONAL] = emotional_arousal

        # Compute circadian arousal
        circadian_arousal = self._compute_circadian_arousal(inputs.circadian_input)
        components[ArousalSource.CIRCADIAN] = circadian_arousal

        # Compute task demand arousal
        task_arousal = self._compute_task_arousal(inputs.task_demand_input)
        components[ArousalSource.TASK_DEMAND] = task_arousal

        # Compute resource-based arousal
        resource_arousal = self._compute_resource_arousal(inputs.resource_input)
        components[ArousalSource.RESOURCE_STATE] = resource_arousal

        # Compute threat arousal (can override)
        threat_arousal = self._compute_threat_arousal(inputs.threat_input)
        components[ArousalSource.THREAT] = threat_arousal

        # Compute novelty arousal
        novelty_arousal = self._compute_novelty_arousal(inputs.novelty_input)
        components[ArousalSource.NOVELTY] = novelty_arousal

        # Weighted integration
        base_arousal = sum(
            self.weights.get(source, 0.0) * value
            for source, value in components.items()
        )

        # Apply threat override if significant
        if threat_arousal > 0.7:
            base_arousal = max(base_arousal, threat_arousal * 0.9)

        # Apply temporal smoothing (inertia)
        smoothed_arousal = (
            self._inertia_factor * self._last_arousal +
            (1 - self._inertia_factor) * base_arousal
        )

        # Clamp to valid range
        final_arousal = max(0.0, min(1.0, smoothed_arousal))

        # Update history
        self._update_history(final_arousal)

        # Determine state and trend
        state = self._level_to_state(final_arousal)
        trend = self._compute_trend()
        stability = self._compute_stability()

        # Determine primary source
        primary_source = max(components, key=components.get)

        return ArousalLevelOutput(
            arousal_level=final_arousal,
            arousal_state=state,
            arousal_trend=trend,
            arousal_stability=stability,
            primary_source=primary_source,
            confidence=self._compute_confidence(components),
            components=components,
        )

    def _compute_environmental_arousal(
        self, inputs: List[SensoryArousalInput]
    ) -> float:
        """Compute arousal from sensory inputs."""
        if not inputs:
            return 0.5

        # Weight by salience and intensity
        weighted_sum = sum(
            inp.salience * inp.intensity * inp.processing_confidence
            for inp in inputs
        )
        return min(1.0, weighted_sum / max(1, len(inputs)))

    def _compute_emotional_arousal(self, inp: Optional[EmotionalInput]) -> float:
        """Compute arousal from emotional state."""
        if not inp:
            return 0.5

        # Emotional arousal component plus specific emotions
        base = inp.arousal_component
        emotion_boost = (
            0.3 * inp.fear +
            0.2 * inp.excitement +
            0.2 * inp.anxiety +
            0.1 * inp.curiosity +
            0.1 * inp.stress -
            0.1 * inp.calm
        )
        return max(0.0, min(1.0, base + emotion_boost))

    def _compute_circadian_arousal(self, inp: Optional[CircadianInput]) -> float:
        """Compute arousal from circadian rhythm."""
        if not inp:
            return 0.5

        # Inverse relationship with melatonin and sleep pressure
        circadian_factor = (
            (1.0 - inp.melatonin_level) * 0.4 +
            inp.cortisol_level * 0.3 +
            (1.0 - inp.sleep_pressure) * 0.3
        )
        return max(0.0, min(1.0, circadian_factor))

    def _compute_task_arousal(self, inp: Optional[TaskDemandInput]) -> float:
        """Compute arousal from task demands."""
        if not inp:
            return 0.5

        task_factor = (
            inp.complexity * 0.3 +
            inp.importance * 0.25 +
            inp.time_pressure * 0.25 +
            inp.sustained_attention_needs * 0.2
        )
        return max(0.0, min(1.0, task_factor))

    def _compute_resource_arousal(self, inp: Optional[ResourceInput]) -> float:
        """Compute arousal from resource state."""
        if not inp:
            return 0.5

        # Higher resources allow for higher arousal
        resource_factor = (
            inp.computational_capacity * 0.3 +
            inp.energy_level * 0.4 +
            inp.attention_capacity * 0.3
        )
        return max(0.0, min(1.0, resource_factor))

    def _compute_threat_arousal(self, inp: Optional[ThreatInput]) -> float:
        """Compute arousal from threat detection."""
        if not inp:
            return 0.0

        threat_factor = (
            inp.threat_level * 0.4 +
            inp.proximity * 0.3 +
            inp.response_urgency * 0.2 +
            inp.certainty * 0.1
        )
        return max(0.0, min(1.0, threat_factor))

    def _compute_novelty_arousal(self, inp: Optional[NoveltyInput]) -> float:
        """Compute arousal from novelty detection."""
        if not inp:
            return 0.0

        novelty_factor = (
            inp.novelty_level * 0.5 +
            inp.exploration_value * 0.3 +
            inp.memory_mismatch * 0.2
        )
        return max(0.0, min(1.0, novelty_factor))

    def _level_to_state(self, level: float) -> ArousalState:
        """Convert arousal level to discrete state."""
        for state, (low, high) in self.STATE_THRESHOLDS.items():
            if low <= level < high:
                return state
        return ArousalState.HYPERAROUSED if level >= 0.9 else ArousalState.SLEEP

    def _update_history(self, level: float) -> None:
        """Update arousal history for trend/stability computation."""
        self._arousal_history.append(level)
        if len(self._arousal_history) > self._max_history:
            self._arousal_history.pop(0)
        self._last_arousal = level

    def _compute_trend(self) -> float:
        """Compute arousal trend from history."""
        if len(self._arousal_history) < 3:
            return 0.0

        recent = self._arousal_history[-5:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        trend = (recent[-1] - recent[0]) / len(recent)
        return max(-1.0, min(1.0, trend * 10))

    def _compute_stability(self) -> float:
        """Compute arousal stability from history."""
        if len(self._arousal_history) < 3:
            return 1.0

        recent = self._arousal_history[-10:]
        if len(recent) < 2:
            return 1.0

        # Inverse of variance
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return max(0.0, 1.0 - math.sqrt(variance) * 5)

    def _compute_confidence(self, components: Dict[ArousalSource, float]) -> float:
        """Compute confidence in arousal assessment."""
        # Higher confidence when inputs are consistent
        values = list(components.values())
        if not values:
            return 0.5

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return max(0.5, 1.0 - math.sqrt(variance))


# ============================================================================
# CONSCIOUSNESS GATING ENGINE
# ============================================================================

class ConsciousnessGatingEngine:
    """
    Engine for computing consciousness gates based on arousal state.

    Gates control information flow to consciousness, with different
    gates for sensory, cognitive, and meta-cognitive processes.
    """

    def __init__(self):
        self._gate_history: Dict[str, List[float]] = {}

    def compute_gates(
        self,
        arousal_output: ArousalLevelOutput,
        threat_level: float = 0.0,
        novelty_level: float = 0.0
    ) -> ConsciousnessGatingOutput:
        """Compute all consciousness gates based on arousal."""
        level = arousal_output.arousal_level
        state = arousal_output.arousal_state

        # Base gate openness follows sigmoid of arousal
        base_openness = self._sigmoid(level, steepness=3.0, midpoint=0.5)

        # Compute sensory gates
        sensory_gates = self._compute_sensory_gates(
            base_openness, threat_level, novelty_level
        )

        # Compute cognitive gates
        cognitive_gates = self._compute_cognitive_gates(base_openness, state)

        # Global threshold adjusts with arousal
        global_threshold = self._compute_global_threshold(level, state)

        # Gate adaptation rate - faster in alert states
        adaptation_rate = self._compute_adaptation_rate(state)

        return ConsciousnessGatingOutput(
            sensory_gates=sensory_gates,
            cognitive_gates=cognitive_gates,
            global_threshold=global_threshold,
            gate_adaptation_rate=adaptation_rate,
        )

    def _sigmoid(
        self, x: float, steepness: float = 1.0, midpoint: float = 0.5
    ) -> float:
        """Sigmoid transformation."""
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

    def _compute_sensory_gates(
        self,
        base_openness: float,
        threat_level: float,
        novelty_level: float
    ) -> Dict[str, float]:
        """Compute gates for each sensory modality."""
        gates = {}

        # Modality-specific modifiers
        modality_modifiers = {
            SensoryModality.VISUAL: 1.1,  # Slightly prioritized
            SensoryModality.AUDITORY: 1.05,
            SensoryModality.SOMATOSENSORY: 1.0,
            SensoryModality.OLFACTORY: 0.9,
            SensoryModality.GUSTATORY: 0.85,
            SensoryModality.INTEROCEPTIVE: 0.95,
        }

        for modality, modifier in modality_modifiers.items():
            # Base gate with modality modifier
            gate = base_openness * modifier

            # Threat boosts relevant modalities
            if threat_level > 0.3:
                if modality in [SensoryModality.VISUAL, SensoryModality.AUDITORY]:
                    gate += threat_level * 0.3

            # Novelty also boosts gates
            gate += novelty_level * 0.1

            gates[modality.value] = max(0.0, min(1.0, gate))

        return gates

    def _compute_cognitive_gates(
        self, base_openness: float, state: ArousalState
    ) -> Dict[str, float]:
        """Compute gates for cognitive processes."""
        gates = {}

        # State-specific adjustments
        state_modifiers = {
            ArousalState.SLEEP: {"memory": 0.3, "executive": 0.1, "meta": 0.1},
            ArousalState.DROWSY: {"memory": 0.5, "executive": 0.4, "meta": 0.3},
            ArousalState.RELAXED: {"memory": 0.8, "executive": 0.7, "meta": 0.7},
            ArousalState.ALERT: {"memory": 1.0, "executive": 1.0, "meta": 1.0},
            ArousalState.FOCUSED: {"memory": 0.9, "executive": 1.2, "meta": 1.1},
            ArousalState.HYPERAROUSED: {"memory": 0.7, "executive": 1.3, "meta": 0.8},
        }

        modifiers = state_modifiers.get(state, {"memory": 1.0, "executive": 1.0, "meta": 1.0})

        gates["memory_access"] = max(0.0, min(1.0, base_openness * modifiers["memory"]))
        gates["executive_control"] = max(0.0, min(1.0, base_openness * modifiers["executive"]))
        gates["meta_cognitive"] = max(0.0, min(1.0, base_openness * modifiers["meta"]))
        gates["emotional_processing"] = max(0.0, min(1.0, base_openness * 1.1))

        return gates

    def _compute_global_threshold(self, level: float, state: ArousalState) -> float:
        """Compute global consciousness threshold."""
        # Lower threshold in alert states (more accessible consciousness)
        base_threshold = 0.5 - (level - 0.5) * 0.3
        return max(0.2, min(0.8, base_threshold))

    def _compute_adaptation_rate(self, state: ArousalState) -> float:
        """Compute how fast gates can change."""
        rates = {
            ArousalState.SLEEP: 0.1,
            ArousalState.DROWSY: 0.3,
            ArousalState.RELAXED: 0.5,
            ArousalState.ALERT: 0.7,
            ArousalState.FOCUSED: 0.8,
            ArousalState.HYPERAROUSED: 1.0,
        }
        return rates.get(state, 0.5)


# ============================================================================
# RESOURCE ALLOCATION ENGINE
# ============================================================================

class ResourceAllocationEngine:
    """
    Engine for allocating processing resources across consciousness forms.

    Uses arousal-gated allocation with priority-based distribution.
    """

    # Form priorities by category
    FORM_PRIORITIES = {
        # Critical forms - always get resources
        "08-arousal": 1.0,
        "13-integrated-information": 0.95,
        "14-global-workspace": 0.95,
        # High priority sensory
        "01-visual": 0.8,
        "02-auditory": 0.75,
        "07-emotional": 0.8,
        # Medium priority
        "03-somatosensory": 0.6,
        "04-olfactory": 0.5,
        "05-gustatory": 0.5,
        "06-interoceptive": 0.6,
        # Cognitive
        "09-perceptual": 0.7,
        "10-self-recognition": 0.6,
        "11-meta-consciousness": 0.65,
        "12-narrative-consciousness": 0.55,
        # Extended forms - lower priority by default
        "28-philosophy": 0.4,
        "29-folk-wisdom": 0.35,
        "30-animal-cognition": 0.35,
    }

    def __init__(self, total_capacity: float = 1.0):
        self.total_capacity = total_capacity
        self.reserve_ratio = 0.1  # Keep 10% in reserve

    def allocate_resources(
        self,
        arousal_output: ArousalLevelOutput,
        form_demands: Dict[str, float],
        active_forms: Set[str]
    ) -> ResourceAllocationOutput:
        """Allocate resources based on arousal and demands."""
        level = arousal_output.arousal_level
        state = arousal_output.arousal_state

        # Available capacity scales with arousal
        available = self._compute_available_capacity(level, state)
        reserve = available * self.reserve_ratio
        distributable = available - reserve

        # Priority-weighted allocation
        allocations = {}
        remaining = distributable

        # Sort forms by priority
        sorted_forms = sorted(
            active_forms,
            key=lambda f: self.FORM_PRIORITIES.get(f, 0.3),
            reverse=True
        )

        for form_id in sorted_forms:
            if remaining <= 0:
                allocations[form_id] = 0.0
                continue

            priority = self.FORM_PRIORITIES.get(form_id, 0.3)
            demand = form_demands.get(form_id, 0.1)

            # Adjust priority by arousal state
            adjusted_priority = self._adjust_priority(priority, state, form_id)

            # Calculate allocation
            requested = demand * adjusted_priority
            allocation = min(requested, remaining * adjusted_priority)

            allocations[form_id] = allocation
            remaining -= allocation

        # Add remaining to reserve
        reserve += remaining

        return ResourceAllocationOutput(
            total_available=available,
            allocations=allocations,
            reserve_capacity=reserve,
            allocation_confidence=arousal_output.confidence,
        )

    def _compute_available_capacity(
        self, level: float, state: ArousalState
    ) -> float:
        """Compute available capacity based on arousal."""
        # Capacity increases with arousal up to a point
        # then decreases in hyperaroused states
        if level < 0.5:
            return 0.5 + level
        elif level < 0.9:
            return 1.0
        else:
            # Hyperarousal reduces efficiency
            return 1.0 - (level - 0.9) * 2

    def _adjust_priority(
        self, base_priority: float, state: ArousalState, form_id: str
    ) -> float:
        """Adjust form priority based on arousal state."""
        # In hyperaroused states, prioritize critical forms
        if state == ArousalState.HYPERAROUSED:
            if form_id in ["08-arousal", "13-integrated-information", "14-global-workspace"]:
                return min(1.0, base_priority * 1.2)
            return base_priority * 0.7

        # In drowsy/sleep, reduce non-critical
        if state in [ArousalState.SLEEP, ArousalState.DROWSY]:
            if form_id in ["08-arousal"]:
                return base_priority
            return base_priority * 0.5

        return base_priority


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class ArousalConsciousnessInterface:
    """
    Main interface for Form 08: Arousal/Vigilance Consciousness.

    This is the CRITICAL gating mechanism for the entire consciousness system.
    It must always be operational as it controls resource allocation and
    information gating for all other forms.
    """

    FORM_ID = "08-arousal"
    FORM_NAME = "Arousal/Vigilance Consciousness"
    IS_CRITICAL = True

    def __init__(self):
        """Initialize the arousal consciousness interface."""
        self.computation_engine = ArousalComputationEngine()
        self.gating_engine = ConsciousnessGatingEngine()
        self.allocation_engine = ResourceAllocationEngine()

        # Current state
        self._current_arousal: Optional[ArousalLevelOutput] = None
        self._current_gating: Optional[ConsciousnessGatingOutput] = None
        self._current_allocation: Optional[ResourceAllocationOutput] = None

        # State transition history
        self._transitions: List[StateTransition] = []
        self._max_transitions = 100

        # Callbacks for state changes
        self._state_callbacks: List[Callable[[ArousalLevelOutput], None]] = []

        # Active forms tracking
        self._active_forms: Set[str] = set()
        self._form_demands: Dict[str, float] = {}

        # Initialize with default state
        self._initialize_default_state()

        logger.info(f"Initialized {self.FORM_NAME}")

    def _initialize_default_state(self) -> None:
        """Initialize to default alert state."""
        self._current_arousal = ArousalLevelOutput(
            arousal_level=0.5,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=1.0,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.8,
            components={source: 0.5 for source in ArousalSource},
        )
        self._current_gating = ConsciousnessGatingOutput(
            sensory_gates={m.value: 0.7 for m in SensoryModality},
            cognitive_gates={
                "memory_access": 0.8,
                "executive_control": 0.8,
                "meta_cognitive": 0.7,
                "emotional_processing": 0.8,
            },
            global_threshold=0.5,
            gate_adaptation_rate=0.7,
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def process_inputs(self, inputs: ArousalInputBundle) -> ArousalLevelOutput:
        """
        Process all arousal inputs and update system state.

        This is the main entry point for arousal updates.
        """
        # Compute new arousal level
        new_arousal = self.computation_engine.compute_arousal(inputs)

        # Check for state transition
        if self._current_arousal and new_arousal.arousal_state != self._current_arousal.arousal_state:
            self._record_transition(
                self._current_arousal.arousal_state,
                new_arousal.arousal_state,
                new_arousal.primary_source,
            )

        # Update current state
        self._current_arousal = new_arousal

        # Recompute gating
        threat_level = 0.0
        novelty_level = 0.0
        if inputs.threat_input:
            threat_level = inputs.threat_input.threat_level
        if inputs.novelty_input:
            novelty_level = inputs.novelty_input.novelty_level

        self._current_gating = self.gating_engine.compute_gates(
            new_arousal, threat_level, novelty_level
        )

        # Recompute resource allocation
        self._current_allocation = self.allocation_engine.allocate_resources(
            new_arousal, self._form_demands, self._active_forms
        )

        # Notify callbacks
        await self._notify_state_change(new_arousal)

        return new_arousal

    def get_arousal_level(self) -> float:
        """Get current arousal level (0.0-1.0)."""
        if self._current_arousal:
            return self._current_arousal.arousal_level
        return 0.5

    def get_arousal_state(self) -> ArousalState:
        """Get current arousal state."""
        if self._current_arousal:
            return self._current_arousal.arousal_state
        return ArousalState.ALERT

    def get_gating_signals(self) -> ConsciousnessGatingOutput:
        """Get current consciousness gating configuration."""
        if self._current_gating:
            return self._current_gating
        return ConsciousnessGatingOutput(
            sensory_gates={},
            cognitive_gates={},
            global_threshold=0.5,
            gate_adaptation_rate=0.5,
        )

    def get_resource_allocation(self) -> ResourceAllocationOutput:
        """Get current resource allocation."""
        if self._current_allocation:
            return self._current_allocation
        return ResourceAllocationOutput(
            total_available=1.0,
            allocations={},
            reserve_capacity=0.1,
            allocation_confidence=0.5,
        )

    def get_gate_for_form(self, form_id: str) -> float:
        """Get the gate openness for a specific form."""
        if not self._current_gating:
            return 0.7

        # Sensory forms
        if form_id.startswith("01"):
            return self._current_gating.sensory_gates.get("visual", 0.7)
        elif form_id.startswith("02"):
            return self._current_gating.sensory_gates.get("auditory", 0.7)
        elif form_id.startswith("03"):
            return self._current_gating.sensory_gates.get("somatosensory", 0.7)
        elif form_id.startswith("04"):
            return self._current_gating.sensory_gates.get("olfactory", 0.7)
        elif form_id.startswith("05"):
            return self._current_gating.sensory_gates.get("gustatory", 0.7)
        elif form_id.startswith("06"):
            return self._current_gating.sensory_gates.get("interoceptive", 0.7)

        # Cognitive forms
        elif form_id.startswith("11") or form_id.startswith("12"):
            return self._current_gating.cognitive_gates.get("meta_cognitive", 0.7)

        # Default
        return self._current_gating.cognitive_gates.get("executive_control", 0.7)

    def is_form_allowed(self, form_id: str) -> bool:
        """Check if a form is allowed to process based on current arousal."""
        if not self._current_arousal:
            return True

        state = self._current_arousal.arousal_state

        # Critical forms always allowed
        if form_id in ["08-arousal", "13-integrated-information", "14-global-workspace"]:
            return True

        # In sleep, only critical forms
        if state == ArousalState.SLEEP:
            return False

        # In drowsy, only high priority
        if state == ArousalState.DROWSY:
            priority = ResourceAllocationEngine.FORM_PRIORITIES.get(form_id, 0.3)
            return priority >= 0.7

        return True

    def register_form(self, form_id: str, base_demand: float = 0.1) -> None:
        """Register a form as active."""
        self._active_forms.add(form_id)
        self._form_demands[form_id] = base_demand

    def unregister_form(self, form_id: str) -> None:
        """Unregister a form."""
        self._active_forms.discard(form_id)
        self._form_demands.pop(form_id, None)

    def on_state_change(self, callback: Callable[[ArousalLevelOutput], None]) -> None:
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    def get_status(self) -> ArousalSystemStatus:
        """Get complete system status."""
        return ArousalSystemStatus(
            current_level=self._current_arousal or self._get_default_arousal(),
            current_gating=self._current_gating or self._get_default_gating(),
            current_allocation=self._current_allocation or self._get_default_allocation(),
            recent_transitions=self._transitions[-10:],
            system_health=self._compute_health(),
            responsiveness=self._compute_responsiveness(),
            stability=self._current_arousal.arousal_stability if self._current_arousal else 1.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "is_critical": self.IS_CRITICAL,
            "arousal": self._current_arousal.to_dict() if self._current_arousal else None,
            "gating": self._current_gating.to_dict() if self._current_gating else None,
            "allocation": self._current_allocation.to_dict() if self._current_allocation else None,
            "active_forms": list(self._active_forms),
            "recent_transitions": len(self._transitions),
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _record_transition(
        self,
        from_state: ArousalState,
        to_state: ArousalState,
        trigger: ArousalSource
    ) -> None:
        """Record a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            transition_type=self._classify_transition(from_state, to_state),
            trigger=trigger,
            duration_ms=0.0,  # Would need timing
        )
        self._transitions.append(transition)
        if len(self._transitions) > self._max_transitions:
            self._transitions.pop(0)

        logger.info(f"Arousal transition: {from_state.value} -> {to_state.value}")

    def _classify_transition(
        self, from_state: ArousalState, to_state: ArousalState
    ) -> StateTransitionType:
        """Classify the type of state transition."""
        state_order = [
            ArousalState.SLEEP,
            ArousalState.DROWSY,
            ArousalState.RELAXED,
            ArousalState.ALERT,
            ArousalState.FOCUSED,
            ArousalState.HYPERAROUSED,
        ]
        from_idx = state_order.index(from_state)
        to_idx = state_order.index(to_state)
        diff = abs(to_idx - from_idx)

        if diff <= 1:
            return StateTransitionType.GRADUAL
        elif diff >= 3:
            return StateTransitionType.RAPID
        else:
            return StateTransitionType.GRADUAL

    async def _notify_state_change(self, arousal: ArousalLevelOutput) -> None:
        """Notify registered callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(arousal)
                else:
                    callback(arousal)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _compute_health(self) -> float:
        """Compute overall system health."""
        if not self._current_arousal:
            return 1.0

        # Health based on stability and appropriate state
        stability = self._current_arousal.arousal_stability
        confidence = self._current_arousal.confidence
        return (stability + confidence) / 2

    def _compute_responsiveness(self) -> float:
        """Compute system responsiveness."""
        if not self._current_gating:
            return 0.7
        return self._current_gating.gate_adaptation_rate

    def _get_default_arousal(self) -> ArousalLevelOutput:
        """Get default arousal output."""
        return ArousalLevelOutput(
            arousal_level=0.5,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=1.0,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.5,
        )

    def _get_default_gating(self) -> ConsciousnessGatingOutput:
        """Get default gating output."""
        return ConsciousnessGatingOutput(
            sensory_gates={m.value: 0.7 for m in SensoryModality},
            cognitive_gates={"memory_access": 0.7, "executive_control": 0.7},
            global_threshold=0.5,
            gate_adaptation_rate=0.5,
        )

    def _get_default_allocation(self) -> ResourceAllocationOutput:
        """Get default allocation output."""
        return ResourceAllocationOutput(
            total_available=1.0,
            allocations={},
            reserve_capacity=0.1,
            allocation_confidence=0.5,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_arousal_interface() -> ArousalConsciousnessInterface:
    """Create and return an arousal consciousness interface."""
    return ArousalConsciousnessInterface()


def create_simple_input(
    sensory_level: float = 0.5,
    emotional_level: float = 0.5,
    circadian_level: float = 0.5,
    threat_level: float = 0.0
) -> ArousalInputBundle:
    """Create a simple input bundle for testing."""
    return ArousalInputBundle(
        sensory_inputs=[
            SensoryArousalInput(
                source_modality=SensoryModality.VISUAL,
                stimulus_type="neutral",
                intensity=sensory_level,
                salience=sensory_level,
                change_rate=0.0,
            )
        ],
        emotional_input=EmotionalInput(
            valence=0.0,
            arousal_component=emotional_level,
        ),
        circadian_input=CircadianInput(
            circadian_phase=12.0,
            melatonin_level=1.0 - circadian_level,
            cortisol_level=circadian_level,
            sleep_pressure=1.0 - circadian_level,
            light_exposure=circadian_level,
        ),
        threat_input=ThreatInput(
            threat_level=threat_level,
            threat_type="unknown",
            proximity=threat_level,
            certainty=threat_level,
            response_urgency=threat_level,
        ) if threat_level > 0 else None,
    )
