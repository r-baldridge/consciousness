#!/usr/bin/env python3
"""
Higher-Order Thought (HOT) Theory Consciousness Interface

Form 15: Implements Higher-Order Thought Theory as proposed by David Rosenthal
and further developed by Hakwan Lau. HOT theory posits that a mental state
is conscious when there is a higher-order representation (a thought about
that thought) that represents oneself as being in that state. Consciousness
requires not just having a mental state, but being aware that one is in
that state - a meta-representation.

This module creates higher-order representations of first-order states,
assesses whether a state qualifies as conscious according to HOT criteria,
and maintains a representation hierarchy.
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

class RepresentationOrder(Enum):
    """
    Order of mental representations in the HOT hierarchy.

    First-order states are basic perceptual/cognitive states.
    Higher orders represent awareness of those states.
    """
    FIRST_ORDER = "first_order"        # Basic perceptual/cognitive state
    SECOND_ORDER = "second_order"      # Thought about first-order state (HOT)
    THIRD_ORDER = "third_order"        # Thought about the HOT (meta-meta)
    FOURTH_ORDER = "fourth_order"      # Rare: further meta-representation


class ConsciousnessType(Enum):
    """Types of consciousness according to HOT theory."""
    UNCONSCIOUS = "unconscious"                # No higher-order representation
    PHENOMENAL = "phenomenal"                  # Basic experiential quality
    ACCESS = "access"                          # Available for report/reasoning
    INTROSPECTIVE = "introspective"            # Full introspective awareness
    SELF_REFLECTIVE = "self_reflective"        # Reflective self-awareness


class RepresentationModality(Enum):
    """Modality of the representation."""
    PERCEPTUAL = "perceptual"              # Sensory/perceptual content
    COGNITIVE = "cognitive"                # Abstract thought content
    EMOTIONAL = "emotional"                # Emotional/affective content
    BODILY = "bodily"                      # Somatic/bodily content
    LINGUISTIC = "linguistic"              # Linguistically structured content
    IMAGISTIC = "imagistic"                # Mental imagery content


class HOTQuality(Enum):
    """Quality attributes of a higher-order thought."""
    CLARITY = "clarity"                    # How clear the HOT is
    CONFIDENCE = "confidence"              # Confidence in the representation
    SPECIFICITY = "specificity"            # How specific vs. vague
    STABILITY = "stability"               # Temporal stability
    ACCURACY = "accuracy"                  # Match with first-order state


class AssessmentCriterion(Enum):
    """Criteria for consciousness assessment."""
    HAS_HOT = "has_hot"                    # Higher-order thought exists
    APPROPRIATE_TARGET = "appropriate_target"  # HOT targets the right state
    SUFFICIENT_QUALITY = "sufficient_quality"  # HOT quality is adequate
    TEMPORAL_MATCH = "temporal_match"        # HOT is temporally concurrent
    SELF_REFERENTIAL = "self_referential"    # HOT references the self


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class FirstOrderState:
    """
    A first-order mental state (perceptual or cognitive content).

    This is the basic state that may or may not be conscious,
    depending on whether a higher-order thought represents it.
    """
    state_id: str
    modality: RepresentationModality
    content: Dict[str, Any]             # The representational content
    intensity: float                    # 0.0-1.0: Signal strength
    distinctness: float                 # 0.0-1.0: How well-defined
    valence: float = 0.0              # -1.0 to 1.0: Affective valence
    activation_level: float = 0.5     # 0.0-1.0: Neural activation
    is_attended: bool = False          # Whether attention is directed here
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "modality": self.modality.value,
            "content": self.content,
            "intensity": round(self.intensity, 4),
            "distinctness": round(self.distinctness, 4),
            "valence": round(self.valence, 4),
            "activation_level": round(self.activation_level, 4),
            "is_attended": self.is_attended,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HOTRequest:
    """Request to create a higher-order thought about a state."""
    target_state_id: str
    requested_order: RepresentationOrder = RepresentationOrder.SECOND_ORDER
    attention_boost: float = 0.0       # Extra attention directed to the state
    introspective_effort: float = 0.5  # 0.0-1.0: How much effort to introspect
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class HigherOrderRepresentation:
    """
    A higher-order representation (thought about a mental state).

    This is the key construct in HOT theory - a meta-representation
    that makes the target state conscious.
    """
    hot_id: str
    target_state_id: str
    order: RepresentationOrder
    content_summary: str                # What the HOT represents
    quality_scores: Dict[HOTQuality, float]  # Quality attributes
    overall_quality: float              # 0.0-1.0: Aggregate quality
    is_conscious_making: bool           # Whether this HOT makes the target conscious
    modality: RepresentationModality = RepresentationModality.COGNITIVE
    self_attribution: bool = True       # Whether the HOT attributes the state to self
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hot_id": self.hot_id,
            "target_state_id": self.target_state_id,
            "order": self.order.value,
            "content_summary": self.content_summary,
            "quality_scores": {k.value: round(v, 4) for k, v in self.quality_scores.items()},
            "overall_quality": round(self.overall_quality, 4),
            "is_conscious_making": self.is_conscious_making,
            "self_attribution": self.self_attribution,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsciousnessAssessment:
    """Assessment of whether a state is conscious according to HOT theory."""
    state_id: str
    is_conscious: bool
    consciousness_type: ConsciousnessType
    criteria_met: Dict[AssessmentCriterion, bool]
    hot_chain: List[HigherOrderRepresentation]  # Chain of HOTs
    highest_order: RepresentationOrder
    confidence: float                   # 0.0-1.0: Assessment confidence
    explanation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "is_conscious": self.is_conscious,
            "consciousness_type": self.consciousness_type.value,
            "criteria_met": {k.value: v for k, v in self.criteria_met.items()},
            "highest_order": self.highest_order.value,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "num_hot_levels": len(self.hot_chain),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RepresentationHierarchy:
    """The full hierarchy of representations for a state."""
    root_state: FirstOrderState
    representations: Dict[RepresentationOrder, List[HigherOrderRepresentation]]
    depth: int                          # How many orders deep
    total_representations: int
    consciousness_assessment: Optional[ConsciousnessAssessment] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_state_id": self.root_state.state_id,
            "depth": self.depth,
            "total_representations": self.total_representations,
            "orders": {
                order.value: [r.to_dict() for r in reps]
                for order, reps in self.representations.items()
            },
            "is_conscious": self.consciousness_assessment.is_conscious if self.consciousness_assessment else None,
        }


@dataclass
class HOTOutput:
    """Complete output from HOT processing."""
    first_order_state: FirstOrderState
    higher_order_thoughts: List[HigherOrderRepresentation]
    assessment: ConsciousnessAssessment
    hierarchy: RepresentationHierarchy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_order_state": self.first_order_state.to_dict(),
            "num_hots": len(self.higher_order_thoughts),
            "assessment": self.assessment.to_dict(),
            "hierarchy": self.hierarchy.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HOTSystemStatus:
    """Status of the HOT system."""
    is_initialized: bool
    total_states_processed: int
    total_hots_created: int
    conscious_state_count: int
    unconscious_state_count: int
    system_health: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# HOT GENERATION ENGINE
# ============================================================================

class HOTGenerationEngine:
    """
    Engine for generating higher-order thoughts about first-order states.

    Implements the core HOT mechanism: creating meta-representations
    that potentially make target states conscious.
    """

    CONSCIOUSNESS_THRESHOLD = 0.4   # Minimum HOT quality for consciousness
    HOT_COUNTER = 0

    def __init__(self):
        self._hot_cache: Dict[str, HigherOrderRepresentation] = {}

    def generate_hot(
        self,
        target_state: FirstOrderState,
        order: RepresentationOrder = RepresentationOrder.SECOND_ORDER,
        introspective_effort: float = 0.5,
        attention_boost: float = 0.0
    ) -> HigherOrderRepresentation:
        """
        Generate a higher-order thought about a target state.

        The quality of the HOT depends on the target state's properties,
        the introspective effort, and attentional resources.
        """
        HOTGenerationEngine.HOT_COUNTER += 1
        hot_id = f"hot_{HOTGenerationEngine.HOT_COUNTER}"

        # Compute quality scores
        quality_scores = self._compute_quality_scores(
            target_state, order, introspective_effort, attention_boost
        )

        # Overall quality is weighted average
        overall_quality = self._compute_overall_quality(quality_scores)

        # Determine if this HOT makes the target conscious
        is_conscious_making = overall_quality >= self.CONSCIOUSNESS_THRESHOLD

        # Generate content summary
        content_summary = self._generate_content_summary(target_state, order)

        hot = HigherOrderRepresentation(
            hot_id=hot_id,
            target_state_id=target_state.state_id,
            order=order,
            content_summary=content_summary,
            quality_scores=quality_scores,
            overall_quality=overall_quality,
            is_conscious_making=is_conscious_making,
            modality=RepresentationModality.COGNITIVE,
            self_attribution=True,
        )

        self._hot_cache[hot_id] = hot
        return hot

    def _compute_quality_scores(
        self,
        state: FirstOrderState,
        order: RepresentationOrder,
        effort: float,
        attention_boost: float
    ) -> Dict[HOTQuality, float]:
        """Compute quality scores for the HOT."""
        # Base quality depends on target state properties
        base = state.intensity * 0.3 + state.distinctness * 0.3 + state.activation_level * 0.2

        # Attention boosts clarity and specificity
        attention_factor = 0.5 + attention_boost * 0.5
        if state.is_attended:
            attention_factor += 0.2

        # Higher orders are generally less clear
        order_penalty = {
            RepresentationOrder.SECOND_ORDER: 0.0,
            RepresentationOrder.THIRD_ORDER: 0.15,
            RepresentationOrder.FOURTH_ORDER: 0.3,
        }.get(order, 0.0)

        clarity = max(0.0, min(1.0, base * attention_factor - order_penalty + effort * 0.3))
        confidence = max(0.0, min(1.0, state.distinctness * 0.5 + effort * 0.3 + attention_factor * 0.2 - order_penalty))
        specificity = max(0.0, min(1.0, state.distinctness * 0.4 + effort * 0.4 + state.intensity * 0.2 - order_penalty))
        stability = max(0.0, min(1.0, state.activation_level * 0.5 + effort * 0.3 + 0.2 - order_penalty))
        accuracy = max(0.0, min(1.0, state.intensity * 0.3 + state.distinctness * 0.3 + attention_factor * 0.4 - order_penalty * 0.5))

        return {
            HOTQuality.CLARITY: clarity,
            HOTQuality.CONFIDENCE: confidence,
            HOTQuality.SPECIFICITY: specificity,
            HOTQuality.STABILITY: stability,
            HOTQuality.ACCURACY: accuracy,
        }

    def _compute_overall_quality(self, scores: Dict[HOTQuality, float]) -> float:
        """Compute overall HOT quality from individual scores."""
        weights = {
            HOTQuality.CLARITY: 0.25,
            HOTQuality.CONFIDENCE: 0.20,
            HOTQuality.SPECIFICITY: 0.20,
            HOTQuality.STABILITY: 0.15,
            HOTQuality.ACCURACY: 0.20,
        }
        return sum(scores.get(q, 0.0) * w for q, w in weights.items())

    def _generate_content_summary(
        self, state: FirstOrderState, order: RepresentationOrder
    ) -> str:
        """Generate a textual summary of the HOT content."""
        modality_desc = state.modality.value
        if order == RepresentationOrder.SECOND_ORDER:
            return f"Awareness of {modality_desc} state '{state.state_id}'"
        elif order == RepresentationOrder.THIRD_ORDER:
            return f"Awareness of being aware of {modality_desc} state '{state.state_id}'"
        elif order == RepresentationOrder.FOURTH_ORDER:
            return f"Meta-awareness of awareness chain for {modality_desc} state '{state.state_id}'"
        return f"Representation of state '{state.state_id}'"


# ============================================================================
# CONSCIOUSNESS ASSESSMENT ENGINE
# ============================================================================

class ConsciousnessAssessmentEngine:
    """
    Engine that assesses whether a state is conscious based on HOT criteria.

    Evaluates the criteria specified by HOT theory to determine
    if a first-order state achieves consciousness.
    """

    def assess(
        self,
        first_order: FirstOrderState,
        hots: List[HigherOrderRepresentation]
    ) -> ConsciousnessAssessment:
        """Assess consciousness status based on available HOTs."""
        criteria = {}

        # Criterion 1: Does a HOT exist?
        has_hot = len(hots) > 0
        criteria[AssessmentCriterion.HAS_HOT] = has_hot

        # Criterion 2: Does the HOT target the right state?
        appropriate_target = any(
            h.target_state_id == first_order.state_id for h in hots
        )
        criteria[AssessmentCriterion.APPROPRIATE_TARGET] = appropriate_target

        # Criterion 3: Is HOT quality sufficient?
        sufficient_quality = any(h.is_conscious_making for h in hots)
        criteria[AssessmentCriterion.SUFFICIENT_QUALITY] = sufficient_quality

        # Criterion 4: Temporal match (simplified: always true if HOT exists)
        temporal_match = has_hot
        criteria[AssessmentCriterion.TEMPORAL_MATCH] = temporal_match

        # Criterion 5: Self-referential
        self_referential = any(h.self_attribution for h in hots)
        criteria[AssessmentCriterion.SELF_REFERENTIAL] = self_referential

        # Determine consciousness type
        is_conscious = all([has_hot, appropriate_target, sufficient_quality])
        consciousness_type = self._determine_type(criteria, hots)

        # Determine highest order
        highest_order = RepresentationOrder.FIRST_ORDER
        for h in hots:
            if h.order.value > highest_order.value:
                highest_order = h.order

        # Compute confidence
        confidence = self._compute_confidence(criteria, hots)

        # Generate explanation
        explanation = self._generate_explanation(criteria, consciousness_type)

        return ConsciousnessAssessment(
            state_id=first_order.state_id,
            is_conscious=is_conscious,
            consciousness_type=consciousness_type,
            criteria_met=criteria,
            hot_chain=hots,
            highest_order=highest_order,
            confidence=confidence,
            explanation=explanation,
        )

    def _determine_type(
        self,
        criteria: Dict[AssessmentCriterion, bool],
        hots: List[HigherOrderRepresentation]
    ) -> ConsciousnessType:
        """Determine the type of consciousness based on criteria."""
        if not criteria.get(AssessmentCriterion.HAS_HOT, False):
            return ConsciousnessType.UNCONSCIOUS

        if not criteria.get(AssessmentCriterion.SUFFICIENT_QUALITY, False):
            return ConsciousnessType.PHENOMENAL

        has_third_order = any(
            h.order == RepresentationOrder.THIRD_ORDER for h in hots
        )
        has_fourth_order = any(
            h.order == RepresentationOrder.FOURTH_ORDER for h in hots
        )

        if has_fourth_order:
            return ConsciousnessType.SELF_REFLECTIVE
        elif has_third_order:
            return ConsciousnessType.INTROSPECTIVE
        else:
            return ConsciousnessType.ACCESS

    def _compute_confidence(
        self,
        criteria: Dict[AssessmentCriterion, bool],
        hots: List[HigherOrderRepresentation]
    ) -> float:
        """Compute assessment confidence."""
        criteria_score = sum(1 for v in criteria.values() if v) / max(1, len(criteria))
        hot_quality = max((h.overall_quality for h in hots), default=0.0)
        return criteria_score * 0.6 + hot_quality * 0.4

    def _generate_explanation(
        self,
        criteria: Dict[AssessmentCriterion, bool],
        consciousness_type: ConsciousnessType
    ) -> str:
        """Generate a human-readable explanation."""
        met = [k.value for k, v in criteria.items() if v]
        not_met = [k.value for k, v in criteria.items() if not v]

        if consciousness_type == ConsciousnessType.UNCONSCIOUS:
            return f"State is unconscious. Missing criteria: {', '.join(not_met)}"
        else:
            return (
                f"State has {consciousness_type.value} consciousness. "
                f"Met criteria: {', '.join(met)}."
            )


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class HigherOrderThoughtInterface:
    """
    Main interface for Form 15: Higher-Order Thought Theory.

    Implements HOT theory's approach to consciousness: a mental state
    becomes conscious when a higher-order thought represents oneself
    as being in that state. The interface creates HOTs, assesses
    consciousness, and maintains the representation hierarchy.
    """

    FORM_ID = "15-higher-order-thought"
    FORM_NAME = "Higher-Order Thought Theory (HOT)"

    def __init__(self):
        """Initialize the Higher-Order Thought interface."""
        self._generation_engine = HOTGenerationEngine()
        self._assessment_engine = ConsciousnessAssessmentEngine()

        # State tracking
        self._first_order_states: Dict[str, FirstOrderState] = {}
        self._hots: Dict[str, List[HigherOrderRepresentation]] = {}
        self._assessments: Dict[str, ConsciousnessAssessment] = {}

        # Counters
        self._is_initialized = False
        self._states_processed = 0
        self._hots_created = 0
        self._conscious_count = 0
        self._unconscious_count = 0

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the HOT system."""
        self._is_initialized = True
        logger.info(f"{self.FORM_NAME} initialized and ready")

    async def create_hot(
        self,
        first_order_state: FirstOrderState,
        order: RepresentationOrder = RepresentationOrder.SECOND_ORDER,
        introspective_effort: float = 0.5,
        attention_boost: float = 0.0,
    ) -> HigherOrderRepresentation:
        """
        Create a higher-order thought about a first-order state.

        This is the core operation in HOT theory - generating
        a meta-representation that may make the target state conscious.
        """
        # Store the first-order state
        self._first_order_states[first_order_state.state_id] = first_order_state

        # Generate the HOT
        hot = self._generation_engine.generate_hot(
            first_order_state, order, introspective_effort, attention_boost
        )

        # Store the HOT
        if first_order_state.state_id not in self._hots:
            self._hots[first_order_state.state_id] = []
        self._hots[first_order_state.state_id].append(hot)

        self._hots_created += 1

        return hot

    async def assess_consciousness(
        self, state_id: str
    ) -> ConsciousnessAssessment:
        """
        Assess whether a state is conscious according to HOT criteria.

        Evaluates all available higher-order thoughts about the state
        and determines consciousness status.
        """
        first_order = self._first_order_states.get(state_id)
        if first_order is None:
            # Create a minimal assessment for unknown states
            return ConsciousnessAssessment(
                state_id=state_id,
                is_conscious=False,
                consciousness_type=ConsciousnessType.UNCONSCIOUS,
                criteria_met={c: False for c in AssessmentCriterion},
                hot_chain=[],
                highest_order=RepresentationOrder.FIRST_ORDER,
                confidence=0.0,
                explanation="State not found in the system.",
            )

        hots = self._hots.get(state_id, [])
        assessment = self._assessment_engine.assess(first_order, hots)

        self._assessments[state_id] = assessment
        self._states_processed += 1

        if assessment.is_conscious:
            self._conscious_count += 1
        else:
            self._unconscious_count += 1

        return assessment

    async def get_representation_hierarchy(
        self, state_id: str
    ) -> RepresentationHierarchy:
        """
        Get the full representation hierarchy for a state.

        Shows all levels of representation from the first-order
        state through all higher-order thoughts.
        """
        first_order = self._first_order_states.get(state_id)
        if first_order is None:
            # Return empty hierarchy
            dummy_state = FirstOrderState(
                state_id=state_id,
                modality=RepresentationModality.COGNITIVE,
                content={},
                intensity=0.0,
                distinctness=0.0,
            )
            return RepresentationHierarchy(
                root_state=dummy_state,
                representations={},
                depth=0,
                total_representations=0,
            )

        hots = self._hots.get(state_id, [])

        # Organize by order
        by_order: Dict[RepresentationOrder, List[HigherOrderRepresentation]] = {}
        for hot in hots:
            if hot.order not in by_order:
                by_order[hot.order] = []
            by_order[hot.order].append(hot)

        depth = len(by_order)
        total = sum(len(v) for v in by_order.values())

        assessment = self._assessments.get(state_id)

        return RepresentationHierarchy(
            root_state=first_order,
            representations=by_order,
            depth=depth,
            total_representations=total,
            consciousness_assessment=assessment,
        )

    async def process_state(
        self,
        first_order_state: FirstOrderState,
        max_order: RepresentationOrder = RepresentationOrder.SECOND_ORDER,
        introspective_effort: float = 0.5,
    ) -> HOTOutput:
        """
        Full pipeline: create HOTs and assess consciousness for a state.

        This is the main entry point that combines HOT creation
        and consciousness assessment.
        """
        hots = []

        # Generate HOTs up to the requested order
        orders = [RepresentationOrder.SECOND_ORDER]
        if max_order in [RepresentationOrder.THIRD_ORDER, RepresentationOrder.FOURTH_ORDER]:
            orders.append(RepresentationOrder.THIRD_ORDER)
        if max_order == RepresentationOrder.FOURTH_ORDER:
            orders.append(RepresentationOrder.FOURTH_ORDER)

        for order in orders:
            hot = await self.create_hot(
                first_order_state, order, introspective_effort
            )
            hots.append(hot)

        assessment = await self.assess_consciousness(first_order_state.state_id)
        hierarchy = await self.get_representation_hierarchy(first_order_state.state_id)

        return HOTOutput(
            first_order_state=first_order_state,
            higher_order_thoughts=hots,
            assessment=assessment,
            hierarchy=hierarchy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "is_initialized": self._is_initialized,
            "states_processed": self._states_processed,
            "hots_created": self._hots_created,
            "conscious_count": self._conscious_count,
            "unconscious_count": self._unconscious_count,
            "tracked_states": len(self._first_order_states),
        }

    def get_status(self) -> HOTSystemStatus:
        """Get current system status."""
        return HOTSystemStatus(
            is_initialized=self._is_initialized,
            total_states_processed=self._states_processed,
            total_hots_created=self._hots_created,
            conscious_state_count=self._conscious_count,
            unconscious_state_count=self._unconscious_count,
            system_health=1.0 if self._is_initialized else 0.5,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_higher_order_thought_interface() -> HigherOrderThoughtInterface:
    """Create and return a Higher-Order Thought interface."""
    return HigherOrderThoughtInterface()


def create_first_order_state(
    state_id: str,
    modality: RepresentationModality = RepresentationModality.PERCEPTUAL,
    intensity: float = 0.5,
    distinctness: float = 0.5,
    content: Optional[Dict[str, Any]] = None,
) -> FirstOrderState:
    """Create a first-order state for testing."""
    return FirstOrderState(
        state_id=state_id,
        modality=modality,
        content=content or {"description": f"State {state_id}"},
        intensity=intensity,
        distinctness=distinctness,
    )


__all__ = [
    # Enums
    "RepresentationOrder",
    "ConsciousnessType",
    "RepresentationModality",
    "HOTQuality",
    "AssessmentCriterion",
    # Input dataclasses
    "FirstOrderState",
    "HOTRequest",
    # Output dataclasses
    "HigherOrderRepresentation",
    "ConsciousnessAssessment",
    "RepresentationHierarchy",
    "HOTOutput",
    "HOTSystemStatus",
    # Engines
    "HOTGenerationEngine",
    "ConsciousnessAssessmentEngine",
    # Main interface
    "HigherOrderThoughtInterface",
    # Convenience functions
    "create_higher_order_thought_interface",
    "create_first_order_state",
]
