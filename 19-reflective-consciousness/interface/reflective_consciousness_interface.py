#!/usr/bin/env python3
"""
Reflective Consciousness Interface

Form 19: Higher-order reflective consciousness -- the capacity for
deliberate self-examination, introspection, and recursive thought about
one's own mental states. Whereas primary consciousness (Form 18) constructs
a scene from sensory data, reflective consciousness takes that scene (or
any mental content) as an object of further thought.

Core capabilities:
- Introspection: examining one's own beliefs, desires, emotions
- Deliberation: weighing options and reasons before acting
- Evaluation: assessing the quality or correctness of thoughts
- Recursive reflection: thinking about thinking (meta-cognition)
- Insight generation: deriving new understanding from reflection

Reflective consciousness is associated with the prefrontal cortex and is
considered a hallmark of higher-order consciousness in humans.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ReflectionType(Enum):
    """
    Types of reflective activity.

    Each type targets a different aspect of mental content and
    serves a different cognitive purpose.
    """
    INTROSPECTIVE = "introspective"      # Examining internal states
    DELIBERATIVE = "deliberative"        # Weighing reasons and options
    EVALUATIVE = "evaluative"            # Judging quality or correctness
    METACOGNITIVE = "metacognitive"      # Thinking about thinking
    COUNTERFACTUAL = "counterfactual"    # Considering alternatives
    PROSPECTIVE = "prospective"          # Anticipating future states


class ReflectionDepth(Enum):
    """
    Depth of reflective processing.

    Deeper reflection involves more recursive self-reference and
    typically requires more time and cognitive resources.
    """
    SURFACE = "surface"            # Quick, automatic self-check
    SHALLOW = "shallow"            # Brief directed reflection
    MODERATE = "moderate"          # Sustained reflective analysis
    DEEP = "deep"                  # Extended recursive examination
    PROFOUND = "profound"          # Transformative reflective insight


class CognitiveStrategy(Enum):
    """
    Strategies used during deliberation and evaluation.
    """
    ANALYTIC = "analytic"              # Step-by-step logical analysis
    INTUITIVE = "intuitive"            # Gut-feeling holistic assessment
    COMPARATIVE = "comparative"        # Comparing alternatives
    DIALECTICAL = "dialectical"        # Thesis-antithesis-synthesis
    NARRATIVE = "narrative"            # Story-based sense-making
    EMBODIED = "embodied"              # Somatic / felt-sense reasoning
    PROBABILISTIC = "probabilistic"    # Weighing likelihoods


class InsightQuality(Enum):
    """Quality of generated insights."""
    TRIVIAL = "trivial"                # Obvious, already known
    INCREMENTAL = "incremental"        # Small step forward
    SIGNIFICANT = "significant"        # Meaningful new understanding
    BREAKTHROUGH = "breakthrough"      # Major shift in perspective
    TRANSFORMATIVE = "transformative"  # Fundamentally changes self-model


class ReflectiveState(Enum):
    """Current state of the reflective system."""
    IDLE = "idle"                      # Not reflecting
    ATTENDING = "attending"            # Focusing on content
    REFLECTING = "reflecting"          # Active reflection in progress
    INTEGRATING = "integrating"        # Incorporating insight
    RESTING = "resting"                # Post-reflection cool-down


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class ThoughtContent:
    """Represents a discrete thought or mental content to reflect upon."""
    content_id: str
    content_type: str              # "belief", "desire", "emotion", "memory", "plan"
    description: str
    confidence: float = 0.5        # 0.0-1.0
    emotional_charge: float = 0.0  # -1.0 to 1.0
    source: str = ""               # Where the thought originated
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "emotional_charge": round(self.emotional_charge, 4),
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReflectiveContext:
    """Context information for a reflective episode."""
    goal: str = ""                 # What the reflection aims to achieve
    constraints: List[str] = field(default_factory=list)
    prior_insights: List[str] = field(default_factory=list)
    time_budget_ms: float = 5000.0  # How long reflection can take
    depth_requested: ReflectionDepth = ReflectionDepth.MODERATE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReflectiveInput:
    """
    Complete input for a reflective episode.
    """
    thought: ThoughtContent
    context: ReflectiveContext = field(default_factory=ReflectiveContext)
    reflection_type: ReflectionType = ReflectionType.INTROSPECTIVE
    related_thoughts: List[ThoughtContent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought.to_dict(),
            "reflection_type": self.reflection_type.value,
            "depth_requested": self.context.depth_requested.value,
            "related_count": len(self.related_thoughts),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class ReflectiveInsight:
    """An insight produced through reflection."""
    insight_id: str
    content: str
    quality: InsightQuality
    confidence: float              # 0.0-1.0
    source_reflection: str         # ID of the reflection that produced it
    supporting_reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "content": self.content,
            "quality": self.quality.value,
            "confidence": round(self.confidence, 4),
            "source_reflection": self.source_reflection,
            "supporting_reasons": self.supporting_reasons,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DeliberationOption:
    """An option considered during deliberation."""
    option_id: str
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    estimated_value: float = 0.5   # 0.0-1.0
    feasibility: float = 0.5       # 0.0-1.0
    risk: float = 0.5              # 0.0-1.0


@dataclass
class ReflectiveOutput:
    """
    Complete output of a reflective episode.
    """
    reflection_id: str
    reflection_type: ReflectionType
    depth_achieved: ReflectionDepth
    strategy_used: CognitiveStrategy
    insight: Optional[ReflectiveInsight]
    evaluation_result: Optional[str]      # For evaluative reflections
    decision: Optional[str]               # For deliberative reflections
    options_considered: List[DeliberationOption] = field(default_factory=list)
    processing_time_ms: float = 0.0
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "reflection_type": self.reflection_type.value,
            "depth_achieved": self.depth_achieved.value,
            "strategy_used": self.strategy_used.value,
            "insight": self.insight.to_dict() if self.insight else None,
            "evaluation_result": self.evaluation_result,
            "decision": self.decision,
            "options_considered": len(self.options_considered),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class ReflectiveConsciousnessInterface:
    """
    Main interface for Form 19: Reflective Consciousness.

    Provides higher-order reflection capabilities including introspection,
    deliberation, evaluation, and recursive meta-cognition. This form
    takes the outputs of primary consciousness (Form 18) and other forms
    as objects of reflective thought.
    """

    FORM_ID = "19-reflective-consciousness"
    FORM_NAME = "Reflective Consciousness"

    def __init__(self):
        """Initialize the reflective consciousness interface."""
        # Reflection history
        self._reflection_history: List[ReflectiveOutput] = []
        self._insight_store: Dict[str, ReflectiveInsight] = {}
        self._reflection_counter: int = 0
        self._insight_counter: int = 0

        # Current state
        self._state: ReflectiveState = ReflectiveState.IDLE
        self._current_depth: ReflectionDepth = ReflectionDepth.SURFACE

        # Strategy selection weights
        self._strategy_weights: Dict[CognitiveStrategy, float] = {
            CognitiveStrategy.ANALYTIC: 0.25,
            CognitiveStrategy.INTUITIVE: 0.15,
            CognitiveStrategy.COMPARATIVE: 0.20,
            CognitiveStrategy.DIALECTICAL: 0.15,
            CognitiveStrategy.NARRATIVE: 0.10,
            CognitiveStrategy.EMBODIED: 0.05,
            CognitiveStrategy.PROBABILISTIC: 0.10,
        }

        # Configuration
        self._max_history: int = 100
        self._depth_time_multiplier: Dict[ReflectionDepth, float] = {
            ReflectionDepth.SURFACE: 0.2,
            ReflectionDepth.SHALLOW: 0.4,
            ReflectionDepth.MODERATE: 1.0,
            ReflectionDepth.DEEP: 2.0,
            ReflectionDepth.PROFOUND: 4.0,
        }

        self._initialized = False
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the reflective consciousness interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def reflect_on_state(
        self, reflective_input: ReflectiveInput
    ) -> ReflectiveOutput:
        """
        Perform an introspective reflection on a mental state.

        Examines the given thought content, assessing its nature, origins,
        and relationship to other mental states. Produces an insight about
        the internal state.

        Args:
            reflective_input: The thought and context for reflection.

        Returns:
            ReflectiveOutput with introspective insight.
        """
        self._state = ReflectiveState.REFLECTING

        self._reflection_counter += 1
        reflection_id = f"refl_{self._reflection_counter:06d}"

        # Select cognitive strategy
        strategy = self._select_strategy(
            reflective_input.reflection_type,
            reflective_input.context.depth_requested,
        )

        # Compute achievable depth
        depth = self._compute_achievable_depth(reflective_input)

        # Generate introspective insight
        insight = self._generate_introspective_insight(
            reflective_input.thought, depth, reflection_id
        )

        output = ReflectiveOutput(
            reflection_id=reflection_id,
            reflection_type=ReflectionType.INTROSPECTIVE,
            depth_achieved=depth,
            strategy_used=strategy,
            insight=insight,
            evaluation_result=None,
            decision=None,
            confidence=insight.confidence if insight else 0.3,
        )

        self._record_reflection(output)
        self._state = ReflectiveState.INTEGRATING
        return output

    async def evaluate_thought(
        self, reflective_input: ReflectiveInput
    ) -> ReflectiveOutput:
        """
        Evaluate a thought for correctness, coherence, or quality.

        Applies evaluative criteria to the given thought content,
        producing a judgment and supporting reasons.

        Args:
            reflective_input: The thought and context for evaluation.

        Returns:
            ReflectiveOutput with evaluation result.
        """
        self._state = ReflectiveState.REFLECTING

        self._reflection_counter += 1
        reflection_id = f"refl_{self._reflection_counter:06d}"

        strategy = self._select_strategy(
            ReflectionType.EVALUATIVE,
            reflective_input.context.depth_requested,
        )
        depth = self._compute_achievable_depth(reflective_input)

        # Evaluate the thought
        evaluation = self._perform_evaluation(reflective_input.thought, depth)

        # Generate evaluative insight
        insight = self._generate_evaluative_insight(
            reflective_input.thought, evaluation, reflection_id
        )

        output = ReflectiveOutput(
            reflection_id=reflection_id,
            reflection_type=ReflectionType.EVALUATIVE,
            depth_achieved=depth,
            strategy_used=strategy,
            insight=insight,
            evaluation_result=evaluation,
            decision=None,
            confidence=insight.confidence if insight else 0.4,
        )

        self._record_reflection(output)
        self._state = ReflectiveState.INTEGRATING
        return output

    async def generate_insight(
        self, reflective_input: ReflectiveInput
    ) -> Optional[ReflectiveInsight]:
        """
        Attempt to generate a novel insight from reflective processing.

        Combines the target thought with related thoughts and prior
        insights to produce new understanding.

        Args:
            reflective_input: The thought and related context.

        Returns:
            A ReflectiveInsight if one is produced, otherwise None.
        """
        self._state = ReflectiveState.REFLECTING

        depth = self._compute_achievable_depth(reflective_input)
        thought = reflective_input.thought

        # Deeper reflection has higher chance of significant insight
        quality = self._determine_insight_quality(depth, thought)

        if quality == InsightQuality.TRIVIAL and depth.value in ["surface", "shallow"]:
            self._state = ReflectiveState.IDLE
            return None

        self._insight_counter += 1
        insight_id = f"insight_{self._insight_counter:06d}"

        # Build insight content from thought and related material
        content = self._synthesize_insight_content(
            thought, reflective_input.related_thoughts, quality
        )

        insight = ReflectiveInsight(
            insight_id=insight_id,
            content=content,
            quality=quality,
            confidence=self._quality_to_confidence(quality),
            source_reflection=f"refl_{self._reflection_counter + 1:06d}",
            supporting_reasons=[
                f"Based on reflection on: {thought.description[:80]}",
                f"Reflection depth: {depth.value}",
            ],
        )

        self._insight_store[insight_id] = insight
        self._state = ReflectiveState.INTEGRATING
        return insight

    async def deliberate(
        self, options: List[DeliberationOption], context: ReflectiveContext
    ) -> ReflectiveOutput:
        """
        Deliberate over a set of options and reach a decision.

        Weighs pros, cons, values, feasibility, and risk for each option,
        then selects the best course of action.

        Args:
            options: Options to deliberate over.
            context: Context including goal and constraints.

        Returns:
            ReflectiveOutput with decision and reasoning.
        """
        self._state = ReflectiveState.REFLECTING

        self._reflection_counter += 1
        reflection_id = f"refl_{self._reflection_counter:06d}"

        strategy = self._select_strategy(
            ReflectionType.DELIBERATIVE,
            context.depth_requested,
        )

        # Score each option
        scored = self._score_options(options, context)

        # Select best
        best_option = max(scored, key=lambda x: x[1]) if scored else (None, 0.0)
        decision = (
            f"Selected: {best_option[0].description} (score: {best_option[1]:.2f})"
            if best_option[0] else "No viable option found"
        )

        # Generate deliberative insight
        insight = None
        if best_option[0]:
            self._insight_counter += 1
            insight = ReflectiveInsight(
                insight_id=f"insight_{self._insight_counter:06d}",
                content=f"Deliberation favors: {best_option[0].description}",
                quality=InsightQuality.INCREMENTAL,
                confidence=min(1.0, best_option[1]),
                source_reflection=reflection_id,
                supporting_reasons=[
                    f"Pros: {', '.join(best_option[0].pros[:3])}",
                    f"Value: {best_option[0].estimated_value:.2f}",
                ],
            )
            self._insight_store[insight.insight_id] = insight

        output = ReflectiveOutput(
            reflection_id=reflection_id,
            reflection_type=ReflectionType.DELIBERATIVE,
            depth_achieved=context.depth_requested,
            strategy_used=strategy,
            insight=insight,
            evaluation_result=None,
            decision=decision,
            options_considered=options,
            confidence=best_option[1] if best_option[0] else 0.2,
        )

        self._record_reflection(output)
        self._state = ReflectiveState.INTEGRATING
        return output

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "state": self._state.value,
            "reflection_count": self._reflection_counter,
            "insight_count": self._insight_counter,
            "history_length": len(self._reflection_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current operational status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "reflective_state": self._state.value,
            "reflections_performed": self._reflection_counter,
            "insights_generated": self._insight_counter,
            "stored_insights": len(self._insight_store),
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _select_strategy(
        self, reflection_type: ReflectionType, depth: ReflectionDepth
    ) -> CognitiveStrategy:
        """Select the most appropriate cognitive strategy."""
        type_preferences = {
            ReflectionType.INTROSPECTIVE: CognitiveStrategy.INTUITIVE,
            ReflectionType.DELIBERATIVE: CognitiveStrategy.COMPARATIVE,
            ReflectionType.EVALUATIVE: CognitiveStrategy.ANALYTIC,
            ReflectionType.METACOGNITIVE: CognitiveStrategy.DIALECTICAL,
            ReflectionType.COUNTERFACTUAL: CognitiveStrategy.NARRATIVE,
            ReflectionType.PROSPECTIVE: CognitiveStrategy.PROBABILISTIC,
        }
        return type_preferences.get(reflection_type, CognitiveStrategy.ANALYTIC)

    def _compute_achievable_depth(self, reflective_input: ReflectiveInput) -> ReflectionDepth:
        """Compute achievable depth based on input quality and resources."""
        requested = reflective_input.context.depth_requested
        thought_confidence = reflective_input.thought.confidence
        related_count = len(reflective_input.related_thoughts)

        depth_order = [
            ReflectionDepth.SURFACE,
            ReflectionDepth.SHALLOW,
            ReflectionDepth.MODERATE,
            ReflectionDepth.DEEP,
            ReflectionDepth.PROFOUND,
        ]
        requested_idx = depth_order.index(requested)

        # Higher confidence and more related thoughts enable deeper reflection
        capacity_score = thought_confidence * 0.5 + min(1.0, related_count / 5) * 0.5
        max_idx = min(len(depth_order) - 1, int(capacity_score * (len(depth_order) - 1)))
        achievable_idx = min(requested_idx, max(0, max_idx))

        return depth_order[achievable_idx]

    def _generate_introspective_insight(
        self, thought: ThoughtContent, depth: ReflectionDepth, reflection_id: str
    ) -> ReflectiveInsight:
        """Generate an introspective insight from a thought."""
        self._insight_counter += 1
        quality = self._determine_insight_quality(depth, thought)

        content = (
            f"Introspective analysis of {thought.content_type} "
            f"'{thought.description[:60]}': "
            f"confidence={thought.confidence:.2f}, "
            f"emotional_charge={thought.emotional_charge:.2f}"
        )

        return ReflectiveInsight(
            insight_id=f"insight_{self._insight_counter:06d}",
            content=content,
            quality=quality,
            confidence=thought.confidence * 0.8,
            source_reflection=reflection_id,
            supporting_reasons=[
                f"Thought type: {thought.content_type}",
                f"Depth achieved: {depth.value}",
            ],
        )

    def _perform_evaluation(
        self, thought: ThoughtContent, depth: ReflectionDepth
    ) -> str:
        """Perform an evaluation of thought quality."""
        score = thought.confidence

        if depth in [ReflectionDepth.DEEP, ReflectionDepth.PROFOUND]:
            # Deeper evaluation is more nuanced
            if thought.emotional_charge > 0.5:
                score *= 0.8  # Emotional bias penalty
            if thought.source:
                score *= 1.1  # Source attribution bonus

        score = max(0.0, min(1.0, score))

        if score > 0.8:
            return "high_quality"
        elif score > 0.5:
            return "adequate"
        elif score > 0.3:
            return "questionable"
        else:
            return "unreliable"

    def _generate_evaluative_insight(
        self, thought: ThoughtContent, evaluation: str, reflection_id: str
    ) -> ReflectiveInsight:
        """Generate an insight from an evaluation."""
        self._insight_counter += 1

        content = (
            f"Evaluation of '{thought.description[:60]}': {evaluation}. "
            f"Confidence in thought: {thought.confidence:.2f}"
        )

        quality_map = {
            "high_quality": InsightQuality.INCREMENTAL,
            "adequate": InsightQuality.INCREMENTAL,
            "questionable": InsightQuality.SIGNIFICANT,
            "unreliable": InsightQuality.SIGNIFICANT,
        }

        return ReflectiveInsight(
            insight_id=f"insight_{self._insight_counter:06d}",
            content=content,
            quality=quality_map.get(evaluation, InsightQuality.TRIVIAL),
            confidence=0.6,
            source_reflection=reflection_id,
            supporting_reasons=[f"Evaluation result: {evaluation}"],
        )

    def _determine_insight_quality(
        self, depth: ReflectionDepth, thought: ThoughtContent
    ) -> InsightQuality:
        """Determine the quality of an insight based on depth and content."""
        depth_quality_map = {
            ReflectionDepth.SURFACE: InsightQuality.TRIVIAL,
            ReflectionDepth.SHALLOW: InsightQuality.INCREMENTAL,
            ReflectionDepth.MODERATE: InsightQuality.INCREMENTAL,
            ReflectionDepth.DEEP: InsightQuality.SIGNIFICANT,
            ReflectionDepth.PROFOUND: InsightQuality.BREAKTHROUGH,
        }
        base = depth_quality_map.get(depth, InsightQuality.TRIVIAL)

        # High emotional charge can boost insight quality
        if abs(thought.emotional_charge) > 0.7:
            qualities = list(InsightQuality)
            idx = qualities.index(base)
            base = qualities[min(idx + 1, len(qualities) - 1)]

        return base

    def _quality_to_confidence(self, quality: InsightQuality) -> float:
        """Map insight quality to confidence."""
        return {
            InsightQuality.TRIVIAL: 0.3,
            InsightQuality.INCREMENTAL: 0.5,
            InsightQuality.SIGNIFICANT: 0.7,
            InsightQuality.BREAKTHROUGH: 0.85,
            InsightQuality.TRANSFORMATIVE: 0.95,
        }.get(quality, 0.5)

    def _synthesize_insight_content(
        self,
        thought: ThoughtContent,
        related: List[ThoughtContent],
        quality: InsightQuality,
    ) -> str:
        """Synthesize insight content from thought and related material."""
        base = f"Reflection on '{thought.description[:50]}'"
        if related:
            base += f" with {len(related)} related thought(s)"
        base += f" yields {quality.value} insight"
        return base

    def _score_options(
        self,
        options: List[DeliberationOption],
        context: ReflectiveContext,
    ) -> List[Tuple[DeliberationOption, float]]:
        """Score deliberation options."""
        scored = []
        for opt in options:
            pros_score = len(opt.pros) * 0.1
            cons_score = len(opt.cons) * 0.1
            value_score = opt.estimated_value * 0.4
            feasibility_score = opt.feasibility * 0.3
            risk_penalty = opt.risk * 0.2

            total = value_score + feasibility_score + pros_score - cons_score - risk_penalty
            total = max(0.0, min(1.0, total))
            scored.append((opt, total))

        return scored

    def _record_reflection(self, output: ReflectiveOutput) -> None:
        """Record a reflection in history."""
        self._reflection_history.append(output)
        if len(self._reflection_history) > self._max_history:
            self._reflection_history.pop(0)
        self._current_depth = output.depth_achieved


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_reflective_consciousness_interface() -> ReflectiveConsciousnessInterface:
    """Create and return a reflective consciousness interface instance."""
    return ReflectiveConsciousnessInterface()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ReflectionType",
    "ReflectionDepth",
    "CognitiveStrategy",
    "InsightQuality",
    "ReflectiveState",
    # Input dataclasses
    "ThoughtContent",
    "ReflectiveContext",
    "ReflectiveInput",
    # Output dataclasses
    "ReflectiveInsight",
    "DeliberationOption",
    "ReflectiveOutput",
    # Interface
    "ReflectiveConsciousnessInterface",
    # Convenience
    "create_reflective_consciousness_interface",
]
