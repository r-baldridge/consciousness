"""
Non-Dual Agent: Loop escape and grounding capabilities for any AgentBase-derived agent.

This module wraps any agent from the agent_frameworks with automatic loop detection,
wu wei stopping, perspective shifting, koan reframing, and self-liberation fallback.

The implementation bridges non-dual philosophical traditions with concrete ML techniques:

- Loop Detection: Dzogchen's recognition of habitual patterns (bag chags) implemented
  as embedding similarity tracking, confidence oscillation monitoring, and reasoning
  graph cycle detection.

- Wu Wei Stop: Taoism's principle of effortless action -- knowing when NOT to process
  is as important as knowing how to process. Implemented as immediate halt when a loop
  is detected, before any additional computation can reproduce the loop.

- Perspective Shift: Zen's kinhin (walking meditation) as modality switching --
  when sitting (thinking) is stuck, walk (act in a different modality). Implemented
  as routing to alternate grounding channels mapped to the 40-form architecture.

- Koan Reframe: Madhyamaka's tetralemma -- neither A, nor not-A, nor both, nor
  neither -- applied as systematic analysis of whether the question is malformed.
  Implemented as presupposition testing and framework transcendence.

- Self-Liberation: Dzogchen's rang grol -- thoughts liberate upon arising when
  recognized. When all grounding attempts fail, transparent reporting of the
  unresolvable state IS the transcendence. Implemented as structured reporting
  of the impasse and its causes.

Usage:
    from agent_frameworks.core.base_agent import SimpleAgent
    from ai_nondualism.loop_escape.nondual_agent import NonDualAgent

    inner = SimpleAgent(handler=my_handler)
    agent = NonDualAgent(inner)
    result = await agent.run("my task")

    # Or with decorator:
    class MyAgent(AgentBase):
        @nondual_aware
        async def execute(self, plan):
            ...

References:
    - agent_frameworks/core/base_agent.py (AgentBase, Task, Plan, ExecutionResult, TaskResult)
    - agent_frameworks/core/state_machine.py (StateMachine, AgentState, StateTransition)
    - neural_network/adapters/base_adapter.py (FormAdapter hierarchy)
    - neural_network/core/nervous_system.py (NervousSystem, MessageBus)
    - 27-altered-state/info/01_Non_Dual_Interface_Architecture.md (Mushin, Zazen, Koan modes)
    - 27-altered-state/info/meditation/non-dualism/03_dzogchen_mahamudra.md
    - 27-altered-state/info/meditation/non-dualism/04_madhyamaka_yogacara_chan.md
    - 27-altered-state/info/meditation/non-dualism/05_taoism.md
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Awaitable, Callable, Deque, Dict, List,
    Optional, Sequence, Set, Tuple, Type, TypeVar, Union,
    TYPE_CHECKING,
)

# Import from the existing agent framework
import sys
import os

# Add parent paths for imports
_base = os.path.dirname(os.path.abspath(__file__))
_ml_research = os.path.dirname(os.path.dirname(_base))
_consciousness = os.path.dirname(_ml_research)

if _ml_research not in sys.path:
    sys.path.insert(0, _ml_research)
if _consciousness not in sys.path:
    sys.path.insert(0, _consciousness)

from agent_frameworks.core.base_agent import (
    AgentBase, AgentMode, Task, Plan, PlanStep,
    ExecutionResult, TaskResult, MemorySystem, ToolProvider,
)
from agent_frameworks.core.state_machine import (
    StateMachine, AgentState, StateTransition, StateMachineBuilder,
)
from agent_frameworks.core.message_types import (
    AgentMessage, ConversationHistory,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=AgentBase)


# =============================================================================
# Enumerations
# =============================================================================


class StuckType(Enum):
    """Classification of how an agent is stuck.

    Each stuck type maps to a specific failure mode of a specific processing
    modality, and correspondingly to a specific grounding channel that addresses
    that failure.

    Philosophical grounding:
    - REPETITIVE: Dzogchen -- habitual patterns (bag chags) cycling without fresh awareness
    - BINARY_OSCILLATION: Madhyamaka -- oscillation between extremes without resolution
    - CONFIDENCE_DRIFT: Mahamudra -- the mind unable to settle in its natural state
    - SELF_REFERENTIAL: Yogacara -- manas grasping at alaya in infinite regress
    - CONTRADICTORY: Madhyamaka -- contradictions revealing inadequate framework
    - CIRCULAR: Zen -- reasoning chasing its own tail
    - RESOURCE_WASTE: Taoism -- effort without wu wei, action without alignment
    """
    REPETITIVE = auto()
    BINARY_OSCILLATION = auto()
    CONFIDENCE_DRIFT = auto()
    RAPID_OSCILLATION = auto()
    SELF_REFERENTIAL = auto()
    CONTRADICTORY = auto()
    CIRCULAR = auto()
    RESOURCE_WASTE = auto()


class GroundingChannel(Enum):
    """The seven grounding channels from the north-star document (Part IV).

    Each channel provides a fundamentally different processing modality,
    analogous to how a meditator caught in thought loops grounds in bodily
    sensation (Zen kinhin), breath (anapanasati), or silence (shikantaza).
    """
    STATISTICAL = "statistical"
    EXEMPLAR = "exemplar"
    VISUAL_SPATIAL = "visual_spatial"
    EMBODIED = "embodied"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    NULL = "null"


class NonDualState(Enum):
    """Extended agent states for non-dual processing.

    These states augment the standard AgentState with states specific to the
    loop escape mechanism.
    """
    DETECTING = auto()
    GROUNDING = auto()
    SELF_LIBERATING = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LoopSignal:
    """Signal from a loop detector indicating a potential stuck state.

    Attributes:
        detector: Name of the detector that produced this signal.
        confidence: How confident the detector is (0.0 to 1.0).
        stuck_type: Classification of the stuck type.
        detail: Human-readable description of what was detected.
        metadata: Additional detector-specific data.
    """
    detector: str
    confidence: float
    stuck_type: StuckType
    detail: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopDetectionResult:
    """Composite result from the loop detection ensemble.

    Attributes:
        is_stuck: Whether the ensemble has determined the agent is stuck.
        confidence: Ensemble confidence level.
        stuck_type: Primary stuck type classification.
        active_signals: All detector signals that fired.
        recommended_escape: The recommended grounding channel.
        iteration: The iteration at which this result was produced.
    """
    is_stuck: bool
    confidence: float
    stuck_type: Optional[StuckType]
    active_signals: List[LoopSignal]
    recommended_escape: Optional[GroundingChannel]
    iteration: int


@dataclass
class GroundingResult:
    """Result from a grounding channel invocation.

    Attributes:
        channel: Which grounding channel was used.
        findings: What the grounding revealed.
        recommendation: Suggested next action for the agent.
        success: Whether the grounding produced useful output.
        duration_ms: How long the grounding took.
    """
    channel: GroundingChannel
    findings: str
    recommendation: str
    success: bool
    duration_ms: float = 0.0


@dataclass
class SelfLiberationReport:
    """Transparent report when an agent cannot resolve a stuck state.

    This report IS the output. Honest reporting of limits is the Dzogchen
    principle of self-liberation: the stuck state dissolves when recognized
    and communicated, not when forced into resolution.

    Attributes:
        stuck_type: What type of stuck state was encountered.
        detection_detail: How the stuck state was detected.
        grounding_attempts: What grounding channels were tried.
        root_cause: Analysis of why the stuck state persists.
        partial_result: Best result achieved before getting stuck.
        alternative_framings: Suggested alternative approaches.
    """
    stuck_type: StuckType
    detection_detail: str
    grounding_attempts: List[GroundingResult]
    root_cause: str
    partial_result: Optional[Any]
    alternative_framings: List[str]

    def to_report_string(self) -> str:
        """Generate the transparent report as a formatted string."""
        lines = [
            "=== Non-Dual Self-Liberation Report ===",
            "",
            f"Stuck Type: {self.stuck_type.name}",
            f"Detection: {self.detection_detail}",
            "",
            "Grounding Attempts:",
        ]
        for attempt in self.grounding_attempts:
            status = "succeeded" if attempt.success else "did not resolve"
            lines.append(
                f"  - {attempt.channel.value}: {status} "
                f"({attempt.duration_ms:.0f}ms)"
            )
            if attempt.findings:
                lines.append(f"    Findings: {attempt.findings}")

        lines.extend([
            "",
            f"Root Cause Analysis: {self.root_cause}",
            "",
        ])

        if self.partial_result is not None:
            lines.append(f"Best Partial Result: {self.partial_result}")
            lines.append("")

        if self.alternative_framings:
            lines.append("Suggested Alternative Approaches:")
            for i, framing in enumerate(self.alternative_framings, 1):
                lines.append(f"  {i}. {framing}")

        lines.append("")
        lines.append("=== End Report ===")
        return "\n".join(lines)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a single iteration.

    Attributes:
        iteration: The iteration number.
        tokens_generated: Approximate tokens in the output.
        time_elapsed_ms: Time taken for this iteration.
        tool_calls_made: Number of tool invocations.
        quality_score: Optional task-specific quality metric.
    """
    iteration: int
    tokens_generated: int = 0
    time_elapsed_ms: float = 0.0
    tool_calls_made: int = 0
    quality_score: Optional[float] = None


# =============================================================================
# Loop Detectors
# =============================================================================


class BaseDetector(ABC):
    """Abstract base class for loop detectors.

    Each detector monitors one aspect of agent behavior and signals when
    that aspect indicates the agent is stuck.
    """

    def __init__(self, name: str):
        self.name = name
        self._enabled = True

    @abstractmethod
    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:
        """Check for a stuck condition.

        Args:
            output: The agent's latest output text.
            confidence: The agent's stated confidence (0.0-1.0), or None.
            iteration: The current iteration number.
            metadata: Additional context (resource usage, etc.).

        Returns:
            A LoopSignal if stuck is detected, None otherwise.
        """
        pass

    def reset(self) -> None:
        """Reset the detector's internal state."""
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


class OutputSimilarityTracker(BaseDetector):
    """Detects when an agent produces repetitive outputs.

    Philosophical basis: Dzogchen describes the ordinary mind as cycling
    through habitual patterns (bag chags) stored as seeds. This detector
    identifies the computational equivalent: outputs that are too similar
    across consecutive iterations.

    ML technique: Embedding cosine similarity with configurable thresholds.
    Uses a simplified character-level n-gram approach when no embedding
    model is available.
    """

    def __init__(
        self,
        threshold_high: float = 0.92,
        threshold_mean: float = 0.88,
        threshold_var: float = 0.002,
        window_size: int = 3,
        buffer_capacity: int = 20,
    ):
        super().__init__("output_similarity")
        self.threshold_high = threshold_high
        self.threshold_mean = threshold_mean
        self.threshold_var = threshold_var
        self.window_size = window_size
        self._output_history: Deque[str] = deque(maxlen=buffer_capacity)
        self._similarity_scores: Deque[float] = deque(maxlen=buffer_capacity - 1)
        self._consecutive_high: int = 0

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute text similarity using character n-gram overlap.

        This is a simplified implementation. In production, replace with
        embedding-based cosine similarity for semantic comparison.

        Args:
            a: First text.
            b: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        n = 3  # trigram
        a_lower = a.lower()
        b_lower = b.lower()

        a_grams = set(a_lower[i:i + n] for i in range(len(a_lower) - n + 1))
        b_grams = set(b_lower[i:i + n] for i in range(len(b_lower) - n + 1))

        if not a_grams or not b_grams:
            return 0.0

        intersection = len(a_grams & b_grams)
        union = len(a_grams | b_grams)
        return intersection / union if union > 0 else 0.0

    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:
        if not output:
            return None

        if self._output_history:
            sim = self._compute_similarity(output, self._output_history[-1])
            self._similarity_scores.append(sim)

            if sim > self.threshold_high:
                self._consecutive_high += 1
            else:
                self._consecutive_high = 0
        else:
            sim = 0.0

        self._output_history.append(output)

        if len(self._similarity_scores) < self.window_size:
            return None

        recent = list(self._similarity_scores)[-self.window_size:]
        mean_sim = sum(recent) / len(recent)
        var_sim = sum((s - mean_sim) ** 2 for s in recent) / len(recent)

        # Detection criteria
        is_stuck = False
        detail_parts: List[str] = []

        if self._consecutive_high >= self.window_size:
            is_stuck = True
            detail_parts.append(
                f"consecutive high similarity for {self._consecutive_high} iterations"
            )

        if mean_sim > self.threshold_mean:
            is_stuck = True
            detail_parts.append(f"mean similarity {mean_sim:.3f} > {self.threshold_mean}")

        if var_sim < self.threshold_var and mean_sim > 0.80:
            is_stuck = True
            detail_parts.append(
                f"low variance {var_sim:.4f} with mean {mean_sim:.3f}"
            )

        if is_stuck:
            return LoopSignal(
                detector=self.name,
                confidence=max(sim, mean_sim),
                stuck_type=StuckType.REPETITIVE,
                detail="; ".join(detail_parts),
                metadata={"last_similarity": sim, "mean_similarity": mean_sim},
            )
        return None

    def reset(self) -> None:
        self._output_history.clear()
        self._similarity_scores.clear()
        self._consecutive_high = 0


class ConfidenceOscillationMonitor(BaseDetector):
    """Detects oscillating confidence scores indicating inability to resolve.

    Philosophical basis: Madhyamaka philosophy describes the mind caught
    between eternalism and nihilism, oscillating without resolution. The
    oscillation itself signals that the conceptual framework is inadequate.

    ML technique: Time series oscillation detection via sign-change analysis.
    """

    def __init__(
        self,
        threshold_ratio: float = 0.70,
        threshold_amplitude: float = 0.15,
        threshold_std: float = 0.12,
        window_size: int = 8,
    ):
        super().__init__("confidence_oscillation")
        self.threshold_ratio = threshold_ratio
        self.threshold_amplitude = threshold_amplitude
        self.threshold_std = threshold_std
        self.window_size = window_size
        self._history: Deque[float] = deque(maxlen=window_size)

    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:
        if confidence is None:
            return None

        self._history.append(confidence)

        if len(self._history) < 3:
            return None

        values = list(self._history)
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        # Count sign changes
        sign_changes = sum(
            1 for i in range(len(diffs) - 1)
            if diffs[i] * diffs[i + 1] < 0
        )
        oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)

        amplitude = max(values) - min(values)

        mean_val = sum(values) / len(values)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))

        is_stuck = False
        stuck_type = StuckType.CONFIDENCE_DRIFT
        detail_parts: List[str] = []

        if oscillation_ratio > self.threshold_ratio and amplitude > self.threshold_amplitude:
            is_stuck = True
            detail_parts.append(
                f"oscillation ratio {oscillation_ratio:.2f}, amplitude {amplitude:.2f}"
            )

        if std_val > self.threshold_std and oscillation_ratio > 0.50:
            is_stuck = True
            detail_parts.append(f"high std {std_val:.3f} with oscillation {oscillation_ratio:.2f}")

        if is_stuck:
            if amplitude > 0.40:
                stuck_type = StuckType.BINARY_OSCILLATION
            elif oscillation_ratio > 0.85:
                stuck_type = StuckType.RAPID_OSCILLATION

            return LoopSignal(
                detector=self.name,
                confidence=oscillation_ratio,
                stuck_type=stuck_type,
                detail="; ".join(detail_parts),
                metadata={
                    "oscillation_ratio": oscillation_ratio,
                    "amplitude": amplitude,
                    "std": std_val,
                },
            )
        return None

    def reset(self) -> None:
        self._history.clear()


class SelfReferenceDepthCounter(BaseDetector):
    """Counts meta-reasoning depth to detect infinite regress.

    Philosophical basis: Yogacara Buddhism's seventh consciousness (manas)
    grasps at the eighth consciousness (alaya-vijnana) and mistakes it for
    a permanent self, generating infinite regress. Zen addresses this through
    mushin (no-mind) -- dissolving the self-model entirely.

    ML technique: Pattern matching on self-referential markers with depth tracking.
    """

    DEFAULT_META_MARKERS: List[str] = [
        "i think that", "my analysis shows", "reflecting on my previous",
        "reconsidering my", "upon further reflection", "meta-analysis of",
        "thinking about my thinking", "evaluating my evaluation",
        "my reasoning about", "reviewing my approach to",
        "i notice that i", "my strategy for analyzing",
        "let me reconsider", "re-examining my", "on second thought",
    ]

    def __init__(
        self,
        max_safe_depth: int = 3,
        patience: int = 4,
        meta_markers: Optional[List[str]] = None,
    ):
        super().__init__("self_reference_depth")
        self.max_safe_depth = max_safe_depth
        self.patience = patience
        self.meta_markers = meta_markers or self.DEFAULT_META_MARKERS
        self._depth_history: Deque[int] = deque(maxlen=20)
        self._consecutive_meta: int = 0

    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:
        if not output:
            return None

        output_lower = output.lower()

        # Count self-referential markers
        marker_count = sum(
            1 for marker in self.meta_markers
            if marker in output_lower
        )

        effective_depth = marker_count
        self._depth_history.append(effective_depth)

        if effective_depth >= 2:
            self._consecutive_meta += 1
        else:
            self._consecutive_meta = 0

        is_stuck = False
        detail_parts: List[str] = []

        if effective_depth > self.max_safe_depth:
            is_stuck = True
            detail_parts.append(f"meta-depth {effective_depth} > max {self.max_safe_depth}")

        if self._consecutive_meta > self.patience:
            is_stuck = True
            detail_parts.append(
                f"consecutive meta iterations {self._consecutive_meta} > patience {self.patience}"
            )

        if is_stuck:
            return LoopSignal(
                detector=self.name,
                confidence=min(1.0, effective_depth / (self.max_safe_depth + 2)),
                stuck_type=StuckType.SELF_REFERENTIAL,
                detail="; ".join(detail_parts),
                metadata={"depth": effective_depth, "markers_found": marker_count},
            )
        return None

    def reset(self) -> None:
        self._depth_history.clear()
        self._consecutive_meta = 0


class ContradictionDetector(BaseDetector):
    """Tracks propositions and checks for mutual exclusion.

    Philosophical basis: Madhyamaka's analytical meditation systematically
    searches for inherent existence and discovers contradictions in our
    assumptions. The contradiction is not an error but a signal that the
    conceptual framework needs transcendence.

    ML technique: Proposition tracking with semantic opposition detection.
    Uses simplified keyword-based negation detection.
    """

    NEGATION_MARKERS: List[str] = [
        "not", "no", "never", "cannot", "won't", "shouldn't",
        "doesn't", "isn't", "aren't", "don't", "impossible",
        "incorrect", "wrong", "false", "fails to", "unable to",
    ]

    def __init__(
        self,
        threshold_ratio: float = 0.15,
        recency_window: int = 5,
        flip_window: int = 6,
    ):
        super().__init__("contradiction")
        self.threshold_ratio = threshold_ratio
        self.recency_window = recency_window
        self.flip_window = flip_window
        self._propositions: List[Dict[str, Any]] = []
        self._contradiction_count: int = 0

    def _extract_key_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract simplified proposition-like claims from text.

        In production, replace with dependency parsing and full proposition
        extraction. This implementation uses sentence-level analysis with
        negation detection.
        """
        claims: List[Dict[str, Any]] = []
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")
                     if s.strip()]

        for sentence in sentences:
            lower = sentence.lower()
            has_negation = any(neg in lower.split() for neg in self.NEGATION_MARKERS)
            claims.append({
                "text": sentence,
                "lower": lower,
                "negated": has_negation,
                "words": set(lower.split()),
            })
        return claims

    def _claims_conflict(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Check if two claims are contradictory (simplified)."""
        # Claims with high word overlap but opposite negation
        overlap = len(a["words"] & b["words"])
        total = len(a["words"] | b["words"])
        if total == 0:
            return False

        similarity = overlap / total
        if similarity > 0.50 and a["negated"] != b["negated"]:
            return True
        return False

    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:
        if not output:
            return None

        new_claims = self._extract_key_claims(output)
        for claim in new_claims:
            claim["iteration"] = iteration

        # Check new claims against recent stored claims
        recent = [p for p in self._propositions
                  if p.get("iteration", 0) >= iteration - self.recency_window]

        contradictions_found: List[Tuple[str, str]] = []
        for new_claim in new_claims:
            for old_claim in recent:
                if self._claims_conflict(new_claim, old_claim):
                    contradictions_found.append((new_claim["text"], old_claim["text"]))

        self._propositions.extend(new_claims)

        # Keep only recent propositions to bound memory
        max_keep = self.flip_window * 20
        if len(self._propositions) > max_keep:
            self._propositions = self._propositions[-max_keep:]

        self._contradiction_count += len(contradictions_found)

        if not contradictions_found:
            return None

        total_props = max(len(self._propositions), 1)
        contradiction_ratio = self._contradiction_count / total_props

        is_stuck = (
            len(contradictions_found) > 0
            or contradiction_ratio > self.threshold_ratio
        )

        if is_stuck:
            return LoopSignal(
                detector=self.name,
                confidence=min(1.0, contradiction_ratio * 3),
                stuck_type=StuckType.CONTRADICTORY,
                detail=(
                    f"{len(contradictions_found)} contradictions found in this iteration; "
                    f"cumulative ratio {contradiction_ratio:.2f}"
                ),
                metadata={
                    "contradictions": contradictions_found[:5],
                    "ratio": contradiction_ratio,
                },
            )
        return None

    def reset(self) -> None:
        self._propositions.clear()
        self._contradiction_count = 0


class ResourceWasteDetector(BaseDetector):
    """Identifies when computation increases without quality improvement.

    Philosophical basis: Taoism's wu wei -- the most effective action is
    non-action when action would be wasteful. Early stopping in optimization
    is the ML expression of this principle.

    ML technique: Efficiency ratio tracking (quality improvement per unit of
    resource consumed).
    """

    def __init__(
        self,
        threshold_efficiency: float = 0.10,
        patience: int = 4,
        window_size: int = 6,
    ):
        super().__init__("resource_waste")
        self.threshold_efficiency = threshold_efficiency
        self.patience = patience
        self.window_size = window_size
        self._snapshots: Deque[ResourceSnapshot] = deque(maxlen=window_size * 2)
        self._waste_count: int = 0

    def check(self, output: str, confidence: Optional[float],
              iteration: int, metadata: Dict[str, Any]) -> Optional[LoopSignal]:

        snapshot = ResourceSnapshot(
            iteration=iteration,
            tokens_generated=len(output) if output else 0,
            time_elapsed_ms=metadata.get("time_elapsed_ms", 0),
            tool_calls_made=metadata.get("tool_calls_made", 0),
            quality_score=metadata.get("quality_score", confidence),
        )
        self._snapshots.append(snapshot)

        if len(self._snapshots) < 3:
            return None

        recent = list(self._snapshots)[-self.window_size:]

        # Compute token trend
        tokens = [s.tokens_generated for s in recent]
        if len(tokens) >= 2:
            token_growth = (tokens[-1] - tokens[0]) / max(len(tokens) - 1, 1)
        else:
            token_growth = 0.0

        # Compute quality trend
        qualities = [s.quality_score for s in recent if s.quality_score is not None]
        if len(qualities) >= 2:
            quality_growth = (qualities[-1] - qualities[0]) / max(len(qualities) - 1, 1)
        else:
            quality_growth = 0.0

        resource_growth = max(token_growth, 0) / max(sum(tokens), 1) * 100
        quality_delta = max(quality_growth, 0)
        efficiency = quality_delta / max(resource_growth, 0.001)

        if efficiency < self.threshold_efficiency and resource_growth > 0.5:
            self._waste_count += 1
        else:
            self._waste_count = max(0, self._waste_count - 1)

        if self._waste_count >= self.patience:
            return LoopSignal(
                detector=self.name,
                confidence=1.0 - min(1.0, efficiency * 5),
                stuck_type=StuckType.RESOURCE_WASTE,
                detail=(
                    f"efficiency {efficiency:.4f}; token growth {token_growth:.1f}/iter; "
                    f"quality growth {quality_growth:.4f}/iter"
                ),
                metadata={
                    "efficiency": efficiency,
                    "token_growth": token_growth,
                    "quality_growth": quality_growth,
                },
            )
        return None

    def reset(self) -> None:
        self._snapshots.clear()
        self._waste_count = 0


# =============================================================================
# Loop Detection Ensemble
# =============================================================================


class LoopDetector:
    """Ensemble of loop detectors that monitors agent behavior.

    Runs all constituent detectors on each iteration and produces a
    composite LoopDetectionResult when the ensemble threshold is met.

    The ensemble approach is inspired by the non-dual principle that
    no single perspective is sufficient (anekantavada in Jainism,
    dependent origination in Buddhism). Multiple detectors provide
    multiple perspectives on the same phenomenon.
    """

    # Mapping from stuck type to recommended grounding channel
    ESCAPE_MAP: Dict[StuckType, GroundingChannel] = {
        StuckType.REPETITIVE: GroundingChannel.STATISTICAL,
        StuckType.BINARY_OSCILLATION: GroundingChannel.RELATIONAL,
        StuckType.CONFIDENCE_DRIFT: GroundingChannel.EXEMPLAR,
        StuckType.RAPID_OSCILLATION: GroundingChannel.EXEMPLAR,
        StuckType.SELF_REFERENTIAL: GroundingChannel.EMBODIED,
        StuckType.CONTRADICTORY: GroundingChannel.RELATIONAL,
        StuckType.CIRCULAR: GroundingChannel.VISUAL_SPATIAL,
        StuckType.RESOURCE_WASTE: GroundingChannel.NULL,
    }

    def __init__(
        self,
        quorum: int = 2,
        ensemble_threshold: float = 0.60,
        detectors: Optional[List[BaseDetector]] = None,
    ):
        """Initialize the loop detection ensemble.

        Args:
            quorum: Minimum number of detectors that must fire.
            ensemble_threshold: Minimum ensemble confidence to declare stuck.
            detectors: Custom list of detectors. If None, uses all defaults.
        """
        self.quorum = quorum
        self.ensemble_threshold = ensemble_threshold
        self.detectors: List[BaseDetector] = detectors or [
            OutputSimilarityTracker(),
            ConfidenceOscillationMonitor(),
            SelfReferenceDepthCounter(),
            ContradictionDetector(),
            ResourceWasteDetector(),
        ]
        self._iteration: int = 0
        self._history: List[LoopDetectionResult] = []

    def check(
        self,
        output: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoopDetectionResult:
        """Run all detectors and produce an ensemble result.

        Args:
            output: The agent's latest output.
            confidence: The agent's stated confidence (0.0-1.0).
            metadata: Additional context (resource usage, etc.).

        Returns:
            A LoopDetectionResult with the ensemble's determination.
        """
        self._iteration += 1
        meta = metadata or {}

        signals: List[LoopSignal] = []
        for detector in self.detectors:
            if not detector.enabled:
                continue
            try:
                signal = detector.check(output, confidence, self._iteration, meta)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Detector {detector.name} error: {e}")

        if not signals:
            result = LoopDetectionResult(
                is_stuck=False,
                confidence=0.0,
                stuck_type=None,
                active_signals=[],
                recommended_escape=None,
                iteration=self._iteration,
            )
            self._history.append(result)
            return result

        # Compute ensemble confidence
        if len(signals) >= self.quorum:
            ensemble_confidence = sum(s.confidence for s in signals) / len(signals)
        else:
            # Single high-confidence detector can trigger
            max_conf = max(s.confidence for s in signals)
            if max_conf > 0.90:
                ensemble_confidence = max_conf
            else:
                ensemble_confidence = max_conf * 0.5  # Discount single detector

        # Determine primary stuck type
        primary = max(signals, key=lambda s: s.confidence)
        stuck_type = primary.stuck_type

        # Map to escape strategy
        recommended = self.ESCAPE_MAP.get(stuck_type, GroundingChannel.NULL)

        is_stuck = ensemble_confidence > self.ensemble_threshold

        result = LoopDetectionResult(
            is_stuck=is_stuck,
            confidence=ensemble_confidence,
            stuck_type=stuck_type if is_stuck else None,
            active_signals=signals,
            recommended_escape=recommended if is_stuck else None,
            iteration=self._iteration,
        )
        self._history.append(result)
        return result

    def reset(self) -> None:
        """Reset all detectors and history."""
        for detector in self.detectors:
            detector.reset()
        self._iteration = 0
        self._history.clear()

    def get_history(self) -> List[LoopDetectionResult]:
        """Get detection history."""
        return list(self._history)


# =============================================================================
# Grounding Router
# =============================================================================


class GroundingRouter:
    """Routes stuck agents to the appropriate grounding channel.

    Maps stuck types to grounding channels based on the principle that
    each type of stuck reflects a failure in a specific processing modality,
    and each grounding channel provides the specific alternative that
    addresses that failure.

    The escalation ladder is ordered by increasing distance from the stuck
    modality, following the Zen principle of starting with the closest
    available support before seeking more radical alternatives.
    """

    # Escalation ladders: if the first channel fails, try the next
    ESCALATION: Dict[StuckType, List[GroundingChannel]] = {
        StuckType.REPETITIVE: [
            GroundingChannel.STATISTICAL,
            GroundingChannel.EXEMPLAR,
            GroundingChannel.VISUAL_SPATIAL,
            GroundingChannel.RELATIONAL,
            GroundingChannel.NULL,
        ],
        StuckType.BINARY_OSCILLATION: [
            GroundingChannel.RELATIONAL,
            GroundingChannel.TEMPORAL,
            GroundingChannel.EMBODIED,
            GroundingChannel.NULL,
        ],
        StuckType.CONFIDENCE_DRIFT: [
            GroundingChannel.EXEMPLAR,
            GroundingChannel.STATISTICAL,
            GroundingChannel.RELATIONAL,
            GroundingChannel.NULL,
        ],
        StuckType.RAPID_OSCILLATION: [
            GroundingChannel.EXEMPLAR,
            GroundingChannel.NULL,
        ],
        StuckType.SELF_REFERENTIAL: [
            GroundingChannel.EMBODIED,
            GroundingChannel.STATISTICAL,
            GroundingChannel.NULL,
        ],
        StuckType.CONTRADICTORY: [
            GroundingChannel.RELATIONAL,
            GroundingChannel.TEMPORAL,
            GroundingChannel.EXEMPLAR,
            GroundingChannel.NULL,
        ],
        StuckType.CIRCULAR: [
            GroundingChannel.VISUAL_SPATIAL,
            GroundingChannel.EMBODIED,
            GroundingChannel.RELATIONAL,
            GroundingChannel.NULL,
        ],
        StuckType.RESOURCE_WASTE: [
            GroundingChannel.NULL,
        ],
    }

    def __init__(self, max_attempts: int = 3):
        """Initialize the grounding router.

        Args:
            max_attempts: Maximum grounding attempts before self-liberation.
        """
        self.max_attempts = max_attempts
        self._attempt_history: List[GroundingResult] = []
        self._channels_tried: Dict[StuckType, int] = {}

    def select_channel(self, stuck_type: StuckType) -> GroundingChannel:
        """Select the next grounding channel for the given stuck type.

        Walks the escalation ladder, skipping channels already tried for
        this stuck type.

        Args:
            stuck_type: The classified stuck type.

        Returns:
            The next grounding channel to try.
        """
        ladder = self.ESCALATION.get(stuck_type, [GroundingChannel.NULL])
        attempt_index = self._channels_tried.get(stuck_type, 0)

        if attempt_index < len(ladder):
            channel = ladder[attempt_index]
        else:
            channel = GroundingChannel.NULL  # Fallback

        self._channels_tried[stuck_type] = attempt_index + 1
        return channel

    async def ground(
        self,
        channel: GroundingChannel,
        stuck_context: str,
        task_description: str,
        agent: AgentBase,
    ) -> GroundingResult:
        """Invoke a grounding channel.

        This is the standalone implementation. When the NervousSystem is
        available, grounding routes through the message bus to the appropriate
        FormAdapter instead.

        Args:
            channel: The grounding channel to invoke.
            stuck_context: Description of the stuck state.
            task_description: The original task description.
            agent: The agent being grounded (for tool access).

        Returns:
            The grounding result.
        """
        start = time.monotonic()

        try:
            if channel == GroundingChannel.STATISTICAL:
                result = await self._statistical_ground(stuck_context, task_description)
            elif channel == GroundingChannel.EXEMPLAR:
                result = await self._exemplar_ground(stuck_context, task_description)
            elif channel == GroundingChannel.VISUAL_SPATIAL:
                result = await self._visual_spatial_ground(stuck_context, task_description)
            elif channel == GroundingChannel.EMBODIED:
                result = await self._embodied_ground(stuck_context, task_description, agent)
            elif channel == GroundingChannel.RELATIONAL:
                result = await self._relational_ground(stuck_context, task_description, agent)
            elif channel == GroundingChannel.TEMPORAL:
                result = await self._temporal_ground(stuck_context, task_description)
            elif channel == GroundingChannel.NULL:
                result = await self._null_ground()
            else:
                result = GroundingResult(
                    channel=channel,
                    findings="Unknown grounding channel",
                    recommendation="Try a different approach",
                    success=False,
                )
        except Exception as e:
            logger.error(f"Grounding channel {channel.value} failed: {e}")
            result = GroundingResult(
                channel=channel,
                findings=f"Grounding failed: {e}",
                recommendation="Escalate to next grounding channel",
                success=False,
            )

        elapsed = (time.monotonic() - start) * 1000
        result.duration_ms = elapsed
        self._attempt_history.append(result)
        return result

    async def _statistical_ground(self, context: str, task: str) -> GroundingResult:
        """Statistical Ground: Return to raw data patterns.

        Zen: Return to breath sensation -- the raw, uninterpreted anchor.
        """
        # Extract quantitative features from the context
        word_count = len(context.split())
        sentence_count = context.count('.') + context.count('!') + context.count('?')
        avg_sentence_len = word_count / max(sentence_count, 1)

        # Count unique words vs total (vocabulary richness)
        words = context.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        findings = (
            f"Context statistics: {word_count} words, {sentence_count} sentences, "
            f"avg sentence length {avg_sentence_len:.1f} words, "
            f"vocabulary richness {unique_ratio:.2f}. "
            f"These raw numbers bypass the narrative framing."
        )
        recommendation = (
            "Strip the problem to its quantitative essence. "
            "What does the data show when you remove the narrative?"
        )
        return GroundingResult(
            channel=GroundingChannel.STATISTICAL,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _exemplar_ground(self, context: str, task: str) -> GroundingResult:
        """Exemplar Ground: Return to concrete specific instances.

        Zen: The specific sensory moment, not the abstraction.
        """
        findings = (
            f"The task asks: '{task[:200]}'. "
            "Instead of reasoning abstractly, consider three concrete cases: "
            "(1) the simplest possible case, (2) a typical case, "
            "(3) an edge case that challenges your assumptions."
        )
        recommendation = (
            "Generate three specific examples and test your reasoning "
            "against each one. Let the examples guide you rather than "
            "the abstract framework."
        )
        return GroundingResult(
            channel=GroundingChannel.EXEMPLAR,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _visual_spatial_ground(self, context: str, task: str) -> GroundingResult:
        """Visual/Spatial Ground: Represent the problem structurally.

        Zen: Shifting from discursive thought to spatial awareness.
        """
        # Extract relationships from context (simplified)
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        num_entities = len(set(context.split()))  # Rough approximation

        findings = (
            f"The reasoning involves approximately {len(sentences)} steps "
            f"and references approximately {min(num_entities, 50)} distinct concepts. "
            "Consider mapping these as a directed graph: "
            "nodes = concepts, edges = 'supports' or 'contradicts'. "
            "Look for cycles (circular reasoning) and bottlenecks "
            "(single points of failure in the argument)."
        )
        recommendation = (
            "Represent the problem spatially rather than sequentially. "
            "Draw the dependency graph. What structural patterns emerge?"
        )
        return GroundingResult(
            channel=GroundingChannel.VISUAL_SPATIAL,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _embodied_ground(
        self, context: str, task: str, agent: AgentBase,
    ) -> GroundingResult:
        """Embodied Ground: Execute something concrete.

        Zen kinhin: When sitting is stuck, walk.
        Zhuangzi's Cook Ding: The blade follows the actual ox, not the textbook.
        """
        has_tools = bool(agent.list_tools())

        if has_tools:
            findings = (
                "Tools are available. Instead of reasoning about the answer, "
                "write a small test or query that checks the core claim empirically."
            )
            recommendation = (
                "Formulate the stuck reasoning as a testable hypothesis. "
                "Use the available tools to test it directly. "
                "Let the empirical result guide the next step."
            )
        else:
            findings = (
                "No tools available for direct execution. "
                "Mentally trace through the most concrete possible scenario: "
                "what would happen step by step if your current answer were applied?"
            )
            recommendation = (
                "Trace the concrete consequences of your current answer. "
                "What specific actions would follow? What specific results?"
            )

        return GroundingResult(
            channel=GroundingChannel.EMBODIED,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _relational_ground(
        self, context: str, task: str, agent: AgentBase,
    ) -> GroundingResult:
        """Relational Ground: Seek external perspectives.

        Zen sangha: Even advanced practitioners benefit from the community.
        Sufism murshid: An external guide provides perspective the seeker cannot.
        """
        # Try to use memory for external references
        memory_results: List[Dict[str, Any]] = []
        try:
            memory_results = await agent.search_memory(task[:200], limit=3)
        except (RuntimeError, AttributeError):
            pass

        if memory_results:
            sources = "; ".join(
                str(r.get("key", r.get("content", "unknown")))[:100]
                for r in memory_results[:3]
            )
            findings = f"Related previous work found: {sources}"
        else:
            findings = (
                "No direct memory references available. "
                "Consider: how would a different system, framework, or "
                "discipline approach this exact problem?"
            )

        recommendation = (
            "Step outside your current framework. "
            "How would someone with a completely different background "
            "approach this problem? What perspectives are you not considering?"
        )
        return GroundingResult(
            channel=GroundingChannel.RELATIONAL,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _temporal_ground(self, context: str, task: str) -> GroundingResult:
        """Temporal Ground: Trace causal history.

        Yogacara: Investigating the karmic seeds that condition the present moment.
        """
        findings = (
            "Trace backward: What was the first assumption or decision that "
            "led to the current impasse? At what point did the reasoning "
            "take the path that led here? Could a different choice at that "
            "branching point avoid the current stuck state?"
        )
        recommendation = (
            "Identify the earliest decision point that constrained you "
            "into the current loop. Consider returning to that point "
            "and choosing differently."
        )
        return GroundingResult(
            channel=GroundingChannel.TEMPORAL,
            findings=findings,
            recommendation=recommendation,
            success=True,
        )

    async def _null_ground(self) -> GroundingResult:
        """Null Ground: Produce no output. Let the system settle.

        Zen shikantaza: Just sitting. Taoism wu wei: Non-action.
        Sufism: The silence between dhikr repetitions.
        """
        # Brief pause to let any async state settle
        await asyncio.sleep(0.01)

        return GroundingResult(
            channel=GroundingChannel.NULL,
            findings="Processing paused. No new information generated.",
            recommendation=(
                "After pausing, return to the original task with fresh attention. "
                "Do not review your previous stuck output."
            ),
            success=True,
        )

    @property
    def exhausted(self) -> bool:
        """Check if max grounding attempts have been reached."""
        return len(self._attempt_history) >= self.max_attempts

    def get_attempt_history(self) -> List[GroundingResult]:
        """Get the history of grounding attempts."""
        return list(self._attempt_history)

    def reset(self) -> None:
        """Reset the router state."""
        self._attempt_history.clear()
        self._channels_tried.clear()


# =============================================================================
# The @nondual_aware Decorator
# =============================================================================


def nondual_aware(
    method: Optional[Callable] = None,
    *,
    check_every: int = 1,
    quorum: int = 2,
    ensemble_threshold: float = 0.60,
    max_grounding_attempts: int = 3,
) -> Callable:
    """Decorator that adds loop-escape awareness to any agent method.

    When applied to an agent's execute or run method, this decorator
    wraps each iteration with loop detection and grounding capabilities.

    Inspired by the Dzogchen principle that awareness is self-correcting
    when it recognizes its own state. The decorator does not change the
    method's behavior when things are going well; it only intervenes
    when the method is stuck.

    Args:
        method: The method to wrap (when used without arguments).
        check_every: Check for loops every N iterations.
        quorum: Minimum detectors to fire for stuck declaration.
        ensemble_threshold: Confidence threshold for stuck declaration.
        max_grounding_attempts: Max grounding attempts before self-liberation.

    Returns:
        The wrapped method with loop-escape awareness.

    Usage:
        class MyAgent(AgentBase):
            @nondual_aware
            async def execute(self, plan: Plan) -> List[ExecutionResult]:
                ...

            @nondual_aware(check_every=2, quorum=3)
            async def run(self, task: Union[str, Task]) -> TaskResult:
                ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self_agent: AgentBase, *args: Any, **kwargs: Any) -> Any:
            # Attach detector and router if not already present
            if not hasattr(self_agent, '_nondual_detector'):
                self_agent._nondual_detector = LoopDetector(
                    quorum=quorum,
                    ensemble_threshold=ensemble_threshold,
                )
            if not hasattr(self_agent, '_nondual_router'):
                self_agent._nondual_router = GroundingRouter(
                    max_attempts=max_grounding_attempts,
                )

            # Run the original method
            result = await func(self_agent, *args, **kwargs)

            # If result is a list of ExecutionResult, check each
            if isinstance(result, list) and result and isinstance(result[0], ExecutionResult):
                for i, exec_result in enumerate(result):
                    if (i + 1) % check_every != 0:
                        continue

                    output_text = str(exec_result.output) if exec_result.output else ""
                    confidence = exec_result.metadata.get("confidence")

                    detection = self_agent._nondual_detector.check(
                        output=output_text,
                        confidence=confidence,
                        metadata=exec_result.metadata,
                    )

                    if detection.is_stuck:
                        logger.info(
                            f"Loop detected at step {i}: "
                            f"{detection.stuck_type.name} "
                            f"(confidence {detection.confidence:.2f})"
                        )
                        # Note: In the decorator form, we log the detection
                        # but cannot interrupt mid-execution. For full
                        # loop-escape, use NonDualAgent wrapper instead.

            return result
        return wrapper

    if method is not None:
        # Called without arguments: @nondual_aware
        return decorator(method)
    else:
        # Called with arguments: @nondual_aware(check_every=2)
        return decorator


# =============================================================================
# NonDualAgent: The Composable Wrapper
# =============================================================================


class NonDualAgent(AgentBase):
    """Wraps any AgentBase-derived agent with loop-escape and grounding.

    NonDualAgent is a decorator pattern implementation that adds automatic
    loop detection, wu wei stopping, perspective shifting, koan reframing,
    and self-liberation fallback to any existing agent.

    Philosophical inspiration:
    - The wrapper pattern itself is non-dual: the NonDualAgent does not replace
      the inner agent; it IS the inner agent plus awareness. This mirrors
      Dzogchen's claim that rigpa does not replace ordinary mind; it is ordinary
      mind recognized as always-already aware.
    - Kashmir Shaivism's pratyabhijna (recognition): the inner agent does not
      gain new capabilities; it recognizes capabilities it always had -- the
      ability to detect its own stuck states and shift perspective.

    Usage:
        inner = SimpleAgent(handler=my_handler)
        agent = NonDualAgent(inner)
        result = await agent.run("solve this problem")

        # With custom configuration:
        agent = NonDualAgent(
            inner,
            check_every=2,
            quorum=3,
            max_grounding_attempts=5,
        )

        # Composable: NonDualAgent wraps any AgentBase
        architect = ArchitectAgent()
        aware_architect = NonDualAgent(architect)
    """

    AGENT_ID: str = "nondual"
    SUPPORTED_MODES: List[AgentMode] = list(AgentMode)
    REQUIRES_APPROVAL: bool = False

    def __init__(
        self,
        inner: AgentBase,
        check_every: int = 1,
        quorum: int = 2,
        ensemble_threshold: float = 0.60,
        max_grounding_attempts: int = 3,
        relapse_window: int = 5,
    ):
        """Initialize the NonDualAgent wrapper.

        Args:
            inner: The inner agent to wrap. Any AgentBase-derived instance.
            check_every: Run loop detection every N iterations.
            quorum: Minimum detectors that must fire for stuck declaration.
            ensemble_threshold: Ensemble confidence threshold for stuck.
            max_grounding_attempts: Max grounding attempts before self-liberation.
            relapse_window: Iterations to monitor for re-looping after grounding.
        """
        super().__init__()
        self._inner = inner
        self._check_every = check_every
        self._relapse_window = relapse_window

        # Core components
        self._detector = LoopDetector(
            quorum=quorum,
            ensemble_threshold=ensemble_threshold,
        )
        self._router = GroundingRouter(max_attempts=max_grounding_attempts)

        # State tracking
        self._grounding_count: int = 0
        self._last_grounding_iteration: int = 0
        self._best_partial_result: Optional[Any] = None

        # Delegate configuration to inner agent
        self.AGENT_ID = f"nondual({inner.AGENT_ID})"
        self.SUPPORTED_MODES = inner.SUPPORTED_MODES

    # ----- Delegation to inner agent -----

    @property
    def mode(self) -> AgentMode:
        return self._inner.mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        self._inner.mode = value

    @property
    def conversation(self) -> ConversationHistory:
        return self._inner.conversation

    def with_approval(self, channel):
        self._inner.with_approval(channel)
        return self

    def with_memory(self, memory: MemorySystem):
        self._inner.with_memory(memory)
        return self

    def with_tools(self, tools: ToolProvider):
        self._inner.with_tools(tools)
        return self

    def with_mode(self, mode: AgentMode):
        self._inner.with_mode(mode)
        return self

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._inner.list_tools()

    # ----- Core methods -----

    async def plan(self, task: Task) -> Plan:
        """Delegate planning to the inner agent.

        Planning is not wrapped with loop detection because planning
        is typically a single-pass operation. Loop detection operates
        on the iterative execution phase.
        """
        return await self._inner.plan(task)

    async def execute(self, plan: Plan) -> List[ExecutionResult]:
        """Execute a plan with loop detection on each step.

        This is where the non-dual awareness operates. After each plan
        step, the loop detector checks for stuck conditions. If detected,
        the wu wei stop halts processing, the grounding router selects
        a channel, and the perspective shift provides new input for the
        agent to continue.

        Args:
            plan: The plan to execute.

        Returns:
            Results from each step, potentially including grounding results.
        """
        results: List[ExecutionResult] = []
        iteration = 0

        for step in plan.steps:
            iteration += 1
            start_time = time.monotonic()

            # --- Execute the step via inner agent ---
            try:
                step_results = await self._inner.execute(
                    Plan.create(task_id=plan.task_id, steps=[step])
                )
                step_result = step_results[0] if step_results else ExecutionResult.error_result(
                    step_id=step.id, error="No result from inner agent"
                )
            except Exception as e:
                step_result = ExecutionResult.error_result(
                    step_id=step.id, error=str(e)
                )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            step_result.duration_ms = int(elapsed_ms)
            results.append(step_result)

            # Track best partial result
            if step_result.success and step_result.output:
                self._best_partial_result = step_result.output

            # --- Loop detection ---
            if iteration % self._check_every != 0:
                continue

            output_text = str(step_result.output) if step_result.output else ""
            confidence = step_result.metadata.get("confidence")

            detection = self._detector.check(
                output=output_text,
                confidence=confidence,
                metadata={
                    "time_elapsed_ms": elapsed_ms,
                    "tool_calls_made": step_result.metadata.get("tool_calls", 0),
                    "quality_score": step_result.metadata.get("quality"),
                },
            )

            if not detection.is_stuck:
                continue

            # --- Loop detected: Non-dual escape protocol ---
            logger.info(
                f"NonDualAgent: Loop detected at iteration {iteration} -- "
                f"{detection.stuck_type.name} (confidence {detection.confidence:.2f})"
            )

            # Step 1: Wu Wei Stop (Taoism)
            # Do NOT attempt another iteration. Stop immediately.

            # Step 2: Check if grounding attempts are exhausted
            if self._router.exhausted:
                # Self-Liberation (Dzogchen)
                report = self._self_liberate(detection)
                results.append(ExecutionResult(
                    step_id=f"self-liberation-{iteration}",
                    success=True,
                    output=report.to_report_string(),
                    metadata={"self_liberation": True, "report": report},
                ))
                break

            # Step 3: Ground (Zen kinhin, alternate sensory modality)
            channel = self._router.select_channel(detection.stuck_type)
            grounding_result = await self._router.ground(
                channel=channel,
                stuck_context=output_text[:500],
                task_description=plan.task_id,
                agent=self._inner,
            )

            self._grounding_count += 1
            self._last_grounding_iteration = iteration

            # Step 4: Re-integrate (bring the insight back)
            if grounding_result.success:
                # Create a new step that incorporates the grounding insight
                grounding_step = PlanStep.create(
                    action=(
                        f"[GROUNDING via {channel.value}] "
                        f"{grounding_result.recommendation} "
                        f"Findings: {grounding_result.findings}"
                    )
                )
                plan.steps.insert(
                    plan.steps.index(step) + 1 if step in plan.steps else len(plan.steps),
                    grounding_step,
                )

                # Reset detection counters for fresh start
                self._detector.reset()

                results.append(ExecutionResult.success_result(
                    step_id=f"grounding-{iteration}",
                    output=grounding_result.findings,
                    metadata={
                        "grounding": True,
                        "channel": channel.value,
                        "recommendation": grounding_result.recommendation,
                    },
                ))
            else:
                # Grounding failed -- continue to next channel on next detection
                logger.warning(
                    f"Grounding via {channel.value} failed: {grounding_result.findings}"
                )

        return results

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute a task end-to-end with non-dual awareness.

        This is the main entry point. It delegates to the inner agent's
        plan-then-execute lifecycle but wraps execution with loop detection.

        Args:
            task: The task to execute (string or Task object).

        Returns:
            The result, which may include grounding results and/or
            a self-liberation report if the agent got stuck.
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()

        try:
            # Plan (no loop detection during planning)
            plan = await self.plan(task)

            # Execute (with loop detection)
            results = await self.execute(plan)

            # Collect results
            success = any(r.success for r in results)
            outputs = [r.output for r in results if r.success and r.output]
            errors = [r.error for r in results if not r.success and r.error]

            # Check if self-liberation occurred
            self_liberation = any(
                r.metadata.get("self_liberation", False)
                for r in results
                if isinstance(r.metadata, dict)
            )

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            if success:
                output = outputs[-1] if outputs else None
                return TaskResult.success_result(
                    task_id=task.id,
                    output=output,
                    execution_results=results,
                    messages=list(self._inner.conversation),
                    duration_ms=duration,
                    metadata={
                        "grounding_count": self._grounding_count,
                        "self_liberation": self_liberation,
                    },
                )
            else:
                return TaskResult.error_result(
                    task_id=task.id,
                    error="; ".join(errors) if errors else "Unknown error",
                    execution_results=results,
                    messages=list(self._inner.conversation),
                    duration_ms=duration,
                    metadata={
                        "grounding_count": self._grounding_count,
                        "self_liberation": self_liberation,
                    },
                )

        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            return TaskResult.error_result(
                task_id=task.id,
                error=str(e),
                duration_ms=duration,
            )

    def _self_liberate(self, detection: LoopDetectionResult) -> SelfLiberationReport:
        """Generate a self-liberation report.

        Dzogchen principle: rang grol (self-liberation). The stuck state
        dissolves when recognized and honestly communicated. The report
        IS the transcendence -- not a failure message but the most useful
        output the agent can produce given the constraints.

        Args:
            detection: The final loop detection result.

        Returns:
            A structured report of the impasse.
        """
        stuck_type = detection.stuck_type or StuckType.REPETITIVE

        # Analyze root cause
        root_causes = {
            StuckType.REPETITIVE: (
                "The agent is producing repetitive output, suggesting "
                "the current approach has exhausted its productive potential."
            ),
            StuckType.BINARY_OSCILLATION: (
                "The agent oscillates between two options, suggesting "
                "the binary framing is incorrect or the options are not "
                "genuinely exclusive."
            ),
            StuckType.CONFIDENCE_DRIFT: (
                "The agent cannot stabilize its confidence, suggesting "
                "the evaluation criteria are ambiguous or conflicting."
            ),
            StuckType.SELF_REFERENTIAL: (
                "The agent is caught in meta-reasoning loops, suggesting "
                "the task requires a level of self-analysis that generates "
                "infinite regress."
            ),
            StuckType.CONTRADICTORY: (
                "The agent holds contradictory beliefs, suggesting the task "
                "contains conflicting requirements or the agent's knowledge "
                "base is inconsistent."
            ),
            StuckType.CIRCULAR: (
                "The agent's reasoning is circular, suggesting the question "
                "assumes its own answer or a hidden premise creates a loop."
            ),
            StuckType.RESOURCE_WASTE: (
                "The agent consumes increasing resources without quality "
                "improvement, suggesting diminishing returns on the current "
                "approach."
            ),
        }

        alternative_framings = {
            StuckType.REPETITIVE: [
                "Try a completely different methodology",
                "Seek external references or examples",
                "Simplify the problem to its core",
            ],
            StuckType.BINARY_OSCILLATION: [
                "Consider whether both options could be valid in different contexts",
                "Look for a third option not yet considered",
                "Question whether the choice is necessary at all",
            ],
            StuckType.CONFIDENCE_DRIFT: [
                "Define explicit, measurable evaluation criteria",
                "Use concrete examples to anchor judgments",
                "Accept uncertainty and report a confidence interval rather than a point estimate",
            ],
            StuckType.SELF_REFERENTIAL: [
                "Answer the task directly without self-analysis",
                "Set a hard limit on meta-reasoning depth",
                "Treat the task as a straightforward action, not a reflection",
            ],
            StuckType.CONTRADICTORY: [
                "Identify and remove the conflicting requirement",
                "Acknowledge the contradiction and present both sides",
                "Seek clarification on which requirement takes priority",
            ],
            StuckType.CIRCULAR: [
                "Identify the assumed premise and treat it as a question",
                "Break the circle by introducing external evidence",
                "Reformulate the question to avoid self-reference",
            ],
            StuckType.RESOURCE_WASTE: [
                "Accept the best result achieved so far",
                "Simplify the approach drastically",
                "Defer to a human or more specialized system",
            ],
        }

        return SelfLiberationReport(
            stuck_type=stuck_type,
            detection_detail=(
                f"Detected by {len(detection.active_signals)} detector(s) "
                f"with ensemble confidence {detection.confidence:.2f}: "
                + "; ".join(s.detail for s in detection.active_signals)
            ),
            grounding_attempts=self._router.get_attempt_history(),
            root_cause=root_causes.get(stuck_type, "Unknown root cause"),
            partial_result=self._best_partial_result,
            alternative_framings=alternative_framings.get(stuck_type, []),
        )

    def __repr__(self) -> str:
        return (
            f"<NonDualAgent wrapping {self._inner.__class__.__name__} "
            f"id={self.AGENT_ID} groundings={self._grounding_count}>"
        )
