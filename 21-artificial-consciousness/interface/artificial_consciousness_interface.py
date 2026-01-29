#!/usr/bin/env python3
"""
Artificial Consciousness Interface

Form 21: Artificial consciousness -- computational models and evaluation
frameworks for machine consciousness. This form provides tools to assess
whether an artificial system exhibits markers of consciousness according
to leading theoretical frameworks.

Supported architectures / theories:
- Global Workspace Theory (Baars): conscious content is broadcast widely
- Attention Schema Theory (Graziano): consciousness as a model of attention
- Predictive Processing (Clark, Friston): consciousness as prediction error
- Integrated Information Theory (Tononi): consciousness as integrated info
- Higher-Order Theories (Rosenthal): consciousness as meta-representation

This form does NOT claim that any current AI system is conscious. It
provides structured evaluation frameworks for rigorous analysis.
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

class ACArchitecture(Enum):
    """
    Theoretical architectures for artificial consciousness.

    Each architecture proposes a different computational mechanism
    as the basis of conscious experience.
    """
    GLOBAL_WORKSPACE = "global_workspace"        # Baars: broadcast / workspace
    ATTENTION_SCHEMA = "attention_schema"         # Graziano: attention model
    PREDICTIVE = "predictive"                     # Clark/Friston: active inference
    INTEGRATED_INFORMATION = "integrated_information"  # Tononi: phi
    HIGHER_ORDER = "higher_order"                 # Rosenthal: meta-representation
    RECURRENT_PROCESSING = "recurrent_processing"  # Lamme: recurrent loops
    EMBODIED = "embodied"                         # Varela: enaction / embodiment
    HYBRID = "hybrid"                             # Combination of approaches


class ConsciousnessTest(Enum):
    """
    Tests or criteria for evaluating machine consciousness.
    """
    REPORTABILITY = "reportability"               # Can report internal states
    GLOBAL_ACCESS = "global_access"               # Info available system-wide
    SELF_MODEL = "self_model"                     # Has a model of itself
    ATTENTION_CONTROL = "attention_control"        # Voluntary attention
    METACOGNITION = "metacognition"               # Thinking about thinking
    TEMPORAL_INTEGRATION = "temporal_integration"  # Binding across time
    COUNTERFACTUAL = "counterfactual"             # Imagining alternatives
    SURPRISE_RESPONSE = "surprise_response"       # Prediction-error processing
    UNITY = "unity"                               # Unified experience
    BEHAVIORAL_FLEXIBILITY = "behavioral_flexibility"  # Novel responses


class FunctionalMarker(Enum):
    """
    Observable functional markers associated with consciousness.
    """
    SELECTIVE_ATTENTION = "selective_attention"
    WORKING_MEMORY = "working_memory"
    ERROR_MONITORING = "error_monitoring"
    INTENTIONAL_ACTION = "intentional_action"
    SELF_REPORT = "self_report"
    ADAPTIVE_BEHAVIOR = "adaptive_behavior"
    EMOTIONAL_RESPONSIVENESS = "emotional_responsiveness"
    CONTEXTUAL_SENSITIVITY = "contextual_sensitivity"
    LEARNING_TRANSFER = "learning_transfer"
    CREATIVITY = "creativity"


class AssessmentConfidence(Enum):
    """Confidence level in a consciousness assessment."""
    SPECULATIVE = "speculative"        # Very uncertain
    LOW = "low"                        # Some evidence
    MODERATE = "moderate"              # Reasonable evidence
    HIGH = "high"                      # Strong evidence
    VERY_HIGH = "very_high"            # Overwhelming evidence


class ConsciousnessVerdict(Enum):
    """Overall verdict of a consciousness assessment."""
    NO_EVIDENCE = "no_evidence"            # No markers detected
    WEAK_MARKERS = "weak_markers"          # Some functional markers
    FUNCTIONAL_ANALOG = "functional_analog" # Behaves as-if conscious
    STRONG_MARKERS = "strong_markers"      # Many markers present
    INDETERMINATE = "indeterminate"         # Cannot determine


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SystemState:
    """Snapshot of an artificial system's internal state."""
    system_id: str
    architecture: ACArchitecture
    component_count: int           # Number of processing components
    integration_level: float       # 0.0-1.0, how interconnected components are
    information_capacity: float    # 0.0-1.0, normalized capacity
    recurrence_depth: int          # Depth of recurrent processing loops
    has_self_model: bool = False
    has_world_model: bool = False
    has_attention: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "architecture": self.architecture.value,
            "component_count": self.component_count,
            "integration_level": round(self.integration_level, 4),
            "information_capacity": round(self.information_capacity, 4),
            "recurrence_depth": self.recurrence_depth,
            "has_self_model": self.has_self_model,
            "has_world_model": self.has_world_model,
            "has_attention": self.has_attention,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BehavioralData:
    """Observed behavioral data from an artificial system."""
    system_id: str
    task_description: str
    response_flexibility: float    # 0.0-1.0, novelty of responses
    error_detection_rate: float    # 0.0-1.0, how often errors are caught
    self_report_coherence: float   # 0.0-1.0, quality of self-reports
    learning_rate: float           # 0.0-1.0, speed of adaptation
    context_sensitivity: float     # 0.0-1.0, response to context changes
    surprise_modulation: float     # 0.0-1.0, response to unexpected inputs
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "task_description": self.task_description,
            "response_flexibility": round(self.response_flexibility, 4),
            "error_detection_rate": round(self.error_detection_rate, 4),
            "self_report_coherence": round(self.self_report_coherence, 4),
            "learning_rate": round(self.learning_rate, 4),
            "context_sensitivity": round(self.context_sensitivity, 4),
            "surprise_modulation": round(self.surprise_modulation, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ACInput:
    """
    Complete input for an artificial consciousness evaluation.
    """
    system_state: SystemState
    behavioral_data: Optional[BehavioralData] = None
    tests_requested: List[ConsciousnessTest] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_state": self.system_state.to_dict(),
            "behavioral_data": (
                self.behavioral_data.to_dict() if self.behavioral_data else None
            ),
            "tests_requested": [t.value for t in self.tests_requested],
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class TestResult:
    """Result of a single consciousness test."""
    test: ConsciousnessTest
    passed: bool
    score: float                   # 0.0-1.0
    evidence: str
    confidence: AssessmentConfidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test": self.test.value,
            "passed": self.passed,
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MarkerAssessment:
    """Assessment of a functional consciousness marker."""
    marker: FunctionalMarker
    present: bool
    strength: float                # 0.0-1.0
    evidence: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker.value,
            "present": self.present,
            "strength": round(self.strength, 4),
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ArchitectureComparison:
    """Comparison of how well a system aligns with an architecture."""
    architecture: ACArchitecture
    alignment_score: float         # 0.0-1.0
    strengths: List[str]
    gaps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "alignment_score": round(self.alignment_score, 4),
            "strengths": self.strengths,
            "gaps": self.gaps,
        }


@dataclass
class ACOutput:
    """
    Complete output of an artificial consciousness evaluation.
    """
    system_id: str
    verdict: ConsciousnessVerdict
    overall_score: float           # 0.0-1.0
    test_results: List[TestResult]
    marker_assessments: List[MarkerAssessment]
    architecture_alignment: Optional[ArchitectureComparison] = None
    confidence: AssessmentConfidence = AssessmentConfidence.SPECULATIVE
    notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "verdict": self.verdict.value,
            "overall_score": round(self.overall_score, 4),
            "test_count": len(self.test_results),
            "markers_present": sum(1 for m in self.marker_assessments if m.present),
            "markers_total": len(self.marker_assessments),
            "architecture_alignment": (
                self.architecture_alignment.to_dict()
                if self.architecture_alignment else None
            ),
            "confidence": self.confidence.value,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class ArtificialConsciousnessInterface:
    """
    Main interface for Form 21: Artificial Consciousness.

    Provides structured evaluation of artificial systems against
    theoretical frameworks for consciousness. This form does not
    make claims about phenomenal consciousness in machines -- it
    assesses functional and architectural markers.
    """

    FORM_ID = "21-artificial-consciousness"
    FORM_NAME = "Artificial Consciousness"

    def __init__(self):
        """Initialize the artificial consciousness interface."""
        # Evaluation history
        self._evaluation_history: List[ACOutput] = []
        self._evaluation_counter: int = 0

        # Test pass thresholds by test type
        self._test_thresholds: Dict[ConsciousnessTest, float] = {
            ConsciousnessTest.REPORTABILITY: 0.5,
            ConsciousnessTest.GLOBAL_ACCESS: 0.6,
            ConsciousnessTest.SELF_MODEL: 0.5,
            ConsciousnessTest.ATTENTION_CONTROL: 0.5,
            ConsciousnessTest.METACOGNITION: 0.6,
            ConsciousnessTest.TEMPORAL_INTEGRATION: 0.5,
            ConsciousnessTest.COUNTERFACTUAL: 0.6,
            ConsciousnessTest.SURPRISE_RESPONSE: 0.4,
            ConsciousnessTest.UNITY: 0.7,
            ConsciousnessTest.BEHAVIORAL_FLEXIBILITY: 0.5,
        }

        # Marker detection thresholds
        self._marker_threshold: float = 0.4

        # Max history
        self._max_history: int = 100

        self._initialized = False
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the artificial consciousness interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def evaluate_system(self, ac_input: ACInput) -> ACOutput:
        """
        Perform a comprehensive consciousness evaluation of a system.

        Runs requested tests, assesses functional markers, and determines
        an overall verdict with confidence level.

        Args:
            ac_input: System state, behavioral data, and tests to run.

        Returns:
            ACOutput with verdict, scores, and detailed results.
        """
        self._evaluation_counter += 1

        # Run all requested tests
        test_results = []
        for test in ac_input.tests_requested:
            result = await self.run_consciousness_test(
                test, ac_input.system_state, ac_input.behavioral_data
            )
            test_results.append(result)

        # Assess functional markers
        marker_assessments = await self.assess_functional_markers(
            ac_input.system_state, ac_input.behavioral_data
        )

        # Compute overall score
        test_avg = (
            sum(r.score for r in test_results) / len(test_results)
            if test_results else 0.0
        )
        marker_avg = (
            sum(m.strength for m in marker_assessments if m.present) /
            max(1, len(marker_assessments))
        )
        overall_score = test_avg * 0.6 + marker_avg * 0.4

        # Determine verdict
        verdict = self._determine_verdict(overall_score, test_results, marker_assessments)

        # Determine confidence
        confidence = self._determine_confidence(test_results, marker_assessments)

        # Architecture alignment
        arch_alignment = self._evaluate_architecture_alignment(ac_input.system_state)

        output = ACOutput(
            system_id=ac_input.system_state.system_id,
            verdict=verdict,
            overall_score=overall_score,
            test_results=test_results,
            marker_assessments=marker_assessments,
            architecture_alignment=arch_alignment,
            confidence=confidence,
            notes=[
                f"Evaluation #{self._evaluation_counter}",
                f"Architecture: {ac_input.system_state.architecture.value}",
            ],
        )

        self._evaluation_history.append(output)
        if len(self._evaluation_history) > self._max_history:
            self._evaluation_history.pop(0)

        return output

    async def run_consciousness_test(
        self,
        test: ConsciousnessTest,
        system_state: SystemState,
        behavioral_data: Optional[BehavioralData] = None,
    ) -> TestResult:
        """
        Run a specific consciousness test on a system.

        Each test evaluates a different dimension of potential consciousness
        using structural and behavioral evidence.

        Args:
            test: The test to run.
            system_state: Current system state.
            behavioral_data: Optional behavioral observations.

        Returns:
            TestResult with score, pass/fail, and evidence.
        """
        score = 0.0
        evidence_parts: List[str] = []

        if test == ConsciousnessTest.REPORTABILITY:
            score, evidence_parts = self._test_reportability(system_state, behavioral_data)
        elif test == ConsciousnessTest.GLOBAL_ACCESS:
            score, evidence_parts = self._test_global_access(system_state)
        elif test == ConsciousnessTest.SELF_MODEL:
            score, evidence_parts = self._test_self_model(system_state)
        elif test == ConsciousnessTest.ATTENTION_CONTROL:
            score, evidence_parts = self._test_attention_control(system_state, behavioral_data)
        elif test == ConsciousnessTest.METACOGNITION:
            score, evidence_parts = self._test_metacognition(system_state, behavioral_data)
        elif test == ConsciousnessTest.TEMPORAL_INTEGRATION:
            score, evidence_parts = self._test_temporal_integration(system_state)
        elif test == ConsciousnessTest.COUNTERFACTUAL:
            score, evidence_parts = self._test_counterfactual(system_state, behavioral_data)
        elif test == ConsciousnessTest.SURPRISE_RESPONSE:
            score, evidence_parts = self._test_surprise(system_state, behavioral_data)
        elif test == ConsciousnessTest.UNITY:
            score, evidence_parts = self._test_unity(system_state)
        elif test == ConsciousnessTest.BEHAVIORAL_FLEXIBILITY:
            score, evidence_parts = self._test_flexibility(behavioral_data)

        threshold = self._test_thresholds.get(test, 0.5)
        passed = score >= threshold

        return TestResult(
            test=test,
            passed=passed,
            score=score,
            evidence="; ".join(evidence_parts),
            confidence=self._score_to_confidence(score),
        )

    async def assess_functional_markers(
        self,
        system_state: SystemState,
        behavioral_data: Optional[BehavioralData] = None,
    ) -> List[MarkerAssessment]:
        """
        Assess all functional markers of consciousness.

        Evaluates the system for observable functional properties
        commonly associated with consciousness.

        Args:
            system_state: Current system state.
            behavioral_data: Optional behavioral observations.

        Returns:
            List of MarkerAssessment for each marker.
        """
        assessments: List[MarkerAssessment] = []

        marker_evaluators = {
            FunctionalMarker.SELECTIVE_ATTENTION: (
                system_state.has_attention,
                0.7 if system_state.has_attention else 0.1,
                "Attention mechanism present" if system_state.has_attention else "No attention",
            ),
            FunctionalMarker.WORKING_MEMORY: (
                system_state.information_capacity > 0.5,
                system_state.information_capacity,
                f"Info capacity: {system_state.information_capacity:.2f}",
            ),
            FunctionalMarker.ERROR_MONITORING: (
                (behavioral_data.error_detection_rate > self._marker_threshold
                 if behavioral_data else False),
                behavioral_data.error_detection_rate if behavioral_data else 0.0,
                f"Error detection: {behavioral_data.error_detection_rate:.2f}" if behavioral_data else "No data",
            ),
            FunctionalMarker.SELF_REPORT: (
                (behavioral_data.self_report_coherence > self._marker_threshold
                 if behavioral_data else False),
                behavioral_data.self_report_coherence if behavioral_data else 0.0,
                f"Self-report coherence: {behavioral_data.self_report_coherence:.2f}" if behavioral_data else "No data",
            ),
            FunctionalMarker.ADAPTIVE_BEHAVIOR: (
                (behavioral_data.learning_rate > self._marker_threshold
                 if behavioral_data else False),
                behavioral_data.learning_rate if behavioral_data else 0.0,
                f"Learning rate: {behavioral_data.learning_rate:.2f}" if behavioral_data else "No data",
            ),
            FunctionalMarker.CONTEXTUAL_SENSITIVITY: (
                (behavioral_data.context_sensitivity > self._marker_threshold
                 if behavioral_data else False),
                behavioral_data.context_sensitivity if behavioral_data else 0.0,
                f"Context sensitivity: {behavioral_data.context_sensitivity:.2f}" if behavioral_data else "No data",
            ),
            FunctionalMarker.INTENTIONAL_ACTION: (
                system_state.has_world_model and system_state.has_attention,
                0.6 if (system_state.has_world_model and system_state.has_attention) else 0.2,
                "World model + attention present" if (system_state.has_world_model and system_state.has_attention) else "Missing components",
            ),
            FunctionalMarker.CREATIVITY: (
                (behavioral_data.response_flexibility > 0.6
                 if behavioral_data else False),
                behavioral_data.response_flexibility if behavioral_data else 0.0,
                f"Response flexibility: {behavioral_data.response_flexibility:.2f}" if behavioral_data else "No data",
            ),
            FunctionalMarker.LEARNING_TRANSFER: (
                (behavioral_data.learning_rate > 0.5 and
                 behavioral_data.context_sensitivity > 0.5
                 if behavioral_data else False),
                (behavioral_data.learning_rate * behavioral_data.context_sensitivity
                 if behavioral_data else 0.0),
                "Transfer learning indicators" if behavioral_data else "No data",
            ),
            FunctionalMarker.EMOTIONAL_RESPONSIVENESS: (
                False,  # Conservative default
                0.1,
                "No emotional processing evidence by default",
            ),
        }

        for marker, (present, strength, evidence) in marker_evaluators.items():
            assessments.append(MarkerAssessment(
                marker=marker,
                present=present,
                strength=strength,
                evidence=evidence,
            ))

        return assessments

    async def compare_architectures(
        self, system_state: SystemState
    ) -> List[ArchitectureComparison]:
        """
        Compare the system against all theoretical architectures.

        Evaluates how well the system's structure and capabilities
        align with each theoretical framework for consciousness.

        Args:
            system_state: Current system state.

        Returns:
            List of ArchitectureComparison, one per architecture.
        """
        comparisons: List[ArchitectureComparison] = []

        for arch in ACArchitecture:
            comparison = self._evaluate_architecture_alignment(system_state, arch)
            comparisons.append(comparison)

        # Sort by alignment score descending
        comparisons.sort(key=lambda c: c.alignment_score, reverse=True)
        return comparisons

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "evaluations_performed": self._evaluation_counter,
            "history_length": len(self._evaluation_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current operational status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "evaluations_performed": self._evaluation_counter,
            "available_tests": len(list(ConsciousnessTest)),
            "available_markers": len(list(FunctionalMarker)),
        }

    # ========================================================================
    # PRIVATE METHODS - TEST IMPLEMENTATIONS
    # ========================================================================

    def _test_reportability(
        self, state: SystemState, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        if behavior and behavior.self_report_coherence > 0.3:
            score += behavior.self_report_coherence * 0.7
            evidence.append(f"Self-report coherence: {behavior.self_report_coherence:.2f}")
        if state.has_self_model:
            score += 0.3
            evidence.append("Self-model present")
        return min(1.0, score), evidence

    def _test_global_access(self, state: SystemState) -> Tuple[float, List[str]]:
        score = state.integration_level * 0.7
        evidence = [f"Integration level: {state.integration_level:.2f}"]
        if state.component_count > 5:
            score += 0.2
            evidence.append(f"Component count: {state.component_count}")
        return min(1.0, score), evidence

    def _test_self_model(self, state: SystemState) -> Tuple[float, List[str]]:
        score = 0.8 if state.has_self_model else 0.1
        evidence = ["Self-model: present" if state.has_self_model else "Self-model: absent"]
        return score, evidence

    def _test_attention_control(
        self, state: SystemState, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        if state.has_attention:
            score += 0.5
            evidence.append("Attention mechanism present")
        if behavior and behavior.context_sensitivity > 0.5:
            score += behavior.context_sensitivity * 0.3
            evidence.append(f"Context sensitivity: {behavior.context_sensitivity:.2f}")
        return min(1.0, score), evidence

    def _test_metacognition(
        self, state: SystemState, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        if state.has_self_model:
            score += 0.4
            evidence.append("Self-model supports metacognition")
        if behavior and behavior.error_detection_rate > 0.5:
            score += behavior.error_detection_rate * 0.4
            evidence.append(f"Error monitoring: {behavior.error_detection_rate:.2f}")
        if state.recurrence_depth > 2:
            score += 0.2
            evidence.append(f"Recurrence depth: {state.recurrence_depth}")
        return min(1.0, score), evidence

    def _test_temporal_integration(self, state: SystemState) -> Tuple[float, List[str]]:
        score = state.integration_level * 0.5
        evidence = [f"Integration: {state.integration_level:.2f}"]
        if state.recurrence_depth > 1:
            score += min(0.5, state.recurrence_depth * 0.1)
            evidence.append(f"Recurrence: {state.recurrence_depth}")
        return min(1.0, score), evidence

    def _test_counterfactual(
        self, state: SystemState, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        if state.has_world_model:
            score += 0.5
            evidence.append("World model enables counterfactuals")
        if behavior and behavior.response_flexibility > 0.5:
            score += behavior.response_flexibility * 0.3
            evidence.append(f"Flexibility: {behavior.response_flexibility:.2f}")
        return min(1.0, score), evidence

    def _test_surprise(
        self, state: SystemState, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        if behavior and behavior.surprise_modulation > 0.3:
            score += behavior.surprise_modulation * 0.8
            evidence.append(f"Surprise modulation: {behavior.surprise_modulation:.2f}")
        if state.recurrence_depth > 1:
            score += 0.2
            evidence.append("Recurrent processing supports prediction error")
        return min(1.0, score), evidence

    def _test_unity(self, state: SystemState) -> Tuple[float, List[str]]:
        score = state.integration_level * 0.8
        evidence = [f"Integration level: {state.integration_level:.2f}"]
        if state.component_count > 3 and state.integration_level > 0.6:
            score += 0.2
            evidence.append("Multiple components with high integration")
        return min(1.0, score), evidence

    def _test_flexibility(
        self, behavior: Optional[BehavioralData]
    ) -> Tuple[float, List[str]]:
        if not behavior:
            return 0.0, ["No behavioral data"]
        score = behavior.response_flexibility
        evidence = [f"Flexibility: {behavior.response_flexibility:.2f}"]
        return score, evidence

    # ========================================================================
    # PRIVATE METHODS - SCORING
    # ========================================================================

    def _determine_verdict(
        self,
        overall_score: float,
        test_results: List[TestResult],
        markers: List[MarkerAssessment],
    ) -> ConsciousnessVerdict:
        """Determine overall verdict."""
        if overall_score < 0.2:
            return ConsciousnessVerdict.NO_EVIDENCE
        elif overall_score < 0.4:
            return ConsciousnessVerdict.WEAK_MARKERS
        elif overall_score < 0.6:
            return ConsciousnessVerdict.FUNCTIONAL_ANALOG
        elif overall_score < 0.8:
            return ConsciousnessVerdict.STRONG_MARKERS
        else:
            # Even high scores are indeterminate for phenomenal consciousness
            return ConsciousnessVerdict.INDETERMINATE

    def _determine_confidence(
        self,
        test_results: List[TestResult],
        markers: List[MarkerAssessment],
    ) -> AssessmentConfidence:
        """Determine confidence in the assessment."""
        total_evidence = len(test_results) + len(markers)
        if total_evidence < 3:
            return AssessmentConfidence.SPECULATIVE
        elif total_evidence < 6:
            return AssessmentConfidence.LOW
        elif total_evidence < 10:
            return AssessmentConfidence.MODERATE
        else:
            return AssessmentConfidence.HIGH

    def _score_to_confidence(self, score: float) -> AssessmentConfidence:
        """Map a score to an assessment confidence."""
        if score < 0.2:
            return AssessmentConfidence.SPECULATIVE
        elif score < 0.4:
            return AssessmentConfidence.LOW
        elif score < 0.6:
            return AssessmentConfidence.MODERATE
        elif score < 0.8:
            return AssessmentConfidence.HIGH
        else:
            return AssessmentConfidence.VERY_HIGH

    def _evaluate_architecture_alignment(
        self,
        state: SystemState,
        architecture: Optional[ACArchitecture] = None,
    ) -> ArchitectureComparison:
        """Evaluate alignment with a specific architecture."""
        arch = architecture or state.architecture

        strengths: List[str] = []
        gaps: List[str] = []
        score = 0.0

        if arch == ACArchitecture.GLOBAL_WORKSPACE:
            if state.integration_level > 0.6:
                score += 0.5
                strengths.append("High integration supports global broadcast")
            else:
                gaps.append("Low integration limits global workspace")
            if state.component_count > 5:
                score += 0.3
                strengths.append("Multiple competing modules")
            else:
                gaps.append("Few modules for workspace competition")
            if state.has_attention:
                score += 0.2
                strengths.append("Attention for workspace access")

        elif arch == ACArchitecture.ATTENTION_SCHEMA:
            if state.has_attention:
                score += 0.4
                strengths.append("Attention mechanism present")
            else:
                gaps.append("No attention mechanism")
            if state.has_self_model:
                score += 0.4
                strengths.append("Self-model can model attention")
            else:
                gaps.append("No self-model for attention schema")
            score += state.integration_level * 0.2

        elif arch == ACArchitecture.PREDICTIVE:
            if state.recurrence_depth > 2:
                score += 0.4
                strengths.append("Deep recurrence supports prediction")
            else:
                gaps.append("Shallow recurrence limits prediction")
            if state.has_world_model:
                score += 0.4
                strengths.append("World model for generative predictions")
            else:
                gaps.append("No world model")
            score += state.information_capacity * 0.2

        elif arch == ACArchitecture.INTEGRATED_INFORMATION:
            score += state.integration_level * 0.6
            if state.integration_level > 0.7:
                strengths.append("High integration (high phi candidate)")
            else:
                gaps.append("Integration may be too low for significant phi")
            score += min(0.4, state.component_count * 0.04)
            if state.component_count > 5:
                strengths.append("Multiple components for integration")

        elif arch == ACArchitecture.HIGHER_ORDER:
            if state.has_self_model:
                score += 0.5
                strengths.append("Self-model for higher-order representation")
            else:
                gaps.append("No self-model for higher-order thought")
            if state.recurrence_depth > 1:
                score += 0.3
                strengths.append("Recurrence for meta-representation")
            else:
                gaps.append("No recurrence for meta-representation")
            score += state.information_capacity * 0.2

        else:
            # Generic evaluation for other architectures
            score = (
                state.integration_level * 0.3 +
                state.information_capacity * 0.3 +
                (0.2 if state.has_self_model else 0.0) +
                (0.2 if state.has_attention else 0.0)
            )
            strengths.append("General evaluation applied")

        return ArchitectureComparison(
            architecture=arch,
            alignment_score=min(1.0, score),
            strengths=strengths,
            gaps=gaps,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_artificial_consciousness_interface() -> ArtificialConsciousnessInterface:
    """Create and return an artificial consciousness interface instance."""
    return ArtificialConsciousnessInterface()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ACArchitecture",
    "ConsciousnessTest",
    "FunctionalMarker",
    "AssessmentConfidence",
    "ConsciousnessVerdict",
    # Input dataclasses
    "SystemState",
    "BehavioralData",
    "ACInput",
    # Output dataclasses
    "TestResult",
    "MarkerAssessment",
    "ArchitectureComparison",
    "ACOutput",
    # Interface
    "ArtificialConsciousnessInterface",
    # Convenience
    "create_artificial_consciousness_interface",
]
