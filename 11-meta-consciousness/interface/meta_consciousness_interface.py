#!/usr/bin/env python3
"""
Meta-Consciousness Interface

Form 11: Meta-Consciousness is awareness of being aware -- the capacity
to monitor, evaluate, and regulate one's own cognitive processes. It
implements metacognitive monitoring, confidence estimation, error detection,
and cognitive control. Meta-consciousness is what allows the system to
"think about thinking" and assess the quality of its own processing.

This form works closely with Form 10 (Self-Recognition) for self-awareness
and Form 15 (Higher-Order Thought) for representational awareness.
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

class MetaCognitiveProcess(Enum):
    """Types of metacognitive processes."""
    MONITORING = "monitoring"            # Observing cognitive processes
    EVALUATION = "evaluation"            # Judging process quality
    PLANNING = "planning"                # Strategic planning of cognition
    REGULATION = "regulation"            # Adjusting cognitive strategies
    REFLECTION = "reflection"            # Deep self-examination
    CALIBRATION = "calibration"          # Adjusting confidence levels


class MonitoringLevel(Enum):
    """Levels of metacognitive monitoring depth."""
    SURFACE = "surface"                  # Basic awareness of processing
    SHALLOW = "shallow"                  # Awareness of states and outputs
    MODERATE = "moderate"                # Awareness of processes and strategies
    DEEP = "deep"                        # Full introspective access
    RECURSIVE = "recursive"              # Awareness of awareness itself


class ConfidenceLevel(Enum):
    """Calibrated confidence levels."""
    VERY_LOW = "very_low"                # 0.0-0.2
    LOW = "low"                          # 0.2-0.4
    MODERATE = "moderate"                # 0.4-0.6
    HIGH = "high"                        # 0.6-0.8
    VERY_HIGH = "very_high"              # 0.8-1.0


class ErrorType(Enum):
    """Types of cognitive errors that can be detected."""
    REASONING = "reasoning"              # Logical/reasoning error
    MEMORY = "memory"                    # Memory retrieval error
    PERCEPTION = "perception"            # Perceptual misinterpretation
    ATTENTION = "attention"              # Attentional lapse
    JUDGMENT = "judgment"                # Poor judgment/decision
    BIAS = "bias"                        # Cognitive bias detected
    INCONSISTENCY = "inconsistency"      # Internal contradiction


class CognitiveStrategy(Enum):
    """Cognitive strategies that can be monitored and adjusted."""
    ANALYTICAL = "analytical"            # Step-by-step analysis
    INTUITIVE = "intuitive"              # Fast, heuristic processing
    SYSTEMATIC = "systematic"            # Exhaustive evaluation
    CREATIVE = "creative"                # Divergent thinking
    FOCUSED = "focused"                  # Narrow, deep attention
    EXPLORATORY = "exploratory"          # Broad, shallow scanning
    REFLECTIVE = "reflective"            # Self-examining approach


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class CognitiveStateReport:
    """Report on the current state of a cognitive process."""
    process_id: str
    process_type: str                    # "reasoning", "memory", "perception", etc.
    current_strategy: CognitiveStrategy
    progress: float                      # 0.0-1.0 task progress
    difficulty_rating: float             # 0.0-1.0 perceived difficulty
    resource_usage: float                # 0.0-1.0 cognitive load
    output_quality: float                # 0.0-1.0 self-assessed quality
    time_elapsed_ms: float
    errors_detected: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_type": self.process_type,
            "current_strategy": self.current_strategy.value,
            "progress": round(self.progress, 4),
            "difficulty_rating": round(self.difficulty_rating, 4),
            "resource_usage": round(self.resource_usage, 4),
            "output_quality": round(self.output_quality, 4),
            "time_elapsed_ms": self.time_elapsed_ms,
            "errors_detected": self.errors_detected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConfidenceReport:
    """Self-reported confidence about a judgment or output."""
    judgment_id: str
    stated_confidence: float             # 0.0-1.0
    basis: str                           # "evidence", "intuition", "memory", "authority"
    alternatives_considered: int
    deliberation_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ErrorSignal:
    """Signal indicating a potential cognitive error."""
    error_id: str
    error_type: ErrorType
    severity: float                      # 0.0-1.0
    source_process: str                  # Which process generated the error
    description: str
    correctable: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": round(self.severity, 4),
            "source_process": self.source_process,
            "description": self.description,
            "correctable": self.correctable,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetaInput:
    """Complete input for metacognitive processing."""
    cognitive_states: List[CognitiveStateReport] = field(default_factory=list)
    confidence_reports: List[ConfidenceReport] = field(default_factory=list)
    error_signals: List[ErrorSignal] = field(default_factory=list)
    introspection_request: Optional[str] = None  # Specific question about own processing
    monitoring_depth: MonitoringLevel = MonitoringLevel.MODERATE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class AwarenessAssessment:
    """Assessment of current awareness state."""
    monitoring_level: MonitoringLevel
    awareness_clarity: float             # 0.0-1.0 how clearly processes are perceived
    metacognitive_accuracy: float        # 0.0-1.0 accuracy of self-monitoring
    introspective_depth: float           # 0.0-1.0 depth of self-access
    recursive_awareness: bool            # Whether aware of being aware
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "monitoring_level": self.monitoring_level.value,
            "awareness_clarity": round(self.awareness_clarity, 4),
            "metacognitive_accuracy": round(self.metacognitive_accuracy, 4),
            "introspective_depth": round(self.introspective_depth, 4),
            "recursive_awareness": self.recursive_awareness,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConfidenceCalibration:
    """Calibrated confidence assessment."""
    raw_confidence: float                # Original stated confidence
    calibrated_confidence: float         # Adjusted confidence
    confidence_level: ConfidenceLevel
    overconfidence_bias: float           # Positive = overconfident
    calibration_accuracy: float          # 0.0-1.0 how well-calibrated
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "confidence_level": self.confidence_level.value,
            "overconfidence_bias": round(self.overconfidence_bias, 4),
            "calibration_accuracy": round(self.calibration_accuracy, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ErrorDetectionResult:
    """Result of error detection and analysis."""
    errors_found: List[ErrorSignal]
    total_error_count: int
    average_severity: float
    most_common_type: Optional[ErrorType]
    correction_suggestions: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "errors_found": [e.to_dict() for e in self.errors_found],
            "total_error_count": self.total_error_count,
            "average_severity": round(self.average_severity, 4),
            "most_common_type": self.most_common_type.value if self.most_common_type else None,
            "correction_suggestions": self.correction_suggestions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetaOutput:
    """Complete output from metacognitive processing."""
    awareness: AwarenessAssessment
    confidence_calibration: Optional[ConfidenceCalibration]
    error_detection: ErrorDetectionResult
    strategy_recommendation: Optional[CognitiveStrategy]
    cognitive_load: float                # 0.0-1.0 overall load
    processing_efficiency: float         # 0.0-1.0 how efficiently resources are used
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "awareness": self.awareness.to_dict(),
            "confidence_calibration": self.confidence_calibration.to_dict() if self.confidence_calibration else None,
            "error_detection": self.error_detection.to_dict(),
            "strategy_recommendation": self.strategy_recommendation.value if self.strategy_recommendation else None,
            "cognitive_load": round(self.cognitive_load, 4),
            "processing_efficiency": round(self.processing_efficiency, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetaSystemStatus:
    """Complete metacognitive system status."""
    current_monitoring_level: MonitoringLevel
    awareness_clarity: float
    active_processes_monitored: int
    errors_detected_total: int
    calibration_accuracy: float
    system_health: float                 # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# METACOGNITIVE MONITORING ENGINE
# ============================================================================

class MetaCognitiveMonitoringEngine:
    """
    Engine for monitoring cognitive processes and assessing awareness.

    Implements metacognitive monitoring at multiple levels,
    from surface awareness to recursive self-reflection.
    """

    MONITORING_COSTS = {
        MonitoringLevel.SURFACE: 0.05,
        MonitoringLevel.SHALLOW: 0.1,
        MonitoringLevel.MODERATE: 0.2,
        MonitoringLevel.DEEP: 0.35,
        MonitoringLevel.RECURSIVE: 0.5,
    }

    def __init__(self):
        self._monitoring_history: List[AwarenessAssessment] = []
        self._max_history = 50

    def assess_awareness(
        self,
        cognitive_states: List[CognitiveStateReport],
        monitoring_depth: MonitoringLevel,
    ) -> AwarenessAssessment:
        """Assess current level of metacognitive awareness."""
        clarity = self._compute_clarity(cognitive_states, monitoring_depth)
        accuracy = self._compute_accuracy(cognitive_states)
        depth = self._compute_depth(monitoring_depth)
        recursive = monitoring_depth == MonitoringLevel.RECURSIVE

        assessment = AwarenessAssessment(
            monitoring_level=monitoring_depth,
            awareness_clarity=clarity,
            metacognitive_accuracy=accuracy,
            introspective_depth=depth,
            recursive_awareness=recursive,
        )

        self._monitoring_history.append(assessment)
        if len(self._monitoring_history) > self._max_history:
            self._monitoring_history.pop(0)

        return assessment

    def compute_cognitive_load(self, cognitive_states: List[CognitiveStateReport]) -> float:
        """Compute overall cognitive load from state reports."""
        if not cognitive_states:
            return 0.0
        return sum(s.resource_usage for s in cognitive_states) / len(cognitive_states)

    def compute_efficiency(self, cognitive_states: List[CognitiveStateReport]) -> float:
        """Compute processing efficiency."""
        if not cognitive_states:
            return 1.0
        quality_sum = sum(s.output_quality for s in cognitive_states)
        load_sum = sum(s.resource_usage for s in cognitive_states)
        if load_sum == 0:
            return 1.0
        return min(1.0, quality_sum / max(0.01, load_sum))

    def recommend_strategy(
        self, cognitive_states: List[CognitiveStateReport]
    ) -> Optional[CognitiveStrategy]:
        """Recommend a cognitive strategy based on current state."""
        if not cognitive_states:
            return None

        avg_difficulty = sum(s.difficulty_rating for s in cognitive_states) / len(cognitive_states)
        avg_quality = sum(s.output_quality for s in cognitive_states) / len(cognitive_states)
        total_errors = sum(s.errors_detected for s in cognitive_states)

        if total_errors > 2:
            return CognitiveStrategy.SYSTEMATIC
        elif avg_difficulty > 0.7 and avg_quality < 0.5:
            return CognitiveStrategy.ANALYTICAL
        elif avg_difficulty < 0.3:
            return CognitiveStrategy.INTUITIVE
        elif avg_quality > 0.8:
            return CognitiveStrategy.CREATIVE
        elif avg_difficulty > 0.5:
            return CognitiveStrategy.FOCUSED
        return None

    def _compute_clarity(
        self, states: List[CognitiveStateReport], depth: MonitoringLevel
    ) -> float:
        """Compute how clearly cognitive processes are perceived."""
        depth_factor = {
            MonitoringLevel.SURFACE: 0.3,
            MonitoringLevel.SHALLOW: 0.5,
            MonitoringLevel.MODERATE: 0.7,
            MonitoringLevel.DEEP: 0.85,
            MonitoringLevel.RECURSIVE: 0.95,
        }
        base = depth_factor.get(depth, 0.5)

        if states:
            avg_quality = sum(s.output_quality for s in states) / len(states)
            return (base + avg_quality) / 2
        return base

    def _compute_accuracy(self, states: List[CognitiveStateReport]) -> float:
        """Compute metacognitive accuracy."""
        if not states:
            return 0.7
        error_penalty = sum(s.errors_detected for s in states) * 0.1
        avg_quality = sum(s.output_quality for s in states) / len(states)
        return max(0.0, min(1.0, avg_quality - error_penalty))

    def _compute_depth(self, depth: MonitoringLevel) -> float:
        """Convert monitoring level to depth score."""
        levels = {
            MonitoringLevel.SURFACE: 0.2,
            MonitoringLevel.SHALLOW: 0.4,
            MonitoringLevel.MODERATE: 0.6,
            MonitoringLevel.DEEP: 0.8,
            MonitoringLevel.RECURSIVE: 1.0,
        }
        return levels.get(depth, 0.5)


# ============================================================================
# CONFIDENCE CALIBRATION ENGINE
# ============================================================================

class ConfidenceCalibrationEngine:
    """
    Engine for calibrating confidence estimates.

    Tracks the relationship between stated confidence and actual
    accuracy to detect and correct overconfidence/underconfidence biases.
    """

    def __init__(self):
        self._calibration_history: List[Tuple[float, float]] = []  # (stated, actual)
        self._max_history = 100
        self._bias_estimate: float = 0.0

    def calibrate_confidence(self, report: ConfidenceReport) -> ConfidenceCalibration:
        """Calibrate a confidence report."""
        raw = report.stated_confidence

        # Apply bias correction
        calibrated = self._apply_correction(raw)

        # Classify confidence level
        level = self._classify_confidence(calibrated)

        # Estimate bias
        bias = self._estimate_bias(raw, report)

        # Calibration accuracy based on history
        accuracy = self._compute_calibration_accuracy()

        return ConfidenceCalibration(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_level=level,
            overconfidence_bias=bias,
            calibration_accuracy=accuracy,
        )

    def record_outcome(self, stated_confidence: float, actual_accuracy: float) -> None:
        """Record outcome for calibration improvement."""
        self._calibration_history.append((stated_confidence, actual_accuracy))
        if len(self._calibration_history) > self._max_history:
            self._calibration_history.pop(0)
        self._update_bias_estimate()

    def _apply_correction(self, raw: float) -> float:
        """Apply bias correction to raw confidence."""
        corrected = raw - self._bias_estimate * 0.5
        return max(0.0, min(1.0, corrected))

    def _classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Classify confidence into a level."""
        if confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.4:
            return ConfidenceLevel.LOW
        elif confidence < 0.6:
            return ConfidenceLevel.MODERATE
        elif confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _estimate_bias(self, raw: float, report: ConfidenceReport) -> float:
        """Estimate overconfidence bias."""
        # Heuristic: high confidence with few alternatives considered suggests overconfidence
        alternatives_factor = max(0.0, 1.0 - report.alternatives_considered * 0.2)
        deliberation_factor = max(0.0, 1.0 - report.deliberation_time_ms / 5000.0)
        bias = (raw - 0.5) * (alternatives_factor * 0.5 + deliberation_factor * 0.5)
        return max(-0.5, min(0.5, bias))

    def _compute_calibration_accuracy(self) -> float:
        """Compute how well-calibrated confidence estimates are."""
        if len(self._calibration_history) < 5:
            return 0.5
        errors = [abs(stated - actual) for stated, actual in self._calibration_history[-20:]]
        avg_error = sum(errors) / len(errors)
        return max(0.0, 1.0 - avg_error * 2)

    def _update_bias_estimate(self) -> None:
        """Update running bias estimate."""
        if not self._calibration_history:
            return
        recent = self._calibration_history[-20:]
        biases = [stated - actual for stated, actual in recent]
        self._bias_estimate = sum(biases) / len(biases)


# ============================================================================
# ERROR DETECTION ENGINE
# ============================================================================

class ErrorDetectionEngine:
    """
    Engine for detecting cognitive errors.

    Monitors cognitive outputs for inconsistencies, biases,
    and processing failures.
    """

    def __init__(self):
        self._error_history: List[ErrorSignal] = []
        self._max_history = 100

    def detect_errors(
        self,
        cognitive_states: List[CognitiveStateReport],
        error_signals: List[ErrorSignal],
    ) -> ErrorDetectionResult:
        """Detect and analyze cognitive errors."""
        all_errors = list(error_signals)

        # Detect additional errors from state analysis
        for state in cognitive_states:
            detected = self._analyze_state_for_errors(state)
            all_errors.extend(detected)

        # Compute statistics
        total = len(all_errors)
        avg_severity = sum(e.severity for e in all_errors) / max(1, total)

        # Find most common type
        type_counts: Dict[ErrorType, int] = {}
        for e in all_errors:
            type_counts[e.error_type] = type_counts.get(e.error_type, 0) + 1
        most_common = max(type_counts, key=type_counts.get) if type_counts else None

        # Generate correction suggestions
        suggestions = self._generate_suggestions(all_errors)

        self._error_history.extend(all_errors)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]

        return ErrorDetectionResult(
            errors_found=all_errors,
            total_error_count=total,
            average_severity=avg_severity,
            most_common_type=most_common,
            correction_suggestions=suggestions,
        )

    def _analyze_state_for_errors(self, state: CognitiveStateReport) -> List[ErrorSignal]:
        """Analyze a cognitive state for potential errors."""
        errors = []

        # High difficulty with high quality might indicate overconfidence
        if state.difficulty_rating > 0.7 and state.output_quality > 0.9:
            errors.append(ErrorSignal(
                error_id=f"err_{state.process_id}_overconfidence",
                error_type=ErrorType.BIAS,
                severity=0.4,
                source_process=state.process_id,
                description="Possible overconfidence: high quality claimed despite high difficulty",
            ))

        # Very low quality suggests fundamental error
        if state.output_quality < 0.2 and state.progress > 0.5:
            errors.append(ErrorSignal(
                error_id=f"err_{state.process_id}_quality",
                error_type=ErrorType.REASONING,
                severity=0.7,
                source_process=state.process_id,
                description="Very low output quality despite significant progress",
            ))

        # Resource overuse
        if state.resource_usage > 0.9 and state.output_quality < 0.5:
            errors.append(ErrorSignal(
                error_id=f"err_{state.process_id}_resource",
                error_type=ErrorType.ATTENTION,
                severity=0.5,
                source_process=state.process_id,
                description="High resource usage with low output quality",
            ))

        return errors

    def _generate_suggestions(self, errors: List[ErrorSignal]) -> List[str]:
        """Generate correction suggestions based on detected errors."""
        suggestions = []
        error_types = set(e.error_type for e in errors)

        if ErrorType.BIAS in error_types:
            suggestions.append("Consider alternative perspectives to reduce bias")
        if ErrorType.REASONING in error_types:
            suggestions.append("Re-examine reasoning steps for logical errors")
        if ErrorType.ATTENTION in error_types:
            suggestions.append("Refocus attention and reduce cognitive load")
        if ErrorType.MEMORY in error_types:
            suggestions.append("Verify memory retrieval against available evidence")
        if ErrorType.INCONSISTENCY in error_types:
            suggestions.append("Check for contradictions between beliefs or outputs")

        return suggestions


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class MetaConsciousnessInterface:
    """
    Main interface for Form 11: Meta-Consciousness.

    Provides awareness of awareness, metacognitive monitoring,
    confidence calibration, and error detection.
    """

    FORM_ID = "11-meta-consciousness"
    FORM_NAME = "Meta-Consciousness"

    def __init__(self):
        """Initialize the meta-consciousness interface."""
        self.monitoring_engine = MetaCognitiveMonitoringEngine()
        self.calibration_engine = ConfidenceCalibrationEngine()
        self.error_engine = ErrorDetectionEngine()

        self._current_output: Optional[MetaOutput] = None
        self._current_awareness: Optional[AwarenessAssessment] = None
        self._monitoring_level: MonitoringLevel = MonitoringLevel.MODERATE
        self._total_errors_detected: int = 0
        self._initialized: bool = False

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the meta-consciousness system."""
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized")

    async def process_metacognition(self, meta_input: MetaInput) -> MetaOutput:
        """
        Process metacognitive input and generate output.

        This is the main entry point for metacognitive processing.
        """
        self._monitoring_level = meta_input.monitoring_depth

        # Assess awareness
        awareness = self.monitoring_engine.assess_awareness(
            meta_input.cognitive_states, meta_input.monitoring_depth
        )
        self._current_awareness = awareness

        # Calibrate confidence
        calibration = None
        if meta_input.confidence_reports:
            calibration = self.calibration_engine.calibrate_confidence(
                meta_input.confidence_reports[0]
            )

        # Detect errors
        error_result = self.error_engine.detect_errors(
            meta_input.cognitive_states, meta_input.error_signals
        )
        self._total_errors_detected += error_result.total_error_count

        # Compute load and efficiency
        load = self.monitoring_engine.compute_cognitive_load(meta_input.cognitive_states)
        efficiency = self.monitoring_engine.compute_efficiency(meta_input.cognitive_states)

        # Strategy recommendation
        strategy = self.monitoring_engine.recommend_strategy(meta_input.cognitive_states)

        output = MetaOutput(
            awareness=awareness,
            confidence_calibration=calibration,
            error_detection=error_result,
            strategy_recommendation=strategy,
            cognitive_load=load,
            processing_efficiency=efficiency,
        )
        self._current_output = output
        return output

    async def introspect(self, question: str) -> str:
        """Perform introspection on a specific question about own processing."""
        if not self._current_awareness:
            return "Insufficient metacognitive state for introspection."

        clarity = self._current_awareness.awareness_clarity
        if clarity > 0.7:
            return f"Clear introspective access: monitoring at {self._monitoring_level.value} level with {clarity:.0%} clarity."
        elif clarity > 0.4:
            return f"Partial introspective access: monitoring at {self._monitoring_level.value} level with {clarity:.0%} clarity."
        else:
            return f"Limited introspective access: monitoring at {self._monitoring_level.value} level with {clarity:.0%} clarity."

    def get_awareness(self) -> Optional[AwarenessAssessment]:
        """Get current awareness assessment."""
        return self._current_awareness

    def get_status(self) -> MetaSystemStatus:
        """Get complete metacognitive system status."""
        clarity = 0.5
        calibration = 0.5
        if self._current_awareness:
            clarity = self._current_awareness.awareness_clarity
        if self._current_output and self._current_output.confidence_calibration:
            calibration = self._current_output.confidence_calibration.calibration_accuracy

        return MetaSystemStatus(
            current_monitoring_level=self._monitoring_level,
            awareness_clarity=clarity,
            active_processes_monitored=0,
            errors_detected_total=self._total_errors_detected,
            calibration_accuracy=calibration,
            system_health=self._compute_health(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "monitoring_level": self._monitoring_level.value,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "total_errors_detected": self._total_errors_detected,
            "initialized": self._initialized,
        }

    def _compute_health(self) -> float:
        """Compute metacognitive system health."""
        if not self._current_awareness:
            return 1.0
        return (
            self._current_awareness.awareness_clarity * 0.4 +
            self._current_awareness.metacognitive_accuracy * 0.4 +
            self._current_awareness.introspective_depth * 0.2
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_meta_consciousness_interface() -> MetaConsciousnessInterface:
    """Create and return a meta-consciousness interface."""
    return MetaConsciousnessInterface()


__all__ = [
    # Enums
    "MetaCognitiveProcess",
    "MonitoringLevel",
    "ConfidenceLevel",
    "ErrorType",
    "CognitiveStrategy",
    # Input dataclasses
    "CognitiveStateReport",
    "ConfidenceReport",
    "ErrorSignal",
    "MetaInput",
    # Output dataclasses
    "AwarenessAssessment",
    "ConfidenceCalibration",
    "ErrorDetectionResult",
    "MetaOutput",
    "MetaSystemStatus",
    # Engines
    "MetaCognitiveMonitoringEngine",
    "ConfidenceCalibrationEngine",
    "ErrorDetectionEngine",
    # Main interface
    "MetaConsciousnessInterface",
    # Convenience
    "create_meta_consciousness_interface",
]
