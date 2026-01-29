#!/usr/bin/env python3
"""
Test Suite for Form 11: Meta-Consciousness.

Tests cover:
- All enumerations (MetaCognitiveProcess, MonitoringLevel, ConfidenceLevel, etc.)
- All input/output dataclasses
- MetaCognitiveMonitoringEngine
- ConfidenceCalibrationEngine
- ErrorDetectionEngine
- MetaConsciousnessInterface (main interface)
- Convenience functions
"""

import asyncio
import unittest
from datetime import datetime, timezone

import sys
from pathlib import Path

# Add parent paths to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interface import (
    # Enums
    MetaCognitiveProcess,
    MonitoringLevel,
    ConfidenceLevel,
    ErrorType,
    CognitiveStrategy,
    # Input dataclasses
    CognitiveStateReport,
    ConfidenceReport,
    ErrorSignal,
    MetaInput,
    # Output dataclasses
    AwarenessAssessment,
    ConfidenceCalibration,
    ErrorDetectionResult,
    MetaOutput,
    MetaSystemStatus,
    # Engines
    MetaCognitiveMonitoringEngine,
    ConfidenceCalibrationEngine,
    ErrorDetectionEngine,
    # Main interface
    MetaConsciousnessInterface,
    # Convenience
    create_meta_consciousness_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestMetaCognitiveProcess(unittest.TestCase):
    """Tests for MetaCognitiveProcess enumeration."""

    def test_all_processes_exist(self):
        """All metacognitive processes should be defined."""
        processes = [
            MetaCognitiveProcess.MONITORING,
            MetaCognitiveProcess.EVALUATION,
            MetaCognitiveProcess.PLANNING,
            MetaCognitiveProcess.REGULATION,
            MetaCognitiveProcess.REFLECTION,
            MetaCognitiveProcess.CALIBRATION,
        ]
        self.assertEqual(len(processes), 6)

    def test_process_values(self):
        """Processes should have expected string values."""
        self.assertEqual(MetaCognitiveProcess.MONITORING.value, "monitoring")
        self.assertEqual(MetaCognitiveProcess.REFLECTION.value, "reflection")


class TestMonitoringLevel(unittest.TestCase):
    """Tests for MonitoringLevel enumeration."""

    def test_all_levels_exist(self):
        """All monitoring levels should be defined."""
        levels = [
            MonitoringLevel.SURFACE,
            MonitoringLevel.SHALLOW,
            MonitoringLevel.MODERATE,
            MonitoringLevel.DEEP,
            MonitoringLevel.RECURSIVE,
        ]
        self.assertEqual(len(levels), 5)

    def test_level_values(self):
        """Levels should have expected string values."""
        self.assertEqual(MonitoringLevel.RECURSIVE.value, "recursive")


class TestConfidenceLevel(unittest.TestCase):
    """Tests for ConfidenceLevel enumeration."""

    def test_all_levels_exist(self):
        """All confidence levels should be defined."""
        levels = [
            ConfidenceLevel.VERY_LOW,
            ConfidenceLevel.LOW,
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]
        self.assertEqual(len(levels), 5)


class TestErrorType(unittest.TestCase):
    """Tests for ErrorType enumeration."""

    def test_all_types_exist(self):
        """All error types should be defined."""
        types = [
            ErrorType.REASONING,
            ErrorType.MEMORY,
            ErrorType.PERCEPTION,
            ErrorType.ATTENTION,
            ErrorType.JUDGMENT,
            ErrorType.BIAS,
            ErrorType.INCONSISTENCY,
        ]
        self.assertEqual(len(types), 7)


class TestCognitiveStrategy(unittest.TestCase):
    """Tests for CognitiveStrategy enumeration."""

    def test_all_strategies_exist(self):
        """All cognitive strategies should be defined."""
        strategies = [
            CognitiveStrategy.ANALYTICAL,
            CognitiveStrategy.INTUITIVE,
            CognitiveStrategy.SYSTEMATIC,
            CognitiveStrategy.CREATIVE,
            CognitiveStrategy.FOCUSED,
            CognitiveStrategy.EXPLORATORY,
            CognitiveStrategy.REFLECTIVE,
        ]
        self.assertEqual(len(strategies), 7)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestCognitiveStateReport(unittest.TestCase):
    """Tests for CognitiveStateReport dataclass."""

    def test_creation(self):
        """Should create cognitive state report."""
        report = CognitiveStateReport(
            process_id="proc_001",
            process_type="reasoning",
            current_strategy=CognitiveStrategy.ANALYTICAL,
            progress=0.6,
            difficulty_rating=0.7,
            resource_usage=0.5,
            output_quality=0.8,
            time_elapsed_ms=1500.0,
        )
        self.assertEqual(report.process_id, "proc_001")
        self.assertEqual(report.current_strategy, CognitiveStrategy.ANALYTICAL)
        self.assertEqual(report.errors_detected, 0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = CognitiveStateReport(
            process_id="p1",
            process_type="memory",
            current_strategy=CognitiveStrategy.FOCUSED,
            progress=0.5,
            difficulty_rating=0.3,
            resource_usage=0.4,
            output_quality=0.7,
            time_elapsed_ms=500.0,
        )
        d = report.to_dict()
        self.assertEqual(d["process_id"], "p1")
        self.assertEqual(d["current_strategy"], "focused")


class TestConfidenceReport(unittest.TestCase):
    """Tests for ConfidenceReport dataclass."""

    def test_creation(self):
        """Should create confidence report."""
        report = ConfidenceReport(
            judgment_id="j001",
            stated_confidence=0.85,
            basis="evidence",
            alternatives_considered=3,
            deliberation_time_ms=2000.0,
        )
        self.assertEqual(report.stated_confidence, 0.85)
        self.assertEqual(report.basis, "evidence")
        self.assertEqual(report.alternatives_considered, 3)


class TestErrorSignal(unittest.TestCase):
    """Tests for ErrorSignal dataclass."""

    def test_creation(self):
        """Should create error signal."""
        error = ErrorSignal(
            error_id="err_001",
            error_type=ErrorType.REASONING,
            severity=0.6,
            source_process="proc_001",
            description="Logical inconsistency detected",
        )
        self.assertEqual(error.error_type, ErrorType.REASONING)
        self.assertTrue(error.correctable)

    def test_to_dict(self):
        """Should convert to dictionary."""
        error = ErrorSignal(
            error_id="err_002",
            error_type=ErrorType.BIAS,
            severity=0.4,
            source_process="proc_002",
            description="Confirmation bias detected",
            correctable=True,
        )
        d = error.to_dict()
        self.assertEqual(d["error_type"], "bias")
        self.assertEqual(d["severity"], 0.4)


class TestMetaInput(unittest.TestCase):
    """Tests for MetaInput dataclass."""

    def test_empty_input(self):
        """Should create empty input."""
        inp = MetaInput()
        self.assertEqual(len(inp.cognitive_states), 0)
        self.assertEqual(inp.monitoring_depth, MonitoringLevel.MODERATE)

    def test_full_input(self):
        """Should create input with all components."""
        inp = MetaInput(
            cognitive_states=[
                CognitiveStateReport(
                    process_id="p1",
                    process_type="reasoning",
                    current_strategy=CognitiveStrategy.ANALYTICAL,
                    progress=0.5,
                    difficulty_rating=0.6,
                    resource_usage=0.4,
                    output_quality=0.7,
                    time_elapsed_ms=1000.0,
                ),
            ],
            confidence_reports=[
                ConfidenceReport(
                    judgment_id="j1",
                    stated_confidence=0.8,
                    basis="evidence",
                    alternatives_considered=2,
                    deliberation_time_ms=1500.0,
                ),
            ],
            monitoring_depth=MonitoringLevel.DEEP,
        )
        self.assertEqual(len(inp.cognitive_states), 1)
        self.assertEqual(len(inp.confidence_reports), 1)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestAwarenessAssessment(unittest.TestCase):
    """Tests for AwarenessAssessment dataclass."""

    def test_creation(self):
        """Should create awareness assessment."""
        assessment = AwarenessAssessment(
            monitoring_level=MonitoringLevel.DEEP,
            awareness_clarity=0.85,
            metacognitive_accuracy=0.8,
            introspective_depth=0.8,
            recursive_awareness=False,
        )
        self.assertEqual(assessment.monitoring_level, MonitoringLevel.DEEP)
        self.assertFalse(assessment.recursive_awareness)

    def test_to_dict(self):
        """Should convert to dictionary."""
        assessment = AwarenessAssessment(
            monitoring_level=MonitoringLevel.RECURSIVE,
            awareness_clarity=0.9,
            metacognitive_accuracy=0.85,
            introspective_depth=1.0,
            recursive_awareness=True,
        )
        d = assessment.to_dict()
        self.assertEqual(d["monitoring_level"], "recursive")
        self.assertTrue(d["recursive_awareness"])


class TestConfidenceCalibration(unittest.TestCase):
    """Tests for ConfidenceCalibration dataclass."""

    def test_creation(self):
        """Should create confidence calibration."""
        cal = ConfidenceCalibration(
            raw_confidence=0.9,
            calibrated_confidence=0.75,
            confidence_level=ConfidenceLevel.HIGH,
            overconfidence_bias=0.15,
            calibration_accuracy=0.7,
        )
        self.assertEqual(cal.raw_confidence, 0.9)
        self.assertEqual(cal.calibrated_confidence, 0.75)

    def test_to_dict(self):
        """Should convert to dictionary."""
        cal = ConfidenceCalibration(
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            confidence_level=ConfidenceLevel.MODERATE,
            overconfidence_bias=0.0,
            calibration_accuracy=0.8,
        )
        d = cal.to_dict()
        self.assertEqual(d["confidence_level"], "moderate")


class TestErrorDetectionResult(unittest.TestCase):
    """Tests for ErrorDetectionResult dataclass."""

    def test_creation_no_errors(self):
        """Should create result with no errors."""
        result = ErrorDetectionResult(
            errors_found=[],
            total_error_count=0,
            average_severity=0.0,
            most_common_type=None,
            correction_suggestions=[],
        )
        self.assertEqual(result.total_error_count, 0)

    def test_to_dict_with_errors(self):
        """Should convert to dictionary with errors."""
        result = ErrorDetectionResult(
            errors_found=[
                ErrorSignal(
                    error_id="e1",
                    error_type=ErrorType.BIAS,
                    severity=0.5,
                    source_process="p1",
                    description="Bias detected",
                ),
            ],
            total_error_count=1,
            average_severity=0.5,
            most_common_type=ErrorType.BIAS,
            correction_suggestions=["Consider alternatives"],
        )
        d = result.to_dict()
        self.assertEqual(d["total_error_count"], 1)
        self.assertEqual(d["most_common_type"], "bias")


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestMetaCognitiveMonitoringEngine(unittest.TestCase):
    """Tests for MetaCognitiveMonitoringEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = MetaCognitiveMonitoringEngine()

    def test_assess_awareness_surface(self):
        """Should assess surface awareness."""
        assessment = self.engine.assess_awareness([], MonitoringLevel.SURFACE)
        self.assertEqual(assessment.monitoring_level, MonitoringLevel.SURFACE)
        self.assertLessEqual(assessment.introspective_depth, 0.3)

    def test_assess_awareness_recursive(self):
        """Should assess recursive awareness."""
        assessment = self.engine.assess_awareness([], MonitoringLevel.RECURSIVE)
        self.assertTrue(assessment.recursive_awareness)
        self.assertGreater(assessment.introspective_depth, 0.8)

    def test_compute_cognitive_load(self):
        """Should compute cognitive load."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.ANALYTICAL,
                progress=0.5, difficulty_rating=0.6,
                resource_usage=0.7, output_quality=0.8,
                time_elapsed_ms=1000.0,
            ),
            CognitiveStateReport(
                process_id="p2", process_type="memory",
                current_strategy=CognitiveStrategy.FOCUSED,
                progress=0.3, difficulty_rating=0.4,
                resource_usage=0.5, output_quality=0.6,
                time_elapsed_ms=800.0,
            ),
        ]
        load = self.engine.compute_cognitive_load(states)
        self.assertAlmostEqual(load, 0.6, places=1)

    def test_recommend_strategy_high_errors(self):
        """Should recommend systematic strategy for high errors."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.INTUITIVE,
                progress=0.5, difficulty_rating=0.8,
                resource_usage=0.6, output_quality=0.3,
                time_elapsed_ms=2000.0,
                errors_detected=3,
            ),
        ]
        strategy = self.engine.recommend_strategy(states)
        self.assertEqual(strategy, CognitiveStrategy.SYSTEMATIC)

    def test_recommend_strategy_easy_task(self):
        """Should recommend intuitive for easy tasks."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="recognition",
                current_strategy=CognitiveStrategy.ANALYTICAL,
                progress=0.8, difficulty_rating=0.2,
                resource_usage=0.3, output_quality=0.9,
                time_elapsed_ms=200.0,
            ),
        ]
        strategy = self.engine.recommend_strategy(states)
        self.assertEqual(strategy, CognitiveStrategy.INTUITIVE)

    def test_compute_efficiency(self):
        """Should compute processing efficiency."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.ANALYTICAL,
                progress=0.8, difficulty_rating=0.5,
                resource_usage=0.3, output_quality=0.9,
                time_elapsed_ms=1000.0,
            ),
        ]
        efficiency = self.engine.compute_efficiency(states)
        self.assertGreater(efficiency, 0.5)


class TestConfidenceCalibrationEngine(unittest.TestCase):
    """Tests for ConfidenceCalibrationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ConfidenceCalibrationEngine()

    def test_calibrate_confidence(self):
        """Should calibrate a confidence report."""
        report = ConfidenceReport(
            judgment_id="j1",
            stated_confidence=0.9,
            basis="evidence",
            alternatives_considered=3,
            deliberation_time_ms=2000.0,
        )
        calibration = self.engine.calibrate_confidence(report)
        self.assertIsInstance(calibration, ConfidenceCalibration)
        self.assertEqual(calibration.raw_confidence, 0.9)

    def test_classify_confidence_levels(self):
        """Should correctly classify confidence levels."""
        for conf, expected in [
            (0.1, ConfidenceLevel.VERY_LOW),
            (0.3, ConfidenceLevel.LOW),
            (0.5, ConfidenceLevel.MODERATE),
            (0.7, ConfidenceLevel.HIGH),
            (0.9, ConfidenceLevel.VERY_HIGH),
        ]:
            report = ConfidenceReport(
                judgment_id="j",
                stated_confidence=conf,
                basis="evidence",
                alternatives_considered=2,
                deliberation_time_ms=1000.0,
            )
            cal = self.engine.calibrate_confidence(report)
            self.assertEqual(cal.confidence_level, expected,
                             f"Expected {expected} for confidence {conf}, got {cal.confidence_level}")

    def test_record_outcome(self):
        """Should record outcome for calibration."""
        self.engine.record_outcome(0.9, 0.6)
        self.engine.record_outcome(0.8, 0.5)
        # Bias should be positive (overconfident)
        self.assertGreater(self.engine._bias_estimate, 0.0)


class TestErrorDetectionEngine(unittest.TestCase):
    """Tests for ErrorDetectionEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ErrorDetectionEngine()

    def test_detect_no_errors(self):
        """Should detect no errors in clean state."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.ANALYTICAL,
                progress=0.5, difficulty_rating=0.5,
                resource_usage=0.4, output_quality=0.7,
                time_elapsed_ms=1000.0,
            ),
        ]
        result = self.engine.detect_errors(states, [])
        self.assertEqual(result.total_error_count, 0)

    def test_detect_overconfidence_error(self):
        """Should detect overconfidence pattern."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.INTUITIVE,
                progress=0.7, difficulty_rating=0.8,
                resource_usage=0.6, output_quality=0.95,
                time_elapsed_ms=500.0,
            ),
        ]
        result = self.engine.detect_errors(states, [])
        bias_errors = [e for e in result.errors_found if e.error_type == ErrorType.BIAS]
        self.assertGreater(len(bias_errors), 0)

    def test_detect_quality_error(self):
        """Should detect low quality error."""
        states = [
            CognitiveStateReport(
                process_id="p1", process_type="reasoning",
                current_strategy=CognitiveStrategy.ANALYTICAL,
                progress=0.8, difficulty_rating=0.9,
                resource_usage=0.7, output_quality=0.1,
                time_elapsed_ms=3000.0,
            ),
        ]
        result = self.engine.detect_errors(states, [])
        reasoning_errors = [e for e in result.errors_found if e.error_type == ErrorType.REASONING]
        self.assertGreater(len(reasoning_errors), 0)

    def test_pass_through_error_signals(self):
        """Should include passed-in error signals."""
        error = ErrorSignal(
            error_id="ext_001",
            error_type=ErrorType.MEMORY,
            severity=0.5,
            source_process="external",
            description="Memory retrieval failure",
        )
        result = self.engine.detect_errors([], [error])
        self.assertGreaterEqual(result.total_error_count, 1)

    def test_generate_suggestions(self):
        """Should generate correction suggestions."""
        errors = [
            ErrorSignal(
                error_id="e1", error_type=ErrorType.BIAS,
                severity=0.5, source_process="p1",
                description="Bias",
            ),
            ErrorSignal(
                error_id="e2", error_type=ErrorType.ATTENTION,
                severity=0.4, source_process="p2",
                description="Lapse",
            ),
        ]
        result = self.engine.detect_errors([], errors)
        self.assertGreater(len(result.correction_suggestions), 0)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestMetaConsciousnessInterface(unittest.TestCase):
    """Tests for MetaConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = MetaConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "11-meta-consciousness")
        self.assertEqual(self.interface.FORM_NAME, "Meta-Consciousness")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_metacognition(self):
        """Should process metacognitive input."""
        inp = MetaInput(
            cognitive_states=[
                CognitiveStateReport(
                    process_id="p1",
                    process_type="reasoning",
                    current_strategy=CognitiveStrategy.ANALYTICAL,
                    progress=0.6,
                    difficulty_rating=0.5,
                    resource_usage=0.4,
                    output_quality=0.8,
                    time_elapsed_ms=1200.0,
                ),
            ],
            confidence_reports=[
                ConfidenceReport(
                    judgment_id="j1",
                    stated_confidence=0.7,
                    basis="evidence",
                    alternatives_considered=2,
                    deliberation_time_ms=1000.0,
                ),
            ],
            monitoring_depth=MonitoringLevel.DEEP,
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_metacognition(inp)
            )
        finally:
            loop.close()

        self.assertIsInstance(output, MetaOutput)
        self.assertIsNotNone(output.awareness)
        self.assertIsNotNone(output.confidence_calibration)
        self.assertIsNotNone(output.error_detection)

    def test_introspect(self):
        """Should perform introspection."""
        # First process something to build state
        inp = MetaInput(
            cognitive_states=[
                CognitiveStateReport(
                    process_id="p1", process_type="reasoning",
                    current_strategy=CognitiveStrategy.ANALYTICAL,
                    progress=0.5, difficulty_rating=0.5,
                    resource_usage=0.4, output_quality=0.8,
                    time_elapsed_ms=1000.0,
                ),
            ],
            monitoring_depth=MonitoringLevel.DEEP,
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_metacognition(inp))
            result = loop.run_until_complete(
                self.interface.introspect("What is my current processing quality?")
            )
        finally:
            loop.close()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, MetaSystemStatus)
        self.assertGreaterEqual(status.system_health, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "11-meta-consciousness")
        self.assertEqual(d["form_name"], "Meta-Consciousness")


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_meta_consciousness_interface(self):
        """Should create new interface."""
        interface = create_meta_consciousness_interface()
        self.assertIsInstance(interface, MetaConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "11-meta-consciousness")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
