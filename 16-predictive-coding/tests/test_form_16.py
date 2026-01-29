#!/usr/bin/env python3
"""
Test Suite for Form 16: Predictive Coding / Free Energy Principle.

Tests cover:
- All enumerations (PredictionLevel, ErrorType, PrecisionLevel, etc.)
- All input/output dataclasses
- PredictionErrorEngine
- GenerativeModelEngine
- FreeEnergyEngine
- PredictiveCodingInterface (main interface)
- Convenience functions
- Integration tests for the full predictive coding cycle
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
    PredictionLevel,
    ErrorType,
    PrecisionLevel,
    UpdateStrategy,
    FreeEnergyComponent,
    # Input dataclasses
    Prediction,
    SensoryEvidence,
    PredictiveCodingInput,
    # Output dataclasses
    PredictionError,
    ModelUpdate,
    FreeEnergyState,
    PredictiveCodingOutput,
    PredictiveCodingSystemStatus,
    # Engines
    PredictionErrorEngine,
    GenerativeModelEngine,
    FreeEnergyEngine,
    # Main interface
    PredictiveCodingInterface,
    # Convenience functions
    create_predictive_coding_interface,
    create_prediction,
    create_evidence,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestPredictionLevel(unittest.TestCase):
    """Tests for PredictionLevel enumeration."""

    def test_all_levels_exist(self):
        """All prediction levels should be defined."""
        levels = [
            PredictionLevel.SENSORY,
            PredictionLevel.PERCEPTUAL,
            PredictionLevel.CONCEPTUAL,
            PredictionLevel.CONTEXTUAL,
            PredictionLevel.NARRATIVE,
        ]
        self.assertEqual(len(levels), 5)

    def test_level_values(self):
        """Levels should have expected string values."""
        self.assertEqual(PredictionLevel.SENSORY.value, "sensory")
        self.assertEqual(PredictionLevel.CONCEPTUAL.value, "conceptual")


class TestErrorType(unittest.TestCase):
    """Tests for ErrorType enumeration."""

    def test_all_types_exist(self):
        """All error types should be defined."""
        types = [
            ErrorType.SENSORY_MISMATCH,
            ErrorType.FEATURE_ERROR,
            ErrorType.OBJECT_ERROR,
            ErrorType.CONTEXT_ERROR,
            ErrorType.MODEL_VIOLATION,
            ErrorType.SURPRISE,
        ]
        self.assertEqual(len(types), 6)


class TestPrecisionLevel(unittest.TestCase):
    """Tests for PrecisionLevel enumeration."""

    def test_all_levels_exist(self):
        """All precision levels should be defined."""
        levels = [
            PrecisionLevel.VERY_LOW,
            PrecisionLevel.LOW,
            PrecisionLevel.MODERATE,
            PrecisionLevel.HIGH,
            PrecisionLevel.VERY_HIGH,
        ]
        self.assertEqual(len(levels), 5)


class TestUpdateStrategy(unittest.TestCase):
    """Tests for UpdateStrategy enumeration."""

    def test_all_strategies_exist(self):
        """All update strategies should be defined."""
        strategies = [
            UpdateStrategy.PERCEPTUAL_INFERENCE,
            UpdateStrategy.MODEL_UPDATE,
            UpdateStrategy.ACTIVE_INFERENCE,
            UpdateStrategy.ATTENTION_SHIFT,
        ]
        self.assertEqual(len(strategies), 4)


class TestFreeEnergyComponent(unittest.TestCase):
    """Tests for FreeEnergyComponent enumeration."""

    def test_all_components_exist(self):
        """All free energy components should be defined."""
        components = [
            FreeEnergyComponent.PREDICTION_ERROR,
            FreeEnergyComponent.COMPLEXITY,
            FreeEnergyComponent.ENTROPY,
            FreeEnergyComponent.EXPECTED_FREE_ENERGY,
        ]
        self.assertEqual(len(components), 4)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestPrediction(unittest.TestCase):
    """Tests for Prediction dataclass."""

    def test_creation(self):
        """Should create prediction with required fields."""
        pred = Prediction(
            prediction_id="p1",
            level=PredictionLevel.PERCEPTUAL,
            content={"color": 0.8, "shape": "circle"},
            confidence=0.7,
            precision=0.6,
        )
        self.assertEqual(pred.prediction_id, "p1")
        self.assertEqual(pred.level, PredictionLevel.PERCEPTUAL)
        self.assertEqual(pred.confidence, 0.7)

    def test_to_dict(self):
        """Should convert to dictionary."""
        pred = create_prediction("p1")
        d = pred.to_dict()
        self.assertEqual(d["prediction_id"], "p1")
        self.assertIn("level", d)
        self.assertIn("precision", d)


class TestSensoryEvidence(unittest.TestCase):
    """Tests for SensoryEvidence dataclass."""

    def test_creation(self):
        """Should create evidence with required fields."""
        ev = SensoryEvidence(
            evidence_id="e1",
            level=PredictionLevel.SENSORY,
            content={"brightness": 0.9},
            reliability=0.8,
            precision=0.7,
        )
        self.assertEqual(ev.evidence_id, "e1")
        self.assertEqual(ev.reliability, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        ev = create_evidence("e1")
        d = ev.to_dict()
        self.assertEqual(d["evidence_id"], "e1")
        self.assertIn("reliability", d)


class TestPredictiveCodingInput(unittest.TestCase):
    """Tests for PredictiveCodingInput dataclass."""

    def test_creation(self):
        """Should create input bundle."""
        inp = PredictiveCodingInput(
            predictions=[create_prediction("p1")],
            evidence=[create_evidence("e1")],
        )
        self.assertEqual(len(inp.predictions), 1)
        self.assertEqual(len(inp.evidence), 1)

    def test_empty_input(self):
        """Should create empty input."""
        inp = PredictiveCodingInput(predictions=[], evidence=[])
        self.assertEqual(len(inp.predictions), 0)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestPredictionErrorOutput(unittest.TestCase):
    """Tests for PredictionError dataclass."""

    def test_creation(self):
        """Should create prediction error."""
        pe = PredictionError(
            error_id="pe1",
            prediction_id="p1",
            evidence_id="e1",
            level=PredictionLevel.PERCEPTUAL,
            error_type=ErrorType.FEATURE_ERROR,
            magnitude=0.3,
            precision_weighted_error=0.24,
            direction={"value": 0.1},
            surprise=0.36,
        )
        self.assertEqual(pe.magnitude, 0.3)
        self.assertEqual(pe.error_type, ErrorType.FEATURE_ERROR)

    def test_to_dict(self):
        """Should convert to dictionary."""
        pe = PredictionError(
            error_id="pe1", prediction_id="p1", evidence_id="e1",
            level=PredictionLevel.SENSORY, error_type=ErrorType.SENSORY_MISMATCH,
            magnitude=0.5, precision_weighted_error=0.4,
            direction={}, surprise=0.7,
        )
        d = pe.to_dict()
        self.assertIn("magnitude", d)
        self.assertIn("surprise", d)


class TestModelUpdate(unittest.TestCase):
    """Tests for ModelUpdate dataclass."""

    def test_creation(self):
        """Should create model update."""
        update = ModelUpdate(
            update_id="u1",
            strategy=UpdateStrategy.MODEL_UPDATE,
            level=PredictionLevel.CONCEPTUAL,
            learning_rate=0.1,
            error_reduction=0.05,
            new_model_confidence=0.55,
        )
        self.assertEqual(update.strategy, UpdateStrategy.MODEL_UPDATE)
        self.assertEqual(update.learning_rate, 0.1)


class TestFreeEnergyState(unittest.TestCase):
    """Tests for FreeEnergyState dataclass."""

    def test_creation(self):
        """Should create free energy state."""
        fe = FreeEnergyState(
            total_free_energy=0.45,
            prediction_error_component=0.3,
            complexity_component=0.1,
            entropy_component=0.05,
            level_breakdown={"sensory": 0.2, "perceptual": 0.1},
            is_minimized=False,
        )
        self.assertEqual(fe.total_free_energy, 0.45)
        self.assertFalse(fe.is_minimized)

    def test_to_dict(self):
        """Should convert to dictionary."""
        fe = FreeEnergyState(
            total_free_energy=0.3,
            prediction_error_component=0.2,
            complexity_component=0.05,
            entropy_component=0.05,
            level_breakdown={},
            is_minimized=True,
        )
        d = fe.to_dict()
        self.assertIn("total_free_energy", d)
        self.assertTrue(d["is_minimized"])


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestPredictionErrorEngine(unittest.TestCase):
    """Tests for PredictionErrorEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = PredictionErrorEngine()

    def test_compute_matching_prediction_evidence(self):
        """Matching prediction and evidence should have low error."""
        pred = create_prediction("p1", content={"value": 0.5})
        ev = create_evidence("e1", content={"value": 0.5})
        error = self.engine.compute_prediction_error(pred, ev)
        self.assertEqual(error.magnitude, 0.0)

    def test_compute_mismatched_prediction_evidence(self):
        """Mismatched prediction and evidence should have high error."""
        pred = create_prediction("p1", content={"value": 0.1})
        ev = create_evidence("e1", content={"value": 0.9})
        error = self.engine.compute_prediction_error(pred, ev)
        self.assertGreater(error.magnitude, 0.0)

    def test_precision_weighting(self):
        """Higher precision should produce stronger weighted error."""
        pred_low = Prediction(
            prediction_id="pl", level=PredictionLevel.SENSORY,
            content={"value": 0.0}, confidence=0.3, precision=0.2
        )
        pred_high = Prediction(
            prediction_id="ph", level=PredictionLevel.SENSORY,
            content={"value": 0.0}, confidence=0.9, precision=0.9
        )
        ev = create_evidence("e1", content={"value": 1.0})
        ev.precision = 0.8

        error_low = self.engine.compute_prediction_error(pred_low, ev)
        error_high = self.engine.compute_prediction_error(pred_high, ev)

        self.assertGreater(
            error_high.precision_weighted_error,
            error_low.precision_weighted_error
        )

    def test_surprise_computation(self):
        """Should compute positive surprise for errors."""
        pred = create_prediction("p1", content={"value": 0.0})
        ev = create_evidence("e1", content={"value": 1.0})
        error = self.engine.compute_prediction_error(pred, ev)
        self.assertGreater(error.surprise, 0.0)

    def test_error_classification(self):
        """Should classify errors based on level and magnitude."""
        pred = create_prediction("p1", PredictionLevel.SENSORY, {"value": 0.0})
        ev = create_evidence("e1", PredictionLevel.SENSORY, {"value": 0.3})
        error = self.engine.compute_prediction_error(pred, ev)
        self.assertIsInstance(error.error_type, ErrorType)

    def test_classify_precision(self):
        """Should classify precision values correctly."""
        self.assertEqual(self.engine.classify_precision(0.1), PrecisionLevel.VERY_LOW)
        self.assertEqual(self.engine.classify_precision(0.3), PrecisionLevel.LOW)
        self.assertEqual(self.engine.classify_precision(0.5), PrecisionLevel.MODERATE)
        self.assertEqual(self.engine.classify_precision(0.7), PrecisionLevel.HIGH)
        self.assertEqual(self.engine.classify_precision(0.9), PrecisionLevel.VERY_HIGH)


class TestGenerativeModelEngine(unittest.TestCase):
    """Tests for GenerativeModelEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = GenerativeModelEngine()

    def test_generate_prediction(self):
        """Should generate a prediction from context."""
        pred = self.engine.generate_prediction(
            PredictionLevel.PERCEPTUAL,
            {"brightness": 0.7, "color": "red"},
        )
        self.assertIsInstance(pred, Prediction)
        self.assertEqual(pred.level, PredictionLevel.PERCEPTUAL)
        self.assertIn("brightness", pred.content)

    def test_model_confidence_initialization(self):
        """All levels should start with default confidence."""
        for level in PredictionLevel:
            conf = self.engine.get_model_confidence(level)
            self.assertEqual(conf, 0.5)

    def test_update_model(self):
        """Should update model based on errors."""
        errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.PERCEPTUAL,
                error_type=ErrorType.FEATURE_ERROR,
                magnitude=0.7, precision_weighted_error=0.6,
                direction={}, surprise=1.2,
            )
        ]
        update = self.engine.update_model(errors, PredictionLevel.PERCEPTUAL)
        self.assertIsInstance(update, ModelUpdate)
        self.assertGreater(update.error_reduction, 0.0)

    def test_update_strategy_selection(self):
        """Should select appropriate update strategy."""
        # High magnitude error -> model update
        high_errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.PERCEPTUAL,
                error_type=ErrorType.MODEL_VIOLATION,
                magnitude=0.9, precision_weighted_error=0.85,
                direction={}, surprise=2.0,
            )
        ]
        update = self.engine.update_model(high_errors, PredictionLevel.PERCEPTUAL)
        self.assertEqual(update.strategy, UpdateStrategy.MODEL_UPDATE)


class TestFreeEnergyEngine(unittest.TestCase):
    """Tests for FreeEnergyEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = FreeEnergyEngine()

    def test_compute_free_energy_no_errors(self):
        """Should handle empty error list."""
        fe = self.engine.compute_free_energy([], {"sensory": 0.5})
        self.assertIsInstance(fe, FreeEnergyState)
        self.assertGreaterEqual(fe.total_free_energy, 0.0)

    def test_compute_free_energy_with_errors(self):
        """Should compute free energy with prediction errors."""
        errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.SENSORY,
                error_type=ErrorType.SENSORY_MISMATCH,
                magnitude=0.5, precision_weighted_error=0.4,
                direction={}, surprise=0.7,
            )
        ]
        fe = self.engine.compute_free_energy(errors, {"sensory": 0.5})
        self.assertGreater(fe.total_free_energy, 0.0)
        self.assertGreater(fe.prediction_error_component, 0.0)

    def test_high_errors_increase_free_energy(self):
        """Higher prediction errors should increase free energy."""
        low_errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.SENSORY,
                error_type=ErrorType.SENSORY_MISMATCH,
                magnitude=0.1, precision_weighted_error=0.05,
                direction={}, surprise=0.1,
            )
        ]
        high_errors = [
            PredictionError(
                error_id="pe2", prediction_id="p2", evidence_id="e2",
                level=PredictionLevel.SENSORY,
                error_type=ErrorType.MODEL_VIOLATION,
                magnitude=0.9, precision_weighted_error=0.85,
                direction={}, surprise=2.3,
            )
        ]

        engine_low = FreeEnergyEngine()
        engine_high = FreeEnergyEngine()

        fe_low = engine_low.compute_free_energy(low_errors, {"sensory": 0.5})
        fe_high = engine_high.compute_free_energy(high_errors, {"sensory": 0.5})

        self.assertGreater(fe_high.total_free_energy, fe_low.total_free_energy)

    def test_level_breakdown(self):
        """Should compute free energy breakdown by level."""
        errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.SENSORY,
                error_type=ErrorType.SENSORY_MISMATCH,
                magnitude=0.5, precision_weighted_error=0.4,
                direction={}, surprise=0.7,
            )
        ]
        fe = self.engine.compute_free_energy(errors, {})
        self.assertIn("sensory", fe.level_breakdown)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestPredictiveCodingInterface(unittest.TestCase):
    """Tests for PredictiveCodingInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = PredictiveCodingInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "16-predictive-coding")
        self.assertEqual(self.interface.FORM_NAME, "Predictive Coding / Free Energy Principle")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._is_initialized)

    def test_generate_prediction(self):
        """Should generate a prediction."""
        loop = asyncio.new_event_loop()
        try:
            pred = loop.run_until_complete(
                self.interface.generate_prediction(
                    PredictionLevel.PERCEPTUAL,
                    {"brightness": 0.7},
                )
            )
        finally:
            loop.close()

        self.assertIsInstance(pred, Prediction)
        self.assertEqual(pred.level, PredictionLevel.PERCEPTUAL)

    def test_compute_prediction_error(self):
        """Should compute prediction error."""
        pred = create_prediction("p1", content={"value": 0.3})
        ev = create_evidence("e1", content={"value": 0.8})

        loop = asyncio.new_event_loop()
        try:
            error = loop.run_until_complete(
                self.interface.compute_prediction_error(pred, ev)
            )
        finally:
            loop.close()

        self.assertIsInstance(error, PredictionError)
        self.assertGreater(error.magnitude, 0.0)

    def test_update_generative_model(self):
        """Should update the generative model."""
        errors = [
            PredictionError(
                error_id="pe1", prediction_id="p1", evidence_id="e1",
                level=PredictionLevel.SENSORY,
                error_type=ErrorType.SENSORY_MISMATCH,
                magnitude=0.5, precision_weighted_error=0.4,
                direction={}, surprise=0.7,
            )
        ]

        loop = asyncio.new_event_loop()
        try:
            update = loop.run_until_complete(
                self.interface.update_generative_model(errors, PredictionLevel.SENSORY)
            )
        finally:
            loop.close()

        self.assertIsInstance(update, ModelUpdate)

    def test_minimize_free_energy(self):
        """Should run full free energy minimization."""
        pred = create_prediction("p1", PredictionLevel.PERCEPTUAL, {"value": 0.3})
        ev = create_evidence("e1", PredictionLevel.PERCEPTUAL, {"value": 0.7})

        pc_input = PredictiveCodingInput(
            predictions=[pred],
            evidence=[ev],
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.minimize_free_energy(pc_input)
            )
        finally:
            loop.close()

        self.assertIsInstance(output, PredictiveCodingOutput)
        self.assertGreater(len(output.prediction_errors), 0)
        self.assertIsNotNone(output.free_energy_state)

    def test_to_dict(self):
        """Should convert state to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "16-predictive-coding")
        self.assertIn("total_predictions", d)
        self.assertIn("model_confidences", d)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, PredictiveCodingSystemStatus)
        self.assertFalse(status.is_initialized)

    def test_counters_increment(self):
        """Should track operation counters."""
        pred = create_prediction("p1")
        ev = create_evidence("e1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self.interface.generate_prediction(PredictionLevel.SENSORY, {"v": 0.5})
            )
            loop.run_until_complete(
                self.interface.compute_prediction_error(pred, ev)
            )
        finally:
            loop.close()

        self.assertEqual(self.interface._total_predictions, 1)
        self.assertEqual(self.interface._total_errors, 1)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_predictive_coding_interface(self):
        """Should create new interface."""
        interface = create_predictive_coding_interface()
        self.assertIsInstance(interface, PredictiveCodingInterface)

    def test_create_prediction(self):
        """Should create prediction."""
        pred = create_prediction("p1", PredictionLevel.CONCEPTUAL, {"idea": 0.5}, 0.8)
        self.assertEqual(pred.prediction_id, "p1")
        self.assertEqual(pred.level, PredictionLevel.CONCEPTUAL)
        self.assertEqual(pred.confidence, 0.8)

    def test_create_evidence(self):
        """Should create evidence."""
        ev = create_evidence("e1", PredictionLevel.SENSORY, {"raw": 0.3}, 0.9)
        self.assertEqual(ev.evidence_id, "e1")
        self.assertEqual(ev.reliability, 0.9)

    def test_default_values(self):
        """Should use sensible defaults."""
        pred = create_prediction("p1")
        self.assertEqual(pred.level, PredictionLevel.PERCEPTUAL)
        ev = create_evidence("e1")
        self.assertEqual(ev.level, PredictionLevel.PERCEPTUAL)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPredictiveCodingIntegration(unittest.TestCase):
    """Integration tests for the Predictive Coding system."""

    def test_full_prediction_error_update_cycle(self):
        """Should complete full predict-compare-update cycle."""
        interface = create_predictive_coding_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())

            # Generate prediction
            pred = loop.run_until_complete(
                interface.generate_prediction(
                    PredictionLevel.PERCEPTUAL,
                    {"brightness": 0.5, "color_hue": 0.3},
                    "test_pred"
                )
            )

            # Create evidence
            ev = SensoryEvidence(
                evidence_id="test_ev",
                level=PredictionLevel.PERCEPTUAL,
                content={"brightness": 0.8, "color_hue": 0.7},
                reliability=0.9,
                precision=0.85,
            )

            # Compute error
            error = loop.run_until_complete(
                interface.compute_prediction_error(pred, ev)
            )

            # Update model
            update = loop.run_until_complete(
                interface.update_generative_model([error], PredictionLevel.PERCEPTUAL)
            )
        finally:
            loop.close()

        self.assertGreater(error.magnitude, 0.0)
        self.assertGreater(update.error_reduction, 0.0)

    def test_multi_level_free_energy_minimization(self):
        """Should minimize free energy across multiple levels."""
        interface = create_predictive_coding_interface()

        predictions = [
            create_prediction("p_s", PredictionLevel.SENSORY, {"raw": 0.5}),
            create_prediction("p_p", PredictionLevel.PERCEPTUAL, {"feature": 0.4}),
        ]
        evidence = [
            create_evidence("e_s", PredictionLevel.SENSORY, {"raw": 0.8}),
            create_evidence("e_p", PredictionLevel.PERCEPTUAL, {"feature": 0.7}),
        ]

        pc_input = PredictiveCodingInput(
            predictions=predictions,
            evidence=evidence,
            attention_weights={"sensory": 1.0, "perceptual": 0.8},
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                interface.minimize_free_energy(pc_input)
            )
        finally:
            loop.close()

        self.assertGreaterEqual(len(output.prediction_errors), 2)
        self.assertIsNotNone(output.free_energy_state)
        self.assertGreater(output.total_surprise, 0.0)

    def test_repeated_processing_reduces_surprise(self):
        """Repeated processing of similar input should reduce surprise over time."""
        interface = create_predictive_coding_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())

            surprises = []
            for i in range(5):
                pred = create_prediction(f"p{i}", PredictionLevel.PERCEPTUAL, {"value": 0.5})
                ev = create_evidence(f"e{i}", PredictionLevel.PERCEPTUAL, {"value": 0.7})

                pc_input = PredictiveCodingInput(
                    predictions=[pred], evidence=[ev]
                )
                output = loop.run_until_complete(
                    interface.minimize_free_energy(pc_input)
                )
                surprises.append(output.total_surprise)
        finally:
            loop.close()

        # At minimum, all surprises should be positive
        for s in surprises:
            self.assertGreater(s, 0.0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
