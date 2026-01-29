#!/usr/bin/env python3
"""
Test Suite for Form 15: Higher-Order Thought (HOT) Theory Consciousness.

Tests cover:
- All enumerations (RepresentationOrder, ConsciousnessType, etc.)
- All input/output dataclasses
- HOTGenerationEngine
- ConsciousnessAssessmentEngine
- HigherOrderThoughtInterface (main interface)
- Convenience functions
- Integration tests for the full HOT pipeline
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
    RepresentationOrder,
    ConsciousnessType,
    RepresentationModality,
    HOTQuality,
    AssessmentCriterion,
    # Input dataclasses
    FirstOrderState,
    HOTRequest,
    # Output dataclasses
    HigherOrderRepresentation,
    ConsciousnessAssessment,
    RepresentationHierarchy,
    HOTOutput,
    HOTSystemStatus,
    # Engines
    HOTGenerationEngine,
    ConsciousnessAssessmentEngine,
    # Main interface
    HigherOrderThoughtInterface,
    # Convenience functions
    create_higher_order_thought_interface,
    create_first_order_state,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestRepresentationOrder(unittest.TestCase):
    """Tests for RepresentationOrder enumeration."""

    def test_all_orders_exist(self):
        """All representation orders should be defined."""
        orders = [
            RepresentationOrder.FIRST_ORDER,
            RepresentationOrder.SECOND_ORDER,
            RepresentationOrder.THIRD_ORDER,
            RepresentationOrder.FOURTH_ORDER,
        ]
        self.assertEqual(len(orders), 4)

    def test_order_values(self):
        """Orders should have expected string values."""
        self.assertEqual(RepresentationOrder.FIRST_ORDER.value, "first_order")
        self.assertEqual(RepresentationOrder.SECOND_ORDER.value, "second_order")
        self.assertEqual(RepresentationOrder.THIRD_ORDER.value, "third_order")


class TestConsciousnessType(unittest.TestCase):
    """Tests for ConsciousnessType enumeration."""

    def test_all_types_exist(self):
        """All consciousness types should be defined."""
        types = [
            ConsciousnessType.UNCONSCIOUS,
            ConsciousnessType.PHENOMENAL,
            ConsciousnessType.ACCESS,
            ConsciousnessType.INTROSPECTIVE,
            ConsciousnessType.SELF_REFLECTIVE,
        ]
        self.assertEqual(len(types), 5)


class TestRepresentationModality(unittest.TestCase):
    """Tests for RepresentationModality enumeration."""

    def test_all_modalities_exist(self):
        """All modalities should be defined."""
        modalities = [
            RepresentationModality.PERCEPTUAL,
            RepresentationModality.COGNITIVE,
            RepresentationModality.EMOTIONAL,
            RepresentationModality.BODILY,
            RepresentationModality.LINGUISTIC,
            RepresentationModality.IMAGISTIC,
        ]
        self.assertEqual(len(modalities), 6)


class TestHOTQuality(unittest.TestCase):
    """Tests for HOTQuality enumeration."""

    def test_all_qualities_exist(self):
        """All quality attributes should be defined."""
        qualities = [
            HOTQuality.CLARITY,
            HOTQuality.CONFIDENCE,
            HOTQuality.SPECIFICITY,
            HOTQuality.STABILITY,
            HOTQuality.ACCURACY,
        ]
        self.assertEqual(len(qualities), 5)


class TestAssessmentCriterion(unittest.TestCase):
    """Tests for AssessmentCriterion enumeration."""

    def test_all_criteria_exist(self):
        """All assessment criteria should be defined."""
        criteria = [
            AssessmentCriterion.HAS_HOT,
            AssessmentCriterion.APPROPRIATE_TARGET,
            AssessmentCriterion.SUFFICIENT_QUALITY,
            AssessmentCriterion.TEMPORAL_MATCH,
            AssessmentCriterion.SELF_REFERENTIAL,
        ]
        self.assertEqual(len(criteria), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestFirstOrderState(unittest.TestCase):
    """Tests for FirstOrderState dataclass."""

    def test_creation(self):
        """Should create first-order state with required fields."""
        state = FirstOrderState(
            state_id="s1",
            modality=RepresentationModality.PERCEPTUAL,
            content={"color": "red", "shape": "circle"},
            intensity=0.8,
            distinctness=0.7,
        )
        self.assertEqual(state.state_id, "s1")
        self.assertEqual(state.modality, RepresentationModality.PERCEPTUAL)
        self.assertEqual(state.intensity, 0.8)
        self.assertFalse(state.is_attended)

    def test_defaults(self):
        """Should have correct defaults."""
        state = FirstOrderState(
            state_id="s1",
            modality=RepresentationModality.COGNITIVE,
            content={},
            intensity=0.5,
            distinctness=0.5,
        )
        self.assertEqual(state.valence, 0.0)
        self.assertEqual(state.activation_level, 0.5)
        self.assertFalse(state.is_attended)

    def test_to_dict(self):
        """Should convert to dictionary."""
        state = FirstOrderState(
            state_id="s1",
            modality=RepresentationModality.EMOTIONAL,
            content={"emotion": "joy"},
            intensity=0.9,
            distinctness=0.8,
            valence=0.7,
        )
        d = state.to_dict()
        self.assertEqual(d["state_id"], "s1")
        self.assertEqual(d["modality"], "emotional")
        self.assertIn("timestamp", d)


class TestHOTRequest(unittest.TestCase):
    """Tests for HOTRequest dataclass."""

    def test_creation(self):
        """Should create HOT request."""
        req = HOTRequest(
            target_state_id="s1",
            requested_order=RepresentationOrder.SECOND_ORDER,
            introspective_effort=0.7,
        )
        self.assertEqual(req.target_state_id, "s1")
        self.assertEqual(req.requested_order, RepresentationOrder.SECOND_ORDER)

    def test_defaults(self):
        """Should have correct defaults."""
        req = HOTRequest(target_state_id="s1")
        self.assertEqual(req.requested_order, RepresentationOrder.SECOND_ORDER)
        self.assertEqual(req.attention_boost, 0.0)
        self.assertEqual(req.introspective_effort, 0.5)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestHigherOrderRepresentation(unittest.TestCase):
    """Tests for HigherOrderRepresentation dataclass."""

    def test_creation(self):
        """Should create higher-order representation."""
        hot = HigherOrderRepresentation(
            hot_id="hot_1",
            target_state_id="s1",
            order=RepresentationOrder.SECOND_ORDER,
            content_summary="Awareness of perceptual state 's1'",
            quality_scores={HOTQuality.CLARITY: 0.8, HOTQuality.CONFIDENCE: 0.7},
            overall_quality=0.75,
            is_conscious_making=True,
        )
        self.assertEqual(hot.hot_id, "hot_1")
        self.assertTrue(hot.is_conscious_making)
        self.assertTrue(hot.self_attribution)

    def test_to_dict(self):
        """Should convert to dictionary."""
        hot = HigherOrderRepresentation(
            hot_id="hot_1",
            target_state_id="s1",
            order=RepresentationOrder.SECOND_ORDER,
            content_summary="Test",
            quality_scores={},
            overall_quality=0.5,
            is_conscious_making=True,
        )
        d = hot.to_dict()
        self.assertEqual(d["order"], "second_order")
        self.assertIn("is_conscious_making", d)


class TestConsciousnessAssessment(unittest.TestCase):
    """Tests for ConsciousnessAssessment dataclass."""

    def test_creation(self):
        """Should create assessment."""
        assessment = ConsciousnessAssessment(
            state_id="s1",
            is_conscious=True,
            consciousness_type=ConsciousnessType.ACCESS,
            criteria_met={AssessmentCriterion.HAS_HOT: True},
            hot_chain=[],
            highest_order=RepresentationOrder.SECOND_ORDER,
            confidence=0.8,
            explanation="State is conscious.",
        )
        self.assertTrue(assessment.is_conscious)
        self.assertEqual(assessment.consciousness_type, ConsciousnessType.ACCESS)

    def test_to_dict(self):
        """Should convert to dictionary."""
        assessment = ConsciousnessAssessment(
            state_id="s1",
            is_conscious=False,
            consciousness_type=ConsciousnessType.UNCONSCIOUS,
            criteria_met={},
            hot_chain=[],
            highest_order=RepresentationOrder.FIRST_ORDER,
            confidence=0.3,
            explanation="No HOT.",
        )
        d = assessment.to_dict()
        self.assertEqual(d["consciousness_type"], "unconscious")


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestHOTGenerationEngine(unittest.TestCase):
    """Tests for HOTGenerationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = HOTGenerationEngine()

    def test_generate_basic_hot(self):
        """Should generate a second-order HOT."""
        state = create_first_order_state("s1", intensity=0.7, distinctness=0.8)
        hot = self.engine.generate_hot(state)
        self.assertIsInstance(hot, HigherOrderRepresentation)
        self.assertEqual(hot.order, RepresentationOrder.SECOND_ORDER)
        self.assertEqual(hot.target_state_id, "s1")

    def test_hot_quality_varies_with_intensity(self):
        """Higher intensity states should produce higher quality HOTs."""
        low_state = create_first_order_state("low", intensity=0.1, distinctness=0.1)
        high_state = create_first_order_state("high", intensity=0.9, distinctness=0.9)

        low_hot = self.engine.generate_hot(low_state)
        high_hot = self.engine.generate_hot(high_state)

        self.assertGreater(high_hot.overall_quality, low_hot.overall_quality)

    def test_higher_orders_lower_quality(self):
        """Higher-order HOTs should generally have lower quality."""
        state = create_first_order_state("s1", intensity=0.7, distinctness=0.7)

        second = self.engine.generate_hot(state, RepresentationOrder.SECOND_ORDER)
        third = self.engine.generate_hot(state, RepresentationOrder.THIRD_ORDER)

        self.assertGreaterEqual(second.overall_quality, third.overall_quality)

    def test_introspective_effort_boosts_quality(self):
        """Higher introspective effort should improve HOT quality."""
        state = create_first_order_state("s1", intensity=0.5, distinctness=0.5)

        low_effort = self.engine.generate_hot(state, introspective_effort=0.1)
        high_effort = self.engine.generate_hot(state, introspective_effort=0.9)

        self.assertGreater(high_effort.overall_quality, low_effort.overall_quality)

    def test_consciousness_making_threshold(self):
        """Strong states should produce consciousness-making HOTs."""
        strong_state = create_first_order_state("strong", intensity=0.9, distinctness=0.9)
        strong_state.activation_level = 0.9
        strong_state.is_attended = True

        hot = self.engine.generate_hot(strong_state, introspective_effort=0.8)
        self.assertTrue(hot.is_conscious_making)

    def test_weak_state_not_conscious(self):
        """Very weak states may not produce consciousness-making HOTs."""
        weak_state = create_first_order_state("weak", intensity=0.05, distinctness=0.05)
        weak_state.activation_level = 0.05

        hot = self.engine.generate_hot(weak_state, introspective_effort=0.1)
        self.assertFalse(hot.is_conscious_making)

    def test_content_summary_generation(self):
        """Should generate appropriate content summaries."""
        state = create_first_order_state("s1", RepresentationModality.PERCEPTUAL)
        hot = self.engine.generate_hot(state, RepresentationOrder.SECOND_ORDER)
        self.assertIn("Awareness", hot.content_summary)
        self.assertIn("s1", hot.content_summary)

    def test_unique_hot_ids(self):
        """Each HOT should have a unique ID."""
        state = create_first_order_state("s1")
        hot1 = self.engine.generate_hot(state)
        hot2 = self.engine.generate_hot(state)
        self.assertNotEqual(hot1.hot_id, hot2.hot_id)


class TestConsciousnessAssessmentEngine(unittest.TestCase):
    """Tests for ConsciousnessAssessmentEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ConsciousnessAssessmentEngine()

    def test_no_hot_means_unconscious(self):
        """State with no HOT should be unconscious."""
        state = create_first_order_state("s1")
        assessment = self.engine.assess(state, [])
        self.assertFalse(assessment.is_conscious)
        self.assertEqual(assessment.consciousness_type, ConsciousnessType.UNCONSCIOUS)

    def test_good_hot_means_conscious(self):
        """State with good HOT should be conscious."""
        state = create_first_order_state("s1", intensity=0.8, distinctness=0.8)
        hot = HigherOrderRepresentation(
            hot_id="hot_test",
            target_state_id="s1",
            order=RepresentationOrder.SECOND_ORDER,
            content_summary="Test",
            quality_scores={},
            overall_quality=0.7,
            is_conscious_making=True,
            self_attribution=True,
        )
        assessment = self.engine.assess(state, [hot])
        self.assertTrue(assessment.is_conscious)
        self.assertEqual(assessment.consciousness_type, ConsciousnessType.ACCESS)

    def test_third_order_gives_introspective(self):
        """Third-order HOT should yield introspective consciousness."""
        state = create_first_order_state("s1")
        hots = [
            HigherOrderRepresentation(
                hot_id="hot_2", target_state_id="s1",
                order=RepresentationOrder.SECOND_ORDER,
                content_summary="Test", quality_scores={},
                overall_quality=0.7, is_conscious_making=True,
            ),
            HigherOrderRepresentation(
                hot_id="hot_3", target_state_id="s1",
                order=RepresentationOrder.THIRD_ORDER,
                content_summary="Test", quality_scores={},
                overall_quality=0.6, is_conscious_making=True,
            ),
        ]
        assessment = self.engine.assess(state, hots)
        self.assertEqual(assessment.consciousness_type, ConsciousnessType.INTROSPECTIVE)

    def test_wrong_target_not_conscious(self):
        """HOT targeting wrong state should not make state conscious."""
        state = create_first_order_state("s1")
        hot = HigherOrderRepresentation(
            hot_id="hot_test", target_state_id="s_other",
            order=RepresentationOrder.SECOND_ORDER,
            content_summary="Test", quality_scores={},
            overall_quality=0.8, is_conscious_making=True,
        )
        assessment = self.engine.assess(state, [hot])
        self.assertFalse(assessment.is_conscious)

    def test_assessment_confidence(self):
        """Assessment should have appropriate confidence."""
        state = create_first_order_state("s1")
        assessment = self.engine.assess(state, [])
        self.assertGreaterEqual(assessment.confidence, 0.0)
        self.assertLessEqual(assessment.confidence, 1.0)

    def test_explanation_generated(self):
        """Should generate an explanation string."""
        state = create_first_order_state("s1")
        assessment = self.engine.assess(state, [])
        self.assertIsInstance(assessment.explanation, str)
        self.assertGreater(len(assessment.explanation), 0)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestHigherOrderThoughtInterface(unittest.TestCase):
    """Tests for HigherOrderThoughtInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = HigherOrderThoughtInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "15-higher-order-thought")
        self.assertEqual(self.interface.FORM_NAME, "Higher-Order Thought Theory (HOT)")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._is_initialized)

    def test_create_hot(self):
        """Should create a HOT for a state."""
        state = create_first_order_state("s1", intensity=0.7, distinctness=0.8)

        loop = asyncio.new_event_loop()
        try:
            hot = loop.run_until_complete(self.interface.create_hot(state))
        finally:
            loop.close()

        self.assertIsInstance(hot, HigherOrderRepresentation)
        self.assertEqual(hot.target_state_id, "s1")

    def test_assess_consciousness(self):
        """Should assess consciousness after creating HOT."""
        state = create_first_order_state("s1", intensity=0.8, distinctness=0.8)
        state.activation_level = 0.8
        state.is_attended = True

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.create_hot(state, introspective_effort=0.8))
            assessment = loop.run_until_complete(self.interface.assess_consciousness("s1"))
        finally:
            loop.close()

        self.assertIsInstance(assessment, ConsciousnessAssessment)

    def test_assess_unknown_state(self):
        """Should handle assessment of unknown state."""
        loop = asyncio.new_event_loop()
        try:
            assessment = loop.run_until_complete(
                self.interface.assess_consciousness("unknown")
            )
        finally:
            loop.close()

        self.assertFalse(assessment.is_conscious)
        self.assertEqual(assessment.consciousness_type, ConsciousnessType.UNCONSCIOUS)

    def test_get_representation_hierarchy(self):
        """Should return representation hierarchy."""
        state = create_first_order_state("s1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.create_hot(state))
            hierarchy = loop.run_until_complete(
                self.interface.get_representation_hierarchy("s1")
            )
        finally:
            loop.close()

        self.assertIsInstance(hierarchy, RepresentationHierarchy)
        self.assertEqual(hierarchy.root_state.state_id, "s1")
        self.assertGreater(hierarchy.total_representations, 0)

    def test_process_state_full_pipeline(self):
        """Should run full processing pipeline."""
        state = create_first_order_state("s1", intensity=0.7, distinctness=0.7)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_state(state))
        finally:
            loop.close()

        self.assertIsInstance(output, HOTOutput)
        self.assertGreater(len(output.higher_order_thoughts), 0)
        self.assertIsNotNone(output.assessment)

    def test_to_dict(self):
        """Should convert state to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "15-higher-order-thought")
        self.assertIn("states_processed", d)
        self.assertIn("hots_created", d)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, HOTSystemStatus)
        self.assertFalse(status.is_initialized)
        self.assertEqual(status.total_states_processed, 0)

    def test_counters_increment(self):
        """Should track processing counters."""
        state = create_first_order_state("s1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.create_hot(state))
            loop.run_until_complete(self.interface.assess_consciousness("s1"))
        finally:
            loop.close()

        self.assertEqual(self.interface._hots_created, 1)
        self.assertEqual(self.interface._states_processed, 1)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_higher_order_thought_interface(self):
        """Should create new interface."""
        interface = create_higher_order_thought_interface()
        self.assertIsInstance(interface, HigherOrderThoughtInterface)

    def test_create_first_order_state(self):
        """Should create first-order state."""
        state = create_first_order_state("test", RepresentationModality.EMOTIONAL, 0.8, 0.6)
        self.assertEqual(state.state_id, "test")
        self.assertEqual(state.modality, RepresentationModality.EMOTIONAL)
        self.assertEqual(state.intensity, 0.8)

    def test_create_first_order_state_defaults(self):
        """Should create state with defaults."""
        state = create_first_order_state("s1")
        self.assertEqual(state.modality, RepresentationModality.PERCEPTUAL)
        self.assertEqual(state.intensity, 0.5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestHOTIntegration(unittest.TestCase):
    """Integration tests for the HOT system."""

    def test_conscious_state_pipeline(self):
        """Strong state should become conscious through HOT processing."""
        interface = create_higher_order_thought_interface()
        state = create_first_order_state("bright", intensity=0.9, distinctness=0.9)
        state.activation_level = 0.9
        state.is_attended = True

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            output = loop.run_until_complete(
                interface.process_state(state, introspective_effort=0.8)
            )
        finally:
            loop.close()

        self.assertTrue(output.assessment.is_conscious)

    def test_weak_state_unconscious(self):
        """Very weak state should remain unconscious."""
        interface = create_higher_order_thought_interface()
        state = create_first_order_state("faint", intensity=0.02, distinctness=0.02)
        state.activation_level = 0.02

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                interface.process_state(state, introspective_effort=0.1)
            )
        finally:
            loop.close()

        self.assertFalse(output.assessment.is_conscious)

    def test_multi_order_processing(self):
        """Should create multi-level HOT hierarchy."""
        interface = create_higher_order_thought_interface()
        state = create_first_order_state("deep", intensity=0.7, distinctness=0.7)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                interface.process_state(
                    state,
                    max_order=RepresentationOrder.THIRD_ORDER,
                    introspective_effort=0.7,
                )
            )
        finally:
            loop.close()

        self.assertGreaterEqual(len(output.higher_order_thoughts), 2)
        orders = [h.order for h in output.higher_order_thoughts]
        self.assertIn(RepresentationOrder.SECOND_ORDER, orders)
        self.assertIn(RepresentationOrder.THIRD_ORDER, orders)

    def test_multiple_states_tracked(self):
        """Should track multiple first-order states."""
        interface = create_higher_order_thought_interface()

        loop = asyncio.new_event_loop()
        try:
            for i in range(5):
                state = create_first_order_state(f"s{i}", intensity=0.5 + i * 0.1)
                loop.run_until_complete(interface.process_state(state))
        finally:
            loop.close()

        self.assertEqual(len(interface._first_order_states), 5)
        self.assertEqual(interface._states_processed, 5)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
