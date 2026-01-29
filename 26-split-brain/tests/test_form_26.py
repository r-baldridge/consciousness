#!/usr/bin/env python3
"""
Test Suite for Form 26: Split-Brain Consciousness.

Tests cover:
- All enumerations (Hemisphere, ProcessingDomain, InterhemisphericState,
  ConflictType, ConfabulationType, LateralizedField)
- All input/output dataclasses
- SplitBrainInterface main interface
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
    Hemisphere,
    ProcessingDomain,
    InterhemisphericState,
    ConflictType,
    ConfabulationType,
    LateralizedField,
    # Input dataclasses
    SplitBrainInput,
    BilateralInput,
    # Output dataclasses
    HemisphereResponse,
    SplitBrainOutput,
    ConflictAnalysis,
    ConfabulationOutput,
    # Interface
    SplitBrainInterface,
    # Convenience
    create_split_brain_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestHemisphere(unittest.TestCase):
    """Tests for Hemisphere enumeration."""

    def test_all_hemispheres_exist(self):
        """Both hemispheres should be defined."""
        hemispheres = [
            Hemisphere.LEFT,
            Hemisphere.RIGHT,
        ]
        self.assertEqual(len(hemispheres), 2)

    def test_hemisphere_values(self):
        """Hemispheres should have expected string values."""
        self.assertEqual(Hemisphere.LEFT.value, "left")
        self.assertEqual(Hemisphere.RIGHT.value, "right")


class TestProcessingDomain(unittest.TestCase):
    """Tests for ProcessingDomain enumeration."""

    def test_all_domains_exist(self):
        """All processing domains should be defined."""
        domains = [
            ProcessingDomain.VERBAL,
            ProcessingDomain.SPATIAL,
            ProcessingDomain.EMOTIONAL,
            ProcessingDomain.ANALYTICAL,
            ProcessingDomain.HOLISTIC,
            ProcessingDomain.MOTOR_SPEECH,
        ]
        self.assertEqual(len(domains), 6)

    def test_domain_values(self):
        """Domains should have expected string values."""
        self.assertEqual(ProcessingDomain.VERBAL.value, "verbal")
        self.assertEqual(ProcessingDomain.SPATIAL.value, "spatial")
        self.assertEqual(ProcessingDomain.EMOTIONAL.value, "emotional")


class TestInterhemisphericState(unittest.TestCase):
    """Tests for InterhemisphericState enumeration."""

    def test_all_states_exist(self):
        """All interhemispheric states should be defined."""
        states = [
            InterhemisphericState.DISCONNECTED,
            InterhemisphericState.PARTIAL,
            InterhemisphericState.CROSS_CUEING,
            InterhemisphericState.INTEGRATED,
        ]
        self.assertEqual(len(states), 4)

    def test_state_values(self):
        """States should have expected string values."""
        self.assertEqual(InterhemisphericState.DISCONNECTED.value, "disconnected")
        self.assertEqual(InterhemisphericState.INTEGRATED.value, "integrated")


class TestConflictType(unittest.TestCase):
    """Tests for ConflictType enumeration."""

    def test_all_types_exist(self):
        """All conflict types should be defined."""
        types = [
            ConflictType.MOTOR_CONFLICT,
            ConflictType.PERCEPTUAL_CONFLICT,
            ConflictType.DECISIONAL_CONFLICT,
            ConflictType.EMOTIONAL_CONFLICT,
        ]
        self.assertEqual(len(types), 4)


class TestConfabulationType(unittest.TestCase):
    """Tests for ConfabulationType enumeration."""

    def test_all_types_exist(self):
        """All confabulation types should be defined."""
        types = [
            ConfabulationType.CAUSAL_ATTRIBUTION,
            ConfabulationType.POST_HOC_RATIONALIZATION,
            ConfabulationType.GAP_FILLING,
            ConfabulationType.NARRATIVE_CONSTRUCTION,
        ]
        self.assertEqual(len(types), 4)


class TestLateralizedField(unittest.TestCase):
    """Tests for LateralizedField enumeration."""

    def test_all_fields_exist(self):
        """All lateralized fields should be defined."""
        fields = [
            LateralizedField.LEFT_VISUAL_FIELD,
            LateralizedField.RIGHT_VISUAL_FIELD,
            LateralizedField.BILATERAL,
            LateralizedField.CENTRAL,
        ]
        self.assertEqual(len(fields), 4)

    def test_field_values(self):
        """Fields should have expected string values."""
        self.assertEqual(LateralizedField.LEFT_VISUAL_FIELD.value, "left_visual_field")
        self.assertEqual(LateralizedField.RIGHT_VISUAL_FIELD.value, "right_visual_field")


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSplitBrainInput(unittest.TestCase):
    """Tests for SplitBrainInput dataclass."""

    def test_basic_creation(self):
        """Test basic input creation."""
        inp = SplitBrainInput(
            visual_field=LateralizedField.LEFT_VISUAL_FIELD,
            stimulus_content="apple",
            stimulus_type="image",
            processing_domain=ProcessingDomain.SPATIAL,
        )
        self.assertEqual(inp.visual_field, LateralizedField.LEFT_VISUAL_FIELD)
        self.assertEqual(inp.stimulus_content, "apple")
        self.assertEqual(inp.processing_domain, ProcessingDomain.SPATIAL)

    def test_default_values(self):
        """Test default values."""
        inp = SplitBrainInput(
            visual_field=LateralizedField.RIGHT_VISUAL_FIELD,
            stimulus_content="word",
            stimulus_type="word",
            processing_domain=ProcessingDomain.VERBAL,
        )
        self.assertAlmostEqual(inp.stimulus_duration_ms, 150.0)
        self.assertEqual(inp.task_type, "identify")
        self.assertEqual(inp.response_modality, "verbal")
        self.assertEqual(inp.interhemispheric_state, InterhemisphericState.DISCONNECTED)

    def test_to_dict(self):
        """Test serialization."""
        inp = SplitBrainInput(
            visual_field=LateralizedField.LEFT_VISUAL_FIELD,
            stimulus_content="face",
            stimulus_type="face",
            processing_domain=ProcessingDomain.EMOTIONAL,
        )
        d = inp.to_dict()
        self.assertEqual(d["visual_field"], "left_visual_field")
        self.assertEqual(d["processing_domain"], "emotional")
        self.assertIn("timestamp", d)


class TestBilateralInput(unittest.TestCase):
    """Tests for BilateralInput dataclass."""

    def test_creation(self):
        """Test bilateral input creation."""
        inp = BilateralInput(
            left_field_stimulus="apple",
            right_field_stimulus="book",
        )
        self.assertEqual(inp.left_field_stimulus, "apple")
        self.assertEqual(inp.right_field_stimulus, "book")

    def test_to_dict(self):
        """Test serialization."""
        inp = BilateralInput(
            left_field_stimulus="cat",
            right_field_stimulus="dog",
            response_modality="verbal",
        )
        d = inp.to_dict()
        self.assertEqual(d["left_field_stimulus"], "cat")
        self.assertEqual(d["right_field_stimulus"], "dog")


class TestHemisphereResponse(unittest.TestCase):
    """Tests for HemisphereResponse dataclass."""

    def test_creation(self):
        """Test hemisphere response creation."""
        response = HemisphereResponse(
            hemisphere=Hemisphere.LEFT,
            response_content="I see a book",
            processing_domain=ProcessingDomain.VERBAL,
            confidence=0.9,
            can_verbalize=True,
        )
        self.assertEqual(response.hemisphere, Hemisphere.LEFT)
        self.assertTrue(response.can_verbalize)

    def test_right_hemisphere(self):
        """Test right hemisphere response."""
        response = HemisphereResponse(
            hemisphere=Hemisphere.RIGHT,
            response_content="spatial_match",
            processing_domain=ProcessingDomain.SPATIAL,
            confidence=0.85,
            can_verbalize=False,
        )
        self.assertFalse(response.can_verbalize)

    def test_to_dict(self):
        """Test serialization."""
        response = HemisphereResponse(
            hemisphere=Hemisphere.LEFT,
            response_content="test",
            processing_domain=ProcessingDomain.ANALYTICAL,
            confidence=0.7,
            can_verbalize=True,
        )
        d = response.to_dict()
        self.assertEqual(d["hemisphere"], "left")
        self.assertTrue(d["can_verbalize"])


class TestSplitBrainOutput(unittest.TestCase):
    """Tests for SplitBrainOutput dataclass."""

    def test_creation(self):
        """Test split-brain output creation."""
        left = HemisphereResponse(
            hemisphere=Hemisphere.LEFT,
            response_content="book",
            processing_domain=ProcessingDomain.VERBAL,
            confidence=0.8,
            can_verbalize=True,
        )
        right = HemisphereResponse(
            hemisphere=Hemisphere.RIGHT,
            response_content="apple",
            processing_domain=ProcessingDomain.SPATIAL,
            confidence=0.7,
            can_verbalize=False,
        )
        output = SplitBrainOutput(
            left_hemisphere=left,
            right_hemisphere=right,
            conflict_detected=True,
            conflict_type=ConflictType.PERCEPTUAL_CONFLICT,
        )
        self.assertTrue(output.conflict_detected)

    def test_to_dict(self):
        """Test serialization."""
        left = HemisphereResponse(
            hemisphere=Hemisphere.LEFT,
            response_content="test",
            processing_domain=ProcessingDomain.VERBAL,
            confidence=0.5,
            can_verbalize=True,
        )
        right = HemisphereResponse(
            hemisphere=Hemisphere.RIGHT,
            response_content="test",
            processing_domain=ProcessingDomain.SPATIAL,
            confidence=0.5,
            can_verbalize=False,
        )
        output = SplitBrainOutput(
            left_hemisphere=left,
            right_hemisphere=right,
            conflict_detected=False,
        )
        d = output.to_dict()
        self.assertFalse(d["conflict_detected"])
        self.assertIn("left_hemisphere", d)
        self.assertIn("right_hemisphere", d)


class TestConfabulationOutput(unittest.TestCase):
    """Tests for ConfabulationOutput dataclass."""

    def test_creation(self):
        """Test confabulation output creation."""
        output = ConfabulationOutput(
            confabulation_generated=True,
            confabulation_content="I chose that because it felt right",
            confabulation_type=ConfabulationType.POST_HOC_RATIONALIZATION,
            plausibility=0.8,
            actual_cause="Right hemisphere recognized the image",
        )
        self.assertTrue(output.confabulation_generated)
        self.assertFalse(output.awareness_of_confabulation)

    def test_to_dict(self):
        """Test serialization."""
        output = ConfabulationOutput(
            confabulation_generated=True,
            confabulation_content="It seemed obvious",
            confabulation_type=ConfabulationType.CAUSAL_ATTRIBUTION,
            plausibility=0.6,
            actual_cause="Subcortical processing",
        )
        d = output.to_dict()
        self.assertEqual(d["confabulation_type"], "causal_attribution")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestSplitBrainInterface(unittest.TestCase):
    """Tests for SplitBrainInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.interface = SplitBrainInterface()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up."""
        self.loop.close()

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return self.loop.run_until_complete(coro)

    def test_form_metadata(self):
        """Test form ID and name."""
        self.assertEqual(self.interface.FORM_ID, "26-split-brain")
        self.assertEqual(self.interface.FORM_NAME, "Split-Brain Consciousness")

    def test_initialization(self):
        """Test interface initializes correctly."""
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_double_initialization(self):
        """Test that double initialization is safe."""
        self._run(self.interface.initialize())
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_domain_lateralization(self):
        """Test that domain lateralization mapping is correct."""
        self.assertEqual(
            self.interface.DOMAIN_LATERALIZATION[ProcessingDomain.VERBAL],
            Hemisphere.LEFT,
        )
        self.assertEqual(
            self.interface.DOMAIN_LATERALIZATION[ProcessingDomain.SPATIAL],
            Hemisphere.RIGHT,
        )
        self.assertEqual(
            self.interface.DOMAIN_LATERALIZATION[ProcessingDomain.EMOTIONAL],
            Hemisphere.RIGHT,
        )

    def test_present_lateralized_right_field(self):
        """Test stimulus to right visual field (left hemisphere)."""
        self._run(self.interface.initialize())
        inp = SplitBrainInput(
            visual_field=LateralizedField.RIGHT_VISUAL_FIELD,
            stimulus_content="book",
            stimulus_type="word",
            processing_domain=ProcessingDomain.VERBAL,
            response_modality="verbal",
        )
        output = self._run(self.interface.present_lateralized(inp))
        self.assertIsInstance(output, SplitBrainOutput)
        # Left hemisphere receives right visual field - should have high accuracy
        self.assertGreater(output.left_hemisphere.accuracy, 0.5)
        self.assertTrue(output.left_hemisphere.can_verbalize)

    def test_present_lateralized_left_field_verbal_response(self):
        """Test stimulus to left visual field with verbal response requirement.

        This is the classic split-brain paradigm: right hemisphere receives
        the stimulus but left hemisphere must give the verbal response,
        potentially leading to confabulation.
        """
        self._run(self.interface.initialize())
        inp = SplitBrainInput(
            visual_field=LateralizedField.LEFT_VISUAL_FIELD,
            stimulus_content="apple",
            stimulus_type="image",
            processing_domain=ProcessingDomain.SPATIAL,
            response_modality="verbal",
        )
        output = self._run(self.interface.present_lateralized(inp))
        # Right hemisphere receives left visual field
        self.assertTrue(output.right_hemisphere.awareness_of_stimulus)
        # Left hemisphere should have no direct information
        self.assertAlmostEqual(output.left_hemisphere.accuracy, 0.0)
        # Confabulation should be generated
        self.assertIsNotNone(output.confabulation)

    def test_present_lateralized_bilateral(self):
        """Test bilateral stimulus presentation."""
        self._run(self.interface.initialize())
        inp = SplitBrainInput(
            visual_field=LateralizedField.BILATERAL,
            stimulus_content="pattern",
            stimulus_type="image",
            processing_domain=ProcessingDomain.HOLISTIC,
        )
        output = self._run(self.interface.present_lateralized(inp))
        # Both hemispheres should be aware
        self.assertTrue(output.left_hemisphere.awareness_of_stimulus)
        self.assertTrue(output.right_hemisphere.awareness_of_stimulus)

    def test_assess_hemisphere_response_specialized(self):
        """Test hemisphere assessment for specialized domain."""
        self._run(self.interface.initialize())
        response = self._run(self.interface.assess_hemisphere_response(
            Hemisphere.LEFT,
            ProcessingDomain.VERBAL,
            "sentence",
        ))
        self.assertIsInstance(response, HemisphereResponse)
        self.assertGreater(response.accuracy, 0.7)
        self.assertTrue(response.can_verbalize)
        self.assertIn("Specialized", response.notes[0])

    def test_assess_hemisphere_response_non_specialized(self):
        """Test hemisphere assessment for non-specialized domain."""
        self._run(self.interface.initialize())
        response = self._run(self.interface.assess_hemisphere_response(
            Hemisphere.LEFT,
            ProcessingDomain.SPATIAL,
            "pattern",
        ))
        self.assertLess(response.accuracy, 0.5)
        self.assertIn("Non-dominant", response.notes[0])

    def test_assess_right_hemisphere_no_verbalize(self):
        """Test that right hemisphere cannot verbalize."""
        self._run(self.interface.initialize())
        response = self._run(self.interface.assess_hemisphere_response(
            Hemisphere.RIGHT,
            ProcessingDomain.SPATIAL,
            "map",
        ))
        self.assertFalse(response.can_verbalize)

    def test_detect_conflict_verbal(self):
        """Test conflict detection with verbal response modality."""
        self._run(self.interface.initialize())
        inp = BilateralInput(
            left_field_stimulus="apple",
            right_field_stimulus="book",
            response_modality="verbal",
        )
        analysis = self._run(self.interface.detect_conflict(inp))
        self.assertIsInstance(analysis, ConflictAnalysis)
        self.assertTrue(analysis.conflict_present)
        self.assertEqual(analysis.conflict_type, ConflictType.PERCEPTUAL_CONFLICT)

    def test_detect_conflict_left_hand(self):
        """Test conflict detection with left hand response."""
        self._run(self.interface.initialize())
        inp = BilateralInput(
            left_field_stimulus="scissors",
            right_field_stimulus="pencil",
            response_modality="left_hand",
        )
        analysis = self._run(self.interface.detect_conflict(inp))
        self.assertTrue(analysis.conflict_present)
        self.assertEqual(analysis.conflict_type, ConflictType.MOTOR_CONFLICT)
        self.assertGreater(analysis.severity, 0.0)

    def test_detect_conflict_same_stimulus(self):
        """Test that same stimuli produce no conflict via generic modality."""
        self._run(self.interface.initialize())
        inp = BilateralInput(
            left_field_stimulus="apple",
            right_field_stimulus="apple",
            response_modality="right_hand",
        )
        analysis = self._run(self.interface.detect_conflict(inp))
        self.assertFalse(analysis.conflict_present)

    def test_model_confabulation(self):
        """Test confabulation modeling."""
        self._run(self.interface.initialize())
        output = self._run(self.interface.model_confabulation(
            right_hemisphere_action="Left hand picked up the spoon",
            actual_cause="Right hemisphere saw image of soup",
        ))
        self.assertIsInstance(output, ConfabulationOutput)
        self.assertTrue(output.confabulation_generated)
        self.assertGreater(output.plausibility, 0.0)
        self.assertFalse(output.awareness_of_confabulation)
        self.assertIn("soup", output.actual_cause)

    def test_confabulation_history(self):
        """Test confabulation history accumulation."""
        self._run(self.interface.initialize())
        self._run(self.interface.model_confabulation(
            right_hemisphere_action="Pointed left",
            actual_cause="Saw arrow pointing left",
        ))
        self._run(self.interface.model_confabulation(
            right_hemisphere_action="Laughed",
            actual_cause="Saw funny image",
        ))
        self.assertEqual(len(self.interface._confabulation_history), 2)

    def test_to_dict(self):
        """Test interface serialization."""
        self._run(self.interface.initialize())
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "26-split-brain")
        self.assertTrue(d["initialized"])
        self.assertEqual(d["interhemispheric_state"], "disconnected")

    def test_get_status(self):
        """Test status retrieval."""
        self._run(self.interface.initialize())
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "26-split-brain")
        self.assertEqual(status["total_trials"], 0)
        self.assertEqual(status["conflicts_detected"], 0)

    def test_auto_initialize(self):
        """Test auto-initialization on method call."""
        inp = SplitBrainInput(
            visual_field=LateralizedField.RIGHT_VISUAL_FIELD,
            stimulus_content="test",
            stimulus_type="word",
            processing_domain=ProcessingDomain.VERBAL,
        )
        output = self._run(self.interface.present_lateralized(inp))
        self.assertTrue(self.interface._initialized)
        self.assertIsInstance(output, SplitBrainOutput)

    def test_trial_history(self):
        """Test trial history accumulates."""
        self._run(self.interface.initialize())
        for field in [LateralizedField.LEFT_VISUAL_FIELD, LateralizedField.RIGHT_VISUAL_FIELD]:
            inp = SplitBrainInput(
                visual_field=field,
                stimulus_content="stimulus",
                stimulus_type="image",
                processing_domain=ProcessingDomain.SPATIAL,
            )
            self._run(self.interface.present_lateralized(inp))
        self.assertEqual(len(self.interface._trial_history), 2)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module convenience functions."""

    def test_create_split_brain_interface(self):
        """Test convenience creation function."""
        interface = create_split_brain_interface()
        self.assertIsInstance(interface, SplitBrainInterface)
        self.assertFalse(interface._initialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
