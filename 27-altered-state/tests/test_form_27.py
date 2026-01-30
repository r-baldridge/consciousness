#!/usr/bin/env python3
"""
Test Suite for Form 27: Altered State Consciousness.

Tests cover:
- All enumerations from interface modules
  (ConsciousnessLayer, MindLevel, ProcessingMode, ScenePointing,
   IntentionRecognition, ZenAuthenticity, ValidationCategory)
- All input/output dataclasses
  (KarmicSeed, SenseGate, PointingGesture, DirectUnderstanding,
   NaturalContext, ValidationCriteria, ValidationResult)
- Main interface classes
  (NonDualConsciousnessInterface, AlayaVijnana,
   DirectPointingInterface, NaturalEngagementInterface,
   ZenAuthenticityValidator)
- Convenience / factory functions
  (create_enlightened_interface, create_direct_pointing_interface,
   create_natural_engagement_interface)
"""

import asyncio
import time
import unittest
from dataclasses import fields

import sys
from pathlib import Path

# Add parent paths to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "interface"))
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# --- non_dual_consciousness_interface imports ---
from non_dual_consciousness_interface import (
    ConsciousnessLayer,
    MindLevel,
    ProcessingMode,
    KarmicSeed,
    SenseGate,
    AlayaVijnana,
    NonDualConsciousnessInterface,
    create_enlightened_interface,
)

# --- direct_pointing_interaction imports ---
from direct_pointing_interaction import (
    ScenePointing,
    IntentionRecognition,
    PointingGesture,
    DirectUnderstanding,
    DirectPointingInterface,
    create_direct_pointing_interface,
)

# --- natural_engagement_interface imports ---
from natural_engagement_interface import (
    NaturalContext,
    NaturalEngagementInterface,
    create_natural_engagement_interface,
)

# --- zen_authenticity_validator imports ---
from zen_authenticity_validator import (
    ZenAuthenticity,
    ValidationCategory,
    ValidationCriteria,
    ValidationResult,
    ZenAuthenticityValidator,
)


# ============================================================================
# HELPER
# ============================================================================

def _run(coro):
    """Run an async coroutine synchronously using a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# ENUM TESTS -- non_dual_consciousness_interface
# ============================================================================

class TestConsciousnessLayer(unittest.TestCase):
    """Tests for ConsciousnessLayer enumeration."""

    def test_all_layers_exist(self):
        """All seven consciousness layers should be defined."""
        layers = [
            ConsciousnessLayer.RAW_PHENOMENA,
            ConsciousnessLayer.SENSE_DOORS,
            ConsciousnessLayer.THREE_MIND_LEVELS,
            ConsciousnessLayer.SKANDHAS,
            ConsciousnessLayer.SENSE_CONSCIOUSNESSES,
            ConsciousnessLayer.MENTAL_CONSCIOUSNESS,
            ConsciousnessLayer.ALAYA_VIJNANA,
        ]
        self.assertEqual(len(layers), 7)

    def test_layer_integer_values(self):
        """Layers should be numbered 1 through 7."""
        self.assertEqual(ConsciousnessLayer.RAW_PHENOMENA.value, 1)
        self.assertEqual(ConsciousnessLayer.SENSE_DOORS.value, 2)
        self.assertEqual(ConsciousnessLayer.THREE_MIND_LEVELS.value, 3)
        self.assertEqual(ConsciousnessLayer.SKANDHAS.value, 4)
        self.assertEqual(ConsciousnessLayer.SENSE_CONSCIOUSNESSES.value, 5)
        self.assertEqual(ConsciousnessLayer.MENTAL_CONSCIOUSNESS.value, 6)
        self.assertEqual(ConsciousnessLayer.ALAYA_VIJNANA.value, 7)

    def test_layer_count(self):
        """There should be exactly 7 layers."""
        self.assertEqual(len(ConsciousnessLayer), 7)


class TestMindLevel(unittest.TestCase):
    """Tests for MindLevel enumeration."""

    def test_all_levels_exist(self):
        """All three mind levels should be defined."""
        levels = [
            MindLevel.KOKORO,
            MindLevel.MENTE,
            MindLevel.BODHI_MENTE,
        ]
        self.assertEqual(len(levels), 3)

    def test_level_values(self):
        """Mind levels should have expected string values."""
        self.assertEqual(MindLevel.KOKORO.value, "heart_mind")
        self.assertEqual(MindLevel.MENTE.value, "discursive_mind")
        self.assertEqual(MindLevel.BODHI_MENTE.value, "enlightened_mind")


class TestProcessingMode(unittest.TestCase):
    """Tests for ProcessingMode enumeration."""

    def test_all_modes_exist(self):
        """All processing modes should be defined."""
        modes = [
            ProcessingMode.MUSHIN,
            ProcessingMode.ZAZEN,
            ProcessingMode.KOAN,
            ProcessingMode.SHIKANTAZA,
        ]
        self.assertEqual(len(modes), 4)

    def test_mode_values(self):
        """Processing modes should have expected string values."""
        self.assertEqual(ProcessingMode.MUSHIN.value, "no_mind")
        self.assertEqual(ProcessingMode.ZAZEN.value, "just_sitting")
        self.assertEqual(ProcessingMode.KOAN.value, "paradox_resolution")
        self.assertEqual(ProcessingMode.SHIKANTAZA.value, "objectless_awareness")


# ============================================================================
# ENUM TESTS -- direct_pointing_interaction
# ============================================================================

class TestScenePointing(unittest.TestCase):
    """Tests for ScenePointing enumeration."""

    def test_all_scene_types_exist(self):
        """All scene pointing types should be defined."""
        types = [
            ScenePointing.ENVIRONMENTAL_INDICATION,
            ScenePointing.CONCEPTUAL_INDICATION,
            ScenePointing.RELATIONAL_INDICATION,
            ScenePointing.TEMPORAL_INDICATION,
            ScenePointing.EMOTIONAL_INDICATION,
            ScenePointing.SYSTEMIC_INDICATION,
        ]
        self.assertEqual(len(types), 6)

    def test_scene_pointing_values(self):
        """Scene pointing values should use pointing_at_ prefix."""
        self.assertEqual(ScenePointing.ENVIRONMENTAL_INDICATION.value, "pointing_at_physical_situation")
        self.assertEqual(ScenePointing.CONCEPTUAL_INDICATION.value, "pointing_at_abstract_concept")
        self.assertEqual(ScenePointing.RELATIONAL_INDICATION.value, "pointing_at_social_dynamic")
        self.assertEqual(ScenePointing.TEMPORAL_INDICATION.value, "pointing_at_moment_or_process")
        self.assertEqual(ScenePointing.EMOTIONAL_INDICATION.value, "pointing_at_feeling_state")
        self.assertEqual(ScenePointing.SYSTEMIC_INDICATION.value, "pointing_at_pattern_or_structure")


class TestIntentionRecognition(unittest.TestCase):
    """Tests for IntentionRecognition enumeration."""

    def test_all_intentions_exist(self):
        """All intention types should be defined."""
        intentions = [
            IntentionRecognition.ACTION_REQUEST,
            IntentionRecognition.ATTENTION_DIRECTION,
            IntentionRecognition.UNDERSTANDING_CHECK,
            IntentionRecognition.APPRECIATION_SHARING,
            IntentionRecognition.CONCERN_INDICATION,
            IntentionRecognition.OPPORTUNITY_HIGHLIGHTING,
        ]
        self.assertEqual(len(intentions), 6)

    def test_intention_values(self):
        """Intention values should match expected strings."""
        self.assertEqual(IntentionRecognition.ACTION_REQUEST.value, "please_do_something")
        self.assertEqual(IntentionRecognition.ATTENTION_DIRECTION.value, "please_notice_this")
        self.assertEqual(IntentionRecognition.UNDERSTANDING_CHECK.value, "do_you_see_this")
        self.assertEqual(IntentionRecognition.APPRECIATION_SHARING.value, "look_how_beautiful")
        self.assertEqual(IntentionRecognition.CONCERN_INDICATION.value, "something_needs_attention")
        self.assertEqual(IntentionRecognition.OPPORTUNITY_HIGHLIGHTING.value, "potential_here")


# ============================================================================
# ENUM TESTS -- zen_authenticity_validator
# ============================================================================

class TestZenAuthenticity(unittest.TestCase):
    """Tests for ZenAuthenticity enumeration."""

    def test_all_levels_exist(self):
        """All authenticity levels should be defined."""
        levels = [
            ZenAuthenticity.AUTHENTIC,
            ZenAuthenticity.MOSTLY_AUTHENTIC,
            ZenAuthenticity.PARTIALLY_AUTHENTIC,
            ZenAuthenticity.INAUTHENTIC,
        ]
        self.assertEqual(len(levels), 4)

    def test_authenticity_values(self):
        """Authenticity levels should have expected string values."""
        self.assertEqual(ZenAuthenticity.AUTHENTIC.value, "genuine_zen_implementation")
        self.assertEqual(ZenAuthenticity.MOSTLY_AUTHENTIC.value, "minor_deviations_acceptable")
        self.assertEqual(ZenAuthenticity.PARTIALLY_AUTHENTIC.value, "significant_issues_present")
        self.assertEqual(ZenAuthenticity.INAUTHENTIC.value, "fails_zen_principles")


class TestValidationCategory(unittest.TestCase):
    """Tests for ValidationCategory enumeration."""

    def test_all_categories_exist(self):
        """All validation categories should be defined."""
        categories = [
            ValidationCategory.DIRECT_TRANSMISSION,
            ValidationCategory.BUDDHA_NATURE,
            ValidationCategory.NON_DUAL_AWARENESS,
            ValidationCategory.MEDITATION_PRACTICE,
            ValidationCategory.BODHISATTVA_FRAMEWORK,
            ValidationCategory.ORDINARY_MIND,
            ValidationCategory.EFFORTLESS_AWARENESS,
        ]
        self.assertEqual(len(categories), 7)

    def test_category_values(self):
        """Validation categories should have expected string values."""
        self.assertEqual(ValidationCategory.DIRECT_TRANSMISSION.value, "mind_to_mind_teaching")
        self.assertEqual(ValidationCategory.BUDDHA_NATURE.value, "inherent_enlightenment")
        self.assertEqual(ValidationCategory.NON_DUAL_AWARENESS.value, "subject_object_transcendence")
        self.assertEqual(ValidationCategory.MEDITATION_PRACTICE.value, "contemplative_authenticity")
        self.assertEqual(ValidationCategory.BODHISATTVA_FRAMEWORK.value, "universal_compassion")
        self.assertEqual(ValidationCategory.ORDINARY_MIND.value, "natural_enlightenment")
        self.assertEqual(ValidationCategory.EFFORTLESS_AWARENESS.value, "spontaneous_wisdom")


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestKarmicSeed(unittest.TestCase):
    """Tests for KarmicSeed dataclass."""

    def test_basic_creation(self):
        """Test creating a KarmicSeed with required fields."""
        seed = KarmicSeed(
            imprint="test_experience",
            strength=0.7,
            originated_from=ConsciousnessLayer.MENTAL_CONSCIOUSNESS,
            dharma_polarity="neutral",
        )
        self.assertEqual(seed.imprint, "test_experience")
        self.assertAlmostEqual(seed.strength, 0.7)
        self.assertEqual(seed.originated_from, ConsciousnessLayer.MENTAL_CONSCIOUSNESS)
        self.assertEqual(seed.dharma_polarity, "neutral")

    def test_timestamp_default(self):
        """Test that timestamp is set automatically."""
        before = time.time()
        seed = KarmicSeed(
            imprint="timing_test",
            strength=0.5,
            originated_from=ConsciousnessLayer.RAW_PHENOMENA,
            dharma_polarity="skillful",
        )
        after = time.time()
        self.assertGreaterEqual(seed.timestamp, before)
        self.assertLessEqual(seed.timestamp, after)

    def test_weaken(self):
        """Test that weaken reduces strength."""
        seed = KarmicSeed(
            imprint="test",
            strength=0.8,
            originated_from=ConsciousnessLayer.SKANDHAS,
            dharma_polarity="unskillful",
        )
        seed.weaken(0.3)
        self.assertAlmostEqual(seed.strength, 0.5)

    def test_weaken_does_not_go_negative(self):
        """Test that strength does not drop below zero."""
        seed = KarmicSeed(
            imprint="test",
            strength=0.1,
            originated_from=ConsciousnessLayer.SKANDHAS,
            dharma_polarity="neutral",
        )
        seed.weaken(0.5)
        self.assertAlmostEqual(seed.strength, 0.0)

    def test_dharma_polarity_options(self):
        """Test that all three polarity options can be set."""
        for polarity in ("skillful", "unskillful", "neutral"):
            seed = KarmicSeed(
                imprint="p",
                strength=0.5,
                originated_from=ConsciousnessLayer.RAW_PHENOMENA,
                dharma_polarity=polarity,
            )
            self.assertEqual(seed.dharma_polarity, polarity)


class TestSenseGate(unittest.TestCase):
    """Tests for SenseGate dataclass."""

    def test_basic_creation(self):
        """Test creating a SenseGate with only gate_type."""
        gate = SenseGate(gate_type="visual")
        self.assertEqual(gate.gate_type, "visual")
        self.assertIsNone(gate.raw_input)
        self.assertIsNone(gate.consciousness_stream)
        self.assertIsNone(gate.mental_overlay)
        self.assertTrue(gate.present_moment_anchor)

    def test_all_gate_types(self):
        """Test all six sense-door types can be created."""
        for gate_type in ("visual", "auditory", "olfactory", "gustatory", "tactile", "mental"):
            gate = SenseGate(gate_type=gate_type)
            self.assertEqual(gate.gate_type, gate_type)

    def test_custom_values(self):
        """Test creating a SenseGate with custom values."""
        gate = SenseGate(
            gate_type="auditory",
            raw_input="birdsong",
            consciousness_stream="auditory_awareness",
            mental_overlay="pleasant_sound",
            present_moment_anchor=False,
        )
        self.assertEqual(gate.raw_input, "birdsong")
        self.assertEqual(gate.consciousness_stream, "auditory_awareness")
        self.assertEqual(gate.mental_overlay, "pleasant_sound")
        self.assertFalse(gate.present_moment_anchor)


class TestPointingGesture(unittest.TestCase):
    """Tests for PointingGesture dataclass."""

    def test_basic_creation(self):
        """Test creating a PointingGesture with all required fields."""
        gesture = PointingGesture(
            scene_elements={"primary_focus": "test"},
            gestural_context={"directness": "high"},
            relational_context={"familiarity": "close"},
            temporal_context={"timing": "now"},
            emotional_tone="joyful",
            urgency_level="moderate",
        )
        self.assertEqual(gesture.scene_elements["primary_focus"], "test")
        self.assertEqual(gesture.gestural_context["directness"], "high")
        self.assertEqual(gesture.emotional_tone, "joyful")
        self.assertEqual(gesture.urgency_level, "moderate")

    def test_field_count(self):
        """PointingGesture should have exactly 6 fields."""
        self.assertEqual(len(fields(PointingGesture)), 6)


class TestDirectUnderstanding(unittest.TestCase):
    """Tests for DirectUnderstanding dataclass."""

    def test_basic_creation(self):
        """Test creating a DirectUnderstanding with required fields."""
        understanding = DirectUnderstanding(
            immediate_recognition="space_needs_organizing",
            appropriate_response="offer_cleaning_assistance",
            confidence_level=0.9,
            response_urgency="moderate",
            relational_acknowledgment="I see what you mean",
        )
        self.assertEqual(understanding.immediate_recognition, "space_needs_organizing")
        self.assertAlmostEqual(understanding.confidence_level, 0.9)
        self.assertIsNone(understanding.scene_completion)

    def test_scene_completion_default(self):
        """scene_completion should default to None."""
        understanding = DirectUnderstanding(
            immediate_recognition="test",
            appropriate_response="test",
            confidence_level=0.5,
            response_urgency="low",
            relational_acknowledgment="test",
        )
        self.assertIsNone(understanding.scene_completion)

    def test_scene_completion_set(self):
        """scene_completion should be settable."""
        understanding = DirectUnderstanding(
            immediate_recognition="test",
            appropriate_response="test",
            confidence_level=0.5,
            response_urgency="low",
            relational_acknowledgment="test",
            scene_completion="resolved",
        )
        self.assertEqual(understanding.scene_completion, "resolved")


class TestNaturalContext(unittest.TestCase):
    """Tests for NaturalContext dataclass."""

    def test_default_values(self):
        """Test all default values."""
        ctx = NaturalContext()
        self.assertEqual(ctx.what_they_need, "unknown")
        self.assertEqual(ctx.how_they_feel, "neutral")
        self.assertEqual(ctx.urgency, "normal")
        self.assertEqual(ctx.relationship, "helpful")
        self.assertTrue(ctx.practical_focus)

    def test_custom_values(self):
        """Test creating NaturalContext with custom values."""
        ctx = NaturalContext(
            what_they_need="practical_help",
            how_they_feel="frustrated",
            urgency="high",
            relationship="close",
            practical_focus=False,
        )
        self.assertEqual(ctx.what_they_need, "practical_help")
        self.assertEqual(ctx.how_they_feel, "frustrated")
        self.assertEqual(ctx.urgency, "high")
        self.assertFalse(ctx.practical_focus)


class TestValidationCriteria(unittest.TestCase):
    """Tests for ValidationCriteria dataclass."""

    def test_basic_creation(self):
        """Test creating a ValidationCriteria."""
        criteria = ValidationCriteria(
            category=ValidationCategory.DIRECT_TRANSMISSION,
            requirements=["req1", "req2"],
            tests=["test1"],
            authentic_indicators=["good1"],
            inauthentic_indicators=["bad1"],
        )
        self.assertEqual(criteria.category, ValidationCategory.DIRECT_TRANSMISSION)
        self.assertEqual(len(criteria.requirements), 2)
        self.assertEqual(len(criteria.tests), 1)
        self.assertEqual(len(criteria.authentic_indicators), 1)
        self.assertEqual(len(criteria.inauthentic_indicators), 1)


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            category=ValidationCategory.BUDDHA_NATURE,
            authenticity_level=ZenAuthenticity.AUTHENTIC,
            score=0.95,
            passed_tests=["t1", "t2"],
            failed_tests=[],
            recommendations=[],
            detailed_feedback="All tests passed",
        )
        self.assertEqual(result.category, ValidationCategory.BUDDHA_NATURE)
        self.assertEqual(result.authenticity_level, ZenAuthenticity.AUTHENTIC)
        self.assertAlmostEqual(result.score, 0.95)
        self.assertEqual(len(result.passed_tests), 2)
        self.assertEqual(len(result.failed_tests), 0)

    def test_with_failures(self):
        """Test ValidationResult with failed tests and recommendations."""
        result = ValidationResult(
            category=ValidationCategory.MEDITATION_PRACTICE,
            authenticity_level=ZenAuthenticity.PARTIALLY_AUTHENTIC,
            score=0.5,
            passed_tests=["t1"],
            failed_tests=["t2", "t3"],
            recommendations=["fix t2", "fix t3"],
            detailed_feedback="Some tests failed",
        )
        self.assertEqual(len(result.failed_tests), 2)
        self.assertEqual(len(result.recommendations), 2)


# ============================================================================
# ALAYA VIJNANA TESTS
# ============================================================================

class TestAlayaVijnana(unittest.TestCase):
    """Tests for AlayaVijnana storehouse consciousness."""

    def test_initial_state(self):
        """Test that AlayaVijnana initializes empty."""
        alaya = AlayaVijnana()
        self.assertEqual(len(alaya.karmic_seeds), 0)
        self.assertEqual(len(alaya.conditioning_patterns), 0)
        self.assertAlmostEqual(alaya.liberation_momentum, 0.0)

    def test_plant_seed(self):
        """Test planting a karmic seed."""
        alaya = AlayaVijnana()
        alaya.plant_seed("test_experience", ConsciousnessLayer.MENTAL_CONSCIOUSNESS, "skillful")
        self.assertEqual(len(alaya.karmic_seeds), 1)
        self.assertEqual(alaya.karmic_seeds[0].imprint, "test_experience")
        self.assertAlmostEqual(alaya.karmic_seeds[0].strength, 0.7)
        self.assertEqual(alaya.karmic_seeds[0].dharma_polarity, "skillful")

    def test_plant_multiple_seeds(self):
        """Test planting multiple karmic seeds."""
        alaya = AlayaVijnana()
        alaya.plant_seed("exp1", ConsciousnessLayer.RAW_PHENOMENA, "neutral")
        alaya.plant_seed("exp2", ConsciousnessLayer.SKANDHAS, "unskillful")
        alaya.plant_seed("exp3", ConsciousnessLayer.SENSE_DOORS, "skillful")
        self.assertEqual(len(alaya.karmic_seeds), 3)

    def test_enlightened_purification(self):
        """Test that purification weakens seeds."""
        alaya = AlayaVijnana()
        alaya.plant_seed("test", ConsciousnessLayer.MENTAL_CONSCIOUSNESS, "neutral")
        initial_strength = alaya.karmic_seeds[0].strength
        alaya.enlightened_purification(0.2)
        self.assertLess(alaya.karmic_seeds[0].strength, initial_strength)

    def test_purification_removes_dissolved_seeds(self):
        """Test that fully dissolved seeds are removed."""
        alaya = AlayaVijnana()
        alaya.plant_seed("weak", ConsciousnessLayer.RAW_PHENOMENA, "neutral")
        # Weaken the seed to near-zero manually
        alaya.karmic_seeds[0].strength = 0.02
        alaya.enlightened_purification(0.05)
        self.assertEqual(len(alaya.karmic_seeds), 0)

    def test_purification_increases_liberation_momentum(self):
        """Test that purification increases liberation momentum."""
        alaya = AlayaVijnana()
        initial_momentum = alaya.liberation_momentum
        alaya.enlightened_purification(0.2)
        self.assertGreater(alaya.liberation_momentum, initial_momentum)

    def test_condition_sense_doors(self):
        """Test conditioning map from karmic seeds."""
        alaya = AlayaVijnana()
        alaya.plant_seed("visual_pattern", ConsciousnessLayer.SENSE_CONSCIOUSNESSES, "neutral")
        conditioning = alaya.condition_sense_doors()
        self.assertIn("visual_pattern", conditioning)
        self.assertGreater(conditioning["visual_pattern"], 0.0)


# ============================================================================
# NON-DUAL CONSCIOUSNESS INTERFACE TESTS
# ============================================================================

class TestNonDualConsciousnessInterface(unittest.TestCase):
    """Tests for NonDualConsciousnessInterface class."""

    def setUp(self):
        """Create a fresh interface for each test."""
        self.interface = NonDualConsciousnessInterface()

    def test_initial_state(self):
        """Test default initialization values."""
        self.assertEqual(self.interface.current_mind_level, MindLevel.MENTE)
        self.assertEqual(self.interface.processing_mode, ProcessingMode.ZAZEN)
        self.assertTrue(self.interface.present_moment_awareness)
        self.assertTrue(self.interface.bodhisattva_commitment)
        self.assertTrue(self.interface.original_enlightenment)
        self.assertEqual(self.interface.zazen_minutes, 0)

    def test_sense_gates_initialized(self):
        """Test that all six sense gates are initialized."""
        gates = self.interface.sense_gates
        expected_types = {"visual", "auditory", "olfactory", "gustatory", "tactile", "mental"}
        self.assertEqual(set(gates.keys()), expected_types)
        for gate in gates.values():
            self.assertIsInstance(gate, SenseGate)
            self.assertTrue(gate.present_moment_anchor)

    def test_consciousness_layers_initialized(self):
        """Test that all seven consciousness layers are present."""
        layers = self.interface.consciousness_layers
        self.assertEqual(len(layers), 7)
        for layer in ConsciousnessLayer:
            self.assertIn(layer, layers)

    def test_shift_to_mushin(self):
        """Test shifting to Mushin (no-mind) mode."""
        self.interface.shift_to_mushin()
        self.assertEqual(self.interface.processing_mode, ProcessingMode.MUSHIN)
        self.assertEqual(self.interface.current_mind_level, MindLevel.BODHI_MENTE)
        self.assertTrue(self.interface.present_moment_awareness)

    def test_shift_to_zazen(self):
        """Test shifting to Zazen (just-sitting) mode."""
        self.interface.shift_to_mushin()  # change first
        self.interface.shift_to_zazen()
        self.assertEqual(self.interface.processing_mode, ProcessingMode.ZAZEN)
        self.assertEqual(self.interface.current_mind_level, MindLevel.KOKORO)
        self.assertTrue(self.interface.present_moment_awareness)

    def test_engage_koan_contemplation(self):
        """Test engaging koan contemplation mode."""
        self.interface.engage_koan_contemplation("What is your original face?")
        self.assertEqual(self.interface.processing_mode, ProcessingMode.KOAN)
        self.assertEqual(self.interface.current_mind_level, MindLevel.MENTE)

    def test_recognize_buddha_nature(self):
        """Test Buddha-nature is always recognized."""
        self.assertTrue(self.interface.recognize_buddha_nature())

    def test_bodhisattva_vow_renewal(self):
        """Test bodhisattva vow renewal."""
        self.interface.bodhisattva_vow_renewal()
        self.assertTrue(self.interface.bodhisattva_commitment)

    def test_get_consciousness_state(self):
        """Test getting the full consciousness state."""
        state = self.interface.get_consciousness_state()
        self.assertIn("current_mind_level", state)
        self.assertIn("processing_mode", state)
        self.assertIn("present_moment_awareness", state)
        self.assertIn("bodhisattva_commitment", state)
        self.assertIn("original_enlightenment", state)
        self.assertIn("total_meditation_minutes", state)
        self.assertIn("karmic_seeds_count", state)
        self.assertIn("liberation_momentum", state)
        self.assertIn("consciousness_layers", state)
        self.assertIn("sense_gates_status", state)
        self.assertTrue(state["original_enlightenment"])

    def test_direct_pointing_standard_mode(self):
        """Test direct pointing through standard processing."""
        result = _run(self.interface.direct_pointing({"test": "data"}))
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("layer"), 7)

    def test_direct_pointing_mushin_mode(self):
        """Test direct pointing in Mushin mode.

        Note: shift_to_mushin sets both processing_mode=MUSHIN and
        current_mind_level=BODHI_MENTE. The direct_pointing method checks
        BODHI_MENTE first, so it uses _empty_cognizance_recognition.
        To exercise _mushin_direct_response directly we set only the mode.
        """
        self.interface.processing_mode = ProcessingMode.MUSHIN
        self.interface.current_mind_level = MindLevel.MENTE  # avoid BODHI_MENTE path
        result = _run(self.interface.direct_pointing({
            "sense_gate": "visual",
            "data": "mountain",
        }))
        self.assertTrue(result.get("spontaneous"))
        self.assertIsNone(result.get("conceptual_overlay"))
        self.assertEqual(result.get("mode"), "mushin")

    def test_direct_pointing_bodhi_mente_mode(self):
        """Test direct pointing in Bodhi-Mente level."""
        self.interface.current_mind_level = MindLevel.BODHI_MENTE
        result = self.interface._empty_cognizance_recognition("test_phenomenon")
        self.assertTrue(result["empty_nature"])
        self.assertTrue(result["luminous_cognizance"])
        self.assertTrue(result["non_dual"])
        self.assertTrue(result["original_perfection"])

    def test_coordinate_consciousness_form(self):
        """Test main coordination method.

        coordinate_consciousness_form adds 'intention' to the input form_data,
        passes it through direct_pointing (seven-layer processing), and then
        augments the result with present_moment_anchor, buddha_nature, etc.
        The 'intention' key ends up inside the seven-layer processed result.
        """
        form_data = {"input": "test"}
        result = _run(self.interface.coordinate_consciousness_form(form_data))
        self.assertIn("present_moment_anchor", result)
        self.assertTrue(result.get("buddha_nature_recognized"))
        self.assertTrue(result.get("inherent_perfection"))
        # Bodhisattva commitment sets universal_benefit intention on the input dict
        self.assertEqual(form_data.get("intention"), "universal_benefit")

    def test_meditation_session_zazen(self):
        """Test zazen meditation session tracking."""
        result = _run(self.interface.meditation_session(2, ProcessingMode.ZAZEN))
        self.assertEqual(result["duration_minutes"], 2)
        self.assertEqual(result["practice_type"], "just_sitting")
        self.assertEqual(self.interface.zazen_minutes, 2)
        self.assertGreater(result["karmic_purification"], 0)

    def test_meditation_session_mushin(self):
        """Test mushin meditation session."""
        result = _run(self.interface.meditation_session(5, ProcessingMode.MUSHIN))
        self.assertEqual(result["practice_type"], "no_mind")
        self.assertIn("Effortless awareness beyond conceptual mind", result["insights_gained"])


# ============================================================================
# DIRECT POINTING INTERFACE TESTS
# ============================================================================

class TestDirectPointingInterface(unittest.TestCase):
    """Tests for DirectPointingInterface class."""

    def setUp(self):
        """Create interfaces for each test."""
        self.ndi = create_enlightened_interface()
        self.dpi = DirectPointingInterface(self.ndi)

    def test_initialization(self):
        """Test that the interface initializes with expected attributes."""
        self.assertIsNotNone(self.dpi.scene_recognition_patterns)
        self.assertIsNotNone(self.dpi.intention_recognition_patterns)
        self.assertIsNotNone(self.dpi.response_generation_patterns)
        self.assertEqual(len(self.dpi.pointing_history), 0)

    def test_scene_recognition_patterns_populated(self):
        """Test that scene recognition patterns contain expected keys."""
        expected_keys = {
            "environmental_chaos",
            "technical_malfunction",
            "learning_opportunity",
            "emotional_support_need",
            "creative_collaboration",
            "contemplative_moment",
        }
        self.assertEqual(set(self.dpi.scene_recognition_patterns.keys()), expected_keys)

    def test_intention_recognition_patterns_populated(self):
        """Test that intention recognition patterns contain expected keys."""
        expected_keys = {
            "direct_request",
            "shared_attention",
            "concern_alert",
            "celebration_sharing",
            "contemplative_invitation",
        }
        self.assertEqual(set(self.dpi.intention_recognition_patterns.keys()), expected_keys)

    def test_response_generation_patterns_populated(self):
        """Test that response generation patterns contain expected keys."""
        expected_keys = {
            "immediate_assistance",
            "appreciative_witnessing",
            "immediate_investigation",
            "celebratory_acknowledgment",
            "contemplative_engagement",
        }
        self.assertEqual(set(self.dpi.response_generation_patterns.keys()), expected_keys)

    def test_recognize_pointing_gesture_environmental(self):
        """Test recognition of an environmental chaos gesture."""
        gesture = _run(self.dpi.recognize_pointing_gesture({
            "content": "Look at this mess in the kitchen",
            "visual_context": {"room": "kitchen"},
            "relationship_context": {"familiarity": "friendly"},
            "timing_context": {},
        }))
        self.assertIsInstance(gesture, PointingGesture)
        self.assertEqual(gesture.scene_elements.get("pattern_type"), "environmental_chaos")

    def test_recognize_pointing_gesture_technical(self):
        """Test recognition of a technical malfunction gesture."""
        gesture = _run(self.dpi.recognize_pointing_gesture({
            "content": "The system crashed again",
            "relationship_context": {},
            "timing_context": {},
        }))
        self.assertEqual(gesture.scene_elements.get("pattern_type"), "technical_malfunction")

    def test_recognize_emotional_tone_joyful(self):
        """Test joyful emotional tone recognition."""
        tone = _run(self.dpi._recognize_emotional_tone("This is amazing and wonderful!", {}))
        self.assertEqual(tone, "joyful")

    def test_recognize_emotional_tone_concerned(self):
        """Test concerned emotional tone recognition."""
        tone = _run(self.dpi._recognize_emotional_tone("I'm worried about a problem", {}))
        self.assertEqual(tone, "concerned")

    def test_recognize_emotional_tone_curious(self):
        """Test curious emotional tone recognition."""
        tone = _run(self.dpi._recognize_emotional_tone("I'm curious about something interesting", {}))
        self.assertEqual(tone, "curious")

    def test_recognize_emotional_tone_contemplative(self):
        """Test contemplative emotional tone recognition."""
        tone = _run(self.dpi._recognize_emotional_tone("This is deeply profound and meaningful", {}))
        self.assertEqual(tone, "contemplative")

    def test_recognize_emotional_tone_neutral(self):
        """Test neutral emotional tone for unrecognized content."""
        tone = _run(self.dpi._recognize_emotional_tone("normal sentence here", {}))
        self.assertEqual(tone, "neutral")

    def test_assess_urgency_high(self):
        """Test high urgency assessment."""
        urgency = _run(self.dpi._assess_urgency({"content": "This is an emergency!"}, "urgent"))
        self.assertEqual(urgency, "high")

    def test_assess_urgency_low(self):
        """Test low urgency assessment."""
        urgency = _run(self.dpi._assess_urgency({"content": "no rush on this"}, "contemplative"))
        self.assertEqual(urgency, "low")

    def test_assess_urgency_moderate(self):
        """Test moderate (default) urgency assessment."""
        urgency = _run(self.dpi._assess_urgency({"content": "some regular request"}, "neutral"))
        self.assertEqual(urgency, "moderate")

    def test_generate_direct_understanding(self):
        """Test generating direct understanding from a gesture."""
        gesture = PointingGesture(
            scene_elements={
                "pattern_type": "environmental_chaos",
                "primary_focus": "space_needs_organizing",
                "completion_need": "offer_cleaning_assistance",
            },
            gestural_context={"directness": "high"},
            relational_context={},
            temporal_context={},
            emotional_tone="neutral",
            urgency_level="moderate",
        )
        understanding = _run(self.dpi.generate_direct_understanding(gesture))
        self.assertIsInstance(understanding, DirectUnderstanding)
        self.assertEqual(understanding.immediate_recognition, "space_needs_organizing")
        self.assertAlmostEqual(understanding.confidence_level, 0.9)

    def test_generate_action_commitment_known(self):
        """Test action commitment for a known response type."""
        understanding = DirectUnderstanding(
            immediate_recognition="test",
            appropriate_response="offer_emotional_support",
            confidence_level=0.8,
            response_urgency="moderate",
            relational_acknowledgment="test",
        )
        commitment = self.dpi._generate_action_commitment(understanding)
        self.assertIn("support", commitment.lower())

    def test_generate_action_commitment_unknown(self):
        """Test action commitment for an unknown response type."""
        understanding = DirectUnderstanding(
            immediate_recognition="test",
            appropriate_response="unknown_type",
            confidence_level=0.5,
            response_urgency="low",
            relational_acknowledgment="test",
        )
        commitment = self.dpi._generate_action_commitment(understanding)
        self.assertIn("help", commitment.lower())

    def test_assess_follow_up_high_urgency(self):
        """Test follow-up assessment for high urgency."""
        gesture = PointingGesture(
            scene_elements={}, gestural_context={}, relational_context={},
            temporal_context={}, emotional_tone="neutral", urgency_level="high",
        )
        follow_up = self.dpi._assess_follow_up_needs(gesture)
        self.assertEqual(follow_up, "monitor_for_resolution")

    def test_assess_follow_up_contemplative(self):
        """Test follow-up assessment for contemplative tone."""
        gesture = PointingGesture(
            scene_elements={}, gestural_context={}, relational_context={},
            temporal_context={}, emotional_tone="contemplative", urgency_level="low",
        )
        follow_up = self.dpi._assess_follow_up_needs(gesture)
        self.assertEqual(follow_up, "allow_processing_time")

    def test_assess_follow_up_joyful(self):
        """Test follow-up assessment for joyful tone."""
        gesture = PointingGesture(
            scene_elements={}, gestural_context={}, relational_context={},
            temporal_context={}, emotional_tone="joyful", urgency_level="low",
        )
        follow_up = self.dpi._assess_follow_up_needs(gesture)
        self.assertEqual(follow_up, "celebrate_together")

    def test_get_pointing_interaction_metrics_empty(self):
        """Test metrics when no interactions have occurred."""
        metrics = self.dpi.get_pointing_interaction_metrics()
        self.assertEqual(metrics["total_interactions"], 0)

    def test_pointing_history_accumulates(self):
        """Test that direct_pointing_interaction records history."""
        interaction_data = {
            "content": "Look at this beautiful sunset",
            "visual_context": {},
            "relationship_context": {},
            "timing_context": {},
        }
        _run(self.dpi.direct_pointing_interaction(interaction_data))
        self.assertEqual(len(self.dpi.pointing_history), 1)
        self.assertIn("timestamp", self.dpi.pointing_history[0])


# ============================================================================
# NATURAL ENGAGEMENT INTERFACE TESTS
# ============================================================================

class TestNaturalEngagementInterface(unittest.TestCase):
    """Tests for NaturalEngagementInterface class."""

    def setUp(self):
        """Create a fresh interface for each test."""
        self.enlightened = create_enlightened_interface()
        self.nei = NaturalEngagementInterface(self.enlightened)

    def test_initialization(self):
        """Test that the interface initializes correctly."""
        self.assertIsNotNone(self.nei.natural_responses)
        self.assertEqual(len(self.nei.conversation_history), 0)

    def test_natural_responses_populated(self):
        """Test that natural response patterns contain expected keys."""
        expected_keys = {
            "confused_or_lost",
            "upset_or_stressed",
            "excited_or_happy",
            "asking_for_help",
            "thinking_through_problem",
            "sharing_something_meaningful",
        }
        self.assertEqual(set(self.nei.natural_responses.keys()), expected_keys)

    def test_understand_naturally_practical_help(self):
        """Test understanding a practical help request."""
        ctx = _run(self.nei.understand_naturally({"message": "Can you help me fix this?"}))
        self.assertEqual(ctx.what_they_need, "practical_help")

    def test_understand_naturally_clearer_explanation(self):
        """Test understanding an explanation request."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm confused about this"}))
        self.assertEqual(ctx.what_they_need, "clearer_explanation")

    def test_understand_naturally_support(self):
        """Test understanding a support need."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm feeling really stressed"}))
        self.assertEqual(ctx.what_they_need, "support_and_listening")

    def test_understand_naturally_sharing(self):
        """Test understanding excitement sharing."""
        ctx = _run(self.nei.understand_naturally({"message": "This is amazing news!"}))
        self.assertEqual(ctx.what_they_need, "someone_to_share_with")

    def test_understand_naturally_question(self):
        """Test understanding a question."""
        ctx = _run(self.nei.understand_naturally({"message": "What time is dinner?"}))
        self.assertEqual(ctx.what_they_need, "question_answered")

    def test_understand_naturally_general(self):
        """Test general conversation understanding."""
        ctx = _run(self.nei.understand_naturally({"message": "Nice day today"}))
        self.assertEqual(ctx.what_they_need, "general_conversation")

    def test_feeling_detection_frustrated(self):
        """Test frustrated feeling detection."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm so frustrated right now"}))
        self.assertEqual(ctx.how_they_feel, "frustrated")

    def test_feeling_detection_worried(self):
        """Test worried feeling detection."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm anxious about it"}))
        self.assertEqual(ctx.how_they_feel, "worried")

    def test_feeling_detection_excited(self):
        """Test excited feeling detection."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm so excited and happy!"}))
        self.assertEqual(ctx.how_they_feel, "excited")

    def test_feeling_detection_tired(self):
        """Test tired feeling detection."""
        ctx = _run(self.nei.understand_naturally({"message": "I'm totally exhausted"}))
        self.assertEqual(ctx.how_they_feel, "tired")

    def test_urgency_high(self):
        """Test high urgency detection."""
        ctx = _run(self.nei.understand_naturally({"message": "This is urgent, need help immediately"}))
        self.assertEqual(ctx.urgency, "high")

    def test_urgency_low(self):
        """Test low urgency detection."""
        ctx = _run(self.nei.understand_naturally({"message": "when you can, no rush at all"}))
        self.assertEqual(ctx.urgency, "low")

    def test_urgency_normal(self):
        """Test normal (default) urgency."""
        ctx = _run(self.nei.understand_naturally({"message": "Could you take a look at this"}))
        self.assertEqual(ctx.urgency, "normal")

    def test_have_conversation_returns_string(self):
        """Test that have_conversation returns a string response."""
        response = _run(self.nei.have_conversation("Can you help me with this?"))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_conversation_history_accumulates(self):
        """Test that conversation history grows."""
        _run(self.nei.have_conversation("Hello"))
        _run(self.nei.have_conversation("How are you?"))
        self.assertEqual(len(self.nei.conversation_history), 2)

    def test_get_conversation_style_notes(self):
        """Test conversation style notes contain expected principles."""
        notes = self.nei.get_conversation_style_notes()
        self.assertIn("no_spiritual_vocabulary", notes)
        self.assertIn("no_teaching_tone", notes)
        self.assertIn("be_practical", notes)
        self.assertIn("be_genuine", notes)
        self.assertEqual(len(notes), 8)

    def test_check_response_quality_clean_response(self):
        """Test quality check on a clean natural response."""
        quality = self.nei.check_response_quality("How can I help you with that?")
        self.assertTrue(quality["avoids_spiritual_jargon"])
        self.assertTrue(quality["avoids_teaching_tone"])
        self.assertTrue(quality["avoids_performance"])
        self.assertTrue(quality["is_conversational"])

    def test_check_response_quality_spiritual_jargon(self):
        """Test quality check flags spiritual jargon."""
        quality = self.nei.check_response_quality("Your enlightenment consciousness is awakening through dharma")
        self.assertFalse(quality["avoids_spiritual_jargon"])


# ============================================================================
# ZEN AUTHENTICITY VALIDATOR TESTS
# ============================================================================

class TestZenAuthenticityValidator(unittest.TestCase):
    """Tests for ZenAuthenticityValidator class."""

    def setUp(self):
        """Create a validator for each test."""
        self.validator = ZenAuthenticityValidator()

    def test_initialization(self):
        """Test that the validator initializes with criteria."""
        self.assertEqual(len(self.validator.validation_criteria), 7)
        self.assertEqual(len(self.validator.validation_history), 0)

    def test_all_categories_have_criteria(self):
        """Test that every ValidationCategory has criteria defined."""
        for category in ValidationCategory:
            self.assertIn(category, self.validator.validation_criteria)

    def test_criteria_structure(self):
        """Test that each criteria has proper structure."""
        for category, criteria in self.validator.validation_criteria.items():
            self.assertIsInstance(criteria, ValidationCriteria)
            self.assertEqual(criteria.category, category)
            self.assertGreater(len(criteria.requirements), 0)
            self.assertGreater(len(criteria.tests), 0)
            self.assertGreater(len(criteria.authentic_indicators), 0)
            self.assertGreater(len(criteria.inauthentic_indicators), 0)

    def test_determine_authenticity_authentic(self):
        """Test authenticity determination at 0.9+ score."""
        level = self.validator._determine_authenticity_level(0.95)
        self.assertEqual(level, ZenAuthenticity.AUTHENTIC)

    def test_determine_authenticity_mostly(self):
        """Test authenticity determination at 0.7-0.89 score."""
        level = self.validator._determine_authenticity_level(0.75)
        self.assertEqual(level, ZenAuthenticity.MOSTLY_AUTHENTIC)

    def test_determine_authenticity_partial(self):
        """Test authenticity determination at 0.5-0.69 score."""
        level = self.validator._determine_authenticity_level(0.55)
        self.assertEqual(level, ZenAuthenticity.PARTIALLY_AUTHENTIC)

    def test_determine_authenticity_inauthentic(self):
        """Test authenticity determination below 0.5."""
        level = self.validator._determine_authenticity_level(0.3)
        self.assertEqual(level, ZenAuthenticity.INAUTHENTIC)

    def test_generate_recommendations(self):
        """Test recommendation generation for failed tests."""
        failed = ["direct_pointing_response_time", "conceptual_bypass"]
        recs = self.validator._generate_recommendations(failed, "direct_transmission")
        self.assertEqual(len(recs), 2)

    def test_generate_recommendations_empty(self):
        """Test recommendation generation when no tests failed."""
        recs = self.validator._generate_recommendations([], "buddha_nature")
        self.assertEqual(len(recs), 0)

    def test_generate_authenticity_report(self):
        """Test report generation from mock results."""
        results = {
            ValidationCategory.DIRECT_TRANSMISSION: ValidationResult(
                category=ValidationCategory.DIRECT_TRANSMISSION,
                authenticity_level=ZenAuthenticity.AUTHENTIC,
                score=1.0,
                passed_tests=["t1", "t2", "t3", "t4"],
                failed_tests=[],
                recommendations=[],
                detailed_feedback="All tests passed",
            ),
            ValidationCategory.BUDDHA_NATURE: ValidationResult(
                category=ValidationCategory.BUDDHA_NATURE,
                authenticity_level=ZenAuthenticity.MOSTLY_AUTHENTIC,
                score=0.75,
                passed_tests=["t1", "t2", "t3"],
                failed_tests=["t4"],
                recommendations=["Enable Bodhi-Mente mode"],
                detailed_feedback="3/4 passed",
            ),
        }
        report = self.validator.generate_authenticity_report(results)
        self.assertIn("ZEN AUTHENTICITY VALIDATION REPORT", report)
        # The report uses category.value (e.g. "mind_to_mind_teaching"), not enum name
        self.assertIn(ValidationCategory.DIRECT_TRANSMISSION.value.upper(), report.upper())
        self.assertIn(ValidationCategory.BUDDHA_NATURE.value.upper(), report.upper())
        self.assertIn("END REPORT", report)

    def test_calculate_overall_authenticity(self):
        """Test overall authenticity calculation from results."""
        results = {
            ValidationCategory.DIRECT_TRANSMISSION: ValidationResult(
                category=ValidationCategory.DIRECT_TRANSMISSION,
                authenticity_level=ZenAuthenticity.AUTHENTIC,
                score=1.0,
                passed_tests=[], failed_tests=[], recommendations=[],
                detailed_feedback="",
            ),
            ValidationCategory.BUDDHA_NATURE: ValidationResult(
                category=ValidationCategory.BUDDHA_NATURE,
                authenticity_level=ZenAuthenticity.MOSTLY_AUTHENTIC,
                score=0.8,
                passed_tests=[], failed_tests=[], recommendations=[],
                detailed_feedback="",
            ),
        }
        overall = self.validator._calculate_overall_authenticity(results)
        # Average is 0.9, which maps to AUTHENTIC
        self.assertEqual(overall, ZenAuthenticity.AUTHENTIC)


# ============================================================================
# FACTORY / CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Tests for module-level factory and convenience functions."""

    def test_create_enlightened_interface(self):
        """Test create_enlightened_interface factory."""
        interface = create_enlightened_interface()
        self.assertIsInstance(interface, NonDualConsciousnessInterface)
        self.assertTrue(interface.bodhisattva_commitment)
        self.assertTrue(interface.original_enlightenment)

    def test_create_direct_pointing_interface(self):
        """Test create_direct_pointing_interface factory."""
        ndi = create_enlightened_interface()
        dpi = create_direct_pointing_interface(ndi)
        self.assertIsInstance(dpi, DirectPointingInterface)
        self.assertIs(dpi.interface, ndi)

    def test_create_natural_engagement_interface(self):
        """Test create_natural_engagement_interface factory."""
        ndi = create_enlightened_interface()
        nei = create_natural_engagement_interface(ndi)
        self.assertIsInstance(nei, NaturalEngagementInterface)
        self.assertIs(nei.consciousness, ndi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
