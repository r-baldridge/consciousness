#!/usr/bin/env python3
"""
Test Suite for Form 06: Interoceptive Consciousness.

Tests cover:
- All enumerations (InteroceptiveChannel, BodySystem, HomeostaticNeed, BodyStateCategory, InteroceptiveAccuracy)
- All input/output dataclasses
- InteroceptiveConsciousnessInterface processing pipeline
- Body state assessment, homeostatic needs, emotional grounding
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
    InteroceptiveChannel,
    BodySystem,
    HomeostaticNeed,
    BodyStateCategory,
    InteroceptiveAccuracy,
    # Input dataclasses
    OrganSignal,
    HomeostaticData,
    InteroceptiveInput,
    # Output dataclasses
    BodyStateAssessment,
    HomeostaticNeedsReport,
    EmotionalGrounding,
    InteroceptiveOutput,
    # Main interface
    InteroceptiveConsciousnessInterface,
    # Convenience functions
    create_interoceptive_interface,
    create_simple_interoceptive_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestInteroceptiveChannel(unittest.TestCase):
    """Tests for InteroceptiveChannel enumeration."""

    def test_all_channels_exist(self):
        """All interoceptive channels should be defined."""
        channels = [
            InteroceptiveChannel.CARDIAC, InteroceptiveChannel.RESPIRATORY,
            InteroceptiveChannel.GASTROINTESTINAL, InteroceptiveChannel.THERMOREGULATORY,
            InteroceptiveChannel.NOCICEPTIVE, InteroceptiveChannel.BLADDER,
            InteroceptiveChannel.IMMUNE, InteroceptiveChannel.METABOLIC,
            InteroceptiveChannel.MUSCULAR, InteroceptiveChannel.OSMOTIC,
        ]
        self.assertEqual(len(channels), 10)

    def test_values(self):
        """Channels should have expected values."""
        self.assertEqual(InteroceptiveChannel.CARDIAC.value, "cardiac")
        self.assertEqual(InteroceptiveChannel.RESPIRATORY.value, "respiratory")


class TestBodySystem(unittest.TestCase):
    """Tests for BodySystem enumeration."""

    def test_all_systems_exist(self):
        """All body systems should be defined."""
        systems = [
            BodySystem.CARDIOVASCULAR, BodySystem.RESPIRATORY_SYSTEM,
            BodySystem.DIGESTIVE, BodySystem.NERVOUS,
            BodySystem.ENDOCRINE, BodySystem.IMMUNE_SYSTEM,
            BodySystem.MUSCULOSKELETAL, BodySystem.URINARY,
        ]
        self.assertEqual(len(systems), 8)


class TestHomeostaticNeed(unittest.TestCase):
    """Tests for HomeostaticNeed enumeration."""

    def test_all_needs_exist(self):
        """All homeostatic needs should be defined."""
        needs = [
            HomeostaticNeed.HUNGER, HomeostaticNeed.THIRST,
            HomeostaticNeed.OXYGEN, HomeostaticNeed.WARMTH,
            HomeostaticNeed.COOLING, HomeostaticNeed.REST,
            HomeostaticNeed.ELIMINATION, HomeostaticNeed.MOVEMENT,
            HomeostaticNeed.SAFETY, HomeostaticNeed.NONE,
        ]
        self.assertEqual(len(needs), 10)

    def test_values(self):
        """Needs should have expected values."""
        self.assertEqual(HomeostaticNeed.HUNGER.value, "hunger")
        self.assertEqual(HomeostaticNeed.THIRST.value, "thirst")


class TestBodyStateCategory(unittest.TestCase):
    """Tests for BodyStateCategory enumeration."""

    def test_all_states_exist(self):
        """All body state categories should be defined."""
        states = [
            BodyStateCategory.OPTIMAL, BodyStateCategory.STRESSED,
            BodyStateCategory.FATIGUED, BodyStateCategory.ILL,
            BodyStateCategory.ENERGIZED, BodyStateCategory.RELAXED,
            BodyStateCategory.ANXIOUS, BodyStateCategory.DEPLETED,
            BodyStateCategory.RECOVERING,
        ]
        self.assertEqual(len(states), 9)


class TestInteroceptiveAccuracy(unittest.TestCase):
    """Tests for InteroceptiveAccuracy enumeration."""

    def test_all_levels_exist(self):
        """All accuracy levels should be defined."""
        levels = [
            InteroceptiveAccuracy.HIGH, InteroceptiveAccuracy.MODERATE,
            InteroceptiveAccuracy.LOW, InteroceptiveAccuracy.HYPERSENSITIVE,
            InteroceptiveAccuracy.DISSOCIATED,
        ]
        self.assertEqual(len(levels), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestOrganSignal(unittest.TestCase):
    """Tests for OrganSignal dataclass."""

    def test_creation(self):
        """Should create organ signal."""
        sig = OrganSignal(
            channel=InteroceptiveChannel.CARDIAC,
            body_system=BodySystem.CARDIOVASCULAR,
            activation_level=0.6,
            deviation_from_baseline=0.1,
            urgency=0.2,
        )
        self.assertEqual(sig.channel, InteroceptiveChannel.CARDIAC)
        self.assertEqual(sig.body_system, BodySystem.CARDIOVASCULAR)
        self.assertEqual(sig.activation_level, 0.6)

    def test_to_dict(self):
        """Should convert to dictionary."""
        sig = OrganSignal(
            channel=InteroceptiveChannel.RESPIRATORY,
            body_system=BodySystem.RESPIRATORY_SYSTEM,
            activation_level=0.5,
            deviation_from_baseline=0.0,
            urgency=0.1,
        )
        d = sig.to_dict()
        self.assertEqual(d["channel"], "respiratory")
        self.assertEqual(d["body_system"], "respiratory_system")


class TestHomeostaticData(unittest.TestCase):
    """Tests for HomeostaticData dataclass."""

    def test_creation_defaults(self):
        """Should create homeostatic data with defaults."""
        hd = HomeostaticData()
        self.assertEqual(hd.body_temperature, 0.5)
        self.assertEqual(hd.blood_glucose, 0.5)
        self.assertEqual(hd.oxygen_saturation, 0.95)

    def test_creation_custom(self):
        """Should create homeostatic data with custom values."""
        hd = HomeostaticData(
            body_temperature=0.6,
            blood_glucose=0.4,
            hydration_level=0.5,
            energy_reserves=0.3,
            stress_hormones=0.7,
            immune_activation=0.6,
        )
        self.assertEqual(hd.body_temperature, 0.6)
        self.assertEqual(hd.stress_hormones, 0.7)
        self.assertEqual(hd.immune_activation, 0.6)

    def test_to_dict(self):
        """Should convert to dictionary."""
        hd = HomeostaticData()
        d = hd.to_dict()
        self.assertIn("body_temperature", d)
        self.assertIn("blood_glucose", d)
        self.assertIn("stress_hormones", d)


class TestInteroceptiveInput(unittest.TestCase):
    """Tests for InteroceptiveInput dataclass."""

    def test_creation_defaults(self):
        """Should create input with defaults."""
        inp = InteroceptiveInput()
        self.assertEqual(len(inp.organ_signals), 0)
        self.assertIsNone(inp.homeostatic_data)
        self.assertEqual(inp.subjective_energy, 0.5)

    def test_creation_full(self):
        """Should create input with full data."""
        inp = InteroceptiveInput(
            organ_signals=[
                OrganSignal(InteroceptiveChannel.CARDIAC, BodySystem.CARDIOVASCULAR, 0.5, 0.0, 0.1),
            ],
            homeostatic_data=HomeostaticData(),
            emotional_body_state=0.3,
            gut_feeling=0.2,
            subjective_energy=0.7,
            body_awareness_focus=0.5,
        )
        self.assertEqual(len(inp.organ_signals), 1)
        self.assertIsNotNone(inp.homeostatic_data)
        self.assertEqual(inp.gut_feeling, 0.2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = InteroceptiveInput(subjective_energy=0.6)
        d = inp.to_dict()
        self.assertAlmostEqual(d["subjective_energy"], 0.6, places=4)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestBodyStateAssessment(unittest.TestCase):
    """Tests for BodyStateAssessment dataclass."""

    def test_creation(self):
        """Should create body state assessment."""
        bsa = BodyStateAssessment(
            state_category=BodyStateCategory.OPTIMAL,
            overall_wellbeing=0.5,
            sympathetic_activation=0.3,
            parasympathetic_activation=0.6,
            autonomic_balance=0.3,
            body_coherence=0.8,
            vitality=0.7,
        )
        self.assertEqual(bsa.state_category, BodyStateCategory.OPTIMAL)
        self.assertEqual(bsa.overall_wellbeing, 0.5)

    def test_to_dict(self):
        """Should convert to dictionary."""
        bsa = BodyStateAssessment(
            state_category=BodyStateCategory.RELAXED,
            overall_wellbeing=0.6,
            sympathetic_activation=0.2,
            parasympathetic_activation=0.7,
            autonomic_balance=0.5,
            body_coherence=0.9,
            vitality=0.8,
        )
        d = bsa.to_dict()
        self.assertEqual(d["state_category"], "relaxed")


class TestHomeostaticNeedsReport(unittest.TestCase):
    """Tests for HomeostaticNeedsReport dataclass."""

    def test_creation(self):
        """Should create homeostatic needs report."""
        hnr = HomeostaticNeedsReport(
            active_needs=[HomeostaticNeed.HUNGER, HomeostaticNeed.THIRST],
            primary_need=HomeostaticNeed.HUNGER,
            primary_urgency=0.7,
            need_intensities={"hunger": 0.7, "thirst": 0.5},
            homeostatic_deviation=0.6,
        )
        self.assertEqual(len(hnr.active_needs), 2)
        self.assertEqual(hnr.primary_need, HomeostaticNeed.HUNGER)

    def test_to_dict(self):
        """Should convert to dictionary."""
        hnr = HomeostaticNeedsReport(
            active_needs=[HomeostaticNeed.NONE],
            primary_need=HomeostaticNeed.NONE,
            primary_urgency=0.0,
            need_intensities={},
            homeostatic_deviation=0.0,
        )
        d = hnr.to_dict()
        self.assertEqual(d["primary_need"], "none")


class TestEmotionalGrounding(unittest.TestCase):
    """Tests for EmotionalGrounding dataclass."""

    def test_creation(self):
        """Should create emotional grounding."""
        eg = EmotionalGrounding(
            body_valence=0.3,
            body_arousal=0.5,
            gut_intuition=0.2,
            felt_sense_clarity=0.7,
            emotional_readiness=0.8,
            somatic_markers=["confidence", "vigor"],
        )
        self.assertEqual(eg.body_valence, 0.3)
        self.assertEqual(len(eg.somatic_markers), 2)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestInteroceptiveConsciousnessInterface(unittest.TestCase):
    """Tests for InteroceptiveConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = InteroceptiveConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "06-interoceptive")
        self.assertEqual(self.interface.FORM_NAME, "Interoceptive Consciousness")

    def test_initialize(self):
        """Should initialize the pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_normal_state(self):
        """Should process normal body state."""
        inp = create_simple_interoceptive_input(energy=0.6, stress=0.2, gut_feeling=0.3)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIsInstance(output, InteroceptiveOutput)
        self.assertIn(output.body_state.state_category, [
            BodyStateCategory.OPTIMAL, BodyStateCategory.RELAXED,
            BodyStateCategory.ENERGIZED, BodyStateCategory.RECOVERING,
        ])

    def test_detect_hunger(self):
        """Should detect hunger when blood glucose is low."""
        inp = create_simple_interoceptive_input(blood_glucose=0.2)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIn(HomeostaticNeed.HUNGER, output.homeostatic_needs.active_needs)

    def test_detect_thirst(self):
        """Should detect thirst when hydration is low."""
        inp = create_simple_interoceptive_input(hydration=0.2)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIn(HomeostaticNeed.THIRST, output.homeostatic_needs.active_needs)

    def test_detect_cold(self):
        """Should detect need for warmth when cold."""
        inp = create_simple_interoceptive_input(body_temperature=0.1)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIn(HomeostaticNeed.WARMTH, output.homeostatic_needs.active_needs)

    def test_detect_rest_need(self):
        """Should detect need for rest when energy is depleted."""
        inp = create_simple_interoceptive_input(energy=0.1)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIn(HomeostaticNeed.REST, output.homeostatic_needs.active_needs)

    def test_stressed_state(self):
        """Should detect stressed state with high stress hormones."""
        inp = create_simple_interoceptive_input(stress=0.8, energy=0.5)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertGreater(output.body_state.sympathetic_activation, 0.4)

    def test_ill_state(self):
        """Should detect ill state with high immune activation."""
        inp = InteroceptiveInput(
            homeostatic_data=HomeostaticData(
                immune_activation=0.7,
                energy_reserves=0.3,
                stress_hormones=0.5,
            ),
            subjective_energy=0.2,
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertEqual(output.body_state.state_category, BodyStateCategory.ILL)

    def test_emotional_grounding(self):
        """Should produce emotional grounding output."""
        inp = create_simple_interoceptive_input(gut_feeling=0.5, energy=0.7)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertIsNotNone(output.emotional_grounding)
        self.assertGreater(output.emotional_grounding.gut_intuition, 0.0)

    def test_somatic_markers_generated(self):
        """Should generate somatic markers for body states."""
        inp = create_simple_interoceptive_input(energy=0.8, stress=0.1, gut_feeling=0.5)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        # Should have at least some somatic markers
        self.assertIsInstance(output.emotional_grounding.somatic_markers, list)

    def test_interoceptive_sensitivity(self):
        """Should allow setting interoceptive sensitivity."""
        self.interface.set_interoceptive_sensitivity(0.8)
        self.assertEqual(self.interface.get_interoceptive_sensitivity(), 0.8)

    def test_action_required_for_urgent_needs(self):
        """Should require action when needs are urgent."""
        inp = create_simple_interoceptive_input(
            energy=0.1, blood_glucose=0.1, hydration=0.2
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_interoceptive_input(inp)
            )
        finally:
            loop.close()
        self.assertTrue(output.requires_action)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "06-interoceptive")
        self.assertIn("interoceptive_sensitivity", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "06-interoceptive")
        self.assertTrue(status["operational"])


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_interoceptive_interface(self):
        """Should create new interface."""
        interface = create_interoceptive_interface()
        self.assertIsInstance(interface, InteroceptiveConsciousnessInterface)

    def test_create_simple_interoceptive_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_interoceptive_input()
        self.assertIsInstance(inp, InteroceptiveInput)
        self.assertIsNotNone(inp.homeostatic_data)
        self.assertEqual(inp.subjective_energy, 0.5)

    def test_create_simple_interoceptive_input_custom(self):
        """Should create input with custom values."""
        inp = create_simple_interoceptive_input(
            energy=0.8,
            stress=0.2,
            gut_feeling=0.5,
            body_temperature=0.6,
        )
        self.assertEqual(inp.homeostatic_data.energy_reserves, 0.8)
        self.assertEqual(inp.homeostatic_data.stress_hormones, 0.2)
        self.assertEqual(inp.gut_feeling, 0.5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete interoceptive pipeline."""

    def test_full_pipeline(self):
        """Should complete full interoceptive processing pipeline."""
        interface = create_interoceptive_interface()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            inp = InteroceptiveInput(
                organ_signals=[
                    OrganSignal(InteroceptiveChannel.CARDIAC, BodySystem.CARDIOVASCULAR, 0.5, 0.0, 0.1),
                    OrganSignal(InteroceptiveChannel.RESPIRATORY, BodySystem.RESPIRATORY_SYSTEM, 0.4, -0.1, 0.0),
                    OrganSignal(InteroceptiveChannel.GASTROINTESTINAL, BodySystem.DIGESTIVE, 0.6, 0.2, 0.3),
                ],
                homeostatic_data=HomeostaticData(
                    body_temperature=0.5,
                    blood_glucose=0.4,
                    hydration_level=0.6,
                    energy_reserves=0.5,
                    stress_hormones=0.3,
                ),
                emotional_body_state=0.2,
                gut_feeling=0.1,
                subjective_energy=0.6,
                body_awareness_focus=0.5,
            )
            output = loop.run_until_complete(interface.process_interoceptive_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, InteroceptiveOutput)
        self.assertIsNotNone(output.body_state)
        self.assertIsNotNone(output.homeostatic_needs)
        self.assertIsNotNone(output.emotional_grounding)

    def test_history_builds(self):
        """Should build state history over time."""
        interface = create_interoceptive_interface()
        inp = create_simple_interoceptive_input()
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(interface.process_interoceptive_input(inp))
        finally:
            loop.close()
        self.assertEqual(len(interface._body_state_history), 5)
        self.assertEqual(len(interface._need_history), 5)

    def test_depleted_to_energized_transition(self):
        """Should track state transitions from depleted to energized."""
        interface = create_interoceptive_interface()
        loop = asyncio.new_event_loop()
        try:
            # Depleted state
            depleted = create_simple_interoceptive_input(energy=0.1, stress=0.2)
            out1 = loop.run_until_complete(interface.process_interoceptive_input(depleted))

            # Energized state
            energized = create_simple_interoceptive_input(energy=0.9, stress=0.1, gut_feeling=0.5)
            out2 = loop.run_until_complete(interface.process_interoceptive_input(energized))
        finally:
            loop.close()

        # Second state should have better wellbeing
        self.assertGreater(out2.body_state.vitality, out1.body_state.vitality)


if __name__ == "__main__":
    unittest.main(verbosity=2)
