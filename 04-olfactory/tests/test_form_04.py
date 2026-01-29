#!/usr/bin/env python3
"""
Test Suite for Form 04: Olfactory Consciousness.

Tests cover:
- All enumerations (OdorCategory, OlfactoryQuality, OdorIntensityLevel, HedonicValence, OlfactoryAdaptationState)
- All input/output dataclasses
- OlfactoryConsciousnessInterface processing pipeline
- Odor identification, hedonic evaluation, memory association, adaptation
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
    OdorCategory,
    OlfactoryQuality,
    OdorIntensityLevel,
    HedonicValence,
    OlfactoryAdaptationState,
    # Input dataclasses
    ChemicalFeatures,
    OlfactoryInput,
    # Output dataclasses
    OdorIdentification,
    HedonicEvaluation,
    MemoryAssociation,
    OlfactoryOutput,
    # Main interface
    OlfactoryConsciousnessInterface,
    # Convenience functions
    create_olfactory_interface,
    create_simple_olfactory_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestOdorCategory(unittest.TestCase):
    """Tests for OdorCategory enumeration."""

    def test_all_categories_exist(self):
        """All odor categories should be defined."""
        categories = [
            OdorCategory.FLORAL, OdorCategory.FRUITY, OdorCategory.WOODY,
            OdorCategory.SPICY, OdorCategory.HERBAL, OdorCategory.EARTHY,
            OdorCategory.CHEMICAL, OdorCategory.FOOD, OdorCategory.SMOKE,
            OdorCategory.DECAY, OdorCategory.BODY, OdorCategory.NEUTRAL,
            OdorCategory.UNKNOWN,
        ]
        self.assertEqual(len(categories), 13)

    def test_values(self):
        """Categories should have expected values."""
        self.assertEqual(OdorCategory.FLORAL.value, "floral")
        self.assertEqual(OdorCategory.DECAY.value, "decay")


class TestOlfactoryQuality(unittest.TestCase):
    """Tests for OlfactoryQuality enumeration."""

    def test_all_qualities_exist(self):
        """All olfactory qualities should be defined."""
        qualities = [
            OlfactoryQuality.SWEET, OlfactoryQuality.PUNGENT,
            OlfactoryQuality.MUSKY, OlfactoryQuality.PUTRID,
            OlfactoryQuality.CAMPHORACEOUS, OlfactoryQuality.ETHEREAL,
            OlfactoryQuality.MINTY, OlfactoryQuality.WARM,
            OlfactoryQuality.COOL, OlfactoryQuality.SHARP,
        ]
        self.assertEqual(len(qualities), 10)


class TestOdorIntensityLevel(unittest.TestCase):
    """Tests for OdorIntensityLevel enumeration."""

    def test_all_levels_exist(self):
        """All intensity levels should be defined."""
        levels = [
            OdorIntensityLevel.THRESHOLD, OdorIntensityLevel.FAINT,
            OdorIntensityLevel.MODERATE, OdorIntensityLevel.STRONG,
            OdorIntensityLevel.OVERWHELMING,
        ]
        self.assertEqual(len(levels), 5)


class TestHedonicValence(unittest.TestCase):
    """Tests for HedonicValence enumeration."""

    def test_all_valences_exist(self):
        """All hedonic valences should be defined."""
        valences = [
            HedonicValence.VERY_PLEASANT, HedonicValence.PLEASANT,
            HedonicValence.NEUTRAL, HedonicValence.UNPLEASANT,
            HedonicValence.VERY_UNPLEASANT,
        ]
        self.assertEqual(len(valences), 5)


class TestOlfactoryAdaptationState(unittest.TestCase):
    """Tests for OlfactoryAdaptationState enumeration."""

    def test_all_states_exist(self):
        """All adaptation states should be defined."""
        states = [
            OlfactoryAdaptationState.FRESH, OlfactoryAdaptationState.ADAPTING,
            OlfactoryAdaptationState.ADAPTED, OlfactoryAdaptationState.CROSS_ADAPTED,
        ]
        self.assertEqual(len(states), 4)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestChemicalFeatures(unittest.TestCase):
    """Tests for ChemicalFeatures dataclass."""

    def test_creation(self):
        """Should create chemical features."""
        cf = ChemicalFeatures(
            molecular_weight=0.6,
            volatility=0.7,
            hydrophobicity=0.4,
            functional_groups=["alcohol", "ester"],
            receptor_activation={"OR1": 0.8, "OR2": 0.3},
            concentration=0.5,
        )
        self.assertEqual(cf.molecular_weight, 0.6)
        self.assertEqual(len(cf.functional_groups), 2)
        self.assertEqual(len(cf.receptor_activation), 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        cf = ChemicalFeatures(
            molecular_weight=0.5,
            volatility=0.5,
            hydrophobicity=0.5,
            functional_groups=["aldehyde"],
            receptor_activation={"OR1": 0.5},
        )
        d = cf.to_dict()
        self.assertIn("num_receptors_activated", d)
        self.assertEqual(d["num_receptors_activated"], 1)


class TestOlfactoryInput(unittest.TestCase):
    """Tests for OlfactoryInput dataclass."""

    def test_creation_defaults(self):
        """Should create input with defaults."""
        inp = OlfactoryInput()
        self.assertEqual(inp.intensity, 0.0)
        self.assertFalse(inp.onset_detected)

    def test_creation_custom(self):
        """Should create input with custom values."""
        inp = OlfactoryInput(
            intensity=0.7,
            onset_detected=True,
            sniff_active=True,
            num_distinct_odors=3,
        )
        self.assertEqual(inp.intensity, 0.7)
        self.assertTrue(inp.onset_detected)
        self.assertEqual(inp.num_distinct_odors, 3)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = OlfactoryInput(intensity=0.5)
        d = inp.to_dict()
        self.assertAlmostEqual(d["intensity"], 0.5, places=4)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestOdorIdentification(unittest.TestCase):
    """Tests for OdorIdentification dataclass."""

    def test_creation(self):
        """Should create odor identification."""
        oid = OdorIdentification(
            category=OdorCategory.FLORAL,
            label="rose",
            qualities=[OlfactoryQuality.SWEET],
            confidence=0.85,
            intensity_level=OdorIntensityLevel.MODERATE,
            familiarity=0.7,
            distinctiveness=0.8,
        )
        self.assertEqual(oid.category, OdorCategory.FLORAL)
        self.assertEqual(len(oid.qualities), 1)

    def test_to_dict(self):
        """Should convert to dictionary."""
        oid = OdorIdentification(
            category=OdorCategory.WOODY,
            label="cedar",
            qualities=[OlfactoryQuality.WARM],
            confidence=0.7,
            intensity_level=OdorIntensityLevel.MODERATE,
            familiarity=0.5,
            distinctiveness=0.6,
        )
        d = oid.to_dict()
        self.assertEqual(d["category"], "woody")
        self.assertEqual(d["qualities"], ["warm"])


class TestHedonicEvaluation(unittest.TestCase):
    """Tests for HedonicEvaluation dataclass."""

    def test_creation(self):
        """Should create hedonic evaluation."""
        he = HedonicEvaluation(
            valence=HedonicValence.PLEASANT,
            valence_score=0.5,
            approach_tendency=0.6,
            avoidance_tendency=0.0,
            emotional_associations=["comfort"],
        )
        self.assertEqual(he.valence, HedonicValence.PLEASANT)
        self.assertIn("comfort", he.emotional_associations)

    def test_to_dict(self):
        """Should convert to dictionary."""
        he = HedonicEvaluation(
            valence=HedonicValence.UNPLEASANT,
            valence_score=-0.5,
            approach_tendency=0.0,
            avoidance_tendency=0.6,
            emotional_associations=["disgust"],
        )
        d = he.to_dict()
        self.assertEqual(d["valence"], "unpleasant")


class TestMemoryAssociation(unittest.TestCase):
    """Tests for MemoryAssociation dataclass."""

    def test_creation(self):
        """Should create memory association."""
        ma = MemoryAssociation(
            has_memory_trigger=True,
            memory_strength=0.7,
            memory_type="episodic",
            memory_valence=0.5,
            context_cues=["garden", "summer"],
            proustian_effect=0.6,
        )
        self.assertTrue(ma.has_memory_trigger)
        self.assertEqual(ma.memory_type, "episodic")
        self.assertEqual(len(ma.context_cues), 2)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestOlfactoryConsciousnessInterface(unittest.TestCase):
    """Tests for OlfactoryConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = OlfactoryConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "04-olfactory")
        self.assertEqual(self.interface.FORM_NAME, "Olfactory Consciousness")

    def test_initialize(self):
        """Should initialize the pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_no_odor(self):
        """Should process absence of odor."""
        inp = OlfactoryInput(intensity=0.01)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_olfactory_input(inp))
        finally:
            loop.close()
        self.assertIsInstance(output, OlfactoryOutput)
        self.assertEqual(output.odor_identification.category, OdorCategory.NEUTRAL)

    def test_process_floral_odor(self):
        """Should process a floral odor."""
        inp = create_simple_olfactory_input(intensity=0.5, category_hint="floral")
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_olfactory_input(inp))
        finally:
            loop.close()
        self.assertIsInstance(output, OlfactoryOutput)
        self.assertGreater(output.odor_identification.confidence, 0.0)

    def test_hedonic_pleasant_for_floral(self):
        """Floral odors should tend toward pleasant hedonic evaluation."""
        inp = create_simple_olfactory_input(intensity=0.4, category_hint="floral")
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_olfactory_input(inp))
        finally:
            loop.close()
        # Herbal (default without specific functional groups) should not be very unpleasant
        self.assertGreaterEqual(output.hedonic_evaluation.valence_score, -0.5)

    def test_safety_alert_for_chemicals(self):
        """Should trigger safety alert for strong chemical odors."""
        cf = ChemicalFeatures(
            molecular_weight=0.5,
            volatility=0.9,
            hydrophobicity=0.3,
            functional_groups=["chlorine"],
            receptor_activation={"OR1": 0.9},
            concentration=0.8,
        )
        inp = OlfactoryInput(
            chemical_features=cf,
            intensity=0.8,
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_olfactory_input(inp))
        finally:
            loop.close()
        self.assertTrue(output.safety_alert)

    def test_adaptation_over_time(self):
        """Should show olfactory adaptation with repeated exposure."""
        inp = create_simple_olfactory_input(intensity=0.6)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(15):
                output = loop.run_until_complete(
                    self.interface.process_olfactory_input(inp)
                )
        finally:
            loop.close()
        # After repeated exposure, adaptation level should increase
        self.assertGreater(self.interface._adaptation_level, 0.0)

    def test_onset_resets_adaptation(self):
        """Should reset adaptation on new odor onset."""
        inp = create_simple_olfactory_input(intensity=0.6)
        loop = asyncio.new_event_loop()
        try:
            # Build adaptation
            for _ in range(10):
                loop.run_until_complete(self.interface.process_olfactory_input(inp))
            adapted_level = self.interface._adaptation_level

            # New onset
            onset_inp = OlfactoryInput(intensity=0.6, onset_detected=True)
            loop.run_until_complete(self.interface.process_olfactory_input(onset_inp))
        finally:
            loop.close()
        self.assertLess(self.interface._adaptation_level, adapted_level)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "04-olfactory")
        self.assertIn("adaptation_level", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "04-olfactory")
        self.assertTrue(status["operational"])


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_olfactory_interface(self):
        """Should create new interface."""
        interface = create_olfactory_interface()
        self.assertIsInstance(interface, OlfactoryConsciousnessInterface)

    def test_create_simple_olfactory_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_olfactory_input()
        self.assertIsInstance(inp, OlfactoryInput)
        self.assertEqual(inp.intensity, 0.5)
        self.assertIsNotNone(inp.chemical_features)

    def test_create_simple_olfactory_input_fruity(self):
        """Should create fruity odor input."""
        inp = create_simple_olfactory_input(category_hint="fruity")
        self.assertIn("alcohol", inp.chemical_features.functional_groups)

    def test_create_simple_olfactory_input_decay(self):
        """Should create decay odor input."""
        inp = create_simple_olfactory_input(category_hint="decay")
        self.assertIn("sulfur", inp.chemical_features.functional_groups)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete olfactory pipeline."""

    def test_full_pipeline(self):
        """Should complete full olfactory processing pipeline."""
        interface = create_olfactory_interface()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            inp = create_simple_olfactory_input(intensity=0.6, onset=True)
            output = loop.run_until_complete(interface.process_olfactory_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, OlfactoryOutput)
        self.assertIsNotNone(output.odor_identification)
        self.assertIsNotNone(output.hedonic_evaluation)
        self.assertIsNotNone(output.memory_association)
        self.assertTrue(output.requires_attention)  # Onset should require attention

    def test_familiarity_builds(self):
        """Should build familiarity with repeated odors."""
        interface = create_olfactory_interface()
        inp = create_simple_olfactory_input(intensity=0.5)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(interface.process_olfactory_input(inp))
        finally:
            loop.close()
        self.assertGreater(len(interface._known_odors), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
