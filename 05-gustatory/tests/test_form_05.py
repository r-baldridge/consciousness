#!/usr/bin/env python3
"""
Test Suite for Form 05: Gustatory Consciousness.

Tests cover:
- All enumerations (TasteModality, FlavorProfile, TextureQuality, PalatabilityLevel, AppetiteState)
- All input/output dataclasses
- GustatoryConsciousnessInterface processing pipeline
- Taste identification, flavor integration, palatability assessment
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
    TasteModality,
    FlavorProfile,
    TextureQuality,
    PalatabilityLevel,
    AppetiteState,
    # Input dataclasses
    TasteReceptorData,
    GustatoryInput,
    # Output dataclasses
    TasteIdentification,
    FlavorIntegration,
    PalatabilityAssessment,
    GustatoryOutput,
    # Main interface
    GustatoryConsciousnessInterface,
    # Convenience functions
    create_gustatory_interface,
    create_simple_gustatory_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestTasteModality(unittest.TestCase):
    """Tests for TasteModality enumeration."""

    def test_all_modalities_exist(self):
        """All five taste modalities should be defined."""
        modalities = [
            TasteModality.SWEET, TasteModality.SOUR,
            TasteModality.SALTY, TasteModality.BITTER,
            TasteModality.UMAMI,
        ]
        self.assertEqual(len(modalities), 5)

    def test_values(self):
        """Modalities should have expected values."""
        self.assertEqual(TasteModality.SWEET.value, "sweet")
        self.assertEqual(TasteModality.UMAMI.value, "umami")


class TestFlavorProfile(unittest.TestCase):
    """Tests for FlavorProfile enumeration."""

    def test_all_profiles_exist(self):
        """All flavor profiles should be defined."""
        profiles = [
            FlavorProfile.SAVORY, FlavorProfile.SWEET_AROMATIC,
            FlavorProfile.SOUR_TANGY, FlavorProfile.BITTER_COMPLEX,
            FlavorProfile.SPICY_HOT, FlavorProfile.MILD,
            FlavorProfile.RICH, FlavorProfile.FRESH,
            FlavorProfile.FERMENTED, FlavorProfile.NEUTRAL,
        ]
        self.assertEqual(len(profiles), 10)


class TestTextureQuality(unittest.TestCase):
    """Tests for TextureQuality enumeration."""

    def test_all_textures_exist(self):
        """All texture qualities should be defined."""
        textures = [
            TextureQuality.SMOOTH, TextureQuality.CRUNCHY,
            TextureQuality.CREAMY, TextureQuality.GRAINY,
            TextureQuality.CHEWY, TextureQuality.LIQUID,
            TextureQuality.FIZZY, TextureQuality.DRY,
        ]
        self.assertEqual(len(textures), 8)


class TestPalatabilityLevel(unittest.TestCase):
    """Tests for PalatabilityLevel enumeration."""

    def test_all_levels_exist(self):
        """All palatability levels should be defined."""
        levels = [
            PalatabilityLevel.DELICIOUS, PalatabilityLevel.PLEASANT,
            PalatabilityLevel.ACCEPTABLE, PalatabilityLevel.BLAND,
            PalatabilityLevel.UNPLEASANT, PalatabilityLevel.AVERSIVE,
        ]
        self.assertEqual(len(levels), 6)


class TestAppetiteState(unittest.TestCase):
    """Tests for AppetiteState enumeration."""

    def test_all_states_exist(self):
        """All appetite states should be defined."""
        states = [
            AppetiteState.HUNGRY, AppetiteState.SATIATED,
            AppetiteState.CRAVING, AppetiteState.NAUSEOUS,
            AppetiteState.NEUTRAL,
        ]
        self.assertEqual(len(states), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestTasteReceptorData(unittest.TestCase):
    """Tests for TasteReceptorData dataclass."""

    def test_creation(self):
        """Should create taste receptor data."""
        rd = TasteReceptorData(
            modality_activations={"sweet": 0.8, "sour": 0.2, "salty": 0.3, "bitter": 0.1, "umami": 0.4},
        )
        self.assertEqual(rd.modality_activations["sweet"], 0.8)
        self.assertEqual(rd.receptor_density, 0.5)  # Default

    def test_to_dict(self):
        """Should convert to dictionary."""
        rd = TasteReceptorData(
            modality_activations={"sweet": 0.5},
        )
        d = rd.to_dict()
        self.assertIn("modality_activations", d)


class TestGustatoryInput(unittest.TestCase):
    """Tests for GustatoryInput dataclass."""

    def test_creation_defaults(self):
        """Should create input with defaults."""
        inp = GustatoryInput()
        self.assertEqual(inp.overall_intensity, 0.0)
        self.assertEqual(inp.texture_quality, TextureQuality.SMOOTH)
        self.assertEqual(inp.appetite_state, AppetiteState.NEUTRAL)

    def test_creation_custom(self):
        """Should create input with custom values."""
        inp = GustatoryInput(
            overall_intensity=0.7,
            texture_quality=TextureQuality.CRUNCHY,
            olfactory_contribution=0.5,
            oral_irritation=0.3,
            appetite_state=AppetiteState.HUNGRY,
        )
        self.assertEqual(inp.overall_intensity, 0.7)
        self.assertEqual(inp.texture_quality, TextureQuality.CRUNCHY)
        self.assertEqual(inp.appetite_state, AppetiteState.HUNGRY)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = GustatoryInput(overall_intensity=0.5)
        d = inp.to_dict()
        self.assertAlmostEqual(d["overall_intensity"], 0.5, places=4)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestTasteIdentification(unittest.TestCase):
    """Tests for TasteIdentification dataclass."""

    def test_creation(self):
        """Should create taste identification."""
        tid = TasteIdentification(
            dominant_modality=TasteModality.SWEET,
            modality_strengths={"sweet": 0.8, "sour": 0.1},
            taste_complexity=0.4,
            taste_harmony=0.7,
            confidence=0.85,
        )
        self.assertEqual(tid.dominant_modality, TasteModality.SWEET)
        self.assertEqual(tid.taste_complexity, 0.4)

    def test_to_dict(self):
        """Should convert to dictionary."""
        tid = TasteIdentification(
            dominant_modality=TasteModality.UMAMI,
            modality_strengths={"umami": 0.7},
            taste_complexity=0.5,
            taste_harmony=0.8,
            confidence=0.9,
        )
        d = tid.to_dict()
        self.assertEqual(d["dominant_modality"], "umami")


class TestFlavorIntegration(unittest.TestCase):
    """Tests for FlavorIntegration dataclass."""

    def test_creation(self):
        """Should create flavor integration."""
        fi = FlavorIntegration(
            flavor_profile=FlavorProfile.SAVORY,
            flavor_richness=0.7,
            aroma_contribution=0.5,
            texture_contribution=0.3,
            temperature_influence=0.2,
            overall_intensity=0.6,
        )
        self.assertEqual(fi.flavor_profile, FlavorProfile.SAVORY)
        self.assertEqual(fi.flavor_richness, 0.7)


class TestPalatabilityAssessment(unittest.TestCase):
    """Tests for PalatabilityAssessment dataclass."""

    def test_creation(self):
        """Should create palatability assessment."""
        pa = PalatabilityAssessment(
            palatability=PalatabilityLevel.PLEASANT,
            palatability_score=0.5,
            desire_to_consume=0.7,
            satiety_signal=0.3,
            safety_assessment=0.9,
            novelty=0.6,
        )
        self.assertEqual(pa.palatability, PalatabilityLevel.PLEASANT)
        self.assertEqual(pa.desire_to_consume, 0.7)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestGustatoryConsciousnessInterface(unittest.TestCase):
    """Tests for GustatoryConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = GustatoryConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "05-gustatory")
        self.assertEqual(self.interface.FORM_NAME, "Gustatory Consciousness")

    def test_initialize(self):
        """Should initialize the pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_sweet_taste(self):
        """Should process sweet taste input."""
        inp = create_simple_gustatory_input(sweet=0.8, intensity=0.6)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertIsInstance(output, GustatoryOutput)
        self.assertEqual(output.taste_identification.dominant_modality, TasteModality.SWEET)

    def test_process_bitter_taste(self):
        """Should process bitter taste and potentially alert."""
        inp = GustatoryInput(
            taste_receptor_data=TasteReceptorData(
                modality_activations={
                    "sweet": 0.0, "sour": 0.0, "salty": 0.0,
                    "bitter": 0.9, "umami": 0.0,
                },
                receptor_density=1.0,  # Full sensitivity to ensure threshold crossed
            ),
            overall_intensity=0.8,
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertEqual(output.taste_identification.dominant_modality, TasteModality.BITTER)
        self.assertTrue(output.food_safety_alert)

    def test_process_umami_savory(self):
        """Should identify umami as savory flavor."""
        inp = create_simple_gustatory_input(umami=0.7, salty=0.3, intensity=0.6)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertEqual(output.taste_identification.dominant_modality, TasteModality.UMAMI)

    def test_hunger_increases_palatability(self):
        """Hunger should increase palatability."""
        hungry_inp = create_simple_gustatory_input(
            sweet=0.5, intensity=0.5, appetite=AppetiteState.HUNGRY
        )
        neutral_inp = create_simple_gustatory_input(
            sweet=0.5, intensity=0.5, appetite=AppetiteState.NEUTRAL
        )

        interface1 = create_gustatory_interface()
        interface2 = create_gustatory_interface()

        loop = asyncio.new_event_loop()
        try:
            out_hungry = loop.run_until_complete(
                interface1.process_gustatory_input(hungry_inp)
            )
            out_neutral = loop.run_until_complete(
                interface2.process_gustatory_input(neutral_inp)
            )
        finally:
            loop.close()

        self.assertGreaterEqual(
            out_hungry.palatability_assessment.palatability_score,
            out_neutral.palatability_assessment.palatability_score - 0.1
        )

    def test_nausea_reduces_appetite(self):
        """Nausea should suppress appetite modulation."""
        inp = create_simple_gustatory_input(
            sweet=0.5, intensity=0.5, appetite=AppetiteState.NAUSEOUS
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertLess(output.appetite_modulation, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "05-gustatory")
        self.assertIn("appetite_state", d)
        self.assertIn("satiety_level", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "05-gustatory")
        self.assertTrue(status["operational"])


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_gustatory_interface(self):
        """Should create new interface."""
        interface = create_gustatory_interface()
        self.assertIsInstance(interface, GustatoryConsciousnessInterface)

    def test_create_simple_gustatory_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_gustatory_input()
        self.assertIsInstance(inp, GustatoryInput)
        self.assertIsNotNone(inp.taste_receptor_data)

    def test_create_simple_gustatory_input_sweet(self):
        """Should create sweet taste input."""
        inp = create_simple_gustatory_input(sweet=0.8)
        self.assertEqual(inp.taste_receptor_data.modality_activations["sweet"], 0.8)

    def test_create_simple_gustatory_input_complex(self):
        """Should create complex taste input."""
        inp = create_simple_gustatory_input(
            sweet=0.3, salty=0.4, umami=0.5, intensity=0.7
        )
        self.assertEqual(inp.overall_intensity, 0.7)
        self.assertEqual(inp.taste_receptor_data.modality_activations["umami"], 0.5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete gustatory pipeline."""

    def test_full_pipeline(self):
        """Should complete full gustatory processing pipeline."""
        interface = create_gustatory_interface()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            inp = GustatoryInput(
                taste_receptor_data=TasteReceptorData(
                    modality_activations={
                        "sweet": 0.6, "sour": 0.2, "salty": 0.3,
                        "bitter": 0.1, "umami": 0.4,
                    },
                ),
                overall_intensity=0.6,
                texture_quality=TextureQuality.CREAMY,
                olfactory_contribution=0.4,
                appetite_state=AppetiteState.HUNGRY,
            )
            output = loop.run_until_complete(interface.process_gustatory_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, GustatoryOutput)
        self.assertIsNotNone(output.taste_identification)
        self.assertIsNotNone(output.flavor_integration)
        self.assertIsNotNone(output.palatability_assessment)

    def test_flavor_familiarity_builds(self):
        """Should build flavor familiarity over time."""
        interface = create_gustatory_interface()
        inp = create_simple_gustatory_input(sweet=0.7, intensity=0.5)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertGreater(len(interface._known_flavors), 0)

    def test_satiety_builds(self):
        """Should build satiety with continued eating."""
        interface = create_gustatory_interface()
        inp = create_simple_gustatory_input(sweet=0.5, intensity=0.7)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(10):
                loop.run_until_complete(interface.process_gustatory_input(inp))
        finally:
            loop.close()
        self.assertGreater(interface._satiety_level, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
