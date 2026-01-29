#!/usr/bin/env python3
"""
Test Suite for Form 02: Auditory Consciousness.

Tests cover:
- All enumerations (SoundCategory, FrequencyBand, AuditoryScene, SpeechContent, SpatialDirection)
- All input/output dataclasses
- AuditoryConsciousnessInterface processing pipeline
- Sound identification, spatial localization, speech processing, scene analysis
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
    SoundCategory,
    FrequencyBand,
    AuditoryScene,
    SpeechContent,
    SpatialDirection,
    # Input dataclasses
    SpectralData,
    AuditoryInput,
    # Output dataclasses
    SoundIdentification,
    SpatialLocation,
    SpeechAnalysis,
    AuditorySceneAnalysis,
    AuditoryOutput,
    # Main interface
    AuditoryConsciousnessInterface,
    # Convenience functions
    create_auditory_interface,
    create_simple_auditory_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestSoundCategory(unittest.TestCase):
    """Tests for SoundCategory enumeration."""

    def test_all_categories_exist(self):
        """All sound categories should be defined."""
        categories = [
            SoundCategory.SPEECH, SoundCategory.MUSIC,
            SoundCategory.ENVIRONMENTAL, SoundCategory.ALARM,
            SoundCategory.ANIMAL, SoundCategory.MECHANICAL,
            SoundCategory.NATURE, SoundCategory.SILENCE,
            SoundCategory.NOISE, SoundCategory.UNKNOWN,
        ]
        self.assertEqual(len(categories), 10)

    def test_category_values(self):
        """Categories should have expected string values."""
        self.assertEqual(SoundCategory.SPEECH.value, "speech")
        self.assertEqual(SoundCategory.ALARM.value, "alarm")


class TestFrequencyBand(unittest.TestCase):
    """Tests for FrequencyBand enumeration."""

    def test_all_bands_exist(self):
        """All frequency bands should be defined."""
        bands = [
            FrequencyBand.SUB_BASS, FrequencyBand.BASS,
            FrequencyBand.LOW_MID, FrequencyBand.MID,
            FrequencyBand.UPPER_MID, FrequencyBand.PRESENCE,
            FrequencyBand.BRILLIANCE,
        ]
        self.assertEqual(len(bands), 7)


class TestAuditoryScene(unittest.TestCase):
    """Tests for AuditoryScene enumeration."""

    def test_all_scenes_exist(self):
        """All auditory scenes should be defined."""
        scenes = [
            AuditoryScene.QUIET, AuditoryScene.CONVERSATION,
            AuditoryScene.CROWDED, AuditoryScene.NATURE_AMBIENT,
            AuditoryScene.URBAN, AuditoryScene.MUSIC_FOCUSED,
            AuditoryScene.ALARM_STATE, AuditoryScene.MIXED,
        ]
        self.assertEqual(len(scenes), 8)


class TestSpeechContent(unittest.TestCase):
    """Tests for SpeechContent enumeration."""

    def test_all_types_exist(self):
        """All speech content types should be defined."""
        types = [
            SpeechContent.DECLARATIVE, SpeechContent.QUESTION,
            SpeechContent.COMMAND, SpeechContent.EMOTIONAL,
            SpeechContent.WHISPER, SpeechContent.SHOUT,
            SpeechContent.SINGING, SpeechContent.NONE,
        ]
        self.assertEqual(len(types), 8)


class TestSpatialDirection(unittest.TestCase):
    """Tests for SpatialDirection enumeration."""

    def test_all_directions_exist(self):
        """All spatial directions should be defined."""
        directions = [
            SpatialDirection.FRONT, SpatialDirection.BEHIND,
            SpatialDirection.LEFT, SpatialDirection.RIGHT,
            SpatialDirection.ABOVE, SpatialDirection.BELOW,
            SpatialDirection.OMNIDIRECTIONAL, SpatialDirection.INTERNAL,
        ]
        self.assertEqual(len(directions), 8)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestSpectralData(unittest.TestCase):
    """Tests for SpectralData dataclass."""

    def test_creation(self):
        """Should create spectral data."""
        sd = SpectralData(
            band_energies={"bass": 0.7, "mid": 0.5},
            dominant_frequency=440.0,
            spectral_centroid=1200.0,
            spectral_flux=0.3,
            zero_crossing_rate=0.2,
        )
        self.assertEqual(sd.dominant_frequency, 440.0)
        self.assertEqual(sd.spectral_flux, 0.3)

    def test_to_dict(self):
        """Should convert to dictionary."""
        sd = SpectralData(
            band_energies={"bass": 0.5},
            dominant_frequency=300.0,
            spectral_centroid=800.0,
            spectral_flux=0.1,
            zero_crossing_rate=0.15,
        )
        d = sd.to_dict()
        self.assertIn("band_energies", d)
        self.assertEqual(d["dominant_frequency"], 300.0)


class TestAuditoryInput(unittest.TestCase):
    """Tests for AuditoryInput dataclass."""

    def test_creation_defaults(self):
        """Should create input with defaults."""
        inp = AuditoryInput()
        self.assertEqual(inp.amplitude, 0.0)
        self.assertFalse(inp.onset_detected)
        self.assertEqual(inp.num_sources, 1)

    def test_creation_custom(self):
        """Should create input with custom values."""
        inp = AuditoryInput(
            amplitude=0.7,
            pitch=0.5,
            rhythm_regularity=0.8,
            spatial_angle=45.0,
            num_sources=3,
        )
        self.assertEqual(inp.amplitude, 0.7)
        self.assertEqual(inp.spatial_angle, 45.0)
        self.assertEqual(inp.num_sources, 3)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = AuditoryInput(amplitude=0.6)
        d = inp.to_dict()
        self.assertAlmostEqual(d["amplitude"], 0.6, places=4)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestSoundIdentification(unittest.TestCase):
    """Tests for SoundIdentification dataclass."""

    def test_creation(self):
        """Should create sound identification."""
        sid = SoundIdentification(
            category=SoundCategory.SPEECH,
            label="voice",
            confidence=0.85,
            loudness=0.6,
            pitch_class="mid",
        )
        self.assertEqual(sid.category, SoundCategory.SPEECH)
        self.assertEqual(sid.pitch_class, "mid")

    def test_to_dict(self):
        """Should convert to dictionary."""
        sid = SoundIdentification(
            category=SoundCategory.MUSIC,
            label="melody",
            confidence=0.9,
            loudness=0.5,
            pitch_class="high",
        )
        d = sid.to_dict()
        self.assertEqual(d["category"], "music")


class TestSpatialLocation(unittest.TestCase):
    """Tests for SpatialLocation dataclass."""

    def test_creation(self):
        """Should create spatial location."""
        loc = SpatialLocation(
            direction=SpatialDirection.LEFT,
            azimuth=-60.0,
            elevation=10.0,
            distance=0.3,
            confidence=0.8,
        )
        self.assertEqual(loc.direction, SpatialDirection.LEFT)
        self.assertEqual(loc.azimuth, -60.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        loc = SpatialLocation(
            direction=SpatialDirection.FRONT,
            azimuth=0.0,
            elevation=0.0,
            distance=0.5,
            confidence=0.7,
        )
        d = loc.to_dict()
        self.assertEqual(d["direction"], "front")


class TestSpeechAnalysis(unittest.TestCase):
    """Tests for SpeechAnalysis dataclass."""

    def test_creation(self):
        """Should create speech analysis."""
        sa = SpeechAnalysis(
            content_type=SpeechContent.DECLARATIVE,
            transcript_fragment="hello world",
            speaker_id="speaker_1",
            speaker_confidence=0.8,
            emotional_tone=0.2,
            speech_rate=0.5,
            clarity=0.9,
        )
        self.assertEqual(sa.content_type, SpeechContent.DECLARATIVE)
        self.assertEqual(sa.transcript_fragment, "hello world")


class TestAuditorySceneAnalysis(unittest.TestCase):
    """Tests for AuditorySceneAnalysis dataclass."""

    def test_creation(self):
        """Should create scene analysis."""
        asa = AuditorySceneAnalysis(
            scene_type=AuditoryScene.CONVERSATION,
            num_streams=2,
            background_level=0.2,
            foreground_salience=0.8,
            scene_complexity=0.5,
            scene_familiarity=0.7,
        )
        self.assertEqual(asa.scene_type, AuditoryScene.CONVERSATION)
        self.assertEqual(asa.num_streams, 2)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestAuditoryConsciousnessInterface(unittest.TestCase):
    """Tests for AuditoryConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = AuditoryConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "02-auditory")
        self.assertEqual(self.interface.FORM_NAME, "Auditory Consciousness")

    def test_initialize(self):
        """Should initialize the pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_silence(self):
        """Should process near-silent input."""
        inp = AuditoryInput(amplitude=0.02)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertIsInstance(output, AuditoryOutput)
        self.assertTrue(any(s.category == SoundCategory.SILENCE for s in output.sounds_identified))

    def test_process_speech_input(self):
        """Should identify speech in auditory input."""
        inp = AuditoryInput(amplitude=0.6, pitch=0.5, num_sources=1)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertIsInstance(output, AuditoryOutput)
        self.assertGreater(len(output.sounds_identified), 0)

    def test_process_music_input(self):
        """Should identify music with high rhythm regularity."""
        inp = AuditoryInput(amplitude=0.5, rhythm_regularity=0.9, pitch=0.6)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertTrue(any(s.category == SoundCategory.MUSIC for s in output.sounds_identified))

    def test_process_alarm(self):
        """Should detect alarm sounds and require attention."""
        inp = AuditoryInput(amplitude=0.95, onset_detected=True)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertTrue(output.requires_attention)

    def test_spatial_localization(self):
        """Should localize sound spatially."""
        inp = AuditoryInput(amplitude=0.5, pitch=0.5, spatial_angle=-60.0)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertGreater(len(output.spatial_locations), 0)
        self.assertEqual(output.spatial_locations[0].direction, SpatialDirection.LEFT)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "02-auditory")
        self.assertIn("current_scene", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "02-auditory")
        self.assertTrue(status["operational"])


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_auditory_interface(self):
        """Should create new interface."""
        interface = create_auditory_interface()
        self.assertIsInstance(interface, AuditoryConsciousnessInterface)

    def test_create_simple_auditory_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_auditory_input()
        self.assertIsInstance(inp, AuditoryInput)
        self.assertEqual(inp.amplitude, 0.5)
        self.assertEqual(inp.num_sources, 1)

    def test_create_simple_auditory_input_custom(self):
        """Should create input with custom values."""
        inp = create_simple_auditory_input(
            amplitude=0.8,
            pitch=0.6,
            rhythm=0.9,
            spatial_angle=45.0,
            num_sources=3,
        )
        self.assertEqual(inp.amplitude, 0.8)
        self.assertEqual(inp.rhythm_regularity, 0.9)
        self.assertEqual(inp.num_sources, 3)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete auditory pipeline."""

    def test_full_pipeline(self):
        """Should complete full auditory processing pipeline."""
        interface = create_auditory_interface()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            inp = AuditoryInput(
                amplitude=0.6,
                pitch=0.5,
                rhythm_regularity=0.3,
                spatial_angle=20.0,
                num_sources=2,
                onset_detected=True,
            )
            output = loop.run_until_complete(interface.process_auditory_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, AuditoryOutput)
        self.assertGreater(len(output.sounds_identified), 0)
        self.assertIsNotNone(output.scene_analysis)

    def test_history_builds(self):
        """Should build sound history over time."""
        interface = create_auditory_interface()
        inp = create_simple_auditory_input(amplitude=0.5)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(interface.process_auditory_input(inp))
        finally:
            loop.close()
        self.assertGreater(len(interface._sound_history), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
