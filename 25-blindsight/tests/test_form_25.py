#!/usr/bin/env python3
"""
Test Suite for Form 25: Blindsight Consciousness.

Tests cover:
- All enumerations (BlindsightType, VisualFieldRegion, ProcessingPathway,
  StimulusProperty, DetectionConfidence)
- All input/output dataclasses
- BlindsightInterface main interface
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
    BlindsightType,
    VisualFieldRegion,
    ProcessingPathway,
    StimulusProperty,
    DetectionConfidence,
    # Input dataclasses
    BlindsightInput,
    ForcedChoiceTrialInput,
    # Output dataclasses
    BlindsightOutput,
    PathwayAnalysis,
    ImplicitDetectionResult,
    BlindsightProfile,
    # Interface
    BlindsightInterface,
    # Convenience
    create_blindsight_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestBlindsightType(unittest.TestCase):
    """Tests for BlindsightType enumeration."""

    def test_all_types_exist(self):
        """All blindsight types should be defined."""
        types = [
            BlindsightType.TYPE_1_GUESSING,
            BlindsightType.TYPE_2_FEELING,
        ]
        self.assertEqual(len(types), 2)

    def test_type_values(self):
        """Types should have expected string values."""
        self.assertEqual(BlindsightType.TYPE_1_GUESSING.value, "type_1_guessing")
        self.assertEqual(BlindsightType.TYPE_2_FEELING.value, "type_2_feeling")


class TestVisualFieldRegion(unittest.TestCase):
    """Tests for VisualFieldRegion enumeration."""

    def test_all_regions_exist(self):
        """All visual field regions should be defined."""
        regions = [
            VisualFieldRegion.INTACT_FOVEAL,
            VisualFieldRegion.INTACT_PERIPHERAL,
            VisualFieldRegion.BLIND_FIELD,
            VisualFieldRegion.TRANSITION_ZONE,
        ]
        self.assertEqual(len(regions), 4)

    def test_region_values(self):
        """Regions should have expected string values."""
        self.assertEqual(VisualFieldRegion.BLIND_FIELD.value, "blind_field")
        self.assertEqual(VisualFieldRegion.TRANSITION_ZONE.value, "transition_zone")


class TestProcessingPathway(unittest.TestCase):
    """Tests for ProcessingPathway enumeration."""

    def test_all_pathways_exist(self):
        """All processing pathways should be defined."""
        pathways = [
            ProcessingPathway.VENTRAL_CONSCIOUS,
            ProcessingPathway.DORSAL_UNCONSCIOUS,
            ProcessingPathway.SUBCORTICAL,
            ProcessingPathway.RESIDUAL_V1,
        ]
        self.assertEqual(len(pathways), 4)

    def test_pathway_values(self):
        """Pathways should have expected string values."""
        self.assertEqual(ProcessingPathway.VENTRAL_CONSCIOUS.value, "ventral_conscious")
        self.assertEqual(ProcessingPathway.DORSAL_UNCONSCIOUS.value, "dorsal_unconscious")


class TestStimulusProperty(unittest.TestCase):
    """Tests for StimulusProperty enumeration."""

    def test_all_properties_exist(self):
        """All stimulus properties should be defined."""
        props = [
            StimulusProperty.MOTION,
            StimulusProperty.ORIENTATION,
            StimulusProperty.COLOR,
            StimulusProperty.SPATIAL_FREQUENCY,
            StimulusProperty.EMOTION,
            StimulusProperty.LUMINANCE,
            StimulusProperty.SHAPE,
        ]
        self.assertEqual(len(props), 7)


class TestDetectionConfidence(unittest.TestCase):
    """Tests for DetectionConfidence enumeration."""

    def test_all_levels_exist(self):
        """All detection confidence levels should be defined."""
        levels = [
            DetectionConfidence.PURE_GUESS,
            DetectionConfidence.SLIGHT_HUNCH,
            DetectionConfidence.MODERATE_FEELING,
            DetectionConfidence.STRONG_FEELING,
        ]
        self.assertEqual(len(levels), 4)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestBlindsightInput(unittest.TestCase):
    """Tests for BlindsightInput dataclass."""

    def test_basic_creation(self):
        """Test basic input creation."""
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.7,
            stimulus_duration_ms=200.0,
        )
        self.assertEqual(inp.stimulus_region, VisualFieldRegion.BLIND_FIELD)
        self.assertEqual(inp.stimulus_property, StimulusProperty.MOTION)
        self.assertAlmostEqual(inp.stimulus_intensity, 0.7)

    def test_default_values(self):
        """Test default values."""
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.ORIENTATION,
            stimulus_intensity=0.5,
            stimulus_duration_ms=100.0,
        )
        self.assertAlmostEqual(inp.stimulus_size_degrees, 2.0)
        self.assertAlmostEqual(inp.contrast, 0.8)
        self.assertEqual(inp.task_type, "forced_choice")
        self.assertEqual(inp.response_options, 2)

    def test_to_dict(self):
        """Test serialization."""
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.EMOTION,
            stimulus_intensity=0.8,
            stimulus_duration_ms=150.0,
        )
        d = inp.to_dict()
        self.assertEqual(d["stimulus_region"], "blind_field")
        self.assertEqual(d["stimulus_property"], "emotion")
        self.assertIn("timestamp", d)


class TestForcedChoiceTrialInput(unittest.TestCase):
    """Tests for ForcedChoiceTrialInput dataclass."""

    def test_creation(self):
        """Test trial input creation."""
        stimulus = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.6,
            stimulus_duration_ms=200.0,
        )
        trial = ForcedChoiceTrialInput(
            stimulus=stimulus,
            correct_response="left",
            trial_number=1,
        )
        self.assertEqual(trial.correct_response, "left")
        self.assertEqual(trial.trial_number, 1)

    def test_to_dict(self):
        """Test serialization."""
        stimulus = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.ORIENTATION,
            stimulus_intensity=0.5,
            stimulus_duration_ms=100.0,
        )
        trial = ForcedChoiceTrialInput(
            stimulus=stimulus,
            correct_response="vertical",
        )
        d = trial.to_dict()
        self.assertIn("stimulus", d)
        self.assertEqual(d["correct_response"], "vertical")


class TestBlindsightOutput(unittest.TestCase):
    """Tests for BlindsightOutput dataclass."""

    def test_creation(self):
        """Test output creation."""
        output = BlindsightOutput(
            implicit_detection=True,
            forced_choice_accuracy=0.75,
            above_chance=True,
            blindsight_type=BlindsightType.TYPE_1_GUESSING,
        )
        self.assertTrue(output.implicit_detection)
        self.assertTrue(output.above_chance)

    def test_to_dict(self):
        """Test serialization."""
        output = BlindsightOutput(
            implicit_detection=False,
            forced_choice_accuracy=0.5,
            above_chance=False,
        )
        d = output.to_dict()
        self.assertFalse(d["implicit_detection"])
        self.assertIsNone(d["blindsight_type"])


class TestPathwayAnalysis(unittest.TestCase):
    """Tests for PathwayAnalysis dataclass."""

    def test_creation(self):
        """Test pathway analysis creation."""
        analysis = PathwayAnalysis(
            pathway=ProcessingPathway.DORSAL_UNCONSCIOUS,
            activation_level=0.7,
            information_transmitted=["motion_direction", "spatial_location"],
            reaches_consciousness=False,
        )
        self.assertAlmostEqual(analysis.activation_level, 0.7)
        self.assertFalse(analysis.reaches_consciousness)

    def test_to_dict(self):
        """Test serialization."""
        analysis = PathwayAnalysis(
            pathway=ProcessingPathway.SUBCORTICAL,
            activation_level=0.5,
        )
        d = analysis.to_dict()
        self.assertEqual(d["pathway"], "subcortical")


class TestBlindsightProfile(unittest.TestCase):
    """Tests for BlindsightProfile dataclass."""

    def test_creation(self):
        """Test profile creation."""
        profile = BlindsightProfile(
            blindsight_type=BlindsightType.TYPE_1_GUESSING,
            affected_field=VisualFieldRegion.BLIND_FIELD,
            preserved_properties=[StimulusProperty.MOTION, StimulusProperty.LUMINANCE],
            active_pathways=[ProcessingPathway.DORSAL_UNCONSCIOUS],
        )
        self.assertEqual(len(profile.preserved_properties), 2)

    def test_to_dict(self):
        """Test serialization."""
        profile = BlindsightProfile(
            blindsight_type=BlindsightType.TYPE_2_FEELING,
            affected_field=VisualFieldRegion.BLIND_FIELD,
        )
        d = profile.to_dict()
        self.assertEqual(d["blindsight_type"], "type_2_feeling")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestBlindsightInterface(unittest.TestCase):
    """Tests for BlindsightInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.interface = BlindsightInterface()
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
        self.assertEqual(self.interface.FORM_ID, "25-blindsight")
        self.assertEqual(self.interface.FORM_NAME, "Blindsight Consciousness")

    def test_initialization(self):
        """Test interface initializes correctly."""
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)
        self.assertEqual(len(self.interface._property_results), len(StimulusProperty))

    def test_double_initialization(self):
        """Test that double initialization is safe."""
        self._run(self.interface.initialize())
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_process_blind_field_stimulus(self):
        """Test processing a stimulus in the blind field."""
        self._run(self.interface.initialize())
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.8,
            stimulus_duration_ms=200.0,
        )
        output = self._run(self.interface.process_blind_field(inp))
        self.assertIsInstance(output, BlindsightOutput)
        self.assertTrue(output.implicit_detection)
        self.assertIsNotNone(output.blindsight_type)
        # Motion at high intensity triggers Type 2 blindsight (vague feeling)
        self.assertIn(output.conscious_report, ("nothing", "vague_feeling"))

    def test_process_intact_field_stimulus(self):
        """Test processing a stimulus in the intact field."""
        self._run(self.interface.initialize())
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.INTACT_FOVEAL,
            stimulus_property=StimulusProperty.COLOR,
            stimulus_intensity=0.9,
            stimulus_duration_ms=500.0,
        )
        output = self._run(self.interface.process_blind_field(inp))
        self.assertIsNone(output.blindsight_type)
        self.assertEqual(output.conscious_report, "clear_perception")

    def test_process_weak_stimulus(self):
        """Test processing a weak stimulus in blind field."""
        self._run(self.interface.initialize())
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.COLOR,
            stimulus_intensity=0.1,
            stimulus_duration_ms=50.0,
            contrast=0.1,
        )
        output = self._run(self.interface.process_blind_field(inp))
        self.assertIsInstance(output, BlindsightOutput)

    def test_forced_choice_test(self):
        """Test forced-choice trial."""
        self._run(self.interface.initialize())
        stimulus = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.7,
            stimulus_duration_ms=200.0,
        )
        trial = ForcedChoiceTrialInput(
            stimulus=stimulus,
            correct_response="left",
            trial_number=1,
        )
        result = self._run(self.interface.forced_choice_test(trial))
        self.assertIn("trial_number", result)
        self.assertIn("correct", result)
        self.assertIn("running_accuracy", result)
        self.assertEqual(result["trial_number"], 1)

    def test_multiple_forced_choice_trials(self):
        """Test accumulation of forced-choice trial data."""
        self._run(self.interface.initialize())
        for i in range(10):
            stimulus = BlindsightInput(
                stimulus_region=VisualFieldRegion.BLIND_FIELD,
                stimulus_property=StimulusProperty.MOTION,
                stimulus_intensity=0.7,
                stimulus_duration_ms=200.0,
            )
            trial = ForcedChoiceTrialInput(
                stimulus=stimulus,
                correct_response="left",
                trial_number=i + 1,
            )
            self._run(self.interface.forced_choice_test(trial))
        self.assertEqual(self.interface._total_trials, 10)

    def test_assess_implicit_detection_no_data(self):
        """Test implicit detection assessment with no trials."""
        self._run(self.interface.initialize())
        result = self._run(
            self.interface.assess_implicit_detection(StimulusProperty.MOTION)
        )
        self.assertIsInstance(result, ImplicitDetectionResult)
        self.assertFalse(result.detected)
        self.assertEqual(result.n_trials, 0)

    def test_assess_implicit_detection_with_data(self):
        """Test implicit detection after running trials."""
        self._run(self.interface.initialize())
        # Run several trials to build data
        for i in range(20):
            stimulus = BlindsightInput(
                stimulus_region=VisualFieldRegion.BLIND_FIELD,
                stimulus_property=StimulusProperty.MOTION,
                stimulus_intensity=0.8,
                stimulus_duration_ms=200.0,
            )
            self._run(self.interface.process_blind_field(stimulus))

        result = self._run(
            self.interface.assess_implicit_detection(StimulusProperty.MOTION)
        )
        self.assertGreater(result.n_trials, 0)
        self.assertGreater(result.accuracy, 0.0)

    def test_analyze_pathway_blind_field(self):
        """Test pathway analysis for blind field stimulus."""
        self._run(self.interface.initialize())
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.7,
            stimulus_duration_ms=200.0,
        )
        analyses = self._run(self.interface.analyze_pathway(inp))
        self.assertIsInstance(analyses, list)
        self.assertGreater(len(analyses), 0)

        # Ventral pathway should be suppressed in blind field
        ventral = [a for a in analyses if a.pathway == ProcessingPathway.VENTRAL_CONSCIOUS]
        self.assertEqual(len(ventral), 1)
        self.assertAlmostEqual(ventral[0].activation_level, 0.0)
        self.assertFalse(ventral[0].reaches_consciousness)

        # Dorsal should be active
        dorsal = [a for a in analyses if a.pathway == ProcessingPathway.DORSAL_UNCONSCIOUS]
        self.assertEqual(len(dorsal), 1)
        self.assertGreater(dorsal[0].activation_level, 0.0)

    def test_analyze_pathway_intact_field(self):
        """Test pathway analysis for intact field stimulus."""
        self._run(self.interface.initialize())
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.INTACT_FOVEAL,
            stimulus_property=StimulusProperty.COLOR,
            stimulus_intensity=0.9,
            stimulus_duration_ms=500.0,
        )
        analyses = self._run(self.interface.analyze_pathway(inp))
        ventral = [a for a in analyses if a.pathway == ProcessingPathway.VENTRAL_CONSCIOUS]
        self.assertGreater(ventral[0].activation_level, 0.0)
        self.assertTrue(ventral[0].reaches_consciousness)

    def test_to_dict(self):
        """Test interface serialization."""
        self._run(self.interface.initialize())
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "25-blindsight")
        self.assertTrue(d["initialized"])
        self.assertEqual(d["total_trials"], 0)

    def test_get_status(self):
        """Test status retrieval."""
        self._run(self.interface.initialize())
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "25-blindsight")
        self.assertEqual(status["properties_tested"], 0)

    def test_auto_initialize(self):
        """Test auto-initialization on method call."""
        inp = BlindsightInput(
            stimulus_region=VisualFieldRegion.BLIND_FIELD,
            stimulus_property=StimulusProperty.MOTION,
            stimulus_intensity=0.5,
            stimulus_duration_ms=100.0,
        )
        output = self._run(self.interface.process_blind_field(inp))
        self.assertTrue(self.interface._initialized)
        self.assertIsInstance(output, BlindsightOutput)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module convenience functions."""

    def test_create_blindsight_interface(self):
        """Test convenience creation function."""
        interface = create_blindsight_interface()
        self.assertIsInstance(interface, BlindsightInterface)
        self.assertFalse(interface._initialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
