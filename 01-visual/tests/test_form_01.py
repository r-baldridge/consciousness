#!/usr/bin/env python3
"""
Test Suite for Form 01: Visual Consciousness.

Tests cover:
- All enumerations (VisualFeatureType, ColorSpace, SceneCategory, ObjectCategory, AttentionMode)
- All input/output dataclasses
- VisualConsciousnessInterface processing pipeline
- Feature extraction, object recognition, scene interpretation, salience
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
    VisualFeatureType,
    ColorSpace,
    SceneCategory,
    ObjectCategory,
    AttentionMode,
    # Input dataclasses
    VisualFeatureVector,
    VisualInput,
    ObjectDetection,
    # Output dataclasses
    FeatureExtractionResult,
    SceneInterpretation,
    SalienceMap,
    VisualOutput,
    # Main interface
    VisualConsciousnessInterface,
    # Convenience functions
    create_visual_interface,
    create_simple_visual_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestVisualFeatureType(unittest.TestCase):
    """Tests for VisualFeatureType enumeration."""

    def test_all_feature_types_exist(self):
        """All visual feature types should be defined."""
        types = [
            VisualFeatureType.EDGE,
            VisualFeatureType.COLOR,
            VisualFeatureType.TEXTURE,
            VisualFeatureType.SHAPE,
            VisualFeatureType.MOTION,
            VisualFeatureType.DEPTH,
            VisualFeatureType.LUMINANCE,
            VisualFeatureType.ORIENTATION,
            VisualFeatureType.SPATIAL_FREQUENCY,
            VisualFeatureType.CONTRAST,
        ]
        self.assertEqual(len(types), 10)

    def test_feature_values(self):
        """Features should have expected string values."""
        self.assertEqual(VisualFeatureType.EDGE.value, "edge")
        self.assertEqual(VisualFeatureType.COLOR.value, "color")
        self.assertEqual(VisualFeatureType.MOTION.value, "motion")


class TestColorSpace(unittest.TestCase):
    """Tests for ColorSpace enumeration."""

    def test_all_color_spaces_exist(self):
        """All color spaces should be defined."""
        spaces = [
            ColorSpace.RGB,
            ColorSpace.HSV,
            ColorSpace.LAB,
            ColorSpace.GRAYSCALE,
            ColorSpace.OPPONENT,
        ]
        self.assertEqual(len(spaces), 5)

    def test_color_space_values(self):
        """Color spaces should have expected values."""
        self.assertEqual(ColorSpace.RGB.value, "rgb")
        self.assertEqual(ColorSpace.OPPONENT.value, "opponent")


class TestSceneCategory(unittest.TestCase):
    """Tests for SceneCategory enumeration."""

    def test_all_categories_exist(self):
        """All scene categories should be defined."""
        categories = [
            SceneCategory.NATURAL,
            SceneCategory.URBAN,
            SceneCategory.INDOOR,
            SceneCategory.ABSTRACT,
            SceneCategory.SOCIAL,
            SceneCategory.THREATENING,
            SceneCategory.FAMILIAR,
            SceneCategory.NOVEL,
        ]
        self.assertEqual(len(categories), 8)


class TestObjectCategory(unittest.TestCase):
    """Tests for ObjectCategory enumeration."""

    def test_all_categories_exist(self):
        """All object categories should be defined."""
        categories = [
            ObjectCategory.FACE,
            ObjectCategory.BODY,
            ObjectCategory.ANIMAL,
            ObjectCategory.TOOL,
            ObjectCategory.VEHICLE,
            ObjectCategory.FOOD,
            ObjectCategory.NATURAL_OBJECT,
            ObjectCategory.TEXT,
            ObjectCategory.UNKNOWN,
        ]
        self.assertEqual(len(categories), 9)


class TestAttentionMode(unittest.TestCase):
    """Tests for AttentionMode enumeration."""

    def test_all_modes_exist(self):
        """All attention modes should be defined."""
        modes = [
            AttentionMode.FOCAL,
            AttentionMode.AMBIENT,
            AttentionMode.SACCADIC,
            AttentionMode.SUSTAINED,
            AttentionMode.DIVIDED,
        ]
        self.assertEqual(len(modes), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestVisualFeatureVector(unittest.TestCase):
    """Tests for VisualFeatureVector dataclass."""

    def test_creation(self):
        """Should create feature vector with all fields."""
        fv = VisualFeatureVector(
            feature_type=VisualFeatureType.EDGE,
            values=[0.5, 0.8, 0.3],
            spatial_location=(0.4, 0.6),
            confidence=0.9,
        )
        self.assertEqual(fv.feature_type, VisualFeatureType.EDGE)
        self.assertEqual(len(fv.values), 3)
        self.assertEqual(fv.spatial_location, (0.4, 0.6))
        self.assertEqual(fv.confidence, 0.9)

    def test_to_dict(self):
        """Should convert to dictionary."""
        fv = VisualFeatureVector(
            feature_type=VisualFeatureType.COLOR,
            values=[0.7],
        )
        d = fv.to_dict()
        self.assertEqual(d["feature_type"], "color")
        self.assertEqual(d["values"], [0.7])

    def test_default_values(self):
        """Should have proper defaults."""
        fv = VisualFeatureVector(
            feature_type=VisualFeatureType.TEXTURE,
            values=[],
        )
        self.assertEqual(fv.spatial_location, (0.5, 0.5))
        self.assertEqual(fv.confidence, 1.0)


class TestVisualInput(unittest.TestCase):
    """Tests for VisualInput dataclass."""

    def test_creation_defaults(self):
        """Should create input with defaults."""
        inp = VisualInput()
        self.assertEqual(len(inp.feature_vectors), 0)
        self.assertEqual(inp.color_space, ColorSpace.RGB)
        self.assertEqual(inp.luminance_level, 0.5)
        self.assertFalse(inp.motion_detected)

    def test_creation_with_features(self):
        """Should create input with feature vectors."""
        fv = VisualFeatureVector(feature_type=VisualFeatureType.EDGE, values=[0.5])
        inp = VisualInput(
            feature_vectors=[fv],
            luminance_level=0.8,
            contrast_level=0.7,
            motion_detected=True,
            motion_velocity=0.5,
        )
        self.assertEqual(len(inp.feature_vectors), 1)
        self.assertTrue(inp.motion_detected)
        self.assertEqual(inp.motion_velocity, 0.5)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = VisualInput(luminance_level=0.6)
        d = inp.to_dict()
        self.assertEqual(d["luminance_level"], 0.6)
        self.assertIn("timestamp", d)


class TestObjectDetection(unittest.TestCase):
    """Tests for ObjectDetection dataclass."""

    def test_creation(self):
        """Should create object detection."""
        obj = ObjectDetection(
            category=ObjectCategory.FACE,
            label="human_face",
            confidence=0.95,
            is_face=True,
            emotional_valence=0.3,
        )
        self.assertEqual(obj.category, ObjectCategory.FACE)
        self.assertTrue(obj.is_face)
        self.assertEqual(obj.emotional_valence, 0.3)

    def test_to_dict(self):
        """Should convert to dictionary."""
        obj = ObjectDetection(
            category=ObjectCategory.ANIMAL,
            label="cat",
            confidence=0.8,
        )
        d = obj.to_dict()
        self.assertEqual(d["category"], "animal")
        self.assertEqual(d["label"], "cat")


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestFeatureExtractionResult(unittest.TestCase):
    """Tests for FeatureExtractionResult dataclass."""

    def test_creation(self):
        """Should create feature extraction result."""
        result = FeatureExtractionResult(
            features_extracted={"edge": 0.7, "color": 0.5},
            dominant_feature=VisualFeatureType.EDGE,
            feature_coherence=0.8,
        )
        self.assertEqual(result.dominant_feature, VisualFeatureType.EDGE)
        self.assertEqual(result.feature_coherence, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = FeatureExtractionResult(
            features_extracted={"edge": 0.7},
            dominant_feature=VisualFeatureType.EDGE,
            feature_coherence=0.9,
        )
        d = result.to_dict()
        self.assertEqual(d["dominant_feature"], "edge")
        self.assertIn("features_extracted", d)


class TestSceneInterpretation(unittest.TestCase):
    """Tests for SceneInterpretation dataclass."""

    def test_creation(self):
        """Should create scene interpretation."""
        scene = SceneInterpretation(
            scene_category=SceneCategory.NATURAL,
            scene_description="outdoor landscape",
            scene_confidence=0.85,
            spatial_layout={"center": 0.7, "periphery": 0.3},
            gist_features=["bright", "static"],
            emotional_tone=0.2,
            novelty_score=0.4,
        )
        self.assertEqual(scene.scene_category, SceneCategory.NATURAL)
        self.assertEqual(len(scene.gist_features), 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        scene = SceneInterpretation(
            scene_category=SceneCategory.URBAN,
            scene_description="city street",
            scene_confidence=0.7,
            spatial_layout={},
            gist_features=[],
            emotional_tone=0.0,
            novelty_score=0.5,
        )
        d = scene.to_dict()
        self.assertEqual(d["scene_category"], "urban")


class TestSalienceMap(unittest.TestCase):
    """Tests for SalienceMap dataclass."""

    def test_creation(self):
        """Should create salience map."""
        sm = SalienceMap(
            salience_values=[0.3, 0.5, 0.8, 0.2],
            peak_location=(0.7, 0.3),
            peak_salience=0.8,
            num_hotspots=1,
            attention_recommendation=AttentionMode.FOCAL,
        )
        self.assertEqual(sm.peak_salience, 0.8)
        self.assertEqual(sm.num_hotspots, 1)
        self.assertEqual(sm.attention_recommendation, AttentionMode.FOCAL)

    def test_to_dict(self):
        """Should convert to dictionary."""
        sm = SalienceMap(
            salience_values=[0.5],
            peak_location=(0.5, 0.5),
            peak_salience=0.5,
            num_hotspots=0,
            attention_recommendation=AttentionMode.AMBIENT,
        )
        d = sm.to_dict()
        self.assertEqual(d["attention_recommendation"], "ambient")


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestVisualConsciousnessInterface(unittest.TestCase):
    """Tests for VisualConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = VisualConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "01-visual")
        self.assertEqual(self.interface.FORM_NAME, "Visual Consciousness")

    def test_initialize(self):
        """Should initialize the processing pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_visual_input(self):
        """Should process visual input and return output."""
        inp = create_simple_visual_input(luminance=0.7, contrast=0.6)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_visual_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, VisualOutput)
        self.assertGreaterEqual(output.overall_confidence, 0.0)
        self.assertLessEqual(output.overall_confidence, 1.0)
        self.assertGreaterEqual(output.visual_clarity, 0.0)
        self.assertLessEqual(output.visual_clarity, 1.0)

    def test_process_with_motion(self):
        """Should detect motion in visual input."""
        inp = create_simple_visual_input(motion=True, motion_speed=0.7)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_visual_input(inp))
        finally:
            loop.close()

        self.assertIn("motion", output.feature_result.features_extracted)

    def test_process_with_features(self):
        """Should process feature vectors."""
        inp = VisualInput(
            feature_vectors=[
                VisualFeatureVector(feature_type=VisualFeatureType.EDGE, values=[0.8]),
                VisualFeatureVector(feature_type=VisualFeatureType.COLOR, values=[0.6]),
                VisualFeatureVector(feature_type=VisualFeatureType.SHAPE, values=[0.7]),
                VisualFeatureVector(feature_type=VisualFeatureType.CONTRAST, values=[0.9]),
            ],
            luminance_level=0.7,
            contrast_level=0.8,
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_visual_input(inp))
        finally:
            loop.close()

        self.assertGreater(len(output.feature_result.features_extracted), 0)
        self.assertGreater(len(output.objects_detected), 0)

    def test_attention_mode(self):
        """Should get and set attention mode."""
        self.assertEqual(self.interface.get_attention_mode(), AttentionMode.AMBIENT)
        self.interface.set_attention_mode(AttentionMode.FOCAL)
        self.assertEqual(self.interface.get_attention_mode(), AttentionMode.FOCAL)

    def test_processing_count(self):
        """Should track processing count."""
        self.assertEqual(self.interface._processing_count, 0)
        inp = create_simple_visual_input()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_visual_input(inp))
            loop.run_until_complete(self.interface.process_visual_input(inp))
        finally:
            loop.close()

        self.assertEqual(self.interface._processing_count, 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "01-visual")
        self.assertEqual(d["form_name"], "Visual Consciousness")
        self.assertIn("attention_mode", d)
        self.assertIn("processing_count", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "01-visual")
        self.assertTrue(status["operational"])
        self.assertIn("visual_clarity", status)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_visual_interface(self):
        """Should create new interface."""
        interface = create_visual_interface()
        self.assertIsInstance(interface, VisualConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "01-visual")

    def test_create_simple_visual_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_visual_input()
        self.assertIsInstance(inp, VisualInput)
        self.assertEqual(len(inp.feature_vectors), 2)
        self.assertEqual(inp.luminance_level, 0.5)
        self.assertFalse(inp.motion_detected)

    def test_create_simple_visual_input_custom(self):
        """Should create input with custom values."""
        inp = create_simple_visual_input(
            luminance=0.8,
            contrast=0.9,
            motion=True,
            motion_speed=0.5,
        )
        self.assertEqual(inp.luminance_level, 0.8)
        self.assertEqual(inp.contrast_level, 0.9)
        self.assertTrue(inp.motion_detected)
        self.assertEqual(inp.motion_velocity, 0.5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete visual pipeline."""

    def test_full_pipeline(self):
        """Should complete full visual processing pipeline."""
        interface = create_visual_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())

            inp = VisualInput(
                feature_vectors=[
                    VisualFeatureVector(feature_type=VisualFeatureType.EDGE, values=[0.8]),
                    VisualFeatureVector(feature_type=VisualFeatureType.COLOR, values=[0.6]),
                    VisualFeatureVector(feature_type=VisualFeatureType.SHAPE, values=[0.7]),
                ],
                luminance_level=0.7,
                contrast_level=0.6,
                motion_detected=True,
                motion_velocity=0.3,
            )

            output = loop.run_until_complete(interface.process_visual_input(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, VisualOutput)
        self.assertIsNotNone(output.feature_result)
        self.assertIsNotNone(output.scene_interpretation)
        self.assertIsNotNone(output.salience_map)
        self.assertGreater(output.overall_confidence, 0.0)

    def test_object_memory_builds(self):
        """Should build object familiarity over repeated exposures."""
        interface = create_visual_interface()
        inp = VisualInput(
            feature_vectors=[
                VisualFeatureVector(feature_type=VisualFeatureType.CONTRAST, values=[0.8]),
                VisualFeatureVector(feature_type=VisualFeatureType.SHAPE, values=[0.6]),
            ],
            luminance_level=0.7,
            contrast_level=0.7,
        )

        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(interface.process_visual_input(inp))
        finally:
            loop.close()

        # Objects should have been tracked
        self.assertGreater(len(interface._object_memory), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
