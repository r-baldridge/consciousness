#!/usr/bin/env python3
"""
Test Suite for Form 09: Perceptual Consciousness.

Tests cover:
- All enumerations (PerceptualBindingType, GestaltPrinciple, AttentionalMode, etc.)
- All input/output dataclasses
- FeatureBindingEngine
- PerceptualOrganizationEngine
- PerceptualConsciousnessInterface (main interface)
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
    PerceptualBindingType,
    GestaltPrinciple,
    AttentionalMode,
    SensoryChannel,
    PerceptualQuality,
    # Input dataclasses
    SensoryFeature,
    PerceptualInput,
    # Output dataclasses
    BoundPercept,
    SceneRepresentation,
    PerceptualOutput,
    PerceptualSystemStatus,
    # Engines
    FeatureBindingEngine,
    PerceptualOrganizationEngine,
    # Main interface
    PerceptualConsciousnessInterface,
    # Convenience
    create_perceptual_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestPerceptualBindingType(unittest.TestCase):
    """Tests for PerceptualBindingType enumeration."""

    def test_all_types_exist(self):
        """All binding types should be defined."""
        types = [
            PerceptualBindingType.SPATIAL,
            PerceptualBindingType.TEMPORAL,
            PerceptualBindingType.FEATURE,
            PerceptualBindingType.OBJECT,
            PerceptualBindingType.CROSS_MODAL,
            PerceptualBindingType.SEMANTIC,
        ]
        self.assertEqual(len(types), 6)

    def test_type_values(self):
        """Types should have expected string values."""
        self.assertEqual(PerceptualBindingType.SPATIAL.value, "spatial")
        self.assertEqual(PerceptualBindingType.CROSS_MODAL.value, "cross_modal")


class TestGestaltPrinciple(unittest.TestCase):
    """Tests for GestaltPrinciple enumeration."""

    def test_all_principles_exist(self):
        """All Gestalt principles should be defined."""
        principles = [
            GestaltPrinciple.PROXIMITY,
            GestaltPrinciple.SIMILARITY,
            GestaltPrinciple.CONTINUITY,
            GestaltPrinciple.CLOSURE,
            GestaltPrinciple.COMMON_FATE,
            GestaltPrinciple.FIGURE_GROUND,
            GestaltPrinciple.PRAGNANZ,
            GestaltPrinciple.COMMON_REGION,
        ]
        self.assertEqual(len(principles), 8)


class TestAttentionalMode(unittest.TestCase):
    """Tests for AttentionalMode enumeration."""

    def test_all_modes_exist(self):
        """All attentional modes should be defined."""
        modes = [
            AttentionalMode.FOCAL,
            AttentionalMode.DIFFUSE,
            AttentionalMode.FEATURE_BASED,
            AttentionalMode.OBJECT_BASED,
            AttentionalMode.SPATIAL,
            AttentionalMode.EXOGENOUS,
            AttentionalMode.ENDOGENOUS,
        ]
        self.assertEqual(len(modes), 7)


class TestSensoryChannel(unittest.TestCase):
    """Tests for SensoryChannel enumeration."""

    def test_all_channels_exist(self):
        """All sensory channels should be defined."""
        channels = [
            SensoryChannel.VISUAL,
            SensoryChannel.AUDITORY,
            SensoryChannel.SOMATOSENSORY,
            SensoryChannel.OLFACTORY,
            SensoryChannel.GUSTATORY,
            SensoryChannel.INTEROCEPTIVE,
        ]
        self.assertEqual(len(channels), 6)


class TestPerceptualQuality(unittest.TestCase):
    """Tests for PerceptualQuality enumeration."""

    def test_all_qualities_exist(self):
        """All perceptual qualities should be defined."""
        qualities = [
            PerceptualQuality.VIVID,
            PerceptualQuality.CLEAR,
            PerceptualQuality.FUZZY,
            PerceptualQuality.FRAGMENTARY,
            PerceptualQuality.ILLUSORY,
        ]
        self.assertEqual(len(qualities), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestSensoryFeature(unittest.TestCase):
    """Tests for SensoryFeature dataclass."""

    def test_creation(self):
        """Should create sensory feature."""
        feature = SensoryFeature(
            feature_id="f001",
            channel=SensoryChannel.VISUAL,
            feature_type="color",
            feature_value="red",
            intensity=0.8,
            spatial_location=(0.5, 0.3, 0.0),
        )
        self.assertEqual(feature.feature_id, "f001")
        self.assertEqual(feature.channel, SensoryChannel.VISUAL)
        self.assertEqual(feature.intensity, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        feature = SensoryFeature(
            feature_id="f002",
            channel=SensoryChannel.AUDITORY,
            feature_type="pitch",
            feature_value=440,
            intensity=0.6,
        )
        d = feature.to_dict()
        self.assertEqual(d["feature_id"], "f002")
        self.assertEqual(d["channel"], "auditory")

    def test_default_values(self):
        """Should have correct default values."""
        feature = SensoryFeature(
            feature_id="f003",
            channel=SensoryChannel.VISUAL,
            feature_type="shape",
            feature_value="circle",
            intensity=0.5,
        )
        self.assertIsNone(feature.spatial_location)
        self.assertEqual(feature.temporal_onset, 0.0)
        self.assertEqual(feature.confidence, 0.8)


class TestPerceptualInput(unittest.TestCase):
    """Tests for PerceptualInput dataclass."""

    def test_empty_input(self):
        """Should create empty input."""
        inp = PerceptualInput()
        self.assertEqual(len(inp.features), 0)
        self.assertEqual(inp.attentional_mode, AttentionalMode.DIFFUSE)

    def test_full_input(self):
        """Should create input with features."""
        inp = PerceptualInput(
            features=[
                SensoryFeature(
                    feature_id="f1",
                    channel=SensoryChannel.VISUAL,
                    feature_type="color",
                    feature_value="blue",
                    intensity=0.7,
                ),
            ],
            attentional_mode=AttentionalMode.FOCAL,
            attentional_focus="color",
        )
        self.assertEqual(len(inp.features), 1)
        self.assertEqual(inp.attentional_mode, AttentionalMode.FOCAL)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestBoundPercept(unittest.TestCase):
    """Tests for BoundPercept dataclass."""

    def test_creation(self):
        """Should create bound percept."""
        percept = BoundPercept(
            percept_id="p001",
            binding_type=PerceptualBindingType.SPATIAL,
            bound_features=["f1", "f2"],
            channels_involved=[SensoryChannel.VISUAL],
            gestalt_principles=[GestaltPrinciple.PROXIMITY],
            coherence=0.9,
            salience=0.8,
        )
        self.assertEqual(percept.percept_id, "p001")
        self.assertEqual(len(percept.bound_features), 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        percept = BoundPercept(
            percept_id="p002",
            binding_type=PerceptualBindingType.CROSS_MODAL,
            bound_features=["f1"],
            channels_involved=[SensoryChannel.VISUAL, SensoryChannel.AUDITORY],
            gestalt_principles=[GestaltPrinciple.COMMON_FATE],
            coherence=0.7,
            salience=0.6,
            label="bell",
        )
        d = percept.to_dict()
        self.assertEqual(d["binding_type"], "cross_modal")
        self.assertEqual(d["label"], "bell")


class TestSceneRepresentation(unittest.TestCase):
    """Tests for SceneRepresentation dataclass."""

    def test_empty_scene(self):
        """Should create empty scene."""
        scene = SceneRepresentation(percepts=[])
        self.assertEqual(len(scene.percepts), 0)
        self.assertIsNone(scene.figure)

    def test_to_dict(self):
        """Should convert to dictionary."""
        scene = SceneRepresentation(
            percepts=[],
            figure="p001",
            ground="p002",
            scene_coherence=0.8,
            complexity=0.5,
        )
        d = scene.to_dict()
        self.assertEqual(d["figure"], "p001")
        self.assertEqual(d["ground"], "p002")


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestFeatureBindingEngine(unittest.TestCase):
    """Tests for FeatureBindingEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = FeatureBindingEngine()

    def test_bind_empty_features(self):
        """Should return empty for no features."""
        result = self.engine.bind_features([])
        self.assertEqual(len(result), 0)

    def test_bind_spatial_features(self):
        """Should bind spatially proximate features."""
        features = [
            SensoryFeature(
                feature_id="f1", channel=SensoryChannel.VISUAL,
                feature_type="color", feature_value="red",
                intensity=0.8, spatial_location=(0.1, 0.1, 0.0),
            ),
            SensoryFeature(
                feature_id="f2", channel=SensoryChannel.VISUAL,
                feature_type="shape", feature_value="circle",
                intensity=0.7, spatial_location=(0.15, 0.12, 0.0),
            ),
        ]
        percepts = self.engine.bind_features(features)
        self.assertGreater(len(percepts), 0)
        # The features should be bound together
        multi_feature = [p for p in percepts if len(p.bound_features) > 1]
        self.assertGreater(len(multi_feature), 0)

    def test_bind_temporal_features(self):
        """Should bind temporally co-occurring features."""
        features = [
            SensoryFeature(
                feature_id="f1", channel=SensoryChannel.VISUAL,
                feature_type="flash", feature_value="bright",
                intensity=0.9, temporal_onset=10.0,
            ),
            SensoryFeature(
                feature_id="f2", channel=SensoryChannel.AUDITORY,
                feature_type="sound", feature_value="bang",
                intensity=0.8, temporal_onset=15.0,
            ),
        ]
        percepts = self.engine.bind_features(features)
        self.assertGreater(len(percepts), 0)

    def test_bind_cross_modal_features(self):
        """Should bind features across modalities."""
        features = [
            SensoryFeature(
                feature_id="f1", channel=SensoryChannel.VISUAL,
                feature_type="motion", feature_value="lip_movement",
                intensity=0.7, temporal_onset=0.0,
            ),
            SensoryFeature(
                feature_id="f2", channel=SensoryChannel.AUDITORY,
                feature_type="speech", feature_value="hello",
                intensity=0.8, temporal_onset=5.0,
            ),
        ]
        percepts = self.engine.bind_features(features)
        cross_modal = [p for p in percepts if p.binding_type == PerceptualBindingType.CROSS_MODAL]
        self.assertGreater(len(cross_modal), 0)

    def test_binding_success_rate(self):
        """Should compute binding success rate."""
        features = [
            SensoryFeature(
                feature_id="f1", channel=SensoryChannel.VISUAL,
                feature_type="color", feature_value="red",
                intensity=0.8,
            ),
        ]
        percepts = self.engine.bind_features(features)
        rate = self.engine.compute_binding_success(features, percepts)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


class TestPerceptualOrganizationEngine(unittest.TestCase):
    """Tests for PerceptualOrganizationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = PerceptualOrganizationEngine()

    def test_organize_empty_scene(self):
        """Should handle empty percept list."""
        scene = self.engine.organize_scene([])
        self.assertEqual(len(scene.percepts), 0)

    def test_organize_with_percepts(self):
        """Should organize percepts into scene with figure-ground."""
        percepts = [
            BoundPercept(
                percept_id="p1",
                binding_type=PerceptualBindingType.SPATIAL,
                bound_features=["f1", "f2"],
                channels_involved=[SensoryChannel.VISUAL],
                gestalt_principles=[GestaltPrinciple.PROXIMITY],
                coherence=0.9,
                salience=0.9,
            ),
            BoundPercept(
                percept_id="p2",
                binding_type=PerceptualBindingType.SPATIAL,
                bound_features=["f3"],
                channels_involved=[SensoryChannel.VISUAL],
                gestalt_principles=[GestaltPrinciple.PRAGNANZ],
                coherence=0.6,
                salience=0.3,
            ),
        ]
        scene = self.engine.organize_scene(percepts)
        self.assertEqual(scene.figure, "p1")  # Most salient
        self.assertEqual(scene.ground, "p2")  # Least salient
        self.assertGreater(scene.scene_coherence, 0.0)

    def test_assess_quality_vivid(self):
        """Should assess high-quality percepts as vivid."""
        percepts = [
            BoundPercept(
                percept_id="p1",
                binding_type=PerceptualBindingType.SPATIAL,
                bound_features=["f1"],
                channels_involved=[SensoryChannel.VISUAL],
                gestalt_principles=[],
                coherence=0.95,
                salience=0.9,
                confidence=0.95,
            ),
        ]
        quality = self.engine.assess_quality(percepts)
        self.assertEqual(quality, PerceptualQuality.VIVID)

    def test_assess_quality_fragmentary(self):
        """Should assess low-quality percepts as fragmentary."""
        percepts = [
            BoundPercept(
                percept_id="p1",
                binding_type=PerceptualBindingType.FEATURE,
                bound_features=["f1"],
                channels_involved=[SensoryChannel.VISUAL],
                gestalt_principles=[],
                coherence=0.2,
                salience=0.3,
                confidence=0.2,
            ),
        ]
        quality = self.engine.assess_quality(percepts)
        self.assertEqual(quality, PerceptualQuality.FRAGMENTARY)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestPerceptualConsciousnessInterface(unittest.TestCase):
    """Tests for PerceptualConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = PerceptualConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "09-perceptual")
        self.assertEqual(self.interface.FORM_NAME, "Perceptual Consciousness")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_perception(self):
        """Should process perceptual input."""
        inp = PerceptualInput(
            features=[
                SensoryFeature(
                    feature_id="f1", channel=SensoryChannel.VISUAL,
                    feature_type="color", feature_value="red",
                    intensity=0.8, spatial_location=(0.1, 0.1, 0.0),
                ),
                SensoryFeature(
                    feature_id="f2", channel=SensoryChannel.VISUAL,
                    feature_type="shape", feature_value="square",
                    intensity=0.7, spatial_location=(0.12, 0.11, 0.0),
                ),
            ],
            attentional_mode=AttentionalMode.FOCAL,
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_perception(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, PerceptualOutput)
        self.assertGreater(len(output.bound_percepts), 0)
        self.assertIsNotNone(output.scene)

    def test_get_active_percepts(self):
        """Should return active percepts."""
        percepts = self.interface.get_active_percepts()
        self.assertIsInstance(percepts, list)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, PerceptualSystemStatus)
        self.assertGreaterEqual(status.system_health, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "09-perceptual")
        self.assertEqual(d["form_name"], "Perceptual Consciousness")

    def test_focus_attention(self):
        """Should change attentional mode."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self.interface.focus_attention(AttentionalMode.FOCAL, "color")
            )
        finally:
            loop.close()
        self.assertEqual(self.interface._attentional_mode, AttentionalMode.FOCAL)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_perceptual_interface(self):
        """Should create new interface."""
        interface = create_perceptual_interface()
        self.assertIsInstance(interface, PerceptualConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "09-perceptual")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
