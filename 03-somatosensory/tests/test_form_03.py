#!/usr/bin/env python3
"""
Test Suite for Form 03: Somatosensory Consciousness.

Tests cover:
- All enumerations (TouchType, PainType, BodyRegion, ProprioceptiveChannel, BodySchemaState)
- All input/output dataclasses
- SomatosensoryConsciousnessInterface processing pipeline
- Touch classification, pain assessment, body schema
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
    TouchType,
    PainType,
    BodyRegion,
    ProprioceptiveChannel,
    BodySchemaState,
    # Input dataclasses
    TouchInput,
    PainInput,
    ProprioceptiveInput,
    SomatosensoryInput,
    # Output dataclasses
    TouchClassification,
    PainAssessment,
    BodySchema,
    SomatosensoryOutput,
    # Main interface
    SomatosensoryConsciousnessInterface,
    # Convenience functions
    create_somatosensory_interface,
    create_simple_touch_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestTouchType(unittest.TestCase):
    """Tests for TouchType enumeration."""

    def test_all_types_exist(self):
        """All touch types should be defined."""
        types = [
            TouchType.LIGHT_TOUCH, TouchType.PRESSURE,
            TouchType.VIBRATION, TouchType.TEXTURE,
            TouchType.TEMPERATURE_WARM, TouchType.TEMPERATURE_COLD,
            TouchType.ITCH, TouchType.TICKLE, TouchType.NONE,
        ]
        self.assertEqual(len(types), 9)

    def test_values(self):
        """Types should have expected string values."""
        self.assertEqual(TouchType.LIGHT_TOUCH.value, "light_touch")
        self.assertEqual(TouchType.PRESSURE.value, "pressure")


class TestPainType(unittest.TestCase):
    """Tests for PainType enumeration."""

    def test_all_types_exist(self):
        """All pain types should be defined."""
        types = [
            PainType.SHARP, PainType.DULL, PainType.BURNING,
            PainType.THROBBING, PainType.ACHING, PainType.TINGLING,
            PainType.NEUROPATHIC, PainType.REFERRED, PainType.NONE,
        ]
        self.assertEqual(len(types), 9)

    def test_values(self):
        """Types should have expected string values."""
        self.assertEqual(PainType.SHARP.value, "sharp")
        self.assertEqual(PainType.BURNING.value, "burning")


class TestBodyRegion(unittest.TestCase):
    """Tests for BodyRegion enumeration."""

    def test_all_regions_exist(self):
        """All body regions should be defined."""
        regions = [
            BodyRegion.HEAD, BodyRegion.FACE, BodyRegion.NECK,
            BodyRegion.SHOULDER_LEFT, BodyRegion.SHOULDER_RIGHT,
            BodyRegion.ARM_LEFT, BodyRegion.ARM_RIGHT,
            BodyRegion.HAND_LEFT, BodyRegion.HAND_RIGHT,
            BodyRegion.TORSO_FRONT, BodyRegion.TORSO_BACK,
            BodyRegion.HIP_LEFT, BodyRegion.HIP_RIGHT,
            BodyRegion.LEG_LEFT, BodyRegion.LEG_RIGHT,
            BodyRegion.FOOT_LEFT, BodyRegion.FOOT_RIGHT,
            BodyRegion.INTERNAL,
        ]
        self.assertEqual(len(regions), 18)


class TestProprioceptiveChannel(unittest.TestCase):
    """Tests for ProprioceptiveChannel enumeration."""

    def test_all_channels_exist(self):
        """All proprioceptive channels should be defined."""
        channels = [
            ProprioceptiveChannel.JOINT_POSITION,
            ProprioceptiveChannel.MUSCLE_TENSION,
            ProprioceptiveChannel.BALANCE,
            ProprioceptiveChannel.MOVEMENT_SENSE,
            ProprioceptiveChannel.FORCE_SENSE,
            ProprioceptiveChannel.BODY_POSITION,
        ]
        self.assertEqual(len(channels), 6)


class TestBodySchemaState(unittest.TestCase):
    """Tests for BodySchemaState enumeration."""

    def test_all_states_exist(self):
        """All body schema states should be defined."""
        states = [
            BodySchemaState.NORMAL, BodySchemaState.HEIGHTENED,
            BodySchemaState.DIMINISHED, BodySchemaState.DISTORTED,
            BodySchemaState.PHANTOM, BodySchemaState.DISSOCIATED,
        ]
        self.assertEqual(len(states), 6)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestTouchInput(unittest.TestCase):
    """Tests for TouchInput dataclass."""

    def test_creation(self):
        """Should create touch input."""
        inp = TouchInput(
            touch_type=TouchType.LIGHT_TOUCH,
            body_region=BodyRegion.HAND_RIGHT,
            intensity=0.5,
            area_size=0.1,
            duration_ms=200.0,
        )
        self.assertEqual(inp.touch_type, TouchType.LIGHT_TOUCH)
        self.assertEqual(inp.body_region, BodyRegion.HAND_RIGHT)
        self.assertEqual(inp.intensity, 0.5)
        self.assertEqual(inp.temperature, 0.5)  # Default

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = TouchInput(
            touch_type=TouchType.PRESSURE,
            body_region=BodyRegion.ARM_LEFT,
            intensity=0.7,
            area_size=0.2,
            duration_ms=500.0,
        )
        d = inp.to_dict()
        self.assertEqual(d["touch_type"], "pressure")
        self.assertEqual(d["body_region"], "arm_left")


class TestPainInput(unittest.TestCase):
    """Tests for PainInput dataclass."""

    def test_creation(self):
        """Should create pain input."""
        inp = PainInput(
            pain_type=PainType.SHARP,
            body_region=BodyRegion.FOOT_LEFT,
            intensity=0.8,
            sharpness=0.9,
            duration_ms=100.0,
            tissue_damage=0.3,
        )
        self.assertEqual(inp.pain_type, PainType.SHARP)
        self.assertEqual(inp.intensity, 0.8)
        self.assertFalse(inp.is_chronic)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = PainInput(
            pain_type=PainType.DULL,
            body_region=BodyRegion.TORSO_BACK,
            intensity=0.4,
            sharpness=0.2,
            duration_ms=5000.0,
            is_chronic=True,
        )
        d = inp.to_dict()
        self.assertEqual(d["pain_type"], "dull")
        self.assertTrue(d["is_chronic"])


class TestProprioceptiveInput(unittest.TestCase):
    """Tests for ProprioceptiveInput dataclass."""

    def test_creation(self):
        """Should create proprioceptive input."""
        inp = ProprioceptiveInput(
            channel=ProprioceptiveChannel.BALANCE,
            body_region=BodyRegion.TORSO_FRONT,
            value=0.7,
        )
        self.assertEqual(inp.channel, ProprioceptiveChannel.BALANCE)
        self.assertEqual(inp.value, 0.7)


class TestSomatosensoryInput(unittest.TestCase):
    """Tests for SomatosensoryInput dataclass."""

    def test_creation_empty(self):
        """Should create empty input."""
        inp = SomatosensoryInput()
        self.assertEqual(len(inp.touch_inputs), 0)
        self.assertEqual(len(inp.pain_inputs), 0)
        self.assertEqual(inp.overall_body_temperature, 0.5)

    def test_creation_full(self):
        """Should create input with all data."""
        inp = SomatosensoryInput(
            touch_inputs=[
                TouchInput(TouchType.LIGHT_TOUCH, BodyRegion.HAND_RIGHT, 0.5, 0.1, 100.0)
            ],
            pain_inputs=[
                PainInput(PainType.DULL, BodyRegion.TORSO_BACK, 0.3, 0.2, 1000.0)
            ],
            proprioceptive_inputs=[
                ProprioceptiveInput(ProprioceptiveChannel.BALANCE, BodyRegion.TORSO_FRONT, 0.7)
            ],
            muscle_tension_global=0.4,
        )
        self.assertEqual(len(inp.touch_inputs), 1)
        self.assertEqual(len(inp.pain_inputs), 1)
        self.assertEqual(len(inp.proprioceptive_inputs), 1)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestTouchClassification(unittest.TestCase):
    """Tests for TouchClassification dataclass."""

    def test_creation(self):
        """Should create touch classification."""
        tc = TouchClassification(
            touch_type=TouchType.LIGHT_TOUCH,
            body_region=BodyRegion.HAND_RIGHT,
            perceived_intensity=0.5,
            pleasantness=0.3,
            novelty=0.6,
            threat_level=0.0,
            confidence=0.8,
        )
        self.assertEqual(tc.pleasantness, 0.3)
        self.assertEqual(tc.threat_level, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        tc = TouchClassification(
            touch_type=TouchType.PRESSURE,
            body_region=BodyRegion.ARM_LEFT,
            perceived_intensity=0.7,
            pleasantness=-0.2,
            novelty=0.3,
            threat_level=0.4,
            confidence=0.7,
        )
        d = tc.to_dict()
        self.assertEqual(d["touch_type"], "pressure")


class TestPainAssessment(unittest.TestCase):
    """Tests for PainAssessment dataclass."""

    def test_creation(self):
        """Should create pain assessment."""
        pa = PainAssessment(
            pain_type=PainType.SHARP,
            body_region=BodyRegion.FOOT_LEFT,
            subjective_intensity=0.8,
            emotional_distress=0.7,
            action_urgency=0.9,
            coping_capacity=0.3,
            requires_attention=True,
        )
        self.assertTrue(pa.requires_attention)
        self.assertEqual(pa.action_urgency, 0.9)


class TestBodySchema(unittest.TestCase):
    """Tests for BodySchema dataclass."""

    def test_creation(self):
        """Should create body schema."""
        bs = BodySchema(
            schema_state=BodySchemaState.NORMAL,
            body_boundary_clarity=0.8,
            postural_stability=0.9,
            body_ownership=0.95,
            region_activation={"hand_right": 0.5, "torso_front": 0.3},
            balance_state=0.7,
        )
        self.assertEqual(bs.schema_state, BodySchemaState.NORMAL)
        self.assertIn("hand_right", bs.region_activation)

    def test_to_dict(self):
        """Should convert to dictionary."""
        bs = BodySchema(
            schema_state=BodySchemaState.HEIGHTENED,
            body_boundary_clarity=0.9,
            postural_stability=0.7,
            body_ownership=0.9,
            region_activation={},
            balance_state=0.5,
        )
        d = bs.to_dict()
        self.assertEqual(d["schema_state"], "heightened")


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestSomatosensoryConsciousnessInterface(unittest.TestCase):
    """Tests for SomatosensoryConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = SomatosensoryConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "03-somatosensory")
        self.assertEqual(self.interface.FORM_NAME, "Somatosensory Consciousness")

    def test_initialize(self):
        """Should initialize the pipeline."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_touch(self):
        """Should process touch input."""
        inp = create_simple_touch_input(intensity=0.6)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_somatosensory_input(inp)
            )
        finally:
            loop.close()
        self.assertIsInstance(output, SomatosensoryOutput)
        self.assertGreater(len(output.touch_classifications), 0)

    def test_process_pain(self):
        """Should process pain input."""
        inp = SomatosensoryInput(
            pain_inputs=[
                PainInput(PainType.SHARP, BodyRegion.HAND_LEFT, 0.8, 0.9, 50.0)
            ]
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_somatosensory_input(inp)
            )
        finally:
            loop.close()
        self.assertGreater(len(output.pain_assessments), 0)
        self.assertTrue(output.pain_assessments[0].requires_attention)

    def test_protective_action_triggered(self):
        """Should trigger protective action for severe pain."""
        inp = SomatosensoryInput(
            pain_inputs=[
                PainInput(PainType.SHARP, BodyRegion.HAND_RIGHT, 0.9, 1.0, 10.0,
                         tissue_damage=0.7)
            ]
        )
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_somatosensory_input(inp)
            )
        finally:
            loop.close()
        self.assertTrue(output.requires_protective_action)

    def test_body_schema_normal(self):
        """Should maintain normal body schema with mild input."""
        inp = create_simple_touch_input(intensity=0.3)
        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_somatosensory_input(inp)
            )
        finally:
            loop.close()
        self.assertIsNotNone(output.body_schema)
        self.assertIn(output.body_schema.schema_state, [BodySchemaState.NORMAL, BodySchemaState.DIMINISHED])

    def test_comfort_decreases_with_pain(self):
        """Should show decreased comfort with pain."""
        # First process without pain
        no_pain = create_simple_touch_input(intensity=0.3)
        # Then with pain
        with_pain = SomatosensoryInput(
            pain_inputs=[
                PainInput(PainType.ACHING, BodyRegion.TORSO_BACK, 0.7, 0.5, 3000.0)
            ]
        )

        interface1 = create_somatosensory_interface()
        interface2 = create_somatosensory_interface()

        loop = asyncio.new_event_loop()
        try:
            out1 = loop.run_until_complete(
                interface1.process_somatosensory_input(no_pain)
            )
            out2 = loop.run_until_complete(
                interface2.process_somatosensory_input(with_pain)
            )
        finally:
            loop.close()

        self.assertGreater(out1.overall_comfort, out2.overall_comfort)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "03-somatosensory")
        self.assertIn("body_schema_state", d)

    def test_get_status(self):
        """Should return status dictionary."""
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "03-somatosensory")
        self.assertTrue(status["operational"])


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_somatosensory_interface(self):
        """Should create new interface."""
        interface = create_somatosensory_interface()
        self.assertIsInstance(interface, SomatosensoryConsciousnessInterface)

    def test_create_simple_touch_input_defaults(self):
        """Should create simple input with defaults."""
        inp = create_simple_touch_input()
        self.assertIsInstance(inp, SomatosensoryInput)
        self.assertEqual(len(inp.touch_inputs), 1)
        self.assertEqual(inp.touch_inputs[0].touch_type, TouchType.LIGHT_TOUCH)

    def test_create_simple_touch_input_custom(self):
        """Should create input with custom values."""
        inp = create_simple_touch_input(
            touch_type=TouchType.PRESSURE,
            body_region=BodyRegion.ARM_LEFT,
            intensity=0.8,
            temperature=0.7,
        )
        self.assertEqual(inp.touch_inputs[0].touch_type, TouchType.PRESSURE)
        self.assertEqual(inp.touch_inputs[0].body_region, BodyRegion.ARM_LEFT)
        self.assertEqual(inp.touch_inputs[0].intensity, 0.8)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete somatosensory pipeline."""

    def test_full_pipeline(self):
        """Should complete full somatosensory processing pipeline."""
        interface = create_somatosensory_interface()
        inp = SomatosensoryInput(
            touch_inputs=[
                TouchInput(TouchType.LIGHT_TOUCH, BodyRegion.HAND_RIGHT, 0.5, 0.1, 200.0),
                TouchInput(TouchType.TEXTURE, BodyRegion.HAND_LEFT, 0.3, 0.2, 300.0),
            ],
            pain_inputs=[
                PainInput(PainType.DULL, BodyRegion.TORSO_BACK, 0.3, 0.2, 2000.0),
            ],
            proprioceptive_inputs=[
                ProprioceptiveInput(ProprioceptiveChannel.BALANCE, BodyRegion.TORSO_FRONT, 0.8),
                ProprioceptiveInput(ProprioceptiveChannel.JOINT_POSITION, BodyRegion.ARM_RIGHT, 0.5),
            ],
            muscle_tension_global=0.3,
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            output = loop.run_until_complete(
                interface.process_somatosensory_input(inp)
            )
        finally:
            loop.close()

        self.assertIsInstance(output, SomatosensoryOutput)
        self.assertEqual(len(output.touch_classifications), 2)
        self.assertEqual(len(output.pain_assessments), 1)
        self.assertIsNotNone(output.body_schema)
        self.assertGreater(len(output.body_schema.region_activation), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
