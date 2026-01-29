#!/usr/bin/env python3
"""
Test Suite for Form 24: Locked-In Consciousness.

Tests cover:
- All enumerations (LockedInType, CommunicationChannel, AwarenessState,
  SignalQuality, CognitiveFunction)
- All input/output dataclasses
- LockedInInterface main interface
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
    LockedInType,
    CommunicationChannel,
    AwarenessState,
    SignalQuality,
    CognitiveFunction,
    # Input dataclasses
    LockedInInput,
    CommunicationAttemptInput,
    # Output dataclasses
    DecodedIntention,
    AwarenessAssessment,
    CommunicationResult,
    ConsciousnessMonitorReading,
    # Interface
    LockedInInterface,
    # Convenience
    create_locked_in_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestLockedInType(unittest.TestCase):
    """Tests for LockedInType enumeration."""

    def test_all_types_exist(self):
        """All locked-in types should be defined."""
        types = [
            LockedInType.CLASSIC,
            LockedInType.INCOMPLETE,
            LockedInType.TOTAL,
        ]
        self.assertEqual(len(types), 3)

    def test_type_values(self):
        """Types should have expected string values."""
        self.assertEqual(LockedInType.CLASSIC.value, "classic")
        self.assertEqual(LockedInType.INCOMPLETE.value, "incomplete")
        self.assertEqual(LockedInType.TOTAL.value, "total")


class TestCommunicationChannel(unittest.TestCase):
    """Tests for CommunicationChannel enumeration."""

    def test_all_channels_exist(self):
        """All communication channels should be defined."""
        channels = [
            CommunicationChannel.EYE_MOVEMENT,
            CommunicationChannel.BRAIN_COMPUTER_INTERFACE,
            CommunicationChannel.MUSCLE_TWITCH,
            CommunicationChannel.PUPIL_DILATION,
            CommunicationChannel.RESPIRATORY,
        ]
        self.assertEqual(len(channels), 5)

    def test_channel_values(self):
        """Channels should have expected string values."""
        self.assertEqual(CommunicationChannel.EYE_MOVEMENT.value, "eye_movement")
        self.assertEqual(
            CommunicationChannel.BRAIN_COMPUTER_INTERFACE.value,
            "brain_computer_interface",
        )


class TestAwarenessState(unittest.TestCase):
    """Tests for AwarenessState enumeration."""

    def test_all_states_exist(self):
        """All awareness states should be defined."""
        states = [
            AwarenessState.UNRESPONSIVE,
            AwarenessState.POSSIBLE_AWARENESS,
            AwarenessState.MINIMAL_CONSCIOUSNESS,
            AwarenessState.FULL_AWARENESS,
        ]
        self.assertEqual(len(states), 4)

    def test_state_values(self):
        """States should have expected string values."""
        self.assertEqual(AwarenessState.UNRESPONSIVE.value, "unresponsive")
        self.assertEqual(AwarenessState.FULL_AWARENESS.value, "full_awareness")


class TestSignalQuality(unittest.TestCase):
    """Tests for SignalQuality enumeration."""

    def test_all_qualities_exist(self):
        """All signal quality levels should be defined."""
        qualities = [
            SignalQuality.NOISE,
            SignalQuality.POOR,
            SignalQuality.FAIR,
            SignalQuality.GOOD,
            SignalQuality.EXCELLENT,
        ]
        self.assertEqual(len(qualities), 5)


class TestCognitiveFunction(unittest.TestCase):
    """Tests for CognitiveFunction enumeration."""

    def test_all_functions_exist(self):
        """All cognitive functions should be defined."""
        functions = [
            CognitiveFunction.LANGUAGE_COMPREHENSION,
            CognitiveFunction.SPATIAL_REASONING,
            CognitiveFunction.EMOTIONAL_PROCESSING,
            CognitiveFunction.MEMORY_RETRIEVAL,
            CognitiveFunction.ATTENTION,
            CognitiveFunction.EXECUTIVE_FUNCTION,
        ]
        self.assertEqual(len(functions), 6)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestLockedInInput(unittest.TestCase):
    """Tests for LockedInInput dataclass."""

    def test_basic_creation(self):
        """Test basic input creation."""
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.7,
        )
        self.assertEqual(inp.locked_in_type, LockedInType.CLASSIC)
        self.assertAlmostEqual(inp.neural_signal_strength, 0.7)

    def test_default_values(self):
        """Test default values are set correctly."""
        inp = LockedInInput(
            locked_in_type=LockedInType.TOTAL,
            neural_signal_strength=0.5,
        )
        self.assertEqual(inp.stimulus_type, "auditory")
        self.assertEqual(inp.medication_state, "stable")
        self.assertEqual(inp.signal_quality, SignalQuality.FAIR)
        self.assertIsNone(inp.eye_tracking_data)

    def test_with_eye_tracking(self):
        """Test input with eye tracking data."""
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.6,
            eye_tracking_data={"gaze_x": 0.5, "gaze_y": 0.3, "blink": True},
        )
        self.assertIsNotNone(inp.eye_tracking_data)
        self.assertTrue(inp.eye_tracking_data["blink"])

    def test_to_dict(self):
        """Test serialization."""
        inp = LockedInInput(
            locked_in_type=LockedInType.INCOMPLETE,
            neural_signal_strength=0.8,
            signal_quality=SignalQuality.GOOD,
        )
        d = inp.to_dict()
        self.assertEqual(d["locked_in_type"], "incomplete")
        self.assertEqual(d["signal_quality"], "good")
        self.assertIn("timestamp", d)


class TestCommunicationAttemptInput(unittest.TestCase):
    """Tests for CommunicationAttemptInput dataclass."""

    def test_creation(self):
        """Test communication attempt input creation."""
        inp = CommunicationAttemptInput(
            channel=CommunicationChannel.EYE_MOVEMENT,
            message_to_patient="Can you hear me?",
        )
        self.assertEqual(inp.channel, CommunicationChannel.EYE_MOVEMENT)
        self.assertEqual(inp.expected_response_type, "yes_no")

    def test_to_dict(self):
        """Test serialization."""
        inp = CommunicationAttemptInput(
            channel=CommunicationChannel.BRAIN_COMPUTER_INTERFACE,
            message_to_patient="Select the letter A",
            expected_response_type="letter_selection",
        )
        d = inp.to_dict()
        self.assertEqual(d["channel"], "brain_computer_interface")


class TestDecodedIntention(unittest.TestCase):
    """Tests for DecodedIntention dataclass."""

    def test_creation(self):
        """Test decoded intention creation."""
        intention = DecodedIntention(
            intention_detected=True,
            decoded_content="affirmative",
            confidence=0.8,
            channel_used=CommunicationChannel.EYE_MOVEMENT,
            signal_quality=SignalQuality.GOOD,
        )
        self.assertTrue(intention.intention_detected)
        self.assertAlmostEqual(intention.confidence, 0.8)

    def test_to_dict(self):
        """Test serialization."""
        intention = DecodedIntention(
            intention_detected=False,
            decoded_content="no_signal",
            confidence=0.1,
            channel_used=CommunicationChannel.BRAIN_COMPUTER_INTERFACE,
            signal_quality=SignalQuality.NOISE,
        )
        d = intention.to_dict()
        self.assertEqual(d["signal_quality"], "noise")
        self.assertFalse(d["intention_detected"])


class TestAwarenessAssessment(unittest.TestCase):
    """Tests for AwarenessAssessment dataclass."""

    def test_creation(self):
        """Test awareness assessment creation."""
        assessment = AwarenessAssessment(
            awareness_state=AwarenessState.FULL_AWARENESS,
            confidence=0.9,
            responsive_channels=[CommunicationChannel.EYE_MOVEMENT],
        )
        self.assertEqual(assessment.awareness_state, AwarenessState.FULL_AWARENESS)
        self.assertTrue(assessment.reassessment_recommended)

    def test_to_dict(self):
        """Test serialization."""
        assessment = AwarenessAssessment(
            awareness_state=AwarenessState.POSSIBLE_AWARENESS,
            confidence=0.4,
        )
        d = assessment.to_dict()
        self.assertEqual(d["awareness_state"], "possible_awareness")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestLockedInInterface(unittest.TestCase):
    """Tests for LockedInInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.interface = LockedInInterface()
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
        self.assertEqual(self.interface.FORM_ID, "24-locked-in")
        self.assertEqual(self.interface.FORM_NAME, "Locked-In Consciousness")

    def test_initialization(self):
        """Test interface initializes correctly."""
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_double_initialization(self):
        """Test that double initialization is safe."""
        self._run(self.interface.initialize())
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_decode_intention_classic(self):
        """Test intention decoding for classic locked-in."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.7,
            eye_tracking_data={"gaze_direction": "up", "blink_count": 2},
        )
        result = self._run(self.interface.decode_intention(inp))
        self.assertIsInstance(result, DecodedIntention)
        self.assertTrue(result.intention_detected)
        self.assertEqual(result.channel_used, CommunicationChannel.EYE_MOVEMENT)

    def test_decode_intention_total(self):
        """Test intention decoding for total locked-in."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.TOTAL,
            neural_signal_strength=0.6,
            bci_signal={"motor_imagery": 0.7, "attention": 0.5},
        )
        result = self._run(self.interface.decode_intention(inp))
        self.assertIsInstance(result, DecodedIntention)
        self.assertEqual(
            result.channel_used, CommunicationChannel.BRAIN_COMPUTER_INTERFACE
        )

    def test_decode_intention_weak_signal(self):
        """Test intention decoding with weak signals."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.TOTAL,
            neural_signal_strength=0.1,
            signal_quality=SignalQuality.NOISE,
        )
        result = self._run(self.interface.decode_intention(inp))
        self.assertLess(result.confidence, 0.5)

    def test_assess_awareness_high_signal(self):
        """Test awareness assessment with high signals."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.9,
            eye_tracking_data={"responsive": True},
            bci_signal={"p300": 0.8},
            signal_quality=SignalQuality.EXCELLENT,
            medication_state="alert",
        )
        result = self._run(self.interface.assess_awareness(inp))
        self.assertIsInstance(result, AwarenessAssessment)
        self.assertEqual(result.awareness_state, AwarenessState.FULL_AWARENESS)

    def test_assess_awareness_low_signal(self):
        """Test awareness assessment with low signals."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.TOTAL,
            neural_signal_strength=0.05,
            signal_quality=SignalQuality.NOISE,
            medication_state="sedated",
        )
        result = self._run(self.interface.assess_awareness(inp))
        self.assertIn(result.awareness_state, [
            AwarenessState.UNRESPONSIVE,
            AwarenessState.POSSIBLE_AWARENESS,
        ])
        self.assertGreater(result.false_negative_risk, 0.0)

    def test_assess_awareness_generates_notes(self):
        """Test that assessment generates clinical notes."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.TOTAL,
            neural_signal_strength=0.3,
            signal_quality=SignalQuality.POOR,
            medication_state="sedated",
        )
        result = self._run(self.interface.assess_awareness(inp))
        self.assertIsInstance(result.assessment_notes, list)
        self.assertGreater(len(result.assessment_notes), 0)

    def test_attempt_communication_yes_no(self):
        """Test yes/no communication attempt."""
        self._run(self.interface.initialize())
        comm = CommunicationAttemptInput(
            channel=CommunicationChannel.EYE_MOVEMENT,
            message_to_patient="Are you in pain?",
            expected_response_type="yes_no",
        )
        result = self._run(self.interface.attempt_communication(comm))
        self.assertIsInstance(result, CommunicationResult)
        self.assertIn(result.decoded_response, ["yes", "no"])

    def test_attempt_communication_letter_selection(self):
        """Test letter selection communication."""
        self._run(self.interface.initialize())
        comm = CommunicationAttemptInput(
            channel=CommunicationChannel.BRAIN_COMPUTER_INTERFACE,
            message_to_patient="Select the first letter of your name",
            expected_response_type="letter_selection",
        )
        result = self._run(self.interface.attempt_communication(comm))
        self.assertIsInstance(result, CommunicationResult)

    def test_monitor_consciousness(self):
        """Test continuous consciousness monitoring."""
        self._run(self.interface.initialize())
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.6,
            eye_tracking_data={"responsive": True},
        )
        reading = self._run(self.interface.monitor_consciousness(inp))
        self.assertIsInstance(reading, ConsciousnessMonitorReading)
        self.assertIsInstance(reading.awareness_state, AwarenessState)
        self.assertGreaterEqual(reading.awareness_level, 0.0)
        self.assertLessEqual(reading.awareness_level, 1.0)

    def test_monitor_multiple_readings(self):
        """Test multiple monitoring readings build history."""
        self._run(self.interface.initialize())
        for i in range(5):
            inp = LockedInInput(
                locked_in_type=LockedInType.CLASSIC,
                neural_signal_strength=0.5 + i * 0.05,
            )
            self._run(self.interface.monitor_consciousness(inp))
        self.assertEqual(len(self.interface._monitor_readings), 5)

    def test_to_dict(self):
        """Test interface serialization."""
        self._run(self.interface.initialize())
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "24-locked-in")
        self.assertTrue(d["initialized"])
        self.assertIn("patient_profile", d)

    def test_get_status(self):
        """Test status retrieval."""
        self._run(self.interface.initialize())
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "24-locked-in")
        self.assertEqual(status["current_awareness"], "unknown")

    def test_auto_initialize(self):
        """Test that methods auto-initialize if needed."""
        inp = LockedInInput(
            locked_in_type=LockedInType.CLASSIC,
            neural_signal_strength=0.5,
        )
        result = self._run(self.interface.decode_intention(inp))
        self.assertTrue(self.interface._initialized)
        self.assertIsInstance(result, DecodedIntention)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module convenience functions."""

    def test_create_locked_in_interface(self):
        """Test convenience creation function."""
        interface = create_locked_in_interface()
        self.assertIsInstance(interface, LockedInInterface)
        self.assertFalse(interface._initialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
