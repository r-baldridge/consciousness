#!/usr/bin/env python3
"""
Test Suite for Form 23: Lucid Dream Consciousness.

Tests cover:
- All enumerations (LucidityLevel, DreamControl, LucidTrigger, DreamStability, DreamPhase)
- All input/output dataclasses
- LucidDreamInterface main interface
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
    LucidityLevel,
    DreamControl,
    LucidTrigger,
    DreamStability,
    DreamPhase,
    # Input dataclasses
    LucidDreamInput,
    DreamControlInput,
    # Output dataclasses
    LucidDreamOutput,
    DreamControlOutput,
    RealityCheckResult,
    DreamStateSnapshot,
    # Interface
    LucidDreamInterface,
    # Convenience
    create_lucid_dream_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestLucidityLevel(unittest.TestCase):
    """Tests for LucidityLevel enumeration."""

    def test_all_levels_exist(self):
        """All lucidity levels should be defined."""
        levels = [
            LucidityLevel.NON_LUCID,
            LucidityLevel.PRE_LUCID,
            LucidityLevel.SEMI_LUCID,
            LucidityLevel.FULLY_LUCID,
        ]
        self.assertEqual(len(levels), 4)

    def test_level_values(self):
        """Levels should have expected string values."""
        self.assertEqual(LucidityLevel.NON_LUCID.value, "non_lucid")
        self.assertEqual(LucidityLevel.PRE_LUCID.value, "pre_lucid")
        self.assertEqual(LucidityLevel.SEMI_LUCID.value, "semi_lucid")
        self.assertEqual(LucidityLevel.FULLY_LUCID.value, "fully_lucid")


class TestDreamControl(unittest.TestCase):
    """Tests for DreamControl enumeration."""

    def test_all_controls_exist(self):
        """All dream control levels should be defined."""
        controls = [
            DreamControl.NONE,
            DreamControl.PARTIAL,
            DreamControl.FULL,
        ]
        self.assertEqual(len(controls), 3)

    def test_control_values(self):
        """Controls should have expected string values."""
        self.assertEqual(DreamControl.NONE.value, "none")
        self.assertEqual(DreamControl.PARTIAL.value, "partial")
        self.assertEqual(DreamControl.FULL.value, "full")


class TestLucidTrigger(unittest.TestCase):
    """Tests for LucidTrigger enumeration."""

    def test_all_triggers_exist(self):
        """All lucid triggers should be defined."""
        triggers = [
            LucidTrigger.REALITY_CHECK,
            LucidTrigger.ANOMALY_RECOGNITION,
            LucidTrigger.WILD,
            LucidTrigger.MILD,
        ]
        self.assertEqual(len(triggers), 4)

    def test_trigger_values(self):
        """Triggers should have expected string values."""
        self.assertEqual(LucidTrigger.REALITY_CHECK.value, "reality_check")
        self.assertEqual(LucidTrigger.ANOMALY_RECOGNITION.value, "anomaly_recognition")
        self.assertEqual(LucidTrigger.WILD.value, "wild")
        self.assertEqual(LucidTrigger.MILD.value, "mild")


class TestDreamStability(unittest.TestCase):
    """Tests for DreamStability enumeration."""

    def test_all_states_exist(self):
        """All stability states should be defined."""
        states = [
            DreamStability.COLLAPSING,
            DreamStability.UNSTABLE,
            DreamStability.STABLE,
            DreamStability.VIVID,
        ]
        self.assertEqual(len(states), 4)


class TestDreamPhase(unittest.TestCase):
    """Tests for DreamPhase enumeration."""

    def test_all_phases_exist(self):
        """All dream phases should be defined."""
        phases = [
            DreamPhase.NREM_LIGHT,
            DreamPhase.NREM_DEEP,
            DreamPhase.REM_EARLY,
            DreamPhase.REM_LATE,
            DreamPhase.REM_EXTENDED,
        ]
        self.assertEqual(len(phases), 5)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestLucidDreamInput(unittest.TestCase):
    """Tests for LucidDreamInput dataclass."""

    def test_basic_creation(self):
        """Test basic input creation with required fields."""
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_LATE,
            awareness_signals=0.6,
            dream_vividness=0.8,
        )
        self.assertEqual(inp.dream_phase, DreamPhase.REM_LATE)
        self.assertAlmostEqual(inp.awareness_signals, 0.6)
        self.assertAlmostEqual(inp.dream_vividness, 0.8)

    def test_default_values(self):
        """Test default values are set correctly."""
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_EARLY,
            awareness_signals=0.3,
            dream_vividness=0.5,
        )
        self.assertEqual(inp.anomaly_count, 0)
        self.assertFalse(inp.reality_check_performed)
        self.assertAlmostEqual(inp.emotional_intensity, 0.5)
        self.assertAlmostEqual(inp.narrative_coherence, 0.5)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_LATE,
            awareness_signals=0.7,
            dream_vividness=0.9,
            anomaly_count=3,
        )
        d = inp.to_dict()
        self.assertEqual(d["dream_phase"], "rem_late")
        self.assertAlmostEqual(d["awareness_signals"], 0.7)
        self.assertEqual(d["anomaly_count"], 3)
        self.assertIn("timestamp", d)


class TestDreamControlInput(unittest.TestCase):
    """Tests for DreamControlInput dataclass."""

    def test_creation(self):
        """Test control input creation."""
        inp = DreamControlInput(
            target_element="environment",
            control_intention="change_scene",
            effort_level=0.6,
        )
        self.assertEqual(inp.target_element, "environment")
        self.assertEqual(inp.control_intention, "change_scene")

    def test_to_dict(self):
        """Test serialization."""
        inp = DreamControlInput(
            target_element="object",
            control_intention="create",
        )
        d = inp.to_dict()
        self.assertEqual(d["target_element"], "object")
        self.assertIn("timestamp", d)


class TestLucidDreamOutput(unittest.TestCase):
    """Tests for LucidDreamOutput dataclass."""

    def test_creation(self):
        """Test output creation with all fields."""
        output = LucidDreamOutput(
            lucidity_level=LucidityLevel.FULLY_LUCID,
            control_degree=DreamControl.PARTIAL,
            dream_stability=DreamStability.STABLE,
            lucidity_score=0.85,
            control_score=0.5,
            stability_score=0.6,
            trigger_detected=LucidTrigger.REALITY_CHECK,
        )
        self.assertEqual(output.lucidity_level, LucidityLevel.FULLY_LUCID)
        self.assertAlmostEqual(output.lucidity_score, 0.85)

    def test_to_dict(self):
        """Test output serialization."""
        output = LucidDreamOutput(
            lucidity_level=LucidityLevel.SEMI_LUCID,
            control_degree=DreamControl.NONE,
            dream_stability=DreamStability.UNSTABLE,
            lucidity_score=0.45,
            control_score=0.1,
            stability_score=0.3,
        )
        d = output.to_dict()
        self.assertEqual(d["lucidity_level"], "semi_lucid")
        self.assertEqual(d["control_degree"], "none")
        self.assertIsNone(d["trigger_detected"])


class TestRealityCheckResult(unittest.TestCase):
    """Tests for RealityCheckResult dataclass."""

    def test_creation(self):
        """Test reality check result creation."""
        result = RealityCheckResult(
            check_type="hand_check",
            result_anomalous=True,
            anomaly_description="Extra fingers visible",
            lucidity_boost=0.3,
        )
        self.assertTrue(result.result_anomalous)
        self.assertAlmostEqual(result.lucidity_boost, 0.3)

    def test_to_dict(self):
        """Test serialization."""
        result = RealityCheckResult(
            check_type="nose_pinch",
            result_anomalous=True,
        )
        d = result.to_dict()
        self.assertEqual(d["check_type"], "nose_pinch")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestLucidDreamInterface(unittest.TestCase):
    """Tests for LucidDreamInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.interface = LucidDreamInterface()
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
        self.assertEqual(self.interface.FORM_ID, "23-lucid-dream")
        self.assertEqual(self.interface.FORM_NAME, "Lucid Dream Consciousness")

    def test_initialization(self):
        """Test interface initializes correctly."""
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_double_initialization(self):
        """Test that double initialization is safe."""
        self._run(self.interface.initialize())
        self._run(self.interface.initialize())
        self.assertTrue(self.interface._initialized)

    def test_assess_lucidity_non_lucid(self):
        """Test assessment of a non-lucid dream state."""
        self._run(self.interface.initialize())
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_EARLY,
            awareness_signals=0.05,
            dream_vividness=0.4,
        )
        output = self._run(self.interface.assess_lucidity(inp))
        self.assertEqual(output.lucidity_level, LucidityLevel.NON_LUCID)
        self.assertLess(output.lucidity_score, 0.2)

    def test_assess_lucidity_fully_lucid(self):
        """Test assessment of a fully lucid dream state."""
        self._run(self.interface.initialize())
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_EXTENDED,
            awareness_signals=0.9,
            dream_vividness=0.9,
            anomaly_count=3,
            reality_check_performed=True,
            prior_lucid_count=5,
        )
        output = self._run(self.interface.assess_lucidity(inp))
        self.assertEqual(output.lucidity_level, LucidityLevel.FULLY_LUCID)
        self.assertGreater(output.lucidity_score, 0.7)
        self.assertIsNotNone(output.trigger_detected)

    def test_assess_lucidity_pre_lucid(self):
        """Test pre-lucid state detection."""
        self._run(self.interface.initialize())
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_LATE,
            awareness_signals=0.3,
            dream_vividness=0.6,
            anomaly_count=1,
        )
        output = self._run(self.interface.assess_lucidity(inp))
        self.assertIn(output.lucidity_level, [
            LucidityLevel.PRE_LUCID,
            LucidityLevel.SEMI_LUCID,
        ])

    def test_attempt_control_without_lucidity(self):
        """Test control attempt without established lucidity."""
        self._run(self.interface.initialize())
        control_input = DreamControlInput(
            target_element="sky",
            control_intention="change_color",
        )
        output = self._run(self.interface.attempt_control(control_input))
        self.assertFalse(output.success)
        self.assertEqual(output.control_degree_achieved, DreamControl.NONE)

    def test_attempt_control_with_lucidity(self):
        """Test control attempt after establishing lucidity."""
        self._run(self.interface.initialize())
        # First establish lucidity
        dream_input = LucidDreamInput(
            dream_phase=DreamPhase.REM_EXTENDED,
            awareness_signals=0.9,
            dream_vividness=0.9,
            reality_check_performed=True,
            prior_lucid_count=5,
        )
        self._run(self.interface.assess_lucidity(dream_input))

        # Then attempt control
        control_input = DreamControlInput(
            target_element="environment",
            control_intention="fly",
            effort_level=0.5,
            technique_used="expectation",
            confidence=0.8,
        )
        output = self._run(self.interface.attempt_control(control_input))
        self.assertIsInstance(output, DreamControlOutput)
        self.assertIsInstance(output.control_degree_achieved, DreamControl)

    def test_reality_check(self):
        """Test reality check method."""
        self._run(self.interface.initialize())
        result = self._run(self.interface.reality_check("hand_check"))
        self.assertIsInstance(result, RealityCheckResult)
        self.assertTrue(result.result_anomalous)
        self.assertEqual(result.check_type, "hand_check")
        self.assertGreater(result.lucidity_boost, 0.0)

    def test_reality_check_types(self):
        """Test multiple reality check types."""
        self._run(self.interface.initialize())
        check_types = ["hand_check", "text_check", "light_switch", "nose_pinch", "clock_check"]
        for check_type in check_types:
            result = self._run(self.interface.reality_check(check_type))
            self.assertTrue(result.result_anomalous)
            self.assertGreater(result.confidence, 0.0)

    def test_stabilize_dream(self):
        """Test dream stabilization."""
        self._run(self.interface.initialize())
        result = self._run(self.interface.stabilize_dream())
        self.assertIn("new_stability", result)
        self.assertIn("techniques_applied", result)
        self.assertIn("success", result)
        self.assertIsInstance(result["techniques_applied"], list)

    def test_get_dream_state(self):
        """Test dream state snapshot retrieval."""
        self._run(self.interface.initialize())
        state = self._run(self.interface.get_dream_state())
        self.assertIsInstance(state, DreamStateSnapshot)
        self.assertIsInstance(state.lucidity, LucidDreamOutput)
        self.assertIsInstance(state.phase, DreamPhase)

    def test_to_dict(self):
        """Test interface serialization."""
        self._run(self.interface.initialize())
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "23-lucid-dream")
        self.assertTrue(d["initialized"])

    def test_get_status(self):
        """Test status retrieval."""
        self._run(self.interface.initialize())
        status = self.interface.get_status()
        self.assertEqual(status["form_id"], "23-lucid-dream")
        self.assertTrue(status["initialized"])

    def test_auto_initialize_on_assess(self):
        """Test that assess_lucidity auto-initializes if needed."""
        inp = LucidDreamInput(
            dream_phase=DreamPhase.REM_LATE,
            awareness_signals=0.5,
            dream_vividness=0.6,
        )
        output = self._run(self.interface.assess_lucidity(inp))
        self.assertTrue(self.interface._initialized)
        self.assertIsInstance(output, LucidDreamOutput)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module convenience functions."""

    def test_create_lucid_dream_interface(self):
        """Test convenience creation function."""
        interface = create_lucid_dream_interface()
        self.assertIsInstance(interface, LucidDreamInterface)
        self.assertFalse(interface._initialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
