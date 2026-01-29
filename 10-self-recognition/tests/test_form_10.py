#!/usr/bin/env python3
"""
Test Suite for Form 10: Self-Recognition Consciousness.

Tests cover:
- All enumerations (SelfAspect, AgencyLevel, OwnershipState, etc.)
- All input/output dataclasses
- SelfModelEngine
- AgencyEngine
- SelfRecognitionInterface (main interface)
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
    SelfAspect,
    AgencyLevel,
    OwnershipState,
    SelfBoundaryType,
    SelfRecognitionMode,
    # Input dataclasses
    BodySignalInput,
    SocialContextInput,
    ActionFeedback,
    SelfInput,
    # Output dataclasses
    SelfModelOutput,
    AgencyAssessment,
    SelfOutput,
    SelfSystemStatus,
    # Engines
    SelfModelEngine,
    AgencyEngine,
    # Main interface
    SelfRecognitionInterface,
    # Convenience
    create_self_recognition_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestSelfAspect(unittest.TestCase):
    """Tests for SelfAspect enumeration."""

    def test_all_aspects_exist(self):
        """All self-aspects should be defined."""
        aspects = [
            SelfAspect.BODILY,
            SelfAspect.MINIMAL,
            SelfAspect.NARRATIVE,
            SelfAspect.SOCIAL,
            SelfAspect.EXPERIENTIAL,
        ]
        self.assertEqual(len(aspects), 5)

    def test_aspect_values(self):
        """Aspects should have expected string values."""
        self.assertEqual(SelfAspect.BODILY.value, "bodily")
        self.assertEqual(SelfAspect.MINIMAL.value, "minimal")
        self.assertEqual(SelfAspect.NARRATIVE.value, "narrative")


class TestAgencyLevel(unittest.TestCase):
    """Tests for AgencyLevel enumeration."""

    def test_all_levels_exist(self):
        """All agency levels should be defined."""
        levels = [
            AgencyLevel.FULL,
            AgencyLevel.PARTIAL,
            AgencyLevel.DIMINISHED,
            AgencyLevel.ABSENT,
            AgencyLevel.INVOLUNTARY,
        ]
        self.assertEqual(len(levels), 5)

    def test_level_values(self):
        """Levels should have expected string values."""
        self.assertEqual(AgencyLevel.FULL.value, "full")
        self.assertEqual(AgencyLevel.INVOLUNTARY.value, "involuntary")


class TestOwnershipState(unittest.TestCase):
    """Tests for OwnershipState enumeration."""

    def test_all_states_exist(self):
        """All ownership states should be defined."""
        states = [
            OwnershipState.OWNED,
            OwnershipState.UNCERTAIN,
            OwnershipState.DISOWNED,
            OwnershipState.EXTENDED,
            OwnershipState.VIRTUAL,
        ]
        self.assertEqual(len(states), 5)


class TestSelfBoundaryType(unittest.TestCase):
    """Tests for SelfBoundaryType enumeration."""

    def test_all_types_exist(self):
        """All boundary types should be defined."""
        types = [
            SelfBoundaryType.PHYSICAL,
            SelfBoundaryType.PSYCHOLOGICAL,
            SelfBoundaryType.SOCIAL,
            SelfBoundaryType.TEMPORAL,
        ]
        self.assertEqual(len(types), 4)


class TestSelfRecognitionMode(unittest.TestCase):
    """Tests for SelfRecognitionMode enumeration."""

    def test_all_modes_exist(self):
        """All recognition modes should be defined."""
        modes = [
            SelfRecognitionMode.MIRROR,
            SelfRecognitionMode.PROPRIOCEPTIVE,
            SelfRecognitionMode.INTEROCEPTIVE,
            SelfRecognitionMode.SOCIAL,
            SelfRecognitionMode.REFLECTIVE,
        ]
        self.assertEqual(len(modes), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestBodySignalInput(unittest.TestCase):
    """Tests for BodySignalInput dataclass."""

    def test_creation(self):
        """Should create body signal input."""
        signals = BodySignalInput(
            proprioceptive_coherence=0.9,
            interoceptive_intensity=0.5,
            vestibular_stability=0.8,
            pain_level=0.1,
            body_temperature=0.5,
            heartbeat_awareness=0.6,
        )
        self.assertEqual(signals.proprioceptive_coherence, 0.9)
        self.assertEqual(signals.pain_level, 0.1)

    def test_to_dict(self):
        """Should convert to dictionary."""
        signals = BodySignalInput(
            proprioceptive_coherence=0.8,
            interoceptive_intensity=0.4,
            vestibular_stability=0.7,
            pain_level=0.0,
            body_temperature=0.5,
            heartbeat_awareness=0.5,
        )
        d = signals.to_dict()
        self.assertIn("proprioceptive_coherence", d)
        self.assertIn("timestamp", d)


class TestSocialContextInput(unittest.TestCase):
    """Tests for SocialContextInput dataclass."""

    def test_creation(self):
        """Should create social context input."""
        ctx = SocialContextInput(
            social_presence=True,
            being_observed=True,
            social_role="peer",
            empathy_activation=0.6,
            social_evaluation_threat=0.3,
            perspective_taking=0.5,
        )
        self.assertTrue(ctx.social_presence)
        self.assertEqual(ctx.social_role, "peer")


class TestActionFeedback(unittest.TestCase):
    """Tests for ActionFeedback dataclass."""

    def test_creation(self):
        """Should create action feedback."""
        feedback = ActionFeedback(
            action_id="act_001",
            intended=True,
            predicted_outcome="button_press",
            actual_outcome="button_press",
            outcome_match=0.95,
            effort_level=0.3,
            timing_accuracy=0.9,
        )
        self.assertEqual(feedback.action_id, "act_001")
        self.assertTrue(feedback.intended)
        self.assertEqual(feedback.outcome_match, 0.95)

    def test_unintended_action(self):
        """Should support unintended actions."""
        feedback = ActionFeedback(
            action_id="act_002",
            intended=False,
            predicted_outcome="none",
            actual_outcome="flinch",
            outcome_match=0.1,
            effort_level=0.0,
            timing_accuracy=0.2,
        )
        self.assertFalse(feedback.intended)


class TestSelfInput(unittest.TestCase):
    """Tests for SelfInput dataclass."""

    def test_empty_input(self):
        """Should create empty input with defaults."""
        inp = SelfInput()
        self.assertIsNone(inp.body_signals)
        self.assertIsNone(inp.social_context)
        self.assertEqual(inp.memory_continuity, 0.8)

    def test_full_input(self):
        """Should create input with all components."""
        inp = SelfInput(
            body_signals=BodySignalInput(
                proprioceptive_coherence=0.9,
                interoceptive_intensity=0.5,
                vestibular_stability=0.8,
                pain_level=0.0,
                body_temperature=0.5,
                heartbeat_awareness=0.6,
            ),
            social_context=SocialContextInput(
                social_presence=True,
                being_observed=False,
                social_role="leader",
                empathy_activation=0.4,
                social_evaluation_threat=0.2,
                perspective_taking=0.6,
            ),
            visual_self_input=True,
            memory_continuity=0.9,
        )
        self.assertIsNotNone(inp.body_signals)
        self.assertIsNotNone(inp.social_context)
        self.assertTrue(inp.visual_self_input)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestSelfModelOutput(unittest.TestCase):
    """Tests for SelfModelOutput dataclass."""

    def test_creation(self):
        """Should create self-model output."""
        model = SelfModelOutput(
            active_aspects=[SelfAspect.MINIMAL, SelfAspect.BODILY],
            body_ownership=OwnershipState.OWNED,
            self_coherence=0.85,
            self_distinctness=0.8,
            embodiment_level=0.75,
            self_continuity=0.9,
        )
        self.assertEqual(len(model.active_aspects), 2)
        self.assertEqual(model.body_ownership, OwnershipState.OWNED)

    def test_to_dict(self):
        """Should convert to dictionary."""
        model = SelfModelOutput(
            active_aspects=[SelfAspect.MINIMAL],
            body_ownership=OwnershipState.OWNED,
            self_coherence=0.8,
            self_distinctness=0.7,
            embodiment_level=0.6,
            self_continuity=0.8,
        )
        d = model.to_dict()
        self.assertIn("minimal", d["active_aspects"])
        self.assertEqual(d["body_ownership"], "owned")


class TestAgencyAssessment(unittest.TestCase):
    """Tests for AgencyAssessment dataclass."""

    def test_creation(self):
        """Should create agency assessment."""
        agency = AgencyAssessment(
            agency_level=AgencyLevel.FULL,
            agency_score=0.9,
            prediction_accuracy=0.95,
            control_feeling=0.85,
            authorship_confidence=0.9,
        )
        self.assertEqual(agency.agency_level, AgencyLevel.FULL)
        self.assertEqual(agency.agency_score, 0.9)

    def test_to_dict(self):
        """Should convert to dictionary."""
        agency = AgencyAssessment(
            agency_level=AgencyLevel.PARTIAL,
            agency_score=0.6,
            prediction_accuracy=0.5,
            control_feeling=0.5,
            authorship_confidence=0.5,
        )
        d = agency.to_dict()
        self.assertEqual(d["agency_level"], "partial")


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestSelfModelEngine(unittest.TestCase):
    """Tests for SelfModelEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = SelfModelEngine()

    def test_update_with_body_signals(self):
        """Should update self-model from body signals."""
        inp = SelfInput(
            body_signals=BodySignalInput(
                proprioceptive_coherence=0.9,
                interoceptive_intensity=0.5,
                vestibular_stability=0.8,
                pain_level=0.1,
                body_temperature=0.5,
                heartbeat_awareness=0.6,
            ),
        )
        model = self.engine.update_self_model(inp)
        self.assertIsInstance(model, SelfModelOutput)
        self.assertIn(SelfAspect.BODILY, model.active_aspects)
        self.assertEqual(model.body_ownership, OwnershipState.OWNED)

    def test_update_with_social_context(self):
        """Should activate social self with social input."""
        inp = SelfInput(
            social_context=SocialContextInput(
                social_presence=True,
                being_observed=True,
                social_role="peer",
                empathy_activation=0.5,
                social_evaluation_threat=0.3,
                perspective_taking=0.4,
            ),
        )
        model = self.engine.update_self_model(inp)
        self.assertIn(SelfAspect.SOCIAL, model.active_aspects)

    def test_minimal_always_active(self):
        """Minimal self should always be active."""
        inp = SelfInput()
        model = self.engine.update_self_model(inp)
        self.assertIn(SelfAspect.MINIMAL, model.active_aspects)

    def test_narrative_with_high_continuity(self):
        """Narrative self should activate with high memory continuity."""
        inp = SelfInput(memory_continuity=0.9)
        model = self.engine.update_self_model(inp)
        self.assertIn(SelfAspect.NARRATIVE, model.active_aspects)

    def test_disowned_body_low_coherence(self):
        """Should report disowned body with low coherence."""
        inp = SelfInput(
            body_signals=BodySignalInput(
                proprioceptive_coherence=0.1,
                interoceptive_intensity=0.2,
                vestibular_stability=0.1,
                pain_level=0.8,
                body_temperature=0.5,
                heartbeat_awareness=0.1,
            ),
        )
        model = self.engine.update_self_model(inp)
        self.assertEqual(model.body_ownership, OwnershipState.DISOWNED)


class TestAgencyEngine(unittest.TestCase):
    """Tests for AgencyEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = AgencyEngine()

    def test_full_agency(self):
        """Should detect full agency with matching predictions."""
        feedback = ActionFeedback(
            action_id="a1",
            intended=True,
            predicted_outcome="click",
            actual_outcome="click",
            outcome_match=0.95,
            effort_level=0.5,
            timing_accuracy=0.9,
        )
        agency = self.engine.assess_agency(feedback)
        self.assertEqual(agency.agency_level, AgencyLevel.FULL)
        self.assertGreater(agency.agency_score, 0.7)

    def test_involuntary_action(self):
        """Should detect involuntary action."""
        feedback = ActionFeedback(
            action_id="a2",
            intended=False,
            predicted_outcome="none",
            actual_outcome="reflex",
            outcome_match=0.1,
            effort_level=0.0,
            timing_accuracy=0.2,
        )
        agency = self.engine.assess_agency(feedback)
        self.assertEqual(agency.agency_level, AgencyLevel.INVOLUNTARY)

    def test_no_feedback(self):
        """Should return default with no feedback."""
        agency = self.engine.assess_agency(None)
        self.assertEqual(agency.agency_level, AgencyLevel.PARTIAL)
        self.assertEqual(agency.agency_score, 0.5)

    def test_diminished_agency(self):
        """Should detect diminished agency with poor prediction match."""
        feedback = ActionFeedback(
            action_id="a3",
            intended=True,
            predicted_outcome="grab",
            actual_outcome="miss",
            outcome_match=0.2,
            effort_level=0.3,
            timing_accuracy=0.3,
        )
        agency = self.engine.assess_agency(feedback)
        self.assertIn(agency.agency_level, [AgencyLevel.DIMINISHED, AgencyLevel.PARTIAL])


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestSelfRecognitionInterface(unittest.TestCase):
    """Tests for SelfRecognitionInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = SelfRecognitionInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "10-self-recognition")
        self.assertEqual(self.interface.FORM_NAME, "Self-Recognition Consciousness")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_self_recognition(self):
        """Should process self-recognition input."""
        inp = SelfInput(
            body_signals=BodySignalInput(
                proprioceptive_coherence=0.9,
                interoceptive_intensity=0.5,
                vestibular_stability=0.85,
                pain_level=0.0,
                body_temperature=0.5,
                heartbeat_awareness=0.7,
            ),
            action_feedback=ActionFeedback(
                action_id="a1",
                intended=True,
                predicted_outcome="reach",
                actual_outcome="reach",
                outcome_match=0.9,
                effort_level=0.4,
                timing_accuracy=0.85,
            ),
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_self_recognition(inp)
            )
        finally:
            loop.close()

        self.assertIsInstance(output, SelfOutput)
        self.assertIsNotNone(output.self_model)
        self.assertIsNotNone(output.agency)
        self.assertTrue(output.mirror_recognition)

    def test_mirror_test(self):
        """Should perform mirror test."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.interface.perform_mirror_test())
        finally:
            loop.close()
        self.assertTrue(result)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, SelfSystemStatus)
        self.assertGreaterEqual(status.system_health, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "10-self-recognition")
        self.assertEqual(d["form_name"], "Self-Recognition Consciousness")
        self.assertTrue(d["mirror_capable"])

    def test_recognition_mode_visual(self):
        """Should set mirror mode with visual self-input."""
        inp = SelfInput(visual_self_input=True)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_self_recognition(inp))
        finally:
            loop.close()
        self.assertEqual(self.interface._recognition_mode, SelfRecognitionMode.MIRROR)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_self_recognition_interface(self):
        """Should create new interface."""
        interface = create_self_recognition_interface()
        self.assertIsInstance(interface, SelfRecognitionInterface)
        self.assertEqual(interface.FORM_ID, "10-self-recognition")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
