#!/usr/bin/env python3
"""
Test Suite for Form 07: Emotional Consciousness.

Tests cover:
- All enumerations (EmotionCategory, EmotionalValence, AffectiveState, MoodState, EmotionalRegulationStrategy)
- All input/output dataclasses
- EmotionProcessingEngine
- MoodTrackingEngine
- EmotionRegulationEngine
- EmotionalConsciousnessInterface (main interface)
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
    EmotionCategory,
    EmotionalValence,
    AffectiveState,
    MoodState,
    EmotionalRegulationStrategy,
    # Input dataclasses
    EmotionalStimulus,
    AppraisalInput,
    BodilySignalInput,
    EmotionalInput,
    # Output dataclasses
    EmotionIdentification,
    EmotionalOutput,
    MoodReport,
    EmotionalSystemStatus,
    # Engines
    EmotionProcessingEngine,
    MoodTrackingEngine,
    EmotionRegulationEngine,
    # Main interface
    EmotionalConsciousnessInterface,
    # Convenience
    create_emotional_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEmotionCategory(unittest.TestCase):
    """Tests for EmotionCategory enumeration."""

    def test_all_categories_exist(self):
        """All primary emotion categories should be defined."""
        categories = [
            EmotionCategory.JOY,
            EmotionCategory.SADNESS,
            EmotionCategory.ANGER,
            EmotionCategory.FEAR,
            EmotionCategory.DISGUST,
            EmotionCategory.SURPRISE,
            EmotionCategory.CONTEMPT,
            EmotionCategory.TRUST,
            EmotionCategory.ANTICIPATION,
        ]
        self.assertEqual(len(categories), 9)

    def test_category_values(self):
        """Categories should have expected string values."""
        self.assertEqual(EmotionCategory.JOY.value, "joy")
        self.assertEqual(EmotionCategory.FEAR.value, "fear")
        self.assertEqual(EmotionCategory.CONTEMPT.value, "contempt")


class TestEmotionalValence(unittest.TestCase):
    """Tests for EmotionalValence enumeration."""

    def test_all_valences_exist(self):
        """All valence categories should be defined."""
        valences = [
            EmotionalValence.VERY_NEGATIVE,
            EmotionalValence.NEGATIVE,
            EmotionalValence.NEUTRAL,
            EmotionalValence.POSITIVE,
            EmotionalValence.VERY_POSITIVE,
        ]
        self.assertEqual(len(valences), 5)

    def test_valence_values(self):
        """Valence categories should have expected string values."""
        self.assertEqual(EmotionalValence.NEUTRAL.value, "neutral")
        self.assertEqual(EmotionalValence.VERY_POSITIVE.value, "very_positive")


class TestAffectiveState(unittest.TestCase):
    """Tests for AffectiveState enumeration."""

    def test_all_states_exist(self):
        """All affective states should be defined."""
        states = [
            AffectiveState.EXCITED,
            AffectiveState.HAPPY,
            AffectiveState.CONTENT,
            AffectiveState.RELAXED,
            AffectiveState.BORED,
            AffectiveState.SAD,
            AffectiveState.DISTRESSED,
            AffectiveState.ALERT,
        ]
        self.assertEqual(len(states), 8)


class TestMoodState(unittest.TestCase):
    """Tests for MoodState enumeration."""

    def test_all_moods_exist(self):
        """All mood states should be defined."""
        moods = [
            MoodState.ELEVATED,
            MoodState.EUTHYMIC,
            MoodState.IRRITABLE,
            MoodState.ANXIOUS,
            MoodState.DEPRESSED,
            MoodState.APATHETIC,
            MoodState.EUPHORIC,
        ]
        self.assertEqual(len(moods), 7)

    def test_euthymic_value(self):
        """Euthymic mood should have expected value."""
        self.assertEqual(MoodState.EUTHYMIC.value, "euthymic")


class TestEmotionalRegulationStrategy(unittest.TestCase):
    """Tests for EmotionalRegulationStrategy enumeration."""

    def test_all_strategies_exist(self):
        """All regulation strategies should be defined."""
        strategies = [
            EmotionalRegulationStrategy.REAPPRAISAL,
            EmotionalRegulationStrategy.SUPPRESSION,
            EmotionalRegulationStrategy.DISTRACTION,
            EmotionalRegulationStrategy.ACCEPTANCE,
            EmotionalRegulationStrategy.PROBLEM_SOLVING,
            EmotionalRegulationStrategy.SITUATION_SELECTION,
        ]
        self.assertEqual(len(strategies), 6)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestEmotionalStimulus(unittest.TestCase):
    """Tests for EmotionalStimulus dataclass."""

    def test_creation(self):
        """Should create stimulus with all fields."""
        stim = EmotionalStimulus(
            stimulus_id="stim_001",
            stimulus_type="event",
            content_description="Unexpected good news",
            intensity=0.8,
            novelty=0.9,
            personal_relevance=0.7,
        )
        self.assertEqual(stim.stimulus_id, "stim_001")
        self.assertEqual(stim.stimulus_type, "event")
        self.assertEqual(stim.intensity, 0.8)
        self.assertEqual(stim.source, "external")

    def test_to_dict(self):
        """Should convert to dictionary."""
        stim = EmotionalStimulus(
            stimulus_id="stim_002",
            stimulus_type="memory",
            content_description="Past loss",
            intensity=0.6,
            novelty=0.2,
            personal_relevance=0.9,
            source="internal",
        )
        d = stim.to_dict()
        self.assertEqual(d["stimulus_id"], "stim_002")
        self.assertEqual(d["source"], "internal")


class TestAppraisalInput(unittest.TestCase):
    """Tests for AppraisalInput dataclass."""

    def test_creation(self):
        """Should create appraisal input."""
        appraisal = AppraisalInput(
            stimulus_id="stim_001",
            goal_relevance=0.8,
            goal_congruence=0.6,
            coping_potential=0.7,
            norm_compatibility=0.9,
            certainty=0.5,
        )
        self.assertEqual(appraisal.goal_relevance, 0.8)
        self.assertEqual(appraisal.agency, "self")

    def test_other_agency(self):
        """Should accept other agency values."""
        appraisal = AppraisalInput(
            stimulus_id="stim_003",
            goal_relevance=0.6,
            goal_congruence=-0.5,
            coping_potential=0.3,
            norm_compatibility=0.4,
            certainty=0.6,
            agency="other",
        )
        self.assertEqual(appraisal.agency, "other")


class TestBodilySignalInput(unittest.TestCase):
    """Tests for BodilySignalInput dataclass."""

    def test_creation(self):
        """Should create bodily signal input."""
        signals = BodilySignalInput(
            heart_rate_change=0.5,
            skin_conductance=0.7,
            muscle_tension=0.6,
            breathing_rate_change=0.3,
            gut_feeling=-0.2,
            temperature_change=0.1,
        )
        self.assertEqual(signals.heart_rate_change, 0.5)
        self.assertEqual(signals.gut_feeling, -0.2)


class TestEmotionalInput(unittest.TestCase):
    """Tests for EmotionalInput dataclass."""

    def test_empty_input(self):
        """Should create empty input."""
        inp = EmotionalInput()
        self.assertIsNone(inp.stimulus)
        self.assertIsNone(inp.appraisal)
        self.assertIsNone(inp.bodily_signals)

    def test_full_input(self):
        """Should create full input with all components."""
        inp = EmotionalInput(
            stimulus=EmotionalStimulus(
                stimulus_id="s1",
                stimulus_type="event",
                content_description="test",
                intensity=0.5,
                novelty=0.5,
                personal_relevance=0.5,
            ),
            appraisal=AppraisalInput(
                stimulus_id="s1",
                goal_relevance=0.6,
                goal_congruence=0.4,
                coping_potential=0.7,
                norm_compatibility=0.8,
                certainty=0.5,
            ),
            social_context="group",
        )
        self.assertIsNotNone(inp.stimulus)
        self.assertIsNotNone(inp.appraisal)
        self.assertEqual(inp.social_context, "group")


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestEmotionIdentification(unittest.TestCase):
    """Tests for EmotionIdentification dataclass."""

    def test_creation(self):
        """Should create emotion identification."""
        eid = EmotionIdentification(
            category=EmotionCategory.JOY,
            intensity=0.7,
            confidence=0.85,
        )
        self.assertEqual(eid.category, EmotionCategory.JOY)
        self.assertEqual(eid.intensity, 0.7)
        self.assertEqual(len(eid.secondary_emotions), 0)

    def test_with_secondary_emotions(self):
        """Should include secondary emotions."""
        eid = EmotionIdentification(
            category=EmotionCategory.ANGER,
            intensity=0.8,
            confidence=0.9,
            secondary_emotions=[
                (EmotionCategory.DISGUST, 0.4),
                (EmotionCategory.CONTEMPT, 0.3),
            ],
        )
        self.assertEqual(len(eid.secondary_emotions), 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        eid = EmotionIdentification(
            category=EmotionCategory.FEAR,
            intensity=0.9,
            confidence=0.8,
        )
        d = eid.to_dict()
        self.assertEqual(d["category"], "fear")
        self.assertIn("intensity", d)


class TestEmotionalOutput(unittest.TestCase):
    """Tests for EmotionalOutput dataclass."""

    def test_creation(self):
        """Should create full emotional output."""
        output = EmotionalOutput(
            emotion=EmotionIdentification(
                category=EmotionCategory.JOY,
                intensity=0.7,
                confidence=0.8,
            ),
            valence=0.6,
            valence_category=EmotionalValence.POSITIVE,
            arousal=0.65,
            affective_state=AffectiveState.HAPPY,
            action_tendency="approach",
        )
        self.assertEqual(output.valence, 0.6)
        self.assertEqual(output.action_tendency, "approach")

    def test_to_dict(self):
        """Should convert to dictionary."""
        output = EmotionalOutput(
            emotion=EmotionIdentification(
                category=EmotionCategory.SADNESS,
                intensity=0.5,
                confidence=0.7,
            ),
            valence=-0.4,
            valence_category=EmotionalValence.NEGATIVE,
            arousal=0.3,
            affective_state=AffectiveState.SAD,
            action_tendency="withdraw",
        )
        d = output.to_dict()
        self.assertEqual(d["affective_state"], "sad")
        self.assertEqual(d["action_tendency"], "withdraw")


class TestMoodReport(unittest.TestCase):
    """Tests for MoodReport dataclass."""

    def test_creation(self):
        """Should create mood report."""
        report = MoodReport(
            mood_state=MoodState.EUTHYMIC,
            stability=0.8,
            duration_minutes=30.0,
            dominant_valence=0.1,
        )
        self.assertEqual(report.mood_state, MoodState.EUTHYMIC)
        self.assertEqual(report.stability, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = MoodReport(
            mood_state=MoodState.ELEVATED,
            stability=0.9,
            duration_minutes=60.0,
            dominant_valence=0.5,
            contributing_emotions=[EmotionCategory.JOY, EmotionCategory.TRUST],
        )
        d = report.to_dict()
        self.assertEqual(d["mood_state"], "elevated")
        self.assertEqual(len(d["contributing_emotions"]), 2)


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestEmotionProcessingEngine(unittest.TestCase):
    """Tests for EmotionProcessingEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = EmotionProcessingEngine()

    def test_detect_emotion_from_appraisal(self):
        """Should detect emotion from appraisal input."""
        inp = EmotionalInput(
            appraisal=AppraisalInput(
                stimulus_id="s1",
                goal_relevance=0.8,
                goal_congruence=0.7,
                coping_potential=0.8,
                norm_compatibility=0.9,
                certainty=0.6,
            ),
        )
        emotion = self.engine.detect_emotion(inp)
        self.assertIsInstance(emotion, EmotionIdentification)
        self.assertGreater(emotion.intensity, 0.0)

    def test_detect_emotion_from_body_signals(self):
        """Should detect emotion from bodily signals."""
        inp = EmotionalInput(
            bodily_signals=BodilySignalInput(
                heart_rate_change=0.8,
                skin_conductance=0.9,
                muscle_tension=0.7,
                breathing_rate_change=0.6,
                gut_feeling=-0.3,
                temperature_change=0.2,
            ),
        )
        emotion = self.engine.detect_emotion(inp)
        self.assertIsInstance(emotion, EmotionIdentification)

    def test_detect_emotion_empty_input(self):
        """Should return default for empty input."""
        inp = EmotionalInput()
        emotion = self.engine.detect_emotion(inp)
        self.assertIsInstance(emotion, EmotionIdentification)
        self.assertEqual(emotion.category, EmotionCategory.SURPRISE)

    def test_compute_valence(self):
        """Should compute correct valence sign for emotions."""
        joy = EmotionIdentification(category=EmotionCategory.JOY, intensity=0.8, confidence=0.9)
        sadness = EmotionIdentification(category=EmotionCategory.SADNESS, intensity=0.8, confidence=0.9)
        self.assertGreater(self.engine.compute_valence(joy), 0.0)
        self.assertLess(self.engine.compute_valence(sadness), 0.0)

    def test_compute_arousal(self):
        """Should compute arousal levels."""
        fear = EmotionIdentification(category=EmotionCategory.FEAR, intensity=0.9, confidence=0.8)
        sadness = EmotionIdentification(category=EmotionCategory.SADNESS, intensity=0.9, confidence=0.8)
        # Fear should have higher arousal than sadness
        self.assertGreater(
            self.engine.compute_arousal(fear),
            self.engine.compute_arousal(sadness)
        )

    def test_classify_valence(self):
        """Should classify valence into categories."""
        self.assertEqual(self.engine.classify_valence(0.8), EmotionalValence.VERY_POSITIVE)
        self.assertEqual(self.engine.classify_valence(-0.8), EmotionalValence.VERY_NEGATIVE)
        self.assertEqual(self.engine.classify_valence(0.0), EmotionalValence.NEUTRAL)

    def test_classify_affect(self):
        """Should classify core affect from valence and arousal."""
        self.assertEqual(self.engine.classify_affect(0.5, 0.8), AffectiveState.EXCITED)
        self.assertEqual(self.engine.classify_affect(-0.5, 0.8), AffectiveState.DISTRESSED)
        self.assertEqual(self.engine.classify_affect(0.5, 0.2), AffectiveState.CONTENT)

    def test_action_tendency(self):
        """Should return correct action tendency."""
        joy = EmotionIdentification(category=EmotionCategory.JOY, intensity=0.7, confidence=0.8)
        fear = EmotionIdentification(category=EmotionCategory.FEAR, intensity=0.7, confidence=0.8)
        self.assertEqual(self.engine.get_action_tendency(joy), "approach")
        self.assertEqual(self.engine.get_action_tendency(fear), "avoid")


class TestMoodTrackingEngine(unittest.TestCase):
    """Tests for MoodTrackingEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = MoodTrackingEngine()

    def test_initial_mood(self):
        """Should start with euthymic mood."""
        mood = self.engine.get_current_mood()
        self.assertEqual(mood.mood_state, MoodState.EUTHYMIC)

    def test_update_mood_positive(self):
        """Should shift mood positive with positive emotions."""
        joy = EmotionIdentification(category=EmotionCategory.JOY, intensity=0.8, confidence=0.9)
        for _ in range(10):
            self.engine.update_mood(joy, 0.7)
        mood = self.engine.get_current_mood()
        self.assertGreater(mood.dominant_valence, 0.0)

    def test_mood_report_structure(self):
        """Should return complete mood report."""
        mood = self.engine.get_current_mood()
        self.assertIsInstance(mood, MoodReport)
        self.assertIsInstance(mood.stability, float)
        self.assertGreaterEqual(mood.duration_minutes, 0.0)


class TestEmotionRegulationEngine(unittest.TestCase):
    """Tests for EmotionRegulationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = EmotionRegulationEngine()

    def test_no_regulation_low_intensity(self):
        """Should not suggest regulation for low intensity."""
        emotion = EmotionIdentification(
            category=EmotionCategory.JOY,
            intensity=0.3,
            confidence=0.8,
        )
        strategy = self.engine.suggest_strategy(emotion)
        self.assertIsNone(strategy)

    def test_suggest_reappraisal_for_fear(self):
        """Should suggest reappraisal for fear."""
        emotion = EmotionIdentification(
            category=EmotionCategory.FEAR,
            intensity=0.8,
            confidence=0.9,
        )
        strategy = self.engine.suggest_strategy(emotion)
        self.assertEqual(strategy, EmotionalRegulationStrategy.REAPPRAISAL)

    def test_apply_regulation_reduces_intensity(self):
        """Should reduce intensity when applying regulation."""
        emotion = EmotionIdentification(
            category=EmotionCategory.ANGER,
            intensity=0.9,
            confidence=0.8,
        )
        regulated = self.engine.apply_regulation(
            emotion, EmotionalRegulationStrategy.REAPPRAISAL
        )
        self.assertLess(regulated.intensity, emotion.intensity)
        self.assertGreater(regulated.intensity, 0.0)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestEmotionalConsciousnessInterface(unittest.TestCase):
    """Tests for EmotionalConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = EmotionalConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "07-emotional")
        self.assertEqual(self.interface.FORM_NAME, "Emotional Consciousness")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_emotion(self):
        """Should process emotional input and produce output."""
        inp = EmotionalInput(
            stimulus=EmotionalStimulus(
                stimulus_id="s1",
                stimulus_type="event",
                content_description="Good news received",
                intensity=0.7,
                novelty=0.6,
                personal_relevance=0.8,
            ),
            appraisal=AppraisalInput(
                stimulus_id="s1",
                goal_relevance=0.8,
                goal_congruence=0.7,
                coping_potential=0.9,
                norm_compatibility=0.8,
                certainty=0.6,
            ),
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_emotion(inp))
        finally:
            loop.close()

        self.assertIsInstance(output, EmotionalOutput)
        self.assertIsNotNone(output.emotion)
        self.assertGreaterEqual(output.arousal, 0.0)
        self.assertLessEqual(output.arousal, 1.0)

    def test_get_current_emotion(self):
        """Should return current emotion after processing."""
        inp = EmotionalInput(
            appraisal=AppraisalInput(
                stimulus_id="s2",
                goal_relevance=0.5,
                goal_congruence=-0.6,
                coping_potential=0.3,
                norm_compatibility=0.5,
                certainty=0.7,
                agency="other",
            ),
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_emotion(inp))
        finally:
            loop.close()

        emotion = self.interface.get_current_emotion()
        self.assertIsNotNone(emotion)

    def test_get_status(self):
        """Should return complete status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, EmotionalSystemStatus)
        self.assertIsInstance(status.system_health, float)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "07-emotional")
        self.assertEqual(d["form_name"], "Emotional Consciousness")
        self.assertIn("current_mood", d)

    def test_regulate_emotion(self):
        """Should regulate current emotion."""
        inp = EmotionalInput(
            appraisal=AppraisalInput(
                stimulus_id="s3",
                goal_relevance=0.9,
                goal_congruence=-0.8,
                coping_potential=0.2,
                norm_compatibility=0.3,
                certainty=0.3,
            ),
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_emotion(inp))
            regulated = loop.run_until_complete(
                self.interface.regulate_emotion(EmotionalRegulationStrategy.REAPPRAISAL)
            )
        finally:
            loop.close()

        self.assertIsNotNone(regulated)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_emotional_interface(self):
        """Should create new interface."""
        interface = create_emotional_interface()
        self.assertIsInstance(interface, EmotionalConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "07-emotional")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
