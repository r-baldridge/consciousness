#!/usr/bin/env python3
"""
Test Suite for Form 14: Global Workspace Theory (GWT) Consciousness.

Tests cover:
- All enumerations (WorkspaceState, ContentType, ProcessorType, etc.)
- All input/output dataclasses
- WorkspaceCompetitionEngine
- BroadcastEngine
- GlobalWorkspaceInterface (main interface)
- Convenience functions
- Integration tests for the full workspace cycle
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
    WorkspaceState,
    ContentType,
    ProcessorType,
    BroadcastStrength,
    CompetitionOutcome,
    # Input dataclasses
    WorkspaceContent,
    ProcessorRegistration,
    # Output dataclasses
    BroadcastEvent,
    CompetitionResult,
    WorkspaceSnapshot,
    GWTSystemStatus,
    # Engines
    WorkspaceCompetitionEngine,
    BroadcastEngine,
    # Main interface
    GlobalWorkspaceInterface,
    # Convenience functions
    create_global_workspace_interface,
    create_workspace_content,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestWorkspaceState(unittest.TestCase):
    """Tests for WorkspaceState enumeration."""

    def test_all_states_exist(self):
        """All workspace states should be defined."""
        states = [
            WorkspaceState.IDLE,
            WorkspaceState.COMPETITION,
            WorkspaceState.BROADCASTING,
            WorkspaceState.REFRACTORY,
            WorkspaceState.CONSOLIDATING,
        ]
        self.assertEqual(len(states), 5)

    def test_state_values(self):
        """States should have expected string values."""
        self.assertEqual(WorkspaceState.IDLE.value, "idle")
        self.assertEqual(WorkspaceState.BROADCASTING.value, "broadcasting")
        self.assertEqual(WorkspaceState.COMPETITION.value, "competition")


class TestContentType(unittest.TestCase):
    """Tests for ContentType enumeration."""

    def test_all_types_exist(self):
        """All content types should be defined."""
        types = [
            ContentType.PERCEPTUAL,
            ContentType.COGNITIVE,
            ContentType.EMOTIONAL,
            ContentType.MOTOR,
            ContentType.MEMORY,
            ContentType.LINGUISTIC,
            ContentType.EXECUTIVE,
            ContentType.METACOGNITIVE,
        ]
        self.assertEqual(len(types), 8)


class TestProcessorType(unittest.TestCase):
    """Tests for ProcessorType enumeration."""

    def test_all_types_exist(self):
        """All processor types should be defined."""
        types = [
            ProcessorType.SENSORY,
            ProcessorType.MOTOR,
            ProcessorType.MEMORY,
            ProcessorType.ATTENTION,
            ProcessorType.LANGUAGE,
            ProcessorType.EMOTION,
            ProcessorType.EXECUTIVE,
            ProcessorType.EVALUATION,
        ]
        self.assertEqual(len(types), 8)


class TestBroadcastStrength(unittest.TestCase):
    """Tests for BroadcastStrength enumeration."""

    def test_all_strengths_exist(self):
        """All broadcast strengths should be defined."""
        strengths = [
            BroadcastStrength.WEAK,
            BroadcastStrength.MODERATE,
            BroadcastStrength.STRONG,
            BroadcastStrength.IGNITION,
        ]
        self.assertEqual(len(strengths), 4)


class TestCompetitionOutcome(unittest.TestCase):
    """Tests for CompetitionOutcome enumeration."""

    def test_all_outcomes_exist(self):
        """All competition outcomes should be defined."""
        outcomes = [
            CompetitionOutcome.WON,
            CompetitionOutcome.LOST,
            CompetitionOutcome.PREEMPTED,
            CompetitionOutcome.EXPIRED,
            CompetitionOutcome.MERGED,
        ]
        self.assertEqual(len(outcomes), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestWorkspaceContent(unittest.TestCase):
    """Tests for WorkspaceContent dataclass."""

    def test_creation(self):
        """Should create workspace content with required fields."""
        content = WorkspaceContent(
            content_id="c1",
            content_type=ContentType.PERCEPTUAL,
            content_data={"description": "red square"},
            salience=0.8,
            source_processor=ProcessorType.SENSORY,
        )
        self.assertEqual(content.content_id, "c1")
        self.assertEqual(content.salience, 0.8)
        self.assertEqual(content.content_type, ContentType.PERCEPTUAL)

    def test_competition_score(self):
        """Should compute competition score correctly."""
        content = WorkspaceContent(
            content_id="c1",
            content_type=ContentType.PERCEPTUAL,
            content_data={},
            salience=1.0,
            source_processor=ProcessorType.SENSORY,
            relevance=1.0,
            urgency=1.0,
            coalition_strength=1.0,
        )
        self.assertAlmostEqual(content.competition_score, 1.0, places=2)

    def test_competition_score_zero(self):
        """Zero salience should give low competition score."""
        content = WorkspaceContent(
            content_id="c1",
            content_type=ContentType.PERCEPTUAL,
            content_data={},
            salience=0.0,
            source_processor=ProcessorType.SENSORY,
            relevance=0.0,
            urgency=0.0,
            coalition_strength=0.0,
        )
        self.assertEqual(content.competition_score, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        content = WorkspaceContent(
            content_id="c1",
            content_type=ContentType.COGNITIVE,
            content_data={"thought": "test"},
            salience=0.6,
            source_processor=ProcessorType.EXECUTIVE,
        )
        d = content.to_dict()
        self.assertEqual(d["content_id"], "c1")
        self.assertEqual(d["content_type"], "cognitive")
        self.assertIn("competition_score", d)


class TestProcessorRegistration(unittest.TestCase):
    """Tests for ProcessorRegistration dataclass."""

    def test_creation(self):
        """Should create processor registration."""
        reg = ProcessorRegistration(
            processor_id="p1",
            processor_type=ProcessorType.SENSORY,
            receptive_content_types=[ContentType.PERCEPTUAL],
        )
        self.assertEqual(reg.processor_id, "p1")
        self.assertTrue(reg.is_active)
        self.assertEqual(reg.processing_capacity, 1.0)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestBroadcastEvent(unittest.TestCase):
    """Tests for BroadcastEvent dataclass."""

    def test_creation(self):
        """Should create broadcast event."""
        content = create_workspace_content("c1", salience=0.8)
        event = BroadcastEvent(
            broadcast_id="b1",
            content=content,
            broadcast_strength=BroadcastStrength.STRONG,
            receiving_processors=["p1", "p2"],
            num_receivers=2,
            duration_ms=3.5,
        )
        self.assertEqual(event.broadcast_id, "b1")
        self.assertEqual(event.num_receivers, 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        content = create_workspace_content("c1")
        event = BroadcastEvent(
            broadcast_id="b1",
            content=content,
            broadcast_strength=BroadcastStrength.MODERATE,
            receiving_processors=["p1"],
            num_receivers=1,
            duration_ms=2.0,
        )
        d = event.to_dict()
        self.assertEqual(d["broadcast_id"], "b1")
        self.assertEqual(d["broadcast_strength"], "moderate")


class TestCompetitionResult(unittest.TestCase):
    """Tests for CompetitionResult dataclass."""

    def test_creation_with_winner(self):
        """Should create result with winner."""
        winner = create_workspace_content("winner", salience=0.9)
        result = CompetitionResult(
            winner=winner,
            competitors=[winner],
            outcome=CompetitionOutcome.WON,
            winning_score=0.9,
            margin=0.3,
            cycle_duration_ms=2.5,
        )
        self.assertEqual(result.outcome, CompetitionOutcome.WON)
        self.assertIsNotNone(result.winner)

    def test_creation_no_winner(self):
        """Should create result without winner."""
        result = CompetitionResult(
            winner=None,
            competitors=[],
            outcome=CompetitionOutcome.EXPIRED,
            winning_score=0.0,
            margin=0.0,
            cycle_duration_ms=1.0,
        )
        self.assertIsNone(result.winner)


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestWorkspaceCompetitionEngine(unittest.TestCase):
    """Tests for WorkspaceCompetitionEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = WorkspaceCompetitionEngine()

    def test_empty_competition(self):
        """Empty competition should have no winner."""
        result = self.engine.run_competition([])
        self.assertIsNone(result.winner)
        self.assertEqual(result.outcome, CompetitionOutcome.EXPIRED)

    def test_single_candidate_above_threshold(self):
        """Single candidate above threshold should win."""
        content = create_workspace_content("c1", salience=0.8)
        content.relevance = 0.7
        result = self.engine.run_competition([content])
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner.content_id, "c1")
        self.assertEqual(result.outcome, CompetitionOutcome.WON)

    def test_highest_salience_wins(self):
        """Highest scoring content should win competition."""
        low = create_workspace_content("low", salience=0.3)
        high = create_workspace_content("high", salience=0.9)
        high.relevance = 0.8

        result = self.engine.run_competition([low, high])
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner.content_id, "high")

    def test_below_threshold_no_winner(self):
        """Content below threshold should not win."""
        low = create_workspace_content("low", salience=0.1)
        low.relevance = 0.1
        low.urgency = 0.0
        low.coalition_strength = 0.0

        result = self.engine.run_competition([low])
        self.assertIsNone(result.winner)

    def test_margin_computation(self):
        """Should compute margin between winner and runner-up."""
        first = create_workspace_content("first", salience=0.9)
        first.relevance = 0.8
        second = create_workspace_content("second", salience=0.6)
        second.relevance = 0.5

        result = self.engine.run_competition([first, second])
        self.assertGreater(result.margin, 0.0)


class TestBroadcastEngine(unittest.TestCase):
    """Tests for BroadcastEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = BroadcastEngine()

    def test_broadcast_to_compatible_processors(self):
        """Should broadcast to compatible processors."""
        content = create_workspace_content("c1", ContentType.PERCEPTUAL, 0.8)
        processors = [
            ProcessorRegistration("p1", ProcessorType.SENSORY, [ContentType.PERCEPTUAL]),
            ProcessorRegistration("p2", ProcessorType.MOTOR, [ContentType.MOTOR]),
            ProcessorRegistration("p3", ProcessorType.ATTENTION, list(ContentType)),
        ]

        event = self.engine.broadcast(content, processors)
        self.assertIn("p1", event.receiving_processors)
        self.assertIn("p3", event.receiving_processors)
        self.assertGreater(event.num_receivers, 0)

    def test_broadcast_strength_determination(self):
        """Should determine appropriate broadcast strength."""
        content = create_workspace_content("c1", salience=0.9)
        content.relevance = 0.9
        processors = [
            ProcessorRegistration("p1", ProcessorType.ATTENTION, list(ContentType)),
        ]

        event = self.engine.broadcast(content, processors)
        self.assertIsInstance(event.broadcast_strength, BroadcastStrength)

    def test_broadcast_history(self):
        """Should maintain broadcast history."""
        content = create_workspace_content("c1")
        processors = [
            ProcessorRegistration("p1", ProcessorType.ATTENTION, list(ContentType)),
        ]

        self.engine.broadcast(content, processors)
        self.engine.broadcast(content, processors)

        history = self.engine.get_history()
        self.assertEqual(len(history), 2)

    def test_broadcast_id_increments(self):
        """Broadcast IDs should increment."""
        content = create_workspace_content("c1")
        processors = []

        e1 = self.engine.broadcast(content, processors)
        e2 = self.engine.broadcast(content, processors)
        self.assertNotEqual(e1.broadcast_id, e2.broadcast_id)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestGlobalWorkspaceInterface(unittest.TestCase):
    """Tests for GlobalWorkspaceInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = GlobalWorkspaceInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "14-global-workspace")
        self.assertEqual(self.interface.FORM_NAME, "Global Workspace Theory (GWT)")

    def test_initialize(self):
        """Should initialize with default processors."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()

        self.assertTrue(self.interface._is_initialized)
        self.assertGreater(len(self.interface._processors), 0)

    def test_submit_content(self):
        """Should accept content submission."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
            content = create_workspace_content("test_content", salience=0.8)
            content.relevance = 0.7
            outcome = loop.run_until_complete(self.interface.submit_content(content))
        finally:
            loop.close()

        self.assertIsInstance(outcome, CompetitionOutcome)

    def test_submit_winning_content(self):
        """High salience content should win when workspace is idle."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
            content = create_workspace_content("winner", salience=0.9)
            content.relevance = 0.9
            outcome = loop.run_until_complete(self.interface.submit_content(content))
        finally:
            loop.close()

        self.assertEqual(outcome, CompetitionOutcome.WON)

    def test_get_workspace_state(self):
        """Should return workspace state."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
            snapshot = loop.run_until_complete(self.interface.get_workspace_state())
        finally:
            loop.close()

        self.assertIsInstance(snapshot, WorkspaceSnapshot)
        self.assertEqual(snapshot.state, WorkspaceState.IDLE)

    def test_get_broadcast_history(self):
        """Should return broadcast history."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
            content = create_workspace_content("c1", salience=0.9)
            content.relevance = 0.8
            loop.run_until_complete(self.interface.submit_content(content))
            history = loop.run_until_complete(self.interface.get_broadcast_history())
        finally:
            loop.close()

        self.assertIsInstance(history, list)

    def test_register_processor(self):
        """Should register new processor."""
        proc = ProcessorRegistration(
            "custom_proc", ProcessorType.EVALUATION,
            [ContentType.COGNITIVE], 1.0, True
        )
        self.interface.register_processor(proc)
        self.assertIn("custom_proc", self.interface._processors)

    def test_unregister_processor(self):
        """Should unregister processor."""
        self.interface.register_processor(
            ProcessorRegistration("temp", ProcessorType.SENSORY, [])
        )
        self.interface.unregister_processor("temp")
        self.assertNotIn("temp", self.interface._processors)

    def test_broadcast_callback(self):
        """Should call broadcast callbacks."""
        callback_data = []

        def callback(event):
            callback_data.append(event)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
            self.interface.on_broadcast(callback)
            content = create_workspace_content("c1", salience=0.9)
            content.relevance = 0.9
            loop.run_until_complete(self.interface.submit_content(content))
        finally:
            loop.close()

        self.assertGreaterEqual(len(callback_data), 1)

    def test_to_dict(self):
        """Should convert state to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "14-global-workspace")
        self.assertIn("workspace_state", d)
        self.assertIn("broadcast_count", d)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, GWTSystemStatus)
        self.assertFalse(status.is_initialized)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_global_workspace_interface(self):
        """Should create new interface."""
        interface = create_global_workspace_interface()
        self.assertIsInstance(interface, GlobalWorkspaceInterface)

    def test_create_workspace_content(self):
        """Should create workspace content."""
        content = create_workspace_content("test_id", ContentType.COGNITIVE, 0.7)
        self.assertEqual(content.content_id, "test_id")
        self.assertEqual(content.content_type, ContentType.COGNITIVE)
        self.assertEqual(content.salience, 0.7)

    def test_create_workspace_content_defaults(self):
        """Should create content with defaults."""
        content = create_workspace_content("c1")
        self.assertEqual(content.content_type, ContentType.PERCEPTUAL)
        self.assertEqual(content.salience, 0.5)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestGWTIntegration(unittest.TestCase):
    """Integration tests for the Global Workspace system."""

    def test_full_competition_broadcast_cycle(self):
        """Should complete full competition and broadcast cycle."""
        interface = create_global_workspace_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())

            # Submit multiple contents
            c1 = create_workspace_content("low", salience=0.4)
            c1.relevance = 0.3
            c2 = create_workspace_content("high", salience=0.9)
            c2.relevance = 0.8
            c2.urgency = 0.7

            outcome1 = loop.run_until_complete(interface.submit_content(c1))
            outcome2 = loop.run_until_complete(interface.submit_content(c2))

            # Get state
            snapshot = loop.run_until_complete(interface.get_workspace_state())
        finally:
            loop.close()

        self.assertGreater(interface._broadcast_count, 0)

    def test_competition_with_multiple_contents(self):
        """Multiple competing contents should result in one winner."""
        engine = WorkspaceCompetitionEngine()

        contents = [
            create_workspace_content(f"c{i}", salience=0.3 + i * 0.15)
            for i in range(5)
        ]
        for c in contents:
            c.relevance = c.salience * 0.9

        result = engine.run_competition(contents)
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner.content_id, "c4")  # Highest salience

    def test_workspace_state_consistency(self):
        """Workspace state should be consistent after operations."""
        interface = create_global_workspace_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())

            content = create_workspace_content("c1", salience=0.8)
            content.relevance = 0.7
            loop.run_until_complete(interface.submit_content(content))

            snapshot = loop.run_until_complete(interface.get_workspace_state())
        finally:
            loop.close()

        # After processing, should be back to idle
        self.assertEqual(snapshot.state, WorkspaceState.IDLE)
        self.assertGreater(snapshot.broadcast_count, 0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
