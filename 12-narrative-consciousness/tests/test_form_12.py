#!/usr/bin/env python3
"""
Test Suite for Form 12: Narrative Consciousness.

Tests cover:
- All enumerations (NarrativeMode, TemporalOrientation, MemoryType, etc.)
- All input/output dataclasses
- NarrativeConstructionEngine
- TemporalConsciousnessEngine
- AutobiographicalMemoryEngine
- NarrativeConsciousnessInterface (main interface)
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
    NarrativeMode,
    TemporalOrientation,
    MemoryType,
    NarrativeCoherence,
    IdentityTheme,
    # Input dataclasses
    EventInput,
    MemoryInput,
    NarrativeInput,
    # Output dataclasses
    NarrativeSegment,
    NarrativeStructure,
    TemporalFrame,
    IdentityCoherenceReport,
    NarrativeOutput,
    NarrativeSystemStatus,
    # Engines
    NarrativeConstructionEngine,
    TemporalConsciousnessEngine,
    AutobiographicalMemoryEngine,
    # Main interface
    NarrativeConsciousnessInterface,
    # Convenience
    create_narrative_consciousness_interface,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestNarrativeMode(unittest.TestCase):
    """Tests for NarrativeMode enumeration."""

    def test_all_modes_exist(self):
        """All narrative modes should be defined."""
        modes = [
            NarrativeMode.AUTOBIOGRAPHICAL,
            NarrativeMode.EPISODIC,
            NarrativeMode.SEMANTIC,
            NarrativeMode.PROSPECTIVE,
            NarrativeMode.CONFABULATION,
            NarrativeMode.REFLECTIVE,
        ]
        self.assertEqual(len(modes), 6)

    def test_mode_values(self):
        """Modes should have expected string values."""
        self.assertEqual(NarrativeMode.AUTOBIOGRAPHICAL.value, "autobiographical")
        self.assertEqual(NarrativeMode.PROSPECTIVE.value, "prospective")


class TestTemporalOrientation(unittest.TestCase):
    """Tests for TemporalOrientation enumeration."""

    def test_all_orientations_exist(self):
        """All temporal orientations should be defined."""
        orientations = [
            TemporalOrientation.PAST_DISTANT,
            TemporalOrientation.PAST_RECENT,
            TemporalOrientation.PRESENT,
            TemporalOrientation.FUTURE_NEAR,
            TemporalOrientation.FUTURE_DISTANT,
            TemporalOrientation.TIMELESS,
        ]
        self.assertEqual(len(orientations), 6)


class TestMemoryType(unittest.TestCase):
    """Tests for MemoryType enumeration."""

    def test_all_types_exist(self):
        """All memory types should be defined."""
        types = [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
            MemoryType.AUTOBIOGRAPHICAL,
            MemoryType.PROSPECTIVE,
            MemoryType.FLASHBULB,
            MemoryType.WORKING,
        ]
        self.assertEqual(len(types), 7)


class TestNarrativeCoherence(unittest.TestCase):
    """Tests for NarrativeCoherence enumeration."""

    def test_all_levels_exist(self):
        """All coherence levels should be defined."""
        levels = [
            NarrativeCoherence.HIGHLY_COHERENT,
            NarrativeCoherence.COHERENT,
            NarrativeCoherence.FRAGMENTED,
            NarrativeCoherence.DISJOINTED,
            NarrativeCoherence.INCOHERENT,
        ]
        self.assertEqual(len(levels), 5)


class TestIdentityTheme(unittest.TestCase):
    """Tests for IdentityTheme enumeration."""

    def test_all_themes_exist(self):
        """All identity themes should be defined."""
        themes = [
            IdentityTheme.AGENCY,
            IdentityTheme.COMMUNION,
            IdentityTheme.GROWTH,
            IdentityTheme.RESILIENCE,
            IdentityTheme.DISCOVERY,
            IdentityTheme.CONTINUITY,
            IdentityTheme.TRANSFORMATION,
        ]
        self.assertEqual(len(themes), 7)

    def test_theme_values(self):
        """Themes should have expected string values."""
        self.assertEqual(IdentityTheme.AGENCY.value, "agency")
        self.assertEqual(IdentityTheme.RESILIENCE.value, "resilience")


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestEventInput(unittest.TestCase):
    """Tests for EventInput dataclass."""

    def test_creation(self):
        """Should create event input."""
        event = EventInput(
            event_id="evt_001",
            description="Graduated from university",
            significance=0.9,
            emotional_valence=0.8,
            emotional_intensity=0.7,
            temporal_location=TemporalOrientation.PAST_DISTANT,
        )
        self.assertEqual(event.event_id, "evt_001")
        self.assertEqual(event.significance, 0.9)
        self.assertEqual(event.temporal_location, TemporalOrientation.PAST_DISTANT)

    def test_to_dict(self):
        """Should convert to dictionary."""
        event = EventInput(
            event_id="evt_002",
            description="Started new job",
            significance=0.7,
            emotional_valence=0.6,
            emotional_intensity=0.5,
            temporal_location=TemporalOrientation.PAST_RECENT,
            participants=["colleague_a"],
        )
        d = event.to_dict()
        self.assertEqual(d["event_id"], "evt_002")
        self.assertEqual(d["temporal_location"], "past_recent")

    def test_causal_links(self):
        """Should support causal links between events."""
        event = EventInput(
            event_id="evt_003",
            description="Got promoted",
            significance=0.8,
            emotional_valence=0.7,
            emotional_intensity=0.6,
            temporal_location=TemporalOrientation.PAST_RECENT,
            causal_links=["evt_001", "evt_002"],
        )
        self.assertEqual(len(event.causal_links), 2)


class TestMemoryInput(unittest.TestCase):
    """Tests for MemoryInput dataclass."""

    def test_creation(self):
        """Should create memory input."""
        memory = MemoryInput(
            memory_id="mem_001",
            memory_type=MemoryType.EPISODIC,
            content_summary="First day of school",
            vividness=0.6,
            confidence=0.7,
            emotional_charge=0.5,
            temporal_distance=0.9,
        )
        self.assertEqual(memory.memory_id, "mem_001")
        self.assertEqual(memory.memory_type, MemoryType.EPISODIC)

    def test_to_dict(self):
        """Should convert to dictionary."""
        memory = MemoryInput(
            memory_id="mem_002",
            memory_type=MemoryType.FLASHBULB,
            content_summary="Vivid childhood event",
            vividness=0.95,
            confidence=0.8,
            emotional_charge=0.9,
            temporal_distance=0.8,
            retrieval_cue="childhood",
        )
        d = memory.to_dict()
        self.assertEqual(d["memory_type"], "flashbulb")
        self.assertEqual(d["retrieval_cue"], "childhood")


class TestNarrativeInput(unittest.TestCase):
    """Tests for NarrativeInput dataclass."""

    def test_empty_input(self):
        """Should create empty input."""
        inp = NarrativeInput()
        self.assertEqual(len(inp.events), 0)
        self.assertEqual(len(inp.memories), 0)
        self.assertEqual(inp.narrative_mode, NarrativeMode.AUTOBIOGRAPHICAL)

    def test_full_input(self):
        """Should create full input."""
        inp = NarrativeInput(
            events=[
                EventInput(
                    event_id="e1", description="Test event",
                    significance=0.5, emotional_valence=0.3,
                    emotional_intensity=0.4,
                    temporal_location=TemporalOrientation.PRESENT,
                ),
            ],
            memories=[
                MemoryInput(
                    memory_id="m1", memory_type=MemoryType.EPISODIC,
                    content_summary="Test memory",
                    vividness=0.6, confidence=0.7,
                    emotional_charge=0.4, temporal_distance=0.5,
                ),
            ],
            narrative_mode=NarrativeMode.REFLECTIVE,
            temporal_focus=TemporalOrientation.PAST_RECENT,
        )
        self.assertEqual(len(inp.events), 1)
        self.assertEqual(len(inp.memories), 1)
        self.assertEqual(inp.narrative_mode, NarrativeMode.REFLECTIVE)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestNarrativeSegment(unittest.TestCase):
    """Tests for NarrativeSegment dataclass."""

    def test_creation(self):
        """Should create narrative segment."""
        segment = NarrativeSegment(
            segment_id="seg_001",
            content="A significant life event occurred",
            temporal_position=TemporalOrientation.PAST_RECENT,
            emotional_tone=0.5,
            significance=0.8,
            related_events=["evt_001"],
            related_memories=["mem_001"],
        )
        self.assertEqual(segment.segment_id, "seg_001")
        self.assertEqual(len(segment.related_events), 1)

    def test_to_dict(self):
        """Should convert to dictionary."""
        segment = NarrativeSegment(
            segment_id="seg_002",
            content="Future plans",
            temporal_position=TemporalOrientation.FUTURE_NEAR,
            emotional_tone=0.6,
            significance=0.7,
            related_events=[],
            related_memories=[],
        )
        d = segment.to_dict()
        self.assertEqual(d["temporal_position"], "future_near")


class TestNarrativeStructure(unittest.TestCase):
    """Tests for NarrativeStructure dataclass."""

    def test_creation(self):
        """Should create narrative structure."""
        structure = NarrativeStructure(
            segments=[],
            coherence=NarrativeCoherence.COHERENT,
            coherence_score=0.7,
            dominant_theme=IdentityTheme.GROWTH,
            temporal_span=[TemporalOrientation.PAST_RECENT, TemporalOrientation.PRESENT],
            narrative_arc="ascending",
        )
        self.assertEqual(structure.coherence, NarrativeCoherence.COHERENT)
        self.assertEqual(structure.narrative_arc, "ascending")

    def test_to_dict(self):
        """Should convert to dictionary."""
        structure = NarrativeStructure(
            segments=[],
            coherence=NarrativeCoherence.FRAGMENTED,
            coherence_score=0.4,
            dominant_theme=IdentityTheme.RESILIENCE,
            temporal_span=[],
            narrative_arc="complex",
        )
        d = structure.to_dict()
        self.assertEqual(d["coherence"], "fragmented")
        self.assertEqual(d["dominant_theme"], "resilience")


class TestTemporalFrame(unittest.TestCase):
    """Tests for TemporalFrame dataclass."""

    def test_creation(self):
        """Should create temporal frame."""
        frame = TemporalFrame(
            orientation=TemporalOrientation.PRESENT,
            subjective_present_duration=2.0,
            temporal_coherence=0.8,
            future_orientation=0.3,
            past_orientation=0.3,
            present_orientation=0.4,
        )
        self.assertEqual(frame.orientation, TemporalOrientation.PRESENT)
        self.assertAlmostEqual(
            frame.future_orientation + frame.past_orientation + frame.present_orientation,
            1.0, places=1
        )

    def test_to_dict(self):
        """Should convert to dictionary."""
        frame = TemporalFrame(
            orientation=TemporalOrientation.PAST_RECENT,
            subjective_present_duration=1.5,
            temporal_coherence=0.7,
            future_orientation=0.2,
            past_orientation=0.5,
            present_orientation=0.3,
        )
        d = frame.to_dict()
        self.assertEqual(d["orientation"], "past_recent")


class TestIdentityCoherenceReport(unittest.TestCase):
    """Tests for IdentityCoherenceReport dataclass."""

    def test_creation(self):
        """Should create identity coherence report."""
        report = IdentityCoherenceReport(
            overall_coherence=0.8,
            dominant_themes=[IdentityTheme.AGENCY, IdentityTheme.GROWTH],
            self_continuity=0.85,
            self_change=0.4,
            narrative_complexity=0.6,
            unresolved_conflicts=1,
        )
        self.assertEqual(report.overall_coherence, 0.8)
        self.assertEqual(len(report.dominant_themes), 2)

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = IdentityCoherenceReport(
            overall_coherence=0.7,
            dominant_themes=[IdentityTheme.CONTINUITY],
            self_continuity=0.8,
            self_change=0.2,
            narrative_complexity=0.4,
            unresolved_conflicts=0,
        )
        d = report.to_dict()
        self.assertIn("continuity", d["dominant_themes"])


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestNarrativeConstructionEngine(unittest.TestCase):
    """Tests for NarrativeConstructionEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = NarrativeConstructionEngine()

    def test_construct_empty_narrative(self):
        """Should handle empty input."""
        narrative = self.engine.construct_narrative([], [], NarrativeMode.AUTOBIOGRAPHICAL)
        self.assertIsInstance(narrative, NarrativeStructure)
        self.assertEqual(len(narrative.segments), 0)

    def test_construct_from_events(self):
        """Should construct narrative from events."""
        events = [
            EventInput(
                event_id="e1", description="Childhood memory",
                significance=0.7, emotional_valence=0.5,
                emotional_intensity=0.6,
                temporal_location=TemporalOrientation.PAST_DISTANT,
            ),
            EventInput(
                event_id="e2", description="Recent achievement",
                significance=0.9, emotional_valence=0.8,
                emotional_intensity=0.7,
                temporal_location=TemporalOrientation.PAST_RECENT,
                causal_links=["e1"],
            ),
        ]
        narrative = self.engine.construct_narrative(events, [], NarrativeMode.AUTOBIOGRAPHICAL)
        self.assertEqual(len(narrative.segments), 2)
        self.assertGreater(narrative.coherence_score, 0.0)

    def test_construct_from_memories(self):
        """Should construct narrative from memories."""
        memories = [
            MemoryInput(
                memory_id="m1", memory_type=MemoryType.EPISODIC,
                content_summary="First day at work",
                vividness=0.7, confidence=0.8,
                emotional_charge=0.5, temporal_distance=0.6,
            ),
        ]
        narrative = self.engine.construct_narrative([], memories, NarrativeMode.EPISODIC)
        self.assertEqual(len(narrative.segments), 1)

    def test_coherence_classification(self):
        """Should correctly classify coherence levels."""
        self.assertEqual(
            self.engine._classify_coherence(0.9),
            NarrativeCoherence.HIGHLY_COHERENT
        )
        self.assertEqual(
            self.engine._classify_coherence(0.5),
            NarrativeCoherence.FRAGMENTED
        )
        self.assertEqual(
            self.engine._classify_coherence(0.1),
            NarrativeCoherence.INCOHERENT
        )

    def test_narrative_arc_ascending(self):
        """Should detect ascending narrative arc."""
        events = [
            EventInput(
                event_id="e1", description="Struggle",
                significance=0.7, emotional_valence=-0.5,
                emotional_intensity=0.6,
                temporal_location=TemporalOrientation.PAST_DISTANT,
            ),
            EventInput(
                event_id="e2", description="Recovery",
                significance=0.7, emotional_valence=0.0,
                emotional_intensity=0.5,
                temporal_location=TemporalOrientation.PAST_RECENT,
            ),
            EventInput(
                event_id="e3", description="Triumph",
                significance=0.9, emotional_valence=0.8,
                emotional_intensity=0.8,
                temporal_location=TemporalOrientation.PRESENT,
            ),
        ]
        narrative = self.engine.construct_narrative(events, [], NarrativeMode.AUTOBIOGRAPHICAL)
        self.assertEqual(narrative.narrative_arc, "ascending")

    def test_theme_identification(self):
        """Should identify narrative themes."""
        events = [
            EventInput(
                event_id="e1", description="Growth event",
                significance=0.8, emotional_valence=0.6,
                emotional_intensity=0.7,
                temporal_location=TemporalOrientation.PAST_RECENT,
            ),
        ]
        theme = self.engine._identify_theme(events, [])
        self.assertIsNotNone(theme)


class TestTemporalConsciousnessEngine(unittest.TestCase):
    """Tests for TemporalConsciousnessEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = TemporalConsciousnessEngine()

    def test_compute_temporal_frame(self):
        """Should compute temporal frame."""
        events = [
            EventInput(
                event_id="e1", description="Past event",
                significance=0.5, emotional_valence=0.3,
                emotional_intensity=0.4,
                temporal_location=TemporalOrientation.PAST_RECENT,
            ),
        ]
        frame = self.engine.compute_temporal_frame(
            events, [], TemporalOrientation.PRESENT
        )
        self.assertIsInstance(frame, TemporalFrame)
        self.assertEqual(frame.orientation, TemporalOrientation.PRESENT)
        self.assertGreater(frame.subjective_present_duration, 0)

    def test_temporal_coherence(self):
        """Should compute temporal coherence."""
        events = [
            EventInput(
                event_id="e1", description="Event A",
                significance=0.5, emotional_valence=0.3,
                emotional_intensity=0.4,
                temporal_location=TemporalOrientation.PAST_RECENT,
                causal_links=["e2"],
            ),
            EventInput(
                event_id="e2", description="Event B",
                significance=0.6, emotional_valence=0.4,
                emotional_intensity=0.5,
                temporal_location=TemporalOrientation.PRESENT,
            ),
        ]
        frame = self.engine.compute_temporal_frame(
            events, [], TemporalOrientation.PRESENT
        )
        self.assertGreater(frame.temporal_coherence, 0.0)

    def test_past_focus_weights(self):
        """Should weight past more when focus is on past."""
        events = [
            EventInput(
                event_id="e1", description="Past event",
                significance=0.5, emotional_valence=0.3,
                emotional_intensity=0.4,
                temporal_location=TemporalOrientation.PAST_RECENT,
            ),
        ]
        frame = self.engine.compute_temporal_frame(
            events, [], TemporalOrientation.PAST_RECENT
        )
        self.assertGreater(frame.past_orientation, frame.future_orientation)


class TestAutobiographicalMemoryEngine(unittest.TestCase):
    """Tests for AutobiographicalMemoryEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = AutobiographicalMemoryEngine()

    def test_store_and_retrieve_memory(self):
        """Should store and retrieve memories."""
        memory = MemoryInput(
            memory_id="m1", memory_type=MemoryType.EPISODIC,
            content_summary="First day at school",
            vividness=0.7, confidence=0.8,
            emotional_charge=0.5, temporal_distance=0.9,
            retrieval_cue="school",
        )
        self.engine.store_memory(memory)
        results = self.engine.retrieve_by_cue("school")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, "m1")

    def test_retrieve_by_type(self):
        """Should retrieve memories by type."""
        self.engine.store_memory(MemoryInput(
            memory_id="m1", memory_type=MemoryType.EPISODIC,
            content_summary="Event 1",
            vividness=0.7, confidence=0.8,
            emotional_charge=0.5, temporal_distance=0.5,
        ))
        self.engine.store_memory(MemoryInput(
            memory_id="m2", memory_type=MemoryType.SEMANTIC,
            content_summary="Fact 1",
            vividness=0.3, confidence=0.9,
            emotional_charge=0.1, temporal_distance=0.0,
        ))
        episodic = self.engine.retrieve_by_type(MemoryType.EPISODIC)
        self.assertEqual(len(episodic), 1)

    def test_store_event(self):
        """Should store events."""
        event = EventInput(
            event_id="e1", description="Test",
            significance=0.5, emotional_valence=0.3,
            emotional_intensity=0.4,
            temporal_location=TemporalOrientation.PRESENT,
        )
        self.engine.store_event(event)
        self.assertEqual(self.engine.get_event_count(), 1)

    def test_identity_coherence_empty(self):
        """Should return default coherence for empty bank."""
        report = self.engine.compute_identity_coherence()
        self.assertIsInstance(report, IdentityCoherenceReport)
        self.assertEqual(report.overall_coherence, 0.5)

    def test_identity_coherence_with_data(self):
        """Should compute identity coherence from stored data."""
        self.engine.store_memory(MemoryInput(
            memory_id="m1", memory_type=MemoryType.AUTOBIOGRAPHICAL,
            content_summary="Life event",
            vividness=0.8, confidence=0.9,
            emotional_charge=0.6, temporal_distance=0.5,
        ))
        self.engine.store_event(EventInput(
            event_id="e1", description="Growth event",
            significance=0.8, emotional_valence=0.6,
            emotional_intensity=0.7,
            temporal_location=TemporalOrientation.PAST_RECENT,
        ))
        report = self.engine.compute_identity_coherence()
        self.assertGreater(report.overall_coherence, 0.0)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestNarrativeConsciousnessInterface(unittest.TestCase):
    """Tests for NarrativeConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = NarrativeConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "12-narrative-consciousness")
        self.assertEqual(self.interface.FORM_NAME, "Narrative Consciousness")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._initialized)

    def test_process_narrative(self):
        """Should process narrative input."""
        inp = NarrativeInput(
            events=[
                EventInput(
                    event_id="e1", description="A formative experience",
                    significance=0.8, emotional_valence=0.5,
                    emotional_intensity=0.6,
                    temporal_location=TemporalOrientation.PAST_RECENT,
                ),
            ],
            memories=[
                MemoryInput(
                    memory_id="m1", memory_type=MemoryType.AUTOBIOGRAPHICAL,
                    content_summary="Related earlier memory",
                    vividness=0.7, confidence=0.8,
                    emotional_charge=0.5, temporal_distance=0.7,
                ),
            ],
            narrative_mode=NarrativeMode.AUTOBIOGRAPHICAL,
            temporal_focus=TemporalOrientation.PRESENT,
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(
                self.interface.process_narrative(inp)
            )
        finally:
            loop.close()

        self.assertIsInstance(output, NarrativeOutput)
        self.assertIsNotNone(output.narrative)
        self.assertIsNotNone(output.temporal_frame)
        self.assertIsNotNone(output.identity_coherence)
        self.assertGreater(len(output.narrative.segments), 0)

    def test_recall_memories(self):
        """Should recall stored memories."""
        inp = NarrativeInput(
            memories=[
                MemoryInput(
                    memory_id="m1", memory_type=MemoryType.EPISODIC,
                    content_summary="School days memory",
                    vividness=0.7, confidence=0.8,
                    emotional_charge=0.5, temporal_distance=0.8,
                    retrieval_cue="school",
                ),
            ],
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.process_narrative(inp))
            results = loop.run_until_complete(self.interface.recall_memories("school"))
        finally:
            loop.close()
        self.assertGreater(len(results), 0)

    def test_get_identity_report(self):
        """Should get identity coherence report."""
        loop = asyncio.new_event_loop()
        try:
            report = loop.run_until_complete(self.interface.get_identity_report())
        finally:
            loop.close()
        self.assertIsInstance(report, IdentityCoherenceReport)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, NarrativeSystemStatus)
        self.assertGreaterEqual(status.system_health, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "12-narrative-consciousness")
        self.assertEqual(d["form_name"], "Narrative Consciousness")
        self.assertIn("active_mode", d)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_narrative_consciousness_interface(self):
        """Should create new interface."""
        interface = create_narrative_consciousness_interface()
        self.assertIsInstance(interface, NarrativeConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "12-narrative-consciousness")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
