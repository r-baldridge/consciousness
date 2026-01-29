#!/usr/bin/env python3
"""
Test suite for Form 18: Primary Consciousness (Edelman)

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- Interface categorization, scene construction, and awareness assessment
- Remembered present and temporal continuity
- Edge cases (empty inputs, absent awareness)
"""

import sys
from pathlib import Path
import unittest
import asyncio

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE_PATH = Path(__file__).parent.parent
INTERFACE_PATH = BASE_PATH / "interface"

interface_module = load_module_from_path(
    "primary_consciousness_interface",
    INTERFACE_PATH / "primary_consciousness_interface.py",
)

# Import all components
PrimaryAwarenessLevel = interface_module.PrimaryAwarenessLevel
SensoryBoundState = interface_module.SensoryBoundState
PerceptualCategory = interface_module.PerceptualCategory
ValueAssignment = interface_module.ValueAssignment
SceneCoherence = interface_module.SceneCoherence
SensoryChannelData = interface_module.SensoryChannelData
PrimarySensoryInput = interface_module.PrimarySensoryInput
CategorizedPercept = interface_module.CategorizedPercept
RememberedPresent = interface_module.RememberedPresent
PrimaryConsciousnessOutput = interface_module.PrimaryConsciousnessOutput
PrimaryConsciousnessInterface = interface_module.PrimaryConsciousnessInterface
create_primary_consciousness_interface = interface_module.create_primary_consciousness_interface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_channel(channel="visual", intensity=0.5, complexity=0.5,
                  novelty=0.3, rate=0.1, reliability=1.0):
    return SensoryChannelData(
        channel=channel,
        raw_intensity=intensity,
        pattern_complexity=complexity,
        novelty_score=novelty,
        temporal_rate=rate,
        reliability=reliability,
    )


def _make_input(channels=None, arousal=0.5):
    if channels is None:
        channels = [_make_channel()]
    return PrimarySensoryInput(channels=channels, ambient_arousal=arousal)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestPrimaryAwarenessLevel(unittest.TestCase):
    """Tests for PrimaryAwarenessLevel enum."""

    def test_level_count(self):
        levels = list(PrimaryAwarenessLevel)
        self.assertEqual(len(levels), 5)

    def test_values_are_strings(self):
        for level in PrimaryAwarenessLevel:
            self.assertIsInstance(level.value, str)

    def test_expected_members(self):
        names = {m.name for m in PrimaryAwarenessLevel}
        for expected in ["ABSENT", "MINIMAL", "PARTIAL", "COHERENT", "HEIGHTENED"]:
            self.assertIn(expected, names)


class TestSensoryBoundState(unittest.TestCase):
    """Tests for SensoryBoundState enum."""

    def test_state_count(self):
        self.assertEqual(len(list(SensoryBoundState)), 5)

    def test_expected_members(self):
        names = {m.name for m in SensoryBoundState}
        for expected in ["UNBOUND", "LOOSELY_BOUND", "BOUND", "TIGHTLY_BOUND", "FRAGMENTED"]:
            self.assertIn(expected, names)


class TestPerceptualCategory(unittest.TestCase):
    """Tests for PerceptualCategory enum."""

    def test_category_count(self):
        self.assertGreaterEqual(len(list(PerceptualCategory)), 10)

    def test_expected_members(self):
        names = {m.name for m in PerceptualCategory}
        for expected in ["OBJECT", "MOTION", "THREAT_CUE", "REWARD_CUE",
                         "SOCIAL_SIGNAL", "AUDITORY_PATTERN"]:
            self.assertIn(expected, names)


class TestValueAssignment(unittest.TestCase):
    """Tests for ValueAssignment enum."""

    def test_value_count(self):
        self.assertEqual(len(list(ValueAssignment)), 6)

    def test_hedonic_range(self):
        names = {m.name for m in ValueAssignment}
        self.assertIn("STRONGLY_AVERSIVE", names)
        self.assertIn("STRONGLY_APPETITIVE", names)
        self.assertIn("NEUTRAL", names)
        self.assertIn("NOVEL", names)


class TestSceneCoherence(unittest.TestCase):
    """Tests for SceneCoherence enum."""

    def test_coherence_count(self):
        self.assertEqual(len(list(SceneCoherence)), 5)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSensoryChannelData(unittest.TestCase):
    """Tests for SensoryChannelData dataclass."""

    def test_creation(self):
        ch = _make_channel()
        self.assertEqual(ch.channel, "visual")
        self.assertAlmostEqual(ch.raw_intensity, 0.5)

    def test_to_dict(self):
        ch = _make_channel(channel="auditory", intensity=0.8)
        d = ch.to_dict()
        self.assertEqual(d["channel"], "auditory")
        self.assertAlmostEqual(d["raw_intensity"], 0.8)
        self.assertIn("timestamp", d)


class TestPrimarySensoryInput(unittest.TestCase):
    """Tests for PrimarySensoryInput dataclass."""

    def test_creation(self):
        inp = _make_input()
        self.assertEqual(len(inp.channels), 1)
        self.assertAlmostEqual(inp.ambient_arousal, 0.5)

    def test_to_dict(self):
        inp = _make_input()
        d = inp.to_dict()
        self.assertIn("channels", d)
        self.assertIn("ambient_arousal", d)


class TestCategorizedPercept(unittest.TestCase):
    """Tests for CategorizedPercept dataclass."""

    def test_creation(self):
        p = CategorizedPercept(
            percept_id="p001",
            category=PerceptualCategory.OBJECT,
            value=ValueAssignment.NEUTRAL,
            salience=0.6,
            source_channels=["visual"],
        )
        self.assertEqual(p.percept_id, "p001")
        self.assertEqual(p.category, PerceptualCategory.OBJECT)

    def test_to_dict(self):
        p = CategorizedPercept(
            percept_id="p002",
            category=PerceptualCategory.THREAT_CUE,
            value=ValueAssignment.STRONGLY_AVERSIVE,
            salience=0.9,
            source_channels=["visual", "auditory"],
        )
        d = p.to_dict()
        self.assertEqual(d["category"], "threat_cue")
        self.assertEqual(d["value"], "strongly_aversive")


class TestRememberedPresent(unittest.TestCase):
    """Tests for RememberedPresent dataclass."""

    def test_creation(self):
        rp = RememberedPresent(
            scene_id="s001",
            percepts=[],
            scene_coherence=SceneCoherence.COHERENT,
            binding_state=SensoryBoundState.BOUND,
            dominant_value=ValueAssignment.NEUTRAL,
            memory_influence=0.6,
            novelty_fraction=0.1,
            temporal_continuity=0.8,
        )
        self.assertEqual(rp.scene_id, "s001")
        self.assertEqual(rp.scene_coherence, SceneCoherence.COHERENT)

    def test_to_dict(self):
        rp = RememberedPresent(
            scene_id="s002",
            percepts=[],
            scene_coherence=SceneCoherence.VIVID,
            binding_state=SensoryBoundState.TIGHTLY_BOUND,
            dominant_value=ValueAssignment.MILDLY_APPETITIVE,
            memory_influence=0.7,
            novelty_fraction=0.05,
            temporal_continuity=0.9,
        )
        d = rp.to_dict()
        self.assertEqual(d["scene_coherence"], "vivid")
        self.assertEqual(d["binding_state"], "tightly_bound")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestPrimaryConsciousnessInterface(unittest.TestCase):
    """Tests for PrimaryConsciousnessInterface class."""

    def _make_interface(self):
        iface = PrimaryConsciousnessInterface()
        run_async(iface.initialize())
        return iface

    def test_form_metadata(self):
        iface = PrimaryConsciousnessInterface()
        self.assertEqual(iface.FORM_ID, "18-primary-consciousness")
        self.assertIn("Edelman", iface.FORM_NAME)

    def test_initialize(self):
        iface = self._make_interface()
        self.assertTrue(iface._initialized)

    def test_categorize_input_basic(self):
        iface = self._make_interface()
        inp = _make_input(channels=[
            _make_channel("visual", intensity=0.6, novelty=0.2),
        ])
        percepts = run_async(iface.categorize_input(inp))
        self.assertGreaterEqual(len(percepts), 1)
        self.assertIsInstance(percepts[0], CategorizedPercept)

    def test_categorize_threat_cue(self):
        iface = self._make_interface()
        inp = _make_input(channels=[
            _make_channel("visual", intensity=0.9, novelty=0.9),
        ])
        percepts = run_async(iface.categorize_input(inp))
        self.assertTrue(len(percepts) >= 1)
        # High novelty + high intensity should yield threat cue
        self.assertEqual(percepts[0].category, PerceptualCategory.THREAT_CUE)

    def test_construct_scene(self):
        iface = self._make_interface()
        inp = _make_input(channels=[
            _make_channel("visual", intensity=0.6),
            _make_channel("auditory", intensity=0.5),
        ])
        percepts = run_async(iface.categorize_input(inp))
        scene = run_async(iface.construct_scene(percepts))
        self.assertIsInstance(scene, RememberedPresent)
        self.assertIn(scene.scene_coherence, list(SceneCoherence))

    def test_remembered_present_after_scene(self):
        iface = self._make_interface()
        inp = _make_input(channels=[_make_channel("visual", intensity=0.6)])
        percepts = run_async(iface.categorize_input(inp))
        run_async(iface.construct_scene(percepts))
        rp = run_async(iface.get_remembered_present())
        self.assertIsNotNone(rp)

    def test_remembered_present_before_scene(self):
        iface = self._make_interface()
        rp = run_async(iface.get_remembered_present())
        self.assertIsNone(rp)

    def test_assess_awareness_absent_no_scene(self):
        iface = self._make_interface()
        level = run_async(iface.assess_primary_awareness())
        self.assertEqual(level, PrimaryAwarenessLevel.ABSENT)

    def test_assess_awareness_after_scene(self):
        iface = self._make_interface()
        inp = _make_input(channels=[
            _make_channel("visual", intensity=0.7),
            _make_channel("auditory", intensity=0.6),
            _make_channel("tactile", intensity=0.5),
        ], arousal=0.7)
        percepts = run_async(iface.categorize_input(inp))
        run_async(iface.construct_scene(percepts))
        level = run_async(iface.assess_primary_awareness())
        self.assertIn(level, [
            PrimaryAwarenessLevel.COHERENT,
            PrimaryAwarenessLevel.HEIGHTENED,
            PrimaryAwarenessLevel.PARTIAL,
        ])

    def test_to_dict(self):
        iface = self._make_interface()
        d = iface.to_dict()
        self.assertEqual(d["form_id"], "18-primary-consciousness")
        self.assertIn("scene_count", d)
        self.assertIn("percept_count", d)

    def test_get_status(self):
        iface = self._make_interface()
        status = iface.get_status()
        self.assertEqual(status["form_id"], "18-primary-consciousness")
        self.assertIn("awareness_level", status)

    def test_temporal_continuity_across_scenes(self):
        iface = self._make_interface()
        # First scene
        inp1 = _make_input(channels=[_make_channel("visual", intensity=0.6)])
        p1 = run_async(iface.categorize_input(inp1))
        s1 = run_async(iface.construct_scene(p1))

        # Second scene with same channel
        inp2 = _make_input(channels=[_make_channel("visual", intensity=0.5)])
        p2 = run_async(iface.categorize_input(inp2))
        s2 = run_async(iface.construct_scene(p2))

        self.assertGreater(s2.temporal_continuity, 0.0)

    def test_empty_input_yields_no_percepts(self):
        iface = self._make_interface()
        inp = _make_input(channels=[])
        percepts = run_async(iface.categorize_input(inp))
        self.assertEqual(len(percepts), 0)

    def test_convenience_factory(self):
        iface = create_primary_consciousness_interface()
        self.assertIsInstance(iface, PrimaryConsciousnessInterface)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
