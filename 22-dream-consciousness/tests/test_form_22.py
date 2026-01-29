#!/usr/bin/env python3
"""
Test suite for Form 22: Dream Consciousness

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- Dream generation across sleep stages
- Content analysis
- Bizarreness computation
- Sleep stage transitions
- Dream type classification
- Edge cases (wake state, no memories, N3 minimal dreaming)
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
    "dream_consciousness_interface",
    INTERFACE_PATH / "dream_consciousness_interface.py",
)

# Import all components
SleepStage = interface_module.SleepStage
DreamType = interface_module.DreamType
DreamEmotion = interface_module.DreamEmotion
BizarrenessSource = interface_module.BizarrenessSource
DreamGenerationModel = interface_module.DreamGenerationModel
SleepStateInput = interface_module.SleepStateInput
RecentMemory = interface_module.RecentMemory
DreamInput = interface_module.DreamInput
DreamElement = interface_module.DreamElement
DreamOutput = interface_module.DreamOutput
DreamConsciousnessInterface = interface_module.DreamConsciousnessInterface
create_dream_consciousness_interface = interface_module.create_dream_consciousness_interface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_sleep_state(
    stage=SleepStage.REM,
    time_min=15.0,
    cycle=2,
    cortisol=0.3,
    melatonin=0.8,
    delta=0.1,
    theta=0.4,
    eye_density=0.7,
    atonia=True,
):
    return SleepStateInput(
        stage=stage,
        time_in_stage_minutes=time_min,
        cycle_number=cycle,
        cortisol_level=cortisol,
        melatonin_level=melatonin,
        eeg_power_delta=delta,
        eeg_power_theta=theta,
        eye_movement_density=eye_density,
        muscle_atonia=atonia,
    )


def _make_memory(
    memory_id="mem_001",
    content="Walking through a forest",
    charge=0.0,
    hours=4.0,
    importance=0.5,
    is_threat=False,
    is_unresolved=False,
):
    return RecentMemory(
        memory_id=memory_id,
        content=content,
        emotional_charge=charge,
        recency_hours=hours,
        importance=importance,
        is_threat=is_threat,
        is_unresolved=is_unresolved,
    )


def _make_dream_input(
    stage=SleepStage.REM,
    memories=None,
    model=DreamGenerationModel.ACTIVATION_SYNTHESIS,
    stress=0.3,
    creativity=0.5,
):
    sleep = _make_sleep_state(stage=stage)
    if memories is None:
        memories = [
            _make_memory("m1", "Walking in the park"),
            _make_memory("m2", "Talking to a friend"),
            _make_memory("m3", "Reading a book"),
        ]
    return DreamInput(
        sleep_state=sleep,
        recent_memories=memories,
        generation_model=model,
        stress_level=stress,
        creativity_factor=creativity,
    )


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestSleepStage(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(list(SleepStage)), 5)

    def test_expected_members(self):
        names = {m.name for m in SleepStage}
        for expected in ["WAKE", "N1", "N2", "N3", "REM"]:
            self.assertIn(expected, names)

    def test_values_are_strings(self):
        for s in SleepStage:
            self.assertIsInstance(s.value, str)


class TestDreamType(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(DreamType)), 8)

    def test_expected_members(self):
        names = {m.name for m in DreamType}
        for expected in ["NARRATIVE", "BIZARRE", "NIGHTMARE", "PROPHETIC",
                         "LUCID", "HYPNAGOGIC", "FRAGMENTARY"]:
            self.assertIn(expected, names)


class TestDreamEmotion(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(DreamEmotion)), 9)

    def test_expected_members(self):
        names = {m.name for m in DreamEmotion}
        for expected in ["FEAR", "ANXIETY", "JOY", "NEUTRAL", "AWE"]:
            self.assertIn(expected, names)


class TestBizarrenessSource(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(BizarrenessSource)), 7)

    def test_expected_members(self):
        names = {m.name for m in BizarrenessSource}
        for expected in ["SPATIAL", "TEMPORAL", "IDENTITY", "PHYSICAL"]:
            self.assertIn(expected, names)


class TestDreamGenerationModel(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(DreamGenerationModel)), 6)

    def test_expected_members(self):
        names = {m.name for m in DreamGenerationModel}
        for expected in ["ACTIVATION_SYNTHESIS", "THREAT_SIMULATION",
                         "MEMORY_CONSOLIDATION"]:
            self.assertIn(expected, names)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSleepStateInput(unittest.TestCase):
    def test_creation(self):
        s = _make_sleep_state()
        self.assertEqual(s.stage, SleepStage.REM)
        self.assertTrue(s.muscle_atonia)

    def test_to_dict(self):
        s = _make_sleep_state(stage=SleepStage.N3)
        d = s.to_dict()
        self.assertEqual(d["stage"], "N3")
        self.assertIn("eeg_power_delta", d)


class TestRecentMemory(unittest.TestCase):
    def test_creation(self):
        m = _make_memory()
        self.assertEqual(m.memory_id, "mem_001")
        self.assertFalse(m.is_threat)

    def test_to_dict(self):
        m = _make_memory(is_threat=True, charge=-0.7)
        d = m.to_dict()
        self.assertTrue(d["is_threat"])
        self.assertAlmostEqual(d["emotional_charge"], -0.7)


class TestDreamInput(unittest.TestCase):
    def test_creation(self):
        di = _make_dream_input()
        self.assertEqual(di.sleep_state.stage, SleepStage.REM)
        self.assertEqual(len(di.recent_memories), 3)

    def test_to_dict(self):
        di = _make_dream_input()
        d = di.to_dict()
        self.assertEqual(d["sleep_stage"], "REM")
        self.assertEqual(d["memory_count"], 3)


class TestDreamElement(unittest.TestCase):
    def test_creation(self):
        e = DreamElement(
            element_id="e001",
            description="Flying over mountains",
            source_memory=None,
            bizarreness=0.8,
            bizarreness_sources=[BizarrenessSource.PHYSICAL],
            emotional_tone=DreamEmotion.AWE,
            vividness=0.9,
        )
        self.assertAlmostEqual(e.bizarreness, 0.8)

    def test_to_dict(self):
        e = DreamElement(
            element_id="e002",
            description="Talking to a cat",
            source_memory="m1",
            bizarreness=0.5,
            bizarreness_sources=[BizarrenessSource.IDENTITY],
            emotional_tone=DreamEmotion.SURPRISE,
            vividness=0.6,
        )
        d = e.to_dict()
        self.assertEqual(d["source_memory"], "m1")
        self.assertIn("identity", d["bizarreness_sources"])


class TestDreamOutput(unittest.TestCase):
    def test_creation(self):
        do = DreamOutput(
            dream_id="d001",
            dream_type=DreamType.NARRATIVE,
            sleep_stage=SleepStage.REM,
            elements=[],
            primary_emotion=DreamEmotion.NEUTRAL,
            emotional_intensity=0.3,
            bizarreness_score=0.4,
            narrative_coherence=0.6,
            vividness=0.7,
            lucidity=0.1,
            memory_sources_used=2,
        )
        self.assertEqual(do.dream_type, DreamType.NARRATIVE)

    def test_to_dict(self):
        do = DreamOutput(
            dream_id="d002",
            dream_type=DreamType.NIGHTMARE,
            sleep_stage=SleepStage.REM,
            elements=[],
            primary_emotion=DreamEmotion.FEAR,
            emotional_intensity=0.9,
            bizarreness_score=0.6,
            narrative_coherence=0.4,
            vividness=0.8,
            lucidity=0.0,
            memory_sources_used=1,
        )
        d = do.to_dict()
        self.assertEqual(d["dream_type"], "nightmare")
        self.assertEqual(d["primary_emotion"], "fear")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestDreamConsciousnessInterface(unittest.TestCase):

    def _make_interface(self):
        iface = DreamConsciousnessInterface()
        run_async(iface.initialize())
        return iface

    def test_form_metadata(self):
        iface = DreamConsciousnessInterface()
        self.assertEqual(iface.FORM_ID, "22-dream-consciousness")
        self.assertIn("Dream", iface.FORM_NAME)

    def test_initialize(self):
        iface = self._make_interface()
        self.assertTrue(iface._initialized)

    def test_generate_dream_rem(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.REM)
        dream = run_async(iface.generate_dream(di))
        self.assertIsInstance(dream, DreamOutput)
        self.assertEqual(dream.sleep_stage, SleepStage.REM)
        self.assertGreaterEqual(len(dream.elements), 1)
        self.assertGreater(dream.vividness, 0.0)

    def test_generate_dream_n1_hypnagogic(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.N1)
        dream = run_async(iface.generate_dream(di))
        self.assertEqual(dream.dream_type, DreamType.HYPNAGOGIC)

    def test_generate_dream_n3_minimal(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.N3, memories=[])
        dream = run_async(iface.generate_dream(di))
        # N3 should have very few elements and low vividness
        self.assertLessEqual(len(dream.elements), 2)
        self.assertLess(dream.vividness, 0.5)

    def test_generate_dream_wake_no_elements(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.WAKE)
        dream = run_async(iface.generate_dream(di))
        self.assertEqual(len(dream.elements), 0)

    def test_nightmare_from_threat_memories(self):
        iface = self._make_interface()
        memories = [
            _make_memory("t1", "Being chased", charge=-0.8, is_threat=True),
            _make_memory("t2", "Falling from height", charge=-0.9, is_threat=True),
            _make_memory("t3", "Dark room", charge=-0.6, is_threat=True),
            _make_memory("t4", "Loud noise", charge=-0.7, is_threat=True),
            _make_memory("t5", "Trapped", charge=-0.8, is_threat=True),
        ]
        di = _make_dream_input(stage=SleepStage.REM, memories=memories, stress=0.9)
        dream = run_async(iface.generate_dream(di))
        self.assertEqual(dream.dream_type, DreamType.NIGHTMARE)
        self.assertEqual(dream.primary_emotion, DreamEmotion.FEAR)

    def test_analyze_content(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.REM)
        dream = run_async(iface.generate_dream(di))
        analysis = run_async(iface.analyze_content(dream))
        self.assertIn("dream_id", analysis)
        self.assertIn("emotion_distribution", analysis)
        self.assertIn("bizarreness_source_distribution", analysis)
        self.assertIn("memory_incorporation_rate", analysis)

    def test_compute_bizarreness(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.REM, creativity=0.8)
        dream = run_async(iface.generate_dream(di))
        biz = run_async(iface.compute_bizarreness(dream))
        self.assertIn("overall_score", biz)
        self.assertIn("element_scores", biz)
        self.assertIn("source_breakdown", biz)
        self.assertGreaterEqual(biz["overall_score"], 0.0)

    def test_compute_bizarreness_empty_dream(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.WAKE)
        dream = run_async(iface.generate_dream(di))
        biz = run_async(iface.compute_bizarreness(dream))
        self.assertAlmostEqual(biz["overall_score"], 0.0)

    def test_get_sleep_stage(self):
        iface = self._make_interface()
        stage = run_async(iface.get_sleep_stage())
        self.assertEqual(stage, SleepStage.WAKE)

    def test_transition_stage(self):
        iface = self._make_interface()
        result = iface.transition_stage(SleepStage.N1)
        self.assertEqual(result["from_stage"], "wake")
        self.assertEqual(result["to_stage"], "N1")

        result2 = iface.transition_stage(SleepStage.REM)
        self.assertEqual(result2["from_stage"], "N1")
        self.assertEqual(result2["to_stage"], "REM")
        self.assertEqual(result2["cycle_number"], 1)

    def test_dream_history(self):
        iface = self._make_interface()
        di = _make_dream_input(stage=SleepStage.REM)
        run_async(iface.generate_dream(di))
        run_async(iface.generate_dream(di))
        self.assertEqual(iface._dream_counter, 2)
        self.assertEqual(len(iface._dream_history), 2)

    def test_to_dict(self):
        iface = self._make_interface()
        d = iface.to_dict()
        self.assertEqual(d["form_id"], "22-dream-consciousness")
        self.assertIn("current_stage", d)
        self.assertIn("dreams_generated", d)

    def test_get_status(self):
        iface = self._make_interface()
        status = iface.get_status()
        self.assertEqual(status["form_id"], "22-dream-consciousness")
        self.assertIn("sleep_stage", status)
        self.assertIn("cycle_number", status)

    def test_convenience_factory(self):
        iface = create_dream_consciousness_interface()
        self.assertIsInstance(iface, DreamConsciousnessInterface)

    def test_vividness_increases_with_cycles(self):
        """Later REM cycles should produce more vivid dreams."""
        iface = self._make_interface()
        sleep_early = _make_sleep_state(stage=SleepStage.REM, cycle=1)
        sleep_late = _make_sleep_state(stage=SleepStage.REM, cycle=5)

        di_early = DreamInput(sleep_state=sleep_early, recent_memories=[_make_memory()])
        di_late = DreamInput(sleep_state=sleep_late, recent_memories=[_make_memory()])

        dream_early = run_async(iface.generate_dream(di_early))
        dream_late = run_async(iface.generate_dream(di_late))

        self.assertGreaterEqual(dream_late.vividness, dream_early.vividness)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
