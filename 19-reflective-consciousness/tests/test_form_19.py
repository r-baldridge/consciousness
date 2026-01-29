#!/usr/bin/env python3
"""
Test suite for Form 19: Reflective Consciousness

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- Interface reflection, evaluation, insight, and deliberation
- Strategy selection and depth computation
- Edge cases and state tracking
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
    "reflective_consciousness_interface",
    INTERFACE_PATH / "reflective_consciousness_interface.py",
)

# Import all components
ReflectionType = interface_module.ReflectionType
ReflectionDepth = interface_module.ReflectionDepth
CognitiveStrategy = interface_module.CognitiveStrategy
InsightQuality = interface_module.InsightQuality
ReflectiveState = interface_module.ReflectiveState
ThoughtContent = interface_module.ThoughtContent
ReflectiveContext = interface_module.ReflectiveContext
ReflectiveInput = interface_module.ReflectiveInput
ReflectiveInsight = interface_module.ReflectiveInsight
DeliberationOption = interface_module.DeliberationOption
ReflectiveOutput = interface_module.ReflectiveOutput
ReflectiveConsciousnessInterface = interface_module.ReflectiveConsciousnessInterface
create_reflective_consciousness_interface = interface_module.create_reflective_consciousness_interface


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


def _make_thought(content_type="belief", description="Test thought",
                  confidence=0.7, emotional_charge=0.0):
    return ThoughtContent(
        content_id="tc_001",
        content_type=content_type,
        description=description,
        confidence=confidence,
        emotional_charge=emotional_charge,
        source="test",
    )


def _make_input(thought=None, depth=ReflectionDepth.MODERATE,
                reflection_type=ReflectionType.INTROSPECTIVE,
                related=None):
    if thought is None:
        thought = _make_thought()
    ctx = ReflectiveContext(
        goal="Test reflection",
        depth_requested=depth,
    )
    return ReflectiveInput(
        thought=thought,
        context=ctx,
        reflection_type=reflection_type,
        related_thoughts=related or [],
    )


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestReflectionType(unittest.TestCase):
    """Tests for ReflectionType enum."""

    def test_count(self):
        self.assertEqual(len(list(ReflectionType)), 6)

    def test_expected_members(self):
        names = {m.name for m in ReflectionType}
        for expected in ["INTROSPECTIVE", "DELIBERATIVE", "EVALUATIVE",
                         "METACOGNITIVE", "COUNTERFACTUAL", "PROSPECTIVE"]:
            self.assertIn(expected, names)

    def test_values_are_strings(self):
        for rt in ReflectionType:
            self.assertIsInstance(rt.value, str)


class TestReflectionDepth(unittest.TestCase):
    """Tests for ReflectionDepth enum."""

    def test_count(self):
        self.assertEqual(len(list(ReflectionDepth)), 5)

    def test_ordering_names(self):
        names = [m.name for m in ReflectionDepth]
        self.assertEqual(names[0], "SURFACE")
        self.assertEqual(names[-1], "PROFOUND")


class TestCognitiveStrategy(unittest.TestCase):
    """Tests for CognitiveStrategy enum."""

    def test_count(self):
        self.assertGreaterEqual(len(list(CognitiveStrategy)), 7)

    def test_expected_members(self):
        names = {m.name for m in CognitiveStrategy}
        for expected in ["ANALYTIC", "INTUITIVE", "COMPARATIVE", "DIALECTICAL"]:
            self.assertIn(expected, names)


class TestInsightQuality(unittest.TestCase):
    """Tests for InsightQuality enum."""

    def test_count(self):
        self.assertEqual(len(list(InsightQuality)), 5)

    def test_range(self):
        names = {m.name for m in InsightQuality}
        self.assertIn("TRIVIAL", names)
        self.assertIn("TRANSFORMATIVE", names)


class TestReflectiveState(unittest.TestCase):
    """Tests for ReflectiveState enum."""

    def test_count(self):
        self.assertEqual(len(list(ReflectiveState)), 5)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestThoughtContent(unittest.TestCase):
    """Tests for ThoughtContent dataclass."""

    def test_creation(self):
        t = _make_thought()
        self.assertEqual(t.content_type, "belief")
        self.assertAlmostEqual(t.confidence, 0.7)

    def test_to_dict(self):
        t = _make_thought(description="I believe X")
        d = t.to_dict()
        self.assertEqual(d["content_type"], "belief")
        self.assertIn("description", d)
        self.assertIn("timestamp", d)


class TestReflectiveInput(unittest.TestCase):
    """Tests for ReflectiveInput dataclass."""

    def test_creation(self):
        inp = _make_input()
        self.assertIsInstance(inp.thought, ThoughtContent)
        self.assertEqual(inp.reflection_type, ReflectionType.INTROSPECTIVE)

    def test_to_dict(self):
        inp = _make_input()
        d = inp.to_dict()
        self.assertIn("thought", d)
        self.assertIn("reflection_type", d)


class TestReflectiveInsight(unittest.TestCase):
    """Tests for ReflectiveInsight dataclass."""

    def test_creation(self):
        insight = ReflectiveInsight(
            insight_id="ins_001",
            content="Test insight",
            quality=InsightQuality.SIGNIFICANT,
            confidence=0.7,
            source_reflection="refl_001",
        )
        self.assertEqual(insight.quality, InsightQuality.SIGNIFICANT)

    def test_to_dict(self):
        insight = ReflectiveInsight(
            insight_id="ins_002",
            content="Another insight",
            quality=InsightQuality.BREAKTHROUGH,
            confidence=0.9,
            source_reflection="refl_002",
        )
        d = insight.to_dict()
        self.assertEqual(d["quality"], "breakthrough")


class TestDeliberationOption(unittest.TestCase):
    """Tests for DeliberationOption dataclass."""

    def test_creation(self):
        opt = DeliberationOption(
            option_id="opt_001",
            description="Option A",
            pros=["fast", "cheap"],
            cons=["risky"],
            estimated_value=0.7,
            feasibility=0.8,
            risk=0.3,
        )
        self.assertEqual(opt.option_id, "opt_001")
        self.assertEqual(len(opt.pros), 2)


class TestReflectiveOutput(unittest.TestCase):
    """Tests for ReflectiveOutput dataclass."""

    def test_creation(self):
        out = ReflectiveOutput(
            reflection_id="refl_001",
            reflection_type=ReflectionType.EVALUATIVE,
            depth_achieved=ReflectionDepth.MODERATE,
            strategy_used=CognitiveStrategy.ANALYTIC,
            insight=None,
            evaluation_result="adequate",
            decision=None,
        )
        self.assertEqual(out.reflection_type, ReflectionType.EVALUATIVE)

    def test_to_dict(self):
        out = ReflectiveOutput(
            reflection_id="refl_002",
            reflection_type=ReflectionType.DELIBERATIVE,
            depth_achieved=ReflectionDepth.DEEP,
            strategy_used=CognitiveStrategy.COMPARATIVE,
            insight=None,
            evaluation_result=None,
            decision="Choose A",
        )
        d = out.to_dict()
        self.assertEqual(d["reflection_type"], "deliberative")
        self.assertEqual(d["decision"], "Choose A")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestReflectiveConsciousnessInterface(unittest.TestCase):
    """Tests for ReflectiveConsciousnessInterface class."""

    def _make_interface(self):
        iface = ReflectiveConsciousnessInterface()
        run_async(iface.initialize())
        return iface

    def test_form_metadata(self):
        iface = ReflectiveConsciousnessInterface()
        self.assertEqual(iface.FORM_ID, "19-reflective-consciousness")
        self.assertIn("Reflective", iface.FORM_NAME)

    def test_initialize(self):
        iface = self._make_interface()
        self.assertTrue(iface._initialized)

    def test_reflect_on_state(self):
        iface = self._make_interface()
        inp = _make_input()
        output = run_async(iface.reflect_on_state(inp))
        self.assertIsInstance(output, ReflectiveOutput)
        self.assertEqual(output.reflection_type, ReflectionType.INTROSPECTIVE)
        self.assertIsNotNone(output.insight)

    def test_evaluate_thought(self):
        iface = self._make_interface()
        inp = _make_input(reflection_type=ReflectionType.EVALUATIVE)
        output = run_async(iface.evaluate_thought(inp))
        self.assertIsInstance(output, ReflectiveOutput)
        self.assertEqual(output.reflection_type, ReflectionType.EVALUATIVE)
        self.assertIsNotNone(output.evaluation_result)
        self.assertIn(output.evaluation_result,
                       ["high_quality", "adequate", "questionable", "unreliable"])

    def test_generate_insight(self):
        iface = self._make_interface()
        inp = _make_input(depth=ReflectionDepth.DEEP)
        insight = run_async(iface.generate_insight(inp))
        # May or may not produce an insight depending on depth achievable
        if insight is not None:
            self.assertIsInstance(insight, ReflectiveInsight)
            self.assertIn(insight.quality, list(InsightQuality))

    def test_deliberate(self):
        iface = self._make_interface()
        options = [
            DeliberationOption(
                option_id="opt_a",
                description="Option A",
                pros=["fast", "cheap"],
                cons=["risky"],
                estimated_value=0.8,
                feasibility=0.9,
                risk=0.2,
            ),
            DeliberationOption(
                option_id="opt_b",
                description="Option B",
                pros=["safe"],
                cons=["slow", "expensive"],
                estimated_value=0.5,
                feasibility=0.7,
                risk=0.1,
            ),
        ]
        ctx = ReflectiveContext(
            goal="Choose best option",
            depth_requested=ReflectionDepth.MODERATE,
        )
        output = run_async(iface.deliberate(options, ctx))
        self.assertIsInstance(output, ReflectiveOutput)
        self.assertEqual(output.reflection_type, ReflectionType.DELIBERATIVE)
        self.assertIsNotNone(output.decision)
        self.assertIn("Selected:", output.decision)

    def test_deliberate_empty_options(self):
        iface = self._make_interface()
        ctx = ReflectiveContext(goal="Choose", depth_requested=ReflectionDepth.SHALLOW)
        output = run_async(iface.deliberate([], ctx))
        self.assertIn("No viable", output.decision)

    def test_to_dict(self):
        iface = self._make_interface()
        d = iface.to_dict()
        self.assertEqual(d["form_id"], "19-reflective-consciousness")
        self.assertIn("reflection_count", d)

    def test_get_status(self):
        iface = self._make_interface()
        status = iface.get_status()
        self.assertEqual(status["form_id"], "19-reflective-consciousness")
        self.assertIn("reflective_state", status)

    def test_reflection_increments_counter(self):
        iface = self._make_interface()
        inp = _make_input()
        run_async(iface.reflect_on_state(inp))
        run_async(iface.reflect_on_state(inp))
        self.assertEqual(iface._reflection_counter, 2)

    def test_state_transitions(self):
        iface = self._make_interface()
        self.assertEqual(iface._state, ReflectiveState.IDLE)
        inp = _make_input()
        run_async(iface.reflect_on_state(inp))
        # After reflection completes, state should be INTEGRATING
        self.assertEqual(iface._state, ReflectiveState.INTEGRATING)

    def test_convenience_factory(self):
        iface = create_reflective_consciousness_interface()
        self.assertIsInstance(iface, ReflectiveConsciousnessInterface)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
