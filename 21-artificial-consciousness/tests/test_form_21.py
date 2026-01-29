#!/usr/bin/env python3
"""
Test suite for Form 21: Artificial Consciousness

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- System evaluation pipeline
- Individual consciousness tests
- Functional marker assessment
- Architecture comparison
- Verdict and confidence determination
- Edge cases (no behavioral data, minimal system)
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
    "artificial_consciousness_interface",
    INTERFACE_PATH / "artificial_consciousness_interface.py",
)

# Import all components
ACArchitecture = interface_module.ACArchitecture
ConsciousnessTest = interface_module.ConsciousnessTest
FunctionalMarker = interface_module.FunctionalMarker
AssessmentConfidence = interface_module.AssessmentConfidence
ConsciousnessVerdict = interface_module.ConsciousnessVerdict
SystemState = interface_module.SystemState
BehavioralData = interface_module.BehavioralData
ACInput = interface_module.ACInput
TestResult = interface_module.TestResult
MarkerAssessment = interface_module.MarkerAssessment
ArchitectureComparison = interface_module.ArchitectureComparison
ACOutput = interface_module.ACOutput
ArtificialConsciousnessInterface = interface_module.ArtificialConsciousnessInterface
create_artificial_consciousness_interface = interface_module.create_artificial_consciousness_interface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_system_state(
    system_id="sys_001",
    architecture=ACArchitecture.GLOBAL_WORKSPACE,
    components=10,
    integration=0.7,
    capacity=0.6,
    recurrence=3,
    self_model=True,
    world_model=True,
    attention=True,
):
    return SystemState(
        system_id=system_id,
        architecture=architecture,
        component_count=components,
        integration_level=integration,
        information_capacity=capacity,
        recurrence_depth=recurrence,
        has_self_model=self_model,
        has_world_model=world_model,
        has_attention=attention,
    )


def _make_behavioral_data(
    system_id="sys_001",
    flexibility=0.7,
    error_rate=0.6,
    self_report=0.7,
    learning=0.6,
    context=0.7,
    surprise=0.5,
):
    return BehavioralData(
        system_id=system_id,
        task_description="General evaluation",
        response_flexibility=flexibility,
        error_detection_rate=error_rate,
        self_report_coherence=self_report,
        learning_rate=learning,
        context_sensitivity=context,
        surprise_modulation=surprise,
    )


def _make_ac_input(tests=None, include_behavior=True):
    state = _make_system_state()
    behavior = _make_behavioral_data() if include_behavior else None
    if tests is None:
        tests = [ConsciousnessTest.REPORTABILITY, ConsciousnessTest.SELF_MODEL,
                 ConsciousnessTest.GLOBAL_ACCESS]
    return ACInput(
        system_state=state,
        behavioral_data=behavior,
        tests_requested=tests,
    )


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestACArchitecture(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(ACArchitecture)), 8)

    def test_expected_members(self):
        names = {m.name for m in ACArchitecture}
        for expected in ["GLOBAL_WORKSPACE", "ATTENTION_SCHEMA", "PREDICTIVE",
                         "INTEGRATED_INFORMATION", "HIGHER_ORDER"]:
            self.assertIn(expected, names)

    def test_values_are_strings(self):
        for a in ACArchitecture:
            self.assertIsInstance(a.value, str)


class TestConsciousnessTest(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(ConsciousnessTest)), 10)

    def test_expected_members(self):
        names = {m.name for m in ConsciousnessTest}
        for expected in ["REPORTABILITY", "SELF_MODEL", "METACOGNITION",
                         "UNITY", "BEHAVIORAL_FLEXIBILITY"]:
            self.assertIn(expected, names)


class TestFunctionalMarker(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(FunctionalMarker)), 10)

    def test_expected_members(self):
        names = {m.name for m in FunctionalMarker}
        for expected in ["SELECTIVE_ATTENTION", "WORKING_MEMORY",
                         "ERROR_MONITORING", "CREATIVITY"]:
            self.assertIn(expected, names)


class TestAssessmentConfidence(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(list(AssessmentConfidence)), 5)


class TestConsciousnessVerdict(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(list(ConsciousnessVerdict)), 5)

    def test_expected_members(self):
        names = {m.name for m in ConsciousnessVerdict}
        self.assertIn("NO_EVIDENCE", names)
        self.assertIn("INDETERMINATE", names)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSystemState(unittest.TestCase):
    def test_creation(self):
        s = _make_system_state()
        self.assertEqual(s.system_id, "sys_001")
        self.assertTrue(s.has_self_model)

    def test_to_dict(self):
        s = _make_system_state()
        d = s.to_dict()
        self.assertEqual(d["architecture"], "global_workspace")
        self.assertIn("integration_level", d)


class TestBehavioralData(unittest.TestCase):
    def test_creation(self):
        b = _make_behavioral_data()
        self.assertAlmostEqual(b.response_flexibility, 0.7)

    def test_to_dict(self):
        b = _make_behavioral_data()
        d = b.to_dict()
        self.assertIn("error_detection_rate", d)
        self.assertIn("self_report_coherence", d)


class TestACInput(unittest.TestCase):
    def test_creation(self):
        inp = _make_ac_input()
        self.assertEqual(len(inp.tests_requested), 3)

    def test_to_dict(self):
        inp = _make_ac_input()
        d = inp.to_dict()
        self.assertIn("system_state", d)
        self.assertIn("tests_requested", d)


class TestTestResult(unittest.TestCase):
    def test_creation(self):
        tr = TestResult(
            test=ConsciousnessTest.SELF_MODEL,
            passed=True,
            score=0.8,
            evidence="Self-model present",
            confidence=AssessmentConfidence.HIGH,
        )
        self.assertTrue(tr.passed)

    def test_to_dict(self):
        tr = TestResult(
            test=ConsciousnessTest.REPORTABILITY,
            passed=False,
            score=0.3,
            evidence="Weak reports",
            confidence=AssessmentConfidence.LOW,
        )
        d = tr.to_dict()
        self.assertEqual(d["test"], "reportability")
        self.assertFalse(d["passed"])


class TestMarkerAssessment(unittest.TestCase):
    def test_creation(self):
        ma = MarkerAssessment(
            marker=FunctionalMarker.SELECTIVE_ATTENTION,
            present=True,
            strength=0.7,
            evidence="Attention mechanism present",
        )
        self.assertTrue(ma.present)

    def test_to_dict(self):
        ma = MarkerAssessment(
            marker=FunctionalMarker.CREATIVITY,
            present=False,
            strength=0.2,
            evidence="Low flexibility",
        )
        d = ma.to_dict()
        self.assertEqual(d["marker"], "creativity")


class TestArchitectureComparison(unittest.TestCase):
    def test_creation(self):
        ac = ArchitectureComparison(
            architecture=ACArchitecture.GLOBAL_WORKSPACE,
            alignment_score=0.7,
            strengths=["High integration"],
            gaps=["Few modules"],
        )
        self.assertAlmostEqual(ac.alignment_score, 0.7)


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestArtificialConsciousnessInterface(unittest.TestCase):

    def _make_interface(self):
        iface = ArtificialConsciousnessInterface()
        run_async(iface.initialize())
        return iface

    def test_form_metadata(self):
        iface = ArtificialConsciousnessInterface()
        self.assertEqual(iface.FORM_ID, "21-artificial-consciousness")
        self.assertIn("Artificial", iface.FORM_NAME)

    def test_initialize(self):
        iface = self._make_interface()
        self.assertTrue(iface._initialized)

    def test_evaluate_system_full(self):
        iface = self._make_interface()
        inp = _make_ac_input(
            tests=[ConsciousnessTest.REPORTABILITY,
                   ConsciousnessTest.SELF_MODEL,
                   ConsciousnessTest.GLOBAL_ACCESS,
                   ConsciousnessTest.METACOGNITION],
            include_behavior=True,
        )
        output = run_async(iface.evaluate_system(inp))
        self.assertIsInstance(output, ACOutput)
        self.assertEqual(output.system_id, "sys_001")
        self.assertIn(output.verdict, list(ConsciousnessVerdict))
        self.assertGreaterEqual(len(output.test_results), 4)
        self.assertGreaterEqual(len(output.marker_assessments), 5)

    def test_evaluate_system_no_behavior(self):
        iface = self._make_interface()
        inp = _make_ac_input(
            tests=[ConsciousnessTest.SELF_MODEL, ConsciousnessTest.UNITY],
            include_behavior=False,
        )
        output = run_async(iface.evaluate_system(inp))
        self.assertIsInstance(output, ACOutput)

    def test_run_single_test(self):
        iface = self._make_interface()
        state = _make_system_state()
        behavior = _make_behavioral_data()
        result = run_async(iface.run_consciousness_test(
            ConsciousnessTest.REPORTABILITY, state, behavior
        ))
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test, ConsciousnessTest.REPORTABILITY)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_run_all_tests(self):
        iface = self._make_interface()
        state = _make_system_state()
        behavior = _make_behavioral_data()
        for test in ConsciousnessTest:
            result = run_async(iface.run_consciousness_test(test, state, behavior))
            self.assertIsInstance(result, TestResult)
            self.assertEqual(result.test, test)

    def test_assess_functional_markers(self):
        iface = self._make_interface()
        state = _make_system_state()
        behavior = _make_behavioral_data()
        assessments = run_async(iface.assess_functional_markers(state, behavior))
        self.assertGreaterEqual(len(assessments), 10)
        present_count = sum(1 for a in assessments if a.present)
        self.assertGreater(present_count, 0)

    def test_assess_markers_no_behavior(self):
        iface = self._make_interface()
        state = _make_system_state()
        assessments = run_async(iface.assess_functional_markers(state, None))
        self.assertGreaterEqual(len(assessments), 5)

    def test_compare_architectures(self):
        iface = self._make_interface()
        state = _make_system_state()
        comparisons = run_async(iface.compare_architectures(state))
        self.assertEqual(len(comparisons), len(list(ACArchitecture)))
        # Should be sorted by score descending
        scores = [c.alignment_score for c in comparisons]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_minimal_system_low_score(self):
        """A minimal system should score low."""
        iface = self._make_interface()
        state = SystemState(
            system_id="minimal",
            architecture=ACArchitecture.GLOBAL_WORKSPACE,
            component_count=1,
            integration_level=0.1,
            information_capacity=0.1,
            recurrence_depth=0,
            has_self_model=False,
            has_world_model=False,
            has_attention=False,
        )
        inp = ACInput(
            system_state=state,
            tests_requested=[ConsciousnessTest.SELF_MODEL,
                             ConsciousnessTest.GLOBAL_ACCESS],
        )
        output = run_async(iface.evaluate_system(inp))
        self.assertIn(output.verdict,
                       [ConsciousnessVerdict.NO_EVIDENCE,
                        ConsciousnessVerdict.WEAK_MARKERS])

    def test_to_dict(self):
        iface = self._make_interface()
        d = iface.to_dict()
        self.assertEqual(d["form_id"], "21-artificial-consciousness")
        self.assertIn("evaluations_performed", d)

    def test_get_status(self):
        iface = self._make_interface()
        status = iface.get_status()
        self.assertIn("available_tests", status)
        self.assertIn("available_markers", status)

    def test_convenience_factory(self):
        iface = create_artificial_consciousness_interface()
        self.assertIsInstance(iface, ArtificialConsciousnessInterface)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
