#!/usr/bin/env python3
"""
Test suite for Form 20: Collective Consciousness (Durkheim)

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- Belief aggregation and shared representation detection
- Emergence detection (effervescence, polarization, herd behavior)
- Cohesion measurement
- Meme propagation simulation
- Edge cases (empty inputs, single agent)
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
    "collective_consciousness_interface",
    INTERFACE_PATH / "collective_consciousness_interface.py",
)

# Import all components
CollectiveType = interface_module.CollectiveType
SocialCohesion = interface_module.SocialCohesion
GroupMindState = interface_module.GroupMindState
BeliefStrength = interface_module.BeliefStrength
PropagationMode = interface_module.PropagationMode
IndividualBelief = interface_module.IndividualBelief
SocialSignal = interface_module.SocialSignal
CollectiveInput = interface_module.CollectiveInput
SharedRepresentation = interface_module.SharedRepresentation
EmergentProperty = interface_module.EmergentProperty
CollectiveOutput = interface_module.CollectiveOutput
CollectiveConsciousnessInterface = interface_module.CollectiveConsciousnessInterface
create_collective_consciousness_interface = interface_module.create_collective_consciousness_interface


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


def _make_belief(agent_id="agent_1", belief_id="b1", content="Sky is blue",
                 strength=BeliefStrength.MODERATE, confidence=0.7, valence=0.0):
    return IndividualBelief(
        agent_id=agent_id,
        belief_id=belief_id,
        content=content,
        strength=strength,
        confidence=confidence,
        emotional_valence=valence,
    )


def _make_signal(signal_type="agreement", source="agent_1", intensity=0.5,
                 reach=0.5, mode=PropagationMode.CONTAGION):
    return SocialSignal(
        signal_id="sig_001",
        signal_type=signal_type,
        source_agent=source,
        intensity=intensity,
        reach=reach,
        propagation_mode=mode,
    )


def _make_input(beliefs=None, signals=None,
                collective_type=CollectiveType.COMMUNITY, group_size=10):
    return CollectiveInput(
        collective_type=collective_type,
        individual_beliefs=beliefs or [],
        social_signals=signals or [],
        group_size=group_size,
    )


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestCollectiveType(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(CollectiveType)), 7)

    def test_expected_members(self):
        names = {m.name for m in CollectiveType}
        for expected in ["CROWD", "INSTITUTION", "CULTURE", "DIGITAL", "MOVEMENT"]:
            self.assertIn(expected, names)

    def test_values_are_strings(self):
        for ct in CollectiveType:
            self.assertIsInstance(ct.value, str)


class TestSocialCohesion(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(list(SocialCohesion)), 5)

    def test_expected_members(self):
        names = {m.name for m in SocialCohesion}
        self.assertIn("FRAGMENTED", names)
        self.assertIn("TIGHTLY_INTEGRATED", names)


class TestGroupMindState(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(GroupMindState)), 7)

    def test_expected_members(self):
        names = {m.name for m in GroupMindState}
        for expected in ["DORMANT", "FOCUSED", "POLARIZED", "EUPHORIC", "PANIC"]:
            self.assertIn(expected, names)


class TestBeliefStrength(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(list(BeliefStrength)), 4)

    def test_sacred_exists(self):
        self.assertIn("SACRED", {m.name for m in BeliefStrength})


class TestPropagationMode(unittest.TestCase):
    def test_count(self):
        self.assertGreaterEqual(len(list(PropagationMode)), 7)

    def test_expected_members(self):
        names = {m.name for m in PropagationMode}
        for expected in ["CONTAGION", "PERSUASION", "VIRAL", "AUTHORITY"]:
            self.assertIn(expected, names)


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestIndividualBelief(unittest.TestCase):
    def test_creation(self):
        b = _make_belief()
        self.assertEqual(b.agent_id, "agent_1")
        self.assertEqual(b.belief_id, "b1")

    def test_to_dict(self):
        b = _make_belief(content="The earth is round")
        d = b.to_dict()
        self.assertEqual(d["content"], "The earth is round")
        self.assertIn("strength", d)


class TestSocialSignal(unittest.TestCase):
    def test_creation(self):
        s = _make_signal()
        self.assertEqual(s.signal_type, "agreement")

    def test_to_dict(self):
        s = _make_signal(signal_type="dissent", intensity=0.9)
        d = s.to_dict()
        self.assertEqual(d["signal_type"], "dissent")
        self.assertAlmostEqual(d["intensity"], 0.9)


class TestSharedRepresentation(unittest.TestCase):
    def test_creation(self):
        sr = SharedRepresentation(
            representation_id="sr_001",
            content="Shared belief",
            adoption_rate=0.8,
            strength=BeliefStrength.STRONG,
            emotional_charge=0.3,
            stability=0.7,
        )
        self.assertAlmostEqual(sr.adoption_rate, 0.8)

    def test_to_dict(self):
        sr = SharedRepresentation(
            representation_id="sr_002",
            content="Another belief",
            adoption_rate=0.9,
            strength=BeliefStrength.SACRED,
            emotional_charge=0.5,
            stability=0.9,
        )
        d = sr.to_dict()
        self.assertEqual(d["strength"], "sacred")


class TestEmergentProperty(unittest.TestCase):
    def test_creation(self):
        ep = EmergentProperty(
            property_id="ep_001",
            name="collective_effervescence",
            description="Shared emotional energy",
            intensity=0.7,
            contributing_agents=50,
        )
        self.assertEqual(ep.name, "collective_effervescence")


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestCollectiveConsciousnessInterface(unittest.TestCase):

    def _make_interface(self):
        iface = CollectiveConsciousnessInterface()
        run_async(iface.initialize())
        return iface

    def test_form_metadata(self):
        iface = CollectiveConsciousnessInterface()
        self.assertEqual(iface.FORM_ID, "20-collective-consciousness")
        self.assertIn("Durkheim", iface.FORM_NAME)

    def test_initialize(self):
        iface = self._make_interface()
        self.assertTrue(iface._initialized)

    def test_aggregate_beliefs_consensus(self):
        """Beliefs shared by enough agents should become shared representations."""
        iface = self._make_interface()
        # 7 out of 10 agents hold the same belief -> 0.7 > 0.6 threshold
        beliefs = [
            _make_belief(agent_id=f"a_{i}", belief_id="b1", content="Water is wet")
            for i in range(7)
        ]
        inp = _make_input(beliefs=beliefs, group_size=10)
        reps = run_async(iface.aggregate_beliefs(inp))
        self.assertGreaterEqual(len(reps), 1)
        self.assertGreaterEqual(reps[0].adoption_rate, 0.6)

    def test_aggregate_beliefs_no_consensus(self):
        """Beliefs held by too few agents should not become shared."""
        iface = self._make_interface()
        beliefs = [
            _make_belief(agent_id=f"a_{i}", belief_id=f"b_{i}", content=f"Unique {i}")
            for i in range(5)
        ]
        inp = _make_input(beliefs=beliefs, group_size=10)
        reps = run_async(iface.aggregate_beliefs(inp))
        self.assertEqual(len(reps), 0)

    def test_detect_emergence_effervescence(self):
        """High enthusiasm signals should trigger collective effervescence."""
        iface = self._make_interface()
        signals = [
            SocialSignal(
                signal_id=f"sig_{i}",
                signal_type="enthusiasm",
                source_agent=f"a_{i}",
                intensity=0.8,
                reach=0.7,
            )
            for i in range(5)
        ]
        inp = _make_input(signals=signals)
        props = run_async(iface.detect_emergence(inp))
        names = [p.name for p in props]
        self.assertIn("collective_effervescence", names)

    def test_detect_emergence_polarization(self):
        """Opposing emotional beliefs should trigger polarization."""
        iface = self._make_interface()
        beliefs = (
            [_make_belief(agent_id=f"pro_{i}", belief_id="issue",
                          valence=0.8) for i in range(5)] +
            [_make_belief(agent_id=f"con_{i}", belief_id="issue",
                          valence=-0.8) for i in range(5)]
        )
        inp = _make_input(beliefs=beliefs)
        props = run_async(iface.detect_emergence(inp))
        names = [p.name for p in props]
        self.assertIn("group_polarization", names)

    def test_detect_emergence_herd(self):
        """High-reach agreement signals should trigger herd behavior."""
        iface = self._make_interface()
        signals = [
            SocialSignal(
                signal_id=f"sig_{i}",
                signal_type="agreement",
                source_agent=f"a_{i}",
                intensity=0.6,
                reach=0.9,
            )
            for i in range(5)
        ]
        inp = _make_input(signals=signals)
        props = run_async(iface.detect_emergence(inp))
        names = [p.name for p in props]
        self.assertIn("herd_behavior", names)

    def test_measure_cohesion_high(self):
        """Agents sharing the same beliefs should yield high cohesion."""
        iface = self._make_interface()
        beliefs = []
        for agent in ["a1", "a2", "a3"]:
            for bid in ["b1", "b2", "b3"]:
                beliefs.append(_make_belief(agent_id=agent, belief_id=bid))
        inp = _make_input(beliefs=beliefs)
        cohesion = run_async(iface.measure_cohesion(inp))
        self.assertIn(cohesion, [SocialCohesion.TIGHTLY_INTEGRATED, SocialCohesion.COHESIVE])

    def test_measure_cohesion_low(self):
        """Agents with completely different beliefs should yield low cohesion."""
        iface = self._make_interface()
        beliefs = [
            _make_belief(agent_id=f"a_{i}", belief_id=f"unique_{i}")
            for i in range(5)
        ]
        inp = _make_input(beliefs=beliefs)
        cohesion = run_async(iface.measure_cohesion(inp))
        self.assertIn(cohesion, [SocialCohesion.FRAGMENTED, SocialCohesion.LOOSELY_CONNECTED])

    def test_measure_cohesion_empty(self):
        iface = self._make_interface()
        inp = _make_input(beliefs=[])
        cohesion = run_async(iface.measure_cohesion(inp))
        self.assertEqual(cohesion, SocialCohesion.FRAGMENTED)

    def test_propagate_meme(self):
        iface = self._make_interface()
        result = run_async(iface.propagate_meme(
            meme_content="Test idea",
            initial_adopters=5,
            group_size=100,
            propagation_mode=PropagationMode.VIRAL,
            generations=10,
        ))
        self.assertIn("meme_id", result)
        self.assertIn("adoption_rate", result)
        self.assertGreater(result["final_adopters"], 5)
        self.assertEqual(len(result["adoption_history"]), 11)  # initial + 10 gens

    def test_propagate_meme_authority(self):
        iface = self._make_interface()
        result = run_async(iface.propagate_meme(
            meme_content="Policy directive",
            initial_adopters=1,
            group_size=50,
            propagation_mode=PropagationMode.AUTHORITY,
            generations=5,
        ))
        self.assertGreater(result["adoption_rate"], 0.0)

    def test_to_dict(self):
        iface = self._make_interface()
        d = iface.to_dict()
        self.assertEqual(d["form_id"], "20-collective-consciousness")
        self.assertIn("shared_representations", d)

    def test_get_status(self):
        iface = self._make_interface()
        status = iface.get_status()
        self.assertEqual(status["form_id"], "20-collective-consciousness")
        self.assertIn("active_meme_count", status)

    def test_convenience_factory(self):
        iface = create_collective_consciousness_interface()
        self.assertIsInstance(iface, CollectiveConsciousnessInterface)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
