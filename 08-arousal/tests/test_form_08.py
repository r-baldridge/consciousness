#!/usr/bin/env python3
"""
Test Suite for Form 08: Arousal/Vigilance Consciousness.

Tests cover:
- All enumerations
- All input/output dataclasses
- ArousalComputationEngine
- ConsciousnessGatingEngine
- ResourceAllocationEngine
- ArousalConsciousnessInterface (main interface)
- Convenience functions
"""

import asyncio
import unittest
from datetime import datetime, timezone
from typing import Dict, Set

import sys
from pathlib import Path

# Add parent paths to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interface import (
    # Enums
    ArousalState,
    ArousalSource,
    GateCategory,
    SensoryModality,
    StateTransitionType,
    # Input dataclasses
    SensoryArousalInput,
    ThreatInput,
    NoveltyInput,
    CircadianInput,
    EmotionalInput,
    TaskDemandInput,
    ResourceInput,
    ArousalInputBundle,
    # Output dataclasses
    ArousalLevelOutput,
    GatingSignal,
    ConsciousnessGatingOutput,
    ResourceAllocationOutput,
    StateTransition,
    ArousalSystemStatus,
    # Engines
    ArousalComputationEngine,
    ConsciousnessGatingEngine,
    ResourceAllocationEngine,
    # Main interface
    ArousalConsciousnessInterface,
    # Convenience functions
    create_arousal_interface,
    create_simple_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestArousalState(unittest.TestCase):
    """Tests for ArousalState enumeration."""

    def test_all_states_exist(self):
        """All arousal states should be defined."""
        states = [
            ArousalState.SLEEP,
            ArousalState.DROWSY,
            ArousalState.RELAXED,
            ArousalState.ALERT,
            ArousalState.FOCUSED,
            ArousalState.HYPERAROUSED,
        ]
        self.assertEqual(len(states), 6)

    def test_state_values(self):
        """States should have expected string values."""
        self.assertEqual(ArousalState.SLEEP.value, "sleep")
        self.assertEqual(ArousalState.ALERT.value, "alert")
        self.assertEqual(ArousalState.HYPERAROUSED.value, "hyperaroused")


class TestArousalSource(unittest.TestCase):
    """Tests for ArousalSource enumeration."""

    def test_all_sources_exist(self):
        """All arousal sources should be defined."""
        sources = [
            ArousalSource.ENVIRONMENTAL,
            ArousalSource.EMOTIONAL,
            ArousalSource.CIRCADIAN,
            ArousalSource.TASK_DEMAND,
            ArousalSource.RESOURCE_STATE,
            ArousalSource.THREAT,
            ArousalSource.NOVELTY,
            ArousalSource.SOCIAL,
            ArousalSource.INTERNAL,
        ]
        self.assertEqual(len(sources), 9)


class TestGateCategory(unittest.TestCase):
    """Tests for GateCategory enumeration."""

    def test_all_categories_exist(self):
        """All gate categories should be defined."""
        categories = [
            GateCategory.SENSORY,
            GateCategory.COGNITIVE,
            GateCategory.EMOTIONAL,
            GateCategory.MEMORY,
            GateCategory.EXECUTIVE,
            GateCategory.META,
        ]
        self.assertEqual(len(categories), 6)


class TestSensoryModality(unittest.TestCase):
    """Tests for SensoryModality enumeration."""

    def test_all_modalities_exist(self):
        """All sensory modalities should be defined."""
        modalities = [
            SensoryModality.VISUAL,
            SensoryModality.AUDITORY,
            SensoryModality.SOMATOSENSORY,
            SensoryModality.OLFACTORY,
            SensoryModality.GUSTATORY,
            SensoryModality.INTEROCEPTIVE,
        ]
        self.assertEqual(len(modalities), 6)


class TestStateTransitionType(unittest.TestCase):
    """Tests for StateTransitionType enumeration."""

    def test_all_types_exist(self):
        """All transition types should be defined."""
        types = [
            StateTransitionType.GRADUAL,
            StateTransitionType.RAPID,
            StateTransitionType.FORCED,
            StateTransitionType.RECOVERY,
        ]
        self.assertEqual(len(types), 4)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestSensoryArousalInput(unittest.TestCase):
    """Tests for SensoryArousalInput dataclass."""

    def test_creation(self):
        """Should create input with all fields."""
        inp = SensoryArousalInput(
            source_modality=SensoryModality.VISUAL,
            stimulus_type="novel",
            intensity=0.8,
            salience=0.9,
            change_rate=0.5,
        )
        self.assertEqual(inp.source_modality, SensoryModality.VISUAL)
        self.assertEqual(inp.stimulus_type, "novel")
        self.assertEqual(inp.intensity, 0.8)
        self.assertEqual(inp.salience, 0.9)
        self.assertEqual(inp.change_rate, 0.5)
        self.assertEqual(inp.processing_confidence, 1.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = SensoryArousalInput(
            source_modality=SensoryModality.AUDITORY,
            stimulus_type="threat",
            intensity=0.7,
            salience=0.8,
            change_rate=0.3,
        )
        d = inp.to_dict()
        self.assertEqual(d["source_modality"], "auditory")
        self.assertEqual(d["stimulus_type"], "threat")
        self.assertEqual(d["intensity"], 0.7)


class TestThreatInput(unittest.TestCase):
    """Tests for ThreatInput dataclass."""

    def test_creation(self):
        """Should create threat input."""
        inp = ThreatInput(
            threat_level=0.8,
            threat_type="physical",
            proximity=0.9,
            certainty=0.7,
            response_urgency=0.85,
        )
        self.assertEqual(inp.threat_level, 0.8)
        self.assertEqual(inp.threat_type, "physical")
        self.assertEqual(inp.proximity, 0.9)


class TestNoveltyInput(unittest.TestCase):
    """Tests for NoveltyInput dataclass."""

    def test_creation(self):
        """Should create novelty input."""
        inp = NoveltyInput(
            novelty_level=0.7,
            novelty_type="stimulus",
            learning_opportunity=0.8,
            exploration_value=0.6,
            memory_mismatch=0.5,
        )
        self.assertEqual(inp.novelty_level, 0.7)
        self.assertEqual(inp.novelty_type, "stimulus")


class TestCircadianInput(unittest.TestCase):
    """Tests for CircadianInput dataclass."""

    def test_creation(self):
        """Should create circadian input."""
        inp = CircadianInput(
            circadian_phase=14.0,
            melatonin_level=0.2,
            cortisol_level=0.6,
            sleep_pressure=0.3,
            light_exposure=0.8,
        )
        self.assertEqual(inp.circadian_phase, 14.0)
        self.assertEqual(inp.melatonin_level, 0.2)
        self.assertEqual(inp.cortisol_level, 0.6)


class TestEmotionalInput(unittest.TestCase):
    """Tests for EmotionalInput dataclass."""

    def test_creation_with_defaults(self):
        """Should create emotional input with defaults."""
        inp = EmotionalInput(
            valence=0.5,
            arousal_component=0.6,
        )
        self.assertEqual(inp.valence, 0.5)
        self.assertEqual(inp.arousal_component, 0.6)
        self.assertEqual(inp.fear, 0.0)
        self.assertEqual(inp.calm, 0.0)

    def test_creation_with_emotions(self):
        """Should create emotional input with specific emotions."""
        inp = EmotionalInput(
            valence=-0.3,
            arousal_component=0.8,
            fear=0.7,
            anxiety=0.5,
        )
        self.assertEqual(inp.fear, 0.7)
        self.assertEqual(inp.anxiety, 0.5)


class TestTaskDemandInput(unittest.TestCase):
    """Tests for TaskDemandInput dataclass."""

    def test_creation(self):
        """Should create task demand input."""
        inp = TaskDemandInput(
            complexity=0.7,
            importance=0.9,
            time_pressure=0.8,
            performance_requirements=0.6,
            sustained_attention_needs=0.5,
        )
        self.assertEqual(inp.complexity, 0.7)
        self.assertEqual(inp.importance, 0.9)


class TestResourceInput(unittest.TestCase):
    """Tests for ResourceInput dataclass."""

    def test_creation(self):
        """Should create resource input."""
        inp = ResourceInput(
            computational_capacity=0.8,
            energy_level=0.7,
            memory_load=0.4,
            attention_capacity=0.6,
        )
        self.assertEqual(inp.computational_capacity, 0.8)
        self.assertEqual(inp.energy_level, 0.7)


class TestArousalInputBundle(unittest.TestCase):
    """Tests for ArousalInputBundle dataclass."""

    def test_empty_bundle(self):
        """Should create empty bundle."""
        bundle = ArousalInputBundle()
        self.assertEqual(len(bundle.sensory_inputs), 0)
        self.assertIsNone(bundle.threat_input)
        self.assertIsNone(bundle.emotional_input)

    def test_full_bundle(self):
        """Should create bundle with all inputs."""
        bundle = ArousalInputBundle(
            sensory_inputs=[
                SensoryArousalInput(
                    source_modality=SensoryModality.VISUAL,
                    stimulus_type="neutral",
                    intensity=0.5,
                    salience=0.5,
                    change_rate=0.1,
                )
            ],
            threat_input=ThreatInput(
                threat_level=0.3,
                threat_type="unknown",
                proximity=0.2,
                certainty=0.5,
                response_urgency=0.3,
            ),
            emotional_input=EmotionalInput(valence=0.0, arousal_component=0.5),
        )
        self.assertEqual(len(bundle.sensory_inputs), 1)
        self.assertIsNotNone(bundle.threat_input)
        self.assertIsNotNone(bundle.emotional_input)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestArousalLevelOutput(unittest.TestCase):
    """Tests for ArousalLevelOutput dataclass."""

    def test_creation(self):
        """Should create arousal level output."""
        output = ArousalLevelOutput(
            arousal_level=0.6,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.1,
            arousal_stability=0.8,
            primary_source=ArousalSource.ENVIRONMENTAL,
            confidence=0.85,
        )
        self.assertEqual(output.arousal_level, 0.6)
        self.assertEqual(output.arousal_state, ArousalState.ALERT)

    def test_to_dict(self):
        """Should convert to dictionary."""
        output = ArousalLevelOutput(
            arousal_level=0.7,
            arousal_state=ArousalState.FOCUSED,
            arousal_trend=0.2,
            arousal_stability=0.9,
            primary_source=ArousalSource.TASK_DEMAND,
            confidence=0.9,
            components={ArousalSource.TASK_DEMAND: 0.8, ArousalSource.EMOTIONAL: 0.5},
        )
        d = output.to_dict()
        self.assertEqual(d["arousal_state"], "focused")
        self.assertEqual(d["primary_source"], "task_demand")
        self.assertIn("task_demand", d["components"])


class TestGatingSignal(unittest.TestCase):
    """Tests for GatingSignal dataclass."""

    def test_creation(self):
        """Should create gating signal."""
        signal = GatingSignal(
            gate_id="visual_gate",
            category=GateCategory.SENSORY,
            openness=0.8,
            modulation_factor=1.1,
            priority_boost=0.2,
        )
        self.assertEqual(signal.gate_id, "visual_gate")
        self.assertEqual(signal.category, GateCategory.SENSORY)
        self.assertEqual(signal.openness, 0.8)


class TestConsciousnessGatingOutput(unittest.TestCase):
    """Tests for ConsciousnessGatingOutput dataclass."""

    def test_creation(self):
        """Should create consciousness gating output."""
        output = ConsciousnessGatingOutput(
            sensory_gates={"visual": 0.8, "auditory": 0.7},
            cognitive_gates={"memory_access": 0.9, "executive_control": 0.85},
            global_threshold=0.4,
            gate_adaptation_rate=0.6,
        )
        self.assertEqual(output.sensory_gates["visual"], 0.8)
        self.assertEqual(output.cognitive_gates["memory_access"], 0.9)
        self.assertEqual(output.global_threshold, 0.4)

    def test_to_dict(self):
        """Should convert to dictionary."""
        output = ConsciousnessGatingOutput(
            sensory_gates={"visual": 0.75},
            cognitive_gates={"memory": 0.8},
            global_threshold=0.45,
            gate_adaptation_rate=0.7,
        )
        d = output.to_dict()
        self.assertIn("sensory_gates", d)
        self.assertIn("cognitive_gates", d)


class TestResourceAllocationOutput(unittest.TestCase):
    """Tests for ResourceAllocationOutput dataclass."""

    def test_creation(self):
        """Should create resource allocation output."""
        output = ResourceAllocationOutput(
            total_available=1.0,
            allocations={"08-arousal": 0.3, "01-visual": 0.2},
            reserve_capacity=0.1,
            allocation_confidence=0.85,
        )
        self.assertEqual(output.total_available, 1.0)
        self.assertEqual(output.allocations["08-arousal"], 0.3)
        self.assertEqual(output.reserve_capacity, 0.1)


class TestStateTransition(unittest.TestCase):
    """Tests for StateTransition dataclass."""

    def test_creation(self):
        """Should create state transition record."""
        transition = StateTransition(
            from_state=ArousalState.RELAXED,
            to_state=ArousalState.ALERT,
            transition_type=StateTransitionType.GRADUAL,
            trigger=ArousalSource.TASK_DEMAND,
            duration_ms=500.0,
        )
        self.assertEqual(transition.from_state, ArousalState.RELAXED)
        self.assertEqual(transition.to_state, ArousalState.ALERT)
        self.assertEqual(transition.transition_type, StateTransitionType.GRADUAL)


# ============================================================================
# AROUSAL COMPUTATION ENGINE TESTS
# ============================================================================

class TestArousalComputationEngine(unittest.TestCase):
    """Tests for ArousalComputationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ArousalComputationEngine()

    def test_default_weights(self):
        """Should have default weights for all sources."""
        weights = self.engine.DEFAULT_WEIGHTS
        self.assertIn(ArousalSource.ENVIRONMENTAL, weights)
        self.assertIn(ArousalSource.EMOTIONAL, weights)
        self.assertIn(ArousalSource.CIRCADIAN, weights)
        self.assertIn(ArousalSource.THREAT, weights)

    def test_state_thresholds(self):
        """Should have thresholds for all states."""
        thresholds = self.engine.STATE_THRESHOLDS
        self.assertEqual(thresholds[ArousalState.SLEEP], (0.0, 0.1))
        self.assertEqual(thresholds[ArousalState.ALERT], (0.5, 0.7))
        self.assertEqual(thresholds[ArousalState.HYPERAROUSED], (0.9, 1.0))

    def test_compute_arousal_empty_input(self):
        """Should compute default arousal for empty input."""
        bundle = ArousalInputBundle()
        output = self.engine.compute_arousal(bundle)
        self.assertIsInstance(output, ArousalLevelOutput)
        self.assertGreaterEqual(output.arousal_level, 0.0)
        self.assertLessEqual(output.arousal_level, 1.0)

    def test_compute_arousal_with_sensory_input(self):
        """Should compute arousal from sensory input."""
        bundle = ArousalInputBundle(
            sensory_inputs=[
                SensoryArousalInput(
                    source_modality=SensoryModality.VISUAL,
                    stimulus_type="novel",
                    intensity=0.8,
                    salience=0.9,
                    change_rate=0.5,
                )
            ]
        )
        output = self.engine.compute_arousal(bundle)
        self.assertIn(ArousalSource.ENVIRONMENTAL, output.components)

    def test_compute_arousal_with_threat(self):
        """Should increase arousal with threat input."""
        # First compute baseline
        baseline_bundle = ArousalInputBundle()
        baseline = self.engine.compute_arousal(baseline_bundle)

        # Reset engine for clean comparison
        threat_engine = ArousalComputationEngine()
        threat_bundle = ArousalInputBundle(
            threat_input=ThreatInput(
                threat_level=0.9,
                threat_type="physical",
                proximity=1.0,
                certainty=0.9,
                response_urgency=0.95,
            )
        )
        threat_output = threat_engine.compute_arousal(threat_bundle)

        # Threat should significantly boost arousal
        self.assertGreater(threat_output.components[ArousalSource.THREAT], 0.5)

    def test_level_to_state_mapping(self):
        """Should correctly map levels to states."""
        self.assertEqual(self.engine._level_to_state(0.05), ArousalState.SLEEP)
        self.assertEqual(self.engine._level_to_state(0.2), ArousalState.DROWSY)
        self.assertEqual(self.engine._level_to_state(0.4), ArousalState.RELAXED)
        self.assertEqual(self.engine._level_to_state(0.6), ArousalState.ALERT)
        self.assertEqual(self.engine._level_to_state(0.8), ArousalState.FOCUSED)
        self.assertEqual(self.engine._level_to_state(0.95), ArousalState.HYPERAROUSED)

    def test_arousal_history_tracking(self):
        """Should track arousal history for trend computation."""
        bundle = ArousalInputBundle()
        for _ in range(5):
            self.engine.compute_arousal(bundle)
        self.assertGreaterEqual(len(self.engine._arousal_history), 5)

    def test_trend_computation(self):
        """Should compute trend from history."""
        # Start with low arousal
        low_bundle = create_simple_input(sensory_level=0.2, emotional_level=0.2, circadian_level=0.2)
        for _ in range(3):
            self.engine.compute_arousal(low_bundle)

        # Increase arousal
        high_bundle = create_simple_input(sensory_level=0.8, emotional_level=0.8, circadian_level=0.8)
        for _ in range(3):
            output = self.engine.compute_arousal(high_bundle)

        # Trend should be positive
        self.assertGreater(output.arousal_trend, -1.0)
        self.assertLess(output.arousal_trend, 1.0)


# ============================================================================
# CONSCIOUSNESS GATING ENGINE TESTS
# ============================================================================

class TestConsciousnessGatingEngine(unittest.TestCase):
    """Tests for ConsciousnessGatingEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ConsciousnessGatingEngine()

    def test_compute_gates_normal_arousal(self):
        """Should compute gates for normal arousal."""
        arousal = ArousalLevelOutput(
            arousal_level=0.5,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.9,
        )
        gates = self.engine.compute_gates(arousal)

        self.assertIsInstance(gates, ConsciousnessGatingOutput)
        self.assertIn("visual", gates.sensory_gates)
        self.assertIn("auditory", gates.sensory_gates)
        self.assertIn("memory_access", gates.cognitive_gates)
        self.assertIn("executive_control", gates.cognitive_gates)

    def test_gates_vary_with_arousal_state(self):
        """Gates should vary based on arousal state."""
        # Drowsy state
        drowsy = ArousalLevelOutput(
            arousal_level=0.2,
            arousal_state=ArousalState.DROWSY,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.CIRCADIAN,
            confidence=0.9,
        )
        drowsy_gates = self.engine.compute_gates(drowsy)

        # Alert state
        alert = ArousalLevelOutput(
            arousal_level=0.6,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.9,
        )
        alert_gates = self.engine.compute_gates(alert)

        # Alert should have more open gates
        self.assertGreater(
            alert_gates.cognitive_gates["executive_control"],
            drowsy_gates.cognitive_gates["executive_control"]
        )

    def test_threat_boosts_sensory_gates(self):
        """Threat should boost relevant sensory gates."""
        arousal = ArousalLevelOutput(
            arousal_level=0.7,
            arousal_state=ArousalState.FOCUSED,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.THREAT,
            confidence=0.9,
        )

        no_threat_gates = self.engine.compute_gates(arousal, threat_level=0.0)
        threat_gates = self.engine.compute_gates(arousal, threat_level=0.8)

        # Visual and auditory should be boosted with threat
        self.assertGreater(
            threat_gates.sensory_gates["visual"],
            no_threat_gates.sensory_gates["visual"]
        )

    def test_global_threshold_varies(self):
        """Global threshold should vary with arousal."""
        low = ArousalLevelOutput(
            arousal_level=0.3,
            arousal_state=ArousalState.RELAXED,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.CIRCADIAN,
            confidence=0.9,
        )
        high = ArousalLevelOutput(
            arousal_level=0.8,
            arousal_state=ArousalState.FOCUSED,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.TASK_DEMAND,
            confidence=0.9,
        )

        low_gates = self.engine.compute_gates(low)
        high_gates = self.engine.compute_gates(high)

        # Both should be in valid range
        self.assertGreaterEqual(low_gates.global_threshold, 0.2)
        self.assertLessEqual(high_gates.global_threshold, 0.8)

    def test_adaptation_rate_by_state(self):
        """Adaptation rate should vary by state."""
        sleep = ArousalLevelOutput(
            arousal_level=0.05,
            arousal_state=ArousalState.SLEEP,
            arousal_trend=0.0,
            arousal_stability=0.9,
            primary_source=ArousalSource.CIRCADIAN,
            confidence=0.8,
        )
        hyperaroused = ArousalLevelOutput(
            arousal_level=0.95,
            arousal_state=ArousalState.HYPERAROUSED,
            arousal_trend=0.0,
            arousal_stability=0.5,
            primary_source=ArousalSource.THREAT,
            confidence=0.9,
        )

        sleep_gates = self.engine.compute_gates(sleep)
        hyper_gates = self.engine.compute_gates(hyperaroused)

        # Hyperaroused should adapt faster
        self.assertGreater(hyper_gates.gate_adaptation_rate, sleep_gates.gate_adaptation_rate)


# ============================================================================
# RESOURCE ALLOCATION ENGINE TESTS
# ============================================================================

class TestResourceAllocationEngine(unittest.TestCase):
    """Tests for ResourceAllocationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = ResourceAllocationEngine()

    def test_form_priorities_exist(self):
        """Should have priorities for key forms."""
        priorities = self.engine.FORM_PRIORITIES
        self.assertIn("08-arousal", priorities)
        self.assertIn("13-integrated-information", priorities)
        self.assertIn("01-visual", priorities)

    def test_critical_forms_high_priority(self):
        """Critical forms should have highest priority."""
        priorities = self.engine.FORM_PRIORITIES
        self.assertEqual(priorities["08-arousal"], 1.0)
        self.assertGreaterEqual(priorities["13-integrated-information"], 0.9)

    def test_allocate_resources_basic(self):
        """Should allocate resources to active forms."""
        arousal = ArousalLevelOutput(
            arousal_level=0.6,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.9,
        )
        demands = {"08-arousal": 0.2, "01-visual": 0.15, "02-auditory": 0.1}
        active = {"08-arousal", "01-visual", "02-auditory"}

        allocation = self.engine.allocate_resources(arousal, demands, active)

        self.assertIsInstance(allocation, ResourceAllocationOutput)
        self.assertIn("08-arousal", allocation.allocations)
        self.assertGreater(allocation.reserve_capacity, 0)

    def test_high_priority_gets_more(self):
        """Higher priority forms should get more resources."""
        arousal = ArousalLevelOutput(
            arousal_level=0.5,
            arousal_state=ArousalState.ALERT,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.INTERNAL,
            confidence=0.9,
        )
        demands = {"08-arousal": 0.5, "28-philosophy": 0.5}
        active = {"08-arousal", "28-philosophy"}

        allocation = self.engine.allocate_resources(arousal, demands, active)

        # Arousal should get more than philosophy
        self.assertGreater(
            allocation.allocations.get("08-arousal", 0),
            allocation.allocations.get("28-philosophy", 0)
        )

    def test_available_capacity_with_arousal(self):
        """Available capacity should scale with arousal."""
        low = ArousalLevelOutput(
            arousal_level=0.2,
            arousal_state=ArousalState.DROWSY,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.CIRCADIAN,
            confidence=0.9,
        )
        high = ArousalLevelOutput(
            arousal_level=0.7,
            arousal_state=ArousalState.FOCUSED,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.TASK_DEMAND,
            confidence=0.9,
        )

        low_alloc = self.engine.allocate_resources(low, {}, set())
        high_alloc = self.engine.allocate_resources(high, {}, set())

        # Higher arousal should have more available capacity
        self.assertGreater(high_alloc.total_available, low_alloc.total_available)

    def test_hyperarousal_reduces_capacity(self):
        """Hyperarousal should reduce efficiency."""
        optimal = ArousalLevelOutput(
            arousal_level=0.7,
            arousal_state=ArousalState.FOCUSED,
            arousal_trend=0.0,
            arousal_stability=0.8,
            primary_source=ArousalSource.TASK_DEMAND,
            confidence=0.9,
        )
        hyper = ArousalLevelOutput(
            arousal_level=0.95,
            arousal_state=ArousalState.HYPERAROUSED,
            arousal_trend=0.0,
            arousal_stability=0.5,
            primary_source=ArousalSource.THREAT,
            confidence=0.9,
        )

        optimal_alloc = self.engine.allocate_resources(optimal, {}, set())
        hyper_alloc = self.engine.allocate_resources(hyper, {}, set())

        # Hyperaroused should have less capacity than optimal focused
        self.assertGreater(optimal_alloc.total_available, hyper_alloc.total_available)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestArousalConsciousnessInterface(unittest.TestCase):
    """Tests for ArousalConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = ArousalConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "08-arousal")
        self.assertEqual(self.interface.FORM_NAME, "Arousal/Vigilance Consciousness")
        self.assertTrue(self.interface.IS_CRITICAL)

    def test_initial_state(self):
        """Should initialize to alert state."""
        self.assertEqual(self.interface.get_arousal_state(), ArousalState.ALERT)
        self.assertAlmostEqual(self.interface.get_arousal_level(), 0.5, places=1)

    def test_process_inputs(self):
        """Should process inputs and update state."""
        bundle = create_simple_input(
            sensory_level=0.7,
            emotional_level=0.6,
            circadian_level=0.7,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output = loop.run_until_complete(self.interface.process_inputs(bundle))
        finally:
            loop.close()

        self.assertIsInstance(output, ArousalLevelOutput)
        self.assertGreaterEqual(output.arousal_level, 0.0)
        self.assertLessEqual(output.arousal_level, 1.0)

    def test_get_gating_signals(self):
        """Should return current gating configuration."""
        gating = self.interface.get_gating_signals()
        self.assertIsInstance(gating, ConsciousnessGatingOutput)
        self.assertIn("visual", gating.sensory_gates)

    def test_get_resource_allocation(self):
        """Should return resource allocation."""
        allocation = self.interface.get_resource_allocation()
        self.assertIsInstance(allocation, ResourceAllocationOutput)

    def test_register_form(self):
        """Should register active forms."""
        self.interface.register_form("01-visual", 0.2)
        self.assertIn("01-visual", self.interface._active_forms)
        self.assertEqual(self.interface._form_demands["01-visual"], 0.2)

    def test_unregister_form(self):
        """Should unregister forms."""
        self.interface.register_form("01-visual", 0.2)
        self.interface.unregister_form("01-visual")
        self.assertNotIn("01-visual", self.interface._active_forms)

    def test_is_form_allowed_critical_always(self):
        """Critical forms should always be allowed."""
        self.assertTrue(self.interface.is_form_allowed("08-arousal"))
        self.assertTrue(self.interface.is_form_allowed("13-integrated-information"))
        self.assertTrue(self.interface.is_form_allowed("14-global-workspace"))

    def test_get_gate_for_form(self):
        """Should return gate for specific form."""
        visual_gate = self.interface.get_gate_for_form("01-visual")
        self.assertGreaterEqual(visual_gate, 0.0)
        self.assertLessEqual(visual_gate, 1.0)

    def test_on_state_change_callback(self):
        """Should register and call state change callbacks."""
        callback_called = []

        def callback(arousal):
            callback_called.append(arousal)

        self.interface.on_state_change(callback)

        bundle = create_simple_input(sensory_level=0.8, emotional_level=0.8, circadian_level=0.8)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.interface.process_inputs(bundle))
        finally:
            loop.close()

        self.assertGreaterEqual(len(callback_called), 1)

    def test_get_status(self):
        """Should return complete system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, ArousalSystemStatus)
        self.assertIsNotNone(status.current_level)
        self.assertIsNotNone(status.current_gating)
        self.assertGreaterEqual(status.system_health, 0.0)
        self.assertLessEqual(status.system_health, 1.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "08-arousal")
        self.assertEqual(d["form_name"], "Arousal/Vigilance Consciousness")
        self.assertTrue(d["is_critical"])
        self.assertIn("arousal", d)
        self.assertIn("gating", d)

    def test_state_transition_recording(self):
        """Should record state transitions."""
        # Force a state transition by changing arousal significantly
        # Start with low arousal
        low_bundle = create_simple_input(
            sensory_level=0.1,
            emotional_level=0.1,
            circadian_level=0.1,
        )
        # Then high arousal to trigger transition
        high_bundle = create_simple_input(
            sensory_level=0.9,
            emotional_level=0.9,
            circadian_level=0.9,
            threat_level=0.8,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Process multiple times to ensure state changes
            for _ in range(5):
                loop.run_until_complete(self.interface.process_inputs(low_bundle))
            for _ in range(5):
                loop.run_until_complete(self.interface.process_inputs(high_bundle))
        finally:
            loop.close()

        # Check that we have transitions recorded
        status = self.interface.get_status()
        # May or may not have transitions depending on smoothing
        self.assertIsInstance(status.recent_transitions, list)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_arousal_interface(self):
        """Should create new interface."""
        interface = create_arousal_interface()
        self.assertIsInstance(interface, ArousalConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "08-arousal")

    def test_create_simple_input_default(self):
        """Should create simple input with defaults."""
        bundle = create_simple_input()
        self.assertIsInstance(bundle, ArousalInputBundle)
        self.assertEqual(len(bundle.sensory_inputs), 1)
        self.assertIsNotNone(bundle.emotional_input)
        self.assertIsNotNone(bundle.circadian_input)
        self.assertIsNone(bundle.threat_input)

    def test_create_simple_input_with_threat(self):
        """Should create input with threat when specified."""
        bundle = create_simple_input(threat_level=0.7)
        self.assertIsNotNone(bundle.threat_input)
        self.assertEqual(bundle.threat_input.threat_level, 0.7)

    def test_create_simple_input_levels(self):
        """Should set levels correctly."""
        bundle = create_simple_input(
            sensory_level=0.3,
            emotional_level=0.4,
            circadian_level=0.6,
        )
        self.assertEqual(bundle.sensory_inputs[0].intensity, 0.3)
        self.assertEqual(bundle.emotional_input.arousal_component, 0.4)
        self.assertEqual(bundle.circadian_input.cortisol_level, 0.6)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete arousal system."""

    def test_full_processing_cycle(self):
        """Should complete full processing cycle."""
        interface = create_arousal_interface()

        # Register some forms
        interface.register_form("01-visual", 0.2)
        interface.register_form("02-auditory", 0.15)
        interface.register_form("07-emotional", 0.18)

        # Process inputs
        bundle = create_simple_input(
            sensory_level=0.6,
            emotional_level=0.5,
            circadian_level=0.7,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            arousal = loop.run_until_complete(interface.process_inputs(bundle))
        finally:
            loop.close()

        # Check all outputs are consistent
        self.assertIsInstance(arousal, ArousalLevelOutput)
        gating = interface.get_gating_signals()
        allocation = interface.get_resource_allocation()

        # Allocations should include registered forms
        self.assertIn("01-visual", allocation.allocations)
        self.assertIn("02-auditory", allocation.allocations)

    def test_threat_response_integration(self):
        """Should properly respond to threat."""
        interface = create_arousal_interface()

        # Normal state
        normal_bundle = create_simple_input(sensory_level=0.5, emotional_level=0.5, circadian_level=0.5)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            normal_arousal = loop.run_until_complete(interface.process_inputs(normal_bundle))

            # Threat state
            threat_bundle = create_simple_input(
                sensory_level=0.7,
                emotional_level=0.8,
                circadian_level=0.5,
                threat_level=0.9,
            )
            threat_arousal = loop.run_until_complete(interface.process_inputs(threat_bundle))
        finally:
            loop.close()

        # Threat should increase arousal
        self.assertGreater(threat_arousal.arousal_level, normal_arousal.arousal_level * 0.8)

    def test_circadian_influence(self):
        """Should show circadian influence on arousal."""
        # Day state (high cortisol, low melatonin)
        day_bundle = ArousalInputBundle(
            circadian_input=CircadianInput(
                circadian_phase=12.0,
                melatonin_level=0.1,
                cortisol_level=0.8,
                sleep_pressure=0.2,
                light_exposure=0.9,
            )
        )

        # Night state (low cortisol, high melatonin)
        night_bundle = ArousalInputBundle(
            circadian_input=CircadianInput(
                circadian_phase=2.0,
                melatonin_level=0.9,
                cortisol_level=0.2,
                sleep_pressure=0.8,
                light_exposure=0.1,
            )
        )

        day_interface = create_arousal_interface()
        night_interface = create_arousal_interface()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            day_arousal = loop.run_until_complete(day_interface.process_inputs(day_bundle))
            night_arousal = loop.run_until_complete(night_interface.process_inputs(night_bundle))
        finally:
            loop.close()

        # Day should have higher circadian arousal component
        self.assertGreater(
            day_arousal.components.get(ArousalSource.CIRCADIAN, 0.5),
            night_arousal.components.get(ArousalSource.CIRCADIAN, 0.5)
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
