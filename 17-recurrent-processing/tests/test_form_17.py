#!/usr/bin/env python3
"""
Test Suite for Form 17: Recurrent Processing Theory (RPT) Consciousness.

Tests cover:
- All enumerations (ProcessingPhase, RecurrenceType, ProcessingLevel, etc.)
- All input/output dataclasses
- FeedforwardEngine
- RecurrenceEngine
- RecurrentProcessingInterface (main interface)
- Convenience functions
- Integration tests for the full recurrent processing pipeline
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
    ProcessingPhase,
    RecurrenceType,
    ProcessingLevel,
    ConsciousnessState,
    MaskingEffect,
    # Input dataclasses
    FeedforwardSweep,
    RecurrentSignal,
    MaskingInput,
    RecurrentProcessingInput,
    # Output dataclasses
    RecurrentState,
    ConsciousnessThresholdResult,
    RecurrentProcessingOutput,
    RPTSystemStatus,
    # Engines
    FeedforwardEngine,
    RecurrenceEngine,
    # Main interface
    RecurrentProcessingInterface,
    # Convenience functions
    create_recurrent_processing_interface,
    create_feedforward_sweep,
    create_masking_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestProcessingPhase(unittest.TestCase):
    """Tests for ProcessingPhase enumeration."""

    def test_all_phases_exist(self):
        """All processing phases should be defined."""
        phases = [
            ProcessingPhase.FEEDFORWARD,
            ProcessingPhase.LOCAL_RECURRENCE,
            ProcessingPhase.GLOBAL_RECURRENCE,
            ProcessingPhase.SUSTAINED,
            ProcessingPhase.DECAYING,
        ]
        self.assertEqual(len(phases), 5)

    def test_phase_values(self):
        """Phases should have expected string values."""
        self.assertEqual(ProcessingPhase.FEEDFORWARD.value, "feedforward")
        self.assertEqual(ProcessingPhase.GLOBAL_RECURRENCE.value, "global_recurrence")


class TestRecurrenceType(unittest.TestCase):
    """Tests for RecurrenceType enumeration."""

    def test_all_types_exist(self):
        """All recurrence types should be defined."""
        types = [
            RecurrenceType.LATERAL,
            RecurrenceType.FEEDBACK,
            RecurrenceType.FEEDFORWARD,
            RecurrenceType.RE_ENTRANT,
            RecurrenceType.TOP_DOWN,
        ]
        self.assertEqual(len(types), 5)


class TestProcessingLevel(unittest.TestCase):
    """Tests for ProcessingLevel enumeration."""

    def test_all_levels_exist(self):
        """All processing levels should be defined."""
        levels = [
            ProcessingLevel.PRIMARY,
            ProcessingLevel.SECONDARY,
            ProcessingLevel.ASSOCIATION,
            ProcessingLevel.PREFRONTAL,
            ProcessingLevel.PARIETAL,
        ]
        self.assertEqual(len(levels), 5)


class TestConsciousnessState(unittest.TestCase):
    """Tests for ConsciousnessState enumeration."""

    def test_all_states_exist(self):
        """All consciousness states should be defined."""
        states = [
            ConsciousnessState.UNCONSCIOUS,
            ConsciousnessState.PHENOMENAL,
            ConsciousnessState.ACCESS,
            ConsciousnessState.FULL,
            ConsciousnessState.FADING,
        ]
        self.assertEqual(len(states), 5)


class TestMaskingEffect(unittest.TestCase):
    """Tests for MaskingEffect enumeration."""

    def test_all_effects_exist(self):
        """All masking effects should be defined."""
        effects = [
            MaskingEffect.NONE,
            MaskingEffect.FORWARD_MASK,
            MaskingEffect.BACKWARD_MASK,
            MaskingEffect.METACONTRAST,
            MaskingEffect.OBJECT_SUBSTITUTION,
        ]
        self.assertEqual(len(effects), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestFeedforwardSweep(unittest.TestCase):
    """Tests for FeedforwardSweep dataclass."""

    def test_creation(self):
        """Should create feedforward sweep with required fields."""
        sweep = FeedforwardSweep(
            sweep_id="ff1",
            stimulus_content={"shape": "circle", "color": "red"},
            stimulus_intensity=0.8,
            onset_time_ms=0.0,
            processing_levels_reached=[ProcessingLevel.PRIMARY, ProcessingLevel.SECONDARY],
            activation_strengths={"primary": 0.8, "secondary": 0.6},
        )
        self.assertEqual(sweep.sweep_id, "ff1")
        self.assertEqual(sweep.stimulus_intensity, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        sweep = create_feedforward_sweep("ff1")
        d = sweep.to_dict()
        self.assertEqual(d["sweep_id"], "ff1")
        self.assertIn("processing_levels", d)
        self.assertIn("activation_strengths", d)

    def test_default_duration(self):
        """Should have default duration."""
        sweep = create_feedforward_sweep("ff1")
        self.assertEqual(sweep.duration_ms, 50.0)


class TestRecurrentSignal(unittest.TestCase):
    """Tests for RecurrentSignal dataclass."""

    def test_creation(self):
        """Should create recurrent signal."""
        signal = RecurrentSignal(
            signal_id="rec1",
            source_level=ProcessingLevel.ASSOCIATION,
            target_level=ProcessingLevel.PRIMARY,
            recurrence_type=RecurrenceType.FEEDBACK,
            signal_strength=0.7,
            latency_ms=30.0,
            content_modulation={"enhancement": 0.35},
        )
        self.assertEqual(signal.signal_id, "rec1")
        self.assertEqual(signal.signal_strength, 0.7)

    def test_to_dict(self):
        """Should convert to dictionary."""
        signal = RecurrentSignal(
            signal_id="rec1",
            source_level=ProcessingLevel.PREFRONTAL,
            target_level=ProcessingLevel.PRIMARY,
            recurrence_type=RecurrenceType.RE_ENTRANT,
            signal_strength=0.6,
            latency_ms=80.0,
            content_modulation={},
        )
        d = signal.to_dict()
        self.assertEqual(d["recurrence_type"], "re_entrant")


class TestMaskingInput(unittest.TestCase):
    """Tests for MaskingInput dataclass."""

    def test_creation(self):
        """Should create masking input."""
        mask = MaskingInput(
            mask_type=MaskingEffect.BACKWARD_MASK,
            mask_strength=0.8,
            mask_onset_ms=50.0,
            stimulus_onset_asynchrony=50.0,
        )
        self.assertEqual(mask.mask_type, MaskingEffect.BACKWARD_MASK)
        self.assertEqual(mask.mask_strength, 0.8)

    def test_to_dict(self):
        """Should convert to dictionary."""
        mask = create_masking_input()
        d = mask.to_dict()
        self.assertIn("mask_type", d)
        self.assertIn("soa", d)


class TestRecurrentProcessingInput(unittest.TestCase):
    """Tests for RecurrentProcessingInput dataclass."""

    def test_creation(self):
        """Should create processing input."""
        sweep = create_feedforward_sweep("ff1")
        rp_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            attention_modulation=0.8,
        )
        self.assertEqual(rp_input.attention_modulation, 0.8)
        self.assertIsNone(rp_input.masking)

    def test_with_masking(self):
        """Should accept masking input."""
        sweep = create_feedforward_sweep("ff1")
        mask = create_masking_input()
        rp_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            masking=mask,
        )
        self.assertIsNotNone(rp_input.masking)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestRecurrentState(unittest.TestCase):
    """Tests for RecurrentState dataclass."""

    def test_creation(self):
        """Should create recurrent state."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.GLOBAL_RECURRENCE,
            consciousness_state=ConsciousnessState.ACCESS,
            active_recurrent_loops=[("prefrontal", "primary")],
            recurrence_strength=0.65,
            loop_duration_ms=100.0,
            is_globally_recurrent=True,
            is_locally_recurrent=True,
            masking_disruption=0.0,
        )
        self.assertTrue(state.is_globally_recurrent)
        self.assertEqual(state.consciousness_state, ConsciousnessState.ACCESS)

    def test_to_dict(self):
        """Should convert to dictionary."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.FEEDFORWARD,
            consciousness_state=ConsciousnessState.UNCONSCIOUS,
            active_recurrent_loops=[],
            recurrence_strength=0.1,
            loop_duration_ms=0.0,
            is_globally_recurrent=False,
            is_locally_recurrent=False,
            masking_disruption=0.0,
        )
        d = state.to_dict()
        self.assertEqual(d["processing_phase"], "feedforward")
        self.assertEqual(d["consciousness_state"], "unconscious")


class TestConsciousnessThresholdResult(unittest.TestCase):
    """Tests for ConsciousnessThresholdResult dataclass."""

    def test_creation(self):
        """Should create threshold result."""
        result = ConsciousnessThresholdResult(
            threshold_reached=True,
            consciousness_state=ConsciousnessState.ACCESS,
            recurrence_strength=0.6,
            required_threshold=0.5,
            margin=0.1,
            explanation="Access consciousness achieved.",
        )
        self.assertTrue(result.threshold_reached)
        self.assertEqual(result.margin, 0.1)

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ConsciousnessThresholdResult(
            threshold_reached=False,
            consciousness_state=ConsciousnessState.UNCONSCIOUS,
            recurrence_strength=0.1,
            required_threshold=0.3,
            margin=-0.2,
            explanation="Below threshold.",
        )
        d = result.to_dict()
        self.assertFalse(d["threshold_reached"])
        self.assertIn("explanation", d)


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestFeedforwardEngine(unittest.TestCase):
    """Tests for FeedforwardEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = FeedforwardEngine()

    def test_process_feedforward(self):
        """Should process feedforward sweep."""
        sweep = create_feedforward_sweep("ff1", stimulus_intensity=0.8)
        result = self.engine.process_feedforward(sweep)

        self.assertEqual(result["sweep_id"], "ff1")
        self.assertIn("features_extracted", result)
        self.assertIn("activation_profile", result)
        self.assertIsNotNone(result["max_level_reached"])

    def test_activation_decay(self):
        """Activation should decay through levels."""
        sweep = create_feedforward_sweep("ff1", stimulus_intensity=0.9)
        result = self.engine.process_feedforward(sweep)

        profile = result["activation_profile"]
        if "primary" in profile and "association" in profile:
            self.assertGreater(profile["primary"], profile["association"])

    def test_feature_extraction(self):
        """Should extract features at appropriate levels."""
        sweep = create_feedforward_sweep("ff1", stimulus_intensity=0.8)
        result = self.engine.process_feedforward(sweep)
        features = result["features_extracted"]

        # Primary level features
        self.assertIn("edges", features)
        self.assertIn("orientation", features)

    def test_latency_computation(self):
        """Should compute total processing latency."""
        sweep = create_feedforward_sweep("ff1")
        result = self.engine.process_feedforward(sweep)
        self.assertGreater(result["total_latency_ms"], 0.0)

    def test_weak_stimulus(self):
        """Weak stimulus should have lower activations."""
        weak_sweep = create_feedforward_sweep("ff_weak", stimulus_intensity=0.1)
        strong_sweep = create_feedforward_sweep("ff_strong", stimulus_intensity=0.9)

        weak_result = self.engine.process_feedforward(weak_sweep)
        strong_result = self.engine.process_feedforward(strong_sweep)

        weak_features = weak_result["features_extracted"]
        strong_features = strong_result["features_extracted"]

        if "edges" in weak_features and "edges" in strong_features:
            self.assertGreater(strong_features["edges"], weak_features["edges"])


class TestRecurrenceEngine(unittest.TestCase):
    """Tests for RecurrenceEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = RecurrenceEngine()
        self.ff_engine = FeedforwardEngine()

    def _get_ff_result(self, intensity: float = 0.8) -> dict:
        """Helper to get feedforward result."""
        sweep = create_feedforward_sweep("ff", stimulus_intensity=intensity)
        return self.ff_engine.process_feedforward(sweep)

    def test_initiate_recurrence(self):
        """Should initiate recurrent processing."""
        ff_result = self._get_ff_result()
        signals, state = self.engine.initiate_recurrence(ff_result, 0.8)

        self.assertIsInstance(signals, list)
        self.assertIsInstance(state, RecurrentState)
        self.assertGreater(len(signals), 0)

    def test_strong_stimulus_produces_recurrence(self):
        """Strong stimulus should produce recurrent signals."""
        ff_result = self._get_ff_result(0.9)
        signals, state = self.engine.initiate_recurrence(ff_result, 0.9, attention_modulation=1.0)

        self.assertGreater(state.recurrence_strength, 0.0)
        self.assertTrue(state.is_locally_recurrent or state.is_globally_recurrent)

    def test_weak_stimulus_less_recurrence(self):
        """Weak stimulus should produce less recurrence."""
        weak_result = self._get_ff_result(0.1)
        strong_result = self._get_ff_result(0.9)

        _, weak_state = self.engine.initiate_recurrence(weak_result, 0.1)
        _, strong_state = self.engine.initiate_recurrence(strong_result, 0.9)

        self.assertLessEqual(weak_state.recurrence_strength, strong_state.recurrence_strength)

    def test_masking_disrupts_recurrence(self):
        """Backward masking should disrupt recurrent processing."""
        ff_result = self._get_ff_result(0.8)
        mask = create_masking_input(MaskingEffect.BACKWARD_MASK, 0.9, 30.0)

        _, no_mask_state = self.engine.initiate_recurrence(ff_result, 0.8)
        _, masked_state = self.engine.initiate_recurrence(ff_result, 0.8, masking=mask)

        self.assertGreater(masked_state.masking_disruption, 0.0)
        self.assertGreaterEqual(
            no_mask_state.recurrence_strength,
            masked_state.recurrence_strength
        )

    def test_attention_modulates_recurrence(self):
        """Attention should modulate recurrence strength."""
        ff_result = self._get_ff_result(0.6)

        _, low_att_state = self.engine.initiate_recurrence(ff_result, 0.6, attention_modulation=0.3)
        _, high_att_state = self.engine.initiate_recurrence(ff_result, 0.6, attention_modulation=1.0)

        self.assertGreaterEqual(
            high_att_state.recurrence_strength,
            low_att_state.recurrence_strength
        )

    def test_check_conscious_threshold_unconscious(self):
        """Below-threshold state should be unconscious."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.FEEDFORWARD,
            consciousness_state=ConsciousnessState.UNCONSCIOUS,
            active_recurrent_loops=[],
            recurrence_strength=0.1,
            loop_duration_ms=0.0,
            is_globally_recurrent=False,
            is_locally_recurrent=False,
            masking_disruption=0.0,
        )
        result = self.engine.check_conscious_threshold(state)
        self.assertFalse(result.threshold_reached)
        self.assertEqual(result.consciousness_state, ConsciousnessState.UNCONSCIOUS)

    def test_check_conscious_threshold_phenomenal(self):
        """Local recurrence should yield phenomenal consciousness."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.LOCAL_RECURRENCE,
            consciousness_state=ConsciousnessState.PHENOMENAL,
            active_recurrent_loops=[("secondary", "primary")],
            recurrence_strength=0.4,
            loop_duration_ms=30.0,
            is_globally_recurrent=False,
            is_locally_recurrent=True,
            masking_disruption=0.0,
        )
        result = self.engine.check_conscious_threshold(state)
        self.assertTrue(result.threshold_reached)
        self.assertEqual(result.consciousness_state, ConsciousnessState.PHENOMENAL)

    def test_check_conscious_threshold_access(self):
        """Global recurrence should yield access consciousness."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.GLOBAL_RECURRENCE,
            consciousness_state=ConsciousnessState.ACCESS,
            active_recurrent_loops=[("prefrontal", "primary")],
            recurrence_strength=0.6,
            loop_duration_ms=80.0,
            is_globally_recurrent=True,
            is_locally_recurrent=True,
            masking_disruption=0.0,
        )
        result = self.engine.check_conscious_threshold(state)
        self.assertTrue(result.threshold_reached)
        self.assertIn(result.consciousness_state,
                      [ConsciousnessState.ACCESS, ConsciousnessState.FULL])

    def test_forward_mask_less_disruptive(self):
        """Forward masking should be less disruptive than backward masking."""
        ff_result = self._get_ff_result(0.8)
        forward_mask = create_masking_input(MaskingEffect.FORWARD_MASK, 0.8, 50.0)
        backward_mask = create_masking_input(MaskingEffect.BACKWARD_MASK, 0.8, 50.0)

        _, forward_state = self.engine.initiate_recurrence(ff_result, 0.8, masking=forward_mask)
        _, backward_state = self.engine.initiate_recurrence(ff_result, 0.8, masking=backward_mask)

        self.assertLessEqual(forward_state.masking_disruption, backward_state.masking_disruption)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestRecurrentProcessingInterface(unittest.TestCase):
    """Tests for RecurrentProcessingInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = RecurrentProcessingInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "17-recurrent-processing")
        self.assertEqual(self.interface.FORM_NAME, "Recurrent Processing Theory (RPT)")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._is_initialized)

    def test_feedforward_sweep(self):
        """Should process feedforward sweep."""
        sweep = create_feedforward_sweep("ff1")

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.interface.feedforward_sweep(sweep))
        finally:
            loop.close()

        self.assertIsInstance(result, dict)
        self.assertIn("features_extracted", result)

    def test_initiate_recurrence(self):
        """Should initiate recurrent processing."""
        sweep = create_feedforward_sweep("ff1", 0.8)
        ff_engine = FeedforwardEngine()
        ff_result = ff_engine.process_feedforward(sweep)

        loop = asyncio.new_event_loop()
        try:
            signals, state = loop.run_until_complete(
                self.interface.initiate_recurrence(ff_result, 0.8)
            )
        finally:
            loop.close()

        self.assertIsInstance(signals, list)
        self.assertIsInstance(state, RecurrentState)

    def test_get_processing_phase(self):
        """Should return processing phase."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.LOCAL_RECURRENCE,
            consciousness_state=ConsciousnessState.PHENOMENAL,
            active_recurrent_loops=[], recurrence_strength=0.4,
            loop_duration_ms=30.0, is_globally_recurrent=False,
            is_locally_recurrent=True, masking_disruption=0.0,
        )

        loop = asyncio.new_event_loop()
        try:
            phase = loop.run_until_complete(self.interface.get_processing_phase(state))
        finally:
            loop.close()

        self.assertEqual(phase, ProcessingPhase.LOCAL_RECURRENCE)

    def test_check_conscious_threshold(self):
        """Should check consciousness threshold."""
        state = RecurrentState(
            processing_phase=ProcessingPhase.GLOBAL_RECURRENCE,
            consciousness_state=ConsciousnessState.ACCESS,
            active_recurrent_loops=[("prefrontal", "primary")],
            recurrence_strength=0.6,
            loop_duration_ms=80.0,
            is_globally_recurrent=True,
            is_locally_recurrent=True,
            masking_disruption=0.0,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.interface.check_conscious_threshold(state)
            )
        finally:
            loop.close()

        self.assertIsInstance(result, ConsciousnessThresholdResult)
        self.assertTrue(result.threshold_reached)

    def test_process_stimulus(self):
        """Should run full stimulus processing pipeline."""
        sweep = create_feedforward_sweep("ff1", stimulus_intensity=0.8)
        rp_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            attention_modulation=1.0,
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.process_stimulus(rp_input))
        finally:
            loop.close()

        self.assertIsInstance(output, RecurrentProcessingOutput)
        self.assertIsNotNone(output.recurrent_state)
        self.assertIsNotNone(output.consciousness_threshold)
        self.assertGreater(output.total_processing_time_ms, 0.0)

    def test_to_dict(self):
        """Should convert state to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "17-recurrent-processing")
        self.assertIn("sweeps_processed", d)
        self.assertIn("conscious_events", d)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, RPTSystemStatus)
        self.assertFalse(status.is_initialized)

    def test_counters_increment(self):
        """Should track processing counters."""
        sweep = create_feedforward_sweep("ff1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.feedforward_sweep(sweep))
        finally:
            loop.close()

        self.assertEqual(self.interface._sweeps_processed, 1)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_recurrent_processing_interface(self):
        """Should create new interface."""
        interface = create_recurrent_processing_interface()
        self.assertIsInstance(interface, RecurrentProcessingInterface)

    def test_create_feedforward_sweep(self):
        """Should create feedforward sweep."""
        sweep = create_feedforward_sweep("test", 0.9, {"shape": "square"})
        self.assertEqual(sweep.sweep_id, "test")
        self.assertEqual(sweep.stimulus_intensity, 0.9)
        self.assertIn("shape", sweep.stimulus_content)

    def test_create_feedforward_sweep_defaults(self):
        """Should create sweep with defaults."""
        sweep = create_feedforward_sweep("ff1")
        self.assertEqual(sweep.stimulus_intensity, 0.7)
        self.assertEqual(len(sweep.processing_levels_reached), 4)

    def test_create_masking_input(self):
        """Should create masking input."""
        mask = create_masking_input(MaskingEffect.METACONTRAST, 0.6, 80.0)
        self.assertEqual(mask.mask_type, MaskingEffect.METACONTRAST)
        self.assertEqual(mask.mask_strength, 0.6)

    def test_create_masking_input_defaults(self):
        """Should create masking with defaults."""
        mask = create_masking_input()
        self.assertEqual(mask.mask_type, MaskingEffect.BACKWARD_MASK)
        self.assertEqual(mask.mask_strength, 0.8)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRPTIntegration(unittest.TestCase):
    """Integration tests for the Recurrent Processing Theory system."""

    def test_strong_stimulus_becomes_conscious(self):
        """Strong unmasked stimulus should reach consciousness."""
        interface = create_recurrent_processing_interface()
        sweep = create_feedforward_sweep("strong", stimulus_intensity=0.9)
        rp_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            attention_modulation=1.0,
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            output = loop.run_until_complete(interface.process_stimulus(rp_input))
        finally:
            loop.close()

        # Strong stimulus should have recurrence
        self.assertGreater(output.recurrent_state.recurrence_strength, 0.0)
        self.assertIn("consciousness_state", output.stimulus_percept)

    def test_masked_stimulus_less_conscious(self):
        """Masked stimulus should have reduced consciousness."""
        interface_no_mask = create_recurrent_processing_interface()
        interface_masked = create_recurrent_processing_interface()

        sweep = create_feedforward_sweep("stim", stimulus_intensity=0.7)
        mask = create_masking_input(MaskingEffect.BACKWARD_MASK, 0.9, 30.0)

        no_mask_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            attention_modulation=1.0,
        )
        masked_input = RecurrentProcessingInput(
            feedforward_sweep=sweep,
            attention_modulation=1.0,
            masking=mask,
        )

        loop = asyncio.new_event_loop()
        try:
            no_mask_out = loop.run_until_complete(
                interface_no_mask.process_stimulus(no_mask_input)
            )
            masked_out = loop.run_until_complete(
                interface_masked.process_stimulus(masked_input)
            )
        finally:
            loop.close()

        self.assertGreaterEqual(
            no_mask_out.recurrent_state.recurrence_strength,
            masked_out.recurrent_state.recurrence_strength
        )

    def test_processing_time_includes_all_stages(self):
        """Total processing time should include feedforward and recurrent stages."""
        interface = create_recurrent_processing_interface()
        sweep = create_feedforward_sweep("ff1")
        rp_input = RecurrentProcessingInput(feedforward_sweep=sweep)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(interface.process_stimulus(rp_input))
        finally:
            loop.close()

        # Should be at least the feedforward duration
        self.assertGreater(output.total_processing_time_ms, 0.0)

    def test_percept_includes_processing_info(self):
        """Final percept should include processing information."""
        interface = create_recurrent_processing_interface()
        sweep = create_feedforward_sweep("ff1", content={"shape": "triangle"})
        rp_input = RecurrentProcessingInput(feedforward_sweep=sweep)

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(interface.process_stimulus(rp_input))
        finally:
            loop.close()

        percept = output.stimulus_percept
        self.assertIn("consciousness_state", percept)
        self.assertIn("processing_phase", percept)
        self.assertIn("recurrence_strength", percept)
        self.assertIn("shape", percept)

    def test_multiple_stimuli_tracked(self):
        """Should track counts for multiple stimuli."""
        interface = create_recurrent_processing_interface()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            for i in range(3):
                sweep = create_feedforward_sweep(f"ff{i}", stimulus_intensity=0.5 + i * 0.2)
                rp_input = RecurrentProcessingInput(feedforward_sweep=sweep)
                loop.run_until_complete(interface.process_stimulus(rp_input))
        finally:
            loop.close()

        self.assertEqual(interface._sweeps_processed, 3)
        status = interface.get_status()
        self.assertEqual(status.total_sweeps_processed, 3)
        total_events = status.conscious_events + status.unconscious_events
        self.assertEqual(total_events, 3)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
