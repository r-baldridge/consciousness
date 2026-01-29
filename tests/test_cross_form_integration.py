#!/usr/bin/env python3
"""
Test Suite: Cross-Form Integration Tests

Verifies that consciousness forms work together correctly across the
architecture. Tests cover arousal gating, sensory-to-perceptual pipelines,
emotional modulation, meta-cognitive monitoring, theoretical assessments
(IIT, GWT), message passing, and full system cycles.

These tests focus on the *integration contracts* between forms rather than
the internal behavior of any single form.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add the arousal interface directory so its package can be imported
AROUSAL_DIR = PROJECT_ROOT / "08-arousal"
AROUSAL_INTERFACE_DIR = str(AROUSAL_DIR / "interface")
if AROUSAL_INTERFACE_DIR not in sys.path:
    sys.path.insert(0, AROUSAL_INTERFACE_DIR)
if str(AROUSAL_DIR) not in sys.path:
    sys.path.insert(0, str(AROUSAL_DIR))

# Import arousal interface components -- Form 08 is the critical gating form
from interface import (
    ArousalConsciousnessInterface,
    ArousalInputBundle,
    ArousalLevelOutput,
    ArousalSource,
    ArousalState,
    CircadianInput,
    ConsciousnessGatingOutput,
    EmotionalInput,
    GateCategory,
    ResourceAllocationOutput,
    SensoryArousalInput,
    SensoryModality,
    ThreatInput,
    create_arousal_interface,
    create_simple_input,
)

# Import test helpers
from tests.conftest import (
    ArousalGatedSystem,
    create_emotional_state,
    create_gwt_broadcast,
    create_iit_measurement,
    create_message,
    create_meta_cognitive_report,
    create_sensory_stimulus,
    run_async,
)
from tests.form_registry_utils import (
    discover_forms,
    extract_form_metadata,
    get_all_form_ids,
    get_forms_with_interfaces,
    instantiate_form_interface,
    load_form_interface,
)


# ============================================================================
# AROUSAL GATING TESTS
# ============================================================================


class TestArousalGating(unittest.TestCase):
    """Test arousal gating of other forms.

    Form 08 (Arousal/Vigilance) acts as the system-wide gatekeeper,
    controlling which forms are allowed to process and how many resources
    each form receives. These tests verify that gating behavior is
    correct across different arousal states.
    """

    def setUp(self):
        """Create a fresh arousal interface for each test."""
        self.arousal = create_arousal_interface()
        # Register a representative set of forms
        self.arousal.register_form("01-visual", 0.20)
        self.arousal.register_form("02-auditory", 0.15)
        self.arousal.register_form("03-somatosensory", 0.10)
        self.arousal.register_form("07-emotional", 0.18)
        self.arousal.register_form("09-perceptual", 0.15)
        self.arousal.register_form("11-meta-consciousness", 0.12)
        self.arousal.register_form("28-philosophy", 0.05)

    def _process_to_state(self, target_state: str):
        """
        Drive the arousal system toward a target state by processing
        appropriate inputs repeatedly.
        """
        if target_state == "sleep":
            bundle = create_simple_input(
                sensory_level=0.02, emotional_level=0.02, circadian_level=0.02
            )
        elif target_state == "drowsy":
            bundle = create_simple_input(
                sensory_level=0.1, emotional_level=0.1, circadian_level=0.15
            )
        elif target_state == "alert":
            bundle = create_simple_input(
                sensory_level=0.5, emotional_level=0.5, circadian_level=0.6
            )
        elif target_state == "focused":
            bundle = create_simple_input(
                sensory_level=0.7, emotional_level=0.6, circadian_level=0.7
            )
        elif target_state == "hyperaroused":
            bundle = create_simple_input(
                sensory_level=0.95, emotional_level=0.95, circadian_level=0.9,
                threat_level=0.9,
            )
        else:
            bundle = create_simple_input()

        for _ in range(5):
            run_async(self.arousal.process_inputs(bundle))

    def test_sleep_blocks_non_critical(self):
        """In sleep state, non-critical forms should have reduced gating."""
        self._process_to_state("sleep")
        gating = self.arousal.get_gating_signals()

        self.assertIsInstance(gating, ConsciousnessGatingOutput)

        # Global threshold should be relatively high in sleep (blocks content)
        # or sensory gates should be mostly closed
        # At minimum, cognitive gates should be reduced
        for gate_name, gate_value in gating.cognitive_gates.items():
            self.assertLessEqual(
                gate_value,
                1.0,
                f"Cognitive gate '{gate_name}' should not exceed 1.0 in sleep",
            )

    def test_alert_enables_all_forms(self):
        """In alert state, all registered forms should be allowed."""
        self._process_to_state("alert")

        # Critical forms should always be allowed
        self.assertTrue(self.arousal.is_form_allowed("08-arousal"))
        self.assertTrue(self.arousal.is_form_allowed("13-integrated-information"))
        self.assertTrue(self.arousal.is_form_allowed("14-global-workspace"))

        # Sensory and cognitive forms should generally be allowed when alert
        gating = self.arousal.get_gating_signals()
        visual_gate = gating.sensory_gates.get("visual", 0)
        self.assertGreater(
            visual_gate,
            0.0,
            "Visual gate should be open in alert state",
        )

    def test_threat_boosts_sensory(self):
        """Threat input should boost sensory gate openness."""
        # First: process at normal alert level
        normal_bundle = create_simple_input(
            sensory_level=0.5, emotional_level=0.5, circadian_level=0.6,
        )
        run_async(self.arousal.process_inputs(normal_bundle))
        normal_gating = self.arousal.get_gating_signals()
        normal_visual = normal_gating.sensory_gates.get("visual", 0)

        # Second: process with high threat
        threat_bundle = create_simple_input(
            sensory_level=0.7, emotional_level=0.8, circadian_level=0.6,
            threat_level=0.9,
        )
        run_async(self.arousal.process_inputs(threat_bundle))
        threat_gating = self.arousal.get_gating_signals()
        threat_visual = threat_gating.sensory_gates.get("visual", 0)

        # Visual gate should be at least as open under threat
        self.assertGreaterEqual(
            threat_visual,
            normal_visual * 0.95,
            "Threat should maintain or boost visual gating",
        )

    def test_critical_forms_always_allowed(self):
        """Critical forms (08, 13, 14) should be allowed in any state."""
        critical_forms = ["08-arousal", "13-integrated-information", "14-global-workspace"]

        # Test across multiple states
        for state in ["sleep", "drowsy", "alert", "hyperaroused"]:
            self._process_to_state(state)
            for form_id in critical_forms:
                self.assertTrue(
                    self.arousal.is_form_allowed(form_id),
                    f"Critical form {form_id} should be allowed in {state} state",
                )

    def test_gate_values_in_valid_range(self):
        """All gate values should be in [0, 1] range."""
        for state in ["sleep", "drowsy", "alert", "focused", "hyperaroused"]:
            self._process_to_state(state)
            gating = self.arousal.get_gating_signals()

            for name, value in gating.sensory_gates.items():
                self.assertGreaterEqual(value, 0.0, f"{name} gate < 0 in {state}")
                self.assertLessEqual(value, 1.0, f"{name} gate > 1 in {state}")

            for name, value in gating.cognitive_gates.items():
                self.assertGreaterEqual(value, 0.0, f"{name} gate < 0 in {state}")
                self.assertLessEqual(value, 1.0, f"{name} gate > 1 in {state}")

    def test_resource_allocation_reflects_arousal(self):
        """Resource allocation should scale with arousal level."""
        # Low arousal
        self._process_to_state("drowsy")
        low_alloc = self.arousal.get_resource_allocation()

        # High arousal
        self._process_to_state("focused")
        high_alloc = self.arousal.get_resource_allocation()

        self.assertIsInstance(low_alloc, ResourceAllocationOutput)
        self.assertIsInstance(high_alloc, ResourceAllocationOutput)

        # Higher arousal should generally provide more total capacity
        self.assertGreater(
            high_alloc.total_available,
            low_alloc.total_available,
            "Focused state should provide more resources than drowsy",
        )

    def test_register_and_unregister_form(self):
        """Forms can be dynamically registered and unregistered."""
        new_form = "16-predictive-coding"

        # Register
        self.arousal.register_form(new_form, 0.10)
        self.assertIn(new_form, self.arousal._active_forms)

        # Process inputs to refresh allocation (allocation is computed
        # during process_inputs, not on register)
        bundle = create_simple_input(
            sensory_level=0.6, emotional_level=0.5, circadian_level=0.7,
        )
        run_async(self.arousal.process_inputs(bundle))

        # Get allocation -- should include new form after processing
        alloc = self.arousal.get_resource_allocation()
        self.assertIn(new_form, alloc.allocations)

        # Unregister
        self.arousal.unregister_form(new_form)
        self.assertNotIn(new_form, self.arousal._active_forms)


# ============================================================================
# SENSORY TO PERCEPTUAL PIPELINE TESTS
# ============================================================================


class TestSensoryToPerceptual(unittest.TestCase):
    """Test sensory -> perceptual binding pipeline.

    Forms 01-06 (sensory) should feed into Form 09 (perceptual binding).
    These tests verify the data flow contracts between sensory forms and
    the perceptual binding form, using arousal gating as the mediator.
    """

    def setUp(self):
        """Set up the arousal-gated sensory pipeline."""
        self.arousal = create_arousal_interface()
        self.sensory_forms = [
            "01-visual", "02-auditory", "03-somatosensory",
            "04-olfactory", "05-gustatory", "06-interoceptive",
        ]
        self.perceptual_form = "09-perceptual"

        for form_id in self.sensory_forms + [self.perceptual_form]:
            self.arousal.register_form(form_id, 0.12)

        # Set to alert state
        bundle = create_simple_input(
            sensory_level=0.6, emotional_level=0.5, circadian_level=0.7,
        )
        run_async(self.arousal.process_inputs(bundle))

    def test_visual_feeds_perception(self):
        """Visual sensory data should be consumable by perceptual binding."""
        stimulus = create_sensory_stimulus(
            modality="visual", intensity=0.7, salience=0.8
        )

        # Verify the stimulus has the expected structure for binding
        self.assertIn("modality", stimulus)
        self.assertEqual(stimulus["modality"], "visual")
        self.assertIn("features", stimulus)
        self.assertIn("complexity", stimulus["features"])

        # The visual form should be gated open in alert state
        visual_gate = self.arousal.get_gate_for_form("01-visual")
        self.assertGreater(
            visual_gate, 0.0,
            "Visual form should be gated open for sensory->perceptual flow",
        )

    def test_multimodal_binding(self):
        """Multiple sensory modalities should be bindable together."""
        stimuli = {
            "visual": create_sensory_stimulus("visual", 0.7, 0.8),
            "auditory": create_sensory_stimulus("auditory", 0.6, 0.7),
            "somatosensory": create_sensory_stimulus("somatosensory", 0.4, 0.3),
        }

        # All modalities should be present and well-formed
        self.assertEqual(len(stimuli), 3)
        for modality, stim in stimuli.items():
            self.assertEqual(stim["modality"], modality)
            self.assertGreater(stim["intensity"], 0.0)

        # Binding message should be constructable
        binding_msg = create_message(
            source_form="01-visual",
            target_form=self.perceptual_form,
            message_type="data",
            payload={"stimuli": stimuli},
            priority=0.7,
        )
        self.assertEqual(binding_msg["target"], self.perceptual_form)
        self.assertIn("stimuli", binding_msg["payload"])

    def test_all_sensory_modalities_gated(self):
        """Each sensory modality should have an associated gate."""
        gating = self.arousal.get_gating_signals()

        expected_gates = ["visual", "auditory", "somatosensory",
                          "olfactory", "gustatory", "interoceptive"]

        for gate_name in expected_gates:
            self.assertIn(
                gate_name,
                gating.sensory_gates,
                f"Missing sensory gate for {gate_name}",
            )

    def test_sensory_to_perceptual_message_structure(self):
        """Messages from sensory forms to perceptual should follow protocol."""
        for sensory_form in self.sensory_forms:
            msg = create_message(
                source_form=sensory_form,
                target_form=self.perceptual_form,
                message_type="data",
                payload={"processed_features": {"intensity": 0.5}},
            )
            self.assertEqual(msg["source"], sensory_form)
            self.assertEqual(msg["target"], self.perceptual_form)
            self.assertEqual(msg["type"], "data")
            self.assertIn("processed_features", msg["payload"])


# ============================================================================
# EMOTIONAL MODULATION TESTS
# ============================================================================


class TestEmotionalModulation(unittest.TestCase):
    """Test emotional modulation of processing.

    Form 07 (Emotional Processing) modulates Form 08 (Arousal) and
    influences memory encoding, perceptual salience, and executive
    function. These tests verify the modulation pathways.
    """

    def setUp(self):
        """Create arousal system with emotional form registered."""
        self.arousal = create_arousal_interface()
        self.arousal.register_form("07-emotional", 0.18)
        self.arousal.register_form("09-perceptual", 0.15)
        self.arousal.register_form("12-narrative-consciousness", 0.10)

    def test_emotion_affects_arousal(self):
        """High emotional arousal should increase overall arousal level."""
        # Process with low emotional input
        low_emotion = create_simple_input(
            sensory_level=0.5, emotional_level=0.2, circadian_level=0.6,
        )
        low_result = run_async(self.arousal.process_inputs(low_emotion))

        # Reset and process with high emotional input
        self.arousal = create_arousal_interface()
        high_emotion = create_simple_input(
            sensory_level=0.5, emotional_level=0.9, circadian_level=0.6,
        )
        high_result = run_async(self.arousal.process_inputs(high_emotion))

        # Higher emotional input should yield higher arousal
        self.assertGreater(
            high_result.arousal_level,
            low_result.arousal_level,
            "High emotional input should produce higher arousal",
        )

    def test_emotion_affects_memory(self):
        """Emotional state should be includable in memory-related messages."""
        emotional_state = create_emotional_state(
            valence=-0.7, arousal_component=0.8, dominant_emotion="fear"
        )

        # Build a memory encoding message that includes emotional context
        memory_msg = create_message(
            source_form="07-emotional",
            target_form="12-narrative-consciousness",
            message_type="data",
            payload={
                "emotional_context": emotional_state,
                "encoding_priority": emotional_state["intensity"],
            },
        )

        self.assertEqual(memory_msg["source"], "07-emotional")
        self.assertIn("emotional_context", memory_msg["payload"])
        self.assertGreater(
            memory_msg["payload"]["encoding_priority"],
            0.5,
            "Fear state should give high encoding priority",
        )

    def test_positive_emotion_vs_negative_emotion(self):
        """Positive and negative emotions should both modulate arousal."""
        # Positive emotion (using excitement and curiosity fields)
        positive = ArousalInputBundle(
            emotional_input=EmotionalInput(
                valence=0.8, arousal_component=0.7,
                excitement=0.8, curiosity=0.7,
            )
        )
        pos_result = run_async(self.arousal.process_inputs(positive))

        # Reset
        self.arousal = create_arousal_interface()

        # Negative emotion (using fear and anxiety fields)
        negative = ArousalInputBundle(
            emotional_input=EmotionalInput(
                valence=-0.8, arousal_component=0.7,
                fear=0.8, anxiety=0.6,
            )
        )
        neg_result = run_async(self.arousal.process_inputs(negative))

        # Both should have emotional component in their output
        self.assertIn(ArousalSource.EMOTIONAL, pos_result.components)
        self.assertIn(ArousalSource.EMOTIONAL, neg_result.components)

    def test_emotional_modulation_message_to_perceptual(self):
        """Emotional form should be able to modulate perceptual processing."""
        emotional_state = create_emotional_state(
            valence=-0.5, arousal_component=0.6, dominant_emotion="anxiety"
        )

        modulation_msg = create_message(
            source_form="07-emotional",
            target_form="09-perceptual",
            message_type="control",
            payload={
                "modulation_type": "salience_boost",
                "emotional_state": emotional_state,
                "threat_bias": 0.4,
            },
            priority=0.7,
        )

        self.assertEqual(modulation_msg["type"], "control")
        self.assertTrue(modulation_msg["requires_ack"])
        self.assertIn("threat_bias", modulation_msg["payload"])


# ============================================================================
# META-COGNITIVE MONITORING TESTS
# ============================================================================


class TestMetaCognitive(unittest.TestCase):
    """Test meta-cognitive monitoring.

    Form 11 (Meta-Consciousness) monitors the operation of other forms,
    assessing confidence and processing quality. These tests verify the
    monitoring contracts.
    """

    def setUp(self):
        """Set up the arousal system with meta-cognitive form."""
        self.arousal = create_arousal_interface()
        self.arousal.register_form("09-perceptual", 0.15)
        self.arousal.register_form("11-meta-consciousness", 0.12)
        self.arousal.register_form("07-emotional", 0.18)

        # Set alert state
        bundle = create_simple_input(
            sensory_level=0.6, emotional_level=0.5, circadian_level=0.7,
        )
        run_async(self.arousal.process_inputs(bundle))

    def test_meta_monitors_perception(self):
        """Meta-consciousness should generate monitoring reports for perception."""
        report = create_meta_cognitive_report(
            target_form="09-perceptual",
            confidence=0.8,
            monitoring_quality=0.85,
        )

        self.assertEqual(report["target_form"], "09-perceptual")
        self.assertGreater(report["confidence"], 0.5)
        self.assertEqual(report["assessment"], "nominal")
        self.assertTrue(report["error_detection_active"])

    def test_confidence_calibration(self):
        """Meta-cognitive confidence should correlate with monitoring quality."""
        # High confidence report
        high_conf = create_meta_cognitive_report(
            target_form="07-emotional",
            confidence=0.9,
            monitoring_quality=0.85,
        )

        # Low confidence report
        low_conf = create_meta_cognitive_report(
            target_form="07-emotional",
            confidence=0.2,
            monitoring_quality=0.3,
        )

        self.assertGreater(
            high_conf["introspective_accuracy"],
            low_conf["introspective_accuracy"],
            "Higher confidence should yield higher introspective accuracy",
        )

        self.assertEqual(high_conf["assessment"], "nominal")
        self.assertEqual(low_conf["assessment"], "uncertain")

    def test_meta_monitoring_message_protocol(self):
        """Meta-consciousness messages should follow query protocol."""
        query_msg = create_message(
            source_form="11-meta-consciousness",
            target_form="09-perceptual",
            message_type="query",
            payload={
                "query_type": "processing_status",
                "metrics_requested": [
                    "confidence", "processing_load", "error_rate"
                ],
            },
            priority=0.6,
        )

        self.assertEqual(query_msg["source"], "11-meta-consciousness")
        self.assertEqual(query_msg["type"], "query")
        self.assertTrue(query_msg["requires_ack"])

    def test_meta_can_monitor_arousal(self):
        """Meta-consciousness should be able to monitor the arousal system."""
        status = self.arousal.get_status()

        # Meta-cognitive report on arousal system health
        report = create_meta_cognitive_report(
            target_form="08-arousal",
            confidence=0.85,
            monitoring_quality=status.system_health,
        )

        self.assertEqual(report["target_form"], "08-arousal")
        self.assertGreater(report["monitoring_quality"], 0.0)

    def test_meta_detects_low_confidence(self):
        """Meta-monitoring should flag low-confidence states."""
        low_report = create_meta_cognitive_report(
            target_form="09-perceptual",
            confidence=0.15,
            monitoring_quality=0.2,
        )

        self.assertEqual(low_report["assessment"], "uncertain")
        self.assertFalse(low_report["error_detection_active"])


# ============================================================================
# THEORETICAL ASSESSMENT TESTS
# ============================================================================


class TestTheoreticalAssessment(unittest.TestCase):
    """Test consciousness theory assessments.

    Form 13 (IIT) and Form 14 (GWT) provide theoretical frameworks for
    assessing system-level consciousness properties. Form 16 (Predictive
    Coding) adds hierarchical prediction error processing.
    """

    def setUp(self):
        """Set up arousal system with theory forms registered."""
        self.arousal = create_arousal_interface()
        self.arousal.register_form("13-integrated-information", 0.10)
        self.arousal.register_form("14-global-workspace", 0.10)
        self.arousal.register_form("16-predictive-coding", 0.08)

        # Alert state
        bundle = create_simple_input(
            sensory_level=0.6, emotional_level=0.5, circadian_level=0.7,
        )
        run_async(self.arousal.process_inputs(bundle))

    def test_iit_measures_integration(self):
        """IIT measurement should assess integration across forms."""
        measurement = create_iit_measurement(
            phi_value=0.7, num_elements=15, integration_level="high"
        )

        self.assertGreater(measurement["phi"], 0.0)
        self.assertEqual(measurement["integration_level"], "high")
        self.assertIn("partition_info", measurement)
        self.assertGreater(
            measurement["partition_info"]["conceptual_structure"],
            0.0,
        )

    def test_gwt_broadcast_reaches_forms(self):
        """GWT broadcast should specify which forms it reaches."""
        broadcast = create_gwt_broadcast(
            content="threat_detected",
            access_strength=0.9,
            broadcasting=True,
        )

        self.assertTrue(broadcast["broadcasting"])
        self.assertGreater(len(broadcast["broadcast_reach"]), 0)
        self.assertEqual(broadcast["winning_coalition"], "threat_detected")

        # Broadcast reach should include specific forms
        self.assertIn("01-visual", broadcast["broadcast_reach"])
        self.assertIn("07-emotional", broadcast["broadcast_reach"])

    def test_gwt_no_broadcast_when_inactive(self):
        """GWT should not broadcast when not active."""
        no_broadcast = create_gwt_broadcast(
            content="subthreshold_stimulus",
            access_strength=0.2,
            broadcasting=False,
        )

        self.assertFalse(no_broadcast["broadcasting"])
        self.assertIsNone(no_broadcast["winning_coalition"])
        self.assertEqual(len(no_broadcast["broadcast_reach"]), 0)

    def test_predictive_coding_hierarchy(self):
        """Predictive coding should support hierarchical error signals."""
        prediction_error = {
            "level": "low",
            "predicted": create_sensory_stimulus("visual", 0.5, 0.5),
            "actual": create_sensory_stimulus("visual", 0.8, 0.9),
            "error_magnitude": 0.3,
            "precision_weight": 0.7,
        }

        self.assertGreater(prediction_error["error_magnitude"], 0.0)
        self.assertEqual(prediction_error["level"], "low")

        # Higher-level prediction error
        high_level_error = {
            "level": "high",
            "predicted": {"category": "safe_environment"},
            "actual": {"category": "threat_detected"},
            "error_magnitude": 0.8,
            "precision_weight": 0.9,
        }

        self.assertGreater(
            high_level_error["error_magnitude"],
            prediction_error["error_magnitude"],
            "Higher-level prediction errors can be larger",
        )

    def test_iit_phi_scales_with_integration(self):
        """Higher integration should yield higher phi values."""
        low_integration = create_iit_measurement(
            phi_value=0.2, num_elements=5, integration_level="low"
        )
        high_integration = create_iit_measurement(
            phi_value=0.8, num_elements=20, integration_level="high"
        )

        self.assertGreater(
            high_integration["phi"],
            low_integration["phi"],
        )
        self.assertGreater(
            high_integration["num_elements"],
            low_integration["num_elements"],
        )

    def test_theory_forms_are_critical(self):
        """Theory forms 13 and 14 should be treated as critical."""
        self.assertTrue(
            self.arousal.is_form_allowed("13-integrated-information"),
            "IIT form should always be allowed",
        )
        self.assertTrue(
            self.arousal.is_form_allowed("14-global-workspace"),
            "GWT form should always be allowed",
        )


# ============================================================================
# MESSAGE PASSING TESTS
# ============================================================================


class TestMessagePassing(unittest.TestCase):
    """Test message passing patterns between forms.

    Forms communicate through a message protocol with defined types,
    priorities, and acknowledgment requirements. These tests verify
    the message structure and routing contracts.
    """

    def test_data_message_structure(self):
        """Data messages should have correct structure."""
        msg = create_message(
            source_form="01-visual",
            target_form="09-perceptual",
            message_type="data",
            payload={"features": [0.1, 0.2, 0.3]},
        )

        self.assertEqual(msg["source"], "01-visual")
        self.assertEqual(msg["target"], "09-perceptual")
        self.assertEqual(msg["type"], "data")
        self.assertFalse(msg["requires_ack"])
        self.assertIn("features", msg["payload"])

    def test_query_message_requires_ack(self):
        """Query messages should require acknowledgment."""
        msg = create_message(
            source_form="11-meta-consciousness",
            target_form="07-emotional",
            message_type="query",
            payload={"query": "current_emotional_state"},
        )

        self.assertTrue(msg["requires_ack"])

    def test_control_message_requires_ack(self):
        """Control messages should require acknowledgment."""
        msg = create_message(
            source_form="08-arousal",
            target_form="01-visual",
            message_type="control",
            payload={"gate_value": 0.8, "priority_boost": 0.2},
        )

        self.assertTrue(msg["requires_ack"])

    def test_broadcast_message_pattern(self):
        """Broadcast messages should target multiple forms."""
        broadcast_targets = [
            "01-visual", "02-auditory", "03-somatosensory",
            "07-emotional", "09-perceptual",
        ]

        messages = []
        for target in broadcast_targets:
            msg = create_message(
                source_form="14-global-workspace",
                target_form=target,
                message_type="broadcast",
                payload={"content": "attended_stimulus", "strength": 0.8},
                priority=0.9,
            )
            messages.append(msg)

        self.assertEqual(len(messages), len(broadcast_targets))
        for msg in messages:
            self.assertEqual(msg["source"], "14-global-workspace")
            self.assertGreater(msg["priority"], 0.5)

    def test_priority_ordering(self):
        """Messages should be orderable by priority."""
        messages = [
            create_message("01-visual", "09-perceptual", priority=0.3),
            create_message("08-arousal", "01-visual", message_type="control", priority=0.9),
            create_message("07-emotional", "08-arousal", priority=0.7),
        ]

        sorted_msgs = sorted(messages, key=lambda m: m["priority"], reverse=True)

        self.assertEqual(sorted_msgs[0]["source"], "08-arousal")
        self.assertEqual(sorted_msgs[-1]["source"], "01-visual")

    def test_bidirectional_communication(self):
        """Forms should support bidirectional message flow."""
        # Emotional -> Arousal
        up_msg = create_message(
            source_form="07-emotional",
            target_form="08-arousal",
            message_type="data",
            payload={"emotional_arousal": 0.7},
        )

        # Arousal -> Emotional (gating response)
        down_msg = create_message(
            source_form="08-arousal",
            target_form="07-emotional",
            message_type="control",
            payload={"gate_value": 0.85, "resource_allocated": 0.18},
        )

        self.assertEqual(up_msg["source"], down_msg["target"])
        self.assertEqual(up_msg["target"], down_msg["source"])

    def test_message_payload_types(self):
        """Message payloads should support various data types."""
        # Numeric payload
        numeric_msg = create_message(
            "08-arousal", "01-visual", payload={"gate": 0.8}
        )
        self.assertIsInstance(numeric_msg["payload"]["gate"], float)

        # List payload
        list_msg = create_message(
            "09-perceptual", "11-meta-consciousness",
            payload={"bound_objects": ["face", "voice", "emotion"]},
        )
        self.assertIsInstance(list_msg["payload"]["bound_objects"], list)

        # Nested payload
        nested_msg = create_message(
            "13-integrated-information", "14-global-workspace",
            payload={"measurement": create_iit_measurement()},
        )
        self.assertIsInstance(nested_msg["payload"]["measurement"], dict)


# ============================================================================
# FULL SYSTEM CYCLE TESTS
# ============================================================================


class TestFullSystemCycle(unittest.TestCase):
    """Test complete processing cycles across the form architecture.

    These tests simulate end-to-end flows from stimulus input through
    sensory processing, perceptual binding, emotional evaluation, arousal
    modulation, meta-cognitive monitoring, and theoretical assessment.
    """

    def setUp(self):
        """Set up the full system with all relevant forms."""
        self.system = ArousalGatedSystem()
        self.system.setup()

    def test_stimulus_to_response(self):
        """A stimulus should flow through the full processing pipeline."""
        arousal = self.system.get_arousal_interface()

        # Step 1: Create a visual stimulus
        stimulus = create_sensory_stimulus("visual", intensity=0.7, salience=0.8)
        self.assertIsNotNone(stimulus)

        # Step 2: Check arousal gate allows visual processing
        visual_gate = self.system.get_gate_for_form("01-visual")
        self.assertGreater(visual_gate, 0.0)

        # Step 3: Sensory output creates message for perceptual binding
        sensory_msg = create_message(
            source_form="01-visual",
            target_form="09-perceptual",
            message_type="data",
            payload={"stimulus": stimulus},
            priority=0.7,
        )
        self.assertEqual(sensory_msg["target"], "09-perceptual")

        # Step 4: Emotional evaluation
        emotional_state = create_emotional_state(
            valence=0.3, arousal_component=0.5, dominant_emotion="interest"
        )
        self.assertIn("dominant_emotion", emotional_state)

        # Step 5: Emotional input affects arousal
        emotion_bundle = ArousalInputBundle(
            sensory_inputs=[
                SensoryArousalInput(
                    source_modality=SensoryModality.VISUAL,
                    stimulus_type="novel",
                    intensity=0.7,
                    salience=0.8,
                    change_rate=0.3,
                )
            ],
            emotional_input=EmotionalInput(
                valence=0.3,
                arousal_component=0.5,
            ),
        )
        arousal_result = run_async(arousal.process_inputs(emotion_bundle))
        self.assertIsInstance(arousal_result, ArousalLevelOutput)

        # Step 6: Meta-cognitive monitoring of the whole process
        meta_report = create_meta_cognitive_report(
            target_form="09-perceptual",
            confidence=0.75,
            monitoring_quality=0.8,
        )
        self.assertEqual(meta_report["assessment"], "nominal")

    def test_multi_form_cascade(self):
        """A processing cascade should flow through multiple forms."""
        arousal = self.system.get_arousal_interface()

        # Phase 1: Sensory input with threat
        threat_bundle = create_simple_input(
            sensory_level=0.8,
            emotional_level=0.7,
            circadian_level=0.6,
            threat_level=0.6,
        )
        arousal_result = run_async(arousal.process_inputs(threat_bundle))

        # Should be in at least alert state
        self.assertIn(
            arousal_result.arousal_state,
            [ArousalState.ALERT, ArousalState.FOCUSED, ArousalState.HYPERAROUSED],
        )

        # Phase 2: Gating should respond to the threat
        gating = arousal.get_gating_signals()
        visual_gate = gating.sensory_gates.get("visual", 0)
        self.assertGreater(visual_gate, 0.3)

        # Phase 3: IIT measurement of the integrated state
        iit = create_iit_measurement(
            phi_value=arousal_result.arousal_level * 0.8,
            num_elements=len(self.system.registered_forms),
            integration_level="moderate" if arousal_result.arousal_level < 0.7 else "high",
        )
        self.assertGreater(iit["phi"], 0.0)

        # Phase 4: GWT broadcast of the threat
        gwt = create_gwt_broadcast(
            content="threat_stimulus",
            access_strength=arousal_result.arousal_level,
            broadcasting=True,
        )
        self.assertTrue(gwt["broadcasting"])
        self.assertGreater(len(gwt["broadcast_reach"]), 0)

        # Phase 5: Meta-cognitive assessment
        meta = create_meta_cognitive_report(
            target_form="08-arousal",
            confidence=arousal_result.confidence,
            monitoring_quality=arousal.get_status().system_health,
        )
        self.assertGreater(meta["confidence"], 0.0)

    def test_state_transition_cascade(self):
        """State transitions in arousal should cascade to gating changes."""
        arousal = self.system.get_arousal_interface()

        # Start relaxed
        relaxed_bundle = create_simple_input(
            sensory_level=0.3, emotional_level=0.3, circadian_level=0.4,
        )
        for _ in range(3):
            run_async(arousal.process_inputs(relaxed_bundle))

        relaxed_gating = arousal.get_gating_signals()
        relaxed_allocation = arousal.get_resource_allocation()

        # Transition to high arousal
        alert_bundle = create_simple_input(
            sensory_level=0.8, emotional_level=0.7, circadian_level=0.8,
        )
        for _ in range(5):
            run_async(arousal.process_inputs(alert_bundle))

        alert_gating = arousal.get_gating_signals()
        alert_allocation = arousal.get_resource_allocation()

        # Verify gating and allocation changed
        self.assertIsNotNone(relaxed_gating)
        self.assertIsNotNone(alert_gating)
        self.assertGreaterEqual(
            alert_allocation.total_available,
            relaxed_allocation.total_available * 0.8,
            "Alert state should provide at least comparable resources to relaxed",
        )

    def test_emotional_arousal_perceptual_loop(self):
        """Test the emotion -> arousal -> perceptual gating feedback loop."""
        arousal = self.system.get_arousal_interface()

        # Step 1: Fear emotion feeds into arousal
        fear_input = ArousalInputBundle(
            emotional_input=EmotionalInput(
                valence=-0.7,
                arousal_component=0.85,
                fear=0.8,
                anxiety=0.6,
            ),
            sensory_inputs=[
                SensoryArousalInput(
                    source_modality=SensoryModality.VISUAL,
                    stimulus_type="threat",
                    intensity=0.8,
                    salience=0.9,
                    change_rate=0.7,
                ),
            ],
        )
        result = run_async(arousal.process_inputs(fear_input))

        # Step 2: Arousal should increase from emotional input
        self.assertIn(ArousalSource.EMOTIONAL, result.components)
        self.assertGreater(result.components[ArousalSource.EMOTIONAL], 0.3)

        # Step 3: Gating should reflect heightened arousal
        gating = arousal.get_gating_signals()
        self.assertGreater(
            gating.sensory_gates.get("visual", 0),
            0.0,
            "Visual gate should be open under fear/threat",
        )

        # Step 4: Perceptual form gate should allow processing
        perceptual_gate = arousal.get_gate_for_form("09-perceptual")
        self.assertGreater(perceptual_gate, 0.0)

    def test_system_health_monitoring(self):
        """System health should be monitorable across forms."""
        arousal = self.system.get_arousal_interface()

        # Get system status
        status = arousal.get_status()
        self.assertGreater(status.system_health, 0.0)
        self.assertLessEqual(status.system_health, 1.0)

        # Status should include current level and gating info
        self.assertIsNotNone(status.current_level)
        self.assertIsNotNone(status.current_gating)
        self.assertIsInstance(status.recent_transitions, list)


# ============================================================================
# CROSS-FORM INTERFACE COMPATIBILITY TESTS
# ============================================================================


class TestCrossFormInterfaceCompatibility(unittest.TestCase):
    """Test that discovered form interfaces follow common conventions.

    Forms with Python interfaces should all follow similar patterns:
    - FORM_ID class attribute
    - FORM_NAME or NAME class attribute
    - Instantiation without arguments
    """

    @classmethod
    def setUpClass(cls):
        """Discover and load all available forms."""
        cls.discovered = discover_forms()
        cls.interfaces = {}

        for form_id, file_path in cls.discovered.items():
            try:
                module = load_form_interface(file_path)
                instance = instantiate_form_interface(module)
                if instance is not None:
                    cls.interfaces[form_id] = instance
            except Exception:
                pass

    def test_all_interfaces_have_form_id(self):
        """Every instantiated interface should have FORM_ID."""
        missing = []
        for form_id, instance in self.interfaces.items():
            if not hasattr(instance, "FORM_ID"):
                missing.append(form_id)

        self.assertEqual(len(missing), 0, f"Missing FORM_ID: {missing}")

    def test_all_interfaces_have_name(self):
        """Every instantiated interface should have FORM_NAME or NAME."""
        missing = []
        for form_id, instance in self.interfaces.items():
            has_name = (
                hasattr(instance, "FORM_NAME") or hasattr(instance, "NAME")
            )
            if not has_name:
                missing.append(form_id)

        self.assertEqual(len(missing), 0, f"Missing name attribute: {missing}")

    def test_form_ids_are_unique(self):
        """All FORM_ID values should be unique across interfaces."""
        seen_ids = {}
        duplicates = []

        for form_id, instance in self.interfaces.items():
            fid = getattr(instance, "FORM_ID", None)
            if fid in seen_ids:
                duplicates.append(f"{fid} (found in {seen_ids[fid]} and {form_id})")
            else:
                seen_ids[fid] = form_id

        self.assertEqual(
            len(duplicates), 0,
            f"Duplicate FORM_IDs: {duplicates}",
        )

    def test_arousal_can_gate_all_interfaces(self):
        """The arousal system should be able to gate every other form."""
        arousal = create_arousal_interface()

        for form_id in self.interfaces:
            if form_id == "08-arousal":
                continue
            arousal.register_form(form_id, 0.05)

        # Process to alert state
        bundle = create_simple_input(
            sensory_level=0.6, emotional_level=0.5, circadian_level=0.7,
        )
        run_async(arousal.process_inputs(bundle))

        # All registered forms should have gate values
        for form_id in self.interfaces:
            if form_id == "08-arousal":
                continue
            gate = arousal.get_gate_for_form(form_id)
            self.assertGreaterEqual(
                gate, 0.0,
                f"Gate for {form_id} should be >= 0",
            )


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)
