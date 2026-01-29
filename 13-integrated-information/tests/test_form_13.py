#!/usr/bin/env python3
"""
Test Suite for Form 13: Integrated Information Theory (IIT) Consciousness.

Tests cover:
- All enumerations (IntegrationLevel, ComplexType, CauseEffectStructure, etc.)
- All input/output dataclasses
- IITComputationEngine
- IITConsciousnessInterface (main interface)
- Convenience functions
- Integration tests for phi computation pipeline
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
    IntegrationLevel,
    ComplexType,
    CauseEffectStructure,
    PartitionType,
    ExperienceQuality,
    # Input dataclasses
    SystemElement,
    IITInput,
    Partition,
    # Output dataclasses
    Concept,
    ConceptualStructure,
    Complex,
    IITOutput,
    IITSystemStatus,
    # Engine
    IITComputationEngine,
    # Main interface
    IITConsciousnessInterface,
    # Convenience functions
    create_iit_interface,
    create_simple_iit_input,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestIntegrationLevel(unittest.TestCase):
    """Tests for IntegrationLevel enumeration."""

    def test_all_levels_exist(self):
        """All integration levels should be defined."""
        levels = [
            IntegrationLevel.NONE,
            IntegrationLevel.MINIMAL,
            IntegrationLevel.LOW,
            IntegrationLevel.MODERATE,
            IntegrationLevel.HIGH,
            IntegrationLevel.VERY_HIGH,
            IntegrationLevel.MAXIMAL,
        ]
        self.assertEqual(len(levels), 7)

    def test_level_values(self):
        """Levels should have expected string values."""
        self.assertEqual(IntegrationLevel.NONE.value, "none")
        self.assertEqual(IntegrationLevel.HIGH.value, "high")
        self.assertEqual(IntegrationLevel.MAXIMAL.value, "maximal")


class TestComplexType(unittest.TestCase):
    """Tests for ComplexType enumeration."""

    def test_all_types_exist(self):
        """All complex types should be defined."""
        types = [
            ComplexType.MAIN_COMPLEX,
            ComplexType.SUB_COMPLEX,
            ComplexType.BACKGROUND,
            ComplexType.ISOLATED,
        ]
        self.assertEqual(len(types), 4)

    def test_type_values(self):
        """Types should have expected string values."""
        self.assertEqual(ComplexType.MAIN_COMPLEX.value, "main_complex")
        self.assertEqual(ComplexType.SUB_COMPLEX.value, "sub_complex")


class TestCauseEffectStructure(unittest.TestCase):
    """Tests for CauseEffectStructure enumeration."""

    def test_all_structures_exist(self):
        """All cause-effect structure types should be defined."""
        structures = [
            CauseEffectStructure.CAUSE,
            CauseEffectStructure.EFFECT,
            CauseEffectStructure.CAUSE_EFFECT,
            CauseEffectStructure.NULL,
        ]
        self.assertEqual(len(structures), 4)


class TestPartitionType(unittest.TestCase):
    """Tests for PartitionType enumeration."""

    def test_all_types_exist(self):
        """All partition types should be defined."""
        types = [
            PartitionType.UNIDIRECTIONAL_CUT,
            PartitionType.BIDIRECTIONAL_CUT,
            PartitionType.MINIMUM_PARTITION,
        ]
        self.assertEqual(len(types), 3)


class TestExperienceQuality(unittest.TestCase):
    """Tests for ExperienceQuality enumeration."""

    def test_all_qualities_exist(self):
        """All experience qualities should be defined."""
        qualities = [
            ExperienceQuality.INTRINSIC,
            ExperienceQuality.STRUCTURED,
            ExperienceQuality.SPECIFIC,
            ExperienceQuality.UNIFIED,
            ExperienceQuality.DEFINITE,
        ]
        self.assertEqual(len(qualities), 5)


# ============================================================================
# INPUT DATACLASS TESTS
# ============================================================================

class TestSystemElement(unittest.TestCase):
    """Tests for SystemElement dataclass."""

    def test_creation(self):
        """Should create element with required fields."""
        elem = SystemElement(element_id="e0", state=1, label="neuron_0")
        self.assertEqual(elem.element_id, "e0")
        self.assertEqual(elem.state, 1)
        self.assertEqual(elem.label, "neuron_0")

    def test_to_dict(self):
        """Should convert to dictionary."""
        elem = SystemElement(element_id="e1", state=0)
        d = elem.to_dict()
        self.assertEqual(d["element_id"], "e1")
        self.assertEqual(d["state"], 0)
        self.assertIn("timestamp", d)


class TestIITInput(unittest.TestCase):
    """Tests for IITInput dataclass."""

    def test_creation(self):
        """Should create IIT input."""
        inp = IITInput(
            system_state=[1, 0, 1],
            connectivity_matrix=[
                [0.0, 0.5, 0.3],
                [0.5, 0.0, 0.4],
                [0.3, 0.4, 0.0],
            ],
        )
        self.assertEqual(len(inp.system_state), 3)
        self.assertEqual(len(inp.connectivity_matrix), 3)

    def test_with_labels(self):
        """Should accept element labels."""
        inp = IITInput(
            system_state=[1, 1],
            connectivity_matrix=[[0.0, 0.8], [0.8, 0.0]],
            element_labels=["A", "B"],
        )
        self.assertEqual(inp.element_labels, ["A", "B"])

    def test_to_dict(self):
        """Should convert to dictionary."""
        inp = IITInput(
            system_state=[1, 0],
            connectivity_matrix=[[0.0, 0.5], [0.5, 0.0]],
        )
        d = inp.to_dict()
        self.assertEqual(d["num_elements"], 2)
        self.assertFalse(d["has_tpm"])


class TestPartition(unittest.TestCase):
    """Tests for Partition dataclass."""

    def test_creation(self):
        """Should create partition."""
        part = Partition(
            part_a=[0, 1],
            part_b=[2, 3],
            partition_type=PartitionType.MINIMUM_PARTITION,
        )
        self.assertEqual(part.part_a, [0, 1])
        self.assertEqual(part.part_b, [2, 3])

    def test_to_dict(self):
        """Should convert to dictionary."""
        part = Partition(part_a=[0], part_b=[1])
        d = part.to_dict()
        self.assertIn("part_a", d)
        self.assertIn("partition_type", d)


# ============================================================================
# OUTPUT DATACLASS TESTS
# ============================================================================

class TestConcept(unittest.TestCase):
    """Tests for Concept dataclass."""

    def test_creation(self):
        """Should create a concept."""
        concept = Concept(
            mechanism=[0, 1],
            cause_repertoire={"0": 0.5, "1": 0.3},
            effect_repertoire={"0": 0.4, "1": 0.6},
            phi_cause=0.4,
            phi_effect=0.5,
            phi=0.4,
        )
        self.assertEqual(concept.mechanism, [0, 1])
        self.assertEqual(concept.phi, 0.4)

    def test_to_dict(self):
        """Should convert to dictionary."""
        concept = Concept(
            mechanism=[0],
            cause_repertoire={"0": 0.5},
            effect_repertoire={"0": 0.6},
            phi_cause=0.3,
            phi_effect=0.4,
            phi=0.3,
        )
        d = concept.to_dict()
        self.assertIn("phi", d)
        self.assertIn("mechanism", d)


class TestConceptualStructure(unittest.TestCase):
    """Tests for ConceptualStructure dataclass."""

    def test_creation(self):
        """Should create conceptual structure."""
        cs = ConceptualStructure(
            concepts=[],
            num_concepts=0,
            total_phi=0.0,
        )
        self.assertEqual(cs.num_concepts, 0)
        self.assertEqual(cs.total_phi, 0.0)

    def test_to_dict(self):
        """Should convert to dictionary."""
        cs = ConceptualStructure(
            concepts=[],
            num_concepts=0,
            total_phi=0.0,
            structure_signature="empty",
        )
        d = cs.to_dict()
        self.assertEqual(d["num_concepts"], 0)
        self.assertEqual(d["structure_signature"], "empty")


class TestComplex(unittest.TestCase):
    """Tests for Complex dataclass."""

    def test_creation(self):
        """Should create complex."""
        c = Complex(
            elements=[0, 1, 2],
            phi=0.65,
            complex_type=ComplexType.MAIN_COMPLEX,
            is_main=True,
        )
        self.assertEqual(c.elements, [0, 1, 2])
        self.assertEqual(c.phi, 0.65)
        self.assertTrue(c.is_main)

    def test_to_dict(self):
        """Should convert to dictionary."""
        c = Complex(elements=[0], phi=0.1, complex_type=ComplexType.SUB_COMPLEX)
        d = c.to_dict()
        self.assertIn("phi", d)
        self.assertEqual(d["complex_type"], "sub_complex")


class TestIITOutput(unittest.TestCase):
    """Tests for IITOutput dataclass."""

    def test_creation(self):
        """Should create IIT output."""
        output = IITOutput(
            phi=0.75,
            integration_level=IntegrationLevel.HIGH,
            main_complex=Complex(elements=[0, 1], phi=0.75, complex_type=ComplexType.MAIN_COMPLEX, is_main=True),
            all_complexes=[],
            conceptual_structure=ConceptualStructure(concepts=[], num_concepts=0, total_phi=0.0),
            minimum_partition=Partition(part_a=[0], part_b=[1]),
            experience_qualities=[ExperienceQuality.INTRINSIC],
            num_elements=2,
        )
        self.assertEqual(output.phi, 0.75)
        self.assertEqual(output.integration_level, IntegrationLevel.HIGH)


# ============================================================================
# ENGINE TESTS
# ============================================================================

class TestIITComputationEngine(unittest.TestCase):
    """Tests for IITComputationEngine."""

    def setUp(self):
        """Set up test engine."""
        self.engine = IITComputationEngine()

    def test_phi_single_element(self):
        """Single element should have zero phi."""
        phi = self.engine.compute_phi_for_subset(
            [0], [1], [[0.0]]
        )
        self.assertEqual(phi, 0.0)

    def test_phi_disconnected_elements(self):
        """Disconnected elements should have zero phi."""
        phi = self.engine.compute_phi_for_subset(
            [0, 1],
            [1, 1],
            [[0.0, 0.0], [0.0, 0.0]]
        )
        self.assertEqual(phi, 0.0)

    def test_phi_connected_elements(self):
        """Connected elements should have positive phi."""
        phi = self.engine.compute_phi_for_subset(
            [0, 1],
            [1, 1],
            [[0.0, 0.8], [0.8, 0.0]]
        )
        self.assertGreater(phi, 0.0)

    def test_phi_increases_with_connectivity(self):
        """Phi should increase with stronger connectivity."""
        weak_phi = self.engine.compute_phi_for_subset(
            [0, 1], [1, 1], [[0.0, 0.2], [0.2, 0.0]]
        )
        strong_phi = self.engine.compute_phi_for_subset(
            [0, 1], [1, 1], [[0.0, 0.9], [0.9, 0.0]]
        )
        self.assertGreater(strong_phi, weak_phi)

    def test_compute_concept(self):
        """Should compute a concept for a mechanism."""
        concept = self.engine.compute_concept(
            [0], [1, 1, 0],
            [[0.0, 0.5, 0.3], [0.5, 0.0, 0.4], [0.3, 0.4, 0.0]]
        )
        self.assertIsInstance(concept, Concept)
        self.assertEqual(concept.mechanism, [0])

    def test_find_minimum_partition(self):
        """Should find the minimum partition."""
        partition = self.engine.find_minimum_partition(
            [0, 1, 2],
            [[0.0, 0.9, 0.1], [0.9, 0.0, 0.1], [0.1, 0.1, 0.0]]
        )
        self.assertIsInstance(partition, Partition)
        self.assertTrue(len(partition.part_a) > 0)
        self.assertTrue(len(partition.part_b) > 0)

    def test_classify_integration_level(self):
        """Should classify phi into correct levels."""
        self.assertEqual(self.engine.classify_integration_level(0.0), IntegrationLevel.NONE)
        self.assertEqual(self.engine.classify_integration_level(0.1), IntegrationLevel.MINIMAL)
        self.assertEqual(self.engine.classify_integration_level(0.3), IntegrationLevel.LOW)
        self.assertEqual(self.engine.classify_integration_level(0.5), IntegrationLevel.MODERATE)
        self.assertEqual(self.engine.classify_integration_level(0.7), IntegrationLevel.HIGH)
        self.assertEqual(self.engine.classify_integration_level(0.9), IntegrationLevel.VERY_HIGH)
        self.assertEqual(self.engine.classify_integration_level(1.5), IntegrationLevel.MAXIMAL)

    def test_determine_experience_qualities(self):
        """Should determine experience qualities based on phi."""
        cs = ConceptualStructure(concepts=[], num_concepts=0, total_phi=0.0)
        # Zero phi should give no qualities
        qualities = self.engine.determine_experience_qualities(0.0, cs)
        self.assertEqual(len(qualities), 0)

        # High phi with concepts should give multiple qualities
        concept = Concept(mechanism=[0], cause_repertoire={}, effect_repertoire={},
                         phi_cause=0.5, phi_effect=0.5, phi=0.5)
        cs_rich = ConceptualStructure(concepts=[concept, concept], num_concepts=2, total_phi=1.0)
        qualities = self.engine.determine_experience_qualities(0.7, cs_rich)
        self.assertIn(ExperienceQuality.INTRINSIC, qualities)
        self.assertIn(ExperienceQuality.STRUCTURED, qualities)
        self.assertIn(ExperienceQuality.UNIFIED, qualities)


# ============================================================================
# MAIN INTERFACE TESTS
# ============================================================================

class TestIITConsciousnessInterface(unittest.TestCase):
    """Tests for IITConsciousnessInterface."""

    def setUp(self):
        """Set up test interface."""
        self.interface = IITConsciousnessInterface()

    def test_form_metadata(self):
        """Should have correct form metadata."""
        self.assertEqual(self.interface.FORM_ID, "13-integrated-information")
        self.assertEqual(self.interface.FORM_NAME, "Integrated Information Theory (IIT)")

    def test_initialize(self):
        """Should initialize successfully."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.initialize())
        finally:
            loop.close()
        self.assertTrue(self.interface._is_initialized)

    def test_compute_phi(self):
        """Should compute phi for a system."""
        iit_input = IITInput(
            system_state=[1, 1, 0, 1],
            connectivity_matrix=[
                [0.0, 0.7, 0.3, 0.1],
                [0.7, 0.0, 0.5, 0.2],
                [0.3, 0.5, 0.0, 0.6],
                [0.1, 0.2, 0.6, 0.0],
            ],
            element_labels=["A", "B", "C", "D"],
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(self.interface.compute_phi(iit_input))
        finally:
            loop.close()

        self.assertIsInstance(output, IITOutput)
        self.assertGreaterEqual(output.phi, 0.0)
        self.assertIsNotNone(output.main_complex)
        self.assertTrue(output.main_complex.is_main)
        self.assertIsNotNone(output.conceptual_structure)
        self.assertEqual(output.num_elements, 4)

    def test_identify_main_complex(self):
        """Should identify the main complex."""
        iit_input = IITInput(
            system_state=[1, 1, 1],
            connectivity_matrix=[
                [0.0, 0.8, 0.1],
                [0.8, 0.0, 0.1],
                [0.1, 0.1, 0.0],
            ],
        )

        loop = asyncio.new_event_loop()
        try:
            main = loop.run_until_complete(self.interface.identify_main_complex(iit_input))
        finally:
            loop.close()

        self.assertIsInstance(main, Complex)
        self.assertTrue(main.is_main)

    def test_get_integration_structure(self):
        """Should return integration structure details."""
        iit_input = IITInput(
            system_state=[1, 0, 1],
            connectivity_matrix=[
                [0.0, 0.5, 0.3],
                [0.5, 0.0, 0.4],
                [0.3, 0.4, 0.0],
            ],
            element_labels=["X", "Y", "Z"],
        )

        loop = asyncio.new_event_loop()
        try:
            structure = loop.run_until_complete(
                self.interface.get_integration_structure(iit_input)
            )
        finally:
            loop.close()

        self.assertIn("system_phi", structure)
        self.assertIn("element_contributions", structure)
        self.assertIn("pairwise_integration", structure)
        self.assertIn("X", structure["element_contributions"])

    def test_to_dict(self):
        """Should convert state to dictionary."""
        d = self.interface.to_dict()
        self.assertEqual(d["form_id"], "13-integrated-information")
        self.assertEqual(d["form_name"], "Integrated Information Theory (IIT)")
        self.assertFalse(d["is_initialized"])
        self.assertEqual(d["computation_count"], 0)

    def test_get_status(self):
        """Should return system status."""
        status = self.interface.get_status()
        self.assertIsInstance(status, IITSystemStatus)
        self.assertFalse(status.is_initialized)
        self.assertEqual(status.computation_count, 0)

    def test_computation_count_increments(self):
        """Should track computation count."""
        iit_input = IITInput(
            system_state=[1, 1],
            connectivity_matrix=[[0.0, 0.5], [0.5, 0.0]],
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.interface.compute_phi(iit_input))
            loop.run_until_complete(self.interface.compute_phi(iit_input))
        finally:
            loop.close()

        self.assertEqual(self.interface._computation_count, 2)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_create_iit_interface(self):
        """Should create a new interface."""
        interface = create_iit_interface()
        self.assertIsInstance(interface, IITConsciousnessInterface)
        self.assertEqual(interface.FORM_ID, "13-integrated-information")

    def test_create_simple_iit_input(self):
        """Should create simple IIT input."""
        inp = create_simple_iit_input(num_elements=3)
        self.assertEqual(len(inp.system_state), 3)
        self.assertEqual(len(inp.connectivity_matrix), 3)
        self.assertEqual(len(inp.element_labels), 3)

    def test_create_simple_iit_input_defaults(self):
        """Should create input with default parameters."""
        inp = create_simple_iit_input()
        self.assertEqual(len(inp.system_state), 4)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIITIntegration(unittest.TestCase):
    """Integration tests for the IIT system."""

    def test_full_computation_pipeline(self):
        """Should complete full IIT computation pipeline."""
        interface = create_iit_interface()

        iit_input = IITInput(
            system_state=[1, 1, 1, 0],
            connectivity_matrix=[
                [0.0, 0.8, 0.2, 0.0],
                [0.8, 0.0, 0.7, 0.1],
                [0.2, 0.7, 0.0, 0.3],
                [0.0, 0.1, 0.3, 0.0],
            ],
            element_labels=["A", "B", "C", "D"],
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(interface.initialize())
            output = loop.run_until_complete(interface.compute_phi(iit_input))
        finally:
            loop.close()

        # Verify output consistency
        self.assertGreaterEqual(output.phi, 0.0)
        self.assertEqual(output.num_elements, 4)
        self.assertIsNotNone(output.main_complex)
        self.assertIsNotNone(output.conceptual_structure)
        self.assertIsNotNone(output.minimum_partition)

        # Status should reflect computation
        status = interface.get_status()
        self.assertTrue(status.is_initialized)
        self.assertEqual(status.computation_count, 1)

    def test_disconnected_system_zero_phi(self):
        """Disconnected system should have zero or near-zero phi."""
        interface = create_iit_interface()

        iit_input = IITInput(
            system_state=[1, 1, 1],
            connectivity_matrix=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

        loop = asyncio.new_event_loop()
        try:
            output = loop.run_until_complete(interface.compute_phi(iit_input))
        finally:
            loop.close()

        self.assertEqual(output.phi, 0.0)
        self.assertEqual(output.integration_level, IntegrationLevel.NONE)

    def test_highly_connected_system(self):
        """Highly connected system should have higher phi than weakly connected."""
        interface_strong = create_iit_interface()
        interface_weak = create_iit_interface()

        strong_input = IITInput(
            system_state=[1, 1, 1],
            connectivity_matrix=[
                [0.0, 0.9, 0.9],
                [0.9, 0.0, 0.9],
                [0.9, 0.9, 0.0],
            ],
        )
        weak_input = IITInput(
            system_state=[1, 1, 1],
            connectivity_matrix=[
                [0.0, 0.1, 0.1],
                [0.1, 0.0, 0.1],
                [0.1, 0.1, 0.0],
            ],
        )

        loop = asyncio.new_event_loop()
        try:
            strong_out = loop.run_until_complete(interface_strong.compute_phi(strong_input))
            weak_out = loop.run_until_complete(interface_weak.compute_phi(weak_input))
        finally:
            loop.close()

        self.assertGreaterEqual(strong_out.phi, weak_out.phi)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
