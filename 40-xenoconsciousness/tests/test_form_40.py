#!/usr/bin/env python3
"""
Test suite for Form 40: Xenoconsciousness (Hypothetical Minds)

Tests cover:
- Enum definitions and completeness
- Dataclass initialization and serialization
- Interface methods and queries
- Seed data initialization
- Cross-form integration readiness
"""

import sys
from pathlib import Path
import pytest
import asyncio


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


# Get the base path
BASE_PATH = Path(__file__).parent.parent
INTERFACE_PATH = BASE_PATH / "interface"

# Load the main interface module
interface_module = load_module_from_path(
    "xenoconsciousness_interface",
    INTERFACE_PATH / "xenoconsciousness_interface.py"
)

# Import Form 40 components from loaded module
XenoMindType = interface_module.XenoMindType
XenoSensoryModality = interface_module.XenoSensoryModality
ConsciousnessIndicatorXeno = interface_module.ConsciousnessIndicatorXeno
SubstrateType = interface_module.SubstrateType
CommunicationParadigm = interface_module.CommunicationParadigm
PhilosophicalFramework = interface_module.PhilosophicalFramework
MaturityLevel = interface_module.MaturityLevel
XenoMindHypothesis = interface_module.XenoMindHypothesis
AlternativeSensoryWorld = interface_module.AlternativeSensoryWorld
SETIConsciousnessProtocol = interface_module.SETIConsciousnessProtocol
SciFiFramework = interface_module.SciFiFramework
CrossSubstrateComparison = interface_module.CrossSubstrateComparison
XenoconsciousnessMaturityState = interface_module.XenoconsciousnessMaturityState
XenoconsciousnessInterface = interface_module.XenoconsciousnessInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestXenoMindType:
    """Tests for XenoMindType enum."""

    def test_type_count(self):
        """Test that we have the expected number of xeno mind types."""
        types = list(XenoMindType)
        assert len(types) >= 10, f"Expected at least 10 types, got {len(types)}"

    def test_biological_types_exist(self):
        """Test biological xeno mind types are defined."""
        biological = [
            XenoMindType.CARBON_BIOLOGICAL,
            XenoMindType.SILICON_BIOLOGICAL,
        ]
        assert len(biological) == 2

    def test_exotic_types_exist(self):
        """Test exotic xeno mind types are defined."""
        exotic = [
            XenoMindType.PLASMA_BASED,
            XenoMindType.QUANTUM_COHERENT,
            XenoMindType.PLANETARY_SCALE,
            XenoMindType.STELLAR_SCALE,
        ]
        assert len(exotic) == 4

    def test_theoretical_types_exist(self):
        """Test theoretical xeno mind types are defined."""
        theoretical = [
            XenoMindType.DIGITAL_SUBSTRATE,
            XenoMindType.BOLTZMANN_BRAIN,
            XenoMindType.HIGHER_DIMENSIONAL,
        ]
        assert len(theoretical) == 3

    def test_type_values_are_strings(self):
        """Test that all type values are strings."""
        for mind_type in XenoMindType:
            assert isinstance(mind_type.value, str)


class TestXenoSensoryModality:
    """Tests for XenoSensoryModality enum."""

    def test_modality_count(self):
        """Test that we have expected sensory modalities."""
        modalities = list(XenoSensoryModality)
        assert len(modalities) >= 8, f"Expected at least 8 modalities, got {len(modalities)}"

    def test_exotic_modalities_exist(self):
        """Test exotic sensory modalities are defined."""
        exotic = [
            XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
            XenoSensoryModality.GRAVITATIONAL,
            XenoSensoryModality.QUANTUM_ENTANGLEMENT,
            XenoSensoryModality.TEMPORAL_PERCEPTION,
        ]
        assert len(exotic) == 4


class TestConsciousnessIndicatorXeno:
    """Tests for ConsciousnessIndicatorXeno enum."""

    def test_indicator_count(self):
        """Test we have expected consciousness indicators."""
        indicators = list(ConsciousnessIndicatorXeno)
        assert len(indicators) >= 6, f"Expected at least 6 indicators, got {len(indicators)}"

    def test_key_indicators_exist(self):
        """Test key consciousness indicators are defined."""
        key_indicators = [
            ConsciousnessIndicatorXeno.BEHAVIORAL_COMPLEXITY,
            ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION,
            ConsciousnessIndicatorXeno.SELF_MODELING,
        ]
        assert len(key_indicators) == 3


class TestSubstrateType:
    """Tests for SubstrateType enum."""

    def test_substrate_types_exist(self):
        """Test substrate types are defined."""
        substrates = list(SubstrateType)
        assert len(substrates) >= 5, f"Expected at least 5 substrates, got {len(substrates)}"

    def test_key_substrates_exist(self):
        """Test key substrate types are defined."""
        key_substrates = [
            SubstrateType.BIOLOGICAL_CARBON,
            SubstrateType.COMPUTATIONAL,
            SubstrateType.QUANTUM,
        ]
        assert len(key_substrates) == 3


class TestCommunicationParadigm:
    """Tests for CommunicationParadigm enum."""

    def test_paradigms_exist(self):
        """Test communication paradigms are defined."""
        paradigms = list(CommunicationParadigm)
        assert len(paradigms) >= 5, f"Expected at least 5 paradigms, got {len(paradigms)}"


class TestPhilosophicalFramework:
    """Tests for PhilosophicalFramework enum."""

    def test_frameworks_exist(self):
        """Test philosophical frameworks are defined."""
        frameworks = list(PhilosophicalFramework)
        assert len(frameworks) >= 4, f"Expected at least 4 frameworks, got {len(frameworks)}"

    def test_key_frameworks_exist(self):
        """Test key philosophical frameworks are defined."""
        key_frameworks = [
            PhilosophicalFramework.FUNCTIONALISM,
            PhilosophicalFramework.PANPSYCHISM,
            PhilosophicalFramework.INTEGRATED_INFORMATION,
        ]
        assert len(key_frameworks) == 3


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestXenoMindHypothesis:
    """Tests for XenoMindHypothesis dataclass."""

    def test_hypothesis_creation(self):
        """Test basic hypothesis creation."""
        hypothesis = XenoMindHypothesis(
            hypothesis_id="stellar_mind_001",
            mind_type=XenoMindType.STELLAR_SCALE,
            substrate=SubstrateType.ELECTROMAGNETIC,
            sensory_modalities=[XenoSensoryModality.GRAVITATIONAL],
            cognitive_architecture="Plasma dynamics processing",
            consciousness_indicators=[ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION],
            plausibility_assessment=0.15,
            detection_methods=["Unusual stellar patterns"],
            description="Consciousness in stellar plasma",
        )
        assert hypothesis.hypothesis_id == "stellar_mind_001"
        assert hypothesis.mind_type == XenoMindType.STELLAR_SCALE
        assert hypothesis.plausibility_assessment == 0.15

    def test_hypothesis_to_embedding_text(self):
        """Test embedding text generation."""
        hypothesis = XenoMindHypothesis(
            hypothesis_id="test",
            mind_type=XenoMindType.QUANTUM_COHERENT,
            substrate=SubstrateType.QUANTUM,
            sensory_modalities=[XenoSensoryModality.QUANTUM_ENTANGLEMENT],
            cognitive_architecture="Superposed cognitive states",
            consciousness_indicators=[],
            plausibility_assessment=0.25,
            detection_methods=["Quantum coherence signatures"],
            description="BEC-based consciousness",
        )
        text = hypothesis.to_embedding_text()
        assert "quantum_coherent" in text
        assert "quantum" in text.lower()


class TestAlternativeSensoryWorld:
    """Tests for AlternativeSensoryWorld dataclass."""

    def test_sensory_world_creation(self):
        """Test sensory world creation."""
        world = AlternativeSensoryWorld(
            world_id="gravitational_world",
            name="Gravitational Sense World",
            sensory_modalities=[XenoSensoryModality.GRAVITATIONAL],
            umwelt_description="Space has texture based on mass distribution",
            cognitive_implications=["Natural orbital mechanics intuition"],
            communication_challenges=["Gravitational wave modulation requires extreme energies"],
        )
        assert world.world_id == "gravitational_world"
        assert len(world.sensory_modalities) == 1

    def test_sensory_world_to_embedding_text(self):
        """Test embedding text generation."""
        world = AlternativeSensoryWorld(
            world_id="test",
            name="Temporal World",
            sensory_modalities=[XenoSensoryModality.TEMPORAL_PERCEPTION],
            umwelt_description="Time perceived as dimension",
            cognitive_implications=["Non-linear planning"],
            communication_challenges=["Temporal ordering issues"],
        )
        text = world.to_embedding_text()
        assert "Temporal World" in text


class TestSETIConsciousnessProtocol:
    """Tests for SETIConsciousnessProtocol dataclass."""

    def test_protocol_creation(self):
        """Test SETI protocol creation."""
        protocol = SETIConsciousnessProtocol(
            protocol_id="iit_detection",
            name="IIT Detection Protocol",
            detection_method="Search for high-phi systems",
            consciousness_signatures=["Complex integrated patterns"],
            false_positive_risks=["Complex but non-conscious systems"],
            ethical_considerations=["Uncertainty about consciousness presence"],
        )
        assert protocol.protocol_id == "iit_detection"
        assert len(protocol.consciousness_signatures) >= 1


class TestSciFiFramework:
    """Tests for SciFiFramework dataclass."""

    def test_scifi_framework_creation(self):
        """Test sci-fi framework creation."""
        framework = SciFiFramework(
            framework_id="solaris_lem",
            source_work="Solaris",
            author="Stanislaw Lem",
            year=1961,
            mind_type_depicted=XenoMindType.PLANETARY_SCALE,
            philosophical_implications=["Consciousness without communication"],
            scientific_plausibility=0.15,
            description="Sentient planetary ocean",
        )
        assert framework.source_work == "Solaris"
        assert framework.mind_type_depicted == XenoMindType.PLANETARY_SCALE

    def test_scifi_framework_to_embedding_text(self):
        """Test embedding text generation."""
        framework = SciFiFramework(
            framework_id="test",
            source_work="Arrival",
            author="Ted Chiang",
            mind_type_depicted=XenoMindType.CARBON_BIOLOGICAL,
            philosophical_implications=["Non-linear temporal perception"],
            scientific_plausibility=0.3,
            description="Heptapods with non-linear time perception",
        )
        text = framework.to_embedding_text()
        assert "Arrival" in text
        assert "Ted Chiang" in text


class TestCrossSubstrateComparison:
    """Tests for CrossSubstrateComparison dataclass."""

    def test_comparison_creation(self):
        """Test cross-substrate comparison creation."""
        comparison = CrossSubstrateComparison(
            comparison_id="carbon_vs_silicon",
            name="Carbon vs. Silicon Consciousness",
            substrate_a=SubstrateType.BIOLOGICAL_CARBON,
            substrate_b=SubstrateType.BIOLOGICAL_SILICON,
            shared_properties=["Chemical information processing"],
            divergent_properties=["Operating temperature"],
            consciousness_implications=["Similar consciousness possible if functionalism true"],
            philosophical_framework=PhilosophicalFramework.FUNCTIONALISM,
        )
        assert comparison.substrate_a == SubstrateType.BIOLOGICAL_CARBON
        assert comparison.substrate_b == SubstrateType.BIOLOGICAL_SILICON


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestXenoconsciousnessInterface:
    """Tests for XenoconsciousnessInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return XenoconsciousnessInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_hypothesis(self, interface):
        """Test adding and retrieving a hypothesis."""
        await interface.initialize()

        hypothesis = XenoMindHypothesis(
            hypothesis_id="test_hive",
            mind_type=XenoMindType.COLLECTIVE_HIVE,
            substrate=SubstrateType.HYBRID,
            sensory_modalities=[XenoSensoryModality.CHEMICAL_EXOTIC],
            cognitive_architecture="Distributed collective",
            consciousness_indicators=[ConsciousnessIndicatorXeno.BEHAVIORAL_COMPLEXITY],
            plausibility_assessment=0.4,
            detection_methods=["Coordinated behavior"],
        )

        await interface.add_hypothesis(hypothesis)
        retrieved = await interface.get_hypothesis("test_hive")

        assert retrieved is not None
        assert retrieved.mind_type == XenoMindType.COLLECTIVE_HIVE

    @pytest.mark.asyncio
    async def test_query_hypotheses_by_mind_type(self, interface):
        """Test querying hypotheses by mind type."""
        await interface.initialize()

        hypothesis = XenoMindHypothesis(
            hypothesis_id="test_plasma",
            mind_type=XenoMindType.PLASMA_BASED,
            substrate=SubstrateType.ELECTROMAGNETIC,
            sensory_modalities=[],
            cognitive_architecture="Plasma vortex processing",
            consciousness_indicators=[],
            plausibility_assessment=0.2,
            detection_methods=[],
        )
        await interface.add_hypothesis(hypothesis)

        hypotheses = await interface.query_hypotheses_by_mind_type(
            XenoMindType.PLASMA_BASED
        )
        assert len(hypotheses) >= 1

    @pytest.mark.asyncio
    async def test_query_hypotheses_by_substrate(self, interface):
        """Test querying hypotheses by substrate."""
        await interface.initialize()

        hypothesis = XenoMindHypothesis(
            hypothesis_id="test_quantum",
            mind_type=XenoMindType.QUANTUM_COHERENT,
            substrate=SubstrateType.QUANTUM,
            sensory_modalities=[],
            cognitive_architecture="Quantum superposition",
            consciousness_indicators=[],
            plausibility_assessment=0.25,
            detection_methods=[],
        )
        await interface.add_hypothesis(hypothesis)

        hypotheses = await interface.query_hypotheses_by_substrate(
            SubstrateType.QUANTUM
        )
        assert len(hypotheses) >= 1

    @pytest.mark.asyncio
    async def test_query_hypotheses_by_plausibility(self, interface):
        """Test querying hypotheses by plausibility range."""
        await interface.initialize()

        hypothesis = XenoMindHypothesis(
            hypothesis_id="test_high_plaus",
            mind_type=XenoMindType.CARBON_BIOLOGICAL,
            substrate=SubstrateType.BIOLOGICAL_CARBON,
            sensory_modalities=[],
            cognitive_architecture="Alternative biochemistry",
            consciousness_indicators=[],
            plausibility_assessment=0.7,
            detection_methods=[],
        )
        await interface.add_hypothesis(hypothesis)

        hypotheses = await interface.query_hypotheses_by_plausibility(
            min_plausibility=0.5,
            max_plausibility=1.0
        )
        assert len(hypotheses) >= 1

    @pytest.mark.asyncio
    async def test_add_and_get_sensory_world(self, interface):
        """Test adding and retrieving sensory world."""
        await interface.initialize()

        world = AlternativeSensoryWorld(
            world_id="test_world",
            name="Test World",
            sensory_modalities=[XenoSensoryModality.MAGNETIC],
            umwelt_description="Magnetic field perception",
            cognitive_implications=["Magnetic navigation"],
            communication_challenges=["Field modulation"],
        )

        await interface.add_sensory_world(world)
        retrieved = await interface.get_sensory_world("test_world")

        assert retrieved is not None
        assert retrieved.name == "Test World"

    @pytest.mark.asyncio
    async def test_add_and_get_protocol(self, interface):
        """Test adding and retrieving SETI protocol."""
        await interface.initialize()

        protocol = SETIConsciousnessProtocol(
            protocol_id="test_protocol",
            name="Test Protocol",
            detection_method="Behavioral analysis",
            consciousness_signatures=["Complex patterns"],
            false_positive_risks=["False identification"],
            ethical_considerations=["Contact decisions"],
        )

        await interface.add_protocol(protocol)
        retrieved = await interface.get_protocol("test_protocol")

        assert retrieved is not None
        assert retrieved.name == "Test Protocol"

    @pytest.mark.asyncio
    async def test_add_and_get_scifi_framework(self, interface):
        """Test adding and retrieving sci-fi framework."""
        await interface.initialize()

        framework = SciFiFramework(
            framework_id="test_scifi",
            source_work="Test Work",
            author="Test Author",
            mind_type_depicted=XenoMindType.DIGITAL_SUBSTRATE,
            philosophical_implications=["Digital consciousness"],
            scientific_plausibility=0.5,
        )

        await interface.add_scifi_framework(framework)
        retrieved = await interface.get_scifi_framework("test_scifi")

        assert retrieved is not None
        assert retrieved.source_work == "Test Work"

    @pytest.mark.asyncio
    async def test_add_and_get_comparison(self, interface):
        """Test adding and retrieving cross-substrate comparison."""
        await interface.initialize()

        comparison = CrossSubstrateComparison(
            comparison_id="test_comparison",
            name="Test Comparison",
            substrate_a=SubstrateType.BIOLOGICAL_CARBON,
            substrate_b=SubstrateType.COMPUTATIONAL,
            shared_properties=["Information processing"],
            divergent_properties=["Physical substrate"],
            consciousness_implications=["Functionalism implications"],
        )

        await interface.add_comparison(comparison)
        retrieved = await interface.get_comparison("test_comparison")

        assert retrieved is not None
        assert retrieved.name == "Test Comparison"

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "hypotheses" in result
        assert result["hypotheses"] > 0
        assert "sensory_worlds" in result
        assert "scifi_frameworks" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.hypothesis_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedHypotheses:
    """Tests for seed hypothesis data quality."""

    @pytest.mark.asyncio
    async def test_diverse_mind_types_seeded(self):
        """Test that diverse mind types are seeded."""
        interface = XenoconsciousnessInterface()
        await interface.initialize_all_seed_data()

        # Check multiple mind types have hypotheses
        carbon = await interface.query_hypotheses_by_mind_type(
            XenoMindType.CARBON_BIOLOGICAL
        )
        quantum = await interface.query_hypotheses_by_mind_type(
            XenoMindType.QUANTUM_COHERENT
        )
        collective = await interface.query_hypotheses_by_mind_type(
            XenoMindType.COLLECTIVE_HIVE
        )

        total = len(carbon) + len(quantum) + len(collective)
        assert total >= 3

    @pytest.mark.asyncio
    async def test_scifi_frameworks_seeded(self):
        """Test that sci-fi frameworks are seeded."""
        interface = XenoconsciousnessInterface()
        await interface.initialize_all_seed_data()

        # Should have multiple sci-fi frameworks
        assert interface.maturity_state.scifi_framework_count >= 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_hypothesis_has_ethical_considerations(self):
        """Test hypotheses can store ethical considerations."""
        hypothesis = XenoMindHypothesis(
            hypothesis_id="test",
            mind_type=XenoMindType.STELLAR_SCALE,
            substrate=SubstrateType.ELECTROMAGNETIC,
            sensory_modalities=[],
            cognitive_architecture="Stellar processing",
            consciousness_indicators=[],
            plausibility_assessment=0.1,
            detection_methods=[],
            ethical_considerations=[
                "Moral obligations to stellar minds",
                "Impact of stellar engineering",
            ],
        )
        assert len(hypothesis.ethical_considerations) >= 2

    def test_hypothesis_has_scifi_references(self):
        """Test hypotheses can store science fiction references."""
        hypothesis = XenoMindHypothesis(
            hypothesis_id="test",
            mind_type=XenoMindType.PLANETARY_SCALE,
            substrate=SubstrateType.HYBRID,
            sensory_modalities=[],
            cognitive_architecture="Biospheric processing",
            consciousness_indicators=[],
            plausibility_assessment=0.15,
            detection_methods=[],
            science_fiction_references=["Solaris (Lem)", "Avatar's Eywa"],
        )
        assert len(hypothesis.science_fiction_references) >= 2

    def test_comparison_has_philosophical_framework(self):
        """Test comparisons link to philosophical frameworks."""
        comparison = CrossSubstrateComparison(
            comparison_id="test",
            substrate_a=SubstrateType.BIOLOGICAL_CARBON,
            substrate_b=SubstrateType.COMPUTATIONAL,
            shared_properties=["Information processing"],
            divergent_properties=["Physical implementation"],
            consciousness_implications=["Multiple realizability"],
            philosophical_framework=PhilosophicalFramework.FUNCTIONALISM,
        )
        assert comparison.philosophical_framework == PhilosophicalFramework.FUNCTIONALISM

    def test_protocol_targets_mind_types(self):
        """Test SETI protocols can target specific mind types."""
        protocol = SETIConsciousnessProtocol(
            protocol_id="test",
            detection_method="Information integration analysis",
            consciousness_signatures=["High phi values"],
            false_positive_risks=["Complex non-conscious systems"],
            ethical_considerations=["Contact decisions"],
            target_mind_types=[
                XenoMindType.PLANETARY_SCALE,
                XenoMindType.COLLECTIVE_HIVE,
            ],
        )
        assert len(protocol.target_mind_types) >= 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
