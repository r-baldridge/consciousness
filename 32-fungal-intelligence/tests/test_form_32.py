#!/usr/bin/env python3
"""
Test suite for Form 32: Fungal Networks & Mycorrhizal Intelligence

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
    "fungal_intelligence_interface",
    INTERFACE_PATH / "fungal_intelligence_interface.py"
)

# Import Form 32 components from loaded module
FungalIntelligenceDomain = interface_module.FungalIntelligenceDomain
FungalType = interface_module.FungalType
NetworkBehavior = interface_module.NetworkBehavior
ResearchParadigm = interface_module.ResearchParadigm
IndigenousFungalTradition = interface_module.IndigenousFungalTradition
MycorrhizalNetworkRole = interface_module.MycorrhizalNetworkRole
CommunicationSignalType = interface_module.CommunicationSignalType
MaturityLevel = interface_module.MaturityLevel
FungalNetworkProfile = interface_module.FungalNetworkProfile
SlimeMoldExperiment = interface_module.SlimeMoldExperiment
MycelialCommunication = interface_module.MycelialCommunication
IndigenousFungalWisdom = interface_module.IndigenousFungalWisdom
FungalIntelligenceMaturityState = interface_module.FungalIntelligenceMaturityState
FungalIntelligenceInterface = interface_module.FungalIntelligenceInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestFungalIntelligenceDomain:
    """Tests for FungalIntelligenceDomain enum."""

    def test_domain_count(self):
        """Test that we have the expected number of intelligence domains."""
        domains = list(FungalIntelligenceDomain)
        assert len(domains) >= 18, f"Expected at least 18 domains, got {len(domains)}"

    def test_network_domains_exist(self):
        """Test network capability domains are defined."""
        network = [
            FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
            FungalIntelligenceDomain.RESOURCE_DISTRIBUTION,
            FungalIntelligenceDomain.NUTRIENT_TRANSFER,
        ]
        assert len(network) == 3

    def test_communication_domains_exist(self):
        """Test communication domains are defined."""
        communication = [
            FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
            FungalIntelligenceDomain.ELECTRICAL_SIGNALING,
            FungalIntelligenceDomain.INTER_KINGDOM_SIGNALING,
        ]
        assert len(communication) == 3

    def test_cognitive_domains_exist(self):
        """Test cognitive function domains are defined."""
        cognitive = [
            FungalIntelligenceDomain.MEMORY_TRACES,
            FungalIntelligenceDomain.SPATIAL_MAPPING,
            FungalIntelligenceDomain.PROBLEM_SOLVING,
        ]
        assert len(cognitive) == 3

    def test_domain_values_are_strings(self):
        """Test that all domain values are strings."""
        for domain in FungalIntelligenceDomain:
            assert isinstance(domain.value, str)


class TestFungalType:
    """Tests for FungalType enum."""

    def test_type_count(self):
        """Test that we have expected fungal types."""
        types = list(FungalType)
        assert len(types) >= 12, f"Expected at least 12 types, got {len(types)}"

    def test_mycorrhizal_types_exist(self):
        """Test mycorrhizal types are defined."""
        mycorrhizal = [
            FungalType.MYCORRHIZAL_ECTO,
            FungalType.MYCORRHIZAL_ARBUSCULAR,
            FungalType.MYCORRHIZAL_ERICOID,
            FungalType.MYCORRHIZAL_ORCHID,
        ]
        assert len(mycorrhizal) == 4

    def test_special_types_exist(self):
        """Test special types are defined."""
        special = [
            FungalType.ENTHEOGENIC,
            FungalType.MEDICINAL,
            FungalType.BIOLUMINESCENT,
            FungalType.SLIME_MOLDS,
        ]
        assert len(special) == 4


class TestIndigenousFungalTradition:
    """Tests for IndigenousFungalTradition enum."""

    def test_tradition_count(self):
        """Test that we have diverse traditions."""
        traditions = list(IndigenousFungalTradition)
        assert len(traditions) >= 10, f"Expected at least 10 traditions, got {len(traditions)}"

    def test_mesoamerican_traditions_exist(self):
        """Test Mesoamerican traditions are defined."""
        mesoamerican = [
            IndigenousFungalTradition.MAZATEC_MUSHROOM,
            IndigenousFungalTradition.NAHUA_TEONANACATL,
        ]
        assert len(mesoamerican) == 2

    def test_asian_traditions_exist(self):
        """Test Asian traditions are defined."""
        asian = [
            IndigenousFungalTradition.CHINESE_LINGZHI,
            IndigenousFungalTradition.JAPANESE_SHIITAKE,
            IndigenousFungalTradition.TIBETAN_CORDYCEPS,
        ]
        assert len(asian) == 3


class TestNetworkBehavior:
    """Tests for NetworkBehavior enum."""

    def test_behavior_count(self):
        """Test that we have expected behaviors."""
        behaviors = list(NetworkBehavior)
        assert len(behaviors) >= 8, f"Expected at least 8 behaviors, got {len(behaviors)}"

    def test_key_behaviors_exist(self):
        """Test key behaviors are defined."""
        behaviors = [
            NetworkBehavior.NUTRIENT_TRANSFER,
            NetworkBehavior.SIGNAL_PROPAGATION,
            NetworkBehavior.DEFENSE_RESPONSE,
            NetworkBehavior.GROWTH_OPTIMIZATION,
        ]
        assert len(behaviors) == 4


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestFungalNetworkProfile:
    """Tests for FungalNetworkProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = FungalNetworkProfile(
            network_id="physarum_polycephalum",
            name="Physarum polycephalum",
            primary_species="Physarum polycephalum",
        )
        assert profile.network_id == "physarum_polycephalum"
        assert profile.name == "Physarum polycephalum"

    def test_profile_with_intelligence_domains(self):
        """Test profile with intelligence domains."""
        profile = FungalNetworkProfile(
            network_id="test_network",
            name="Test Network",
            primary_species="Test species",
            intelligence_domains=[
                FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
                FungalIntelligenceDomain.PROBLEM_SOLVING,
            ],
            fungal_type=FungalType.SLIME_MOLDS,
        )
        assert len(profile.intelligence_domains) == 2
        assert profile.fungal_type == FungalType.SLIME_MOLDS

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = FungalNetworkProfile(
            network_id="test",
            name="Test Fungus",
            primary_species="Testus fungus",
            fungal_type=FungalType.MEDICINAL,
            description="A test fungal network",
        )
        text = profile.to_embedding_text()
        assert "Test Fungus" in text
        assert "medicinal" in text


class TestSlimeMoldExperiment:
    """Tests for SlimeMoldExperiment dataclass."""

    def test_experiment_creation(self):
        """Test experiment creation."""
        experiment = SlimeMoldExperiment(
            experiment_id="maze_solving_001",
            name="Maze Solving Experiment",
            species="Physarum polycephalum",
            paradigm=ResearchParadigm.UNCONVENTIONAL_COMPUTING,
            year=2000,
            researchers=["Toshiyuki Nakagaki"],
            findings="Slime mold found shortest path through maze",
        )
        assert experiment.paradigm == ResearchParadigm.UNCONVENTIONAL_COMPUTING
        assert experiment.year == 2000

    def test_experiment_to_embedding_text(self):
        """Test experiment embedding text generation."""
        experiment = SlimeMoldExperiment(
            experiment_id="test",
            name="Test Experiment",
            species="Physarum polycephalum",
            paradigm=ResearchParadigm.NETWORK_SCIENCE,
            findings="Network optimization demonstrated",
            computational_analog="Shortest path algorithm",
        )
        text = experiment.to_embedding_text()
        assert "Test Experiment" in text
        assert "Physarum" in text


class TestIndigenousFungalWisdom:
    """Tests for IndigenousFungalWisdom dataclass."""

    def test_wisdom_creation(self):
        """Test indigenous wisdom creation with Form 29 link."""
        wisdom = IndigenousFungalWisdom(
            wisdom_id="mazatec_velada",
            name="Mazatec Velada Ceremony",
            fungal_species="Psilocybe mexicana",
            tradition=IndigenousFungalTradition.MAZATEC_MUSHROOM,
            ceremonial_use="Nocturnal healing ceremony",
            form_29_link="mesoamerican_folk",
        )
        assert wisdom.form_29_link == "mesoamerican_folk"
        assert wisdom.tradition == IndigenousFungalTradition.MAZATEC_MUSHROOM


class TestMycelialCommunication:
    """Tests for MycelialCommunication dataclass."""

    def test_communication_creation(self):
        """Test communication record creation."""
        comm = MycelialCommunication(
            communication_id="carbon_transfer_001",
            name="Carbon Transfer Event",
            sender="Douglas fir",
            receiver="Seedling",
            signal_type=CommunicationSignalType.DIFFUSIBLE_CHEMICAL,
            ecological_context="Forest understory",
            purpose="Resource provisioning",
        )
        assert comm.signal_type == CommunicationSignalType.DIFFUSIBLE_CHEMICAL
        assert comm.purpose == "Resource provisioning"


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestFungalIntelligenceInterface:
    """Tests for FungalIntelligenceInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return FungalIntelligenceInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_network_profile(self, interface):
        """Test adding and retrieving a network profile."""
        await interface.initialize()

        profile = FungalNetworkProfile(
            network_id="test_network",
            name="Test Network",
            primary_species="Test species",
            fungal_type=FungalType.SLIME_MOLDS,
        )

        await interface.add_network_profile(profile)
        retrieved = await interface.get_network_profile("test_network")

        assert retrieved is not None
        assert retrieved.name == "Test Network"

    @pytest.mark.asyncio
    async def test_query_profiles_by_type(self, interface):
        """Test querying profiles by fungal type."""
        await interface.initialize()

        profile1 = FungalNetworkProfile(
            network_id="slime_1",
            name="Slime 1",
            primary_species="Slime sp.",
            fungal_type=FungalType.SLIME_MOLDS,
        )
        profile2 = FungalNetworkProfile(
            network_id="medicinal_1",
            name="Medicinal 1",
            primary_species="Medicinal sp.",
            fungal_type=FungalType.MEDICINAL,
        )

        await interface.add_network_profile(profile1)
        await interface.add_network_profile(profile2)

        slimes = await interface.query_profiles_by_type(FungalType.SLIME_MOLDS)
        assert len(slimes) == 1
        assert slimes[0].network_id == "slime_1"

    @pytest.mark.asyncio
    async def test_query_profiles_by_domain(self, interface):
        """Test querying profiles by intelligence domain."""
        await interface.initialize()

        profile = FungalNetworkProfile(
            network_id="problem_solver",
            name="Problem Solver",
            primary_species="Solver sp.",
            fungal_type=FungalType.SLIME_MOLDS,
            intelligence_domains=[FungalIntelligenceDomain.PROBLEM_SOLVING],
        )
        await interface.add_network_profile(profile)

        solvers = await interface.query_profiles_by_domain(
            FungalIntelligenceDomain.PROBLEM_SOLVING
        )
        assert len(solvers) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_experiment(self, interface):
        """Test adding and retrieving an experiment."""
        await interface.initialize()

        experiment = SlimeMoldExperiment(
            experiment_id="test_exp",
            name="Test Experiment",
            species="Physarum polycephalum",
            paradigm=ResearchParadigm.UNCONVENTIONAL_COMPUTING,
        )

        await interface.add_experiment(experiment)
        retrieved = await interface.get_experiment("test_exp")

        assert retrieved is not None
        assert retrieved.name == "Test Experiment"

    @pytest.mark.asyncio
    async def test_add_and_get_wisdom(self, interface):
        """Test adding and retrieving wisdom."""
        await interface.initialize()

        wisdom = IndigenousFungalWisdom(
            wisdom_id="test_wisdom",
            name="Test Wisdom",
            fungal_species="Test species",
            tradition=IndigenousFungalTradition.CHINESE_LINGZHI,
        )

        await interface.add_wisdom(wisdom)
        retrieved = await interface.get_wisdom("test_wisdom")

        assert retrieved is not None
        assert retrieved.name == "Test Wisdom"

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "network_profiles" in result
        assert result["network_profiles"] >= 5  # Should have at least 5 seed profiles
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.network_profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedProfiles:
    """Tests for seed profile data quality."""

    @pytest.mark.asyncio
    async def test_physarum_profile_exists(self):
        """Test Physarum polycephalum seed profile exists."""
        interface = FungalIntelligenceInterface()
        await interface.initialize_all_seed_data()

        physarum = await interface.get_network_profile("physarum_polycephalum")
        assert physarum is not None
        assert "Physarum" in physarum.name
        assert FungalIntelligenceDomain.PROBLEM_SOLVING in physarum.intelligence_domains

    @pytest.mark.asyncio
    async def test_humongous_fungus_exists(self):
        """Test Humongous Fungus seed profile exists."""
        interface = FungalIntelligenceInterface()
        await interface.initialize_all_seed_data()

        humongous = await interface.get_network_profile("armillaria_ostoyae_humongous")
        assert humongous is not None
        assert "Humongous" in humongous.name

    @pytest.mark.asyncio
    async def test_maze_experiment_exists(self):
        """Test maze solving experiment seed data exists."""
        interface = FungalIntelligenceInterface()
        await interface.initialize_all_seed_data()

        maze_exp = await interface.get_experiment("nakagaki_maze_2000")
        assert maze_exp is not None
        assert maze_exp.year == 2000


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_form_32_link_field_exists(self):
        """Test Form 32 wisdom has link field for Form 29."""
        wisdom = IndigenousFungalWisdom(
            wisdom_id="test",
            name="Test Wisdom",
            fungal_species="Test species",
            tradition=IndigenousFungalTradition.MAZATEC_MUSHROOM,
            form_29_link="mesoamerican_folk",
        )
        assert wisdom.form_29_link is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
