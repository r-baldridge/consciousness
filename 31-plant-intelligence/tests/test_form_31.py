#!/usr/bin/env python3
"""
Test suite for Form 31: Plant Intelligence & Vegetal Consciousness

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
    "plant_intelligence_interface",
    INTERFACE_PATH / "plant_intelligence_interface.py"
)

# Import Form 31 components from loaded module
PlantCognitionDomain = interface_module.PlantCognitionDomain
PlantTaxonomicGroup = interface_module.PlantTaxonomicGroup
PlantSensoryModality = interface_module.PlantSensoryModality
ResearchParadigm = interface_module.ResearchParadigm
IndigenousTraditionType = interface_module.IndigenousTraditionType
PlantSignalingType = interface_module.PlantSignalingType
MaturityLevel = interface_module.MaturityLevel
PlantSpeciesProfile = interface_module.PlantSpeciesProfile
PlantBehaviorInsight = interface_module.PlantBehaviorInsight
PlantCommunicationEvent = interface_module.PlantCommunicationEvent
IndigenousPlantWisdom = interface_module.IndigenousPlantWisdom
PlantLearningExperiment = interface_module.PlantLearningExperiment
PlantIntelligenceMaturityState = interface_module.PlantIntelligenceMaturityState
PlantIntelligenceInterface = interface_module.PlantIntelligenceInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestPlantCognitionDomain:
    """Tests for PlantCognitionDomain enum."""

    def test_domain_count(self):
        """Test that we have the expected number of cognition domains."""
        domains = list(PlantCognitionDomain)
        # Should have a comprehensive set of domains
        assert len(domains) >= 30, f"Expected at least 30 domains, got {len(domains)}"

    def test_communication_domains_exist(self):
        """Test communication-related domains are defined."""
        communication = [
            PlantCognitionDomain.CHEMICAL_SIGNALING,
            PlantCognitionDomain.ROOT_COMMUNICATION,
            PlantCognitionDomain.VOLATILE_COMMUNICATION,
            PlantCognitionDomain.MYCORRHIZAL_NETWORKING,
        ]
        assert len(communication) == 4

    def test_memory_domains_exist(self):
        """Test memory-related domains are defined."""
        memory = [
            PlantCognitionDomain.MEMORY_HABITUATION,
            PlantCognitionDomain.ASSOCIATIVE_LEARNING,
            PlantCognitionDomain.EPIGENETIC_MEMORY,
            PlantCognitionDomain.STRESS_MEMORY,
        ]
        assert len(memory) == 4

    def test_sensing_domains_exist(self):
        """Test sensing-related domains are defined."""
        sensing = [
            PlantCognitionDomain.CIRCADIAN_AWARENESS,
            PlantCognitionDomain.GRAVITROPISM,
            PlantCognitionDomain.PHOTOTROPISM,
            PlantCognitionDomain.PROPRIOCEPTION,
        ]
        assert len(sensing) == 4

    def test_domain_values_are_strings(self):
        """Test that all domain values are strings."""
        for domain in PlantCognitionDomain:
            assert isinstance(domain.value, str)


class TestPlantTaxonomicGroup:
    """Tests for PlantTaxonomicGroup enum."""

    def test_group_count(self):
        """Test that we have expected taxonomic groups."""
        groups = list(PlantTaxonomicGroup)
        assert len(groups) >= 15, f"Expected at least 15 groups, got {len(groups)}"

    def test_tree_groups_exist(self):
        """Test tree groups are defined."""
        trees = [
            PlantTaxonomicGroup.TREES_DECIDUOUS,
            PlantTaxonomicGroup.TREES_CONIFEROUS,
            PlantTaxonomicGroup.TREES_TROPICAL,
        ]
        assert len(trees) == 3

    def test_special_groups_exist(self):
        """Test special groups are defined."""
        special = [
            PlantTaxonomicGroup.CARNIVOROUS_PLANTS,
            PlantTaxonomicGroup.SENSITIVE_PLANTS,
            PlantTaxonomicGroup.SACRED_PLANTS,
            PlantTaxonomicGroup.ENTHEOGENIC_PLANTS,
        ]
        assert len(special) == 4


class TestPlantSensoryModality:
    """Tests for PlantSensoryModality enum."""

    def test_modality_count(self):
        """Test that we have comprehensive sensory modalities."""
        modalities = list(PlantSensoryModality)
        assert len(modalities) >= 10, f"Expected at least 10 modalities, got {len(modalities)}"

    def test_key_modalities_exist(self):
        """Test key modalities are defined."""
        modalities = [
            PlantSensoryModality.TOUCH_MECHANOSENSING,
            PlantSensoryModality.LIGHT_PHOTORECEPTION,
            PlantSensoryModality.GRAVITY_GRAVISENSING,
            PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
        ]
        assert len(modalities) == 4


class TestIndigenousTraditionType:
    """Tests for IndigenousTraditionType enum."""

    def test_tradition_count(self):
        """Test that we have diverse traditions."""
        traditions = list(IndigenousTraditionType)
        assert len(traditions) >= 15, f"Expected at least 15 traditions, got {len(traditions)}"

    def test_amazonian_traditions_exist(self):
        """Test Amazonian traditions are defined."""
        amazonian = [
            IndigenousTraditionType.AMAZONIAN_PLANT_TEACHER,
            IndigenousTraditionType.DIETA_TRADITION,
        ]
        assert len(amazonian) == 2

    def test_asian_traditions_exist(self):
        """Test Asian traditions are defined."""
        asian = [
            IndigenousTraditionType.VEDIC_PLANT_CONSCIOUSNESS,
            IndigenousTraditionType.AYURVEDIC,
            IndigenousTraditionType.CHINESE_MEDICINE,
        ]
        assert len(asian) == 3


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestPlantSpeciesProfile:
    """Tests for PlantSpeciesProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = PlantSpeciesProfile(
            species_id="mimosa_pudica",
            common_name="Sensitive Plant",
            scientific_name="Mimosa pudica",
            taxonomic_group=PlantTaxonomicGroup.SENSITIVE_PLANTS,
        )
        assert profile.species_id == "mimosa_pudica"
        assert profile.common_name == "Sensitive Plant"
        assert profile.taxonomic_group == PlantTaxonomicGroup.SENSITIVE_PLANTS

    def test_profile_with_cognition_domains(self):
        """Test profile with cognition domain scores."""
        profile = PlantSpeciesProfile(
            species_id="dionaea_muscipula",
            common_name="Venus Flytrap",
            scientific_name="Dionaea muscipula",
            taxonomic_group=PlantTaxonomicGroup.CARNIVOROUS_PLANTS,
            cognition_domains={
                PlantCognitionDomain.DECISION_MAKING: 0.90,
                PlantCognitionDomain.MEMORY_HABITUATION: 0.85,
            },
        )
        assert profile.cognition_domains[PlantCognitionDomain.DECISION_MAKING] == 0.90
        assert len(profile.cognition_domains) == 2

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = PlantSpeciesProfile(
            species_id="test",
            common_name="Test Plant",
            scientific_name="Testus plantus",
            taxonomic_group=PlantTaxonomicGroup.FLOWERING_PLANTS,
            cognition_domains={PlantCognitionDomain.CHEMICAL_SIGNALING: 0.8},
            notable_behaviors=["Volatile signaling", "Root communication"],
        )
        text = profile.to_embedding_text()
        assert "Test Plant" in text
        assert "flowering_plants" in text


class TestPlantBehaviorInsight:
    """Tests for PlantBehaviorInsight dataclass."""

    def test_insight_creation(self):
        """Test insight creation."""
        insight = PlantBehaviorInsight(
            insight_id="mimosa_habituation_001",
            species_id="mimosa_pudica",
            domain=PlantCognitionDomain.MEMORY_HABITUATION,
            description="Habituation to repeated drop stimuli",
            evidence_type=ResearchParadigm.BEHAVIORAL_EXPERIMENTAL,
            methodology="Repeated dropping experiments",
        )
        assert insight.domain == PlantCognitionDomain.MEMORY_HABITUATION
        assert insight.species_id == "mimosa_pudica"


class TestIndigenousPlantWisdom:
    """Tests for IndigenousPlantWisdom dataclass."""

    def test_wisdom_creation(self):
        """Test indigenous wisdom creation with Form 29 link."""
        wisdom = IndigenousPlantWisdom(
            wisdom_id="ayahuasca_wisdom",
            plant_name="Ayahuasca",
            scientific_name="Banisteriopsis caapi",
            tradition=IndigenousTraditionType.AMAZONIAN_PLANT_TEACHER,
            spiritual_significance="Master plant teacher",
            form_29_link="amazonian_folk_wisdom",
        )
        assert wisdom.form_29_link == "amazonian_folk_wisdom"
        assert wisdom.tradition == IndigenousTraditionType.AMAZONIAN_PLANT_TEACHER


class TestPlantCommunicationEvent:
    """Tests for PlantCommunicationEvent dataclass."""

    def test_event_creation(self):
        """Test communication event creation."""
        event = PlantCommunicationEvent(
            event_id="voc_signal_001",
            sender_species="Solanum lycopersicum",
            receiver_species="Solanum lycopersicum",
            signal_type=PlantSignalingType.VOLATILE_ORGANIC_COMPOUNDS,
            medium="air",
            ecological_context="Herbivore attack",
            trigger="Leaf damage",
            signal_content="Methyl jasmonate",
            response_observed="Defense gene upregulation",
        )
        assert event.signal_type == PlantSignalingType.VOLATILE_ORGANIC_COMPOUNDS
        assert event.medium == "air"


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestPlantIntelligenceInterface:
    """Tests for PlantIntelligenceInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return PlantIntelligenceInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_species_profile(self, interface):
        """Test adding and retrieving a species profile."""
        await interface.initialize()

        profile = PlantSpeciesProfile(
            species_id="test_species",
            common_name="Test Plant",
            scientific_name="Testus plantus",
            taxonomic_group=PlantTaxonomicGroup.FLOWERING_PLANTS,
            cognition_domains={PlantCognitionDomain.CHEMICAL_SIGNALING: 0.8},
        )

        await interface.add_species_profile(profile)
        retrieved = await interface.get_species_profile("test_species")

        assert retrieved is not None
        assert retrieved.common_name == "Test Plant"

    @pytest.mark.asyncio
    async def test_query_by_taxonomic_group(self, interface):
        """Test querying species by taxonomic group."""
        await interface.initialize()

        profile1 = PlantSpeciesProfile(
            species_id="carnivore_1",
            common_name="Carnivore 1",
            scientific_name="Carnivorus sp.",
            taxonomic_group=PlantTaxonomicGroup.CARNIVOROUS_PLANTS,
        )
        profile2 = PlantSpeciesProfile(
            species_id="tree_1",
            common_name="Tree 1",
            scientific_name="Treeus sp.",
            taxonomic_group=PlantTaxonomicGroup.TREES_DECIDUOUS,
        )

        await interface.add_species_profile(profile1)
        await interface.add_species_profile(profile2)

        carnivores = await interface.query_by_taxonomic_group(
            PlantTaxonomicGroup.CARNIVOROUS_PLANTS
        )
        assert len(carnivores) == 1
        assert carnivores[0].species_id == "carnivore_1"

    @pytest.mark.asyncio
    async def test_query_by_cognition_domain(self, interface):
        """Test querying species by cognition domain."""
        await interface.initialize()

        profile = PlantSpeciesProfile(
            species_id="memory_plant",
            common_name="Memory Plant",
            scientific_name="Memorius sp.",
            taxonomic_group=PlantTaxonomicGroup.SENSITIVE_PLANTS,
            cognition_domains={PlantCognitionDomain.MEMORY_HABITUATION: 0.9},
        )
        await interface.add_species_profile(profile)

        memory_plants = await interface.query_by_cognition_domain(
            PlantCognitionDomain.MEMORY_HABITUATION
        )
        assert len(memory_plants) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_indigenous_wisdom(self, interface):
        """Test adding and retrieving indigenous wisdom."""
        await interface.initialize()

        wisdom = IndigenousPlantWisdom(
            wisdom_id="test_wisdom",
            plant_name="Test Plant",
            tradition=IndigenousTraditionType.CELTIC_OGHAM,
            spiritual_significance="Sacred tree",
        )

        await interface.add_indigenous_wisdom(wisdom)
        results = await interface.get_indigenous_wisdom(wisdom_id="test_wisdom")

        assert len(results) == 1
        assert results[0].plant_name == "Test Plant"

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "species_profiles" in result
        assert result["species_profiles"] > 0
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.species_profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedSpecies:
    """Tests for seed species data quality."""

    @pytest.mark.asyncio
    async def test_mimosa_profile_exists(self):
        """Test Mimosa pudica seed profile exists."""
        interface = PlantIntelligenceInterface()
        await interface.initialize_all_seed_data()

        mimosa = await interface.get_species_profile("mimosa_pudica")
        assert mimosa is not None
        assert mimosa.common_name == "Sensitive Plant"
        assert PlantCognitionDomain.MEMORY_HABITUATION in mimosa.cognition_domains

    @pytest.mark.asyncio
    async def test_venus_flytrap_profile_exists(self):
        """Test Venus Flytrap seed profile exists."""
        interface = PlantIntelligenceInterface()
        await interface.initialize_all_seed_data()

        flytrap = await interface.get_species_profile("dionaea_muscipula")
        assert flytrap is not None
        assert flytrap.common_name == "Venus Flytrap"

    @pytest.mark.asyncio
    async def test_mother_tree_profile_exists(self):
        """Test Douglas Fir/Mother Tree seed profile exists."""
        interface = PlantIntelligenceInterface()
        await interface.initialize_all_seed_data()

        mother_tree = await interface.get_species_profile("pseudotsuga_menziesii")
        assert mother_tree is not None
        assert PlantCognitionDomain.MYCORRHIZAL_NETWORKING in mother_tree.cognition_domains


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_form_31_link_field_exists(self):
        """Test Form 31 wisdom has link field for Form 29."""
        wisdom = IndigenousPlantWisdom(
            wisdom_id="test",
            plant_name="Test Plant",
            tradition=IndigenousTraditionType.AMAZONIAN_PLANT_TEACHER,
            form_29_link="amazonian_folk_wisdom",
        )
        assert wisdom.form_29_link is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
