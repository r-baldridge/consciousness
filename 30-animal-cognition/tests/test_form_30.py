#!/usr/bin/env python3
"""
Test suite for Form 30: Animal Cognition & Ethology

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
    "animal_cognition_interface",
    INTERFACE_PATH / "animal_cognition_interface.py"
)

# Import Form 30 components from loaded module
TaxonomicGroup = interface_module.TaxonomicGroup
CognitionDomain = interface_module.CognitionDomain
ConsciousnessIndicator = interface_module.ConsciousnessIndicator
ResearchParadigm = interface_module.ResearchParadigm
EvidenceStrength = interface_module.EvidenceStrength
MaturityLevel = interface_module.MaturityLevel
SpeciesCognitionProfile = interface_module.SpeciesCognitionProfile
AnimalBehaviorInsight = interface_module.AnimalBehaviorInsight
CrossSpeciesSynthesis = interface_module.CrossSpeciesSynthesis
IndigenousAnimalKnowledge = interface_module.IndigenousAnimalKnowledge
ConsciousnessTheoryApplication = interface_module.ConsciousnessTheoryApplication
AnimalCognitionMaturityState = interface_module.AnimalCognitionMaturityState
AnimalCognitionInterface = interface_module.AnimalCognitionInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestTaxonomicGroup:
    """Tests for TaxonomicGroup enum."""

    def test_group_count(self):
        """Test that we have expected taxonomic groups."""
        groups = list(TaxonomicGroup)
        assert len(groups) >= 20, f"Expected at least 20 groups, got {len(groups)}"

    def test_mammal_groups_exist(self):
        """Test mammal groups are defined."""
        mammals = [
            TaxonomicGroup.GREAT_APES,
            TaxonomicGroup.CETACEANS,
            TaxonomicGroup.ELEPHANTS,
            TaxonomicGroup.CANIDS,
            TaxonomicGroup.FELIDS,
        ]
        assert len(mammals) == 5

    def test_bird_groups_exist(self):
        """Test bird groups are defined."""
        birds = [
            TaxonomicGroup.CORVIDS,
            TaxonomicGroup.PARROTS,
            TaxonomicGroup.RAPTORS,
        ]
        assert len(birds) == 3

    def test_invertebrate_groups_exist(self):
        """Test invertebrate groups are defined."""
        invertebrates = [
            TaxonomicGroup.CEPHALOPODS,
            TaxonomicGroup.SOCIAL_INSECTS,
            TaxonomicGroup.ARACHNIDS,
        ]
        assert len(invertebrates) == 3


class TestCognitionDomain:
    """Tests for CognitionDomain enum."""

    def test_domain_count(self):
        """Test we have comprehensive cognition domains."""
        domains = list(CognitionDomain)
        assert len(domains) >= 25, f"Expected at least 25 domains, got {len(domains)}"

    def test_memory_domains_exist(self):
        """Test memory-related domains exist."""
        memory = [
            CognitionDomain.EPISODIC_MEMORY,
            CognitionDomain.WORKING_MEMORY,
            CognitionDomain.SPATIAL_COGNITION,
        ]
        assert len(memory) == 3

    def test_social_domains_exist(self):
        """Test social cognition domains exist."""
        social = [
            CognitionDomain.THEORY_OF_MIND,
            CognitionDomain.COOPERATION,
            CognitionDomain.EMPATHY,
            CognitionDomain.DECEPTION,
        ]
        assert len(social) == 4

    def test_self_awareness_domains_exist(self):
        """Test self-awareness domains exist."""
        awareness = [
            CognitionDomain.SELF_RECOGNITION,
            CognitionDomain.MIRROR_TEST,
            CognitionDomain.METACOGNITION,
        ]
        assert len(awareness) == 3


class TestConsciousnessIndicator:
    """Tests for ConsciousnessIndicator enum."""

    def test_indicator_types(self):
        """Test all indicator types exist."""
        indicators = [
            ConsciousnessIndicator.BEHAVIORAL,
            ConsciousnessIndicator.NEUROANATOMICAL,
            ConsciousnessIndicator.NEUROPHYSIOLOGICAL,
            ConsciousnessIndicator.PHARMACOLOGICAL,
            ConsciousnessIndicator.SELF_REPORT_PROXY,
            ConsciousnessIndicator.INDIGENOUS_OBSERVATION,
        ]
        assert len(indicators) == 6

    def test_indigenous_indicator_included(self):
        """Test indigenous observation is included as evidence type."""
        assert ConsciousnessIndicator.INDIGENOUS_OBSERVATION is not None


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSpeciesCognitionProfile:
    """Tests for SpeciesCognitionProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = SpeciesCognitionProfile(
            species_id="pan_troglodytes",
            common_name="Chimpanzee",
            scientific_name="Pan troglodytes",
            taxonomic_group=TaxonomicGroup.GREAT_APES,
        )
        assert profile.species_id == "pan_troglodytes"
        assert profile.common_name == "Chimpanzee"
        assert profile.taxonomic_group == TaxonomicGroup.GREAT_APES

    def test_profile_with_cognition_domains(self):
        """Test profile with cognition domain scores."""
        profile = SpeciesCognitionProfile(
            species_id="corvus_moneduloides",
            common_name="New Caledonian Crow",
            scientific_name="Corvus moneduloides",
            taxonomic_group=TaxonomicGroup.CORVIDS,
            cognition_domains={
                CognitionDomain.TOOL_USE: 0.95,
                CognitionDomain.TOOL_MANUFACTURE: 0.95,
                CognitionDomain.CAUSAL_REASONING: 0.85,
            },
        )
        assert profile.cognition_domains[CognitionDomain.TOOL_USE] == 0.95
        assert len(profile.cognition_domains) == 3

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = SpeciesCognitionProfile(
            species_id="tursiops_truncatus",
            common_name="Bottlenose Dolphin",
            scientific_name="Tursiops truncatus",
            taxonomic_group=TaxonomicGroup.CETACEANS,
            cognition_domains={CognitionDomain.SELF_RECOGNITION: 0.9},
            notable_individuals=["Kelly"],
        )
        text = profile.to_embedding_text()
        assert "Bottlenose Dolphin" in text
        assert "cetaceans" in text
        assert "Kelly" in text


class TestAnimalBehaviorInsight:
    """Tests for AnimalBehaviorInsight dataclass."""

    def test_insight_creation(self):
        """Test insight creation."""
        insight = AnimalBehaviorInsight(
            insight_id="crow_tool_001",
            species_id="corvus_moneduloides",
            domain=CognitionDomain.TOOL_USE,
            description="Uses hooked stick tools to extract larvae",
            evidence_type=ConsciousnessIndicator.BEHAVIORAL,
            research_paradigm=ResearchParadigm.FIELD_OBSERVATION,
            evidence_strength=EvidenceStrength.STRONG,
        )
        assert insight.domain == CognitionDomain.TOOL_USE
        assert insight.evidence_strength == EvidenceStrength.STRONG


class TestIndigenousAnimalKnowledge:
    """Tests for IndigenousAnimalKnowledge dataclass."""

    def test_knowledge_creation(self):
        """Test indigenous knowledge creation with Form 29 link."""
        knowledge = IndigenousAnimalKnowledge(
            knowledge_id="inuit_whale_001",
            species_id="orcinus_orca",
            folk_wisdom_id="inuit_orca_wisdom",  # Link to Form 29
            behavioral_claim="Orcas communicate across generations",
            scientific_corroboration="Cultural transmission confirmed by studies",
            unique_indigenous_insight="Hunters observed pod-specific traditions centuries ago",
        )
        assert knowledge.folk_wisdom_id == "inuit_orca_wisdom"
        assert "tradition" in knowledge.unique_indigenous_insight.lower()


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestAnimalCognitionInterface:
    """Tests for AnimalCognitionInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return AnimalCognitionInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_species(self, interface):
        """Test adding and retrieving a species profile."""
        await interface.initialize()

        profile = SpeciesCognitionProfile(
            species_id="test_species",
            common_name="Test Animal",
            scientific_name="Testus animalis",
            taxonomic_group=TaxonomicGroup.CORVIDS,
            cognition_domains={CognitionDomain.PROBLEM_SOLVING: 0.8},
        )

        await interface.add_species_profile(profile)
        retrieved = await interface.get_species_profile("test_species")

        assert retrieved is not None
        assert retrieved.common_name == "Test Animal"

    @pytest.mark.asyncio
    async def test_query_by_taxonomic_group(self, interface):
        """Test querying species by taxonomic group."""
        await interface.initialize()

        profile1 = SpeciesCognitionProfile(
            species_id="corvid_1",
            common_name="Crow 1",
            scientific_name="Corvus sp.",
            taxonomic_group=TaxonomicGroup.CORVIDS,
        )
        profile2 = SpeciesCognitionProfile(
            species_id="ape_1",
            common_name="Ape 1",
            scientific_name="Pan sp.",
            taxonomic_group=TaxonomicGroup.GREAT_APES,
        )

        await interface.add_species_profile(profile1)
        await interface.add_species_profile(profile2)

        corvids = await interface.query_by_taxonomic_group(TaxonomicGroup.CORVIDS)
        assert len(corvids) == 1
        assert corvids[0].species_id == "corvid_1"

    @pytest.mark.asyncio
    async def test_query_by_cognition_domain(self, interface):
        """Test querying species by cognition domain."""
        await interface.initialize()

        profile = SpeciesCognitionProfile(
            species_id="tool_user",
            common_name="Tool User",
            scientific_name="Toolus userus",
            taxonomic_group=TaxonomicGroup.CORVIDS,
            cognition_domains={CognitionDomain.TOOL_USE: 0.9},
        )
        await interface.add_species_profile(profile)

        tool_users = await interface.query_by_cognition_domain(
            CognitionDomain.TOOL_USE,
            min_evidence=0.5
        )
        assert len(tool_users) == 1

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "species" in result
        assert result["species"] >= 10  # Should have at least 10 seed species
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.species_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedSpecies:
    """Tests for seed species data quality."""

    @pytest.fixture
    def interface_with_seeds(self):
        """Create interface with seed data."""
        interface = AnimalCognitionInterface()
        asyncio.get_event_loop().run_until_complete(
            interface.initialize_all_seed_data()
        )
        return interface

    @pytest.mark.asyncio
    async def test_chimp_profile_exists(self):
        """Test chimpanzee seed profile exists."""
        interface = AnimalCognitionInterface()
        await interface.initialize_all_seed_data()

        chimp = await interface.get_species_profile("pan_troglodytes")
        assert chimp is not None
        assert chimp.common_name == "Chimpanzee"
        assert CognitionDomain.TOOL_USE in chimp.cognition_domains

    @pytest.mark.asyncio
    async def test_dolphin_profile_exists(self):
        """Test dolphin seed profile exists."""
        interface = AnimalCognitionInterface()
        await interface.initialize_all_seed_data()

        dolphin = await interface.get_species_profile("tursiops_truncatus")
        assert dolphin is not None
        assert dolphin.common_name == "Bottlenose Dolphin"

    @pytest.mark.asyncio
    async def test_octopus_profile_exists(self):
        """Test octopus seed profile exists."""
        interface = AnimalCognitionInterface()
        await interface.initialize_all_seed_data()

        octopus = await interface.get_species_profile("octopus_vulgaris")
        assert octopus is not None
        assert octopus.taxonomic_group == TaxonomicGroup.CEPHALOPODS


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_indigenous_knowledge_link_to_form_29(self):
        """Test Form 30 can link to Form 29 animal wisdom."""
        knowledge = IndigenousAnimalKnowledge(
            knowledge_id="link_test",
            species_id="orcinus_orca",
            folk_wisdom_id="inuit_orca_wisdom",  # Would link to Form 29
            behavioral_claim="Traditional knowledge claim",
            unique_indigenous_insight="Indigenous insight",
        )
        assert knowledge.folk_wisdom_id is not None

    def test_species_profile_has_indigenous_perspectives(self):
        """Test species profiles can store indigenous perspectives."""
        profile = SpeciesCognitionProfile(
            species_id="test",
            common_name="Test",
            scientific_name="Test species",
            taxonomic_group=TaxonomicGroup.CETACEANS,
            indigenous_perspectives=[
                "Inuit knowledge of whale behavior",
                "Pacific Northwest orca traditions",
            ],
        )
        assert len(profile.indigenous_perspectives) == 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
