#!/usr/bin/env python3
"""
Test suite for Form 29: Folk & Indigenous Wisdom

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
    "folk_wisdom_interface",
    INTERFACE_PATH / "folk_wisdom_interface.py"
)

# Import Form 29 components from loaded module
FolkTraditionRegion = interface_module.FolkTraditionRegion
WisdomTransmissionMode = interface_module.WisdomTransmissionMode
AnimisticDomain = interface_module.AnimisticDomain
EthicalPrinciple = interface_module.EthicalPrinciple
CosmologicalElement = interface_module.CosmologicalElement
MaturityLevel = interface_module.MaturityLevel
FolkWisdomTeaching = interface_module.FolkWisdomTeaching
AnimisticPractice = interface_module.AnimisticPractice
IndigenousCosmology = interface_module.IndigenousCosmology
OralTradition = interface_module.OralTradition
IndigenousAnimalWisdom = interface_module.IndigenousAnimalWisdom
FolkWisdomMaturityState = interface_module.FolkWisdomMaturityState
FolkWisdomInterface = interface_module.FolkWisdomInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestFolkTraditionRegion:
    """Tests for FolkTraditionRegion enum."""

    def test_region_count(self):
        """Test that we have the expected number of regions."""
        regions = list(FolkTraditionRegion)
        # 5 Africa + 6 Europe + 5 Asia + 5 Oceania + 7 Americas = 28
        assert len(regions) >= 26, f"Expected at least 26 regions, got {len(regions)}"

    def test_african_regions_exist(self):
        """Test African regions are defined."""
        african = [
            FolkTraditionRegion.WEST_AFRICAN,
            FolkTraditionRegion.EAST_AFRICAN,
            FolkTraditionRegion.SOUTHERN_AFRICAN,
            FolkTraditionRegion.CENTRAL_AFRICAN,
            FolkTraditionRegion.NORTH_AFRICAN_BERBER,
        ]
        assert len(african) == 5

    def test_european_regions_exist(self):
        """Test European regions are defined."""
        european = [
            FolkTraditionRegion.CELTIC,
            FolkTraditionRegion.NORSE_GERMANIC,
            FolkTraditionRegion.SLAVIC,
            FolkTraditionRegion.BALTIC,
            FolkTraditionRegion.MEDITERRANEAN_FOLK,
            FolkTraditionRegion.FINNO_UGRIC,
        ]
        assert len(european) == 6

    def test_americas_regions_exist(self):
        """Test Americas regions are defined."""
        americas = [
            FolkTraditionRegion.ARCTIC_INDIGENOUS,
            FolkTraditionRegion.PACIFIC_NORTHWEST,
            FolkTraditionRegion.PLAINS_INDIGENOUS,
            FolkTraditionRegion.EASTERN_WOODLANDS,
            FolkTraditionRegion.MESOAMERICAN_FOLK,
            FolkTraditionRegion.AMAZONIAN,
            FolkTraditionRegion.ANDEAN_FOLK,
        ]
        assert len(americas) == 7

    def test_region_values_are_strings(self):
        """Test that all region values are strings."""
        for region in FolkTraditionRegion:
            assert isinstance(region.value, str)


class TestWisdomTransmissionMode:
    """Tests for WisdomTransmissionMode enum."""

    def test_transmission_modes_exist(self):
        """Test transmission modes are defined."""
        modes = [
            WisdomTransmissionMode.ORAL_NARRATIVE,
            WisdomTransmissionMode.SONG_CHANT,
            WisdomTransmissionMode.RITUAL_CEREMONY,
            WisdomTransmissionMode.DANCE_MOVEMENT,
            WisdomTransmissionMode.VISUAL_SYMBOLIC,
            WisdomTransmissionMode.MATERIAL_CRAFT,
            WisdomTransmissionMode.APPRENTICESHIP,
            WisdomTransmissionMode.DREAM_VISION,
        ]
        assert len(modes) >= 8


class TestAnimisticDomain:
    """Tests for AnimisticDomain enum."""

    def test_domains_exist(self):
        """Test animistic domains are defined."""
        domains = [
            AnimisticDomain.NATURE_SPIRITS,
            AnimisticDomain.ANCESTOR_RELATIONS,
            AnimisticDomain.ANIMAL_POWERS,
            AnimisticDomain.PLANT_INTELLIGENCE,
            AnimisticDomain.ELEMENTAL_FORCES,
            AnimisticDomain.LAND_CONSCIOUSNESS,
            AnimisticDomain.CELESTIAL_BEINGS,
            AnimisticDomain.UNDERWORLD_ENTITIES,
        ]
        assert len(domains) >= 8


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestFolkWisdomTeaching:
    """Tests for FolkWisdomTeaching dataclass."""

    def test_teaching_creation(self):
        """Test basic teaching creation."""
        teaching = FolkWisdomTeaching(
            teaching_id="test_teaching",
            name="Test Teaching",
            region=FolkTraditionRegion.WEST_AFRICAN,
            domains=[AnimisticDomain.ANCESTOR_RELATIONS],
            transmission_modes=[WisdomTransmissionMode.ORAL_NARRATIVE],
            core_teaching="Test core teaching content",
        )
        assert teaching.teaching_id == "test_teaching"
        assert teaching.name == "Test Teaching"
        assert teaching.region == FolkTraditionRegion.WEST_AFRICAN

    def test_teaching_to_embedding_text(self):
        """Test embedding text generation."""
        teaching = FolkWisdomTeaching(
            teaching_id="test",
            name="Ashe",
            region=FolkTraditionRegion.WEST_AFRICAN,
            domains=[AnimisticDomain.NATURE_SPIRITS],
            transmission_modes=[WisdomTransmissionMode.RITUAL_CEREMONY],
            core_teaching="Vital force of creation",
        )
        text = teaching.to_embedding_text()
        assert "Ashe" in text
        assert "west_african" in text
        assert "Vital force" in text


class TestAnimisticPractice:
    """Tests for AnimisticPractice dataclass."""

    def test_practice_creation(self):
        """Test basic practice creation."""
        practice = AnimisticPractice(
            practice_id="test_practice",
            name="Ancestor Offering",
            region=FolkTraditionRegion.WEST_AFRICAN,
            domain=AnimisticDomain.ANCESTOR_RELATIONS,
            description="Making offerings to ancestors",
            purpose="Maintain ancestral connection",
        )
        assert practice.practice_id == "test_practice"
        assert practice.domain == AnimisticDomain.ANCESTOR_RELATIONS


class TestIndigenousAnimalWisdom:
    """Tests for IndigenousAnimalWisdom dataclass."""

    def test_animal_wisdom_creation(self):
        """Test animal wisdom creation."""
        wisdom = IndigenousAnimalWisdom(
            wisdom_id="test_animal",
            animal_name="Raven",
            region=FolkTraditionRegion.PACIFIC_NORTHWEST,
            spiritual_significance="Trickster and creator",
            behavioral_observations=["Intelligent tool user"],
            human_relationship="Clan totem",
        )
        assert wisdom.animal_name == "Raven"
        assert "tool user" in wisdom.behavioral_observations[0]


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestFolkWisdomInterface:
    """Tests for FolkWisdomInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return FolkWisdomInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_teaching(self, interface):
        """Test adding and retrieving a teaching."""
        await interface.initialize()

        teaching = FolkWisdomTeaching(
            teaching_id="celtic_awen",
            name="Awen",
            region=FolkTraditionRegion.CELTIC,
            domains=[AnimisticDomain.NATURE_SPIRITS],
            transmission_modes=[WisdomTransmissionMode.SONG_CHANT],
            core_teaching="Flowing spirit of inspiration",
        )

        await interface.add_teaching(teaching)
        retrieved = await interface.get_teaching("celtic_awen")

        assert retrieved is not None
        assert retrieved.name == "Awen"
        assert retrieved.region == FolkTraditionRegion.CELTIC

    @pytest.mark.asyncio
    async def test_query_by_region(self, interface):
        """Test querying teachings by region."""
        await interface.initialize()

        teaching1 = FolkWisdomTeaching(
            teaching_id="celtic_1",
            name="Teaching 1",
            region=FolkTraditionRegion.CELTIC,
            domains=[AnimisticDomain.NATURE_SPIRITS],
            transmission_modes=[WisdomTransmissionMode.ORAL_NARRATIVE],
            core_teaching="Test 1",
        )
        teaching2 = FolkWisdomTeaching(
            teaching_id="norse_1",
            name="Teaching 2",
            region=FolkTraditionRegion.NORSE_GERMANIC,
            domains=[AnimisticDomain.ELEMENTAL_FORCES],
            transmission_modes=[WisdomTransmissionMode.ORAL_NARRATIVE],
            core_teaching="Test 2",
        )

        await interface.add_teaching(teaching1)
        await interface.add_teaching(teaching2)

        celtic_teachings = await interface.query_teachings_by_region(
            FolkTraditionRegion.CELTIC
        )
        assert len(celtic_teachings) == 1
        assert celtic_teachings[0].teaching_id == "celtic_1"

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "teachings" in result
        assert result["teachings"] > 0
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.teaching_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_form_29_link_field_exists(self):
        """Test Form 29 animal wisdom has link field for Form 30."""
        wisdom = IndigenousAnimalWisdom(
            wisdom_id="test",
            animal_name="Whale",
            region=FolkTraditionRegion.ARCTIC_INDIGENOUS,
            spiritual_significance="Ocean spirit",
            human_relationship="Sacred hunting relationship",
            form_30_link="tursiops_truncatus",  # Dolphin
        )
        assert wisdom.form_30_link is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
