#!/usr/bin/env python3
"""
Test suite for Form 37: Psychedelic/Entheogenic Consciousness

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
    "psychedelic_consciousness_interface",
    INTERFACE_PATH / "psychedelic_consciousness_interface.py"
)

# Import Form 37 components from loaded module
PsychedelicSubstance = interface_module.PsychedelicSubstance
ExperienceType = interface_module.ExperienceType
EntheogenicTradition = interface_module.EntheogenicTradition
TherapeuticApplication = interface_module.TherapeuticApplication
NeuralMechanism = interface_module.NeuralMechanism
SetSettingFactor = interface_module.SetSettingFactor
MaturityLevel = interface_module.MaturityLevel
SubstanceProfile = interface_module.SubstanceProfile
PsychedelicExperience = interface_module.PsychedelicExperience
EntheogenicCeremony = interface_module.EntheogenicCeremony
TherapeuticProtocol = interface_module.TherapeuticProtocol
ExperiencePhenomenology = interface_module.ExperiencePhenomenology
PsychedelicConsciousnessMaturityState = interface_module.PsychedelicConsciousnessMaturityState
PsychedelicConsciousnessInterface = interface_module.PsychedelicConsciousnessInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestPsychedelicSubstance:
    """Tests for PsychedelicSubstance enum."""

    def test_substance_count(self):
        """Test that we have the expected number of substances."""
        substances = list(PsychedelicSubstance)
        assert len(substances) >= 10, f"Expected at least 10 substances, got {len(substances)}"

    def test_classical_psychedelics_exist(self):
        """Test classical psychedelics are defined."""
        classical = [
            PsychedelicSubstance.PSILOCYBIN,
            PsychedelicSubstance.LSD,
            PsychedelicSubstance.DMT,
            PsychedelicSubstance.MESCALINE,
        ]
        assert len(classical) == 4

    def test_therapeutic_substances_exist(self):
        """Test therapeutic substances are defined."""
        therapeutic = [
            PsychedelicSubstance.PSILOCYBIN,
            PsychedelicSubstance.MDMA,
            PsychedelicSubstance.KETAMINE,
        ]
        assert len(therapeutic) == 3

    def test_traditional_entheogens_exist(self):
        """Test traditional entheogens are defined."""
        traditional = [
            PsychedelicSubstance.AYAHUASCA,
            PsychedelicSubstance.IBOGAINE,
            PsychedelicSubstance.SAN_PEDRO,
        ]
        assert len(traditional) == 3

    def test_substance_values_are_strings(self):
        """Test that all substance values are strings."""
        for substance in PsychedelicSubstance:
            assert isinstance(substance.value, str)


class TestExperienceType:
    """Tests for ExperienceType enum."""

    def test_experience_type_count(self):
        """Test that we have expected experience types."""
        types = list(ExperienceType)
        assert len(types) >= 10, f"Expected at least 10 types, got {len(types)}"

    def test_key_experience_types_exist(self):
        """Test key experience types are defined."""
        key_types = [
            ExperienceType.VISUAL_GEOMETRY,
            ExperienceType.ENTITY_ENCOUNTER,
            ExperienceType.EGO_DISSOLUTION,
            ExperienceType.MYSTICAL_UNITY,
            ExperienceType.EMOTIONAL_CATHARSIS,
        ]
        assert len(key_types) == 5


class TestEntheogenicTradition:
    """Tests for EntheogenicTradition enum."""

    def test_tradition_count(self):
        """Test we have expected number of traditions."""
        traditions = list(EntheogenicTradition)
        assert len(traditions) >= 8, f"Expected at least 8 traditions, got {len(traditions)}"

    def test_geographic_traditions_exist(self):
        """Test geographic traditions are defined."""
        geographic = [
            EntheogenicTradition.AMAZONIAN_AYAHUASCA,
            EntheogenicTradition.NATIVE_AMERICAN_PEYOTE,
            EntheogenicTradition.MESOAMERICAN_MUSHROOM,
            EntheogenicTradition.AFRICAN_IBOGA,
        ]
        assert len(geographic) == 4


class TestTherapeuticApplication:
    """Tests for TherapeuticApplication enum."""

    def test_therapeutic_applications_exist(self):
        """Test therapeutic applications are defined."""
        applications = [
            TherapeuticApplication.DEPRESSION,
            TherapeuticApplication.PTSD,
            TherapeuticApplication.ADDICTION,
            TherapeuticApplication.END_OF_LIFE_ANXIETY,
        ]
        assert len(applications) == 4


class TestNeuralMechanism:
    """Tests for NeuralMechanism enum."""

    def test_mechanisms_exist(self):
        """Test neural mechanisms are defined."""
        mechanisms = [
            NeuralMechanism.DMN_DISRUPTION,
            NeuralMechanism.SEROTONIN_5HT2A,
            NeuralMechanism.NEURAL_ENTROPY,
            NeuralMechanism.NEUROPLASTICITY,
        ]
        assert len(mechanisms) == 4


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestSubstanceProfile:
    """Tests for SubstanceProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = SubstanceProfile(
            substance_id="test_psilocybin",
            substance=PsychedelicSubstance.PSILOCYBIN,
            chemical_class="Tryptamine",
            receptor_targets=["5-HT2A"],
            typical_dose="25mg",
            duration="4-6 hours",
            subjective_effects=["Visual enhancement", "Introspection"],
        )
        assert profile.substance_id == "test_psilocybin"
        assert profile.substance == PsychedelicSubstance.PSILOCYBIN
        assert len(profile.subjective_effects) == 2

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = SubstanceProfile(
            substance_id="test",
            substance=PsychedelicSubstance.LSD,
            chemical_class="Lysergamide",
            receptor_targets=["5-HT2A", "D2"],
            typical_dose="100ug",
            duration="8-12 hours",
            subjective_effects=["Visual distortions", "Cognitive enhancement"],
        )
        text = profile.to_embedding_text()
        assert "lsd" in text
        assert "Lysergamide" in text


class TestPsychedelicExperience:
    """Tests for PsychedelicExperience dataclass."""

    def test_experience_creation(self):
        """Test experience creation."""
        experience = PsychedelicExperience(
            experience_id="exp_001",
            substance=PsychedelicSubstance.PSILOCYBIN,
            dose="3.5g dried mushrooms",
            experience_types=[ExperienceType.MYSTICAL_UNITY, ExperienceType.EGO_DISSOLUTION],
            intensity_level=4,
        )
        assert experience.experience_id == "exp_001"
        assert len(experience.experience_types) == 2
        assert experience.intensity_level == 4


class TestEntheogenicCeremony:
    """Tests for EntheogenicCeremony dataclass."""

    def test_ceremony_creation(self):
        """Test ceremony creation with Form 29 link."""
        ceremony = EntheogenicCeremony(
            ceremony_id="ayahuasca_001",
            tradition=EntheogenicTradition.AMAZONIAN_AYAHUASCA,
            substance=PsychedelicSubstance.AYAHUASCA,
            ritual_structure=["Opening prayers", "Medicine serving", "Icaros"],
            spiritual_framework="Plant spirit communication",
            form_29_link="amazonian_plant_medicine",
        )
        assert ceremony.tradition == EntheogenicTradition.AMAZONIAN_AYAHUASCA
        assert ceremony.form_29_link is not None


class TestTherapeuticProtocol:
    """Tests for TherapeuticProtocol dataclass."""

    def test_protocol_creation(self):
        """Test therapeutic protocol creation."""
        protocol = TherapeuticProtocol(
            protocol_id="psilocybin_depression_001",
            indication=TherapeuticApplication.DEPRESSION,
            substance=PsychedelicSubstance.PSILOCYBIN,
            dose_schedule=["25mg"],
            preparation_sessions=3,
            dosing_sessions=2,
            integration_sessions=3,
            research_institution="Johns Hopkins",
        )
        assert protocol.indication == TherapeuticApplication.DEPRESSION
        assert protocol.preparation_sessions == 3


class TestExperiencePhenomenology:
    """Tests for ExperiencePhenomenology dataclass."""

    def test_phenomenology_creation(self):
        """Test phenomenology creation."""
        phenom = ExperiencePhenomenology(
            phenomenology_id="ego_dissolution_001",
            experience_type=ExperienceType.EGO_DISSOLUTION,
            description="Loss of self-boundaries",
            subjective_features=["Unity with environment", "Loss of personal narrative"],
            neural_correlates=[NeuralMechanism.DMN_DISRUPTION],
        )
        assert phenom.experience_type == ExperienceType.EGO_DISSOLUTION
        assert len(phenom.subjective_features) == 2


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestPsychedelicConsciousnessInterface:
    """Tests for PsychedelicConsciousnessInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return PsychedelicConsciousnessInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_substance(self, interface):
        """Test adding and retrieving a substance profile."""
        await interface.initialize()

        profile = SubstanceProfile(
            substance_id="test_dmt",
            substance=PsychedelicSubstance.DMT,
            chemical_class="Tryptamine",
            receptor_targets=["5-HT2A"],
            typical_dose="30mg",
            duration="15-30 minutes",
        )

        await interface.add_substance(profile)
        retrieved = await interface.get_substance("test_dmt")

        assert retrieved is not None
        assert retrieved.substance == PsychedelicSubstance.DMT

    @pytest.mark.asyncio
    async def test_query_by_substance_type(self, interface):
        """Test querying substances by type."""
        await interface.initialize()

        profile = SubstanceProfile(
            substance_id="test_mdma",
            substance=PsychedelicSubstance.MDMA,
            chemical_class="Empathogen",
            receptor_targets=["Serotonin release"],
            typical_dose="125mg",
            duration="4-6 hours",
        )
        await interface.add_substance(profile)

        mdma_profiles = await interface.query_substances_by_type(
            PsychedelicSubstance.MDMA
        )
        assert len(mdma_profiles) >= 1

    @pytest.mark.asyncio
    async def test_add_and_get_ceremony(self, interface):
        """Test adding and retrieving a ceremony."""
        await interface.initialize()

        ceremony = EntheogenicCeremony(
            ceremony_id="test_peyote",
            tradition=EntheogenicTradition.NATIVE_AMERICAN_PEYOTE,
            substance=PsychedelicSubstance.MESCALINE,
            ritual_structure=["Tipi ceremony"],
        )

        await interface.add_ceremony(ceremony)
        retrieved = await interface.get_ceremony("test_peyote")

        assert retrieved is not None
        assert retrieved.tradition == EntheogenicTradition.NATIVE_AMERICAN_PEYOTE

    @pytest.mark.asyncio
    async def test_query_ceremonies_by_tradition(self, interface):
        """Test querying ceremonies by tradition."""
        await interface.initialize()

        ceremony = EntheogenicCeremony(
            ceremony_id="test_santo_daime",
            tradition=EntheogenicTradition.SANTO_DAIME,
            substance=PsychedelicSubstance.AYAHUASCA,
        )
        await interface.add_ceremony(ceremony)

        ceremonies = await interface.query_ceremonies_by_tradition(
            EntheogenicTradition.SANTO_DAIME
        )
        assert len(ceremonies) >= 1

    @pytest.mark.asyncio
    async def test_add_and_get_protocol(self, interface):
        """Test adding and retrieving a therapeutic protocol."""
        await interface.initialize()

        protocol = TherapeuticProtocol(
            protocol_id="test_mdma_ptsd",
            indication=TherapeuticApplication.PTSD,
            substance=PsychedelicSubstance.MDMA,
            dose_schedule=["100mg"],
        )

        await interface.add_protocol(protocol)
        retrieved = await interface.get_protocol("test_mdma_ptsd")

        assert retrieved is not None
        assert retrieved.indication == TherapeuticApplication.PTSD

    @pytest.mark.asyncio
    async def test_query_protocols_by_indication(self, interface):
        """Test querying protocols by indication."""
        await interface.initialize()

        protocol = TherapeuticProtocol(
            protocol_id="test_depression",
            indication=TherapeuticApplication.DEPRESSION,
            substance=PsychedelicSubstance.KETAMINE,
        )
        await interface.add_protocol(protocol)

        protocols = await interface.query_protocols_by_indication(
            TherapeuticApplication.DEPRESSION
        )
        assert len(protocols) >= 1

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "substances" in result
        assert result["substances"] > 0
        assert "ceremonies" in result
        assert "protocols" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.substance_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedSubstances:
    """Tests for seed substance data quality."""

    @pytest.mark.asyncio
    async def test_psilocybin_profile_exists(self):
        """Test psilocybin seed profile exists."""
        interface = PsychedelicConsciousnessInterface()
        await interface.initialize_all_seed_data()

        psilocybin = await interface.get_substance("psilocybin_profile")
        assert psilocybin is not None
        assert psilocybin.substance == PsychedelicSubstance.PSILOCYBIN

    @pytest.mark.asyncio
    async def test_ayahuasca_profile_exists(self):
        """Test ayahuasca seed profile exists."""
        interface = PsychedelicConsciousnessInterface()
        await interface.initialize_all_seed_data()

        ayahuasca = await interface.get_substance("ayahuasca_profile")
        assert ayahuasca is not None
        assert ayahuasca.substance == PsychedelicSubstance.AYAHUASCA

    @pytest.mark.asyncio
    async def test_mdma_profile_exists(self):
        """Test MDMA seed profile exists."""
        interface = PsychedelicConsciousnessInterface()
        await interface.initialize_all_seed_data()

        mdma = await interface.get_substance("mdma_profile")
        assert mdma is not None
        assert mdma.substance == PsychedelicSubstance.MDMA


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_ceremony_link_to_form_29(self):
        """Test ceremonies can link to Form 29 folk wisdom."""
        ceremony = EntheogenicCeremony(
            ceremony_id="test",
            tradition=EntheogenicTradition.AMAZONIAN_AYAHUASCA,
            substance=PsychedelicSubstance.AYAHUASCA,
            form_29_link="amazonian_plant_wisdom",
        )
        assert ceremony.form_29_link is not None

    def test_therapeutic_protocol_has_efficacy_data(self):
        """Test therapeutic protocols can store efficacy data."""
        protocol = TherapeuticProtocol(
            protocol_id="test",
            indication=TherapeuticApplication.DEPRESSION,
            substance=PsychedelicSubstance.PSILOCYBIN,
            efficacy_data={
                "response_rate": "60%",
                "effect_size": "Large",
            },
        )
        assert len(protocol.efficacy_data) >= 2

    def test_experience_has_mystical_score(self):
        """Test experiences can store mystical and ego dissolution scores."""
        experience = PsychedelicExperience(
            experience_id="test",
            substance=PsychedelicSubstance.PSILOCYBIN,
            dose="25mg",
            mystical_score=0.85,
            ego_dissolution_score=0.75,
        )
        assert experience.mystical_score is not None
        assert experience.ego_dissolution_score is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
