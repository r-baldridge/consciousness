#!/usr/bin/env python3
"""
Test suite for Form 39: Trauma & Dissociative Consciousness

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
    "trauma_consciousness_interface",
    INTERFACE_PATH / "trauma_consciousness_interface.py"
)

# Import Form 39 components from loaded module
TraumaType = interface_module.TraumaType
DissociativeState = interface_module.DissociativeState
HealingModality = interface_module.HealingModality
NervousSystemState = interface_module.NervousSystemState
TraumaResponse = interface_module.TraumaResponse
IntegrationStage = interface_module.IntegrationStage
MaturityLevel = interface_module.MaturityLevel
TraumaProfile = interface_module.TraumaProfile
DissociativeExperience = interface_module.DissociativeExperience
HealingApproach = interface_module.HealingApproach
NervousSystemAssessment = interface_module.NervousSystemAssessment
IntergenerationalPattern = interface_module.IntergenerationalPattern
TraumaConsciousnessMaturityState = interface_module.TraumaConsciousnessMaturityState
TraumaConsciousnessInterface = interface_module.TraumaConsciousnessInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestTraumaType:
    """Tests for TraumaType enum."""

    def test_type_count(self):
        """Test that we have the expected number of trauma types."""
        types = list(TraumaType)
        assert len(types) >= 5, f"Expected at least 5 types, got {len(types)}"

    def test_temporal_types_exist(self):
        """Test temporal/pattern-based trauma types are defined."""
        temporal = [
            TraumaType.ACUTE_SINGLE,
            TraumaType.COMPLEX_DEVELOPMENTAL,
            TraumaType.CHRONIC_ONGOING,
        ]
        assert len(temporal) == 3

    def test_type_values_are_strings(self):
        """Test that all type values are strings."""
        for trauma_type in TraumaType:
            assert isinstance(trauma_type.value, str)


class TestDissociativeState:
    """Tests for DissociativeState enum."""

    def test_state_count(self):
        """Test that we have expected dissociative states."""
        states = list(DissociativeState)
        assert len(states) >= 5, f"Expected at least 5 states, got {len(states)}"


class TestHealingModality:
    """Tests for HealingModality enum."""

    def test_modality_count(self):
        """Test we have expected healing modalities."""
        modalities = list(HealingModality)
        assert len(modalities) >= 8, f"Expected at least 8 modalities, got {len(modalities)}"


class TestNervousSystemState:
    """Tests for NervousSystemState enum."""

    def test_states_exist(self):
        """Test nervous system states are defined."""
        states = list(NervousSystemState)
        assert len(states) >= 3, f"Expected at least 3 states, got {len(states)}"


class TestTraumaResponse:
    """Tests for TraumaResponse enum."""

    def test_responses_exist(self):
        """Test trauma responses are defined."""
        responses = list(TraumaResponse)
        assert len(responses) >= 4, f"Expected at least 4 responses, got {len(responses)}"


class TestIntegrationStage:
    """Tests for IntegrationStage enum."""

    def test_stages_exist(self):
        """Test integration stages are defined."""
        stages = list(IntegrationStage)
        assert len(stages) >= 3, f"Expected at least 3 stages, got {len(stages)}"


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestTraumaProfile:
    """Tests for TraumaProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = TraumaProfile(
            trauma_id="complex_ptsd_001",
            trauma_type=TraumaType.COMPLEX_DEVELOPMENTAL,
            description="Complex developmental trauma profile",
            consciousness_effects=["Fragmented memory", "Dissociation"],
        )
        assert profile.trauma_id == "complex_ptsd_001"
        assert profile.trauma_type == TraumaType.COMPLEX_DEVELOPMENTAL
        assert len(profile.consciousness_effects) == 2

    def test_profile_with_responses(self):
        """Test profile with trauma responses."""
        profile = TraumaProfile(
            trauma_id="test",
            trauma_type=TraumaType.ACUTE_SINGLE,
            trauma_responses=[TraumaResponse.FIGHT, TraumaResponse.FLIGHT],
        )
        assert len(profile.trauma_responses) == 2


class TestDissociativeExperience:
    """Tests for DissociativeExperience dataclass."""

    def test_experience_creation(self):
        """Test dissociative experience creation."""
        experience = DissociativeExperience(
            experience_id="deperson_001",
            dissociative_state=DissociativeState.DEPERSONALIZATION,
            description="Feeling disconnected from body",
            phenomenology=["Out-of-body sensation", "Observing self from outside"],
        )
        assert experience.dissociative_state == DissociativeState.DEPERSONALIZATION
        assert len(experience.phenomenology) == 2


class TestHealingApproach:
    """Tests for HealingApproach dataclass."""

    def test_approach_creation(self):
        """Test healing approach creation."""
        approach = HealingApproach(
            approach_id="emdr_001",
            modality=HealingModality.EMDR,
            description="Eye Movement Desensitization and Reprocessing",
            evidence_base="Strong empirical support for PTSD",
            trauma_types_addressed=[TraumaType.ACUTE_SINGLE, TraumaType.COMPLEX_DEVELOPMENTAL],
        )
        assert approach.modality == HealingModality.EMDR
        assert len(approach.trauma_types_addressed) == 2


class TestNervousSystemAssessment:
    """Tests for NervousSystemAssessment dataclass."""

    def test_assessment_creation(self):
        """Test nervous system assessment creation."""
        assessment = NervousSystemAssessment(
            assessment_id="assessment_001",
            current_state=NervousSystemState.HYPERAROUSAL,
            description="Elevated nervous system state",
            indicators=["Racing heart", "Hypervigilance"],
        )
        assert assessment.current_state == NervousSystemState.HYPERAROUSAL
        assert len(assessment.indicators) == 2


class TestIntergenerationalPattern:
    """Tests for IntergenerationalPattern dataclass."""

    def test_pattern_creation(self):
        """Test intergenerational pattern creation."""
        pattern = IntergenerationalPattern(
            pattern_id="pattern_001",
            description="Ancestral trauma transmission",
            transmission_mechanisms=["Attachment patterns", "Epigenetic changes"],
            cultural_context="Collective trauma experience",
        )
        assert pattern.pattern_id == "pattern_001"
        assert len(pattern.transmission_mechanisms) == 2


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestTraumaConsciousnessInterface:
    """Tests for TraumaConsciousnessInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return TraumaConsciousnessInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_trauma_profile(self, interface):
        """Test adding and retrieving a trauma profile."""
        await interface.initialize()

        profile = TraumaProfile(
            trauma_id="test_trauma",
            trauma_type=TraumaType.ACUTE_SINGLE,
            description="Single event trauma",
        )

        await interface.add_trauma_profile(profile)
        retrieved = await interface.get_trauma_profile("test_trauma")

        assert retrieved is not None
        assert retrieved.trauma_type == TraumaType.ACUTE_SINGLE

    @pytest.mark.asyncio
    async def test_query_profiles_by_type(self, interface):
        """Test querying profiles by trauma type."""
        await interface.initialize()

        profile = TraumaProfile(
            trauma_id="complex_001",
            trauma_type=TraumaType.COMPLEX_DEVELOPMENTAL,
            description="Complex trauma",
        )
        await interface.add_trauma_profile(profile)

        profiles = await interface.query_profiles_by_type(
            TraumaType.COMPLEX_DEVELOPMENTAL
        )
        assert len(profiles) >= 1

    @pytest.mark.asyncio
    async def test_add_and_get_dissociative_experience(self, interface):
        """Test adding and retrieving dissociative experience."""
        await interface.initialize()

        experience = DissociativeExperience(
            experience_id="test_dissoc",
            dissociative_state=DissociativeState.DEREALIZATION,
            description="World feels unreal",
        )

        await interface.add_dissociative_experience(experience)
        retrieved = await interface.get_dissociative_experience("test_dissoc")

        assert retrieved is not None
        assert retrieved.dissociative_state == DissociativeState.DEREALIZATION

    @pytest.mark.asyncio
    async def test_add_and_get_healing_approach(self, interface):
        """Test adding and retrieving healing approach."""
        await interface.initialize()

        approach = HealingApproach(
            approach_id="test_somatic",
            modality=HealingModality.SOMATIC_EXPERIENCING,
            description="Body-based trauma therapy",
        )

        await interface.add_healing_approach(approach)
        retrieved = await interface.get_healing_approach("test_somatic")

        assert retrieved is not None
        assert retrieved.modality == HealingModality.SOMATIC_EXPERIENCING

    @pytest.mark.asyncio
    async def test_add_and_get_nervous_system_assessment(self, interface):
        """Test adding and retrieving nervous system assessment."""
        await interface.initialize()

        assessment = NervousSystemAssessment(
            assessment_id="test_assessment",
            current_state=NervousSystemState.HYPOAROUSAL,
            description="Shutdown state",
        )

        await interface.add_nervous_system_assessment(assessment)
        retrieved = await interface.get_nervous_system_assessment("test_assessment")

        assert retrieved is not None
        assert retrieved.current_state == NervousSystemState.HYPOAROUSAL

    @pytest.mark.asyncio
    async def test_add_and_get_intergenerational_pattern(self, interface):
        """Test adding and retrieving intergenerational pattern."""
        await interface.initialize()

        pattern = IntergenerationalPattern(
            pattern_id="test_pattern",
            description="Generational trauma pattern",
        )

        await interface.add_intergenerational_pattern(pattern)
        retrieved = await interface.get_intergenerational_pattern("test_pattern")

        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "profiles" in result
        assert result["profiles"] > 0
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedProfiles:
    """Tests for seed profile data quality."""

    @pytest.mark.asyncio
    async def test_trauma_profiles_comprehensive(self):
        """Test trauma profiles cover key types."""
        interface = TraumaConsciousnessInterface()
        await interface.initialize_all_seed_data()

        # Check that multiple trauma types are represented
        acute_profiles = await interface.query_profiles_by_type(
            TraumaType.ACUTE_SINGLE
        )
        complex_profiles = await interface.query_profiles_by_type(
            TraumaType.COMPLEX_DEVELOPMENTAL
        )

        total_profiles = len(acute_profiles) + len(complex_profiles)
        assert total_profiles >= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_trauma_profile_has_consciousness_effects(self):
        """Test trauma profiles can store consciousness effects."""
        profile = TraumaProfile(
            trauma_id="test",
            trauma_type=TraumaType.COMPLEX_DEVELOPMENTAL,
            consciousness_effects=[
                "Altered sense of time",
                "Fragmented identity",
                "Dissociative episodes",
            ],
        )
        assert len(profile.consciousness_effects) >= 3

    def test_healing_approach_addresses_multiple_types(self):
        """Test healing approaches can address multiple trauma types."""
        approach = HealingApproach(
            approach_id="test",
            modality=HealingModality.IFS,
            description="Internal Family Systems",
            trauma_types_addressed=[
                TraumaType.ACUTE_SINGLE,
                TraumaType.COMPLEX_DEVELOPMENTAL,
                TraumaType.CHRONIC_ONGOING,
            ],
        )
        assert len(approach.trauma_types_addressed) >= 3

    def test_intergenerational_pattern_has_cultural_context(self):
        """Test intergenerational patterns include cultural context."""
        pattern = IntergenerationalPattern(
            pattern_id="test",
            description="Historical trauma pattern",
            cultural_context="Collective historical trauma",
            transmission_mechanisms=["Narrative transmission", "Behavioral patterns"],
        )
        assert pattern.cultural_context is not None
        assert len(pattern.transmission_mechanisms) >= 2

    def test_trauma_informed_framework(self):
        """Test that the framework supports trauma-informed principles."""
        # Safety - profiles can track current nervous system state
        assessment = NervousSystemAssessment(
            assessment_id="test",
            current_state=NervousSystemState.WINDOW_OF_TOLERANCE,
            description="Regulated state",
        )
        assert assessment.current_state == NervousSystemState.WINDOW_OF_TOLERANCE

        # Healing - approaches exist for recovery
        approach = HealingApproach(
            approach_id="test",
            modality=HealingModality.SOMATIC_EXPERIENCING,
            description="Body-based healing",
        )
        assert approach.modality == HealingModality.SOMATIC_EXPERIENCING


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
