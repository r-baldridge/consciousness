#!/usr/bin/env python3
"""
Test suite for Form 35: Developmental Consciousness

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
    "developmental_consciousness_interface",
    INTERFACE_PATH / "developmental_consciousness_interface.py"
)

# Import Form 35 components from loaded module
DevelopmentalStage = interface_module.DevelopmentalStage
DevelopmentalCapacity = interface_module.DevelopmentalCapacity
ConsciousnessMarker = interface_module.ConsciousnessMarker
ResearchMethodology = interface_module.ResearchMethodology
LifespanDomain = interface_module.LifespanDomain
MaturityLevel = interface_module.MaturityLevel
DevelopmentalStageProfile = interface_module.DevelopmentalStageProfile
CapacityEmergence = interface_module.CapacityEmergence
ConsciousnessTransition = interface_module.ConsciousnessTransition
LifespanTrajectory = interface_module.LifespanTrajectory
EndOfLifeAwareness = interface_module.EndOfLifeAwareness
DevelopmentalConsciousnessMaturityState = interface_module.DevelopmentalConsciousnessMaturityState
DevelopmentalConsciousnessInterface = interface_module.DevelopmentalConsciousnessInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestDevelopmentalStage:
    """Tests for DevelopmentalStage enum."""

    def test_stage_count(self):
        """Test that we have the expected number of stages."""
        stages = list(DevelopmentalStage)
        assert len(stages) >= 13, f"Expected at least 13 stages, got {len(stages)}"

    def test_prenatal_stages_exist(self):
        """Test prenatal stages are defined."""
        prenatal = [
            DevelopmentalStage.PRENATAL_EARLY,
            DevelopmentalStage.PRENATAL_LATE,
        ]
        assert len(prenatal) == 2

    def test_early_life_stages_exist(self):
        """Test early life stages are defined."""
        early_life = [
            DevelopmentalStage.NEONATAL,
            DevelopmentalStage.INFANT_EARLY,
            DevelopmentalStage.INFANT_LATE,
            DevelopmentalStage.TODDLER,
        ]
        assert len(early_life) == 4

    def test_childhood_stages_exist(self):
        """Test childhood stages are defined."""
        childhood = [
            DevelopmentalStage.EARLY_CHILDHOOD,
            DevelopmentalStage.MIDDLE_CHILDHOOD,
        ]
        assert len(childhood) == 2

    def test_adult_stages_exist(self):
        """Test adult stages are defined."""
        adult = [
            DevelopmentalStage.ADOLESCENCE,
            DevelopmentalStage.YOUNG_ADULT,
            DevelopmentalStage.MIDDLE_ADULT,
            DevelopmentalStage.LATE_ADULT,
        ]
        assert len(adult) == 4

    def test_end_of_life_stage_exists(self):
        """Test end of life stage is defined."""
        assert DevelopmentalStage.END_OF_LIFE is not None

    def test_stage_values_are_strings(self):
        """Test that all stage values are strings."""
        for stage in DevelopmentalStage:
            assert isinstance(stage.value, str)


class TestDevelopmentalCapacity:
    """Tests for DevelopmentalCapacity enum."""

    def test_capacity_count(self):
        """Test that we have expected capacities."""
        capacities = list(DevelopmentalCapacity)
        assert len(capacities) >= 10, f"Expected at least 10 capacities, got {len(capacities)}"

    def test_key_capacities_exist(self):
        """Test key cognitive capacities are defined."""
        capacities = [
            DevelopmentalCapacity.SENSORY_AWARENESS,
            DevelopmentalCapacity.OBJECT_PERMANENCE,
            DevelopmentalCapacity.SELF_RECOGNITION,
            DevelopmentalCapacity.THEORY_OF_MIND,
            DevelopmentalCapacity.METACOGNITION,
        ]
        assert len(capacities) == 5

    def test_late_life_capacities_exist(self):
        """Test late life capacities are defined."""
        late_life = [
            DevelopmentalCapacity.MORTALITY_AWARENESS,
            DevelopmentalCapacity.WISDOM_INTEGRATION,
        ]
        assert len(late_life) == 2


class TestConsciousnessMarker:
    """Tests for ConsciousnessMarker enum."""

    def test_marker_count(self):
        """Test that we have expected markers."""
        markers = list(ConsciousnessMarker)
        assert len(markers) >= 5, f"Expected at least 5 markers, got {len(markers)}"

    def test_key_markers_exist(self):
        """Test key markers are defined."""
        markers = [
            ConsciousnessMarker.NEURAL_CORRELATE,
            ConsciousnessMarker.BEHAVIORAL_INDICATOR,
            ConsciousnessMarker.COGNITIVE_MILESTONE,
        ]
        assert len(markers) == 3


class TestResearchMethodology:
    """Tests for ResearchMethodology enum."""

    def test_methodology_count(self):
        """Test that we have expected methodologies."""
        methodologies = list(ResearchMethodology)
        assert len(methodologies) >= 6, f"Expected at least 6 methodologies, got {len(methodologies)}"

    def test_key_methodologies_exist(self):
        """Test key methodologies are defined."""
        methodologies = [
            ResearchMethodology.LOOKING_TIME,
            ResearchMethodology.HABITUATION,
            ResearchMethodology.EEG_INFANT,
            ResearchMethodology.LONGITUDINAL_STUDY,
        ]
        assert len(methodologies) == 4


class TestLifespanDomain:
    """Tests for LifespanDomain enum."""

    def test_domain_count(self):
        """Test that we have expected domains."""
        domains = list(LifespanDomain)
        assert len(domains) >= 7, f"Expected at least 7 domains, got {len(domains)}"

    def test_key_domains_exist(self):
        """Test key domains are defined."""
        domains = [
            LifespanDomain.PERCEPTUAL,
            LifespanDomain.COGNITIVE,
            LifespanDomain.EMOTIONAL,
            LifespanDomain.SOCIAL,
            LifespanDomain.SELF_AWARENESS,
        ]
        assert len(domains) == 5


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestDevelopmentalStageProfile:
    """Tests for DevelopmentalStageProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = DevelopmentalStageProfile(
            stage_id="neonatal",
            stage=DevelopmentalStage.NEONATAL,
            age_range="Birth to 28 days",
            consciousness_characteristics=["Primary sensory awareness"],
            emerging_capacities=[DevelopmentalCapacity.SENSORY_AWARENESS],
            neural_developments=["Noradrenergic surge at birth"],
            key_milestones=["First breath"],
        )
        assert profile.stage_id == "neonatal"
        assert profile.stage == DevelopmentalStage.NEONATAL

    def test_profile_with_research_methods(self):
        """Test profile with research methods."""
        profile = DevelopmentalStageProfile(
            stage_id="infant_early",
            stage=DevelopmentalStage.INFANT_EARLY,
            age_range="1-6 months",
            consciousness_characteristics=["Sensory-affective awareness"],
            emerging_capacities=[DevelopmentalCapacity.OBJECT_PERMANENCE],
            neural_developments=["Visual cortex maturation"],
            key_milestones=["Social smiling"],
            research_methods=[
                ResearchMethodology.LOOKING_TIME,
                ResearchMethodology.HABITUATION,
            ],
        )
        assert len(profile.research_methods) == 2

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = DevelopmentalStageProfile(
            stage_id="test",
            stage=DevelopmentalStage.TODDLER,
            age_range="2-3 years",
            consciousness_characteristics=["Egocentric awareness", "Emerging narrative"],
            emerging_capacities=[DevelopmentalCapacity.SELF_RECOGNITION],
            neural_developments=["Language network expansion"],
            key_milestones=["Sentence use", "Symbolic play"],
        )
        text = profile.to_embedding_text()
        assert "toddler" in text
        assert "2-3 years" in text


class TestCapacityEmergence:
    """Tests for CapacityEmergence dataclass."""

    def test_emergence_creation(self):
        """Test capacity emergence creation."""
        emergence = CapacityEmergence(
            capacity_id="theory_of_mind",
            capacity=DevelopmentalCapacity.THEORY_OF_MIND,
            typical_emergence_age="~4 years",
            prerequisites=[
                DevelopmentalCapacity.SELF_RECOGNITION,
                DevelopmentalCapacity.OBJECT_PERMANENCE,
            ],
            neural_correlates=[
                "Temporoparietal junction",
                "Medial prefrontal cortex",
            ],
        )
        assert emergence.capacity == DevelopmentalCapacity.THEORY_OF_MIND
        assert len(emergence.prerequisites) == 2
        assert len(emergence.neural_correlates) == 2


class TestConsciousnessTransition:
    """Tests for ConsciousnessTransition dataclass."""

    def test_transition_creation(self):
        """Test transition creation."""
        transition = ConsciousnessTransition(
            transition_id="birth_awakening",
            from_stage=DevelopmentalStage.PRENATAL_LATE,
            to_stage=DevelopmentalStage.NEONATAL,
            key_changes=["From muted awareness to overwhelming input"],
            duration="Hours to days",
            individual_variation="Birth circumstances affect intensity",
        )
        assert transition.from_stage == DevelopmentalStage.PRENATAL_LATE
        assert transition.to_stage == DevelopmentalStage.NEONATAL


class TestLifespanTrajectory:
    """Tests for LifespanTrajectory dataclass."""

    def test_trajectory_creation(self):
        """Test trajectory creation."""
        trajectory = LifespanTrajectory(
            trajectory_id="cognitive_trajectory",
            domain=LifespanDomain.COGNITIVE,
            developmental_curve="Rise, peak, gradual decline",
            peak_age="25-30 years",
            decline_pattern="Fluid declines, crystallized stable",
            protective_factors=["Education", "Physical activity", "Social engagement"],
        )
        assert trajectory.domain == LifespanDomain.COGNITIVE
        assert len(trajectory.protective_factors) == 3


class TestEndOfLifeAwareness:
    """Tests for EndOfLifeAwareness dataclass."""

    def test_awareness_creation(self):
        """Test end of life awareness creation."""
        awareness = EndOfLifeAwareness(
            awareness_id="terminal_awareness",
            stage=DevelopmentalStage.END_OF_LIFE,
            consciousness_changes=["Liminal awareness", "Transitional states"],
            common_experiences=["Deathbed visions", "Terminal lucidity"],
            cultural_interpretations=["Spiritual transition", "Journey metaphors"],
        )
        assert awareness.stage == DevelopmentalStage.END_OF_LIFE
        assert len(awareness.consciousness_changes) == 2


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestDevelopmentalConsciousnessInterface:
    """Tests for DevelopmentalConsciousnessInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return DevelopmentalConsciousnessInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_stage_profile(self, interface):
        """Test adding and retrieving a stage profile."""
        await interface.initialize()

        profile = DevelopmentalStageProfile(
            stage_id="test_stage",
            stage=DevelopmentalStage.NEONATAL,
            age_range="Birth to 28 days",
            consciousness_characteristics=["Test characteristic"],
            emerging_capacities=[DevelopmentalCapacity.SENSORY_AWARENESS],
            neural_developments=["Test development"],
            key_milestones=["Test milestone"],
        )

        await interface.add_stage_profile(profile)
        retrieved = await interface.get_stage_profile("test_stage")

        assert retrieved is not None
        assert retrieved.stage == DevelopmentalStage.NEONATAL

    @pytest.mark.asyncio
    async def test_get_profile_by_stage(self, interface):
        """Test getting profile by developmental stage."""
        await interface.initialize()

        profile = DevelopmentalStageProfile(
            stage_id="toddler_profile",
            stage=DevelopmentalStage.TODDLER,
            age_range="2-3 years",
            consciousness_characteristics=["Egocentric awareness"],
            emerging_capacities=[DevelopmentalCapacity.SELF_RECOGNITION],
            neural_developments=["Language network expansion"],
            key_milestones=["Sentence use"],
        )
        await interface.add_stage_profile(profile)

        retrieved = await interface.get_profile_by_stage(DevelopmentalStage.TODDLER)
        assert retrieved is not None
        assert retrieved.stage_id == "toddler_profile"

    @pytest.mark.asyncio
    async def test_query_stages_by_capacity(self, interface):
        """Test querying stages by emerging capacity."""
        await interface.initialize()

        profile = DevelopmentalStageProfile(
            stage_id="tom_stage",
            stage=DevelopmentalStage.EARLY_CHILDHOOD,
            age_range="3-6 years",
            consciousness_characteristics=["Full naive theory of mind"],
            emerging_capacities=[DevelopmentalCapacity.THEORY_OF_MIND],
            neural_developments=["Theory of mind network maturation"],
            key_milestones=["False belief threshold"],
        )
        await interface.add_stage_profile(profile)

        stages = await interface.query_stages_by_capacity(DevelopmentalCapacity.THEORY_OF_MIND)
        assert len(stages) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_capacity_emergence(self, interface):
        """Test adding and retrieving capacity emergence."""
        await interface.initialize()

        emergence = CapacityEmergence(
            capacity_id="test_capacity",
            capacity=DevelopmentalCapacity.METACOGNITION,
            typical_emergence_age="~6 years",
            prerequisites=[DevelopmentalCapacity.THEORY_OF_MIND],
            neural_correlates=["Prefrontal cortex"],
        )

        await interface.add_capacity_emergence(emergence)
        retrieved = await interface.get_capacity_emergence("test_capacity")

        assert retrieved is not None
        assert retrieved.capacity == DevelopmentalCapacity.METACOGNITION

    @pytest.mark.asyncio
    async def test_add_and_get_transition(self, interface):
        """Test adding and retrieving a transition."""
        await interface.initialize()

        transition = ConsciousnessTransition(
            transition_id="test_transition",
            from_stage=DevelopmentalStage.INFANT_EARLY,
            to_stage=DevelopmentalStage.INFANT_LATE,
            key_changes=["Joint attention emergence"],
            duration="Weeks to months",
            individual_variation="Variable",
        )

        await interface.add_transition(transition)
        retrieved = await interface.get_transition("test_transition")

        assert retrieved is not None
        assert retrieved.from_stage == DevelopmentalStage.INFANT_EARLY

    @pytest.mark.asyncio
    async def test_get_transitions_from_stage(self, interface):
        """Test getting transitions from a specific stage."""
        await interface.initialize()

        transition = ConsciousnessTransition(
            transition_id="from_infant",
            from_stage=DevelopmentalStage.INFANT_LATE,
            to_stage=DevelopmentalStage.TODDLER,
            key_changes=["Self-recognition emerges"],
            duration="Months",
            individual_variation="Some variation",
        )
        await interface.add_transition(transition)

        transitions = await interface.get_transitions_from_stage(DevelopmentalStage.INFANT_LATE)
        assert len(transitions) == 1
        assert transitions[0].to_stage == DevelopmentalStage.TODDLER

    @pytest.mark.asyncio
    async def test_add_and_get_trajectory(self, interface):
        """Test adding and retrieving a lifespan trajectory."""
        await interface.initialize()

        trajectory = LifespanTrajectory(
            trajectory_id="test_trajectory",
            domain=LifespanDomain.EMOTIONAL,
            developmental_curve="Improves with age",
            peak_age="60+ years",
            decline_pattern="Minimal decline",
            protective_factors=["Social support"],
        )

        await interface.add_trajectory(trajectory)
        retrieved = await interface.get_trajectory("test_trajectory")

        assert retrieved is not None
        assert retrieved.domain == LifespanDomain.EMOTIONAL

    @pytest.mark.asyncio
    async def test_add_and_get_end_of_life_awareness(self, interface):
        """Test adding and retrieving end of life awareness."""
        await interface.initialize()

        awareness = EndOfLifeAwareness(
            awareness_id="test_awareness",
            stage=DevelopmentalStage.END_OF_LIFE,
            consciousness_changes=["Transitional states"],
            common_experiences=["Deathbed visions"],
            cultural_interpretations=["Spiritual journey"],
        )

        await interface.add_end_of_life_awareness(awareness)
        retrieved = await interface.get_end_of_life_awareness("test_awareness")

        assert retrieved is not None
        assert retrieved.stage == DevelopmentalStage.END_OF_LIFE

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "stage_profiles" in result
        assert result["stage_profiles"] >= 10  # Should have profiles for all stages
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.stage_profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedStageProfiles:
    """Tests for seed stage profile data quality."""

    @pytest.mark.asyncio
    async def test_neonatal_profile_exists(self):
        """Test neonatal stage seed profile exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        neonatal = await interface.get_stage_profile("neonatal")
        assert neonatal is not None
        assert neonatal.stage == DevelopmentalStage.NEONATAL
        assert "birth" in neonatal.age_range.lower() or "28 days" in neonatal.age_range

    @pytest.mark.asyncio
    async def test_toddler_profile_exists(self):
        """Test toddler stage seed profile exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        toddler = await interface.get_stage_profile("toddler")
        assert toddler is not None
        assert toddler.stage == DevelopmentalStage.TODDLER

    @pytest.mark.asyncio
    async def test_adolescence_profile_exists(self):
        """Test adolescence stage seed profile exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        adolescence = await interface.get_stage_profile("adolescence")
        assert adolescence is not None
        assert adolescence.stage == DevelopmentalStage.ADOLESCENCE
        assert DevelopmentalCapacity.ABSTRACT_REASONING in adolescence.emerging_capacities

    @pytest.mark.asyncio
    async def test_end_of_life_profile_exists(self):
        """Test end of life stage seed profile exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        eol = await interface.get_stage_profile("end_of_life")
        assert eol is not None
        assert eol.stage == DevelopmentalStage.END_OF_LIFE

    @pytest.mark.asyncio
    async def test_theory_of_mind_emergence_exists(self):
        """Test theory of mind capacity emergence exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        tom = await interface.get_capacity_emergence("theory_of_mind_emergence")
        assert tom is not None
        assert tom.capacity == DevelopmentalCapacity.THEORY_OF_MIND

    @pytest.mark.asyncio
    async def test_birth_transition_exists(self):
        """Test birth transition seed data exists."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        birth = await interface.get_transition("birth_awakening")
        assert birth is not None
        assert birth.from_stage == DevelopmentalStage.PRENATAL_LATE
        assert birth.to_stage == DevelopmentalStage.NEONATAL


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_profile_has_related_stages_field(self):
        """Test profiles can link to related stages."""
        profile = DevelopmentalStageProfile(
            stage_id="test",
            stage=DevelopmentalStage.INFANT_LATE,
            age_range="6-24 months",
            consciousness_characteristics=["Test"],
            emerging_capacities=[DevelopmentalCapacity.OBJECT_PERMANENCE],
            neural_developments=["Test"],
            key_milestones=["Test"],
            related_stages=["infant_early", "toddler"],
        )
        assert len(profile.related_stages) == 2

    @pytest.mark.asyncio
    async def test_all_13_stages_covered(self):
        """Test that all 13 developmental stages have profiles."""
        interface = DevelopmentalConsciousnessInterface()
        await interface.initialize_all_seed_data()

        # Check that we can get a profile for each stage
        stages_covered = 0
        for stage in DevelopmentalStage:
            profile = await interface.get_profile_by_stage(stage)
            if profile is not None:
                stages_covered += 1

        assert stages_covered >= 13, f"Expected 13 stages covered, got {stages_covered}"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
