#!/usr/bin/env python3
"""
Test suite for Form 38: Neurodivergent Consciousness

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
    "neurodivergent_consciousness_interface",
    INTERFACE_PATH / "neurodivergent_consciousness_interface.py"
)

# Import Form 38 components from loaded module
NeurodivergentType = interface_module.NeurodivergentType
CognitiveStrength = interface_module.CognitiveStrength
SynesthesiaType = interface_module.SynesthesiaType
ProcessingStyle = interface_module.ProcessingStyle
SensoryProfile = interface_module.SensoryProfile
AccommodationType = interface_module.AccommodationType
MaturityLevel = interface_module.MaturityLevel
NeurodivergentProfile = interface_module.NeurodivergentProfile
SynesthesiaProfile = interface_module.SynesthesiaProfile
CognitiveStrengthEvidence = interface_module.CognitiveStrengthEvidence
FirstPersonAccount = interface_module.FirstPersonAccount
AccommodationStrategy = interface_module.AccommodationStrategy
NeurodivergentConsciousnessMaturityState = interface_module.NeurodivergentConsciousnessMaturityState
NeurodivergentConsciousnessInterface = interface_module.NeurodivergentConsciousnessInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestNeurodivergentType:
    """Tests for NeurodivergentType enum."""

    def test_type_count(self):
        """Test that we have the expected number of neurodivergent types."""
        types = list(NeurodivergentType)
        assert len(types) >= 8, f"Expected at least 8 types, got {len(types)}"

    def test_core_types_exist(self):
        """Test core neurodivergent types are defined."""
        core_types = [
            NeurodivergentType.AUTISM_SPECTRUM,
            NeurodivergentType.ADHD,
            NeurodivergentType.SYNESTHESIA,
        ]
        assert len(core_types) == 3

    def test_type_values_are_strings(self):
        """Test that all type values are strings."""
        for nd_type in NeurodivergentType:
            assert isinstance(nd_type.value, str)


class TestCognitiveStrength:
    """Tests for CognitiveStrength enum."""

    def test_strength_count(self):
        """Test that we have expected cognitive strengths."""
        strengths = list(CognitiveStrength)
        assert len(strengths) >= 10, f"Expected at least 10 strengths, got {len(strengths)}"

    def test_key_strengths_exist(self):
        """Test key cognitive strengths are defined."""
        key_strengths = [
            CognitiveStrength.PATTERN_RECOGNITION,
            CognitiveStrength.HYPERFOCUS,
            CognitiveStrength.ATTENTION_TO_DETAIL,
        ]
        assert len(key_strengths) == 3


class TestSynesthesiaType:
    """Tests for SynesthesiaType enum."""

    def test_synesthesia_type_count(self):
        """Test we have expected synesthesia types."""
        types = list(SynesthesiaType)
        assert len(types) >= 8, f"Expected at least 8 synesthesia types, got {len(types)}"

    def test_common_synesthesia_types_exist(self):
        """Test common synesthesia types are defined."""
        common = [
            SynesthesiaType.GRAPHEME_COLOR,
            SynesthesiaType.SOUND_COLOR,
            SynesthesiaType.SPATIAL_SEQUENCE,
        ]
        assert len(common) == 3


class TestProcessingStyle:
    """Tests for ProcessingStyle enum."""

    def test_processing_styles_exist(self):
        """Test processing styles are defined."""
        styles = list(ProcessingStyle)
        assert len(styles) >= 5, f"Expected at least 5 processing styles, got {len(styles)}"


class TestSensoryProfile:
    """Tests for SensoryProfile enum."""

    def test_sensory_profiles_exist(self):
        """Test sensory profiles are defined."""
        profiles = list(SensoryProfile)
        assert len(profiles) >= 4, f"Expected at least 4 sensory profiles, got {len(profiles)}"


class TestAccommodationType:
    """Tests for AccommodationType enum."""

    def test_accommodation_types_exist(self):
        """Test accommodation types are defined."""
        types = list(AccommodationType)
        assert len(types) >= 5, f"Expected at least 5 accommodation types, got {len(types)}"


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestNeurodivergentProfile:
    """Tests for NeurodivergentProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = NeurodivergentProfile(
            profile_id="autism_profile_001",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            cognitive_strengths=[CognitiveStrength.PATTERN_RECOGNITION],
            processing_style=ProcessingStyle.DETAIL_FOCUSED,
            description="Autism spectrum profile",
        )
        assert profile.profile_id == "autism_profile_001"
        assert profile.neurotype == NeurodivergentType.AUTISM_SPECTRUM
        assert len(profile.cognitive_strengths) == 1

    def test_profile_with_sensory_info(self):
        """Test profile with sensory information."""
        profile = NeurodivergentProfile(
            profile_id="test",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            cognitive_strengths=[],
            sensory_profile=SensoryProfile.HYPERSENSITIVE,
        )
        assert profile.sensory_profile == SensoryProfile.HYPERSENSITIVE


class TestSynesthesiaProfile:
    """Tests for SynesthesiaProfile dataclass."""

    def test_synesthesia_creation(self):
        """Test synesthesia profile creation."""
        synesthesia = SynesthesiaProfile(
            synesthesia_id="grapheme_color_001",
            synesthesia_type=SynesthesiaType.GRAPHEME_COLOR,
            inducer="Letters and numbers",
            concurrent="Colors",
            description="Letters trigger color experiences",
        )
        assert synesthesia.synesthesia_type == SynesthesiaType.GRAPHEME_COLOR
        assert "Letters" in synesthesia.inducer


class TestCognitiveStrengthEvidence:
    """Tests for CognitiveStrengthEvidence dataclass."""

    def test_evidence_creation(self):
        """Test cognitive strength evidence creation."""
        evidence = CognitiveStrengthEvidence(
            evidence_id="pattern_evidence_001",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            strength=CognitiveStrength.PATTERN_RECOGNITION,
            description="Superior performance on pattern recognition tasks",
            research_references=["Baron-Cohen et al. (2009)"],
        )
        assert evidence.strength == CognitiveStrength.PATTERN_RECOGNITION
        assert len(evidence.research_references) >= 1


class TestFirstPersonAccount:
    """Tests for FirstPersonAccount dataclass."""

    def test_account_creation(self):
        """Test first-person account creation."""
        account = FirstPersonAccount(
            account_id="account_001",
            neurotype=NeurodivergentType.ADHD,
            experience_type="Time blindness",
            description="Time passes differently for me",
            themes=["Time perception", "Executive function"],
        )
        assert account.neurotype == NeurodivergentType.ADHD
        assert len(account.themes) == 2


class TestAccommodationStrategy:
    """Tests for AccommodationStrategy dataclass."""

    def test_accommodation_creation(self):
        """Test accommodation strategy creation."""
        accommodation = AccommodationStrategy(
            strategy_id="sensory_001",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            accommodation_type=AccommodationType.SENSORY,
            description="Noise-canceling headphones",
            implementation="Provide quiet workspace",
        )
        assert accommodation.accommodation_type == AccommodationType.SENSORY


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestNeurodivergentConsciousnessInterface:
    """Tests for NeurodivergentConsciousnessInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return NeurodivergentConsciousnessInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_profile(self, interface):
        """Test adding and retrieving a profile."""
        await interface.initialize()

        profile = NeurodivergentProfile(
            profile_id="test_adhd",
            neurotype=NeurodivergentType.ADHD,
            cognitive_strengths=[CognitiveStrength.CREATIVITY],
            description="ADHD profile",
        )

        await interface.add_profile(profile)
        retrieved = await interface.get_profile("test_adhd")

        assert retrieved is not None
        assert retrieved.neurotype == NeurodivergentType.ADHD

    @pytest.mark.asyncio
    async def test_get_profile_by_neurotype(self, interface):
        """Test retrieving profile by neurotype."""
        await interface.initialize()

        profile = NeurodivergentProfile(
            profile_id="autism_test",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            cognitive_strengths=[CognitiveStrength.PATTERN_RECOGNITION],
        )
        await interface.add_profile(profile)

        retrieved = await interface.get_profile_by_neurotype(
            NeurodivergentType.AUTISM_SPECTRUM
        )
        assert retrieved is not None
        assert retrieved.profile_id == "autism_test"

    @pytest.mark.asyncio
    async def test_add_and_get_synesthesia(self, interface):
        """Test adding and retrieving synesthesia profile."""
        await interface.initialize()

        synesthesia = SynesthesiaProfile(
            synesthesia_id="test_syn",
            synesthesia_type=SynesthesiaType.GRAPHEME_COLOR,
            inducer="Letters",
            concurrent="Colors",
        )

        await interface.add_synesthesia(synesthesia)
        retrieved = await interface.get_synesthesia("test_syn")

        assert retrieved is not None
        assert retrieved.synesthesia_type == SynesthesiaType.GRAPHEME_COLOR

    @pytest.mark.asyncio
    async def test_add_and_get_first_person_account(self, interface):
        """Test adding and retrieving first-person account."""
        await interface.initialize()

        account = FirstPersonAccount(
            account_id="test_account",
            neurotype=NeurodivergentType.ADHD,
            experience_type="Hyperfocus",
            description="When engaged, time disappears",
        )

        await interface.add_first_person_account(account)
        retrieved = await interface.get_first_person_account("test_account")

        assert retrieved is not None
        assert retrieved.experience_type == "Hyperfocus"

    @pytest.mark.asyncio
    async def test_add_and_get_accommodation(self, interface):
        """Test adding and retrieving accommodation strategy."""
        await interface.initialize()

        accommodation = AccommodationStrategy(
            strategy_id="test_acc",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            accommodation_type=AccommodationType.ENVIRONMENTAL,
            description="Reduced lighting",
        )

        await interface.add_accommodation(accommodation)
        retrieved = await interface.get_accommodation("test_acc")

        assert retrieved is not None
        assert retrieved.accommodation_type == AccommodationType.ENVIRONMENTAL

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
    async def test_autism_profile_exists(self):
        """Test autism seed profile exists."""
        interface = NeurodivergentConsciousnessInterface()
        await interface.initialize_all_seed_data()

        profile = await interface.get_profile_by_neurotype(
            NeurodivergentType.AUTISM_SPECTRUM
        )
        assert profile is not None

    @pytest.mark.asyncio
    async def test_adhd_profile_exists(self):
        """Test ADHD seed profile exists."""
        interface = NeurodivergentConsciousnessInterface()
        await interface.initialize_all_seed_data()

        profile = await interface.get_profile_by_neurotype(
            NeurodivergentType.ADHD
        )
        assert profile is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_profile_has_strengths_list(self):
        """Test profiles can store multiple cognitive strengths."""
        profile = NeurodivergentProfile(
            profile_id="test",
            neurotype=NeurodivergentType.AUTISM_SPECTRUM,
            cognitive_strengths=[
                CognitiveStrength.PATTERN_RECOGNITION,
                CognitiveStrength.ATTENTION_TO_DETAIL,
                CognitiveStrength.SYSTEMATIC_THINKING,
            ],
        )
        assert len(profile.cognitive_strengths) >= 3

    def test_account_has_themes(self):
        """Test first-person accounts can store thematic tags."""
        account = FirstPersonAccount(
            account_id="test",
            neurotype=NeurodivergentType.SYNESTHESIA,
            experience_type="Grapheme-color",
            themes=["Perception", "Creativity", "Memory"],
        )
        assert len(account.themes) >= 3

    def test_neurodiversity_affirming_framework(self):
        """Test that profiles support strength-based framing."""
        profile = NeurodivergentProfile(
            profile_id="test",
            neurotype=NeurodivergentType.ADHD,
            cognitive_strengths=[
                CognitiveStrength.CREATIVITY,
                CognitiveStrength.DIVERGENT_THINKING,
                CognitiveStrength.CRISIS_RESPONSE,
            ],
            description="Neurodiversity-affirming ADHD profile",
        )
        # Profile should have strengths, not just deficits
        assert len(profile.cognitive_strengths) >= 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
