#!/usr/bin/env python3
"""
Test suite for Form 36: Contemplative & Meditative States

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
    "contemplative_states_interface",
    INTERFACE_PATH / "contemplative_states_interface.py"
)

# Import Form 36 components from loaded module
ContemplativeState = interface_module.ContemplativeState
ContemplativeTradition = interface_module.ContemplativeTradition
PhenomenologicalQuality = interface_module.PhenomenologicalQuality
NeuralCorrelate = interface_module.NeuralCorrelate
PracticeType = interface_module.PracticeType
MaturityLevel = interface_module.MaturityLevel
ContemplativeStateProfile = interface_module.ContemplativeStateProfile
MeditationSession = interface_module.MeditationSession
TraditionProfile = interface_module.TraditionProfile
NeuralFinding = interface_module.NeuralFinding
ContemplativeStatesMaturityState = interface_module.ContemplativeStatesMaturityState
ContemplativeStatesInterface = interface_module.ContemplativeStatesInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestContemplativeState:
    """Tests for ContemplativeState enum."""

    def test_state_count(self):
        """Test that we have the expected number of contemplative states."""
        states = list(ContemplativeState)
        assert len(states) >= 15, f"Expected at least 15 states, got {len(states)}"

    def test_jhana_states_exist(self):
        """Test jhana states are defined."""
        jhanas = [
            ContemplativeState.ACCESS_CONCENTRATION,
            ContemplativeState.FIRST_JHANA,
            ContemplativeState.SECOND_JHANA,
            ContemplativeState.THIRD_JHANA,
            ContemplativeState.FOURTH_JHANA,
        ]
        assert len(jhanas) == 5

    def test_formless_jhanas_exist(self):
        """Test formless jhana states are defined."""
        formless = [
            ContemplativeState.FIFTH_JHANA_INFINITE_SPACE,
            ContemplativeState.SIXTH_JHANA_INFINITE_CONSCIOUSNESS,
            ContemplativeState.SEVENTH_JHANA_NOTHINGNESS,
            ContemplativeState.EIGHTH_JHANA_NEITHER_PERCEPTION,
        ]
        assert len(formless) == 4

    def test_cross_tradition_states_exist(self):
        """Test cross-tradition states are defined."""
        cross = [
            ContemplativeState.KENSHO,
            ContemplativeState.SATORI,
            ContemplativeState.TURIYA,
            ContemplativeState.FANA,
            ContemplativeState.SAMADHI,
        ]
        assert len(cross) == 5

    def test_state_values_are_strings(self):
        """Test that all state values are strings."""
        for state in ContemplativeState:
            assert isinstance(state.value, str)


class TestContemplativeTradition:
    """Tests for ContemplativeTradition enum."""

    def test_tradition_count(self):
        """Test that we have expected number of traditions."""
        traditions = list(ContemplativeTradition)
        assert len(traditions) >= 10, f"Expected at least 10 traditions, got {len(traditions)}"

    def test_buddhist_traditions_exist(self):
        """Test Buddhist traditions are defined."""
        buddhist = [
            ContemplativeTradition.THERAVADA_BUDDHIST,
            ContemplativeTradition.ZEN_BUDDHIST,
            ContemplativeTradition.TIBETAN_BUDDHIST,
        ]
        assert len(buddhist) == 3

    def test_hindu_traditions_exist(self):
        """Test Hindu traditions are defined."""
        hindu = [
            ContemplativeTradition.HINDU_YOGIC,
            ContemplativeTradition.ADVAITA_VEDANTA,
        ]
        assert len(hindu) == 2


class TestPhenomenologicalQuality:
    """Tests for PhenomenologicalQuality enum."""

    def test_quality_count(self):
        """Test we have expected phenomenological qualities."""
        qualities = list(PhenomenologicalQuality)
        assert len(qualities) >= 8, f"Expected at least 8 qualities, got {len(qualities)}"

    def test_key_qualities_exist(self):
        """Test key phenomenological qualities exist."""
        key_qualities = [
            PhenomenologicalQuality.SPACIOUSNESS,
            PhenomenologicalQuality.LUMINOSITY,
            PhenomenologicalQuality.STILLNESS,
            PhenomenologicalQuality.BLISS,
            PhenomenologicalQuality.EQUANIMITY,
            PhenomenologicalQuality.UNITY,
        ]
        assert len(key_qualities) == 6


class TestNeuralCorrelate:
    """Tests for NeuralCorrelate enum."""

    def test_correlate_count(self):
        """Test we have expected neural correlates."""
        correlates = list(NeuralCorrelate)
        assert len(correlates) >= 5, f"Expected at least 5 correlates, got {len(correlates)}"

    def test_key_correlates_exist(self):
        """Test key neural correlates exist."""
        key_correlates = [
            NeuralCorrelate.GAMMA_WAVES,
            NeuralCorrelate.DMN_DEACTIVATION,
            NeuralCorrelate.ANTERIOR_CINGULATE,
        ]
        assert len(key_correlates) == 3


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestContemplativeStateProfile:
    """Tests for ContemplativeStateProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = ContemplativeStateProfile(
            state_id="test_state",
            state=ContemplativeState.FIRST_JHANA,
            tradition_origin=ContemplativeTradition.THERAVADA_BUDDHIST,
            phenomenology=[PhenomenologicalQuality.BLISS, PhenomenologicalQuality.STILLNESS],
            description="Test jhana state",
        )
        assert profile.state_id == "test_state"
        assert profile.state == ContemplativeState.FIRST_JHANA
        assert profile.tradition_origin == ContemplativeTradition.THERAVADA_BUDDHIST
        assert len(profile.phenomenology) == 2

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = ContemplativeStateProfile(
            state_id="test",
            state=ContemplativeState.KENSHO,
            tradition_origin=ContemplativeTradition.ZEN_BUDDHIST,
            phenomenology=[PhenomenologicalQuality.CLARITY, PhenomenologicalQuality.UNITY],
            description="Initial awakening experience",
        )
        text = profile.to_embedding_text()
        assert "kensho" in text
        assert "zen_buddhist" in text
        assert "awakening" in text


class TestMeditationSession:
    """Tests for MeditationSession dataclass."""

    def test_session_creation(self):
        """Test meditation session creation."""
        session = MeditationSession(
            session_id="session_001",
            practitioner_id="practitioner_001",
            practice_type=PracticeType.CONCENTRATION,
            duration=45,
            states_accessed=[ContemplativeState.ACCESS_CONCENTRATION],
            phenomenological_report="Experienced deep stillness",
        )
        assert session.session_id == "session_001"
        assert session.duration == 45
        assert session.practice_type == PracticeType.CONCENTRATION


class TestTraditionProfile:
    """Tests for TraditionProfile dataclass."""

    def test_tradition_creation(self):
        """Test tradition profile creation."""
        tradition = TraditionProfile(
            tradition_id="test_tradition",
            tradition=ContemplativeTradition.THERAVADA_BUDDHIST,
            origin="Sri Lanka, Myanmar, Thailand",
            key_practices=["Samatha", "Vipassana"],
            notable_teachers=["Mahasi Sayadaw"],
        )
        assert tradition.tradition_id == "test_tradition"
        assert len(tradition.key_practices) == 2


class TestNeuralFinding:
    """Tests for NeuralFinding dataclass."""

    def test_finding_creation(self):
        """Test neural finding creation."""
        finding = NeuralFinding(
            finding_id="test_finding",
            state=ContemplativeState.SAMADHI,
            neural_correlate=NeuralCorrelate.GAMMA_WAVES,
            study_reference="Test Study (2024)",
            methodology="EEG",
            sample_size=20,
            key_result="Increased gamma synchrony",
        )
        assert finding.finding_id == "test_finding"
        assert finding.neural_correlate == NeuralCorrelate.GAMMA_WAVES


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestContemplativeStatesInterface:
    """Tests for ContemplativeStatesInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return ContemplativeStatesInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_state_profile(self, interface):
        """Test adding and retrieving a state profile."""
        await interface.initialize()

        profile = ContemplativeStateProfile(
            state_id="test_jhana",
            state=ContemplativeState.FIRST_JHANA,
            tradition_origin=ContemplativeTradition.THERAVADA_BUDDHIST,
            phenomenology=[PhenomenologicalQuality.BLISS],
            description="First jhana state",
        )

        await interface.add_state_profile(profile)
        retrieved = await interface.get_state_profile("test_jhana")

        assert retrieved is not None
        assert retrieved.state == ContemplativeState.FIRST_JHANA

    @pytest.mark.asyncio
    async def test_query_by_state(self, interface):
        """Test querying profiles by state."""
        await interface.initialize()

        profile1 = ContemplativeStateProfile(
            state_id="jhana_1",
            state=ContemplativeState.FIRST_JHANA,
            tradition_origin=ContemplativeTradition.THERAVADA_BUDDHIST,
            phenomenology=[PhenomenologicalQuality.BLISS],
        )
        profile2 = ContemplativeStateProfile(
            state_id="samadhi_1",
            state=ContemplativeState.SAMADHI,
            tradition_origin=ContemplativeTradition.HINDU_YOGIC,
            phenomenology=[PhenomenologicalQuality.UNITY],
        )

        await interface.add_state_profile(profile1)
        await interface.add_state_profile(profile2)

        jhana_profiles = await interface.query_profiles_by_state(
            ContemplativeState.FIRST_JHANA
        )
        assert len(jhana_profiles) == 1
        assert jhana_profiles[0].state_id == "jhana_1"

    @pytest.mark.asyncio
    async def test_query_by_tradition(self, interface):
        """Test querying profiles by tradition."""
        await interface.initialize()

        profile = ContemplativeStateProfile(
            state_id="zen_kensho",
            state=ContemplativeState.KENSHO,
            tradition_origin=ContemplativeTradition.ZEN_BUDDHIST,
            phenomenology=[PhenomenologicalQuality.CLARITY],
        )
        await interface.add_state_profile(profile)

        zen_profiles = await interface.query_profiles_by_tradition(
            ContemplativeTradition.ZEN_BUDDHIST
        )
        assert len(zen_profiles) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_tradition(self, interface):
        """Test adding and retrieving a tradition."""
        await interface.initialize()

        tradition = TraditionProfile(
            tradition_id="test_zen",
            tradition=ContemplativeTradition.ZEN_BUDDHIST,
            origin="China (Chan), Japan (Zen)",
            key_practices=["Zazen", "Koan study"],
        )

        await interface.add_tradition(tradition)
        retrieved = await interface.get_tradition("test_zen")

        assert retrieved is not None
        assert retrieved.tradition == ContemplativeTradition.ZEN_BUDDHIST

    @pytest.mark.asyncio
    async def test_add_and_get_finding(self, interface):
        """Test adding and retrieving a neural finding."""
        await interface.initialize()

        finding = NeuralFinding(
            finding_id="test_gamma",
            state=ContemplativeState.SAMADHI,
            neural_correlate=NeuralCorrelate.GAMMA_WAVES,
            study_reference="Test (2024)",
            methodology="EEG",
            sample_size=10,
            key_result="Elevated gamma",
        )

        await interface.add_finding(finding)
        retrieved = await interface.get_finding("test_gamma")

        assert retrieved is not None
        assert retrieved.neural_correlate == NeuralCorrelate.GAMMA_WAVES

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "state_profiles" in result
        assert result["state_profiles"] > 0
        assert "traditions" in result
        assert result["traditions"] > 0
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.state_profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedData:
    """Tests for seed data quality."""

    @pytest.mark.asyncio
    async def test_theravada_jhana_profiles_exist(self):
        """Test Theravada jhana seed profiles exist."""
        interface = ContemplativeStatesInterface()
        await interface.initialize_all_seed_data()

        first_jhana = await interface.get_state_profile("first_jhana_theravada")
        assert first_jhana is not None
        assert first_jhana.state == ContemplativeState.FIRST_JHANA

    @pytest.mark.asyncio
    async def test_cross_tradition_profiles_exist(self):
        """Test cross-tradition state profiles exist."""
        interface = ContemplativeStatesInterface()
        await interface.initialize_all_seed_data()

        kensho = await interface.get_state_profile("kensho_zen")
        assert kensho is not None
        assert kensho.tradition_origin == ContemplativeTradition.ZEN_BUDDHIST

    @pytest.mark.asyncio
    async def test_tradition_profiles_comprehensive(self):
        """Test tradition profiles are comprehensive."""
        interface = ContemplativeStatesInterface()
        await interface.initialize_all_seed_data()

        traditions = await interface.get_all_traditions()
        assert len(traditions) >= 5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_neural_correlate_link_to_neuroscience(self):
        """Test neural findings can link to neuroscience forms."""
        finding = NeuralFinding(
            finding_id="test",
            state=ContemplativeState.SAMADHI,
            neural_correlate=NeuralCorrelate.GAMMA_WAVES,
            study_reference="Test Study",
            methodology="EEG",
            sample_size=10,
            key_result="Test result",
            doi="10.1234/test",
        )
        assert finding.doi is not None

    def test_phenomenology_descriptions_comprehensive(self):
        """Test phenomenological qualities are well-defined."""
        profile = ContemplativeStateProfile(
            state_id="test",
            state=ContemplativeState.UNITIVE_STATE,
            tradition_origin=ContemplativeTradition.SECULAR_MINDFULNESS,
            phenomenology=[
                PhenomenologicalQuality.UNITY,
                PhenomenologicalQuality.PEACE,
                PhenomenologicalQuality.TIMELESSNESS,
            ],
            description="Cross-tradition unitive experience",
        )
        assert len(profile.phenomenology) >= 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
