#!/usr/bin/env python3
"""
Test suite for Form 34: Ecological/Planetary Intelligence (Gaia)

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
    "gaia_intelligence_interface",
    INTERFACE_PATH / "gaia_intelligence_interface.py"
)

# Import Form 34 components from loaded module
GaiaSystem = interface_module.GaiaSystem
PlanetaryBoundary = interface_module.PlanetaryBoundary
EcologicalIntelligenceType = interface_module.EcologicalIntelligenceType
FeedbackType = interface_module.FeedbackType
IndigenousEarthTradition = interface_module.IndigenousEarthTradition
GaiaPhilosophy = interface_module.GaiaPhilosophy
MaturityLevel = interface_module.MaturityLevel
BoundaryStatus = interface_module.BoundaryStatus
EarthSystemComponent = interface_module.EarthSystemComponent
PlanetaryBoundaryState = interface_module.PlanetaryBoundaryState
ClimateFeedback = interface_module.ClimateFeedback
IndigenousEarthPerspective = interface_module.IndigenousEarthPerspective
TippingPoint = interface_module.TippingPoint
GaiaIntelligenceMaturityState = interface_module.GaiaIntelligenceMaturityState
GaiaIntelligenceInterface = interface_module.GaiaIntelligenceInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestGaiaSystem:
    """Tests for GaiaSystem enum."""

    def test_system_count(self):
        """Test that we have the expected number of Gaia systems."""
        systems = list(GaiaSystem)
        assert len(systems) >= 10, f"Expected at least 10 systems, got {len(systems)}"

    def test_core_systems_exist(self):
        """Test core Earth systems are defined."""
        core = [
            GaiaSystem.ATMOSPHERE,
            GaiaSystem.OCEAN_CIRCULATION,
            GaiaSystem.CARBON_CYCLE,
            GaiaSystem.WATER_CYCLE,
        ]
        assert len(core) == 4

    def test_feedback_systems_exist(self):
        """Test feedback systems are defined."""
        feedback = [
            GaiaSystem.CLIMATE_FEEDBACK,
            GaiaSystem.ALBEDO_REGULATION,
        ]
        assert len(feedback) == 2

    def test_system_values_are_strings(self):
        """Test that all system values are strings."""
        for system in GaiaSystem:
            assert isinstance(system.value, str)


class TestPlanetaryBoundary:
    """Tests for PlanetaryBoundary enum."""

    def test_boundary_count(self):
        """Test that we have all 9 planetary boundaries."""
        boundaries = list(PlanetaryBoundary)
        assert len(boundaries) == 9, f"Expected 9 boundaries, got {len(boundaries)}"

    def test_all_nine_boundaries_exist(self):
        """Test all nine planetary boundaries are defined."""
        boundaries = [
            PlanetaryBoundary.CLIMATE_CHANGE,
            PlanetaryBoundary.BIOSPHERE_INTEGRITY,
            PlanetaryBoundary.LAND_SYSTEM_CHANGE,
            PlanetaryBoundary.FRESHWATER_USE,
            PlanetaryBoundary.BIOGEOCHEMICAL_FLOWS,
            PlanetaryBoundary.OCEAN_ACIDIFICATION,
            PlanetaryBoundary.ATMOSPHERIC_AEROSOLS,
            PlanetaryBoundary.STRATOSPHERIC_OZONE,
            PlanetaryBoundary.NOVEL_ENTITIES,
        ]
        assert len(boundaries) == 9


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_types_exist(self):
        """Test all feedback types are defined."""
        types = [
            FeedbackType.POSITIVE,
            FeedbackType.NEGATIVE,
            FeedbackType.COMPLEX,
        ]
        assert len(types) == 3


class TestBoundaryStatus:
    """Tests for BoundaryStatus enum."""

    def test_status_types_exist(self):
        """Test all status types are defined."""
        statuses = [
            BoundaryStatus.SAFE,
            BoundaryStatus.INCREASING_RISK,
            BoundaryStatus.HIGH_RISK,
            BoundaryStatus.UNCERTAIN,
        ]
        assert len(statuses) == 4


class TestIndigenousEarthTradition:
    """Tests for IndigenousEarthTradition enum."""

    def test_tradition_count(self):
        """Test that we have diverse traditions."""
        traditions = list(IndigenousEarthTradition)
        assert len(traditions) >= 5, f"Expected at least 5 traditions, got {len(traditions)}"

    def test_key_traditions_exist(self):
        """Test key traditions are defined."""
        traditions = [
            IndigenousEarthTradition.PACHAMAMA_ANDEAN,
            IndigenousEarthTradition.ABORIGINAL_COUNTRY,
            IndigenousEarthTradition.LAKOTA_EARTH,
            IndigenousEarthTradition.MAORI_PAPATUANUKU,
        ]
        assert len(traditions) == 4


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestEarthSystemComponent:
    """Tests for EarthSystemComponent dataclass."""

    def test_component_creation(self):
        """Test basic component creation."""
        component = EarthSystemComponent(
            system_id="atmosphere",
            name="Atmosphere",
            gaia_system=GaiaSystem.ATMOSPHERE,
            description="Earth's gaseous envelope",
        )
        assert component.system_id == "atmosphere"
        assert component.gaia_system == GaiaSystem.ATMOSPHERE

    def test_component_to_embedding_text(self):
        """Test embedding text generation."""
        component = EarthSystemComponent(
            system_id="test",
            name="Test System",
            gaia_system=GaiaSystem.CARBON_CYCLE,
            description="Test description",
            current_state="Active",
        )
        text = component.to_embedding_text()
        assert "Test System" in text
        assert "carbon_cycle" in text


class TestPlanetaryBoundaryState:
    """Tests for PlanetaryBoundaryState dataclass."""

    def test_boundary_creation(self):
        """Test boundary state creation."""
        boundary = PlanetaryBoundaryState(
            boundary_id="climate_change",
            name="Climate Change",
            boundary_type=PlanetaryBoundary.CLIMATE_CHANGE,
            status=BoundaryStatus.HIGH_RISK,
            current_value="420 ppm CO2",
            safe_threshold="350 ppm CO2",
        )
        assert boundary.status == BoundaryStatus.HIGH_RISK
        assert boundary.boundary_type == PlanetaryBoundary.CLIMATE_CHANGE

    def test_boundary_to_embedding_text(self):
        """Test boundary embedding text generation."""
        boundary = PlanetaryBoundaryState(
            boundary_id="test",
            name="Test Boundary",
            boundary_type=PlanetaryBoundary.BIOSPHERE_INTEGRITY,
            status=BoundaryStatus.INCREASING_RISK,
            current_value="Test value",
            safe_threshold="Test threshold",
        )
        text = boundary.to_embedding_text()
        assert "Test Boundary" in text
        assert "increasing_risk" in text


class TestClimateFeedback:
    """Tests for ClimateFeedback dataclass."""

    def test_feedback_creation(self):
        """Test feedback creation."""
        feedback = ClimateFeedback(
            feedback_id="ice_albedo",
            name="Ice-Albedo Feedback",
            feedback_type=FeedbackType.POSITIVE,
            description="Warming melts ice, reducing reflectivity",
            systems_involved=[GaiaSystem.ALBEDO_REGULATION, GaiaSystem.CLIMATE_FEEDBACK],
            tipping_potential=True,
        )
        assert feedback.feedback_type == FeedbackType.POSITIVE
        assert feedback.tipping_potential is True

    def test_feedback_to_embedding_text(self):
        """Test feedback embedding text generation."""
        feedback = ClimateFeedback(
            feedback_id="test",
            name="Test Feedback",
            feedback_type=FeedbackType.NEGATIVE,
            mechanism="Test mechanism",
            timescale="Centuries",
        )
        text = feedback.to_embedding_text()
        assert "Test Feedback" in text
        assert "negative" in text


class TestIndigenousEarthPerspective:
    """Tests for IndigenousEarthPerspective dataclass."""

    def test_perspective_creation(self):
        """Test perspective creation with Form 29 link."""
        perspective = IndigenousEarthPerspective(
            perspective_id="pachamama",
            name="Pachamama",
            tradition=IndigenousEarthTradition.PACHAMAMA_ANDEAN,
            earth_conception="Earth as conscious maternal being",
            human_role="Children with obligation to reciprocate",
            form_29_link="andean_folk_wisdom",
        )
        assert perspective.form_29_link == "andean_folk_wisdom"
        assert perspective.tradition == IndigenousEarthTradition.PACHAMAMA_ANDEAN

    def test_perspective_to_embedding_text(self):
        """Test perspective embedding text generation."""
        perspective = IndigenousEarthPerspective(
            perspective_id="test",
            name="Test Tradition",
            tradition=IndigenousEarthTradition.LAKOTA_EARTH,
            earth_conception="Test conception",
            human_role="Test role",
            ethical_principles=["Principle 1", "Principle 2"],
        )
        text = perspective.to_embedding_text()
        assert "Test Tradition" in text


class TestTippingPoint:
    """Tests for TippingPoint dataclass."""

    def test_tipping_point_creation(self):
        """Test tipping point creation."""
        tipping_point = TippingPoint(
            tipping_id="greenland_ice",
            name="Greenland Ice Sheet Collapse",
            system=GaiaSystem.ALBEDO_REGULATION,
            threshold="~1.5-2.5C global warming",
            reversibility="Essentially irreversible on human timescales",
            consequences=["7m sea level rise", "AMOC disruption"],
        )
        assert tipping_point.system == GaiaSystem.ALBEDO_REGULATION
        assert len(tipping_point.consequences) == 2

    def test_tipping_point_to_embedding_text(self):
        """Test tipping point embedding text generation."""
        tp = TippingPoint(
            tipping_id="test",
            name="Test Tipping Point",
            system=GaiaSystem.OCEAN_CIRCULATION,
            threshold="Test threshold",
            reversibility="Irreversible",
        )
        text = tp.to_embedding_text()
        assert "Test Tipping Point" in text


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestGaiaIntelligenceInterface:
    """Tests for GaiaIntelligenceInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return GaiaIntelligenceInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_earth_system(self, interface):
        """Test adding and retrieving an Earth system."""
        await interface.initialize()

        system = EarthSystemComponent(
            system_id="test_system",
            name="Test System",
            gaia_system=GaiaSystem.CARBON_CYCLE,
        )

        await interface.add_earth_system(system)
        retrieved = await interface.get_earth_system("test_system")

        assert retrieved is not None
        assert retrieved.name == "Test System"

    @pytest.mark.asyncio
    async def test_add_and_get_boundary(self, interface):
        """Test adding and retrieving a planetary boundary."""
        await interface.initialize()

        boundary = PlanetaryBoundaryState(
            boundary_id="test_boundary",
            name="Test Boundary",
            boundary_type=PlanetaryBoundary.CLIMATE_CHANGE,
            status=BoundaryStatus.HIGH_RISK,
        )

        await interface.add_boundary(boundary)
        retrieved = await interface.get_boundary("test_boundary")

        assert retrieved is not None
        assert retrieved.status == BoundaryStatus.HIGH_RISK

    @pytest.mark.asyncio
    async def test_query_boundaries_by_status(self, interface):
        """Test querying boundaries by status."""
        await interface.initialize()

        boundary1 = PlanetaryBoundaryState(
            boundary_id="high_risk_1",
            name="High Risk 1",
            boundary_type=PlanetaryBoundary.CLIMATE_CHANGE,
            status=BoundaryStatus.HIGH_RISK,
        )
        boundary2 = PlanetaryBoundaryState(
            boundary_id="safe_1",
            name="Safe 1",
            boundary_type=PlanetaryBoundary.FRESHWATER_USE,
            status=BoundaryStatus.SAFE,
        )

        await interface.add_boundary(boundary1)
        await interface.add_boundary(boundary2)

        high_risk = await interface.query_boundaries_by_status(BoundaryStatus.HIGH_RISK)
        assert len(high_risk) == 1
        assert high_risk[0].boundary_id == "high_risk_1"

    @pytest.mark.asyncio
    async def test_add_and_get_feedback(self, interface):
        """Test adding and retrieving a climate feedback."""
        await interface.initialize()

        feedback = ClimateFeedback(
            feedback_id="test_feedback",
            name="Test Feedback",
            feedback_type=FeedbackType.POSITIVE,
        )

        await interface.add_feedback(feedback)
        retrieved = await interface.get_feedback("test_feedback")

        assert retrieved is not None
        assert retrieved.feedback_type == FeedbackType.POSITIVE

    @pytest.mark.asyncio
    async def test_query_feedbacks_by_type(self, interface):
        """Test querying feedbacks by type."""
        await interface.initialize()

        feedback = ClimateFeedback(
            feedback_id="positive_1",
            name="Positive Feedback",
            feedback_type=FeedbackType.POSITIVE,
        )
        await interface.add_feedback(feedback)

        positive = await interface.query_feedbacks_by_type(FeedbackType.POSITIVE)
        assert len(positive) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_perspective(self, interface):
        """Test adding and retrieving an indigenous perspective."""
        await interface.initialize()

        perspective = IndigenousEarthPerspective(
            perspective_id="test_perspective",
            name="Test Perspective",
            tradition=IndigenousEarthTradition.PACHAMAMA_ANDEAN,
        )

        await interface.add_perspective(perspective)
        retrieved = await interface.get_perspective("test_perspective")

        assert retrieved is not None
        assert retrieved.tradition == IndigenousEarthTradition.PACHAMAMA_ANDEAN

    @pytest.mark.asyncio
    async def test_add_and_get_tipping_point(self, interface):
        """Test adding and retrieving a tipping point."""
        await interface.initialize()

        tp = TippingPoint(
            tipping_id="test_tp",
            name="Test Tipping Point",
            system=GaiaSystem.OCEAN_CIRCULATION,
        )

        await interface.add_tipping_point(tp)
        retrieved = await interface.get_tipping_point("test_tp")

        assert retrieved is not None
        assert retrieved.system == GaiaSystem.OCEAN_CIRCULATION

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "boundaries" in result
        assert result["boundaries"] >= 5
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.boundary_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedBoundaries:
    """Tests for seed boundary data quality."""

    @pytest.mark.asyncio
    async def test_climate_change_boundary_exists(self):
        """Test climate change boundary seed data exists."""
        interface = GaiaIntelligenceInterface()
        await interface.initialize_all_seed_data()

        climate = await interface.get_boundary("climate_change")
        assert climate is not None
        assert climate.status == BoundaryStatus.HIGH_RISK
        assert climate.boundary_type == PlanetaryBoundary.CLIMATE_CHANGE

    @pytest.mark.asyncio
    async def test_biosphere_integrity_exists(self):
        """Test biosphere integrity boundary exists."""
        interface = GaiaIntelligenceInterface()
        await interface.initialize_all_seed_data()

        biosphere = await interface.get_boundary("biosphere_integrity")
        assert biosphere is not None
        assert biosphere.status == BoundaryStatus.HIGH_RISK

    @pytest.mark.asyncio
    async def test_ice_albedo_feedback_exists(self):
        """Test ice-albedo feedback seed data exists."""
        interface = GaiaIntelligenceInterface()
        await interface.initialize_all_seed_data()

        ice_albedo = await interface.get_feedback("ice_albedo")
        assert ice_albedo is not None
        assert ice_albedo.feedback_type == FeedbackType.POSITIVE

    @pytest.mark.asyncio
    async def test_pachamama_perspective_exists(self):
        """Test Pachamama perspective seed data exists."""
        interface = GaiaIntelligenceInterface()
        await interface.initialize_all_seed_data()

        pachamama = await interface.get_perspective("pachamama")
        assert pachamama is not None
        assert pachamama.tradition == IndigenousEarthTradition.PACHAMAMA_ANDEAN


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_perspective_has_form_29_link(self):
        """Test perspectives can link to Form 29."""
        perspective = IndigenousEarthPerspective(
            perspective_id="test",
            name="Test",
            tradition=IndigenousEarthTradition.LAKOTA_EARTH,
            form_29_link="lakota_folk_wisdom",
        )
        assert perspective.form_29_link is not None

    @pytest.mark.asyncio
    async def test_transgressed_boundaries_query(self):
        """Test query for transgressed boundaries."""
        interface = GaiaIntelligenceInterface()
        await interface.initialize_all_seed_data()

        transgressed = await interface.get_transgressed_boundaries()
        # Should have several transgressed boundaries based on seed data
        assert len(transgressed) >= 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
