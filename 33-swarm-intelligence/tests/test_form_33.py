#!/usr/bin/env python3
"""
Test suite for Form 33: Swarm & Collective Intelligence

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
    "swarm_intelligence_interface",
    INTERFACE_PATH / "swarm_intelligence_interface.py"
)

# Import Form 33 components from loaded module
SwarmBehaviorType = interface_module.SwarmBehaviorType
CollectiveSystem = interface_module.CollectiveSystem
CoordinationMechanism = interface_module.CoordinationMechanism
SwarmAlgorithm = interface_module.SwarmAlgorithm
EmergentPropertyType = interface_module.EmergentPropertyType
MaturityLevel = interface_module.MaturityLevel
CollectiveSystemProfile = interface_module.CollectiveSystemProfile
SwarmBehaviorObservation = interface_module.SwarmBehaviorObservation
SwarmAlgorithmModel = interface_module.SwarmAlgorithmModel
EmergenceEvent = interface_module.EmergenceEvent
SwarmIntelligenceMaturityState = interface_module.SwarmIntelligenceMaturityState
SwarmIntelligenceInterface = interface_module.SwarmIntelligenceInterface


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestSwarmBehaviorType:
    """Tests for SwarmBehaviorType enum."""

    def test_behavior_count(self):
        """Test that we have the expected number of behavior types."""
        behaviors = list(SwarmBehaviorType)
        assert len(behaviors) >= 10, f"Expected at least 10 behaviors, got {len(behaviors)}"

    def test_resource_behaviors_exist(self):
        """Test resource-related behaviors are defined."""
        resource = [
            SwarmBehaviorType.FORAGING_OPTIMIZATION,
            SwarmBehaviorType.NEST_CONSTRUCTION,
            SwarmBehaviorType.RESOURCE_ALLOCATION,
        ]
        assert len(resource) == 3

    def test_coordination_behaviors_exist(self):
        """Test coordination behaviors are defined."""
        coordination = [
            SwarmBehaviorType.COMMUNICATION_NETWORKS,
            SwarmBehaviorType.SELF_ORGANIZATION,
            SwarmBehaviorType.SYNCHRONIZATION,
        ]
        assert len(coordination) == 3

    def test_behavior_values_are_strings(self):
        """Test that all behavior values are strings."""
        for behavior in SwarmBehaviorType:
            assert isinstance(behavior.value, str)


class TestCollectiveSystem:
    """Tests for CollectiveSystem enum."""

    def test_system_count(self):
        """Test that we have expected collective systems."""
        systems = list(CollectiveSystem)
        assert len(systems) >= 10, f"Expected at least 10 systems, got {len(systems)}"

    def test_insect_systems_exist(self):
        """Test insect society systems are defined."""
        insects = [
            CollectiveSystem.ANT_COLONIES,
            CollectiveSystem.BEE_HIVES,
            CollectiveSystem.TERMITE_MOUNDS,
        ]
        assert len(insects) == 3

    def test_vertebrate_systems_exist(self):
        """Test vertebrate group systems are defined."""
        vertebrates = [
            CollectiveSystem.BIRD_FLOCKS,
            CollectiveSystem.FISH_SCHOOLS,
            CollectiveSystem.WOLF_PACKS,
        ]
        assert len(vertebrates) == 3

    def test_human_systems_exist(self):
        """Test human collective systems are defined."""
        human = [
            CollectiveSystem.HUMAN_CROWDS,
            CollectiveSystem.MARKETS_ECONOMIES,
        ]
        assert len(human) == 2


class TestCoordinationMechanism:
    """Tests for CoordinationMechanism enum."""

    def test_mechanism_count(self):
        """Test that we have expected coordination mechanisms."""
        mechanisms = list(CoordinationMechanism)
        assert len(mechanisms) >= 6, f"Expected at least 6 mechanisms, got {len(mechanisms)}"

    def test_key_mechanisms_exist(self):
        """Test key mechanisms are defined."""
        mechanisms = [
            CoordinationMechanism.STIGMERGY,
            CoordinationMechanism.PHEROMONE_TRAILS,
            CoordinationMechanism.VISUAL_SIGNALS,
            CoordinationMechanism.QUORUM_SENSING,
        ]
        assert len(mechanisms) == 4


class TestSwarmAlgorithm:
    """Tests for SwarmAlgorithm enum."""

    def test_algorithm_count(self):
        """Test that we have expected algorithms."""
        algorithms = list(SwarmAlgorithm)
        assert len(algorithms) >= 4, f"Expected at least 4 algorithms, got {len(algorithms)}"

    def test_key_algorithms_exist(self):
        """Test key algorithms are defined."""
        algorithms = [
            SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
            SwarmAlgorithm.PARTICLE_SWARM,
            SwarmAlgorithm.BOIDS,
            SwarmAlgorithm.CELLULAR_AUTOMATA,
        ]
        assert len(algorithms) == 4


class TestEmergentPropertyType:
    """Tests for EmergentPropertyType enum."""

    def test_property_count(self):
        """Test that we have expected emergent properties."""
        properties = list(EmergentPropertyType)
        assert len(properties) >= 5, f"Expected at least 5 properties, got {len(properties)}"

    def test_key_properties_exist(self):
        """Test key properties are defined."""
        properties = [
            EmergentPropertyType.ROBUSTNESS,
            EmergentPropertyType.SCALABILITY,
            EmergentPropertyType.FLEXIBILITY,
            EmergentPropertyType.SELF_HEALING,
        ]
        assert len(properties) == 4


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestCollectiveSystemProfile:
    """Tests for CollectiveSystemProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = CollectiveSystemProfile(
            system_id="ant_colony",
            name="Ant Colony",
            system_type=CollectiveSystem.ANT_COLONIES,
        )
        assert profile.system_id == "ant_colony"
        assert profile.name == "Ant Colony"
        assert profile.system_type == CollectiveSystem.ANT_COLONIES

    def test_profile_with_behaviors(self):
        """Test profile with behaviors and mechanisms."""
        profile = CollectiveSystemProfile(
            system_id="test_system",
            name="Test System",
            system_type=CollectiveSystem.BEE_HIVES,
            behaviors=[
                SwarmBehaviorType.FORAGING_OPTIMIZATION,
                SwarmBehaviorType.COMMUNICATION_NETWORKS,
            ],
            coordination_mechanisms=[
                CoordinationMechanism.PHEROMONE_TRAILS,
            ],
            emergent_properties=[
                EmergentPropertyType.ROBUSTNESS,
            ],
        )
        assert len(profile.behaviors) == 2
        assert len(profile.coordination_mechanisms) == 1
        assert len(profile.emergent_properties) == 1

    def test_profile_to_embedding_text(self):
        """Test embedding text generation."""
        profile = CollectiveSystemProfile(
            system_id="test",
            name="Test Colony",
            system_type=CollectiveSystem.ANT_COLONIES,
            species_involved=["Atta sp."],
            behaviors=[SwarmBehaviorType.FORAGING_OPTIMIZATION],
            description="A test ant colony",
        )
        text = profile.to_embedding_text()
        assert "Test Colony" in text
        assert "ant_colonies" in text


class TestSwarmBehaviorObservation:
    """Tests for SwarmBehaviorObservation dataclass."""

    def test_observation_creation(self):
        """Test observation creation."""
        observation = SwarmBehaviorObservation(
            observation_id="foraging_001",
            system_id="leafcutter_ants",
            behavior_type=SwarmBehaviorType.FORAGING_OPTIMIZATION,
            description="Efficient trail formation to food source",
            conditions=["Sunny day", "Abundant food"],
            emergent_outcomes=["Optimal path found"],
        )
        assert observation.behavior_type == SwarmBehaviorType.FORAGING_OPTIMIZATION
        assert len(observation.conditions) == 2


class TestSwarmAlgorithmModel:
    """Tests for SwarmAlgorithmModel dataclass."""

    def test_algorithm_creation(self):
        """Test algorithm model creation."""
        algorithm = SwarmAlgorithmModel(
            algorithm_id="aco_test",
            name="Ant Colony Optimization",
            algorithm_type=SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
            inspired_by=[CollectiveSystem.ANT_COLONIES],
            inventor="Marco Dorigo",
            year_introduced=1992,
        )
        assert algorithm.algorithm_type == SwarmAlgorithm.ANT_COLONY_OPTIMIZATION
        assert algorithm.year_introduced == 1992

    def test_algorithm_with_parameters(self):
        """Test algorithm with parameters."""
        algorithm = SwarmAlgorithmModel(
            algorithm_id="pso_test",
            name="Particle Swarm Optimization",
            algorithm_type=SwarmAlgorithm.PARTICLE_SWARM,
            parameters={
                "w": "Inertia weight",
                "c1": "Cognitive coefficient",
                "c2": "Social coefficient",
            },
            applications=["Function optimization", "Neural network training"],
        )
        assert len(algorithm.parameters) == 3
        assert len(algorithm.applications) == 2


class TestEmergenceEvent:
    """Tests for EmergenceEvent dataclass."""

    def test_event_creation(self):
        """Test emergence event creation."""
        event = EmergenceEvent(
            event_id="murmuration_001",
            system_id="starling_murmurations",
            trigger="Predator approach",
            emergent_pattern="Shape-shifting cloud formation",
            emergent_properties=[EmergentPropertyType.ROBUSTNESS],
            scale="Thousands of birds",
        )
        assert event.trigger == "Predator approach"
        assert len(event.emergent_properties) == 1


# ============================================================================
# INTERFACE TESTS
# ============================================================================

class TestSwarmIntelligenceInterface:
    """Tests for SwarmIntelligenceInterface class."""

    @pytest.fixture
    def interface(self):
        """Create interface fixture."""
        return SwarmIntelligenceInterface()

    @pytest.mark.asyncio
    async def test_interface_initialization(self, interface):
        """Test interface initializes correctly."""
        await interface.initialize()
        assert interface._initialized is True

    @pytest.mark.asyncio
    async def test_add_and_get_system_profile(self, interface):
        """Test adding and retrieving a system profile."""
        await interface.initialize()

        profile = CollectiveSystemProfile(
            system_id="test_system",
            name="Test System",
            system_type=CollectiveSystem.ANT_COLONIES,
        )

        await interface.add_system_profile(profile)
        retrieved = await interface.get_system_profile("test_system")

        assert retrieved is not None
        assert retrieved.name == "Test System"

    @pytest.mark.asyncio
    async def test_query_systems_by_type(self, interface):
        """Test querying systems by type."""
        await interface.initialize()

        profile1 = CollectiveSystemProfile(
            system_id="ant_1",
            name="Ant Colony 1",
            system_type=CollectiveSystem.ANT_COLONIES,
        )
        profile2 = CollectiveSystemProfile(
            system_id="bee_1",
            name="Bee Hive 1",
            system_type=CollectiveSystem.BEE_HIVES,
        )

        await interface.add_system_profile(profile1)
        await interface.add_system_profile(profile2)

        ants = await interface.query_systems_by_type(CollectiveSystem.ANT_COLONIES)
        assert len(ants) == 1
        assert ants[0].system_id == "ant_1"

    @pytest.mark.asyncio
    async def test_query_systems_by_behavior(self, interface):
        """Test querying systems by behavior type."""
        await interface.initialize()

        profile = CollectiveSystemProfile(
            system_id="foraging_system",
            name="Foraging System",
            system_type=CollectiveSystem.ANT_COLONIES,
            behaviors=[SwarmBehaviorType.FORAGING_OPTIMIZATION],
        )
        await interface.add_system_profile(profile)

        foragers = await interface.query_systems_by_behavior(
            SwarmBehaviorType.FORAGING_OPTIMIZATION
        )
        assert len(foragers) == 1

    @pytest.mark.asyncio
    async def test_query_systems_by_mechanism(self, interface):
        """Test querying systems by coordination mechanism."""
        await interface.initialize()

        profile = CollectiveSystemProfile(
            system_id="pheromone_system",
            name="Pheromone System",
            system_type=CollectiveSystem.ANT_COLONIES,
            coordination_mechanisms=[CoordinationMechanism.PHEROMONE_TRAILS],
        )
        await interface.add_system_profile(profile)

        pheromone_systems = await interface.query_systems_by_mechanism(
            CoordinationMechanism.PHEROMONE_TRAILS
        )
        assert len(pheromone_systems) == 1

    @pytest.mark.asyncio
    async def test_add_and_get_algorithm(self, interface):
        """Test adding and retrieving an algorithm."""
        await interface.initialize()

        algorithm = SwarmAlgorithmModel(
            algorithm_id="test_algo",
            name="Test Algorithm",
            algorithm_type=SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
        )

        await interface.add_algorithm(algorithm)
        retrieved = await interface.get_algorithm("test_algo")

        assert retrieved is not None
        assert retrieved.name == "Test Algorithm"

    @pytest.mark.asyncio
    async def test_add_and_get_behavior_observation(self, interface):
        """Test adding and retrieving a behavior observation."""
        await interface.initialize()

        observation = SwarmBehaviorObservation(
            observation_id="test_obs",
            system_id="test_system",
            behavior_type=SwarmBehaviorType.FORAGING_OPTIMIZATION,
            description="Test observation",
        )

        await interface.add_behavior_observation(observation)
        retrieved = await interface.get_behavior_observation("test_obs")

        assert retrieved is not None
        assert retrieved.description == "Test observation"

    @pytest.mark.asyncio
    async def test_seed_data_initialization(self, interface):
        """Test seed data initializes correctly."""
        result = await interface.initialize_all_seed_data()

        assert "system_profiles" in result
        assert result["system_profiles"] >= 5  # Should have at least 5 seed profiles
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_maturity_tracking(self, interface):
        """Test maturity state updates."""
        await interface.initialize_all_seed_data()

        maturity = await interface.get_maturity_state()
        assert maturity.system_profile_count > 0
        assert maturity.overall_maturity > 0


# ============================================================================
# SEED DATA TESTS
# ============================================================================

class TestSeedProfiles:
    """Tests for seed profile data quality."""

    @pytest.mark.asyncio
    async def test_leafcutter_profile_exists(self):
        """Test leafcutter ant seed profile exists."""
        interface = SwarmIntelligenceInterface()
        await interface.initialize_all_seed_data()

        leafcutter = await interface.get_system_profile("leafcutter_ants")
        assert leafcutter is not None
        assert "Leafcutter" in leafcutter.name
        assert SwarmBehaviorType.FORAGING_OPTIMIZATION in leafcutter.behaviors

    @pytest.mark.asyncio
    async def test_honeybee_profile_exists(self):
        """Test honeybee seed profile exists."""
        interface = SwarmIntelligenceInterface()
        await interface.initialize_all_seed_data()

        honeybees = await interface.get_system_profile("honeybees")
        assert honeybees is not None
        assert honeybees.system_type == CollectiveSystem.BEE_HIVES

    @pytest.mark.asyncio
    async def test_starling_profile_exists(self):
        """Test starling murmuration seed profile exists."""
        interface = SwarmIntelligenceInterface()
        await interface.initialize_all_seed_data()

        starlings = await interface.get_system_profile("starling_murmurations")
        assert starlings is not None
        assert starlings.system_type == CollectiveSystem.BIRD_FLOCKS

    @pytest.mark.asyncio
    async def test_aco_algorithm_exists(self):
        """Test ACO algorithm seed data exists."""
        interface = SwarmIntelligenceInterface()
        await interface.initialize_all_seed_data()

        aco = await interface.get_algorithm("aco_standard")
        assert aco is not None
        assert aco.algorithm_type == SwarmAlgorithm.ANT_COLONY_OPTIMIZATION


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossFormReadiness:
    """Tests for cross-form integration readiness."""

    def test_profile_has_related_systems_field(self):
        """Test profiles can link to related systems."""
        profile = CollectiveSystemProfile(
            system_id="test",
            name="Test",
            system_type=CollectiveSystem.ANT_COLONIES,
            related_systems=["termite_mounds", "bee_hives"],
        )
        assert len(profile.related_systems) == 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
