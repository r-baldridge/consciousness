#!/usr/bin/env python3
"""
Swarm & Collective Intelligence Interface

Form 33: The comprehensive interface for swarm intelligence, collective behavior,
and emergent systems across biological and artificial domains. This form examines
how simple local interactions produce sophisticated collective behaviors from
ant colonies to financial markets.

Key Principles:
- Emergence from local interactions
- No centralized control
- Stigmergic coordination
- Scale-free collective computation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SwarmBehaviorType(Enum):
    """
    Classification of collective behaviors exhibited by swarm systems.

    These behaviors emerge from local interactions without central control,
    producing sophisticated group-level adaptations.
    """

    # Resource Acquisition & Movement
    FORAGING_OPTIMIZATION = "foraging_optimization"
    NEST_CONSTRUCTION = "nest_construction"
    COLLECTIVE_DEFENSE = "collective_defense"
    MIGRATION_COORDINATION = "migration_coordination"
    RESOURCE_ALLOCATION = "resource_allocation"

    # Communication & Coordination
    COMMUNICATION_NETWORKS = "communication_networks"
    EMERGENT_PATTERNS = "emergent_patterns"
    SELF_ORGANIZATION = "self_organization"

    # Specialized Functions
    THERMOREGULATION = "thermoregulation"
    DIVISION_OF_LABOR = "division_of_labor"
    QUORUM_SENSING = "quorum_sensing"
    COLLECTIVE_TRANSPORT = "collective_transport"
    SYNCHRONIZATION = "synchronization"


class CollectiveSystem(Enum):
    """
    Types of collective systems exhibiting swarm intelligence.

    Spans biological systems from insects to mammals, as well as
    human-created systems exhibiting emergent collective behavior.
    """

    # Insect Societies
    ANT_COLONIES = "ant_colonies"
    BEE_HIVES = "bee_hives"
    TERMITE_MOUNDS = "termite_mounds"

    # Vertebrate Groups
    BIRD_FLOCKS = "bird_flocks"
    FISH_SCHOOLS = "fish_schools"
    WOLF_PACKS = "wolf_packs"

    # Human Systems
    HUMAN_CROWDS = "human_crowds"
    MARKETS_ECONOMIES = "markets_economies"
    INTERNET_NETWORKS = "internet_networks"

    # Microbial Systems
    BACTERIAL_COLONIES = "bacterial_colonies"
    SLIME_MOLDS = "slime_molds"

    # Mass Movement
    LOCUST_SWARMS = "locust_swarms"


class CoordinationMechanism(Enum):
    """
    Mechanisms by which individuals coordinate collective behavior.

    These are the channels through which information flows to enable
    emergent coordination without central control.
    """

    # Indirect Coordination
    STIGMERGY = "stigmergy"
    PHEROMONE_TRAILS = "pheromone_trails"

    # Direct Signaling
    VISUAL_SIGNALS = "visual_signals"
    ACOUSTIC_SIGNALS = "acoustic_signals"
    DIRECT_CONTACT = "direct_contact"

    # Chemical/Density
    QUORUM_SENSING = "quorum_sensing"


class SwarmAlgorithm(Enum):
    """
    Computational algorithms inspired by swarm intelligence.

    These algorithms harness principles observed in natural swarms
    for optimization and problem-solving.
    """

    ANT_COLONY_OPTIMIZATION = "ant_colony_optimization"
    PARTICLE_SWARM = "particle_swarm"
    BOIDS = "boids"
    CELLULAR_AUTOMATA = "cellular_automata"


class EmergentPropertyType(Enum):
    """
    Types of emergent properties observed in collective systems.

    These properties arise from collective dynamics and cannot be
    predicted from individual component analysis alone.
    """

    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    FLEXIBILITY = "flexibility"
    SELF_HEALING = "self_healing"
    DISTRIBUTED_COMPUTATION = "distributed_computation"


class MaturityLevel(Enum):
    """Depth of knowledge coverage."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CollectiveSystemProfile:
    """
    Represents a profile of a natural or artificial collective system.

    Captures the essential characteristics of swarm systems including
    their behaviors, coordination mechanisms, and emergent properties.
    """
    system_id: str
    name: str
    system_type: CollectiveSystem
    species_involved: List[str] = field(default_factory=list)
    population_scale: str = ""  # e.g., "thousands", "millions"
    behaviors: List[SwarmBehaviorType] = field(default_factory=list)
    coordination_mechanisms: List[CoordinationMechanism] = field(default_factory=list)
    emergent_properties: List[EmergentPropertyType] = field(default_factory=list)
    description: str = ""
    key_researchers: List[str] = field(default_factory=list)
    related_systems: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"System: {self.name}",
            f"Type: {self.system_type.value}",
            f"Species: {', '.join(self.species_involved)}",
            f"Behaviors: {', '.join(b.value for b in self.behaviors)}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class SwarmBehaviorObservation:
    """
    Represents an observation of collective behavior in a swarm system.

    Documents specific instances of swarm behaviors including the
    conditions under which they occur and their emergent outcomes.
    """
    observation_id: str
    system_id: str
    behavior_type: SwarmBehaviorType
    description: str
    conditions: List[str] = field(default_factory=list)
    emergent_outcomes: List[str] = field(default_factory=list)
    scale: str = ""  # e.g., "colony-wide", "local group"
    duration: Optional[str] = None
    observer: Optional[str] = None
    related_observations: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SwarmAlgorithmModel:
    """
    Represents a computational swarm algorithm and its characteristics.

    Documents algorithms inspired by collective behavior including
    their parameters, applications, and performance characteristics.
    """
    algorithm_id: str
    name: str
    algorithm_type: SwarmAlgorithm
    inspired_by: List[CollectiveSystem] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    applications: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    inventor: Optional[str] = None
    year_introduced: Optional[int] = None
    key_equations: List[str] = field(default_factory=list)
    related_algorithms: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EmergenceEvent:
    """
    Represents an emergence event in a collective system.

    Documents specific instances where emergent patterns or behaviors
    arise from collective dynamics.
    """
    event_id: str
    system_id: str
    trigger: str
    emergent_pattern: str
    emergent_properties: List[EmergentPropertyType] = field(default_factory=list)
    scale: str = ""  # spatial/population scale
    duration: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    description: str = ""
    related_events: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SwarmIntelligenceMaturityState:
    """Tracks the maturity of swarm intelligence knowledge."""
    overall_maturity: float = 0.0
    system_coverage: Dict[str, float] = field(default_factory=dict)
    system_profile_count: int = 0
    behavior_observation_count: int = 0
    algorithm_count: int = 0
    emergence_event_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class SwarmIntelligenceInterface:
    """
    Main interface for Form 33: Swarm & Collective Intelligence.

    Provides methods for storing, retrieving, and querying collective
    system profiles, behavior observations, swarm algorithms, and
    emergence events across natural and artificial swarm systems.
    """

    FORM_ID = "33-swarm-intelligence"
    FORM_NAME = "Swarm & Collective Intelligence"

    def __init__(self):
        """Initialize the Swarm Intelligence Interface."""
        # Knowledge indexes
        self.system_profile_index: Dict[str, CollectiveSystemProfile] = {}
        self.behavior_observation_index: Dict[str, SwarmBehaviorObservation] = {}
        self.algorithm_index: Dict[str, SwarmAlgorithmModel] = {}
        self.emergence_event_index: Dict[str, EmergenceEvent] = {}

        # Cross-reference indexes
        self.system_type_index: Dict[CollectiveSystem, List[str]] = {}
        self.behavior_type_index: Dict[SwarmBehaviorType, List[str]] = {}
        self.mechanism_index: Dict[CoordinationMechanism, List[str]] = {}

        # Maturity tracking
        self.maturity_state = SwarmIntelligenceMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize system type index
        for system_type in CollectiveSystem:
            self.system_type_index[system_type] = []

        # Initialize behavior type index
        for behavior_type in SwarmBehaviorType:
            self.behavior_type_index[behavior_type] = []

        # Initialize mechanism index
        for mechanism in CoordinationMechanism:
            self.mechanism_index[mechanism] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # SYSTEM PROFILE METHODS
    # ========================================================================

    async def add_system_profile(self, profile: CollectiveSystemProfile) -> None:
        """Add a collective system profile to the index."""
        self.system_profile_index[profile.system_id] = profile

        # Update system type index
        if profile.system_type in self.system_type_index:
            self.system_type_index[profile.system_type].append(profile.system_id)

        # Update behavior type index
        for behavior in profile.behaviors:
            if behavior in self.behavior_type_index:
                self.behavior_type_index[behavior].append(profile.system_id)

        # Update mechanism index
        for mechanism in profile.coordination_mechanisms:
            if mechanism in self.mechanism_index:
                self.mechanism_index[mechanism].append(profile.system_id)

        # Update maturity
        self.maturity_state.system_profile_count = len(self.system_profile_index)
        await self._update_maturity()

    async def get_system_profile(self, system_id: str) -> Optional[CollectiveSystemProfile]:
        """Retrieve a system profile by ID."""
        return self.system_profile_index.get(system_id)

    async def query_systems_by_type(
        self,
        system_type: CollectiveSystem,
        limit: int = 10
    ) -> List[CollectiveSystemProfile]:
        """Query systems by type."""
        system_ids = self.system_type_index.get(system_type, [])[:limit]
        return [
            self.system_profile_index[sid]
            for sid in system_ids
            if sid in self.system_profile_index
        ]

    async def query_systems_by_behavior(
        self,
        behavior_type: SwarmBehaviorType,
        limit: int = 10
    ) -> List[CollectiveSystemProfile]:
        """Query systems by behavior type."""
        system_ids = self.behavior_type_index.get(behavior_type, [])[:limit]
        return [
            self.system_profile_index[sid]
            for sid in system_ids
            if sid in self.system_profile_index
        ]

    async def query_systems_by_mechanism(
        self,
        mechanism: CoordinationMechanism,
        limit: int = 10
    ) -> List[CollectiveSystemProfile]:
        """Query systems by coordination mechanism."""
        system_ids = self.mechanism_index.get(mechanism, [])[:limit]
        return [
            self.system_profile_index[sid]
            for sid in system_ids
            if sid in self.system_profile_index
        ]

    # ========================================================================
    # BEHAVIOR OBSERVATION METHODS
    # ========================================================================

    async def add_behavior_observation(self, observation: SwarmBehaviorObservation) -> None:
        """Add a behavior observation to the index."""
        self.behavior_observation_index[observation.observation_id] = observation

        # Update behavior type index
        if observation.behavior_type in self.behavior_type_index:
            self.behavior_type_index[observation.behavior_type].append(
                observation.observation_id
            )

        # Update maturity
        self.maturity_state.behavior_observation_count = len(self.behavior_observation_index)
        await self._update_maturity()

    async def get_behavior_observation(
        self, observation_id: str
    ) -> Optional[SwarmBehaviorObservation]:
        """Retrieve a behavior observation by ID."""
        return self.behavior_observation_index.get(observation_id)

    async def query_observations_by_system(
        self,
        system_id: str,
        limit: int = 10
    ) -> List[SwarmBehaviorObservation]:
        """Query observations for a specific system."""
        results = []
        for observation in self.behavior_observation_index.values():
            if observation.system_id == system_id:
                results.append(observation)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # ALGORITHM METHODS
    # ========================================================================

    async def add_algorithm(self, algorithm: SwarmAlgorithmModel) -> None:
        """Add a swarm algorithm to the index."""
        self.algorithm_index[algorithm.algorithm_id] = algorithm

        # Update maturity
        self.maturity_state.algorithm_count = len(self.algorithm_index)
        await self._update_maturity()

    async def get_algorithm(self, algorithm_id: str) -> Optional[SwarmAlgorithmModel]:
        """Retrieve an algorithm by ID."""
        return self.algorithm_index.get(algorithm_id)

    async def query_algorithms_by_type(
        self,
        algorithm_type: SwarmAlgorithm,
        limit: int = 10
    ) -> List[SwarmAlgorithmModel]:
        """Query algorithms by type."""
        results = []
        for algorithm in self.algorithm_index.values():
            if algorithm.algorithm_type == algorithm_type:
                results.append(algorithm)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # EMERGENCE EVENT METHODS
    # ========================================================================

    async def add_emergence_event(self, event: EmergenceEvent) -> None:
        """Add an emergence event to the index."""
        self.emergence_event_index[event.event_id] = event

        # Update maturity
        self.maturity_state.emergence_event_count = len(self.emergence_event_index)
        await self._update_maturity()

    async def get_emergence_event(self, event_id: str) -> Optional[EmergenceEvent]:
        """Retrieve an emergence event by ID."""
        return self.emergence_event_index.get(event_id)

    async def query_emergence_by_system(
        self,
        system_id: str,
        limit: int = 10
    ) -> List[EmergenceEvent]:
        """Query emergence events for a specific system."""
        results = []
        for event in self.emergence_event_index.values():
            if event.system_id == system_id:
                results.append(event)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.system_profile_count +
            self.maturity_state.behavior_observation_count +
            self.maturity_state.algorithm_count +
            self.maturity_state.emergence_event_count
        )

        # Simple maturity calculation
        target_items = 200  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update system coverage
        for system_type in CollectiveSystem:
            count = len(self.system_type_index.get(system_type, []))
            target_per_type = 10
            self.maturity_state.system_coverage[system_type.value] = min(
                1.0, count / target_per_type
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> SwarmIntelligenceMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_system_profiles(self) -> List[Dict[str, Any]]:
        """Return seed system profiles for initialization."""
        return [
            # Leafcutter Ants (Agriculture)
            {
                "system_id": "leafcutter_ants",
                "name": "Leafcutter Ant Colonies",
                "system_type": CollectiveSystem.ANT_COLONIES,
                "species_involved": ["Atta", "Acromyrmex"],
                "population_scale": "millions (up to 8 million individuals)",
                "behaviors": [
                    SwarmBehaviorType.FORAGING_OPTIMIZATION,
                    SwarmBehaviorType.DIVISION_OF_LABOR,
                    SwarmBehaviorType.NEST_CONSTRUCTION,
                    SwarmBehaviorType.COMMUNICATION_NETWORKS,
                    SwarmBehaviorType.RESOURCE_ALLOCATION,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.PHEROMONE_TRAILS,
                    CoordinationMechanism.DIRECT_CONTACT,
                    CoordinationMechanism.STIGMERGY,
                ],
                "emergent_properties": [
                    EmergentPropertyType.ROBUSTNESS,
                    EmergentPropertyType.SCALABILITY,
                    EmergentPropertyType.DISTRIBUTED_COMPUTATION,
                ],
                "description": (
                    "Leafcutter ants practice sophisticated fungus agriculture, "
                    "cultivating fungal gardens from harvested leaf material. "
                    "Their colonies exhibit complex division of labor with distinct "
                    "castes (minims, minors, mediae, majors) optimized for different "
                    "tasks from leaf cutting to fungus tending. This represents one "
                    "of the most complex non-human agricultural systems on Earth."
                ),
                "key_researchers": ["Bert Holldobler", "E.O. Wilson"],
            },

            # Honeybees (Waggle Dance & Democracy)
            {
                "system_id": "honeybees",
                "name": "Honeybee Colonies",
                "system_type": CollectiveSystem.BEE_HIVES,
                "species_involved": ["Apis mellifera"],
                "population_scale": "tens of thousands (20,000-80,000)",
                "behaviors": [
                    SwarmBehaviorType.FORAGING_OPTIMIZATION,
                    SwarmBehaviorType.COMMUNICATION_NETWORKS,
                    SwarmBehaviorType.THERMOREGULATION,
                    SwarmBehaviorType.DIVISION_OF_LABOR,
                    SwarmBehaviorType.NEST_CONSTRUCTION,
                    SwarmBehaviorType.COLLECTIVE_DEFENSE,
                    SwarmBehaviorType.QUORUM_SENSING,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.VISUAL_SIGNALS,
                    CoordinationMechanism.ACOUSTIC_SIGNALS,
                    CoordinationMechanism.PHEROMONE_TRAILS,
                    CoordinationMechanism.DIRECT_CONTACT,
                    CoordinationMechanism.QUORUM_SENSING,
                ],
                "emergent_properties": [
                    EmergentPropertyType.DISTRIBUTED_COMPUTATION,
                    EmergentPropertyType.ROBUSTNESS,
                    EmergentPropertyType.FLEXIBILITY,
                ],
                "description": (
                    "Honeybees exhibit remarkable collective intelligence through "
                    "the waggle dance, which communicates direction, distance, and "
                    "quality of food sources. During swarming, scout bees engage in "
                    "democratic nest-site selection through competitive dancing, "
                    "achieving consensus without any queen involvement. Thomas Seeley's "
                    "research shows five principles of their collective intelligence: "
                    "diversity of knowledge, open information sharing, independent "
                    "evaluation, unbiased aggregation, and non-dominant leadership."
                ),
                "key_researchers": ["Thomas D. Seeley", "Karl von Frisch"],
            },

            # Army Ants (Bridge Formation)
            {
                "system_id": "army_ants",
                "name": "Army Ant Swarms",
                "system_type": CollectiveSystem.ANT_COLONIES,
                "species_involved": ["Eciton burchellii", "Dorylus"],
                "population_scale": "hundreds of thousands to millions",
                "behaviors": [
                    SwarmBehaviorType.FORAGING_OPTIMIZATION,
                    SwarmBehaviorType.COLLECTIVE_TRANSPORT,
                    SwarmBehaviorType.MIGRATION_COORDINATION,
                    SwarmBehaviorType.EMERGENT_PATTERNS,
                    SwarmBehaviorType.SELF_ORGANIZATION,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.PHEROMONE_TRAILS,
                    CoordinationMechanism.DIRECT_CONTACT,
                    CoordinationMechanism.STIGMERGY,
                ],
                "emergent_properties": [
                    EmergentPropertyType.SELF_HEALING,
                    EmergentPropertyType.FLEXIBILITY,
                    EmergentPropertyType.ROBUSTNESS,
                ],
                "description": (
                    "Army ants are nomadic predators that form massive swarm raids "
                    "and create living structures from their own bodies. They construct "
                    "living bridges by linking together to span gaps, with the structure "
                    "dynamically self-repairing when damaged. Workers are functionally "
                    "blind and coordinate entirely through chemical and tactile signals. "
                    "Bivouacs (temporary nests) are formed entirely from interlocked "
                    "worker bodies, demonstrating extraordinary collective construction."
                ),
                "key_researchers": ["Deborah M. Gordon", "Iain Couzin"],
            },

            # Starling Murmurations
            {
                "system_id": "starling_murmurations",
                "name": "Starling Murmurations",
                "system_type": CollectiveSystem.BIRD_FLOCKS,
                "species_involved": ["Sturnus vulgaris"],
                "population_scale": "thousands to hundreds of thousands",
                "behaviors": [
                    SwarmBehaviorType.EMERGENT_PATTERNS,
                    SwarmBehaviorType.COLLECTIVE_DEFENSE,
                    SwarmBehaviorType.SYNCHRONIZATION,
                    SwarmBehaviorType.SELF_ORGANIZATION,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.VISUAL_SIGNALS,
                ],
                "emergent_properties": [
                    EmergentPropertyType.SCALABILITY,
                    EmergentPropertyType.ROBUSTNESS,
                    EmergentPropertyType.DISTRIBUTED_COMPUTATION,
                ],
                "description": (
                    "Starling murmurations are among the most spectacular examples "
                    "of emergent collective behavior. Up to 750,000 birds move as a "
                    "unified entity, creating shape-shifting clouds, teardrops, and "
                    "figure-eights. Each bird tracks approximately 7 nearest neighbors "
                    "using simple rules (separation, alignment, cohesion). Scale-free "
                    "correlations enable information to propagate uncorrupted across "
                    "the entire flock - a 'game of telephone that always works.' "
                    "No leader or plan exists; all complexity emerges from local "
                    "interactions."
                ),
                "key_researchers": ["Iain Couzin", "Andrea Cavagna"],
            },

            # Sardine Bait Balls
            {
                "system_id": "sardine_bait_balls",
                "name": "Sardine Bait Ball Formation",
                "system_type": CollectiveSystem.FISH_SCHOOLS,
                "species_involved": ["Sardinops sagax", "Sardina pilchardus"],
                "population_scale": "thousands to millions",
                "behaviors": [
                    SwarmBehaviorType.COLLECTIVE_DEFENSE,
                    SwarmBehaviorType.EMERGENT_PATTERNS,
                    SwarmBehaviorType.SYNCHRONIZATION,
                    SwarmBehaviorType.SELF_ORGANIZATION,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.VISUAL_SIGNALS,
                    CoordinationMechanism.DIRECT_CONTACT,
                ],
                "emergent_properties": [
                    EmergentPropertyType.SELF_HEALING,
                    EmergentPropertyType.FLEXIBILITY,
                    EmergentPropertyType.ROBUSTNESS,
                ],
                "description": (
                    "Sardine bait balls form as a defensive response to predator "
                    "attack, with fish compacting into tight spherical formations. "
                    "The lateral line system enables fish to detect pressure changes "
                    "from neighbors' movements, allowing real-time coordination. "
                    "Startle cascades propagate faster than predator approach speed, "
                    "and the school dynamically changes shape in response to threats. "
                    "The confusion effect (synchronized movements) and dilution effect "
                    "(larger groups) reduce individual capture probability."
                ),
                "key_researchers": ["Iain Couzin", "Julia Parrish"],
            },

            # Termite Mound Construction
            {
                "system_id": "termite_mounds",
                "name": "Termite Mound Architecture",
                "system_type": CollectiveSystem.TERMITE_MOUNDS,
                "species_involved": ["Macrotermes bellicosus", "Nasutitermes"],
                "population_scale": "millions",
                "behaviors": [
                    SwarmBehaviorType.NEST_CONSTRUCTION,
                    SwarmBehaviorType.THERMOREGULATION,
                    SwarmBehaviorType.DIVISION_OF_LABOR,
                    SwarmBehaviorType.SELF_ORGANIZATION,
                ],
                "coordination_mechanisms": [
                    CoordinationMechanism.STIGMERGY,
                    CoordinationMechanism.PHEROMONE_TRAILS,
                    CoordinationMechanism.DIRECT_CONTACT,
                ],
                "emergent_properties": [
                    EmergentPropertyType.DISTRIBUTED_COMPUTATION,
                    EmergentPropertyType.ROBUSTNESS,
                    EmergentPropertyType.SCALABILITY,
                ],
                "description": (
                    "Termite mounds are architectural marvels built through stigmergy - "
                    "indirect coordination where traces left by one individual stimulate "
                    "subsequent actions by others. Pierre-Paul Grasse observed termites "
                    "depositing mud that then stimulated others to add more, creating "
                    "cathedral-like structures with interlocking arches. These mounds "
                    "maintain internal temperature around 30C regardless of external "
                    "conditions through sophisticated ventilation systems. The Eastgate "
                    "Centre in Harare uses termite-inspired design for 90% energy savings."
                ),
                "key_researchers": ["Pierre-Paul Grasse", "J. Scott Turner"],
            },
        ]

    def _get_seed_algorithms(self) -> List[Dict[str, Any]]:
        """Return seed swarm algorithms for initialization."""
        return [
            {
                "algorithm_id": "aco_standard",
                "name": "Ant Colony Optimization",
                "algorithm_type": SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
                "inspired_by": [CollectiveSystem.ANT_COLONIES],
                "parameters": {
                    "alpha": "Pheromone importance (typical: 1.0)",
                    "beta": "Heuristic importance (typical: 2.0)",
                    "rho": "Evaporation rate (typical: 0.1)",
                    "Q": "Pheromone deposit constant",
                },
                "applications": [
                    "Traveling salesman problem",
                    "Vehicle routing",
                    "Network routing",
                    "Scheduling",
                    "Graph coloring",
                ],
                "description": (
                    "ACO was invented by Marco Dorigo in 1992, inspired by ant foraging "
                    "behavior. Artificial ants construct solutions by moving through a "
                    "graph, depositing pheromone on edges proportional to solution quality. "
                    "Future ants probabilistically prefer high-pheromone edges. Evaporation "
                    "prevents convergence to local optima."
                ),
                "inventor": "Marco Dorigo",
                "year_introduced": 1992,
                "key_equations": [
                    "tau_ij(t+1) = (1-rho)*tau_ij(t) + sum_k(delta_tau_ij_k)",
                    "p_ij = [tau_ij^alpha * eta_ij^beta] / sum_l([tau_il^alpha * eta_il^beta])",
                ],
            },
            {
                "algorithm_id": "pso_standard",
                "name": "Particle Swarm Optimization",
                "algorithm_type": SwarmAlgorithm.PARTICLE_SWARM,
                "inspired_by": [CollectiveSystem.BIRD_FLOCKS, CollectiveSystem.FISH_SCHOOLS],
                "parameters": {
                    "w": "Inertia weight (typical: 0.7)",
                    "c1": "Cognitive coefficient (typical: 1.5)",
                    "c2": "Social coefficient (typical: 1.5)",
                },
                "applications": [
                    "Function optimization",
                    "Neural network training",
                    "Feature selection",
                    "Engineering design",
                ],
                "description": (
                    "PSO was proposed by Kennedy and Eberhart in 1995, inspired by "
                    "bird flocking and fish schooling. Particles move through search "
                    "space influenced by their personal best position and the swarm's "
                    "global best. Computationally inexpensive, requiring only primitive "
                    "mathematical operators."
                ),
                "inventor": "James Kennedy & Russell Eberhart",
                "year_introduced": 1995,
                "key_equations": [
                    "v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)",
                    "x_i(t+1) = x_i(t) + v_i(t+1)",
                ],
            },
            {
                "algorithm_id": "boids",
                "name": "Boids Flocking Algorithm",
                "algorithm_type": SwarmAlgorithm.BOIDS,
                "inspired_by": [CollectiveSystem.BIRD_FLOCKS, CollectiveSystem.FISH_SCHOOLS],
                "parameters": {
                    "separation_weight": "Avoidance force weight",
                    "alignment_weight": "Velocity matching weight",
                    "cohesion_weight": "Centering force weight",
                    "neighbor_radius": "Perception distance",
                },
                "applications": [
                    "Computer graphics (film, games)",
                    "Swarm robotics",
                    "Crowd simulation",
                    "Traffic modeling",
                ],
                "description": (
                    "Craig Reynolds created Boids in 1986 to simulate flocking through "
                    "three simple steering behaviors: Separation (avoid crowding), "
                    "Alignment (match heading with neighbors), and Cohesion (move toward "
                    "center of mass). Each boid reacts only to neighbors within a local "
                    "neighborhood. Used in films including Batman Returns (1992)."
                ),
                "inventor": "Craig Reynolds",
                "year_introduced": 1986,
                "key_equations": [
                    "separation = sum(self - neighbor) / count",
                    "alignment = avg_velocity - self_velocity",
                    "cohesion = center_of_mass - self_position",
                ],
            },
            {
                "algorithm_id": "cellular_automata",
                "name": "Cellular Automata",
                "algorithm_type": SwarmAlgorithm.CELLULAR_AUTOMATA,
                "inspired_by": [],
                "parameters": {
                    "grid_size": "Dimensions of cell grid",
                    "neighborhood": "Moore or von Neumann",
                    "rule_set": "State transition function",
                },
                "applications": [
                    "Pattern formation",
                    "Traffic simulation",
                    "Biological modeling",
                    "Cryptography",
                ],
                "description": (
                    "Cellular automata model systems as grids where each cell updates "
                    "based on neighbor states. Conway's Game of Life demonstrates how "
                    "four simple rules produce Turing-complete computation. Emergent "
                    "patterns include gliders, oscillators, and spaceships. The model "
                    "shows how complex behavior emerges from minimal local rules."
                ),
                "inventor": "John von Neumann & Stanislaw Ulam",
                "year_introduced": 1940,
                "key_equations": [
                    "state_i(t+1) = f(state_neighbors(t))",
                ],
            },
        ]

    async def initialize_seed_system_profiles(self) -> int:
        """Initialize with seed system profiles."""
        seed_profiles = self._get_seed_system_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = CollectiveSystemProfile(
                system_id=profile_data["system_id"],
                name=profile_data["name"],
                system_type=profile_data["system_type"],
                species_involved=profile_data.get("species_involved", []),
                population_scale=profile_data.get("population_scale", ""),
                behaviors=profile_data.get("behaviors", []),
                coordination_mechanisms=profile_data.get("coordination_mechanisms", []),
                emergent_properties=profile_data.get("emergent_properties", []),
                description=profile_data.get("description", ""),
                key_researchers=profile_data.get("key_researchers", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_system_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed system profiles")
        return count

    async def initialize_seed_algorithms(self) -> int:
        """Initialize with seed algorithms."""
        seed_algorithms = self._get_seed_algorithms()
        count = 0

        for algo_data in seed_algorithms:
            algorithm = SwarmAlgorithmModel(
                algorithm_id=algo_data["algorithm_id"],
                name=algo_data["name"],
                algorithm_type=algo_data["algorithm_type"],
                inspired_by=algo_data.get("inspired_by", []),
                parameters=algo_data.get("parameters", {}),
                applications=algo_data.get("applications", []),
                description=algo_data.get("description", ""),
                inventor=algo_data.get("inventor"),
                year_introduced=algo_data.get("year_introduced"),
                key_equations=algo_data.get("key_equations", []),
            )
            await self.add_algorithm(algorithm)
            count += 1

        logger.info(f"Initialized {count} seed algorithms")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_system_profiles()
        algorithms_count = await self.initialize_seed_algorithms()

        return {
            "system_profiles": profiles_count,
            "algorithms": algorithms_count,
            "total": profiles_count + algorithms_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "SwarmBehaviorType",
    "CollectiveSystem",
    "CoordinationMechanism",
    "SwarmAlgorithm",
    "EmergentPropertyType",
    "MaturityLevel",
    # Dataclasses
    "CollectiveSystemProfile",
    "SwarmBehaviorObservation",
    "SwarmAlgorithmModel",
    "EmergenceEvent",
    "SwarmIntelligenceMaturityState",
    # Interface
    "SwarmIntelligenceInterface",
]
