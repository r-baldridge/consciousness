# Swarm Intelligence Data Structures
**Form 33: Swarm Intelligence**
**Task B3: Data Structures Specification**
**Date:** January 2026

## Overview

This document defines the comprehensive data structures required for implementing swarm intelligence and collective behavior systems. These structures support agent modeling, swarm simulation, stigmergic communication, emergence detection, and optimization algorithm execution across biological and artificial collective systems.

## Core Data Models

### 1. Agent and Swarm Entity Structures

#### 1.1 Swarm Agent Structure

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


@dataclass
class SwarmAgent:
    """Individual agent within a swarm collective"""

    # Core Identity
    agent_id: str                          # Unique agent identifier
    swarm_id: str                          # Parent swarm identifier
    agent_type: 'AgentType'                # Agent classification
    creation_timestamp: datetime

    # Spatial State
    position: np.ndarray                   # Position vector (2D or 3D)
    velocity: np.ndarray                   # Velocity vector
    heading: float                         # Heading angle in radians
    acceleration: np.ndarray               # Current acceleration

    # Behavioral State
    current_behavior: 'AgentBehavior'      # Active behavior mode
    behavioral_rules: List['BehavioralRule']  # Rule set governing actions
    rule_weights: Dict[str, float]         # Weights for each rule
    internal_state: Dict[str, Any]         # Internal state variables

    # Sensory Capabilities
    perception_radius: float               # Maximum perception distance
    perception_angle: float                # Field of view (radians)
    sensory_modalities: List['SensoryModality']  # Available senses
    detected_neighbors: List[str]          # Currently perceived neighbor IDs

    # Action Capabilities
    max_speed: float                       # Maximum movement speed
    max_acceleration: float                # Maximum acceleration
    max_turn_rate: float                   # Maximum angular velocity
    signal_emission_capable: bool          # Can emit signals

    # Energy and Resources
    energy_level: float                    # Current energy (0.0-1.0)
    energy_consumption_rate: float         # Energy drain per tick
    carrying_capacity: float               # Resource carrying capacity
    carried_resources: float               # Currently carried resources

    # Communication State
    pheromone_emission_rate: float         # Chemical signal emission rate
    signal_strength: float                 # Active signal strength
    received_signals: List['Signal']       # Buffered received signals
    communication_history: List['CommEvent']  # Recent communications

    # Performance Metrics
    distance_traveled: float               # Total distance traveled
    tasks_completed: int                   # Number of tasks completed
    interactions_count: int                # Total neighbor interactions
    lifetime_ticks: int                    # Ticks since creation

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = generate_unique_agent_id()
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)


class AgentType(Enum):
    """Classification of swarm agent types"""
    SIMPLE_REACTIVE = "simple_reactive"        # Stimulus-response only
    RULE_BASED = "rule_based"                  # Fixed rule set
    ADAPTIVE = "adaptive"                      # Learns and adapts rules
    NEURAL = "neural"                          # Neural network controller
    HYBRID = "hybrid"                          # Mixed control architecture
    ANT_WORKER = "ant_worker"                  # Ant colony worker
    ANT_SCOUT = "ant_scout"                    # Ant colony scout
    BEE_FORAGER = "bee_forager"                # Honeybee forager
    BEE_SCOUT = "bee_scout"                    # Honeybee scout
    BIRD_FLOCKING = "bird_flocking"            # Flocking bird agent
    FISH_SCHOOLING = "fish_schooling"          # Schooling fish agent


class AgentBehavior(Enum):
    """Current behavioral mode of an agent"""
    IDLE = "idle"
    FORAGING = "foraging"
    RETURNING = "returning"
    FOLLOWING_TRAIL = "following_trail"
    EXPLORING = "exploring"
    RECRUITING = "recruiting"
    DEFENDING = "defending"
    FLOCKING = "flocking"
    SWARMING = "swarming"
    MIGRATING = "migrating"
    RESTING = "resting"
    DANCING = "dancing"                        # Waggle dance for bees
```

#### 1.2 Swarm Collective Structure

```python
@dataclass
class SwarmCollective:
    """Representation of an entire swarm collective"""

    # Core Identity
    swarm_id: str                          # Unique swarm identifier
    swarm_name: str                        # Human-readable name
    swarm_type: 'SwarmType'                # Type of swarm system
    creation_timestamp: datetime

    # Population
    agents: Dict[str, SwarmAgent]          # All agents by ID
    population_size: int                   # Current population count
    target_population: int                 # Desired population size
    population_limits: 'PopulationLimits'  # Min/max constraints

    # Environment
    environment: 'SwarmEnvironment'        # Environmental context
    boundary_condition: 'BoundaryCondition'  # Edge behavior
    spatial_extent: Tuple[float, ...]      # Environment dimensions

    # Collective State
    center_of_mass: np.ndarray             # Swarm centroid
    dispersion: float                      # Spatial spread measure
    polarization: float                    # Alignment measure (0.0-1.0)
    rotation: float                        # Angular momentum
    density: float                         # Agent density

    # Behavioral State
    dominant_behavior: AgentBehavior       # Most common behavior
    behavior_distribution: Dict[str, float]  # Proportion per behavior
    coordination_mechanism: 'CoordinationMechanism'
    synchronization_level: float           # 0.0-1.0

    # Emergence Metrics
    emergence_state: 'EmergenceState'      # Current emergence assessment
    order_parameter: float                 # Global order (0.0-1.0)
    complexity_index: float                # Kolmogorov-like complexity
    integrated_information: float          # Phi estimate

    # Performance
    collective_fitness: float              # Swarm-level fitness
    task_completion_rate: float            # Tasks completed per tick
    resource_efficiency: float             # Resources gathered / energy spent
    adaptation_rate: float                 # Speed of behavioral adaptation

    # History
    state_history: List['SwarmSnapshot']   # Historical state snapshots
    emergence_events: List['EmergenceEvent']  # Detected emergence events
    phase_transitions: List['PhaseTransition']  # Phase transition records


class SwarmType(Enum):
    """Classification of swarm system types"""
    ANT_COLONY = "ant_colony"
    BEE_HIVE = "bee_hive"
    BIRD_FLOCK = "bird_flock"
    FISH_SCHOOL = "fish_school"
    TERMITE_COLONY = "termite_colony"
    BACTERIAL_COLONY = "bacterial_colony"
    SLIME_MOLD = "slime_mold"
    HUMAN_CROWD = "human_crowd"
    PARTICLE_SWARM = "particle_swarm"
    BOIDS_FLOCK = "boids_flock"
    CELLULAR_AUTOMATA = "cellular_automata"
    CUSTOM = "custom"


@dataclass
class PopulationLimits:
    """Population size constraints"""
    minimum: int = 2
    maximum: int = 10000
    optimal_range: Tuple[int, int] = (50, 5000)
    growth_rate: float = 0.0               # Agents per tick
    death_rate: float = 0.0                # Agent removal rate
```

### 2. Behavioral Rule Structures

#### 2.1 Behavioral Rule Definition

```python
@dataclass
class BehavioralRule:
    """A single behavioral rule for agent decision-making"""

    rule_id: str                           # Unique rule identifier
    rule_name: str                         # Human-readable name
    rule_type: 'RuleType'                  # Classification
    priority: int                          # Execution priority (higher = first)

    # Conditions
    activation_conditions: List['Condition']  # When rule applies
    inhibition_conditions: List['Condition']  # When rule is suppressed

    # Parameters
    weight: float                          # Rule weight in blending
    radius: float                          # Spatial radius of effect
    threshold: float                       # Activation threshold
    parameters: Dict[str, float]           # Rule-specific parameters

    # Response
    response_type: 'ResponseType'          # Type of response generated
    response_magnitude: float              # Strength of response
    response_direction: Optional[str]      # Direction specification

    # Adaptation
    is_adaptive: bool = False              # Whether rule adapts
    learning_rate: float = 0.0             # Adaptation speed
    adaptation_history: List[float] = field(default_factory=list)


class RuleType(Enum):
    """Types of behavioral rules"""
    SEPARATION = "separation"              # Avoid crowding neighbors
    ALIGNMENT = "alignment"                # Steer toward average heading
    COHESION = "cohesion"                  # Steer toward average position
    AVOIDANCE = "avoidance"                # Obstacle avoidance
    ATTRACTION = "attraction"              # Attraction to targets
    WANDER = "wander"                      # Random exploration
    BOUNDARY = "boundary"                  # Stay within boundaries
    PHEROMONE_FOLLOW = "pheromone_follow"  # Follow chemical trails
    PHEROMONE_DEPOSIT = "pheromone_deposit"  # Deposit chemical signals
    WAGGLE_DANCE = "waggle_dance"          # Communicate food location
    QUORUM_SENSE = "quorum_sense"          # Density-dependent behavior
    GRADIENT_CLIMB = "gradient_climb"      # Follow gradient ascent
    TASK_ALLOCATION = "task_allocation"    # Dynamic task switching


class ResponseType(Enum):
    """Types of behavioral responses"""
    STEERING_FORCE = "steering_force"      # Change velocity direction
    SPEED_CHANGE = "speed_change"          # Modify speed
    STATE_CHANGE = "state_change"          # Change behavioral state
    SIGNAL_EMISSION = "signal_emission"    # Emit communication signal
    RESOURCE_ACTION = "resource_action"    # Pick up / deposit resource
    RECRUITMENT = "recruitment"            # Recruit other agents
```

#### 2.2 Interaction Network Structure

```python
@dataclass
class InteractionNetwork:
    """Network of agent interactions within the swarm"""

    network_id: str
    swarm_id: str
    timestamp: datetime

    # Network Structure
    adjacency_matrix: np.ndarray           # Agent-agent connections
    interaction_weights: np.ndarray        # Connection strengths
    interaction_types: Dict[Tuple[str, str], str]  # Type per edge

    # Network Metrics
    clustering_coefficient: float          # Local clustering
    average_path_length: float             # Mean shortest path
    degree_distribution: List[int]         # Degree histogram
    connectivity: float                    # Network connectivity (0.0-1.0)
    modularity: float                      # Community structure strength
    small_world_coefficient: float         # Small-world property

    # Dynamic Properties
    interaction_rate: float                # Interactions per tick
    network_turnover: float                # Rate of edge creation/deletion
    information_flow_rate: float           # Bits per tick through network
    cascade_potential: float               # Susceptibility to cascades


@dataclass
class InteractionEvent:
    """A single interaction between two agents"""
    event_id: str
    timestamp: datetime
    agent_a_id: str
    agent_b_id: str
    interaction_type: str                  # direct, stigmergic, broadcast
    information_exchanged: Dict[str, Any]
    duration_ticks: int
    outcome: str
```

### 3. Environment and Stigmergy Structures

#### 3.1 Swarm Environment Structure

```python
@dataclass
class SwarmEnvironment:
    """Environment in which the swarm operates"""

    environment_id: str
    dimensions: int                        # 2D or 3D
    size: Tuple[float, ...]                # Environment size per dimension
    boundary_type: 'BoundaryCondition'

    # Spatial Features
    obstacles: List['Obstacle']            # Physical obstacles
    resources: List['Resource']            # Available resources
    landmarks: List['Landmark']            # Reference points
    terrain: Optional['TerrainMap']        # Terrain data

    # Pheromone Fields
    pheromone_layers: Dict[str, 'PheromoneField']  # Named pheromone layers
    pheromone_diffusion_rate: float        # Diffusion coefficient
    pheromone_evaporation_rate: float      # Evaporation coefficient

    # Environmental Gradients
    gradients: Dict[str, 'GradientField']  # Named gradient fields
    temperature_field: Optional['ScalarField']
    light_field: Optional['ScalarField']

    # Dynamic Properties
    wind_field: Optional['VectorField']    # Wind or current vectors
    time_of_day: float                     # Cyclical time (0.0-1.0)
    season: float                          # Cyclical season (0.0-1.0)


class BoundaryCondition(Enum):
    """Environment boundary behavior"""
    PERIODIC = "periodic"                  # Wrap-around (toroidal)
    REFLECTIVE = "reflective"              # Bounce off walls
    ABSORBING = "absorbing"               # Remove at boundary
    REPULSIVE = "repulsive"               # Force away from boundary
    OPEN = "open"                          # No boundaries


@dataclass
class PheromoneField:
    """Spatially distributed pheromone concentration"""
    field_id: str
    pheromone_type: str                    # food_trail, home_trail, alarm, etc.
    grid_resolution: Tuple[int, ...]       # Grid dimensions
    concentration: np.ndarray              # Concentration values
    diffusion_rate: float                  # Spatial diffusion coefficient
    evaporation_rate: float                # Temporal decay rate
    max_concentration: float               # Saturation limit
    deposit_amount: float                  # Default deposit per agent
    last_updated: datetime


@dataclass
class Resource:
    """A resource in the environment"""
    resource_id: str
    position: np.ndarray
    resource_type: str                     # food, material, nest_site, etc.
    quantity: float                         # Available amount
    quality: float                         # Quality measure (0.0-1.0)
    renewal_rate: float                    # Regeneration rate
    discovery_time: Optional[datetime]     # When first found by agents
    depletion_time: Optional[datetime]     # When fully consumed
```

#### 3.2 Signal and Communication Structures

```python
@dataclass
class Signal:
    """Communication signal between agents or via environment"""

    signal_id: str
    sender_id: str
    signal_type: 'SignalType'
    timestamp: datetime

    # Content
    content: Dict[str, Any]                # Signal payload
    encoded_information: Optional[bytes]   # Encoded data

    # Propagation
    origin_position: np.ndarray            # Emission location
    propagation_range: float               # Maximum signal range
    current_strength: float                # Current signal strength
    attenuation_rate: float                # Strength decay with distance

    # Targeting
    target_type: str                       # broadcast, directed, stigmergic
    target_ids: Optional[List[str]]        # Specific targets if directed

    # Lifetime
    creation_tick: int
    expiration_tick: int
    is_active: bool = True


class SignalType(Enum):
    """Types of signals used in swarm communication"""
    PHEROMONE_TRAIL = "pheromone_trail"     # Chemical trail deposit
    ALARM_PHEROMONE = "alarm_pheromone"     # Alarm signal
    RECRUITMENT = "recruitment"             # Recruit to location
    WAGGLE_DANCE = "waggle_dance"           # Direction and distance info
    VIBRATION = "vibration"                 # Substrate vibration
    VISUAL_DISPLAY = "visual_display"       # Visual signaling
    ACOUSTIC = "acoustic"                   # Sound-based signal
    QUORUM_SIGNAL = "quorum_signal"         # Quorum sensing molecule
    CONTACT = "contact"                     # Direct contact signal


@dataclass
class CommEvent:
    """A completed communication event"""
    event_id: str
    sender_id: str
    receiver_ids: List[str]
    signal_type: SignalType
    timestamp: datetime
    success: bool
    information_transferred: float         # Bits of information
    latency_ticks: int                     # Delivery delay
```

### 4. Emergence Detection Structures

#### 4.1 Emergence State Structure

```python
@dataclass
class EmergenceState:
    """Current state of emergent phenomena in the swarm"""

    assessment_id: str
    swarm_id: str
    timestamp: datetime

    # Global Metrics
    order_parameter: float                 # Global order (0.0-1.0)
    complexity_index: float                # Behavioral complexity
    entropy: float                         # Shannon entropy of states
    mutual_information: float              # Information shared among agents
    integrated_information_phi: float      # Tononi phi estimate

    # Pattern Detection
    detected_patterns: List['EmergentPattern']  # Currently active patterns
    pattern_stability: Dict[str, float]    # Stability per pattern
    phase_state: 'PhaseState'              # Current thermodynamic phase

    # Transition Indicators
    criticality_measure: float             # Proximity to phase transition
    susceptibility: float                  # Response to perturbation
    correlation_length: float              # Spatial correlation extent
    fluctuation_magnitude: float           # State fluctuation size

    # Collective Intelligence Indicators
    decision_quality: float                # Quality of collective decisions
    information_processing_rate: float     # Bits processed per tick
    problem_solving_efficiency: float      # Solution quality / time
    adaptability_score: float              # Response to environmental change


@dataclass
class EmergentPattern:
    """A detected emergent pattern in the swarm"""
    pattern_id: str
    pattern_type: 'EmergenceType'
    description: str
    detection_timestamp: datetime

    # Pattern Characteristics
    scale: str                             # local, mesoscale, global
    stability: float                       # 0.0-1.0
    persistence_ticks: int                 # How long pattern has lasted
    participating_agents: int              # Number of agents involved

    # Metrics
    order_contribution: float              # Contribution to global order
    complexity_contribution: float         # Contribution to complexity
    novelty_score: float                   # How unexpected the pattern is

    # Spatial Properties
    spatial_extent: float                  # Physical extent of pattern
    center_position: Optional[np.ndarray]  # Pattern center
    spatial_symmetry: str                  # rotational, bilateral, none


class EmergenceType(Enum):
    """Types of emergent phenomena"""
    PATTERN_FORMATION = "pattern_formation"        # Spatial patterns
    PHASE_TRANSITION = "phase_transition"           # State change
    COLLECTIVE_DECISION = "collective_decision"     # Group choice
    SYNCHRONIZATION = "synchronization"             # Temporal alignment
    SELF_ORGANIZATION = "self_organization"         # Spontaneous order
    LANE_FORMATION = "lane_formation"               # Directional lanes
    VORTEX_FORMATION = "vortex_formation"           # Rotating structures
    CLUSTER_FORMATION = "cluster_formation"         # Group clustering
    TASK_DIFFERENTIATION = "task_differentiation"   # Role emergence
    COLLECTIVE_MEMORY = "collective_memory"         # Distributed memory


class PhaseState(Enum):
    """Thermodynamic-like phase states of the swarm"""
    DISORDERED = "disordered"              # No collective order (gas-like)
    TRANSITIONAL = "transitional"          # Near phase transition
    ORDERED = "ordered"                    # High collective order (solid-like)
    CRITICAL = "critical"                  # At phase transition point
    FLOCKING = "flocking"                  # Ordered collective motion
    MILLING = "milling"                    # Rotating collective motion
    SWARMING = "swarming"                  # Dense disordered motion
```

#### 4.2 Phase Transition Structure

```python
@dataclass
class PhaseTransition:
    """Record of a detected phase transition in the swarm"""

    transition_id: str
    swarm_id: str
    detection_timestamp: datetime

    # Transition Description
    from_phase: PhaseState
    to_phase: PhaseState
    transition_type: str                   # continuous, discontinuous, critical

    # Transition Parameters
    control_parameter: str                 # Parameter driving transition
    critical_value: float                  # Critical parameter value
    actual_value: float                    # Observed parameter value

    # Pre-Transition Indicators
    pre_transition_fluctuations: float     # Fluctuation amplitude before
    pre_transition_correlation: float      # Correlation length before
    early_warning_signals: List[str]       # Detected warning signs

    # Post-Transition State
    new_order_parameter: float             # Order after transition
    relaxation_time: int                   # Ticks to reach new equilibrium
    hysteresis: bool                       # Whether transition is reversible
    reversibility_parameter: Optional[float]
```

### 5. Optimization Algorithm Structures

#### 5.1 Ant Colony Optimization Structures

```python
@dataclass
class ACOState:
    """State of an Ant Colony Optimization algorithm"""

    aco_id: str
    problem_id: str
    iteration: int

    # Colony State
    colony_size: int
    ants: List['ACOAnt']
    pheromone_matrix: np.ndarray           # Pheromone on each edge
    heuristic_matrix: np.ndarray           # Static heuristic values

    # Parameters
    alpha: float                           # Pheromone weight exponent
    beta: float                            # Heuristic weight exponent
    evaporation_rate: float                # Pheromone evaporation (rho)
    deposit_amount: float                  # Pheromone deposit (Q)
    min_pheromone: float                   # Minimum pheromone level
    max_pheromone: float                   # Maximum pheromone level

    # Solution Tracking
    best_solution: Optional[List[int]]     # Best solution found
    best_cost: float                       # Cost of best solution
    iteration_best: Optional[List[int]]    # Best in current iteration
    iteration_best_cost: float
    solution_history: List[Tuple[int, float]]  # (iteration, best_cost)

    # Convergence
    convergence_metric: float              # Pheromone trail convergence
    diversity_metric: float                # Solution diversity
    stagnation_counter: int                # Iterations without improvement


@dataclass
class ACOAnt:
    """A single ant in ACO"""
    ant_id: str
    current_node: int
    visited_nodes: List[int]
    tour_cost: float
    pheromone_deposit: float
```

#### 5.2 Particle Swarm Optimization Structures

```python
@dataclass
class PSOState:
    """State of a Particle Swarm Optimization algorithm"""

    pso_id: str
    problem_id: str
    iteration: int

    # Swarm State
    swarm_size: int
    particles: List['PSOParticle']
    dimensions: int                        # Problem dimensionality

    # Parameters
    inertia_weight: float                  # w - velocity momentum
    cognitive_coefficient: float           # c1 - personal best weight
    social_coefficient: float              # c2 - global best weight
    velocity_clamp: float                  # Maximum velocity magnitude
    position_bounds: Tuple[np.ndarray, np.ndarray]  # Min/max per dim

    # Solution Tracking
    global_best_position: np.ndarray       # Best position found
    global_best_fitness: float             # Fitness at best position
    fitness_history: List[float]           # Best fitness per iteration

    # Convergence
    velocity_magnitude_avg: float          # Average velocity
    position_diversity: float              # Spread of positions
    stagnation_counter: int


@dataclass
class PSOParticle:
    """A single particle in PSO"""
    particle_id: str
    position: np.ndarray                   # Current position
    velocity: np.ndarray                   # Current velocity
    fitness: float                         # Current fitness value
    personal_best_position: np.ndarray     # Best position found
    personal_best_fitness: float           # Best fitness found
    neighborhood: List[str]                # Neighbor particle IDs
```

#### 5.3 Boids Simulation Structures

```python
@dataclass
class BoidsState:
    """State of a Boids flocking simulation"""

    boids_id: str
    iteration: int

    # Flock State
    boid_count: int
    boids: List['Boid']
    dimensions: int                        # 2D or 3D

    # Rule Parameters
    separation_radius: float               # Minimum separation distance
    alignment_radius: float                # Alignment neighborhood radius
    cohesion_radius: float                 # Cohesion neighborhood radius
    separation_weight: float               # Separation force weight
    alignment_weight: float                # Alignment force weight
    cohesion_weight: float                 # Cohesion force weight

    # Optional Rule Parameters
    obstacle_avoidance_radius: float = 0.0
    goal_attraction_weight: float = 0.0
    predator_avoidance_weight: float = 0.0

    # Flock Metrics
    flock_center: np.ndarray               # Center of mass
    flock_velocity: np.ndarray             # Average velocity
    polarization: float                    # Alignment measure (0.0-1.0)
    angular_momentum: float                # Collective rotation
    nearest_neighbor_distance: float       # Average NND


@dataclass
class Boid:
    """A single boid in the simulation"""
    boid_id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    max_speed: float
    max_force: float
    neighbors: List[str]
```

## Enumeration Types

### Coordination Mechanism Enumerations

```python
class CoordinationMechanism(Enum):
    """Mechanisms by which agents coordinate"""
    STIGMERGY = "stigmergy"                # Indirect via environment
    DIRECT_SIGNALING = "direct_signaling"  # Direct agent-to-agent
    VISUAL = "visual"                      # Visual observation
    CHEMICAL = "chemical"                  # Chemical signals
    ACOUSTIC = "acoustic"                  # Sound-based coordination
    TACTILE = "tactile"                    # Physical contact
    QUORUM_SENSING = "quorum_sensing"      # Density-dependent
    GRADIENT_FOLLOWING = "gradient_following"  # Environmental gradients
    LEADER_FOLLOWING = "leader_following"   # Follow designated leader
    HYBRID = "hybrid"                      # Multiple mechanisms


class SensoryModality(Enum):
    """Sensory modalities available to agents"""
    VISUAL = "visual"
    CHEMICAL = "chemical"
    ACOUSTIC = "acoustic"
    TACTILE = "tactile"
    MAGNETIC = "magnetic"
    ELECTRIC = "electric"
    THERMAL = "thermal"
    VIBRATION = "vibration"
```

### Simulation Control Enumerations

```python
class SimulationStatus(Enum):
    """Status of a swarm simulation"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    CONVERGED = "converged"
    TERMINATED = "terminated"
    ERROR = "error"


class TerminationReason(Enum):
    """Reason for simulation termination"""
    MAX_ITERATIONS = "max_iterations"
    CONVERGENCE = "convergence"
    USER_TERMINATED = "user_terminated"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    POPULATION_COLLAPSE = "population_collapse"
    ERROR = "error"
    TIMEOUT = "timeout"
```

## Input/Output Structures

### Simulation Input Structures

```python
@dataclass
class SimulationConfiguration:
    """Complete configuration for a swarm simulation"""

    # Simulation Identity
    simulation_id: str
    simulation_name: str
    swarm_type: SwarmType
    description: str = ""

    # Population Configuration
    initial_population: int = 100
    population_limits: PopulationLimits = field(default_factory=PopulationLimits)
    agent_configuration: 'AgentConfiguration' = None

    # Environment Configuration
    environment_size: Tuple[float, ...] = (100.0, 100.0)
    boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC
    obstacles: List[Dict] = field(default_factory=list)
    resources: List[Dict] = field(default_factory=list)

    # Behavioral Parameters
    behavioral_rules: List[BehavioralRule] = field(default_factory=list)
    coordination_mechanism: CoordinationMechanism = CoordinationMechanism.STIGMERGY

    # Simulation Parameters
    time_step: float = 0.1
    max_iterations: int = 10000
    convergence_threshold: float = 0.01
    random_seed: Optional[int] = None

    # Output Configuration
    snapshot_interval: int = 100           # Ticks between state snapshots
    metrics_interval: int = 10             # Ticks between metrics recording
    output_format: str = "json"            # json, csv, hdf5


@dataclass
class AgentConfiguration:
    """Configuration template for swarm agents"""
    agent_type: AgentType
    perception_radius: float = 10.0
    perception_angle: float = 2.0 * np.pi  # Full circle
    max_speed: float = 1.0
    max_acceleration: float = 0.5
    max_turn_rate: float = 0.3
    initial_energy: float = 1.0
    energy_consumption_rate: float = 0.001
    signal_emission_capable: bool = True
    sensory_modalities: List[SensoryModality] = field(
        default_factory=lambda: [SensoryModality.VISUAL]
    )
```

### Simulation Output Structures

```python
@dataclass
class SimulationResult:
    """Complete result of a swarm simulation"""

    simulation_id: str
    configuration: SimulationConfiguration
    status: SimulationStatus
    termination_reason: TerminationReason
    total_iterations: int
    wall_clock_time_seconds: float

    # Final State
    final_swarm_state: SwarmCollective
    final_emergence_state: EmergenceState

    # Time Series Data
    metrics_history: List['MetricsSnapshot']
    state_snapshots: List['SwarmSnapshot']
    emergence_events: List[EmergenceEvent]
    phase_transitions: List[PhaseTransition]

    # Summary Statistics
    summary: 'SimulationSummary'


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics capture"""
    tick: int
    timestamp: datetime
    population_size: int
    center_of_mass: np.ndarray
    dispersion: float
    polarization: float
    order_parameter: float
    complexity_index: float
    average_speed: float
    average_energy: float
    interaction_rate: float
    resource_collected: float


@dataclass
class SwarmSnapshot:
    """Complete swarm state at a point in time"""
    tick: int
    timestamp: datetime
    agent_positions: np.ndarray            # Nx3 array
    agent_velocities: np.ndarray           # Nx3 array
    agent_behaviors: List[str]             # Behavior per agent
    agent_energies: List[float]            # Energy per agent
    pheromone_state: Dict[str, np.ndarray] # Pheromone fields
    emergence_state: EmergenceState


@dataclass
class SimulationSummary:
    """Summary statistics for a completed simulation"""
    total_ticks: int
    total_agents_created: int
    total_agents_died: int
    total_interactions: int
    total_signals_emitted: int
    total_resources_collected: float
    total_distance_traveled: float
    average_order_parameter: float
    peak_order_parameter: float
    emergence_event_count: int
    phase_transition_count: int
    convergence_tick: Optional[int]
    final_fitness: float
```

## Internal State Structures

### Emergence Event Structure

```python
@dataclass
class EmergenceEvent:
    """Record of a detected emergence event"""

    event_id: str
    swarm_id: str
    timestamp: datetime
    tick: int

    # Event Classification
    emergence_type: EmergenceType
    significance: float                    # 0.0-1.0
    novelty: float                         # 0.0-1.0
    confidence: float                      # Detection confidence

    # Event Details
    description: str
    trigger: str                           # What triggered emergence
    participating_agents: int
    spatial_extent: float
    duration_ticks: int

    # Metrics at Event
    order_parameter_before: float
    order_parameter_after: float
    complexity_change: float
    information_integration_change: float

    # Impact Assessment
    impact_on_performance: float           # Change in collective fitness
    impact_on_behavior: str                # Description of behavioral impact
    cascading_effects: List[str]           # Subsequent events triggered
```

### Cross-System Observation Structure

```python
@dataclass
class CrossSystemObservation:
    """Observation data for comparing swarm systems"""

    observation_id: str
    timestamp: datetime

    # System Information
    system_type: SwarmType
    system_name: str
    species_or_algorithm: str

    # Observed Metrics
    population_size: int
    spatial_scale: float                   # Physical extent
    temporal_scale: float                  # Observation duration
    order_parameter: float
    complexity_index: float

    # Behavioral Observations
    coordination_mechanisms: List[CoordinationMechanism]
    dominant_behavior: AgentBehavior
    observed_emergence: List[EmergenceType]

    # Environmental Context
    environment_type: str
    resource_availability: float           # 0.0-1.0
    threat_level: float                    # 0.0-1.0

    # Comparative Metrics
    efficiency_ratio: float                # Task output / energy input
    robustness_score: float                # Resilience to perturbation
    scalability_factor: float              # Performance scaling with size
```

## Relationship Mappings

### Cross-Form Data Exchange Structures

```python
@dataclass
class CrossFormDataExchange:
    """Data exchange structures for inter-form communication"""

    # To/From Form 20 (Collective Consciousness)
    collective_consciousness_data: Dict[str, Any] = field(default_factory=lambda: {
        'swarm_collective_state': 'SwarmCollective',
        'emergence_metrics': 'EmergenceState',
        'coordination_quality': 'float',
        'collective_decision_data': 'Dict',
        'shared_knowledge_state': 'Dict'
    })

    # To/From Form 30 (Animal Cognition)
    animal_cognition_data: Dict[str, Any] = field(default_factory=lambda: {
        'individual_agent_behavior': 'AgentBehavior',
        'cognitive_complexity': 'float',
        'learning_observations': 'Dict',
        'communication_patterns': 'List[CommEvent]'
    })

    # To/From Form 32 (Fungal Intelligence)
    fungal_intelligence_data: Dict[str, Any] = field(default_factory=lambda: {
        'network_topology': 'InteractionNetwork',
        'stigmergic_communication': 'List[Signal]',
        'resource_transport_efficiency': 'float',
        'network_optimization_state': 'Dict'
    })

    # To/From Form 34 (Gaia Intelligence)
    gaia_intelligence_data: Dict[str, Any] = field(default_factory=lambda: {
        'ecosystem_level_emergence': 'EmergenceState',
        'population_dynamics': 'Dict',
        'ecological_interactions': 'InteractionNetwork',
        'environmental_feedback': 'Dict'
    })

    # To/From Form 13 (Integrated Information Theory)
    iit_data: Dict[str, Any] = field(default_factory=lambda: {
        'integrated_information_phi': 'float',
        'integration_structure': 'Dict',
        'partitioning_results': 'Dict',
        'consciousness_assessment': 'Dict'
    })

    # To/From Form 14 (Global Workspace)
    gwt_data: Dict[str, Any] = field(default_factory=lambda: {
        'emergence_events_for_broadcast': 'List[EmergenceEvent]',
        'collective_patterns': 'List[EmergentPattern]',
        'attention_allocation_request': 'Dict',
        'workspace_integration_quality': 'float'
    })


@dataclass
class FormIntegrationProtocol:
    """Protocol definition for cross-form data exchange"""
    source_form: str
    target_form: str
    data_type: str
    exchange_format: str                   # json, protobuf, msgpack
    update_frequency_hz: float
    max_latency_ms: float
    validation_schema: Dict[str, Any]
    bidirectional: bool = True
```

## Data Validation and Constraints

### Validation Rules

```python
class SwarmDataValidation:
    """Validation rules for swarm intelligence data structures"""

    @staticmethod
    def validate_agent(agent: SwarmAgent) -> 'ValidationResult':
        """Validate swarm agent data"""
        errors = []

        if not agent.agent_id:
            errors.append("Agent ID is required")
        if agent.energy_level < 0.0 or agent.energy_level > 1.0:
            errors.append("Energy level must be between 0.0 and 1.0")
        if agent.perception_radius < 0.0:
            errors.append("Perception radius must be non-negative")
        if agent.max_speed < 0.0:
            errors.append("Max speed must be non-negative")
        if not agent.behavioral_rules:
            errors.append("Agent must have at least one behavioral rule")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_swarm(swarm: SwarmCollective) -> 'ValidationResult':
        """Validate swarm collective data"""
        errors = []

        if not swarm.swarm_id:
            errors.append("Swarm ID is required")
        if swarm.population_size < 2:
            errors.append("Swarm must have at least 2 agents")
        if swarm.polarization < 0.0 or swarm.polarization > 1.0:
            errors.append("Polarization must be between 0.0 and 1.0")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_simulation_config(config: SimulationConfiguration) -> 'ValidationResult':
        """Validate simulation configuration"""
        errors = []

        if config.initial_population < 2:
            errors.append("Initial population must be at least 2")
        if config.time_step <= 0:
            errors.append("Time step must be positive")
        if config.max_iterations <= 0:
            errors.append("Max iterations must be positive")
        if len(config.environment_size) < 2:
            errors.append("Environment must be at least 2D")

        return ValidationResult(valid=len(errors) == 0, errors=errors)


@dataclass
class ValidationResult:
    """Result of data validation"""
    valid: bool
    errors: List[str]
    warnings: List[str] = field(default_factory=list)
```

## Data Serialization

### Serialization Support

```python
class SwarmDataSerialization:
    """Handles serialization of swarm intelligence data structures"""

    @staticmethod
    def to_json(obj: Any) -> str:
        """Serialize swarm data to JSON"""
        return json.dumps(obj, cls=SwarmDataEncoder, indent=2)

    @staticmethod
    def from_json(json_str: str, target_type: type) -> Any:
        """Deserialize JSON to swarm data object"""
        data = json.loads(json_str)
        return SwarmDataDecoder.decode(data, target_type)

    @staticmethod
    def snapshot_to_hdf5(snapshot: SwarmSnapshot, filepath: str):
        """Save swarm snapshot to HDF5 for efficient array storage"""
        import h5py
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('positions', data=snapshot.agent_positions)
            f.create_dataset('velocities', data=snapshot.agent_velocities)
            f.attrs['tick'] = snapshot.tick


class SwarmDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for swarm data structures"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)
```

---

This data structures specification provides the foundation for implementing swarm intelligence systems with comprehensive agent modeling, environmental simulation, emergence detection, and optimization algorithm support. All structures are designed for efficient serialization, cross-form data exchange, and real-time simulation performance.
