# Fungal Intelligence Data Structures

## Overview

This document defines the core data structures for the Fungal Networks and Mycorrhizal Intelligence system (Form 32). These structures model the distributed computational capabilities of fungal organisms, including mycorrhizal network topology, chemical and electrical signaling, slime mold problem-solving, resource transfer dynamics, and inter-kingdom communication. The structures support both macro-scale forest network simulation and micro-scale hyphal growth and decision-making.

All data models use Python dataclasses with standard library typing. Enum types encode the discrete states and categories spanning network topology roles, signaling modalities, foraging strategies, and fungal life cycle phases.

---

## Core Data Models

### Network Topology Structures

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import uuid


@dataclass
class HyphalSegment:
    """Individual segment of a fungal hypha."""
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_segment_id: Optional[str] = None
    child_segment_ids: List[str] = field(default_factory=list)
    length_um: float = 0.0
    diameter_um: float = 5.0
    orientation_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    growth_rate_um_per_hr: float = 0.0
    is_tip: bool = False
    is_septate: bool = True
    cytoplasmic_flow_rate: float = 0.0  # um/s
    nutrient_content: Dict[str, float] = field(default_factory=dict)
    age_hours: float = 0.0
    is_alive: bool = True


@dataclass
class HyphalTip:
    """Active growing tip of a hypha with sensing capabilities."""
    tip_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    growth_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    growth_rate_um_per_hr: float = 50.0
    branching_probability: float = 0.05
    chemotropic_sensitivity: float = 1.0
    sensed_chemicals: Dict[str, float] = field(default_factory=dict)
    sensed_physical: Dict[str, float] = field(default_factory=dict)
    turgor_pressure_MPa: float = 0.5
    vesicle_supply_rate: float = 1.0  # normalized
    spitzenkorper_active: bool = True


@dataclass
class AnastomosisJunction:
    """Junction where two hyphae fuse, creating network connections."""
    junction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_a_id: str = ""
    segment_b_id: str = ""
    junction_type: str = "hyphal_fusion"  # "hyphal_fusion", "clamp_connection", "bridge"
    conductance: float = 1.0  # relative transport capacity
    formed_at: datetime = field(default_factory=datetime.now)
    is_functional: bool = True
    self_nonself_verified: bool = True
    incompatibility_risk: float = 0.0


@dataclass
class MycelialNetwork:
    """Complete mycelial network topology for a single fungal individual or genet."""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    species: str = ""
    genet_id: str = ""
    segments: Dict[str, HyphalSegment] = field(default_factory=dict)
    tips: Dict[str, HyphalTip] = field(default_factory=dict)
    junctions: Dict[str, AnastomosisJunction] = field(default_factory=dict)
    total_length_m: float = 0.0
    total_biomass_g: float = 0.0
    network_density: float = 0.0  # junctions per unit volume
    fractal_dimension: float = 0.0
    connectivity_index: float = 0.0  # graph connectivity metric
    age_days: float = 0.0
    territory_volume_m3: float = 0.0
```

### Common Mycorrhizal Network (CMN) Structures

```python
@dataclass
class MycorrhizalConnection:
    """Single mycorrhizal connection between a fungus and a plant root."""
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fungal_network_id: str = ""
    plant_id: str = ""
    connection_type: str = ""  # "ectomycorrhizal", "arbuscular", "ericoid", "orchid"
    colonization_percentage: float = 0.0
    carbon_flux_ug_per_hr: float = 0.0   # plant -> fungus
    nitrogen_flux_ug_per_hr: float = 0.0  # fungus -> plant
    phosphorus_flux_ug_per_hr: float = 0.0  # fungus -> plant
    water_flux_uL_per_hr: float = 0.0
    signal_molecules_transferred: List[str] = field(default_factory=list)
    connection_age_days: float = 0.0
    health_score: float = 1.0
    is_active: bool = True


@dataclass
class CMNNode:
    """A node in the Common Mycorrhizal Network representing a connected plant."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plant_id: str = ""
    plant_species: str = ""
    is_hub_tree: bool = False
    is_mother_tree: bool = False
    connection_count: int = 0
    connections: List[str] = field(default_factory=list)  # connection_ids
    net_carbon_balance: float = 0.0  # positive = net donor
    net_nutrient_balance: float = 0.0
    kin_group_id: Optional[str] = None
    stress_level: float = 0.0
    age_years: float = 0.0


@dataclass
class CommonMycorrhizalNetwork:
    """Complete CMN connecting multiple plants through shared fungal networks."""
    cmn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fungal_networks: Dict[str, MycelialNetwork] = field(default_factory=dict)
    plant_nodes: Dict[str, CMNNode] = field(default_factory=dict)
    connections: Dict[str, MycorrhizalConnection] = field(default_factory=dict)
    total_plants_connected: int = 0
    total_fungal_species: int = 0
    network_diameter_m: float = 0.0
    hub_node_ids: List[str] = field(default_factory=list)
    carbon_redistribution_active: bool = False
    defense_signaling_active: bool = False
    topology_type: str = "scale_free"  # "scale_free", "random", "small_world"
    clustering_coefficient: float = 0.0
    average_path_length: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
```

### Signaling Structures

```python
@dataclass
class FungalElectricalSpike:
    """Electrical spike event in fungal mycelium (action potential analog)."""
    spike_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    origin_segment_id: str = ""
    amplitude_mV: float = 0.0
    duration_hours: float = 1.0  # fungal spikes range 1-21 hours
    spike_pattern_class: int = 0  # classification index (up to ~50 classes)
    propagation_velocity_mm_per_s: float = 0.5
    propagation_path: List[str] = field(default_factory=list)
    inter_spike_interval_hours: Optional[float] = None
    triggered_by: str = ""  # "chemical", "mechanical", "spontaneous", "relay"
    information_content_bits: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FungalVOCSignal:
    """Volatile organic compound signal emitted or detected by fungal mycelium."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compound_name: str = ""
    compound_class: str = ""  # "sesquiterpene", "oxylipid", "ketone", "alcohol", etc.
    concentration_ppb: float = 0.0
    emission_rate_ng_per_hr: float = 0.0
    ecological_function: str = ""  # "defense", "attraction", "repulsion", "species_recognition"
    target_organism: str = ""  # "conspecific", "plant", "insect", "bacteria"
    diffusion_radius_m: float = 0.0
    half_life_minutes: float = 0.0
    is_emitted: bool = True  # True = emitting, False = detecting
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnzymeSecretion:
    """Extracellular enzyme secretion for substrate decomposition."""
    secretion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enzyme_class: str = ""  # "laccase", "peroxidase", "cellulase", "protease", etc.
    secretion_rate: float = 0.0
    target_substrate: str = ""  # "lignin", "cellulose", "chitin", "protein"
    pH_optimum: float = 5.0
    temperature_optimum_C: float = 25.0
    activity_level: float = 1.0  # normalized 0-1
    segment_ids: List[str] = field(default_factory=list)  # where secretion occurs


@dataclass
class IntegratedFungalSignalState:
    """Combined multi-modal signal state across the fungal network."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    electrical_spikes: List[FungalElectricalSpike] = field(default_factory=list)
    voc_signals: List[FungalVOCSignal] = field(default_factory=list)
    enzyme_secretions: List[EnzymeSecretion] = field(default_factory=list)
    cytoplasmic_streaming_rates: Dict[str, float] = field(default_factory=dict)
    dominant_signal_mode: Optional[str] = None
    network_activation_level: float = 0.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
```

### Slime Mold Computation Structures

```python
@dataclass
class PhysarumNode:
    """Node in a Physarum polycephalum transport network."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position: Tuple[float, float] = (0.0, 0.0)
    is_food_source: bool = False
    nutrient_concentration: float = 0.0
    oscillation_frequency_Hz: float = 0.01
    oscillation_phase: float = 0.0
    tube_connections: List[str] = field(default_factory=list)


@dataclass
class PhysarumTube:
    """Transport tube in Physarum network connecting two nodes."""
    tube_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_a_id: str = ""
    node_b_id: str = ""
    length_mm: float = 0.0
    diameter_um: float = 50.0
    flow_rate_uL_per_s: float = 0.0
    flow_direction: int = 1  # 1 = A->B, -1 = B->A, 0 = oscillating
    conductance: float = 1.0
    reinforcement_level: float = 0.5  # determines growth/shrinkage
    age_hours: float = 0.0


@dataclass
class PhysarumNetwork:
    """Complete Physarum transport network for optimization computation."""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, PhysarumNode] = field(default_factory=dict)
    tubes: Dict[str, PhysarumTube] = field(default_factory=dict)
    food_sources: List[str] = field(default_factory=list)
    total_biomass_mg: float = 0.0
    network_efficiency: float = 0.0  # ratio of actual to optimal path costs
    fault_tolerance: float = 0.0     # robustness to random edge removal
    optimization_objective: str = "shortest_path"
    convergence_metric: float = 0.0
    iteration_count: int = 0


@dataclass
class PhysarumComputationResult:
    """Result of a Physarum-based optimization computation."""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_type: str = ""  # "shortest_path", "network_design", "multi_objective"
    input_network: str = ""  # network_id
    solution_path: List[str] = field(default_factory=list)  # ordered node_ids
    solution_cost: float = 0.0
    optimality_ratio: float = 0.0  # solution_cost / known_optimal
    iterations_to_converge: int = 0
    computation_time_seconds: float = 0.0
    is_converged: bool = False
    pareto_front: Optional[List[Tuple[float, float]]] = None  # for multi-objective
```

---

## Enumeration Types

```python
class FungalLifePhase(Enum):
    """Life cycle phases of fungal organisms."""
    SPORE_DORMANT = auto()
    GERMINATION = auto()
    VEGETATIVE_GROWTH = auto()
    EXPLORATION = auto()
    EXPLOITATION = auto()
    REPRODUCTIVE = auto()
    SENESCENCE = auto()
    DORMANCY = auto()


class MycorrhizalType(Enum):
    """Types of mycorrhizal associations."""
    ECTOMYCORRHIZAL = auto()
    ARBUSCULAR = auto()
    ERICOID = auto()
    ORCHID = auto()
    MONOTROPOID = auto()
    ARBUTOID = auto()


class NetworkRole(Enum):
    """Role of a node within the mycorrhizal network."""
    HUB_NODE = auto()          # Highly connected mother tree
    CONNECTOR_NODE = auto()    # Bridges between clusters
    PERIPHERAL_NODE = auto()   # Low-connectivity edge node
    SEEDLING_NODE = auto()     # Young plant receiving support
    DONOR_NODE = auto()        # Net carbon/nutrient provider
    RECEIVER_NODE = auto()     # Net carbon/nutrient consumer
    RELAY_NODE = auto()        # Signal/resource pass-through


class DecompositionStrategy(Enum):
    """Fungal decomposition strategies."""
    WHITE_ROT = auto()    # Lignin + cellulose degradation
    BROWN_ROT = auto()    # Cellulose preferential degradation
    SOFT_ROT = auto()     # Cellulose in wet conditions
    LITTER_DECAY = auto()
    WOOD_DECAY = auto()
    COPROPHILOUS = auto()  # Dung decomposition


class ForagingStrategy(Enum):
    """Mycelial foraging strategies observed in fungi."""
    PHALANX = auto()        # Dense, slow-advancing front
    GUERRILLA = auto()      # Long, sparse exploratory hyphae
    FAN_SHAPED = auto()     # Broad exploration front
    CORD_FORMING = auto()   # Long-distance rhizomorph transport
    DIFFUSE = auto()        # Even, undirected spread
    PATCH_EXPLOITATION = auto()  # Concentrated on resource patches


class ElectricalSpikeClass(Enum):
    """Classification of fungal electrical spike patterns."""
    SHORT_BURST = auto()       # 1-3 hour duration
    LONG_SUSTAINED = auto()    # 10-21 hour duration
    RHYTHMIC_TRAIN = auto()    # Regular repeating pattern
    IRREGULAR_BURST = auto()   # Aperiodic burst sequence
    PROPAGATING_WAVE = auto()  # Coordinated wave across network
    RESPONSE_SPIKE = auto()    # Stimulus-evoked response


class InterKingdomSignalTarget(Enum):
    """Target organisms for inter-kingdom fungal signaling."""
    PLANT_HOST = auto()
    PLANT_NON_HOST = auto()
    BACTERIA = auto()
    OTHER_FUNGUS_CONSPECIFIC = auto()
    OTHER_FUNGUS_HETEROSPECIFIC = auto()
    INSECT_MUTUALIST = auto()
    INSECT_ANTAGONIST = auto()
    NEMATODE = auto()
    VERTEBRATE = auto()


class HostManipulationType(Enum):
    """Types of host behavioral manipulation by parasitic fungi."""
    SUMMIT_DISEASE = auto()       # Ophiocordyceps: climbing and biting behavior
    SPORULATION_POSITIONING = auto()
    FEEDING_BEHAVIOR_CHANGE = auto()
    MOVEMENT_ALTERATION = auto()
    AGGREGATION_INDUCTION = auto()
    NONE = auto()
```

---

## Input/Output Structures

### System Input Structures

```python
@dataclass
class FungalIntelligenceInput:
    """Top-level input to the fungal intelligence processing system."""
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    substrate_environment: 'SubstrateEnvironment' = None
    current_network_state: Optional[MycelialNetwork] = None
    cmn_state: Optional[CommonMycorrhizalNetwork] = None
    incoming_signals: IntegratedFungalSignalState = field(default_factory=IntegratedFungalSignalState)
    plant_partner_states: List[Dict[str, any]] = field(default_factory=list)
    cross_form_inputs: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_priority: str = "normal"


@dataclass
class SubstrateEnvironment:
    """Environmental conditions of the substrate the fungus inhabits."""
    substrate_type: str = ""  # "soil", "wood", "leaf_litter", "dung", "living_tissue"
    moisture_content_percent: float = 30.0
    temperature_C: float = 20.0
    pH: float = 5.5
    oxygen_level_percent: float = 18.0
    organic_matter_percent: float = 5.0
    nutrient_concentrations: Dict[str, float] = field(default_factory=dict)
    competing_organism_density: float = 0.0
    substrate_density_g_per_cm3: float = 1.2
    toxin_levels: Dict[str, float] = field(default_factory=dict)
```

### System Output Structures

```python
@dataclass
class FungalIntelligenceOutput:
    """Top-level output from the fungal intelligence processing system."""
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_id: str = ""
    life_phase: FungalLifePhase = FungalLifePhase.VEGETATIVE_GROWTH
    foraging_strategy: ForagingStrategy = ForagingStrategy.DIFFUSE
    growth_directives: List['HyphalGrowthDirective'] = field(default_factory=list)
    network_remodeling: List['NetworkRemodelingAction'] = field(default_factory=list)
    resource_transfers: List['ResourceTransferDirective'] = field(default_factory=list)
    signal_emissions: List[FungalVOCSignal] = field(default_factory=list)
    electrical_outputs: List[FungalElectricalSpike] = field(default_factory=list)
    enzyme_directives: List[EnzymeSecretion] = field(default_factory=list)
    consciousness_indicators: Dict[str, float] = field(default_factory=dict)
    processing_latency_ms: float = 0.0
    cross_form_outputs: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HyphalGrowthDirective:
    """Directive for hyphal tip growth or branching."""
    directive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tip_id: str = ""
    action: str = ""  # "extend", "branch", "fuse", "retract", "dormancy"
    direction_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    rate_modifier: float = 1.0
    chemotropic_target: Optional[str] = None
    priority: int = 5
    triggered_by: str = ""


@dataclass
class NetworkRemodelingAction:
    """Action to remodel network topology by strengthening or pruning connections."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""  # "reinforce", "prune", "redirect", "anastomose"
    target_segment_ids: List[str] = field(default_factory=list)
    transport_capacity_change: float = 0.0  # positive = reinforce, negative = prune
    rationale: str = ""  # "unused_path", "redundancy_reduction", "shortcut_creation"


@dataclass
class ResourceTransferDirective:
    """Directive for transferring resources through the CMN."""
    transfer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    destination_node_id: str = ""
    resource_type: str = ""  # "carbon", "nitrogen", "phosphorus", "water", "defense_signal"
    quantity: float = 0.0
    unit: str = ""
    priority: int = 5
    is_kin_preferential: bool = False
    is_defense_related: bool = False
```

---

## Internal State Structures

```python
@dataclass
class FungalDecisionState:
    """Internal decision-making state for resource allocation and growth strategy."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_foraging_strategy: ForagingStrategy = ForagingStrategy.DIFFUSE
    exploration_exploitation_balance: float = 0.5
    resource_budget: Dict[str, float] = field(default_factory=dict)
    growth_front_priorities: List[Tuple[str, float]] = field(default_factory=list)
    threat_assessments: Dict[str, float] = field(default_factory=dict)
    partner_plant_rankings: List[Tuple[str, float]] = field(default_factory=list)
    strategy_confidence: float = 0.5
    last_strategy_switch: Optional[datetime] = None


@dataclass
class FungalMemoryState:
    """Memory and learning state in fungal systems."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    habituation_records: List[Dict[str, any]] = field(default_factory=list)
    spatial_memory_map: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    nutrient_patch_history: List[Dict[str, any]] = field(default_factory=list)
    substrate_quality_learned: Dict[str, float] = field(default_factory=dict)
    threat_memory: List[Dict[str, any]] = field(default_factory=list)
    network_topology_history: List[str] = field(default_factory=list)


@dataclass
class NetworkFlowState:
    """Current state of resource flow through the mycelial network."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    flow_graph: Dict[str, Dict[str, float]] = field(default_factory=dict)  # edge -> flow
    source_nodes: List[str] = field(default_factory=list)
    sink_nodes: List[str] = field(default_factory=list)
    total_carbon_flux_ug_per_hr: float = 0.0
    total_nutrient_flux_ug_per_hr: float = 0.0
    flow_efficiency: float = 0.0
    bottleneck_segments: List[str] = field(default_factory=list)
    redistribution_active: bool = False


@dataclass
class FungalConsciousnessMetrics:
    """Consciousness and intelligence assessment metrics for fungal systems."""
    metrics_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    network_integration_score: float = 0.0      # information integration across network
    problem_solving_score: float = 0.0           # optimization capability measure
    learning_capacity_score: float = 0.0         # habituation and spatial memory
    communication_complexity: float = 0.0        # signal vocabulary richness
    self_nonself_discrimination: float = 0.0     # vegetative compatibility accuracy
    anticipatory_behavior: float = 0.0           # predictive resource allocation
    collective_intelligence_score: float = 0.0   # CMN-level emergent behavior
    overall_consciousness_estimate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Relationship Mappings

### Cross-Form Data Exchange

```python
@dataclass
class PlantInterfaceExchange:
    """Data exchanged with Form 31 (Plant Intelligence) through mycorrhizal connections."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connected_plant_ids: List[str] = field(default_factory=list)
    carbon_received_total: float = 0.0
    nutrients_provided_total: Dict[str, float] = field(default_factory=dict)
    defense_signals_relayed: int = 0
    plant_stress_indicators: Dict[str, float] = field(default_factory=dict)
    kin_recognition_data: Dict[str, bool] = field(default_factory=dict)
    connection_health_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SwarmIntelligenceExchange:
    """Data exchanged with Form 33 (Swarm Intelligence) for distributed computation parallels."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    network_topology_summary: Dict[str, any] = field(default_factory=dict)
    optimization_problem_type: str = ""
    solution_quality_metric: float = 0.0
    distributed_decision_algorithm: str = ""
    emergent_pattern_type: str = ""
    agent_count: int = 0  # number of active hyphal tips


@dataclass
class AnimalCognitionExchange:
    """Data exchanged with Form 30 (Animal Cognition) for comparative intelligence studies."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    host_manipulation_data: Optional[Dict[str, any]] = None  # Ophiocordyceps etc.
    spatial_memory_comparison: Dict[str, float] = field(default_factory=dict)
    problem_solving_comparison: Dict[str, float] = field(default_factory=dict)
    neural_analog_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class CrossFormMessage:
    """Generic message structure for cross-form data exchange."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: int = 32
    target_form: int = 0
    message_type: str = ""  # "data_request", "data_response", "event_notification", "sync"
    payload: Dict[str, any] = field(default_factory=dict)
    priority: int = 5
    requires_response: bool = False
    ttl_seconds: int = 300
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
```

---

## Data Validation and Constraints

```python
@dataclass
class FungalDataConstraints:
    """Validation constraints for fungal intelligence data structures."""

    # Hyphal growth bounds
    HYPHAL_DIAMETER_MIN_UM: float = 1.0
    HYPHAL_DIAMETER_MAX_UM: float = 30.0
    GROWTH_RATE_MAX_UM_PER_HR: float = 1000.0  # fast-growing species
    BRANCHING_ANGLE_MIN_DEG: float = 30.0
    BRANCHING_ANGLE_MAX_DEG: float = 90.0

    # Electrical spike bounds (from Adamatzky research)
    SPIKE_AMPLITUDE_MIN_MV: float = 0.03
    SPIKE_AMPLITUDE_MAX_MV: float = 2.1
    SPIKE_DURATION_MIN_HR: float = 1.0
    SPIKE_DURATION_MAX_HR: float = 21.0
    SPIKE_VOCABULARY_SIZE: int = 50

    # Network topology bounds
    MAX_NETWORK_SEGMENTS: int = 1_000_000
    MAX_CMN_PLANT_NODES: int = 10_000
    MAX_ANASTOMOSIS_JUNCTIONS: int = 500_000

    # Environmental bounds
    TEMPERATURE_MIN_C: float = -5.0
    TEMPERATURE_MAX_C: float = 45.0
    PH_MIN: float = 2.0
    PH_MAX: float = 9.0
    MOISTURE_MIN_PERCENT: float = 5.0
    MOISTURE_MAX_PERCENT: float = 100.0

    # Physarum computation bounds
    PHYSARUM_MAX_NODES: int = 1000
    PHYSARUM_MAX_TUBES: int = 10_000
    PHYSARUM_MAX_ITERATIONS: int = 100_000

    # System bounds
    MAX_CROSS_FORM_MESSAGES_PER_SECOND: int = 200
    CONFIDENCE_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)
    VOC_VOCABULARY_SIZE: int = 300  # over 300 distinct VOCs identified
```
