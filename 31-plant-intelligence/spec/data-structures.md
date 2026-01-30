# Plant Intelligence Data Structures

## Overview

This document defines the core data structures for the Plant Intelligence and Vegetal Consciousness system (Form 31). These structures model plant cognition, chemical signaling, distributed decision-making, learning and memory, sensory integration, and social behavior in plant systems. They are designed to integrate with the broader consciousness framework while respecting the unique characteristics of non-neural, distributed intelligence.

All data models use Python dataclasses and standard library typing for clarity. Enum types codify the discrete states and categories found across plant intelligence domains including chemical signaling, resource allocation, phenotypic plasticity, and mycorrhizal network interactions.

---

## Core Data Models

### Environmental Sensing State

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import uuid


@dataclass
class PhotoreceptorReading:
    """Individual photoreceptor channel measurement."""
    wavelength_nm: float
    intensity_umol_m2_s: float
    direction_vector: Tuple[float, float, float]
    duration_exposure: timedelta
    receptor_type: str  # e.g., "phytochrome_A", "cryptochrome_1", "phototropin"
    photoequilibrium_ratio: Optional[float] = None


@dataclass
class ChemicalGradient:
    """Spatial chemical concentration gradient in root zone or canopy air."""
    compound_id: str
    compound_name: str
    concentration_mol_per_L: float
    gradient_vector: Tuple[float, float, float]
    diffusion_rate: float
    source_distance_estimate_m: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MechanicalStimulus:
    """Thigmotropic or wind-induced mechanical stimulus."""
    force_newtons: float
    direction_vector: Tuple[float, float, float]
    contact_area_mm2: float
    duration: timedelta
    stimulus_type: str  # "wind", "touch", "vibration", "gravity_shift"
    frequency_hz: Optional[float] = None


@dataclass
class EnvironmentalSnapshot:
    """Complete environmental state at a given moment."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    photoreceptor_readings: List[PhotoreceptorReading] = field(default_factory=list)
    chemical_gradients: List[ChemicalGradient] = field(default_factory=list)
    mechanical_stimuli: List[MechanicalStimulus] = field(default_factory=list)
    soil_moisture_fraction: float = 0.0
    soil_nutrient_concentrations: Dict[str, float] = field(default_factory=dict)
    air_temperature_celsius: float = 20.0
    soil_temperature_celsius: float = 18.0
    relative_humidity_percent: float = 50.0
    co2_concentration_ppm: float = 415.0
    photoperiod_hours: float = 12.0
    season_estimate: Optional[str] = None
```

### Signal Propagation Model

```python
@dataclass
class ElectricalSignal:
    """Action potential or variation potential in plant tissue."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: str = "action_potential"  # "action_potential", "variation_potential", "system_potential"
    origin_tissue: str = ""
    propagation_velocity_m_per_s: float = 0.0
    amplitude_mV: float = 0.0
    duration_ms: float = 0.0
    propagation_path: List[str] = field(default_factory=list)
    attenuation_factor: float = 1.0
    triggered_by: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChemicalSignal:
    """Volatile or soluble chemical signal between plant organs or individuals."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compound_id: str = ""
    compound_class: str = ""  # "VOC", "hormone", "peptide", "RNA", "exudate"
    concentration_mol_per_L: float = 0.0
    emission_rate: float = 0.0
    half_life_seconds: float = 0.0
    target_tissue: Optional[str] = None
    is_systemic: bool = False
    ecological_function: str = ""  # "defense", "attraction", "growth", "stress"
    diffusion_medium: str = "air"  # "air", "soil_solution", "phloem", "xylem"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HydraulicSignal:
    """Pressure-based signal propagated through xylem or turgor changes."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pressure_change_MPa: float = 0.0
    propagation_velocity_m_per_s: float = 0.0
    origin_organ: str = ""
    affected_organs: List[str] = field(default_factory=list)
    signal_type: str = "turgor_change"  # "turgor_change", "cavitation", "xylem_pressure_wave"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegratedSignalState:
    """Combined multi-modal signal state at a tissue or whole-plant level."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    electrical_signals: List[ElectricalSignal] = field(default_factory=list)
    chemical_signals: List[ChemicalSignal] = field(default_factory=list)
    hydraulic_signals: List[HydraulicSignal] = field(default_factory=list)
    dominant_signal_mode: Optional[str] = None
    integration_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
```

### Learning and Memory Structures

```python
@dataclass
class HabituationRecord:
    """Record of habituation learning to repeated stimuli."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stimulus_type: str = ""
    initial_response_magnitude: float = 0.0
    current_response_magnitude: float = 0.0
    exposure_count: int = 0
    habituation_rate: float = 0.0  # rate of response decline per exposure
    recovery_time_estimate: timedelta = timedelta(0)
    is_dishabituated: bool = False
    first_exposure: Optional[datetime] = None
    last_exposure: Optional[datetime] = None


@dataclass
class AssociativeMemory:
    """Associative learning record linking co-occurring stimuli."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conditioned_stimulus: str = ""
    unconditioned_stimulus: str = ""
    association_strength: float = 0.0  # 0.0 to 1.0
    pairing_count: int = 0
    extinction_trials: int = 0
    last_reinforcement: Optional[datetime] = None
    formed_at: datetime = field(default_factory=datetime.now)
    is_extinguished: bool = False


@dataclass
class EpigeneticMemoryState:
    """Epigenetic modifications encoding long-term stress memory."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modification_type: str = ""  # "DNA_methylation", "histone_modification", "chromatin_remodeling"
    target_gene_region: str = ""
    stress_type_encoded: str = ""
    modification_level: float = 0.0  # normalized 0.0 to 1.0
    generation_acquired: int = 0
    transgenerational: bool = False
    stability_estimate: float = 0.0  # probability of retention per generation
    acquired_at: Optional[datetime] = None


@dataclass
class PlantMemoryStore:
    """Aggregate memory store for a plant individual or colony."""
    store_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    habituation_records: List[HabituationRecord] = field(default_factory=list)
    associative_memories: List[AssociativeMemory] = field(default_factory=list)
    epigenetic_states: List[EpigeneticMemoryState] = field(default_factory=list)
    stress_priming_history: List[str] = field(default_factory=list)
    total_memory_capacity_estimate: float = 0.0
    oldest_memory_age: Optional[timedelta] = None
```

---

## Enumeration Types

```python
class SignalType(Enum):
    """Types of signals in plant communication systems."""
    ACTION_POTENTIAL = auto()
    VARIATION_POTENTIAL = auto()
    SYSTEM_POTENTIAL = auto()
    VOLATILE_ORGANIC_COMPOUND = auto()
    ROOT_EXUDATE = auto()
    HORMONE = auto()
    PEPTIDE_SIGNAL = auto()
    RNA_SIGNAL = auto()
    HYDRAULIC_PRESSURE = auto()
    MECHANICAL_WAVE = auto()


class SensoryModality(Enum):
    """Sensory modalities available to plant systems."""
    PHOTORECEPTION = auto()       # Light quality, quantity, direction
    CHEMORECEPTION = auto()       # Chemical detection in air and soil
    MECHANORECEPTION = auto()     # Touch, wind, vibration
    GRAVITROPISM = auto()         # Gravity sensing via statoliths
    THERMORECEPTION = auto()      # Temperature sensing
    HYDROTROPISM = auto()         # Water gradient sensing
    ELECTRORECEPTION = auto()     # Electrical field detection
    ACOUSTIC_SENSING = auto()     # Sound/vibration frequency detection
    MAGNETORECEPTION = auto()     # Magnetic field orientation


class ResourceAllocationStrategy(Enum):
    """Strategies for distributing resources across plant body."""
    BALANCED_GROWTH = auto()
    ROOT_PRIORITY = auto()
    SHOOT_PRIORITY = auto()
    REPRODUCTIVE_INVESTMENT = auto()
    DEFENSE_MOBILIZATION = auto()
    STORAGE_ACCUMULATION = auto()
    STRESS_SURVIVAL = auto()
    COMPETITIVE_DISPLACEMENT = auto()


class PlantBehaviorState(Enum):
    """High-level behavioral states of the plant intelligence system."""
    VEGETATIVE_GROWTH = auto()
    REPRODUCTIVE_TRANSITION = auto()
    STRESS_RESPONSE = auto()
    DORMANCY = auto()
    FORAGING = auto()
    DEFENSE_ACTIVE = auto()
    RECOVERY = auto()
    SENESCENCE = auto()


class MycorrhizalInteractionType(Enum):
    """Types of mycorrhizal network interactions."""
    CARBON_TRANSFER = auto()
    NUTRIENT_EXCHANGE = auto()
    DEFENSE_SIGNAL_RELAY = auto()
    WATER_REDISTRIBUTION = auto()
    INFORMATION_SHARING = auto()
    KIN_PREFERENTIAL = auto()
    COMPETITIVE_SUPPRESSION = auto()


class LearningMechanism(Enum):
    """Mechanisms of learning observed in plant systems."""
    HABITUATION = auto()
    SENSITIZATION = auto()
    ASSOCIATIVE_CONDITIONING = auto()
    EPIGENETIC_MEMORY = auto()
    STRESS_PRIMING = auto()
    TRANSGENERATIONAL = auto()
    PHENOTYPIC_PLASTICITY = auto()


class DefenseStrategy(Enum):
    """Plant defense response strategies."""
    CONSTITUTIVE_PHYSICAL = auto()   # Thorns, trichomes, thick cell walls
    INDUCED_CHEMICAL = auto()        # Toxins, deterrents produced on attack
    INDIRECT_DEFENSE = auto()        # Attracting predators of herbivores
    VOLATILE_SIGNALING = auto()      # Warning neighboring plants
    SYSTEMIC_ACQUIRED = auto()       # Whole-plant immune priming
    JASMONIC_ACID_PATHWAY = auto()   # JA-mediated herbivore defense
    SALICYLIC_ACID_PATHWAY = auto()  # SA-mediated pathogen defense
    TOLERANCE = auto()               # Growth compensation after damage
```

---

## Input/Output Structures

### System Input Structures

```python
@dataclass
class PlantIntelligenceInput:
    """Top-level input to the plant intelligence processing system."""
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    environment: EnvironmentalSnapshot = field(default_factory=EnvironmentalSnapshot)
    current_signals: IntegratedSignalState = field(default_factory=IntegratedSignalState)
    memory_context: PlantMemoryStore = field(default_factory=PlantMemoryStore)
    neighboring_plant_signals: List[ChemicalSignal] = field(default_factory=list)
    mycorrhizal_network_data: Optional['MycorrhizalNetworkState'] = None
    cross_form_inputs: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_priority: str = "normal"  # "urgent", "normal", "background"


@dataclass
class EnvironmentalQuery:
    """Query for specific environmental information."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: str = ""  # "resource_availability", "threat_assessment", "season_phase"
    spatial_scope_m: float = 1.0
    temporal_scope: timedelta = timedelta(hours=1)
    required_modalities: List[SensoryModality] = field(default_factory=list)
    resolution_requirement: str = "standard"
```

### System Output Structures

```python
@dataclass
class PlantIntelligenceOutput:
    """Top-level output from the plant intelligence processing system."""
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_id: str = ""
    behavioral_state: PlantBehaviorState = PlantBehaviorState.VEGETATIVE_GROWTH
    resource_allocation: ResourceAllocationStrategy = ResourceAllocationStrategy.BALANCED_GROWTH
    defense_responses: List[DefenseStrategy] = field(default_factory=list)
    growth_directives: List['GrowthDirective'] = field(default_factory=list)
    signal_emissions: List[ChemicalSignal] = field(default_factory=list)
    memory_updates: List[Dict[str, any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_latency_ms: float = 0.0
    cross_form_outputs: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GrowthDirective:
    """Directive for growth or morphological change."""
    directive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_organ: str = ""  # "root_tip", "shoot_apex", "leaf", "lateral_root"
    growth_type: str = ""   # "elongation", "branching", "thickening", "abscission"
    direction_vector: Optional[Tuple[float, float, float]] = None
    rate_modifier: float = 1.0  # multiplier on base growth rate
    duration_estimate: Optional[timedelta] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    triggered_by: str = ""  # signal or condition that triggered this directive


@dataclass
class ConsciousnessAssessment:
    """Assessment of plant consciousness indicators for cross-form integration."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    information_integration_score: float = 0.0    # phi-like measure
    behavioral_complexity_score: float = 0.0
    learning_capacity_score: float = 0.0
    self_nonself_discrimination: float = 0.0
    anticipatory_behavior_score: float = 0.0
    agency_indicators: Dict[str, float] = field(default_factory=dict)
    overall_consciousness_estimate: float = 0.0   # 0.0 to 1.0
    confidence: float = 0.0
    methodology_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Internal State Structures

```python
@dataclass
class PlantBodyModel:
    """Internal representation of the plant body and its organ systems."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_system: 'OrganSystem' = None
    shoot_system: 'OrganSystem' = None
    reproductive_organs: Optional['OrganSystem'] = None
    total_biomass_g: float = 0.0
    root_shoot_ratio: float = 1.0
    age_days: int = 0
    developmental_stage: str = "vegetative"
    organ_count: Dict[str, int] = field(default_factory=dict)
    vascular_connectivity: float = 1.0  # 0.0 to 1.0 integrity measure


@dataclass
class OrganSystem:
    """Representation of a plant organ system (root or shoot)."""
    system_type: str = ""
    organ_count: int = 0
    total_mass_g: float = 0.0
    active_meristems: int = 0
    signal_processing_nodes: int = 0
    resource_status: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    growth_rate_mm_per_day: float = 0.0


@dataclass
class DecisionState:
    """Internal decision-making state tracking resource trade-offs."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active_decisions: List[str] = field(default_factory=list)
    resource_budget: Dict[str, float] = field(default_factory=dict)
    competing_demands: List[Tuple[str, float]] = field(default_factory=list)
    current_strategy: ResourceAllocationStrategy = ResourceAllocationStrategy.BALANCED_GROWTH
    strategy_confidence: float = 0.5
    time_horizon: timedelta = timedelta(hours=24)
    last_strategy_switch: Optional[datetime] = None
    switch_threshold: float = 0.3  # confidence delta to trigger strategy change


@dataclass
class ForagingState:
    """State of root or shoot foraging behavior."""
    foraging_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    foraging_type: str = ""  # "root_nutrient", "root_water", "shoot_light"
    target_resource: str = ""
    search_pattern: str = "random"  # "random", "directed", "exploitation", "exploration"
    patch_quality_estimate: float = 0.0
    exploration_exploitation_ratio: float = 0.5
    marginal_value_threshold: float = 0.0
    time_in_current_patch: timedelta = timedelta(0)
    patches_visited: int = 0


@dataclass
class CircadianState:
    """Internal circadian clock state governing temporal coordination."""
    phase_hours: float = 0.0         # current phase in 24-hour cycle
    period_hours: float = 24.0       # free-running period
    amplitude: float = 1.0           # rhythm amplitude (0.0 to 1.0)
    entrainment_strength: float = 0.8
    dawn_anticipation_active: bool = False
    dusk_anticipation_active: bool = False
    seasonal_phase: str = "equinox"  # "spring", "summer", "autumn", "winter", "equinox"
    photoperiod_memory_days: int = 0
```

---

## Relationship Mappings

### Cross-Form Data Exchange

```python
@dataclass
class MycorrhizalNetworkState:
    """Data exchanged with Form 32 (Fungal Intelligence) via mycorrhizal networks."""
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connected_plant_ids: List[str] = field(default_factory=list)
    fungal_partner_ids: List[str] = field(default_factory=list)
    carbon_flow_direction: Dict[str, str] = field(default_factory=dict)
    nutrient_transfer_rates: Dict[str, float] = field(default_factory=dict)
    defense_signals_received: List[ChemicalSignal] = field(default_factory=list)
    network_topology: str = "unknown"  # "hub_spoke", "distributed", "linear"
    kin_recognition_active: bool = False
    network_health: float = 1.0


@dataclass
class SwarmIntelligenceExchange:
    """Data exchanged with Form 33 (Swarm Intelligence) for distributed processing parallels."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    distributed_decision_type: str = ""
    agent_count: int = 0  # number of meristems or root tips acting as agents
    consensus_metric: float = 0.0
    local_rule_set: List[str] = field(default_factory=list)
    emergent_behavior_detected: bool = False
    collective_fitness_estimate: float = 0.0


@dataclass
class AnimalCognitionExchange:
    """Data exchanged with Form 30 (Animal Cognition) for comparative learning studies."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    learning_type_compared: LearningMechanism = LearningMechanism.HABITUATION
    plant_performance_metric: float = 0.0
    animal_baseline_metric: float = 0.0
    convergence_indicators: List[str] = field(default_factory=list)
    neural_analog_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class FolkWisdomExchange:
    """Data exchanged with Form 29 (Folk Wisdom) for indigenous plant knowledge integration."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_tradition: str = ""         # "Amazonian", "Celtic", "Vedic", etc.
    plant_species: str = ""
    traditional_properties: List[str] = field(default_factory=list)
    empirical_validation_status: str = "unvalidated"
    cultural_sensitivity_level: str = "high"
    attribution_source: str = ""
    consent_obtained: bool = False


@dataclass
class CrossFormMessage:
    """Generic message structure for cross-form data exchange."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: int = 31
    target_form: int = 0
    message_type: str = ""    # "data_request", "data_response", "event_notification", "sync"
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
class DataConstraints:
    """Validation constraints for plant intelligence data structures."""

    # Environmental bounds
    TEMPERATURE_MIN_C: float = -40.0
    TEMPERATURE_MAX_C: float = 60.0
    HUMIDITY_MIN_PERCENT: float = 0.0
    HUMIDITY_MAX_PERCENT: float = 100.0
    CO2_MIN_PPM: float = 150.0
    CO2_MAX_PPM: float = 2000.0
    LIGHT_MAX_UMOL: float = 2500.0

    # Signal propagation bounds
    ACTION_POTENTIAL_MAX_VELOCITY: float = 0.04   # m/s
    VARIATION_POTENTIAL_MAX_VELOCITY: float = 0.01 # m/s
    VOC_MAX_DIFFUSION_RATE: float = 0.1           # m/s in still air

    # Learning bounds
    HABITUATION_MIN_EXPOSURES: int = 3
    ASSOCIATION_MIN_PAIRINGS: int = 5
    MEMORY_MAX_DURATION_DAYS: int = 40  # based on Gagliano experiments

    # System bounds
    MAX_SIMULTANEOUS_SIGNALS: int = 500
    MAX_MEMORY_RECORDS: int = 10000
    MAX_CROSS_FORM_MESSAGES_PER_SECOND: int = 100
    CONFIDENCE_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)
```
