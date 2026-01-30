# Developmental Consciousness Data Structures

## Overview

This document defines the core data structures for the Developmental Consciousness system (Form 35). These structures model the emergence, transformation, and decline of consciousness across the human lifespan, from prenatal development through end of life. They encode developmental stages, cognitive capacity trajectories, sensory and perceptual maturation, theory of mind emergence, metacognitive development, and end-of-life consciousness states. The structures integrate with the broader consciousness framework to provide developmental context for all other forms.

All data models use Python dataclasses with standard library typing. Enum types encode the discrete developmental stages, cognitive capacities, assessment methods, and critical period classifications that span the full ontogenetic trajectory of human consciousness.

---

## Core Data Models

### Developmental Stage Representation

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta, date
from enum import Enum, auto
import uuid


@dataclass
class DevelopmentalAge:
    """Representation of developmental age with multiple reference frames."""
    chronological_age_days: int = 0
    gestational_age_weeks: Optional[float] = None  # prenatal only
    developmental_age_equivalent_days: Optional[int] = None  # may differ from chronological
    biological_maturation_score: float = 0.0  # 0.0 to 1.0 normalized
    experience_adjusted_age: Optional[int] = None  # accounts for enrichment/deprivation
    birth_date: Optional[date] = None
    conception_date_estimate: Optional[date] = None
    is_prenatal: bool = False
    is_premature: bool = False
    prematurity_weeks: Optional[float] = None


@dataclass
class DevelopmentalStageProfile:
    """Complete profile of a developmental stage with associated capabilities."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: 'DevelopmentalStage' = None
    sub_stage: Optional[str] = None
    age_range_start_days: int = 0
    age_range_end_days: int = 0
    cognitive_capabilities: Dict[str, float] = field(default_factory=dict)
    sensory_capabilities: Dict[str, float] = field(default_factory=dict)
    motor_capabilities: Dict[str, float] = field(default_factory=dict)
    social_capabilities: Dict[str, float] = field(default_factory=dict)
    language_capabilities: Dict[str, float] = field(default_factory=dict)
    consciousness_indicators: Dict[str, float] = field(default_factory=dict)
    critical_periods_active: List[str] = field(default_factory=list)
    typical_milestones: List['DevelopmentalMilestone'] = field(default_factory=list)
    piaget_stage: Optional[str] = None
    attachment_style_influence: Optional[str] = None


@dataclass
class DevelopmentalMilestone:
    """Individual developmental milestone with timing and assessment data."""
    milestone_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    domain: str = ""  # "cognitive", "motor", "language", "social", "emotional", "consciousness"
    typical_age_days: int = 0
    age_range_min_days: int = 0
    age_range_max_days: int = 0
    achieved: bool = False
    achieved_at_days: Optional[int] = None
    assessment_method: str = ""
    confidence_score: float = 0.0
    prerequisite_milestones: List[str] = field(default_factory=list)
    dependent_milestones: List[str] = field(default_factory=list)
    clinical_significance: str = "standard"  # "standard", "critical", "red_flag"
```

### Consciousness Capacity Models

```python
@dataclass
class ConsciousnessCapacityProfile:
    """Composite profile of consciousness-related cognitive capacities."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    developmental_age: DevelopmentalAge = field(default_factory=DevelopmentalAge)
    sensory_awareness: 'SensoryAwarenessState' = None
    perceptual_integration: float = 0.0       # 0.0 to 1.0
    attentional_capacity: float = 0.0          # working memory slots equivalent
    object_permanence_level: float = 0.0       # 0.0 to 1.0
    self_recognition_level: float = 0.0        # 0.0 to 1.0
    theory_of_mind_level: float = 0.0          # 0.0 to 1.0
    metacognitive_capacity: float = 0.0        # 0.0 to 1.0
    temporal_consciousness: float = 0.0        # 0.0 to 1.0
    narrative_identity_coherence: float = 0.0  # 0.0 to 1.0
    abstract_reasoning_capacity: float = 0.0   # 0.0 to 1.0
    mortality_awareness: float = 0.0           # 0.0 to 1.0
    wisdom_integration: float = 0.0            # 0.0 to 1.0
    global_workspace_capacity: float = 0.0     # information integration measure
    overall_consciousness_level: float = 0.0   # composite estimate 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SensoryAwarenessState:
    """State of sensory awareness across modalities at a developmental stage."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    visual_acuity: float = 0.0         # normalized, 1.0 = adult 20/20
    auditory_sensitivity: float = 0.0   # normalized, 1.0 = adult typical
    tactile_sensitivity: float = 0.0
    proprioceptive_accuracy: float = 0.0
    vestibular_function: float = 0.0
    olfactory_sensitivity: float = 0.0
    gustatory_sensitivity: float = 0.0
    pain_sensitivity: float = 0.0
    interoceptive_awareness: float = 0.0
    cross_modal_integration: float = 0.0  # ability to integrate across senses
    sensory_gating: float = 0.0           # filtering of irrelevant input
    habituation_rate: float = 0.0         # rate of response decline to repeated stimuli


@dataclass
class CognitiveCapacityTrajectory:
    """Longitudinal trajectory of a single cognitive capacity over development."""
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capacity_name: str = ""
    measurements: List[Tuple[int, float]] = field(default_factory=list)  # (age_days, value)
    growth_curve_type: str = "sigmoid"  # "sigmoid", "linear", "step", "u_shaped", "inverted_u"
    onset_age_days: Optional[int] = None
    peak_age_days: Optional[int] = None
    plateau_age_days: Optional[int] = None
    decline_onset_days: Optional[int] = None
    growth_rate_parameter: float = 0.0
    asymptote_value: float = 1.0
    current_value: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
```

### Theory of Mind Structures

```python
@dataclass
class TheoryOfMindState:
    """State of theory of mind development and assessment."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    joint_attention_capacity: float = 0.0       # following/directing gaze
    intention_attribution: float = 0.0          # understanding others have goals
    desire_understanding: float = 0.0           # understanding others have wants
    belief_understanding: float = 0.0           # understanding others have beliefs
    false_belief_understanding: float = 0.0     # Sally-Anne test competence
    second_order_belief: float = 0.0            # beliefs about beliefs
    deception_capability: float = 0.0           # ability to intentionally deceive
    empathy_level: float = 0.0                  # affective perspective taking
    cognitive_empathy: float = 0.0              # cognitive perspective taking
    social_referencing: float = 0.0             # looking to others for cues
    shared_intentionality: float = 0.0          # Tomasello's shared goals
    assessment_results: List['ToMAssessment'] = field(default_factory=list)
    developmental_age: Optional[DevelopmentalAge] = None


@dataclass
class ToMAssessment:
    """Result of a specific theory of mind assessment task."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = ""  # "sally_anne", "smarties", "unexpected_transfer", "appearance_reality"
    task_type: str = ""  # "false_belief", "desire", "intention", "emotion", "knowledge_access"
    passed: bool = False
    response_latency_ms: float = 0.0
    confidence: float = 0.0
    age_at_assessment_days: int = 0
    assessment_method: str = ""  # "behavioral", "looking_time", "verbal", "ERP"
    notes: str = ""
```

### Metacognitive Development Structures

```python
@dataclass
class MetacognitiveState:
    """State of metacognitive development."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowing_about_knowing: float = 0.0          # awareness of own knowledge
    feeling_of_knowing: float = 0.0             # FOK judgments
    judgment_of_learning: float = 0.0           # JOL accuracy
    monitoring_accuracy: float = 0.0            # how well one tracks own cognition
    control_of_cognition: float = 0.0           # strategy use and regulation
    source_monitoring: float = 0.0              # knowing where knowledge came from
    reality_monitoring: float = 0.0             # distinguishing real from imagined
    introspective_accuracy: float = 0.0         # accuracy of self-reports
    confidence_calibration: float = 0.0         # alignment of confidence and accuracy
    cognitive_flexibility: float = 0.0          # ability to shift strategies
    planning_capacity: float = 0.0              # forward-looking behavior
    error_detection: float = 0.0                # recognizing own mistakes
    developmental_age: Optional[DevelopmentalAge] = None


@dataclass
class SelfRecognitionState:
    """State of self-recognition and self-concept development."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mirror_self_recognition: float = 0.0       # rouge test competence
    body_schema: float = 0.0                   # body ownership and boundaries
    self_other_distinction: float = 0.0        # distinguishing self from others
    personal_pronoun_use: float = 0.0          # "I", "me", "mine" usage
    autobiographical_memory: float = 0.0       # personal event recall
    self_continuity: float = 0.0               # sense of persistence over time
    narrative_self: float = 0.0                # story-based identity
    self_evaluation: float = 0.0               # ability to judge own performance
    social_self: float = 0.0                   # understanding how others see self
    existential_self: float = 0.0              # awareness of own existence
    age_at_mirror_recognition_days: Optional[int] = None
```

### End-of-Life Consciousness Structures

```python
@dataclass
class EndOfLifeState:
    """Consciousness state during end-of-life processes."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = ""  # "approaching_death", "active_dying", "near_death", "death_threshold"
    lucidity_level: float = 0.0              # 0.0 to 1.0
    terminal_lucidity_event: bool = False    # paradoxical lucidity near death
    pain_awareness: float = 0.0
    environmental_awareness: float = 0.0
    social_awareness: float = 0.0
    temporal_orientation: float = 0.0
    near_death_experience_indicators: Dict[str, float] = field(default_factory=dict)
    greyson_scale_score: Optional[float] = None   # Greyson NDE Scale
    consciousness_fading_pattern: str = ""  # "gradual", "fluctuating", "sudden", "plateau"
    last_responsive_modality: Optional[str] = None  # "visual", "auditory", "tactile"
    dignity_preservation_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NearDeathExperienceRecord:
    """Record of near-death experience indicators and phenomenology."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    greyson_scale_total: float = 0.0
    cognitive_features: Dict[str, float] = field(default_factory=dict)
    affective_features: Dict[str, float] = field(default_factory=dict)
    paranormal_features: Dict[str, float] = field(default_factory=dict)
    transcendental_features: Dict[str, float] = field(default_factory=dict)
    life_review_reported: bool = False
    out_of_body_reported: bool = False
    tunnel_experience_reported: bool = False
    light_experience_reported: bool = False
    deceased_encounter_reported: bool = False
    border_or_limit_reported: bool = False
    time_distortion_reported: bool = False
    verifiable_perception_claims: List[str] = field(default_factory=list)
    clinical_context: str = ""  # "cardiac_arrest", "trauma", "illness", "surgical"
    duration_estimate_minutes: Optional[float] = None
    timestamp: Optional[datetime] = None
```

### Neural Maturation Structures

```python
@dataclass
class NeuralMaturationState:
    """State of neural development and maturation relevant to consciousness."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gestational_week: Optional[float] = None
    cortical_layers_formed: int = 0         # 0-6 layers
    thalamocortical_connectivity: float = 0.0  # 0.0 to 1.0
    callosal_myelination: float = 0.0       # 0.0 to 1.0
    prefrontal_maturation: float = 0.0      # 0.0 to 1.0
    synaptic_density_normalized: float = 0.0  # relative to adult, can exceed 1.0
    pruning_progress: float = 0.0           # 0.0 to 1.0
    myelination_progress: float = 0.0       # 0.0 to 1.0
    eeg_pattern: str = ""                   # "trace_discontinue", "trace_alternant", "continuous"
    dominant_frequency_Hz: float = 0.0
    sleep_wake_differentiation: float = 0.0
    pain_pathway_maturity: float = 0.0
    sensory_cortex_maturity: Dict[str, float] = field(default_factory=dict)
    default_mode_network_emergence: float = 0.0
    global_workspace_connectivity: float = 0.0
```

---

## Enumeration Types

```python
class DevelopmentalStage(Enum):
    """Major developmental stages of consciousness."""
    EARLY_PRENATAL = auto()        # Conception to 20 weeks
    LATE_PRENATAL = auto()         # 20-40 weeks gestational
    NEONATAL = auto()              # Birth to 4 weeks
    EARLY_INFANCY = auto()         # 1-6 months
    LATE_INFANCY = auto()          # 6-24 months
    TODDLER = auto()               # 2-3 years
    EARLY_CHILDHOOD = auto()       # 3-6 years
    MIDDLE_CHILDHOOD = auto()      # 6-12 years
    ADOLESCENCE = auto()           # 12-18 years
    YOUNG_ADULTHOOD = auto()       # 18-35 years
    MIDDLE_ADULTHOOD = auto()      # 35-65 years
    LATE_ADULTHOOD = auto()        # 65+ years
    END_OF_LIFE = auto()           # Terminal phase


class PiagetStage(Enum):
    """Piaget's stages of cognitive development."""
    SENSORIMOTOR = auto()          # Birth to ~2 years
    PREOPERATIONAL = auto()        # ~2 to ~7 years
    CONCRETE_OPERATIONAL = auto()  # ~7 to ~11 years
    FORMAL_OPERATIONAL = auto()    # ~11+ years


class ConsciousnessCapacity(Enum):
    """Trackable consciousness-related capacities."""
    SENSORY_AWARENESS = auto()
    PERCEPTUAL_BINDING = auto()
    ATTENTIONAL_SELECTION = auto()
    WORKING_MEMORY = auto()
    OBJECT_PERMANENCE = auto()
    SELF_RECOGNITION = auto()
    JOINT_ATTENTION = auto()
    THEORY_OF_MIND = auto()
    METACOGNITION = auto()
    ABSTRACT_REASONING = auto()
    TEMPORAL_CONSCIOUSNESS = auto()
    NARRATIVE_IDENTITY = auto()
    MORTALITY_AWARENESS = auto()
    WISDOM = auto()


class CriticalPeriodType(Enum):
    """Types of critical/sensitive periods in development."""
    VISUAL_ACUITY = auto()
    BINOCULAR_VISION = auto()
    LANGUAGE_ACQUISITION = auto()
    ATTACHMENT_FORMATION = auto()
    PHONEME_DISCRIMINATION = auto()
    ABSOLUTE_PITCH = auto()
    SOCIAL_COGNITION = auto()
    EMOTIONAL_REGULATION = auto()
    EXECUTIVE_FUNCTION = auto()
    MORAL_REASONING = auto()


class AssessmentParadigm(Enum):
    """Experimental paradigms for assessing infant and child cognition."""
    HABITUATION = auto()
    VIOLATION_OF_EXPECTATION = auto()
    PREFERENTIAL_LOOKING = auto()
    VISUAL_CLIFF = auto()
    ROUGE_TEST = auto()
    SALLY_ANNE = auto()
    SMARTIES_TEST = auto()
    A_NOT_B_TASK = auto()
    OBJECT_SEARCH = auto()
    DELAYED_IMITATION = auto()
    POINTING_TASK = auto()
    STRANGE_SITUATION = auto()
    DIMENSIONAL_CHANGE_CARD_SORT = auto()


class CognitiveDevelopmentDomain(Enum):
    """Domains of cognitive development tracked."""
    PERCEPTION = auto()
    ATTENTION = auto()
    MEMORY = auto()
    LANGUAGE = auto()
    EXECUTIVE_FUNCTION = auto()
    SOCIAL_COGNITION = auto()
    EMOTIONAL_REGULATION = auto()
    MOTOR_COGNITION = auto()
    SPATIAL_REASONING = auto()
    NUMERICAL_COGNITION = auto()
    MORAL_REASONING = auto()
    CAUSAL_REASONING = auto()


class EndOfLifePhase(Enum):
    """Phases of consciousness change at end of life."""
    APPROACHING_DEATH = auto()     # Weeks to months
    ACTIVE_DYING = auto()          # Days to hours
    NEAR_DEATH = auto()            # Minutes
    DEATH_THRESHOLD = auto()       # Transition point
    TERMINAL_LUCIDITY = auto()     # Paradoxical return of clarity


class NeuralDevelopmentPhase(Enum):
    """Phases of neural development relevant to consciousness emergence."""
    NEURULATION = auto()           # Neural tube formation
    PROLIFERATION = auto()         # Neuronal multiplication
    MIGRATION = auto()             # Neurons move to cortical positions
    DIFFERENTIATION = auto()       # Neurons specialize
    SYNAPTOGENESIS = auto()        # Synapse formation (peaks early childhood)
    MYELINATION = auto()           # Axon insulation (continues to ~25 years)
    PRUNING = auto()               # Synaptic elimination
    DEGENERATION = auto()          # Age-related decline
```

---

## Input/Output Structures

### System Input Structures

```python
@dataclass
class DevelopmentalConsciousnessInput:
    """Top-level input to the developmental consciousness processing system."""
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    developmental_age: DevelopmentalAge = field(default_factory=DevelopmentalAge)
    neural_maturation: Optional[NeuralMaturationState] = None
    behavioral_observations: List['BehavioralObservation'] = field(default_factory=list)
    assessment_results: List[ToMAssessment] = field(default_factory=list)
    environmental_context: 'DevelopmentalEnvironment' = None
    cross_form_inputs: Dict[str, any] = field(default_factory=dict)
    clinical_context: Optional[Dict[str, any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_priority: str = "normal"


@dataclass
class BehavioralObservation:
    """Observation of behavior relevant to consciousness assessment."""
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    behavior_type: str = ""  # "looking_time", "reaching", "pointing", "vocalizing", "imitating"
    domain: str = ""  # cognitive development domain
    paradigm: Optional[str] = None  # assessment paradigm used
    stimulus_description: str = ""
    response_description: str = ""
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    frequency: float = 0.0
    age_at_observation_days: int = 0
    observer_reliability: float = 0.0
    context: str = ""


@dataclass
class DevelopmentalEnvironment:
    """Environmental context relevant to developmental trajectory."""
    environment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enrichment_level: float = 0.5          # 0.0 (deprived) to 1.0 (highly enriched)
    social_interaction_frequency: float = 0.5
    language_exposure_hours_per_day: float = 0.0
    caregiver_responsiveness: float = 0.5
    attachment_security: float = 0.5
    nutritional_adequacy: float = 1.0
    sleep_quality: float = 0.5
    stress_exposure_level: float = 0.0     # 0.0 (none) to 1.0 (severe)
    cultural_context: str = ""
    educational_environment: str = ""
    peer_interaction_frequency: float = 0.0
    screen_time_hours_per_day: float = 0.0
```

### System Output Structures

```python
@dataclass
class DevelopmentalConsciousnessOutput:
    """Top-level output from the developmental consciousness processing system."""
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_id: str = ""
    current_stage: DevelopmentalStage = DevelopmentalStage.NEONATAL
    piaget_stage: Optional[PiagetStage] = None
    consciousness_capacity_profile: ConsciousnessCapacityProfile = field(
        default_factory=ConsciousnessCapacityProfile
    )
    milestone_status: List[DevelopmentalMilestone] = field(default_factory=list)
    capacity_trajectories: List[CognitiveCapacityTrajectory] = field(default_factory=list)
    theory_of_mind_state: Optional[TheoryOfMindState] = None
    metacognitive_state: Optional[MetacognitiveState] = None
    self_recognition_state: Optional[SelfRecognitionState] = None
    end_of_life_state: Optional[EndOfLifeState] = None
    developmental_alerts: List['DevelopmentalAlert'] = field(default_factory=list)
    stage_transition_probability: float = 0.0
    next_expected_milestone: Optional[str] = None
    cross_form_outputs: Dict[str, any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DevelopmentalAlert:
    """Alert for developmental concerns or notable events."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = ""  # "delayed_milestone", "regression", "precocious", "critical_period_closing"
    severity: str = "informational"  # "informational", "warning", "critical"
    domain: str = ""
    description: str = ""
    recommended_action: str = ""
    age_at_alert_days: int = 0
    confidence: float = 0.0
    ethical_sensitivity: str = "standard"  # "standard", "elevated", "high"
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Internal State Structures

```python
@dataclass
class DevelopmentalTrajectoryState:
    """Internal state tracking the full developmental trajectory."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_age: DevelopmentalAge = field(default_factory=DevelopmentalAge)
    current_stage: DevelopmentalStage = DevelopmentalStage.NEONATAL
    stage_history: List[Tuple[DevelopmentalStage, int]] = field(default_factory=list)
    milestone_registry: Dict[str, DevelopmentalMilestone] = field(default_factory=dict)
    active_critical_periods: List[CriticalPeriodType] = field(default_factory=list)
    capacity_curves: Dict[str, CognitiveCapacityTrajectory] = field(default_factory=dict)
    regression_events: List[Dict[str, any]] = field(default_factory=list)
    environmental_history: List[DevelopmentalEnvironment] = field(default_factory=list)
    cumulative_experience_score: float = 0.0


@dataclass
class ConsciousnessEmergenceModel:
    """Model of how consciousness emerges and develops."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thalamocortical_threshold: float = 0.3   # connectivity level for minimal consciousness
    current_thalamocortical_level: float = 0.0
    information_integration_phi: float = 0.0
    global_workspace_active: bool = False
    recurrent_processing_level: float = 0.0
    minimal_consciousness_achieved: bool = False
    access_consciousness_achieved: bool = False
    phenomenal_consciousness_estimate: float = 0.0
    first_consciousness_estimate_week: Optional[float] = None
    consciousness_emergence_confidence: float = 0.0


@dataclass
class LifespanProgressModel:
    """Model tracking position across the full lifespan."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    lifespan_fraction: float = 0.0    # 0.0 = conception, 1.0 = death
    peak_capacity_reached: Dict[str, bool] = field(default_factory=dict)
    decline_onset_detected: Dict[str, bool] = field(default_factory=dict)
    cognitive_reserve_estimate: float = 0.0
    brain_age_vs_chronological: float = 0.0  # positive = older than expected
    successful_aging_indicators: Dict[str, float] = field(default_factory=dict)
    dementia_risk_factors: Dict[str, float] = field(default_factory=dict)
    end_of_life_proximity: float = 0.0  # 0.0 = far, 1.0 = imminent
```

---

## Relationship Mappings

### Cross-Form Data Exchange

```python
@dataclass
class InteroceptiveExchange:
    """Data exchanged with Form 6 (Interoceptive Consciousness) for body awareness development."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    body_awareness_level: float = 0.0
    interoceptive_accuracy: float = 0.0
    body_schema_maturity: float = 0.0
    pain_awareness_level: float = 0.0
    hunger_thirst_discrimination: float = 0.0
    developmental_stage: Optional[DevelopmentalStage] = None


@dataclass
class EmotionalDevelopmentExchange:
    """Data exchanged with Form 7 (Emotional Consciousness) for emotional development tracking."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    basic_emotion_recognition: float = 0.0
    complex_emotion_understanding: float = 0.0
    emotion_regulation_capacity: float = 0.0
    empathy_development_level: float = 0.0
    emotional_vocabulary_size: int = 0
    attachment_security_score: float = 0.0
    developmental_stage: Optional[DevelopmentalStage] = None


@dataclass
class SelfRecognitionExchange:
    """Data exchanged with Form 10 (Self-Recognition) for self-awareness emergence tracking."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mirror_recognition_achieved: bool = False
    self_other_distinction_level: float = 0.0
    autobiographical_memory_capacity: float = 0.0
    narrative_self_coherence: float = 0.0
    existential_awareness_level: float = 0.0
    developmental_stage: Optional[DevelopmentalStage] = None
    age_at_mirror_recognition_days: Optional[int] = None


@dataclass
class MetaConsciousnessExchange:
    """Data exchanged with Form 11 (Meta-Consciousness) for metacognition development tracking."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metacognitive_capacity: float = 0.0
    monitoring_accuracy: float = 0.0
    control_of_cognition: float = 0.0
    introspective_accuracy: float = 0.0
    error_detection_capacity: float = 0.0
    developmental_stage: Optional[DevelopmentalStage] = None


@dataclass
class NarrativeExchange:
    """Data exchanged with Form 12 (Narrative Consciousness) for autobiographical development."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    narrative_capacity: float = 0.0
    autobiographical_memory_onset_days: Optional[int] = None
    childhood_amnesia_boundary_days: Optional[int] = None
    life_story_coherence: float = 0.0
    temporal_self_continuity: float = 0.0
    generativity_vs_stagnation: Optional[float] = None
    ego_integrity_vs_despair: Optional[float] = None
    developmental_stage: Optional[DevelopmentalStage] = None


@dataclass
class CrossFormMessage:
    """Generic message structure for cross-form data exchange."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: int = 35
    target_form: int = 0
    message_type: str = ""  # "data_request", "data_response", "event_notification", "sync"
    payload: Dict[str, any] = field(default_factory=dict)
    priority: int = 5
    requires_response: bool = False
    ttl_seconds: int = 300
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    ethical_sensitivity: str = "standard"
```

---

## Data Validation and Constraints

```python
@dataclass
class DevelopmentalDataConstraints:
    """Validation constraints for developmental consciousness data structures."""

    # Age bounds
    GESTATIONAL_AGE_MIN_WEEKS: float = 4.0
    GESTATIONAL_AGE_MAX_WEEKS: float = 44.0
    CHRONOLOGICAL_AGE_MAX_DAYS: int = 43800  # ~120 years
    PREMATURE_THRESHOLD_WEEKS: float = 37.0

    # Capacity score bounds
    CAPACITY_SCORE_MIN: float = 0.0
    CAPACITY_SCORE_MAX: float = 1.0
    SYNAPTIC_DENSITY_MAX_NORMALIZED: float = 2.0  # can exceed adult levels

    # Theory of mind timing bounds (typical development)
    JOINT_ATTENTION_ONSET_MIN_DAYS: int = 180    # ~6 months
    JOINT_ATTENTION_ONSET_MAX_DAYS: int = 540    # ~18 months
    FALSE_BELIEF_ONSET_MIN_DAYS: int = 1095      # ~3 years
    FALSE_BELIEF_ONSET_MAX_DAYS: int = 1825      # ~5 years
    MIRROR_RECOGNITION_MIN_DAYS: int = 365       # ~12 months
    MIRROR_RECOGNITION_MAX_DAYS: int = 730       # ~24 months

    # Neural maturation bounds
    CORTICAL_LAYERS_MAX: int = 6
    THALAMOCORTICAL_ONSET_WEEK: float = 23.0
    MYELINATION_COMPLETION_YEARS: float = 25.0

    # Greyson NDE Scale bounds
    GREYSON_SCALE_MIN: float = 0.0
    GREYSON_SCALE_MAX: float = 32.0
    GREYSON_NDE_THRESHOLD: float = 7.0  # score >= 7 classified as NDE

    # System bounds
    MAX_MILESTONES_TRACKED: int = 500
    MAX_TRAJECTORY_MEASUREMENTS: int = 10000
    MAX_BEHAVIORAL_OBSERVATIONS: int = 50000
    MAX_CROSS_FORM_MESSAGES_PER_SECOND: int = 100
    CONFIDENCE_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)
```
