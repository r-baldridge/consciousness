# Form 23: Lucid Dream Consciousness - Data Structures

## Core Data Models

### 1. Dream State Representation

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import numpy as np
import uuid

@dataclass
class SensoryEnvironment:
    """Represents the sensory environment of a dream state."""

    visual_complexity: float = 0.0
    auditory_richness: float = 0.0
    tactile_presence: float = 0.0
    olfactory_elements: float = 0.0
    gustatory_elements: float = 0.0
    proprioceptive_awareness: float = 0.0

    # Environmental characteristics
    lighting_conditions: Dict[str, float] = field(default_factory=dict)
    spatial_properties: Dict[str, Any] = field(default_factory=dict)
    temporal_flow: float = 1.0
    physics_consistency: float = 1.0

    # Reality markers
    anomaly_indicators: List[str] = field(default_factory=list)
    consistency_score: float = 1.0

@dataclass
class CognitiveState:
    """Represents cognitive functioning during dream state."""

    # Core cognitive functions
    working_memory_capacity: float = 1.0
    attention_focus: float = 1.0
    executive_control: float = 1.0
    critical_thinking: float = 1.0

    # Memory access
    episodic_memory_access: float = 1.0
    semantic_memory_access: float = 1.0
    procedural_memory_access: float = 1.0

    # Metacognitive awareness
    self_awareness_level: float = 0.0
    reality_monitoring: float = 0.0
    intention_tracking: float = 0.0

    # Emotional state
    emotional_intensity: float = 0.5
    emotional_regulation: float = 1.0
    emotional_coherence: float = 1.0

@dataclass
class DreamStateSnapshot:
    """Complete snapshot of dream state at a specific moment."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # State classification
    processing_state: str = "unknown"
    state_confidence: float = 0.0
    lucidity_level: float = 0.0

    # Environment and cognition
    sensory_environment: SensoryEnvironment = field(default_factory=SensoryEnvironment)
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)

    # Content information
    narrative_context: Dict[str, Any] = field(default_factory=dict)
    character_information: List[Dict[str, Any]] = field(default_factory=list)
    environmental_details: Dict[str, Any] = field(default_factory=dict)

    # Stability and transition information
    state_stability: float = 0.0
    transition_probability: Dict[str, float] = field(default_factory=dict)
    duration_in_state: float = 0.0
```

### 2. Lucidity and Awareness Structures

```python
class LucidityMarker(Enum):
    """Markers indicating different aspects of lucid awareness."""
    DREAM_RECOGNITION = "dream_recognition"
    REALITY_QUESTIONING = "reality_questioning"
    MEMORY_ACCESS = "memory_access"
    INTENTION_FORMATION = "intention_formation"
    CONTROL_ATTEMPT = "control_attempt"
    METACOGNITIVE_REFLECTION = "metacognitive_reflection"

@dataclass
class LucidityIndicator:
    """Individual indicator of lucid awareness."""

    marker_type: LucidityMarker
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    onset_time: datetime
    duration: float
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

@dataclass
class AwarenessProfile:
    """Comprehensive profile of awareness state."""

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Overall lucidity metrics
    overall_lucidity: float = 0.0
    lucidity_stability: float = 0.0
    awareness_quality: float = 0.0

    # Specific awareness components
    dream_recognition_strength: float = 0.0
    reality_testing_frequency: float = 0.0
    metacognitive_engagement: float = 0.0
    control_capability: float = 0.0

    # Active indicators
    active_indicators: List[LucidityIndicator] = field(default_factory=list)

    # Historical context
    previous_lucidity_episode: Optional[datetime] = None
    lucidity_trend: float = 0.0

    # Predictive information
    lucidity_maintenance_probability: float = 0.0
    expected_duration: float = 0.0

@dataclass
class ConsciousnessGradient:
    """Represents the gradient of consciousness across different dimensions."""

    # Awareness dimensions
    perceptual_awareness: float = 0.0
    cognitive_awareness: float = 0.0
    emotional_awareness: float = 0.0
    bodily_awareness: float = 0.0
    environmental_awareness: float = 0.0

    # Control dimensions
    attention_control: float = 0.0
    memory_control: float = 0.0
    narrative_control: float = 0.0
    environmental_control: float = 0.0
    character_control: float = 0.0

    # Meta-awareness
    state_awareness: float = 0.0
    process_awareness: float = 0.0
    intention_awareness: float = 0.0
```

### 3. Dream Content and Narrative Structures

```python
@dataclass
class DreamCharacter:
    """Represents a character within a dream scenario."""

    character_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None

    # Character properties
    appearance: Dict[str, Any] = field(default_factory=dict)
    personality_traits: List[str] = field(default_factory=list)
    relationship_to_dreamer: str = "unknown"

    # Behavioral characteristics
    behavior_patterns: List[str] = field(default_factory=list)
    dialogue_style: Dict[str, Any] = field(default_factory=dict)
    emotional_state: Dict[str, float] = field(default_factory=dict)

    # Reality assessment
    consistency_score: float = 1.0
    reality_violations: List[str] = field(default_factory=list)

    # Control information
    player_controlled: bool = False
    responsiveness_to_control: float = 0.0

@dataclass
class DreamScene:
    """Represents a scene or setting within a dream."""

    scene_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Scene properties
    location_description: str = ""
    environmental_details: Dict[str, Any] = field(default_factory=dict)
    atmospheric_qualities: Dict[str, float] = field(default_factory=dict)

    # Spatial information
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    navigational_elements: List[str] = field(default_factory=list)
    accessibility: Dict[str, bool] = field(default_factory=dict)

    # Temporal aspects
    time_of_day: Optional[str] = None
    temporal_anomalies: List[str] = field(default_factory=list)
    scene_duration: float = 0.0

    # Characters and objects
    present_characters: List[str] = field(default_factory=list)  # character_ids
    objects: List[Dict[str, Any]] = field(default_factory=list)
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class NarrativeElement:
    """Represents a narrative component of the dream story."""

    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: str = ""  # "event", "dialogue", "action", "thought", "emotion"

    # Content
    description: str = ""
    participants: List[str] = field(default_factory=list)  # character_ids
    emotional_tone: Dict[str, float] = field(default_factory=dict)

    # Narrative function
    plot_significance: float = 0.0
    character_development: Dict[str, Any] = field(default_factory=dict)
    thematic_relevance: List[str] = field(default_factory=list)

    # Temporal placement
    sequence_position: int = 0
    duration: float = 0.0
    causal_connections: List[str] = field(default_factory=list)  # element_ids

@dataclass
class DreamNarrative:
    """Complete narrative structure of a dream episode."""

    narrative_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Narrative components
    scenes: List[DreamScene] = field(default_factory=list)
    characters: List[DreamCharacter] = field(default_factory=list)
    narrative_elements: List[NarrativeElement] = field(default_factory=list)

    # Story structure
    narrative_arc: Dict[str, Any] = field(default_factory=dict)
    themes: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolutions: List[Dict[str, Any]] = field(default_factory=list)

    # Coherence and quality
    narrative_coherence: float = 0.0
    emotional_coherence: float = 0.0
    logical_consistency: float = 0.0

    # Control influence
    user_influenced_elements: List[str] = field(default_factory=list)  # element_ids
    control_success_rate: float = 0.0
```

### 4. Reality Testing and Validation Structures

```python
class AnomalyType(Enum):
    """Types of reality anomalies that can be detected."""
    PHYSICAL_IMPOSSIBILITY = "physical_impossibility"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SPATIAL_IMPOSSIBILITY = "spatial_impossibility"
    CHARACTER_INCONSISTENCY = "character_inconsistency"
    MEMORY_MISMATCH = "memory_mismatch"
    CAUSAL_VIOLATION = "causal_violation"

@dataclass
class RealityAnomaly:
    """Represents a detected anomaly in dream content."""

    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Anomaly classification
    anomaly_type: AnomalyType
    severity: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0

    # Description and context
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Dict[str, Any]] = None

    # Detection information
    detection_method: str = ""
    detection_confidence: float = 0.0

    # Impact assessment
    lucidity_trigger_potential: float = 0.0
    reality_testing_relevance: float = 0.0

@dataclass
class ConsistencyCheck:
    """Represents a consistency verification operation."""

    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Check parameters
    check_type: str = ""
    target_element: str = ""
    verification_criteria: Dict[str, Any] = field(default_factory=dict)

    # Results
    is_consistent: bool = True
    consistency_score: float = 1.0
    anomalies_detected: List[RealityAnomaly] = field(default_factory=list)

    # Processing information
    processing_time: float = 0.0
    computational_cost: float = 0.0

@dataclass
class RealityTestingSession:
    """Complete reality testing session data."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Testing configuration
    testing_scope: List[str] = field(default_factory=list)
    sensitivity_settings: Dict[str, float] = field(default_factory=dict)

    # Results
    consistency_checks: List[ConsistencyCheck] = field(default_factory=list)
    overall_reality_score: float = 1.0
    total_anomalies: int = 0

    # Impact on lucidity
    lucidity_triggers_activated: int = 0
    awareness_enhancement: float = 0.0
```

### 5. Control and Manipulation Structures

```python
class ControlDomain(Enum):
    """Domains of dream control capability."""
    ENVIRONMENTAL = "environmental"
    NARRATIVE = "narrative"
    CHARACTER = "character"
    SENSORY = "sensory"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    PERSPECTIVE = "perspective"

@dataclass
class ControlCapability:
    """Represents capability level in a specific control domain."""

    domain: ControlDomain
    proficiency_level: float = 0.0  # 0.0 to 1.0
    reliability: float = 0.0  # 0.0 to 1.0
    response_time: float = 0.0  # seconds

    # Historical performance
    success_rate: float = 0.0
    average_effort: float = 0.0
    learning_progress: float = 0.0

    # Constraints and limitations
    scope_limitations: List[str] = field(default_factory=list)
    stability_requirements: Dict[str, float] = field(default_factory=dict)

@dataclass
class ControlAction:
    """Represents a specific control action attempt."""

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Action specification
    control_domain: ControlDomain
    action_type: str = ""
    target_description: str = ""
    intended_outcome: Dict[str, Any] = field(default_factory=dict)

    # Execution details
    effort_required: float = 0.0
    execution_time: float = 0.0
    success: bool = False

    # Results
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    stability_impact: float = 0.0

    # Learning information
    difficulty_assessment: float = 0.0
    technique_used: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class ControlProfile:
    """Comprehensive profile of control capabilities."""

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Domain capabilities
    domain_capabilities: Dict[ControlDomain, ControlCapability] = field(default_factory=dict)

    # Overall metrics
    overall_control_level: float = 0.0
    control_stability: float = 0.0
    learning_rate: float = 0.0

    # Recent performance
    recent_actions: List[ControlAction] = field(default_factory=list)
    success_trends: Dict[ControlDomain, float] = field(default_factory=dict)

    # Optimal conditions
    optimal_lucidity_level: float = 0.0
    optimal_stability_level: float = 0.0
    preferred_techniques: Dict[ControlDomain, List[str]] = field(default_factory=dict)
```

### 6. Memory and Integration Structures

```python
class DreamMemoryType(Enum):
    """Types of dream-related memories."""
    EXPERIENTIAL = "experiential"  # Direct dream experience
    CONTROL = "control"  # Control actions and outcomes
    INSIGHT = "insight"  # Insights and realizations
    LEARNING = "learning"  # Skills and knowledge acquired
    EMOTIONAL = "emotional"  # Emotional experiences and processing
    THERAPEUTIC = "therapeutic"  # Therapeutic work and healing

@dataclass
class DreamMemoryFragment:
    """Individual fragment of dream memory."""

    fragment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Memory classification
    memory_type: DreamMemoryType
    content_category: str = ""
    significance_level: float = 0.0

    # Content
    description: str = ""
    sensory_details: Dict[str, Any] = field(default_factory=dict)
    emotional_context: Dict[str, float] = field(default_factory=dict)
    cognitive_context: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    vividness: float = 0.0
    accuracy_confidence: float = 0.0
    detail_preservation: float = 0.0

    # Connections
    related_fragments: List[str] = field(default_factory=list)  # fragment_ids
    causal_connections: List[str] = field(default_factory=list)
    thematic_connections: List[str] = field(default_factory=list)

@dataclass
class DreamMemoryCluster:
    """Cluster of related dream memory fragments."""

    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_time: datetime = field(default_factory=datetime.now)

    # Cluster composition
    fragments: List[DreamMemoryFragment] = field(default_factory=list)
    central_theme: str = ""
    coherence_score: float = 0.0

    # Integration information
    integration_status: str = "pending"  # "pending", "integrated", "archived"
    integration_quality: float = 0.0

    # Learning outcomes
    insights_extracted: List[str] = field(default_factory=list)
    skills_developed: List[str] = field(default_factory=list)
    knowledge_gained: List[str] = field(default_factory=list)

    # Therapeutic value
    therapeutic_progress: Dict[str, float] = field(default_factory=dict)
    healing_outcomes: List[str] = field(default_factory=list)

@dataclass
class IntegrationRecord:
    """Record of memory integration with autobiographical memory."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    integration_time: datetime = field(default_factory=datetime.now)

    # Source information
    dream_memory_cluster: str = ""  # cluster_id
    integration_trigger: str = ""

    # Integration process
    integration_method: str = ""
    processing_time: float = 0.0
    conflicts_resolved: int = 0

    # Outcomes
    autobiographical_connections: List[str] = field(default_factory=list)
    narrative_contributions: List[str] = field(default_factory=list)
    identity_impact: Dict[str, float] = field(default_factory=dict)

    # Quality assessment
    integration_completeness: float = 0.0
    coherence_maintenance: float = 0.0
    accuracy_preservation: float = 0.0
```

### 7. Session and Performance Tracking

```python
@dataclass
class LucidDreamSession:
    """Complete record of a lucid dreaming session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Session configuration
    intended_goals: List[str] = field(default_factory=list)
    target_lucidity_level: float = 0.0
    planned_activities: List[str] = field(default_factory=list)

    # Session progression
    state_snapshots: List[DreamStateSnapshot] = field(default_factory=list)
    awareness_profiles: List[AwarenessProfile] = field(default_factory=list)
    control_actions: List[ControlAction] = field(default_factory=list)

    # Content and narrative
    dream_narrative: Optional[DreamNarrative] = None
    reality_testing_sessions: List[RealityTestingSession] = field(default_factory=list)

    # Outcomes
    goals_achieved: List[str] = field(default_factory=list)
    insights_gained: List[str] = field(default_factory=list)
    skills_practiced: List[str] = field(default_factory=list)
    therapeutic_progress: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    overall_lucidity: float = 0.0
    control_success_rate: float = 0.0
    memory_quality: float = 0.0
    session_satisfaction: float = 0.0

@dataclass
class PerformanceHistogram:
    """Histogram of performance metrics over time."""

    metric_name: str = ""
    time_period: str = ""  # "daily", "weekly", "monthly"

    # Data points
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    # Statistical summary
    mean_value: float = 0.0
    median_value: float = 0.0
    std_deviation: float = 0.0
    trend_direction: float = 0.0  # -1 to 1, negative=declining, positive=improving

    # Quality indicators
    data_completeness: float = 0.0
    measurement_reliability: float = 0.0

@dataclass
class UserProfile:
    """Comprehensive user profile for lucid dreaming system."""

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_creation: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # User characteristics
    experience_level: str = "beginner"  # "beginner", "intermediate", "advanced"
    natural_lucidity_frequency: float = 0.0
    preferred_induction_methods: List[str] = field(default_factory=list)

    # Performance history
    sessions_completed: int = 0
    total_lucid_time: float = 0.0
    average_lucidity_level: float = 0.0
    control_proficiency: Dict[ControlDomain, float] = field(default_factory=dict)

    # Learning and adaptation
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    adaptation_parameters: Dict[str, float] = field(default_factory=dict)
    personalization_settings: Dict[str, Any] = field(default_factory=dict)

    # Goals and progress
    current_goals: List[str] = field(default_factory=list)
    goal_progress: Dict[str, float] = field(default_factory=dict)
    therapeutic_objectives: List[str] = field(default_factory=list)

    # Performance trends
    performance_histograms: Dict[str, PerformanceHistogram] = field(default_factory=dict)
    improvement_trajectory: Dict[str, float] = field(default_factory=dict)
```

These comprehensive data structures provide the foundation for representing all aspects of lucid dream consciousness, from basic state detection through advanced control capabilities and long-term learning outcomes. They support both real-time processing requirements and historical analysis needs for research and therapeutic applications.