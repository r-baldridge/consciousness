# Form 22: Dream Consciousness - Data Models

## Overview

This document defines the comprehensive data models for dream consciousness systems, including dream state representations, memory integration structures, sleep cycle data, and consciousness level specifications. These models ensure consistent data handling and interoperability across all dream consciousness components.

## Core Data Models

### Dream State Models

#### DreamConsciousnessState

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid

@dataclass
class DreamConsciousnessState:
    """Primary data model for dream consciousness state representation"""

    # Identity and temporal information
    dream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration: timedelta = field(default=timedelta(0))

    # Sleep context
    sleep_phase: 'SleepPhase' = None
    sleep_cycle_number: int = 0
    circadian_phase: float = 0.0  # 0.0-1.0 representing 24-hour cycle

    # Dream content
    dream_narrative: 'DreamNarrative' = field(default=None)
    sensory_experience: 'SensoryExperience' = field(default=None)
    emotional_content: 'EmotionalContent' = field(default=None)

    # Consciousness characteristics
    consciousness_level: float = 0.5  # 0.0-1.0 scale
    lucidity_level: float = 0.0  # 0.0-1.0 scale
    critical_thinking_level: float = 0.3  # 0.0-1.0 scale
    self_awareness_level: float = 0.4  # 0.0-1.0 scale

    # Memory integration
    memory_consolidation: 'MemoryConsolidation' = field(default=None)
    episodic_memories: List['EpisodicMemory'] = field(default_factory=list)
    semantic_integration: 'SemanticIntegration' = field(default=None)

    # Quality metrics
    narrative_coherence: float = 0.0  # 0.0-1.0 scale
    sensory_vividness: float = 0.0  # 0.0-1.0 scale
    emotional_intensity: float = 0.0  # 0.0-1.0 scale
    reality_distortion: float = 0.0  # 0.0-1.0 scale

    # Integration data
    cross_form_connections: Dict[str, Any] = field(default_factory=dict)
    integration_status: 'IntegrationStatus' = None

    # Metadata
    generation_method: str = "default"
    safety_flags: List[str] = field(default_factory=list)
    therapeutic_markers: Dict[str, Any] = field(default_factory=dict)
    research_annotations: Dict[str, Any] = field(default_factory=dict)

class SleepPhase(Enum):
    WAKE = "wake"
    NREM_STAGE_1 = "nrem_1"
    NREM_STAGE_2 = "nrem_2"
    NREM_STAGE_3 = "nrem_3"
    REM = "rem"
    TRANSITION = "transition"
```

#### DreamNarrative

```python
@dataclass
class DreamNarrative:
    """Model for dream narrative structure and content"""

    narrative_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    narrative_type: 'NarrativeType' = None

    # Narrative structure
    story_elements: List['StoryElement'] = field(default_factory=list)
    character_profiles: List['DreamCharacter'] = field(default_factory=list)
    setting_descriptions: List['DreamSetting'] = field(default_factory=list)
    plot_progression: List['PlotPoint'] = field(default_factory=list)

    # Narrative qualities
    coherence_score: float = 0.0  # 0.0-1.0 scale
    complexity_level: float = 0.0  # 0.0-1.0 scale
    temporal_consistency: float = 0.0  # 0.0-1.0 scale
    logical_consistency: float = 0.0  # 0.0-1.0 scale

    # Content themes
    primary_themes: List[str] = field(default_factory=list)
    symbolic_content: List['SymbolicElement'] = field(default_factory=list)
    archetypal_patterns: List['ArchetypalPattern'] = field(default_factory=list)

    # Narrative dynamics
    tension_curve: List[float] = field(default_factory=list)  # Tension over time
    pacing_rhythm: List[float] = field(default_factory=list)  # Pacing changes
    perspective_shifts: List['PerspectiveShift'] = field(default_factory=list)

    # Memory connections
    autobiographical_elements: List['AutobiographicalElement'] = field(default_factory=list)
    recent_memory_incorporations: List['MemoryIncorporation'] = field(default_factory=list)
    knowledge_integrations: List['KnowledgeIntegration'] = field(default_factory=list)

class NarrativeType(Enum):
    LINEAR = "linear"
    NON_LINEAR = "non_linear"
    FRAGMENTED = "fragmented"
    CYCLICAL = "cyclical"
    SURREAL = "surreal"
    LUCID = "lucid"
```

#### SensoryExperience

```python
@dataclass
class SensoryExperience:
    """Model for multi-modal sensory experiences in dreams"""

    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Visual experience
    visual_content: 'VisualContent' = field(default=None)
    visual_clarity: float = 0.0  # 0.0-1.0 scale
    color_saturation: float = 0.0  # 0.0-1.0 scale
    spatial_coherence: float = 0.0  # 0.0-1.0 scale

    # Auditory experience
    auditory_content: 'AudioContent' = field(default=None)
    audio_clarity: float = 0.0  # 0.0-1.0 scale
    audio_complexity: float = 0.0  # 0.0-1.0 scale

    # Tactile experience
    tactile_sensations: List['TactileSensation'] = field(default_factory=list)
    tactile_intensity: float = 0.0  # 0.0-1.0 scale

    # Olfactory experience
    olfactory_content: List['OlfactoryStimulus'] = field(default_factory=list)
    olfactory_intensity: float = 0.0  # 0.0-1.0 scale

    # Gustatory experience
    gustatory_content: List['GustatoryStimulus'] = field(default_factory=list)
    gustatory_intensity: float = 0.0  # 0.0-1.0 scale

    # Proprioceptive experience
    body_schema: 'BodySchema' = field(default=None)
    movement_sensations: List['MovementSensation'] = field(default_factory=list)

    # Synesthetic experiences
    synesthetic_mappings: List['SynestheticMapping'] = field(default_factory=list)

    # Overall sensory integration
    multi_modal_coherence: float = 0.0  # 0.0-1.0 scale
    sensory_synchronization: float = 0.0  # 0.0-1.0 scale
    overall_vividness: float = 0.0  # 0.0-1.0 scale

@dataclass
class VisualContent:
    scenes: List['VisualScene'] = field(default_factory=list)
    objects: List['VisualObject'] = field(default_factory=list)
    lighting_conditions: 'LightingConditions' = field(default=None)
    perspective_type: str = "first_person"
    field_of_view: float = 180.0  # degrees
    depth_perception: float = 0.8  # 0.0-1.0 scale
```

#### EmotionalContent

```python
@dataclass
class EmotionalContent:
    """Model for emotional aspects of dream experiences"""

    emotion_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Primary emotions
    primary_emotions: List['EmotionInstance'] = field(default_factory=list)
    emotional_trajectory: List['EmotionalState'] = field(default_factory=list)
    peak_emotional_moments: List['PeakEmotion'] = field(default_factory=list)

    # Emotional characteristics
    overall_valence: float = 0.0  # -1.0 to 1.0 (negative to positive)
    overall_arousal: float = 0.0  # 0.0-1.0 scale
    emotional_complexity: float = 0.0  # 0.0-1.0 scale
    emotional_stability: float = 0.0  # 0.0-1.0 scale

    # Emotional regulation
    regulation_attempts: List['EmotionRegulation'] = field(default_factory=list)
    regulation_effectiveness: float = 0.0  # 0.0-1.0 scale

    # Safety and well-being
    distress_indicators: List['DistressIndicator'] = field(default_factory=list)
    well_being_markers: List['WellBeingMarker'] = field(default_factory=list)
    safety_threshold_breaches: List['SafetyBreach'] = field(default_factory=list)

    # Therapeutic relevance
    therapeutic_significance: float = 0.0  # 0.0-1.0 scale
    processing_indicators: List['ProcessingIndicator'] = field(default_factory=list)

@dataclass
class EmotionInstance:
    emotion_type: str  # fear, joy, anger, sadness, etc.
    intensity: float  # 0.0-1.0 scale
    duration: timedelta
    onset_time: float  # seconds from dream start
    trigger_event: Optional[str] = None
    physiological_correlates: Dict[str, float] = field(default_factory=dict)
```

### Memory Integration Models

#### MemoryConsolidation

```python
@dataclass
class MemoryConsolidation:
    """Model for memory consolidation during dreams"""

    consolidation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Consolidation targets
    target_memories: List['TargetMemory'] = field(default_factory=list)
    consolidation_goals: List['ConsolidationGoal'] = field(default_factory=list)

    # Consolidation processes
    replay_sessions: List['MemoryReplay'] = field(default_factory=list)
    integration_processes: List['IntegrationProcess'] = field(default_factory=list)
    strengthening_activities: List['StrengtheningActivity'] = field(default_factory=list)

    # Consolidation outcomes
    consolidation_success_rate: float = 0.0  # 0.0-1.0 scale
    memory_strength_changes: Dict[str, float] = field(default_factory=dict)
    new_associations_formed: List['MemoryAssociation'] = field(default_factory=list)

    # Quality metrics
    consolidation_efficiency: float = 0.0  # 0.0-1.0 scale
    interference_level: float = 0.0  # 0.0-1.0 scale
    consolidation_completeness: float = 0.0  # 0.0-1.0 scale

@dataclass
class EpisodicMemory:
    """Model for episodic memory representation in dreams"""

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Memory content
    event_description: str = ""
    temporal_context: datetime = field(default_factory=datetime.utcnow)
    spatial_context: 'SpatialContext' = field(default=None)
    participants: List[str] = field(default_factory=list)
    sensory_details: 'SensoryDetails' = field(default=None)

    # Memory characteristics
    memory_strength: float = 0.0  # 0.0-1.0 scale
    emotional_significance: float = 0.0  # 0.0-1.0 scale
    personal_relevance: float = 0.0  # 0.0-1.0 scale

    # Dream integration
    dream_incorporation_method: str = ""
    modification_level: float = 0.0  # 0.0-1.0 scale
    symbolic_transformation: List['SymbolicTransformation'] = field(default_factory=list)

    # Consolidation tracking
    pre_dream_strength: float = 0.0
    post_dream_strength: float = 0.0
    consolidation_delta: float = 0.0
```

### Sleep Cycle Models

#### SleepCycleData

```python
@dataclass
class SleepCycleData:
    """Model for sleep cycle information and dream timing"""

    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Cycle structure
    cycle_number: int = 0
    cycle_start_time: datetime = field(default_factory=datetime.utcnow)
    cycle_duration: timedelta = field(default=timedelta(0))

    # Sleep stages
    stage_progression: List['SleepStage'] = field(default_factory=list)
    rem_periods: List['REMPeriod'] = field(default_factory=list)
    nrem_periods: List['NREMPeriod'] = field(default_factory=list)

    # Physiological data
    brain_activity: 'BrainActivityData' = field(default=None)
    autonomic_measures: 'AutonomicMeasures' = field(default=None)
    sleep_architecture: 'SleepArchitecture' = field(default=None)

    # Dream occurrence
    dream_episodes: List['DreamEpisode'] = field(default_factory=list)
    dream_density: float = 0.0  # dreams per hour
    dream_intensity_progression: List[float] = field(default_factory=list)

    # Quality metrics
    sleep_efficiency: float = 0.0  # 0.0-1.0 scale
    stage_transition_quality: float = 0.0  # 0.0-1.0 scale
    circadian_alignment: float = 0.0  # 0.0-1.0 scale

@dataclass
class REMPeriod:
    """Model for REM sleep periods"""

    rem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    duration: timedelta
    rem_density: float = 0.0  # eye movements per minute

    # Physiological characteristics
    eeg_patterns: 'EEGPatterns' = field(default=None)
    muscle_atonia_level: float = 0.0  # 0.0-1.0 scale
    autonomic_activity: 'AutonomicActivity' = field(default=None)

    # Dream characteristics
    dream_reports: List['DreamReport'] = field(default_factory=list)
    dream_vividness: float = 0.0  # 0.0-1.0 scale
    dream_complexity: float = 0.0  # 0.0-1.0 scale
    lucidity_potential: float = 0.0  # 0.0-1.0 scale
```

### Consciousness Level Models

#### ConsciousnessLevelMetrics

```python
@dataclass
class ConsciousnessLevelMetrics:
    """Model for consciousness level measurement and tracking"""

    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Core consciousness dimensions
    awareness_level: float = 0.0  # 0.0-1.0 scale
    attention_focus: float = 0.0  # 0.0-1.0 scale
    self_recognition: float = 0.0  # 0.0-1.0 scale
    meta_cognition: float = 0.0  # 0.0-1.0 scale

    # Dream-specific consciousness features
    dream_awareness: float = 0.0  # awareness of being in a dream
    reality_testing: float = 0.0  # ability to question reality
    volitional_control: float = 0.0  # control over dream content
    memory_access: float = 0.0  # access to waking memories

    # Cognitive functions
    working_memory_capacity: float = 0.0
    executive_function: float = 0.0
    logical_reasoning: float = 0.0
    creative_processing: float = 0.0

    # Integration metrics
    unified_experience: float = 0.0  # integration of dream elements
    temporal_continuity: float = 0.0  # continuity over time
    narrative_coherence: float = 0.0  # story coherence

    # Transition indicators
    lucidity_markers: List['LucidityMarker'] = field(default_factory=list)
    transition_triggers: List['TransitionTrigger'] = field(default_factory=list)

@dataclass
class LucidityMarker:
    """Model for lucidity indicators in dreams"""

    marker_type: str  # reality_check, anomaly_recognition, etc.
    detection_time: float  # seconds from dream start
    marker_strength: float  # 0.0-1.0 scale
    response_quality: float  # 0.0-1.0 scale
    lucidity_outcome: bool  # whether lucidity was achieved
```

### Integration Models

#### CrossFormIntegration

```python
@dataclass
class CrossFormIntegration:
    """Model for integration with other consciousness forms"""

    integration_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Integration targets
    target_forms: List[str] = field(default_factory=list)  # Form IDs
    integration_mode: 'IntegrationMode' = None

    # Data synchronization
    shared_data_elements: List['SharedDataElement'] = field(default_factory=list)
    synchronization_status: Dict[str, 'SyncStatus'] = field(default_factory=dict)

    # Cross-form consciousness coordination
    consciousness_state_alignment: float = 0.0  # 0.0-1.0 scale
    memory_system_coordination: float = 0.0  # 0.0-1.0 scale

    # Integration quality
    integration_coherence: float = 0.0  # 0.0-1.0 scale
    data_consistency: float = 0.0  # 0.0-1.0 scale
    performance_impact: float = 0.0  # 0.0-1.0 scale (negative impact)

class IntegrationMode(Enum):
    PASSIVE = "passive"  # receive data only
    ACTIVE = "active"    # bidirectional data exchange
    COORDINATED = "coordinated"  # synchronized processing
    MERGED = "merged"    # unified processing
```

### Therapeutic Models

#### TherapeuticSession

```python
@dataclass
class TherapeuticSession:
    """Model for therapeutic dream sessions"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Session context
    therapy_type: 'TherapyType' = None
    therapeutic_goals: List['TherapeuticGoal'] = field(default_factory=list)
    session_duration: timedelta = field(default=timedelta(0))

    # Therapeutic interventions
    interventions: List['TherapeuticIntervention'] = field(default_factory=list)
    guided_imagery: List['GuidedImagery'] = field(default_factory=list)
    dream_modifications: List['DreamModification'] = field(default_factory=list)

    # Progress tracking
    baseline_measures: Dict[str, float] = field(default_factory=dict)
    session_outcomes: Dict[str, float] = field(default_factory=dict)
    therapeutic_progress: float = 0.0  # 0.0-1.0 scale

    # Safety monitoring
    safety_indicators: List['SafetyIndicator'] = field(default_factory=list)
    adverse_events: List['AdverseEvent'] = field(default_factory=list)
    safety_interventions: List['SafetyIntervention'] = field(default_factory=list)

class TherapyType(Enum):
    NIGHTMARE_THERAPY = "nightmare_therapy"
    TRAUMA_PROCESSING = "trauma_processing"
    LUCID_DREAM_THERAPY = "lucid_dream_therapy"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CREATIVE_ENHANCEMENT = "creative_enhancement"
```

### Quality Assurance Models

#### DreamQualityAssessment

```python
@dataclass
class DreamQualityAssessment:
    """Model for comprehensive dream quality evaluation"""

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Quality dimensions
    narrative_quality: 'NarrativeQuality' = field(default=None)
    sensory_quality: 'SensoryQuality' = field(default=None)
    emotional_quality: 'EmotionalQuality' = field(default=None)
    consciousness_quality: 'ConsciousnessQuality' = field(default=None)

    # Overall quality metrics
    overall_quality_score: float = 0.0  # 0.0-1.0 scale
    user_satisfaction: float = 0.0  # 0.0-1.0 scale
    therapeutic_value: float = 0.0  # 0.0-1.0 scale

    # Comparative metrics
    quality_improvement: float = 0.0  # compared to baseline
    peer_comparison: float = 0.0  # compared to similar users
    normative_comparison: float = 0.0  # compared to population norms

    # Quality factors
    technical_factors: Dict[str, float] = field(default_factory=dict)
    user_factors: Dict[str, float] = field(default_factory=dict)
    environmental_factors: Dict[str, float] = field(default_factory=dict)

@dataclass
class NarrativeQuality:
    coherence: float = 0.0
    complexity: float = 0.0
    creativity: float = 0.0
    engagement: float = 0.0
    symbolic_richness: float = 0.0
    emotional_resonance: float = 0.0
```

## Data Relationships

### Relationship Mappings

```python
@dataclass
class DataRelationshipMap:
    """Defines relationships between different data models"""

    # Primary relationships
    dream_state_relationships: Dict[str, List[str]] = field(default_factory=dict)
    memory_relationships: Dict[str, List[str]] = field(default_factory=dict)
    consciousness_relationships: Dict[str, List[str]] = field(default_factory=dict)

    # Cross-form relationships
    form_integration_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Temporal relationships
    temporal_sequences: List['TemporalSequence'] = field(default_factory=list)
    causal_relationships: List['CausalRelationship'] = field(default_factory=list)

@dataclass
class TemporalSequence:
    sequence_id: str
    sequence_type: str
    elements: List[str]  # IDs of related elements
    temporal_order: List[int]  # ordering indices
```

## Validation Schemas

### Data Validation Rules

```python
class DreamDataValidator:
    """Validation rules for dream consciousness data models"""

    @staticmethod
    def validate_consciousness_state(state: DreamConsciousnessState) -> ValidationResult:
        """Validate dream consciousness state data"""
        errors = []
        warnings = []

        # Validate consciousness levels
        if not (0.0 <= state.consciousness_level <= 1.0):
            errors.append("Consciousness level must be between 0.0 and 1.0")

        if not (0.0 <= state.lucidity_level <= 1.0):
            errors.append("Lucidity level must be between 0.0 and 1.0")

        # Validate temporal consistency
        if state.duration.total_seconds() < 0:
            errors.append("Duration cannot be negative")

        # Validate safety constraints
        if state.emotional_content and state.emotional_content.overall_valence < -0.8:
            warnings.append("Extremely negative emotional content detected")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def validate_memory_consolidation(consolidation: MemoryConsolidation) -> ValidationResult:
        """Validate memory consolidation data"""
        errors = []

        # Validate success rates
        if not (0.0 <= consolidation.consolidation_success_rate <= 1.0):
            errors.append("Consolidation success rate must be between 0.0 and 1.0")

        # Validate memory strength changes
        for memory_id, strength_change in consolidation.memory_strength_changes.items():
            if not (-1.0 <= strength_change <= 1.0):
                errors.append(f"Memory strength change for {memory_id} out of range")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[]
        )

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
```

## Serialization Specifications

### JSON Schema Definitions

```python
DREAM_CONSCIOUSNESS_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "dream_id": {"type": "string", "format": "uuid"},
        "session_id": {"type": "string", "format": "uuid"},
        "user_id": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "consciousness_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "lucidity_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "dream_narrative": {"$ref": "#/definitions/DreamNarrative"},
        "sensory_experience": {"$ref": "#/definitions/SensoryExperience"},
        "emotional_content": {"$ref": "#/definitions/EmotionalContent"}
    },
    "required": ["dream_id", "session_id", "user_id", "timestamp"],
    "additionalProperties": False
}
```

## Database Schema

### Table Structures

```sql
-- Primary dream consciousness states table
CREATE TABLE dream_consciousness_states (
    dream_id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    duration INTERVAL,
    sleep_phase VARCHAR(50),
    consciousness_level DECIMAL(3,2) CHECK (consciousness_level >= 0 AND consciousness_level <= 1),
    lucidity_level DECIMAL(3,2) CHECK (lucidity_level >= 0 AND lucidity_level <= 1),
    narrative_coherence DECIMAL(3,2),
    sensory_vividness DECIMAL(3,2),
    emotional_intensity DECIMAL(3,2),
    safety_flags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Dream narrative content table
CREATE TABLE dream_narratives (
    narrative_id UUID PRIMARY KEY,
    dream_id UUID REFERENCES dream_consciousness_states(dream_id),
    narrative_type VARCHAR(50),
    coherence_score DECIMAL(3,2),
    complexity_level DECIMAL(3,2),
    primary_themes TEXT[],
    story_elements JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memory consolidation tracking table
CREATE TABLE memory_consolidations (
    consolidation_id UUID PRIMARY KEY,
    dream_id UUID REFERENCES dream_consciousness_states(dream_id),
    consolidation_success_rate DECIMAL(3,2),
    consolidation_efficiency DECIMAL(3,2),
    target_memories JSONB,
    consolidation_outcomes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_dream_states_user_timestamp ON dream_consciousness_states(user_id, timestamp);
CREATE INDEX idx_dream_states_session ON dream_consciousness_states(session_id);
CREATE INDEX idx_narratives_dream ON dream_narratives(dream_id);
CREATE INDEX idx_consolidations_dream ON memory_consolidations(dream_id);
```

## Conclusion

These comprehensive data models provide a robust foundation for representing, storing, and processing dream consciousness information. The models support complex dream experiences, memory integration, consciousness level tracking, therapeutic applications, and quality assurance while maintaining data integrity and enabling sophisticated analysis capabilities.