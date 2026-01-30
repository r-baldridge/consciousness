# Neurodivergent Consciousness Data Structures
**Form 38: Neurodivergent Consciousness**
**Task 38.B.3: Data Structures Specification**
**Date:** January 29, 2026

## Overview

This document defines the comprehensive data structures required for implementing neurodivergent consciousness systems. These structures support neurotype profiling, cognitive strength identification, sensory processing characterization, accommodation planning, synesthesia modeling, first-person account preservation, and cross-form integration. All data structures are designed with neurodiversity-affirming principles, framing neurological differences as natural variations rather than deficits.

## Core Data Models

### 1. Neurotype Profile Structures

#### 1.1 Individual Neurotype Profile

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set

@dataclass
class NeurodivergentProfile:
    """Comprehensive profile for an individual's neurodivergent characteristics"""

    # Core Identity
    profile_id: str  # Unique profile identifier
    created_at: datetime  # Profile creation timestamp
    last_updated: datetime  # Last modification timestamp
    consent_status: ConsentRecord  # Data consent tracking

    # Neurotype Information
    primary_neurotype: NeurotypeClassification  # Primary neurotype
    additional_neurotypes: List[NeurotypeClassification]  # Co-occurring neurotypes
    self_identified_labels: List[str]  # Self-chosen identity labels
    diagnostic_status: DiagnosticStatus  # Formal diagnosis status

    # Cognitive Profile
    cognitive_profile: CognitiveStyleProfile  # Cognitive processing patterns
    strengths_profile: StrengthsProfile  # Identified strengths
    processing_preferences: ProcessingPreferences  # Preferred processing modes

    # Sensory Profile
    sensory_profile: SensoryProfile  # Sensory processing characteristics
    synesthesia_profile: Optional[SynesthesiaProfile]  # If synesthete

    # Executive Function Profile
    executive_profile: ExecutiveFunctionProfile  # Executive function patterns

    # Social and Communication
    communication_profile: CommunicationProfile  # Communication preferences
    social_profile: SocialProfile  # Social interaction patterns

    # Support and Accommodation
    accommodation_needs: AccommodationProfile  # Current accommodations
    support_preferences: SupportPreferences  # Preferred support types
    environmental_preferences: EnvironmentalPreferences  # Optimal environments

    # First-Person Account
    lived_experience_records: List[LivedExperienceRecord]  # First-person accounts

    # Maturity and Quality
    profile_completeness: float  # Profile completeness (0.0-1.0)
    data_quality_score: float  # Overall data quality (0.0-1.0)
    self_report_proportion: float  # Proportion from self-report (0.0-1.0)

    def __post_init__(self):
        if not self.profile_id:
            self.profile_id = generate_profile_id()


class NeurotypeClassification(Enum):
    AUTISM_SPECTRUM = "autism_spectrum"
    ADHD_INATTENTIVE = "adhd_inattentive"
    ADHD_HYPERACTIVE_IMPULSIVE = "adhd_hyperactive_impulsive"
    ADHD_COMBINED = "adhd_combined"
    DYSLEXIA = "dyslexia"
    DYSCALCULIA = "dyscalculia"
    DYSPRAXIA = "dyspraxia"
    DYSGRAPHIA = "dysgraphia"
    TOURETTE_SYNDROME = "tourette_syndrome"
    OCD = "ocd"
    SYNESTHESIA = "synesthesia"
    HIGHLY_SENSITIVE_PERSON = "highly_sensitive_person"
    BIPOLAR_SPECTRUM = "bipolar_spectrum"
    SCHIZOPHRENIA_SPECTRUM = "schizophrenia_spectrum"
    GIFTEDNESS = "giftedness"
    TWICE_EXCEPTIONAL = "twice_exceptional"
    AUDITORY_PROCESSING_DIFFERENCE = "auditory_processing_difference"
    VISUAL_PROCESSING_DIFFERENCE = "visual_processing_difference"
    SENSORY_PROCESSING_DIFFERENCE = "sensory_processing_difference"
    NONVERBAL_LEARNING_DIFFERENCE = "nonverbal_learning_difference"
    CUSTOM = "custom"


class DiagnosticStatus(Enum):
    FORMALLY_DIAGNOSED = "formally_diagnosed"
    SELF_IDENTIFIED = "self_identified"
    SUSPECTED = "suspected"
    UNDER_ASSESSMENT = "under_assessment"
    LATE_DIAGNOSED = "late_diagnosed"
    NOT_SEEKING_DIAGNOSIS = "not_seeking_diagnosis"


@dataclass
class ConsentRecord:
    """Data consent tracking for neurodivergent profiles"""

    consent_given: bool
    consent_date: datetime
    consent_scope: List[str]  # What data uses are consented
    data_control_preferences: DataControlPreferences  # Individual data control
    withdrawal_option: bool  # Can withdraw at any time
    last_reviewed: datetime  # Last consent review date
```

#### 1.2 Cognitive Style Profile

```python
@dataclass
class CognitiveStyleProfile:
    """Cognitive processing style characterization"""

    # Attention Patterns
    attention_style: AttentionStyleProfile  # Attention characteristics
    hyperfocus_patterns: HyperfocusProfile  # Hyperfocus capabilities

    # Processing Style
    detail_global_balance: float  # Detail vs. global orientation (-1.0 to 1.0)
    sequential_simultaneous: float  # Sequential vs. simultaneous (-1.0 to 1.0)
    verbal_visual_orientation: float  # Verbal vs. visual preference (-1.0 to 1.0)
    processing_speed_profile: ProcessingSpeedProfile  # Speed across domains

    # Learning Style
    learning_preferences: List[LearningPreference]  # Preferred learning modes
    optimal_input_modalities: List[str]  # Best input channels
    challenge_areas: List[str]  # Areas requiring more support

    # Pattern Recognition
    pattern_recognition_strength: float  # Pattern detection ability (0.0-1.0)
    systemizing_quotient: float  # Systemizing drive (0.0-1.0)
    detail_detection: float  # Fine detail detection (0.0-1.0)

    # Memory Patterns
    memory_profile: MemoryProfile  # Memory strengths and patterns
    special_interest_depth: float  # Depth of knowledge in interests (0.0-1.0)

    # Creativity and Divergent Thinking
    divergent_thinking_score: float  # Divergent thinking ability (0.0-1.0)
    creative_flow_accessibility: float  # Ease of entering flow (0.0-1.0)
    novel_association_generation: float  # Novel connections (0.0-1.0)


@dataclass
class AttentionStyleProfile:
    """Attention pattern characterization"""

    attention_distribution: AttentionDistribution  # How attention is distributed
    sustained_focus_duration: Optional[timedelta]  # Typical focus duration
    interest_attention_coupling: float  # Interest-attention link strength (0.0-1.0)
    context_switching_ease: float  # Ease of switching contexts (0.0-1.0)
    attention_to_detail: float  # Detail attention level (0.0-1.0)
    environmental_sensitivity: float  # Distraction sensitivity (0.0-1.0)
    optimal_stimulation_level: float  # Preferred stimulation (0.0-1.0)
    attention_strengths: List[str]  # Specific attention strengths
    attention_challenges: List[str]  # Context-specific challenges


class AttentionDistribution(Enum):
    BROAD_DISTRIBUTED = "broad_distributed"
    NARROW_FOCUSED = "narrow_focused"
    VARIABLE_CONTEXT_DEPENDENT = "variable_context_dependent"
    INTEREST_DRIVEN = "interest_driven"
    HYPERFOCUSED = "hyperfocused"
    DIFFUSE = "diffuse"


@dataclass
class HyperfocusProfile:
    """Hyperfocus capability characterization"""

    hyperfocus_capable: bool  # Whether hyperfocus occurs
    typical_triggers: List[str]  # What triggers hyperfocus
    typical_duration: Optional[timedelta]  # Typical hyperfocus duration
    productivity_during: float  # Productivity level (0.0-1.0)
    transition_difficulty: float  # Difficulty exiting (0.0-1.0)
    positive_applications: List[str]  # How hyperfocus is used beneficially
    management_strategies: List[str]  # Strategies for managing hyperfocus
```

### 2. Strength Identification Structures

#### 2.1 Strengths Profile

```python
@dataclass
class StrengthsProfile:
    """Comprehensive strengths identification and documentation"""

    profile_id: str
    strengths: List[IdentifiedStrength]  # All identified strengths
    strength_categories: Dict[str, List[str]]  # Strengths by category
    strength_applications: List[StrengthApplication]  # Real-world applications
    development_history: List[StrengthDevelopmentRecord]  # How strengths developed
    overall_strength_narrative: str  # Narrative summary of strengths


@dataclass
class IdentifiedStrength:
    """Individual identified strength"""

    strength_id: str
    strength_name: str  # Name of the strength
    strength_category: StrengthCategory  # Category classification
    description: str  # Detailed description
    neurotype_association: List[NeurotypeClassification]  # Associated neurotypes
    evidence_type: EvidenceType  # Type of evidence supporting identification
    confidence_level: float  # Confidence in identification (0.0-1.0)

    # Contextual Information
    contexts_where_expressed: List[str]  # Where this strength manifests
    optimal_conditions: List[str]  # Best conditions for this strength
    frequency: str  # How often this strength is expressed
    development_stage: str  # Current development level

    # Impact
    personal_significance: float  # How important to the individual (0.0-1.0)
    practical_applications: List[str]  # How it is applied
    contribution_areas: List[str]  # Where it contributes value


class StrengthCategory(Enum):
    PATTERN_RECOGNITION = "pattern_recognition"
    DETAIL_ORIENTATION = "detail_orientation"
    HYPERFOCUS = "hyperfocus"
    DIVERGENT_THINKING = "divergent_thinking"
    SENSORY_ACUITY = "sensory_acuity"
    MEMORY_SPECIALIZATION = "memory_specialization"
    SPATIAL_REASONING = "spatial_reasoning"
    SYSTEMIZING = "systemizing"
    CREATIVE_FLOW = "creative_flow"
    EMPATHIC_INTENSITY = "empathic_intensity"
    HOLISTIC_PROCESSING = "holistic_processing"
    VISUAL_THINKING = "visual_thinking"
    VERBAL_FLUENCY = "verbal_fluency"
    RAPID_PROCESSING = "rapid_processing"
    INTENSE_FOCUS = "intense_focus"
    NOVEL_CONNECTIONS = "novel_connections"
    JUSTICE_SENSITIVITY = "justice_sensitivity"
    PERSISTENCE = "persistence"
    AUTHENTICITY = "authenticity"
    DEEP_PROCESSING = "deep_processing"


class EvidenceType(Enum):
    SELF_REPORT = "self_report"
    OBSERVATIONAL = "observational"
    ASSESSMENT_BASED = "assessment_based"
    PERFORMANCE_DEMONSTRATED = "performance_demonstrated"
    PEER_IDENTIFIED = "peer_identified"
    PROFESSIONALLY_ASSESSED = "professionally_assessed"
    COMBINED = "combined"
```

### 3. Sensory Processing Structures

#### 3.1 Sensory Profile

```python
@dataclass
class SensoryProfile:
    """Comprehensive sensory processing characterization"""

    profile_id: str
    overall_sensitivity_level: SensitivityLevel  # General sensitivity
    sensory_channels: Dict[str, SensoryChannelProfile]  # Per-channel profiles

    # Modality-Specific Profiles
    visual_processing: VisualSensoryProfile
    auditory_processing: AuditorySensoryProfile
    tactile_processing: TactileSensoryProfile
    olfactory_processing: OlfactorySensoryProfile
    gustatory_processing: GustatorySensoryProfile
    vestibular_processing: VestibularSensoryProfile
    proprioceptive_processing: ProprioceptiveSensoryProfile
    interoceptive_processing: InteroceptiveSensoryProfile

    # Regulation
    regulation_strategies: List[RegulationStrategy]  # Sensory regulation methods
    sensory_diet: Optional[SensoryDiet]  # Structured sensory plan
    overload_indicators: List[str]  # Signs of sensory overload
    recovery_needs: RecoveryNeeds  # Post-overload recovery requirements

    # Environment
    optimal_sensory_environment: EnvironmentalSpecification
    challenging_environments: List[str]  # Environments to modify or avoid
    environmental_modifications: List[EnvironmentalModification]


class SensitivityLevel(Enum):
    HYPOSENSITIVE = "hyposensitive"
    LOW_REGISTRATION = "low_registration"
    TYPICAL = "typical"
    SENSITIVE = "sensitive"
    HYPERSENSITIVE = "hypersensitive"
    VARIABLE = "variable"


@dataclass
class SensoryChannelProfile:
    """Profile for a single sensory channel"""

    channel: str  # Sensory channel name
    sensitivity_level: SensitivityLevel  # Sensitivity for this channel
    seeking_behaviors: List[str]  # Sensory seeking in this channel
    avoiding_behaviors: List[str]  # Sensory avoidance in this channel
    strengths: List[str]  # Sensory strengths
    preferences: List[str]  # Preferences within this channel
    thresholds: SensoryThresholds  # Detection and tolerance thresholds
    processing_speed: str  # Processing speed in this channel


@dataclass
class SensoryThresholds:
    """Sensory detection and tolerance thresholds"""

    detection_threshold: float  # Sensitivity of detection (0.0-1.0, lower = more sensitive)
    comfort_range_low: float  # Lower bound of comfort (0.0-1.0)
    comfort_range_high: float  # Upper bound of comfort (0.0-1.0)
    overload_threshold: float  # Point of overload (0.0-1.0)
    habituation_rate: float  # How quickly habituation occurs (0.0-1.0)
```

#### 3.2 Synesthesia Profile

```python
@dataclass
class SynesthesiaProfile:
    """Comprehensive synesthesia characterization"""

    profile_id: str
    synesthesia_types: List[SynesthesiaMapping]  # All synesthesia types
    consistency_score: float  # Test-retest consistency (0.0-1.0)
    automaticity: float  # How automatic the experience is (0.0-1.0)
    age_of_earliest_memory: Optional[int]  # When first noticed
    family_history: bool  # Family members with synesthesia

    # Phenomenology
    phenomenological_description: str  # Subjective description
    projector_associator: ProjectorAssociatorType  # Where experience occurs
    vividness: float  # Vividness of synesthetic experience (0.0-1.0)

    # Impact
    memory_benefits: List[str]  # How synesthesia aids memory
    creative_applications: List[str]  # Creative uses
    daily_life_impact: str  # Impact on daily functioning
    emotional_associations: List[str]  # Emotional qualities


@dataclass
class SynesthesiaMapping:
    """Individual synesthesia type mapping"""

    synesthesia_type: SynesthesiaType
    inducer: str  # What triggers the experience
    concurrent: str  # What is experienced
    consistency: float  # Consistency of mapping (0.0-1.0)
    bidirectional: bool  # Whether mapping goes both directions
    specific_mappings: Dict[str, str]  # Specific inducer-concurrent pairs
    phenomenological_notes: str  # Qualitative description


class SynesthesiaType(Enum):
    GRAPHEME_COLOR = "grapheme_color"
    CHROMESTHESIA = "chromesthesia"
    SPATIAL_SEQUENCE = "spatial_sequence"
    NUMBER_FORM = "number_form"
    MIRROR_TOUCH = "mirror_touch"
    LEXICAL_GUSTATORY = "lexical_gustatory"
    ORDINAL_LINGUISTIC_PERSONIFICATION = "ordinal_linguistic_personification"
    AUDITORY_TACTILE = "auditory_tactile"
    EMOTION_COLOR = "emotion_color"
    SMELL_COLOR = "smell_color"
    PAIN_COLOR = "pain_color"
    TEMPERATURE_COLOR = "temperature_color"
    VISION_SOUND = "vision_sound"
    TIME_SPACE = "time_space"
    PERSONALITY_COLOR = "personality_color"
    CUSTOM = "custom"


class ProjectorAssociatorType(Enum):
    PROJECTOR = "projector"  # Experiences concurrents in external space
    ASSOCIATOR = "associator"  # Experiences concurrents in mind's eye
    MIXED = "mixed"
```

### 4. Accommodation and Support Structures

#### 4.1 Accommodation Profile

```python
@dataclass
class AccommodationProfile:
    """Accommodation needs and current provisions"""

    profile_id: str
    accommodations: List[Accommodation]  # All accommodations
    priority_accommodations: List[str]  # Most important accommodation IDs
    accommodation_contexts: Dict[str, List[str]]  # Accommodations by context
    effectiveness_tracking: Dict[str, float]  # Effectiveness scores
    unmet_needs: List[str]  # Identified but unmet needs
    accommodation_history: List[AccommodationHistoryRecord]


@dataclass
class Accommodation:
    """Individual accommodation specification"""

    accommodation_id: str
    accommodation_type: AccommodationType
    description: str  # Detailed description
    context: AccommodationContext  # Where it applies
    implementation_details: str  # How to implement
    importance: float  # Importance level (0.0-1.0)
    currently_in_place: bool  # Whether currently provided
    effectiveness: Optional[float]  # Effectiveness rating (0.0-1.0)
    self_determined: bool  # Whether self-identified
    evidence_basis: str  # Evidence supporting this accommodation


class AccommodationType(Enum):
    ENVIRONMENTAL = "environmental"
    COMMUNICATION = "communication"
    TASK_MODIFICATION = "task_modification"
    TEMPORAL = "temporal"
    SENSORY = "sensory"
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"
    INFORMATIONAL = "informational"
    PHYSICAL = "physical"
    PROCESS = "process"


class AccommodationContext(Enum):
    WORKPLACE = "workplace"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    SOCIAL = "social"
    HOME = "home"
    PUBLIC_SPACES = "public_spaces"
    DIGITAL = "digital"
    TRANSPORTATION = "transportation"
    ALL_CONTEXTS = "all_contexts"
```

### 5. First-Person Account Structures

#### 5.1 Lived Experience Record

```python
@dataclass
class LivedExperienceRecord:
    """First-person account of neurodivergent experience"""

    record_id: str
    record_type: AccountType
    neurotype_context: List[NeurotypeClassification]  # Relevant neurotypes
    account_text: str  # The first-person narrative
    themes: List[str]  # Identified themes
    domain: ExperienceDomain  # Life domain
    anonymized: bool  # Whether de-identified
    composite: bool  # Whether composite of multiple accounts
    consent_for_sharing: bool  # Consent for use
    emotional_tone: str  # Overall emotional tone
    key_insights: List[str]  # Key insights from the account
    related_strengths: List[str]  # Strengths mentioned
    related_challenges: List[str]  # Challenges mentioned
    resilience_elements: List[str]  # Resilience aspects
    created_at: datetime = field(default_factory=datetime.utcnow)


class AccountType(Enum):
    SENSORY_EXPERIENCE = "sensory_experience"
    SOCIAL_EXPERIENCE = "social_experience"
    COGNITIVE_EXPERIENCE = "cognitive_experience"
    EMOTIONAL_EXPERIENCE = "emotional_experience"
    IDENTITY_NARRATIVE = "identity_narrative"
    STRENGTH_STORY = "strength_story"
    CHALLENGE_NAVIGATION = "challenge_navigation"
    ACCOMMODATION_EXPERIENCE = "accommodation_experience"
    DIAGNOSIS_JOURNEY = "diagnosis_journey"
    MASKING_EXPERIENCE = "masking_experience"
    BURNOUT_RECOVERY = "burnout_recovery"
    ADVOCACY_EXPERIENCE = "advocacy_experience"


class ExperienceDomain(Enum):
    WORK = "work"
    EDUCATION = "education"
    RELATIONSHIPS = "relationships"
    DAILY_LIFE = "daily_life"
    HEALTHCARE = "healthcare"
    CREATIVITY = "creativity"
    SELF_UNDERSTANDING = "self_understanding"
    COMMUNITY = "community"
    PARENTING = "parenting"
    IDENTITY = "identity"
```

## Enumeration Types

### Classification Enumerations

```python
class CognitiveDomain(Enum):
    """Cognitive processing domains"""
    ATTENTION = "attention"
    MEMORY = "memory"
    LANGUAGE = "language"
    SPATIAL = "spatial"
    EXECUTIVE_FUNCTION = "executive_function"
    PROCESSING_SPEED = "processing_speed"
    SENSORY_PROCESSING = "sensory_processing"
    SOCIAL_COGNITION = "social_cognition"
    CREATIVE_THINKING = "creative_thinking"
    PATTERN_RECOGNITION = "pattern_recognition"
    MATHEMATICAL = "mathematical"
    MOTOR_PLANNING = "motor_planning"


class ProfileDataSource(Enum):
    """Source of profile data"""
    SELF_REPORT_NARRATIVE = "self_report_narrative"
    SELF_REPORT_QUESTIONNAIRE = "self_report_questionnaire"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    NEUROPSYCHOLOGICAL_TESTING = "neuropsychological_testing"
    BEHAVIORAL_OBSERVATION = "behavioral_observation"
    PEER_REPORT = "peer_report"
    FAMILY_REPORT = "family_report"
    EDUCATIONAL_RECORDS = "educational_records"
    OCCUPATIONAL_ASSESSMENT = "occupational_assessment"


class LanguageFraming(Enum):
    """Language framing preferences"""
    IDENTITY_FIRST = "identity_first"  # e.g., "autistic person"
    PERSON_FIRST = "person_first"  # e.g., "person with autism"
    NO_PREFERENCE = "no_preference"
    CUSTOM = "custom"


class MaskingLevel(Enum):
    """Level of social masking/camouflaging"""
    NO_MASKING = "no_masking"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"
    CHRONIC = "chronic"
    VARIABLE = "variable"


class EnergyState(Enum):
    """Current energy/regulation state"""
    WELL_REGULATED = "well_regulated"
    SLIGHTLY_DEPLETED = "slightly_depleted"
    MODERATELY_DEPLETED = "moderately_depleted"
    SIGNIFICANTLY_DEPLETED = "significantly_depleted"
    BURNOUT = "burnout"
    RECOVERY = "recovery"
    ENERGIZED = "energized"
    OVERSTIMULATED = "overstimulated"
```

## Input/Output Structures

### Processing Input Structure

```python
@dataclass
class NeurodivergentProcessingInput:
    """Input structure for neurodivergent consciousness processing"""

    input_id: str  # Unique input identifier
    input_type: NDProcessingInputType  # Type of processing request
    profile_data: Optional[NeurodivergentProfile]  # Profile information
    context_data: Dict[str, Any]  # Context for the request
    query_parameters: Dict[str, Any]  # Processing parameters
    requested_outputs: List[str]  # Desired output types
    language_preferences: LanguageFraming  # Language framing preference
    consent_verified: bool  # Consent verification status
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NDProcessingInputType(Enum):
    PROFILE_CREATION = "profile_creation"
    PROFILE_UPDATE = "profile_update"
    STRENGTH_IDENTIFICATION = "strength_identification"
    ACCOMMODATION_PLANNING = "accommodation_planning"
    SENSORY_ASSESSMENT = "sensory_assessment"
    SYNESTHESIA_PROFILING = "synesthesia_profiling"
    SUPPORT_PLANNING = "support_planning"
    COMMUNICATION_SUPPORT = "communication_support"
    ENVIRONMENTAL_ASSESSMENT = "environmental_assessment"
    CROSS_FORM_INTEGRATION = "cross_form_integration"
```

### Processing Output Structure

```python
@dataclass
class NeurodivergentProcessingOutput:
    """Output structure from neurodivergent consciousness processing"""

    output_id: str
    input_id: str  # Reference to input
    output_type: str  # Type of output
    results: Dict[str, Any]  # Processing results
    strengths_highlighted: List[str]  # Strengths emphasized in output
    language_framing: LanguageFraming  # Language framing used
    affirming_language_verified: bool  # Affirming language check passed
    quality_assessment: QualityAssessment  # Output quality metrics
    accessibility_formats: List[str]  # Available accessible formats
    recommendations: List[str]  # Actionable recommendations
    metadata: ProcessingMetadata
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityAssessment:
    """Output quality assessment"""

    overall_quality: float  # Overall quality (0.0-1.0)
    affirming_language_score: float  # Language affirming-ness (0.0-1.0)
    strength_representation: float  # Strength emphasis (0.0-1.0)
    self_report_alignment: float  # Alignment with self-report (0.0-1.0)
    cultural_sensitivity: float  # Cultural sensitivity (0.0-1.0)
    actionability: float  # How actionable outputs are (0.0-1.0)
    limitations: List[str]  # Known limitations
```

## Internal State Structures

### System State Management

```python
@dataclass
class NeurodivergentSystemState:
    """Internal state of the neurodivergent consciousness system"""

    # Processing State
    active_profiles: Dict[str, ProfileState]  # Active profile sessions
    processing_queue: List[str]  # Queued requests
    resource_utilization: ResourceUtilization  # Resource usage

    # Knowledge State
    neurotype_database_version: str  # Neurotype knowledge version
    accommodation_database_version: str  # Accommodation DB version
    research_database_version: str  # Research evidence version
    first_person_accounts_count: int  # Number of stored accounts
    last_knowledge_update: datetime  # Last update timestamp

    # Ethical Compliance
    affirming_language_model_version: str  # Language model version
    consent_compliance_status: str  # Overall consent compliance
    last_ethical_audit: datetime  # Last ethical audit date

    # Model State
    active_models: Dict[str, ModelState]  # Active processing models
    model_bias_metrics: Dict[str, float]  # Bias monitoring metrics
    model_fairness_scores: Dict[str, float]  # Fairness metrics
```

## Relationship Mappings

### Cross-Form Data Exchange

```python
@dataclass
class CrossFormRelationshipMap:
    """Cross-form relationship mappings for neurodivergent consciousness"""

    # Form 01 - Visual Consciousness
    visual_processing_mapping: Dict[str, str]  # Visual processing differences
    visual_thinking_mapping: Dict[str, str]  # Visual thinking strengths

    # Form 02 - Auditory Consciousness
    auditory_processing_mapping: Dict[str, str]  # Auditory processing differences
    chromesthesia_mapping: Dict[str, str]  # Sound-color synesthesia

    # Form 03 - Somatosensory Consciousness
    tactile_processing_mapping: Dict[str, str]  # Touch sensitivity
    proprioceptive_mapping: Dict[str, str]  # Body awareness differences

    # Form 06 - Interoceptive Consciousness
    interoceptive_mapping: Dict[str, str]  # Internal body signal processing

    # Form 07 - Emotional Consciousness
    emotional_intensity_mapping: Dict[str, str]  # Emotional processing
    alexithymia_mapping: Dict[str, str]  # Emotion identification differences

    # Form 08 - Arousal and Alertness
    arousal_regulation_mapping: Dict[str, str]  # Arousal patterns
    energy_management_mapping: Dict[str, str]  # Energy regulation

    # Form 09 - Perceptual Consciousness
    perceptual_binding_mapping: Dict[str, str]  # Synesthesia binding
    perception_style_mapping: Dict[str, str]  # Perceptual differences

    # Form 37 - Psychedelic Consciousness
    psychedelic_synesthesia_mapping: Dict[str, str]  # Synesthesia-psychedelic overlap

    # Form 39 - Trauma Consciousness
    trauma_neurodivergent_mapping: Dict[str, str]  # Intersection of trauma and ND


@dataclass
class CrossFormDataExchange:
    """Data exchange record between forms"""

    exchange_id: str
    source_form: str  # "38-neurodivergent-consciousness"
    target_form: str
    data_type: str
    data_payload: Dict[str, Any]
    exchange_protocol: str
    affirming_language_verified: bool  # Language check for exchange
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_status: str = "pending"
```

## Data Validation and Constraints

### Validation Rules

```python
class NeurodivergentDataValidation:
    """Validation rules for neurodivergent consciousness data structures"""

    @staticmethod
    def validate_profile(profile: NeurodivergentProfile) -> ValidationResult:
        """Validate neurodivergent profile"""
        errors = []
        warnings = []

        if not profile.profile_id:
            errors.append("Profile ID is required")
        if not profile.consent_status or not profile.consent_status.consent_given:
            errors.append("Valid consent is required for profile creation")
        if profile.self_report_proportion < 0.3:
            warnings.append("Self-report proportion is low; prioritize first-person data")
        if not profile.strengths_profile or len(profile.strengths_profile.strengths) == 0:
            warnings.append("No strengths identified; strength identification is recommended")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_language_framing(text: str) -> LanguageValidationResult:
        """Validate text for neurodiversity-affirming language"""
        violations = []
        suggestions = []

        deficit_terms = ['suffering from', 'afflicted with', 'disorder', 'deficit',
                        'impairment', 'abnormal', 'normal brain']
        for term in deficit_terms:
            if term.lower() in text.lower():
                violations.append(f"Deficit-framing term detected: '{term}'")
                suggestions.append(f"Consider replacing '{term}' with affirming language")

        return LanguageValidationResult(
            affirming=len(violations) == 0,
            violations=violations,
            suggestions=suggestions
        )

    @staticmethod
    def validate_accommodation(accommodation: Accommodation) -> ValidationResult:
        """Validate accommodation specification"""
        errors = []
        warnings = []

        if not accommodation.description:
            errors.append("Accommodation description is required")
        if not accommodation.context:
            errors.append("Accommodation context must be specified")
        if not accommodation.self_determined:
            warnings.append("Accommodation not self-determined; verify with individual")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


@dataclass
class ValidationResult:
    """Result of data validation"""
    valid: bool
    errors: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class LanguageValidationResult:
    """Result of language framing validation"""
    affirming: bool
    violations: List[str]
    suggestions: List[str]
```

## Data Serialization

### Serialization Support

```python
class NeurodivergentDataSerializer:
    """Serialization utilities for neurodivergent consciousness data"""

    @staticmethod
    def serialize_profile(profile: NeurodivergentProfile) -> str:
        """Serialize profile to JSON with privacy protection"""
        sanitized = NeurodivergentDataSerializer._apply_privacy_rules(profile)
        return json.dumps(sanitized, cls=NeurodivergentEncoder, indent=2)

    @staticmethod
    def _apply_privacy_rules(profile: NeurodivergentProfile) -> Dict:
        """Apply privacy rules before serialization"""
        data = profile.__dict__.copy()
        # Redact identifying information based on consent scope
        if 'identifying_info' not in profile.consent_status.consent_scope:
            data.pop('identifying_info', None)
        return data


class NeurodivergentEncoder(json.JSONEncoder):
    """Custom JSON encoder for neurodivergent data structures"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)
```

This comprehensive data structure specification provides the foundation for implementing neurodiversity-affirming consciousness systems with proper cognitive profiling, strength identification, sensory characterization, accommodation planning, and cross-form integration while centering individual voice and self-determination.
