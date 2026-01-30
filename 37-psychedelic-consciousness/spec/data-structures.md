# Psychedelic Consciousness Data Structures
**Form 37: Psychedelic/Entheogenic Consciousness**
**Task 37.B.3: Data Structures Specification**
**Date:** January 29, 2026

## Overview

This document defines the comprehensive data structures required for implementing psychedelic consciousness systems. These structures support substance profiling, experience phenomenology modeling, ceremonial context representation, therapeutic protocol management, neural mechanism correlation, and cross-form integration for research and clinical applications.

## Core Data Structures

### 1. Substance and Pharmacology Structures

#### 1.1 Substance Profile Structure

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class SubstanceProfile:
    """Comprehensive profile for a psychedelic/entheogenic substance"""

    # Core Identity
    substance_id: str  # Unique substance identifier
    common_name: str  # Common/street name
    scientific_name: str  # IUPAC or systematic name
    chemical_class: ChemicalClass  # Chemical classification
    cas_number: Optional[str]  # CAS registry number

    # Chemical Properties
    molecular_formula: str  # Molecular formula
    molecular_weight: float  # g/mol
    structural_class: StructuralClass  # Structural classification

    # Pharmacology
    receptor_profile: ReceptorBindingProfile  # Receptor binding affinities
    pharmacokinetics: PharmacokineticsProfile  # Absorption, distribution, metabolism
    dose_response: DoseResponseProfile  # Dose-response relationships

    # Safety
    safety_profile: SafetyProfile  # Safety and contraindication data
    interaction_profile: DrugInteractionProfile  # Drug interactions
    toxicology: ToxicologyProfile  # Toxicological data

    # Therapeutic Potential
    therapeutic_indications: List[TherapeuticIndication]  # Supported indications
    clinical_trial_data: List[ClinicalTrialReference]  # Referenced clinical trials
    efficacy_scores: Dict[str, float]  # Efficacy by indication (0.0-1.0)

    # Legal and Regulatory
    legal_status: Dict[str, LegalStatus]  # Legal status by jurisdiction
    scheduling: Dict[str, str]  # Drug scheduling by country

    # Traditional Use
    traditional_contexts: List[TraditionalUseContext]  # Traditional ceremonial uses
    ethnobotanical_history: str  # Historical usage narrative

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    data_quality_score: float = 0.0  # Overall data quality (0.0-1.0)
    source_references: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.substance_id:
            self.substance_id = generate_substance_id(self.common_name)


class ChemicalClass(Enum):
    TRYPTAMINE = "tryptamine"
    LYSERGAMIDE = "lysergamide"
    PHENETHYLAMINE = "phenethylamine"
    EMPATHOGEN = "empathogen"
    DISSOCIATIVE = "dissociative"
    CANNABINOID = "cannabinoid"
    IBOGAINE_CLASS = "ibogaine_class"
    SALVINORIN_CLASS = "salvinorin_class"
    AMANITA_CLASS = "amanita_class"
    OTHER = "other"


class StructuralClass(Enum):
    CLASSICAL_TRYPTAMINE = "classical_tryptamine"  # Psilocybin, DMT, 5-MeO-DMT
    ERGOLINE = "ergoline"  # LSD
    SUBSTITUTED_PHENETHYLAMINE = "substituted_phenethylamine"  # Mescaline
    SUBSTITUTED_AMPHETAMINE = "substituted_amphetamine"  # MDMA
    ARYLCYCLOHEXYLAMINE = "arylcyclohexylamine"  # Ketamine
    TERPENOID = "terpenoid"  # Salvinorin A
    ISOXAZOLE = "isoxazole"  # Muscimol
    INDOLE_ALKALOID = "indole_alkaloid"  # Ibogaine
    OTHER = "other"
```

#### 1.2 Receptor Binding Structure

```python
@dataclass
class ReceptorBindingProfile:
    """Receptor binding affinity profile for a substance"""

    substance_id: str
    bindings: List[ReceptorBinding]  # Individual receptor bindings
    primary_target: str  # Primary receptor target
    selectivity_ratios: Dict[str, float]  # Selectivity between receptors

    def get_binding_affinity(self, receptor: str) -> Optional[float]:
        """Get Ki value for a specific receptor"""
        for binding in self.bindings:
            if binding.receptor_name == receptor:
                return binding.ki_nm
        return None


@dataclass
class ReceptorBinding:
    """Individual receptor binding data point"""

    receptor_name: str  # e.g., '5-HT2A', '5-HT2C', 'D2', 'NMDA'
    receptor_subtype: Optional[str]  # Receptor subtype
    binding_type: BindingType  # Agonist, antagonist, partial agonist
    ki_nm: float  # Binding affinity in nanomolar
    efficacy: float  # Intrinsic efficacy (0.0-1.0)
    functional_selectivity: Optional[float]  # Biased agonism measure
    species: str = "human"  # Species for binding data
    assay_method: str = ""  # Assay methodology used
    source_reference: str = ""  # Citation for data


class BindingType(Enum):
    FULL_AGONIST = "full_agonist"
    PARTIAL_AGONIST = "partial_agonist"
    ANTAGONIST = "antagonist"
    INVERSE_AGONIST = "inverse_agonist"
    ALLOSTERIC_MODULATOR = "allosteric_modulator"
    REUPTAKE_INHIBITOR = "reuptake_inhibitor"
    RELEASING_AGENT = "releasing_agent"


@dataclass
class PharmacokineticsProfile:
    """Pharmacokinetics for a substance across administration routes"""

    substance_id: str
    routes: Dict[str, RouteProfile]  # Profiles by administration route
    metabolism_pathway: str  # Primary metabolic pathway
    metabolites: List[MetaboliteInfo]  # Active/notable metabolites
    half_life_hours: float  # Elimination half-life
    protein_binding_pct: float  # Plasma protein binding percentage
    volume_of_distribution: float  # Vd in L/kg
    cyp_interactions: List[CYPInteraction]  # Cytochrome P450 interactions


@dataclass
class RouteProfile:
    """Pharmacokinetic profile for a specific administration route"""

    route: AdministrationRoute
    bioavailability: float  # Fraction absorbed (0.0-1.0)
    onset_minutes: float  # Time to onset
    peak_minutes: float  # Time to peak effect
    duration_hours: float  # Total duration of effects
    typical_dose_range: DoseRange  # Typical dosing range


class AdministrationRoute(Enum):
    ORAL = "oral"
    SUBLINGUAL = "sublingual"
    INTRANASAL = "intranasal"
    SMOKED = "smoked"
    VAPORIZED = "vaporized"
    INTRAMUSCULAR = "intramuscular"
    INTRAVENOUS = "intravenous"
    RECTAL = "rectal"
    TRANSDERMAL = "transdermal"
    INSUFFLATED = "insufflated"
```

### 2. Experience Phenomenology Structures

#### 2.1 Psychedelic Experience Structure

```python
@dataclass
class PsychedelicExperience:
    """Comprehensive psychedelic experience record"""

    # Identity
    experience_id: str  # Unique experience identifier
    session_id: Optional[str]  # Clinical session identifier if applicable

    # Substance Information
    substance_id: str  # Substance used
    dose: DoseRecord  # Dosing information
    administration_route: AdministrationRoute  # Route of administration

    # Context
    set_assessment: SetAssessment  # Psychological set factors
    setting_assessment: SettingAssessment  # Physical/social setting
    context_type: ExperienceContextType  # Clinical, ceremonial, etc.

    # Temporal Dynamics
    timeline: ExperienceTimeline  # Phase-by-phase timeline
    total_duration_hours: float  # Total experience duration
    onset_time: datetime  # When effects began
    peak_time: Optional[datetime]  # When peak occurred

    # Phenomenological Content
    experience_types: List[ExperienceType]  # Types of experiences reported
    intensity_profile: IntensityProfile  # Intensity over time
    visual_content: List[VisualExperienceRecord]  # Visual phenomenology
    auditory_content: List[AuditoryExperienceRecord]  # Auditory phenomenology
    somatic_content: List[SomaticExperienceRecord]  # Body-based experiences

    # Psychological Dimensions
    ego_dissolution: EgoDissolutionRecord  # Ego dissolution metrics
    mystical_experience: MysticalExperienceRecord  # MEQ-30 and related
    emotional_content: List[EmotionalExperienceRecord]  # Emotional experiences
    cognitive_content: List[CognitiveExperienceRecord]  # Insights, realizations

    # Challenging Aspects
    challenging_experiences: List[ChallengingExperienceRecord]  # Difficult moments
    safety_events: List[SafetyEvent]  # Any safety concerns

    # Outcomes
    acute_outcomes: AcuteOutcomeRecord  # Immediate post-session outcomes
    integration_notes: List[IntegrationNote]  # Integration session notes
    long_term_outcomes: Optional[LongTermOutcomeRecord]  # Follow-up data

    # Questionnaire Scores
    questionnaire_scores: Dict[str, QuestionnaireScore]  # Standardized measures

    # Metadata
    recorded_by: str  # Recorder identity
    recording_method: str  # How data was captured
    data_quality: float  # Quality score (0.0-1.0)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ExperienceContextType(Enum):
    CLINICAL_THERAPEUTIC = "clinical_therapeutic"
    CLINICAL_RESEARCH = "clinical_research"
    CEREMONIAL_TRADITIONAL = "ceremonial_traditional"
    CEREMONIAL_SYNCRETIC = "ceremonial_syncretic"
    RETREAT_SETTING = "retreat_setting"
    SUPERVISED_NON_CLINICAL = "supervised_non_clinical"
    NATURALISTIC = "naturalistic"


class ExperienceType(Enum):
    VISUAL_GEOMETRY = "visual_geometry"  # Form constants, fractals
    ENTITY_ENCOUNTER = "entity_encounter"  # Contact with perceived entities
    EGO_DISSOLUTION = "ego_dissolution"  # Loss of self-boundaries
    MYSTICAL_UNITY = "mystical_unity"  # Unitive consciousness
    TIME_DISTORTION = "time_distortion"  # Altered temporal perception
    SYNESTHESIA = "synesthesia"  # Cross-modal sensory blending
    EMOTIONAL_CATHARSIS = "emotional_catharsis"  # Intense emotional release
    DEATH_REBIRTH = "death_rebirth"  # Psychological death-rebirth
    INSIGHT_REVELATION = "insight_revelation"  # Noetic insights
    COSMIC_CONSCIOUSNESS = "cosmic_consciousness"  # Universal awareness
    HEALING_VISION = "healing_vision"  # Healing-focused imagery
    AUTOBIOGRAPHICAL = "autobiographical"  # Memory re-processing
    SOMATIC_RELEASE = "somatic_release"  # Body-based release
    NATURE_COMMUNION = "nature_communion"  # Deep connection with nature


@dataclass
class IntensityProfile:
    """Intensity measurements across the experience"""

    scale_type: IntensityScale
    time_series: List[Tuple[float, float]]  # (minutes_from_onset, intensity)
    peak_intensity: float  # Maximum intensity reached
    peak_time_minutes: float  # Time of peak intensity
    average_intensity: float  # Mean intensity over experience
    intensity_variability: float  # Variance of intensity

    # Standardized scores
    mckenna_level: Optional[int]  # McKenna 1-5 scale
    shulgin_rating: Optional[str]  # Shulgin +/- scale
    subjective_0_10: Optional[float]  # Simple 0-10 subjective


class IntensityScale(Enum):
    MCKENNA_LEVELS = "mckenna_levels"  # 1-5 levels
    SHULGIN_RATING = "shulgin_rating"  # +/- to ++++
    SUBJECTIVE_0_10 = "subjective_0_10"  # 0-10 linear
    EDI_SCALE = "edi_scale"  # Ego Dissolution Inventory 0-100
    ASC_INTENSITY = "asc_intensity"  # ASC questionnaire intensity
```

#### 2.2 Ego Dissolution and Mystical Experience Structures

```python
@dataclass
class EgoDissolutionRecord:
    """Ego dissolution measurement and phenomenology"""

    edi_total_score: float  # Ego Dissolution Inventory total (0-100)
    edi_subscale_scores: Dict[str, float]  # Subscale scores
    onset_time_minutes: float  # When ego dissolution began
    duration_minutes: float  # How long it lasted
    completeness: EgoDissolutionLevel  # Degree of dissolution
    phenomenological_description: str  # Qualitative description
    associated_emotions: List[str]  # Emotions during dissolution
    self_boundary_dissolution: float  # Self-boundary loss (0.0-1.0)
    oceanic_boundlessness: float  # Oceanic boundlessness score (0.0-1.0)
    anxious_ego_dissolution: float  # Anxious dissolution score (0.0-1.0)


class EgoDissolutionLevel(Enum):
    NONE = "none"
    MILD_LOOSENING = "mild_loosening"
    MODERATE_DISSOLUTION = "moderate_dissolution"
    STRONG_DISSOLUTION = "strong_dissolution"
    COMPLETE_DISSOLUTION = "complete_dissolution"


@dataclass
class MysticalExperienceRecord:
    """Mystical experience measurement using MEQ-30 and related instruments"""

    # MEQ-30 Scores
    meq30_total: float  # Total MEQ-30 score (0.0-1.0 normalized)
    meq30_subscales: Dict[str, float]  # Subscale scores

    # MEQ-30 Subscale Detail
    internal_unity: float  # Internal unity/sacredness (0.0-1.0)
    external_unity: float  # External unity (0.0-1.0)
    transcendence_time_space: float  # Transcendence of time/space (0.0-1.0)
    ineffability: float  # Ineffability (0.0-1.0)
    sacredness: float  # Sense of sacredness (0.0-1.0)
    noetic_quality: float  # Noetic quality (0.0-1.0)
    deeply_felt_positive_mood: float  # Positive mood (0.0-1.0)

    # Criteria Met
    complete_mystical_experience: bool  # Meets Griffiths criteria
    criteria_met: List[str]  # Which specific criteria met
    criteria_threshold: float = 0.6  # Threshold for "complete" (default 60%)

    # Phenomenological Detail
    sense_of_unity_description: str  # Qualitative unity description
    transcendence_description: str  # Time/space transcendence description
    noetic_insights: List[str]  # Specific noetic insights reported
```

### 3. Set and Setting Structures

#### 3.1 Set Assessment Structure

```python
@dataclass
class SetAssessment:
    """Psychological set factors assessment"""

    # Intention
    primary_intention: str  # Primary intention for the experience
    secondary_intentions: List[str]  # Additional intentions
    intention_clarity: float  # Clarity of intention (0.0-1.0)

    # Psychological State
    baseline_mood: MoodAssessment  # Pre-session mood
    anxiety_level: float  # Pre-session anxiety (0.0-1.0)
    expectation_set: ExpectationProfile  # Expectations about the experience
    readiness_score: float  # Overall readiness (0.0-1.0)

    # Preparation
    preparation_activities: List[PreparationActivity]  # Preparation done
    preparation_duration_days: int  # Days of preparation
    dietary_preparation: Optional[DietaryPreparation]  # Diet modifications

    # Background
    prior_psychedelic_experience: ExperienceHistory  # Past experience level
    psychiatric_history: PsychiatricScreening  # Relevant psychiatric history
    meditation_practice: Optional[MeditationHistory]  # Contemplative practice
    therapy_history: Optional[TherapyHistory]  # Prior therapy experience

    # Trust and Relationship
    trust_in_guide: float  # Trust in facilitator/therapist (0.0-1.0)
    trust_in_process: float  # Trust in the overall process (0.0-1.0)
    support_system_quality: float  # Quality of support system (0.0-1.0)


@dataclass
class SettingAssessment:
    """Physical and social setting factors assessment"""

    # Physical Environment
    location_type: LocationType  # Type of location
    physical_comfort: float  # Comfort level (0.0-1.0)
    safety_rating: float  # Physical safety (0.0-1.0)
    aesthetic_quality: float  # Aesthetic/beauty of space (0.0-1.0)

    # Sensory Environment
    lighting: str  # Lighting conditions
    music_playlist: Optional[str]  # Music/sound design
    nature_access: bool  # Access to nature/outdoors
    temperature_comfort: float  # Temperature comfort (0.0-1.0)

    # Social Environment
    guide_present: bool  # Facilitator/therapist present
    guide_type: Optional[str]  # Type of guide
    group_size: int  # Number of participants
    social_safety: float  # Social safety (0.0-1.0)

    # Ceremonial Elements
    ceremonial_context: Optional[CeremonialContext]  # If ceremonial
    ritual_elements: List[str]  # Ritual elements present

    # Clinical Elements
    clinical_protocol: Optional[str]  # Clinical protocol identifier
    monitoring_level: str  # Level of medical monitoring
    emergency_preparedness: float  # Emergency readiness (0.0-1.0)


class LocationType(Enum):
    CLINICAL_TREATMENT_ROOM = "clinical_treatment_room"
    RESEARCH_LABORATORY = "research_laboratory"
    CEREMONIAL_MALOKA = "ceremonial_maloka"
    RETREAT_CENTER = "retreat_center"
    NATURAL_OUTDOOR = "natural_outdoor"
    HOME_SETTING = "home_setting"
    THERAPEUTIC_OFFICE = "therapeutic_office"
```

### 4. Therapeutic Protocol Structures

#### 4.1 Therapeutic Protocol Structure

```python
@dataclass
class TherapeuticProtocol:
    """Comprehensive therapeutic protocol definition"""

    # Protocol Identity
    protocol_id: str  # Unique protocol identifier
    protocol_name: str  # Human-readable name
    protocol_version: str  # Version number
    indication: TherapeuticIndication  # Target condition

    # Substance Parameters
    substance_id: str  # Substance used
    dose_schedule: DoseSchedule  # Dosing protocol
    administration_route: AdministrationRoute  # Route of administration
    number_of_sessions: int  # Total medication sessions

    # Preparation Phase
    preparation_sessions: List[SessionTemplate]  # Preparation session templates
    screening_requirements: ScreeningRequirements  # Eligibility screening
    informed_consent: InformedConsentTemplate  # Consent documentation

    # Medication Sessions
    session_protocol: MedicationSessionProtocol  # Session-day protocol
    monitoring_protocol: MonitoringProtocol  # Vital signs and safety monitoring
    music_protocol: Optional[MusicProtocol]  # Music/playlist protocol
    therapist_protocol: TherapistProtocol  # Therapist behavior guidelines

    # Integration Phase
    integration_sessions: List[SessionTemplate]  # Integration session templates
    integration_practices: List[IntegrationPractice]  # Recommended practices
    follow_up_schedule: FollowUpSchedule  # Follow-up assessment schedule

    # Outcome Measures
    primary_outcome_measures: List[OutcomeMeasure]  # Primary endpoints
    secondary_outcome_measures: List[OutcomeMeasure]  # Secondary endpoints
    assessment_schedule: AssessmentSchedule  # When to administer measures

    # Safety
    contraindications: List[Contraindication]  # Absolute/relative contraindications
    adverse_event_protocol: AdverseEventProtocol  # AE management
    stopping_rules: List[StoppingRule]  # Study/treatment stopping criteria

    # Evidence Base
    evidence_level: EvidenceLevel  # Level of supporting evidence
    key_references: List[str]  # Key supporting citations
    regulatory_status: str  # FDA/EMA status


class TherapeuticIndication(Enum):
    MAJOR_DEPRESSIVE_DISORDER = "major_depressive_disorder"
    TREATMENT_RESISTANT_DEPRESSION = "treatment_resistant_depression"
    PTSD = "ptsd"
    ALCOHOL_USE_DISORDER = "alcohol_use_disorder"
    TOBACCO_USE_DISORDER = "tobacco_use_disorder"
    OPIOID_USE_DISORDER = "opioid_use_disorder"
    END_OF_LIFE_ANXIETY = "end_of_life_anxiety"
    EXISTENTIAL_DISTRESS = "existential_distress"
    OCD = "ocd"
    ANOREXIA_NERVOSA = "anorexia_nervosa"
    CLUSTER_HEADACHE = "cluster_headache"
    SOCIAL_ANXIETY = "social_anxiety"


class EvidenceLevel(Enum):
    PHASE_3_COMPLETED = "phase_3_completed"
    PHASE_3_ONGOING = "phase_3_ongoing"
    PHASE_2_COMPLETED = "phase_2_completed"
    PHASE_2_ONGOING = "phase_2_ongoing"
    PHASE_1_COMPLETED = "phase_1_completed"
    OPEN_LABEL_STUDY = "open_label_study"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    PRECLINICAL = "preclinical"
    TRADITIONAL_USE = "traditional_use"
```

### 5. Neural Mechanism Structures

#### 5.1 Neural Correlate Structure

```python
@dataclass
class NeuralCorrelateProfile:
    """Neural correlates of psychedelic experience"""

    # Network Effects
    default_mode_network: DMNEffect  # DMN disruption metrics
    salience_network: NetworkEffect  # Salience network changes
    executive_network: NetworkEffect  # Executive network changes
    visual_network: NetworkEffect  # Visual cortex changes

    # Connectivity Changes
    global_connectivity: ConnectivityChange  # Overall connectivity shift
    between_network_connectivity: List[NetworkPairConnectivity]  # Cross-network
    within_network_connectivity: List[NetworkConnectivity]  # Within-network
    entropy_measures: EntropyMeasures  # Neural entropy metrics

    # Receptor-Level
    receptor_occupancy: Dict[str, float]  # Receptor occupancy estimates
    neurotransmitter_effects: List[NeurotransmitterEffect]  # NT level changes
    neuroplasticity_markers: NeuroplasticityMarkers  # Plasticity indicators

    # Temporal Dynamics
    neural_state_timeline: List[NeuralStateSnapshot]  # Time-series
    criticality_measures: CriticalityMeasures  # Edge-of-chaos metrics
    oscillation_changes: OscillationChanges  # Brainwave band changes

    # Correlation with Experience
    experience_correlations: List[ExperienceNeuralCorrelation]  # Exp-neural links
    prediction_accuracy: float  # How well neural data predicts experience


@dataclass
class DMNEffect:
    """Default Mode Network effect characterization"""

    connectivity_reduction: float  # Reduction in DMN connectivity (0.0-1.0)
    hub_disruption: float  # PCC/mPFC hub disruption (0.0-1.0)
    ego_dissolution_correlation: float  # Correlation with ego dissolution
    temporal_dynamics: List[Tuple[float, float]]  # Time-series of DMN changes
    recovery_time_hours: float  # Time to return to baseline


@dataclass
class EntropyMeasures:
    """Neural entropy measurements under psychedelic influence"""

    lempel_ziv_complexity: float  # LZ complexity score
    spectral_entropy: float  # Spectral entropy
    sample_entropy: float  # Sample entropy
    permutation_entropy: float  # Permutation entropy
    entropy_increase_pct: float  # Percentage increase from baseline
    criticality_index: float  # Proximity to critical regime (0.0-1.0)
```

### 6. Ceremonial and Traditional Use Structures

#### 6.1 Ceremonial Context Structure

```python
@dataclass
class CeremonialContext:
    """Traditional or syncretic ceremonial context"""

    # Identity
    ceremony_id: str  # Unique ceremony identifier
    tradition: CeremonialTradition  # Tradition classification
    lineage: str  # Specific lineage or school

    # Structure
    ceremony_structure: List[CeremonialPhase]  # Phases of the ceremony
    total_duration_hours: float  # Total ceremony duration
    participants_count: int  # Number of participants

    # Leadership
    facilitator_type: str  # e.g., curandero, ayahuasquero, roadman
    facilitator_qualifications: List[str]  # Training and lineage
    support_team: List[str]  # Additional support roles

    # Sacred Elements
    sacred_plants: List[str]  # Sacred plants/substances used
    ritual_objects: List[str]  # Objects used in ceremony
    songs_icaros: Optional[str]  # Musical/vocal elements
    prayers_invocations: List[str]  # Prayer traditions

    # Cultural Protocol
    cultural_protocols: List[str]  # Cultural protocols observed
    dietary_requirements: Optional[DietaryPreparation]  # Dieta requirements
    ethical_commitments: List[str]  # Ethical agreements

    # Form 29 Link
    folk_wisdom_reference: Optional[str]  # Cross-reference to Form 29


class CeremonialTradition(Enum):
    AMAZONIAN_AYAHUASCA_INDIGENOUS = "amazonian_ayahuasca_indigenous"
    AMAZONIAN_AYAHUASCA_MESTIZO = "amazonian_ayahuasca_mestizo"
    SANTO_DAIME = "santo_daime"
    UNIAO_DO_VEGETAL = "uniao_do_vegetal"
    MAZATEC_MUSHROOM = "mazatec_mushroom"
    NATIVE_AMERICAN_CHURCH = "native_american_church"
    BWITI_IBOGA = "bwiti_iboga"
    HUICHOL_PEYOTE = "huichol_peyote"
    NEO_SHAMANIC = "neo_shamanic"
    SYNCRETIC_MODERN = "syncretic_modern"
    CLINICAL_THERAPEUTIC = "clinical_therapeutic"
```

## Enumeration Types

### Classification Enumerations

```python
class ExperiencePhase(Enum):
    """Phases of a psychedelic experience"""
    BASELINE = "baseline"
    ONSET = "onset"
    COME_UP = "come_up"
    PEAK = "peak"
    PLATEAU = "plateau"
    COME_DOWN = "come_down"
    AFTER_EFFECTS = "after_effects"
    INTEGRATION = "integration"


class SafetyRiskLevel(Enum):
    """Safety risk classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DataQualityLevel(Enum):
    """Data quality classification"""
    CLINICAL_GRADE = "clinical_grade"
    RESEARCH_GRADE = "research_grade"
    OBSERVATIONAL = "observational"
    SELF_REPORT = "self_report"
    ANECDOTAL = "anecdotal"


class LegalStatus(Enum):
    """Legal status classification"""
    APPROVED_MEDICINE = "approved_medicine"
    BREAKTHROUGH_THERAPY = "breakthrough_therapy"
    INVESTIGATIONAL = "investigational"
    DECRIMINALIZED = "decriminalized"
    CONTROLLED_SCHEDULE_I = "controlled_schedule_i"
    CONTROLLED_SCHEDULE_II = "controlled_schedule_ii"
    CONTROLLED_SCHEDULE_III = "controlled_schedule_iii"
    UNSCHEDULED = "unscheduled"
    TRADITIONAL_EXCEPTION = "traditional_exception"


class QuestionnaireType(Enum):
    """Standard psychedelic research questionnaires"""
    MEQ_30 = "meq_30"  # Mystical Experience Questionnaire
    EDI = "edi"  # Ego Dissolution Inventory
    ASC = "asc"  # Altered States of Consciousness
    FIVE_D_ASC = "5d_asc"  # 5-Dimensional ASC
    CEQ = "ceq"  # Challenging Experience Questionnaire
    PANAS = "panas"  # Positive and Negative Affect Schedule
    BDI_II = "bdi_ii"  # Beck Depression Inventory
    PCL_5 = "pcl_5"  # PTSD Checklist
    CAPS_5 = "caps_5"  # Clinician-Administered PTSD Scale
    PHQ_9 = "phq_9"  # Patient Health Questionnaire
    QIDS_SR = "qids_sr"  # Quick Inventory of Depressive Symptoms
    AUDIT = "audit"  # Alcohol Use Disorders Identification Test
```

## Input/Output Structures

### Processing Input Structure

```python
@dataclass
class PsychedelicProcessingInput:
    """Input structure for psychedelic consciousness processing"""

    input_id: str  # Unique input identifier
    input_type: ProcessingInputType  # Type of processing request
    substance_data: Optional[SubstanceProfile]  # Substance information
    experience_data: Optional[PsychedelicExperience]  # Experience data
    context_data: Optional[Dict[str, Any]]  # Context information
    query_parameters: Dict[str, Any]  # Processing parameters
    requested_outputs: List[str]  # Requested output types
    priority: str = "normal"  # Processing priority
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ProcessingInputType(Enum):
    SUBSTANCE_QUERY = "substance_query"
    EXPERIENCE_ANALYSIS = "experience_analysis"
    PROTOCOL_RECOMMENDATION = "protocol_recommendation"
    RISK_ASSESSMENT = "risk_assessment"
    OUTCOME_PREDICTION = "outcome_prediction"
    RESEARCH_ANALYSIS = "research_analysis"
    CEREMONIAL_CONTEXT = "ceremonial_context"
    CROSS_FORM_INTEGRATION = "cross_form_integration"
```

### Processing Output Structure

```python
@dataclass
class PsychedelicProcessingOutput:
    """Output structure from psychedelic consciousness processing"""

    output_id: str  # Unique output identifier
    input_id: str  # Reference to input
    output_type: str  # Type of output generated
    results: Dict[str, Any]  # Processing results
    confidence_scores: Dict[str, float]  # Confidence by result component
    quality_assessment: QualityAssessment  # Output quality metrics
    warnings: List[str]  # Any warnings generated
    recommendations: List[str]  # Additional recommendations
    metadata: ProcessingMetadata  # Processing metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityAssessment:
    """Assessment of output quality"""

    overall_quality: float  # Overall quality (0.0-1.0)
    data_completeness: float  # Input data completeness (0.0-1.0)
    classification_confidence: float  # Classification confidence (0.0-1.0)
    prediction_confidence: float  # Prediction confidence (0.0-1.0)
    limitations: List[str]  # Known limitations of output
```

## Internal State Structures

### System State Management

```python
@dataclass
class PsychedelicSystemState:
    """Internal state of the psychedelic consciousness system"""

    # Processing State
    active_sessions: Dict[str, SessionState]  # Active processing sessions
    processing_queue: List[str]  # Queued processing requests
    resource_utilization: ResourceUtilization  # Current resource usage

    # Knowledge State
    substance_database_version: str  # Current substance DB version
    protocol_database_version: str  # Current protocol DB version
    research_database_version: str  # Current research DB version
    last_knowledge_update: datetime  # Last knowledge base update

    # Model State
    active_models: Dict[str, ModelState]  # Active prediction models
    model_performance_metrics: Dict[str, float]  # Model performance tracking
    calibration_status: Dict[str, CalibrationStatus]  # Model calibration

    # Safety State
    safety_monitoring_active: bool  # Safety monitoring status
    active_alerts: List[SafetyAlert]  # Current safety alerts
    last_safety_check: datetime  # Last safety system check


@dataclass
class SessionState:
    """State of an active processing session"""

    session_id: str
    session_type: str
    start_time: datetime
    current_phase: str
    progress_pct: float
    safety_status: str
    resources_allocated: Dict[str, Any]
```

## Relationship Mappings

### Cross-Form Data Exchange

```python
@dataclass
class CrossFormRelationshipMap:
    """Cross-form relationship mappings for psychedelic consciousness"""

    # Form 01 - Visual Consciousness
    visual_mapping: Dict[str, str]  # Visual hallucination -> Visual processing
    geometry_mapping: Dict[str, str]  # Form constants -> Visual patterns

    # Form 07 - Emotional Consciousness
    emotional_mapping: Dict[str, str]  # Emotional catharsis -> Emotion processing
    affect_modulation: Dict[str, str]  # Affect changes -> Emotional regulation

    # Form 10 - Self Recognition
    ego_mapping: Dict[str, str]  # Ego dissolution -> Self-model processing
    identity_mapping: Dict[str, str]  # Identity shifts -> Self-recognition

    # Form 27 - Altered States
    state_mapping: Dict[str, str]  # Psychedelic states -> Altered state taxonomy

    # Form 29 - Folk Wisdom
    traditional_mapping: Dict[str, str]  # Ceremonial contexts -> Indigenous knowledge

    # Form 36 - Contemplative States
    meditation_mapping: Dict[str, str]  # Meditation-psychedelic overlap

    # Form 39 - Trauma Consciousness
    therapeutic_mapping: Dict[str, str]  # MDMA-assisted therapy -> Trauma processing


@dataclass
class CrossFormDataExchange:
    """Data exchange record between forms"""

    exchange_id: str
    source_form: str  # e.g., "37-psychedelic-consciousness"
    target_form: str  # e.g., "07-emotional"
    data_type: str  # Type of data exchanged
    data_payload: Dict[str, Any]  # Exchanged data
    exchange_protocol: str  # Protocol used
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_status: str = "pending"
```

## Data Validation and Constraints

### Validation Rules

```python
class PsychedelicDataValidation:
    """Validation rules for psychedelic consciousness data structures"""

    @staticmethod
    def validate_substance_profile(profile: SubstanceProfile) -> ValidationResult:
        """Validate substance profile completeness and consistency"""
        errors = []
        warnings = []

        if not profile.substance_id:
            errors.append("Substance ID is required")
        if not profile.common_name:
            errors.append("Common name is required")
        if not profile.chemical_class:
            errors.append("Chemical class is required")
        if profile.receptor_profile and not profile.receptor_profile.primary_target:
            warnings.append("Primary receptor target not specified")
        if not profile.safety_profile:
            errors.append("Safety profile is required for all substances")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_experience_record(experience: PsychedelicExperience) -> ValidationResult:
        """Validate experience record"""
        errors = []
        warnings = []

        if not experience.substance_id:
            errors.append("Substance ID is required")
        if not experience.dose:
            errors.append("Dose information is required")
        if experience.intensity_profile and experience.intensity_profile.peak_intensity > 1.0:
            errors.append("Peak intensity must be normalized to 0.0-1.0")
        if not experience.set_assessment:
            warnings.append("Set assessment recommended for clinical records")
        if not experience.setting_assessment:
            warnings.append("Setting assessment recommended for clinical records")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_protocol(protocol: TherapeuticProtocol) -> ValidationResult:
        """Validate therapeutic protocol"""
        errors = []
        warnings = []

        if not protocol.protocol_id:
            errors.append("Protocol ID is required")
        if not protocol.contraindications:
            errors.append("Contraindications must be specified")
        if not protocol.adverse_event_protocol:
            errors.append("Adverse event protocol is required")
        if protocol.number_of_sessions < 1:
            errors.append("Must specify at least one medication session")
        if not protocol.primary_outcome_measures:
            warnings.append("Primary outcome measures not specified")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


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
class PsychedelicDataSerializer:
    """Serialization utilities for psychedelic consciousness data"""

    @staticmethod
    def serialize_substance(profile: SubstanceProfile) -> str:
        """Serialize substance profile to JSON"""
        return json.dumps(profile, cls=PsychedelicEncoder, indent=2)

    @staticmethod
    def serialize_experience(experience: PsychedelicExperience) -> str:
        """Serialize experience record to JSON"""
        return json.dumps(experience, cls=PsychedelicEncoder, indent=2)

    @staticmethod
    def deserialize_substance(json_str: str) -> SubstanceProfile:
        """Deserialize JSON to substance profile"""
        data = json.loads(json_str)
        return PsychedelicDecoder.decode_substance(data)


class PsychedelicEncoder(json.JSONEncoder):
    """Custom JSON encoder for psychedelic data structures"""

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

This comprehensive data structure specification provides the foundation for implementing robust psychedelic consciousness systems with proper pharmacological modeling, experience phenomenology tracking, therapeutic protocol management, and cross-form integration capabilities.
