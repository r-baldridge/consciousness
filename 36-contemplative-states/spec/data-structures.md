# Contemplative States Data Structures

## Overview

This document defines the core data structures used by Form 36 (Contemplative & Meditative States) for representing, storing, and exchanging information about contemplative practices, phenomenological reports, neural correlates, and tradition-specific state models. All structures use Python dataclasses and typing for clarity and validation.

---

## Core Data Models

### ContemplativeStateProfile

The canonical representation of a contemplative state, combining phenomenological description with neuroscientific correlates and tradition-specific context.

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum, auto
import uuid


@dataclass
class ContemplativeStateProfile:
    """Canonical profile for a contemplative or meditative state."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tradition: str = ""
    state_category: "ContemplativeStateCategory" = None
    canonical_description: str = ""
    phenomenological_markers: List["PhenomenologicalMarker"] = field(default_factory=list)
    neural_correlates: List["NeuralCorrelate"] = field(default_factory=list)
    prerequisite_states: List[str] = field(default_factory=list)
    successor_states: List[str] = field(default_factory=list)
    cross_tradition_equivalences: Dict[str, str] = field(default_factory=dict)
    depth_index: float = 0.0  # 0.0 (surface) to 1.0 (deepest absorption)
    stability_rating: float = 0.0  # 0.0 (fleeting) to 1.0 (stable)
    accessibility_level: "AccessibilityLevel" = None
    typical_duration_minutes: Optional[Tuple[float, float]] = None  # (min, max)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)
```

### MeditationSession

Records an individual practice session with real-time state tracking, physiological data, and self-report annotations.

```python
@dataclass
class MeditationSession:
    """Individual meditation or contemplative practice session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    practitioner_id: str = ""
    tradition: str = ""
    practice_type: "PracticeType" = None
    technique: str = ""
    start_time: datetime = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    session_context: "SessionContext" = None
    state_timeline: List["StateTransitionEvent"] = field(default_factory=list)
    phenomenological_reports: List["PhenomenologicalReport"] = field(default_factory=list)
    neural_recordings: List["NeuralRecording"] = field(default_factory=list)
    physiological_readings: List["PhysiologicalReading"] = field(default_factory=list)
    self_assessment: Optional["SessionSelfAssessment"] = None
    instructor_assessment: Optional["InstructorAssessment"] = None
    quality_score: Optional[float] = None  # 0.0 to 1.0
    disturbances: List["SessionDisturbance"] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
```

### TraditionProfile

Comprehensive documentation of a contemplative tradition including its state taxonomy, practices, and developmental maps.

```python
@dataclass
class TraditionProfile:
    """Profile of a contemplative or meditative tradition."""
    tradition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    lineage: str = ""
    geographic_origin: str = ""
    historical_period: str = ""
    primary_language: str = ""
    canonical_texts: List["CanonicalText"] = field(default_factory=list)
    state_taxonomy: List["TraditionStateTaxonomy"] = field(default_factory=list)
    practice_hierarchy: List["PracticeHierarchy"] = field(default_factory=list)
    developmental_map: Optional["DevelopmentalMap"] = None
    key_concepts: Dict[str, str] = field(default_factory=dict)
    teacher_lineages: List["TeacherLineage"] = field(default_factory=list)
    contemporary_representatives: List[str] = field(default_factory=list)
    research_corpus: List["ResearchReference"] = field(default_factory=list)
    cross_tradition_mappings: Dict[str, "CrossTraditionMapping"] = field(default_factory=dict)
```

### NeuralFinding

A neuroscientific research finding associated with a contemplative state or practice.

```python
@dataclass
class NeuralFinding:
    """Neuroscience research finding for a contemplative state."""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    researchers: List[str] = field(default_factory=list)
    publication_year: int = 0
    journal: str = ""
    doi: Optional[str] = None
    associated_state: str = ""
    associated_tradition: str = ""
    imaging_modality: "ImagingModality" = None
    brain_regions: List["BrainRegionActivation"] = field(default_factory=list)
    connectivity_changes: List["ConnectivityChange"] = field(default_factory=list)
    neurotransmitter_findings: List["NeurotransmitterFinding"] = field(default_factory=list)
    sample_size: int = 0
    practitioner_experience_hours: Optional[Tuple[float, float]] = None
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    replication_status: "ReplicationStatus" = None
    confidence_level: float = 0.0
    summary: str = ""
```

### PractitionerProfile

Profile of an individual engaged in contemplative practice, tracking their developmental history and current capacities.

```python
@dataclass
class PractitionerProfile:
    """Profile of a contemplative practitioner."""
    practitioner_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anonymized_id: str = ""
    primary_tradition: str = ""
    secondary_traditions: List[str] = field(default_factory=list)
    total_practice_hours: float = 0.0
    years_of_practice: float = 0.0
    retreat_days: int = 0
    current_daily_practice_minutes: float = 0.0
    attained_states: List[str] = field(default_factory=list)
    stable_attainments: List[str] = field(default_factory=list)
    developmental_stage: Optional["DevelopmentalStage"] = None
    teacher_verified: bool = False
    research_participant: bool = False
    baseline_neural_data: Optional["NeuralBaseline"] = None
    session_history_summary: "SessionHistorySummary" = None
    consent_status: "ConsentStatus" = None
```

---

## Enumeration Types

### ContemplativeStateCategory

```python
class ContemplativeStateCategory(Enum):
    """Primary classification of contemplative states."""
    ABSORPTION = auto()          # Jhana, samadhi, one-pointed concentration
    INSIGHT = auto()             # Vipassana nanas, kensho, direct seeing
    NON_DUAL = auto()            # Rigpa, turiya, witness consciousness
    MYSTICAL = auto()            # Fana, unitive experience, theosis
    FLOW = auto()                # Csikszentmihalyi flow, engaged presence
    DEVOTIONAL = auto()          # Bhakti absorption, centering prayer
    ENERGETIC = auto()           # Kundalini, tummo, chi states
    CESSATION = auto()           # Nirodha samapatti, fruition
    TRANSITIONAL = auto()        # Between-state liminal awareness
    ORDINARY_MINDFULNESS = auto()  # Present-centered non-reactive awareness
```

### PracticeType

```python
class PracticeType(Enum):
    """Type of contemplative practice."""
    CONCENTRATION = auto()       # Shamatha, samadhi, focused attention
    INSIGHT_MEDITATION = auto()  # Vipassana, shikantaza, inquiry
    OPEN_AWARENESS = auto()      # Choiceless awareness, dzogchen
    LOVING_KINDNESS = auto()     # Metta, tonglen, compassion practices
    MANTRA = auto()              # Japa, zikr, prayer repetition
    VISUALIZATION = auto()       # Deity yoga, guided imagery
    BODY_SCAN = auto()           # Progressive relaxation, yoga nidra
    MOVEMENT = auto()            # Walking meditation, tai chi, whirling
    BREATHWORK = auto()          # Pranayama, holotropic, tummo
    CONTEMPLATIVE_PRAYER = auto()  # Centering prayer, lectio divina
    KOAN_INQUIRY = auto()        # Zen koan work, self-inquiry
    CHANTING = auto()            # Kirtan, gregorian, sutra recitation
```

### ImagingModality

```python
class ImagingModality(Enum):
    """Neuroimaging modality used in contemplative research."""
    EEG = auto()
    FMRI = auto()
    SPECT = auto()
    PET = auto()
    MEG = auto()
    FNIRS = auto()
    DTI = auto()
    STRUCTURAL_MRI = auto()
    COMBINED_EEG_FMRI = auto()
```

### AccessibilityLevel

```python
class AccessibilityLevel(Enum):
    """How accessible a contemplative state is to practitioners."""
    BEGINNER = auto()            # Accessible within first year of practice
    INTERMEDIATE = auto()        # Typically 1-5 years of regular practice
    ADVANCED = auto()            # 5-15 years of dedicated practice
    EXPERT = auto()              # 15+ years, often retreat-intensive
    RARE = auto()                # Exceptional, few verified reports
```

### ReplicationStatus

```python
class ReplicationStatus(Enum):
    """Replication status of a neural finding."""
    INITIAL_FINDING = auto()
    PARTIALLY_REPLICATED = auto()
    WELL_REPLICATED = auto()
    FAILED_REPLICATION = auto()
    META_ANALYSIS_CONFIRMED = auto()
    CONTESTED = auto()
```

### DevelopmentalStage

```python
class DevelopmentalStage(Enum):
    """Broad developmental stages in contemplative practice."""
    NOVICE = auto()              # Learning basic techniques
    DEVELOPING = auto()          # Building stability and consistency
    PROFICIENT = auto()          # Reliable access to core states
    ADVANCED = auto()            # Deep states, cross-practice integration
    ADEPT = auto()               # Teacher-level realization
    MASTERY = auto()             # Verified attainment by qualified teacher
```

### ConsentStatus

```python
class ConsentStatus(Enum):
    """Consent status for research participation."""
    NOT_CONSENTED = auto()
    CONSENTED_BASIC = auto()
    CONSENTED_FULL = auto()
    CONSENTED_WITH_RESTRICTIONS = auto()
    CONSENT_WITHDRAWN = auto()
    CONSENT_EXPIRED = auto()
```

---

## Input/Output Structures

### PhenomenologicalReport

```python
@dataclass
class PhenomenologicalReport:
    """First-person report of contemplative experience."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: datetime = None
    report_type: str = ""  # "real-time", "retrospective", "interview"
    content: str = ""
    structured_ratings: "ExperienceRatings" = None
    identified_states: List[str] = field(default_factory=list)
    confidence: float = 0.0
    language: str = "en"
    tradition_vocabulary_used: List[str] = field(default_factory=list)


@dataclass
class ExperienceRatings:
    """Structured numerical ratings of phenomenological dimensions."""
    absorption_depth: float = 0.0        # 0-10 scale
    clarity: float = 0.0                 # 0-10 scale
    equanimity: float = 0.0              # 0-10 scale
    energy: float = 0.0                  # 0-10 scale
    bliss_pleasure: float = 0.0          # 0-10 scale
    spaciousness: float = 0.0            # 0-10 scale
    nondual_quality: float = 0.0         # 0-10 scale
    sense_of_self: float = 0.0           # 0-10 (strong self to no-self)
    time_distortion: float = 0.0         # 0-10 scale
    body_awareness: float = 0.0          # 0-10 scale
    emotional_tone: float = -5.0         # -5 to +5 (negative to positive)
    narrative_thought: float = 0.0       # 0-10 (none to continuous)
```

### StateTransitionEvent

```python
@dataclass
class StateTransitionEvent:
    """A detected or reported transition between contemplative states."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: datetime = None
    elapsed_minutes: float = 0.0
    from_state: str = ""
    to_state: str = ""
    transition_type: str = ""  # "gradual", "sudden", "oscillating"
    trigger: Optional[str] = None
    confidence: float = 0.0
    detection_method: str = ""  # "self-report", "neural", "physiological", "combined"
    neural_signature: Optional[Dict[str, float]] = None
    duration_seconds: float = 0.0  # How long the transition itself took
```

### NeuralRecording

```python
@dataclass
class NeuralRecording:
    """Neural data recording from a contemplative session."""
    recording_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    modality: ImagingModality = None
    sampling_rate_hz: float = 0.0
    channels: List[str] = field(default_factory=list)
    start_time: datetime = None
    duration_seconds: float = 0.0
    data_uri: str = ""  # Reference to raw data storage
    preprocessed_data_uri: Optional[str] = None
    artifact_rejection_applied: bool = False
    quality_score: float = 0.0
    annotations: List["RecordingAnnotation"] = field(default_factory=list)
    frequency_band_powers: Optional[Dict[str, List[float]]] = None
    connectivity_matrices: Optional[List["ConnectivityMatrix"]] = None


@dataclass
class RecordingAnnotation:
    """Time-stamped annotation on a neural recording."""
    timestamp_seconds: float = 0.0
    label: str = ""
    annotator: str = ""  # "practitioner", "researcher", "algorithm"
    confidence: float = 0.0
    note: str = ""
```

### PhysiologicalReading

```python
@dataclass
class PhysiologicalReading:
    """Physiological measurement from a contemplative session."""
    reading_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    measurement_type: str = ""  # "heart_rate", "hrv", "gsr", "respiration", "temperature"
    timestamp: datetime = None
    value: float = 0.0
    unit: str = ""
    quality: float = 0.0
    device: str = ""
```

### SessionSelfAssessment

```python
@dataclass
class SessionSelfAssessment:
    """Post-session self-assessment by the practitioner."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    overall_quality: float = 0.0  # 0-10
    deepest_state_reached: str = ""
    primary_hindrance: Optional[str] = None  # "sloth", "restlessness", "doubt", etc.
    breakthrough_event: bool = False
    breakthrough_description: Optional[str] = None
    practice_notes: str = ""
    comparative_rating: str = ""  # "below_average", "average", "above_average", "exceptional"
    follow_up_intentions: List[str] = field(default_factory=list)
```

---

## Internal State Structures

### ContemplativeStateDetector

```python
@dataclass
class ContemplativeStateDetector:
    """Internal model for real-time contemplative state detection."""
    detector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active_model: str = ""
    model_version: str = ""
    supported_states: List[str] = field(default_factory=list)
    current_state_estimate: str = ""
    state_probabilities: Dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.7
    transition_smoothing_window_seconds: float = 5.0
    last_update: datetime = None
    buffer_size: int = 256
    feature_weights: Dict[str, float] = field(default_factory=dict)
    calibration_data: Optional["CalibrationProfile"] = None


@dataclass
class CalibrationProfile:
    """Practitioner-specific calibration for state detection."""
    practitioner_id: str = ""
    calibration_date: datetime = None
    baseline_eeg_features: Dict[str, float] = field(default_factory=dict)
    baseline_physiology: Dict[str, float] = field(default_factory=dict)
    state_signatures: Dict[str, Dict[str, float]] = field(default_factory=dict)
    accuracy_estimates: Dict[str, float] = field(default_factory=dict)
```

### DevelopmentalMap

```python
@dataclass
class DevelopmentalMap:
    """Hierarchical map of contemplative development within a tradition."""
    map_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tradition: str = ""
    stages: List["DevelopmentalMapStage"] = field(default_factory=list)
    transition_requirements: Dict[str, List[str]] = field(default_factory=dict)
    typical_timelines: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    verification_criteria: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DevelopmentalMapStage:
    """A single stage in a developmental map."""
    stage_id: str = ""
    name: str = ""
    tradition_name: str = ""  # Name in the tradition's language
    description: str = ""
    characteristic_states: List[str] = field(default_factory=list)
    characteristic_capacities: List[str] = field(default_factory=list)
    common_challenges: List[str] = field(default_factory=list)
    order_index: int = 0
```

### CrossTraditionMapping

```python
@dataclass
class CrossTraditionMapping:
    """Mapping between equivalent states or stages across traditions."""
    mapping_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_tradition: str = ""
    target_tradition: str = ""
    source_state: str = ""
    target_state: str = ""
    equivalence_type: str = ""  # "strong", "partial", "analogous", "debated"
    supporting_evidence: List[str] = field(default_factory=list)
    scholarly_references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
```

---

## Relationship Mappings (Cross-Form Data Exchange)

### Form 36 exchanges data with multiple other consciousness forms. The following structures define the interchange formats.

### Integration with Form 39 (Trauma Consciousness)

```python
@dataclass
class ContemplativeTraumaInterface:
    """Data exchange structure between Form 36 and Form 39 (Trauma Consciousness).

    Tracks how contemplative states interact with trauma processing,
    including contraindications and trauma-sensitive adaptations.
    """
    practitioner_id: str = ""
    trauma_flags: List[str] = field(default_factory=list)
    contraindicated_practices: List[str] = field(default_factory=list)
    trauma_sensitive_modifications: Dict[str, str] = field(default_factory=dict)
    dissociative_risk_level: float = 0.0  # 0.0 to 1.0
    grounding_protocols: List[str] = field(default_factory=list)
    safe_depth_limit: float = 0.0  # Maximum safe absorption depth
    monitoring_requirements: List[str] = field(default_factory=list)
    therapeutic_integration_notes: str = ""
```

### Integration with Form 40 (Xenoconsciousness)

```python
@dataclass
class ContemplativeXenoInterface:
    """Data exchange structure between Form 36 and Form 40 (Xenoconsciousness).

    Enables comparison of contemplative states with non-human or
    hypothetical alien consciousness models.
    """
    state_id: str = ""
    human_specificity_score: float = 0.0  # How human-specific this state is
    universal_consciousness_markers: List[str] = field(default_factory=list)
    substrate_independence_level: float = 0.0
    cross_species_analogues: Dict[str, str] = field(default_factory=dict)
    phenomenological_universals: List[str] = field(default_factory=list)
    embodiment_dependency: float = 0.0  # 0.0 (fully disembodied) to 1.0 (fully embodied)
```

### General Cross-Form Exchange Envelope

```python
@dataclass
class ContemplativeDataEnvelope:
    """Standard envelope for cross-form data exchange."""
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: str = "form-36-contemplative-states"
    target_form: str = ""
    payload_type: str = ""
    payload: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    protocol_version: str = "1.0"
    requires_acknowledgment: bool = False
    priority: str = "normal"  # "low", "normal", "high", "critical"
    expiry: Optional[datetime] = None
```

---

## Appendix: Type Aliases and Utility Types

```python
from typing import TypeAlias, NewType

# Identifiers
StateID: TypeAlias = str
SessionID: TypeAlias = str
PractitionerID: TypeAlias = str
TraditionID: TypeAlias = str

# Measurements
DepthScore = NewType("DepthScore", float)       # 0.0 to 1.0
ConfidenceScore = NewType("ConfidenceScore", float)  # 0.0 to 1.0
FrequencyHz = NewType("FrequencyHz", float)
DurationMinutes = NewType("DurationMinutes", float)

# Collections
StateTimeline: TypeAlias = List[StateTransitionEvent]
FrequencyBandPowers: TypeAlias = Dict[str, List[float]]  # band_name -> power values
NeuralFeatureVector: TypeAlias = List[float]
```
