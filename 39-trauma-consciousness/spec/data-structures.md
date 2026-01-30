# Trauma Consciousness Data Structures

## Overview

This document defines the core data structures for Form 39 (Trauma & Dissociative Consciousness), covering representations of trauma experiences, dissociative states, autonomic nervous system profiles, recovery trajectories, and survivor-centered data exchange. All structures adhere to trauma-informed principles: safety, trustworthiness, choice, collaboration, and empowerment. Python dataclasses, enums, and typing are used throughout for clarity and validation.

---

## Core Data Models

### TraumaProfile

The central representation of an individual's trauma history, current state, and recovery trajectory. Designed to be survivor-controlled with granular consent.

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, date
from enum import Enum, auto
import uuid


@dataclass
class TraumaProfile:
    """Comprehensive trauma profile for an individual.

    This profile is survivor-owned. The survivor controls what is
    recorded, who can access it, and can request deletion at any time.
    """
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anonymized_id: str = ""
    creation_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    consent_status: "TraumaConsentStatus" = None
    trauma_history: List["TraumaEvent"] = field(default_factory=list)
    current_diagnoses: List["TraumaDiagnosis"] = field(default_factory=list)
    dissociative_profile: Optional["DissociativeProfile"] = None
    polyvagal_profile: Optional["PolyvagalProfile"] = None
    window_of_tolerance: Optional["WindowOfTolerance"] = None
    attachment_pattern: Optional["AttachmentPattern"] = None
    recovery_stage: "RecoveryStage" = None
    resilience_factors: List["ResilienceFactor"] = field(default_factory=list)
    active_safety_plan: Optional["SafetyPlan"] = None
    support_network: List["SupportContact"] = field(default_factory=list)
    treatment_history: List["TreatmentRecord"] = field(default_factory=list)
    cultural_context: Optional["CulturalContext"] = None
    access_permissions: Dict[str, "AccessLevel"] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
```

### TraumaEvent

Records a traumatic experience with sensitivity to the survivor's readiness to disclose and level of detail they choose to share.

```python
@dataclass
class TraumaEvent:
    """A recorded traumatic event or experience.

    Detail level is always survivor-controlled. The system supports
    minimal to full documentation based on the survivor's choice.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    trauma_type: "TraumaType" = None
    trauma_category: "TraumaCategory" = None
    disclosure_level: "DisclosureLevel" = None
    age_at_event: Optional[float] = None
    developmental_period: Optional["DevelopmentalPeriod"] = None
    duration_type: "TraumaDurationType" = None
    duration_description: Optional[str] = None
    perpetrator_relationship: Optional["PerpetratorRelationship"] = None
    somatic_impacts: List["SomaticImpact"] = field(default_factory=list)
    consciousness_impacts: List["ConsciousnessImpact"] = field(default_factory=list)
    adaptive_responses_developed: List[str] = field(default_factory=list)
    current_activation_triggers: List["Trigger"] = field(default_factory=list)
    narrative_fragment: Optional[str] = None  # Only if survivor chooses to share
    processing_status: "ProcessingStatus" = None
    associated_parts: List[str] = field(default_factory=list)  # IFS parts connected to this event
    intergenerational: bool = False
    collective: bool = False
    recorded_date: datetime = field(default_factory=datetime.utcnow)
```

### DissociativeProfile

Models the dissociative landscape of a survivor according to the Structural Dissociation model (van der Hart, Nijenhuis).

```python
@dataclass
class DissociativeProfile:
    """Dissociative profile based on Structural Dissociation theory."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    structural_dissociation_level: "StructuralDissociationLevel" = None
    apparently_normal_parts: List["PersonalityPart"] = field(default_factory=list)
    emotional_parts: List["PersonalityPart"] = field(default_factory=list)
    dissociative_symptoms: List["DissociativeSymptom"] = field(default_factory=list)
    des_score: Optional[float] = None  # Dissociative Experiences Scale (0-100)
    mdd_score: Optional[float] = None  # Multidimensional Inventory of Dissociation
    switching_frequency: Optional[str] = None  # "rare", "occasional", "frequent", "very_frequent"
    co_consciousness_level: Optional[float] = None  # 0.0 (none) to 1.0 (full)
    amnesia_barriers: List["AmnesiaBarrier"] = field(default_factory=list)
    system_map: Optional["InternalSystemMap"] = None
    integration_progress: float = 0.0  # 0.0 to 1.0
    last_assessment_date: Optional[datetime] = None


@dataclass
class PersonalityPart:
    """A personality part in the structural dissociation framework.

    Represents an ANP (Apparently Normal Part) or EP (Emotional Part)
    with its own characteristics, roles, and relationships.
    """
    part_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    part_type: str = ""  # "ANP" or "EP"
    name_or_label: Optional[str] = None  # Survivor-chosen identifier
    role: str = ""  # "protector", "exile", "manager", "firefighter" (IFS), or custom
    age_presentation: Optional[float] = None
    emotional_range: List[str] = field(default_factory=list)
    primary_affect: str = ""
    somatic_signature: Optional[str] = None
    triggers: List[str] = field(default_factory=list)
    defensive_action_tendencies: List[str] = field(default_factory=list)
    cognitive_style: Optional[str] = None
    relational_stance: Optional[str] = None
    associated_memories: List[str] = field(default_factory=list)
    communication_capacity: float = 0.0  # 0.0 (no internal communication) to 1.0 (full)
    integration_readiness: float = 0.0
```

### PolyvagalProfile

Models the autonomic nervous system state based on Stephen Porges' Polyvagal Theory.

```python
@dataclass
class PolyvagalProfile:
    """Autonomic nervous system profile based on Polyvagal Theory."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_state: "PolyvagalState" = None
    baseline_state: "PolyvagalState" = None
    ventral_vagal_capacity: float = 0.0  # 0.0 to 1.0
    sympathetic_reactivity: float = 0.0  # 0.0 to 1.0
    dorsal_vagal_tendency: float = 0.0   # 0.0 to 1.0
    state_flexibility: float = 0.0       # Ability to shift between states
    social_engagement_capacity: float = 0.0
    neuroception_accuracy: float = 0.0   # How accurately the system detects safety/danger
    vagal_tone_hrv: Optional[float] = None  # Heart rate variability metric
    state_history: List["PolyvagalStateEvent"] = field(default_factory=list)
    co_regulation_capacity: float = 0.0
    self_regulation_capacity: float = 0.0
    glimmers: List[str] = field(default_factory=list)  # Micro-moments of ventral vagal activation
    anchors: List[str] = field(default_factory=list)    # Reliable ventral vagal resources


@dataclass
class PolyvagalStateEvent:
    """A recorded autonomic state change."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = None
    from_state: "PolyvagalState" = None
    to_state: "PolyvagalState" = None
    trigger: Optional[str] = None
    duration_minutes: Optional[float] = None
    regulation_method_used: Optional[str] = None
    return_to_baseline_minutes: Optional[float] = None
```

### SafetyPlan

An active safety plan with grounding resources, contacts, and escalation procedures.

```python
@dataclass
class SafetyPlan:
    """Active safety plan for a trauma survivor."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_reviewed: datetime = field(default_factory=datetime.utcnow)
    review_frequency_days: int = 30
    warning_signs: List[str] = field(default_factory=list)
    grounding_techniques: List["GroundingTechnique"] = field(default_factory=list)
    internal_coping_strategies: List[str] = field(default_factory=list)
    safe_people: List["SupportContact"] = field(default_factory=list)
    professional_contacts: List["SupportContact"] = field(default_factory=list)
    crisis_resources: List["CrisisResource"] = field(default_factory=list)
    environment_safety_steps: List[str] = field(default_factory=list)
    reasons_for_living: List[str] = field(default_factory=list)
    lethal_means_restriction: Optional[str] = None
    active: bool = True


@dataclass
class GroundingTechnique:
    """A grounding technique with effectiveness tracking."""
    technique_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""  # "sensory", "cognitive", "somatic", "relational", "movement"
    instructions: str = ""
    effectiveness_rating: float = 0.0  # 0.0 to 1.0, survivor-rated
    applicable_states: List["PolyvagalState"] = field(default_factory=list)
    contraindicated_states: List[str] = field(default_factory=list)
    duration_minutes: float = 0.0
    requires_props: bool = False
    props_needed: List[str] = field(default_factory=list)
```

### WindowOfTolerance

Models the survivor's current window of tolerance and zone tracking.

```python
@dataclass
class WindowOfTolerance:
    """Window of tolerance model for arousal regulation."""
    profile_id: str = ""
    current_arousal_level: float = 0.0  # -1.0 (hypoarousal) to +1.0 (hyperarousal), 0.0 = optimal
    window_width: float = 0.5  # How wide the tolerance window is (0.0 = very narrow, 1.0 = very wide)
    current_zone: "ArousalZone" = None
    hyperarousal_threshold: float = 0.5
    hypoarousal_threshold: float = -0.5
    expansion_trend: float = 0.0  # Positive = expanding over time, negative = contracting
    zone_history: List["ZoneEvent"] = field(default_factory=list)
    regulation_resources: List[str] = field(default_factory=list)
    known_triggers_hyperarousal: List[str] = field(default_factory=list)
    known_triggers_hypoarousal: List[str] = field(default_factory=list)


@dataclass
class ZoneEvent:
    """A recorded window-of-tolerance zone transition."""
    timestamp: datetime = None
    from_zone: "ArousalZone" = None
    to_zone: "ArousalZone" = None
    arousal_level: float = 0.0
    trigger: Optional[str] = None
    regulation_applied: Optional[str] = None
    duration_in_zone_minutes: Optional[float] = None
```

---

## Enumeration Types

### TraumaType

```python
class TraumaType(Enum):
    """Primary trauma type classification."""
    ACUTE_SINGLE_INCIDENT = auto()
    COMPLEX_DEVELOPMENTAL = auto()
    CHRONIC_ONGOING = auto()
    ATTACHMENT_TRAUMA = auto()
    BETRAYAL_TRAUMA = auto()
    RELATIONAL_TRAUMA = auto()
    MEDICAL_TRAUMA = auto()
    BIRTH_PRENATAL = auto()
    COMBAT_WAR = auto()
    NATURAL_DISASTER = auto()
    INTERPERSONAL_VIOLENCE = auto()
    SEXUAL_TRAUMA = auto()
    INTERGENERATIONAL = auto()
    COLLECTIVE_HISTORICAL = auto()
    RACIAL_TRAUMA = auto()
    MORAL_INJURY = auto()
    VICARIOUS_SECONDARY = auto()
```

### TraumaCategory

```python
class TraumaCategory(Enum):
    """Broad category of trauma by context."""
    INTERPERSONAL = auto()
    NON_INTERPERSONAL = auto()
    SYSTEMIC = auto()
    COLLECTIVE = auto()
    DEVELOPMENTAL = auto()
    INTERGENERATIONAL = auto()
```

### StructuralDissociationLevel

```python
class StructuralDissociationLevel(Enum):
    """Level of structural dissociation (van der Hart/Nijenhuis)."""
    PRIMARY = auto()     # One ANP, one EP (simple PTSD)
    SECONDARY = auto()   # One ANP, multiple EPs (Complex PTSD, OSDD)
    TERTIARY = auto()    # Multiple ANPs, multiple EPs (DID)
```

### PolyvagalState

```python
class PolyvagalState(Enum):
    """Autonomic nervous system state per Polyvagal Theory."""
    VENTRAL_VAGAL = auto()         # Safe, socially engaged
    SYMPATHETIC_ACTIVATED = auto()  # Fight/flight mobilization
    DORSAL_VAGAL = auto()          # Shutdown, collapse, conservation
    SYMPATHETIC_DORSAL_BLEND = auto()  # Freeze (immobilization with fear)
    VENTRAL_SYMPATHETIC_BLEND = auto()  # Play, healthy mobilization
    VENTRAL_DORSAL_BLEND = auto()      # Stillness without fear, intimacy
```

### ArousalZone

```python
class ArousalZone(Enum):
    """Zone within the window of tolerance model."""
    HYPERAROUSAL = auto()   # Above window: anxiety, panic, hypervigilance
    OPTIMAL = auto()        # Within window: flexible, regulated
    HYPOAROUSAL = auto()    # Below window: numbness, dissociation, collapse
    TRANSITIONAL = auto()   # Shifting between zones
```

### RecoveryStage

```python
class RecoveryStage(Enum):
    """Recovery stage per Judith Herman's model."""
    PRE_TREATMENT = auto()          # Before formal recovery begins
    STAGE_1_SAFETY = auto()         # Establishing safety and stabilization
    STAGE_2_REMEMBRANCE = auto()    # Remembrance and mourning
    STAGE_3_RECONNECTION = auto()   # Reconnection with ordinary life
    POST_RECOVERY_GROWTH = auto()   # Post-traumatic growth phase
```

### DisclosureLevel

```python
class DisclosureLevel(Enum):
    """Level of detail the survivor chooses to share."""
    MINIMAL = auto()       # Existence acknowledged only
    BASIC = auto()         # Type and general timeframe
    MODERATE = auto()      # Type, timeframe, impacts described
    DETAILED = auto()      # Full narrative with impacts
    COMPREHENSIVE = auto() # All available detail including somatic/neural
```

### ProcessingStatus

```python
class ProcessingStatus(Enum):
    """Processing status of a trauma memory."""
    UNPROCESSED = auto()           # Not yet addressed in treatment
    STABILIZATION_PHASE = auto()   # Building resources before processing
    ACTIVE_PROCESSING = auto()     # Currently being processed in treatment
    PARTIALLY_PROCESSED = auto()   # Some integration achieved
    INTEGRATED = auto()            # Fully processed and integrated
    REACTIVATED = auto()           # Previously processed but re-triggered
```

### DevelopmentalPeriod

```python
class DevelopmentalPeriod(Enum):
    """Developmental period during which trauma occurred."""
    PRENATAL = auto()
    INFANCY = auto()         # 0-2 years
    EARLY_CHILDHOOD = auto() # 2-6 years
    MIDDLE_CHILDHOOD = auto() # 6-12 years
    ADOLESCENCE = auto()     # 12-18 years
    EARLY_ADULTHOOD = auto() # 18-25 years
    ADULTHOOD = auto()       # 25-65 years
    LATE_ADULTHOOD = auto()  # 65+ years
```

### TraumaConsentStatus

```python
class TraumaConsentStatus(Enum):
    """Granular consent status for trauma data."""
    FULL_CONSENT = auto()
    LIMITED_CONSENT = auto()          # Specific restrictions documented
    ANONYMIZED_RESEARCH_ONLY = auto()
    TREATMENT_ONLY = auto()
    REVOKED = auto()
    PENDING_REVIEW = auto()
```

### PerpetratorRelationship

```python
class PerpetratorRelationship(Enum):
    """Relationship of perpetrator to survivor (if applicable)."""
    PARENT_CAREGIVER = auto()
    FAMILY_MEMBER = auto()
    INTIMATE_PARTNER = auto()
    AUTHORITY_FIGURE = auto()
    PEER = auto()
    STRANGER = auto()
    INSTITUTIONAL = auto()
    STATE_SYSTEMIC = auto()
    NOT_APPLICABLE = auto()
    UNDISCLOSED = auto()
```

---

## Input/Output Structures

### TraumaAssessmentInput

```python
@dataclass
class TraumaAssessmentInput:
    """Input structure for trauma assessment processing."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    assessment_type: str = ""  # "initial", "periodic", "crisis", "progress"
    presenting_concerns: List[str] = field(default_factory=list)
    current_symptoms: List["TraumaSymptom"] = field(default_factory=list)
    current_polyvagal_state: Optional["PolyvagalState"] = None
    current_arousal_level: Optional[float] = None
    recent_triggers: List["Trigger"] = field(default_factory=list)
    current_safety_level: float = 0.0  # 0.0 (unsafe) to 1.0 (fully safe)
    available_resources: List[str] = field(default_factory=list)
    survivor_stated_goals: List[str] = field(default_factory=list)
    assessor_observations: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TraumaSymptom:
    """A reported trauma symptom with severity and context."""
    symptom_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""  # "intrusion", "avoidance", "negative_cognition", "arousal", "dissociative", "somatic"
    description: str = ""
    severity: float = 0.0  # 0.0 to 1.0
    frequency: str = ""    # "rarely", "sometimes", "often", "constantly"
    onset: Optional[str] = None
    associated_triggers: List[str] = field(default_factory=list)
    functional_impact: float = 0.0  # 0.0 (no impact) to 1.0 (completely disabling)
```

### TraumaAssessmentOutput

```python
@dataclass
class TraumaAssessmentOutput:
    """Output from trauma assessment processing."""
    assessment_id: str = ""
    profile_id: str = ""
    risk_level: "RiskLevel" = None
    recommended_recovery_stage: "RecoveryStage" = None
    safety_plan_update_needed: bool = False
    recommended_interventions: List["InterventionRecommendation"] = field(default_factory=list)
    resource_connections: List["ResourceConnection"] = field(default_factory=list)
    grounding_needs: List[str] = field(default_factory=list)
    cross_form_alerts: List["CrossFormAlert"] = field(default_factory=list)
    confidence: float = 0.0
    clinician_review_required: bool = True  # Always true for safety
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InterventionRecommendation:
    """A recommended therapeutic intervention."""
    intervention_type: str = ""  # "EMDR", "SE", "IFS", "CPT", "DBT", "sensorimotor"
    rationale: str = ""
    priority: str = ""  # "immediate", "short_term", "ongoing"
    contraindications: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    evidence_level: str = ""  # "strong", "moderate", "emerging", "clinical_consensus"
```

### Trigger

```python
@dataclass
class Trigger:
    """A trauma trigger with context and management strategies."""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""  # "sensory", "relational", "situational", "temporal", "somatic", "emotional"
    description: str = ""
    sensory_modality: Optional[str] = None  # "visual", "auditory", "olfactory", "tactile", "gustatory"
    intensity: float = 0.0  # 0.0 to 1.0
    predictability: float = 0.0  # 0.0 (unpredictable) to 1.0 (fully predictable)
    associated_events: List[str] = field(default_factory=list)
    management_strategies: List[str] = field(default_factory=list)
    avoidance_level: float = 0.0  # Current avoidance behavior intensity
```

---

## Internal State Structures

### TraumaProcessingEngine

```python
@dataclass
class TraumaProcessingEngine:
    """Internal processing state for trauma consciousness modeling."""
    engine_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active_profile_id: str = ""
    safety_check_status: "SafetyCheckStatus" = None
    current_processing_mode: str = ""  # "assessment", "stabilization", "processing", "integration"
    containment_active: bool = False
    containment_resources: List[str] = field(default_factory=list)
    titration_level: float = 0.5  # 0.0 (minimal exposure) to 1.0 (full exposure)
    pendulation_state: str = ""  # "resource", "activation", "settling" (SE concept)
    window_of_tolerance_monitor: Optional["WindowOfTolerance"] = None
    emergency_protocol_armed: bool = True
    session_safety_score: float = 1.0
    last_safety_check: datetime = None
    intervention_history: List["InterventionEvent"] = field(default_factory=list)


@dataclass
class InterventionEvent:
    """A recorded intervention during processing."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = None
    intervention_type: str = ""
    reason: str = ""
    outcome: str = ""  # "effective", "partially_effective", "ineffective", "escalated"
    arousal_change: float = 0.0
    notes: str = ""
```

### InternalSystemMap

```python
@dataclass
class InternalSystemMap:
    """Map of the internal system of parts (IFS/Structural Dissociation).

    This structure represents the survivor's internal landscape of parts,
    their relationships, and communication patterns.
    """
    map_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    parts: List["PersonalityPart"] = field(default_factory=list)
    part_relationships: List["PartRelationship"] = field(default_factory=list)
    self_energy_access: float = 0.0  # IFS: access to core Self (0.0 to 1.0)
    internal_communication_level: float = 0.0  # Overall system communication
    dominant_part_id: Optional[str] = None  # Currently fronting part
    recently_active_parts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PartRelationship:
    """Relationship between two internal parts."""
    part_a_id: str = ""
    part_b_id: str = ""
    relationship_type: str = ""  # "protective", "conflicted", "cooperative", "unknown", "blended"
    communication_quality: float = 0.0  # 0.0 (none) to 1.0 (clear communication)
    amnesia_barrier: bool = False
    notes: str = ""
```

### SomaticImpact

```python
@dataclass
class SomaticImpact:
    """Somatic (body-based) impact of trauma."""
    impact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    body_region: str = ""  # "head", "throat", "chest", "abdomen", "pelvis", "limbs", "whole_body"
    sensation_type: str = ""  # "tension", "pain", "numbness", "tingling", "constriction", "trembling"
    intensity: float = 0.0  # 0.0 to 1.0
    chronic: bool = False
    associated_trauma_event: Optional[str] = None
    associated_emotional_state: Optional[str] = None
    body_keeps_score_pattern: Optional[str] = None  # van der Kolk pattern description
    somatic_resource: Optional[str] = None  # Known somatic resource for this impact
```

---

## Relationship Mappings (Cross-Form Data Exchange)

### Integration with Form 36 (Contemplative States)

```python
@dataclass
class TraumaContemplativeInterface:
    """Data exchange between Form 39 and Form 36 (Contemplative States).

    Manages safety considerations when trauma survivors engage in
    contemplative practices, including contraindications and
    trauma-sensitive adaptations.
    """
    profile_id: str = ""
    risk_assessment: "ContemplativeRiskAssessment" = None
    contraindicated_practices: List[str] = field(default_factory=list)
    safe_practices: List[str] = field(default_factory=list)
    required_modifications: Dict[str, str] = field(default_factory=dict)
    maximum_safe_depth: float = 0.0
    dissociative_risk_during_practice: float = 0.0
    grounding_protocols_required: bool = True
    teacher_notification_required: bool = False
    monitoring_frequency: str = ""  # "continuous", "periodic", "standard"
    emergency_contact_info: Optional["SupportContact"] = None


@dataclass
class ContemplativeRiskAssessment:
    """Risk assessment for contemplative practice engagement."""
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    overall_risk: str = ""  # "low", "moderate", "high", "contraindicated"
    specific_risks: Dict[str, str] = field(default_factory=dict)
    protective_factors: List[str] = field(default_factory=list)
    recommended_review_interval_days: int = 30
```

### Integration with Form 40 (Xenoconsciousness)

```python
@dataclass
class TraumaXenoInterface:
    """Data exchange between Form 39 and Form 40 (Xenoconsciousness).

    Explores how trauma responses may be universal across conscious
    entities or specific to human embodiment and social structures.
    """
    trauma_response_type: str = ""
    human_specificity: float = 0.0  # 0.0 (universal) to 1.0 (uniquely human)
    biological_substrate_dependency: float = 0.0
    social_structure_dependency: float = 0.0
    cross_species_analogues: Dict[str, str] = field(default_factory=dict)
    universal_threat_responses: List[str] = field(default_factory=list)
    embodiment_requirements: List[str] = field(default_factory=list)
    consciousness_fragmentation_universality: float = 0.0
```

### Cross-Form Alert Structure

```python
@dataclass
class CrossFormAlert:
    """Safety alert sent from Form 39 to other forms."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: str = "form-39-trauma-consciousness"
    target_form: str = ""
    alert_type: str = ""  # "safety_warning", "contraindication", "risk_update", "emergency"
    severity: str = ""    # "informational", "caution", "warning", "critical"
    profile_id: str = ""
    message: str = ""
    recommended_action: str = ""
    expiry: Optional[datetime] = None
    requires_acknowledgment: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### General Cross-Form Exchange Envelope

```python
@dataclass
class TraumaDataEnvelope:
    """Standard envelope for cross-form data exchange from Form 39.

    All outbound data is filtered through safety and consent checks
    before transmission.
    """
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_form: str = "form-39-trauma-consciousness"
    target_form: str = ""
    payload_type: str = ""
    payload: Dict = field(default_factory=dict)
    consent_verified: bool = False
    safety_checked: bool = False
    anonymized: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    protocol_version: str = "1.0"
    priority: str = "normal"
    sensitivity_level: str = "high"  # Trauma data defaults to high sensitivity
```

---

## Appendix: Type Aliases and Utility Types

```python
from typing import TypeAlias, NewType

# Identifiers
ProfileID: TypeAlias = str
EventID: TypeAlias = str
PartID: TypeAlias = str
PlanID: TypeAlias = str

# Measurements
SafetyScore = NewType("SafetyScore", float)          # 0.0 to 1.0
ArousalLevel = NewType("ArousalLevel", float)         # -1.0 to +1.0
SeverityScore = NewType("SeverityScore", float)       # 0.0 to 1.0
IntegrationScore = NewType("IntegrationScore", float) # 0.0 to 1.0
RiskScore = NewType("RiskScore", float)               # 0.0 to 1.0

# Enums for quick reference
class RiskLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

class SafetyCheckStatus(Enum):
    PASSED = auto()
    WARNING = auto()
    FAILED = auto()
    REQUIRES_REVIEW = auto()

class AccessLevel(Enum):
    NONE = auto()
    READ_SUMMARY = auto()
    READ_FULL = auto()
    READ_WRITE = auto()
    ADMIN = auto()

class TraumaDurationType(Enum):
    SINGLE_EVENT = auto()
    SHORT_TERM = auto()       # Days to weeks
    MEDIUM_TERM = auto()      # Weeks to months
    LONG_TERM = auto()        # Months to years
    ONGOING = auto()          # Currently continuing
    INTERGENERATIONAL = auto() # Transmitted across generations
```

---

## Appendix: Supporting Structures

```python
@dataclass
class SupportContact:
    """A support contact in the survivor's network."""
    contact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # "therapist", "friend", "family", "crisis_line", "peer_support"
    name_or_label: str = ""
    contact_method: str = ""  # "phone", "text", "in_person", "online"
    availability: str = ""
    is_trauma_informed: bool = False
    priority_order: int = 0


@dataclass
class CrisisResource:
    """A crisis resource with contact information."""
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    phone: Optional[str] = None
    text_line: Optional[str] = None
    url: Optional[str] = None
    hours: str = ""  # "24/7", business hours, etc.
    specialization: Optional[str] = None
    language_support: List[str] = field(default_factory=list)


@dataclass
class TreatmentRecord:
    """Record of a treatment modality used."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modality: str = ""  # "EMDR", "SE", "IFS", "CPT", "DBT", "sensorimotor", etc.
    provider_type: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    sessions_completed: int = 0
    effectiveness_rating: Optional[float] = None
    survivor_satisfaction: Optional[float] = None
    notes: str = ""


@dataclass
class CulturalContext:
    """Cultural context relevant to trauma experience and healing."""
    cultural_identity: List[str] = field(default_factory=list)
    language_preferences: List[str] = field(default_factory=list)
    culturally_specific_healing: List[str] = field(default_factory=list)
    spiritual_resources: List[str] = field(default_factory=list)
    community_resources: List[str] = field(default_factory=list)
    systemic_factors: List[str] = field(default_factory=list)
    immigration_status_relevant: bool = False
    intergenerational_context: Optional[str] = None


@dataclass
class ConsciousnessImpact:
    """Impact of trauma on consciousness structures."""
    impact_type: str = ""  # "fragmentation", "constriction", "hypervigilance", "dissociation", "derealization"
    severity: float = 0.0
    areas_affected: List[str] = field(default_factory=list)  # "identity", "memory", "perception", "agency", "continuity"
    adaptive_function: str = ""  # What protective purpose this serves
    current_functionality: float = 0.0  # 0.0 (severely impaired) to 1.0 (fully functional)


@dataclass
class ResilienceFactor:
    """A resilience factor supporting recovery."""
    factor_type: str = ""  # "internal", "relational", "community", "spiritual", "biological"
    description: str = ""
    strength: float = 0.0  # 0.0 to 1.0
    accessibility: float = 0.0  # How readily available this resource is
    stability: float = 0.0  # How stable/reliable this factor is


@dataclass
class AttachmentPattern:
    """Attachment pattern assessment."""
    primary_style: str = ""  # "secure", "anxious_preoccupied", "dismissive_avoidant", "fearful_avoidant", "disorganized"
    earned_security: bool = False
    attachment_history: List[str] = field(default_factory=list)
    current_relational_patterns: List[str] = field(default_factory=list)
    repair_capacity: float = 0.0


@dataclass
class AmnesiaBarrier:
    """An amnesia barrier between parts or for specific memories."""
    barrier_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    between_parts: Tuple[str, str] = ("", "")
    barrier_strength: float = 0.0  # 0.0 (permeable) to 1.0 (complete)
    bidirectional: bool = True
    content_type: str = ""  # "autobiographical", "procedural", "emotional", "somatic"
    clinically_significant: bool = False


@dataclass
class ResourceConnection:
    """A recommended resource connection."""
    resource_type: str = ""  # "therapist", "support_group", "hotline", "education", "community"
    name: str = ""
    description: str = ""
    trauma_informed: bool = True
    specialization: List[str] = field(default_factory=list)
    accessibility: str = ""
    contact_info: Optional[str] = None
```
