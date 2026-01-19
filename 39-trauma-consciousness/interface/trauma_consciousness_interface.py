#!/usr/bin/env python3
"""
Trauma & Dissociative Consciousness Interface

Form 39: The comprehensive interface for understanding how traumatic experiences
alter consciousness structure, function, and phenomenology. This form explores
trauma as a transformative force that fragments, reorganizes, and creates
adaptive alterations in conscious experience.

Core Principles:
- Trauma-informed, survivor-centered approach
- Recognition of resilience and healing capacity
- Respect for the adaptive wisdom of protective mechanisms
- Cultural sensitivity in understanding trauma and healing
- Emphasis on safety, choice, and empowerment
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class TraumaType(Enum):
    """
    Classification of trauma types based on nature, context, and pattern.

    Trauma types are not mutually exclusive; individuals often experience
    multiple types. This taxonomy supports comprehensive assessment while
    honoring the unique experience of each survivor.
    """

    # === Temporal/Pattern-Based ===
    ACUTE_SINGLE = "acute_single"  # Single overwhelming event
    COMPLEX_DEVELOPMENTAL = "complex_developmental"  # Repeated, prolonged during development
    CHRONIC_ONGOING = "chronic_ongoing"  # Continuous without resolution

    # === Generational/Collective ===
    INTERGENERATIONAL = "intergenerational"  # Transmitted across generations
    COLLECTIVE_HISTORICAL = "collective_historical"  # Affecting communities/populations

    # === Relational ===
    ATTACHMENT = "attachment"  # Early caregiver disruptions
    BETRAYAL = "betrayal"  # By trusted/depended-upon person

    # === Medical/Physical ===
    MEDICAL = "medical"  # Healthcare experiences
    BIRTH = "birth"  # Birth process related
    PRENATAL = "prenatal"  # In-utero experiences

    # === Violence-Related ===
    COMBAT = "combat"  # Military/war related
    NATURAL_DISASTER = "natural_disaster"  # Environmental catastrophes
    INTERPERSONAL_VIOLENCE = "interpersonal_violence"  # Assault, abuse, domestic violence


class DissociativeState(Enum):
    """
    Types of dissociative experiences and protective states.

    Dissociation is understood as an adaptive response to overwhelming
    experience. These states represent the nervous system's wisdom in
    protecting consciousness from what cannot be integrated in the moment.
    """

    # === Core Dissociative Phenomena ===
    DEPERSONALIZATION = "depersonalization"  # Detachment from self/body
    DEREALIZATION = "derealization"  # World seems unreal/dreamlike
    AMNESIA = "amnesia"  # Inability to recall important information
    IDENTITY_CONFUSION = "identity_confusion"  # Uncertainty about identity
    IDENTITY_ALTERATION = "identity_alteration"  # Shifts between identity states

    # === Intrusive Phenomena ===
    FLASHBACK = "flashback"  # Past intrudes into present consciousness

    # === Numbing/Constriction ===
    EMOTIONAL_NUMBING = "emotional_numbing"  # Reduced emotional capacity

    # === Structural ===
    FRAGMENTATION = "fragmentation"  # Disrupted integration of consciousness

    # === Autonomic States ===
    FREEZE_RESPONSE = "freeze_response"  # Immobilization with hypervigilance
    STRUCTURAL_DISSOCIATION = "structural_dissociation"  # ANP/EP division


class HealingModality(Enum):
    """
    Therapeutic and healing approaches for trauma recovery.

    Multiple pathways to healing are recognized. Different modalities
    serve different aspects of trauma recovery, and individuals may
    benefit from various combinations based on their unique needs.
    """

    # === Evidence-Based Trauma Therapies ===
    EMDR = "emdr"  # Eye Movement Desensitization and Reprocessing
    SOMATIC_EXPERIENCING = "somatic_experiencing"  # Peter Levine's approach
    IFS_PARTS_WORK = "ifs_parts_work"  # Internal Family Systems
    SENSORIMOTOR_PSYCHOTHERAPY = "sensorimotor_psychotherapy"  # Pat Ogden's approach

    # === Neuroscience-Based ===
    NEUROFEEDBACK = "neurofeedback"  # Brain state training

    # === Psychedelic-Assisted ===
    PSYCHEDELIC_ASSISTED = "psychedelic_assisted"  # MDMA, psilocybin, etc.

    # === Traditional/Indigenous ===
    TRADITIONAL_CEREMONY = "traditional_ceremony"  # Cultural healing practices

    # === Narrative/Cognitive ===
    NARRATIVE_THERAPY = "narrative_therapy"  # Story integration
    CPPT = "cppt"  # Cognitive Processing Therapy
    DBT = "dbt"  # Dialectical Behavior Therapy


class NervousSystemState(Enum):
    """
    Autonomic nervous system states per Polyvagal Theory.

    Stephen Porges' framework provides understanding of how the
    nervous system responds to safety, danger, and life threat.
    These states are adaptive, not pathological.
    """

    VENTRAL_VAGAL = "ventral_vagal"  # Safe, social engagement active
    SYMPATHETIC_ACTIVATION = "sympathetic_activation"  # Fight/flight mobilization
    DORSAL_VAGAL = "dorsal_vagal"  # Shutdown, collapse, conservation
    MIXED_STATE = "mixed_state"  # Blended activation patterns
    WINDOW_OF_TOLERANCE = "window_of_tolerance"  # Optimal regulation zone


class TraumaResponse(Enum):
    """
    Survival defense responses activated by perceived threat.

    These responses are intelligent adaptations, not character flaws.
    The nervous system chooses the response most likely to ensure
    survival based on past experience and current threat assessment.
    """

    FIGHT = "fight"  # Active resistance/aggression
    FLIGHT = "flight"  # Escape/avoidance
    FREEZE = "freeze"  # Immobilization with alertness
    FAWN = "fawn"  # Appeasement/people-pleasing
    COLLAPSE = "collapse"  # Dorsal vagal shutdown


class IntegrationStage(Enum):
    """
    Stages of trauma recovery and integration.

    Based on Judith Herman's foundational model, expanded to include
    the possibility of post-traumatic growth. Recovery is not linear;
    individuals may move between stages as needed.
    """

    STABILIZATION = "stabilization"  # Establishing safety and regulation
    PROCESSING = "processing"  # Working through traumatic material
    INTEGRATION = "integration"  # Incorporating into life narrative
    POST_TRAUMATIC_GROWTH = "post_traumatic_growth"  # Finding meaning/transformation


class MaturityLevel(Enum):
    """Depth of knowledge coverage within this form."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TraumaProfile:
    """
    Comprehensive profile describing a type of traumatic experience.

    This represents the general characteristics and patterns of a trauma
    type, not an individual's specific experience. Used for educational
    purposes and to support understanding of diverse trauma presentations.
    """
    trauma_id: str
    trauma_type: TraumaType
    description: str
    nervous_system_impact: List[NervousSystemState] = field(default_factory=list)
    dissociative_patterns: List[DissociativeState] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    common_responses: List[TraumaResponse] = field(default_factory=list)
    healing_approaches: List[HealingModality] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)
    related_trauma_types: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for semantic embedding."""
        parts = [
            f"Trauma Type: {self.trauma_type.value}",
            f"Description: {self.description}",
            f"Impact: {', '.join(s.value for s in self.nervous_system_impact)}",
        ]
        return " | ".join(parts)


@dataclass
class DissociativeExperience:
    """
    Represents a type of dissociative experience with its phenomenology.

    Dissociation is understood as protective adaptation. This dataclass
    captures the subjective experience, triggers, and adaptive function
    of different dissociative states.
    """
    experience_id: str
    state: DissociativeState
    phenomenology: str  # Subjective description of the experience
    triggers: List[str] = field(default_factory=list)
    duration: Optional[str] = None  # Typical duration pattern
    function: str = ""  # Protective/adaptive function
    coping_strategies: List[str] = field(default_factory=list)
    grounding_techniques: List[str] = field(default_factory=list)
    related_states: List[str] = field(default_factory=list)
    neurobiological_basis: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for semantic embedding."""
        parts = [
            f"State: {self.state.value}",
            f"Experience: {self.phenomenology}",
            f"Function: {self.function}",
        ]
        return " | ".join(parts)


@dataclass
class HealingApproach:
    """
    Represents a healing modality with its theoretical basis and application.

    Multiple pathways to healing are honored. This dataclass captures
    the essential elements of each approach to support informed
    understanding and appropriate application.
    """
    approach_id: str
    modality: HealingModality
    theoretical_basis: str
    key_techniques: List[str] = field(default_factory=list)
    indications: List[str] = field(default_factory=list)  # When this approach may help
    contraindications: List[str] = field(default_factory=list)  # When to use caution
    evidence_base: str = ""  # Research support summary
    developer: Optional[str] = None
    training_requirements: Optional[str] = None
    integration_stages: List[IntegrationStage] = field(default_factory=list)
    related_modalities: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for semantic embedding."""
        parts = [
            f"Modality: {self.modality.value}",
            f"Theory: {self.theoretical_basis}",
            f"Techniques: {', '.join(self.key_techniques[:3])}",
        ]
        return " | ".join(parts)


@dataclass
class NervousSystemAssessment:
    """
    Framework for understanding nervous system state and regulation.

    Based on Polyvagal Theory and the Window of Tolerance concept.
    Supports understanding of autonomic states without pathologizing
    protective responses.
    """
    assessment_id: str
    baseline_state: NervousSystemState
    triggers: List[str] = field(default_factory=list)  # What shifts state
    resources: List[str] = field(default_factory=list)  # What supports regulation
    window_of_tolerance_range: Tuple[float, float] = (0.3, 0.7)  # Arousal range
    regulation_capacity: float = 0.5  # 0-1 scale
    co_regulation_needs: List[str] = field(default_factory=list)
    self_regulation_skills: List[str] = field(default_factory=list)
    polyvagal_ladder_position: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_embedding_text(self) -> str:
        """Generate text for semantic embedding."""
        parts = [
            f"State: {self.baseline_state.value}",
            f"Regulation: {self.regulation_capacity}",
            f"Resources: {', '.join(self.resources[:3])}",
        ]
        return " | ".join(parts)


@dataclass
class IntergenerationalPattern:
    """
    Represents patterns of trauma transmission across generations.

    Recognizes that trauma effects can be carried forward through
    biological, psychological, and cultural mechanisms. Also captures
    pathways for healing and breaking cycles.
    """
    pattern_id: str
    trauma_origin: str  # Original traumatic event/context
    transmission_mechanism: List[str] = field(default_factory=list)
    epigenetic_factors: List[str] = field(default_factory=list)
    cultural_context: Optional[str] = None
    healing_approaches: List[str] = field(default_factory=list)
    resilience_factors: List[str] = field(default_factory=list)
    generations_affected: int = 1
    cycle_interruption_strategies: List[str] = field(default_factory=list)
    community_healing_needs: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for semantic embedding."""
        parts = [
            f"Origin: {self.trauma_origin}",
            f"Transmission: {', '.join(self.transmission_mechanism)}",
            f"Context: {self.cultural_context or 'Not specified'}",
        ]
        return " | ".join(parts)


@dataclass
class TraumaConsciousnessMaturityState:
    """Tracks the maturity and completeness of trauma consciousness knowledge."""
    overall_maturity: float = 0.0
    trauma_type_coverage: Dict[str, float] = field(default_factory=dict)
    profile_count: int = 0
    dissociative_experience_count: int = 0
    healing_approach_count: int = 0
    nervous_system_assessment_count: int = 0
    intergenerational_pattern_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class TraumaConsciousnessInterface:
    """
    Main interface for Form 39: Trauma & Dissociative Consciousness.

    Provides methods for storing, retrieving, and querying information
    about trauma types, dissociative states, healing modalities, nervous
    system states, and intergenerational patterns.

    This interface takes a trauma-informed, survivor-centered approach,
    emphasizing resilience, healing capacity, and the adaptive wisdom
    of protective mechanisms.
    """

    FORM_ID = "39-trauma-consciousness"
    FORM_NAME = "Trauma & Dissociative Consciousness"

    def __init__(self):
        """Initialize the Trauma Consciousness Interface."""
        # Knowledge indexes
        self.trauma_profile_index: Dict[str, TraumaProfile] = {}
        self.dissociative_experience_index: Dict[str, DissociativeExperience] = {}
        self.healing_approach_index: Dict[str, HealingApproach] = {}
        self.nervous_system_index: Dict[str, NervousSystemAssessment] = {}
        self.intergenerational_index: Dict[str, IntergenerationalPattern] = {}

        # Cross-reference indexes
        self.trauma_type_index: Dict[TraumaType, List[str]] = {}
        self.dissociative_state_index: Dict[DissociativeState, List[str]] = {}
        self.modality_index: Dict[HealingModality, List[str]] = {}
        self.nervous_state_index: Dict[NervousSystemState, List[str]] = {}

        # Maturity tracking
        self.maturity_state = TraumaConsciousnessMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and prepare indexes."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize trauma type index
        for trauma_type in TraumaType:
            self.trauma_type_index[trauma_type] = []

        # Initialize dissociative state index
        for state in DissociativeState:
            self.dissociative_state_index[state] = []

        # Initialize modality index
        for modality in HealingModality:
            self.modality_index[modality] = []

        # Initialize nervous system state index
        for state in NervousSystemState:
            self.nervous_state_index[state] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # TRAUMA PROFILE METHODS
    # ========================================================================

    async def add_trauma_profile(self, profile: TraumaProfile) -> None:
        """Add a trauma profile to the index."""
        self.trauma_profile_index[profile.trauma_id] = profile

        # Update trauma type index
        if profile.trauma_type in self.trauma_type_index:
            self.trauma_type_index[profile.trauma_type].append(profile.trauma_id)

        # Update maturity
        self.maturity_state.profile_count = len(self.trauma_profile_index)
        await self._update_maturity()

    async def get_trauma_profile(self, profile_id: str) -> Optional[TraumaProfile]:
        """Retrieve a trauma profile by ID."""
        return self.trauma_profile_index.get(profile_id)

    async def query_profiles_by_type(
        self,
        trauma_type: TraumaType,
        limit: int = 10
    ) -> List[TraumaProfile]:
        """Query trauma profiles by type."""
        profile_ids = self.trauma_type_index.get(trauma_type, [])[:limit]
        return [
            self.trauma_profile_index[pid]
            for pid in profile_ids
            if pid in self.trauma_profile_index
        ]

    # ========================================================================
    # DISSOCIATIVE EXPERIENCE METHODS
    # ========================================================================

    async def add_dissociative_experience(
        self,
        experience: DissociativeExperience
    ) -> None:
        """Add a dissociative experience to the index."""
        self.dissociative_experience_index[experience.experience_id] = experience

        # Update state index
        if experience.state in self.dissociative_state_index:
            self.dissociative_state_index[experience.state].append(
                experience.experience_id
            )

        # Update maturity
        self.maturity_state.dissociative_experience_count = len(
            self.dissociative_experience_index
        )
        await self._update_maturity()

    async def get_dissociative_experience(
        self,
        experience_id: str
    ) -> Optional[DissociativeExperience]:
        """Retrieve a dissociative experience by ID."""
        return self.dissociative_experience_index.get(experience_id)

    async def query_experiences_by_state(
        self,
        state: DissociativeState,
        limit: int = 10
    ) -> List[DissociativeExperience]:
        """Query dissociative experiences by state."""
        experience_ids = self.dissociative_state_index.get(state, [])[:limit]
        return [
            self.dissociative_experience_index[eid]
            for eid in experience_ids
            if eid in self.dissociative_experience_index
        ]

    # ========================================================================
    # HEALING APPROACH METHODS
    # ========================================================================

    async def add_healing_approach(self, approach: HealingApproach) -> None:
        """Add a healing approach to the index."""
        self.healing_approach_index[approach.approach_id] = approach

        # Update modality index
        if approach.modality in self.modality_index:
            self.modality_index[approach.modality].append(approach.approach_id)

        # Update maturity
        self.maturity_state.healing_approach_count = len(self.healing_approach_index)
        await self._update_maturity()

    async def get_healing_approach(
        self,
        approach_id: str
    ) -> Optional[HealingApproach]:
        """Retrieve a healing approach by ID."""
        return self.healing_approach_index.get(approach_id)

    async def query_approaches_by_modality(
        self,
        modality: HealingModality,
        limit: int = 10
    ) -> List[HealingApproach]:
        """Query healing approaches by modality."""
        approach_ids = self.modality_index.get(modality, [])[:limit]
        return [
            self.healing_approach_index[aid]
            for aid in approach_ids
            if aid in self.healing_approach_index
        ]

    async def query_approaches_for_stage(
        self,
        stage: IntegrationStage
    ) -> List[HealingApproach]:
        """Query healing approaches appropriate for a recovery stage."""
        return [
            approach for approach in self.healing_approach_index.values()
            if stage in approach.integration_stages
        ]

    # ========================================================================
    # NERVOUS SYSTEM ASSESSMENT METHODS
    # ========================================================================

    async def add_nervous_system_assessment(
        self,
        assessment: NervousSystemAssessment
    ) -> None:
        """Add a nervous system assessment to the index."""
        self.nervous_system_index[assessment.assessment_id] = assessment

        # Update state index
        if assessment.baseline_state in self.nervous_state_index:
            self.nervous_state_index[assessment.baseline_state].append(
                assessment.assessment_id
            )

        # Update maturity
        self.maturity_state.nervous_system_assessment_count = len(
            self.nervous_system_index
        )
        await self._update_maturity()

    async def get_nervous_system_assessment(
        self,
        assessment_id: str
    ) -> Optional[NervousSystemAssessment]:
        """Retrieve a nervous system assessment by ID."""
        return self.nervous_system_index.get(assessment_id)

    # ========================================================================
    # INTERGENERATIONAL PATTERN METHODS
    # ========================================================================

    async def add_intergenerational_pattern(
        self,
        pattern: IntergenerationalPattern
    ) -> None:
        """Add an intergenerational pattern to the index."""
        self.intergenerational_index[pattern.pattern_id] = pattern

        # Update maturity
        self.maturity_state.intergenerational_pattern_count = len(
            self.intergenerational_index
        )
        await self._update_maturity()

    async def get_intergenerational_pattern(
        self,
        pattern_id: str
    ) -> Optional[IntergenerationalPattern]:
        """Retrieve an intergenerational pattern by ID."""
        return self.intergenerational_index.get(pattern_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.profile_count +
            self.maturity_state.dissociative_experience_count +
            self.maturity_state.healing_approach_count +
            self.maturity_state.nervous_system_assessment_count +
            self.maturity_state.intergenerational_pattern_count
        )

        # Maturity calculation
        target_items = 200  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update trauma type coverage
        for trauma_type in TraumaType:
            count = len(self.trauma_type_index.get(trauma_type, []))
            target_per_type = 5
            self.maturity_state.trauma_type_coverage[trauma_type.value] = min(
                1.0, count / target_per_type
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> TraumaConsciousnessMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_trauma_profiles(self) -> List[Dict[str, Any]]:
        """Return seed trauma profiles for initialization."""
        return [
            {
                "trauma_id": "acute_single_profile",
                "trauma_type": TraumaType.ACUTE_SINGLE,
                "description": "A single overwhelming event that exceeds the individual's capacity to cope. Examples include accidents, assaults, natural disasters, or sudden loss. Characterized by clear before/after demarcation and often involves acute threat to life or safety.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL
                ],
                "dissociative_patterns": [
                    DissociativeState.FLASHBACK,
                    DissociativeState.DEPERSONALIZATION
                ],
                "protective_factors": [
                    "Strong social support network",
                    "Prior adaptive coping skills",
                    "Secure attachment history",
                    "Access to early intervention",
                    "Meaning-making capacity"
                ],
                "risk_factors": [
                    "Prior trauma history",
                    "Lack of social support",
                    "Peritraumatic dissociation",
                    "Ongoing life stressors"
                ],
                "common_responses": [TraumaResponse.FIGHT, TraumaResponse.FLIGHT, TraumaResponse.FREEZE],
                "healing_approaches": [HealingModality.EMDR, HealingModality.SOMATIC_EXPERIENCING],
                "key_researchers": ["Bessel van der Kolk", "Peter Levine", "Francine Shapiro"],
            },
            {
                "trauma_id": "complex_developmental_profile",
                "trauma_type": TraumaType.COMPLEX_DEVELOPMENTAL,
                "description": "Repeated, prolonged exposure to traumatic stressors during critical developmental periods. Often involves caregiver abuse, neglect, or chronic household dysfunction. Affects personality development, attachment patterns, and nervous system regulation capacity.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.STRUCTURAL_DISSOCIATION,
                    DissociativeState.EMOTIONAL_NUMBING,
                    DissociativeState.IDENTITY_CONFUSION
                ],
                "protective_factors": [
                    "At least one stable, caring adult",
                    "Innate temperament and resilience",
                    "Access to education and opportunity",
                    "Positive peer relationships",
                    "Development of personal strengths"
                ],
                "risk_factors": [
                    "Multiple types of maltreatment",
                    "Chronicity of exposure",
                    "Earlier age of onset",
                    "Lack of protective caregiving"
                ],
                "common_responses": [TraumaResponse.FAWN, TraumaResponse.FREEZE, TraumaResponse.COLLAPSE],
                "healing_approaches": [
                    HealingModality.IFS_PARTS_WORK,
                    HealingModality.SENSORIMOTOR_PSYCHOTHERAPY,
                    HealingModality.DBT
                ],
                "key_researchers": ["Judith Herman", "Bessel van der Kolk", "Pat Ogden", "Daniel Siegel"],
            },
            {
                "trauma_id": "intergenerational_profile",
                "trauma_type": TraumaType.INTERGENERATIONAL,
                "description": "Trauma effects transmitted across generations through biological (epigenetic), psychological (attachment, modeling), and cultural (narrative, silence) mechanisms. Descendants may carry biological vulnerability and psychological patterns from traumas they never directly experienced.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.EMOTIONAL_NUMBING,
                    DissociativeState.DEPERSONALIZATION
                ],
                "protective_factors": [
                    "Breaking silence about family history",
                    "Earned secure attachment",
                    "Cultural connection and identity",
                    "Community healing efforts",
                    "Conscious parenting practices"
                ],
                "risk_factors": [
                    "Family silence about trauma",
                    "Unresolved parental trauma",
                    "Ongoing systemic oppression",
                    "Disrupted cultural transmission"
                ],
                "common_responses": [TraumaResponse.FREEZE, TraumaResponse.FAWN],
                "healing_approaches": [
                    HealingModality.NARRATIVE_THERAPY,
                    HealingModality.IFS_PARTS_WORK,
                    HealingModality.TRADITIONAL_CEREMONY
                ],
                "key_researchers": ["Rachel Yehuda", "Maria Yellow Horse Brave Heart", "Joy DeGruy"],
            },
            {
                "trauma_id": "collective_historical_profile",
                "trauma_type": TraumaType.COLLECTIVE_HISTORICAL,
                "description": "Trauma experienced by entire communities, cultures, or populations. Includes genocide, slavery, colonization, war, and other mass atrocities. Affects cultural identity, creates collective memory, and may require community-level healing alongside individual work.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL
                ],
                "dissociative_patterns": [
                    DissociativeState.EMOTIONAL_NUMBING,
                    DissociativeState.FRAGMENTATION
                ],
                "protective_factors": [
                    "Cultural revitalization efforts",
                    "Community solidarity",
                    "Reclaiming traditional practices",
                    "Land-based healing",
                    "Intergenerational dialogue"
                ],
                "risk_factors": [
                    "Ongoing oppression and discrimination",
                    "Loss of cultural practices",
                    "Forced assimilation",
                    "Economic marginalization"
                ],
                "common_responses": [TraumaResponse.FLIGHT, TraumaResponse.COLLAPSE],
                "healing_approaches": [
                    HealingModality.TRADITIONAL_CEREMONY,
                    HealingModality.NARRATIVE_THERAPY
                ],
                "key_researchers": ["Maria Yellow Horse Brave Heart", "Joy DeGruy", "Eduardo Duran"],
            },
            {
                "trauma_id": "attachment_profile",
                "trauma_type": TraumaType.ATTACHMENT,
                "description": "Disruptions in early caregiver relationships that impair development of secure attachment. Includes caregiver abuse, neglect, frightening/frightened caregiver behavior, or early separation. Creates lasting patterns in how relationships are approached and experienced.",
                "nervous_system_impact": [
                    NervousSystemState.MIXED_STATE,
                    NervousSystemState.SYMPATHETIC_ACTIVATION
                ],
                "dissociative_patterns": [
                    DissociativeState.STRUCTURAL_DISSOCIATION,
                    DissociativeState.EMOTIONAL_NUMBING
                ],
                "protective_factors": [
                    "Later corrective attachment experiences",
                    "Therapeutic relationship",
                    "Reflective capacity development",
                    "Earned secure attachment possible"
                ],
                "risk_factors": [
                    "Disorganized attachment pattern",
                    "Multiple placement changes",
                    "Lack of consistent caregiver",
                    "Caregiver substance abuse or mental illness"
                ],
                "common_responses": [TraumaResponse.FAWN, TraumaResponse.FREEZE, TraumaResponse.FLIGHT],
                "healing_approaches": [
                    HealingModality.SENSORIMOTOR_PSYCHOTHERAPY,
                    HealingModality.IFS_PARTS_WORK,
                    HealingModality.SOMATIC_EXPERIENCING
                ],
                "key_researchers": ["Allan Schore", "Daniel Siegel", "Mary Main", "Karlen Lyons-Ruth"],
            },
            {
                "trauma_id": "betrayal_profile",
                "trauma_type": TraumaType.BETRAYAL,
                "description": "Trauma perpetrated by someone the survivor depends upon and trusts. Creates unique challenges because exposing the perpetrator may threaten the survivor's survival needs. Often involves amnesia or unawareness as adaptive protection of the necessary relationship.",
                "nervous_system_impact": [
                    NervousSystemState.DORSAL_VAGAL,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.AMNESIA,
                    DissociativeState.DEPERSONALIZATION,
                    DissociativeState.STRUCTURAL_DISSOCIATION
                ],
                "protective_factors": [
                    "Validation of experience",
                    "Safe relationships outside betrayal context",
                    "Independence from perpetrator",
                    "Community support"
                ],
                "risk_factors": [
                    "Continued dependence on perpetrator",
                    "Institutional betrayal compounding individual",
                    "Gaslighting and denial",
                    "Social pressure to maintain silence"
                ],
                "common_responses": [TraumaResponse.FAWN, TraumaResponse.FREEZE, TraumaResponse.COLLAPSE],
                "healing_approaches": [
                    HealingModality.IFS_PARTS_WORK,
                    HealingModality.NARRATIVE_THERAPY,
                    HealingModality.SOMATIC_EXPERIENCING
                ],
                "key_researchers": ["Jennifer Freyd", "Judith Herman"],
            },
            {
                "trauma_id": "medical_profile",
                "trauma_type": TraumaType.MEDICAL,
                "description": "Traumatic stress resulting from medical events, procedures, or healthcare experiences. May include life-threatening diagnoses, invasive procedures, ICU stays, or traumatic childbirth. Often unrecognized as 'real' trauma, leading to invalidation.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL
                ],
                "dissociative_patterns": [
                    DissociativeState.DEPERSONALIZATION,
                    DissociativeState.FLASHBACK
                ],
                "protective_factors": [
                    "Trauma-informed medical care",
                    "Sense of control and choice during procedures",
                    "Support person present",
                    "Preparation and information"
                ],
                "risk_factors": [
                    "Prior trauma history",
                    "Loss of bodily autonomy",
                    "Pain with inadequate management",
                    "Healthcare provider dismissiveness"
                ],
                "common_responses": [TraumaResponse.FREEZE, TraumaResponse.COLLAPSE],
                "healing_approaches": [
                    HealingModality.SOMATIC_EXPERIENCING,
                    HealingModality.EMDR
                ],
                "key_researchers": ["Peter Levine", "Bessel van der Kolk"],
            },
            {
                "trauma_id": "birth_profile",
                "trauma_type": TraumaType.BIRTH,
                "description": "Traumatic stress related to the birth process, affecting birthing parent and/or infant. May include emergency interventions, complications, lack of control, consent violations, or NICU experiences. Often minimized culturally despite profound impact on bonding and wellbeing.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL
                ],
                "dissociative_patterns": [
                    DissociativeState.DEPERSONALIZATION,
                    DissociativeState.EMOTIONAL_NUMBING
                ],
                "protective_factors": [
                    "Continuous support during labor",
                    "Sense of being heard and respected",
                    "Skin-to-skin contact after birth",
                    "Processing and validation of experience"
                ],
                "risk_factors": [
                    "Emergency cesarean",
                    "Perceived threat to self or baby",
                    "History of sexual trauma",
                    "Isolation during birth"
                ],
                "common_responses": [TraumaResponse.FREEZE, TraumaResponse.FLIGHT],
                "healing_approaches": [
                    HealingModality.SOMATIC_EXPERIENCING,
                    HealingModality.EMDR,
                    HealingModality.NARRATIVE_THERAPY
                ],
                "key_researchers": ["Penny Simkin", "Phyllis Klaus"],
            },
            {
                "trauma_id": "prenatal_profile",
                "trauma_type": TraumaType.PRENATAL,
                "description": "Traumatic experiences occurring in utero or affecting the pregnant person, impacting fetal development. Includes maternal stress during pregnancy, violence, disasters, or near-death experiences in utero. Creates preverbal, implicit memories stored somatically.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.FRAGMENTATION
                ],
                "protective_factors": [
                    "Maternal support during pregnancy",
                    "Post-birth secure attachment",
                    "Body-based healing approaches",
                    "Recognition of prenatal experience"
                ],
                "risk_factors": [
                    "High maternal cortisol during pregnancy",
                    "Domestic violence during pregnancy",
                    "Substance exposure",
                    "Prenatal loss of twin"
                ],
                "common_responses": [TraumaResponse.FREEZE, TraumaResponse.COLLAPSE],
                "healing_approaches": [
                    HealingModality.SOMATIC_EXPERIENCING,
                    HealingModality.SENSORIMOTOR_PSYCHOTHERAPY
                ],
                "key_researchers": ["Thomas Verny", "William Emerson", "Rachel Yehuda"],
            },
            {
                "trauma_id": "combat_profile",
                "trauma_type": TraumaType.COMBAT,
                "description": "Trauma related to military service and combat exposure. May include direct combat, witnessing death, moral injury, military sexual trauma, and the challenges of reintegration. Often complicated by stigma around seeking help and loss of military identity.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.FLASHBACK,
                    DissociativeState.EMOTIONAL_NUMBING,
                    DissociativeState.DEPERSONALIZATION
                ],
                "protective_factors": [
                    "Unit cohesion",
                    "Leadership support",
                    "Homecoming welcome",
                    "Peer support from veterans",
                    "Meaning-making about service"
                ],
                "risk_factors": [
                    "Multiple deployments",
                    "Moral injury experiences",
                    "Lack of social support upon return",
                    "Prior trauma history"
                ],
                "common_responses": [TraumaResponse.FIGHT, TraumaResponse.FREEZE],
                "healing_approaches": [
                    HealingModality.EMDR,
                    HealingModality.CPPT,
                    HealingModality.PSYCHEDELIC_ASSISTED
                ],
                "key_researchers": ["Jonathan Shay", "Brett Litz", "Rachel Yehuda"],
            },
            {
                "trauma_id": "natural_disaster_profile",
                "trauma_type": TraumaType.NATURAL_DISASTER,
                "description": "Trauma resulting from environmental catastrophes such as earthquakes, hurricanes, floods, wildfires, or pandemics. Often involves community-wide impact, displacement, loss of home and possessions, and ongoing uncertainty about safety.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL
                ],
                "dissociative_patterns": [
                    DissociativeState.FLASHBACK,
                    DissociativeState.DEREALIZATION
                ],
                "protective_factors": [
                    "Community solidarity",
                    "Practical support and resources",
                    "Prior disaster preparedness",
                    "Social connection maintained"
                ],
                "risk_factors": [
                    "Displacement from home",
                    "Loss of loved ones",
                    "Prior trauma history",
                    "Limited resources for recovery"
                ],
                "common_responses": [TraumaResponse.FLIGHT, TraumaResponse.FREEZE],
                "healing_approaches": [
                    HealingModality.SOMATIC_EXPERIENCING,
                    HealingModality.NARRATIVE_THERAPY
                ],
                "key_researchers": ["Stevan Hobfoll", "Yuval Neria"],
            },
            {
                "trauma_id": "interpersonal_violence_profile",
                "trauma_type": TraumaType.INTERPERSONAL_VIOLENCE,
                "description": "Trauma resulting from violence perpetrated by another person, including assault, abuse, domestic violence, or hate crimes. The interpersonal nature creates unique challenges around trust, safety in relationships, and worldview assumptions.",
                "nervous_system_impact": [
                    NervousSystemState.SYMPATHETIC_ACTIVATION,
                    NervousSystemState.DORSAL_VAGAL,
                    NervousSystemState.MIXED_STATE
                ],
                "dissociative_patterns": [
                    DissociativeState.FLASHBACK,
                    DissociativeState.FREEZE_RESPONSE,
                    DissociativeState.STRUCTURAL_DISSOCIATION
                ],
                "protective_factors": [
                    "Safety from perpetrator",
                    "Validation and belief",
                    "Access to advocacy and resources",
                    "Social support",
                    "Sense of agency restored"
                ],
                "risk_factors": [
                    "Ongoing exposure to perpetrator",
                    "Victim-blaming responses",
                    "Prior trauma history",
                    "Economic dependence on perpetrator"
                ],
                "common_responses": [
                    TraumaResponse.FIGHT,
                    TraumaResponse.FLIGHT,
                    TraumaResponse.FREEZE,
                    TraumaResponse.FAWN
                ],
                "healing_approaches": [
                    HealingModality.EMDR,
                    HealingModality.SOMATIC_EXPERIENCING,
                    HealingModality.IFS_PARTS_WORK
                ],
                "key_researchers": ["Judith Herman", "Bessel van der Kolk", "Patricia Resick"],
            },
        ]

    def _get_seed_dissociative_experiences(self) -> List[Dict[str, Any]]:
        """Return seed dissociative experiences for initialization."""
        return [
            {
                "experience_id": "depersonalization_exp",
                "state": DissociativeState.DEPERSONALIZATION,
                "phenomenology": "Feeling detached from one's own self, body, or mental processes. May include watching oneself from outside, feeling like a robot or automaton, sense that self is unreal, emotional numbing, or actions feeling mechanical.",
                "triggers": [
                    "High stress or anxiety",
                    "Trauma reminders",
                    "Overwhelming emotions",
                    "Sleep deprivation",
                    "Sensory overload"
                ],
                "duration": "Seconds to hours; may become chronic",
                "function": "Creates distance from overwhelming emotional experience. Protects consciousness from affect that cannot be processed in the moment.",
                "coping_strategies": [
                    "Grounding through senses",
                    "Naming current experience",
                    "Orienting to environment",
                    "Self-compassion practices"
                ],
                "grounding_techniques": [
                    "5-4-3-2-1 sensory grounding",
                    "Cold water on face or hands",
                    "Strong sensory input (peppermint, citrus)",
                    "Movement and physical activity",
                    "Saying name and date aloud"
                ],
                "neurobiological_basis": "Involves altered activity in insula and anterior cingulate cortex, affecting interoception and self-referential processing.",
            },
            {
                "experience_id": "derealization_exp",
                "state": DissociativeState.DEREALIZATION,
                "phenomenology": "Feeling that the external world is unreal, dreamlike, or unfamiliar. World may look flat, 2D, or artificial. Colors seem dull or overly bright. Familiar places seem strange. Sounds may seem far away or muffled.",
                "triggers": [
                    "Anxiety and panic",
                    "Trauma reminders",
                    "Overwhelming sensory input",
                    "Fatigue",
                    "Unfamiliar environments"
                ],
                "duration": "Minutes to hours typically",
                "function": "Creates perceptual distance from threatening environment. When the world becomes too overwhelming, perception itself provides protective distance.",
                "coping_strategies": [
                    "Focusing on physical sensations",
                    "Engaging with concrete objects",
                    "Connection with safe person",
                    "Naming what is actually present"
                ],
                "grounding_techniques": [
                    "Touching textured objects",
                    "Describing environment in detail",
                    "Eye contact with safe person",
                    "Physical movement through space",
                    "Temperature changes"
                ],
                "neurobiological_basis": "Associated with altered visual processing and temporal-parietal junction activity affecting spatial perception.",
            },
            {
                "experience_id": "amnesia_exp",
                "state": DissociativeState.AMNESIA,
                "phenomenology": "Inability to recall important autobiographical information, typically of a traumatic or stressful nature. Different from ordinary forgetting. Information is encoded but inaccessible to conscious recall.",
                "triggers": [
                    "Proximity to trauma reminders",
                    "High stress states",
                    "Emotional overwhelm",
                    "Anniversary dates"
                ],
                "duration": "May be ongoing until memory recovery",
                "function": "Protects from overwhelming traumatic content. Particularly adaptive when trauma was perpetrated by a needed caregiver (betrayal trauma).",
                "coping_strategies": [
                    "Patience with memory recovery",
                    "Journaling and documentation",
                    "Working with trauma-informed therapist",
                    "Body-based approaches to access implicit memory"
                ],
                "grounding_techniques": [
                    "Orientation to present time and place",
                    "Reviewing known facts about self",
                    "Connecting with continuous sense of self"
                ],
                "neurobiological_basis": "Involves hippocampal function disruption affecting explicit memory consolidation and retrieval.",
            },
            {
                "experience_id": "identity_confusion_exp",
                "state": DissociativeState.IDENTITY_CONFUSION,
                "phenomenology": "Subjective sense of uncertainty, puzzlement, or conflict about one's identity. May include being uncertain who one really is, experiencing conflicting values or preferences, or difficulty knowing one's own wants and needs.",
                "triggers": [
                    "Identity-related decisions",
                    "Relational contexts",
                    "Major life transitions",
                    "Encountering different aspects of self"
                ],
                "duration": "May be ongoing or episodic",
                "function": "Reflects the impact of trauma on identity development and integration. May protect from recognizing incompatible aspects of experience.",
                "coping_strategies": [
                    "Exploration of values and preferences",
                    "Parts work approaches",
                    "Self-compassion for confusion",
                    "Journaling about different self-states"
                ],
                "grounding_techniques": [
                    "Connecting to core values when known",
                    "Physical grounding in body",
                    "Naming what is currently true"
                ],
                "neurobiological_basis": "Relates to default mode network alterations affecting self-referential processing and autobiographical integration.",
            },
            {
                "experience_id": "identity_alteration_exp",
                "state": DissociativeState.IDENTITY_ALTERATION,
                "phenomenology": "Observable shifts between distinct identity states or personality parts. Different parts may have distinct names, ages, genders, memories, emotional patterns, and even physical sensations or skills.",
                "triggers": [
                    "Trauma reminders",
                    "Specific relational contexts",
                    "Emotional states",
                    "Environmental cues",
                    "Internal communication or conflict"
                ],
                "duration": "Switches may last minutes to extended periods",
                "function": "Allows compartmentalization of traumatic experience. Different parts can hold what other parts cannot yet integrate. Adaptive response to early, severe trauma.",
                "coping_strategies": [
                    "Internal communication development",
                    "Parts mapping and relationship building",
                    "Safety planning across parts",
                    "IFS or similar approaches"
                ],
                "grounding_techniques": [
                    "Orienting all parts to present safety",
                    "Internal communication about current situation",
                    "Shared grounding practices"
                ],
                "neurobiological_basis": "Shows distinct brain activation patterns in different identity states. Related to developmental failure of identity integration under extreme trauma.",
            },
            {
                "experience_id": "flashback_exp",
                "state": DissociativeState.FLASHBACK,
                "phenomenology": "Reliving experiences where past traumatic events intrude into present consciousness. May include visual, somatic, emotional, or auditory components. Characterized by loss of time orientation where trauma feels like it is happening now.",
                "triggers": [
                    "Sensory similarities to trauma (smells, sounds, sights)",
                    "Body positions similar to trauma",
                    "Emotional states similar to trauma",
                    "Anniversary dates",
                    "Relational dynamics similar to trauma"
                ],
                "duration": "Seconds to hours",
                "function": "Represents traumatic memory that was not fully processed. The nervous system is attempting to complete unfinished processing.",
                "coping_strategies": [
                    "Dual awareness practices",
                    "Distinguishing then from now",
                    "Safety anchors",
                    "Planned grounding sequences"
                ],
                "grounding_techniques": [
                    "Stating 'I am here now, the trauma is in the past'",
                    "Looking around and naming current environment",
                    "Feeling feet on floor",
                    "Holding a grounding object",
                    "Contact with safe person"
                ],
                "neurobiological_basis": "Involves amygdala hyperactivation and hippocampal underactivation, preventing temporal contextualization of memory.",
            },
            {
                "experience_id": "emotional_numbing_exp",
                "state": DissociativeState.EMOTIONAL_NUMBING,
                "phenomenology": "Reduced capacity to experience emotions, particularly positive ones. Feeling flat, empty, or blank. Difficulty accessing love, joy, or excitement. Feeling detached from others emotionally.",
                "triggers": [
                    "Emotional overwhelm",
                    "Chronic stress",
                    "Reminders of loss",
                    "Intimacy or connection"
                ],
                "duration": "May be chronic or episodic",
                "function": "Protects from overwhelming affect. When emotions become too intense, the system shuts down emotional processing to survive.",
                "coping_strategies": [
                    "Gradual emotional reconnection",
                    "Titrated exposure to positive emotions",
                    "Body-based approaches to access feeling",
                    "Patience with thawing process"
                ],
                "grounding_techniques": [
                    "Noticing subtle body sensations",
                    "Allowing small amounts of emotion",
                    "Connection with nature or animals",
                    "Creative expression"
                ],
                "neurobiological_basis": "Associated with increased dorsal vagal activity and emotional processing shutdown in limbic system.",
            },
            {
                "experience_id": "fragmentation_exp",
                "state": DissociativeState.FRAGMENTATION,
                "phenomenology": "Disruption in the normally integrated functions of consciousness, memory, identity, and perception. Thoughts feel disconnected. Memory fragments do not form coherent narrative. Sense of self in pieces.",
                "triggers": [
                    "Acute overwhelm",
                    "Retraumatization",
                    "Severe stress",
                    "Processing traumatic material"
                ],
                "duration": "Variable, often tied to stress level",
                "function": "Prevents overwhelming experience from being fully registered. Protective compartmentalization during and after trauma.",
                "coping_strategies": [
                    "Reducing overall stress load",
                    "Integration-focused therapy",
                    "Building coherent narrative gradually",
                    "Patience with integration process"
                ],
                "grounding_techniques": [
                    "Simple, concrete tasks",
                    "Orientation to basic facts",
                    "Connection with body",
                    "Safe, predictable environment"
                ],
                "neurobiological_basis": "Reflects disrupted connectivity between brain networks during and after traumatic encoding.",
            },
            {
                "experience_id": "freeze_response_exp",
                "state": DissociativeState.FREEZE_RESPONSE,
                "phenomenology": "Immobilization with maintained consciousness and hypervigilance. Body frozen, unable to move. Mind may be racing or blank. Hyperalert to danger while physically still.",
                "triggers": [
                    "Perceived threat",
                    "Trauma reminders",
                    "Inescapable situations",
                    "Social evaluation"
                ],
                "duration": "Seconds to minutes typically",
                "function": "Adaptive when movement would increase danger. Creates 'attentive stillness' for assessment of threat. Prepares for fight or flight if opportunity arises.",
                "coping_strategies": [
                    "Allowing small movements to complete",
                    "Gentle pendulation in body",
                    "Reminding system of current safety",
                    "Slow, deep breathing when possible"
                ],
                "grounding_techniques": [
                    "Gentle movement of fingers or toes",
                    "Slow head turning to orient",
                    "Pushing against floor or wall",
                    "Shaking or trembling if it arises naturally"
                ],
                "neurobiological_basis": "Represents simultaneous sympathetic (mobilization) and dorsal vagal (immobilization) activation - 'brake and gas at once'.",
            },
            {
                "experience_id": "structural_dissociation_exp",
                "state": DissociativeState.STRUCTURAL_DISSOCIATION,
                "phenomenology": "Division of personality into Apparently Normal Part (ANP) focused on daily life and Emotional Parts (EPs) holding traumatic material. ANP manages survival needs while EPs remain frozen in trauma time.",
                "triggers": [
                    "Trauma reminders activate EP",
                    "Daily life tasks engage ANP",
                    "Stress may increase EP intrusion",
                    "Safety allows more integration"
                ],
                "duration": "Ongoing structural pattern",
                "function": "Allows continuation of daily life functioning while traumatic material is compartmentalized. ANP can work, care for children, maintain relationships while EPs hold what cannot yet be integrated.",
                "coping_strategies": [
                    "Building communication between parts",
                    "Honoring protective function of structure",
                    "Gradual, safe trauma processing",
                    "IFS or structural dissociation-informed therapy"
                ],
                "grounding_techniques": [
                    "ANP grounding in present responsibilities",
                    "EP orientation to current safety",
                    "Internal communication about what each part needs"
                ],
                "neurobiological_basis": "Van der Hart, Nijenhuis, and Steele's model based on action systems and developmental failure of integration under chronic trauma.",
            },
        ]

    def _get_seed_healing_approaches(self) -> List[Dict[str, Any]]:
        """Return seed healing approaches for initialization."""
        return [
            {
                "approach_id": "emdr_approach",
                "modality": HealingModality.EMDR,
                "theoretical_basis": "Bilateral stimulation facilitates processing of traumatic memories, similar to REM sleep memory consolidation. Helps integrate fragmented traumatic memories into adaptive memory networks.",
                "key_techniques": [
                    "Bilateral eye movements or tapping",
                    "Target memory identification",
                    "Negative and positive cognition work",
                    "Body scan for residual disturbance",
                    "Resource development and installation"
                ],
                "indications": [
                    "Single-incident trauma with clear memory",
                    "PTSD with discrete traumatic events",
                    "Phobias and anxiety with known origin",
                    "Stabilized complex trauma clients"
                ],
                "contraindications": [
                    "Active psychosis",
                    "Severe dissociative disorders without preparation",
                    "Lack of stabilization skills",
                    "Active suicidality",
                    "Medical conditions affecting eye movement"
                ],
                "evidence_base": "Strong research support. WHO and VA recommended. Multiple RCTs demonstrating efficacy for PTSD.",
                "developer": "Francine Shapiro, PhD",
                "training_requirements": "EMDRIA-approved basic training (40+ hours) plus consultation",
                "integration_stages": [IntegrationStage.PROCESSING, IntegrationStage.INTEGRATION],
            },
            {
                "approach_id": "se_approach",
                "modality": HealingModality.SOMATIC_EXPERIENCING,
                "theoretical_basis": "Trauma is primarily a physiological phenomenon. The body has innate capacity to complete interrupted defensive responses and discharge held traumatic activation through titration and pendulation.",
                "key_techniques": [
                    "Tracking body sensations (SIBAM)",
                    "Titration - working with small amounts",
                    "Pendulation between activation and calm",
                    "Supporting completion of defensive responses",
                    "Resourcing before and during processing"
                ],
                "indications": [
                    "Somatic symptoms and chronic pain",
                    "Developmental and complex trauma",
                    "Medical and procedural trauma",
                    "Clients who dissociate with talk therapy",
                    "Preverbal trauma"
                ],
                "contraindications": [
                    "Acute psychosis",
                    "Active substance abuse without support",
                    "Client preference for cognitive approaches",
                    "Lack of body awareness capacity"
                ],
                "evidence_base": "Growing research base. Studies show efficacy for PTSD, physical symptoms, and emotional regulation.",
                "developer": "Peter Levine, PhD",
                "training_requirements": "Three-year training program through SE International",
                "integration_stages": [
                    IntegrationStage.STABILIZATION,
                    IntegrationStage.PROCESSING,
                    IntegrationStage.INTEGRATION
                ],
            },
            {
                "approach_id": "ifs_approach",
                "modality": HealingModality.IFS_PARTS_WORK,
                "theoretical_basis": "All people have sub-personalities or 'parts.' Trauma creates exiled parts carrying pain, protected by manager and firefighter parts. The core Self is always intact and can lead healing when accessed.",
                "key_techniques": [
                    "Accessing Self energy (8 C's)",
                    "Identifying and mapping parts",
                    "Building relationships with protector parts",
                    "Witnessing and unburdening exiles",
                    "Internal communication and negotiation"
                ],
                "indications": [
                    "Complex and developmental trauma",
                    "Dissociative presentations",
                    "Internal conflict and self-criticism",
                    "Addictions and compulsive behaviors",
                    "Parts-based experience of self"
                ],
                "contraindications": [
                    "Acute crisis without stabilization",
                    "Client discomfort with parts language",
                    "Severe cognitive impairment",
                    "Resistance to internal exploration"
                ],
                "evidence_base": "Growing research base including RCTs. Recognized as evidence-based by NREPP. Effective for depression, anxiety, PTSD.",
                "developer": "Richard Schwartz, PhD",
                "training_requirements": "IFS Institute training levels 1-3 for certification",
                "integration_stages": [
                    IntegrationStage.STABILIZATION,
                    IntegrationStage.PROCESSING,
                    IntegrationStage.INTEGRATION,
                    IntegrationStage.POST_TRAUMATIC_GROWTH
                ],
            },
            {
                "approach_id": "sensorimotor_approach",
                "modality": HealingModality.SENSORIMOTOR_PSYCHOTHERAPY,
                "theoretical_basis": "Integration of body-oriented interventions with psychodynamic and cognitive approaches. Works bottom-up: body first, then emotions, then meaning. Trauma disrupts sensorimotor processing that must be addressed directly.",
                "key_techniques": [
                    "Tracking body sensation and movement",
                    "Exploring somatic gestures",
                    "Boundary and containment exercises",
                    "Mindful movement experiments",
                    "Posture and alignment work"
                ],
                "indications": [
                    "Developmental and attachment trauma",
                    "Somatic symptoms and body disconnection",
                    "Dissociative presentations",
                    "Complex PTSD",
                    "Clients who 'talk about' but don't process"
                ],
                "contraindications": [
                    "Acute crisis",
                    "Severe body image disturbance",
                    "Active eating disorder without support",
                    "Client preference for purely verbal approaches"
                ],
                "evidence_base": "Research studies support efficacy for PTSD and complex trauma. Integrates well with other evidence-based approaches.",
                "developer": "Pat Ogden, PhD",
                "training_requirements": "Sensorimotor Psychotherapy Institute training programs",
                "integration_stages": [
                    IntegrationStage.STABILIZATION,
                    IntegrationStage.PROCESSING,
                    IntegrationStage.INTEGRATION
                ],
            },
            {
                "approach_id": "neurofeedback_approach",
                "modality": HealingModality.NEUROFEEDBACK,
                "theoretical_basis": "Real-time training of brain function using EEG biofeedback. Trauma alters brain patterns; direct training can help the brain develop more regulated states and improve flexibility.",
                "key_techniques": [
                    "EEG monitoring and feedback",
                    "Alpha-theta training for deep states",
                    "SMR training for calm focus",
                    "Connectivity training for network integration",
                    "Personalized protocols based on assessment"
                ],
                "indications": [
                    "Developmental trauma with regulation difficulties",
                    "PTSD with hyperarousal",
                    "Dissociative symptoms",
                    "Attention and concentration problems",
                    "Sleep disturbance related to trauma"
                ],
                "contraindications": [
                    "Seizure disorders (some protocols)",
                    "Active psychosis",
                    "Inability to participate in sessions",
                    "Certain medications may interact"
                ],
                "evidence_base": "Growing evidence base. Bessel van der Kolk and others have published research showing benefits for trauma populations.",
                "developer": "Various researchers and clinicians",
                "training_requirements": "BCIA certification or equivalent training",
                "integration_stages": [IntegrationStage.STABILIZATION, IntegrationStage.PROCESSING],
            },
            {
                "approach_id": "psychedelic_approach",
                "modality": HealingModality.PSYCHEDELIC_ASSISTED,
                "theoretical_basis": "Substances like MDMA, psilocybin, and ketamine can facilitate trauma processing by reducing fear response, increasing self-compassion, and opening windows for approaching avoided material with support.",
                "key_techniques": [
                    "Preparation sessions for set and setting",
                    "Medicine session with trained therapists",
                    "Integration sessions following",
                    "Inner healing intelligence model",
                    "Support without direction during session"
                ],
                "indications": [
                    "Treatment-resistant PTSD",
                    "Complex trauma with extensive avoidance",
                    "Moral injury",
                    "End-of-life distress with trauma history"
                ],
                "contraindications": [
                    "Psychotic disorders or high risk",
                    "Certain cardiac conditions",
                    "Some medications (SSRIs with MDMA)",
                    "Lack of adequate support system",
                    "Active suicidality"
                ],
                "evidence_base": "MDMA-assisted therapy received FDA breakthrough therapy designation. Phase 3 trials show 67%+ no longer meeting PTSD criteria.",
                "developer": "MAPS (MDMA research), Johns Hopkins (psilocybin), various researchers",
                "training_requirements": "Specialized training through MAPS or similar organizations",
                "integration_stages": [
                    IntegrationStage.PROCESSING,
                    IntegrationStage.INTEGRATION,
                    IntegrationStage.POST_TRAUMATIC_GROWTH
                ],
            },
            {
                "approach_id": "traditional_ceremony_approach",
                "modality": HealingModality.TRADITIONAL_CEREMONY,
                "theoretical_basis": "Indigenous and traditional healing practices address trauma through community connection, spiritual relationship, cultural identity, and ceremonial containers for processing and transformation.",
                "key_techniques": [
                    "Sweat lodge / purification ceremonies",
                    "Talking circles and community witnessing",
                    "Connection with ancestors and spirits",
                    "Soul retrieval practices",
                    "Land-based healing"
                ],
                "indications": [
                    "Historical and collective trauma",
                    "Cultural disconnection",
                    "Spiritual aspects of trauma",
                    "Community-level healing",
                    "Indigenous individuals seeking cultural approaches"
                ],
                "contraindications": [
                    "Appropriation without proper cultural connection",
                    "Facilitator lacking proper training and lineage",
                    "Physical health conditions (sweat lodge)",
                    "Active psychosis"
                ],
                "evidence_base": "Thousands of years of traditional knowledge. Growing academic research on effectiveness of culturally-based approaches.",
                "developer": "Indigenous peoples and traditional healers worldwide",
                "training_requirements": "Varies by tradition. Requires proper cultural transmission and authorization.",
                "integration_stages": [
                    IntegrationStage.STABILIZATION,
                    IntegrationStage.PROCESSING,
                    IntegrationStage.INTEGRATION,
                    IntegrationStage.POST_TRAUMATIC_GROWTH
                ],
            },
            {
                "approach_id": "narrative_approach",
                "modality": HealingModality.NARRATIVE_THERAPY,
                "theoretical_basis": "Creating coherent life narrative integrates traumatic experiences into meaningful story. Placing trauma in timeline with beginning, middle, and end helps locate it in the past and integrate into identity.",
                "key_techniques": [
                    "Life narrative construction",
                    "Placing trauma in timeline",
                    "Externalizing problems",
                    "Finding unique outcomes and strengths",
                    "Re-authoring life story"
                ],
                "indications": [
                    "Fragmented sense of identity",
                    "Multiple traumas needing organization",
                    "Shame and self-blame",
                    "Meaning-making difficulties",
                    "Cultural or collective trauma"
                ],
                "contraindications": [
                    "Active crisis",
                    "Severe dissociation without stabilization",
                    "Premature focus on narrative before body work",
                    "Overwhelming flooding when approaching story"
                ],
                "evidence_base": "Narrative Exposure Therapy (NET) has strong evidence for refugee and torture survivor populations.",
                "developer": "Michael White and David Epston (Narrative Therapy); Schauer, Neuner, Elbert (NET)",
                "training_requirements": "Training through Dulwich Centre or NET training programs",
                "integration_stages": [IntegrationStage.INTEGRATION, IntegrationStage.POST_TRAUMATIC_GROWTH],
            },
            {
                "approach_id": "cppt_approach",
                "modality": HealingModality.CPPT,
                "theoretical_basis": "Cognitive Processing Therapy addresses 'stuck points' - maladaptive beliefs developed from trauma. By examining and challenging trauma-related cognitions, integration becomes possible.",
                "key_techniques": [
                    "Identifying stuck points",
                    "Written trauma account",
                    "Socratic questioning of beliefs",
                    "Cognitive worksheets",
                    "Themes of safety, trust, power, esteem, intimacy"
                ],
                "indications": [
                    "PTSD with prominent cognitive distortions",
                    "Self-blame and guilt",
                    "Trust and safety belief disruptions",
                    "Clients who prefer structured approaches"
                ],
                "contraindications": [
                    "Severe dissociation",
                    "Active suicidality",
                    "Cognitive impairment",
                    "Acute substance abuse"
                ],
                "evidence_base": "Strong research support. VA/DoD recommended. Multiple RCTs demonstrating efficacy for PTSD.",
                "developer": "Patricia Resick, PhD",
                "training_requirements": "CPT training through certified trainers",
                "integration_stages": [IntegrationStage.PROCESSING, IntegrationStage.INTEGRATION],
            },
            {
                "approach_id": "dbt_approach",
                "modality": HealingModality.DBT,
                "theoretical_basis": "Dialectical Behavior Therapy builds distress tolerance, emotion regulation, interpersonal effectiveness, and mindfulness skills. Originally developed for BPD, highly effective for complex trauma presentations.",
                "key_techniques": [
                    "Mindfulness skills",
                    "Distress tolerance (TIPP, ACCEPTS, etc.)",
                    "Emotion regulation",
                    "Interpersonal effectiveness",
                    "Walking the middle path"
                ],
                "indications": [
                    "Complex trauma with emotional dysregulation",
                    "Self-harm behaviors",
                    "Suicidality",
                    "Borderline presentations",
                    "Need for skill building before processing"
                ],
                "contraindications": [
                    "Active psychosis",
                    "Severe cognitive impairment",
                    "Inability to participate in group",
                    "Client seeking only trauma processing without stabilization"
                ],
                "evidence_base": "Extensive research support. Gold standard for BPD. Strong evidence for suicidality and self-harm reduction.",
                "developer": "Marsha Linehan, PhD",
                "training_requirements": "DBT certification through Linehan Institute or equivalent",
                "integration_stages": [IntegrationStage.STABILIZATION, IntegrationStage.PROCESSING],
            },
        ]

    def _get_seed_nervous_system_states(self) -> List[Dict[str, Any]]:
        """Return seed nervous system state information."""
        return [
            {
                "assessment_id": "ventral_vagal_state",
                "baseline_state": NervousSystemState.VENTRAL_VAGAL,
                "triggers": [
                    "Perception of threat",
                    "Unexpected events",
                    "Trauma reminders",
                    "Social rejection cues"
                ],
                "resources": [
                    "Safe social connection",
                    "Familiar environments",
                    "Regulated breath",
                    "Positive memories",
                    "Co-regulation with safe other"
                ],
                "window_of_tolerance_range": (0.3, 0.7),
                "regulation_capacity": 0.8,
                "co_regulation_needs": [
                    "Attuned presence of safe other",
                    "Warm vocal prosody",
                    "Soft eye contact"
                ],
                "self_regulation_skills": [
                    "Deep breathing",
                    "Orienting to safety cues",
                    "Self-soothing touch",
                    "Positive visualization"
                ],
                "polyvagal_ladder_position": "Top of ladder - Social Engagement System active",
            },
            {
                "assessment_id": "sympathetic_state",
                "baseline_state": NervousSystemState.SYMPATHETIC_ACTIVATION,
                "triggers": [
                    "Perceived danger",
                    "Time pressure",
                    "Conflict",
                    "Trauma reminders",
                    "Caffeine/stimulants"
                ],
                "resources": [
                    "Physical movement/exercise",
                    "Completing fight/flight response safely",
                    "Grounding techniques",
                    "Slow exhale breathing"
                ],
                "window_of_tolerance_range": (0.7, 1.0),
                "regulation_capacity": 0.5,
                "co_regulation_needs": [
                    "Calm presence of regulated other",
                    "Reassurance of safety",
                    "Physical containment if welcome"
                ],
                "self_regulation_skills": [
                    "Extended exhale",
                    "Cold water on face",
                    "Physical grounding",
                    "Progressive muscle relaxation"
                ],
                "polyvagal_ladder_position": "Middle of ladder - Fight/Flight activated",
            },
            {
                "assessment_id": "dorsal_vagal_state",
                "baseline_state": NervousSystemState.DORSAL_VAGAL,
                "triggers": [
                    "Overwhelming threat",
                    "Inescapable danger",
                    "Extreme exhaustion",
                    "Severe betrayal",
                    "Life threat"
                ],
                "resources": [
                    "Gentle, warm social engagement",
                    "Slow, careful activation",
                    "Reassurance of safety",
                    "Time and patience"
                ],
                "window_of_tolerance_range": (0.0, 0.3),
                "regulation_capacity": 0.2,
                "co_regulation_needs": [
                    "Patient, gentle presence",
                    "Non-demanding connection",
                    "Slow approach",
                    "Warmth and safety cues"
                ],
                "self_regulation_skills": [
                    "Gentle movement",
                    "Warm beverages",
                    "Soft sensory input",
                    "Orientation to environment"
                ],
                "polyvagal_ladder_position": "Bottom of ladder - Conservation/Shutdown mode",
            },
            {
                "assessment_id": "mixed_state",
                "baseline_state": NervousSystemState.MIXED_STATE,
                "triggers": [
                    "Trauma during immobilization",
                    "Helplessness with high arousal",
                    "Freeze with fight energy"
                ],
                "resources": [
                    "Gradual, titrated discharge",
                    "Supporting completion of thwarted responses",
                    "Dual awareness practices",
                    "Somatic Experiencing approaches"
                ],
                "window_of_tolerance_range": (0.2, 0.8),
                "regulation_capacity": 0.3,
                "co_regulation_needs": [
                    "Steady, regulated presence",
                    "Permission for whatever arises",
                    "Patience with mixed signals"
                ],
                "self_regulation_skills": [
                    "Pendulation between activation and calm",
                    "Small movements to begin discharge",
                    "Noticing shifts between states"
                ],
                "polyvagal_ladder_position": "Blended state - both brake and gas activated",
            },
            {
                "assessment_id": "window_tolerance_state",
                "baseline_state": NervousSystemState.WINDOW_OF_TOLERANCE,
                "triggers": [
                    "Stress accumulation",
                    "Trauma reminders",
                    "Sleep deprivation",
                    "Interpersonal conflict"
                ],
                "resources": [
                    "Maintaining daily routines",
                    "Regular sleep and nutrition",
                    "Social support",
                    "Mindfulness practice",
                    "Physical exercise"
                ],
                "window_of_tolerance_range": (0.35, 0.65),
                "regulation_capacity": 0.7,
                "co_regulation_needs": [
                    "Attuned relationships",
                    "Community connection",
                    "Safe base relationships"
                ],
                "self_regulation_skills": [
                    "Breath awareness",
                    "Body scan",
                    "Grounding practices",
                    "Self-compassion"
                ],
                "polyvagal_ladder_position": "Within optimal zone - flexible access to all states",
            },
        ]

    def _get_seed_intergenerational_patterns(self) -> List[Dict[str, Any]]:
        """Return seed intergenerational patterns."""
        return [
            {
                "pattern_id": "holocaust_transmission",
                "trauma_origin": "Holocaust genocide and survival",
                "transmission_mechanism": [
                    "Epigenetic methylation of stress genes (FKBP5)",
                    "Altered cortisol patterns in offspring",
                    "Attachment disruptions",
                    "Family atmosphere of threat",
                    "Narrative silences and overwhelming stories"
                ],
                "epigenetic_factors": [
                    "Glucocorticoid receptor methylation",
                    "Altered HPA axis functioning",
                    "Enhanced cortisol suppression",
                    "Changes persisting to third generation"
                ],
                "cultural_context": "Jewish communities worldwide carrying collective memory of genocide, creating both vulnerability and remarkable resilience through cultural practices, meaning-making, and 'never forget' commitments.",
                "healing_approaches": [
                    "Intergenerational dialogue",
                    "Testimony projects and witnessing",
                    "Cultural and religious practices",
                    "Third generation engagement with history",
                    "Therapy addressing inherited trauma patterns"
                ],
                "resilience_factors": [
                    "Strong cultural identity",
                    "Community cohesion",
                    "Meaning-making through religious practice",
                    "Commitment to justice and remembrance",
                    "Educational achievement and advocacy"
                ],
                "generations_affected": 3,
                "cycle_interruption_strategies": [
                    "Speaking about trauma in age-appropriate ways",
                    "Building secure attachment",
                    "Cultural education without overwhelming",
                    "Individual and family therapy"
                ],
                "community_healing_needs": [
                    "Holocaust education and remembrance",
                    "Support for aging survivors",
                    "Intergenerational dialogue programs",
                    "Addressing ongoing antisemitism"
                ],
            },
            {
                "pattern_id": "slavery_transmission",
                "trauma_origin": "American chattel slavery and ongoing racism",
                "transmission_mechanism": [
                    "Epigenetic changes from generations of extreme stress",
                    "Attachment disruptions from family separations",
                    "Internalized oppression",
                    "Survival behaviors becoming patterns",
                    "Ongoing systemic racism compounding effects"
                ],
                "epigenetic_factors": [
                    "Stress response alterations",
                    "Inflammatory markers",
                    "Weathering effects on health",
                    "Allostatic load accumulation"
                ],
                "cultural_context": "African American communities carrying effects of slavery, Jim Crow, and ongoing structural racism. Joy DeGruy's Post Traumatic Slave Syndrome framework addresses unique patterns including vacant esteem, ever-present anger, and racist socialization.",
                "healing_approaches": [
                    "Community-based healing circles",
                    "Cultural reclamation and Afrocentrism",
                    "Addressing internalized racism",
                    "Economic justice work",
                    "Culturally-responsive therapy"
                ],
                "resilience_factors": [
                    "Strong family and community bonds",
                    "Faith traditions and spiritual practice",
                    "Cultural creativity and expression",
                    "Civil rights organizing and advocacy",
                    "Racial pride and identity"
                ],
                "generations_affected": 10,
                "cycle_interruption_strategies": [
                    "Teaching accurate history",
                    "Building racial identity and pride",
                    "Addressing systemic racism",
                    "Economic empowerment",
                    "Healing-centered parenting"
                ],
                "community_healing_needs": [
                    "Reparations and acknowledgment",
                    "Dismantling structural racism",
                    "Community mental health resources",
                    "Economic opportunity",
                    "Police and criminal justice reform"
                ],
            },
            {
                "pattern_id": "indigenous_boarding_school",
                "trauma_origin": "Residential/boarding school system and colonization",
                "transmission_mechanism": [
                    "Forced separation breaking attachment bonds",
                    "Loss of parenting models",
                    "Language and cultural suppression",
                    "Physical, sexual, emotional abuse",
                    "Internalized colonialism"
                ],
                "epigenetic_factors": [
                    "Intergenerational stress response changes",
                    "Health disparities across generations",
                    "Trauma effects on pregnancy outcomes"
                ],
                "cultural_context": "Indigenous communities in US, Canada, Australia, and elsewhere experienced government policies of forced assimilation through residential schools. Recent discoveries of unmarked graves have brought renewed attention to this genocide.",
                "healing_approaches": [
                    "Return to traditional practices",
                    "Language revitalization",
                    "Ceremony and spiritual reconnection",
                    "Land-based healing",
                    "Truth and reconciliation processes",
                    "Community-led healing initiatives"
                ],
                "resilience_factors": [
                    "Cultural survival despite genocide",
                    "Elder wisdom and teaching",
                    "Land connection",
                    "Traditional practices maintained",
                    "Sovereignty movements"
                ],
                "generations_affected": 5,
                "cycle_interruption_strategies": [
                    "Cultural and language programs for youth",
                    "Healing-centered tribal programs",
                    "Reconnecting families and communities",
                    "Land return and sovereignty",
                    "Traditional parenting recovery"
                ],
                "community_healing_needs": [
                    "Truth-telling and acknowledgment",
                    "Return of remains and records",
                    "Reparations and land return",
                    "Self-determination in healing approaches",
                    "Mental health services by and for Indigenous peoples"
                ],
            },
        ]

    async def initialize_seed_trauma_profiles(self) -> int:
        """Initialize with seed trauma profiles."""
        seed_profiles = self._get_seed_trauma_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = TraumaProfile(
                trauma_id=profile_data["trauma_id"],
                trauma_type=profile_data["trauma_type"],
                description=profile_data["description"],
                nervous_system_impact=profile_data.get("nervous_system_impact", []),
                dissociative_patterns=profile_data.get("dissociative_patterns", []),
                protective_factors=profile_data.get("protective_factors", []),
                risk_factors=profile_data.get("risk_factors", []),
                common_responses=profile_data.get("common_responses", []),
                healing_approaches=profile_data.get("healing_approaches", []),
                key_researchers=profile_data.get("key_researchers", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_trauma_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed trauma profiles")
        return count

    async def initialize_seed_dissociative_experiences(self) -> int:
        """Initialize with seed dissociative experiences."""
        seed_experiences = self._get_seed_dissociative_experiences()
        count = 0

        for exp_data in seed_experiences:
            experience = DissociativeExperience(
                experience_id=exp_data["experience_id"],
                state=exp_data["state"],
                phenomenology=exp_data["phenomenology"],
                triggers=exp_data.get("triggers", []),
                duration=exp_data.get("duration"),
                function=exp_data.get("function", ""),
                coping_strategies=exp_data.get("coping_strategies", []),
                grounding_techniques=exp_data.get("grounding_techniques", []),
                neurobiological_basis=exp_data.get("neurobiological_basis"),
            )
            await self.add_dissociative_experience(experience)
            count += 1

        logger.info(f"Initialized {count} seed dissociative experiences")
        return count

    async def initialize_seed_healing_approaches(self) -> int:
        """Initialize with seed healing approaches."""
        seed_approaches = self._get_seed_healing_approaches()
        count = 0

        for approach_data in seed_approaches:
            approach = HealingApproach(
                approach_id=approach_data["approach_id"],
                modality=approach_data["modality"],
                theoretical_basis=approach_data["theoretical_basis"],
                key_techniques=approach_data.get("key_techniques", []),
                indications=approach_data.get("indications", []),
                contraindications=approach_data.get("contraindications", []),
                evidence_base=approach_data.get("evidence_base", ""),
                developer=approach_data.get("developer"),
                training_requirements=approach_data.get("training_requirements"),
                integration_stages=approach_data.get("integration_stages", []),
            )
            await self.add_healing_approach(approach)
            count += 1

        logger.info(f"Initialized {count} seed healing approaches")
        return count

    async def initialize_seed_nervous_system_states(self) -> int:
        """Initialize with seed nervous system state information."""
        seed_states = self._get_seed_nervous_system_states()
        count = 0

        for state_data in seed_states:
            assessment = NervousSystemAssessment(
                assessment_id=state_data["assessment_id"],
                baseline_state=state_data["baseline_state"],
                triggers=state_data.get("triggers", []),
                resources=state_data.get("resources", []),
                window_of_tolerance_range=state_data.get("window_of_tolerance_range", (0.3, 0.7)),
                regulation_capacity=state_data.get("regulation_capacity", 0.5),
                co_regulation_needs=state_data.get("co_regulation_needs", []),
                self_regulation_skills=state_data.get("self_regulation_skills", []),
                polyvagal_ladder_position=state_data.get("polyvagal_ladder_position"),
            )
            await self.add_nervous_system_assessment(assessment)
            count += 1

        logger.info(f"Initialized {count} seed nervous system states")
        return count

    async def initialize_seed_intergenerational_patterns(self) -> int:
        """Initialize with seed intergenerational patterns."""
        seed_patterns = self._get_seed_intergenerational_patterns()
        count = 0

        for pattern_data in seed_patterns:
            pattern = IntergenerationalPattern(
                pattern_id=pattern_data["pattern_id"],
                trauma_origin=pattern_data["trauma_origin"],
                transmission_mechanism=pattern_data.get("transmission_mechanism", []),
                epigenetic_factors=pattern_data.get("epigenetic_factors", []),
                cultural_context=pattern_data.get("cultural_context"),
                healing_approaches=pattern_data.get("healing_approaches", []),
                resilience_factors=pattern_data.get("resilience_factors", []),
                generations_affected=pattern_data.get("generations_affected", 1),
                cycle_interruption_strategies=pattern_data.get("cycle_interruption_strategies", []),
                community_healing_needs=pattern_data.get("community_healing_needs", []),
            )
            await self.add_intergenerational_pattern(pattern)
            count += 1

        logger.info(f"Initialized {count} seed intergenerational patterns")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_trauma_profiles()
        experiences_count = await self.initialize_seed_dissociative_experiences()
        approaches_count = await self.initialize_seed_healing_approaches()
        states_count = await self.initialize_seed_nervous_system_states()
        patterns_count = await self.initialize_seed_intergenerational_patterns()

        total = (
            profiles_count +
            experiences_count +
            approaches_count +
            states_count +
            patterns_count
        )

        return {
            "trauma_profiles": profiles_count,
            "dissociative_experiences": experiences_count,
            "healing_approaches": approaches_count,
            "nervous_system_states": states_count,
            "intergenerational_patterns": patterns_count,
            "total": total
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "TraumaType",
    "DissociativeState",
    "HealingModality",
    "NervousSystemState",
    "TraumaResponse",
    "IntegrationStage",
    "MaturityLevel",
    # Dataclasses
    "TraumaProfile",
    "DissociativeExperience",
    "HealingApproach",
    "NervousSystemAssessment",
    "IntergenerationalPattern",
    "TraumaConsciousnessMaturityState",
    # Interface
    "TraumaConsciousnessInterface",
]
