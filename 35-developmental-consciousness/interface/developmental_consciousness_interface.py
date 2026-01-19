#!/usr/bin/env python3
"""
Developmental Consciousness Interface

Form 35: The comprehensive interface for tracking consciousness development
across the human lifespan, from prenatal stages through end-of-life experiences.
This form integrates developmental psychology, cognitive neuroscience,
neurophenomenology, and end-of-life studies.

Key Features:
- 13 major developmental stages with detailed characteristics
- Capacity emergence tracking (theory of mind, metacognition, etc.)
- Consciousness transitions between stages
- Lifespan trajectory modeling
- End-of-life awareness documentation
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

class DevelopmentalStage(Enum):
    """
    Major developmental stages across the human lifespan.

    These represent the primary phases of consciousness development,
    from prenatal awareness through end-of-life experiences.
    """
    # Prenatal
    PRENATAL_EARLY = "prenatal_early"      # Conception through ~20 weeks
    PRENATAL_LATE = "prenatal_late"        # ~20 weeks through birth

    # Early Life
    NEONATAL = "neonatal"                  # Birth through 28 days
    INFANT_EARLY = "infant_early"          # 1-6 months
    INFANT_LATE = "infant_late"            # 6-24 months
    TODDLER = "toddler"                    # 2-3 years

    # Childhood
    EARLY_CHILDHOOD = "early_childhood"    # 3-6 years
    MIDDLE_CHILDHOOD = "middle_childhood"  # 6-12 years

    # Adolescence and Beyond
    ADOLESCENCE = "adolescence"            # 12-18 years
    YOUNG_ADULT = "young_adult"            # 18-40 years
    MIDDLE_ADULT = "middle_adult"          # 40-65 years
    LATE_ADULT = "late_adult"              # 65+ years

    # End of Life
    END_OF_LIFE = "end_of_life"            # Approaching death


class DevelopmentalCapacity(Enum):
    """
    Core cognitive and consciousness capacities tracked across development.

    Each capacity has a typical emergence age and follows a developmental
    trajectory that may include peak performance and potential decline.
    """
    # Basic Awareness
    SENSORY_AWARENESS = "sensory_awareness"

    # Object Cognition
    OBJECT_PERMANENCE = "object_permanence"

    # Self Capacities
    SELF_RECOGNITION = "self_recognition"

    # Social Cognition
    THEORY_OF_MIND = "theory_of_mind"

    # Memory Systems
    AUTOBIOGRAPHICAL_MEMORY = "autobiographical_memory"

    # Higher Cognition
    METACOGNITION = "metacognition"
    ABSTRACT_REASONING = "abstract_reasoning"

    # Temporal Awareness
    TEMPORAL_CONSCIOUSNESS = "temporal_consciousness"

    # Late Life Specific
    MORTALITY_AWARENESS = "mortality_awareness"
    WISDOM_INTEGRATION = "wisdom_integration"


class ConsciousnessMarker(Enum):
    """
    Types of markers used to assess consciousness at different stages.

    These represent different methodological approaches and evidence types
    for documenting consciousness development.
    """
    NEURAL_CORRELATE = "neural_correlate"
    BEHAVIORAL_INDICATOR = "behavioral_indicator"
    COGNITIVE_MILESTONE = "cognitive_milestone"
    EMOTIONAL_DEVELOPMENT = "emotional_development"
    SOCIAL_COGNITION = "social_cognition"


class ResearchMethodology(Enum):
    """
    Research methods used to study developmental consciousness.

    Different methodologies are appropriate for different age groups
    and research questions.
    """
    LOOKING_TIME = "looking_time"
    HABITUATION = "habituation"
    EEG_INFANT = "eeg_infant"
    FMRI_DEVELOPMENTAL = "fmri_developmental"
    BEHAVIORAL_OBSERVATION = "behavioral_observation"
    LONGITUDINAL_STUDY = "longitudinal_study"


class LifespanDomain(Enum):
    """
    Domains of development tracked across the lifespan.

    Each domain has its own developmental trajectory with
    characteristic patterns of emergence, peak, and potential decline.
    """
    PERCEPTUAL = "perceptual"
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    SELF_AWARENESS = "self_awareness"
    TEMPORAL = "temporal"
    MORAL = "moral"


class MaturityLevel(Enum):
    """Depth of knowledge coverage for the interface."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DevelopmentalStageProfile:
    """
    Complete profile of a developmental stage.

    Captures the key characteristics, emerging capacities, neural developments,
    and milestones associated with each major stage of consciousness development.
    """
    stage_id: str
    stage: DevelopmentalStage
    age_range: str
    consciousness_characteristics: List[str]
    emerging_capacities: List[DevelopmentalCapacity]
    neural_developments: List[str]
    key_milestones: List[str]
    research_methods: List[ResearchMethodology] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)
    related_stages: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Stage: {self.stage.value}",
            f"Age Range: {self.age_range}",
            f"Characteristics: {', '.join(self.consciousness_characteristics[:3])}",
            f"Milestones: {', '.join(self.key_milestones[:3])}"
        ]
        return " | ".join(parts)


@dataclass
class CapacityEmergence:
    """
    Documents the emergence of a specific cognitive capacity.

    Tracks when capacities typically emerge, what prerequisites are needed,
    the neural correlates involved, and how emergence varies across cultures.
    """
    capacity_id: str
    capacity: DevelopmentalCapacity
    typical_emergence_age: str
    prerequisites: List[DevelopmentalCapacity]
    neural_correlates: List[str]
    cultural_variations: List[str] = field(default_factory=list)
    assessment_methods: List[ResearchMethodology] = field(default_factory=list)
    key_studies: List[str] = field(default_factory=list)
    related_capacities: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ConsciousnessTransition:
    """
    Represents a major transition between developmental stages.

    Documents the key changes that occur during transitions, the typical
    duration of the transition period, and individual variation patterns.
    """
    transition_id: str
    from_stage: DevelopmentalStage
    to_stage: DevelopmentalStage
    key_changes: List[str]
    duration: str
    individual_variation: str
    triggering_factors: List[str] = field(default_factory=list)
    supporting_conditions: List[str] = field(default_factory=list)
    related_transitions: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LifespanTrajectory:
    """
    Models the trajectory of a domain across the entire lifespan.

    Captures the developmental curve, peak ages, decline patterns,
    and factors that can protect against decline.
    """
    trajectory_id: str
    domain: LifespanDomain
    developmental_curve: str  # Description of the trajectory shape
    peak_age: str
    decline_pattern: str
    protective_factors: List[str]
    individual_differences: List[str] = field(default_factory=list)
    measurement_approaches: List[str] = field(default_factory=list)
    related_trajectories: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EndOfLifeAwareness:
    """
    Documents consciousness changes and experiences at end of life.

    Captures the characteristic consciousness changes, common experiences
    reported, and how different cultures interpret these phenomena.
    """
    awareness_id: str
    stage: DevelopmentalStage  # Should be END_OF_LIFE
    consciousness_changes: List[str]
    common_experiences: List[str]
    cultural_interpretations: List[str]
    research_findings: List[str] = field(default_factory=list)
    phenomenological_reports: List[str] = field(default_factory=list)
    care_implications: List[str] = field(default_factory=list)
    related_phenomena: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DevelopmentalConsciousnessMaturityState:
    """Tracks the maturity of developmental consciousness knowledge."""
    overall_maturity: float = 0.0
    stage_coverage: Dict[str, float] = field(default_factory=dict)
    stage_profile_count: int = 0
    capacity_emergence_count: int = 0
    transition_count: int = 0
    trajectory_count: int = 0
    end_of_life_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class DevelopmentalConsciousnessInterface:
    """
    Main interface for Form 35: Developmental Consciousness.

    Provides methods for storing, retrieving, and querying information about
    consciousness development across the human lifespan, from prenatal stages
    through end-of-life experiences.
    """

    FORM_ID = "35-developmental-consciousness"
    FORM_NAME = "Developmental Consciousness"

    def __init__(self):
        """Initialize the Developmental Consciousness Interface."""
        # Knowledge indexes
        self.stage_profile_index: Dict[str, DevelopmentalStageProfile] = {}
        self.capacity_emergence_index: Dict[str, CapacityEmergence] = {}
        self.transition_index: Dict[str, ConsciousnessTransition] = {}
        self.trajectory_index: Dict[str, LifespanTrajectory] = {}
        self.end_of_life_index: Dict[str, EndOfLifeAwareness] = {}

        # Cross-reference indexes
        self.stage_index: Dict[DevelopmentalStage, List[str]] = {}
        self.capacity_index: Dict[DevelopmentalCapacity, List[str]] = {}
        self.domain_index: Dict[LifespanDomain, List[str]] = {}

        # Maturity tracking
        self.maturity_state = DevelopmentalConsciousnessMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize stage index
        for stage in DevelopmentalStage:
            self.stage_index[stage] = []

        # Initialize capacity index
        for capacity in DevelopmentalCapacity:
            self.capacity_index[capacity] = []

        # Initialize domain index
        for domain in LifespanDomain:
            self.domain_index[domain] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # STAGE PROFILE METHODS
    # ========================================================================

    async def add_stage_profile(self, profile: DevelopmentalStageProfile) -> None:
        """Add a developmental stage profile to the index."""
        self.stage_profile_index[profile.stage_id] = profile

        # Update stage index
        if profile.stage in self.stage_index:
            self.stage_index[profile.stage].append(profile.stage_id)

        # Update capacity index for emerging capacities
        for capacity in profile.emerging_capacities:
            if capacity in self.capacity_index:
                self.capacity_index[capacity].append(profile.stage_id)

        # Update maturity
        self.maturity_state.stage_profile_count = len(self.stage_profile_index)
        await self._update_maturity()

    async def get_stage_profile(self, stage_id: str) -> Optional[DevelopmentalStageProfile]:
        """Retrieve a stage profile by ID."""
        return self.stage_profile_index.get(stage_id)

    async def get_profile_by_stage(
        self,
        stage: DevelopmentalStage
    ) -> Optional[DevelopmentalStageProfile]:
        """Get the profile for a specific developmental stage."""
        stage_ids = self.stage_index.get(stage, [])
        if stage_ids:
            return self.stage_profile_index.get(stage_ids[0])
        return None

    async def query_stages_by_capacity(
        self,
        capacity: DevelopmentalCapacity,
        limit: int = 10
    ) -> List[DevelopmentalStageProfile]:
        """Query stages where a capacity emerges."""
        stage_ids = self.capacity_index.get(capacity, [])[:limit]
        return [
            self.stage_profile_index[sid]
            for sid in stage_ids
            if sid in self.stage_profile_index
        ]

    # ========================================================================
    # CAPACITY EMERGENCE METHODS
    # ========================================================================

    async def add_capacity_emergence(self, emergence: CapacityEmergence) -> None:
        """Add a capacity emergence record to the index."""
        self.capacity_emergence_index[emergence.capacity_id] = emergence

        # Update capacity index
        if emergence.capacity in self.capacity_index:
            self.capacity_index[emergence.capacity].append(emergence.capacity_id)

        # Update maturity
        self.maturity_state.capacity_emergence_count = len(self.capacity_emergence_index)
        await self._update_maturity()

    async def get_capacity_emergence(self, capacity_id: str) -> Optional[CapacityEmergence]:
        """Retrieve a capacity emergence by ID."""
        return self.capacity_emergence_index.get(capacity_id)

    async def get_emergence_by_capacity(
        self,
        capacity: DevelopmentalCapacity
    ) -> Optional[CapacityEmergence]:
        """Get the emergence record for a specific capacity."""
        for emergence in self.capacity_emergence_index.values():
            if emergence.capacity == capacity:
                return emergence
        return None

    # ========================================================================
    # TRANSITION METHODS
    # ========================================================================

    async def add_transition(self, transition: ConsciousnessTransition) -> None:
        """Add a consciousness transition to the index."""
        self.transition_index[transition.transition_id] = transition

        # Update stage indexes
        if transition.from_stage in self.stage_index:
            self.stage_index[transition.from_stage].append(transition.transition_id)
        if transition.to_stage in self.stage_index:
            self.stage_index[transition.to_stage].append(transition.transition_id)

        # Update maturity
        self.maturity_state.transition_count = len(self.transition_index)
        await self._update_maturity()

    async def get_transition(self, transition_id: str) -> Optional[ConsciousnessTransition]:
        """Retrieve a transition by ID."""
        return self.transition_index.get(transition_id)

    async def get_transitions_from_stage(
        self,
        stage: DevelopmentalStage
    ) -> List[ConsciousnessTransition]:
        """Get all transitions from a given stage."""
        return [
            t for t in self.transition_index.values()
            if t.from_stage == stage
        ]

    # ========================================================================
    # TRAJECTORY METHODS
    # ========================================================================

    async def add_trajectory(self, trajectory: LifespanTrajectory) -> None:
        """Add a lifespan trajectory to the index."""
        self.trajectory_index[trajectory.trajectory_id] = trajectory

        # Update domain index
        if trajectory.domain in self.domain_index:
            self.domain_index[trajectory.domain].append(trajectory.trajectory_id)

        # Update maturity
        self.maturity_state.trajectory_count = len(self.trajectory_index)
        await self._update_maturity()

    async def get_trajectory(self, trajectory_id: str) -> Optional[LifespanTrajectory]:
        """Retrieve a trajectory by ID."""
        return self.trajectory_index.get(trajectory_id)

    async def get_trajectory_by_domain(
        self,
        domain: LifespanDomain
    ) -> Optional[LifespanTrajectory]:
        """Get the trajectory for a specific domain."""
        trajectory_ids = self.domain_index.get(domain, [])
        if trajectory_ids:
            return self.trajectory_index.get(trajectory_ids[0])
        return None

    # ========================================================================
    # END OF LIFE METHODS
    # ========================================================================

    async def add_end_of_life_awareness(self, awareness: EndOfLifeAwareness) -> None:
        """Add an end-of-life awareness record to the index."""
        self.end_of_life_index[awareness.awareness_id] = awareness

        # Update stage index
        if awareness.stage in self.stage_index:
            self.stage_index[awareness.stage].append(awareness.awareness_id)

        # Update maturity
        self.maturity_state.end_of_life_count = len(self.end_of_life_index)
        await self._update_maturity()

    async def get_end_of_life_awareness(
        self,
        awareness_id: str
    ) -> Optional[EndOfLifeAwareness]:
        """Retrieve an end-of-life awareness record by ID."""
        return self.end_of_life_index.get(awareness_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.stage_profile_count +
            self.maturity_state.capacity_emergence_count +
            self.maturity_state.transition_count +
            self.maturity_state.trajectory_count +
            self.maturity_state.end_of_life_count
        )

        # Simple maturity calculation
        target_items = 100  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update stage coverage
        for stage in DevelopmentalStage:
            count = len(self.stage_index.get(stage, []))
            target_per_stage = 5
            self.maturity_state.stage_coverage[stage.value] = min(
                1.0, count / target_per_stage
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> DevelopmentalConsciousnessMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_stage_profiles(self) -> List[Dict[str, Any]]:
        """Return seed stage profiles for initialization."""
        return [
            # Prenatal Early
            {
                "stage_id": "prenatal_early",
                "stage": DevelopmentalStage.PRENATAL_EARLY,
                "age_range": "Conception to ~20 weeks gestation",
                "consciousness_characteristics": [
                    "No consciousness (early neural tube formation)",
                    "Reflexive responses begin (not conscious)",
                    "Basic neural architecture establishing",
                    "Sleep-like states predominate"
                ],
                "emerging_capacities": [DevelopmentalCapacity.SENSORY_AWARENESS],
                "neural_developments": [
                    "Neural tube formation (weeks 3-4)",
                    "Brain vesicle differentiation (weeks 4-6)",
                    "Neurogenesis and migration (weeks 6-10)",
                    "Subplate zone developing"
                ],
                "key_milestones": [
                    "First neurons appear (~week 5)",
                    "Spontaneous neural activity (~week 8)",
                    "Basic reflexive movements",
                    "Thalamic connections beginning"
                ],
                "research_methods": [ResearchMethodology.BEHAVIORAL_OBSERVATION],
                "key_researchers": ["Hugo Lagercrantz", "Jean-Pierre Changeux"]
            },
            # Prenatal Late
            {
                "stage_id": "prenatal_late",
                "stage": DevelopmentalStage.PRENATAL_LATE,
                "age_range": "~20 weeks to birth",
                "consciousness_characteristics": [
                    "Minimal consciousness emerges (~24 weeks)",
                    "Thalamocortical connectivity sufficient for awareness",
                    "Sedated/muted consciousness state",
                    "Sensory awareness likely present but muted"
                ],
                "emerging_capacities": [DevelopmentalCapacity.SENSORY_AWARENESS],
                "neural_developments": [
                    "Thalamocortical connections mature (~24 weeks)",
                    "EEG patterns emerge",
                    "Pain pathways functional (~24-26 weeks)",
                    "REM sleep emerges",
                    "Cortical folding and myelination"
                ],
                "key_milestones": [
                    "Viability threshold (~22-24 weeks)",
                    "Responds to sounds",
                    "Habituation demonstrated (basic learning)",
                    "Sleep-wake cycles establish",
                    "Recognizes mother's voice"
                ],
                "research_methods": [
                    ResearchMethodology.EEG_INFANT,
                    ResearchMethodology.BEHAVIORAL_OBSERVATION
                ],
                "key_researchers": [
                    "Hugo Lagercrantz",
                    "Anthony DeCasper",
                    "Peter Hepper"
                ]
            },
            # Neonatal
            {
                "stage_id": "neonatal",
                "stage": DevelopmentalStage.NEONATAL,
                "age_range": "Birth to 28 days",
                "consciousness_characteristics": [
                    "The 'great awakening' - consciousness flooding in",
                    "Primary sensory awareness",
                    "Overwhelming sensory input transition",
                    "Already recognizes mother's voice and smell"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.SENSORY_AWARENESS
                ],
                "neural_developments": [
                    "Noradrenergic surge at birth",
                    "Rapid synaptic development",
                    "Visual cortex activation begins",
                    "Sleep state regulation developing"
                ],
                "key_milestones": [
                    "First breath and environmental transition",
                    "Face preference established",
                    "Imitation of facial expressions (Meltzoff)",
                    "Recognition memory for mother's face",
                    "Beginning social smile (~end of period)"
                ],
                "research_methods": [
                    ResearchMethodology.LOOKING_TIME,
                    ResearchMethodology.HABITUATION,
                    ResearchMethodology.EEG_INFANT
                ],
                "key_researchers": [
                    "Andrew Meltzoff",
                    "Patricia Kuhl"
                ]
            },
            # Infant Early
            {
                "stage_id": "infant_early",
                "stage": DevelopmentalStage.INFANT_EARLY,
                "age_range": "1-6 months",
                "consciousness_characteristics": [
                    "Sensory-affective awareness",
                    "Object-directed awareness emerging",
                    "Social responsiveness developing",
                    "Ecological self (body as agent in world)"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.SENSORY_AWARENESS,
                    DevelopmentalCapacity.OBJECT_PERMANENCE
                ],
                "neural_developments": [
                    "Visual cortex maturation",
                    "Color vision developing",
                    "Depth perception emerging",
                    "Social brain regions activating"
                ],
                "key_milestones": [
                    "Social smiling (6-8 weeks)",
                    "Color vision and depth perception",
                    "Differentiated emotions (joy, anger, sadness)",
                    "Object solidity expectations",
                    "Contact causality understanding (~6 months)"
                ],
                "research_methods": [
                    ResearchMethodology.LOOKING_TIME,
                    ResearchMethodology.HABITUATION,
                    ResearchMethodology.EEG_INFANT
                ],
                "key_researchers": [
                    "Elizabeth Spelke",
                    "Renee Baillargeon"
                ]
            },
            # Infant Late
            {
                "stage_id": "infant_late",
                "stage": DevelopmentalStage.INFANT_LATE,
                "age_range": "6-24 months",
                "consciousness_characteristics": [
                    "Intentional awareness",
                    "Joint intentionality emergence (9-12 months)",
                    "Symbolic awareness emerging (12-18 months)",
                    "Self-recognition threshold (18-24 months)"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.OBJECT_PERMANENCE,
                    DevelopmentalCapacity.SELF_RECOGNITION,
                    DevelopmentalCapacity.THEORY_OF_MIND
                ],
                "neural_developments": [
                    "Prefrontal cortex developing",
                    "Language areas activating",
                    "Mirror neuron system maturation",
                    "Memory systems consolidating"
                ],
                "key_milestones": [
                    "Joint attention and gaze following (9-12 months)",
                    "9-month revolution in social cognition",
                    "First words and rich gesture use",
                    "Mirror self-recognition (18-24 months)",
                    "Diverse desires understood (~18 months)"
                ],
                "research_methods": [
                    ResearchMethodology.LOOKING_TIME,
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.EEG_INFANT
                ],
                "key_researchers": [
                    "Michael Tomasello",
                    "Philippe Rochat",
                    "Alison Gopnik"
                ]
            },
            # Toddler
            {
                "stage_id": "toddler",
                "stage": DevelopmentalStage.TODDLER,
                "age_range": "2-3 years",
                "consciousness_characteristics": [
                    "Egocentric symbolic awareness",
                    "Emerging narrative consciousness",
                    "Explicit self-concept ('I', 'me', 'mine')",
                    "Temporal self-extension begins"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.SELF_RECOGNITION,
                    DevelopmentalCapacity.AUTOBIOGRAPHICAL_MEMORY,
                    DevelopmentalCapacity.THEORY_OF_MIND
                ],
                "neural_developments": [
                    "Language network expansion",
                    "Hippocampal maturation for episodic memory",
                    "Prefrontal development continuing",
                    "Theory of mind network emerging"
                ],
                "key_milestones": [
                    "Sentence use and vocabulary expansion",
                    "Symbolic/pretend play flourishes",
                    "Others have beliefs (both true) understood",
                    "Autobiographical memories forming",
                    "Childhood amnesia barrier emerging"
                ],
                "research_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_researchers": [
                    "Jean Piaget",
                    "Katherine Nelson",
                    "Robyn Fivush"
                ]
            },
            # Early Childhood
            {
                "stage_id": "early_childhood",
                "stage": DevelopmentalStage.EARLY_CHILDHOOD,
                "age_range": "3-6 years",
                "consciousness_characteristics": [
                    "Pre-operational awareness",
                    "Full naive theory of mind",
                    "Narrative self emerging",
                    "Metacognition beginning"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.THEORY_OF_MIND,
                    DevelopmentalCapacity.METACOGNITION,
                    DevelopmentalCapacity.AUTOBIOGRAPHICAL_MEMORY
                ],
                "neural_developments": [
                    "Theory of mind network maturation",
                    "Prefrontal-temporal connectivity",
                    "Executive function circuits developing",
                    "Memory consolidation pathways"
                ],
                "key_milestones": [
                    "False belief threshold (~4 years)",
                    "Childhood amnesia offset (~3.5 years)",
                    "Knowledge vs. ignorance distinction (3-4 years)",
                    "Hidden emotion understood (~5-6 years)",
                    "Dreams understood as not real"
                ],
                "research_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_researchers": [
                    "Henry Wellman",
                    "Josef Perner",
                    "Simon Baron-Cohen"
                ]
            },
            # Middle Childhood
            {
                "stage_id": "middle_childhood",
                "stage": DevelopmentalStage.MIDDLE_CHILDHOOD,
                "age_range": "6-12 years",
                "consciousness_characteristics": [
                    "Concrete operational awareness",
                    "Expanding concrete operations",
                    "Transitional to abstract thought",
                    "Recursive theory of mind"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.METACOGNITION,
                    DevelopmentalCapacity.ABSTRACT_REASONING,
                    DevelopmentalCapacity.TEMPORAL_CONSCIOUSNESS
                ],
                "neural_developments": [
                    "Continued prefrontal maturation",
                    "White matter development",
                    "Improved connectivity between regions",
                    "Executive function refinement"
                ],
                "key_milestones": [
                    "Conservation mastered",
                    "Second-order false belief (8-10 years)",
                    "Explicit memory strategies",
                    "Longer time horizons",
                    "Early abstract reasoning in familiar domains"
                ],
                "research_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY,
                    ResearchMethodology.FMRI_DEVELOPMENTAL
                ],
                "key_researchers": [
                    "Jean Piaget",
                    "John Flavell"
                ]
            },
            # Adolescence
            {
                "stage_id": "adolescence",
                "stage": DevelopmentalStage.ADOLESCENCE,
                "age_range": "12-18 years",
                "consciousness_characteristics": [
                    "Formal operational emergence",
                    "Abstract self-consciousness",
                    "Recursive self-reflection",
                    "Imaginary audience and personal fable"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.ABSTRACT_REASONING,
                    DevelopmentalCapacity.METACOGNITION,
                    DevelopmentalCapacity.TEMPORAL_CONSCIOUSNESS
                ],
                "neural_developments": [
                    "Prefrontal pruning and myelination",
                    "Reward system sensitivity peaks",
                    "Executive control circuits maturing",
                    "Social brain network refinement"
                ],
                "key_milestones": [
                    "Hypothetical-deductive reasoning",
                    "Full formal operations (~16-18 years)",
                    "Identity exploration and commitment",
                    "Extended future time perspective",
                    "Post-conventional moral reasoning possible"
                ],
                "research_methods": [
                    ResearchMethodology.FMRI_DEVELOPMENTAL,
                    ResearchMethodology.LONGITUDINAL_STUDY,
                    ResearchMethodology.BEHAVIORAL_OBSERVATION
                ],
                "key_researchers": [
                    "Laurence Steinberg",
                    "Sarah-Jayne Blakemore",
                    "Erik Erikson"
                ]
            },
            # Young Adult
            {
                "stage_id": "young_adult",
                "stage": DevelopmentalStage.YOUNG_ADULT,
                "age_range": "18-40 years",
                "consciousness_characteristics": [
                    "Peak integration",
                    "Post-formal exploration",
                    "Full adult metacognitive capacity",
                    "Generative self emerging"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.WISDOM_INTEGRATION,
                    DevelopmentalCapacity.TEMPORAL_CONSCIOUSNESS
                ],
                "neural_developments": [
                    "Prefrontal reaches full maturity (~25)",
                    "Peak cognitive performance many domains",
                    "Expertise-related plasticity",
                    "Stable neural architecture"
                ],
                "key_milestones": [
                    "Prefrontal full maturity (~25 years)",
                    "Dialectical thinking emerging",
                    "Individuation from family of origin",
                    "Long-term planning capability",
                    "Intimacy vs. isolation (Erikson)"
                ],
                "research_methods": [
                    ResearchMethodology.LONGITUDINAL_STUDY,
                    ResearchMethodology.FMRI_DEVELOPMENTAL
                ],
                "key_researchers": [
                    "Erik Erikson",
                    "Robert Kegan"
                ]
            },
            # Middle Adult
            {
                "stage_id": "middle_adult",
                "stage": DevelopmentalStage.MIDDLE_ADULT,
                "age_range": "40-65 years",
                "consciousness_characteristics": [
                    "Generative awareness",
                    "Crystallized intelligence rising",
                    "Awareness of mortality increasing",
                    "Integration of experience, values, emotion"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.WISDOM_INTEGRATION,
                    DevelopmentalCapacity.MORTALITY_AWARENESS
                ],
                "neural_developments": [
                    "Fluid intelligence slow decline",
                    "Crystallized intelligence stable/increasing",
                    "Compensatory mechanisms developing",
                    "Expertise networks strengthened"
                ],
                "key_milestones": [
                    "Generativity vs. stagnation (Erikson)",
                    "Legacy concerns emerge",
                    "Knowing what one knows well",
                    "Life experience accumulation",
                    "Wisdom development potential"
                ],
                "research_methods": [
                    ResearchMethodology.LONGITUDINAL_STUDY,
                    ResearchMethodology.FMRI_DEVELOPMENTAL
                ],
                "key_researchers": [
                    "Erik Erikson",
                    "Paul Baltes"
                ]
            },
            # Late Adult
            {
                "stage_id": "late_adult",
                "stage": DevelopmentalStage.LATE_ADULT,
                "age_range": "65+ years",
                "consciousness_characteristics": [
                    "Integrative reflection",
                    "Selective optimization with compensation",
                    "Essential awareness in oldest-old",
                    "Present-focused, memory-rich"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.WISDOM_INTEGRATION,
                    DevelopmentalCapacity.MORTALITY_AWARENESS
                ],
                "neural_developments": [
                    "Gray matter decline (variable)",
                    "Compensatory mechanisms active",
                    "Preserved domains possible",
                    "Emotional regulation often improved"
                ],
                "key_milestones": [
                    "Life review processes",
                    "Ego integrity vs. despair (Erikson)",
                    "Peak wisdom possible",
                    "Coming to terms with life lived",
                    "Positive affect often increases"
                ],
                "research_methods": [
                    ResearchMethodology.LONGITUDINAL_STUDY,
                    ResearchMethodology.BEHAVIORAL_OBSERVATION
                ],
                "key_researchers": [
                    "Erik Erikson",
                    "Paul Baltes",
                    "Laura Carstensen"
                ]
            },
            # End of Life
            {
                "stage_id": "end_of_life",
                "stage": DevelopmentalStage.END_OF_LIFE,
                "age_range": "Approaching death (any age)",
                "consciousness_characteristics": [
                    "Transitional states",
                    "Liminal awareness",
                    "Nearing death awareness",
                    "Possible transcendent experiences"
                ],
                "emerging_capacities": [
                    DevelopmentalCapacity.MORTALITY_AWARENESS
                ],
                "neural_developments": [
                    "Multi-organ changes",
                    "Possible surge of neuromodulators",
                    "Terminal lucidity phenomenon",
                    "Hearing may persist longest"
                ],
                "key_milestones": [
                    "Deathbed visions",
                    "Seeing deceased relatives",
                    "Symbolic language about journeys",
                    "Near-death experiences (if cardiac events)",
                    "Terminal lucidity possible"
                ],
                "research_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_researchers": [
                    "Bruce Greyson",
                    "Pim van Lommel",
                    "Peter Fenwick"
                ]
            }
        ]

    def _get_seed_capacity_emergences(self) -> List[Dict[str, Any]]:
        """Return seed capacity emergence data."""
        return [
            {
                "capacity_id": "theory_of_mind_emergence",
                "capacity": DevelopmentalCapacity.THEORY_OF_MIND,
                "typical_emergence_age": "~4 years (false belief)",
                "prerequisites": [
                    DevelopmentalCapacity.SELF_RECOGNITION,
                    DevelopmentalCapacity.OBJECT_PERMANENCE
                ],
                "neural_correlates": [
                    "Temporoparietal junction (TPJ)",
                    "Medial prefrontal cortex (mPFC)",
                    "Posterior superior temporal sulcus (pSTS)"
                ],
                "cultural_variations": [
                    "Western children ~4 years",
                    "Some cultures show earlier/later emergence",
                    "Language and social interaction affect timing"
                ],
                "assessment_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LOOKING_TIME
                ],
                "key_studies": [
                    "Baron-Cohen et al. (1985) - Sally-Anne task",
                    "Wellman & Liu (2004) - ToM scale",
                    "Onishi & Baillargeon (2005) - Implicit false belief"
                ]
            },
            {
                "capacity_id": "metacognition_emergence",
                "capacity": DevelopmentalCapacity.METACOGNITION,
                "typical_emergence_age": "~6 years (explicit)",
                "prerequisites": [
                    DevelopmentalCapacity.THEORY_OF_MIND,
                    DevelopmentalCapacity.AUTOBIOGRAPHICAL_MEMORY
                ],
                "neural_correlates": [
                    "Prefrontal cortex",
                    "Anterior cingulate cortex",
                    "Precuneus"
                ],
                "cultural_variations": [
                    "Basic metacognition universal",
                    "Schooling affects explicit metacognitive strategies",
                    "Cultural emphasis on reflection varies"
                ],
                "assessment_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_studies": [
                    "Flavell (1979) - Metacognitive development",
                    "Koriat & Shitzer-Reichert (2002) - Children's metacognition"
                ]
            },
            {
                "capacity_id": "self_recognition_emergence",
                "capacity": DevelopmentalCapacity.SELF_RECOGNITION,
                "typical_emergence_age": "18-24 months",
                "prerequisites": [
                    DevelopmentalCapacity.SENSORY_AWARENESS,
                    DevelopmentalCapacity.OBJECT_PERMANENCE
                ],
                "neural_correlates": [
                    "Medial prefrontal cortex",
                    "Temporal-parietal junction",
                    "Insula"
                ],
                "cultural_variations": [
                    "Mirror test timing relatively consistent",
                    "Some cultural variation in self-concept emphasis",
                    "Independent vs. interdependent self-construal"
                ],
                "assessment_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION
                ],
                "key_studies": [
                    "Amsterdam (1972) - Mirror self-recognition",
                    "Rochat (2003) - Levels of self-awareness"
                ]
            },
            {
                "capacity_id": "object_permanence_emergence",
                "capacity": DevelopmentalCapacity.OBJECT_PERMANENCE,
                "typical_emergence_age": "4-8 months (implicit), 8-12 months (behavioral)",
                "prerequisites": [
                    DevelopmentalCapacity.SENSORY_AWARENESS
                ],
                "neural_correlates": [
                    "Prefrontal cortex (working memory)",
                    "Parietal cortex (object representation)",
                    "Temporal cortex (object recognition)"
                ],
                "cultural_variations": [
                    "Universal development pattern",
                    "Timing relatively consistent across cultures"
                ],
                "assessment_methods": [
                    ResearchMethodology.LOOKING_TIME,
                    ResearchMethodology.HABITUATION,
                    ResearchMethodology.BEHAVIORAL_OBSERVATION
                ],
                "key_studies": [
                    "Piaget (1954) - Original theory",
                    "Baillargeon (1987) - Earlier competence",
                    "Spelke (1992) - Core knowledge"
                ]
            },
            {
                "capacity_id": "autobiographical_memory_emergence",
                "capacity": DevelopmentalCapacity.AUTOBIOGRAPHICAL_MEMORY,
                "typical_emergence_age": "~3.5 years (childhood amnesia offset)",
                "prerequisites": [
                    DevelopmentalCapacity.SELF_RECOGNITION,
                    DevelopmentalCapacity.TEMPORAL_CONSCIOUSNESS
                ],
                "neural_correlates": [
                    "Hippocampus",
                    "Medial prefrontal cortex",
                    "Medial temporal lobe"
                ],
                "cultural_variations": [
                    "Age of earliest memory varies by culture",
                    "Elaborative parenting style affects memory",
                    "Narrative practices influence structure"
                ],
                "assessment_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_studies": [
                    "Nelson & Fivush (2004) - Autobiographical memory development",
                    "Howe & Courage (1997) - Self and memory"
                ]
            },
            {
                "capacity_id": "abstract_reasoning_emergence",
                "capacity": DevelopmentalCapacity.ABSTRACT_REASONING,
                "typical_emergence_age": "~12 years (formal operations)",
                "prerequisites": [
                    DevelopmentalCapacity.METACOGNITION,
                    DevelopmentalCapacity.TEMPORAL_CONSCIOUSNESS
                ],
                "neural_correlates": [
                    "Prefrontal cortex (fully mature ~25)",
                    "Parietal cortex",
                    "Frontal-parietal network"
                ],
                "cultural_variations": [
                    "Formal schooling affects expression",
                    "Not universally achieved in all domains",
                    "Culture-specific applications"
                ],
                "assessment_methods": [
                    ResearchMethodology.BEHAVIORAL_OBSERVATION,
                    ResearchMethodology.LONGITUDINAL_STUDY
                ],
                "key_studies": [
                    "Piaget & Inhelder (1958) - Formal operations",
                    "Kuhn (1999) - Scientific thinking development"
                ]
            }
        ]

    def _get_seed_transitions(self) -> List[Dict[str, Any]]:
        """Return seed consciousness transitions."""
        return [
            {
                "transition_id": "birth_awakening",
                "from_stage": DevelopmentalStage.PRENATAL_LATE,
                "to_stage": DevelopmentalStage.NEONATAL,
                "key_changes": [
                    "From muted fetal awareness to overwhelming sensory input",
                    "Noradrenergic surge at birth",
                    "First breath and environmental transition",
                    "Temperature, light, sound bombardment"
                ],
                "duration": "Hours to days",
                "individual_variation": "Birth circumstances affect transition intensity",
                "triggering_factors": ["Labor hormones", "First breath", "Umbilical separation"],
                "supporting_conditions": ["Maternal contact", "Warmth", "Immediate breastfeeding"]
            },
            {
                "transition_id": "joint_attention_revolution",
                "from_stage": DevelopmentalStage.INFANT_EARLY,
                "to_stage": DevelopmentalStage.INFANT_LATE,
                "key_changes": [
                    "Emergence of shared intentionality",
                    "Joint attention and gaze following",
                    "Understanding others have intentions",
                    "9-month revolution in social cognition"
                ],
                "duration": "Weeks to months (9-12 months)",
                "individual_variation": "Social interaction quality affects timing",
                "triggering_factors": ["Social engagement", "Caregiver responsiveness"],
                "supporting_conditions": ["Rich social environment", "Responsive caregiving"]
            },
            {
                "transition_id": "self_recognition_threshold",
                "from_stage": DevelopmentalStage.INFANT_LATE,
                "to_stage": DevelopmentalStage.TODDLER,
                "key_changes": [
                    "Mirror self-recognition emerges",
                    "Explicit self-concept ('I', 'me', 'mine')",
                    "Self-conscious emotions (pride, shame)",
                    "Language explosion enables self-reference"
                ],
                "duration": "Months (18-24 months)",
                "individual_variation": "Some individual timing variation",
                "triggering_factors": ["Mirror exposure", "Language development"],
                "supporting_conditions": ["Self-referential talk", "Mirror play"]
            },
            {
                "transition_id": "theory_of_mind_threshold",
                "from_stage": DevelopmentalStage.TODDLER,
                "to_stage": DevelopmentalStage.EARLY_CHILDHOOD,
                "key_changes": [
                    "False belief understanding emerges",
                    "Understanding others can be wrong",
                    "Deception becomes possible",
                    "Full naive theory of mind"
                ],
                "duration": "Months to year (3-5 years)",
                "individual_variation": "Language ability, sibling interaction affect timing",
                "triggering_factors": ["Social experience", "Pretend play", "Sibling interaction"],
                "supporting_conditions": ["Rich language environment", "Mental state talk"]
            },
            {
                "transition_id": "concrete_operations_emergence",
                "from_stage": DevelopmentalStage.EARLY_CHILDHOOD,
                "to_stage": DevelopmentalStage.MIDDLE_CHILDHOOD,
                "key_changes": [
                    "Conservation mastered",
                    "Logical operations on concrete objects",
                    "Reversibility understood",
                    "Classification and seriation"
                ],
                "duration": "Gradual (years)",
                "individual_variation": "Domain-specific emergence",
                "triggering_factors": ["Schooling", "Hands-on experience"],
                "supporting_conditions": ["Educational opportunities", "Concrete manipulatives"]
            },
            {
                "transition_id": "formal_operations_emergence",
                "from_stage": DevelopmentalStage.MIDDLE_CHILDHOOD,
                "to_stage": DevelopmentalStage.ADOLESCENCE,
                "key_changes": [
                    "Hypothetical-deductive reasoning",
                    "Abstract thinking capability",
                    "Recursive self-reflection",
                    "Future orientation expands"
                ],
                "duration": "Years (11-15)",
                "individual_variation": "Not universal; domain-specific",
                "triggering_factors": ["Brain maturation", "Educational demands"],
                "supporting_conditions": ["Abstract curriculum", "Scientific reasoning practice"]
            },
            {
                "transition_id": "end_of_life_transition",
                "from_stage": DevelopmentalStage.LATE_ADULT,
                "to_stage": DevelopmentalStage.END_OF_LIFE,
                "key_changes": [
                    "Increased sleep and withdrawal",
                    "Nearing death awareness",
                    "Deathbed visions possible",
                    "Transitional consciousness states"
                ],
                "duration": "Days to weeks typically",
                "individual_variation": "Highly variable based on cause of death",
                "triggering_factors": ["Organ decline", "Disease progression"],
                "supporting_conditions": ["Palliative care", "Family presence", "Peaceful environment"]
            }
        ]

    async def initialize_seed_stage_profiles(self) -> int:
        """Initialize with seed stage profiles."""
        seed_profiles = self._get_seed_stage_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = DevelopmentalStageProfile(
                stage_id=profile_data["stage_id"],
                stage=profile_data["stage"],
                age_range=profile_data["age_range"],
                consciousness_characteristics=profile_data["consciousness_characteristics"],
                emerging_capacities=profile_data["emerging_capacities"],
                neural_developments=profile_data["neural_developments"],
                key_milestones=profile_data["key_milestones"],
                research_methods=profile_data.get("research_methods", []),
                key_researchers=profile_data.get("key_researchers", []),
                maturity_level=MaturityLevel.DEVELOPING
            )
            await self.add_stage_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed stage profiles")
        return count

    async def initialize_seed_capacity_emergences(self) -> int:
        """Initialize with seed capacity emergences."""
        seed_emergences = self._get_seed_capacity_emergences()
        count = 0

        for emergence_data in seed_emergences:
            emergence = CapacityEmergence(
                capacity_id=emergence_data["capacity_id"],
                capacity=emergence_data["capacity"],
                typical_emergence_age=emergence_data["typical_emergence_age"],
                prerequisites=emergence_data["prerequisites"],
                neural_correlates=emergence_data["neural_correlates"],
                cultural_variations=emergence_data.get("cultural_variations", []),
                assessment_methods=emergence_data.get("assessment_methods", []),
                key_studies=emergence_data.get("key_studies", [])
            )
            await self.add_capacity_emergence(emergence)
            count += 1

        logger.info(f"Initialized {count} seed capacity emergences")
        return count

    async def initialize_seed_transitions(self) -> int:
        """Initialize with seed transitions."""
        seed_transitions = self._get_seed_transitions()
        count = 0

        for transition_data in seed_transitions:
            transition = ConsciousnessTransition(
                transition_id=transition_data["transition_id"],
                from_stage=transition_data["from_stage"],
                to_stage=transition_data["to_stage"],
                key_changes=transition_data["key_changes"],
                duration=transition_data["duration"],
                individual_variation=transition_data["individual_variation"],
                triggering_factors=transition_data.get("triggering_factors", []),
                supporting_conditions=transition_data.get("supporting_conditions", [])
            )
            await self.add_transition(transition)
            count += 1

        logger.info(f"Initialized {count} seed transitions")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_stage_profiles()
        emergences_count = await self.initialize_seed_capacity_emergences()
        transitions_count = await self.initialize_seed_transitions()

        return {
            "stage_profiles": profiles_count,
            "capacity_emergences": emergences_count,
            "transitions": transitions_count,
            "total": profiles_count + emergences_count + transitions_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "DevelopmentalStage",
    "DevelopmentalCapacity",
    "ConsciousnessMarker",
    "ResearchMethodology",
    "LifespanDomain",
    "MaturityLevel",
    # Dataclasses
    "DevelopmentalStageProfile",
    "CapacityEmergence",
    "ConsciousnessTransition",
    "LifespanTrajectory",
    "EndOfLifeAwareness",
    "DevelopmentalConsciousnessMaturityState",
    # Interface
    "DevelopmentalConsciousnessInterface",
]
