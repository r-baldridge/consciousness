#!/usr/bin/env python3
"""
Neurodivergent Consciousness Interface

Form 38: The comprehensive interface for understanding and representing
neurodivergent consciousness - the diverse ways human brains can be
structured and function. This form embraces the neurodiversity paradigm,
recognizing neurological differences as natural variations rather than
deficits to be corrected.

Ethical Principles:
- Neurodiversity-affirming language throughout
- Differences framed as variations, not deficits
- First-person accounts centered and respected
- Strengths recognized alongside challenges
- Individual variation acknowledged within neurotypes
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class NeurodivergentType(Enum):
    """
    Categories of neurodivergent conditions and cognitive styles.

    Each type represents a distinct profile of cognitive, perceptual,
    and experiential differences. These are recognized as natural
    variations in human neurology.
    """

    # === DEVELOPMENTAL VARIATIONS ===
    AUTISM_SPECTRUM = "autism_spectrum"
    """Differences in social communication, sensory processing, and focused interests."""

    ADHD = "adhd"
    """Differences in attention regulation, executive function, and emotional processing."""

    SYNESTHESIA = "synesthesia"
    """Cross-modal sensory experiences where one sense triggers another."""

    DYSLEXIA = "dyslexia"
    """Different language processing with associated spatial and holistic strengths."""

    DYSCALCULIA = "dyscalculia"
    """Different mathematical and numerical processing patterns."""

    DYSPRAXIA = "dyspraxia"
    """Motor coordination and planning differences (Developmental Coordination Disorder)."""

    # === NEUROLOGICAL VARIATIONS ===
    TOURETTE_SYNDROME = "tourette_syndrome"
    """Tic conditions with associated cognitive speed and creativity."""

    OCD = "ocd"
    """Differences in certainty-seeking, threat detection, and thoroughness."""

    BIPOLAR_SPECTRUM = "bipolar_spectrum"
    """Cyclical mood and energy patterns with creativity associations."""

    SCHIZOPHRENIA_SPECTRUM = "schizophrenia_spectrum"
    """Altered perception and cognition patterns; inclusion acknowledges both diversity and challenges."""

    # === PROCESSING VARIATIONS ===
    HIGHLY_SENSITIVE_PERSON = "highly_sensitive_person"
    """Heightened sensory processing sensitivity trait (Elaine Aron's research)."""

    GIFTEDNESS = "giftedness"
    """Significantly above-average cognitive ability with intensity and overexcitabilities."""

    TWICE_EXCEPTIONAL = "twice_exceptional"
    """Giftedness co-occurring with other neurodivergent conditions."""


class CognitiveStrength(Enum):
    """
    Cognitive strengths associated with neurodivergent conditions.

    These represent capabilities and advantages that neurodivergent
    individuals often possess, contributing valuable perspectives
    and skills to human diversity.
    """

    PATTERN_RECOGNITION = "pattern_recognition"
    """Enhanced ability to detect patterns, regularities, and underlying structures."""

    DETAIL_ORIENTATION = "detail_orientation"
    """Heightened attention to fine details and ability to maintain focus on specifics."""

    HYPERFOCUS = "hyperfocus"
    """Ability to achieve deep, sustained concentration on tasks of interest."""

    DIVERGENT_THINKING = "divergent_thinking"
    """Ability to generate multiple, varied, and original solutions to problems."""

    SENSORY_ACUITY = "sensory_acuity"
    """Heightened perception across one or more sensory modalities."""

    MEMORY_SPECIALIZATION = "memory_specialization"
    """Enhanced memory in specific domains or for specific types of information."""

    SPATIAL_REASONING = "spatial_reasoning"
    """Superior ability to mentally manipulate spatial information and visualize objects."""

    SYSTEMIZING = "systemizing"
    """Drive to analyze and construct systems with predictable rules."""

    CREATIVITY_FLOW = "creativity_flow"
    """Enhanced creative abilities and ease of entering flow states."""

    EMPATHIC_INTENSITY = "empathic_intensity"
    """Profound capacity for emotional attunement and empathic connection."""

    HOLISTIC_PROCESSING = "holistic_processing"
    """Ability to perceive and integrate the whole picture and see interconnections."""


class SynesthesiaType(Enum):
    """
    Types of synesthesia - cross-modal sensory experiences.

    Synesthesia is a neurological phenomenon in which stimulation of
    one sensory or cognitive pathway automatically triggers experiences
    in another pathway. These are genuine perceptual experiences, not
    metaphors or imagination.
    """

    GRAPHEME_COLOR = "grapheme_color"
    """Letters and/or numbers perceived as having inherent colors."""

    CHROMESTHESIA = "chromesthesia"
    """Sounds, music, and voices trigger color experiences."""

    SPATIAL_SEQUENCE = "spatial_sequence"
    """Ordered sequences (numbers, months, days) occupy specific locations in space."""

    MIRROR_TOUCH = "mirror_touch"
    """Feeling tactile sensations when observing touch to another person."""

    LEXICAL_GUSTATORY = "lexical_gustatory"
    """Words trigger specific taste experiences."""

    ORDINAL_LINGUISTIC_PERSONIFICATION = "ordinal_linguistic_personification"
    """Ordered sequences perceived as having personalities, genders, or characteristics."""

    AUDITORY_TACTILE = "auditory_tactile"
    """Sounds trigger tactile sensations on the body."""

    EMOTION_COLOR = "emotion_color"
    """Emotional states trigger color experiences; may appear as 'auras' around people."""

    TIME_SPACE = "time_space"
    """Time units perceived as having specific spatial locations or arrangements."""


class ProcessingStyle(Enum):
    """
    Cognitive processing style variations.

    Different neurotypes often favor different approaches to
    processing information. These are preferences and tendencies,
    not absolute categories.
    """

    LOCAL_PROCESSING = "local_processing"
    """Detail-focused processing; strengths in perceiving parts and embedded figures."""

    GLOBAL_PROCESSING = "global_processing"
    """Big-picture processing; strengths in perceiving wholes and overall patterns."""

    BOTTOM_UP = "bottom_up"
    """Building understanding from details to whole; data-driven processing."""

    TOP_DOWN = "top_down"
    """Applying frameworks to details; concept-driven processing."""

    SEQUENTIAL = "sequential"
    """Step-by-step, linear processing of information."""

    SIMULTANEOUS = "simultaneous"
    """Parallel, holistic processing of multiple elements at once."""


class SensoryProfile(Enum):
    """
    Sensory processing profiles.

    Neurodivergent conditions often involve differences in how
    sensory information is processed. These profiles can vary
    by modality and context.
    """

    HYPERSENSITIVE = "hypersensitive"
    """Lower thresholds for sensory input; sounds louder, lights brighter, textures more vivid."""

    HYPOSENSITIVE = "hyposensitive"
    """Higher thresholds; may seek more stimulation and be less responsive to input."""

    SENSORY_SEEKING = "sensory_seeking"
    """Actively seeking sensory experiences; craving stimulation."""

    SENSORY_AVOIDING = "sensory_avoiding"
    """Actively avoiding certain sensory experiences; need for reduced input."""

    MIXED = "mixed"
    """Different profiles across modalities; may be hyper- in some senses, hypo- in others."""


class AccommodationType(Enum):
    """
    Types of accommodations that support neurodivergent flourishing.

    Accommodations recognize that challenges often arise from
    environmental mismatch rather than intrinsic deficit.
    """

    ENVIRONMENTAL = "environmental"
    """Physical environment modifications: lighting, sound, space design."""

    COMMUNICATION = "communication"
    """Alternative communication formats, clear expectations, written vs. verbal options."""

    TASK_MODIFICATION = "task_modification"
    """Breaking down tasks, flexible approaches, alternative methods of completion."""

    ASSISTIVE_TECHNOLOGY = "assistive_technology"
    """Tools and technologies that support different processing styles."""

    SOCIAL_SUPPORT = "social_support"
    """Mentorship, peer support, understanding colleagues, communication partners."""


class MaturityLevel(Enum):
    """Depth of knowledge coverage for this domain."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NeurodivergentProfile:
    """
    Represents a profile for a specific neurodivergent type.

    Profiles document the characteristics, strengths, and support
    considerations for each neurotype while affirming these as
    natural variations.
    """
    profile_id: str
    neurotype: NeurodivergentType
    cognitive_strengths: List[CognitiveStrength] = field(default_factory=list)
    processing_styles: List[ProcessingStyle] = field(default_factory=list)
    sensory_profile: Optional[SensoryProfile] = None
    support_needs: List[str] = field(default_factory=list)
    self_description: str = ""
    """Description framed from strengths-based, neurodiversity-affirming perspective."""

    # Additional metadata
    prevalence: Optional[str] = None
    key_researchers: List[str] = field(default_factory=list)
    related_neurotypes: List[str] = field(default_factory=list)
    common_co_occurrences: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Neurotype: {self.neurotype.value}",
            f"Strengths: {', '.join(s.value for s in self.cognitive_strengths)}",
            f"Description: {self.self_description}"
        ]
        return " | ".join(parts)


@dataclass
class SynesthesiaProfile:
    """
    Represents a synesthesia profile documenting cross-modal experiences.

    Synesthesia is framed as a variant form of perception that
    illuminates how the brain integrates sensory information,
    not as a disorder.
    """
    profile_id: str
    synesthesia_types: List[SynesthesiaType] = field(default_factory=list)
    consistency_score: Optional[float] = None
    """Measure of consistency over time (hallmark of genuine synesthesia)."""

    inducer_concurrent_pairs: List[Dict[str, str]] = field(default_factory=list)
    """Examples of inducer-concurrent mappings (e.g., 'A' -> 'red')."""

    developmental_history: str = ""
    """When and how the synesthesia was first noticed."""

    # Additional metadata
    associated_benefits: List[str] = field(default_factory=list)
    famous_synesthetes: List[str] = field(default_factory=list)
    neural_basis_notes: str = ""
    prevalence: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CognitiveStrengthEvidence:
    """
    Documents evidence for a cognitive strength associated with a neurotype.

    This dataclass supports the strengths-based approach by
    providing research backing and real-world examples.
    """
    evidence_id: str
    neurotype: NeurodivergentType
    strength: CognitiveStrength
    research_basis: str = ""
    """Summary of research supporting this strength."""

    real_world_examples: List[str] = field(default_factory=list)
    """Examples of how this strength manifests in practice."""

    notable_individuals: List[str] = field(default_factory=list)
    """Public figures who have demonstrated this strength (with consent/public disclosure)."""

    applications: List[str] = field(default_factory=list)
    """Fields or activities where this strength is particularly valuable."""

    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FirstPersonAccount:
    """
    Preserves first-person accounts from neurodivergent individuals.

    These accounts honor the "nothing about us without us" principle
    and illuminate phenomenological aspects that research alone
    cannot capture.
    """
    account_id: str
    neurotype: NeurodivergentType
    experience_domain: str
    """Area of experience described (sensory, social, cognitive, temporal, etc.)."""

    description: str
    """The first-person account itself."""

    source_attribution: str
    """Attribution (anonymized, composite, or with permission)."""

    # Additional metadata
    themes: List[str] = field(default_factory=list)
    """Key themes in the account."""

    is_composite: bool = False
    """Whether this is a composite account from multiple sources."""

    is_anonymized: bool = True
    """Whether identifying information has been removed."""

    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AccommodationStrategy:
    """
    Documents evidence-based accommodation strategies.

    Accommodations recognize that challenges often arise from
    environmental mismatch and aim to enable flourishing.
    """
    strategy_id: str
    neurotype: NeurodivergentType
    accommodation_type: AccommodationType
    description: str
    effectiveness: str = ""
    """Notes on effectiveness and evidence base."""

    implementation_notes: str = ""
    """Practical notes on implementing this accommodation."""

    # Additional metadata
    context: str = ""
    """Context where this accommodation is most applicable (work, school, home, etc.)."""

    cost_level: str = ""
    """Approximate resource requirements (low/medium/high)."""

    related_strategies: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class NeurodivergentConsciousnessMaturityState:
    """Tracks the maturity of neurodivergent consciousness knowledge."""
    overall_maturity: float = 0.0
    neurotype_coverage: Dict[str, float] = field(default_factory=dict)
    profile_count: int = 0
    synesthesia_profile_count: int = 0
    strength_evidence_count: int = 0
    first_person_account_count: int = 0
    accommodation_strategy_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class NeurodivergentConsciousnessInterface:
    """
    Main interface for Form 38: Neurodivergent Consciousness.

    Provides methods for storing, retrieving, and querying information
    about neurodivergent conditions, cognitive strengths, synesthesia,
    first-person accounts, and accommodation strategies.

    Built on the neurodiversity paradigm, this interface frames
    neurological differences as natural variations rather than
    deficits, while still acknowledging real challenges and
    support needs.
    """

    FORM_ID = "38-neurodivergent-consciousness"
    FORM_NAME = "Neurodivergent Consciousness"

    def __init__(self):
        """Initialize the Neurodivergent Consciousness Interface."""
        # Knowledge indexes
        self.profile_index: Dict[str, NeurodivergentProfile] = {}
        self.synesthesia_index: Dict[str, SynesthesiaProfile] = {}
        self.strength_evidence_index: Dict[str, CognitiveStrengthEvidence] = {}
        self.first_person_account_index: Dict[str, FirstPersonAccount] = {}
        self.accommodation_index: Dict[str, AccommodationStrategy] = {}

        # Cross-reference indexes
        self.neurotype_index: Dict[NeurodivergentType, List[str]] = {}
        self.strength_index: Dict[CognitiveStrength, List[str]] = {}
        self.synesthesia_type_index: Dict[SynesthesiaType, List[str]] = {}
        self.accommodation_type_index: Dict[AccommodationType, List[str]] = {}

        # Maturity tracking
        self.maturity_state = NeurodivergentConsciousnessMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and prepare indexes."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize neurotype index
        for neurotype in NeurodivergentType:
            self.neurotype_index[neurotype] = []

        # Initialize strength index
        for strength in CognitiveStrength:
            self.strength_index[strength] = []

        # Initialize synesthesia type index
        for syn_type in SynesthesiaType:
            self.synesthesia_type_index[syn_type] = []

        # Initialize accommodation type index
        for acc_type in AccommodationType:
            self.accommodation_type_index[acc_type] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # PROFILE METHODS
    # ========================================================================

    async def add_profile(self, profile: NeurodivergentProfile) -> None:
        """Add a neurodivergent profile to the index."""
        self.profile_index[profile.profile_id] = profile

        # Update neurotype index
        if profile.neurotype in self.neurotype_index:
            self.neurotype_index[profile.neurotype].append(profile.profile_id)

        # Update strength index
        for strength in profile.cognitive_strengths:
            if strength in self.strength_index:
                self.strength_index[strength].append(profile.profile_id)

        # Update maturity
        self.maturity_state.profile_count = len(self.profile_index)
        await self._update_maturity()

    async def get_profile(self, profile_id: str) -> Optional[NeurodivergentProfile]:
        """Retrieve a profile by ID."""
        return self.profile_index.get(profile_id)

    async def get_profile_by_neurotype(
        self,
        neurotype: NeurodivergentType
    ) -> Optional[NeurodivergentProfile]:
        """Retrieve the profile for a specific neurotype."""
        profile_ids = self.neurotype_index.get(neurotype, [])
        if profile_ids:
            return self.profile_index.get(profile_ids[0])
        return None

    async def query_profiles_by_strength(
        self,
        strength: CognitiveStrength,
        limit: int = 10
    ) -> List[NeurodivergentProfile]:
        """Query profiles that include a specific cognitive strength."""
        profile_ids = self.strength_index.get(strength, [])[:limit]
        return [
            self.profile_index[pid]
            for pid in profile_ids
            if pid in self.profile_index
        ]

    # ========================================================================
    # SYNESTHESIA METHODS
    # ========================================================================

    async def add_synesthesia_profile(self, profile: SynesthesiaProfile) -> None:
        """Add a synesthesia profile to the index."""
        self.synesthesia_index[profile.profile_id] = profile

        # Update synesthesia type index
        for syn_type in profile.synesthesia_types:
            if syn_type in self.synesthesia_type_index:
                self.synesthesia_type_index[syn_type].append(profile.profile_id)

        # Update maturity
        self.maturity_state.synesthesia_profile_count = len(self.synesthesia_index)
        await self._update_maturity()

    async def get_synesthesia_profile(
        self,
        profile_id: str
    ) -> Optional[SynesthesiaProfile]:
        """Retrieve a synesthesia profile by ID."""
        return self.synesthesia_index.get(profile_id)

    async def query_synesthesia_by_type(
        self,
        syn_type: SynesthesiaType,
        limit: int = 10
    ) -> List[SynesthesiaProfile]:
        """Query synesthesia profiles by type."""
        profile_ids = self.synesthesia_type_index.get(syn_type, [])[:limit]
        return [
            self.synesthesia_index[pid]
            for pid in profile_ids
            if pid in self.synesthesia_index
        ]

    # ========================================================================
    # COGNITIVE STRENGTH EVIDENCE METHODS
    # ========================================================================

    async def add_strength_evidence(
        self,
        evidence: CognitiveStrengthEvidence
    ) -> None:
        """Add cognitive strength evidence to the index."""
        self.strength_evidence_index[evidence.evidence_id] = evidence

        # Update strength index
        if evidence.strength in self.strength_index:
            self.strength_index[evidence.strength].append(evidence.evidence_id)

        # Update neurotype index
        if evidence.neurotype in self.neurotype_index:
            self.neurotype_index[evidence.neurotype].append(evidence.evidence_id)

        # Update maturity
        self.maturity_state.strength_evidence_count = len(self.strength_evidence_index)
        await self._update_maturity()

    async def get_strength_evidence(
        self,
        evidence_id: str
    ) -> Optional[CognitiveStrengthEvidence]:
        """Retrieve cognitive strength evidence by ID."""
        return self.strength_evidence_index.get(evidence_id)

    async def query_evidence_by_neurotype(
        self,
        neurotype: NeurodivergentType,
        limit: int = 10
    ) -> List[CognitiveStrengthEvidence]:
        """Query strength evidence for a specific neurotype."""
        results = []
        for evidence in self.strength_evidence_index.values():
            if evidence.neurotype == neurotype:
                results.append(evidence)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # FIRST-PERSON ACCOUNT METHODS
    # ========================================================================

    async def add_first_person_account(self, account: FirstPersonAccount) -> None:
        """Add a first-person account to the index."""
        self.first_person_account_index[account.account_id] = account

        # Update neurotype index
        if account.neurotype in self.neurotype_index:
            self.neurotype_index[account.neurotype].append(account.account_id)

        # Update maturity
        self.maturity_state.first_person_account_count = len(
            self.first_person_account_index
        )
        await self._update_maturity()

    async def get_first_person_account(
        self,
        account_id: str
    ) -> Optional[FirstPersonAccount]:
        """Retrieve a first-person account by ID."""
        return self.first_person_account_index.get(account_id)

    async def query_accounts_by_neurotype(
        self,
        neurotype: NeurodivergentType,
        limit: int = 10
    ) -> List[FirstPersonAccount]:
        """Query first-person accounts for a specific neurotype."""
        results = []
        for account in self.first_person_account_index.values():
            if account.neurotype == neurotype:
                results.append(account)
                if len(results) >= limit:
                    break
        return results

    async def query_accounts_by_domain(
        self,
        domain: str,
        limit: int = 10
    ) -> List[FirstPersonAccount]:
        """Query first-person accounts by experience domain."""
        results = []
        domain_lower = domain.lower()
        for account in self.first_person_account_index.values():
            if domain_lower in account.experience_domain.lower():
                results.append(account)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # ACCOMMODATION STRATEGY METHODS
    # ========================================================================

    async def add_accommodation(self, strategy: AccommodationStrategy) -> None:
        """Add an accommodation strategy to the index."""
        self.accommodation_index[strategy.strategy_id] = strategy

        # Update accommodation type index
        if strategy.accommodation_type in self.accommodation_type_index:
            self.accommodation_type_index[strategy.accommodation_type].append(
                strategy.strategy_id
            )

        # Update neurotype index
        if strategy.neurotype in self.neurotype_index:
            self.neurotype_index[strategy.neurotype].append(strategy.strategy_id)

        # Update maturity
        self.maturity_state.accommodation_strategy_count = len(
            self.accommodation_index
        )
        await self._update_maturity()

    async def get_accommodation(
        self,
        strategy_id: str
    ) -> Optional[AccommodationStrategy]:
        """Retrieve an accommodation strategy by ID."""
        return self.accommodation_index.get(strategy_id)

    async def query_accommodations_by_neurotype(
        self,
        neurotype: NeurodivergentType,
        limit: int = 10
    ) -> List[AccommodationStrategy]:
        """Query accommodations for a specific neurotype."""
        results = []
        for strategy in self.accommodation_index.values():
            if strategy.neurotype == neurotype:
                results.append(strategy)
                if len(results) >= limit:
                    break
        return results

    async def query_accommodations_by_type(
        self,
        acc_type: AccommodationType,
        limit: int = 10
    ) -> List[AccommodationStrategy]:
        """Query accommodations by type."""
        strategy_ids = self.accommodation_type_index.get(acc_type, [])[:limit]
        return [
            self.accommodation_index[sid]
            for sid in strategy_ids
            if sid in self.accommodation_index
        ]

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.profile_count +
            self.maturity_state.synesthesia_profile_count +
            self.maturity_state.strength_evidence_count +
            self.maturity_state.first_person_account_count +
            self.maturity_state.accommodation_strategy_count
        )

        # Simple maturity calculation
        target_items = 200  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update neurotype coverage
        for neurotype in NeurodivergentType:
            count = len(self.neurotype_index.get(neurotype, []))
            target_per_neurotype = 10
            self.maturity_state.neurotype_coverage[neurotype.value] = min(
                1.0, count / target_per_neurotype
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> NeurodivergentConsciousnessMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_profiles(self) -> List[Dict[str, Any]]:
        """Return seed profiles for initialization."""
        return [
            # Autism Spectrum
            {
                "profile_id": "autism_spectrum_profile",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "cognitive_strengths": [
                    CognitiveStrength.PATTERN_RECOGNITION,
                    CognitiveStrength.DETAIL_ORIENTATION,
                    CognitiveStrength.HYPERFOCUS,
                    CognitiveStrength.SYSTEMIZING,
                    CognitiveStrength.MEMORY_SPECIALIZATION,
                ],
                "processing_styles": [
                    ProcessingStyle.LOCAL_PROCESSING,
                    ProcessingStyle.BOTTOM_UP,
                ],
                "sensory_profile": SensoryProfile.MIXED,
                "support_needs": [
                    "Predictable environments and clear expectations",
                    "Sensory accommodations (lighting, sound, textures)",
                    "Processing time for social and verbal communication",
                    "Respect for focused interests as valuable engagement",
                ],
                "self_description": "Autism involves a different way of perceiving and processing the world, often characterized by heightened pattern recognition, deep focus on areas of interest, and different sensory experiences. Many autistic individuals describe thinking in pictures, patterns, or systems. Social communication may work differently but is not absent - autistic people often connect deeply with others who share their interests or communication style.",
                "prevalence": "1-2% of population",
                "key_researchers": ["Temple Grandin", "Simon Baron-Cohen", "Uta Frith", "Kamila Markram"],
                "common_co_occurrences": ["ADHD", "Synesthesia", "Dyspraxia", "Giftedness"],
            },
            # ADHD
            {
                "profile_id": "adhd_profile",
                "neurotype": NeurodivergentType.ADHD,
                "cognitive_strengths": [
                    CognitiveStrength.HYPERFOCUS,
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.CREATIVITY_FLOW,
                    CognitiveStrength.HOLISTIC_PROCESSING,
                ],
                "processing_styles": [
                    ProcessingStyle.GLOBAL_PROCESSING,
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.SENSORY_SEEKING,
                "support_needs": [
                    "External scaffolding for time management",
                    "Breaking large tasks into smaller, engaging pieces",
                    "Movement and novelty to maintain engagement",
                    "Understanding of hyperfocus as both strength and challenge",
                ],
                "self_description": "ADHD involves a different relationship with attention, time, and executive function. Rather than an attention deficit, many describe having attention that is abundant but difficult to direct at will. When engaged with interesting tasks, individuals can achieve remarkable hyperfocus. The ADHD brain often excels at divergent thinking, crisis performance, and creative problem-solving. Time may be experienced differently, with 'now' and 'not now' being the primary temporal categories.",
                "prevalence": "5-7% of children, 2.5-4% of adults",
                "key_researchers": ["Russell Barkley", "Edward Hallowell", "Thomas Brown"],
                "common_co_occurrences": ["Autism Spectrum", "Dyslexia", "Giftedness"],
            },
            # Synesthesia
            {
                "profile_id": "synesthesia_profile",
                "neurotype": NeurodivergentType.SYNESTHESIA,
                "cognitive_strengths": [
                    CognitiveStrength.SENSORY_ACUITY,
                    CognitiveStrength.MEMORY_SPECIALIZATION,
                    CognitiveStrength.CREATIVITY_FLOW,
                ],
                "processing_styles": [
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Recognition that synesthetic experiences are real perceptions",
                    "Flexibility when synesthetic responses affect preferences",
                    "Understanding that 'wrong' colors or associations can feel disturbing",
                ],
                "self_description": "Synesthesia involves automatic cross-modal experiences where stimulation of one sense triggers perception in another. A synesthete might genuinely see colors when hearing music, taste shapes, or perceive numbers as having personalities. These experiences are consistent over time, involuntary, and felt as real perceptions rather than imagination. Synesthesia often provides memory advantages and contributes to creative and artistic abilities.",
                "prevalence": "2-4% of population",
                "key_researchers": ["Richard Cytowic", "David Eagleman", "Jamie Ward"],
                "common_co_occurrences": ["Autism Spectrum", "Giftedness"],
            },
            # Dyslexia
            {
                "profile_id": "dyslexia_profile",
                "neurotype": NeurodivergentType.DYSLEXIA,
                "cognitive_strengths": [
                    CognitiveStrength.SPATIAL_REASONING,
                    CognitiveStrength.HOLISTIC_PROCESSING,
                    CognitiveStrength.PATTERN_RECOGNITION,
                    CognitiveStrength.DIVERGENT_THINKING,
                ],
                "processing_styles": [
                    ProcessingStyle.GLOBAL_PROCESSING,
                    ProcessingStyle.TOP_DOWN,
                ],
                "sensory_profile": None,
                "support_needs": [
                    "Multiple modalities for information access (audio, visual, text)",
                    "Extended time for reading-intensive tasks",
                    "Recognition of strong spatial and visualization abilities",
                    "Alternative methods for demonstrating knowledge",
                ],
                "self_description": "Dyslexia involves a different way of processing language, often accompanied by notable strengths in spatial reasoning, big-picture thinking, and three-dimensional visualization. While reading and spelling may require more effort, dyslexic individuals frequently excel in fields requiring visual-spatial skills like architecture, engineering, and design. Many successful entrepreneurs are dyslexic, perhaps because their holistic thinking style helps them see market opportunities others miss.",
                "prevalence": "5-17% depending on definition",
                "key_researchers": ["Sally Shaywitz", "Thomas West", "Brock Eide", "Fernette Eide"],
                "common_co_occurrences": ["ADHD", "Dyscalculia"],
            },
            # Dyscalculia
            {
                "profile_id": "dyscalculia_profile",
                "neurotype": NeurodivergentType.DYSCALCULIA,
                "cognitive_strengths": [
                    CognitiveStrength.HOLISTIC_PROCESSING,
                ],
                "processing_styles": [
                    ProcessingStyle.GLOBAL_PROCESSING,
                ],
                "sensory_profile": None,
                "support_needs": [
                    "Visual and concrete representations of mathematical concepts",
                    "Extended time for numerical tasks",
                    "Calculator and technology support",
                    "Alternative approaches to quantitative reasoning",
                ],
                "self_description": "Dyscalculia involves a different relationship with numbers, mathematical facts, and quantitative reasoning. While number sense and procedural math may be challenging, individuals with dyscalculia often have strengths in verbal reasoning, qualitative analysis, and narrative understanding. The condition highlights that mathematical ability is just one form of intelligence among many.",
                "prevalence": "3-6% of population",
                "key_researchers": [],
                "common_co_occurrences": ["Dyslexia", "ADHD"],
            },
            # Dyspraxia
            {
                "profile_id": "dyspraxia_profile",
                "neurotype": NeurodivergentType.DYSPRAXIA,
                "cognitive_strengths": [
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.EMPATHIC_INTENSITY,
                ],
                "processing_styles": [
                    ProcessingStyle.GLOBAL_PROCESSING,
                ],
                "sensory_profile": SensoryProfile.MIXED,
                "support_needs": [
                    "Additional time for tasks requiring motor coordination",
                    "Technology alternatives for handwriting",
                    "Breaking complex motor sequences into steps",
                    "Patience and understanding for coordination challenges",
                ],
                "self_description": "Dyspraxia (Developmental Coordination Disorder) involves differences in motor planning and coordination. While fine and gross motor tasks may require more effort, many individuals with dyspraxia develop strong verbal abilities, creative thinking, empathy, and determined problem-solving skills from adapting to challenges throughout life.",
                "prevalence": "5-6% of children",
                "key_researchers": [],
                "common_co_occurrences": ["Autism Spectrum", "ADHD", "Dyslexia"],
            },
            # Tourette Syndrome
            {
                "profile_id": "tourette_profile",
                "neurotype": NeurodivergentType.TOURETTE_SYNDROME,
                "cognitive_strengths": [
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.CREATIVITY_FLOW,
                ],
                "processing_styles": [
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Understanding that tics are involuntary",
                    "Ability to take breaks when tic suppression is exhausting",
                    "Reduction of stress which can exacerbate tics",
                    "Acceptance in social and professional environments",
                ],
                "self_description": "Tourette Syndrome involves motor and vocal tics that are largely involuntary. Many individuals develop enhanced cognitive control from years of managing tics, and some report heightened creativity and quick cognitive processing. The premonitory urge before tics represents a unique form of body awareness.",
                "prevalence": "0.3-0.8% of children",
                "key_researchers": [],
                "common_co_occurrences": ["ADHD", "OCD"],
            },
            # OCD
            {
                "profile_id": "ocd_profile",
                "neurotype": NeurodivergentType.OCD,
                "cognitive_strengths": [
                    CognitiveStrength.DETAIL_ORIENTATION,
                    CognitiveStrength.PATTERN_RECOGNITION,
                    CognitiveStrength.SYSTEMIZING,
                ],
                "processing_styles": [
                    ProcessingStyle.LOCAL_PROCESSING,
                    ProcessingStyle.SEQUENTIAL,
                ],
                "sensory_profile": None,
                "support_needs": [
                    "Understanding that compulsions are driven by genuine distress",
                    "Environments that reduce triggers when possible",
                    "Patience with rituals and checking behaviors",
                    "Access to evidence-based support when desired",
                ],
                "self_description": "OCD involves intrusive thoughts (obsessions) and repetitive behaviors (compulsions), reflecting differences in certainty-seeking and threat detection. While the distress from intrusive thoughts is real, the underlying tendency toward thoroughness, attention to detail, and conscientiousness can be channeled productively in many contexts.",
                "prevalence": "1-2% of population",
                "key_researchers": [],
                "common_co_occurrences": ["Tourette Syndrome", "Autism Spectrum"],
            },
            # Bipolar Spectrum
            {
                "profile_id": "bipolar_profile",
                "neurotype": NeurodivergentType.BIPOLAR_SPECTRUM,
                "cognitive_strengths": [
                    CognitiveStrength.CREATIVITY_FLOW,
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.EMPATHIC_INTENSITY,
                ],
                "processing_styles": [
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Stable routines, especially around sleep",
                    "Monitoring for mood shifts",
                    "Understanding of productivity variations",
                    "Support during depressive phases",
                ],
                "self_description": "Bipolar involves cyclical mood states including periods of elevated energy (mania/hypomania) and depression. During elevated states, many experience enhanced creativity, rapid ideation, and prolific productivity. The condition is overrepresented among artists, writers, and musicians. While mood episodes present genuine challenges, many with bipolar conditions achieve at high levels and contribute significantly to creative fields.",
                "prevalence": "1-4% depending on spectrum definition",
                "key_researchers": ["Kay Redfield Jamison"],
                "common_co_occurrences": ["ADHD", "Anxiety"],
            },
            # Schizophrenia Spectrum
            {
                "profile_id": "schizophrenia_profile",
                "neurotype": NeurodivergentType.SCHIZOPHRENIA_SPECTRUM,
                "cognitive_strengths": [
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.CREATIVITY_FLOW,
                ],
                "processing_styles": [
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Stable, low-stress environments",
                    "Social support and connection",
                    "Access to appropriate healthcare",
                    "Reduction of stigma and isolation",
                ],
                "self_description": "Schizophrenia spectrum conditions involve differences in perception, thought processes, and reality testing. Inclusion in neurodiversity frameworks is debated, as these experiences can cause significant distress. However, the divergent thinking patterns are sometimes associated with enhanced creativity, and first-degree relatives show increased creative achievement. Understanding these experiences as part of human cognitive diversity can reduce stigma while still acknowledging support needs.",
                "prevalence": "Approximately 1%",
                "key_researchers": [],
                "common_co_occurrences": [],
            },
            # Highly Sensitive Person
            {
                "profile_id": "hsp_profile",
                "neurotype": NeurodivergentType.HIGHLY_SENSITIVE_PERSON,
                "cognitive_strengths": [
                    CognitiveStrength.SENSORY_ACUITY,
                    CognitiveStrength.EMPATHIC_INTENSITY,
                    CognitiveStrength.CREATIVITY_FLOW,
                    CognitiveStrength.DETAIL_ORIENTATION,
                ],
                "processing_styles": [
                    ProcessingStyle.BOTTOM_UP,
                    ProcessingStyle.LOCAL_PROCESSING,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Low-stimulation environments or ability to retreat",
                    "Processing time after intense experiences",
                    "Recognition of depth of processing as valuable",
                    "Respect for need to avoid overstimulation",
                ],
                "self_description": "Sensory Processing Sensitivity (the HSP trait) involves deeper cognitive processing of stimuli and greater emotional responsiveness. HSPs notice subtleties others miss, process experiences more deeply, and respond more intensely to both positive and negative stimulation. While overstimulation can be challenging, the trait contributes to heightened aesthetic sensitivity, conscientiousness, and deep reflection. The trait is found in approximately 20% of humans and over 100 other species, suggesting evolutionary advantage.",
                "prevalence": "15-20% of population",
                "key_researchers": ["Elaine Aron"],
                "common_co_occurrences": ["Giftedness"],
            },
            # Giftedness
            {
                "profile_id": "giftedness_profile",
                "neurotype": NeurodivergentType.GIFTEDNESS,
                "cognitive_strengths": [
                    CognitiveStrength.PATTERN_RECOGNITION,
                    CognitiveStrength.MEMORY_SPECIALIZATION,
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.CREATIVITY_FLOW,
                    CognitiveStrength.SYSTEMIZING,
                ],
                "processing_styles": [
                    ProcessingStyle.GLOBAL_PROCESSING,
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.HYPERSENSITIVE,
                "support_needs": [
                    "Appropriate intellectual challenge",
                    "Understanding of asynchronous development",
                    "Support for emotional intensity",
                    "Peer connections with intellectual equals",
                ],
                "self_description": "Giftedness involves significantly above-average cognitive ability often accompanied by intensity and overexcitabilities (Dabrowski). Gifted individuals may experience intellectual, emotional, imaginational, sensual, and psychomotor intensities that affect their engagement with the world. Asynchronous development means cognitive ability may far outpace social or emotional development. Giftedness brings unique social-emotional needs alongside cognitive capabilities.",
                "prevalence": "2-5% depending on definition",
                "key_researchers": ["Kazimierz Dabrowski", "Leta Hollingworth", "Linda Silverman"],
                "common_co_occurrences": ["ADHD", "Autism Spectrum", "HSP"],
            },
            # Twice-Exceptional
            {
                "profile_id": "twice_exceptional_profile",
                "neurotype": NeurodivergentType.TWICE_EXCEPTIONAL,
                "cognitive_strengths": [
                    CognitiveStrength.PATTERN_RECOGNITION,
                    CognitiveStrength.DIVERGENT_THINKING,
                    CognitiveStrength.CREATIVITY_FLOW,
                ],
                "processing_styles": [
                    ProcessingStyle.SIMULTANEOUS,
                ],
                "sensory_profile": SensoryProfile.MIXED,
                "support_needs": [
                    "Recognition of both giftedness and other conditions",
                    "Support that addresses challenges without limiting strengths",
                    "Understanding that abilities may mask difficulties and vice versa",
                    "Individualized approaches that recognize the full profile",
                ],
                "self_description": "Twice-exceptional (2e) individuals are gifted while also having one or more other neurodivergent conditions (such as ADHD, autism, or dyslexia). The giftedness may mask the other condition, and the other condition may mask the giftedness, leading to late or missed identification of both. 2e individuals often experience frustration from uneven abilities - being brilliant in some areas while struggling in others. They may not qualify for either gifted or disability services, falling through the cracks of systems designed for more uniform profiles.",
                "prevalence": "Unknown, but significant portion of gifted population",
                "key_researchers": ["Linda Silverman", "Susan Baum"],
                "common_co_occurrences": ["Giftedness with ADHD", "Giftedness with Autism", "Giftedness with Dyslexia"],
            },
        ]

    def _get_seed_synesthesia_profiles(self) -> List[Dict[str, Any]]:
        """Return seed synesthesia profiles for initialization."""
        return [
            {
                "profile_id": "grapheme_color_synesthesia",
                "synesthesia_types": [SynesthesiaType.GRAPHEME_COLOR],
                "consistency_score": 0.95,
                "inducer_concurrent_pairs": [
                    {"inducer": "A", "concurrent": "red"},
                    {"inducer": "B", "concurrent": "blue"},
                    {"inducer": "7", "concurrent": "green"},
                ],
                "developmental_history": "Typically present from earliest memories of reading. Many synesthetes do not realize until adulthood that others do not share their experience, assuming 'seeing' letters in color is universal.",
                "associated_benefits": [
                    "Enhanced memory for text and numbers",
                    "Automatic visual encoding aids recall",
                    "May contribute to spelling ability",
                ],
                "famous_synesthetes": ["Vladimir Nabokov (author)", "Daniel Tammet (savant mathematician)"],
                "neural_basis_notes": "V4 color area activates when viewing graphemes. Increased connectivity between grapheme recognition and color processing areas. May result from reduced developmental pruning.",
                "prevalence": "1-2% of population (most common form)",
            },
            {
                "profile_id": "chromesthesia_synesthesia",
                "synesthesia_types": [SynesthesiaType.CHROMESTHESIA],
                "consistency_score": 0.92,
                "inducer_concurrent_pairs": [
                    {"inducer": "C major chord", "concurrent": "white/bright"},
                    {"inducer": "D major chord", "concurrent": "yellow"},
                    {"inducer": "violin timbre", "concurrent": "silver/crystalline"},
                ],
                "developmental_history": "Often discovered through music education or discussion with other musicians. Common among musicians and may influence musical composition and performance.",
                "associated_benefits": [
                    "Enriched musical experience",
                    "May facilitate perfect pitch development",
                    "Compositional inspiration",
                    "Enhanced memory for melodies",
                ],
                "famous_synesthetes": [
                    "Franz Liszt (composer)",
                    "Olivier Messiaen (composer)",
                    "Billy Joel (musician)",
                    "Pharrell Williams (musician/producer)",
                ],
                "neural_basis_notes": "Connections between auditory and visual cortices. Specific pitches may map to specific colors consistently. Timbres influence color qualities.",
                "prevalence": "Common among synesthetes, especially musicians",
            },
            {
                "profile_id": "spatial_sequence_synesthesia",
                "synesthesia_types": [SynesthesiaType.SPATIAL_SEQUENCE],
                "consistency_score": 0.97,
                "inducer_concurrent_pairs": [
                    {"inducer": "numbers 1-10", "concurrent": "curve upward to the right"},
                    {"inducer": "months of year", "concurrent": "oval shape with January at top"},
                    {"inducer": "days of week", "concurrent": "horizontal line extending forward"},
                ],
                "developmental_history": "Often not recognized as unusual until discussing calendars or number lines with others. Many believe everyone experiences sequences spatially.",
                "associated_benefits": [
                    "Enhanced calendar and date memory",
                    "Strong sense of temporal organization",
                    "Mathematical intuition",
                    "Easy navigation of sequences",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "Involves parietal regions associated with spatial processing and numerical cognition. The 'SNARC effect' (spatial-numerical association) may be related.",
                "prevalence": "10-15% may have some form of number form",
            },
            {
                "profile_id": "mirror_touch_synesthesia",
                "synesthesia_types": [SynesthesiaType.MIRROR_TOUCH],
                "consistency_score": 0.88,
                "inducer_concurrent_pairs": [
                    {"inducer": "seeing someone's cheek touched", "concurrent": "sensation on own cheek"},
                    {"inducer": "watching someone receive injection", "concurrent": "pain sensation in same location"},
                ],
                "developmental_history": "May cause difficulty watching medical procedures or violent content. Often correlates with high empathy measures.",
                "associated_benefits": [
                    "Heightened empathic attunement",
                    "Deep understanding of others' physical experiences",
                    "May facilitate caregiving roles",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "Hyperactive mirror neuron system. Reduced self-other distinction in somatosensory processing. Enhanced activity in empathy-related brain regions.",
                "prevalence": "Approximately 1.5% of population",
            },
            {
                "profile_id": "lexical_gustatory_synesthesia",
                "synesthesia_types": [SynesthesiaType.LEXICAL_GUSTATORY],
                "consistency_score": 0.94,
                "inducer_concurrent_pairs": [
                    {"inducer": "word 'Philip'", "concurrent": "taste of oranges"},
                    {"inducer": "word 'Tuesday'", "concurrent": "taste of ground beef"},
                    {"inducer": "number 4", "concurrent": "taste of toast"},
                ],
                "developmental_history": "Relatively rare form. May influence food preferences and word choices. Some tastes pleasant, others not, affecting language use.",
                "associated_benefits": [
                    "Unique relationship with language",
                    "Enhanced memory through gustatory encoding",
                ],
                "famous_synesthetes": ["James Wannerton (most documented case)"],
                "neural_basis_notes": "Connections between language processing areas and gustatory cortex. Word sounds may drive associations more than meanings.",
                "prevalence": "Less than 0.2% (relatively rare)",
            },
            {
                "profile_id": "olp_synesthesia",
                "synesthesia_types": [SynesthesiaType.ORDINAL_LINGUISTIC_PERSONIFICATION],
                "consistency_score": 0.91,
                "inducer_concurrent_pairs": [
                    {"inducer": "number 7", "concurrent": "serious older man"},
                    {"inducer": "number 8", "concurrent": "friendly woman"},
                    {"inducer": "letter A", "concurrent": "proud, red personality"},
                    {"inducer": "Wednesday", "concurrent": "middle-aged grumpy male"},
                ],
                "developmental_history": "Often present from childhood. May facilitate mathematical and calendar memory through personification.",
                "associated_benefits": [
                    "Enhanced sequence memory",
                    "Richer relationship with abstract concepts",
                    "May aid mathematical intuition",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "May involve social cognition regions being activated by ordinal sequences. Personification is experienced as intrinsic, not assigned.",
                "prevalence": "Approximately 1% of population",
            },
            {
                "profile_id": "auditory_tactile_synesthesia",
                "synesthesia_types": [SynesthesiaType.AUDITORY_TACTILE],
                "consistency_score": 0.89,
                "inducer_concurrent_pairs": [
                    {"inducer": "bass notes", "concurrent": "pressure on chest"},
                    {"inducer": "high-pitched sounds", "concurrent": "tingling in hands"},
                    {"inducer": "certain voices", "concurrent": "velvet texture on skin"},
                ],
                "developmental_history": "May significantly affect music experience. Some sounds produce pleasant sensations, others unpleasant.",
                "associated_benefits": [
                    "Enriched auditory experience",
                    "Multi-sensory engagement with music",
                    "Enhanced body awareness",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "Cross-activation between auditory and somatosensory cortices.",
                "prevalence": "Relatively uncommon",
            },
            {
                "profile_id": "emotion_color_synesthesia",
                "synesthesia_types": [SynesthesiaType.EMOTION_COLOR],
                "consistency_score": 0.86,
                "inducer_concurrent_pairs": [
                    {"inducer": "anger", "concurrent": "red tinge to perception"},
                    {"inducer": "sadness in others", "concurrent": "blue aura"},
                    {"inducer": "anxiety", "concurrent": "yellow-green"},
                ],
                "developmental_history": "May explain traditional 'aura' perceptions across cultures. Provides neurological basis for reported aura seeing.",
                "associated_benefits": [
                    "Visual cues for emotional states",
                    "Enhanced emotional awareness",
                    "May facilitate emotional intelligence",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "Cross-activation between emotional processing and visual color areas. May explain cross-cultural consistency in emotion-color associations.",
                "prevalence": "Unknown, may overlap with other forms",
            },
            {
                "profile_id": "time_space_synesthesia",
                "synesthesia_types": [SynesthesiaType.TIME_SPACE],
                "consistency_score": 0.93,
                "inducer_concurrent_pairs": [
                    {"inducer": "future", "concurrent": "extends to the right"},
                    {"inducer": "10 AM", "concurrent": "up high"},
                    {"inducer": "1800s", "concurrent": "behind and to the left"},
                ],
                "developmental_history": "Similar to spatial sequence synesthesia but specifically for time. Provides spatial scaffolding for temporal reasoning.",
                "associated_benefits": [
                    "Strong temporal organization",
                    "Enhanced date and historical memory",
                    "Intuitive grasp of timelines",
                ],
                "famous_synesthetes": [],
                "neural_basis_notes": "Involves parietal regions. May be related to how the brain spatially encodes temporal information.",
                "prevalence": "Related to spatial sequence synesthesia, fairly common",
            },
        ]

    def _get_seed_strength_evidence(self) -> List[Dict[str, Any]]:
        """Return seed cognitive strength evidence for initialization."""
        return [
            {
                "evidence_id": "autism_pattern_recognition",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "strength": CognitiveStrength.PATTERN_RECOGNITION,
                "research_basis": "Studies consistently show superior performance on embedded figures tests, enhanced statistical learning in some domains, and exceptional ability to detect regularities in visual and auditory information. The 'Pattern Seekers' research by Baron-Cohen documents how autistic pattern recognition has contributed to human innovation.",
                "real_world_examples": [
                    "Identifying trends and anomalies in large datasets",
                    "Detecting subtle changes or errors in code",
                    "Recognizing musical patterns and structures",
                    "Quality assurance and testing roles",
                ],
                "notable_individuals": [
                    "Temple Grandin (animal behavior patterns)",
                    "Many successful software developers",
                ],
                "applications": ["Data science", "Software testing", "Scientific research", "Music analysis"],
            },
            {
                "evidence_id": "adhd_divergent_thinking",
                "neurotype": NeurodivergentType.ADHD,
                "strength": CognitiveStrength.DIVERGENT_THINKING,
                "research_basis": "Multiple studies show higher divergent thinking scores in ADHD populations. The tendency for mind-wandering, often framed as a deficit, may facilitate creative associations. Research by Holly White and others documents enhanced creativity in ADHD.",
                "real_world_examples": [
                    "Generating multiple solutions to open-ended problems",
                    "Making unexpected connections between ideas",
                    "Entrepreneurial opportunity recognition",
                    "Creative writing and artistic production",
                ],
                "notable_individuals": [
                    "Richard Branson (entrepreneur)",
                    "David Neeleman (JetBlue founder)",
                    "Many artists and creative professionals",
                ],
                "applications": ["Entrepreneurship", "Creative industries", "Brainstorming", "Innovation"],
            },
            {
                "evidence_id": "dyslexia_spatial_reasoning",
                "neurotype": NeurodivergentType.DYSLEXIA,
                "strength": CognitiveStrength.SPATIAL_REASONING,
                "research_basis": "MIT studies found dyslexic students overrepresented in engineering programs. Research by the Eides ('The Dyslexic Advantage') documents superior 3D visualization, mental rotation, and spatial memory in dyslexic populations.",
                "real_world_examples": [
                    "Architectural visualization and design",
                    "Engineering problem-solving",
                    "Surgical planning and execution",
                    "Art and sculpture creation",
                ],
                "notable_individuals": [
                    "Many architects (overrepresentation documented)",
                    "Many engineers",
                    "Many surgeons",
                ],
                "applications": ["Architecture", "Engineering", "Surgery", "Design", "Art"],
            },
            {
                "evidence_id": "synesthesia_memory",
                "neurotype": NeurodivergentType.SYNESTHESIA,
                "strength": CognitiveStrength.MEMORY_SPECIALIZATION,
                "research_basis": "Synesthetes show enhanced memory for information that triggers their synesthesia. Multi-modal encoding provides additional retrieval cues. Famous cases like Daniel Tammet demonstrate exceptional memory facilitated by synesthetic experiences.",
                "real_world_examples": [
                    "Superior recall for names, numbers, and dates",
                    "Enhanced vocabulary acquisition",
                    "Better memory for music and melodies",
                    "Improved spelling through color consistency",
                ],
                "notable_individuals": [
                    "Daniel Tammet (memorized pi to 22,514 digits using synesthesia)",
                    "Solomon Shereshevsky (famous mnemonist with multiple synesthesias)",
                ],
                "applications": ["Academic learning", "Language acquisition", "Music", "Memory sports"],
            },
            {
                "evidence_id": "hsp_empathic_intensity",
                "neurotype": NeurodivergentType.HIGHLY_SENSITIVE_PERSON,
                "strength": CognitiveStrength.EMPATHIC_INTENSITY,
                "research_basis": "Elaine Aron's research documents that HSPs show greater activation in brain regions associated with empathy and emotional processing. They pick up on subtle emotional cues others miss and process emotional information more deeply.",
                "real_world_examples": [
                    "Sensing emotional atmospheres in groups",
                    "Responding sensitively to others' needs",
                    "Deep emotional connections in relationships",
                    "Being moved profoundly by art and injustice",
                ],
                "notable_individuals": [],
                "applications": ["Counseling", "Healthcare", "Teaching", "Leadership", "Art"],
            },
            {
                "evidence_id": "bipolar_creativity",
                "neurotype": NeurodivergentType.BIPOLAR_SPECTRUM,
                "strength": CognitiveStrength.CREATIVITY_FLOW,
                "research_basis": "Kay Redfield Jamison's research ('Touched with Fire') documents overrepresentation of bipolar spectrum conditions among eminent artists, writers, and musicians. Hypomanic states are associated with enhanced ideation, reduced inhibition, and prolific output.",
                "real_world_examples": [
                    "Prolific artistic production during elevated states",
                    "Novel ideation and creative breakthroughs",
                    "Cross-domain creative connections",
                    "High achievement in creative fields",
                ],
                "notable_individuals": [
                    "Virginia Woolf (author)",
                    "Vincent van Gogh (artist)",
                    "Ernest Hemingway (author)",
                    "Robert Schumann (composer)",
                ],
                "applications": ["Arts", "Writing", "Music composition", "Creative leadership"],
            },
            {
                "evidence_id": "giftedness_systemizing",
                "neurotype": NeurodivergentType.GIFTEDNESS,
                "strength": CognitiveStrength.SYSTEMIZING,
                "research_basis": "Gifted individuals often show exceptional ability to understand and construct complex systems, whether in mathematics, science, language, or other domains. Dabrowski's intellectual overexcitability describes the intense drive to understand how things work.",
                "real_world_examples": [
                    "Mastering complex theoretical frameworks",
                    "Creating elaborate classification systems",
                    "Understanding intricate rule-based systems",
                    "Advanced programming and algorithm design",
                ],
                "notable_individuals": [],
                "applications": ["Academia", "Research", "Programming", "Law", "Medicine"],
            },
        ]

    def _get_seed_first_person_accounts(self) -> List[Dict[str, Any]]:
        """Return seed first-person accounts for initialization."""
        return [
            {
                "account_id": "autism_sensory_account",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "experience_domain": "sensory",
                "description": "It's like having a radio that can't be turned down, and sometimes picking up multiple stations at once. The fluorescent lights buzz at a frequency that drills into my head. The texture of certain fabrics makes my skin crawl. But then there's the flip side - I can hear the beauty in sounds others miss, see patterns in visual noise, and the right textures bring genuine joy. It's not that my senses are broken; they're just tuned differently.",
                "source_attribution": "Composite account from autistic self-advocacy communities",
                "themes": ["sensory intensity", "both challenge and gift", "different tuning"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "autism_social_account",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "experience_domain": "social",
                "description": "I don't lack empathy - I feel others' emotions so intensely it's overwhelming. What I lack is the ability to perform the expected social responses automatically. It's like everyone else has the script to a play memorized, and I'm trying to improvise while also translating from a different language. With other autistic people, suddenly the translation isn't needed. We communicate differently, but we communicate well.",
                "source_attribution": "Composite account from autistic self-advocacy communities",
                "themes": ["intense empathy", "social translation effort", "autistic connection"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "adhd_attention_account",
                "neurotype": NeurodivergentType.ADHD,
                "experience_domain": "attention",
                "description": "My mind is like a browser with 100 tabs open, all playing different videos. I don't have an attention deficit - I have attention abundance that I can't always direct where others want it to go. When something captures me, I can hyperfocus for hours and produce incredible work. The challenge is that my brain decides what's interesting, not my to-do list. It's not about trying harder; it's about working with how my brain actually functions.",
                "source_attribution": "Composite account from ADHD communities",
                "themes": ["attention abundance", "hyperfocus", "working with the brain"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "adhd_time_account",
                "neurotype": NeurodivergentType.ADHD,
                "experience_domain": "time",
                "description": "There are two times: 'now' and 'not now.' Everything that isn't now might as well be equally far away - whether it's tomorrow or next year. Deadlines exist in 'not now' until suddenly they're in 'now' and I'm in crisis mode. Time blindness isn't a choice or laziness; it's a genuine difference in how I perceive time passing. I've learned to use external scaffolding - alarms, visual timers, body doubling - to bridge the gap.",
                "source_attribution": "Composite account from ADHD communities",
                "themes": ["time blindness", "now vs not-now", "external scaffolding"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "synesthesia_discovery_account",
                "neurotype": NeurodivergentType.SYNESTHESIA,
                "experience_domain": "perceptual",
                "description": "I didn't know everyone didn't see music. I thought 'seeing' music was just a common phrase, like 'feeling blue.' I was in my twenties before I realized my experience was unusual. When I play piano, I see cascades of color - C major is bright white, D major warm yellow, minor keys have darker, more complex hues. It's not imagination; it's as real as seeing the keys themselves. Finding out others don't share this felt like learning everyone else is colorblind.",
                "source_attribution": "Composite account from synesthesia communities",
                "themes": ["late discovery", "music-color experience", "reality of perception"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "dyslexia_reading_account",
                "neurotype": NeurodivergentType.DYSLEXIA,
                "experience_domain": "reading",
                "description": "The letters seem to move, or I read 'was' as 'saw' and have to go back. It's exhausting in a way that's hard to explain to fluent readers. But here's what people miss: give me a complex 3D problem to solve, and I can rotate objects in my mind like they're nothing. I see the whole system while others are still looking at parts. Dyslexia isn't just about reading difficulties - it's a different cognitive architecture that struggles with some things and excels at others.",
                "source_attribution": "Composite account from dyslexia communities",
                "themes": ["reading effort", "spatial strength", "different architecture"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "hsp_processing_account",
                "neurotype": NeurodivergentType.HIGHLY_SENSITIVE_PERSON,
                "experience_domain": "processing",
                "description": "I process everything more deeply - it's not a choice, it's how I'm wired. A conversation that seems simple to others, I'm still mulling over hours later, noticing subtleties in tone, word choice, body language. Art moves me to tears. Injustice keeps me up at night. I need quiet time after stimulation not because I'm antisocial, but because my brain has been working overtime on everything. When I'm rested and in the right environment, this depth of processing is my superpower.",
                "source_attribution": "Composite account from HSP communities",
                "themes": ["deep processing", "need for recovery", "depth as strength"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "giftedness_intensity_account",
                "neurotype": NeurodivergentType.GIFTEDNESS,
                "experience_domain": "emotional",
                "description": "Being gifted isn't just about being smart - it's about experiencing everything with more intensity. Dabrowski called them 'overexcitabilities' and I feel them all: the intellectual hunger that's never satisfied, the emotional intensity that makes small slights feel devastating, the imagination that creates whole worlds, the sensory sensitivity that makes some environments unbearable. It's exhausting and wonderful. The hardest part is feeling out of sync with peers who don't share this intensity.",
                "source_attribution": "Composite account from gifted communities",
                "themes": ["intensity", "overexcitabilities", "asynchrony"],
                "is_composite": True,
                "is_anonymized": True,
            },
            {
                "account_id": "twice_exceptional_account",
                "neurotype": NeurodivergentType.TWICE_EXCEPTIONAL,
                "experience_domain": "identity",
                "description": "I'm gifted and ADHD, which means I understand quantum physics but can't remember to eat lunch. My giftedness masked my ADHD for years - I could compensate, so no one noticed I was struggling. My ADHD masked my giftedness - my scattered presentation hid my intellectual capabilities. I fell through every crack in the system, not disabled enough for support, not 'normal' enough for regular expectations. Finding the 2e community was like finding my people for the first time.",
                "source_attribution": "Composite account from twice-exceptional communities",
                "themes": ["masking both ways", "falling through cracks", "finding community"],
                "is_composite": True,
                "is_anonymized": True,
            },
        ]

    def _get_seed_accommodations(self) -> List[Dict[str, Any]]:
        """Return seed accommodation strategies for initialization."""
        return [
            # Autism Accommodations
            {
                "strategy_id": "autism_sensory_environment",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "accommodation_type": AccommodationType.ENVIRONMENTAL,
                "description": "Modify sensory environment to reduce overwhelming input: adjustable lighting (avoiding fluorescents), quiet spaces or noise-cancelling options, consideration of textures and smells in the environment.",
                "effectiveness": "High effectiveness when tailored to individual sensory profile. Reduces stress and enables focus on tasks rather than managing sensory input.",
                "implementation_notes": "Assess individual sensory profile first - some may need reduced input, others may seek specific inputs. Provide options and flexibility rather than one-size-fits-all solutions.",
                "context": "Work, school, home",
                "cost_level": "Low to medium",
            },
            {
                "strategy_id": "autism_communication_clarity",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "accommodation_type": AccommodationType.COMMUNICATION,
                "description": "Provide clear, explicit communication without relying on unstated expectations or implicit meanings. Written instructions to complement verbal. Advance notice of changes.",
                "effectiveness": "Very high. Reduces anxiety from uncertainty and cognitive load from decoding indirect communication.",
                "implementation_notes": "Say what you mean directly. Provide agendas and expectations in advance. Allow processing time. Written follow-ups after meetings.",
                "context": "Work, school, healthcare",
                "cost_level": "Low",
            },
            # ADHD Accommodations
            {
                "strategy_id": "adhd_task_modification",
                "neurotype": NeurodivergentType.ADHD,
                "accommodation_type": AccommodationType.TASK_MODIFICATION,
                "description": "Break large tasks into smaller, time-bound chunks with clear milestones. Build in novelty and movement. Allow flexibility in how and when work is completed.",
                "effectiveness": "High. Smaller chunks maintain engagement and provide dopamine hits from completion. Flexibility allows working with natural rhythms.",
                "implementation_notes": "Focus on outcomes rather than process. Allow movement breaks. Consider whether in-person presence is always necessary. Use body doubling when helpful.",
                "context": "Work, school",
                "cost_level": "Low",
            },
            {
                "strategy_id": "adhd_external_scaffolding",
                "neurotype": NeurodivergentType.ADHD,
                "accommodation_type": AccommodationType.ASSISTIVE_TECHNOLOGY,
                "description": "External tools for time management and organization: visual timers, calendar apps with reminders, project management software, noise-cancelling headphones, fidget tools.",
                "effectiveness": "High when tools match individual preferences. External scaffolding replaces executive functions that work differently.",
                "implementation_notes": "Individual must find tools that work for them - there's no universal solution. Regular review and adjustment needed. Tools should be normalized, not stigmatized.",
                "context": "Work, school, home",
                "cost_level": "Low to medium",
            },
            # Dyslexia Accommodations
            {
                "strategy_id": "dyslexia_multimodal_access",
                "neurotype": NeurodivergentType.DYSLEXIA,
                "accommodation_type": AccommodationType.ASSISTIVE_TECHNOLOGY,
                "description": "Provide information in multiple formats: audio versions of text, text-to-speech software, video explanations, visual diagrams alongside written content.",
                "effectiveness": "Very high. Bypasses the specific challenge while leveraging other processing strengths.",
                "implementation_notes": "Make audio/visual options standard, not special requests. Use dyslexia-friendly fonts and formatting. Allow oral responses as alternatives to written.",
                "context": "Work, school",
                "cost_level": "Low to medium",
            },
            {
                "strategy_id": "dyslexia_extended_time",
                "neurotype": NeurodivergentType.DYSLEXIA,
                "accommodation_type": AccommodationType.TASK_MODIFICATION,
                "description": "Extended time for reading-intensive tasks and assessments. This accommodation recognizes that reading takes more effort, not that the person is less capable.",
                "effectiveness": "High for timed assessments. Allows demonstration of actual knowledge without time pressure disadvantaging.",
                "implementation_notes": "Typically 50-100% additional time. Should be normalized and private. Focus on whether assessment measures knowledge vs. reading speed.",
                "context": "School, certification exams",
                "cost_level": "Low",
            },
            # Sensory Processing Accommodations
            {
                "strategy_id": "hsp_recovery_space",
                "neurotype": NeurodivergentType.HIGHLY_SENSITIVE_PERSON,
                "accommodation_type": AccommodationType.ENVIRONMENTAL,
                "description": "Access to quiet, low-stimulation spaces for recovery during the day. Ability to take breaks after intense meetings or experiences. Control over workspace sensory environment.",
                "effectiveness": "High. Prevents cumulative overwhelm and allows sustained productivity rather than burnout.",
                "implementation_notes": "Frame as productivity enhancement, not special treatment. Recovery time is how HSPs maintain their valuable depth of processing.",
                "context": "Work",
                "cost_level": "Low",
            },
            # Social Support Accommodations
            {
                "strategy_id": "general_peer_mentorship",
                "neurotype": NeurodivergentType.AUTISM_SPECTRUM,
                "accommodation_type": AccommodationType.SOCIAL_SUPPORT,
                "description": "Connection with neurodivergent mentors and peers who share similar experiences. May include formal mentorship programs, employee resource groups, or informal connections.",
                "effectiveness": "High for wellbeing and professional development. Reduces isolation and provides practical strategies from those with lived experience.",
                "implementation_notes": "Facilitate but don't force connections. Ensure mentors are supported too. Value neurodivergent expertise in neurodivergent support.",
                "context": "Work, school",
                "cost_level": "Low to medium",
            },
            # Giftedness Accommodations
            {
                "strategy_id": "giftedness_challenge",
                "neurotype": NeurodivergentType.GIFTEDNESS,
                "accommodation_type": AccommodationType.TASK_MODIFICATION,
                "description": "Provide appropriate intellectual challenge rather than more of the same work. Allow acceleration, compacting, or independent projects aligned with abilities and interests.",
                "effectiveness": "High for engagement and wellbeing. Under-challenge is as problematic as over-challenge for gifted individuals.",
                "implementation_notes": "Focus on depth and complexity, not just quantity. Allow exploration of interests. Connect with intellectual peers regardless of age.",
                "context": "School, work",
                "cost_level": "Low",
            },
            # OCD Accommodations
            {
                "strategy_id": "ocd_flexibility",
                "neurotype": NeurodivergentType.OCD,
                "accommodation_type": AccommodationType.TASK_MODIFICATION,
                "description": "Flexibility around rituals and compulsions when they don't affect work quality. Understanding that some behaviors serve an anxiety-management function. Patience without judgment.",
                "effectiveness": "Moderate to high. Reduces shame and allows energy for work rather than hiding symptoms.",
                "implementation_notes": "Don't draw attention to rituals. Focus on outcomes not process. Reduce environmental triggers where possible. Support access to evidence-based treatment if desired.",
                "context": "Work",
                "cost_level": "Low",
            },
        ]

    async def initialize_seed_profiles(self) -> int:
        """Initialize with seed profiles."""
        seed_profiles = self._get_seed_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = NeurodivergentProfile(
                profile_id=profile_data["profile_id"],
                neurotype=profile_data["neurotype"],
                cognitive_strengths=profile_data.get("cognitive_strengths", []),
                processing_styles=profile_data.get("processing_styles", []),
                sensory_profile=profile_data.get("sensory_profile"),
                support_needs=profile_data.get("support_needs", []),
                self_description=profile_data.get("self_description", ""),
                prevalence=profile_data.get("prevalence"),
                key_researchers=profile_data.get("key_researchers", []),
                common_co_occurrences=profile_data.get("common_co_occurrences", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed profiles")
        return count

    async def initialize_seed_synesthesia(self) -> int:
        """Initialize with seed synesthesia profiles."""
        seed_profiles = self._get_seed_synesthesia_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = SynesthesiaProfile(
                profile_id=profile_data["profile_id"],
                synesthesia_types=profile_data["synesthesia_types"],
                consistency_score=profile_data.get("consistency_score"),
                inducer_concurrent_pairs=profile_data.get("inducer_concurrent_pairs", []),
                developmental_history=profile_data.get("developmental_history", ""),
                associated_benefits=profile_data.get("associated_benefits", []),
                famous_synesthetes=profile_data.get("famous_synesthetes", []),
                neural_basis_notes=profile_data.get("neural_basis_notes", ""),
                prevalence=profile_data.get("prevalence"),
            )
            await self.add_synesthesia_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed synesthesia profiles")
        return count

    async def initialize_seed_strength_evidence(self) -> int:
        """Initialize with seed cognitive strength evidence."""
        seed_evidence = self._get_seed_strength_evidence()
        count = 0

        for evidence_data in seed_evidence:
            evidence = CognitiveStrengthEvidence(
                evidence_id=evidence_data["evidence_id"],
                neurotype=evidence_data["neurotype"],
                strength=evidence_data["strength"],
                research_basis=evidence_data.get("research_basis", ""),
                real_world_examples=evidence_data.get("real_world_examples", []),
                notable_individuals=evidence_data.get("notable_individuals", []),
                applications=evidence_data.get("applications", []),
            )
            await self.add_strength_evidence(evidence)
            count += 1

        logger.info(f"Initialized {count} seed strength evidence entries")
        return count

    async def initialize_seed_first_person_accounts(self) -> int:
        """Initialize with seed first-person accounts."""
        seed_accounts = self._get_seed_first_person_accounts()
        count = 0

        for account_data in seed_accounts:
            account = FirstPersonAccount(
                account_id=account_data["account_id"],
                neurotype=account_data["neurotype"],
                experience_domain=account_data["experience_domain"],
                description=account_data["description"],
                source_attribution=account_data["source_attribution"],
                themes=account_data.get("themes", []),
                is_composite=account_data.get("is_composite", False),
                is_anonymized=account_data.get("is_anonymized", True),
            )
            await self.add_first_person_account(account)
            count += 1

        logger.info(f"Initialized {count} seed first-person accounts")
        return count

    async def initialize_seed_accommodations(self) -> int:
        """Initialize with seed accommodation strategies."""
        seed_accommodations = self._get_seed_accommodations()
        count = 0

        for acc_data in seed_accommodations:
            strategy = AccommodationStrategy(
                strategy_id=acc_data["strategy_id"],
                neurotype=acc_data["neurotype"],
                accommodation_type=acc_data["accommodation_type"],
                description=acc_data["description"],
                effectiveness=acc_data.get("effectiveness", ""),
                implementation_notes=acc_data.get("implementation_notes", ""),
                context=acc_data.get("context", ""),
                cost_level=acc_data.get("cost_level", ""),
            )
            await self.add_accommodation(strategy)
            count += 1

        logger.info(f"Initialized {count} seed accommodation strategies")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_profiles()
        synesthesia_count = await self.initialize_seed_synesthesia()
        strength_evidence_count = await self.initialize_seed_strength_evidence()
        accounts_count = await self.initialize_seed_first_person_accounts()
        accommodations_count = await self.initialize_seed_accommodations()

        total = (
            profiles_count +
            synesthesia_count +
            strength_evidence_count +
            accounts_count +
            accommodations_count
        )

        return {
            "profiles": profiles_count,
            "synesthesia_profiles": synesthesia_count,
            "strength_evidence": strength_evidence_count,
            "first_person_accounts": accounts_count,
            "accommodations": accommodations_count,
            "total": total,
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "NeurodivergentType",
    "CognitiveStrength",
    "SynesthesiaType",
    "ProcessingStyle",
    "SensoryProfile",
    "AccommodationType",
    "MaturityLevel",
    # Dataclasses
    "NeurodivergentProfile",
    "SynesthesiaProfile",
    "CognitiveStrengthEvidence",
    "FirstPersonAccount",
    "AccommodationStrategy",
    "NeurodivergentConsciousnessMaturityState",
    # Interface
    "NeurodivergentConsciousnessInterface",
]
