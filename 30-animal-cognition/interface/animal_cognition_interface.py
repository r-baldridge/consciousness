#!/usr/bin/env python3
"""
Animal Cognition & Ethology Interface

Form 30: The comprehensive interface for animal cognition, behavior, and
consciousness research. Integrates Western scientific findings with indigenous
perspectives on animal intelligence and human-animal relationships.

This form bridges comparative psychology, cognitive ethology, and traditional
ecological knowledge to provide a nuanced understanding of animal minds.
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

class TaxonomicGroup(Enum):
    """
    Major taxonomic groups covered in animal cognition research.

    Organized by broad categories, not strict taxonomy.
    """

    # === MAMMALS ===
    GREAT_APES = "great_apes"  # Chimps, bonobos, gorillas, orangutans
    CETACEANS = "cetaceans"  # Dolphins, whales, orcas
    ELEPHANTS = "elephants"  # African, Asian
    CANIDS = "canids"  # Wolves, dogs, foxes
    FELIDS = "felids"  # Cats, lions, tigers
    PRIMATES_OTHER = "primates_other"  # Capuchins, macaques, lemurs
    MARINE_MAMMALS_OTHER = "marine_mammals_other"  # Seals, sea lions
    RODENTS = "rodents"  # Rats, mice (research subjects)
    UNGULATES = "ungulates"  # Horses, pigs, deer
    BATS = "bats"  # Chiroptera

    # === BIRDS ===
    CORVIDS = "corvids"  # Crows, ravens, jays, magpies
    PARROTS = "parrots"  # African Grey, Kea, macaws
    RAPTORS = "raptors"  # Eagles, owls, hawks
    SONGBIRDS = "songbirds"  # Vocal learners
    WATERFOWL = "waterfowl"  # Geese, swans

    # === OTHER VERTEBRATES ===
    REPTILES = "reptiles"  # Monitors, crocodilians, turtles
    AMPHIBIANS = "amphibians"  # Complex-behaving frogs
    FISH = "fish"  # Cleaner wrasse, rays, cichlids

    # === INVERTEBRATES ===
    CEPHALOPODS = "cephalopods"  # Octopuses, cuttlefish, squid
    SOCIAL_INSECTS = "social_insects"  # Bees, ants, termites
    ARACHNIDS = "arachnids"  # Jumping spiders


class CognitionDomain(Enum):
    """Domains of cognitive ability assessed across species."""

    # === MEMORY SYSTEMS ===
    EPISODIC_MEMORY = "episodic_memory"  # What-where-when
    WORKING_MEMORY = "working_memory"  # Capacity and duration
    SPATIAL_COGNITION = "spatial_cognition"  # Navigation, mental maps
    LONG_TERM_MEMORY = "long_term_memory"  # Retention over time

    # === LEARNING & REASONING ===
    SOCIAL_LEARNING = "social_learning"  # Learning from conspecifics
    OBSERVATIONAL_LEARNING = "observational_learning"  # Imitation
    CAUSAL_REASONING = "causal_reasoning"  # Understanding cause-effect
    NUMERICAL_COGNITION = "numerical_cognition"  # Quantity discrimination
    PROBLEM_SOLVING = "problem_solving"  # Novel problem solutions
    PLANNING = "planning"  # Future-directed behavior
    INSIGHT = "insight"  # Sudden solution without trial-and-error

    # === TOOL USE ===
    TOOL_USE = "tool_use"  # Using objects as tools
    TOOL_MANUFACTURE = "tool_manufacture"  # Making tools

    # === SOCIAL COGNITION ===
    THEORY_OF_MIND = "theory_of_mind"  # Understanding others' mental states
    PERSPECTIVE_TAKING = "perspective_taking"  # Visual/knowledge perspective
    COOPERATION = "cooperation"  # Coordinated action
    RECIPROCITY = "reciprocity"  # Tit-for-tat, mutual aid
    DECEPTION = "deception"  # Tactical deception
    FAIRNESS_SENSITIVITY = "fairness_sensitivity"  # Inequity aversion
    EMPATHY = "empathy"  # Emotional sharing
    EMOTIONAL_CONTAGION = "emotional_contagion"  # Catching emotions

    # === SELF-AWARENESS ===
    SELF_RECOGNITION = "self_recognition"  # Body self-recognition
    MIRROR_TEST = "mirror_test"  # Mark test performance
    METACOGNITION = "metacognition"  # Knowing what you know
    SELF_AGENCY = "self_agency"  # Understanding own actions

    # === COMMUNICATION ===
    REFERENTIAL_SIGNALS = "referential_signals"  # Signals referring to objects
    SYNTAX = "syntax"  # Combinatorial signals
    LANGUAGE_COMPREHENSION = "language_comprehension"  # Understanding human language
    VOCAL_LEARNING = "vocal_learning"  # Acquiring new vocalizations

    # === EMOTIONAL/OTHER ===
    EMOTIONAL_PROCESSING = "emotional_processing"  # Emotional expressions
    GRIEF_MOURNING = "grief_mourning"  # Death-related behavior
    PLAY_BEHAVIOR = "play_behavior"  # Social and object play
    CULTURAL_TRANSMISSION = "cultural_transmission"  # Cross-generational learning


class ConsciousnessIndicator(Enum):
    """Types of evidence for consciousness in animals."""
    BEHAVIORAL = "behavioral"  # Observable behavior
    NEUROANATOMICAL = "neuroanatomical"  # Brain structures
    NEUROPHYSIOLOGICAL = "neurophysiological"  # Neural activity
    PHARMACOLOGICAL = "pharmacological"  # Drug responses
    SELF_REPORT_PROXY = "self_report_proxy"  # Trained language use
    INDIGENOUS_OBSERVATION = "indigenous_observation"  # Traditional knowledge


class ResearchParadigm(Enum):
    """Research approaches in animal cognition."""
    COMPARATIVE_PSYCHOLOGY = "comparative_psychology"  # Lab comparisons
    COGNITIVE_ETHOLOGY = "cognitive_ethology"  # Natural behavior focus
    FIELD_OBSERVATION = "field_observation"  # Wild populations
    LABORATORY_EXPERIMENTAL = "laboratory_experimental"  # Controlled experiments
    NEUROSCIENCE = "neuroscience"  # Brain-based approaches
    INDIGENOUS_KNOWLEDGE = "indigenous_knowledge"  # Traditional perspectives


class EvidenceStrength(Enum):
    """Strength of evidence for cognitive claims."""
    ANECDOTAL = "anecdotal"  # Single observations
    PRELIMINARY = "preliminary"  # Initial studies
    MODERATE = "moderate"  # Multiple studies
    STRONG = "strong"  # Replicated, peer-reviewed
    CONSENSUS = "consensus"  # Field-wide agreement


class MaturityLevel(Enum):
    """Depth of knowledge coverage."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SpeciesCognitionProfile:
    """
    Comprehensive cognitive profile for a species or species group.

    This is the primary data structure for Form 30, containing all
    known information about a species' cognitive abilities.
    """
    species_id: str
    common_name: str
    scientific_name: str
    taxonomic_group: TaxonomicGroup
    cognition_domains: Dict[CognitionDomain, float] = field(default_factory=dict)  # 0-1 evidence strength
    consciousness_indicators: Dict[ConsciousnessIndicator, List[str]] = field(default_factory=dict)
    key_studies: List[str] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)
    notable_individuals: List[str] = field(default_factory=list)  # e.g., Koko, Alex, Washoe
    indigenous_perspectives: List[str] = field(default_factory=list)  # Links to Form 29
    brain_features: List[str] = field(default_factory=list)  # Relevant neuroanatomy
    social_structure: Optional[str] = None
    communication_system: Optional[str] = None
    tool_use_description: Optional[str] = None
    cultural_variation: Optional[str] = None
    conservation_status: Optional[str] = None
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Species: {self.common_name} ({self.scientific_name})",
            f"Group: {self.taxonomic_group.value}",
            f"Key domains: {', '.join(d.value for d in self.cognition_domains.keys())}",
        ]
        if self.notable_individuals:
            parts.append(f"Notable: {', '.join(self.notable_individuals)}")
        return " | ".join(parts)


@dataclass
class AnimalBehaviorInsight:
    """
    Represents a specific behavioral observation or experimental finding.

    These are the building blocks of species profiles - individual
    pieces of evidence for cognitive abilities.
    """
    insight_id: str
    species_id: str
    domain: CognitionDomain
    description: str
    evidence_type: ConsciousnessIndicator
    research_paradigm: ResearchParadigm
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    source_citation: str = ""
    year_published: Optional[int] = None
    researcher: Optional[str] = None
    location: Optional[str] = None  # Lab, field site, etc.
    replication_status: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CrossSpeciesSynthesis:
    """
    Comparative analysis across multiple species.

    These represent synthesized understanding of how cognitive
    abilities vary or converge across different species.
    """
    synthesis_id: str
    topic: str  # e.g., "Mirror self-recognition across taxa"
    species_ids: List[str]
    domain: CognitionDomain
    description: str
    key_findings: List[str] = field(default_factory=list)
    evolutionary_implications: Optional[str] = None
    methodological_notes: Optional[str] = None
    open_questions: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class IndigenousAnimalKnowledge:
    """
    Links to Form 29's IndigenousAnimalWisdom.

    This bridges scientific findings with traditional knowledge,
    noting where they converge or provide unique insights.
    """
    knowledge_id: str
    species_id: str
    folk_wisdom_id: str  # Reference to Form 29
    behavioral_claim: str
    scientific_corroboration: Optional[str] = None
    unique_indigenous_insight: str = ""
    cultural_context: Optional[str] = None
    ecological_relevance: Optional[str] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ConsciousnessTheoryApplication:
    """
    How consciousness theories apply to a species.

    Maps theoretical frameworks to evidence for/against
    consciousness in specific species.
    """
    application_id: str
    species_id: str
    theory_name: str  # e.g., "Global Workspace Theory"
    relevant_evidence: List[str] = field(default_factory=list)
    theory_prediction: Optional[str] = None
    evidence_alignment: Optional[str] = None  # supports, contradicts, neutral
    discussion: Optional[str] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AnimalCognitionMaturityState:
    """Tracks the maturity of animal cognition knowledge."""
    overall_maturity: float = 0.0
    taxonomic_coverage: Dict[str, float] = field(default_factory=dict)
    species_count: int = 0
    insight_count: int = 0
    synthesis_count: int = 0
    indigenous_links: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class AnimalCognitionInterface:
    """
    Main interface for Form 30: Animal Cognition & Ethology.

    Provides methods for storing, retrieving, and querying species
    cognition profiles, behavioral insights, and cross-species syntheses.
    """

    FORM_ID = "30-animal-cognition"
    FORM_NAME = "Animal Cognition & Ethology"

    def __init__(self):
        """Initialize the Animal Cognition Interface."""
        # Knowledge indexes
        self.species_index: Dict[str, SpeciesCognitionProfile] = {}
        self.insight_index: Dict[str, AnimalBehaviorInsight] = {}
        self.synthesis_index: Dict[str, CrossSpeciesSynthesis] = {}
        self.indigenous_knowledge_index: Dict[str, IndigenousAnimalKnowledge] = {}
        self.theory_application_index: Dict[str, ConsciousnessTheoryApplication] = {}

        # Cross-reference indexes
        self.taxonomic_index: Dict[TaxonomicGroup, List[str]] = {}
        self.domain_index: Dict[CognitionDomain, List[str]] = {}

        # Maturity tracking
        self.maturity_state = AnimalCognitionMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize taxonomic index
        for group in TaxonomicGroup:
            self.taxonomic_index[group] = []

        # Initialize domain index
        for domain in CognitionDomain:
            self.domain_index[domain] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # SPECIES PROFILE METHODS
    # ========================================================================

    async def add_species_profile(self, profile: SpeciesCognitionProfile) -> None:
        """Add a species cognition profile to the index."""
        self.species_index[profile.species_id] = profile

        # Update taxonomic index
        if profile.taxonomic_group in self.taxonomic_index:
            self.taxonomic_index[profile.taxonomic_group].append(profile.species_id)

        # Update domain index
        for domain in profile.cognition_domains.keys():
            if domain in self.domain_index:
                self.domain_index[domain].append(profile.species_id)

        # Update maturity
        self.maturity_state.species_count = len(self.species_index)
        await self._update_maturity()

    async def get_species_profile(self, species_id: str) -> Optional[SpeciesCognitionProfile]:
        """Retrieve a species profile by ID."""
        return self.species_index.get(species_id)

    async def query_by_taxonomic_group(
        self,
        group: TaxonomicGroup,
        limit: int = 10
    ) -> List[SpeciesCognitionProfile]:
        """Query species by taxonomic group."""
        species_ids = self.taxonomic_index.get(group, [])[:limit]
        return [
            self.species_index[sid]
            for sid in species_ids
            if sid in self.species_index
        ]

    async def query_by_cognition_domain(
        self,
        domain: CognitionDomain,
        min_evidence: float = 0.5,
        limit: int = 10
    ) -> List[SpeciesCognitionProfile]:
        """Query species showing evidence in a cognition domain."""
        results = []
        for species in self.species_index.values():
            if domain in species.cognition_domains:
                if species.cognition_domains[domain] >= min_evidence:
                    results.append(species)
        return results[:limit]

    # ========================================================================
    # INSIGHT METHODS
    # ========================================================================

    async def add_insight(self, insight: AnimalBehaviorInsight) -> None:
        """Add a behavioral insight to the index."""
        self.insight_index[insight.insight_id] = insight

        # Update domain index
        if insight.domain in self.domain_index:
            self.domain_index[insight.domain].append(insight.insight_id)

        # Update maturity
        self.maturity_state.insight_count = len(self.insight_index)
        await self._update_maturity()

    async def get_insight(self, insight_id: str) -> Optional[AnimalBehaviorInsight]:
        """Retrieve an insight by ID."""
        return self.insight_index.get(insight_id)

    async def query_insights_by_species(
        self,
        species_id: str
    ) -> List[AnimalBehaviorInsight]:
        """Get all insights for a species."""
        return [
            i for i in self.insight_index.values()
            if i.species_id == species_id
        ]

    # ========================================================================
    # SYNTHESIS METHODS
    # ========================================================================

    async def add_synthesis(self, synthesis: CrossSpeciesSynthesis) -> None:
        """Add a cross-species synthesis."""
        self.synthesis_index[synthesis.synthesis_id] = synthesis
        self.maturity_state.synthesis_count = len(self.synthesis_index)
        await self._update_maturity()

    async def get_synthesis(self, synthesis_id: str) -> Optional[CrossSpeciesSynthesis]:
        """Retrieve a synthesis by ID."""
        return self.synthesis_index.get(synthesis_id)

    # ========================================================================
    # INDIGENOUS KNOWLEDGE METHODS
    # ========================================================================

    async def add_indigenous_knowledge(self, knowledge: IndigenousAnimalKnowledge) -> None:
        """Add indigenous animal knowledge link."""
        self.indigenous_knowledge_index[knowledge.knowledge_id] = knowledge
        self.maturity_state.indigenous_links = len(self.indigenous_knowledge_index)
        await self._update_maturity()

    async def get_indigenous_knowledge(
        self,
        knowledge_id: str
    ) -> Optional[IndigenousAnimalKnowledge]:
        """Retrieve indigenous knowledge by ID."""
        return self.indigenous_knowledge_index.get(knowledge_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.species_count +
            self.maturity_state.insight_count +
            self.maturity_state.synthesis_count +
            self.maturity_state.indigenous_links
        )

        # Simple maturity calculation
        target_items = 300
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update taxonomic coverage
        for group in TaxonomicGroup:
            count = len(self.taxonomic_index.get(group, []))
            target_per_group = 5
            self.maturity_state.taxonomic_coverage[group.value] = min(
                1.0, count / target_per_group
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> AnimalCognitionMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_species(self) -> List[Dict[str, Any]]:
        """Return seed species profiles for initialization."""
        return [
            # Great Apes
            {
                "species_id": "pan_troglodytes",
                "common_name": "Chimpanzee",
                "scientific_name": "Pan troglodytes",
                "taxonomic_group": TaxonomicGroup.GREAT_APES,
                "cognition_domains": {
                    CognitionDomain.TOOL_USE: 0.95,
                    CognitionDomain.TOOL_MANUFACTURE: 0.9,
                    CognitionDomain.SOCIAL_LEARNING: 0.9,
                    CognitionDomain.THEORY_OF_MIND: 0.7,
                    CognitionDomain.SELF_RECOGNITION: 0.95,
                    CognitionDomain.COOPERATION: 0.85,
                    CognitionDomain.DECEPTION: 0.8,
                    CognitionDomain.CULTURAL_TRANSMISSION: 0.9,
                    CognitionDomain.PLANNING: 0.75,
                },
                "notable_individuals": ["Washoe", "Kanzi", "Ai"],
                "key_researchers": ["Jane Goodall", "Frans de Waal", "Tetsuro Matsuzawa"],
                "key_studies": ["Termite fishing", "Nut cracking", "Sign language studies"],
            },
            {
                "species_id": "pan_paniscus",
                "common_name": "Bonobo",
                "scientific_name": "Pan paniscus",
                "taxonomic_group": TaxonomicGroup.GREAT_APES,
                "cognition_domains": {
                    CognitionDomain.COOPERATION: 0.9,
                    CognitionDomain.EMPATHY: 0.9,
                    CognitionDomain.SOCIAL_LEARNING: 0.85,
                    CognitionDomain.SELF_RECOGNITION: 0.95,
                    CognitionDomain.LANGUAGE_COMPREHENSION: 0.8,
                    CognitionDomain.TOOL_USE: 0.7,
                },
                "notable_individuals": ["Kanzi", "Panbanisha"],
                "key_researchers": ["Sue Savage-Rumbaugh", "Frans de Waal"],
                "key_studies": ["Lexigram communication", "Cooperation studies", "Empathy research"],
            },

            # Cetaceans
            {
                "species_id": "tursiops_truncatus",
                "common_name": "Bottlenose Dolphin",
                "scientific_name": "Tursiops truncatus",
                "taxonomic_group": TaxonomicGroup.CETACEANS,
                "cognition_domains": {
                    CognitionDomain.SELF_RECOGNITION: 0.9,
                    CognitionDomain.SOCIAL_LEARNING: 0.9,
                    CognitionDomain.COOPERATION: 0.85,
                    CognitionDomain.LANGUAGE_COMPREHENSION: 0.75,
                    CognitionDomain.PROBLEM_SOLVING: 0.8,
                    CognitionDomain.REFERENTIAL_SIGNALS: 0.8,
                    CognitionDomain.CULTURAL_TRANSMISSION: 0.85,
                },
                "notable_individuals": ["Kelly (bubble ring maker)", "Dolphins at Kewalo Basin"],
                "key_researchers": ["Louis Herman", "Diana Reiss", "Lori Marino"],
                "key_studies": ["Mirror self-recognition", "Artificial language comprehension", "Signature whistles"],
            },
            {
                "species_id": "orcinus_orca",
                "common_name": "Orca (Killer Whale)",
                "scientific_name": "Orcinus orca",
                "taxonomic_group": TaxonomicGroup.CETACEANS,
                "cognition_domains": {
                    CognitionDomain.CULTURAL_TRANSMISSION: 0.95,
                    CognitionDomain.COOPERATION: 0.95,
                    CognitionDomain.SOCIAL_LEARNING: 0.9,
                    CognitionDomain.PROBLEM_SOLVING: 0.85,
                    CognitionDomain.VOCAL_LEARNING: 0.9,
                    CognitionDomain.GRIEF_MOURNING: 0.8,
                },
                "key_researchers": ["John Ford", "Hal Whitehead", "Ken Balcomb"],
                "key_studies": ["Cultural hunting traditions", "Matrilineal knowledge transfer", "Dialect studies"],
                "cultural_variation": "Different ecotypes have distinct hunting techniques, dialects, and social customs passed through matrilines",
            },

            # Elephants
            {
                "species_id": "loxodonta_africana",
                "common_name": "African Elephant",
                "scientific_name": "Loxodonta africana",
                "taxonomic_group": TaxonomicGroup.ELEPHANTS,
                "cognition_domains": {
                    CognitionDomain.SELF_RECOGNITION: 0.85,
                    CognitionDomain.EMPATHY: 0.9,
                    CognitionDomain.GRIEF_MOURNING: 0.95,
                    CognitionDomain.COOPERATION: 0.85,
                    CognitionDomain.LONG_TERM_MEMORY: 0.95,
                    CognitionDomain.SOCIAL_LEARNING: 0.85,
                    CognitionDomain.PROBLEM_SOLVING: 0.8,
                },
                "notable_individuals": ["Happy (Bronx Zoo - mirror test)"],
                "key_researchers": ["Cynthia Moss", "Joyce Poole", "Joshua Plotnik"],
                "key_studies": ["Mirror self-recognition", "Grief behaviors at remains", "Cooperative problem solving"],
            },

            # Corvids
            {
                "species_id": "corvus_corax",
                "common_name": "Common Raven",
                "scientific_name": "Corvus corax",
                "taxonomic_group": TaxonomicGroup.CORVIDS,
                "cognition_domains": {
                    CognitionDomain.PLANNING: 0.85,
                    CognitionDomain.TOOL_USE: 0.7,
                    CognitionDomain.THEORY_OF_MIND: 0.75,
                    CognitionDomain.CAUSAL_REASONING: 0.8,
                    CognitionDomain.SOCIAL_LEARNING: 0.8,
                    CognitionDomain.DECEPTION: 0.75,
                    CognitionDomain.PLAY_BEHAVIOR: 0.85,
                },
                "key_researchers": ["Bernd Heinrich", "Thomas Bugnyar"],
                "key_studies": ["Gaze following", "Future planning", "Social manipulation"],
            },
            {
                "species_id": "corvus_moneduloides",
                "common_name": "New Caledonian Crow",
                "scientific_name": "Corvus moneduloides",
                "taxonomic_group": TaxonomicGroup.CORVIDS,
                "cognition_domains": {
                    CognitionDomain.TOOL_USE: 0.95,
                    CognitionDomain.TOOL_MANUFACTURE: 0.95,
                    CognitionDomain.CAUSAL_REASONING: 0.85,
                    CognitionDomain.PROBLEM_SOLVING: 0.9,
                    CognitionDomain.PLANNING: 0.8,
                    CognitionDomain.INSIGHT: 0.75,
                },
                "key_researchers": ["Gavin Hunt", "Russell Gray", "Alex Taylor"],
                "key_studies": ["Hook tool manufacture", "Metatool use", "Aesop's fable paradigm"],
                "tool_use_description": "Creates hooked stick tools from twigs, manufactures stepped-cut pandanus tools",
            },

            # Parrots
            {
                "species_id": "psittacus_erithacus",
                "common_name": "African Grey Parrot",
                "scientific_name": "Psittacus erithacus",
                "taxonomic_group": TaxonomicGroup.PARROTS,
                "cognition_domains": {
                    CognitionDomain.LANGUAGE_COMPREHENSION: 0.9,
                    CognitionDomain.NUMERICAL_COGNITION: 0.85,
                    CognitionDomain.CAUSAL_REASONING: 0.75,
                    CognitionDomain.SELF_AGENCY: 0.7,
                    CognitionDomain.SOCIAL_LEARNING: 0.8,
                    CognitionDomain.VOCAL_LEARNING: 0.95,
                },
                "notable_individuals": ["Alex", "Griffin"],
                "key_researchers": ["Irene Pepperberg"],
                "key_studies": ["Alex studies - labels, categories, zero concept", "Model/rival training"],
            },

            # Cephalopods
            {
                "species_id": "octopus_vulgaris",
                "common_name": "Common Octopus",
                "scientific_name": "Octopus vulgaris",
                "taxonomic_group": TaxonomicGroup.CEPHALOPODS,
                "cognition_domains": {
                    CognitionDomain.PROBLEM_SOLVING: 0.85,
                    CognitionDomain.OBSERVATIONAL_LEARNING: 0.75,
                    CognitionDomain.TOOL_USE: 0.7,
                    CognitionDomain.SPATIAL_COGNITION: 0.8,
                    CognitionDomain.PLAY_BEHAVIOR: 0.65,
                },
                "key_researchers": ["Jennifer Mather", "Peter Godfrey-Smith"],
                "key_studies": ["Jar opening", "Coconut shell use", "Play with objects"],
                "brain_features": ["Distributed nervous system", "500 million neurons", "Independent arm control"],
            },

            # Canids
            {
                "species_id": "canis_lupus_familiaris",
                "common_name": "Domestic Dog",
                "scientific_name": "Canis lupus familiaris",
                "taxonomic_group": TaxonomicGroup.CANIDS,
                "cognition_domains": {
                    CognitionDomain.SOCIAL_LEARNING: 0.85,
                    CognitionDomain.LANGUAGE_COMPREHENSION: 0.8,
                    CognitionDomain.EMOTIONAL_PROCESSING: 0.85,
                    CognitionDomain.EMPATHY: 0.75,
                    CognitionDomain.THEORY_OF_MIND: 0.6,
                    CognitionDomain.REFERENTIAL_SIGNALS: 0.7,
                },
                "notable_individuals": ["Chaser (1000+ words)", "Rico"],
                "key_researchers": ["Brian Hare", "Ádám Miklósi", "Juliane Kaminski"],
                "key_studies": ["Word learning", "Human gesture following", "Emotional responsiveness"],
            },

            # Social Insects
            {
                "species_id": "apis_mellifera",
                "common_name": "Western Honey Bee",
                "scientific_name": "Apis mellifera",
                "taxonomic_group": TaxonomicGroup.SOCIAL_INSECTS,
                "cognition_domains": {
                    CognitionDomain.SPATIAL_COGNITION: 0.8,
                    CognitionDomain.SOCIAL_LEARNING: 0.7,
                    CognitionDomain.REFERENTIAL_SIGNALS: 0.85,
                    CognitionDomain.NUMERICAL_COGNITION: 0.7,
                    CognitionDomain.PROBLEM_SOLVING: 0.6,
                },
                "key_researchers": ["Karl von Frisch", "Lars Chittka"],
                "key_studies": ["Waggle dance communication", "Cognitive map formation", "Numerical discrimination"],
                "communication_system": "Waggle dance encodes direction, distance, and quality of resources",
            },
        ]

    async def initialize_seed_species(self) -> int:
        """Initialize with seed species profiles."""
        seed_species = self._get_seed_species()
        count = 0

        for species_data in seed_species:
            profile = SpeciesCognitionProfile(
                species_id=species_data["species_id"],
                common_name=species_data["common_name"],
                scientific_name=species_data["scientific_name"],
                taxonomic_group=species_data["taxonomic_group"],
                cognition_domains=species_data.get("cognition_domains", {}),
                notable_individuals=species_data.get("notable_individuals", []),
                key_researchers=species_data.get("key_researchers", []),
                key_studies=species_data.get("key_studies", []),
                tool_use_description=species_data.get("tool_use_description"),
                cultural_variation=species_data.get("cultural_variation"),
                communication_system=species_data.get("communication_system"),
                brain_features=species_data.get("brain_features", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_species_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed species profiles")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        species_count = await self.initialize_seed_species()

        return {
            "species": species_count,
            "total": species_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "TaxonomicGroup",
    "CognitionDomain",
    "ConsciousnessIndicator",
    "ResearchParadigm",
    "EvidenceStrength",
    "MaturityLevel",
    # Dataclasses
    "SpeciesCognitionProfile",
    "AnimalBehaviorInsight",
    "CrossSpeciesSynthesis",
    "IndigenousAnimalKnowledge",
    "ConsciousnessTheoryApplication",
    "AnimalCognitionMaturityState",
    # Interface
    "AnimalCognitionInterface",
]
