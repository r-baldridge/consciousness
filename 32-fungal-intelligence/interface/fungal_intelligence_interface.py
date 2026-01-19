#!/usr/bin/env python3
"""
Fungal Networks & Mycorrhizal Intelligence Interface

Form 32: The comprehensive interface for fungal intelligence, mycorrhizal
networks, slime mold cognition, and indigenous fungal wisdom traditions.
This form explores non-neural intelligence and distributed computation
exhibited by fungi across diverse ecological and cultural contexts.

Ethical Principles:
- Indigenous knowledge attribution
- Conservation awareness
- Respect for traditional practices
- Recognition of fungal ecological importance
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

class FungalIntelligenceDomain(Enum):
    """
    Domains of fungal intelligence and cognition.

    These represent the various ways fungi demonstrate intelligent behavior
    without centralized nervous systems.
    """

    # === NETWORK CAPABILITIES ===
    NETWORK_OPTIMIZATION = "network_optimization"  # Efficient topology creation
    RESOURCE_DISTRIBUTION = "resource_distribution"  # Allocation decisions
    NUTRIENT_TRANSFER = "nutrient_transfer"  # Carbon, nitrogen, phosphorus sharing

    # === COMMUNICATION ===
    CHEMICAL_COMMUNICATION = "chemical_communication"  # VOCs, diffusible signals
    ELECTRICAL_SIGNALING = "electrical_signaling"  # Action potential-like spikes
    INTER_KINGDOM_SIGNALING = "inter_kingdom_signaling"  # Plant-fungal communication

    # === COGNITIVE FUNCTIONS ===
    MEMORY_TRACES = "memory_traces"  # Information storage in structure
    SPATIAL_MAPPING = "spatial_mapping"  # Environmental navigation
    TEMPORAL_ANTICIPATION = "temporal_anticipation"  # Predicting periodic events
    PROBLEM_SOLVING = "problem_solving"  # Maze solving, optimization

    # === SYMBIOTIC ===
    SYMBIOTIC_NEGOTIATION = "symbiotic_negotiation"  # Exchange rate determination
    HOST_MANIPULATION = "host_manipulation"  # Behavioral control (Ophiocordyceps)
    CHEATER_DETECTION = "cheater_detection"  # Identifying non-reciprocal partners

    # === ADAPTIVE ===
    STRESS_RESPONSE = "stress_response"  # Environmental adaptation
    WOUND_HEALING = "wound_healing"  # Network repair
    COLLECTIVE_DECISION = "collective_decision"  # Unified choices without central control

    # === SPECIALIZED ===
    PREDATORY_BEHAVIOR = "predatory_behavior"  # Nematode-trapping
    SELF_RECOGNITION = "self_recognition"  # het gene systems
    INFORMATION_INTEGRATION = "information_integration"  # Multi-signal processing


class FungalType(Enum):
    """
    Major functional and ecological types of fungi.

    Categorized by ecological role, symbiotic relationships,
    and special characteristics relevant to intelligence studies.
    """

    # === MYCORRHIZAL ===
    MYCORRHIZAL_ECTO = "mycorrhizal_ecto"  # Ectomycorrhizal (Hartig net)
    MYCORRHIZAL_ARBUSCULAR = "mycorrhizal_arbuscular"  # Arbuscular (AMF)
    MYCORRHIZAL_ERICOID = "mycorrhizal_ericoid"  # Ericaceae specialists
    MYCORRHIZAL_ORCHID = "mycorrhizal_orchid"  # Orchid mycorrhizae

    # === TROPHIC TYPES ===
    SAPROPHYTIC = "saprophytic"  # Decomposers
    PARASITIC = "parasitic"  # Living host feeders
    MYCOPARASITIC = "mycoparasitic"  # Parasites of other fungi
    PREDATORY = "predatory"  # Nematode-trapping fungi

    # === SPECIAL GROUPS ===
    LICHENIZED = "lichenized"  # Algal/cyanobacterial symbiosis
    SLIME_MOLDS = "slime_molds"  # Myxomycetes (not true fungi)
    BIOLUMINESCENT = "bioluminescent"  # Light-producing species

    # === BIOACTIVE ===
    ENTHEOGENIC = "entheogenic"  # Psychoactive species
    MEDICINAL = "medicinal"  # Therapeutic compounds
    TOXIC = "toxic"  # Poisonous species


class NetworkBehavior(Enum):
    """
    Observable behaviors in fungal networks.

    These represent measurable network-level phenomena
    that suggest intelligent or adaptive behavior.
    """

    NUTRIENT_TRANSFER = "nutrient_transfer"  # Resource sharing
    SIGNAL_PROPAGATION = "signal_propagation"  # Information transmission
    DEFENSE_RESPONSE = "defense_response"  # Coordinated protection
    GROWTH_OPTIMIZATION = "growth_optimization"  # Efficient expansion
    RESOURCE_HOARDING = "resource_hoarding"  # Storage behavior
    NETWORK_PRUNING = "network_pruning"  # Removing inefficient connections
    HUB_FORMATION = "hub_formation"  # Creating central nodes
    REDUNDANCY_CREATION = "redundancy_creation"  # Fault tolerance


class ResearchParadigm(Enum):
    """
    Research approaches to studying fungal intelligence.
    """

    NETWORK_SCIENCE = "network_science"  # Graph theory, topology
    UNCONVENTIONAL_COMPUTING = "unconventional_computing"  # Bio-inspired computation
    NEUROSCIENCE_COMPARATIVE = "neuroscience_comparative"  # Neural analogs
    ECOLOGY = "ecology"  # Ecosystem function
    ETHNOMYCOLOGY = "ethnomycology"  # Cultural/indigenous knowledge
    BIOCHEMISTRY = "biochemistry"  # Chemical signaling
    ELECTROPHYSIOLOGY = "electrophysiology"  # Electrical signals
    EVOLUTIONARY_BIOLOGY = "evolutionary_biology"  # Origins of cognition
    CONSCIOUSNESS_STUDIES = "consciousness_studies"  # Non-neural awareness


class IndigenousFungalTradition(Enum):
    """
    Indigenous and traditional cultures with documented fungal practices.
    """

    # === MESOAMERICAN ===
    MAZATEC_MUSHROOM = "mazatec_mushroom"  # Psilocybe velada ceremonies
    NAHUA_TEONANACATL = "nahua_teonanacatl"  # Aztec "flesh of the gods"

    # === SIBERIAN ===
    SIBERIAN_AMANITA = "siberian_amanita"  # Amanita muscaria shamanism
    KORYAK_FLY_AGARIC = "koryak_fly_agaric"  # Kamchatka traditions

    # === EAST ASIAN ===
    CHINESE_LINGZHI = "chinese_lingzhi"  # Reishi in TCM
    JAPANESE_SHIITAKE = "japanese_shiitake"  # Cultivation traditions
    TIBETAN_CORDYCEPS = "tibetan_cordyceps"  # High altitude medicine

    # === OCEANIC ===
    ABORIGINAL_TRUFFLE = "aboriginal_truffle"  # Australian desert truffles
    ABORIGINAL_MEDICINAL = "aboriginal_medicinal"  # Bracket fungi medicine

    # === EUROPEAN ===
    EUROPEAN_FOLK = "european_folk"  # Fairy rings, folk medicine
    NORSE_BERSERKER = "norse_berserker"  # Fly agaric hypothesis
    ALPINE_OTZI = "alpine_otzi"  # Birch polypore traditions

    # === AFRICAN ===
    AFRICAN_TERMITOMYCES = "african_termitomyces"  # Termite-cultivated mushrooms


class MycorrhizalNetworkRole(Enum):
    """
    Roles that organisms play in mycorrhizal networks.
    """

    HUB_TREE = "hub_tree"  # Central node (Mother Tree)
    SEEDLING = "seedling"  # Young recipient
    DONOR = "donor"  # Resource provider
    RECEIVER = "receiver"  # Resource recipient
    BRIDGE = "bridge"  # Connecting separate networks
    MYCOHETEROTROPH = "mycoheterotroph"  # Non-photosynthetic plant


class CommunicationSignalType(Enum):
    """
    Types of signals used in fungal communication.
    """

    VOLATILE_ORGANIC = "volatile_organic"  # Airborne chemicals
    DIFFUSIBLE_CHEMICAL = "diffusible_chemical"  # Solution-phase
    ELECTRICAL = "electrical"  # Ion-based signals
    HYDRAULIC = "hydraulic"  # Pressure changes
    MECHANICAL = "mechanical"  # Physical contact


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
class FungalNetworkProfile:
    """
    Represents a documented fungal network or notable fungal organism.

    Profiles can represent individual species, mycorrhizal networks,
    or notable organisms studied for their intelligence properties.
    """
    network_id: str
    name: str
    primary_species: str
    host_species: List[str] = field(default_factory=list)
    network_size: Optional[str] = None  # e.g., "9.6 km^2", "single cell"
    intelligence_domains: List[FungalIntelligenceDomain] = field(default_factory=list)
    fungal_type: FungalType = FungalType.SAPROPHYTIC
    ecological_role: str = ""
    geographic_range: List[str] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)
    landmark_discoveries: List[str] = field(default_factory=list)
    description: str = ""
    related_profiles: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Network: {self.name}",
            f"Species: {self.primary_species}",
            f"Type: {self.fungal_type.value}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class SlimeMoldExperiment:
    """
    Represents a documented slime mold intelligence experiment.

    Captures the methodology, findings, and computational analogs
    of experiments demonstrating slime mold cognitive abilities.
    """
    experiment_id: str
    name: str
    species: str  # Usually Physarum polycephalum
    paradigm: ResearchParadigm
    year: Optional[int] = None
    researchers: List[str] = field(default_factory=list)
    methodology: str = ""
    findings: str = ""
    computational_analog: str = ""  # What computational problem it solves
    performance_vs_algorithms: Optional[str] = None  # How it compares
    intelligence_domains: List[FungalIntelligenceDomain] = field(default_factory=list)
    replication_rate: Optional[float] = None  # How reproducible
    publication: Optional[str] = None
    related_experiments: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Experiment: {self.name}",
            f"Species: {self.species}",
            f"Findings: {self.findings}",
            f"Computational analog: {self.computational_analog}"
        ]
        return " | ".join(parts)


@dataclass
class MycelialCommunication:
    """
    Represents a documented instance of fungal communication.

    Captures chemical, electrical, or other signaling between
    fungi and their partners (plants, other fungi, animals).
    """
    communication_id: str
    name: str
    sender: str  # Species or node type
    receiver: str  # Species or node type
    signal_type: CommunicationSignalType
    signal_compounds: List[str] = field(default_factory=list)  # e.g., VOCs
    distance: Optional[str] = None  # How far signal travels
    ecological_context: str = ""
    purpose: str = ""  # Defense, nutrient, mating, etc.
    response_time: Optional[str] = None
    research_method: str = ""
    related_communications: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class IndigenousFungalWisdom:
    """
    Represents traditional/indigenous knowledge about fungi.

    Captures ceremonial, medicinal, and practical knowledge
    with appropriate cultural attribution and ethical considerations.
    """
    wisdom_id: str
    name: str
    fungal_species: str
    indigenous_name: Optional[str] = None
    tradition: IndigenousFungalTradition = IndigenousFungalTradition.MAZATEC_MUSHROOM
    source_communities: List[str] = field(default_factory=list)
    ceremonial_use: str = ""
    medicinal_use: str = ""
    preparation_methods: List[str] = field(default_factory=list)
    cultural_significance: str = ""
    taboos_restrictions: List[str] = field(default_factory=list)
    documented_antiquity: Optional[str] = None  # How far back documented
    current_status: str = ""  # Active, declining, revived
    ip_concerns: List[str] = field(default_factory=list)
    form_29_link: Optional[str] = None  # Link to Folk Wisdom form
    related_wisdom: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FungalIntelligenceMaturityState:
    """Tracks the maturity of fungal intelligence knowledge."""
    overall_maturity: float = 0.0
    domain_coverage: Dict[str, float] = field(default_factory=dict)
    network_profile_count: int = 0
    experiment_count: int = 0
    communication_count: int = 0
    wisdom_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class FungalIntelligenceInterface:
    """
    Main interface for Form 32: Fungal Networks & Mycorrhizal Intelligence.

    Provides methods for storing, retrieving, and querying fungal network
    profiles, slime mold experiments, communication records, and indigenous
    fungal wisdom from cultures worldwide.
    """

    FORM_ID = "32-fungal-intelligence"
    FORM_NAME = "Fungal Networks & Mycorrhizal Intelligence"

    def __init__(self):
        """Initialize the Fungal Intelligence Interface."""
        # Knowledge indexes
        self.network_profile_index: Dict[str, FungalNetworkProfile] = {}
        self.experiment_index: Dict[str, SlimeMoldExperiment] = {}
        self.communication_index: Dict[str, MycelialCommunication] = {}
        self.wisdom_index: Dict[str, IndigenousFungalWisdom] = {}

        # Cross-reference indexes
        self.domain_index: Dict[FungalIntelligenceDomain, List[str]] = {}
        self.type_index: Dict[FungalType, List[str]] = {}
        self.tradition_index: Dict[IndigenousFungalTradition, List[str]] = {}

        # Maturity tracking
        self.maturity_state = FungalIntelligenceMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize domain index
        for domain in FungalIntelligenceDomain:
            self.domain_index[domain] = []

        # Initialize type index
        for fungal_type in FungalType:
            self.type_index[fungal_type] = []

        # Initialize tradition index
        for tradition in IndigenousFungalTradition:
            self.tradition_index[tradition] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # NETWORK PROFILE METHODS
    # ========================================================================

    async def add_network_profile(self, profile: FungalNetworkProfile) -> None:
        """Add a fungal network profile to the index."""
        self.network_profile_index[profile.network_id] = profile

        # Update type index
        if profile.fungal_type in self.type_index:
            self.type_index[profile.fungal_type].append(profile.network_id)

        # Update domain index
        for domain in profile.intelligence_domains:
            if domain in self.domain_index:
                self.domain_index[domain].append(profile.network_id)

        # Update maturity
        self.maturity_state.network_profile_count = len(self.network_profile_index)
        await self._update_maturity()

    async def get_network_profile(self, network_id: str) -> Optional[FungalNetworkProfile]:
        """Retrieve a network profile by ID."""
        return self.network_profile_index.get(network_id)

    async def query_profiles_by_type(
        self,
        fungal_type: FungalType,
        limit: int = 10
    ) -> List[FungalNetworkProfile]:
        """Query network profiles by fungal type."""
        profile_ids = self.type_index.get(fungal_type, [])[:limit]
        return [
            self.network_profile_index[pid]
            for pid in profile_ids
            if pid in self.network_profile_index
        ]

    async def query_profiles_by_domain(
        self,
        domain: FungalIntelligenceDomain,
        limit: int = 10
    ) -> List[FungalNetworkProfile]:
        """Query network profiles by intelligence domain."""
        profile_ids = self.domain_index.get(domain, [])[:limit]
        return [
            self.network_profile_index[pid]
            for pid in profile_ids
            if pid in self.network_profile_index
        ]

    # ========================================================================
    # EXPERIMENT METHODS
    # ========================================================================

    async def add_experiment(self, experiment: SlimeMoldExperiment) -> None:
        """Add a slime mold experiment to the index."""
        self.experiment_index[experiment.experiment_id] = experiment

        # Update domain index
        for domain in experiment.intelligence_domains:
            if domain in self.domain_index:
                self.domain_index[domain].append(experiment.experiment_id)

        # Update maturity
        self.maturity_state.experiment_count = len(self.experiment_index)
        await self._update_maturity()

    async def get_experiment(self, experiment_id: str) -> Optional[SlimeMoldExperiment]:
        """Retrieve an experiment by ID."""
        return self.experiment_index.get(experiment_id)

    async def query_experiments_by_paradigm(
        self,
        paradigm: ResearchParadigm,
        limit: int = 10
    ) -> List[SlimeMoldExperiment]:
        """Query experiments by research paradigm."""
        results = []
        for experiment in self.experiment_index.values():
            if experiment.paradigm == paradigm:
                results.append(experiment)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # COMMUNICATION METHODS
    # ========================================================================

    async def add_communication(self, communication: MycelialCommunication) -> None:
        """Add a communication record to the index."""
        self.communication_index[communication.communication_id] = communication

        # Update maturity
        self.maturity_state.communication_count = len(self.communication_index)
        await self._update_maturity()

    async def get_communication(self, communication_id: str) -> Optional[MycelialCommunication]:
        """Retrieve a communication record by ID."""
        return self.communication_index.get(communication_id)

    async def query_communications_by_signal_type(
        self,
        signal_type: CommunicationSignalType,
        limit: int = 10
    ) -> List[MycelialCommunication]:
        """Query communications by signal type."""
        results = []
        for comm in self.communication_index.values():
            if comm.signal_type == signal_type:
                results.append(comm)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # WISDOM METHODS
    # ========================================================================

    async def add_wisdom(self, wisdom: IndigenousFungalWisdom) -> None:
        """Add indigenous fungal wisdom to the index."""
        self.wisdom_index[wisdom.wisdom_id] = wisdom

        # Update tradition index
        if wisdom.tradition in self.tradition_index:
            self.tradition_index[wisdom.tradition].append(wisdom.wisdom_id)

        # Update maturity
        self.maturity_state.wisdom_count = len(self.wisdom_index)
        await self._update_maturity()

    async def get_wisdom(self, wisdom_id: str) -> Optional[IndigenousFungalWisdom]:
        """Retrieve wisdom by ID."""
        return self.wisdom_index.get(wisdom_id)

    async def query_wisdom_by_tradition(
        self,
        tradition: IndigenousFungalTradition,
        limit: int = 10
    ) -> List[IndigenousFungalWisdom]:
        """Query wisdom by tradition."""
        wisdom_ids = self.tradition_index.get(tradition, [])[:limit]
        return [
            self.wisdom_index[wid]
            for wid in wisdom_ids
            if wid in self.wisdom_index
        ]

    async def query_wisdom_by_species(
        self,
        species_name: str
    ) -> List[IndigenousFungalWisdom]:
        """Query wisdom by fungal species name."""
        results = []
        species_lower = species_name.lower()
        for wisdom in self.wisdom_index.values():
            if species_lower in wisdom.fungal_species.lower():
                results.append(wisdom)
        return results

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.network_profile_count +
            self.maturity_state.experiment_count +
            self.maturity_state.communication_count +
            self.maturity_state.wisdom_count
        )

        # Simple maturity calculation
        target_items = 200  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update domain coverage
        for domain in FungalIntelligenceDomain:
            count = len(self.domain_index.get(domain, []))
            target_per_domain = 10
            self.maturity_state.domain_coverage[domain.value] = min(
                1.0, count / target_per_domain
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> FungalIntelligenceMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_network_profiles(self) -> List[Dict[str, Any]]:
        """Return seed network profiles for initialization."""
        return [
            # Physarum polycephalum - Slime Mold
            {
                "network_id": "physarum_polycephalum",
                "name": "Physarum polycephalum",
                "primary_species": "Physarum polycephalum",
                "network_size": "Single multinucleate cell, up to several m^2",
                "intelligence_domains": [
                    FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
                    FungalIntelligenceDomain.PROBLEM_SOLVING,
                    FungalIntelligenceDomain.MEMORY_TRACES,
                    FungalIntelligenceDomain.SPATIAL_MAPPING,
                    FungalIntelligenceDomain.TEMPORAL_ANTICIPATION,
                    FungalIntelligenceDomain.COLLECTIVE_DECISION,
                ],
                "fungal_type": FungalType.SLIME_MOLDS,
                "ecological_role": "Decomposer of decaying plant matter",
                "geographic_range": ["Worldwide", "Temperate forests", "Decaying logs"],
                "key_researchers": ["Toshiyuki Nakagaki", "Andrew Adamatzky", "Audrey Dussutour"],
                "landmark_discoveries": [
                    "Maze solving (Nakagaki 2000)",
                    "Tokyo rail network recreation (Tero 2010)",
                    "Habituation learning (Boisseau 2016)",
                    "Memory transfer through fusion",
                ],
                "description": "An acellular slime mold (not a true fungus) that demonstrates remarkable cognitive abilities despite lacking a brain or nervous system. A single giant cell with millions of nuclei that can solve mazes, optimize networks, learn, and remember.",
            },
            # Armillaria ostoyae - Largest Organism
            {
                "network_id": "armillaria_ostoyae_humongous",
                "name": "Humongous Fungus",
                "primary_species": "Armillaria ostoyae",
                "host_species": ["Conifers", "Various trees"],
                "network_size": "9.6 km^2 (965 hectares)",
                "intelligence_domains": [
                    FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
                    FungalIntelligenceDomain.RESOURCE_DISTRIBUTION,
                    FungalIntelligenceDomain.WOUND_HEALING,
                ],
                "fungal_type": FungalType.PARASITIC,
                "ecological_role": "Forest pathogen and decomposer",
                "geographic_range": ["Malheur National Forest, Oregon, USA"],
                "key_researchers": ["Catherine Parks", "Tom Volk"],
                "landmark_discoveries": [
                    "Largest known organism by area (2.4 miles across)",
                    "Estimated 2,400-8,650 years old",
                    "Single genetic individual",
                ],
                "description": "The largest known organism on Earth, a single fungal individual spanning 965 hectares in Oregon's Blue Mountains. Despite its vast size, it maintains genetic unity and coordinated behavior across the entire network.",
            },
            # Douglas Fir Mycorrhizal Network
            {
                "network_id": "douglas_fir_cmn",
                "name": "Douglas Fir Mother Tree Network",
                "primary_species": "Multiple ectomycorrhizal species",
                "host_species": ["Pseudotsuga menziesii (Douglas fir)", "Betula papyrifera (Paper birch)"],
                "network_size": "Forest-scale, hundreds of meters",
                "intelligence_domains": [
                    FungalIntelligenceDomain.RESOURCE_DISTRIBUTION,
                    FungalIntelligenceDomain.NUTRIENT_TRANSFER,
                    FungalIntelligenceDomain.INTER_KINGDOM_SIGNALING,
                    FungalIntelligenceDomain.SYMBIOTIC_NEGOTIATION,
                ],
                "fungal_type": FungalType.MYCORRHIZAL_ECTO,
                "ecological_role": "Forest communication and resource sharing network",
                "geographic_range": ["Pacific Northwest", "British Columbia"],
                "key_researchers": ["Suzanne Simard", "David Perry"],
                "landmark_discoveries": [
                    "Trees share carbon through mycorrhizal networks (1997)",
                    "Mother trees preferentially support kin seedlings",
                    "Dying trees transfer resources to neighbors",
                    "Wood Wide Web concept",
                ],
                "description": "The paradigmatic example of mycorrhizal network intelligence, where hub 'mother trees' serve as central nodes distributing resources to seedlings, with preferential allocation to genetic kin. Demonstrates that forests function as superorganisms.",
            },
            # Psilocybe cubensis
            {
                "network_id": "psilocybe_cubensis",
                "name": "Psilocybe cubensis",
                "primary_species": "Psilocybe cubensis",
                "network_size": "Colony-scale mycelium",
                "intelligence_domains": [
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.ENTHEOGENIC,
                "ecological_role": "Saprophyte on cattle dung and enriched soils",
                "geographic_range": ["Tropical and subtropical worldwide"],
                "key_researchers": ["R. Gordon Wasson", "Albert Hofmann", "Paul Stamets"],
                "landmark_discoveries": [
                    "Psilocybin isolation and synthesis",
                    "Serotonin receptor agonism (5-HT2A)",
                    "Default mode network suppression",
                    "Neuroplasticity enhancement",
                ],
                "description": "The most commonly cultivated entheogenic mushroom, producing psilocybin and psilocin. Currently being researched for treatment-resistant depression, end-of-life anxiety, and addiction.",
            },
            # Psilocybe mexicana
            {
                "network_id": "psilocybe_mexicana",
                "name": "Psilocybe mexicana (Teonanacatl)",
                "primary_species": "Psilocybe mexicana",
                "network_size": "Colony-scale mycelium",
                "intelligence_domains": [
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.ENTHEOGENIC,
                "ecological_role": "Saprophyte in grassy areas",
                "geographic_range": ["Mexico", "Central America"],
                "key_researchers": ["R. Gordon Wasson", "Maria Sabina", "Albert Hofmann"],
                "landmark_discoveries": [
                    "Used in Mazatec velada ceremonies",
                    "First species from which psilocybin was isolated (1958)",
                    "Traditional 'flesh of the gods' (teonanacatl)",
                ],
                "description": "The sacred mushroom used in traditional Mazatec velada ceremonies, famously practiced by Maria Sabina. Known as teonanacatl ('flesh of the gods') to the Aztecs.",
            },
            # Amanita muscaria
            {
                "network_id": "amanita_muscaria",
                "name": "Amanita muscaria (Fly Agaric)",
                "primary_species": "Amanita muscaria",
                "host_species": ["Pinus", "Picea", "Betula", "Various conifers and deciduous trees"],
                "network_size": "Extensive ectomycorrhizal networks",
                "intelligence_domains": [
                    FungalIntelligenceDomain.SYMBIOTIC_NEGOTIATION,
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.ENTHEOGENIC,
                "ecological_role": "Ectomycorrhizal symbiont",
                "geographic_range": ["Northern Hemisphere", "Temperate and boreal forests"],
                "key_researchers": ["R. Gordon Wasson", "Ethnomycological researchers"],
                "landmark_discoveries": [
                    "Muscimol as primary active compound",
                    "GABAergic mechanism (different from psilocybin)",
                    "Siberian shamanic use possibly 6000+ years old",
                    "Reindeer behavior connection to flying reindeer myths",
                ],
                "description": "The iconic red-capped mushroom with white spots, used in Siberian shamanic traditions for millennia. Contains muscimol (GABAergic) rather than psilocybin, producing distinctly different effects.",
            },
            # Hericium erinaceus - Lion's Mane
            {
                "network_id": "hericium_erinaceus",
                "name": "Hericium erinaceus (Lion's Mane)",
                "primary_species": "Hericium erinaceus",
                "network_size": "Wood-colonizing mycelium",
                "intelligence_domains": [
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.MEDICINAL,
                "ecological_role": "Saprophyte on hardwood trees",
                "geographic_range": ["North America", "Europe", "Asia"],
                "key_researchers": ["Hirokazu Kawagishi", "Paul Stamets"],
                "landmark_discoveries": [
                    "Hericenones stimulate NGF synthesis",
                    "Erinacines cross blood-brain barrier",
                    "Hippocampal neurogenesis promotion",
                    "Improved cognitive function in mild cognitive impairment trials",
                ],
                "description": "A medicinal mushroom renowned for its neurotrophic effects. Contains compounds that stimulate nerve growth factor (NGF) synthesis and promote neurogenesis, showing promise for cognitive enhancement and neurodegenerative conditions.",
            },
            # Ganoderma lucidum - Reishi
            {
                "network_id": "ganoderma_lucidum",
                "name": "Ganoderma lucidum (Reishi/Lingzhi)",
                "primary_species": "Ganoderma lucidum",
                "network_size": "Wood-colonizing mycelium",
                "intelligence_domains": [
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.MEDICINAL,
                "ecological_role": "Saprophyte/weak parasite on hardwoods",
                "geographic_range": ["Worldwide", "Temperate and subtropical"],
                "key_researchers": ["Traditional Chinese Medicine practitioners"],
                "landmark_discoveries": [
                    "Over 400 bioactive compounds identified",
                    "380+ triterpenoids (ganoderic acids)",
                    "Immunomodulatory bidirectional effects",
                    "2000+ years of documented use in TCM",
                ],
                "description": "The 'mushroom of immortality' in Traditional Chinese Medicine, documented for over 2,000 years. Contains extensive bioactive compounds with immunomodulatory effects, able to both enhance weakened and suppress overactive immune responses.",
            },
            # Ophiocordyceps unilateralis - Zombie Ant Fungus
            {
                "network_id": "ophiocordyceps_unilateralis",
                "name": "Ophiocordyceps unilateralis (Zombie Ant Fungus)",
                "primary_species": "Ophiocordyceps unilateralis",
                "host_species": ["Camponotus leonardi", "Various carpenter ants"],
                "network_size": "Individual host infection",
                "intelligence_domains": [
                    FungalIntelligenceDomain.HOST_MANIPULATION,
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                    FungalIntelligenceDomain.SPATIAL_MAPPING,
                ],
                "fungal_type": FungalType.PARASITIC,
                "ecological_role": "Obligate ant parasite",
                "geographic_range": ["Tropical forests worldwide"],
                "key_researchers": ["David Hughes", "Charissa de Bekker"],
                "landmark_discoveries": [
                    "Precise behavioral manipulation of host",
                    "Ant brain remains intact - control is peripheral",
                    "Sphingosine and GBA compounds affect ant nervous system",
                    "~320 species, ~35 manipulate ant behavior",
                ],
                "description": "A remarkable parasitic fungus that manipulates ant behavior with extraordinary precision, causing infected ants to climb to optimal height and grip leaves before death, creating ideal conditions for spore dispersal.",
            },
            # Mycena chlorophos - Bioluminescent
            {
                "network_id": "mycena_chlorophos",
                "name": "Mycena chlorophos (Glowing Mushroom)",
                "primary_species": "Mycena chlorophos",
                "network_size": "Small colony mycelium",
                "intelligence_domains": [
                    FungalIntelligenceDomain.CHEMICAL_COMMUNICATION,
                ],
                "fungal_type": FungalType.BIOLUMINESCENT,
                "ecological_role": "Saprophyte on dead wood",
                "geographic_range": ["Tropical Asia", "Australasia", "Japan", "Taiwan"],
                "key_researchers": ["Cassius Stevani", "Anderson Oliveira"],
                "landmark_discoveries": [
                    "Brightest known bioluminescent fungus",
                    "Circadian rhythm in luminescence",
                    "Hispidin-based luciferin pathway",
                    "Emission peak at 520-530 nm",
                ],
                "description": "One of the brightest bioluminescent fungi, producing an eerie green glow visible to the naked eye. Bioluminescence follows a circadian rhythm and may function to attract spore-dispersing insects.",
            },
        ]

    def _get_seed_experiments(self) -> List[Dict[str, Any]]:
        """Return seed experiments for initialization."""
        return [
            # Maze Solving
            {
                "experiment_id": "nakagaki_maze_2000",
                "name": "Physarum Maze Solving",
                "species": "Physarum polycephalum",
                "paradigm": ResearchParadigm.UNCONVENTIONAL_COMPUTING,
                "year": 2000,
                "researchers": ["Toshiyuki Nakagaki", "Hiroyasu Yamada", "Agnes Toth"],
                "methodology": "Physarum placed in agar maze with food sources at entrance and exit. Observed network formation over time.",
                "findings": "Slime mold consistently found shortest path through maze, retracting from dead ends and optimizing route.",
                "computational_analog": "Shortest path problem, maze solving algorithms",
                "intelligence_domains": [
                    FungalIntelligenceDomain.PROBLEM_SOLVING,
                    FungalIntelligenceDomain.SPATIAL_MAPPING,
                    FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
                ],
                "publication": "Nature 407:470 (2000)",
            },
            # Tokyo Rail Network
            {
                "experiment_id": "tero_tokyo_2010",
                "name": "Tokyo Rail Network Recreation",
                "species": "Physarum polycephalum",
                "paradigm": ResearchParadigm.NETWORK_SCIENCE,
                "year": 2010,
                "researchers": ["Atsushi Tero", "Seiji Takagi", "Tetsu Saigusa", "Toshiyuki Nakagaki"],
                "methodology": "Oat flakes placed on agar in positions matching 36 cities around Tokyo. Physarum placed at Tokyo position. Network formation observed over 26 hours.",
                "findings": "Slime mold created networks with comparable efficiency, fault tolerance, and cost to actual Tokyo rail system. Different trials produced various solutions, but many showed striking similarity to actual rail networks.",
                "computational_analog": "Multi-objective network optimization, Steiner tree problem",
                "performance_vs_algorithms": "Comparable to or better than human-designed networks on efficiency/robustness tradeoff",
                "intelligence_domains": [
                    FungalIntelligenceDomain.NETWORK_OPTIMIZATION,
                    FungalIntelligenceDomain.COLLECTIVE_DECISION,
                    FungalIntelligenceDomain.PROBLEM_SOLVING,
                ],
                "publication": "Science 327:439 (2010)",
            },
            # Habituation Learning
            {
                "experiment_id": "boisseau_habituation_2016",
                "name": "Habituation Learning in Slime Molds",
                "species": "Physarum polycephalum",
                "paradigm": ResearchParadigm.NEUROSCIENCE_COMPARATIVE,
                "year": 2016,
                "researchers": ["Romain P. Boisseau", "David Vogel", "Audrey Dussutour"],
                "methodology": "Slime molds exposed to aversive but harmless substances (quinine, caffeine) blocking path to food. Response measured over repeated exposures.",
                "findings": "Slime molds learned to ignore aversive stimuli after repeated exposure (habituation). Learning was specific to the substance encountered and persisted for days.",
                "computational_analog": "Adaptive filtering, habituation learning algorithms",
                "intelligence_domains": [
                    FungalIntelligenceDomain.MEMORY_TRACES,
                    FungalIntelligenceDomain.STRESS_RESPONSE,
                ],
                "publication": "Proceedings of the Royal Society B 283:20160446 (2016)",
            },
            # Memory Transfer
            {
                "experiment_id": "vogel_memory_transfer_2019",
                "name": "Memory Transfer Through Fusion",
                "species": "Physarum polycephalum",
                "paradigm": ResearchParadigm.NEUROSCIENCE_COMPARATIVE,
                "year": 2019,
                "researchers": ["David Vogel", "Audrey Dussutour"],
                "methodology": "Slime molds habituated to sodium (aversive) fused with naive individuals. Behavior of fused organism tested.",
                "findings": "Habituated slime molds could transfer learned information to naive individuals through cell fusion. The mechanism involves circulating memory through absorbed substances.",
                "computational_analog": "Distributed memory systems, knowledge transfer",
                "intelligence_domains": [
                    FungalIntelligenceDomain.MEMORY_TRACES,
                    FungalIntelligenceDomain.INFORMATION_INTEGRATION,
                ],
                "publication": "Philosophical Transactions of the Royal Society B 374:20180368 (2019)",
            },
            # Temporal Anticipation
            {
                "experiment_id": "saigusa_anticipation_2008",
                "name": "Temporal Anticipation in Physarum",
                "species": "Physarum polycephalum",
                "paradigm": ResearchParadigm.NEUROSCIENCE_COMPARATIVE,
                "year": 2008,
                "researchers": ["Tetsu Saigusa", "Atsushi Tero", "Toshiyuki Nakagaki"],
                "methodology": "Slime molds exposed to cold temperatures at regular intervals. After several cycles, tested whether they anticipated upcoming cold.",
                "findings": "Slime molds learned to anticipate periodic events and prepared in advance, demonstrating primitive temporal memory.",
                "computational_analog": "Periodic event prediction, circadian rhythm modeling",
                "intelligence_domains": [
                    FungalIntelligenceDomain.TEMPORAL_ANTICIPATION,
                    FungalIntelligenceDomain.MEMORY_TRACES,
                ],
                "publication": "Physical Review Letters 100:018101 (2008)",
            },
        ]

    def _get_seed_communications(self) -> List[Dict[str, Any]]:
        """Return seed communication records for initialization."""
        return [
            {
                "communication_id": "tree_to_tree_carbon",
                "name": "Inter-tree Carbon Transfer",
                "sender": "Pseudotsuga menziesii (Douglas fir - mature)",
                "receiver": "Pseudotsuga menziesii (Douglas fir - seedling)",
                "signal_type": CommunicationSignalType.DIFFUSIBLE_CHEMICAL,
                "signal_compounds": ["Carbon-13 labeled sugars", "Carbon-14 labeled sugars"],
                "distance": "Up to 30 meters through CMN",
                "ecological_context": "Forest understory, shaded seedlings",
                "purpose": "Resource provisioning to kin seedlings",
                "research_method": "Isotope tracing",
            },
            {
                "communication_id": "defense_signal_aphid",
                "name": "Defense Signaling Against Aphid Attack",
                "sender": "Vicia faba (Broad bean) - infested",
                "receiver": "Vicia faba (Broad bean) - neighboring",
                "signal_type": CommunicationSignalType.DIFFUSIBLE_CHEMICAL,
                "signal_compounds": ["Jasmonic acid derivatives", "Volatile terpenes"],
                "distance": "Through mycorrhizal network",
                "ecological_context": "Agricultural/garden setting",
                "purpose": "Warning neighboring plants to upregulate defense",
                "research_method": "VOC analysis, gene expression",
            },
            {
                "communication_id": "fungal_electrical_spikes",
                "name": "Oyster Mushroom Electrical Spiking",
                "sender": "Pleurotus djamor mycelium",
                "receiver": "Same mycelium (internal propagation)",
                "signal_type": CommunicationSignalType.ELECTRICAL,
                "signal_compounds": [],
                "distance": "Centimeters to meters through substrate",
                "ecological_context": "Wood substrate colonization",
                "purpose": "Proposed: coordination, environmental sensing",
                "response_time": "Spike duration 1-21 hours",
                "research_method": "Differential electrode recording",
            },
            {
                "communication_id": "voc_species_recognition",
                "name": "Fungal VOC Species Recognition",
                "sender": "Various fungal species",
                "receiver": "Conspecifics and heterospecifics",
                "signal_type": CommunicationSignalType.VOLATILE_ORGANIC,
                "signal_compounds": ["1-octen-3-ol", "Various alcohols", "Terpenes", "Sesquiterpenes"],
                "distance": "Meters through air/soil",
                "ecological_context": "Forest floor, decomposing matter",
                "purpose": "Species recognition, mating signals",
                "research_method": "GC-MS volatile analysis",
            },
        ]

    def _get_seed_wisdom(self) -> List[Dict[str, Any]]:
        """Return seed indigenous wisdom for initialization."""
        return [
            # Mazatec Velada
            {
                "wisdom_id": "mazatec_velada",
                "name": "Mazatec Velada Ceremony",
                "fungal_species": "Psilocybe caerulescens, Psilocybe mexicana, Psilocybe cubensis",
                "indigenous_name": "ndi xijtho (little ones that sprout)",
                "tradition": IndigenousFungalTradition.MAZATEC_MUSHROOM,
                "source_communities": ["Mazatec people", "Huautla de Jimenez, Oaxaca"],
                "ceremonial_use": "Nocturnal healing ceremony led by curandero/curandera. Mushrooms consumed to diagnose illness, communicate with spirits, and receive divine guidance. Not recreational but deeply therapeutic and spiritual.",
                "medicinal_use": "Diagnosis and treatment of physical and spiritual illness",
                "preparation_methods": ["Fresh consumption", "Paired consumption (curandero and patient)"],
                "cultural_significance": "Mushrooms are viewed as conscious entities - 'holy children' capable of communicating wisdom. The ceremony maintains collective myths and community health.",
                "taboos_restrictions": [
                    "Four days sexual abstinence before and after",
                    "Special diet restrictions",
                    "Ceremony conducted at night in darkness",
                    "Not for recreational use",
                ],
                "documented_antiquity": "Pre-Columbian, continues to present",
                "current_status": "Active but impacted by Western interest",
                "ip_concerns": [
                    "Western appropriation without benefit to Mazatec people",
                    "Multi-billion dollar psilocybin industry with no indigenous compensation",
                    "Traditional practices disrupted by influx of seekers",
                ],
                "form_29_link": "mesoamerican_folk",
            },
            # Siberian Amanita
            {
                "wisdom_id": "siberian_fly_agaric",
                "name": "Siberian Fly Agaric Shamanism",
                "fungal_species": "Amanita muscaria",
                "indigenous_name": "wapaq (Koryak)",
                "tradition": IndigenousFungalTradition.SIBERIAN_AMANITA,
                "source_communities": ["Koryak", "Chukchi", "Kamchadal", "Evenki", "Yakut"],
                "ceremonial_use": "Shamanic journeys to upper and lower worlds. Communication with spirits and ancestors. Divination and healing rituals.",
                "medicinal_use": "Healing rituals, vision quests",
                "preparation_methods": [
                    "Drying over fire to convert ibotenic acid to muscimol",
                    "Consumption of dried caps",
                    "Reindeer or human urine recycling to extend effects",
                ],
                "cultural_significance": "Divine offering from Big Raven, first shaman. Central to shamanic practice and worldview. Possible origin of flying reindeer mythology.",
                "taboos_restrictions": [
                    "Proper preparation required for safety",
                    "Shamanic training traditionally required",
                ],
                "documented_antiquity": "Possibly 6,000-10,000 years",
                "current_status": "Diminished due to Russian colonization and Orthodox Christianity",
                "ip_concerns": ["Cultural appropriation concerns"],
                "form_29_link": "siberian",
            },
            # Aboriginal Truffles
            {
                "wisdom_id": "aboriginal_desert_truffle",
                "name": "Aboriginal Desert Truffle Knowledge",
                "fungal_species": "Various native Australian truffles (~300 species)",
                "indigenous_name": "Varies by region and language group",
                "tradition": IndigenousFungalTradition.ABORIGINAL_TRUFFLE,
                "source_communities": ["Various Aboriginal Australian groups"],
                "ceremonial_use": "Part of traditional diet and seasonal knowledge",
                "medicinal_use": "Fluid applied to sores and sore eyes, rubbed into skin",
                "preparation_methods": [
                    "Eaten raw",
                    "Baked or roasted in ashes",
                ],
                "cultural_significance": "Part of Traditional Ecological Knowledge passed through generations. Women traditionally primary truffle hunters.",
                "taboos_restrictions": [],
                "documented_antiquity": "Thousands of years of oral tradition",
                "current_status": "Much knowledge potentially lost due to colonization",
                "ip_concerns": ["Large geographic blank areas with no recorded uses"],
                "form_29_link": "aboriginal_australian",
            },
            # Chinese Lingzhi
            {
                "wisdom_id": "chinese_lingzhi",
                "name": "Chinese Lingzhi (Reishi) Tradition",
                "fungal_species": "Ganoderma lucidum",
                "indigenous_name": "Lingzhi (divine mushroom), Ruizhi, Xiancao",
                "tradition": IndigenousFungalTradition.CHINESE_LINGZHI,
                "source_communities": ["Chinese traditional medicine practitioners"],
                "ceremonial_use": "Associated with longevity, spiritual potency, and immortality in Taoist traditions",
                "medicinal_use": "Qi enhancement, immune support, calming shen (spirit), adaptogenic effects",
                "preparation_methods": [
                    "Decoction (water extraction)",
                    "Alcohol tincture",
                    "Powdered form",
                ],
                "cultural_significance": "One of the most revered herbs in TCM. Symbol of good fortune and longevity. Mentioned in earliest Chinese pharmacopeias.",
                "taboos_restrictions": [],
                "documented_antiquity": "Over 2,000 years of documented use",
                "current_status": "Active, widely cultivated",
                "ip_concerns": [],
                "form_29_link": "east_asian_folk",
            },
            # Lion's Mane Cognitive
            {
                "wisdom_id": "lions_mane_cognitive",
                "name": "Lion's Mane Traditional and Modern Use",
                "fungal_species": "Hericium erinaceus",
                "indigenous_name": "Yamabushitake (Japanese), Hou tou gu (Chinese)",
                "tradition": IndigenousFungalTradition.JAPANESE_SHIITAKE,
                "source_communities": ["Japanese Buddhist monks (Yamabushi)", "Chinese medicine practitioners"],
                "ceremonial_use": "Associated with Buddhist mountain ascetics (Yamabushi) for meditation support",
                "medicinal_use": "Cognitive enhancement, nerve regeneration, digestive health",
                "preparation_methods": [
                    "Culinary preparation",
                    "Hot water extraction",
                    "Alcohol extraction",
                ],
                "cultural_significance": "Named for resemblance to lion's mane in English, mountain priest's beard in Japanese",
                "taboos_restrictions": [],
                "documented_antiquity": "Centuries of use in TCM and Japanese traditions",
                "current_status": "Active, increasing popularity for nootropic effects",
                "ip_concerns": [],
                "form_29_link": "east_asian_folk",
            },
        ]

    async def initialize_seed_profiles(self) -> int:
        """Initialize with seed network profiles."""
        seed_profiles = self._get_seed_network_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = FungalNetworkProfile(
                network_id=profile_data["network_id"],
                name=profile_data["name"],
                primary_species=profile_data["primary_species"],
                host_species=profile_data.get("host_species", []),
                network_size=profile_data.get("network_size"),
                intelligence_domains=profile_data.get("intelligence_domains", []),
                fungal_type=profile_data.get("fungal_type", FungalType.SAPROPHYTIC),
                ecological_role=profile_data.get("ecological_role", ""),
                geographic_range=profile_data.get("geographic_range", []),
                key_researchers=profile_data.get("key_researchers", []),
                landmark_discoveries=profile_data.get("landmark_discoveries", []),
                description=profile_data.get("description", ""),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_network_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed network profiles")
        return count

    async def initialize_seed_experiments(self) -> int:
        """Initialize with seed experiments."""
        seed_experiments = self._get_seed_experiments()
        count = 0

        for exp_data in seed_experiments:
            experiment = SlimeMoldExperiment(
                experiment_id=exp_data["experiment_id"],
                name=exp_data["name"],
                species=exp_data["species"],
                paradigm=exp_data["paradigm"],
                year=exp_data.get("year"),
                researchers=exp_data.get("researchers", []),
                methodology=exp_data.get("methodology", ""),
                findings=exp_data.get("findings", ""),
                computational_analog=exp_data.get("computational_analog", ""),
                performance_vs_algorithms=exp_data.get("performance_vs_algorithms"),
                intelligence_domains=exp_data.get("intelligence_domains", []),
                publication=exp_data.get("publication"),
            )
            await self.add_experiment(experiment)
            count += 1

        logger.info(f"Initialized {count} seed experiments")
        return count

    async def initialize_seed_communications(self) -> int:
        """Initialize with seed communications."""
        seed_comms = self._get_seed_communications()
        count = 0

        for comm_data in seed_comms:
            communication = MycelialCommunication(
                communication_id=comm_data["communication_id"],
                name=comm_data["name"],
                sender=comm_data["sender"],
                receiver=comm_data["receiver"],
                signal_type=comm_data["signal_type"],
                signal_compounds=comm_data.get("signal_compounds", []),
                distance=comm_data.get("distance"),
                ecological_context=comm_data.get("ecological_context", ""),
                purpose=comm_data.get("purpose", ""),
                response_time=comm_data.get("response_time"),
                research_method=comm_data.get("research_method", ""),
            )
            await self.add_communication(communication)
            count += 1

        logger.info(f"Initialized {count} seed communications")
        return count

    async def initialize_seed_wisdom(self) -> int:
        """Initialize with seed wisdom."""
        seed_wisdom = self._get_seed_wisdom()
        count = 0

        for wisdom_data in seed_wisdom:
            wisdom = IndigenousFungalWisdom(
                wisdom_id=wisdom_data["wisdom_id"],
                name=wisdom_data["name"],
                fungal_species=wisdom_data["fungal_species"],
                indigenous_name=wisdom_data.get("indigenous_name"),
                tradition=wisdom_data["tradition"],
                source_communities=wisdom_data.get("source_communities", []),
                ceremonial_use=wisdom_data.get("ceremonial_use", ""),
                medicinal_use=wisdom_data.get("medicinal_use", ""),
                preparation_methods=wisdom_data.get("preparation_methods", []),
                cultural_significance=wisdom_data.get("cultural_significance", ""),
                taboos_restrictions=wisdom_data.get("taboos_restrictions", []),
                documented_antiquity=wisdom_data.get("documented_antiquity"),
                current_status=wisdom_data.get("current_status", ""),
                ip_concerns=wisdom_data.get("ip_concerns", []),
                form_29_link=wisdom_data.get("form_29_link"),
            )
            await self.add_wisdom(wisdom)
            count += 1

        logger.info(f"Initialized {count} seed wisdom entries")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_profiles()
        experiments_count = await self.initialize_seed_experiments()
        communications_count = await self.initialize_seed_communications()
        wisdom_count = await self.initialize_seed_wisdom()

        return {
            "network_profiles": profiles_count,
            "experiments": experiments_count,
            "communications": communications_count,
            "wisdom": wisdom_count,
            "total": profiles_count + experiments_count + communications_count + wisdom_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "FungalIntelligenceDomain",
    "FungalType",
    "NetworkBehavior",
    "ResearchParadigm",
    "IndigenousFungalTradition",
    "MycorrhizalNetworkRole",
    "CommunicationSignalType",
    "MaturityLevel",
    # Dataclasses
    "FungalNetworkProfile",
    "SlimeMoldExperiment",
    "MycelialCommunication",
    "IndigenousFungalWisdom",
    "FungalIntelligenceMaturityState",
    # Interface
    "FungalIntelligenceInterface",
]
