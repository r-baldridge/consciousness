#!/usr/bin/env python3
"""
Plant Intelligence & Vegetal Consciousness Interface

Form 31: The comprehensive interface for plant cognition, non-neural intelligence,
and traditional knowledge about plant consciousness. This form bridges empirical
research on plant behavior with indigenous wisdom traditions and philosophical
inquiry into vegetal minds.

Ethical Principles:
- Respectful integration of indigenous knowledge
- Recognition of plants as subjects, not mere objects
- Precautionary approach to plant experience
- Support for traditional ecological knowledge
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

class PlantCognitionDomain(Enum):
    """
    Domains of cognitive capacity observed or hypothesized in plants.
    Each domain represents a functional category of plant intelligence.
    """
    # Communication Domains
    CHEMICAL_SIGNALING = "chemical_signaling"
    ROOT_COMMUNICATION = "root_communication"
    VOLATILE_COMMUNICATION = "volatile_communication"
    MYCORRHIZAL_NETWORKING = "mycorrhizal_networking"

    # Memory and Learning Domains
    MEMORY_HABITUATION = "memory_habituation"
    ASSOCIATIVE_LEARNING = "associative_learning"
    EPIGENETIC_MEMORY = "epigenetic_memory"
    STRESS_MEMORY = "stress_memory"

    # Sensing Domains
    CIRCADIAN_AWARENESS = "circadian_awareness"
    GRAVITROPISM = "gravitropism"
    PHOTOTROPISM = "phototropism"
    PROPRIOCEPTION = "proprioception"
    CHEMORECEPTION = "chemoreception"
    THERMOSENSING = "thermosensing"
    HYDROTROPISM = "hydrotropism"
    ACOUSTIC_SENSING = "acoustic_sensing"
    MAGNETORECEPTION = "magnetoreception"

    # Decision and Behavior Domains
    DECISION_MAKING = "decision_making"
    RESOURCE_ALLOCATION = "resource_allocation"
    DEFENSE_COORDINATION = "defense_coordination"
    SPATIAL_NAVIGATION = "spatial_navigation"
    FORAGING_OPTIMIZATION = "foraging_optimization"

    # Social Domains
    KIN_RECOGNITION = "kin_recognition"
    SOCIAL_BEHAVIOR = "social_behavior"
    SYMBIOTIC_INTELLIGENCE = "symbiotic_intelligence"
    ALLELOPATHIC_COMPETITION = "allelopathic_competition"

    # Temporal Domains
    ANTICIPATORY_BEHAVIOR = "anticipatory_behavior"
    SEASONAL_TIMING = "seasonal_timing"
    REPRODUCTIVE_TIMING = "reproductive_timing"

    # Integration Domains
    STRESS_INTEGRATION = "stress_integration"
    IMMUNE_RESPONSE = "immune_response"
    SELF_RECOGNITION = "self_recognition"


class PlantTaxonomicGroup(Enum):
    """
    Taxonomic and functional groupings of plants for consciousness studies.
    Groups are organized by shared cognitive or behavioral characteristics.
    """
    # Tree Groups
    TREES_DECIDUOUS = "trees_deciduous"
    TREES_CONIFEROUS = "trees_coniferous"
    TREES_TROPICAL = "trees_tropical"

    # Functional Groups
    FLOWERING_PLANTS = "flowering_plants"
    VINES_CLIMBERS = "vines_climbers"
    CARNIVOROUS_PLANTS = "carnivorous_plants"
    AQUATIC_PLANTS = "aquatic_plants"
    DESERT_SUCCULENTS = "desert_succulents"
    EPIPHYTES = "epiphytes"
    PARASITIC_PLANTS = "parasitic_plants"

    # Special Groups
    SENSITIVE_PLANTS = "sensitive_plants"
    LEGUMES = "legumes"
    GRASSES = "grasses"
    COLONIAL_CLONAL = "colonial_clonal"

    # Ethnobotanical Groups
    SACRED_PLANTS = "sacred_plants"
    ENTHEOGENIC_PLANTS = "entheogenic_plants"
    MEDICINAL_PLANTS = "medicinal_plants"
    TOXIC_DEFENSIVE = "toxic_defensive"

    # Research Groups
    MODEL_ORGANISMS = "model_organisms"
    ANCIENT_LINEAGES = "ancient_lineages"
    AGRICULTURAL_CROPS = "agricultural_crops"


class PlantSensoryModality(Enum):
    """
    Sensory modalities documented in plants.
    """
    TOUCH_MECHANOSENSING = "touch_mechanosensing"
    LIGHT_PHOTORECEPTION = "light_photoreception"
    GRAVITY_GRAVISENSING = "gravity_gravisensing"
    CHEMICAL_CHEMORECEPTION = "chemical_chemoreception"
    TEMPERATURE_THERMOSENSING = "temperature_thermosensing"
    WATER_HYDROTROPISM = "water_hydrotropism"
    SOUND_VIBRATION = "sound_vibration"
    ELECTROMAGNETIC = "electromagnetic"
    HUMIDITY = "humidity"
    NUTRIENT_SENSING = "nutrient_sensing"
    OXYGEN_SENSING = "oxygen_sensing"
    DAMAGE_WOUNDING = "damage_wounding"


class ResearchParadigm(Enum):
    """
    Research paradigms used to study plant cognition.
    """
    BEHAVIORAL_EXPERIMENTAL = "behavioral_experimental"
    ELECTROPHYSIOLOGICAL = "electrophysiological"
    MOLECULAR_GENETIC = "molecular_genetic"
    ECOLOGICAL_OBSERVATIONAL = "ecological_observational"
    COMPUTATIONAL_MODELING = "computational_modeling"
    COMPARATIVE = "comparative"
    ETHNOBOTANICAL = "ethnobotanical"
    PHARMACOLOGICAL = "pharmacological"
    IMAGING_VISUALIZATION = "imaging_visualization"


class IndigenousTraditionType(Enum):
    """
    Indigenous and traditional frameworks for understanding plant consciousness.
    """
    # Amazonian
    AMAZONIAN_PLANT_TEACHER = "amazonian_plant_teacher"
    DIETA_TRADITION = "dieta_tradition"

    # North American
    NATIVE_AMERICAN_MEDICINE = "native_american_medicine"
    HAUDENOSAUNEE_THREE_SISTERS = "haudenosaunee_three_sisters"
    NATIVE_AMERICAN_CHURCH_PEYOTE = "native_american_church_peyote"

    # Central American
    MAZATEC_MUSHROOM = "mazatec_mushroom"
    MAYA_WORLD_TREE = "maya_world_tree"

    # European
    CELTIC_OGHAM = "celtic_ogham"
    EUROPEAN_HERBALISM = "european_herbalism"

    # Asian
    VEDIC_SOMA = "vedic_soma"
    VEDIC_PLANT_CONSCIOUSNESS = "vedic_plant_consciousness"
    AYURVEDIC = "ayurvedic"
    CHINESE_MEDICINE = "chinese_medicine"
    JAPANESE_SHINRIN_YOKU = "japanese_shinrin_yoku"

    # African
    YORUBA_OSAIN = "yoruba_osain"
    BWITI_IBOGA = "bwiti_iboga"
    SANGOMA_PLANT_SPIRIT = "sangoma_plant_spirit"

    # Oceanian
    ABORIGINAL_AUSTRALIAN = "aboriginal_australian"
    POLYNESIAN = "polynesian"


class PlantSignalingType(Enum):
    """
    Types of signaling mechanisms in plant communication.
    """
    # Electrical Signaling
    ACTION_POTENTIAL = "action_potential"
    VARIATION_POTENTIAL = "variation_potential"
    SYSTEM_POTENTIAL = "system_potential"

    # Chemical Signaling
    VOLATILE_ORGANIC_COMPOUNDS = "volatile_organic_compounds"
    ROOT_EXUDATES = "root_exudates"
    HORMONAL_SYSTEMIC = "hormonal_systemic"
    NEUROTRANSMITTER_LIKE = "neurotransmitter_like"

    # Physical Signaling
    HYDRAULIC = "hydraulic"
    MECHANICAL = "mechanical"

    # Network Signaling
    MYCORRHIZAL_MEDIATED = "mycorrhizal_mediated"
    VASCULAR_TRANSMITTED = "vascular_transmitted"


class MaturityLevel(Enum):
    """
    Depth of knowledge coverage for plant intelligence domains.
    """
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"
    TRANSCENDENT = "transcendent"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PlantSpeciesProfile:
    """
    Comprehensive profile of cognitive capacities for a plant species.

    Represents the core knowledge unit for Form 31, tracking demonstrated
    cognitive abilities, sensory capabilities, and cultural significance.
    """
    species_id: str
    common_name: str
    scientific_name: str
    taxonomic_group: PlantTaxonomicGroup
    cognition_domains: Dict[PlantCognitionDomain, float] = field(default_factory=dict)
    sensory_capabilities: List[PlantSensoryModality] = field(default_factory=list)
    notable_behaviors: List[str] = field(default_factory=list)
    indigenous_perspectives: List[str] = field(default_factory=list)
    research_evidence: List[str] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)
    related_species: List[str] = field(default_factory=list)
    cultural_significance: Optional[str] = None
    conservation_status: Optional[str] = None
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Species: {self.common_name} ({self.scientific_name})",
            f"Group: {self.taxonomic_group.value}",
            f"Cognition: {', '.join(d.value for d in self.cognition_domains.keys())}",
            f"Behaviors: {', '.join(self.notable_behaviors[:5])}"
        ]
        return " | ".join(parts)


@dataclass
class PlantBehaviorInsight:
    """
    Represents a specific observation or insight about plant behavior.

    These are documented instances of plant cognitive behavior with
    supporting evidence and methodology.
    """
    insight_id: str
    species_id: str
    domain: PlantCognitionDomain
    description: str
    evidence_type: ResearchParadigm
    methodology: str
    key_findings: List[str] = field(default_factory=list)
    researchers: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    year_documented: Optional[int] = None
    replicated: bool = False
    confidence_level: str = "moderate"
    controversies: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PlantCommunicationEvent:
    """
    Documented instance of plant-to-plant or plant-environment communication.

    Represents the dynamic communicative relationships between plants
    and their environment.
    """
    event_id: str
    sender_species: str
    receiver_species: Optional[str]  # None for broadcast
    signal_type: PlantSignalingType
    medium: str  # air, soil, mycorrhizal network, etc.
    ecological_context: str
    trigger: str
    signal_content: str
    response_observed: str
    timescale: Optional[str] = None
    distance: Optional[str] = None
    evidence_type: ResearchParadigm = ResearchParadigm.ECOLOGICAL_OBSERVATIONAL
    study_reference: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class IndigenousPlantWisdom:
    """
    Traditional/indigenous knowledge about plant consciousness and behavior.

    Bridges to Form 29 (Folk Wisdom) and honors traditional ecological
    knowledge systems.
    """
    wisdom_id: str
    plant_name: str
    scientific_name: Optional[str] = None
    tradition: IndigenousTraditionType = IndigenousTraditionType.AMAZONIAN_PLANT_TEACHER
    spiritual_significance: str = ""
    practical_uses: List[str] = field(default_factory=list)
    ceremonial_role: Optional[str] = None
    harvesting_protocols: List[str] = field(default_factory=list)
    traditional_preparation: Optional[str] = None
    relationship_principles: List[str] = field(default_factory=list)
    source_community: Optional[str] = None
    public_knowledge: bool = True
    sacred_restricted: bool = False
    form_29_link: Optional[str] = None  # Link to Folk Wisdom form
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PlantLearningExperiment:
    """
    Documented learning experiment in plants.

    Captures the key details of experiments demonstrating plant
    learning and memory capabilities.
    """
    experiment_id: str
    species: str
    learning_type: str  # habituation, associative, sensitization
    methodology: str
    sample_size: int
    controls: List[str] = field(default_factory=list)
    stimulus_conditioned: Optional[str] = None
    stimulus_unconditioned: Optional[str] = None
    learning_demonstrated: bool = False
    memory_duration: Optional[str] = None
    extinction_observed: bool = False
    researchers: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publication: Optional[str] = None
    replicated: bool = False
    significance: Optional[str] = None
    criticisms: List[str] = field(default_factory=list)


@dataclass
class PlantIntelligenceMaturityState:
    """
    Tracks the maturity of plant intelligence knowledge coverage.
    """
    overall_maturity: float = 0.0
    domain_coverage: Dict[str, float] = field(default_factory=dict)
    species_profile_count: int = 0
    behavior_insight_count: int = 0
    communication_event_count: int = 0
    indigenous_wisdom_count: int = 0
    learning_experiment_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class PlantIntelligenceInterface:
    """
    Main interface for Form 31: Plant Intelligence & Vegetal Consciousness.

    Provides methods for storing, retrieving, and querying plant cognition
    profiles, behavioral insights, communication events, and indigenous
    wisdom traditions.
    """

    FORM_ID = "31-plant-intelligence"
    FORM_NAME = "Plant Intelligence & Vegetal Consciousness"

    def __init__(self):
        """Initialize the Plant Intelligence Interface."""
        # Knowledge indexes
        self.species_index: Dict[str, PlantSpeciesProfile] = {}
        self.insight_index: Dict[str, PlantBehaviorInsight] = {}
        self.communication_index: Dict[str, PlantCommunicationEvent] = {}
        self.indigenous_wisdom_index: Dict[str, IndigenousPlantWisdom] = {}
        self.experiment_index: Dict[str, PlantLearningExperiment] = {}

        # Cross-reference indexes
        self.cognition_domain_index: Dict[PlantCognitionDomain, List[str]] = {}
        self.taxonomic_group_index: Dict[PlantTaxonomicGroup, List[str]] = {}
        self.tradition_index: Dict[IndigenousTraditionType, List[str]] = {}

        # Maturity tracking
        self.maturity_state = PlantIntelligenceMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and prepare indexes."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize cognition domain index
        for domain in PlantCognitionDomain:
            self.cognition_domain_index[domain] = []

        # Initialize taxonomic group index
        for group in PlantTaxonomicGroup:
            self.taxonomic_group_index[group] = []

        # Initialize tradition index
        for tradition in IndigenousTraditionType:
            self.tradition_index[tradition] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # SPECIES PROFILE METHODS
    # ========================================================================

    async def add_species_profile(self, profile: PlantSpeciesProfile) -> None:
        """Add a plant species profile to the index."""
        self.species_index[profile.species_id] = profile

        # Update taxonomic group index
        if profile.taxonomic_group in self.taxonomic_group_index:
            self.taxonomic_group_index[profile.taxonomic_group].append(profile.species_id)

        # Update cognition domain index
        for domain in profile.cognition_domains.keys():
            if domain in self.cognition_domain_index:
                self.cognition_domain_index[domain].append(profile.species_id)

        # Update maturity
        self.maturity_state.species_profile_count = len(self.species_index)
        await self._update_maturity()

    async def get_species_profile(self, species_id: str) -> Optional[PlantSpeciesProfile]:
        """Retrieve a species profile by ID."""
        return self.species_index.get(species_id)

    async def query_by_cognition_domain(
        self,
        domain: PlantCognitionDomain,
        limit: int = 10
    ) -> List[PlantSpeciesProfile]:
        """Query species profiles by cognitive domain."""
        species_ids = self.cognition_domain_index.get(domain, [])[:limit]
        return [
            self.species_index[sid]
            for sid in species_ids
            if sid in self.species_index
        ]

    async def query_by_taxonomic_group(
        self,
        group: PlantTaxonomicGroup,
        limit: int = 10
    ) -> List[PlantSpeciesProfile]:
        """Query species profiles by taxonomic group."""
        species_ids = self.taxonomic_group_index.get(group, [])[:limit]
        return [
            self.species_index[sid]
            for sid in species_ids
            if sid in self.species_index
        ]

    # ========================================================================
    # BEHAVIOR INSIGHT METHODS
    # ========================================================================

    async def add_behavior_insight(self, insight: PlantBehaviorInsight) -> None:
        """Add a behavior insight to the index."""
        self.insight_index[insight.insight_id] = insight

        # Update domain index
        if insight.domain in self.cognition_domain_index:
            self.cognition_domain_index[insight.domain].append(insight.insight_id)

        # Update maturity
        self.maturity_state.behavior_insight_count = len(self.insight_index)
        await self._update_maturity()

    async def get_behavior_insight(self, insight_id: str) -> Optional[PlantBehaviorInsight]:
        """Retrieve a behavior insight by ID."""
        return self.insight_index.get(insight_id)

    # ========================================================================
    # COMMUNICATION EVENT METHODS
    # ========================================================================

    async def add_communication_event(self, event: PlantCommunicationEvent) -> None:
        """Add a communication event to the index."""
        self.communication_index[event.event_id] = event

        # Update maturity
        self.maturity_state.communication_event_count = len(self.communication_index)
        await self._update_maturity()

    async def get_communication_event(self, event_id: str) -> Optional[PlantCommunicationEvent]:
        """Retrieve a communication event by ID."""
        return self.communication_index.get(event_id)

    # ========================================================================
    # INDIGENOUS WISDOM METHODS
    # ========================================================================

    async def add_indigenous_wisdom(self, wisdom: IndigenousPlantWisdom) -> None:
        """Add indigenous plant wisdom to the index."""
        self.indigenous_wisdom_index[wisdom.wisdom_id] = wisdom

        # Update tradition index
        if wisdom.tradition in self.tradition_index:
            self.tradition_index[wisdom.tradition].append(wisdom.wisdom_id)

        # Update maturity
        self.maturity_state.indigenous_wisdom_count = len(self.indigenous_wisdom_index)
        await self._update_maturity()

    async def get_indigenous_wisdom(
        self,
        wisdom_id: Optional[str] = None,
        tradition: Optional[IndigenousTraditionType] = None,
        plant_name: Optional[str] = None
    ) -> List[IndigenousPlantWisdom]:
        """
        Retrieve indigenous wisdom records with optional filters.

        Args:
            wisdom_id: Specific wisdom ID to retrieve
            tradition: Filter by tradition type
            plant_name: Filter by plant name (partial match)

        Returns:
            List of matching wisdom records
        """
        if wisdom_id:
            wisdom = self.indigenous_wisdom_index.get(wisdom_id)
            return [wisdom] if wisdom else []

        results = []

        if tradition:
            wisdom_ids = self.tradition_index.get(tradition, [])
            results = [
                self.indigenous_wisdom_index[wid]
                for wid in wisdom_ids
                if wid in self.indigenous_wisdom_index
            ]
        else:
            results = list(self.indigenous_wisdom_index.values())

        if plant_name:
            plant_lower = plant_name.lower()
            results = [
                w for w in results
                if plant_lower in w.plant_name.lower()
            ]

        # Filter out sacred/restricted unless explicitly public
        results = [w for w in results if w.public_knowledge and not w.sacred_restricted]

        return results

    # ========================================================================
    # LEARNING EXPERIMENT METHODS
    # ========================================================================

    async def add_learning_experiment(self, experiment: PlantLearningExperiment) -> None:
        """Add a learning experiment to the index."""
        self.experiment_index[experiment.experiment_id] = experiment

        # Update maturity
        self.maturity_state.learning_experiment_count = len(self.experiment_index)
        await self._update_maturity()

    async def get_learning_experiment(self, experiment_id: str) -> Optional[PlantLearningExperiment]:
        """Retrieve a learning experiment by ID."""
        return self.experiment_index.get(experiment_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.species_profile_count +
            self.maturity_state.behavior_insight_count +
            self.maturity_state.communication_event_count +
            self.maturity_state.indigenous_wisdom_count +
            self.maturity_state.learning_experiment_count
        )

        # Simple maturity calculation (can be refined)
        target_items = 500  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update domain coverage
        for domain in PlantCognitionDomain:
            count = len(self.cognition_domain_index.get(domain, []))
            target_per_domain = 15
            self.maturity_state.domain_coverage[domain.value] = min(
                1.0, count / target_per_domain
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> PlantIntelligenceMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_species_profiles(self) -> List[Dict[str, Any]]:
        """Return seed species profiles for initialization."""
        return [
            # === SENSITIVE PLANTS ===
            {
                "species_id": "mimosa_pudica",
                "common_name": "Sensitive Plant",
                "scientific_name": "Mimosa pudica",
                "taxonomic_group": PlantTaxonomicGroup.SENSITIVE_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.MEMORY_HABITUATION: 0.95,
                    PlantCognitionDomain.PROPRIOCEPTION: 0.90,
                    PlantCognitionDomain.DEFENSE_COORDINATION: 0.80,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.TOUCH_MECHANOSENSING,
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.GRAVITY_GRAVISENSING,
                ],
                "notable_behaviors": [
                    "Rapid leaf folding in response to touch",
                    "Habituation to repeated non-threatening stimuli",
                    "Memory retention for 28+ days",
                    "Stimulus-specific learning",
                ],
                "indigenous_perspectives": [
                    "Used in traditional medicine across tropical regions",
                    "Associated with sensitivity and awareness in folk traditions",
                ],
                "research_evidence": [
                    "Gagliano et al. 2014 - Habituation learning demonstration",
                    "Multiple replication studies confirming memory",
                ],
                "key_researchers": ["Monica Gagliano", "Stefano Mancuso"],
            },
            # === CARNIVOROUS PLANTS ===
            {
                "species_id": "dionaea_muscipula",
                "common_name": "Venus Flytrap",
                "scientific_name": "Dionaea muscipula",
                "taxonomic_group": PlantTaxonomicGroup.CARNIVOROUS_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.DECISION_MAKING: 0.90,
                    PlantCognitionDomain.MEMORY_HABITUATION: 0.85,
                    PlantCognitionDomain.PROPRIOCEPTION: 0.95,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.TOUCH_MECHANOSENSING,
                    PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
                ],
                "notable_behaviors": [
                    "Action potential generation for trap closure",
                    "Counting of trigger hair stimulations",
                    "Cost-benefit optimization avoiding false closures",
                    "Digestion timing based on prey size",
                ],
                "indigenous_perspectives": [
                    "Native to Carolina bogs, limited traditional knowledge",
                ],
                "research_evidence": [
                    "Burdon-Sanderson 1873 - First plant action potential",
                    "Forterre et al. 2005 - Snap-trap mechanics",
                    "Hedrich & Neher 2018 - Electrical signaling",
                ],
                "key_researchers": ["Rainer Hedrich", "Erwin Neher"],
            },
            # === LEGUMES ===
            {
                "species_id": "pisum_sativum",
                "common_name": "Pea Plant",
                "scientific_name": "Pisum sativum",
                "taxonomic_group": PlantTaxonomicGroup.LEGUMES,
                "cognition_domains": {
                    PlantCognitionDomain.ASSOCIATIVE_LEARNING: 0.90,
                    PlantCognitionDomain.FORAGING_OPTIMIZATION: 0.85,
                    PlantCognitionDomain.SYMBIOTIC_INTELLIGENCE: 0.80,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.SOUND_VIBRATION,
                    PlantSensoryModality.GRAVITY_GRAVISENSING,
                ],
                "notable_behaviors": [
                    "Associative learning (Pavlovian conditioning)",
                    "Growth toward fan predicting light",
                    "Root-rhizobium symbiosis management",
                ],
                "indigenous_perspectives": [
                    "Ancient cultivar with deep agricultural traditions",
                ],
                "research_evidence": [
                    "Gagliano et al. 2016 - Associative learning demonstration",
                    "Landmark experiment showing Pavlovian conditioning",
                ],
                "key_researchers": ["Monica Gagliano"],
            },
            # === TREES - DECIDUOUS ===
            {
                "species_id": "pseudotsuga_menziesii",
                "common_name": "Douglas Fir / Mother Tree",
                "scientific_name": "Pseudotsuga menziesii",
                "taxonomic_group": PlantTaxonomicGroup.TREES_CONIFEROUS,
                "cognition_domains": {
                    PlantCognitionDomain.MYCORRHIZAL_NETWORKING: 0.95,
                    PlantCognitionDomain.KIN_RECOGNITION: 0.90,
                    PlantCognitionDomain.SOCIAL_BEHAVIOR: 0.90,
                    PlantCognitionDomain.RESOURCE_ALLOCATION: 0.85,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.NUTRIENT_SENSING,
                ],
                "notable_behaviors": [
                    "Carbon sharing through mycorrhizal networks",
                    "Recognition and preferential nurturing of offspring",
                    "Resource transfer to stressed neighbors",
                    "Legacy resource transfer before death",
                ],
                "indigenous_perspectives": [
                    "Sacred in Pacific Northwest indigenous traditions",
                    "Recognition of forest as interconnected community",
                ],
                "research_evidence": [
                    "Simard et al. 1997 - Mycorrhizal carbon transfer",
                    "Simard 2021 - Finding the Mother Tree",
                ],
                "key_researchers": ["Suzanne Simard"],
            },
            # === ENTHEOGENIC PLANTS ===
            {
                "species_id": "banisteriopsis_caapi",
                "common_name": "Ayahuasca Vine",
                "scientific_name": "Banisteriopsis caapi",
                "taxonomic_group": PlantTaxonomicGroup.ENTHEOGENIC_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.CHEMICAL_SIGNALING: 0.85,
                    PlantCognitionDomain.SYMBIOTIC_INTELLIGENCE: 0.80,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.TOUCH_MECHANOSENSING,
                ],
                "notable_behaviors": [
                    "Climbing vine behavior",
                    "Complex alkaloid production (MAOIs)",
                    "Synergistic pairing with DMT-containing plants",
                ],
                "indigenous_perspectives": [
                    "Master plant teacher in Amazonian traditions",
                    "Central to Shipibo, Ashaninka healing practices",
                    "Requires dieta relationship for proper use",
                    "Considered conscious being capable of teaching",
                ],
                "research_evidence": [
                    "Ethnobotanical documentation extensive",
                    "Pharmacological research on harmala alkaloids",
                ],
                "key_researchers": ["Richard Evans Schultes", "Dennis McKenna"],
                "cultural_significance": "Sacred plant teacher in Amazonian shamanism, central to ayahuasca brew",
            },
            {
                "species_id": "nicotiana_rustica",
                "common_name": "Sacred Tobacco / Mapacho",
                "scientific_name": "Nicotiana rustica",
                "taxonomic_group": PlantTaxonomicGroup.ENTHEOGENIC_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.DEFENSE_COORDINATION: 0.90,
                    PlantCognitionDomain.VOLATILE_COMMUNICATION: 0.85,
                    PlantCognitionDomain.CHEMICAL_SIGNALING: 0.90,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.DAMAGE_WOUNDING,
                    PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
                ],
                "notable_behaviors": [
                    "Volatile signaling to attract predators of herbivores",
                    "Nicotine production as sophisticated defense",
                    "Systemic acquired resistance",
                ],
                "indigenous_perspectives": [
                    "Sacred plant across Native American traditions",
                    "Used for prayer, ceremony, protection, cleansing",
                    "Considered masculine, protective plant spirit",
                    "Not recreational - deeply sacramental",
                ],
                "research_evidence": [
                    "Baldwin lab - Volatile signaling research",
                    "Model organism for plant defense studies",
                ],
                "key_researchers": ["Ian Baldwin"],
                "cultural_significance": "Sacred medicine plant in Native American and Amazonian traditions",
            },
            {
                "species_id": "lophophora_williamsii",
                "common_name": "Peyote",
                "scientific_name": "Lophophora williamsii",
                "taxonomic_group": PlantTaxonomicGroup.ENTHEOGENIC_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.STRESS_INTEGRATION: 0.85,
                    PlantCognitionDomain.CHEMICAL_SIGNALING: 0.90,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.TEMPERATURE_THERMOSENSING,
                    PlantSensoryModality.WATER_HYDROTROPISM,
                ],
                "notable_behaviors": [
                    "Extreme slow growth adaptation",
                    "Complex mescaline alkaloid production",
                    "Desert stress tolerance",
                ],
                "indigenous_perspectives": [
                    "Sacred sacrament of Native American Church",
                    "Centuries of ceremonial use by Huichol, Tarahumara",
                    "Peyote as teacher and healer spirit",
                    "Protected religious use in United States",
                ],
                "research_evidence": [
                    "Extensive ethnobotanical documentation",
                    "Conservation concerns due to overharvesting",
                ],
                "key_researchers": ["Weston La Barre", "Peter Furst"],
                "cultural_significance": "Legal sacrament for Native American Church; centuries of indigenous use",
            },
            {
                "species_id": "echinopsis_pachanoi",
                "common_name": "San Pedro Cactus",
                "scientific_name": "Echinopsis pachanoi",
                "taxonomic_group": PlantTaxonomicGroup.ENTHEOGENIC_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.STRESS_MEMORY: 0.80,
                    PlantCognitionDomain.CIRCADIAN_AWARENESS: 0.75,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.TEMPERATURE_THERMOSENSING,
                ],
                "notable_behaviors": [
                    "CAM photosynthesis timing",
                    "Mescaline production",
                    "Fast-growing for a cactus",
                ],
                "indigenous_perspectives": [
                    "3000+ years of ceremonial use in Andes",
                    "Named for Saint Peter as gatekeeper",
                    "Used in traditional healing ceremonies",
                ],
                "research_evidence": [
                    "Archaeological evidence of ancient use",
                ],
                "key_researchers": ["Douglas Sharon"],
                "cultural_significance": "Ancient Andean medicine plant with continuous ceremonial tradition",
            },
            # === VINES AND CLIMBERS ===
            {
                "species_id": "cuscuta_spp",
                "common_name": "Dodder",
                "scientific_name": "Cuscuta spp.",
                "taxonomic_group": PlantTaxonomicGroup.PARASITIC_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.DECISION_MAKING: 0.90,
                    PlantCognitionDomain.CHEMORECEPTION: 0.90,
                    PlantCognitionDomain.FORAGING_OPTIMIZATION: 0.85,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
                    PlantSensoryModality.TOUCH_MECHANOSENSING,
                ],
                "notable_behaviors": [
                    "Host plant evaluation and selection",
                    "Quality assessment of potential hosts",
                    "Decision to accept or reject hosts",
                    "Comparison of multiple hosts before committing",
                ],
                "indigenous_perspectives": [
                    "Known in traditional herbalism",
                ],
                "research_evidence": [
                    "Host selection experiments demonstrating decision-making",
                ],
                "key_researchers": ["Consuelo De Moraes"],
            },
            # === AGRICULTURAL CROPS ===
            {
                "species_id": "solanum_lycopersicum",
                "common_name": "Tomato",
                "scientific_name": "Solanum lycopersicum",
                "taxonomic_group": PlantTaxonomicGroup.AGRICULTURAL_CROPS,
                "cognition_domains": {
                    PlantCognitionDomain.VOLATILE_COMMUNICATION: 0.90,
                    PlantCognitionDomain.DEFENSE_COORDINATION: 0.85,
                    PlantCognitionDomain.ROOT_COMMUNICATION: 0.80,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.DAMAGE_WOUNDING,
                    PlantSensoryModality.CHEMICAL_CHEMORECEPTION,
                ],
                "notable_behaviors": [
                    "Release of volatile alarm signals when damaged",
                    "Priming of neighboring plant defenses",
                    "Systemic acquired resistance",
                ],
                "indigenous_perspectives": [
                    "Mesoamerican origin with rich cultural history",
                ],
                "research_evidence": [
                    "Talking trees experiments - early VOC research",
                ],
                "key_researchers": ["Jack Schultz", "Ian Baldwin"],
            },
            # === ANCIENT LINEAGES ===
            {
                "species_id": "ginkgo_biloba",
                "common_name": "Ginkgo / Maidenhair Tree",
                "scientific_name": "Ginkgo biloba",
                "taxonomic_group": PlantTaxonomicGroup.ANCIENT_LINEAGES,
                "cognition_domains": {
                    PlantCognitionDomain.STRESS_INTEGRATION: 0.85,
                    PlantCognitionDomain.SEASONAL_TIMING: 0.90,
                    PlantCognitionDomain.DEFENSE_COORDINATION: 0.80,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.TEMPERATURE_THERMOSENSING,
                ],
                "notable_behaviors": [
                    "270 million year survival",
                    "Remarkable stress tolerance (survived Hiroshima)",
                    "Coordinated autumn leaf drop",
                ],
                "indigenous_perspectives": [
                    "Sacred in Chinese and Japanese Buddhist traditions",
                    "Temple trees planted for centuries",
                    "Symbol of longevity and resilience",
                ],
                "research_evidence": [
                    "Extensive research on ginkgolides and stress tolerance",
                ],
                "key_researchers": [],
                "cultural_significance": "Living fossil; sacred temple tree in East Asian traditions",
            },
            # === SACRED PLANTS ===
            {
                "species_id": "ficus_religiosa",
                "common_name": "Bodhi Tree / Sacred Fig",
                "scientific_name": "Ficus religiosa",
                "taxonomic_group": PlantTaxonomicGroup.SACRED_PLANTS,
                "cognition_domains": {
                    PlantCognitionDomain.MYCORRHIZAL_NETWORKING: 0.80,
                    PlantCognitionDomain.SYMBIOTIC_INTELLIGENCE: 0.85,
                },
                "sensory_capabilities": [
                    PlantSensoryModality.LIGHT_PHOTORECEPTION,
                    PlantSensoryModality.GRAVITY_GRAVISENSING,
                ],
                "notable_behaviors": [
                    "Fig-wasp mutualism",
                    "Strangler fig growth patterns",
                    "Long lifespan",
                ],
                "indigenous_perspectives": [
                    "Tree of Buddha's enlightenment",
                    "Sacred in Hindu, Jain, Buddhist traditions",
                    "Temple plantings worldwide",
                ],
                "research_evidence": [
                    "Fig-wasp coevolution studies",
                ],
                "key_researchers": [],
                "cultural_significance": "Tree under which Buddha attained enlightenment; sacred across South Asian religions",
            },
        ]

    def _get_seed_indigenous_wisdom(self) -> List[Dict[str, Any]]:
        """Return seed indigenous wisdom for initialization."""
        return [
            {
                "wisdom_id": "ayahuasca_dieta",
                "plant_name": "Ayahuasca",
                "scientific_name": "Banisteriopsis caapi",
                "tradition": IndigenousTraditionType.DIETA_TRADITION,
                "spiritual_significance": "The master plant teacher, considered a conscious being capable of revealing hidden knowledge, healing, and spiritual guidance through visions and dreams.",
                "practical_uses": [
                    "Healing ceremonies for physical and psychological ailments",
                    "Divination and spiritual insight",
                    "Training of healers (curanderos/vegetalistas)",
                ],
                "ceremonial_role": "Central to Amazonian healing ceremonies; requires experienced guide (ayahuasquero)",
                "harvesting_protocols": [
                    "Harvested with prayers and offerings",
                    "Specific lunar timing observed",
                    "Permission asked from plant spirit",
                ],
                "relationship_principles": [
                    "Dieta required - dietary and behavioral restrictions",
                    "Long-term relationship developed over years",
                    "Plant teaches through dreams and visions",
                    "Reciprocity through offerings and respect",
                ],
                "source_community": "Shipibo-Conibo, Ashuar, various Amazonian peoples",
                "public_knowledge": True,
                "sacred_restricted": False,
            },
            {
                "wisdom_id": "celtic_oak",
                "plant_name": "Oak",
                "scientific_name": "Quercus spp.",
                "tradition": IndigenousTraditionType.CELTIC_OGHAM,
                "spiritual_significance": "Duir in Ogham alphabet; represents strength, doorways, and connection to divine. Sacred to Druids as seat of wisdom.",
                "practical_uses": [
                    "Sacred groves for ceremony and teaching",
                    "Acorns as food and medicine",
                    "Wood for sacred fires",
                ],
                "ceremonial_role": "Center of Druidic practice; doorway between worlds",
                "harvesting_protocols": [
                    "Mistletoe harvested from oak with golden sickle",
                    "Specific timing (sixth night of moon)",
                ],
                "relationship_principles": [
                    "Trees as teachers and ancestors",
                    "Sacred groves protected",
                    "Oak as king of forest, center of wisdom",
                ],
                "source_community": "Celtic peoples - Irish, Welsh, Scottish, Breton",
                "public_knowledge": True,
                "sacred_restricted": False,
            },
            {
                "wisdom_id": "vedic_tulsi",
                "plant_name": "Tulsi / Holy Basil",
                "scientific_name": "Ocimum tenuiflorum",
                "tradition": IndigenousTraditionType.VEDIC_PLANT_CONSCIOUSNESS,
                "spiritual_significance": "Sacred to Vishnu; considered a living goddess (Tulsi Devi). Purifying presence in home and temple.",
                "practical_uses": [
                    "Daily worship and offerings",
                    "Ayurvedic medicine (adaptogen)",
                    "Protection of household",
                    "Sanctification of food and water",
                ],
                "ceremonial_role": "Central to Hindu household worship; marriage ceremonies",
                "harvesting_protocols": [
                    "Never harvested on certain days",
                    "Prayers offered before taking leaves",
                    "Specific persons may harvest",
                ],
                "relationship_principles": [
                    "Treated as living deity, not just plant",
                    "Daily care and worship",
                    "Reciprocal relationship - plant gives protection, devotee gives care",
                ],
                "source_community": "Hindu traditions across South Asia",
                "public_knowledge": True,
                "sacred_restricted": False,
            },
            {
                "wisdom_id": "three_sisters",
                "plant_name": "Three Sisters (Corn, Beans, Squash)",
                "scientific_name": "Zea mays, Phaseolus vulgaris, Cucurbita spp.",
                "tradition": IndigenousTraditionType.HAUDENOSAUNEE_THREE_SISTERS,
                "spiritual_significance": "Three sister spirits who support each other and humanity. Gift from Sky Woman. Model of cooperation and mutual support.",
                "practical_uses": [
                    "Companion planting for sustainable agriculture",
                    "Nutritionally complete food system",
                    "Teaching model for cooperation",
                ],
                "ceremonial_role": "Green Corn Ceremony; planting and harvest ceremonies",
                "harvesting_protocols": [
                    "First fruits offered in ceremony",
                    "Thanks given to plant spirits",
                    "Seeds saved with prayers",
                ],
                "relationship_principles": [
                    "Plants as teachers of cooperation",
                    "Human responsibility to continue relationship",
                    "Reciprocity through ceremony and care",
                ],
                "source_community": "Haudenosaunee (Iroquois) and many Eastern Woodlands peoples",
                "public_knowledge": True,
                "sacred_restricted": False,
            },
        ]

    async def initialize_seed_species(self) -> int:
        """Initialize with seed species profiles."""
        seed_profiles = self._get_seed_species_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = PlantSpeciesProfile(
                species_id=profile_data["species_id"],
                common_name=profile_data["common_name"],
                scientific_name=profile_data["scientific_name"],
                taxonomic_group=profile_data["taxonomic_group"],
                cognition_domains=profile_data.get("cognition_domains", {}),
                sensory_capabilities=profile_data.get("sensory_capabilities", []),
                notable_behaviors=profile_data.get("notable_behaviors", []),
                indigenous_perspectives=profile_data.get("indigenous_perspectives", []),
                research_evidence=profile_data.get("research_evidence", []),
                key_researchers=profile_data.get("key_researchers", []),
                cultural_significance=profile_data.get("cultural_significance"),
                maturity_level=MaturityLevel.DEVELOPING,
                created_at=datetime.now(timezone.utc),
            )
            await self.add_species_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed species profiles")
        return count

    async def initialize_seed_wisdom(self) -> int:
        """Initialize with seed indigenous wisdom."""
        seed_wisdom = self._get_seed_indigenous_wisdom()
        count = 0

        for wisdom_data in seed_wisdom:
            wisdom = IndigenousPlantWisdom(
                wisdom_id=wisdom_data["wisdom_id"],
                plant_name=wisdom_data["plant_name"],
                scientific_name=wisdom_data.get("scientific_name"),
                tradition=wisdom_data["tradition"],
                spiritual_significance=wisdom_data["spiritual_significance"],
                practical_uses=wisdom_data.get("practical_uses", []),
                ceremonial_role=wisdom_data.get("ceremonial_role"),
                harvesting_protocols=wisdom_data.get("harvesting_protocols", []),
                relationship_principles=wisdom_data.get("relationship_principles", []),
                source_community=wisdom_data.get("source_community"),
                public_knowledge=wisdom_data.get("public_knowledge", True),
                sacred_restricted=wisdom_data.get("sacred_restricted", False),
            )
            await self.add_indigenous_wisdom(wisdom)
            count += 1

        logger.info(f"Initialized {count} seed indigenous wisdom records")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        species_count = await self.initialize_seed_species()
        wisdom_count = await self.initialize_seed_wisdom()

        return {
            "species_profiles": species_count,
            "indigenous_wisdom": wisdom_count,
            "total": species_count + wisdom_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "PlantCognitionDomain",
    "PlantTaxonomicGroup",
    "PlantSensoryModality",
    "ResearchParadigm",
    "IndigenousTraditionType",
    "PlantSignalingType",
    "MaturityLevel",
    # Dataclasses
    "PlantSpeciesProfile",
    "PlantBehaviorInsight",
    "PlantCommunicationEvent",
    "IndigenousPlantWisdom",
    "PlantLearningExperiment",
    "PlantIntelligenceMaturityState",
    # Interface
    "PlantIntelligenceInterface",
]
