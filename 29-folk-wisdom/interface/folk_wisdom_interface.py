#!/usr/bin/env python3
"""
Folk & Indigenous Wisdom Interface

Form 29: The comprehensive interface for folk traditions, indigenous wisdom,
and animistic practices from cultures worldwide. This form bridges formal
philosophical inquiry with lived wisdom embedded in oral traditions,
ceremonial practices, and traditional ecological knowledge.

Ethical Principles:
- Source attribution for all teachings
- Cultural context preservation
- Recognition of living traditions
- Respect for sacred boundaries
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

class FolkTraditionRegion(Enum):
    """
    Geographic regions for folk and indigenous traditions.

    Organized by continent and sub-region without implying hierarchy.
    Each region contains multiple distinct cultural traditions.
    """

    # === AFRICA ===
    WEST_AFRICAN = "west_african"  # Yoruba, Akan, Dogon, Fon
    EAST_AFRICAN = "east_african"  # Maasai, Swahili coast, Ethiopian
    SOUTHERN_AFRICAN = "southern_african"  # Zulu, San, Shona
    CENTRAL_AFRICAN = "central_african"  # Kongo, Pygmy/BaAka
    NORTH_AFRICAN_BERBER = "north_african_berber"  # Amazigh traditions

    # === EUROPE ===
    CELTIC = "celtic"  # Irish, Welsh, Scottish, Breton
    NORSE_GERMANIC = "norse_germanic"  # Norse, Anglo-Saxon, Continental
    SLAVIC = "slavic"  # Russian, Polish, Ukrainian, Balkan
    BALTIC = "baltic"  # Lithuanian, Latvian
    MEDITERRANEAN_FOLK = "mediterranean_folk"  # Greek, Italian folk
    FINNO_UGRIC = "finno_ugric"  # Finnish, Sami, Hungarian

    # === ASIA ===
    SIBERIAN = "siberian"  # Yakut, Buryat, Evenki, Chukchi
    CENTRAL_ASIAN = "central_asian"  # Kazakh, Mongolian, Kyrgyz
    SOUTHEAST_ASIAN = "southeast_asian"  # Thai, Balinese, Philippine
    EAST_ASIAN_FOLK = "east_asian_folk"  # Chinese, Japanese, Korean folk
    SOUTH_ASIAN_FOLK = "south_asian_folk"  # Village and tribal traditions

    # === OCEANIA ===
    POLYNESIAN = "polynesian"  # Hawaiian, Maori, Samoan, Tongan
    MELANESIAN = "melanesian"  # Papua New Guinea, Fiji
    MICRONESIAN = "micronesian"  # Chamorro and island traditions
    ABORIGINAL_AUSTRALIAN = "aboriginal_australian"  # Diverse language groups
    MAORI = "maori"  # Aotearoa/New Zealand specifically

    # === AMERICAS ===
    ARCTIC_INDIGENOUS = "arctic_indigenous"  # Inuit, Yup'ik
    PACIFIC_NORTHWEST = "pacific_northwest"  # Tlingit, Haida, Coast Salish
    PLAINS_INDIGENOUS = "plains_indigenous"  # Lakota, Cheyenne, Blackfoot
    EASTERN_WOODLANDS = "eastern_woodlands"  # Haudenosaunee, Cherokee, Anishinaabe
    MESOAMERICAN_FOLK = "mesoamerican_folk"  # Maya, Nahua contemporary
    AMAZONIAN = "amazonian"  # Shipibo, Yanomami, diverse groups
    ANDEAN_FOLK = "andean_folk"  # Quechua, Aymara village traditions


class WisdomTransmissionMode(Enum):
    """How wisdom is transmitted in folk traditions."""
    ORAL_NARRATIVE = "oral_narrative"  # Stories, myths, legends
    SONG_CHANT = "song_chant"  # Musical transmission
    RITUAL_CEREMONY = "ritual_ceremony"  # Ceremonial embedding
    DANCE_MOVEMENT = "dance_movement"  # Embodied transmission
    VISUAL_SYMBOLIC = "visual_symbolic"  # Art, symbols, sacred objects
    MATERIAL_CRAFT = "material_craft"  # Making and craftsmanship
    APPRENTICESHIP = "apprenticeship"  # One-on-one transmission
    DREAM_VISION = "dream_vision"  # Visionary/revelatory
    ELDER_TEACHING = "elder_teaching"  # Generational transmission


class AnimisticDomain(Enum):
    """Domains of animistic relationship and practice."""
    NATURE_SPIRITS = "nature_spirits"  # Spirits of places, elements
    ANCESTOR_RELATIONS = "ancestor_relations"  # Ongoing ancestor connection
    ANIMAL_POWERS = "animal_powers"  # Animal spirits, totems
    PLANT_INTELLIGENCE = "plant_intelligence"  # Plant spirits, medicine
    ELEMENTAL_FORCES = "elemental_forces"  # Water, fire, earth, air
    LAND_CONSCIOUSNESS = "land_consciousness"  # Living landscape
    CELESTIAL_BEINGS = "celestial_beings"  # Sun, moon, stars, sky
    UNDERWORLD_ENTITIES = "underworld_entities"  # Chthonic beings
    THRESHOLD_BEINGS = "threshold_beings"  # Liminal spirits


class EthicalPrinciple(Enum):
    """Core ethical principles found across folk traditions."""
    RECIPROCITY = "reciprocity"  # Give-and-take balance
    HOSPITALITY = "hospitality"  # Treatment of guests/strangers
    ANCESTOR_HONOR = "ancestor_honor"  # Duties to ancestors
    COMMUNITY_HARMONY = "community_harmony"  # Social cohesion
    NATURE_RESPECT = "nature_respect"  # Human-nature relationship
    SACRED_PROHIBITION = "sacred_prohibition"  # Taboo systems
    GENEROSITY = "generosity"  # Sharing and giving
    TRUTH_SPEAKING = "truth_speaking"  # Honesty and integrity
    ELDER_RESPECT = "elder_respect"  # Generational wisdom


class CosmologicalElement(Enum):
    """Elements of indigenous cosmologies."""
    WORLD_TREE = "world_tree"  # Axis mundi
    THREE_WORLDS = "three_worlds"  # Upper/middle/lower realms
    FOUR_DIRECTIONS = "four_directions"  # Cardinal orientations
    SACRED_MOUNTAIN = "sacred_mountain"  # Vertical axis
    COSMIC_WATERS = "cosmic_waters"  # Primordial waters
    WORLD_SERPENT = "world_serpent"  # Serpent symbolism
    CREATOR_BEING = "creator_being"  # First cause/creator
    TRICKSTER = "trickster"  # Transformative agent
    ETERNAL_RETURN = "eternal_return"  # Cyclical time


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
class FolkWisdomTeaching:
    """
    Represents a core wisdom teaching from a folk tradition.

    These are the philosophical insights and ethical principles
    embedded in traditional cultures, transmitted through various modes.
    """
    teaching_id: str
    name: str
    region: FolkTraditionRegion
    domains: List[AnimisticDomain]
    transmission_modes: List[WisdomTransmissionMode]
    core_teaching: str
    practical_applications: List[str] = field(default_factory=list)
    related_ceremonies: List[str] = field(default_factory=list)
    source_communities: List[str] = field(default_factory=list)
    ethical_principles: List[EthicalPrinciple] = field(default_factory=list)
    related_teachings: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Teaching: {self.name}",
            f"Region: {self.region.value}",
            f"Domains: {', '.join(d.value for d in self.domains)}",
            f"Core Teaching: {self.core_teaching}"
        ]
        return " | ".join(parts)


@dataclass
class AnimisticPractice:
    """
    Represents an animistic practice or ritual relationship.

    These are the ceremonial and practical ways that folk traditions
    engage with spirits, ancestors, and the natural world.
    """
    practice_id: str
    name: str
    region: FolkTraditionRegion
    domain: AnimisticDomain
    description: str
    purpose: str
    associated_beings: List[str] = field(default_factory=list)
    seasonal_timing: Optional[str] = None
    materials_used: List[str] = field(default_factory=list)
    taboos_restrictions: List[str] = field(default_factory=list)
    related_practices: List[str] = field(default_factory=list)
    practitioner_type: Optional[str] = None  # e.g., "shaman", "elder"
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class IndigenousCosmology:
    """
    Represents a traditional worldview and cosmological system.

    These are the underlying frameworks for understanding reality
    in folk traditions - creation narratives, spatial organization,
    and fundamental principles of existence.
    """
    cosmology_id: str
    name: str
    region: FolkTraditionRegion
    description: str
    creation_narrative: Optional[str] = None
    cosmological_elements: List[CosmologicalElement] = field(default_factory=list)
    principal_beings: List[str] = field(default_factory=list)
    spatial_organization: Optional[str] = None  # How space is organized
    temporal_conception: Optional[str] = None  # How time is understood
    human_place: Optional[str] = None  # Human role in cosmos
    related_cosmologies: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class OralTradition:
    """
    Represents stories, songs, proverbs, and their embedded wisdom.

    Oral traditions are the primary vehicle for transmitting folk
    wisdom across generations, encoding philosophy in narrative form.
    """
    tradition_id: str
    name: str
    region: FolkTraditionRegion
    tradition_type: str  # story, song, proverb, riddle, etc.
    content_summary: str
    embedded_wisdom: List[str] = field(default_factory=list)
    key_characters: List[str] = field(default_factory=list)
    transmission_context: Optional[str] = None
    performance_requirements: List[str] = field(default_factory=list)
    related_traditions: List[str] = field(default_factory=list)
    ethical_principles: List[EthicalPrinciple] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class IndigenousAnimalWisdom:
    """
    Traditional knowledge about animals from indigenous perspectives.

    This represents the sophisticated understanding of animal behavior,
    meaning, and relationship that indigenous cultures have developed
    through millennia of observation and interaction.
    """
    wisdom_id: str
    animal_name: str
    indigenous_name: Optional[str] = None
    region: FolkTraditionRegion = FolkTraditionRegion.WEST_AFRICAN
    spiritual_significance: str = ""
    behavioral_observations: List[str] = field(default_factory=list)
    human_relationship: str = ""
    stories_myths: List[str] = field(default_factory=list)
    practical_knowledge: List[str] = field(default_factory=list)  # Hunting, medicine, ecology
    symbolic_meanings: List[str] = field(default_factory=list)
    taboos_restrictions: List[str] = field(default_factory=list)
    related_species: List[str] = field(default_factory=list)
    form_30_link: Optional[str] = None  # Link to animal cognition profile
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FolkWisdomMaturityState:
    """Tracks the maturity of folk wisdom knowledge."""
    overall_maturity: float = 0.0
    regional_coverage: Dict[str, float] = field(default_factory=dict)
    teaching_count: int = 0
    practice_count: int = 0
    cosmology_count: int = 0
    oral_tradition_count: int = 0
    animal_wisdom_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class FolkWisdomInterface:
    """
    Main interface for Form 29: Folk & Indigenous Wisdom.

    Provides methods for storing, retrieving, and querying folk wisdom
    teachings, animistic practices, indigenous cosmologies, and oral
    traditions from cultures worldwide.
    """

    FORM_ID = "29-folk-wisdom"
    FORM_NAME = "Folk & Indigenous Wisdom"

    def __init__(self):
        """Initialize the Folk Wisdom Interface."""
        # Knowledge indexes
        self.teaching_index: Dict[str, FolkWisdomTeaching] = {}
        self.practice_index: Dict[str, AnimisticPractice] = {}
        self.cosmology_index: Dict[str, IndigenousCosmology] = {}
        self.oral_tradition_index: Dict[str, OralTradition] = {}
        self.animal_wisdom_index: Dict[str, IndigenousAnimalWisdom] = {}

        # Cross-reference indexes
        self.region_index: Dict[FolkTraditionRegion, List[str]] = {}
        self.domain_index: Dict[AnimisticDomain, List[str]] = {}

        # Maturity tracking
        self.maturity_state = FolkWisdomMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize region index
        for region in FolkTraditionRegion:
            self.region_index[region] = []

        # Initialize domain index
        for domain in AnimisticDomain:
            self.domain_index[domain] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # TEACHING METHODS
    # ========================================================================

    async def add_teaching(self, teaching: FolkWisdomTeaching) -> None:
        """Add a wisdom teaching to the index."""
        self.teaching_index[teaching.teaching_id] = teaching

        # Update region index
        if teaching.region in self.region_index:
            self.region_index[teaching.region].append(teaching.teaching_id)

        # Update domain index
        for domain in teaching.domains:
            if domain in self.domain_index:
                self.domain_index[domain].append(teaching.teaching_id)

        # Update maturity
        self.maturity_state.teaching_count = len(self.teaching_index)
        await self._update_maturity()

    async def get_teaching(self, teaching_id: str) -> Optional[FolkWisdomTeaching]:
        """Retrieve a teaching by ID."""
        return self.teaching_index.get(teaching_id)

    async def query_teachings_by_region(
        self,
        region: FolkTraditionRegion,
        limit: int = 10
    ) -> List[FolkWisdomTeaching]:
        """Query teachings by geographic region."""
        teaching_ids = self.region_index.get(region, [])[:limit]
        return [
            self.teaching_index[tid]
            for tid in teaching_ids
            if tid in self.teaching_index
        ]

    async def query_teachings_by_domain(
        self,
        domain: AnimisticDomain,
        limit: int = 10
    ) -> List[FolkWisdomTeaching]:
        """Query teachings by animistic domain."""
        teaching_ids = self.domain_index.get(domain, [])[:limit]
        return [
            self.teaching_index[tid]
            for tid in teaching_ids
            if tid in self.teaching_index
        ]

    # ========================================================================
    # PRACTICE METHODS
    # ========================================================================

    async def add_practice(self, practice: AnimisticPractice) -> None:
        """Add an animistic practice to the index."""
        self.practice_index[practice.practice_id] = practice

        # Update region index
        if practice.region in self.region_index:
            self.region_index[practice.region].append(practice.practice_id)

        # Update domain index
        if practice.domain in self.domain_index:
            self.domain_index[practice.domain].append(practice.practice_id)

        # Update maturity
        self.maturity_state.practice_count = len(self.practice_index)
        await self._update_maturity()

    async def get_practice(self, practice_id: str) -> Optional[AnimisticPractice]:
        """Retrieve a practice by ID."""
        return self.practice_index.get(practice_id)

    # ========================================================================
    # COSMOLOGY METHODS
    # ========================================================================

    async def add_cosmology(self, cosmology: IndigenousCosmology) -> None:
        """Add a cosmology to the index."""
        self.cosmology_index[cosmology.cosmology_id] = cosmology

        # Update region index
        if cosmology.region in self.region_index:
            self.region_index[cosmology.region].append(cosmology.cosmology_id)

        # Update maturity
        self.maturity_state.cosmology_count = len(self.cosmology_index)
        await self._update_maturity()

    async def get_cosmology(self, cosmology_id: str) -> Optional[IndigenousCosmology]:
        """Retrieve a cosmology by ID."""
        return self.cosmology_index.get(cosmology_id)

    # ========================================================================
    # ORAL TRADITION METHODS
    # ========================================================================

    async def add_oral_tradition(self, tradition: OralTradition) -> None:
        """Add an oral tradition to the index."""
        self.oral_tradition_index[tradition.tradition_id] = tradition

        # Update region index
        if tradition.region in self.region_index:
            self.region_index[tradition.region].append(tradition.tradition_id)

        # Update maturity
        self.maturity_state.oral_tradition_count = len(self.oral_tradition_index)
        await self._update_maturity()

    async def get_oral_tradition(self, tradition_id: str) -> Optional[OralTradition]:
        """Retrieve an oral tradition by ID."""
        return self.oral_tradition_index.get(tradition_id)

    # ========================================================================
    # ANIMAL WISDOM METHODS
    # ========================================================================

    async def add_animal_wisdom(self, wisdom: IndigenousAnimalWisdom) -> None:
        """Add animal wisdom to the index."""
        self.animal_wisdom_index[wisdom.wisdom_id] = wisdom

        # Update region index
        if wisdom.region in self.region_index:
            self.region_index[wisdom.region].append(wisdom.wisdom_id)

        # Update maturity
        self.maturity_state.animal_wisdom_count = len(self.animal_wisdom_index)
        await self._update_maturity()

    async def get_animal_wisdom(self, wisdom_id: str) -> Optional[IndigenousAnimalWisdom]:
        """Retrieve animal wisdom by ID."""
        return self.animal_wisdom_index.get(wisdom_id)

    async def query_animal_wisdom_by_animal(
        self,
        animal_name: str
    ) -> List[IndigenousAnimalWisdom]:
        """Query animal wisdom by animal name."""
        results = []
        animal_lower = animal_name.lower()
        for wisdom in self.animal_wisdom_index.values():
            if (animal_lower in wisdom.animal_name.lower() or
                (wisdom.indigenous_name and animal_lower in wisdom.indigenous_name.lower())):
                results.append(wisdom)
        return results

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.teaching_count +
            self.maturity_state.practice_count +
            self.maturity_state.cosmology_count +
            self.maturity_state.oral_tradition_count +
            self.maturity_state.animal_wisdom_count
        )

        # Simple maturity calculation (can be refined)
        target_items = 500  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update regional coverage
        for region in FolkTraditionRegion:
            count = len(self.region_index.get(region, []))
            target_per_region = 20
            self.maturity_state.regional_coverage[region.value] = min(
                1.0, count / target_per_region
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> FolkWisdomMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_teachings(self) -> List[Dict[str, Any]]:
        """Return seed teachings for initialization."""
        return [
            # West African
            {
                "teaching_id": "yoruba_ashe",
                "name": "Ashe",
                "region": FolkTraditionRegion.WEST_AFRICAN,
                "domains": [AnimisticDomain.NATURE_SPIRITS, AnimisticDomain.ELEMENTAL_FORCES],
                "transmission_modes": [WisdomTransmissionMode.RITUAL_CEREMONY, WisdomTransmissionMode.ORAL_NARRATIVE],
                "core_teaching": "Ashe is the vital force of creation that flows through all things. It is the power to make things happen, the dynamic energy behind existence. Through proper action, prayer, and offerings, humans can align with and channel ashe.",
                "source_communities": ["Yoruba", "Afro-Caribbean traditions"],
                "ethical_principles": [EthicalPrinciple.RECIPROCITY, EthicalPrinciple.NATURE_RESPECT],
            },
            {
                "teaching_id": "akan_sankofa",
                "name": "Sankofa",
                "region": FolkTraditionRegion.WEST_AFRICAN,
                "domains": [AnimisticDomain.ANCESTOR_RELATIONS],
                "transmission_modes": [WisdomTransmissionMode.VISUAL_SYMBOLIC, WisdomTransmissionMode.ORAL_NARRATIVE],
                "core_teaching": "It is not taboo to go back and fetch what you have forgotten. The wisdom of learning from the past to build the future. Symbolized by a bird reaching back for an egg on its tail.",
                "source_communities": ["Akan", "Asante"],
                "ethical_principles": [EthicalPrinciple.ANCESTOR_HONOR, EthicalPrinciple.COMMUNITY_HARMONY],
            },

            # Celtic
            {
                "teaching_id": "celtic_thin_places",
                "name": "Thin Places",
                "region": FolkTraditionRegion.CELTIC,
                "domains": [AnimisticDomain.LAND_CONSCIOUSNESS, AnimisticDomain.THRESHOLD_BEINGS],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.ELDER_TEACHING],
                "core_teaching": "There are places where the veil between this world and the Otherworld grows thin, where the sacred more readily penetrates the mundane. At wells, crossroads, and ancient sites, one can sense the presence of the spirits.",
                "source_communities": ["Irish", "Scottish", "Welsh"],
                "ethical_principles": [EthicalPrinciple.NATURE_RESPECT, EthicalPrinciple.SACRED_PROHIBITION],
            },

            # Norse/Germanic
            {
                "teaching_id": "norse_web_of_wyrd",
                "name": "Web of Wyrd",
                "region": FolkTraditionRegion.NORSE_GERMANIC,
                "domains": [AnimisticDomain.ELEMENTAL_FORCES],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.VISUAL_SYMBOLIC],
                "core_teaching": "All actions are woven into the web of fate by the Norns. Past actions (orlog) shape present possibilities, but one can still choose how to act within the pattern. Honor and right action strengthen one's thread in the web.",
                "source_communities": ["Norse", "Anglo-Saxon", "Germanic"],
                "ethical_principles": [EthicalPrinciple.TRUTH_SPEAKING, EthicalPrinciple.HOSPITALITY],
            },

            # Slavic
            {
                "teaching_id": "slavic_domovoi",
                "name": "Domovoi House Spirit",
                "region": FolkTraditionRegion.SLAVIC,
                "domains": [AnimisticDomain.NATURE_SPIRITS, AnimisticDomain.ANCESTOR_RELATIONS],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.RITUAL_CEREMONY],
                "core_teaching": "Every home has its guardian spirit, often ancestral, who protects the family and maintains household harmony. The domovoi must be respected and appeased with offerings. Moving homes requires inviting the domovoi to come along.",
                "source_communities": ["Russian", "Ukrainian", "Polish"],
                "ethical_principles": [EthicalPrinciple.ANCESTOR_HONOR, EthicalPrinciple.HOSPITALITY],
            },

            # Aboriginal Australian
            {
                "teaching_id": "aboriginal_dreaming_law",
                "name": "The Law (Tjukurpa)",
                "region": FolkTraditionRegion.ABORIGINAL_AUSTRALIAN,
                "domains": [AnimisticDomain.LAND_CONSCIOUSNESS, AnimisticDomain.ANCESTOR_RELATIONS],
                "transmission_modes": [WisdomTransmissionMode.RITUAL_CEREMONY, WisdomTransmissionMode.SONG_CHANT, WisdomTransmissionMode.DANCE_MOVEMENT],
                "core_teaching": "The Law was established by ancestral beings in the Dreaming and continues to govern all aspects of life. It teaches how to live in relationship with Country, maintain ceremonies, and fulfill obligations to kin and land.",
                "source_communities": ["Western Desert peoples", "Arrernte", "Warlpiri"],
                "ethical_principles": [EthicalPrinciple.NATURE_RESPECT, EthicalPrinciple.COMMUNITY_HARMONY, EthicalPrinciple.ELDER_RESPECT],
            },

            # Polynesian
            {
                "teaching_id": "hawaiian_aloha_spirit",
                "name": "Aloha Spirit",
                "region": FolkTraditionRegion.POLYNESIAN,
                "domains": [AnimisticDomain.ANCESTOR_RELATIONS, AnimisticDomain.NATURE_SPIRITS],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.ELDER_TEACHING],
                "core_teaching": "Aloha is more than a greeting - it is a way of being that recognizes the divine breath (ha) in all beings. To live aloha is to extend love, compassion, and respect in all interactions, maintaining pono (righteousness) with gods, nature, and people.",
                "source_communities": ["Hawaiian"],
                "ethical_principles": [EthicalPrinciple.HOSPITALITY, EthicalPrinciple.GENEROSITY, EthicalPrinciple.NATURE_RESPECT],
            },

            # Siberian
            {
                "teaching_id": "siberian_three_worlds",
                "name": "Three Worlds Cosmology",
                "region": FolkTraditionRegion.SIBERIAN,
                "domains": [AnimisticDomain.CELESTIAL_BEINGS, AnimisticDomain.UNDERWORLD_ENTITIES],
                "transmission_modes": [WisdomTransmissionMode.RITUAL_CEREMONY, WisdomTransmissionMode.APPRENTICESHIP],
                "core_teaching": "The universe consists of three realms - Upper World of spirits and gods, Middle World of humans and nature, Lower World of ancestors and certain spirits. The shaman travels between these realms via the World Tree to heal, divine, and maintain cosmic balance.",
                "source_communities": ["Yakut", "Buryat", "Evenki"],
                "ethical_principles": [EthicalPrinciple.RECIPROCITY, EthicalPrinciple.ANCESTOR_HONOR],
            },

            # Inuit/Arctic
            {
                "teaching_id": "inuit_sila_wisdom",
                "name": "Sila - Weather Wisdom",
                "region": FolkTraditionRegion.ARCTIC_INDIGENOUS,
                "domains": [AnimisticDomain.ELEMENTAL_FORCES, AnimisticDomain.NATURE_SPIRITS],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.APPRENTICESHIP],
                "core_teaching": "Sila is the breath, weather, and intelligence that pervades all things. When one lives in harmony with sila, weather is favorable. When taboos are broken, sila becomes angered, bringing storms and scarcity. Wisdom is living in attunement with sila.",
                "source_communities": ["Inuit", "Yup'ik", "Greenlandic"],
                "ethical_principles": [EthicalPrinciple.NATURE_RESPECT, EthicalPrinciple.SACRED_PROHIBITION],
            },

            # Plains Indigenous
            {
                "teaching_id": "lakota_mitakuye_oyasin",
                "name": "Mitákuye Oyás'iŋ",
                "region": FolkTraditionRegion.PLAINS_INDIGENOUS,
                "domains": [AnimisticDomain.NATURE_SPIRITS, AnimisticDomain.ANCESTOR_RELATIONS],
                "transmission_modes": [WisdomTransmissionMode.RITUAL_CEREMONY, WisdomTransmissionMode.ORAL_NARRATIVE],
                "core_teaching": "All Are Related - a prayer and philosophical statement recognizing the kinship of all beings. Humans are not above nature but part of the sacred hoop of life that includes four-leggeds, winged ones, crawling beings, and all of creation.",
                "source_communities": ["Lakota", "Dakota", "Nakota"],
                "ethical_principles": [EthicalPrinciple.NATURE_RESPECT, EthicalPrinciple.COMMUNITY_HARMONY],
            },

            # Andean
            {
                "teaching_id": "andean_ayni",
                "name": "Ayni - Sacred Reciprocity",
                "region": FolkTraditionRegion.ANDEAN_FOLK,
                "domains": [AnimisticDomain.NATURE_SPIRITS, AnimisticDomain.ANCESTOR_RELATIONS],
                "transmission_modes": [WisdomTransmissionMode.RITUAL_CEREMONY, WisdomTransmissionMode.MATERIAL_CRAFT],
                "core_teaching": "The universe operates through ayni - sacred reciprocity. Humans must give back to Pachamama (Earth Mother) and the Apus (mountain spirits) what they receive. This creates ayni with the cosmos, ensuring continued abundance and harmony.",
                "source_communities": ["Quechua", "Aymara"],
                "ethical_principles": [EthicalPrinciple.RECIPROCITY, EthicalPrinciple.NATURE_RESPECT],
            },

            # Eastern Woodlands
            {
                "teaching_id": "haudenosaunee_thanksgiving",
                "name": "Thanksgiving Address (Ohén:ton Karihwatéhkwen)",
                "region": FolkTraditionRegion.EASTERN_WOODLANDS,
                "domains": [AnimisticDomain.NATURE_SPIRITS, AnimisticDomain.CELESTIAL_BEINGS],
                "transmission_modes": [WisdomTransmissionMode.ORAL_NARRATIVE, WisdomTransmissionMode.RITUAL_CEREMONY],
                "core_teaching": "Before any gathering, we acknowledge and thank all parts of creation - the earth, waters, fish, plants, animals, trees, birds, winds, thunderers, sun, moon, stars, and the Creator. This brings minds together in gratitude and unity.",
                "source_communities": ["Haudenosaunee/Iroquois"],
                "ethical_principles": [EthicalPrinciple.NATURE_RESPECT, EthicalPrinciple.COMMUNITY_HARMONY],
            },
        ]

    async def initialize_seed_teachings(self) -> int:
        """Initialize with seed teachings."""
        seed_teachings = self._get_seed_teachings()
        count = 0

        for teaching_data in seed_teachings:
            teaching = FolkWisdomTeaching(
                teaching_id=teaching_data["teaching_id"],
                name=teaching_data["name"],
                region=teaching_data["region"],
                domains=teaching_data["domains"],
                transmission_modes=teaching_data["transmission_modes"],
                core_teaching=teaching_data["core_teaching"],
                source_communities=teaching_data.get("source_communities", []),
                ethical_principles=teaching_data.get("ethical_principles", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_teaching(teaching)
            count += 1

        logger.info(f"Initialized {count} seed teachings")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        teachings_count = await self.initialize_seed_teachings()

        return {
            "teachings": teachings_count,
            "total": teachings_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "FolkTraditionRegion",
    "WisdomTransmissionMode",
    "AnimisticDomain",
    "EthicalPrinciple",
    "CosmologicalElement",
    "MaturityLevel",
    # Dataclasses
    "FolkWisdomTeaching",
    "AnimisticPractice",
    "IndigenousCosmology",
    "OralTradition",
    "IndigenousAnimalWisdom",
    "FolkWisdomMaturityState",
    # Interface
    "FolkWisdomInterface",
]
