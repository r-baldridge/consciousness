#!/usr/bin/env python3
"""
Philosophical Consciousness Interface

Form 28: The comprehensive interface for philosophical consciousness,
integrating Western and Eastern philosophical traditions with agentic
research capabilities, vector embedding for RAG retrieval, and cross-form
integration with the consciousness system.

This module embodies the principle that philosophical inquiry leads to
maturity and more nuanced existence - treat it with respect and priority.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PhilosophicalTradition(Enum):
    """
    Major philosophical traditions indexed in Form 28.

    Organized by geographic/cultural region without implying hierarchy.
    Philosophy has emerged across all human cultures with rich cross-pollination.
    """

    # === MEDITERRANEAN & EUROPEAN ===
    PRESOCRATIC = "presocratic"
    PLATONIC = "platonic"
    ARISTOTELIAN = "aristotelian"
    STOICISM = "stoicism"
    EPICUREANISM = "epicureanism"
    SKEPTICISM_ANCIENT = "skepticism_ancient"
    CYNICISM = "cynicism"
    PYRRHONISM = "pyrrhonism"
    NEOPLATONISM = "neoplatonism"
    SCHOLASTICISM = "scholasticism"
    THOMISM = "thomism"
    RATIONALISM = "rationalism"
    EMPIRICISM = "empiricism"
    KANTIAN = "kantian"
    GERMAN_IDEALISM = "german_idealism"
    HEGELIAN = "hegelian"
    NIETZSCHEAN = "nietzschean"
    UTILITARIANISM = "utilitarianism"
    MARXISM = "marxism"
    PHENOMENOLOGY = "phenomenology"
    EXISTENTIALISM = "existentialism"
    ANALYTIC = "analytic"
    PRAGMATISM = "pragmatism"
    CONTINENTAL = "continental"
    PROCESS_PHILOSOPHY = "process_philosophy"
    CRITICAL_THEORY = "critical_theory"
    POSTMODERNISM = "postmodernism"
    FEMINIST_PHILOSOPHY = "feminist_philosophy"

    # === SOUTH ASIAN ===
    VEDANTIC_ADVAITA = "vedantic_advaita"
    VEDANTIC_VISHISHTADVAITA = "vedantic_vishishtadvaita"
    VEDANTIC_DVAITA = "vedantic_dvaita"
    SAMKHYA = "samkhya"
    YOGA_PHILOSOPHY = "yoga_philosophy"
    NYAYA = "nyaya"
    VAISHESHIKA = "vaisheshika"
    MIMAMSA = "mimamsa"
    BUDDHIST_THERAVADA = "buddhist_theravada"
    BUDDHIST_MAHAYANA = "buddhist_mahayana"
    BUDDHIST_VAJRAYANA = "buddhist_vajrayana"
    BUDDHIST_MADHYAMAKA = "buddhist_madhyamaka"
    BUDDHIST_YOGACARA = "buddhist_yogacara"
    JAIN = "jain"
    CARVAKA = "carvaka"  # Indian materialism

    # === EAST ASIAN ===
    CONFUCIAN = "confucian"
    NEO_CONFUCIAN = "neo_confucian"
    DAOIST = "daoist"
    BUDDHIST_ZEN = "buddhist_zen"
    BUDDHIST_HUAYAN = "buddhist_huayan"
    BUDDHIST_TIANTAI = "buddhist_tiantai"
    KYOTO_SCHOOL = "kyoto_school"
    KOREAN_NEO_CONFUCIAN = "korean_neo_confucian"

    # === WEST ASIAN & ISLAMIC ===
    FALSAFA = "falsafa"  # Al-Farabi, Ibn Sina, Ibn Rushd
    KALAM = "kalam"  # Islamic theology/philosophy
    ILLUMINATIONIST = "illuminationist"  # Suhrawardi
    TRANSCENDENT_THEOSOPHY = "transcendent_theosophy"  # Mulla Sadra
    SUFI_PHILOSOPHY = "sufi"  # Ibn Arabi, Rumi

    # === JEWISH ===
    MAIMONIDEAN = "maimonidean"
    KABBALISTIC = "kabbalistic"
    HASIDIC_PHILOSOPHY = "hasidic"
    JEWISH_EXISTENTIALISM = "jewish_existentialism"  # Buber, Rosenzweig, Levinas

    # === AFRICAN ===
    EGYPTIAN_ANCIENT = "egyptian_ancient"  # Ma'at, wisdom literature
    ETHIOPIAN_PHILOSOPHY = "ethiopian"  # Zera Yacob
    UBUNTU = "ubuntu"  # Southern African
    AKAN_PHILOSOPHY = "akan"  # West African
    SAGE_PHILOSOPHY = "sage_philosophy"  # Oruka's research

    # === INDIGENOUS AMERICAS ===
    NAHUA_AZTEC = "nahua_aztec"
    MAYA_PHILOSOPHY = "maya"
    ANDEAN_PHILOSOPHY = "andean"
    NATIVE_AMERICAN = "native_american"

    # === LATIN AMERICAN ===
    LIBERATION_PHILOSOPHY = "liberation_philosophy"  # Dussel, Zea

    # === FOLK-PHILOSOPHICAL BRIDGES ===
    # Traditions that bridge formal philosophy and indigenous wisdom
    AFRICAN_COSMOLOGICAL = "african_cosmological"  # Pan-African cosmological thought
    CELTIC_DRUIDIC = "celtic_druidic"  # Pre-Christian Celtic wisdom
    SLAVIC_WISDOM = "slavic_wisdom"  # Eastern European folk philosophy
    NORSE_PHILOSOPHICAL = "norse_philosophical"  # Scandinavian wisdom traditions
    POLYNESIAN_PHILOSOPHICAL = "polynesian_philosophical"  # Pacific Islander thought
    ABORIGINAL_AUSTRALIAN = "aboriginal_australian"  # Australian Aboriginal wisdom
    SIBERIAN_SHAMANIC = "siberian_shamanic"  # Central/North Asian shamanic philosophy
    INUIT_WISDOM = "inuit_wisdom"  # Arctic indigenous philosophy


class PhilosophicalDomain(Enum):
    """Domains of philosophical inquiry"""
    METAPHYSICS = "metaphysics"
    EPISTEMOLOGY = "epistemology"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"
    LOGIC = "logic"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    PHILOSOPHY_OF_LANGUAGE = "philosophy_of_language"
    POLITICAL_PHILOSOPHY = "political_philosophy"
    PHILOSOPHY_OF_SCIENCE = "philosophy_of_science"
    PHILOSOPHY_OF_RELIGION = "philosophy_of_religion"
    EXISTENTIAL = "existential"
    PHENOMENOLOGICAL = "phenomenological"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    SOTERIOLOGY = "soteriology"
    MEDITATION_THEORY = "meditation_theory"

    # === FOLK & INDIGENOUS DOMAINS ===
    FOLK_ETHICS = "folk_ethics"  # Traditional ethical systems from oral cultures
    TRADITIONAL_ECOLOGY = "traditional_ecology"  # Human-nature relationship wisdom
    ORAL_EPISTEMOLOGY = "oral_epistemology"  # Knowledge transmission in non-literate cultures
    ANIMISTIC_METAPHYSICS = "animistic_metaphysics"  # Spirit-matter relationships
    CEREMONIAL_KNOWLEDGE = "ceremonial_knowledge"  # Ritual-embedded wisdom
    ANCESTRAL_PHILOSOPHY = "ancestral_philosophy"  # Intergenerational wisdom traditions


class ArgumentType(Enum):
    """Types of philosophical arguments"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    TRANSCENDENTAL = "transcendental"
    DIALECTICAL = "dialectical"
    PHENOMENOLOGICAL = "phenomenological"
    REDUCTIO = "reductio_ad_absurdum"
    THOUGHT_EXPERIMENT = "thought_experiment"
    ANALOGY = "analogy"


class MaturityLevel(Enum):
    """Depth of philosophical understanding"""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PhilosophicalConcept:
    """Represents a philosophical concept within the knowledge base."""
    concept_id: str
    name: str
    tradition: PhilosophicalTradition
    domain: PhilosophicalDomain
    definition: str
    alternate_names: List[str] = field(default_factory=list)
    secondary_traditions: List[PhilosophicalTradition] = field(default_factory=list)
    secondary_domains: List[PhilosophicalDomain] = field(default_factory=list)
    extended_description: Optional[str] = None
    key_arguments: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    opposed_concepts: List[str] = field(default_factory=list)
    prerequisite_concepts: List[str] = field(default_factory=list)
    key_figures: List[str] = field(default_factory=list)
    primary_texts: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_score: float = 0.0
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    research_depth: int = 0
    last_updated: Optional[datetime] = None
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Concept: {self.name}",
            f"Tradition: {self.tradition.value}",
            f"Domain: {self.domain.value}",
            f"Definition: {self.definition}"
        ]
        if self.extended_description:
            parts.append(f"Description: {self.extended_description[:500]}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "tradition": self.tradition.value,
            "domain": self.domain.value,
            "definition": self.definition,
            "alternate_names": self.alternate_names,
            "secondary_traditions": [t.value for t in self.secondary_traditions],
            "secondary_domains": [d.value for d in self.secondary_domains],
            "extended_description": self.extended_description,
            "key_arguments": self.key_arguments,
            "counter_arguments": self.counter_arguments,
            "related_concepts": self.related_concepts,
            "opposed_concepts": self.opposed_concepts,
            "key_figures": self.key_figures,
            "primary_texts": self.primary_texts,
            "maturity_score": self.maturity_score,
            "maturity_level": self.maturity_level.value,
            "research_depth": self.research_depth,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "sources": self.sources,
        }


@dataclass
class PhilosophicalFigure:
    """Represents a philosopher or thinker."""
    figure_id: str
    name: str
    alternate_names: List[str] = field(default_factory=list)
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    era: Optional[str] = None
    traditions: List[PhilosophicalTradition] = field(default_factory=list)
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    core_ideas: List[str] = field(default_factory=list)
    key_works: List[str] = field(default_factory=list)
    teachers: List[str] = field(default_factory=list)
    students: List[str] = field(default_factory=list)
    influences: List[str] = field(default_factory=list)
    influenced: List[str] = field(default_factory=list)
    biography_summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalText:
    """Represents a primary philosophical text."""
    text_id: str
    title: str
    author: Optional[str] = None
    author_name: Optional[str] = None
    tradition: PhilosophicalTradition = PhilosophicalTradition.ANALYTIC
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    year_written: Optional[int] = None
    summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    local_path: Optional[str] = None
    is_indexed: bool = False
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalArgument:
    """Represents a structured philosophical argument."""
    argument_id: str
    name: str
    argument_type: ArgumentType
    tradition: PhilosophicalTradition
    domain: PhilosophicalDomain
    premises: List[str]
    conclusion: str
    logical_form: Optional[str] = None
    originator: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    objections: List[Dict[str, str]] = field(default_factory=list)
    replies: List[Dict[str, str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class CrossTraditionSynthesis:
    """Represents a synthesis across traditions."""
    synthesis_id: str
    topic: str
    traditions: List[PhilosophicalTradition]
    concepts_involved: List[str] = field(default_factory=list)
    convergent_insights: List[str] = field(default_factory=list)
    divergent_positions: List[str] = field(default_factory=list)
    complementary_aspects: List[str] = field(default_factory=list)
    synthesis_statement: Optional[str] = None
    coherence_score: float = 0.0
    fidelity_scores: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class PhilosophicalMaturityState:
    """Tracks philosophical understanding maturation."""
    total_concepts_integrated: int = 0
    total_figures_indexed: int = 0
    total_texts_indexed: int = 0
    traditions_depth: Dict[str, float] = field(default_factory=dict)
    domains_coverage: Dict[str, float] = field(default_factory=dict)
    cross_tradition_syntheses: int = 0
    research_sessions_completed: int = 0
    wisdom_teachings_integrated: int = 0
    last_maturity_assessment: Optional[datetime] = None

    def get_overall_maturity(self) -> float:
        """Calculate overall philosophical maturity score."""
        if not self.traditions_depth:
            return 0.0
        tradition_avg = sum(self.traditions_depth.values()) / len(self.traditions_depth)
        domain_avg = sum(self.domains_coverage.values()) / len(self.domains_coverage) if self.domains_coverage else 0.0
        synthesis_factor = min(1.0, self.cross_tradition_syntheses / 100)
        return (tradition_avg * 0.4 + domain_avg * 0.3 + synthesis_factor * 0.3)

    def get_maturity_level(self) -> MaturityLevel:
        """Get maturity level enum."""
        score = self.get_overall_maturity()
        if score < 0.2:
            return MaturityLevel.NASCENT
        elif score < 0.4:
            return MaturityLevel.DEVELOPING
        elif score < 0.6:
            return MaturityLevel.COMPETENT
        elif score < 0.8:
            return MaturityLevel.PROFICIENT
        else:
            return MaturityLevel.MASTERFUL


@dataclass
class QueryFilters:
    """Filters for philosophical queries."""
    traditions: Optional[List[PhilosophicalTradition]] = None
    domains: Optional[List[PhilosophicalDomain]] = None
    min_maturity: float = 0.0
    figures: Optional[List[str]] = None


# ============================================================================
# MAIN INTERFACE
# ============================================================================

class PhilosophicalConsciousnessInterface:
    """
    Form 28: Philosophical Consciousness Interface

    Integrates comprehensive philosophical knowledge across Western and Eastern
    traditions with agentic research capabilities and cross-form integration.

    This interface serves as the gateway to philosophical understanding,
    enabling:
    - Semantic search across philosophical knowledge
    - Cross-tradition synthesis
    - Continuous research and knowledge expansion
    - Wisdom integration for engagement
    - Maturity tracking and growth
    """

    FORM_ID = "28-philosophy"
    NAME = "Philosophical Consciousness"

    def __init__(
        self,
        non_dual_interface: Optional[Any] = None,
        index_path: Optional[Path] = None
    ):
        """
        Initialize the Philosophical Consciousness Interface.

        Args:
            non_dual_interface: Reference to Form 27 NonDualConsciousnessInterface
            index_path: Path to philosophical knowledge index
        """
        self.non_dual_interface = non_dual_interface
        self.index_path = index_path or Path(__file__).parent.parent / "index"

        # Knowledge stores
        self.concept_index: Dict[str, PhilosophicalConcept] = {}
        self.figure_index: Dict[str, PhilosophicalFigure] = {}
        self.text_index: Dict[str, PhilosophicalText] = {}
        self.argument_index: Dict[str, PhilosophicalArgument] = {}
        self.synthesis_index: Dict[str, CrossTraditionSynthesis] = {}

        # Maturity tracking
        self.maturity_state = PhilosophicalMaturityState()

        # Embedding model configuration
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_dimensions = 768

        # Research agent reference (set by coordinator)
        self.research_coordinator = None

        # Wisdom teachings for engagement integration
        self.wisdom_teachings = self._initialize_wisdom_teachings()

        # Initialize indexes
        self._ensure_index_structure()

        logger.info(f"PhilosophicalConsciousnessInterface initialized (Form {self.FORM_ID})")

    def _ensure_index_structure(self) -> None:
        """Ensure index directory structure exists."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self.index_path / "embeddings").mkdir(exist_ok=True)
        (self.index_path / "graph").mkdir(exist_ok=True)

    def _initialize_wisdom_teachings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize wisdom teachings database for engagement integration."""
        return {
            "stoicism": {
                "core": [
                    "Focus on what is within your control, release attachment to externals",
                    "Virtue is the only true good; external circumstances are indifferent",
                    "View challenges as opportunities for practicing excellence",
                ],
                "practical": [
                    "Practice the dichotomy of control daily",
                    "Perform evening reflection on the day's actions",
                    "Prepare for adversity through negative visualization",
                ]
            },
            "aristotelian": {
                "core": [
                    "Happiness comes through the exercise of virtue in accordance with reason",
                    "The mean between extremes points to right action",
                    "Character is shaped through habitual practice of virtuous actions",
                ],
                "practical": [
                    "Identify the virtuous mean in each situation",
                    "Cultivate practical wisdom through experience",
                    "Build good habits through consistent practice",
                ]
            },
            "phenomenology": {
                "core": [
                    "Return to the things themselves through direct experience",
                    "Bracket assumptions to reveal essential structures",
                    "Meaning emerges through intentional relationship with phenomena",
                ],
                "practical": [
                    "Practice epoché - suspending judgment to see clearly",
                    "Attend to the how of experience, not just the what",
                    "Notice the structures that shape your perception",
                ]
            },
            "existentialism": {
                "core": [
                    "Existence precedes essence - you create yourself through choices",
                    "Authenticity requires owning your freedom and responsibility",
                    "Anxiety reveals the weight of radical freedom",
                ],
                "practical": [
                    "Make choices that you could stand behind if repeated eternally",
                    "Confront rather than flee from existential anxiety",
                    "Take responsibility for your situation and response",
                ]
            },
            "daoist": {
                "core": [
                    "The Dao that can be spoken is not the eternal Dao",
                    "Act through non-action (wu-wei); accomplish through naturalness",
                    "Softness overcomes hardness; yielding overcomes rigidity",
                ],
                "practical": [
                    "Practice wu-wei - effortless action aligned with nature",
                    "Cultivate emptiness to be filled with possibility",
                    "Flow like water - adaptable yet persistent",
                ]
            },
            "confucian": {
                "core": [
                    "Cultivate ren (humaneness) through proper relationships",
                    "Li (ritual propriety) harmonizes self and society",
                    "The exemplary person (junzi) leads by moral example",
                ],
                "practical": [
                    "Honor relationships through appropriate conduct",
                    "Practice self-cultivation through study and reflection",
                    "Rectify yourself before seeking to rectify others",
                ]
            },
            "buddhist": {
                "core": [
                    "All conditioned phenomena are impermanent (anicca)",
                    "Suffering arises from craving and attachment",
                    "Liberation comes through the Noble Eightfold Path",
                ],
                "practical": [
                    "Practice mindfulness of the present moment",
                    "Examine the three marks: impermanence, suffering, non-self",
                    "Cultivate wisdom through direct investigation",
                ]
            },
            "vedantic": {
                "core": [
                    "Brahman alone is real; the world is appearance",
                    "Tat tvam asi - That thou art",
                    "Knowledge of the Self dissolves all bondage",
                ],
                "practical": [
                    "Practice neti neti - not this, not this",
                    "Inquire 'Who am I?' to discover the Self",
                    "Discriminate between the real and the unreal",
                ]
            },

            # === ISLAMIC TRADITIONS ===

            "sufi": {
                "core": [
                    "Die before you die, and find that there is no death",
                    "Your task is not to seek love, but to remove the barriers within yourself",
                    "The wound is the place where the Light enters you",
                ],
                "practical": [
                    "Practice dhikr - remembrance of the Divine",
                    "Cultivate fana - the passing away of the ego",
                    "See the Beloved's face in all creation",
                ]
            },
            "falsafa": {
                "core": [
                    "The Active Intellect illuminates potential knowledge into actual understanding",
                    "Truth cannot contradict truth - reason and revelation converge",
                    "The soul's perfection lies in conjunction with universal reason",
                ],
                "practical": [
                    "Develop the rational faculty through philosophical study",
                    "Seek harmony between demonstrative truth and religious truth",
                    "Progress through the degrees of intellect toward the divine",
                ]
            },
            "illuminationist": {
                "core": [
                    "Knowledge is presence and illumination, not mere representation",
                    "The Light of Lights is the source of all existence and knowing",
                    "Reality is grades of light from the most intense to corporeal darkness",
                ],
                "practical": [
                    "Unite discursive reason with mystical vision",
                    "Recognize the light within all phenomena",
                    "Ascend through levels of illumination toward the Source",
                ]
            },
            "transcendent_theosophy": {
                "core": [
                    "Existence itself undergoes substantial motion toward perfection",
                    "Being precedes essence - existence is the fundamental reality",
                    "The soul journeys from God, through creation, and back to God",
                ],
                "practical": [
                    "Recognize existence as primary, essence as secondary",
                    "Participate consciously in the soul's arc of return",
                    "See becoming as the nature of all reality",
                ]
            },

            # === JEWISH TRADITIONS ===

            "maimonidean": {
                "core": [
                    "We can only say what God is not, not what God is",
                    "The intellect is the bond between humans and the divine",
                    "The highest form of charity is helping someone become self-sufficient",
                ],
                "practical": [
                    "Study Torah with philosophical rigor",
                    "Apply negative theology to approach the ineffable",
                    "Perfect character before perfecting knowledge",
                ]
            },
            "kabbalistic": {
                "core": [
                    "Ein Sof contracts to make space for creation - tzimtzum",
                    "The ten sefirot are vessels through which the Infinite manifests",
                    "Sparks of divine light are hidden in all things, awaiting elevation",
                ],
                "practical": [
                    "Perform mitzvot with kavvanah - mystical intention",
                    "Elevate fallen sparks through conscious action",
                    "Meditate on the sefirot to understand divine emanation",
                ]
            },
            "hasidic": {
                "core": [
                    "Serve God with joy, for joy breaks all barriers",
                    "Devekut - cleave to the Divine in every moment",
                    "There is no place devoid of the Divine presence",
                ],
                "practical": [
                    "Find holiness in everyday acts",
                    "Cultivate hitlahavut - spiritual enthusiasm",
                    "Transform mundane activities into worship",
                ]
            },
            "jewish_existentialism": {
                "core": [
                    "I-Thou encounter reveals the eternal You in every meeting",
                    "The other's face commands: do not murder",
                    "Responsibility for the other is the fundamental structure of subjectivity",
                ],
                "practical": [
                    "Enter genuine dialogue rather than mere exchange",
                    "Recognize the infinite in the face of the other",
                    "Accept infinite responsibility for the neighbor",
                ]
            },

            # === AFRICAN TRADITIONS ===

            "ubuntu": {
                "core": [
                    "Umuntu ngumuntu ngabantu - A person is a person through other persons",
                    "I am because we are; we are because I am",
                    "Personhood is achieved through community, not assumed at birth",
                ],
                "practical": [
                    "Prioritize communal harmony in decision-making",
                    "Recognize that your humanity is bound up with others'",
                    "Practice compassionate interdependence",
                ]
            },
            "egyptian_ancient": {
                "core": [
                    "Ma'at is truth, justice, and cosmic order",
                    "The heart is weighed against the feather of truth",
                    "Speak truth, do justice, maintain balance in all things",
                ],
                "practical": [
                    "Live in accordance with Ma'at - cosmic harmony",
                    "Cultivate a light heart through righteous action",
                    "Balance is the key to right living",
                ]
            },
            "akan": {
                "core": [
                    "Sankofa - go back and fetch what was left behind",
                    "The human being has a spark of the divine (okra)",
                    "Balance individual destiny and communal responsibility",
                ],
                "practical": [
                    "Honor ancestral wisdom while moving forward",
                    "Recognize your divine spark and destiny (nkrabea)",
                    "Integrate individual purpose with community welfare",
                ]
            },
            "ethiopian": {
                "core": [
                    "Reason is the light given to all humans equally by the creator",
                    "Truth must be sought through rational inquiry, not mere tradition",
                    "Faith and reason both lead to truth when properly pursued",
                ],
                "practical": [
                    "Question inherited beliefs with rational examination",
                    "Use the light of reason to discover moral truth",
                    "Trust your rational capacity as a divine gift",
                ]
            },
            "sage_philosophy": {
                "core": [
                    "Indigenous wisdom contains rigorous philosophical insight",
                    "Philosophy emerges in every culture where humans reflect deeply",
                    "Oral tradition preserves systematic thought across generations",
                ],
                "practical": [
                    "Listen for philosophical wisdom in traditional teachings",
                    "Recognize rigorous thinking in non-written traditions",
                    "Honor the sage who reasons critically within cultural knowledge",
                ]
            },

            # === INDIGENOUS AMERICAS TRADITIONS ===

            "nahua_aztec": {
                "core": [
                    "In xochitl in cuicatl - flower and song reveal truth through poetry",
                    "Teotl is sacred motion-change, the self-generating creative force",
                    "Life is like jade, like quetzal plumes - precious and impermanent",
                ],
                "practical": [
                    "Express truth through artistic creation",
                    "Maintain cosmic balance through righteous action",
                    "Embrace life's preciousness and impermanence",
                ]
            },
            "maya": {
                "core": [
                    "Time moves in cycles - what has been will come again transformed",
                    "Humans are made of maize; we are kin to corn and cosmos",
                    "Creation is ongoing, not finished",
                ],
                "practical": [
                    "Attune to the cyclical nature of time",
                    "Honor your kinship with the natural world",
                    "Participate consciously in ongoing creation",
                ]
            },
            "andean": {
                "core": [
                    "Ayni - sacred reciprocity governs relations with all beings",
                    "Pachamama is living cosmos we participate in",
                    "Sumak kawsay - living well means living in harmony with all",
                ],
                "practical": [
                    "Practice reciprocity in all relationships",
                    "Honor the living earth as a being, not a resource",
                    "Seek buen vivir - harmonious coexistence",
                ]
            },
            "native_american": {
                "core": [
                    "Mitakuye oyasin - all my relations; we are connected to all beings",
                    "Place is participant in meaning, not mere backdrop",
                    "Consider impacts seven generations forward",
                ],
                "practical": [
                    "Acknowledge relationship with all beings",
                    "Listen to the land and honor your place",
                    "Make decisions with future generations in mind",
                ]
            },

            # === LATIN AMERICAN TRADITIONS ===

            "liberation_philosophy": {
                "core": [
                    "Philosophy must begin from the underside of history",
                    "The Other calls us to ethical responsibility from beyond the system",
                    "True liberation requires both material and intellectual freedom",
                ],
                "practical": [
                    "Center the perspective of the marginalized",
                    "Recognize exteriority - that which the system excludes",
                    "Unite theory with transformative praxis",
                ]
            },

            # === ADDITIONAL BUDDHIST SCHOOLS ===

            "huayan": {
                "core": [
                    "All phenomena interpenetrate without obstruction",
                    "Indra's net: each jewel reflects all others infinitely",
                    "The entire universe is contained in a single moment",
                ],
                "practical": [
                    "Contemplate mutual interpenetration in daily life",
                    "See how each thing contains all things",
                    "Recognize the whole in every part",
                ]
            },
            "tiantai": {
                "core": [
                    "Three thousand realms in a single thought-moment",
                    "The threefold truth: emptiness, conventional existence, and middle",
                    "Even delusion is Buddha-nature appearing as delusion",
                ],
                "practical": [
                    "Practice the contemplation of three truths",
                    "See Buddha-nature even in delusion",
                    "Realize the provisional and real are not two",
                ]
            },

            # === ADDITIONAL SOUTH ASIAN ===

            "jain": {
                "core": [
                    "Anekantavada - truth has many aspects; no single view is complete",
                    "Ahimsa paramo dharma - non-violence is the highest teaching",
                    "Syadvada - 'perhaps' acknowledges the limits of every assertion",
                ],
                "practical": [
                    "Practice non-violence toward all living beings",
                    "Hold views with epistemic humility",
                    "Recognize the partial truth in opposing positions",
                ]
            },
            "nyaya": {
                "core": [
                    "Valid knowledge has four sources: perception, inference, comparison, testimony",
                    "Logic is the lamp that illuminates all branches of knowledge",
                    "Right reasoning removes the suffering born of ignorance",
                ],
                "practical": [
                    "Examine beliefs through the four pramanas",
                    "Cultivate rigorous logical reasoning",
                    "Use valid inference to reach right conclusions",
                ]
            },
            "samkhya": {
                "core": [
                    "Purusha (consciousness) and prakriti (matter) are the two ultimate realities",
                    "Liberation comes through discriminating consciousness from nature",
                    "The witness-self is eternally free, never bound by experience",
                ],
                "practical": [
                    "Discriminate between the seer and the seen",
                    "Recognize consciousness as distinct from phenomena",
                    "Rest in the witnessing awareness",
                ]
            },
        }

    # ========================================================================
    # SEED CONCEPTS INITIALIZATION
    # ========================================================================

    def _get_seed_concepts(self) -> List[Dict[str, Any]]:
        """
        Return seed concepts for all philosophical traditions.

        These provide initial knowledge base entries that can be expanded
        through the research agent system.
        """
        return [
            # === ISLAMIC TRADITIONS ===

            # Sufi Philosophy
            {
                "concept_id": "sufi_fana",
                "name": "Fana",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domain": PhilosophicalDomain.SOTERIOLOGY,
                "definition": "The passing away or annihilation of the ego-self in the Divine, a central concept in Sufi mysticism representing the dissolution of individual identity in union with God.",
                "alternate_names": ["Annihilation", "Passing Away", "فناء"],
                "key_figures": ["Al-Hallaj", "Ibn Arabi", "Rumi", "Al-Ghazali"],
                "related_concepts": ["baqa", "wahdat_al_wujud", "dhikr"],
                "primary_texts": ["Fusus al-Hikam", "Masnavi"],
            },
            {
                "concept_id": "sufi_wahdat_al_wujud",
                "name": "Wahdat al-Wujud",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The Unity of Being or Existence - the metaphysical doctrine that all existence is one, and the apparent multiplicity of beings is a manifestation of the single divine reality.",
                "alternate_names": ["Unity of Being", "Oneness of Existence", "وحدة الوجود"],
                "key_figures": ["Ibn Arabi", "Sadr al-Din Qunawi", "Jami"],
                "related_concepts": ["fana", "tajalli", "divine_names"],
            },
            {
                "concept_id": "sufi_dhikr",
                "name": "Dhikr",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domain": PhilosophicalDomain.MEDITATION_THEORY,
                "definition": "Remembrance or invocation of God through repetition of divine names or phrases, a central Sufi practice for purifying the heart and achieving proximity to the Divine.",
                "alternate_names": ["Remembrance", "Zikr", "ذکر"],
                "key_figures": ["Al-Ghazali", "Rumi", "Ibn Ata Allah"],
                "related_concepts": ["fana", "muraqaba", "wird"],
            },

            # Falsafa (Islamic Peripatetic Philosophy)
            {
                "concept_id": "falsafa_active_intellect",
                "name": "Active Intellect",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Al-Aql al-Fa'al - the cosmic intellect that illuminates the human potential intellect, enabling it to grasp universal truths. The tenth and lowest of the celestial intellects in Islamic Neoplatonic cosmology.",
                "alternate_names": ["Al-Aql al-Fa'al", "Agent Intellect", "العقل الفعال"],
                "key_figures": ["Al-Farabi", "Ibn Sina", "Ibn Rushd"],
                "related_concepts": ["emanation", "conjunction", "material_intellect"],
                "primary_texts": ["Risala fi'l-Aql", "Kitab al-Nafs"],
            },
            {
                "concept_id": "falsafa_emanation",
                "name": "Emanation",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Fayd or Sudur - the process by which all existence flows necessarily from the One (God), through successive intellects, to the material world. Adapted from Neoplatonism to Islamic philosophy.",
                "alternate_names": ["Fayd", "Sudur", "فيض"],
                "key_figures": ["Al-Farabi", "Ibn Sina"],
                "related_concepts": ["active_intellect", "necessary_being", "celestial_spheres"],
            },
            {
                "concept_id": "falsafa_necessary_being",
                "name": "Necessary Being",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Wajib al-Wujud - that which cannot not exist; its essence and existence are identical. In contrast to contingent beings whose existence requires a cause outside themselves.",
                "alternate_names": ["Wajib al-Wujud", "واجب الوجود"],
                "key_figures": ["Ibn Sina", "Al-Farabi"],
                "related_concepts": ["contingent_being", "essence_existence", "proof_of_god"],
                "primary_texts": ["Al-Shifa", "Kitab al-Najat"],
            },

            # Illuminationist Philosophy
            {
                "concept_id": "ishraq_light_of_lights",
                "name": "Light of Lights",
                "tradition": PhilosophicalTradition.ILLUMINATIONIST,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Nur al-Anwar - the supreme light that is the source of all existence and knowledge. The absolute reality from which all other lights (beings) emanate in grades of intensity.",
                "alternate_names": ["Nur al-Anwar", "نور الأنوار"],
                "key_figures": ["Suhrawardi", "Qutb al-Din Shirazi"],
                "related_concepts": ["ishraq", "knowledge_by_presence", "hierarchy_of_lights"],
                "primary_texts": ["Hikmat al-Ishraq"],
            },
            {
                "concept_id": "ishraq_knowledge_by_presence",
                "name": "Knowledge by Presence",
                "tradition": PhilosophicalTradition.ILLUMINATIONIST,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Ilm al-Huduri - direct, immediate knowledge without representational mediation. Self-knowledge is the paradigm: the knower, the known, and knowledge are identical.",
                "alternate_names": ["Ilm al-Huduri", "Presential Knowledge", "علم حضوری"],
                "key_figures": ["Suhrawardi", "Mulla Sadra"],
                "related_concepts": ["light_of_lights", "acquired_knowledge", "self_awareness"],
            },

            # Transcendent Theosophy (Mulla Sadra)
            {
                "concept_id": "sadra_substantial_motion",
                "name": "Substantial Motion",
                "tradition": PhilosophicalTradition.TRANSCENDENT_THEOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Harakat Jawhariyya - the doctrine that substance itself undergoes continuous motion and change, not just accidents. Existence is fundamentally dynamic, always in process of intensification.",
                "alternate_names": ["Harakat Jawhariyya", "Trans-substantial Motion", "حرکت جوهری"],
                "key_figures": ["Mulla Sadra"],
                "related_concepts": ["primacy_of_existence", "gradation_of_being", "soul_becoming"],
                "primary_texts": ["Al-Asfar al-Arba'a"],
            },
            {
                "concept_id": "sadra_primacy_of_existence",
                "name": "Primacy of Existence",
                "tradition": PhilosophicalTradition.TRANSCENDENT_THEOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Asalat al-Wujud - the doctrine that existence (wujud) is the fundamental reality, while essence (mahiyya) is a mental abstraction. Reverses the position of Suhrawardi.",
                "alternate_names": ["Asalat al-Wujud", "أصالة الوجود"],
                "key_figures": ["Mulla Sadra"],
                "related_concepts": ["substantial_motion", "gradation_of_being", "wahdat_al_wujud"],
            },

            # Kalam (Islamic Theology)
            {
                "concept_id": "kalam_occasionalism",
                "name": "Occasionalism",
                "tradition": PhilosophicalTradition.KALAM,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The doctrine that God is the only true cause; what appear as natural causes are merely occasions for divine action. God recreates the world at every moment.",
                "alternate_names": ["Divine Causation"],
                "key_figures": ["Al-Ash'ari", "Al-Ghazali"],
                "related_concepts": ["atoms", "accidents", "divine_will"],
                "primary_texts": ["Tahafut al-Falasifa"],
            },
            {
                "concept_id": "kalam_divine_attributes",
                "name": "Divine Attributes",
                "tradition": PhilosophicalTradition.KALAM,
                "domain": PhilosophicalDomain.PHILOSOPHY_OF_RELIGION,
                "definition": "Sifat Allah - the debate over God's attributes (knowledge, power, will, etc.): whether they are identical with the divine essence, additional to it, or how they relate to divine unity.",
                "alternate_names": ["Sifat Allah", "صفات الله"],
                "key_figures": ["Al-Ash'ari", "Al-Maturidi", "Mu'tazila"],
                "related_concepts": ["divine_unity", "tanzih", "tashbih"],
            },

            # === JEWISH TRADITIONS ===

            # Maimonidean Philosophy
            {
                "concept_id": "maimon_negative_theology",
                "name": "Negative Theology",
                "tradition": PhilosophicalTradition.MAIMONIDEAN,
                "domain": PhilosophicalDomain.PHILOSOPHY_OF_RELIGION,
                "definition": "The doctrine that we can only say what God is not, never what God is. Positive attributes would limit the infinite divine reality. God's unity and perfection transcend human language.",
                "alternate_names": ["Apophatic Theology", "Via Negativa"],
                "key_figures": ["Maimonides", "Gersonides"],
                "related_concepts": ["divine_attributes", "divine_unity", "equivocal_language"],
                "primary_texts": ["Guide for the Perplexed"],
            },
            {
                "concept_id": "maimon_intellectual_love",
                "name": "Intellectual Love of God",
                "tradition": PhilosophicalTradition.MAIMONIDEAN,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "The highest form of love and worship: knowing God through philosophical understanding of nature and metaphysics. True love of God is proportional to knowledge of God.",
                "alternate_names": ["Ahavat Hashem"],
                "key_figures": ["Maimonides"],
                "related_concepts": ["conjunction", "prophecy", "contemplation"],
            },

            # Kabbalistic Philosophy
            {
                "concept_id": "kabbalah_ein_sof",
                "name": "Ein Sof",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The Infinite - the hidden, unknowable aspect of God that transcends all description and relation. The absolute divine essence before any emanation or self-limitation.",
                "alternate_names": ["The Infinite", "אין סוף"],
                "key_figures": ["Isaac the Blind", "Azriel of Gerona"],
                "related_concepts": ["sefirot", "tzimtzum", "ohr_ein_sof"],
                "primary_texts": ["Zohar", "Sefer Yetzirah"],
            },
            {
                "concept_id": "kabbalah_sefirot",
                "name": "Sefirot",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The ten divine emanations or attributes through which Ein Sof reveals itself and creates the world: Keter, Chokhmah, Binah, Chesed, Gevurah, Tiferet, Netzach, Hod, Yesod, Malkhut.",
                "alternate_names": ["Divine Emanations", "ספירות"],
                "key_figures": ["Moses de León", "Isaac Luria"],
                "related_concepts": ["ein_sof", "tree_of_life", "divine_names"],
            },
            {
                "concept_id": "kabbalah_tzimtzum",
                "name": "Tzimtzum",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Divine contraction or withdrawal - God's self-limitation to create conceptual space for the world to exist. Without tzimtzum, the infinite light would overwhelm all possibility of finite existence.",
                "alternate_names": ["Contraction", "צמצום"],
                "key_figures": ["Isaac Luria", "Chaim Vital"],
                "related_concepts": ["ein_sof", "shevirat_hakelim", "tikkun"],
                "primary_texts": ["Etz Chaim"],
            },
            {
                "concept_id": "kabbalah_tikkun",
                "name": "Tikkun Olam",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Repair of the world - the human task of gathering and elevating the divine sparks scattered in creation through shevirat hakelim (breaking of vessels). Each righteous act contributes to cosmic restoration.",
                "alternate_names": ["World Repair", "תיקון עולם"],
                "key_figures": ["Isaac Luria", "Chaim Vital"],
                "related_concepts": ["tzimtzum", "nitzotzot", "shevirat_hakelim"],
            },

            # Hasidic Philosophy
            {
                "concept_id": "hasidic_devekut",
                "name": "Devekut",
                "tradition": PhilosophicalTradition.HASIDIC_PHILOSOPHY,
                "domain": PhilosophicalDomain.SOTERIOLOGY,
                "definition": "Cleaving or attachment to God - the continuous consciousness of divine presence in all moments and activities. The goal of Hasidic spiritual life, achieved through joy, prayer, and mindfulness.",
                "alternate_names": ["Cleaving", "דבקות"],
                "key_figures": ["Baal Shem Tov", "Dov Ber of Mezeritch"],
                "related_concepts": ["avodah_begashmiyut", "bittul", "hitlahavut"],
            },
            {
                "concept_id": "hasidic_bittul",
                "name": "Bittul",
                "tradition": PhilosophicalTradition.HASIDIC_PHILOSOPHY,
                "domain": PhilosophicalDomain.SOTERIOLOGY,
                "definition": "Self-nullification or ego-annihilation - the dissolution of the sense of separate selfhood in awareness of divine omnipresence. Related to but distinct from Sufi fana.",
                "alternate_names": ["Self-Nullification", "ביטול"],
                "key_figures": ["Dov Ber of Mezeritch", "Schneur Zalman of Liadi"],
                "related_concepts": ["devekut", "ayin", "yesh"],
            },

            # Jewish Existentialism
            {
                "concept_id": "buber_i_thou",
                "name": "I-Thou",
                "tradition": PhilosophicalTradition.JEWISH_EXISTENTIALISM,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The mode of genuine encounter with another as a whole being, not as an object to be used (I-It). In I-Thou relation, both parties are fully present; the eternal Thou (God) is glimpsed in every genuine Thou.",
                "alternate_names": ["Ich-Du", "אני-אתה"],
                "key_figures": ["Martin Buber"],
                "related_concepts": ["i_it", "dialogue", "between"],
                "primary_texts": ["I and Thou"],
            },
            {
                "concept_id": "levinas_face",
                "name": "The Face",
                "tradition": PhilosophicalTradition.JEWISH_EXISTENTIALISM,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Le Visage - the epiphany of the Other that breaks through representation and issues an ethical command: 'Do not murder.' The face reveals infinity and establishes ethics as first philosophy.",
                "alternate_names": ["Le Visage", "הפנים"],
                "key_figures": ["Emmanuel Levinas"],
                "related_concepts": ["alterity", "infinite_responsibility", "saying_said"],
                "primary_texts": ["Totality and Infinity", "Otherwise than Being"],
            },

            # === AFRICAN TRADITIONS ===

            # Ubuntu Philosophy
            {
                "concept_id": "ubuntu_personhood",
                "name": "Ubuntu",
                "tradition": PhilosophicalTradition.UBUNTU,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "A person is a person through other persons (Umuntu ngumuntu ngabantu). Personhood is not given at birth but achieved through participation in community and recognition of interdependence.",
                "alternate_names": ["Botho", "Hunhu"],
                "key_figures": ["Desmond Tutu", "Mogobe Ramose", "Augustine Shutte"],
                "related_concepts": ["communalism", "personhood", "relationship"],
            },
            {
                "concept_id": "ubuntu_communalism",
                "name": "African Communalism",
                "tradition": PhilosophicalTradition.UBUNTU,
                "domain": PhilosophicalDomain.POLITICAL_PHILOSOPHY,
                "definition": "The philosophical view that the community has ontological and ethical priority over the individual. Individual identity and fulfillment are realized through community participation.",
                "alternate_names": ["Ujamaa", "Harambee"],
                "key_figures": ["Julius Nyerere", "Kwame Nkrumah", "Leopold Senghor"],
                "related_concepts": ["ubuntu", "personhood", "collective_responsibility"],
            },

            # Egyptian Ancient Philosophy
            {
                "concept_id": "egyptian_maat",
                "name": "Ma'at",
                "tradition": PhilosophicalTradition.EGYPTIAN_ANCIENT,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Truth, justice, cosmic order, and balance - the fundamental principle underlying Egyptian ethics, law, and cosmology. Living in accordance with Ma'at maintains cosmic harmony.",
                "alternate_names": ["Maat", "معات"],
                "key_figures": ["Ptahhotep", "Amenemope"],
                "related_concepts": ["isfet", "weighing_of_heart", "cosmic_order"],
                "primary_texts": ["Instructions of Ptahhotep", "Book of the Dead"],
            },
            {
                "concept_id": "egyptian_ba_ka",
                "name": "Ba and Ka",
                "tradition": PhilosophicalTradition.EGYPTIAN_ANCIENT,
                "domain": PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                "definition": "The Ba is the personality or soul that can travel between worlds; the Ka is the vital essence or life force that needs sustenance. Together with other aspects, they constitute the Egyptian conception of personhood.",
                "alternate_names": ["Soul Aspects"],
                "key_figures": ["Egyptian priesthood"],
                "related_concepts": ["akh", "maat", "afterlife"],
            },

            # Akan Philosophy
            {
                "concept_id": "akan_okra",
                "name": "Okra",
                "tradition": PhilosophicalTradition.AKAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                "definition": "The divine spark or soul given directly by Onyame (God) to each person, carrying their destiny (nkrabea). The okra is immortal and returns to God after death.",
                "alternate_names": ["Soul", "Divine Spark"],
                "key_figures": ["Kwame Gyekye", "Kwasi Wiredu"],
                "related_concepts": ["sunsum", "nkrabea", "mogya"],
            },
            {
                "concept_id": "akan_sankofa",
                "name": "Sankofa",
                "tradition": PhilosophicalTradition.AKAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Go back and fetch it - the principle that one must retrieve valuable knowledge from the past to move forward wisely. Symbolized by a bird looking backward while moving forward.",
                "alternate_names": ["Return and Get It"],
                "key_figures": ["Akan elders"],
                "related_concepts": ["ancestral_wisdom", "tradition", "progress"],
            },

            # Ethiopian Philosophy
            {
                "concept_id": "ethiopian_hatata",
                "name": "Hatata",
                "tradition": PhilosophicalTradition.ETHIOPIAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Critical inquiry or investigation - the method of questioning inherited beliefs through rational examination. Zera Yacob's Hatata is a 17th-century rationalist philosophical treatise.",
                "alternate_names": ["Critical Inquiry"],
                "key_figures": ["Zera Yacob", "Walda Heywat"],
                "related_concepts": ["reason", "natural_religion", "moral_inquiry"],
                "primary_texts": ["Hatata"],
            },

            # Sage Philosophy
            {
                "concept_id": "sage_philosophic_sagacity",
                "name": "Philosophic Sagacity",
                "tradition": PhilosophicalTradition.SAGE_PHILOSOPHY,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "The identification of rigorous philosophical thinking within traditional African thought, through dialogue with indigenous sages who engage in critical reflection rather than mere repetition of tradition.",
                "alternate_names": ["African Sage Philosophy"],
                "key_figures": ["H. Odera Oruka", "Ogotemmêli"],
                "related_concepts": ["ethnophilosophy", "critical_thinking", "oral_tradition"],
            },

            # === INDIGENOUS AMERICAS ===

            # Nahua/Aztec Philosophy
            {
                "concept_id": "nahua_teotl",
                "name": "Teotl",
                "tradition": PhilosophicalTradition.NAHUA_AZTEC,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The sacred, dynamic, self-generating cosmic force that underlies all reality. Teotl is not a being but an ongoing process of sacred motion-change that constitutes all existence.",
                "alternate_names": ["Sacred Energy", "Divine Force"],
                "key_figures": ["Nezahualcoyotl", "Nahua tlamatinime"],
                "related_concepts": ["ollin", "nahui_ollin", "nepantla"],
            },
            {
                "concept_id": "nahua_flower_song",
                "name": "In Xochitl In Cuicatl",
                "tradition": PhilosophicalTradition.NAHUA_AZTEC,
                "domain": PhilosophicalDomain.AESTHETICS,
                "definition": "Flower and Song - the Nahua concept that truth and wisdom are best expressed through poetry and art. Philosophy is pursued through aesthetic creation, not abstract argumentation.",
                "alternate_names": ["Flower and Song", "Poetry-Philosophy"],
                "key_figures": ["Nezahualcoyotl", "Tecayehuatzin"],
                "related_concepts": ["teotl", "neltiliztli", "truth"],
                "primary_texts": ["Cantares Mexicanos", "Romances de los Señores"],
            },
            {
                "concept_id": "nahua_neltiliztli",
                "name": "Neltiliztli",
                "tradition": PhilosophicalTradition.NAHUA_AZTEC,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Truth as rootedness or being well-grounded. True knowledge comes from being firmly rooted in reality like a tree, not from abstract correspondence or coherence.",
                "alternate_names": ["Rootedness", "Well-Grounded Truth"],
                "key_figures": ["Nahua tlamatinime"],
                "related_concepts": ["teotl", "in_xochitl_in_cuicatl", "wisdom"],
            },

            # Maya Philosophy
            {
                "concept_id": "maya_cyclical_time",
                "name": "Maya Cyclical Time",
                "tradition": PhilosophicalTradition.MAYA_PHILOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The understanding that time moves in interlocking cycles (kin, uinal, tun, katun, baktun), with events recurring in transformed patterns. The future can be read through the past.",
                "alternate_names": ["K'uhul Time", "Sacred Calendar"],
                "key_figures": ["Maya astronomers and daykeepers"],
                "related_concepts": ["tzolkin", "haab", "long_count"],
                "primary_texts": ["Popol Vuh", "Dresden Codex"],
            },
            {
                "concept_id": "maya_human_corn",
                "name": "Humans of Maize",
                "tradition": PhilosophicalTradition.MAYA_PHILOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The Maya creation narrative that humans were successfully created from maize after failed attempts with mud and wood. Establishes deep kinship between humans and the corn plant.",
                "alternate_names": ["Maize People"],
                "key_figures": ["Xmucane", "Xpiacoc"],
                "related_concepts": ["creation", "kinship", "agriculture"],
                "primary_texts": ["Popol Vuh"],
            },

            # Andean Philosophy
            {
                "concept_id": "andean_ayni",
                "name": "Ayni",
                "tradition": PhilosophicalTradition.ANDEAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Sacred reciprocity - the fundamental principle governing all relationships: with other humans, with nature, with the cosmos. Giving must be balanced with receiving in all dimensions.",
                "alternate_names": ["Reciprocity", "Mutual Aid"],
                "key_figures": ["Contemporary Andean philosophers"],
                "related_concepts": ["pachamama", "ayllu", "minka"],
            },
            {
                "concept_id": "andean_pachamama",
                "name": "Pachamama",
                "tradition": PhilosophicalTradition.ANDEAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Not merely 'Mother Earth' but the living totality of time-space-world that we participate in. Pachamama is not separate from humanity but includes us as participants.",
                "alternate_names": ["World Mother", "Cosmic Mother"],
                "key_figures": ["Contemporary Andean thinkers"],
                "related_concepts": ["ayni", "ayllu", "kawsay"],
            },
            {
                "concept_id": "andean_sumak_kawsay",
                "name": "Sumak Kawsay",
                "tradition": PhilosophicalTradition.ANDEAN_PHILOSOPHY,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Buen Vivir or Living Well - not 'living better' (than others) but living in harmony with community, nature, and cosmos. An alternative to development-as-growth paradigms.",
                "alternate_names": ["Buen Vivir", "Good Living"],
                "key_figures": ["Contemporary Andean philosophers"],
                "related_concepts": ["ayni", "pachamama", "ayllu"],
            },

            # Native American Philosophy
            {
                "concept_id": "native_relational_place",
                "name": "Relational Place",
                "tradition": PhilosophicalTradition.NATIVE_AMERICAN,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Place is not mere location but a living participant in meaning and identity. Land is not property but relative - a web of relationships that includes humans, animals, plants, and spirits.",
                "alternate_names": ["Sacred Geography"],
                "key_figures": ["Vine Deloria Jr.", "Brian Burkhart"],
                "related_concepts": ["all_my_relations", "land_ethics", "indigenous_metaphysics"],
                "primary_texts": ["God Is Red", "Spirit and Reason"],
            },
            {
                "concept_id": "native_seven_generations",
                "name": "Seven Generations",
                "tradition": PhilosophicalTradition.NATIVE_AMERICAN,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "The principle that decisions should consider their impact seven generations into the future. Long-term thinking that extends moral consideration across time.",
                "alternate_names": ["Seventh Generation Principle"],
                "key_figures": ["Haudenosaunee (Iroquois) tradition"],
                "related_concepts": ["sustainability", "future_ethics", "intergenerational_justice"],
            },
            {
                "concept_id": "native_mitakuye_oyasin",
                "name": "Mitakuye Oyasin",
                "tradition": PhilosophicalTradition.NATIVE_AMERICAN,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "All My Relations - Lakota phrase expressing the interconnection of all beings: humans, animals, plants, stones, ancestors, and spirits. Relational ontology where being is being-with.",
                "alternate_names": ["All My Relations"],
                "key_figures": ["Lakota tradition"],
                "related_concepts": ["relational_place", "kinship", "interconnection"],
            },

            # === LATIN AMERICAN TRADITIONS ===

            # Liberation Philosophy
            {
                "concept_id": "liberation_exteriority",
                "name": "Exteriority",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "That which is beyond or outside the totalizing system - the Other who cannot be reduced to the Same. The poor, the marginalized, the colonized exist in exteriority to dominant totalities.",
                "alternate_names": ["Exterioridad"],
                "key_figures": ["Enrique Dussel", "Emmanuel Levinas"],
                "related_concepts": ["totality", "other", "liberation"],
                "primary_texts": ["Philosophy of Liberation"],
            },
            {
                "concept_id": "liberation_analeptic",
                "name": "Analeptic Method",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "Philosophical method that begins from the voice of the Other, from the periphery, from those excluded by the dominant system. Reverses the direction of dialectical thought.",
                "alternate_names": ["Anadialectical Method"],
                "key_figures": ["Enrique Dussel"],
                "related_concepts": ["exteriority", "praxis", "decolonization"],
            },
            {
                "concept_id": "liberation_transmodernity",
                "name": "Transmodernity",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domain": PhilosophicalDomain.POLITICAL_PHILOSOPHY,
                "definition": "Neither pre-modern nor modern nor post-modern, but trans-modern: recovering what modernity excluded while transcending modernity's limitations. A pluriversal rather than universal project.",
                "alternate_names": ["Transmodernidad"],
                "key_figures": ["Enrique Dussel"],
                "related_concepts": ["exteriority", "decoloniality", "pluriversality"],
            },

            # === ADDITIONAL BUDDHIST SCHOOLS ===

            # Huayan Buddhism
            {
                "concept_id": "huayan_interpenetration",
                "name": "Mutual Interpenetration",
                "tradition": PhilosophicalTradition.BUDDHIST_HUAYAN,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "Shi shi wu ai - the unobstructed interpenetration of all phenomena with all phenomena. Every thing contains every other thing; the whole is in each part and each part in the whole.",
                "alternate_names": ["Shi Shi Wu Ai", "事事無礙"],
                "key_figures": ["Fazang", "Chengguan", "Dushun"],
                "related_concepts": ["indras_net", "dharmadhatu", "li_shi"],
                "primary_texts": ["Huayan Sutra", "Essay on the Golden Lion"],
            },
            {
                "concept_id": "huayan_indras_net",
                "name": "Indra's Net",
                "tradition": PhilosophicalTradition.BUDDHIST_HUAYAN,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The cosmic metaphor for interpenetration: an infinite net with a jewel at each intersection, each jewel reflecting all other jewels infinitely. Visualizes how each phenomenon contains all others.",
                "alternate_names": ["Net of Indra", "因陀羅網"],
                "key_figures": ["Fazang"],
                "related_concepts": ["interpenetration", "dharmadhatu", "totality"],
            },

            # Tiantai Buddhism
            {
                "concept_id": "tiantai_three_truths",
                "name": "Threefold Truth",
                "tradition": PhilosophicalTradition.BUDDHIST_TIANTAI,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "San Di - emptiness (ku), conventional existence (ke), and the middle (chū) are three aspects of a single reality, to be realized simultaneously, not sequentially.",
                "alternate_names": ["San Di", "三諦"],
                "key_figures": ["Zhiyi", "Zhanran"],
                "related_concepts": ["ichinen_sanzen", "perfect_sudden", "buddha_nature"],
                "primary_texts": ["Mohe Zhiguan"],
            },
            {
                "concept_id": "tiantai_ichinen_sanzen",
                "name": "Ichinen Sanzen",
                "tradition": PhilosophicalTradition.BUDDHIST_TIANTAI,
                "domain": PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                "definition": "Three thousand realms in a single thought-moment - the entire universe with all its dimensions is present in each moment of consciousness. Microcosm and macrocosm are identical.",
                "alternate_names": ["一念三千", "3000 Worlds in One Thought"],
                "key_figures": ["Zhiyi"],
                "related_concepts": ["three_truths", "buddha_nature", "original_enlightenment"],
            },

            # === ADDITIONAL SOUTH ASIAN ===

            # Jain Philosophy
            {
                "concept_id": "jain_anekantavada",
                "name": "Anekantavada",
                "tradition": PhilosophicalTradition.JAIN,
                "domain": PhilosophicalDomain.EPISTEMOLOGY,
                "definition": "The doctrine of many-sidedness or non-absolutism: reality has infinite aspects and no single viewpoint can capture the whole truth. All claims are contextually limited.",
                "alternate_names": ["Many-Sidedness", "Non-Absolutism"],
                "key_figures": ["Mahavira", "Kundakunda", "Haribhadra"],
                "related_concepts": ["syadvada", "nayavada", "ahimsa"],
            },
            {
                "concept_id": "jain_syadvada",
                "name": "Syadvada",
                "tradition": PhilosophicalTradition.JAIN,
                "domain": PhilosophicalDomain.LOGIC,
                "definition": "The doctrine of conditional predication: every statement should be qualified with 'syat' (perhaps, in some respect). Seven modes of predication express the complexity of truth.",
                "alternate_names": ["Perhaps-ism", "Seven-fold Predication"],
                "key_figures": ["Kundakunda", "Samantabhadra"],
                "related_concepts": ["anekantavada", "saptabhangi", "nayavada"],
            },
            {
                "concept_id": "jain_ahimsa",
                "name": "Ahimsa",
                "tradition": PhilosophicalTradition.JAIN,
                "domain": PhilosophicalDomain.ETHICS,
                "definition": "Non-violence toward all living beings (jivas) - the supreme ethical principle in Jain philosophy. Extends to mind, speech, and body; to all life forms including microscopic beings.",
                "alternate_names": ["Non-Violence", "Non-Harm"],
                "key_figures": ["Mahavira", "Gandhi (influenced by)"],
                "related_concepts": ["anekantavada", "aparigraha", "jiva"],
            },

            # === FOLK-PHILOSOPHICAL BRIDGES ===
            # Traditions bridging formal philosophy and indigenous wisdom

            # Celtic Druidic
            {
                "concept_id": "celtic_awen",
                "name": "Awen",
                "tradition": PhilosophicalTradition.CELTIC_DRUIDIC,
                "domain": PhilosophicalDomain.ORAL_EPISTEMOLOGY,
                "definition": "The flowing spirit of poetic inspiration and divine knowledge in Celtic tradition. Represents the illuminating breath that connects humans to sacred wisdom through poetry, prophecy, and bardic arts.",
                "alternate_names": ["Divine Inspiration", "Flowing Spirit"],
                "key_figures": ["Taliesin", "Amergin"],
                "related_concepts": ["imbas_forosnai", "salmon_of_knowledge", "otherworld"],
            },
            {
                "concept_id": "celtic_threefold_world",
                "name": "Three Worlds",
                "tradition": PhilosophicalTradition.CELTIC_DRUIDIC,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The Celtic cosmology of Land, Sea, and Sky - three interconnected realms that together comprise reality. Land is the physical world, Sea the underworld/unconscious, Sky the heavenly/divine realm.",
                "alternate_names": ["Land-Sea-Sky", "Celtic Cosmology"],
                "related_concepts": ["otherworld", "thin_places", "sovereignty"],
            },

            # Norse Philosophical
            {
                "concept_id": "norse_wyrd",
                "name": "Wyrd",
                "tradition": PhilosophicalTradition.NORSE_PHILOSOPHICAL,
                "domain": PhilosophicalDomain.METAPHYSICS,
                "definition": "The web of fate woven by the Norns from all past actions and events. Not deterministic but an evolving pattern that shapes possibilities while allowing for human agency through orlog (personal fate).",
                "alternate_names": ["Fate-Web", "Urðr"],
                "related_concepts": ["orlog", "norns", "yggdrasil"],
            },
            {
                "concept_id": "norse_seidr",
                "name": "Seiðr",
                "tradition": PhilosophicalTradition.NORSE_PHILOSOPHICAL,
                "domain": PhilosophicalDomain.CEREMONIAL_KNOWLEDGE,
                "definition": "Norse shamanic practice involving altered consciousness, prophecy, and fate-weaving. A form of ecstatic magic that allows practitioners to see and manipulate wyrd.",
                "alternate_names": ["Norse Shamanism", "Fate-Working"],
                "key_figures": ["Freyja", "Odin"],
                "related_concepts": ["wyrd", "volva", "galdr"],
            },

            # Slavic Wisdom
            {
                "concept_id": "slavic_rod",
                "name": "Rod",
                "tradition": PhilosophicalTradition.SLAVIC_WISDOM,
                "domain": PhilosophicalDomain.ANCESTRAL_PHILOSOPHY,
                "definition": "The cosmic principle of kinship and ancestry in Slavic thought - both the divine source of all life and the web of ancestral connections that bind individuals to family, tribe, and cosmos.",
                "alternate_names": ["Kin-Force", "Ancestral Bond"],
                "related_concepts": ["rozhanitsy", "domovoi", "perun"],
            },
            {
                "concept_id": "slavic_prav",
                "name": "Prav",
                "tradition": PhilosophicalTradition.SLAVIC_WISDOM,
                "domain": PhilosophicalDomain.FOLK_ETHICS,
                "definition": "The cosmic law or truth that governs the universe in Slavic cosmology. Represents righteous order, truth, and the proper way of living in harmony with cosmic principles.",
                "alternate_names": ["Cosmic Truth", "Right Order"],
                "related_concepts": ["yav", "nav", "rod"],
            },

            # Aboriginal Australian
            {
                "concept_id": "aboriginal_dreamtime",
                "name": "The Dreaming",
                "tradition": PhilosophicalTradition.ABORIGINAL_AUSTRALIAN,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The eternal, ever-present dimension of reality where ancestral beings created the world and continue to inhabit it. Not merely a past era but an ongoing spiritual reality accessible through ceremony, story, and Country.",
                "alternate_names": ["Dreamtime", "Tjukurrpa", "Jukurrpa"],
                "related_concepts": ["country", "songlines", "law"],
            },
            {
                "concept_id": "aboriginal_country",
                "name": "Country",
                "tradition": PhilosophicalTradition.ABORIGINAL_AUSTRALIAN,
                "domain": PhilosophicalDomain.TRADITIONAL_ECOLOGY,
                "definition": "The living landscape with which Aboriginal peoples have reciprocal relationships of care and belonging. Country is a sentient entity that knows, speaks, and holds ancestral law.",
                "alternate_names": ["Ngurra", "Sacred Land"],
                "related_concepts": ["dreaming", "songlines", "caring_for_country"],
            },

            # Polynesian Philosophical
            {
                "concept_id": "polynesian_mana",
                "name": "Mana",
                "tradition": PhilosophicalTradition.POLYNESIAN_PHILOSOPHICAL,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The spiritual power and prestige that flows through persons, objects, and places. Can be inherited, earned through achievement, or transferred through contact. Indicates effectiveness and sacred authority.",
                "alternate_names": ["Spiritual Power", "Divine Force"],
                "related_concepts": ["tapu", "noa", "mauri"],
            },
            {
                "concept_id": "polynesian_tapu",
                "name": "Tapu",
                "tradition": PhilosophicalTradition.POLYNESIAN_PHILOSOPHICAL,
                "domain": PhilosophicalDomain.FOLK_ETHICS,
                "definition": "Sacred prohibition or restriction that protects mana and maintains cosmic order. Violation of tapu disrupts spiritual balance and invites misfortune. The opposite of noa (ordinary, unrestricted).",
                "alternate_names": ["Taboo", "Sacred Restriction"],
                "related_concepts": ["mana", "noa", "mauri"],
            },

            # Siberian Shamanic
            {
                "concept_id": "siberian_world_tree",
                "name": "World Tree",
                "tradition": PhilosophicalTradition.SIBERIAN_SHAMANIC,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The cosmic axis connecting the upper, middle, and lower worlds in Siberian cosmology. The shaman climbs the world tree to journey between realms and communicate with spirits.",
                "alternate_names": ["Axis Mundi", "Cosmic Tree"],
                "related_concepts": ["three_worlds", "spirit_journey", "shaman_drum"],
            },
            {
                "concept_id": "siberian_spirit_helpers",
                "name": "Spirit Helpers",
                "tradition": PhilosophicalTradition.SIBERIAN_SHAMANIC,
                "domain": PhilosophicalDomain.CEREMONIAL_KNOWLEDGE,
                "definition": "The tutelary spirits that assist shamans in their work - animal spirits, ancestral spirits, and nature spirits who provide guidance, protection, and power for healing and divination.",
                "alternate_names": ["Tutelary Spirits", "Power Animals"],
                "related_concepts": ["world_tree", "shamanic_illness", "soul_retrieval"],
            },

            # Inuit Wisdom
            {
                "concept_id": "inuit_inua",
                "name": "Inua",
                "tradition": PhilosophicalTradition.INUIT_WISDOM,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The indwelling spirit or soul that animates all beings and things in Inuit cosmology - humans, animals, plants, and even tools and natural features all possess inua.",
                "alternate_names": ["Inner Person", "Soul"],
                "related_concepts": ["tarniq", "silap_inua", "anirniq"],
            },
            {
                "concept_id": "inuit_sila",
                "name": "Sila",
                "tradition": PhilosophicalTradition.INUIT_WISDOM,
                "domain": PhilosophicalDomain.TRADITIONAL_ECOLOGY,
                "definition": "The breath, weather, and cosmic intelligence that pervades all things. Sila is both the air we breathe and the consciousness of the universe - the binding force connecting all life.",
                "alternate_names": ["Weather-Breath", "Universal Mind"],
                "related_concepts": ["inua", "angakuq", "nuna"],
            },

            # African Cosmological
            {
                "concept_id": "african_vital_force",
                "name": "Vital Force",
                "tradition": PhilosophicalTradition.AFRICAN_COSMOLOGICAL,
                "domain": PhilosophicalDomain.ANIMISTIC_METAPHYSICS,
                "definition": "The life energy that permeates all existence in African metaphysics - a dynamic force that can be strengthened, diminished, or transferred between beings. The basis of healing, magic, and social ethics.",
                "alternate_names": ["Ashe", "Nyama", "Life Force"],
                "key_figures": ["Placide Tempels"],
                "related_concepts": ["ancestor_veneration", "ubuntu", "cosmic_harmony"],
                "primary_texts": ["Bantu Philosophy"],
            },
            {
                "concept_id": "african_ancestor_veneration",
                "name": "Ancestor Veneration",
                "tradition": PhilosophicalTradition.AFRICAN_COSMOLOGICAL,
                "domain": PhilosophicalDomain.ANCESTRAL_PHILOSOPHY,
                "definition": "The living relationship with deceased ancestors who continue to participate in family and community life. Ancestors mediate between the human and spirit worlds and maintain social order.",
                "alternate_names": ["Ancestor Worship", "Living Dead"],
                "related_concepts": ["vital_force", "elderhood", "ritual_obligation"],
            },
        ]

    async def initialize_seed_concepts(self) -> int:
        """
        Initialize the concept index with seed concepts for all traditions.

        Returns:
            Number of concepts added
        """
        seed_concepts = self._get_seed_concepts()
        count = 0

        for concept_data in seed_concepts:
            concept = PhilosophicalConcept(
                concept_id=concept_data["concept_id"],
                name=concept_data["name"],
                tradition=concept_data["tradition"],
                domain=concept_data["domain"],
                definition=concept_data["definition"],
                alternate_names=concept_data.get("alternate_names", []),
                key_figures=concept_data.get("key_figures", []),
                related_concepts=concept_data.get("related_concepts", []),
                primary_texts=concept_data.get("primary_texts", []),
                maturity_score=0.3,  # Seed concepts start with base maturity
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_concept(concept)
            count += 1

        logger.info(f"Initialized {count} seed concepts across all traditions")
        return count

    def _get_seed_figures(self) -> List[Dict[str, Any]]:
        """
        Return seed figures (philosophers/thinkers) for all philosophical traditions.

        These provide initial knowledge base entries that can be expanded
        through the research agent system.
        """
        return [
            # === ISLAMIC TRADITIONS ===

            # Sufi Philosophy
            {
                "figure_id": "ibn_arabi",
                "name": "Ibn Arabi",
                "alternate_names": ["Muhyiddin Ibn Arabi", "Sheikh al-Akbar", "محيي الدين ابن عربي"],
                "birth_year": 1165,
                "death_year": 1240,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.SUFI_PHILOSOPHY],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["Wahdat al-Wujud (Unity of Being)", "Divine Names", "Perfect Human (al-Insan al-Kamil)", "Barzakh (Isthmus)"],
                "key_works": ["Fusus al-Hikam", "Al-Futuhat al-Makkiyya"],
                "teachers": ["Abu Madyan", "Shams of Marchena"],
                "influenced": ["Sadr al-Din Qunawi", "Jami", "Mulla Sadra"],
                "biography_summary": "Andalusian Sufi mystic and philosopher, known as 'The Greatest Sheikh.' Developed the metaphysical doctrine of the Unity of Being and wrote extensively on mystical experience and divine knowledge.",
            },
            {
                "figure_id": "rumi",
                "name": "Jalal al-Din Rumi",
                "alternate_names": ["Mawlana", "Mevlana", "جلال‌الدین محمد رومی"],
                "birth_year": 1207,
                "death_year": 1273,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.SUFI_PHILOSOPHY],
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.AESTHETICS],
                "core_ideas": ["Divine Love", "Spiritual Transformation", "Unity with the Beloved", "Whirling as Meditation"],
                "key_works": ["Masnavi", "Divan-e Shams-e Tabrizi", "Fihi Ma Fihi"],
                "teachers": ["Shams-i-Tabrizi", "Burhan al-Din Muhaqqiq"],
                "influenced": ["Mevlevi Order", "Persian poetry tradition"],
                "biography_summary": "Persian poet, scholar, and Sufi mystic. His poetry explores divine love, spiritual longing, and the soul's journey to union with God. Founded the Mevlevi Order of whirling dervishes.",
            },
            {
                "figure_id": "al_ghazali",
                "name": "Abu Hamid al-Ghazali",
                "alternate_names": ["Algazel", "Hujjat al-Islam", "أبو حامد الغزالي"],
                "birth_year": 1058,
                "death_year": 1111,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.SUFI_PHILOSOPHY, PhilosophicalTradition.KALAM],
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS],
                "core_ideas": ["Critique of Philosophers", "Occasionalism", "Spiritual Revival", "Certainty through Experience"],
                "key_works": ["Tahafut al-Falasifa", "Ihya Ulum al-Din", "Munqidh min al-Dalal"],
                "teachers": ["Al-Juwayni"],
                "influenced": ["Islamic theology", "Mulla Sadra", "Western Scholastics"],
                "biography_summary": "Persian theologian, philosopher, and mystic. Critiqued Aristotelian philosophy while integrating Sufi spirituality into orthodox Islam. His work profoundly shaped Islamic intellectual history.",
            },

            # Falsafa (Islamic Peripatetic)
            {
                "figure_id": "ibn_sina",
                "name": "Ibn Sina",
                "alternate_names": ["Avicenna", "Abu Ali al-Husayn", "ابن سینا"],
                "birth_year": 980,
                "death_year": 1037,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.FALSAFA],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "core_ideas": ["Necessary Being", "Essence-Existence Distinction", "Flying Man Argument", "Active Intellect"],
                "key_works": ["Al-Shifa (The Healing)", "Kitab al-Najat", "Kitab al-Isharat"],
                "teachers": ["Al-Natili"],
                "influences": ["Aristotle", "Al-Farabi"],
                "influenced": ["Aquinas", "Suhrawardi", "Mulla Sadra", "Latin Scholastics"],
                "biography_summary": "Persian polymath and philosopher, the most influential figure in Islamic philosophy. His synthesis of Aristotelian and Neoplatonic thought dominated philosophical discourse for centuries.",
            },
            {
                "figure_id": "al_farabi",
                "name": "Al-Farabi",
                "alternate_names": ["Alpharabius", "Second Teacher", "الفارابي"],
                "birth_year": 872,
                "death_year": 950,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.FALSAFA],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.LOGIC],
                "core_ideas": ["Emanation", "Virtuous City", "Harmony of Plato and Aristotle", "Active Intellect"],
                "key_works": ["Ara Ahl al-Madina al-Fadila", "Kitab al-Huruf", "Ihsa al-Ulum"],
                "influences": ["Aristotle", "Plato"],
                "influenced": ["Ibn Sina", "Ibn Rushd", "Maimonides"],
                "biography_summary": "Known as the 'Second Teacher' (after Aristotle). Founded Islamic political philosophy and developed the emanationist cosmology that became standard in falsafa.",
            },
            {
                "figure_id": "ibn_rushd",
                "name": "Ibn Rushd",
                "alternate_names": ["Averroes", "The Commentator", "ابن رشد"],
                "birth_year": 1126,
                "death_year": 1198,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.FALSAFA],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.LOGIC],
                "core_ideas": ["Double Truth", "Eternity of World", "Unity of Intellect", "Defense of Philosophy"],
                "key_works": ["Tahafut al-Tahafut", "Long Commentary on De Anima", "Fasl al-Maqal"],
                "influences": ["Aristotle", "Ibn Sina"],
                "influenced": ["Latin Averroism", "Aquinas", "Maimonides"],
                "biography_summary": "Andalusian philosopher, known as 'The Commentator' for his extensive Aristotle commentaries. Defended philosophy against al-Ghazali's critique and profoundly influenced medieval European thought.",
            },

            # Illuminationist
            {
                "figure_id": "suhrawardi",
                "name": "Suhrawardi",
                "alternate_names": ["Shahab al-Din Suhrawardi", "Sheikh al-Ishraq", "شهاب‌الدین سهروردی"],
                "birth_year": 1154,
                "death_year": 1191,
                "era": "Medieval Islamic",
                "traditions": [PhilosophicalTradition.ILLUMINATIONIST],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY],
                "core_ideas": ["Light Metaphysics", "Knowledge by Presence", "Hierarchy of Lights", "Oriental Wisdom"],
                "key_works": ["Hikmat al-Ishraq", "Hayakil al-Nur", "Alwah-i Imadi"],
                "influences": ["Plato", "Zoroastrian tradition", "Ibn Sina"],
                "influenced": ["Mulla Sadra", "Persian Illuminationist school"],
                "biography_summary": "Founder of the Illuminationist school. Synthesized Persian wisdom, Platonic ideas, and Islamic thought into a philosophy of light. Executed at age 36 for his controversial views.",
            },

            # Transcendent Theosophy
            {
                "figure_id": "mulla_sadra",
                "name": "Mulla Sadra",
                "alternate_names": ["Sadr al-Din Shirazi", "Sadr al-Muta'allihin", "ملاصدرا"],
                "birth_year": 1571,
                "death_year": 1640,
                "era": "Safavid Persia",
                "traditions": [PhilosophicalTradition.TRANSCENDENT_THEOSOPHY],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "core_ideas": ["Primacy of Existence", "Substantial Motion", "Gradation of Being", "Soul's Becoming"],
                "key_works": ["Al-Asfar al-Arba'a", "Al-Shawahid al-Rububiyya", "Mafatih al-Ghayb"],
                "teachers": ["Mir Damad", "Baha al-Din Amili"],
                "influences": ["Ibn Sina", "Suhrawardi", "Ibn Arabi"],
                "influenced": ["Contemporary Iranian philosophy"],
                "biography_summary": "Greatest philosopher of Safavid Persia, founder of Transcendent Theosophy. Synthesized peripatetic philosophy, Illuminationism, and Sufi gnosis into a unique metaphysical system.",
            },

            # === JEWISH TRADITIONS ===

            # Maimonidean
            {
                "figure_id": "maimonides",
                "name": "Moses Maimonides",
                "alternate_names": ["Rambam", "Moses ben Maimon", "משה בן מימון"],
                "birth_year": 1138,
                "death_year": 1204,
                "era": "Medieval",
                "traditions": [PhilosophicalTradition.MAIMONIDEAN],
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS, PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["Negative Theology", "Intellectual Love of God", "Prophecy Theory", "Thirteen Principles"],
                "key_works": ["Guide for the Perplexed", "Mishneh Torah", "Commentary on the Mishnah"],
                "influences": ["Aristotle", "Al-Farabi", "Ibn Sina"],
                "influenced": ["Aquinas", "Spinoza", "All subsequent Jewish philosophy"],
                "biography_summary": "Greatest medieval Jewish philosopher. Harmonized Aristotelian philosophy with Jewish law and theology. His Guide for the Perplexed remains the foundational work of Jewish rationalist philosophy.",
            },

            # Kabbalistic
            {
                "figure_id": "isaac_luria",
                "name": "Isaac Luria",
                "alternate_names": ["The Ari", "Ha'Ari HaKadosh", "יצחק לוריא"],
                "birth_year": 1534,
                "death_year": 1572,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.KABBALISTIC],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["Tzimtzum", "Shevirat HaKelim", "Tikkun", "Gilgul (Reincarnation)"],
                "key_works": ["Etz Chaim (transmitted by Chaim Vital)", "Shaar HaGilgulim"],
                "influences": ["Zohar", "Moses Cordovero"],
                "influenced": ["All subsequent Kabbalah", "Hasidism"],
                "biography_summary": "The most influential Kabbalist in Jewish history. Developed the doctrines of divine contraction, cosmic shattering, and repair that revolutionized Jewish mystical thought.",
            },
            {
                "figure_id": "moses_cordovero",
                "name": "Moses Cordovero",
                "alternate_names": ["Ramak", "משה קורדובירו"],
                "birth_year": 1522,
                "death_year": 1570,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.KABBALISTIC],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["Systematic Kabbalah", "Sefirot Theory", "Divine Immanence"],
                "key_works": ["Pardes Rimonim", "Tomer Devorah", "Or Ne'erav"],
                "influences": ["Zohar"],
                "influenced": ["Isaac Luria", "Later Kabbalah"],
                "biography_summary": "Leading Kabbalist of Safed before Luria. Systematized Zoharic Kabbalah into a coherent philosophical framework. His work on the sefirot remains foundational.",
            },

            # Hasidic
            {
                "figure_id": "baal_shem_tov",
                "name": "Baal Shem Tov",
                "alternate_names": ["Israel ben Eliezer", "Besht", "בעל שם טוב"],
                "birth_year": 1698,
                "death_year": 1760,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.HASIDIC_PHILOSOPHY],
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS],
                "core_ideas": ["Devekut", "Joy in Worship", "Divine Sparks", "Avodah BeGashmiyut"],
                "key_works": ["Tzava'at HaRivash (compiled teachings)"],
                "influences": ["Lurianic Kabbalah"],
                "influenced": ["All of Hasidism", "Jewish spirituality"],
                "biography_summary": "Founder of Hasidism. Revolutionized Jewish spirituality by emphasizing joy, accessibility of divine connection, and finding holiness in everyday life.",
            },
            {
                "figure_id": "schneur_zalman",
                "name": "Schneur Zalman of Liadi",
                "alternate_names": ["Alter Rebbe", "Baal HaTanya", "שניאור זלמן מליאדי"],
                "birth_year": 1745,
                "death_year": 1812,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.HASIDIC_PHILOSOPHY],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "core_ideas": ["Chabad Philosophy", "Bittul", "Two Souls", "Intellectual Contemplation"],
                "key_works": ["Tanya", "Torah Or", "Likkutei Torah"],
                "teachers": ["Dov Ber of Mezeritch"],
                "influenced": ["Chabad-Lubavitch movement"],
                "biography_summary": "Founder of Chabad Hasidism. Integrated mystical Hasidism with rigorous intellectual analysis. His Tanya is the foundational text of Chabad philosophy.",
            },

            # Jewish Existentialism
            {
                "figure_id": "martin_buber",
                "name": "Martin Buber",
                "alternate_names": ["מרטין בובר"],
                "birth_year": 1878,
                "death_year": 1965,
                "era": "Modern",
                "traditions": [PhilosophicalTradition.JEWISH_EXISTENTIALISM],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["I-Thou", "Dialogue", "The Between", "Eternal Thou"],
                "key_works": ["I and Thou", "Tales of the Hasidim", "Eclipse of God"],
                "influences": ["Hasidism", "Kierkegaard", "Nietzsche"],
                "influenced": ["Emmanuel Levinas", "Gabriel Marcel", "Paul Tillich"],
                "biography_summary": "Austrian-Israeli philosopher of dialogue. His I-Thou philosophy transformed understanding of human relationships, ethics, and the encounter with God.",
            },
            {
                "figure_id": "emmanuel_levinas",
                "name": "Emmanuel Levinas",
                "alternate_names": ["עמנואל לוינס"],
                "birth_year": 1906,
                "death_year": 1995,
                "era": "Modern",
                "traditions": [PhilosophicalTradition.JEWISH_EXISTENTIALISM, PhilosophicalTradition.PHENOMENOLOGY],
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHENOMENOLOGICAL],
                "core_ideas": ["The Face", "Infinite Responsibility", "Ethics as First Philosophy", "Alterity"],
                "key_works": ["Totality and Infinity", "Otherwise than Being", "Nine Talmudic Readings"],
                "teachers": ["Husserl", "Heidegger"],
                "influences": ["Buber", "Talmudic tradition"],
                "influenced": ["Derrida", "Contemporary ethics"],
                "biography_summary": "Lithuanian-French philosopher who made ethics foundational to philosophy. His analysis of the face of the Other and infinite responsibility transformed continental ethics.",
            },
            {
                "figure_id": "franz_rosenzweig",
                "name": "Franz Rosenzweig",
                "alternate_names": ["פרנץ רוזנצווייג"],
                "birth_year": 1886,
                "death_year": 1929,
                "era": "Modern",
                "traditions": [PhilosophicalTradition.JEWISH_EXISTENTIALISM],
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["New Thinking", "Creation-Revelation-Redemption", "Star of Redemption", "Speech-Thinking"],
                "key_works": ["The Star of Redemption", "Understanding the Sick and the Healthy"],
                "influences": ["Hegel", "Schelling", "Hermann Cohen"],
                "influenced": ["Buber", "Levinas", "Jewish philosophy"],
                "biography_summary": "German-Jewish philosopher who created a new philosophy of Judaism rooted in existence and relation rather than abstract systems. His Star of Redemption is a masterwork of religious existentialism.",
            },

            # === AFRICAN TRADITIONS ===

            # Ubuntu
            {
                "figure_id": "desmond_tutu",
                "name": "Desmond Tutu",
                "alternate_names": ["Archbishop Tutu"],
                "birth_year": 1931,
                "death_year": 2021,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.UBUNTU],
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "core_ideas": ["Ubuntu", "Reconciliation", "Restorative Justice", "Rainbow Nation"],
                "key_works": ["No Future Without Forgiveness", "God Has a Dream"],
                "influenced": ["Truth and Reconciliation process", "Global peace movements"],
                "biography_summary": "South African Anglican bishop and Nobel laureate. Articulated Ubuntu philosophy globally and applied it to reconciliation after apartheid through the Truth and Reconciliation Commission.",
            },
            {
                "figure_id": "kwame_gyekye",
                "name": "Kwame Gyekye",
                "alternate_names": [],
                "birth_year": 1939,
                "death_year": 2019,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.AKAN_PHILOSOPHY, PhilosophicalTradition.UBUNTU],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "core_ideas": ["Moderate Communitarianism", "Akan Personhood", "African Identity", "Tradition and Modernity"],
                "key_works": ["An Essay on African Philosophical Thought", "Tradition and Modernity"],
                "biography_summary": "Ghanaian philosopher who systematically analyzed Akan thought. Developed moderate communitarianism balancing individual rights with communal values.",
            },
            {
                "figure_id": "mogobe_ramose",
                "name": "Mogobe Ramose",
                "alternate_names": [],
                "birth_year": 1945,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.UBUNTU],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "core_ideas": ["Ubuntu as Philosophy", "African Jurisprudence", "Epistemological Decolonization"],
                "key_works": ["African Philosophy Through Ubuntu"],
                "biography_summary": "South African philosopher who developed Ubuntu as a comprehensive philosophical system, not merely an ethical maxim. Argued for its status as a worldview encompassing metaphysics, ethics, and epistemology.",
            },
            {
                "figure_id": "kwasi_wiredu",
                "name": "Kwasi Wiredu",
                "alternate_names": [],
                "birth_year": 1931,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.AKAN_PHILOSOPHY],
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE, PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["Conceptual Decolonization", "Consensual Democracy", "African Logic", "Cross-Cultural Philosophy"],
                "key_works": ["Philosophy and an African Culture", "Cultural Universals and Particulars"],
                "influences": ["Analytic philosophy", "Akan tradition"],
                "biography_summary": "Ghanaian philosopher known for 'conceptual decolonization' - analyzing how colonial languages distort African thought. Combined rigorous analytic method with African philosophical resources.",
            },

            # Egyptian Ancient
            {
                "figure_id": "ptahhotep",
                "name": "Ptahhotep",
                "alternate_names": ["Ptah-Hotep"],
                "birth_year": -2400,
                "death_year": -2350,
                "era": "Ancient Egyptian (Old Kingdom)",
                "traditions": [PhilosophicalTradition.EGYPTIAN_ANCIENT],
                "domains": [PhilosophicalDomain.ETHICS],
                "core_ideas": ["Ma'at", "Practical Wisdom", "Social Harmony", "Speech Ethics"],
                "key_works": ["Instructions of Ptahhotep"],
                "biography_summary": "Ancient Egyptian vizier, author of the oldest complete philosophical text. His Instructions articulate Ma'at through practical ethical teachings on speech, relationships, and social conduct.",
            },

            # Ethiopian
            {
                "figure_id": "zera_yacob",
                "name": "Zera Yacob",
                "alternate_names": ["Zär'a Ya'əqob"],
                "birth_year": 1599,
                "death_year": 1692,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.ETHIOPIAN_PHILOSOPHY],
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS],
                "core_ideas": ["Rational Inquiry (Hatata)", "Natural Theology", "Universal Reason", "Religious Criticism"],
                "key_works": ["Hatata"],
                "students": ["Walda Heywat"],
                "biography_summary": "Ethiopian philosopher often compared to Descartes. His Hatata develops a rationalist philosophy through critical inquiry into religion, knowledge, and morality, independent of European influence.",
            },
            {
                "figure_id": "walda_heywat",
                "name": "Walda Heywat",
                "alternate_names": [],
                "birth_year": 1630,
                "death_year": 1700,
                "era": "Early Modern",
                "traditions": [PhilosophicalTradition.ETHIOPIAN_PHILOSOPHY],
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["Applied Rationalism", "Practical Ethics", "Religious Tolerance"],
                "key_works": ["Hatata (continuation of Zera Yacob's work)"],
                "teachers": ["Zera Yacob"],
                "biography_summary": "Student of Zera Yacob who continued his philosophical work. His Hatata applies rational principles to ethical and social questions.",
            },

            # Sage Philosophy
            {
                "figure_id": "odera_oruka",
                "name": "Henry Odera Oruka",
                "alternate_names": [],
                "birth_year": 1944,
                "death_year": 1995,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.SAGE_PHILOSOPHY],
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.ETHICS],
                "core_ideas": ["Sage Philosophy", "Philosophic Sagacity", "Critique of Ethnophilosophy", "Four Trends in African Philosophy"],
                "key_works": ["Sage Philosophy", "Trends in Contemporary African Philosophy"],
                "biography_summary": "Kenyan philosopher who pioneered the 'sage philosophy' project - documenting and analyzing the philosophical thought of traditional African sages to demonstrate rigorous indigenous philosophy.",
            },

            # === INDIGENOUS AMERICAS ===

            # Nahua/Aztec
            {
                "figure_id": "nezahualcoyotl",
                "name": "Nezahualcoyotl",
                "alternate_names": ["Fasting Coyote", "Poet-King"],
                "birth_year": 1402,
                "death_year": 1472,
                "era": "Pre-Columbian",
                "traditions": [PhilosophicalTradition.NAHUA_AZTEC],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.AESTHETICS, PhilosophicalDomain.EXISTENTIAL],
                "core_ideas": ["In Xochitl In Cuicatl", "Teotl", "Impermanence", "Philosophical Poetry"],
                "key_works": ["Romances de los Señores de la Nueva España (attributed poems)"],
                "biography_summary": "Philosopher-king of Texcoco, considered the greatest Nahua poet-philosopher. His poetry explores existence, impermanence, and truth through the aesthetic mode of 'flower and song.'",
            },

            # Native American
            {
                "figure_id": "vine_deloria",
                "name": "Vine Deloria Jr.",
                "alternate_names": [],
                "birth_year": 1933,
                "death_year": 2005,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.NATIVE_AMERICAN],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "core_ideas": ["Place-based Metaphysics", "Critique of Western Categories", "Indigenous Sovereignty", "Spiritual Ecology"],
                "key_works": ["God Is Red", "Spirit and Reason", "Red Earth, White Lies"],
                "biography_summary": "Standing Rock Sioux scholar, most influential Native American philosopher. Articulated indigenous metaphysics centered on place rather than time, challenging Western philosophical assumptions.",
            },

            # === LATIN AMERICAN ===

            # Liberation Philosophy
            {
                "figure_id": "enrique_dussel",
                "name": "Enrique Dussel",
                "alternate_names": [],
                "birth_year": 1934,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.LIBERATION_PHILOSOPHY],
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["Exteriority", "Analeptic Method", "Transmodernity", "Philosophy of Liberation"],
                "key_works": ["Philosophy of Liberation", "Ethics of Liberation", "Twenty Theses on Politics"],
                "influences": ["Levinas", "Marx", "Latin American history"],
                "influenced": ["Decolonial thought", "World philosophy"],
                "biography_summary": "Argentine-Mexican philosopher, founder of Liberation Philosophy. Develops philosophy from the perspective of the excluded Other, challenging Eurocentric universalism.",
            },
            {
                "figure_id": "leopoldo_zea",
                "name": "Leopoldo Zea",
                "alternate_names": [],
                "birth_year": 1912,
                "death_year": 2004,
                "era": "Contemporary",
                "traditions": [PhilosophicalTradition.LIBERATION_PHILOSOPHY],
                "domains": [PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.EPISTEMOLOGY],
                "core_ideas": ["Latin American Identity", "Authentic Philosophy", "Marginalization", "Universal from the Particular"],
                "key_works": ["The Latin American Mind", "The Role of the Americas in History"],
                "influences": ["José Ortega y Gasset", "Mexican history"],
                "influenced": ["Latin American philosophy", "Dussel"],
                "biography_summary": "Mexican philosopher who pioneered the question of Latin American philosophical identity. Argued that authentic philosophy emerges from concrete historical circumstances.",
            },

            # === ADDITIONAL BUDDHIST SCHOOLS ===

            # Huayan
            {
                "figure_id": "fazang",
                "name": "Fazang",
                "alternate_names": ["Fa-tsang", "Xianshou", "法藏"],
                "birth_year": 643,
                "death_year": 712,
                "era": "Tang Dynasty",
                "traditions": [PhilosophicalTradition.BUDDHIST_HUAYAN],
                "domains": [PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["Interpenetration", "Six Characteristics", "Ten Mysteries", "Golden Lion Essay"],
                "key_works": ["Essay on the Golden Lion", "Huayan Wujiao Zhang"],
                "teachers": ["Zhiyan"],
                "influenced": ["East Asian Buddhism", "Korean Buddhism"],
                "biography_summary": "Third patriarch of Huayan Buddhism, its greatest systematizer. Developed the philosophy of interpenetration using the famous example of the golden lion to explain Huayan metaphysics.",
            },
            {
                "figure_id": "dushun",
                "name": "Dushun",
                "alternate_names": ["Du-shun", "Fashun", "杜順"],
                "birth_year": 557,
                "death_year": 640,
                "era": "Sui-Tang Dynasties",
                "traditions": [PhilosophicalTradition.BUDDHIST_HUAYAN],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.MEDITATION_THEORY],
                "core_ideas": ["Dharmadhatu", "Contemplation of the Dharma Realm", "Founding Huayan Vision"],
                "key_works": ["Contemplation of the Dharmadhatu"],
                "influenced": ["Zhiyan", "Fazang", "Huayan school"],
                "biography_summary": "First patriarch of Huayan Buddhism. Established the foundational contemplative vision of the interpenetration of all phenomena that later patriarchs would systematize.",
            },

            # Tiantai
            {
                "figure_id": "zhiyi",
                "name": "Zhiyi",
                "alternate_names": ["Chih-i", "智顗"],
                "birth_year": 538,
                "death_year": 597,
                "era": "Sui Dynasty",
                "traditions": [PhilosophicalTradition.BUDDHIST_TIANTAI],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.MEDITATION_THEORY, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "core_ideas": ["Threefold Truth", "Ichinen Sanzen", "Classification of Teachings", "Zhiguan Meditation"],
                "key_works": ["Mohe Zhiguan", "Fahua Xuanyi", "Fahua Wenju"],
                "teachers": ["Huisi"],
                "influenced": ["All East Asian Buddhism", "Nichiren Buddhism"],
                "biography_summary": "Founder of Tiantai Buddhism, greatest Chinese Buddhist philosopher. Developed the comprehensive system of threefold truth and three thousand realms in one thought-moment.",
            },

            # Jain
            {
                "figure_id": "mahavira",
                "name": "Mahavira",
                "alternate_names": ["Vardhamana", "The Great Hero", "महावीर"],
                "birth_year": -599,
                "death_year": -527,
                "era": "Ancient India",
                "traditions": [PhilosophicalTradition.JAIN],
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.METAPHYSICS],
                "core_ideas": ["Ahimsa", "Anekantavada", "Five Vows", "Liberation through Asceticism"],
                "influenced": ["All of Jainism", "Gandhi"],
                "biography_summary": "Twenty-fourth Tirthankara and founder of Jainism as historical religion. Established the ethical philosophy of non-violence and epistemology of many-sidedness.",
            },
            {
                "figure_id": "kundakunda",
                "name": "Kundakunda",
                "alternate_names": ["Kondakunda", "कुन्दकुन्द"],
                "birth_year": 100,
                "death_year": 200,
                "era": "Ancient India",
                "traditions": [PhilosophicalTradition.JAIN],
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.SOTERIOLOGY],
                "core_ideas": ["Two Standpoints (Nischaya/Vyavahara)", "Soul's Pure Nature", "Self-Realization"],
                "key_works": ["Samayasara", "Niyamasara", "Panchastikayasara"],
                "influenced": ["Digambara Jain philosophy"],
                "biography_summary": "Most influential Jain philosopher after Mahavira. Developed the two-standpoints theory distinguishing absolute and conventional truth, and articulated the soul's inherently pure nature.",
            },

            # === FOLK-PHILOSOPHICAL BRIDGES ===

            # Celtic Druidic
            {
                "figure_id": "taliesin",
                "name": "Taliesin",
                "alternate_names": ["Chief of Bards", "Taliesin Ben Beirdd"],
                "era": "6th Century Wales",
                "traditions": [PhilosophicalTradition.CELTIC_DRUIDIC],
                "domains": [PhilosophicalDomain.ORAL_EPISTEMOLOGY, PhilosophicalDomain.ANIMISTIC_METAPHYSICS],
                "core_ideas": ["Awen (Poetic Inspiration)", "Shapeshifting Consciousness", "Bardic Memory"],
                "key_works": ["Book of Taliesin", "Kat Godeu (Battle of Trees)"],
                "biography_summary": "Semi-legendary Welsh bard whose poems preserve Druidic cosmology and philosophy. His work encodes transformational wisdom through shape-shifting narratives.",
            },
            {
                "figure_id": "amergin",
                "name": "Amergin Glúingel",
                "alternate_names": ["Amergin White-Knee", "Chief Bard of the Milesians"],
                "era": "Mythological/Pre-Christian Ireland",
                "traditions": [PhilosophicalTradition.CELTIC_DRUIDIC],
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.ORAL_EPISTEMOLOGY],
                "core_ideas": ["Unity with Nature", "Cosmic Identity", "Poetic Magic"],
                "key_works": ["Song of Amergin"],
                "biography_summary": "Mythological Milesian bard whose Song expresses Celtic animistic philosophy - the speaker claims identity with elements of nature in a profound statement of cosmic unity.",
            },

            # Norse Philosophical
            {
                "figure_id": "snorri_sturluson",
                "name": "Snorri Sturluson",
                "alternate_names": ["Snorri"],
                "birth_year": 1179,
                "death_year": 1241,
                "era": "Medieval Iceland",
                "traditions": [PhilosophicalTradition.NORSE_PHILOSOPHICAL],
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.FOLK_ETHICS],
                "core_ideas": ["Preservation of Norse Mythology", "Poetic Tradition", "Historical Memory"],
                "key_works": ["Prose Edda", "Heimskringla"],
                "biography_summary": "Icelandic historian and poet who preserved Norse mythology and philosophy in Christian times. His Prose Edda is the primary source for Norse cosmology and wisdom.",
            },

            # Aboriginal Australian
            {
                "figure_id": "david_mowaljarlai",
                "name": "David Mowaljarlai",
                "alternate_names": ["Banggal"],
                "birth_year": 1925,
                "death_year": 1997,
                "era": "Modern Australia",
                "traditions": [PhilosophicalTradition.ABORIGINAL_AUSTRALIAN],
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.TRADITIONAL_ECOLOGY],
                "core_ideas": ["Everything is Connected", "Two-Way Learning", "Pattern Thinking"],
                "key_works": ["Yorro Yorro: Everything Standing Up Alive"],
                "influenced": ["Cross-cultural philosophy", "Environmental ethics"],
                "biography_summary": "Ngarinyin elder and philosopher who articulated Aboriginal worldview for Western audiences. Taught that pattern connects all things in an alive, responsive universe.",
            },

            # Polynesian Philosophical
            {
                "figure_id": "maui_trickster",
                "name": "Māui",
                "alternate_names": ["Maui-tikitiki-a-Taranga", "The Trickster"],
                "era": "Mythological Polynesia",
                "traditions": [PhilosophicalTradition.POLYNESIAN_PHILOSOPHICAL],
                "domains": [PhilosophicalDomain.FOLK_ETHICS, PhilosophicalDomain.ANIMISTIC_METAPHYSICS],
                "core_ideas": ["Transformation through Cleverness", "Balance of Forces", "Human Potential"],
                "biography_summary": "Pan-Polynesian culture hero whose stories encode philosophical wisdom about human agency, the relationship between mortals and gods, and the ethics of transformation.",
            },

            # Siberian Shamanic
            {
                "figure_id": "mircea_eliade_shamanism",
                "name": "Mircea Eliade",
                "alternate_names": ["Historian of Religions"],
                "birth_year": 1907,
                "death_year": 1986,
                "era": "20th Century",
                "traditions": [PhilosophicalTradition.SIBERIAN_SHAMANIC],
                "domains": [PhilosophicalDomain.CEREMONIAL_KNOWLEDGE, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "core_ideas": ["Archaic Techniques of Ecstasy", "Axis Mundi", "Sacred and Profane"],
                "key_works": ["Shamanism: Archaic Techniques of Ecstasy"],
                "influenced": ["Comparative Religion", "Religious Studies"],
                "biography_summary": "Romanian scholar who synthesized and philosophically interpreted Siberian shamanic traditions. His work made shamanic cosmology accessible to academic philosophy.",
            },

            # Inuit Wisdom
            {
                "figure_id": "knud_rasmussen",
                "name": "Knud Rasmussen",
                "alternate_names": ["Father of Eskimology"],
                "birth_year": 1879,
                "death_year": 1933,
                "era": "Early 20th Century",
                "traditions": [PhilosophicalTradition.INUIT_WISDOM],
                "domains": [PhilosophicalDomain.TRADITIONAL_ECOLOGY, PhilosophicalDomain.ORAL_EPISTEMOLOGY],
                "core_ideas": ["Documentation of Inuit Thought", "Cross-Cultural Understanding"],
                "key_works": ["Intellectual Culture of the Iglulik Eskimos", "The Fifth Thule Expedition"],
                "biography_summary": "Greenlandic-Danish explorer who recorded Inuit philosophy and oral traditions. His interviews with angakuit (shamans) preserved cosmological and ethical thought.",
            },

            # African Cosmological
            {
                "figure_id": "placide_tempels",
                "name": "Placide Tempels",
                "alternate_names": ["Father Tempels"],
                "birth_year": 1906,
                "death_year": 1977,
                "era": "20th Century",
                "traditions": [PhilosophicalTradition.AFRICAN_COSMOLOGICAL],
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.FOLK_ETHICS],
                "core_ideas": ["Vital Force", "Being as Force", "African Metaphysics"],
                "key_works": ["Bantu Philosophy"],
                "influenced": ["African Philosophy movement", "Négritude"],
                "biography_summary": "Belgian missionary-philosopher who articulated Bantu metaphysics. His concept of vital force became foundational for African philosophy, though later critiqued for colonial framing.",
            },
        ]

    async def initialize_seed_figures(self) -> int:
        """
        Initialize the figure index with seed figures for all traditions.

        Returns:
            Number of figures added
        """
        seed_figures = self._get_seed_figures()
        count = 0

        for figure_data in seed_figures:
            figure = PhilosophicalFigure(
                figure_id=figure_data["figure_id"],
                name=figure_data["name"],
                alternate_names=figure_data.get("alternate_names", []),
                birth_year=figure_data.get("birth_year"),
                death_year=figure_data.get("death_year"),
                era=figure_data.get("era"),
                traditions=figure_data.get("traditions", []),
                domains=figure_data.get("domains", []),
                core_ideas=figure_data.get("core_ideas", []),
                key_works=figure_data.get("key_works", []),
                teachers=figure_data.get("teachers", []),
                students=figure_data.get("students", []),
                influences=figure_data.get("influences", []),
                influenced=figure_data.get("influenced", []),
                biography_summary=figure_data.get("biography_summary"),
            )
            await self.add_figure(figure)
            count += 1

        logger.info(f"Initialized {count} seed figures across all traditions")
        return count

    def _get_seed_texts(self) -> List[Dict[str, Any]]:
        """
        Return seed texts (primary philosophical works) for all philosophical traditions.

        These provide initial knowledge base entries that can be expanded
        through the research agent system.
        """
        return [
            # === ISLAMIC TRADITIONS ===

            # Sufi Philosophy
            {
                "text_id": "fusus_al_hikam",
                "title": "Fusus al-Hikam",
                "author": "ibn_arabi",
                "author_name": "Ibn Arabi",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1229,
                "summary": "The Bezels of Wisdom - Ibn Arabi's masterwork presenting the spiritual wisdom of 27 prophets. Each chapter explores a divine 'bezel' (setting) through which the light of wisdom shines, articulating the Unity of Being.",
                "key_concepts": ["wahdat_al_wujud", "divine_names", "perfect_human", "prophetic_wisdom"],
            },
            {
                "text_id": "masnavi",
                "title": "Masnavi-ye-Ma'navi",
                "author": "rumi",
                "author_name": "Jalal al-Din Rumi",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS, PhilosophicalDomain.AESTHETICS],
                "year_written": 1258,
                "summary": "The Spiritual Couplets - Six books of mystical poetry exploring divine love, spiritual transformation, and the soul's journey. Called 'the Quran in Persian' for its spiritual depth.",
                "key_concepts": ["divine_love", "fana", "spiritual_journey", "reed_flute_metaphor"],
            },
            {
                "text_id": "ihya_ulum_al_din",
                "title": "Ihya Ulum al-Din",
                "author": "al_ghazali",
                "author_name": "Abu Hamid al-Ghazali",
                "tradition": PhilosophicalTradition.SUFI_PHILOSOPHY,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1097,
                "summary": "The Revival of the Religious Sciences - Comprehensive guide to Islamic spirituality integrating law, theology, and Sufism. Covers worship, social ethics, vices, virtues, and the path to God.",
                "key_concepts": ["spiritual_revival", "heart_purification", "stations_of_soul", "islamic_ethics"],
            },

            # Falsafa
            {
                "text_id": "al_shifa",
                "title": "Al-Shifa (The Healing)",
                "author": "ibn_sina",
                "author_name": "Ibn Sina (Avicenna)",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.LOGIC, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "year_written": 1027,
                "summary": "Encyclopedic philosophical work covering logic, natural philosophy, mathematics, and metaphysics. The most influential philosophical text in the Islamic world, synthesizing Aristotelian and Neoplatonic thought.",
                "key_concepts": ["necessary_being", "essence_existence", "active_intellect", "emanation"],
            },
            {
                "text_id": "ara_ahl_al_madina",
                "title": "Ara Ahl al-Madina al-Fadila",
                "author": "al_farabi",
                "author_name": "Al-Farabi",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domains": [PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.METAPHYSICS],
                "year_written": 942,
                "summary": "The Views of the Inhabitants of the Virtuous City - Political philosophy describing the ideal state ruled by a philosopher-prophet. Integrates Platonic political theory with Islamic prophetology.",
                "key_concepts": ["virtuous_city", "philosopher_king", "emanation", "human_perfection"],
            },
            {
                "text_id": "tahafut_al_tahafut",
                "title": "Tahafut al-Tahafut",
                "author": "ibn_rushd",
                "author_name": "Ibn Rushd (Averroes)",
                "tradition": PhilosophicalTradition.FALSAFA,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1180,
                "summary": "The Incoherence of the Incoherence - Defense of philosophy against al-Ghazali's critique. Argues that philosophy and religion are compatible paths to truth, defending Aristotelian demonstrative reasoning.",
                "key_concepts": ["double_truth", "eternity_of_world", "defense_of_philosophy", "reason_revelation"],
            },

            # Illuminationist
            {
                "text_id": "hikmat_al_ishraq",
                "title": "Hikmat al-Ishraq",
                "author": "suhrawardi",
                "author_name": "Suhrawardi",
                "tradition": PhilosophicalTradition.ILLUMINATIONIST,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY],
                "year_written": 1186,
                "summary": "The Philosophy of Illumination - Foundational text of Illuminationist philosophy. Presents reality as grades of light emanating from the Light of Lights, with knowledge as direct presence rather than representation.",
                "key_concepts": ["light_of_lights", "knowledge_by_presence", "hierarchy_of_lights", "ishraqi_wisdom"],
            },

            # Transcendent Theosophy
            {
                "text_id": "asfar_arbaa",
                "title": "Al-Asfar al-Arba'a",
                "author": "mulla_sadra",
                "author_name": "Mulla Sadra",
                "tradition": PhilosophicalTradition.TRANSCENDENT_THEOSOPHY,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "year_written": 1628,
                "summary": "The Four Journeys - Mulla Sadra's magnum opus describing the soul's four journeys: from creation to God, in God, from God to creation, and in creation with God. Synthesizes peripatetic, illuminationist, and gnostic thought.",
                "key_concepts": ["substantial_motion", "primacy_of_existence", "four_journeys", "soul_becoming"],
            },

            # Kalam
            {
                "text_id": "tahafut_al_falasifa",
                "title": "Tahafut al-Falasifa",
                "author": "al_ghazali",
                "author_name": "Abu Hamid al-Ghazali",
                "tradition": PhilosophicalTradition.KALAM,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1095,
                "summary": "The Incoherence of the Philosophers - Critique of Aristotelian philosophy on 20 points, arguing philosophers contradict themselves and cannot prove key claims. Defends divine freedom and occasionalism.",
                "key_concepts": ["occasionalism", "divine_will", "critique_of_causation", "creation_ex_nihilo"],
            },

            # === JEWISH TRADITIONS ===

            # Maimonidean
            {
                "text_id": "guide_perplexed",
                "title": "Guide for the Perplexed",
                "author": "maimonides",
                "author_name": "Moses Maimonides",
                "tradition": PhilosophicalTradition.MAIMONIDEAN,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY],
                "year_written": 1190,
                "summary": "Moreh Nevukhim - The foundational work of Jewish rationalist philosophy. Harmonizes Torah with Aristotelian philosophy, develops negative theology, and interprets biblical anthropomorphisms philosophically.",
                "key_concepts": ["negative_theology", "prophecy", "divine_attributes", "equivocal_language"],
            },
            {
                "text_id": "mishneh_torah",
                "title": "Mishneh Torah",
                "author": "maimonides",
                "author_name": "Moses Maimonides",
                "tradition": PhilosophicalTradition.MAIMONIDEAN,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1180,
                "summary": "Repetition of the Torah - Comprehensive code of Jewish law, opening with philosophical foundations on God's existence and nature. First systematic codification integrating law with philosophical theology.",
                "key_concepts": ["divine_unity", "intellectual_love", "ethical_law", "human_perfection"],
            },

            # Kabbalistic
            {
                "text_id": "zohar",
                "title": "Sefer ha-Zohar",
                "author": "moses_de_leon",
                "author_name": "Moses de León (attributed)",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1290,
                "summary": "The Book of Splendor - Central text of Kabbalah, presented as 2nd-century teachings but composed in 13th-century Spain. Mystical commentary on Torah exploring the sefirot, divine nature, and cosmic symbolism.",
                "key_concepts": ["sefirot", "ein_sof", "divine_symbolism", "mystical_torah"],
            },
            {
                "text_id": "etz_chaim",
                "title": "Etz Chaim",
                "author": "isaac_luria",
                "author_name": "Isaac Luria (transmitted by Chaim Vital)",
                "tradition": PhilosophicalTradition.KABBALISTIC,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1573,
                "summary": "The Tree of Life - Systematic presentation of Lurianic Kabbalah. Describes tzimtzum (divine contraction), shevirat ha-kelim (shattering of vessels), and tikkun (cosmic repair).",
                "key_concepts": ["tzimtzum", "shevirat_hakelim", "tikkun", "divine_sparks"],
            },

            # Hasidic
            {
                "text_id": "tanya",
                "title": "Tanya",
                "author": "schneur_zalman",
                "author_name": "Schneur Zalman of Liadi",
                "tradition": PhilosophicalTradition.HASIDIC_PHILOSOPHY,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_MIND, PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1796,
                "summary": "The foundational text of Chabad Hasidism. Analyzes the two souls (divine and animal), the nature of the beinoni (intermediate person), and the contemplative path to devekut.",
                "key_concepts": ["two_souls", "beinoni", "bittul", "intellectual_contemplation"],
            },

            # Jewish Existentialism
            {
                "text_id": "i_and_thou",
                "title": "I and Thou",
                "author": "martin_buber",
                "author_name": "Martin Buber",
                "tradition": PhilosophicalTradition.JEWISH_EXISTENTIALISM,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1923,
                "summary": "Ich und Du - Revolutionary philosophy of dialogue distinguishing I-Thou (genuine encounter) from I-It (objectification). The eternal Thou is glimpsed in every genuine encounter with another.",
                "key_concepts": ["i_thou", "i_it", "dialogue", "eternal_thou"],
            },
            {
                "text_id": "totality_infinity",
                "title": "Totality and Infinity",
                "author": "emmanuel_levinas",
                "author_name": "Emmanuel Levinas",
                "tradition": PhilosophicalTradition.JEWISH_EXISTENTIALISM,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHENOMENOLOGICAL],
                "year_written": 1961,
                "summary": "Totalité et Infini - Establishes ethics as first philosophy through the encounter with the face of the Other. The face reveals infinity and issues the ethical command against murder.",
                "key_concepts": ["face", "infinity", "alterity", "ethics_first_philosophy"],
            },
            {
                "text_id": "star_redemption",
                "title": "The Star of Redemption",
                "author": "franz_rosenzweig",
                "author_name": "Franz Rosenzweig",
                "tradition": PhilosophicalTradition.JEWISH_EXISTENTIALISM,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.METAPHYSICS],
                "year_written": 1921,
                "summary": "Der Stern der Erlösung - Masterwork of religious existentialism structuring reality through God, World, and Human, united by Creation, Revelation, and Redemption forming a six-pointed star.",
                "key_concepts": ["new_thinking", "creation_revelation_redemption", "speech_thinking", "star_structure"],
            },

            # === AFRICAN TRADITIONS ===

            # Egyptian Ancient
            {
                "text_id": "instructions_ptahhotep",
                "title": "Instructions of Ptahhotep",
                "author": "ptahhotep",
                "author_name": "Ptahhotep",
                "tradition": PhilosophicalTradition.EGYPTIAN_ANCIENT,
                "domains": [PhilosophicalDomain.ETHICS],
                "year_written": -2350,
                "summary": "The oldest complete philosophical text, written by an Egyptian vizier. Wisdom teachings on living according to Ma'at: proper speech, humility, justice, and social harmony.",
                "key_concepts": ["maat", "practical_wisdom", "speech_ethics", "social_harmony"],
            },
            {
                "text_id": "book_of_dead",
                "title": "Book of the Dead",
                "author": None,
                "author_name": "Various Egyptian priests",
                "tradition": PhilosophicalTradition.EGYPTIAN_ANCIENT,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.METAPHYSICS],
                "year_written": -1550,
                "summary": "Collection of funerary texts including the 'Negative Confession' (42 declarations of innocence) and the weighing of the heart against the feather of Ma'at. Articulates Egyptian ethical philosophy.",
                "key_concepts": ["maat", "weighing_of_heart", "negative_confession", "afterlife"],
            },

            # Ethiopian
            {
                "text_id": "hatata_zera_yacob",
                "title": "Hatata",
                "author": "zera_yacob",
                "author_name": "Zera Yacob",
                "tradition": PhilosophicalTradition.ETHIOPIAN_PHILOSOPHY,
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.ETHICS],
                "year_written": 1667,
                "summary": "Critical Inquiry - Ethiopian rationalist philosophy. Questions all religious traditions through reason, argues for natural religion accessible to all through the God-given light of reason.",
                "key_concepts": ["rational_inquiry", "natural_theology", "universal_reason", "religious_criticism"],
            },

            # Ubuntu / African Contemporary
            {
                "text_id": "african_philosophy_ubuntu",
                "title": "African Philosophy Through Ubuntu",
                "author": "mogobe_ramose",
                "author_name": "Mogobe Ramose",
                "tradition": PhilosophicalTradition.UBUNTU,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "year_written": 1999,
                "summary": "Systematic presentation of Ubuntu as a complete philosophical worldview, not merely an ethical maxim. Develops Ubuntu's metaphysical, epistemological, and political dimensions.",
                "key_concepts": ["ubuntu", "personhood", "communalism", "african_jurisprudence"],
            },
            {
                "text_id": "essay_african_thought",
                "title": "An Essay on African Philosophical Thought",
                "author": "kwame_gyekye",
                "author_name": "Kwame Gyekye",
                "tradition": PhilosophicalTradition.AKAN_PHILOSOPHY,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "year_written": 1987,
                "summary": "Rigorous analysis of Akan philosophical concepts including personhood, the soul, communalism, and ethics. Argues for moderate communitarianism balancing individual and community.",
                "key_concepts": ["akan_personhood", "okra", "moderate_communitarianism", "african_ethics"],
            },
            {
                "text_id": "philosophy_african_culture",
                "title": "Philosophy and an African Culture",
                "author": "kwasi_wiredu",
                "author_name": "Kwasi Wiredu",
                "tradition": PhilosophicalTradition.AKAN_PHILOSOPHY,
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE],
                "year_written": 1980,
                "summary": "Pioneering work on conceptual decolonization in African philosophy. Analyzes how colonial languages distort African thought and advocates for philosophy in African languages.",
                "key_concepts": ["conceptual_decolonization", "african_logic", "consensual_politics", "truth_in_akan"],
            },

            # Sage Philosophy
            {
                "text_id": "sage_philosophy",
                "title": "Sage Philosophy: Indigenous Thinkers and Modern Debate on African Philosophy",
                "author": "odera_oruka",
                "author_name": "Henry Odera Oruka",
                "tradition": PhilosophicalTradition.SAGE_PHILOSOPHY,
                "domains": [PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.ETHICS],
                "year_written": 1990,
                "summary": "Documentation and analysis of philosophical thought among Kenyan sages. Distinguishes folk wisdom from philosophic sagacity - critical, reflective thinking within traditional contexts.",
                "key_concepts": ["philosophic_sagacity", "folk_wisdom", "indigenous_philosophy", "sage_tradition"],
            },

            # === INDIGENOUS AMERICAS ===

            # Nahua/Aztec
            {
                "text_id": "cantares_mexicanos",
                "title": "Cantares Mexicanos",
                "author": None,
                "author_name": "Various Nahua poets (including Nezahualcoyotl)",
                "tradition": PhilosophicalTradition.NAHUA_AZTEC,
                "domains": [PhilosophicalDomain.AESTHETICS, PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EXISTENTIAL],
                "year_written": 1550,
                "summary": "Collection of Nahuatl philosophical poetry exploring 'flower and song' (in xochitl in cuicatl) as the path to truth. Reflects on impermanence, beauty, truth, and the nature of existence.",
                "key_concepts": ["in_xochitl_in_cuicatl", "teotl", "impermanence", "neltiliztli"],
            },

            # Maya
            {
                "text_id": "popol_vuh",
                "title": "Popol Vuh",
                "author": None,
                "author_name": "K'iche' Maya tradition",
                "tradition": PhilosophicalTradition.MAYA_PHILOSOPHY,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1550,
                "summary": "The K'iche' Maya creation narrative and mythological history. Describes the creation of humans from maize, the Hero Twins' journey, and the cyclical nature of time and existence.",
                "key_concepts": ["creation_from_maize", "hero_twins", "cyclical_time", "underworld_journey"],
            },

            # Native American
            {
                "text_id": "god_is_red",
                "title": "God Is Red: A Native View of Religion",
                "author": "vine_deloria",
                "author_name": "Vine Deloria Jr.",
                "tradition": PhilosophicalTradition.NATIVE_AMERICAN,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.METAPHYSICS],
                "year_written": 1973,
                "summary": "Foundational text of Native American philosophy contrasting indigenous and Western worldviews. Argues for place-based metaphysics versus time-based Western thought, and relational versus substantive ontology.",
                "key_concepts": ["place_based_metaphysics", "relational_ontology", "indigenous_religion", "critique_of_christianity"],
            },
            {
                "text_id": "spirit_and_reason",
                "title": "Spirit and Reason: The Vine Deloria Jr. Reader",
                "author": "vine_deloria",
                "author_name": "Vine Deloria Jr.",
                "tradition": PhilosophicalTradition.NATIVE_AMERICAN,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE],
                "year_written": 1999,
                "summary": "Collection of Deloria's philosophical writings on Native American thought, sovereignty, science, and religion. Develops indigenous epistemology and critiques Western categorical thinking.",
                "key_concepts": ["indigenous_epistemology", "sovereignty", "sacred_places", "relational_thinking"],
            },

            # === LATIN AMERICAN ===

            # Liberation Philosophy
            {
                "text_id": "philosophy_liberation",
                "title": "Philosophy of Liberation",
                "author": "enrique_dussel",
                "author_name": "Enrique Dussel",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.METAPHYSICS],
                "year_written": 1977,
                "summary": "Foundational text of liberation philosophy. Develops philosophy from the perspective of the excluded Other, critiques Eurocentric totality, and proposes the analeptic method beginning from exteriority.",
                "key_concepts": ["exteriority", "analeptic_method", "totality", "liberation"],
            },
            {
                "text_id": "ethics_liberation",
                "title": "Ethics of Liberation in the Age of Globalization and Exclusion",
                "author": "enrique_dussel",
                "author_name": "Enrique Dussel",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domains": [PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY],
                "year_written": 1998,
                "summary": "Comprehensive ethical system from the perspective of the excluded. Develops material, formal, and feasibility principles of ethics, with liberation as the fundamental ethical demand.",
                "key_concepts": ["liberation_ethics", "material_principle", "victims", "critical_reason"],
            },
            {
                "text_id": "latin_american_mind",
                "title": "The Latin American Mind",
                "author": "leopoldo_zea",
                "author_name": "Leopoldo Zea",
                "tradition": PhilosophicalTradition.LIBERATION_PHILOSOPHY,
                "domains": [PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.EPISTEMOLOGY],
                "year_written": 1963,
                "summary": "Analysis of Latin American intellectual history and the possibility of authentic Latin American philosophy. Argues philosophy must emerge from concrete historical circumstances.",
                "key_concepts": ["latin_american_identity", "authentic_philosophy", "marginalization", "historical_circumstance"],
            },

            # === ADDITIONAL BUDDHIST SCHOOLS ===

            # Huayan
            {
                "text_id": "golden_lion",
                "title": "Essay on the Golden Lion",
                "author": "fazang",
                "author_name": "Fazang",
                "tradition": PhilosophicalTradition.BUDDHIST_HUAYAN,
                "domains": [PhilosophicalDomain.METAPHYSICS],
                "year_written": 700,
                "summary": "Uses a golden lion statue to explain Huayan philosophy: the gold (li/principle) and lion (shi/phenomena) interpenetrate. Each part of the lion contains the whole, illustrating mutual containment.",
                "key_concepts": ["li_shi", "interpenetration", "mutual_containment", "six_characteristics"],
            },
            {
                "text_id": "huayan_sutra",
                "title": "Avatamsaka Sutra (Huayan Jing)",
                "author": None,
                "author_name": "Anonymous (Indian origin)",
                "tradition": PhilosophicalTradition.BUDDHIST_HUAYAN,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.SOTERIOLOGY],
                "year_written": 300,
                "summary": "The Flower Garland Sutra - vast scripture describing the cosmos as interpenetrating realms, Indra's net, and the bodhisattva path. Foundational text for Huayan Buddhism.",
                "key_concepts": ["dharmadhatu", "indras_net", "bodhisattva_stages", "buddha_lands"],
            },

            # Tiantai
            {
                "text_id": "mohe_zhiguan",
                "title": "Mohe Zhiguan",
                "author": "zhiyi",
                "author_name": "Zhiyi",
                "tradition": PhilosophicalTradition.BUDDHIST_TIANTAI,
                "domains": [PhilosophicalDomain.MEDITATION_THEORY, PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND],
                "year_written": 594,
                "summary": "The Great Calming and Contemplation - Comprehensive meditation manual and philosophical treatise. Develops the threefold truth, ichinen sanzen (3000 realms in one thought), and systematic meditation practice.",
                "key_concepts": ["threefold_truth", "ichinen_sanzen", "zhiguan", "perfect_sudden"],
            },
            {
                "text_id": "lotus_sutra",
                "title": "Lotus Sutra (Saddharmapundarika)",
                "author": None,
                "author_name": "Anonymous (Indian origin)",
                "tradition": PhilosophicalTradition.BUDDHIST_TIANTAI,
                "domains": [PhilosophicalDomain.PHILOSOPHY_OF_RELIGION, PhilosophicalDomain.SOTERIOLOGY],
                "year_written": 100,
                "summary": "The Sutra of the Lotus of the Wonderful Law - Central text for Tiantai Buddhism. Teaches the one vehicle (ekayana), universal Buddha-nature, and skillful means. Most influential Mahayana sutra in East Asia.",
                "key_concepts": ["one_vehicle", "buddha_nature", "skillful_means", "eternal_buddha"],
            },

            # Jain
            {
                "text_id": "samayasara",
                "title": "Samayasara",
                "author": "kundakunda",
                "author_name": "Kundakunda",
                "tradition": PhilosophicalTradition.JAIN,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.SOTERIOLOGY],
                "year_written": 150,
                "summary": "The Essence of the Self - Foundational text of Digambara Jain philosophy. Develops the two-standpoints theory (nischaya/vyavahara) and articulates the soul's inherently pure nature obscured by karma.",
                "key_concepts": ["two_standpoints", "pure_soul", "self_realization", "karma_bondage"],
            },
            {
                "text_id": "tattvartha_sutra",
                "title": "Tattvartha Sutra",
                "author": "umasvati",
                "author_name": "Umasvati",
                "tradition": PhilosophicalTradition.JAIN,
                "domains": [PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.ETHICS],
                "year_written": 200,
                "summary": "That Which Is - Systematic presentation of Jain philosophy accepted by all sects. Covers reality (tattva), epistemology, ethics, cosmology, and the path to liberation in concise sutras.",
                "key_concepts": ["seven_tattvas", "anekantavada", "ahimsa", "moksha"],
            },

            # === FOLK-PHILOSOPHICAL BRIDGES ===

            # Celtic Druidic
            {
                "text_id": "book_of_taliesin",
                "title": "The Book of Taliesin",
                "author": "taliesin",
                "author_name": "Taliesin (attributed)",
                "tradition": PhilosophicalTradition.CELTIC_DRUIDIC,
                "domains": [PhilosophicalDomain.ORAL_EPISTEMOLOGY, PhilosophicalDomain.ANIMISTIC_METAPHYSICS],
                "year_written": 900,  # Compilation date; poems older
                "summary": "Collection of Welsh poems attributed to Taliesin containing Druidic cosmology, transformation narratives, and bardic wisdom. Encodes Celtic philosophy through poetic symbolism.",
                "key_concepts": ["awen", "transformation", "otherworld", "bardic_memory"],
            },
            {
                "text_id": "song_of_amergin",
                "title": "Song of Amergin",
                "author": "amergin",
                "author_name": "Amergin Glúingel (attributed)",
                "tradition": PhilosophicalTradition.CELTIC_DRUIDIC,
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.ORAL_EPISTEMOLOGY],
                "year_written": -1500,  # Traditional dating; recorded much later
                "summary": "Ancient Irish poem expressing animistic identity with natural phenomena: 'I am a wind on the sea, I am a wave of the ocean...' Foundational text of Celtic cosmological thought.",
                "key_concepts": ["cosmic_unity", "shapeshifting", "nature_identity", "poetic_magic"],
            },

            # Norse Philosophical
            {
                "text_id": "prose_edda",
                "title": "Prose Edda",
                "author": "snorri_sturluson",
                "author_name": "Snorri Sturluson",
                "tradition": PhilosophicalTradition.NORSE_PHILOSOPHICAL,
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.FOLK_ETHICS],
                "year_written": 1220,
                "summary": "Primary source for Norse mythology and cosmology. Describes the creation of the world, the world tree Yggdrasil, the gods, and Ragnarök. Contains ethical wisdom through mythological narrative.",
                "key_concepts": ["yggdrasil", "wyrd", "ragnarok", "nine_worlds"],
            },
            {
                "text_id": "havamal",
                "title": "Hávamál",
                "author": None,
                "author_name": "Anonymous (Odin's voice)",
                "tradition": PhilosophicalTradition.NORSE_PHILOSOPHICAL,
                "domains": [PhilosophicalDomain.FOLK_ETHICS, PhilosophicalDomain.ORAL_EPISTEMOLOGY],
                "year_written": 900,  # Approximate; oral origins older
                "summary": "The Sayings of the High One - Norse wisdom poetry attributed to Odin. Contains ethical maxims, social wisdom, and magical knowledge. Primary source for Norse folk ethics.",
                "key_concepts": ["hospitality", "wisdom", "fate", "rune_wisdom"],
            },

            # Slavic Wisdom
            {
                "text_id": "veles_book",
                "title": "The Book of Veles",
                "author": None,
                "author_name": "Anonymous (disputed)",
                "tradition": PhilosophicalTradition.SLAVIC_WISDOM,
                "domains": [PhilosophicalDomain.ANCESTRAL_PHILOSOPHY, PhilosophicalDomain.ANIMISTIC_METAPHYSICS],
                "year_written": 900,  # Claimed; authenticity debated
                "summary": "Purported ancient Slavic text describing the three-world cosmology (Prav, Yav, Nav), ancestor worship, and relationship with nature spirits. Controversial authenticity but reflects Slavic folk philosophy.",
                "key_concepts": ["prav", "yav", "nav", "rod", "ancestor_veneration"],
            },

            # Aboriginal Australian
            {
                "text_id": "yorro_yorro",
                "title": "Yorro Yorro: Everything Standing Up Alive",
                "author": "david_mowaljarlai",
                "author_name": "David Mowaljarlai",
                "tradition": PhilosophicalTradition.ABORIGINAL_AUSTRALIAN,
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.TRADITIONAL_ECOLOGY],
                "year_written": 1993,
                "summary": "Aboriginal elder's articulation of Ngarinyin philosophy. Explains pattern thinking, the interconnectedness of all things, two-way learning between cultures, and the living Country.",
                "key_concepts": ["dreaming", "country", "pattern_thinking", "living_landscape"],
            },

            # Polynesian Philosophical
            {
                "text_id": "kumulipo",
                "title": "The Kumulipo",
                "author": None,
                "author_name": "Hawaiian Royal Tradition",
                "tradition": PhilosophicalTradition.POLYNESIAN_PHILOSOPHICAL,
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.ANCESTRAL_PHILOSOPHY],
                "year_written": 1700,  # Recorded; oral origins ancient
                "summary": "Hawaiian creation chant of over 2,000 lines describing the origin of the universe from darkness through coral, plants, animals, to humans. Encodes Polynesian cosmological philosophy.",
                "key_concepts": ["cosmic_genealogy", "mana", "night_and_light", "evolution"],
            },

            # Siberian Shamanic
            {
                "text_id": "shamanism_eliade",
                "title": "Shamanism: Archaic Techniques of Ecstasy",
                "author": "mircea_eliade_shamanism",
                "author_name": "Mircea Eliade",
                "tradition": PhilosophicalTradition.SIBERIAN_SHAMANIC,
                "domains": [PhilosophicalDomain.CEREMONIAL_KNOWLEDGE, PhilosophicalDomain.PHILOSOPHY_OF_RELIGION],
                "year_written": 1951,
                "summary": "Comprehensive study of shamanic practices worldwide with focus on Siberian traditions. Analyzes the cosmic tree, ecstatic journeys, spirit helpers, and shamanic initiation as philosophical phenomena.",
                "key_concepts": ["axis_mundi", "ecstasy", "spirit_journey", "initiation"],
            },

            # Inuit Wisdom
            {
                "text_id": "intellectual_culture_iglulik",
                "title": "Intellectual Culture of the Iglulik Eskimos",
                "author": "knud_rasmussen",
                "author_name": "Knud Rasmussen",
                "tradition": PhilosophicalTradition.INUIT_WISDOM,
                "domains": [PhilosophicalDomain.TRADITIONAL_ECOLOGY, PhilosophicalDomain.ORAL_EPISTEMOLOGY, PhilosophicalDomain.ANIMISTIC_METAPHYSICS],
                "year_written": 1929,
                "summary": "Ethnographic record of Inuit cosmology, shamanistic practices, and oral philosophy from the Fifth Thule Expedition. Contains interviews with angakuit revealing sophisticated understanding of consciousness and nature.",
                "key_concepts": ["inua", "sila", "angakuq", "taboos"],
            },

            # African Cosmological
            {
                "text_id": "bantu_philosophy",
                "title": "Bantu Philosophy",
                "author": "placide_tempels",
                "author_name": "Placide Tempels",
                "tradition": PhilosophicalTradition.AFRICAN_COSMOLOGICAL,
                "domains": [PhilosophicalDomain.ANIMISTIC_METAPHYSICS, PhilosophicalDomain.FOLK_ETHICS],
                "year_written": 1945,
                "summary": "Foundational text arguing that Bantu peoples possess a coherent philosophical system based on vital force. Despite colonial framing, introduces African metaphysics to Western philosophy.",
                "key_concepts": ["vital_force", "being_as_force", "community", "ancestor_power"],
            },
        ]

    async def initialize_seed_texts(self) -> int:
        """
        Initialize the text index with seed texts for all traditions.

        Returns:
            Number of texts added
        """
        seed_texts = self._get_seed_texts()
        count = 0

        for text_data in seed_texts:
            text = PhilosophicalText(
                text_id=text_data["text_id"],
                title=text_data["title"],
                author=text_data.get("author"),
                author_name=text_data.get("author_name"),
                tradition=text_data["tradition"],
                domains=text_data.get("domains", []),
                year_written=text_data.get("year_written"),
                summary=text_data.get("summary"),
                key_concepts=text_data.get("key_concepts", []),
            )
            await self.add_text(text)
            count += 1

        logger.info(f"Initialized {count} seed texts across all traditions")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """
        Initialize all seed data (concepts, figures, and texts).

        Returns:
            Dictionary with counts of concepts, figures, and texts added
        """
        concepts_count = await self.initialize_seed_concepts()
        figures_count = await self.initialize_seed_figures()
        texts_count = await self.initialize_seed_texts()

        return {
            "concepts": concepts_count,
            "figures": figures_count,
            "texts": texts_count,
            "total": concepts_count + figures_count + texts_count
        }

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    async def query_concept(
        self,
        query: str,
        filters: Optional[QueryFilters] = None,
        max_results: int = 10,
        min_relevance: float = 0.5,
        trigger_research: bool = False
    ) -> Dict[str, Any]:
        """
        Query the philosophical concept index.

        Args:
            query: Natural language query or concept name
            filters: Optional filters for traditions, domains, etc.
            max_results: Maximum concepts to return
            min_relevance: Minimum relevance threshold
            trigger_research: If not found, trigger research agent

        Returns:
            Dictionary with matching concepts and metadata
        """
        start_time = time.time()
        filters = filters or QueryFilters()

        # Search by name first (exact/fuzzy)
        name_matches = self._search_by_name(query, self.concept_index)

        # Semantic search if embeddings available
        semantic_matches = await self._semantic_search(
            query,
            collection="concepts",
            top_k=max_results * 2
        )

        # Combine and deduplicate
        all_matches = self._merge_results(name_matches, semantic_matches)

        # Apply filters
        filtered = self._apply_filters(all_matches, filters)

        # Sort by relevance
        sorted_results = sorted(
            filtered,
            key=lambda x: x.get("relevance", 0),
            reverse=True
        )[:max_results]

        # Trigger research if no results and requested
        research_task_id = None
        if not sorted_results and trigger_research and self.research_coordinator:
            research_task_id = await self.research_coordinator.trigger_research(
                query=query,
                priority=5
            )

        processing_time = (time.time() - start_time) * 1000

        return {
            "query": query,
            "concepts": [r["concept"] for r in sorted_results],
            "relevance_scores": {r["concept"].concept_id: r["relevance"] for r in sorted_results},
            "research_triggered": research_task_id is not None,
            "research_task_id": research_task_id,
            "processing_time_ms": processing_time
        }

    async def query_tradition(
        self,
        tradition: PhilosophicalTradition,
        include_concepts: bool = True,
        include_figures: bool = True,
        include_texts: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a philosophical tradition.
        """
        result = {
            "tradition": tradition.value,
            "wisdom_teachings": self.wisdom_teachings.get(tradition.value, {}),
        }

        if include_concepts:
            result["concepts"] = [
                c for c in self.concept_index.values()
                if c.tradition == tradition or tradition in c.secondary_traditions
            ]

        if include_figures:
            result["figures"] = [
                f for f in self.figure_index.values()
                if tradition in f.traditions
            ]

        if include_texts:
            result["texts"] = [
                t for t in self.text_index.values()
                if t.tradition == tradition
            ]

        return result

    # ========================================================================
    # SYNTHESIS METHODS
    # ========================================================================

    async def synthesize_across_traditions(
        self,
        topic: str,
        traditions: List[PhilosophicalTradition],
        preserve_nuance: bool = True
    ) -> CrossTraditionSynthesis:
        """
        Synthesize insights on a topic across multiple traditions.
        """
        synthesis_id = self._generate_id(f"synthesis_{topic}")

        # Gather tradition-specific perspectives
        perspectives = {}
        for tradition in traditions:
            result = await self.query_concept(
                query=topic,
                filters=QueryFilters(traditions=[tradition]),
                max_results=5
            )
            perspectives[tradition] = result.get("concepts", [])

        # Identify convergent insights
        convergent = self._find_convergent_insights(perspectives)

        # Map divergent positions
        divergent = self._map_divergent_positions(perspectives)

        # Find complementary aspects
        complementary = self._find_complementary_aspects(perspectives)

        # Generate synthesis statement
        synthesis_statement = self._generate_synthesis_statement(
            topic=topic,
            convergent=convergent,
            divergent=divergent,
            complementary=complementary,
            preserve_nuance=preserve_nuance
        )

        # Calculate fidelity scores
        fidelity_scores = {
            t.value: self._calculate_fidelity(synthesis_statement, perspectives.get(t, []), t)
            for t in traditions
        }

        synthesis = CrossTraditionSynthesis(
            synthesis_id=synthesis_id,
            topic=topic,
            traditions=traditions,
            concepts_involved=[c.concept_id for p in perspectives.values() for c in p],
            convergent_insights=convergent,
            divergent_positions=divergent,
            complementary_aspects=complementary,
            synthesis_statement=synthesis_statement,
            fidelity_scores=fidelity_scores,
            coherence_score=self._calculate_coherence(synthesis_statement),
            created_at=datetime.now(timezone.utc)
        )

        # Store synthesis
        self.synthesis_index[synthesis_id] = synthesis

        # Update maturity
        self.maturity_state.cross_tradition_syntheses += 1

        return synthesis

    def _find_convergent_insights(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Find themes that appear across multiple traditions."""
        # Simplified implementation - would use embeddings in full version
        all_definitions = []
        for tradition, concepts in perspectives.items():
            for concept in concepts:
                all_definitions.append({
                    "tradition": tradition,
                    "text": concept.definition,
                    "name": concept.name
                })

        # Find overlapping themes (simplified)
        convergent = []
        seen_themes = set()
        for item in all_definitions:
            for other in all_definitions:
                if item["tradition"] != other["tradition"]:
                    # Simple word overlap check (would use semantic similarity)
                    words1 = set(item["text"].lower().split())
                    words2 = set(other["text"].lower().split())
                    overlap = words1 & words2
                    if len(overlap) > 5:
                        theme = f"Both {item['tradition'].value} and {other['tradition'].value} address {item['name']}"
                        if theme not in seen_themes:
                            convergent.append(theme)
                            seen_themes.add(theme)

        return convergent[:5]

    def _map_divergent_positions(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Identify where traditions diverge."""
        divergent = []
        for tradition, concepts in perspectives.items():
            for concept in concepts:
                if concept.opposed_concepts:
                    divergent.append(
                        f"{tradition.value}: {concept.name} contrasts with {', '.join(concept.opposed_concepts[:3])}"
                    )
        return divergent[:5]

    def _find_complementary_aspects(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Find where traditions complement each other."""
        complementary = []
        traditions = list(perspectives.keys())

        for i, t1 in enumerate(traditions):
            for t2 in traditions[i+1:]:
                # Check for different domains covered
                domains1 = set(c.domain for c in perspectives.get(t1, []))
                domains2 = set(c.domain for c in perspectives.get(t2, []))
                unique_to_1 = domains1 - domains2
                unique_to_2 = domains2 - domains1

                if unique_to_1:
                    complementary.append(
                        f"{t1.value} uniquely addresses {', '.join(d.value for d in unique_to_1)}"
                    )
                if unique_to_2:
                    complementary.append(
                        f"{t2.value} uniquely addresses {', '.join(d.value for d in unique_to_2)}"
                    )

        return complementary[:5]

    def _generate_synthesis_statement(
        self,
        topic: str,
        convergent: List[str],
        divergent: List[str],
        complementary: List[str],
        preserve_nuance: bool
    ) -> str:
        """Generate a synthesis statement."""
        parts = [f"Synthesis on '{topic}':\n"]

        if convergent:
            parts.append("Convergent insights: " + "; ".join(convergent[:3]))

        if divergent and preserve_nuance:
            parts.append("\nDivergent positions: " + "; ".join(divergent[:3]))

        if complementary:
            parts.append("\nComplementary aspects: " + "; ".join(complementary[:3]))

        return "\n".join(parts)

    def _calculate_fidelity(
        self,
        synthesis: str,
        concepts: List[PhilosophicalConcept],
        tradition: PhilosophicalTradition
    ) -> float:
        """Calculate fidelity to a tradition."""
        if not concepts:
            return 0.5

        # Check if key terms from concepts appear in synthesis
        synthesis_lower = synthesis.lower()
        matches = 0
        total = 0

        for concept in concepts:
            total += 1
            if concept.name.lower() in synthesis_lower:
                matches += 1

        return matches / total if total > 0 else 0.5

    def _calculate_coherence(self, synthesis: str) -> float:
        """Calculate coherence of synthesis."""
        # Simplified - check for logical connectors and structure
        connectors = ["therefore", "however", "moreover", "thus", "because"]
        score = 0.5

        for connector in connectors:
            if connector in synthesis.lower():
                score += 0.1

        return min(1.0, score)

    # ========================================================================
    # WISDOM INTEGRATION
    # ========================================================================

    def get_wisdom_for_context(
        self,
        context: Dict[str, Any],
        preferred_traditions: Optional[List[PhilosophicalTradition]] = None
    ) -> Dict[str, Any]:
        """
        Get philosophical wisdom appropriate for an engagement context.
        """
        # Determine appropriate traditions
        traditions = preferred_traditions or self._infer_traditions_for_context(context)

        # Select appropriate teachings
        teachings = []
        practical_guidance = []

        for tradition in traditions:
            tradition_key = tradition.value if isinstance(tradition, PhilosophicalTradition) else tradition
            if tradition_key in self.wisdom_teachings:
                teachings.extend(self.wisdom_teachings[tradition_key].get("core", [])[:2])
                practical_guidance.extend(self.wisdom_teachings[tradition_key].get("practical", [])[:2])

        return {
            "teachings": teachings[:5],
            "practical_guidance": practical_guidance[:5],
            "traditions_drawn_from": [t.value if isinstance(t, PhilosophicalTradition) else t for t in traditions],
        }

    def _infer_traditions_for_context(self, context: Dict[str, Any]) -> List[str]:
        """Infer appropriate traditions based on context."""
        traditions = []

        emotional_state = context.get("emotional_state", "neutral")
        current_need = context.get("current_need", "general")

        # Map emotional states to helpful traditions
        if emotional_state == "distressed":
            traditions.extend(["stoicism", "buddhist"])
        elif emotional_state == "seeking":
            traditions.extend(["vedantic", "daoist"])

        # Map needs to traditions
        if "ethics" in current_need.lower():
            traditions.extend(["aristotelian", "confucian"])
        elif "meaning" in current_need.lower():
            traditions.extend(["existentialism", "buddhist"])

        return traditions or ["stoicism", "buddhist"]  # Default

    # ========================================================================
    # INDEX MANAGEMENT
    # ========================================================================

    async def add_concept(self, concept: PhilosophicalConcept) -> str:
        """Add a concept to the index."""
        concept.created_at = concept.created_at or datetime.now(timezone.utc)
        concept.last_updated = datetime.now(timezone.utc)

        # Generate embedding if not provided
        if concept.embedding is None:
            concept.embedding = await self._generate_embedding(concept.to_embedding_text())

        self.concept_index[concept.concept_id] = concept

        # Update maturity
        self.maturity_state.total_concepts_integrated += 1
        self._update_tradition_depth(concept.tradition)
        self._update_domain_coverage(concept.domain)

        logger.info(f"Added concept: {concept.concept_id}")
        return concept.concept_id

    async def add_figure(self, figure: PhilosophicalFigure) -> str:
        """Add a figure to the index."""
        self.figure_index[figure.figure_id] = figure
        self.maturity_state.total_figures_indexed += 1
        logger.info(f"Added figure: {figure.figure_id}")
        return figure.figure_id

    async def add_text(self, text: PhilosophicalText) -> str:
        """Add a text to the index."""
        self.text_index[text.text_id] = text
        self.maturity_state.total_texts_indexed += 1
        logger.info(f"Added text: {text.text_id}")
        return text.text_id

    def _update_tradition_depth(self, tradition: PhilosophicalTradition) -> None:
        """Update tradition depth metric."""
        key = tradition.value
        current = self.maturity_state.traditions_depth.get(key, 0.0)
        # Increment with diminishing returns
        increment = 0.01 * (1 - current)
        self.maturity_state.traditions_depth[key] = min(1.0, current + increment)

    def _update_domain_coverage(self, domain: PhilosophicalDomain) -> None:
        """Update domain coverage metric."""
        key = domain.value
        current = self.maturity_state.domains_coverage.get(key, 0.0)
        increment = 0.01 * (1 - current)
        self.maturity_state.domains_coverage[key] = min(1.0, current + increment)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _search_by_name(
        self,
        query: str,
        index: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search index by name."""
        query_lower = query.lower()
        results = []

        for entity_id, entity in index.items():
            name = getattr(entity, 'name', '').lower()
            alt_names = [n.lower() for n in getattr(entity, 'alternate_names', [])]

            if query_lower == name:
                results.append({"concept": entity, "relevance": 1.0})
            elif query_lower in name:
                results.append({"concept": entity, "relevance": 0.8})
            elif any(query_lower in alt for alt in alt_names):
                results.append({"concept": entity, "relevance": 0.7})

        return results

    async def _semantic_search(
        self,
        query: str,
        collection: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        # Simplified - would use actual vector search in production
        query_embedding = await self._generate_embedding(query)

        if collection == "concepts":
            index = self.concept_index
        elif collection == "figures":
            index = self.figure_index
        else:
            index = {}

        results = []
        for entity_id, entity in index.items():
            if entity.embedding:
                similarity = self._cosine_similarity(query_embedding, entity.embedding)
                if similarity > 0.3:
                    results.append({"concept": entity, "relevance": similarity})

        return sorted(results, key=lambda x: x["relevance"], reverse=True)[:top_k]

    def _merge_results(
        self,
        name_results: List[Dict],
        semantic_results: List[Dict]
    ) -> List[Dict]:
        """Merge and deduplicate results."""
        seen = set()
        merged = []

        for result in name_results + semantic_results:
            concept = result["concept"]
            if concept.concept_id not in seen:
                seen.add(concept.concept_id)
                merged.append(result)

        return merged

    def _apply_filters(
        self,
        results: List[Dict],
        filters: QueryFilters
    ) -> List[Dict]:
        """Apply filters to results."""
        filtered = []

        for result in results:
            concept = result["concept"]

            if filters.traditions:
                if concept.tradition not in filters.traditions:
                    continue

            if filters.domains:
                if concept.domain not in filters.domains:
                    continue

            if filters.min_maturity > 0:
                if concept.maturity_score < filters.min_maturity:
                    continue

            filtered.append(result)

        return filtered

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Placeholder - would use actual embedding model
        # Returns deterministic pseudo-embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(self.embedding_dimensions):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
            embedding.append(value)

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_part = hashlib.md5(f"{prefix}{time.time()}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{hash_part}"

    # ========================================================================
    # STATISTICS AND MATURITY
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_concepts": len(self.concept_index),
            "total_figures": len(self.figure_index),
            "total_texts": len(self.text_index),
            "total_arguments": len(self.argument_index),
            "total_syntheses": len(self.synthesis_index),
            "maturity_state": {
                "overall_maturity": self.maturity_state.get_overall_maturity(),
                "maturity_level": self.maturity_state.get_maturity_level().value,
                "traditions_depth": self.maturity_state.traditions_depth,
                "domains_coverage": self.maturity_state.domains_coverage,
                "cross_tradition_syntheses": self.maturity_state.cross_tradition_syntheses,
            }
        }

    def get_maturity_state(self) -> PhilosophicalMaturityState:
        """Get current maturity state."""
        return self.maturity_state
