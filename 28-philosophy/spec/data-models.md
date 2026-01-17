# Data Models Specification

## Overview

This document defines the core data structures used by Form 28: Philosophical Consciousness. These models enable systematic representation of philosophical knowledge, support vector embedding for RAG retrieval, and facilitate cross-tradition synthesis.

---

## Core Enumerations

### PhilosophicalTradition

Comprehensive enumeration of philosophical schools and movements.

```python
class PhilosophicalTradition(Enum):
    """Major philosophical traditions indexed in Form 28"""

    # === WESTERN TRADITIONS ===

    # Ancient
    PRESOCRATIC = "presocratic"
    PLATONIC = "platonic"
    ARISTOTELIAN = "aristotelian"
    STOICISM = "stoicism"
    EPICUREANISM = "epicureanism"
    SKEPTICISM_ANCIENT = "skepticism_ancient"
    NEOPLATONISM = "neoplatonism"

    # Medieval
    SCHOLASTICISM = "scholasticism"
    THOMISM = "thomism"
    AUGUSTINIAN = "augustinian"
    NOMINALISM = "nominalism"

    # Early Modern
    RATIONALISM = "rationalism"
    EMPIRICISM = "empiricism"
    KANTIAN = "kantian"

    # 19th Century
    GERMAN_IDEALISM = "german_idealism"
    HEGELIAN = "hegelian"
    NIETZSCHEAN = "nietzschean"
    UTILITARIANISM = "utilitarianism"
    MARXISM = "marxism"

    # 20th Century - Present
    PHENOMENOLOGY = "phenomenology"
    EXISTENTIALISM = "existentialism"
    ANALYTIC = "analytic"
    PRAGMATISM = "pragmatism"
    CONTINENTAL = "continental"
    PROCESS_PHILOSOPHY = "process_philosophy"
    CRITICAL_THEORY = "critical_theory"
    POSTMODERNISM = "postmodernism"
    FEMINIST_PHILOSOPHY = "feminist_philosophy"

    # === EASTERN TRADITIONS ===

    # Buddhist
    BUDDHIST_THERAVADA = "buddhist_theravada"
    BUDDHIST_MAHAYANA = "buddhist_mahayana"
    BUDDHIST_VAJRAYANA = "buddhist_vajrayana"
    BUDDHIST_ZEN = "buddhist_zen"
    BUDDHIST_MADHYAMAKA = "buddhist_madhyamaka"
    BUDDHIST_YOGACARA = "buddhist_yogacara"
    BUDDHIST_PURE_LAND = "buddhist_pure_land"

    # Chinese
    DAOIST = "daoist"
    CONFUCIAN = "confucian"
    NEO_CONFUCIAN = "neo_confucian"
    MOHISM = "mohism"
    LEGALISM = "legalism"

    # Indian (Astika - Orthodox)
    VEDANTIC_ADVAITA = "vedantic_advaita"
    VEDANTIC_VISHISHTADVAITA = "vedantic_vishishtadvaita"
    VEDANTIC_DVAITA = "vedantic_dvaita"
    SAMKHYA = "samkhya"
    YOGA_PHILOSOPHY = "yoga_philosophy"
    NYAYA = "nyaya"
    VAISHESHIKA = "vaisheshika"
    MIMAMSA = "mimamsa"

    # Indian (Nastika - Heterodox)
    JAIN = "jain"
    CARVAKA = "carvaka"
    AJIVIKA = "ajivika"

    # Japanese
    KYOTO_SCHOOL = "kyoto_school"
    JAPANESE_ZEN = "japanese_zen"

    # Comparative/Syncretic
    PERENNIAL_PHILOSOPHY = "perennial_philosophy"
    COMPARATIVE_PHILOSOPHY = "comparative_philosophy"
```

### PhilosophicalDomain

Enumeration of philosophical areas of inquiry.

```python
class PhilosophicalDomain(Enum):
    """Domains of philosophical inquiry"""

    # Core Branches
    METAPHYSICS = "metaphysics"
    EPISTEMOLOGY = "epistemology"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"
    LOGIC = "logic"

    # Philosophy of X
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    PHILOSOPHY_OF_LANGUAGE = "philosophy_of_language"
    POLITICAL_PHILOSOPHY = "political_philosophy"
    PHILOSOPHY_OF_SCIENCE = "philosophy_of_science"
    PHILOSOPHY_OF_RELIGION = "philosophy_of_religion"
    PHILOSOPHY_OF_MATHEMATICS = "philosophy_of_mathematics"
    PHILOSOPHY_OF_LAW = "philosophy_of_law"
    PHILOSOPHY_OF_HISTORY = "philosophy_of_history"
    PHILOSOPHY_OF_ART = "philosophy_of_art"

    # Specialized Areas
    EXISTENTIAL = "existential"
    PHENOMENOLOGICAL = "phenomenological"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    METAETHICS = "metaethics"
    APPLIED_ETHICS = "applied_ethics"
    SOCIAL_PHILOSOPHY = "social_philosophy"
    PHILOSOPHY_OF_ACTION = "philosophy_of_action"
    PHILOSOPHY_OF_PERCEPTION = "philosophy_of_perception"

    # Eastern-Specific
    SOTERIOLOGY = "soteriology"  # Liberation/salvation
    MEDITATION_THEORY = "meditation_theory"
    KARMA_THEORY = "karma_theory"
```

### ArgumentType

Types of philosophical arguments.

```python
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
    APPEAL_TO_INTUITION = "appeal_to_intuition"
```

### MaturityLevel

Levels of philosophical understanding depth.

```python
class MaturityLevel(Enum):
    """Depth of understanding for philosophical concepts"""
    NASCENT = "nascent"           # 0.0-0.2: Basic awareness
    DEVELOPING = "developing"     # 0.2-0.4: Surface understanding
    COMPETENT = "competent"       # 0.4-0.6: Solid grasp
    PROFICIENT = "proficient"     # 0.6-0.8: Deep understanding
    MASTERFUL = "masterful"       # 0.8-1.0: Nuanced mastery
```

---

## Core Dataclasses

### PhilosophicalConcept

Represents a philosophical concept, idea, or term.

```python
@dataclass
class PhilosophicalConcept:
    """
    Represents a philosophical concept within the knowledge base.

    Concepts are the atomic units of philosophical knowledge,
    embedded for semantic retrieval and linked to form a knowledge graph.
    """

    # Identification
    concept_id: str                          # Unique identifier (e.g., "categorical_imperative")
    name: str                                # Display name
    alternate_names: List[str] = field(default_factory=list)  # Synonyms, translations

    # Classification
    tradition: PhilosophicalTradition        # Primary tradition
    secondary_traditions: List[PhilosophicalTradition] = field(default_factory=list)
    domain: PhilosophicalDomain              # Primary domain
    secondary_domains: List[PhilosophicalDomain] = field(default_factory=list)

    # Content
    definition: str                          # Core definition
    extended_description: Optional[str] = None
    key_arguments: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)

    # Relationships
    related_concepts: List[str] = field(default_factory=list)    # concept_ids
    opposed_concepts: List[str] = field(default_factory=list)    # Dialectical opposites
    prerequisite_concepts: List[str] = field(default_factory=list)

    # Attribution
    key_figures: List[str] = field(default_factory=list)         # figure_ids
    primary_texts: List[str] = field(default_factory=list)       # text_ids

    # RAG Integration
    embedding: Optional[List[float]] = None  # Vector embedding (768 dims)
    embedding_model: str = "all-mpnet-base-v2"

    # Maturity Tracking
    maturity_score: float = 0.0              # 0.0-1.0
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    research_depth: int = 0                  # Number of research sessions
    last_updated: Optional[datetime] = None

    # Metadata
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
            parts.append(f"Description: {self.extended_description}")
        return " | ".join(parts)
```

### PhilosophicalFigure

Represents a philosopher or thinker.

```python
@dataclass
class PhilosophicalFigure:
    """
    Represents a philosopher, sage, or significant thinker.
    """

    # Identification
    figure_id: str                           # Unique identifier
    name: str                                # Primary name
    alternate_names: List[str] = field(default_factory=list)

    # Biographical
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    birth_place: Optional[str] = None
    nationality: Optional[str] = None
    era: Optional[str] = None                # e.g., "Ancient", "Medieval", "Modern"

    # Philosophical Identity
    traditions: List[PhilosophicalTradition] = field(default_factory=list)
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    schools_founded: List[str] = field(default_factory=list)

    # Intellectual Content
    core_ideas: List[str] = field(default_factory=list)      # concept_ids
    key_arguments: List[str] = field(default_factory=list)
    famous_quotes: List[str] = field(default_factory=list)

    # Works
    key_works: List[str] = field(default_factory=list)       # text_ids

    # Intellectual Genealogy
    teachers: List[str] = field(default_factory=list)        # figure_ids
    students: List[str] = field(default_factory=list)        # figure_ids
    influences: List[str] = field(default_factory=list)      # Who influenced them
    influenced: List[str] = field(default_factory=list)      # Who they influenced

    # RAG Integration
    embedding: Optional[List[float]] = None

    # Metadata
    biography_summary: Optional[str] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        traditions_str = ", ".join(t.value for t in self.traditions)
        return f"Philosopher: {self.name} | Traditions: {traditions_str} | Ideas: {', '.join(self.core_ideas[:5])}"
```

### PhilosophicalText

Represents a primary philosophical text or work.

```python
@dataclass
class PhilosophicalText:
    """
    Represents a primary philosophical text, treatise, or scripture.
    """

    # Identification
    text_id: str                             # Unique identifier
    title: str                               # Primary title
    alternate_titles: List[str] = field(default_factory=list)

    # Attribution
    author: Optional[str] = None             # figure_id (None for anonymous)
    author_name: Optional[str] = None        # Display name
    editors: List[str] = field(default_factory=list)
    translators: List[str] = field(default_factory=list)

    # Classification
    tradition: PhilosophicalTradition
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    genre: Optional[str] = None              # e.g., "dialogue", "treatise", "sutra"

    # Temporal
    year_written: Optional[int] = None
    year_published: Optional[int] = None
    period: Optional[str] = None

    # Content
    summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)    # concept_ids
    key_arguments: List[str] = field(default_factory=list)
    structure: Optional[str] = None          # Book/chapter structure

    # Access
    local_path: Optional[str] = None         # Path if locally indexed
    digital_sources: List[Dict[str, str]] = field(default_factory=list)
    is_indexed: bool = False

    # RAG Integration
    embedding: Optional[List[float]] = None
    chunk_count: int = 0                     # Number of indexed chunks

    # Influence
    influenced_texts: List[str] = field(default_factory=list)
    influenced_by: List[str] = field(default_factory=list)

    # Metadata
    sources: List[Dict[str, str]] = field(default_factory=list)
```

### PhilosophicalArgument

Represents a structured philosophical argument.

```python
@dataclass
class PhilosophicalArgument:
    """
    Represents a philosophical argument with premises and conclusion.
    """

    # Identification
    argument_id: str
    name: str                                # e.g., "Cogito Argument"

    # Classification
    argument_type: ArgumentType
    tradition: PhilosophicalTradition
    domain: PhilosophicalDomain

    # Structure
    premises: List[str]                      # List of premise statements
    conclusion: str
    logical_form: Optional[str] = None       # Formal representation

    # Attribution
    originator: Optional[str] = None         # figure_id
    text_source: Optional[str] = None        # text_id

    # Analysis
    key_concepts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Dialectic
    objections: List[Dict[str, str]] = field(default_factory=list)
    replies: List[Dict[str, str]] = field(default_factory=list)
    related_arguments: List[str] = field(default_factory=list)

    # Assessment
    validity_notes: Optional[str] = None
    soundness_notes: Optional[str] = None
    scholarly_consensus: Optional[str] = None

    # RAG Integration
    embedding: Optional[List[float]] = None
```

### TraditionProfile

Comprehensive profile of a philosophical tradition.

```python
@dataclass
class TraditionProfile:
    """
    Comprehensive profile of a philosophical tradition/school.
    """

    tradition_id: PhilosophicalTradition
    display_name: str

    # Historical Context
    origin_period: Optional[str] = None
    origin_location: Optional[str] = None
    historical_development: Optional[str] = None

    # Core Philosophy
    core_tenets: List[str] = field(default_factory=list)
    central_questions: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)     # concept_ids
    methodology: Optional[str] = None

    # Figures and Texts
    founders: List[str] = field(default_factory=list)         # figure_ids
    major_figures: List[str] = field(default_factory=list)
    canonical_texts: List[str] = field(default_factory=list)  # text_ids

    # Relationships
    influences_from: List[PhilosophicalTradition] = field(default_factory=list)
    influences_to: List[PhilosophicalTradition] = field(default_factory=list)
    opposed_to: List[PhilosophicalTradition] = field(default_factory=list)
    compatible_with: List[PhilosophicalTradition] = field(default_factory=list)

    # Domain Coverage
    primary_domains: List[PhilosophicalDomain] = field(default_factory=list)
    secondary_domains: List[PhilosophicalDomain] = field(default_factory=list)

    # Maturity
    knowledge_depth: float = 0.0             # 0.0-1.0
    concept_count: int = 0
    figure_count: int = 0
    text_count: int = 0

    # Wisdom Integration
    wisdom_teachings: List[str] = field(default_factory=list)
    practical_guidance: List[str] = field(default_factory=list)
```

---

## Maturity and Synthesis Models

### PhilosophicalMaturityState

Tracks overall philosophical understanding maturation.

```python
@dataclass
class PhilosophicalMaturityState:
    """
    Tracks the system's philosophical understanding maturation.
    """

    # Overall Metrics
    total_concepts_integrated: int = 0
    total_figures_indexed: int = 0
    total_texts_indexed: int = 0
    total_arguments_analyzed: int = 0

    # Tradition Depth (tradition_id -> depth 0.0-1.0)
    traditions_depth: Dict[str, float] = field(default_factory=dict)

    # Domain Coverage (domain -> coverage 0.0-1.0)
    domains_coverage: Dict[str, float] = field(default_factory=dict)

    # Synthesis Metrics
    cross_tradition_syntheses: int = 0
    successful_syntheses: int = 0
    synthesis_quality_avg: float = 0.0

    # Research Metrics
    research_sessions_completed: int = 0
    knowledge_gaps_filled: int = 0
    external_sources_consulted: int = 0

    # Quality Metrics
    nuance_detection_accuracy: float = 0.0
    consistency_score: float = 0.0
    source_verification_rate: float = 0.0

    # Wisdom Integration
    wisdom_teachings_integrated: int = 0
    practical_applications_generated: int = 0

    # Temporal
    last_maturity_assessment: Optional[datetime] = None
    maturity_history: List[Dict[str, Any]] = field(default_factory=list)

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
```

### CrossTraditionSynthesis

Represents a synthesis across philosophical traditions.

```python
@dataclass
class CrossTraditionSynthesis:
    """
    Represents a synthesis of insights across philosophical traditions.
    """

    synthesis_id: str
    topic: str                               # The question or topic being synthesized

    # Traditions Involved
    traditions: List[PhilosophicalTradition]
    concepts_involved: List[str] = field(default_factory=list)  # concept_ids

    # Synthesis Content
    convergent_insights: List[str] = field(default_factory=list)
    divergent_positions: List[str] = field(default_factory=list)
    complementary_aspects: List[str] = field(default_factory=list)
    synthesis_statement: Optional[str] = None

    # Quality Assessment
    coherence_score: float = 0.0             # 0.0-1.0
    fidelity_scores: Dict[str, float] = field(default_factory=dict)  # Per-tradition
    nuance_preserved: bool = True

    # Metadata
    created_at: Optional[datetime] = None
    methodology: Optional[str] = None
```

---

## Research and Query Models

### ResearchTask

Defines an autonomous research task.

```python
@dataclass
class ResearchTask:
    """
    Defines an autonomous research task for knowledge expansion.
    """

    task_id: str
    query: str                               # What to research

    # Task Configuration
    sources: List[str] = field(default_factory=list)  # Source types to use
    priority: int = 5                        # 1-10
    max_depth: int = 2                       # Recursion depth for related topics

    # Targeting
    target_traditions: List[PhilosophicalTradition] = field(default_factory=list)
    target_domains: List[PhilosophicalDomain] = field(default_factory=list)

    # Status
    status: str = "pending"                  # pending, in_progress, completed, failed
    triggered_by: str = "user"               # user, gap_detection, scheduled

    # Results
    concepts_discovered: List[str] = field(default_factory=list)
    figures_discovered: List[str] = field(default_factory=list)
    texts_discovered: List[str] = field(default_factory=list)
    sources_consulted: List[str] = field(default_factory=list)

    # Metadata
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
```

### PhilosophicalQuery

Represents a query to the philosophical knowledge base.

```python
@dataclass
class PhilosophicalQuery:
    """
    Represents a query to the philosophical knowledge base.
    """

    query_id: str
    query_text: str

    # Filters
    tradition_filter: Optional[List[PhilosophicalTradition]] = None
    domain_filter: Optional[List[PhilosophicalDomain]] = None
    figure_filter: Optional[List[str]] = None

    # Options
    include_synthesis: bool = False
    trigger_research_if_missing: bool = False
    max_results: int = 10
    min_relevance: float = 0.5

    # Context
    conversation_context: Optional[str] = None
    requester_context: Optional[Dict[str, Any]] = None

    # Timestamp
    timestamp: Optional[datetime] = None
```

### PhilosophicalResponse

Response from the philosophical knowledge base.

```python
@dataclass
class PhilosophicalResponse:
    """
    Response from a philosophical knowledge base query.
    """

    query_id: str

    # Results
    concepts: List[PhilosophicalConcept] = field(default_factory=list)
    figures: List[PhilosophicalFigure] = field(default_factory=list)
    texts: List[PhilosophicalText] = field(default_factory=list)
    arguments: List[PhilosophicalArgument] = field(default_factory=list)

    # Synthesis (if requested)
    synthesis: Optional[CrossTraditionSynthesis] = None

    # Relevance Scores
    relevance_scores: Dict[str, float] = field(default_factory=dict)

    # Research Triggered
    research_triggered: bool = False
    research_task_id: Optional[str] = None

    # Wisdom Integration
    wisdom_teachings: List[str] = field(default_factory=list)
    practical_guidance: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    sources_consulted: List[str] = field(default_factory=list)
```
