# Form 29: Folk & Indigenous Wisdom - Data Structures

## Overview

This document defines the core data structures for Form 29: Folk & Indigenous Wisdom. These structures represent cultural knowledge, oral traditions, animistic practices, indigenous cosmologies, and cross-cultural wisdom synthesis. The data models support respectful encoding of traditional knowledge systems while enabling computational retrieval, synthesis, and cross-form integration.

## Core Data Models

### Wisdom Teaching Structures

**FolkWisdomTeaching**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import time
import uuid


@dataclass
class FolkWisdomTeaching:
    """Core representation of a folk or indigenous wisdom teaching."""

    teaching_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Teaching content
    title: str = ""
    summary: str = ""
    core_principle: str = ""
    full_text: Optional[str] = None
    embedded_values: List[str] = field(default_factory=list)

    # Cultural context
    source_tradition: str = ""
    source_region: str = ""
    source_community: str = ""
    cultural_context: Dict[str, Any] = field(default_factory=dict)

    # Transmission metadata
    transmission_mode: 'TransmissionMode' = None
    transmission_lineage: List[str] = field(default_factory=list)
    authorized_source: bool = False
    permission_status: 'PermissionStatus' = None

    # Classification
    domain: 'WisdomDomain' = None
    themes: List[str] = field(default_factory=list)
    ethical_dimensions: List[str] = field(default_factory=list)
    related_teachings: List[str] = field(default_factory=list)

    # Maturity and quality
    maturity_level: 'MaturityLevel' = None
    source_diversity_score: float = 0.0
    scholarly_references: List[str] = field(default_factory=list)
    cross_reference_count: int = 0

    # Embedding for semantic retrieval
    embedding_vector: Optional[List[float]] = None
    embedding_model: str = ""
    embedding_timestamp: Optional[float] = None

    def calculate_quality_score(self) -> float:
        """Calculate composite quality score for the teaching record."""
        attribution_score = 1.0 if self.authorized_source else 0.5
        context_score = min(1.0, len(self.cultural_context) / 5.0)
        reference_score = min(1.0, len(self.scholarly_references) / 3.0)
        return (attribution_score + context_score + reference_score) / 3.0
```

**OralTradition**
```python
@dataclass
class OralTradition:
    """Representation of an oral tradition element (story, song, proverb)."""

    tradition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Content
    title: str = ""
    tradition_type: 'OralTraditionType' = None
    narrative_text: str = ""
    original_language: str = ""
    translation_notes: str = ""

    # Story structure (for narrative traditions)
    characters: List[Dict[str, Any]] = field(default_factory=list)
    plot_elements: List[str] = field(default_factory=list)
    moral_teachings: List[str] = field(default_factory=list)
    symbolic_elements: Dict[str, str] = field(default_factory=dict)

    # Cultural context
    source_tradition: str = ""
    source_region: str = ""
    performance_context: str = ""
    seasonal_associations: List[str] = field(default_factory=list)
    ceremonial_role: Optional[str] = None

    # Wisdom extraction
    extracted_principles: List[str] = field(default_factory=list)
    practical_applications: List[str] = field(default_factory=list)
    related_teaching_ids: List[str] = field(default_factory=list)

    # Attribution
    collector: str = ""
    collection_date: Optional[str] = None
    permission_status: 'PermissionStatus' = None
    scholarly_source: str = ""

    # Embedding
    embedding_vector: Optional[List[float]] = None
```

### Animistic Practice Structures

**AnimisticPractice**
```python
@dataclass
class AnimisticPractice:
    """Representation of an animistic or ceremonial practice."""

    practice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Practice description
    name: str = ""
    description: str = ""
    purpose: str = ""
    practice_type: 'AnimisticPracticeType' = None

    # Spirit/entity relationships
    associated_spirits: List[Dict[str, Any]] = field(default_factory=list)
    natural_entities: List[str] = field(default_factory=list)
    ancestor_connections: List[str] = field(default_factory=list)
    power_dynamics: Dict[str, str] = field(default_factory=dict)

    # Ritual structure
    ritual_elements: List[Dict[str, Any]] = field(default_factory=list)
    required_materials: List[str] = field(default_factory=list)
    spatial_requirements: Dict[str, Any] = field(default_factory=dict)
    temporal_requirements: Dict[str, Any] = field(default_factory=dict)

    # Restrictions and protocols
    access_restrictions: 'AccessRestriction' = None
    gender_restrictions: Optional[str] = None
    age_restrictions: Optional[str] = None
    initiation_required: bool = False
    seasonal_restrictions: List[str] = field(default_factory=list)

    # Cultural context
    source_tradition: str = ""
    source_region: str = ""
    community_role: str = ""
    living_tradition: bool = True

    # Taboos and sacred boundaries
    taboos: List[str] = field(default_factory=list)
    sacred_boundary_notes: str = ""
    public_shareable: bool = False

    # Scholarly documentation
    ethnographic_references: List[str] = field(default_factory=list)
    documentation_quality: float = 0.0
```

### Cosmology Structures

**IndigenousCosmology**
```python
@dataclass
class IndigenousCosmology:
    """Representation of an indigenous or folk cosmological worldview."""

    cosmology_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Worldview description
    tradition_name: str = ""
    worldview_summary: str = ""
    cosmological_type: 'CosmologicalType' = None

    # Spatial organization
    spatial_structure: Dict[str, Any] = field(default_factory=dict)
    world_layers: List[Dict[str, Any]] = field(default_factory=list)
    cardinal_directions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sacred_geography: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal conception
    time_structure: Dict[str, Any] = field(default_factory=dict)
    cyclical_patterns: List[str] = field(default_factory=list)
    creation_narrative: Optional[str] = None
    eschatological_view: Optional[str] = None

    # Being categories
    being_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    consciousness_attribution: Dict[str, str] = field(default_factory=dict)
    human_nonhuman_relations: Dict[str, Any] = field(default_factory=dict)
    spirit_categories: List[Dict[str, Any]] = field(default_factory=list)

    # Ethical framework
    ethical_principles: List[str] = field(default_factory=list)
    relational_obligations: Dict[str, List[str]] = field(default_factory=dict)
    reciprocity_norms: List[str] = field(default_factory=list)
    taboo_systems: List[Dict[str, Any]] = field(default_factory=list)

    # Cultural context
    source_region: str = ""
    source_community: str = ""
    language_family: str = ""
    related_cosmologies: List[str] = field(default_factory=list)

    # Attribution
    scholarly_references: List[str] = field(default_factory=list)
    indigenous_scholars: List[str] = field(default_factory=list)
    documentation_quality: float = 0.0
```

### Animal Wisdom Structures

**IndigenousAnimalWisdom**
```python
@dataclass
class IndigenousAnimalWisdom:
    """Traditional knowledge about animal behavior and meaning."""

    wisdom_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Animal identification
    animal_common_name: str = ""
    animal_scientific_name: Optional[str] = None
    local_names: Dict[str, str] = field(default_factory=dict)

    # Traditional knowledge
    behavioral_observations: List[str] = field(default_factory=list)
    ecological_knowledge: List[str] = field(default_factory=list)
    medicinal_associations: List[str] = field(default_factory=list)
    spiritual_significance: str = ""

    # Cultural meaning
    symbolic_meanings: Dict[str, str] = field(default_factory=dict)
    story_appearances: List[str] = field(default_factory=list)
    ceremonial_roles: List[str] = field(default_factory=list)
    totem_clan_associations: List[str] = field(default_factory=list)

    # Scientific cross-reference
    form_30_species_id: Optional[str] = None
    scientific_corroboration: List[Dict[str, Any]] = field(default_factory=list)
    unique_insights: List[str] = field(default_factory=list)

    # Cultural context
    source_tradition: str = ""
    source_region: str = ""
    permission_status: 'PermissionStatus' = None
    scholarly_references: List[str] = field(default_factory=list)
```

## Enumeration Types

### Wisdom Domain Enumerations

```python
class WisdomDomain(Enum):
    """Categories of folk wisdom teachings."""
    ETHICS_AND_CONDUCT = "ethics_and_conduct"
    NATURE_AND_ECOLOGY = "nature_and_ecology"
    HEALING_AND_MEDICINE = "healing_and_medicine"
    SOCIAL_RELATIONS = "social_relations"
    SPIRITUAL_PRACTICE = "spiritual_practice"
    COSMOLOGY_AND_ORIGIN = "cosmology_and_origin"
    DEATH_AND_AFTERLIFE = "death_and_afterlife"
    SEASONAL_KNOWLEDGE = "seasonal_knowledge"
    PRACTICAL_SKILLS = "practical_skills"
    GOVERNANCE_AND_LAW = "governance_and_law"


class TransmissionMode(Enum):
    """How wisdom is transmitted across generations."""
    ORAL_NARRATIVE = "oral_narrative"
    SONG = "song"
    RITUAL = "ritual"
    APPRENTICESHIP = "apprenticeship"
    VISUAL_ART = "visual_art"
    DANCE = "dance"
    PROVERB = "proverb"
    DREAM_VISION = "dream_vision"
    CEREMONIAL_INSTRUCTION = "ceremonial_instruction"
    COMMUNITY_PRACTICE = "community_practice"


class OralTraditionType(Enum):
    """Types of oral tradition elements."""
    MYTH = "myth"
    LEGEND = "legend"
    FOLKTALE = "folktale"
    FABLE = "fable"
    PROVERB = "proverb"
    RIDDLE = "riddle"
    SONG = "song"
    CHANT = "chant"
    PRAYER = "prayer"
    EPIC = "epic"
    GENEALOGY = "genealogy"
    PLACE_NARRATIVE = "place_narrative"


class AnimisticPracticeType(Enum):
    """Categories of animistic practices."""
    SPIRIT_COMMUNICATION = "spirit_communication"
    ANCESTOR_VENERATION = "ancestor_veneration"
    NATURE_CEREMONY = "nature_ceremony"
    HEALING_RITUAL = "healing_ritual"
    DIVINATION = "divination"
    VISION_QUEST = "vision_quest"
    SEASONAL_CEREMONY = "seasonal_ceremony"
    RITE_OF_PASSAGE = "rite_of_passage"
    PROTECTIVE_PRACTICE = "protective_practice"
    SHAMANIC_JOURNEY = "shamanic_journey"


class CosmologicalType(Enum):
    """Types of cosmological worldviews."""
    LAYERED_UNIVERSE = "layered_universe"
    CYCLICAL_TIME = "cyclical_time"
    ANIMISTIC_FIELD = "animistic_field"
    ANCESTOR_CENTERED = "ancestor_centered"
    EARTH_CENTERED = "earth_centered"
    SKY_CENTERED = "sky_centered"
    WATER_CENTERED = "water_centered"
    TREE_AXIS = "tree_axis"
    MOUNTAIN_AXIS = "mountain_axis"
    DREAMTIME = "dreamtime"
```

### Status and Access Enumerations

```python
class PermissionStatus(Enum):
    """Permission status for sharing indigenous knowledge."""
    FREELY_SHARED = "freely_shared"
    COMMUNITY_APPROVED = "community_approved"
    SCHOLARLY_PUBLISHED = "scholarly_published"
    RESTRICTED_ACCESS = "restricted_access"
    SACRED_PRIVATE = "sacred_private"
    UNKNOWN = "unknown"


class AccessRestriction(Enum):
    """Access restriction levels for practices and teachings."""
    PUBLIC = "public"
    COMMUNITY_ONLY = "community_only"
    INITIATED_ONLY = "initiated_only"
    GENDER_RESTRICTED = "gender_restricted"
    ELDER_ONLY = "elder_only"
    SACRED_RESTRICTED = "sacred_restricted"


class MaturityLevel(Enum):
    """Maturity level of a wisdom record."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


class RegionCode(Enum):
    """Major world regions for folk wisdom coverage."""
    WEST_AFRICA = "west_africa"
    EAST_AFRICA = "east_africa"
    SOUTHERN_AFRICA = "southern_africa"
    CENTRAL_AFRICA = "central_africa"
    NORTH_AFRICA = "north_africa"
    CELTIC = "celtic"
    NORSE_GERMANIC = "norse_germanic"
    SLAVIC = "slavic"
    BALTIC = "baltic"
    MEDITERRANEAN = "mediterranean"
    FINNO_UGRIC = "finno_ugric"
    SIBERIAN = "siberian"
    CENTRAL_ASIAN = "central_asian"
    SOUTHEAST_ASIAN = "southeast_asian"
    EAST_ASIAN_FOLK = "east_asian_folk"
    SOUTH_ASIAN_FOLK = "south_asian_folk"
    POLYNESIAN = "polynesian"
    MELANESIAN = "melanesian"
    MICRONESIAN = "micronesian"
    ABORIGINAL_AUSTRALIAN = "aboriginal_australian"
    ARCTIC_INUIT = "arctic_inuit"
    PACIFIC_NORTHWEST = "pacific_northwest"
    PLAINS = "plains"
    EASTERN_WOODLANDS = "eastern_woodlands"
    MESOAMERICAN = "mesoamerican"
    AMAZONIAN = "amazonian"
    ANDEAN = "andean"
```

## Input/Output Structures

### Input Structures

**WisdomIngestionInput**
```python
@dataclass
class WisdomIngestionInput:
    """Input structure for ingesting new wisdom into the system."""

    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Raw content
    raw_text: str = ""
    source_type: str = ""  # ethnographic, oral_recording, scholarly, community
    original_language: str = ""

    # Source metadata
    source_reference: str = ""
    collector_name: str = ""
    collection_context: Dict[str, Any] = field(default_factory=dict)
    community_name: str = ""
    region: str = ""

    # Processing directives
    target_categories: List[str] = field(default_factory=list)
    extract_teachings: bool = True
    extract_narratives: bool = True
    extract_practices: bool = True
    generate_embeddings: bool = True

    # Ethical checks
    permission_documentation: Optional[str] = None
    sensitivity_flags: List[str] = field(default_factory=list)
    sacred_content_check: bool = True


@dataclass
class WisdomQueryInput:
    """Input structure for querying wisdom."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Query content
    query_text: str = ""
    query_type: str = ""  # thematic, regional, situational, cross_cultural

    # Filters
    region_filter: Optional[List[str]] = None
    tradition_filter: Optional[List[str]] = None
    domain_filter: Optional[List[str]] = None
    maturity_minimum: Optional[str] = None
    permission_filter: List[str] = field(
        default_factory=lambda: ["freely_shared", "community_approved", "scholarly_published"]
    )

    # Context
    user_context: Dict[str, Any] = field(default_factory=dict)
    situational_context: Optional[str] = None
    ethical_sensitivity_level: str = "standard"

    # Response parameters
    max_results: int = 10
    include_narratives: bool = True
    include_practices: bool = False
    include_cross_cultural: bool = True
    output_format: str = "standard"
```

### Output Structures

**WisdomRetrievalOutput**
```python
@dataclass
class WisdomRetrievalOutput:
    """Output structure for wisdom retrieval results."""

    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Retrieved content
    teachings: List[FolkWisdomTeaching] = field(default_factory=list)
    narratives: List[OralTradition] = field(default_factory=list)
    practices: List[AnimisticPractice] = field(default_factory=list)
    animal_wisdom: List[IndigenousAnimalWisdom] = field(default_factory=list)

    # Synthesis
    cross_cultural_themes: List[Dict[str, Any]] = field(default_factory=list)
    principle_summary: str = ""
    contradictions_noted: List[str] = field(default_factory=list)

    # Quality metrics
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    cultural_diversity_score: float = 0.0
    source_quality_score: float = 0.0
    attribution_completeness: float = 0.0

    # Sensitivity checks
    sensitivity_warnings: List[str] = field(default_factory=list)
    filtered_content_count: int = 0
    attribution_notes: List[str] = field(default_factory=list)

    # Processing metadata
    processing_time_ms: float = 0.0
    total_candidates_evaluated: int = 0
    retrieval_confidence: float = 0.0
```

**CrossCulturalSynthesisOutput**
```python
@dataclass
class CrossCulturalSynthesisOutput:
    """Output structure for cross-cultural wisdom synthesis."""

    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Synthesis content
    common_themes: List[Dict[str, Any]] = field(default_factory=list)
    cultural_variations: List[Dict[str, Any]] = field(default_factory=list)
    complementary_insights: List[Dict[str, Any]] = field(default_factory=list)
    tensions_and_contradictions: List[Dict[str, Any]] = field(default_factory=list)

    # Participating traditions
    traditions_included: List[str] = field(default_factory=list)
    regions_represented: List[str] = field(default_factory=list)
    teaching_count_by_tradition: Dict[str, int] = field(default_factory=dict)

    # Integration quality
    coherence_score: float = 0.0
    respect_score: float = 0.0
    accuracy_score: float = 0.0
    utility_score: float = 0.0

    # Attribution
    source_attributions: List[Dict[str, str]] = field(default_factory=list)
    limitation_acknowledgments: List[str] = field(default_factory=list)
```

## Internal State Structures

### System State

**FolkWisdomSystemState**
```python
@dataclass
class FolkWisdomSystemState:
    """Complete system state for the Folk Wisdom consciousness form."""

    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Knowledge base state
    total_teachings: int = 0
    total_narratives: int = 0
    total_practices: int = 0
    total_cosmologies: int = 0
    total_animal_wisdom: int = 0

    # Regional coverage
    region_coverage: Dict[str, int] = field(default_factory=dict)
    tradition_coverage: Dict[str, int] = field(default_factory=dict)
    domain_coverage: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    average_maturity_level: float = 0.0
    average_source_quality: float = 0.0
    attribution_completeness: float = 0.0
    cross_reference_density: float = 0.0

    # Processing state
    active_queries: int = 0
    pending_ingestions: int = 0
    recent_retrievals: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-form state
    form_30_links: int = 0  # Animal cognition cross-references
    form_28_links: int = 0  # Philosophy cross-references
    form_27_links: int = 0  # Altered states cross-references

    # Health metrics
    system_health: float = 1.0
    error_count: int = 0
    last_health_check: float = field(default_factory=time.time)

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Generate a summary of regional and domain coverage."""
        return {
            "total_records": (
                self.total_teachings + self.total_narratives +
                self.total_practices + self.total_cosmologies +
                self.total_animal_wisdom
            ),
            "regions_covered": len(self.region_coverage),
            "traditions_covered": len(self.tradition_coverage),
            "domains_covered": len(self.domain_coverage),
            "quality_score": (
                self.average_maturity_level + self.average_source_quality +
                self.attribution_completeness
            ) / 3.0
        }
```

### Processing Buffers

**WisdomProcessingBuffer**
```python
@dataclass
class WisdomProcessingBuffer:
    """Buffer for in-progress wisdom processing operations."""

    buffer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Ingestion buffer
    pending_teachings: List[Dict[str, Any]] = field(default_factory=list)
    pending_narratives: List[Dict[str, Any]] = field(default_factory=list)
    pending_practices: List[Dict[str, Any]] = field(default_factory=list)

    # Retrieval buffer
    active_queries: List[WisdomQueryInput] = field(default_factory=list)
    cached_results: Dict[str, WisdomRetrievalOutput] = field(default_factory=dict)
    cache_ttl_seconds: float = 300.0

    # Synthesis buffer
    pending_syntheses: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_cache: Dict[str, CrossCulturalSynthesisOutput] = field(default_factory=dict)

    # Sensitivity check queue
    pending_sensitivity_checks: List[str] = field(default_factory=list)
    flagged_content: List[Dict[str, Any]] = field(default_factory=list)

    # Buffer capacity
    max_pending_items: int = 1000
    max_cache_entries: int = 500

    def is_cache_valid(self, key: str) -> bool:
        """Check if a cached result is still valid."""
        if key not in self.cached_results:
            return False
        result = self.cached_results[key]
        age = time.time() - result.timestamp
        return age < self.cache_ttl_seconds
```

## Relationship Mappings

### Cross-Form Data Exchange

**Form29ToForm30Exchange**
```python
@dataclass
class Form29ToForm30Exchange:
    """Data exchange format between Folk Wisdom (29) and Animal Cognition (30)."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    direction: str = "bidirectional"

    # From Form 29 to Form 30
    indigenous_animal_observations: List[IndigenousAnimalWisdom] = field(default_factory=list)
    traditional_ecological_claims: List[Dict[str, Any]] = field(default_factory=list)
    cultural_significance_data: List[Dict[str, Any]] = field(default_factory=list)

    # From Form 30 to Form 29
    scientific_corroboration_results: List[Dict[str, Any]] = field(default_factory=list)
    species_profile_summaries: List[Dict[str, Any]] = field(default_factory=list)
    cognition_domain_mappings: List[Dict[str, Any]] = field(default_factory=list)

    # Exchange metadata
    species_ids_linked: List[str] = field(default_factory=list)
    teaching_ids_linked: List[str] = field(default_factory=list)
    exchange_quality: float = 0.0
    last_sync_timestamp: float = 0.0


@dataclass
class Form29ToForm28Exchange:
    """Data exchange format between Folk Wisdom (29) and Philosophy (28)."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # From Form 29 to Form 28
    folk_philosophical_insights: List[Dict[str, Any]] = field(default_factory=list)
    animistic_metaphysics: List[Dict[str, Any]] = field(default_factory=list)
    oral_epistemology_data: List[Dict[str, Any]] = field(default_factory=list)
    indigenous_ethics_data: List[Dict[str, Any]] = field(default_factory=list)

    # From Form 28 to Form 29
    philosophical_framework_mappings: List[Dict[str, Any]] = field(default_factory=list)
    ontological_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    epistemological_analyses: List[Dict[str, Any]] = field(default_factory=list)

    # Exchange metadata
    tradition_ids_linked: List[str] = field(default_factory=list)
    framework_ids_linked: List[str] = field(default_factory=list)
    exchange_quality: float = 0.0


@dataclass
class Form29ToForm27Exchange:
    """Data exchange format between Folk Wisdom (29) and Altered States (27)."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # From Form 29 to Form 27
    shamanic_practice_data: List[Dict[str, Any]] = field(default_factory=list)
    vision_quest_traditions: List[Dict[str, Any]] = field(default_factory=list)
    ceremonial_state_descriptions: List[Dict[str, Any]] = field(default_factory=list)

    # From Form 27 to Form 29
    altered_state_research_data: List[Dict[str, Any]] = field(default_factory=list)
    neurological_correlates: List[Dict[str, Any]] = field(default_factory=list)
    consciousness_state_mappings: List[Dict[str, Any]] = field(default_factory=dict)

    # Exchange metadata
    practice_ids_linked: List[str] = field(default_factory=list)
    state_ids_linked: List[str] = field(default_factory=list)
    exchange_quality: float = 0.0
```

### Configuration Structure

**FolkWisdomConfiguration**
```python
@dataclass
class FolkWisdomConfiguration:
    """Configuration parameters for the Folk Wisdom system."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)

    # Ethical settings
    default_permission_filter: List[str] = field(
        default_factory=lambda: ["freely_shared", "community_approved", "scholarly_published"]
    )
    sacred_content_filtering: bool = True
    attribution_required: bool = True
    sensitivity_check_enabled: bool = True

    # Retrieval settings
    default_max_results: int = 10
    embedding_similarity_threshold: float = 0.7
    cross_cultural_synthesis_enabled: bool = True
    minimum_maturity_for_synthesis: str = "developing"

    # Performance settings
    max_query_latency_ms: float = 200.0
    cache_ttl_seconds: float = 300.0
    max_concurrent_queries: int = 50
    embedding_batch_size: int = 32

    # Integration settings
    form_30_integration_enabled: bool = True
    form_28_integration_enabled: bool = True
    form_27_integration_enabled: bool = True
    cross_form_sync_interval_seconds: float = 60.0

    # Monitoring settings
    metrics_collection_interval: float = 5.0
    health_check_interval: float = 30.0
    log_retention_days: int = 90

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.embedding_similarity_threshold < 0.0 or self.embedding_similarity_threshold > 1.0:
            errors.append("embedding_similarity_threshold must be between 0.0 and 1.0")
        if self.max_query_latency_ms <= 0:
            errors.append("max_query_latency_ms must be positive")
        if self.max_concurrent_queries <= 0:
            errors.append("max_concurrent_queries must be positive")
        return errors
```

These data structures provide a comprehensive foundation for representing all aspects of folk and indigenous wisdom, from core teachings and oral traditions to animistic practices and cross-cultural synthesis. The structures prioritize cultural sensitivity, proper attribution, and respectful handling of sacred knowledge while enabling computational retrieval and cross-form integration.
