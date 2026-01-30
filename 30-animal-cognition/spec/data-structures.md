# Form 30: Animal Cognition & Ethology - Data Structures

## Overview

This document defines the core data structures for Form 30: Animal Cognition & Ethology. These structures represent species cognition profiles, behavioral insights, consciousness indicators, cross-species synthesis, and integration with indigenous knowledge systems. The models support encoding, retrieval, and comparison of cognitive capabilities across the animal kingdom, from invertebrates to great apes.

## Core Data Models

### Species Cognition Profile

**SpeciesCognitionProfile**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import time
import uuid


@dataclass
class SpeciesCognitionProfile:
    """Comprehensive cognitive profile for a species."""

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Species identification
    species_id: str = ""
    common_name: str = ""
    scientific_name: str = ""
    taxonomic_group: 'TaxonomicGroup' = None
    taxonomic_order: str = ""
    taxonomic_family: str = ""

    # Conservation and context
    conservation_status: str = ""  # IUCN classification
    habitat_description: str = ""
    brain_weight_grams: Optional[float] = None
    encephalization_quotient: Optional[float] = None
    social_structure: str = ""

    # Cognition domain scores (0.0 to 1.0 evidence strength)
    cognition_scores: Dict['CognitionDomain', float] = field(default_factory=dict)
    cognition_evidence_counts: Dict['CognitionDomain', int] = field(default_factory=dict)

    # Consciousness indicators
    consciousness_indicators: Dict['ConsciousnessIndicator', 'IndicatorEvidence'] = field(
        default_factory=dict
    )
    overall_consciousness_score: float = 0.0

    # Notable individuals and research
    notable_individuals: List['NotableIndividual'] = field(default_factory=list)
    key_studies: List['KeyStudy'] = field(default_factory=list)
    key_researchers: List[str] = field(default_factory=list)

    # Indigenous knowledge links (Form 29)
    indigenous_knowledge_ids: List[str] = field(default_factory=list)
    traditional_observations: List[str] = field(default_factory=list)

    # Maturity and quality
    maturity_level: 'ProfileMaturity' = None
    last_updated: float = field(default_factory=time.time)
    data_completeness: float = 0.0

    # Embedding for similarity search
    embedding_vector: Optional[List[float]] = None

    def calculate_overall_consciousness_score(self) -> float:
        """Calculate composite consciousness score from indicators."""
        if not self.consciousness_indicators:
            return 0.0
        total = sum(
            indicator.strength for indicator in self.consciousness_indicators.values()
        )
        return total / len(self.consciousness_indicators)

    def get_top_cognitive_abilities(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N cognitive domains by evidence strength."""
        sorted_domains = sorted(
            self.cognition_scores.items(),
            key=lambda x: x[1] if isinstance(x[1], float) else 0.0,
            reverse=True
        )
        return sorted_domains[:n]
```

### Behavioral Insight Structures

**AnimalBehaviorInsight**
```python
@dataclass
class AnimalBehaviorInsight:
    """A documented behavioral observation or experimental finding."""

    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Species reference
    species_id: str = ""
    species_common_name: str = ""
    species_scientific_name: str = ""

    # Behavior description
    title: str = ""
    description: str = ""
    behavior_type: 'BehaviorType' = None
    cognition_domain: 'CognitionDomain' = None

    # Evidence details
    evidence_type: 'EvidenceType' = None
    evidence_strength: 'EvidenceStrength' = None
    research_paradigm: str = ""
    sample_size: Optional[int] = None
    replication_status: 'ReplicationStatus' = None

    # Source information
    source_citation: str = ""
    publication_year: Optional[int] = None
    researchers: List[str] = field(default_factory=list)
    journal: str = ""

    # Context
    study_context: 'StudyContext' = None
    wild_vs_captive: str = ""  # "wild", "captive", "semi-wild", "both"
    ecological_context: str = ""

    # Notable individuals
    individual_subjects: List['NotableIndividual'] = field(default_factory=list)

    # Cross-references
    related_insight_ids: List[str] = field(default_factory=list)
    form_29_wisdom_ids: List[str] = field(default_factory=list)

    # Quality assessment
    methodological_quality: float = 0.0
    ecological_validity: float = 0.0
    anthropomorphism_risk: 'AnthropomorphismRisk' = None

    # Embedding
    embedding_vector: Optional[List[float]] = None

    def calculate_evidence_score(self) -> float:
        """Calculate composite evidence score."""
        strength_map = {
            EvidenceStrength.ANECDOTAL: 0.2,
            EvidenceStrength.OBSERVATIONAL: 0.4,
            EvidenceStrength.EXPERIMENTAL: 0.6,
            EvidenceStrength.REPLICATED: 0.8,
            EvidenceStrength.META_ANALYZED: 1.0
        }
        base = strength_map.get(self.evidence_strength, 0.0)
        quality_modifier = self.methodological_quality * 0.3
        return min(1.0, base + quality_modifier)
```

### Consciousness Assessment Structures

**IndicatorEvidence**
```python
@dataclass
class IndicatorEvidence:
    """Evidence for a specific consciousness indicator in a species."""

    indicator: 'ConsciousnessIndicator' = None
    species_id: str = ""

    # Evidence assessment
    strength: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    evidence_count: int = 0
    evidence_type_distribution: Dict['EvidenceType', int] = field(default_factory=dict)

    # Key evidence
    strongest_evidence: str = ""
    key_study_ids: List[str] = field(default_factory=list)
    counterevidence: List[str] = field(default_factory=list)

    # Theoretical context
    relevant_theories: List[str] = field(default_factory=list)
    theory_predictions: Dict[str, str] = field(default_factory=dict)
    open_questions: List[str] = field(default_factory=list)

    # Assessment metadata
    last_assessed: float = field(default_factory=time.time)
    assessor_notes: str = ""


@dataclass
class ConsciousnessAssessment:
    """Complete consciousness assessment for a species."""

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    species_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Indicator evidence
    indicator_evidence: Dict['ConsciousnessIndicator', IndicatorEvidence] = field(
        default_factory=dict
    )

    # Overall assessment
    overall_score: float = 0.0
    confidence_level: float = 0.0
    assessment_summary: str = ""

    # Theoretical lens assessments
    theory_assessments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g., {"global_workspace": {"score": 0.7, "rationale": "..."}}

    # Comparative context
    comparative_ranking: Optional[int] = None
    similar_species: List[str] = field(default_factory=list)
    evolutionary_context: str = ""

    # Ethical implications
    moral_status_implications: str = ""
    welfare_considerations: List[str] = field(default_factory=list)

    # Uncertainty and limitations
    key_uncertainties: List[str] = field(default_factory=list)
    methodological_limitations: List[str] = field(default_factory=list)
    anthropomorphism_notes: str = ""
```

### Notable Individual Structure

**NotableIndividual**
```python
@dataclass
class NotableIndividual:
    """A notable individual animal known for cognitive abilities."""

    individual_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identification
    name: str = ""
    species_id: str = ""
    species_common_name: str = ""
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    sex: str = ""  # "male", "female", "unknown"

    # Context
    facility: str = ""  # zoo, research center, sanctuary
    wild_born: bool = False
    study_history: str = ""

    # Notable achievements
    achievements: List[str] = field(default_factory=list)
    cognitive_milestones: List[Dict[str, Any]] = field(default_factory=list)
    communication_abilities: Dict[str, Any] = field(default_factory=dict)
    tool_use_abilities: Dict[str, Any] = field(default_factory=dict)

    # Associated research
    key_study_ids: List[str] = field(default_factory=list)
    researchers: List[str] = field(default_factory=list)

    # Cultural significance
    public_recognition: str = ""
    media_appearances: List[str] = field(default_factory=list)
    conservation_impact: str = ""
```

### Cross-Species Synthesis Structure

**CrossSpeciesSynthesis**
```python
@dataclass
class CrossSpeciesSynthesis:
    """Comparison and synthesis across multiple species."""

    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Synthesis parameters
    species_compared: List[str] = field(default_factory=list)
    cognition_domain: Optional['CognitionDomain'] = None
    synthesis_type: 'SynthesisType' = None

    # Comparison data
    comparison_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ranked_species: List[Tuple[str, float]] = field(default_factory=list)
    capability_clusters: List[Dict[str, Any]] = field(default_factory=list)

    # Evolutionary analysis
    convergent_evolution_examples: List[Dict[str, Any]] = field(default_factory=list)
    shared_ancestry_patterns: List[Dict[str, Any]] = field(default_factory=list)
    independent_origins: List[Dict[str, Any]] = field(default_factory=list)
    phylogenetic_context: str = ""

    # Key findings
    key_findings: List[str] = field(default_factory=list)
    patterns_identified: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)

    # Methodological context
    comparison_limitations: List[str] = field(default_factory=list)
    testing_paradigm_differences: List[str] = field(default_factory=list)
    anthropocentric_biases: List[str] = field(default_factory=list)

    # Quality metrics
    species_coverage_quality: float = 0.0
    evidence_consistency: float = 0.0
    synthesis_confidence: float = 0.0
```

### Key Study Structure

**KeyStudy**
```python
@dataclass
class KeyStudy:
    """A key research study in animal cognition."""

    study_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Citation
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: Optional[str] = None

    # Study details
    species_studied: List[str] = field(default_factory=list)
    cognition_domains: List['CognitionDomain'] = field(default_factory=list)
    methodology: str = ""
    sample_size: Optional[int] = None
    study_context: 'StudyContext' = None

    # Findings
    key_findings: List[str] = field(default_factory=list)
    evidence_strength: 'EvidenceStrength' = None
    replication_status: 'ReplicationStatus' = None
    significance: str = ""

    # Impact
    citation_count: Optional[int] = None
    landmark_status: bool = False
    paradigm_shifting: bool = False
    follow_up_studies: List[str] = field(default_factory=list)
```

## Enumeration Types

### Taxonomic Enumerations

```python
class TaxonomicGroup(Enum):
    """Major taxonomic groups covered by Form 30."""
    GREAT_APES = "great_apes"
    OTHER_PRIMATES = "other_primates"
    CETACEANS = "cetaceans"
    ELEPHANTS = "elephants"
    CANIDS = "canids"
    FELIDS = "felids"
    MARINE_MAMMALS = "marine_mammals"
    RODENTS = "rodents"
    UNGULATES = "ungulates"
    OTHER_MAMMALS = "other_mammals"
    CORVIDS = "corvids"
    PARROTS = "parrots"
    RAPTORS = "raptors"
    SONGBIRDS = "songbirds"
    WATERFOWL = "waterfowl"
    OTHER_BIRDS = "other_birds"
    REPTILES = "reptiles"
    AMPHIBIANS = "amphibians"
    FISH = "fish"
    CEPHALOPODS = "cephalopods"
    SOCIAL_INSECTS = "social_insects"
    ARACHNIDS = "arachnids"
    OTHER_INVERTEBRATES = "other_invertebrates"
```

### Cognition Domain Enumerations

```python
class CognitionDomain(Enum):
    """Cognitive domains tracked per species."""
    # Memory systems
    EPISODIC_MEMORY = "episodic_memory"
    WORKING_MEMORY = "working_memory"
    SPATIAL_COGNITION = "spatial_cognition"
    LONG_TERM_MEMORY = "long_term_memory"

    # Learning and problem solving
    SOCIAL_LEARNING = "social_learning"
    TOOL_USE = "tool_use"
    TOOL_MANUFACTURE = "tool_manufacture"
    CAUSAL_REASONING = "causal_reasoning"
    PROBLEM_SOLVING = "problem_solving"
    PLANNING = "planning"
    INSIGHT = "insight"

    # Social cognition
    THEORY_OF_MIND = "theory_of_mind"
    COOPERATION = "cooperation"
    RECIPROCITY = "reciprocity"
    DECEPTION = "deception"
    EMPATHY = "empathy"
    EMOTIONAL_CONTAGION = "emotional_contagion"
    FAIRNESS = "fairness"

    # Self-awareness
    MIRROR_SELF_RECOGNITION = "mirror_self_recognition"
    METACOGNITION = "metacognition"
    SELF_AGENCY = "self_agency"

    # Communication
    REFERENTIAL_SIGNALS = "referential_signals"
    SYNTAX = "syntax"
    LANGUAGE_COMPREHENSION = "language_comprehension"
    VOCAL_LEARNING = "vocal_learning"

    # Emotional processing
    EMOTIONAL_EXPRESSION = "emotional_expression"
    GRIEF_MOURNING = "grief_mourning"
    PLAY_BEHAVIOR = "play_behavior"
    JOY_DISPLAY = "joy_display"


class ConsciousnessIndicator(Enum):
    """Types of consciousness evidence tracked."""
    BEHAVIORAL = "behavioral"
    NEUROANATOMICAL = "neuroanatomical"
    NEUROPHYSIOLOGICAL = "neurophysiological"
    PHARMACOLOGICAL = "pharmacological"
    SELF_REPORT_PROXY = "self_report_proxy"
    INDIGENOUS_OBSERVATION = "indigenous_observation"
```

### Evidence and Quality Enumerations

```python
class EvidenceType(Enum):
    """Types of evidence for cognitive claims."""
    ANECDOTAL_REPORT = "anecdotal_report"
    NATURALISTIC_OBSERVATION = "naturalistic_observation"
    FIELD_EXPERIMENT = "field_experiment"
    LAB_EXPERIMENT = "lab_experiment"
    LONGITUDINAL_STUDY = "longitudinal_study"
    COMPARATIVE_STUDY = "comparative_study"
    NEUROIMAGING = "neuroimaging"
    PHARMACOLOGICAL_STUDY = "pharmacological_study"
    META_ANALYSIS = "meta_analysis"
    INDIGENOUS_KNOWLEDGE = "indigenous_knowledge"


class EvidenceStrength(Enum):
    """Strength of evidence for a finding."""
    ANECDOTAL = "anecdotal"
    OBSERVATIONAL = "observational"
    EXPERIMENTAL = "experimental"
    REPLICATED = "replicated"
    META_ANALYZED = "meta_analyzed"


class ReplicationStatus(Enum):
    """Replication status of findings."""
    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED_FAILED = "attempted_failed"
    PARTIALLY_REPLICATED = "partially_replicated"
    SUCCESSFULLY_REPLICATED = "successfully_replicated"
    MULTI_LAB_REPLICATED = "multi_lab_replicated"


class AnthropomorphismRisk(Enum):
    """Risk level of anthropomorphic interpretation."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class StudyContext(Enum):
    """Context in which a study was conducted."""
    WILD = "wild"
    CAPTIVE = "captive"
    SEMI_WILD = "semi_wild"
    SANCTUARY = "sanctuary"
    LABORATORY = "laboratory"
    FIELD_STATION = "field_station"


class BehaviorType(Enum):
    """Type of animal behavior documented."""
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    COMMUNICATIVE = "communicative"
    TOOL_RELATED = "tool_related"
    SELF_DIRECTED = "self_directed"
    EMOTIONAL = "emotional"
    CULTURAL = "cultural"
    PLAYFUL = "playful"


class ProfileMaturity(Enum):
    """Maturity level of a species cognition profile."""
    STUB = "stub"
    BASIC = "basic"
    DEVELOPING = "developing"
    COMPREHENSIVE = "comprehensive"
    AUTHORITATIVE = "authoritative"


class SynthesisType(Enum):
    """Type of cross-species synthesis."""
    DOMAIN_FOCUSED = "domain_focused"
    SPECIES_FOCUSED = "species_focused"
    TAXONOMIC_GROUP = "taxonomic_group"
    EVOLUTIONARY = "evolutionary"
    CONSCIOUSNESS_ASSESSMENT = "consciousness_assessment"
    COMPREHENSIVE = "comprehensive"
```

## Input/Output Structures

### Input Structures

**SpeciesQueryInput**
```python
@dataclass
class SpeciesQueryInput:
    """Input structure for querying species cognition profiles."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Query type
    query_type: str = ""  # "species", "domain", "consciousness", "comparison"

    # Species filters
    species_id: Optional[str] = None
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    taxonomic_group: Optional['TaxonomicGroup'] = None

    # Domain filters
    cognition_domains: Optional[List['CognitionDomain']] = None
    minimum_evidence_strength: Optional['EvidenceStrength'] = None
    consciousness_indicators: Optional[List['ConsciousnessIndicator']] = None

    # Comparison parameters
    compare_species: Optional[List[str]] = None
    comparison_domain: Optional['CognitionDomain'] = None

    # Response parameters
    include_indigenous_knowledge: bool = True
    include_consciousness_assessment: bool = True
    include_notable_individuals: bool = True
    format_type: str = "standard"  # "brief", "standard", "detailed", "comprehensive"
    max_results: int = 10

    # Context
    query_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightIngestionInput:
    """Input structure for adding new behavioral insights."""

    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Species identification
    species_id: str = ""
    common_name: str = ""
    scientific_name: str = ""

    # Insight content
    title: str = ""
    description: str = ""
    cognition_domain: 'CognitionDomain' = None
    behavior_type: 'BehaviorType' = None

    # Evidence
    evidence_type: 'EvidenceType' = None
    evidence_strength: 'EvidenceStrength' = None
    source_citation: str = ""
    researchers: List[str] = field(default_factory=list)

    # Context
    study_context: 'StudyContext' = None
    sample_size: Optional[int] = None

    # Quality flags
    peer_reviewed: bool = False
    replication_status: 'ReplicationStatus' = None

    # Processing directives
    update_profile: bool = True
    generate_embedding: bool = True
    check_for_duplicates: bool = True
```

### Output Structures

**SpeciesProfileOutput**
```python
@dataclass
class SpeciesProfileOutput:
    """Output structure for species profile query results."""

    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Profile data
    profiles: List[SpeciesCognitionProfile] = field(default_factory=list)

    # Associated insights
    top_insights: Dict[str, List[AnimalBehaviorInsight]] = field(default_factory=dict)

    # Consciousness assessments
    consciousness_assessments: Dict[str, ConsciousnessAssessment] = field(default_factory=dict)

    # Indigenous perspectives
    indigenous_knowledge: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Quality metrics
    result_confidence: float = 0.0
    coverage_assessment: Dict[str, float] = field(default_factory=dict)
    data_recency: Dict[str, float] = field(default_factory=dict)

    # Processing metadata
    processing_time_ms: float = 0.0
    total_candidates_evaluated: int = 0
    sources_consulted: List[str] = field(default_factory=list)


@dataclass
class CrossSpeciesComparisonOutput:
    """Output structure for cross-species comparison queries."""

    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Comparison data
    synthesis: CrossSpeciesSynthesis = field(default_factory=CrossSpeciesSynthesis)

    # Visual comparison aids
    comparison_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    capability_rankings: List[Tuple[str, float]] = field(default_factory=list)

    # Evolutionary context
    phylogenetic_tree_data: Optional[Dict[str, Any]] = None
    convergence_highlights: List[str] = field(default_factory=list)

    # Quality metrics
    comparison_validity: float = 0.0
    methodological_notes: List[str] = field(default_factory=list)

    # Processing metadata
    processing_time_ms: float = 0.0
```

## Internal State Structures

### System State

**AnimalCognitionSystemState**
```python
@dataclass
class AnimalCognitionSystemState:
    """Complete system state for the Animal Cognition consciousness form."""

    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Knowledge base state
    total_species_profiles: int = 0
    total_behavioral_insights: int = 0
    total_key_studies: int = 0
    total_notable_individuals: int = 0

    # Coverage metrics
    taxonomic_coverage: Dict[str, int] = field(default_factory=dict)
    domain_coverage: Dict[str, int] = field(default_factory=dict)
    evidence_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    average_profile_maturity: float = 0.0
    average_evidence_strength: float = 0.0
    replication_rate: float = 0.0
    indigenous_integration_rate: float = 0.0

    # Cross-form links
    form_29_links: int = 0  # Indigenous animal wisdom links
    form_28_links: int = 0  # Philosophy of mind links
    form_11_links: int = 0  # Meta-consciousness links

    # Processing state
    active_queries: int = 0
    pending_ingestions: int = 0
    active_syntheses: int = 0

    # Health
    system_health: float = 1.0
    error_count: int = 0
    last_health_check: float = field(default_factory=time.time)

    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate system coverage report."""
        return {
            "total_species": self.total_species_profiles,
            "total_insights": self.total_behavioral_insights,
            "taxonomic_groups_covered": len(self.taxonomic_coverage),
            "cognition_domains_covered": len(self.domain_coverage),
            "average_maturity": self.average_profile_maturity,
            "indigenous_integration": self.indigenous_integration_rate,
            "cross_form_links": self.form_29_links + self.form_28_links + self.form_11_links
        }
```

### Processing Buffers

**CognitionProcessingBuffer**
```python
@dataclass
class CognitionProcessingBuffer:
    """Buffer for in-progress animal cognition processing operations."""

    buffer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Query buffers
    active_species_queries: List[SpeciesQueryInput] = field(default_factory=list)
    active_comparison_queries: List[Dict[str, Any]] = field(default_factory=list)
    cached_profiles: Dict[str, SpeciesProfileOutput] = field(default_factory=dict)
    cache_ttl_seconds: float = 600.0

    # Ingestion buffers
    pending_insights: List[InsightIngestionInput] = field(default_factory=list)
    pending_profile_updates: List[Dict[str, Any]] = field(default_factory=list)

    # Synthesis buffers
    pending_syntheses: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_cache: Dict[str, CrossSpeciesComparisonOutput] = field(default_factory=dict)

    # Species resolution cache
    name_resolution_cache: Dict[str, str] = field(default_factory=dict)

    # Buffer limits
    max_active_queries: int = 100
    max_pending_ingestions: int = 500
    max_cache_entries: int = 1000
```

## Relationship Mappings

### Cross-Form Data Exchange

**Form30ToForm29Exchange**
```python
@dataclass
class Form30ToForm29Exchange:
    """Data exchange format between Animal Cognition (30) and Folk Wisdom (29)."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # From Form 30 to Form 29
    species_profile_summaries: List[Dict[str, Any]] = field(default_factory=list)
    scientific_corroboration_of_traditional_claims: List[Dict[str, Any]] = field(
        default_factory=list
    )
    cognition_domain_data: List[Dict[str, Any]] = field(default_factory=list)

    # From Form 29 to Form 30
    indigenous_animal_observations: List[Dict[str, Any]] = field(default_factory=list)
    traditional_ecological_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    cultural_significance_mappings: List[Dict[str, Any]] = field(default_factory=list)

    # Link metadata
    species_ids_linked: List[str] = field(default_factory=list)
    wisdom_ids_linked: List[str] = field(default_factory=list)
    corroboration_count: int = 0
    novel_insight_count: int = 0
    exchange_quality: float = 0.0


@dataclass
class Form30ToForm28Exchange:
    """Data exchange format between Animal Cognition (30) and Philosophy (28)."""

    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # From Form 30 to Form 28
    consciousness_evidence_summaries: List[Dict[str, Any]] = field(default_factory=list)
    moral_status_data: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_capacity_rankings: List[Dict[str, Any]] = field(default_factory=list)

    # From Form 28 to Form 30
    consciousness_theory_applications: List[Dict[str, Any]] = field(default_factory=list)
    ethical_framework_analyses: List[Dict[str, Any]] = field(default_factory=list)
    philosophical_debate_context: List[Dict[str, Any]] = field(default_factory=list)

    # Link metadata
    species_ids: List[str] = field(default_factory=list)
    theory_ids: List[str] = field(default_factory=list)
    exchange_quality: float = 0.0
```

### Configuration Structure

**AnimalCognitionConfiguration**
```python
@dataclass
class AnimalCognitionConfiguration:
    """Configuration parameters for the Animal Cognition system."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)

    # Query settings
    default_max_results: int = 10
    species_resolution_confidence_threshold: float = 0.8
    evidence_minimum_strength: str = "observational"
    embedding_similarity_threshold: float = 0.7

    # Profile settings
    auto_update_profiles_on_insight: bool = True
    consciousness_score_recalculation: bool = True
    maturity_recalculation_on_update: bool = True

    # Integration settings
    form_29_integration_enabled: bool = True
    form_28_integration_enabled: bool = True
    form_11_integration_enabled: bool = True
    cross_form_sync_interval_seconds: float = 120.0

    # Quality settings
    anthropomorphism_warnings_enabled: bool = True
    methodological_quality_minimum: float = 0.3
    require_peer_reviewed_for_profile_update: bool = False

    # Performance settings
    max_query_latency_ms: float = 200.0
    max_synthesis_latency_ms: float = 1000.0
    cache_ttl_seconds: float = 600.0
    max_concurrent_queries: int = 100

    # Monitoring settings
    metrics_collection_interval: float = 5.0
    health_check_interval: float = 30.0
    log_retention_days: int = 60

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if not 0.0 <= self.species_resolution_confidence_threshold <= 1.0:
            errors.append("species_resolution_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.embedding_similarity_threshold <= 1.0:
            errors.append("embedding_similarity_threshold must be between 0.0 and 1.0")
        if self.max_query_latency_ms <= 0:
            errors.append("max_query_latency_ms must be positive")
        return errors
```

These data structures provide a comprehensive foundation for representing animal cognition knowledge, from individual species profiles to cross-species synthesis and consciousness assessment. The structures support integration with indigenous knowledge systems (Form 29) and philosophical frameworks (Form 28) while maintaining rigorous evidence tracking and quality metrics appropriate to comparative psychology and cognitive ethology.
