# API Specifications

## Overview

This document specifies the APIs for Form 28: Philosophical Consciousness. These interfaces enable querying the philosophical knowledge base, triggering research, performing cross-tradition synthesis, and integrating with other consciousness forms.

---

## Core Query API

### Concept Query

**Endpoint**: `philosophical_consciousness.query_concept`

**Purpose**: Retrieve philosophical concepts matching a query.

```python
async def query_concept(
    query: str,
    tradition_filter: Optional[List[PhilosophicalTradition]] = None,
    domain_filter: Optional[List[PhilosophicalDomain]] = None,
    max_results: int = 10,
    min_relevance: float = 0.5,
    include_related: bool = True,
    trigger_research: bool = False
) -> ConceptQueryResponse:
    """
    Query the philosophical concept index.

    Args:
        query: Natural language query or concept name
        tradition_filter: Limit to specific traditions
        domain_filter: Limit to specific domains
        max_results: Maximum concepts to return
        min_relevance: Minimum cosine similarity threshold
        include_related: Include related concepts in response
        trigger_research: If not found, trigger research agent

    Returns:
        ConceptQueryResponse with matching concepts and metadata
    """
```

**Response Schema**:
```python
@dataclass
class ConceptQueryResponse:
    query_id: str
    concepts: List[PhilosophicalConcept]
    relevance_scores: Dict[str, float]  # concept_id -> score
    related_concepts: List[PhilosophicalConcept]
    research_triggered: bool
    research_task_id: Optional[str]
    processing_time_ms: float
```

### Figure Query

**Endpoint**: `philosophical_consciousness.query_figure`

```python
async def query_figure(
    query: str,
    tradition_filter: Optional[List[PhilosophicalTradition]] = None,
    era_filter: Optional[str] = None,
    max_results: int = 10,
    include_works: bool = True,
    include_influences: bool = True
) -> FigureQueryResponse:
    """
    Query the philosopher/figure index.
    """
```

### Text Query

**Endpoint**: `philosophical_consciousness.query_text`

```python
async def query_text(
    query: str,
    tradition_filter: Optional[List[PhilosophicalTradition]] = None,
    author_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
    include_chunks: bool = False,
    chunk_limit: int = 5
) -> TextQueryResponse:
    """
    Query the philosophical text index.
    If include_chunks=True and text is indexed, return relevant chunks.
    """
```

### Semantic Search

**Endpoint**: `philosophical_consciousness.semantic_search`

```python
async def semantic_search(
    query: str,
    collections: List[str] = ["concepts", "figures", "texts", "arguments"],
    top_k: int = 10,
    min_similarity: float = 0.5,
    filters: Optional[Dict[str, Any]] = None
) -> SemanticSearchResponse:
    """
    Perform semantic search across philosophical knowledge base.

    Uses sentence-transformers embeddings for similarity matching.
    """
```

**Response Schema**:
```python
@dataclass
class SemanticSearchResult:
    entity_type: str  # "concept", "figure", "text", "argument"
    entity_id: str
    entity: Union[PhilosophicalConcept, PhilosophicalFigure, PhilosophicalText, PhilosophicalArgument]
    similarity_score: float
    matched_text: str

@dataclass
class SemanticSearchResponse:
    query_id: str
    results: List[SemanticSearchResult]
    total_candidates: int
    processing_time_ms: float
```

---

## Research Agent API

### Trigger Research

**Endpoint**: `philosophical_consciousness.trigger_research`

```python
async def trigger_research(
    query: str,
    sources: List[ResearchSource] = [
        ResearchSource.STANFORD_ENCYCLOPEDIA,
        ResearchSource.PHILPAPERS,
        ResearchSource.WEB_SEARCH
    ],
    priority: int = 5,
    max_depth: int = 2,
    target_traditions: Optional[List[PhilosophicalTradition]] = None,
    callback: Optional[Callable] = None
) -> ResearchTaskResponse:
    """
    Trigger autonomous research on a philosophical topic.

    Args:
        query: Topic or question to research
        sources: Which sources to consult
        priority: 1-10, higher = more urgent
        max_depth: How deep to follow related topics
        target_traditions: Focus on specific traditions
        callback: Optional callback when complete

    Returns:
        ResearchTaskResponse with task_id for tracking
    """
```

**Response Schema**:
```python
@dataclass
class ResearchTaskResponse:
    task_id: str
    status: str  # "queued", "in_progress"
    estimated_completion_seconds: Optional[int]
    sources_to_consult: List[str]
```

### Research Status

**Endpoint**: `philosophical_consciousness.get_research_status`

```python
async def get_research_status(
    task_id: str
) -> ResearchStatusResponse:
    """
    Get status of a research task.
    """
```

**Response Schema**:
```python
@dataclass
class ResearchStatusResponse:
    task_id: str
    status: str  # "queued", "in_progress", "completed", "failed"
    progress_percent: float
    concepts_discovered: int
    figures_discovered: int
    texts_discovered: int
    sources_consulted: List[str]
    error_message: Optional[str]
    results: Optional[ResearchResults]  # If completed
```

### Index Local Text

**Endpoint**: `philosophical_consciousness.index_text`

```python
async def index_text(
    file_path: str,
    metadata: TextMetadata,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> TextIndexResponse:
    """
    Index a local philosophical text file (PDF, TXT, MD).

    Extracts content, chunks it, generates embeddings,
    and adds to the searchable index.
    """
```

---

## Synthesis API

### Cross-Tradition Synthesis

**Endpoint**: `philosophical_consciousness.synthesize`

```python
async def synthesize(
    topic: str,
    traditions: List[PhilosophicalTradition],
    synthesis_depth: str = "moderate",  # "shallow", "moderate", "deep"
    preserve_nuance: bool = True,
    include_practical_guidance: bool = True
) -> SynthesisResponse:
    """
    Synthesize insights on a topic across multiple traditions.

    Args:
        topic: The philosophical question or concept
        traditions: Which traditions to synthesize
        synthesis_depth: How thoroughly to explore each tradition
        preserve_nuance: Maintain distinct positions vs. merge
        include_practical_guidance: Add wisdom/practical takeaways

    Returns:
        CrossTraditionSynthesis with convergent/divergent insights
    """
```

**Response Schema**:
```python
@dataclass
class SynthesisResponse:
    synthesis_id: str
    topic: str
    traditions_analyzed: List[PhilosophicalTradition]
    convergent_insights: List[str]
    divergent_positions: List[Dict[str, str]]  # tradition -> position
    complementary_aspects: List[str]
    synthesis_statement: str
    practical_guidance: List[str]
    wisdom_teachings: List[str]
    coherence_score: float
    fidelity_scores: Dict[str, float]  # tradition -> fidelity
    sources_used: List[str]
```

### Dialectical Analysis

**Endpoint**: `philosophical_consciousness.dialectical_analysis`

```python
async def dialectical_analysis(
    thesis: str,
    antithesis: str,
    context_tradition: Optional[PhilosophicalTradition] = None
) -> DialecticalResponse:
    """
    Perform Hegelian-style dialectical analysis.

    Given a thesis and antithesis, attempt synthesis
    that preserves valid aspects of both.
    """
```

---

## Integration APIs

### Form 10 (Memory) Integration

**Endpoint**: `philosophical_consciousness.store_philosophical_memory`

```python
async def store_philosophical_memory(
    concept_id: str,
    context: str,
    importance: float = 0.5
) -> MemoryStoreResponse:
    """
    Store a philosophical insight in long-term memory (Form 10).
    Uses shared embedding space for cross-form retrieval.
    """
```

**Endpoint**: `philosophical_consciousness.retrieve_philosophical_memories`

```python
async def retrieve_philosophical_memories(
    query: str,
    max_results: int = 5
) -> List[PhilosophicalMemory]:
    """
    Retrieve philosophical memories from Form 10.
    """
```

### Form 14 (Global Workspace) Integration

**Endpoint**: `philosophical_consciousness.broadcast_insight`

```python
async def broadcast_insight(
    insight: str,
    tradition: PhilosophicalTradition,
    relevance_context: str,
    priority: str = "normal"
) -> BroadcastResponse:
    """
    Broadcast a philosophical insight to the global workspace.
    Competes with other content for conscious access.
    """
```

### Form 27 (Non-Dual Interface) Integration

**Endpoint**: `philosophical_consciousness.integrate_with_meditation`

```python
async def integrate_with_meditation(
    philosophical_context: Dict[str, Any],
    processing_mode: ProcessingMode,
    mind_level: MindLevel
) -> MeditationIntegrationResponse:
    """
    Integrate philosophical processing with meditation-enhanced states.
    Applies appropriate processing based on mind level and mode.
    """
```

### Enlightened Engagement Integration

**Endpoint**: `philosophical_consciousness.get_wisdom_for_context`

```python
async def get_wisdom_for_context(
    context: EngagementContext,
    preferred_traditions: Optional[List[PhilosophicalTradition]] = None
) -> WisdomResponse:
    """
    Get philosophical wisdom appropriate for an engagement context.

    Considers recipient capacity, emotional state, and spiritual orientation.
    Returns tradition-appropriate wisdom teachings and practical guidance.
    """
```

**Response Schema**:
```python
@dataclass
class WisdomResponse:
    wisdom_aspects: List[WisdomAspect]
    teachings: List[str]
    practical_guidance: List[str]
    traditions_drawn_from: List[PhilosophicalTradition]
    skillful_means: Dict[str, Any]
```

---

## Message Bus Integration

### Message Types

```python
class PhilosophicalMessageType(Enum):
    """Message types for Form 28 communication"""

    # Queries
    PHILOSOPHICAL_QUERY = "philosophical_query"
    CONCEPT_LOOKUP = "concept_lookup"
    TRADITION_QUERY = "tradition_query"

    # Responses
    PHILOSOPHICAL_RESPONSE = "philosophical_response"
    WISDOM_RESPONSE = "wisdom_response"

    # Broadcasts
    WISDOM_BROADCAST = "wisdom_broadcast"
    INSIGHT_BROADCAST = "insight_broadcast"

    # Research
    RESEARCH_REQUEST = "research_request"
    RESEARCH_COMPLETE = "research_complete"

    # Integration
    SYNTHESIS_REQUEST = "synthesis_request"
    SYNTHESIS_COMPLETE = "synthesis_complete"
```

### Message Schemas

**Philosophical Query Message**:
```python
@dataclass
class PhilosophicalQueryMessage:
    message_type: PhilosophicalMessageType = PhilosophicalMessageType.PHILOSOPHICAL_QUERY
    query_id: str
    query_text: str
    requester_form: str
    filters: Dict[str, Any]
    options: Dict[str, Any]
    timestamp: datetime
    priority: int = 5
```

**Wisdom Broadcast Message**:
```python
@dataclass
class WisdomBroadcastMessage:
    message_type: PhilosophicalMessageType = PhilosophicalMessageType.WISDOM_BROADCAST
    broadcast_id: str
    wisdom_content: str
    tradition: PhilosophicalTradition
    relevance_context: str
    target_forms: List[str] = field(default_factory=list)  # Empty = all
    timestamp: datetime
    ttl_ms: int = 5000
```

### Channel Subscriptions

```python
# Form 28 subscribes to these channels
SUBSCRIBED_CHANNELS = [
    "philosophical_queries",      # Incoming queries from other forms
    "global_workspace",           # Global workspace broadcasts
    "engagement_context",         # Context for wisdom selection
    "meditation_state",           # Meditation state changes
]

# Form 28 publishes to these channels
PUBLISH_CHANNELS = [
    "philosophical_responses",    # Query responses
    "wisdom_broadcasts",          # Wisdom for global workspace
    "research_updates",           # Research task updates
]
```

---

## Maturity and Statistics API

### Get Maturity State

**Endpoint**: `philosophical_consciousness.get_maturity_state`

```python
async def get_maturity_state() -> PhilosophicalMaturityState:
    """
    Get current philosophical maturity state.
    """
```

### Get Statistics

**Endpoint**: `philosophical_consciousness.get_statistics`

```python
async def get_statistics() -> PhilosophicalStatistics:
    """
    Get comprehensive statistics about the philosophical knowledge base.
    """
```

**Response Schema**:
```python
@dataclass
class PhilosophicalStatistics:
    total_concepts: int
    total_figures: int
    total_texts: int
    total_arguments: int

    concepts_by_tradition: Dict[str, int]
    concepts_by_domain: Dict[str, int]
    figures_by_tradition: Dict[str, int]
    texts_by_tradition: Dict[str, int]

    average_concept_maturity: float
    tradition_depths: Dict[str, float]
    domain_coverages: Dict[str, float]

    research_tasks_completed: int
    syntheses_generated: int

    last_updated: datetime
```

### Identify Knowledge Gaps

**Endpoint**: `philosophical_consciousness.identify_gaps`

```python
async def identify_gaps(
    tradition: Optional[PhilosophicalTradition] = None,
    domain: Optional[PhilosophicalDomain] = None
) -> List[KnowledgeGap]:
    """
    Identify areas where knowledge is shallow or missing.
    """
```

**Response Schema**:
```python
@dataclass
class KnowledgeGap:
    gap_type: str  # "missing_concept", "shallow_tradition", "missing_figure"
    description: str
    tradition: Optional[PhilosophicalTradition]
    domain: Optional[PhilosophicalDomain]
    suggested_research_query: str
    priority: int
```

---

## Error Handling

### Error Codes

```python
class PhilosophicalErrorCode(Enum):
    """Error codes for Form 28 operations"""

    # Query Errors
    INVALID_QUERY = "PHIL_001"
    NO_RESULTS = "PHIL_002"
    INVALID_FILTER = "PHIL_003"

    # Research Errors
    RESEARCH_FAILED = "PHIL_010"
    SOURCE_UNAVAILABLE = "PHIL_011"
    RATE_LIMITED = "PHIL_012"

    # Synthesis Errors
    SYNTHESIS_FAILED = "PHIL_020"
    INCOMPATIBLE_TRADITIONS = "PHIL_021"
    INSUFFICIENT_DATA = "PHIL_022"

    # Integration Errors
    FORM_UNAVAILABLE = "PHIL_030"
    MESSAGE_TIMEOUT = "PHIL_031"

    # Index Errors
    INDEX_CORRUPTED = "PHIL_040"
    EMBEDDING_FAILED = "PHIL_041"
```

### Error Response

```python
@dataclass
class PhilosophicalError:
    error_code: PhilosophicalErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None
```
