# Research Agent Architecture

## Overview

The Philosophical Research Agent system enables continuous, autonomous expansion of philosophical knowledge. It operates as a coordinated set of agents that fetch, extract, validate, and integrate philosophical content from external sources.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │     Research Agent Coordinator       │
                    │  (Task Queue, Priority, Scheduling)  │
                    └─────────────────┬───────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
    │ SEP Agent   │           │ PhilPapers  │           │ Web Search  │
    │             │           │   Agent     │           │   Agent     │
    └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │        Content Extractor            │
                    │ (Concepts, Figures, Texts, Arguments)│
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         Validation Layer            │
                    │   (Accuracy, Consistency, Nuance)   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Knowledge Integration          │
                    │  (Embedding, Indexing, Graph Update) │
                    └─────────────────────────────────────┘
```

---

## Research Sources

### Stanford Encyclopedia of Philosophy (SEP)

**Priority**: Primary academic source
**URL**: https://plato.stanford.edu/

**Capabilities**:
- Fetch full article content
- Parse structured sections
- Extract related entries
- Retrieve bibliography

**Rate Limiting**: 2 seconds between requests

**Extraction Strategy**:
1. Parse HTML structure
2. Extract preamble for definition
3. Extract section headings for structure
4. Parse related entries for connections
5. Extract bibliography for texts

### PhilPapers

**Priority**: Primary bibliography source
**URL**: https://philpapers.org/

**Capabilities**:
- Search by topic, author, category
- Retrieve paper metadata
- Access abstracts
- Get citation counts

**Rate Limiting**: 1 second between requests

**Extraction Strategy**:
1. Search API for relevant papers
2. Extract metadata (author, title, year)
3. Parse abstracts for concepts
4. Track citations for importance

### Internet Encyclopedia of Philosophy (IEP)

**Priority**: Secondary reference
**URL**: https://iep.utm.edu/

**Capabilities**:
- Accessible explanations
- Complementary to SEP
- Different author perspectives

**Rate Limiting**: 2 seconds between requests

### Local Corpus

**Priority**: User-provided texts
**Path**: Configurable

**Capabilities**:
- Index PDF, TXT, MD files
- Chunk and embed content
- Full-text search
- Concept extraction

### Web Search

**Priority**: Tertiary/discovery
**Method**: General philosophical web search

**Capabilities**:
- Discover new sources
- Find recent discussions
- Identify emerging topics

---

## Task Management

### Task Lifecycle

```
Created → Queued → In Progress → Completed/Failed
```

### Task Priority (1-10)

| Priority | Description | Trigger Type |
|----------|-------------|--------------|
| 10 | Emergency gap fill | System critical |
| 8-9 | User explicit request | User command |
| 6-7 | Query miss | Automatic |
| 4-5 | Gap detection | Proactive |
| 1-3 | Scheduled expansion | Background |

### Concurrency

- Maximum concurrent tasks: 3 (configurable)
- Per-source parallelism: 1 (respect rate limits)
- Background tasks yield to higher priority

### Task Schema

```python
@dataclass
class ResearchTask:
    task_id: str
    query: str
    sources: List[ResearchSource]
    priority: int
    max_depth: int
    status: ResearchStatus
    triggered_by: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    progress: float  # 0.0-100.0
```

---

## Content Extraction

### Concept Extraction Pipeline

1. **Text Preprocessing**
   - Clean HTML/formatting
   - Segment into sections
   - Identify definition patterns

2. **Entity Recognition**
   - Philosophical terms
   - Philosopher names
   - Text titles
   - Argument patterns

3. **Relationship Extraction**
   - Related concepts
   - Influences
   - Oppositions
   - Prerequisites

4. **Metadata Enrichment**
   - Tradition classification
   - Domain classification
   - Time period
   - Key figures

### Extraction Patterns

**Definition Patterns**:
```regex
"X is (defined as|understood as|the view that)..."
"The concept of X refers to..."
"By X we mean..."
```

**Philosopher Patterns**:
```regex
"(Plato|Aristotle|Kant|...) (argued|believed|claimed)..."
"According to (philosopher name)..."
```

**Argument Patterns**:
```regex
"(Therefore|Thus|Hence|It follows that)..."
"If P then Q; P; therefore Q"
"Premise 1:... Premise 2:... Conclusion:..."
```

---

## Validation Layer

### Accuracy Checks

1. **Source Verification**
   - Cross-reference multiple sources
   - Check against SEP as gold standard
   - Flag conflicting information

2. **Terminology Validation**
   - Verify technical terms used correctly
   - Check tradition-specific usage
   - Flag anachronisms

3. **Logical Consistency**
   - Check internal consistency
   - Verify claimed relationships
   - Validate argument structures

### Quality Scoring

```python
def calculate_quality_score(extracted: Dict) -> float:
    source_score = score_source_reliability(extracted["source"])
    completeness_score = score_completeness(extracted)
    consistency_score = score_consistency(extracted)

    return (
        source_score * 0.4 +
        completeness_score * 0.3 +
        consistency_score * 0.3
    )
```

### Rejection Criteria

- Quality score < 0.5
- Contradicts established knowledge
- Source unreliable
- Obvious errors detected

---

## Knowledge Integration

### Embedding Generation

```python
async def generate_embedding(concept: PhilosophicalConcept) -> List[float]:
    text = concept.to_embedding_text()
    embedding = await embedding_model.encode(text)
    return normalize(embedding)
```

### Index Update

1. Check for existing entry
2. If exists: merge or update
3. If new: add to index
4. Update embeddings
5. Update knowledge graph

### Graph Update

```python
async def update_graph(concept: PhilosophicalConcept):
    # Add node
    add_node(concept.concept_id, type="concept")

    # Add relationships
    for related in concept.related_concepts:
        add_edge(concept.concept_id, related, "RELATED_TO")

    for opposed in concept.opposed_concepts:
        add_edge(concept.concept_id, opposed, "OPPOSED_TO")

    for figure in concept.key_figures:
        add_edge(figure, concept.concept_id, "ORIGINATED")
```

---

## Scheduling

### Background Research

- Run during low activity periods
- Priority: 1-3
- Focus: Gap filling, depth expansion

### Triggered Research

- Immediate on query miss (if enabled)
- Priority: 6-7
- Focus: Specific topic

### Scheduled Expansion

```python
EXPANSION_SCHEDULE = {
    "daily": [
        {"query": "recent_philosophy_of_mind", "depth": 1},
    ],
    "weekly": [
        {"query": "expand_shallow_traditions", "depth": 2},
    ],
    "monthly": [
        {"query": "comprehensive_gap_analysis", "depth": 3},
    ]
}
```

---

## Monitoring

### Metrics Tracked

- Tasks completed/failed
- Concepts discovered
- Sources consulted
- Processing time
- Quality scores

### Health Indicators

- Source availability
- Rate limit status
- Queue depth
- Error rate

### Logging

```python
logger.info(f"Research task {task_id} completed: "
           f"{concepts_discovered} concepts, "
           f"{figures_discovered} figures")
```
