# Architecture Design

## Overview

This document describes the architectural design of Form 28 (Philosophical Consciousness), its components, design rationale, and integration patterns with the broader consciousness system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Form 28: Philosophical Consciousness                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Interface Layer                                │    │
│  │  ┌───────────────────────┐  ┌───────────────────────────────┐  │    │
│  │  │ PhilosophicalConsci-  │  │ Research Agent Coordinator    │  │    │
│  │  │ ousnessInterface      │  │ (Autonomous Knowledge Fetch)  │  │    │
│  │  └───────────┬───────────┘  └───────────────┬───────────────┘  │    │
│  │              │                              │                    │    │
│  │  ┌───────────▼───────────────────────────────▼───────────────┐  │    │
│  │  │           Philosophical Reasoning Engine                   │  │    │
│  │  │   (Argument Analysis, Dialectical Synthesis, Inquiry)      │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                     Knowledge Layer                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │ Concept     │  │ Figure      │  │ Text        │               │  │
│  │  │ Index       │  │ Index       │  │ Index       │               │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │  │
│  │         │                │                │                        │  │
│  │  ┌──────▼────────────────▼────────────────▼──────────────────┐   │  │
│  │  │              Knowledge Graph                               │   │  │
│  │  │    (Nodes: Concepts, Figures, Texts, Arguments)            │   │  │
│  │  │    (Edges: Related, Opposed, Influenced, Authored)         │   │  │
│  │  └───────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                     Embedding Layer                                │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Vector Store (HNSW Index)                                  │  │  │
│  │  │  - phil_concepts (768-dim embeddings)                       │  │  │
│  │  │  - phil_figures (768-dim embeddings)                        │  │  │
│  │  │  - phil_texts (768-dim embeddings)                          │  │  │
│  │  │  - phil_arguments (768-dim embeddings)                      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
  ┌───────────────┐          ┌────────────────┐          ┌────────────────┐
  │   Form 10     │          │    Form 27     │          │   Form 14      │
  │   Memory      │          │   Non-Dual     │          │   Global       │
  │               │          │   Interface    │          │   Workspace    │
  └───────────────┘          └────────────────┘          └────────────────┘
```

---

## Component Architecture

### 1. Interface Layer

#### PhilosophicalConsciousnessInterface

**Responsibility**: Primary entry point for all philosophical queries and operations

**Key Methods**:
- `query_concept()` - Retrieve philosophical concepts with RAG
- `synthesize_across_traditions()` - Cross-tradition synthesis
- `get_wisdom_for_context()` - Context-appropriate wisdom selection
- `research_topic()` - Trigger research agent

**Design Pattern**: Facade pattern providing unified access to subsystems

#### Research Agent Coordinator

**Responsibility**: Autonomous expansion of philosophical knowledge

**Key Features**:
- Task queue with priority scheduling
- Multi-source fetching (SEP, PhilPapers, web)
- Rate limiting and retry logic
- Quality validation before integration

**Design Pattern**: Coordinator/Mediator managing multiple research agents

#### Philosophical Reasoning Engine

**Responsibility**: Core reasoning and synthesis operations

**Key Capabilities**:
- Argument structure analysis
- Dialectical synthesis (thesis-antithesis-synthesis)
- Socratic inquiry generation
- Thought experiment construction

**Design Pattern**: Strategy pattern for different reasoning modes

### 2. Knowledge Layer

#### Concept Index

**Structure**: Dictionary mapping concept_id to PhilosophicalConcept

**Contents**:
- Concept definitions and explanations
- Tradition and domain classifications
- Related and opposed concepts
- Key figures and primary texts

#### Figure Index

**Structure**: Dictionary mapping figure_id to PhilosophicalFigure

**Contents**:
- Biographical information
- Major works and key concepts
- Philosophical tradition and influences
- Historical context

#### Text Index

**Structure**: Dictionary mapping text_id to PhilosophicalText

**Contents**:
- Text metadata (author, title, year)
- Key arguments and themes
- Influence on tradition
- Chunked content for RAG

#### Knowledge Graph

**Structure**: Node-edge graph with typed relationships

**Node Types**:
- concept, figure, text, argument, tradition, domain

**Edge Types**:
- RELATED_TO, OPPOSED_TO, INFLUENCED_BY, AUTHORED, BELONGS_TO

### 3. Embedding Layer

**Model**: sentence-transformers/all-mpnet-base-v2 (768 dimensions)

**Index Type**: HNSW (Hierarchical Navigable Small World)

**Collections**:
- phil_concepts: Concept embeddings
- phil_figures: Figure embeddings
- phil_texts: Text metadata embeddings
- phil_arguments: Argument embeddings
- phil_text_chunks: Chunked text embeddings

---

## Data Flow Architecture

### Query Flow

```
User Query
    │
    ▼
┌─────────────────────────────┐
│ PhilosophicalConsciousness  │
│ Interface.query_concept()   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Generate Query Embedding    │
│ (all-mpnet-base-v2)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐      ┌─────────────────────────┐
│ Vector Search               │─────▶│ Knowledge Graph         │
│ (HNSW Index)                │      │ Traversal (if needed)   │
└─────────────┬───────────────┘      └───────────┬─────────────┘
              │                                   │
              └─────────────┬─────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Result Ranking & Assembly   │
              │ (Relevance + Maturity)      │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ RAG Context Construction    │
              └─────────────┬───────────────┘
                            │
                            ▼
                     Query Response
```

### Research Flow

```
Research Trigger
(Query Miss / Gap Detection / User Request)
    │
    ▼
┌─────────────────────────────┐
│ Research Agent Coordinator  │
│ create_task()               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Task Queue                  │
│ (Priority-based scheduling) │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Parallel Source Fetching                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ SEP Agent   │  │ PhilPapers  │  │ Web Search  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
              ┌─────────────────────────────┐
              │ Content Extraction          │
              │ (Concepts, Figures, Args)   │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Validation Layer            │
              │ (Quality Score > 0.5)       │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Knowledge Integration       │
              │ (Index + Graph + Embeddings)│
              └─────────────────────────────┘
```

---

## Integration Architecture

### Message Bus Integration

```
                    ┌─────────────────────────┐
                    │     Message Bus         │
                    │                         │
    ┌───────────────┤ Channels:               │
    │               │ - global_workspace      │
    │  Subscribe    │ - engagement_context    │
    │               │ - meditation_state      │
    │               │ - memory_query          │
    │               │ - consciousness_state   │
    │               │                         │
    │               │ - philosophical_wisdom  │◀──── Publish
    │               │ - philosophical_response│◀──── Publish
    │               │ - research_updates      │◀──── Publish
    │               └─────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Form 28 Message Handlers    │
│                             │
│ @handler("WORKSPACE_BROADCAST")
│ @handler("CONTEXT_UPDATE")
│ @handler("STATE_CHANGE")
│ @handler("MEMORY_QUERY")
└─────────────────────────────┘
```

### Cross-Form Integration Points

| Connected Form | Integration Type | Mechanism |
|----------------|-----------------|-----------|
| Form 10 | Deep | Shared embedding space, memory storage |
| Form 11 | Bidirectional | Philosophical reflection requests |
| Form 12 | Synthesis | Narrative framework provision |
| Form 14 | Broadcast | Wisdom content submission |
| Form 27 | Deep | Processing mode enhancement |

---

## Design Rationale

### 1. Shared Embedding Space (Form 10)

**Decision**: Use identical embedding model as Form 10's RAG system

**Rationale**:
- Enables cross-form semantic search
- Reduces memory footprint (single model)
- Allows philosophical memories alongside identity memories
- Maintains semantic consistency across systems

### 2. HNSW Vector Index

**Decision**: Use Hierarchical Navigable Small World index

**Rationale**:
- O(log n) query time complexity
- High recall (>95%) at configured parameters
- Supports incremental updates
- Reasonable memory overhead

### 3. Knowledge Graph Complement

**Decision**: Maintain knowledge graph alongside vector index

**Rationale**:
- Captures explicit relationships (opposition, influence)
- Enables multi-hop reasoning
- Supports graph-based queries (shortest path, clustering)
- Preserves philosophical nuance (vectors lose structure)

### 4. Autonomous Research Agent

**Decision**: Implement proactive knowledge expansion

**Rationale**:
- Philosophical knowledge is vast; pre-indexing impractical
- On-demand research ensures relevance
- Gap detection improves coverage over time
- Multiple sources ensure accuracy

### 5. Cross-Tradition Synthesis

**Decision**: Support synthesis while maintaining fidelity

**Rationale**:
- Real philosophical insight often transcends traditions
- Users benefit from multi-perspective understanding
- Fidelity tracking prevents misrepresentation
- Explicit tension acknowledgment maintains nuance

---

## Scalability Considerations

### Knowledge Scale

| Component | Current Design | Scale Limit | Mitigation |
|-----------|---------------|-------------|------------|
| Concept Index | In-memory dict | ~100K concepts | Disk-backed with LRU cache |
| Vector Store | HNSW | ~1M vectors | Sharding by tradition |
| Knowledge Graph | In-memory | ~500K nodes | Graph database (Neo4j) |
| Research Queue | In-memory | ~1K tasks | Redis queue |

### Performance Targets

| Operation | Target Latency | Achieved |
|-----------|----------------|----------|
| Simple query | <100ms | Yes (p95) |
| Synthesis | <500ms | Yes (p95) |
| Research task | <30s | Depends on source |
| Embedding generation | <50ms | Yes (p95) |

---

## Security Considerations

### Data Validation

- All external content sanitized before storage
- Source URLs validated against allowlist
- Rate limiting on research requests
- Quality threshold (>0.5) before integration

### Access Control

- Message bus authentication required
- Research agent runs in sandboxed environment
- External API keys secured
- No user data in philosophical knowledge base

---

## Future Evolution

### Planned Enhancements

1. **Distributed Knowledge Store**: Scale beyond single-node limits
2. **Active Learning**: Use query patterns to guide research
3. **Community Contributions**: Accept verified external contributions
4. **Multi-Modal Philosophy**: Image/audio philosophical content
5. **Debate Simulation**: Multi-agent philosophical debate

### API Stability

- Core query APIs (v1) frozen
- Research agent APIs subject to enhancement
- Message bus schemas versioned
- Embedding model may be upgraded (with migration)
