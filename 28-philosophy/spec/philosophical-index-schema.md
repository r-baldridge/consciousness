# Philosophical Index Schema

## Overview

This document specifies the schema and organization of the philosophical knowledge index. The index serves as the central repository for philosophical knowledge, enabling efficient retrieval, cross-referencing, and continuous expansion.

---

## Index Architecture

### Multi-Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY INTERFACE                          │
│         (Semantic Search, Filters, RAG Retrieval)           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                          │
│        Vector Store (768-dim sentence-transformers)         │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│   │Concepts │ │Figures  │ │Texts    │ │Arguments│          │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE GRAPH                           │
│              (Entity Relationships)                         │
│   Concept ←→ Concept    Figure ←→ Figure                   │
│   Concept ←→ Figure     Text ←→ Concept                    │
│   Tradition ←→ All      Domain ←→ All                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   PRIMARY STORAGE                           │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│   │ concepts.json │ │ figures.json  │ │  texts.json   │    │
│   └───────────────┘ └───────────────┘ └───────────────┘    │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│   │arguments.json │ │traditions.json│ │ syntheses.json│    │
│   └───────────────┘ └───────────────┘ └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Entity Schemas

### Concept Index

**File**: `index/concepts.json`

```json
{
  "schema_version": "1.0",
  "entity_type": "concept",
  "count": 0,
  "concepts": {
    "concept_id": {
      "id": "categorical_imperative",
      "name": "Categorical Imperative",
      "alternate_names": ["Kategorischer Imperativ"],
      "tradition": "KANTIAN",
      "secondary_traditions": ["GERMAN_IDEALISM"],
      "domain": "ETHICS",
      "secondary_domains": ["METAETHICS"],
      "definition": "Act only according to that maxim whereby you can at the same time will that it should become a universal law",
      "extended_description": "...",
      "key_arguments": ["universalizability_test", "humanity_formula"],
      "counter_arguments": ["consequentialist_objections"],
      "related_concepts": ["hypothetical_imperative", "autonomy", "moral_law"],
      "opposed_concepts": ["consequentialism"],
      "prerequisite_concepts": ["practical_reason", "good_will"],
      "key_figures": ["kant"],
      "primary_texts": ["groundwork_metaphysics_morals"],
      "maturity_score": 0.75,
      "maturity_level": "PROFICIENT",
      "research_depth": 3,
      "last_updated": "2024-01-15T10:30:00Z",
      "sources": [
        {"type": "SEP", "entry": "kant-moral", "accessed": "2024-01-10"}
      ],
      "created_at": "2024-01-01T00:00:00Z"
    }
  }
}
```

### Figure Index

**File**: `index/figures.json`

```json
{
  "schema_version": "1.0",
  "entity_type": "figure",
  "count": 0,
  "figures": {
    "figure_id": {
      "id": "kant",
      "name": "Immanuel Kant",
      "alternate_names": ["Kant"],
      "birth_year": 1724,
      "death_year": 1804,
      "birth_place": "Konigsberg, Prussia",
      "nationality": "Prussian/German",
      "era": "Enlightenment",
      "traditions": ["KANTIAN", "GERMAN_IDEALISM"],
      "domains": ["EPISTEMOLOGY", "ETHICS", "METAPHYSICS", "AESTHETICS"],
      "schools_founded": ["Kantian Critical Philosophy"],
      "core_ideas": [
        "categorical_imperative",
        "transcendental_idealism",
        "synthetic_a_priori",
        "thing_in_itself"
      ],
      "key_arguments": ["transcendental_deduction"],
      "famous_quotes": [
        "Two things fill the mind with ever new and increasing admiration and awe..."
      ],
      "key_works": [
        "critique_pure_reason",
        "critique_practical_reason",
        "groundwork_metaphysics_morals"
      ],
      "teachers": ["martin_knutzen"],
      "students": ["fichte", "schelling_early"],
      "influences": ["hume", "leibniz", "newton", "rousseau"],
      "influenced": ["fichte", "schelling", "hegel", "schopenhauer"],
      "biography_summary": "German philosopher who synthesized rationalism and empiricism...",
      "sources": []
    }
  }
}
```

### Text Index

**File**: `index/texts.json`

```json
{
  "schema_version": "1.0",
  "entity_type": "text",
  "count": 0,
  "texts": {
    "text_id": {
      "id": "critique_pure_reason",
      "title": "Critique of Pure Reason",
      "alternate_titles": ["Kritik der reinen Vernunft", "KrV", "First Critique"],
      "author": "kant",
      "author_name": "Immanuel Kant",
      "editors": [],
      "translators": ["norman_kemp_smith", "paul_guyer"],
      "tradition": "KANTIAN",
      "domains": ["EPISTEMOLOGY", "METAPHYSICS"],
      "genre": "treatise",
      "year_written": 1781,
      "year_published": 1781,
      "period": "Enlightenment",
      "summary": "Kant's magnum opus examining the limits and possibilities of human reason...",
      "key_concepts": [
        "synthetic_a_priori",
        "transcendental_idealism",
        "categories_understanding",
        "antinomies"
      ],
      "key_arguments": ["transcendental_deduction", "refutation_idealism"],
      "structure": "Transcendental Aesthetic, Transcendental Analytic, Transcendental Dialectic",
      "local_path": null,
      "digital_sources": [
        {"name": "Gutenberg", "url": "https://www.gutenberg.org/ebooks/4280"}
      ],
      "is_indexed": false,
      "chunk_count": 0,
      "influenced_texts": ["phenomenology_of_spirit", "world_as_will"],
      "influenced_by": ["treatise_human_nature"],
      "sources": []
    }
  }
}
```

### Tradition Index

**File**: `index/traditions.json`

```json
{
  "schema_version": "1.0",
  "entity_type": "tradition",
  "count": 0,
  "traditions": {
    "tradition_id": {
      "id": "KANTIAN",
      "display_name": "Kantian Philosophy",
      "origin_period": "18th century",
      "origin_location": "Prussia/Germany",
      "historical_development": "Developed by Kant in the 1780s...",
      "core_tenets": [
        "Synthesis of rationalism and empiricism",
        "Phenomena vs noumena distinction",
        "Transcendental idealism",
        "Categorical imperative",
        "Autonomy of reason"
      ],
      "central_questions": [
        "What can I know?",
        "What ought I to do?",
        "What may I hope?",
        "What is man?"
      ],
      "key_concepts": [
        "categorical_imperative",
        "thing_in_itself",
        "synthetic_a_priori"
      ],
      "methodology": "Transcendental critique",
      "founders": ["kant"],
      "major_figures": ["kant", "fichte", "reinhold"],
      "canonical_texts": [
        "critique_pure_reason",
        "critique_practical_reason",
        "groundwork_metaphysics_morals"
      ],
      "influences_from": ["RATIONALISM", "EMPIRICISM"],
      "influences_to": ["GERMAN_IDEALISM", "PHENOMENOLOGY", "ANALYTIC"],
      "opposed_to": [],
      "compatible_with": ["PHENOMENOLOGY"],
      "primary_domains": ["EPISTEMOLOGY", "ETHICS"],
      "secondary_domains": ["METAPHYSICS", "AESTHETICS"],
      "knowledge_depth": 0.0,
      "concept_count": 0,
      "figure_count": 0,
      "text_count": 0,
      "wisdom_teachings": [
        "Act only on principles you could will to be universal laws",
        "Treat humanity never merely as means but always as ends"
      ],
      "practical_guidance": [
        "Test maxims by universalizability",
        "Respect rational agency in self and others"
      ]
    }
  }
}
```

### Argument Index

**File**: `index/arguments.json`

```json
{
  "schema_version": "1.0",
  "entity_type": "argument",
  "count": 0,
  "arguments": {
    "argument_id": {
      "id": "cogito",
      "name": "Cogito Argument",
      "argument_type": "TRANSCENDENTAL",
      "tradition": "RATIONALISM",
      "domain": "EPISTEMOLOGY",
      "premises": [
        "If I am deceived, then I exist (for something must exist to be deceived)",
        "If I think, then I exist (thinking requires a thinker)",
        "I cannot doubt that I am thinking (even doubt is a form of thinking)"
      ],
      "conclusion": "I think, therefore I am (Cogito ergo sum)",
      "logical_form": "∀x(Thinks(x) → Exists(x)), Thinks(I) ⊢ Exists(I)",
      "originator": "descartes",
      "text_source": "meditations",
      "key_concepts": ["certainty", "doubt", "self_knowledge", "substance"],
      "assumptions": [
        "Thinking is self-evident",
        "Existence of thought implies existence of thinker"
      ],
      "objections": [
        {"source": "Hume", "objection": "No impression of a substantial self, only fleeting perceptions"},
        {"source": "Lichtenberg", "objection": "Can only conclude 'there is thinking', not 'I think'"}
      ],
      "replies": [
        {"to": "Hume", "reply": "The unity of apperception provides continuity"},
        {"to": "Lichtenberg", "reply": "Thinking necessarily involves a first-person perspective"}
      ],
      "related_arguments": ["transcendental_deduction", "private_language"],
      "validity_notes": "Valid if premises accepted",
      "soundness_notes": "Disputed whether conclusion is substantial",
      "scholarly_consensus": "Foundational but contested"
    }
  }
}
```

---

## Embedding Schema

### Vector Store Structure

```json
{
  "embedding_config": {
    "model": "sentence-transformers/all-mpnet-base-v2",
    "dimensions": 768,
    "normalization": "L2",
    "similarity_metric": "cosine"
  },
  "collections": {
    "concepts": {
      "count": 0,
      "index_type": "HNSW",
      "ef_construction": 200,
      "M": 16
    },
    "figures": {
      "count": 0,
      "index_type": "HNSW",
      "ef_construction": 200,
      "M": 16
    },
    "texts": {
      "count": 0,
      "index_type": "HNSW",
      "ef_construction": 200,
      "M": 16
    },
    "arguments": {
      "count": 0,
      "index_type": "HNSW",
      "ef_construction": 200,
      "M": 16
    },
    "text_chunks": {
      "count": 0,
      "index_type": "HNSW",
      "ef_construction": 200,
      "M": 16
    }
  }
}
```

### Embedding Record Format

```json
{
  "id": "concept_categorical_imperative",
  "entity_type": "concept",
  "entity_id": "categorical_imperative",
  "embedding": [0.123, -0.456, ...],
  "text_embedded": "Concept: Categorical Imperative | Tradition: kantian | Domain: ethics | Definition: Act only according to that maxim...",
  "metadata": {
    "tradition": "KANTIAN",
    "domain": "ETHICS",
    "maturity_score": 0.75,
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

---

## Knowledge Graph Schema

### Relationship Types

```yaml
relationships:
  # Concept-Concept
  - type: "RELATED_TO"
    description: "Concepts that are thematically related"
    bidirectional: true

  - type: "OPPOSED_TO"
    description: "Concepts that represent opposing views"
    bidirectional: true

  - type: "PREREQUISITE_OF"
    description: "Understanding A is needed to understand B"
    bidirectional: false

  - type: "DERIVED_FROM"
    description: "B is developed from or based on A"
    bidirectional: false

  - type: "ANALOGOUS_TO"
    description: "Similar concepts across traditions"
    bidirectional: true

  # Figure-Figure
  - type: "INFLUENCED"
    description: "A intellectually influenced B"
    bidirectional: false

  - type: "STUDENT_OF"
    description: "A was a student of B"
    bidirectional: false

  - type: "CONTEMPORARY_OF"
    description: "A and B were contemporaries"
    bidirectional: true

  - type: "RESPONDED_TO"
    description: "A's work responds to B's work"
    bidirectional: false

  # Figure-Concept
  - type: "ORIGINATED"
    description: "Figure originated the concept"
    bidirectional: false

  - type: "DEVELOPED"
    description: "Figure further developed the concept"
    bidirectional: false

  - type: "CRITICIZED"
    description: "Figure critiqued the concept"
    bidirectional: false

  # Figure-Text
  - type: "AUTHORED"
    description: "Figure wrote the text"
    bidirectional: false

  - type: "COMMENTED_ON"
    description: "Figure wrote commentary on text"
    bidirectional: false

  # Text-Concept
  - type: "INTRODUCES"
    description: "Text introduces the concept"
    bidirectional: false

  - type: "ANALYZES"
    description: "Text analyzes the concept"
    bidirectional: false

  # Tradition relationships
  - type: "BELONGS_TO"
    description: "Entity belongs to tradition"
    bidirectional: false

  - type: "EVOLVED_FROM"
    description: "Tradition B evolved from A"
    bidirectional: false
```

### Graph Storage Format

```json
{
  "nodes": [
    {
      "id": "concept_categorical_imperative",
      "type": "concept",
      "entity_id": "categorical_imperative",
      "labels": ["Concept", "Ethics", "Kantian"]
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "concept_categorical_imperative",
      "target": "concept_hypothetical_imperative",
      "type": "OPPOSED_TO",
      "properties": {
        "weight": 0.9,
        "description": "Contrasting types of imperatives"
      }
    }
  ]
}
```

---

## Index Operations

### Query Operations

```python
# Semantic search
search(
    query: str,
    collections: List[str] = ["concepts", "figures", "texts"],
    top_k: int = 10,
    min_similarity: float = 0.5,
    filters: Dict[str, Any] = {}
) -> List[SearchResult]

# Graph traversal
traverse(
    start_node: str,
    relationship_types: List[str],
    max_depth: int = 2,
    direction: str = "both"
) -> List[Node]

# Combined RAG query
rag_query(
    query: str,
    context_size: int = 5,
    include_graph_context: bool = True
) -> RAGResult

# Tradition-filtered search
search_tradition(
    query: str,
    tradition: PhilosophicalTradition,
    include_related: bool = False
) -> List[SearchResult]
```

### Update Operations

```python
# Add new entity
add_entity(
    entity_type: str,
    entity: Union[Concept, Figure, Text, Argument],
    compute_embedding: bool = True,
    update_graph: bool = True
) -> str

# Update existing entity
update_entity(
    entity_type: str,
    entity_id: str,
    updates: Dict[str, Any],
    recompute_embedding: bool = False
) -> bool

# Add relationship
add_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    properties: Dict[str, Any] = {}
) -> str

# Bulk import
bulk_import(
    entities: List[Dict[str, Any]],
    entity_type: str
) -> ImportResult
```

---

## Indexing Pipeline

### New Entity Flow

```
1. Entity Creation
   └─ Validate against schema
   └─ Generate unique ID

2. Embedding Generation
   └─ Create embedding text
   └─ Generate 768-dim vector
   └─ Normalize vector

3. Storage
   └─ Add to primary index (JSON)
   └─ Add to vector store
   └─ Add to knowledge graph

4. Relationship Extraction
   └─ Analyze for related entities
   └─ Create graph edges

5. Maturity Update
   └─ Update tradition depth
   └─ Update domain coverage
```

### Text Indexing Flow

```
1. Document Ingestion
   └─ Parse document format
   └─ Extract metadata

2. Chunking
   └─ Split into semantic chunks
   └─ Preserve section context

3. Chunk Embedding
   └─ Embed each chunk
   └─ Store with metadata

4. Concept Extraction
   └─ Identify philosophical concepts
   └─ Link to existing concepts
   └─ Create new concepts as needed

5. Full-Text Index
   └─ Add to searchable index
```

---

## Directory Structure

```
consciousness/28-philosophy/
└── index/
    ├── concepts.json           # Concept entities
    ├── figures.json            # Figure entities
    ├── texts.json              # Text entities
    ├── arguments.json          # Argument entities
    ├── traditions.json         # Tradition profiles
    ├── syntheses.json          # Cross-tradition syntheses
    ├── embeddings/
    │   ├── concepts.bin        # Concept vectors
    │   ├── figures.bin         # Figure vectors
    │   ├── texts.bin           # Text vectors
    │   ├── arguments.bin       # Argument vectors
    │   └── chunks.bin          # Text chunk vectors
    ├── graph/
    │   ├── nodes.json          # Graph nodes
    │   └── edges.json          # Graph edges
    └── metadata/
        ├── schema_version.json
        ├── statistics.json
        └── maturity_state.json
```
