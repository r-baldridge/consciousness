# Memory Integration

## Overview

Form 28 integrates with the consciousness system's memory architecture, sharing infrastructure with Form 10 (Self-Recognition/Long-Term Memory) while maintaining philosophical-specific indexing and retrieval capabilities.

---

## Shared Embedding Space

### Embedding Model

Both Form 28 and Form 10 use the same embedding model:

```yaml
model_name: "sentence-transformers/all-mpnet-base-v2"
dimensions: 768
normalization: "L2"
similarity_metric: "cosine"
```

**Rationale**: Shared embedding space enables:
- Cross-form semantic search
- Memory queries spanning philosophical and identity contexts
- Unified relevance scoring

### Embedding Generation

```python
async def generate_embedding(text: str) -> List[float]:
    """Generate 768-dim embedding using shared model."""
    # Use Form 10's embedding infrastructure
    embedding = await embedding_model.encode(text)
    return normalize_l2(embedding)
```

---

## Vector Store Architecture

### Collections

Form 28 maintains separate collections within the shared vector store:

| Collection | Content | Count |
|------------|---------|-------|
| `phil_concepts` | Philosophical concepts | Variable |
| `phil_figures` | Philosopher profiles | Variable |
| `phil_texts` | Primary text metadata | Variable |
| `phil_arguments` | Argument structures | Variable |
| `phil_text_chunks` | Indexed text chunks | Variable |

### Index Configuration

```python
VECTOR_INDEX_CONFIG = {
    "index_type": "HNSW",  # Hierarchical Navigable Small World
    "ef_construction": 200,  # Build-time parameter
    "M": 16,  # Max connections per node
    "ef_search": 100,  # Query-time parameter
}
```

### Storage Structure

```
consciousness/28-philosophy/index/
├── embeddings/
│   ├── concepts.bin      # Concept vectors
│   ├── figures.bin       # Figure vectors
│   ├── texts.bin         # Text metadata vectors
│   ├── arguments.bin     # Argument vectors
│   └── chunks.bin        # Text chunk vectors
├── metadata/
│   ├── concepts.json     # Concept metadata
│   ├── figures.json      # Figure metadata
│   └── ...
└── graph/
    ├── nodes.json        # Knowledge graph nodes
    └── edges.json        # Knowledge graph edges
```

---

## RAG Processing

### Query Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Embedding │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Parallel Vector Search          │
│  ┌─────────┐  ┌─────────┐          │
│  │Form 28  │  │Form 10  │          │
│  │Concepts │  │Memories │          │
│  └────┬────┘  └────┬────┘          │
└───────┼────────────┼────────────────┘
        │            │
        ▼            ▼
┌─────────────────────────────────────┐
│         Result Merging              │
│   Deduplicate + Rank by Relevance   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      Context Construction           │
│   Build RAG context from results    │
└────────────────┬────────────────────┘
                 │
                 ▼
            Response
```

### Retrieval Function

```python
async def retrieve_with_rag(
    query: str,
    top_k: int = 10,
    include_memories: bool = True
) -> RAGContext:
    """Retrieve philosophical content with RAG."""

    query_embedding = await generate_embedding(query)

    # Search Form 28 philosophical index
    phil_results = await vector_search(
        collection="phil_concepts",
        query_vector=query_embedding,
        top_k=top_k
    )

    # Optionally include Form 10 memories
    if include_memories:
        memory_results = await form_10.search_memories(
            query_vector=query_embedding,
            memory_type="philosophical",
            top_k=top_k // 2
        )
        phil_results.extend(memory_results)

    # Rank and construct context
    ranked = rank_by_relevance(phil_results, query_embedding)
    context = construct_rag_context(ranked[:top_k])

    return context
```

### Context Construction

```python
def construct_rag_context(results: List[SearchResult]) -> RAGContext:
    """Build context from search results."""
    context_parts = []

    for result in results:
        if result.entity_type == "concept":
            context_parts.append(
                f"Concept: {result.entity.name}\n"
                f"Tradition: {result.entity.tradition.value}\n"
                f"Definition: {result.entity.definition}\n"
            )
        elif result.entity_type == "memory":
            context_parts.append(
                f"Related knowledge: {result.content}\n"
            )

    return RAGContext(
        content="\n---\n".join(context_parts),
        sources=[r.source for r in results],
        relevance_scores={r.id: r.score for r in results}
    )
```

---

## Form 10 Integration

### Memory Storage

Philosophical insights can be stored as Form 10 memories:

```python
async def store_philosophical_memory(
    concept: PhilosophicalConcept,
    context: str
) -> str:
    """Store philosophical insight in long-term memory."""

    memory_content = {
        "type": "philosophical_concept",
        "concept_id": concept.concept_id,
        "name": concept.name,
        "tradition": concept.tradition.value,
        "definition": concept.definition,
        "context": context,
    }

    memory_id = await form_10.store_memory(
        content=json.dumps(memory_content),
        memory_type="philosophical",
        embedding=concept.embedding,
        importance=calculate_importance(concept)
    )

    return memory_id
```

### Memory Retrieval

```python
async def retrieve_philosophical_memories(
    query: str,
    max_results: int = 5
) -> List[PhilosophicalMemory]:
    """Retrieve philosophical memories from Form 10."""

    results = await form_10.search_memories(
        query=query,
        memory_type="philosophical",
        max_results=max_results
    )

    memories = []
    for result in results:
        content = json.loads(result.content)
        memories.append(PhilosophicalMemory(
            memory_id=result.memory_id,
            concept_id=content.get("concept_id"),
            name=content.get("name"),
            tradition=content.get("tradition"),
            definition=content.get("definition"),
            context=content.get("context"),
            relevance=result.similarity_score
        ))

    return memories
```

---

## Agentic Document Querying

### Multi-Hop Query

For complex philosophical questions requiring multiple reasoning steps:

```python
async def agentic_query(
    question: str,
    max_hops: int = 3
) -> AgenticQueryResult:
    """Perform multi-hop agentic query."""

    hops = []
    current_context = []

    for hop in range(max_hops):
        # Generate sub-query based on accumulated context
        sub_query = generate_sub_query(question, current_context)

        # Retrieve relevant content
        results = await retrieve_with_rag(sub_query, top_k=5)

        # Extract answer for this hop
        hop_answer = extract_answer(sub_query, results)

        hops.append(QueryHop(
            query=sub_query,
            results=results,
            answer=hop_answer
        ))

        current_context.append(hop_answer)

        # Check if we have sufficient answer
        if is_answer_complete(question, current_context):
            break

    # Synthesize final answer
    final_answer = synthesize_answer(question, hops)

    return AgenticQueryResult(
        question=question,
        hops=hops,
        final_answer=final_answer
    )
```

### Sub-Query Generation

```python
def generate_sub_query(
    original_question: str,
    accumulated_context: List[str]
) -> str:
    """Generate next sub-query based on context."""

    if not accumulated_context:
        # First hop: identify main concepts
        return f"What are the key philosophical concepts related to: {original_question}"

    # Subsequent hops: dig deeper based on what we've learned
    last_context = accumulated_context[-1]
    return f"Given that {last_context}, what philosophical insights address: {original_question}"
```

---

## Persistence

### Save Operations

```python
async def persist_index():
    """Persist philosophical index to disk."""

    # Save embeddings
    save_embeddings("embeddings/concepts.bin", concept_embeddings)
    save_embeddings("embeddings/figures.bin", figure_embeddings)

    # Save metadata
    save_json("metadata/concepts.json", concept_index)
    save_json("metadata/figures.json", figure_index)

    # Save graph
    save_json("graph/nodes.json", graph_nodes)
    save_json("graph/edges.json", graph_edges)

    # Save maturity state
    save_json("metadata/maturity_state.json", maturity_state)
```

### Load Operations

```python
async def load_index():
    """Load philosophical index from disk."""

    # Load embeddings
    concept_embeddings = load_embeddings("embeddings/concepts.bin")
    figure_embeddings = load_embeddings("embeddings/figures.bin")

    # Load metadata
    concept_index = load_json("metadata/concepts.json")
    figure_index = load_json("metadata/figures.json")

    # Load graph
    graph_nodes = load_json("graph/nodes.json")
    graph_edges = load_json("graph/edges.json")

    # Load maturity state
    maturity_state = load_json("metadata/maturity_state.json")

    # Rebuild in-memory structures
    await rebuild_indices()
```

---

## Caching

### Query Cache

```python
QUERY_CACHE = LRUCache(max_size=1000, ttl_seconds=3600)

async def cached_query(query: str, **kwargs) -> QueryResult:
    cache_key = hash_query(query, kwargs)

    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]

    result = await execute_query(query, **kwargs)
    QUERY_CACHE[cache_key] = result

    return result
```

### Embedding Cache

```python
EMBEDDING_CACHE = LRUCache(max_size=10000, ttl_seconds=86400)

async def cached_embedding(text: str) -> List[float]:
    cache_key = hash(text)

    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]

    embedding = await generate_embedding(text)
    EMBEDDING_CACHE[cache_key] = embedding

    return embedding
```
