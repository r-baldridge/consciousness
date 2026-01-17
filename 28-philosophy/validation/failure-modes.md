# Failure Modes

## Overview

This document catalogs potential failure modes for Form 28 (Philosophical Consciousness), their detection mechanisms, mitigation strategies, and recovery procedures.

---

## Failure Categories

| Category | Severity | Description |
|----------|----------|-------------|
| **Critical** | P0 | System unusable, data loss possible |
| **Major** | P1 | Significant functionality impaired |
| **Moderate** | P2 | Partial functionality affected |
| **Minor** | P3 | Cosmetic or edge case issues |

---

## Critical Failures (P0)

### F-001: Index Corruption

**Description**: Philosophical index becomes corrupted, preventing concept retrieval.

**Symptoms**:
- `query_concept()` returns exceptions
- Embedding lookups fail silently
- Inconsistent results between queries

**Detection**:
```python
async def check_index_integrity():
    # Verify all concept_ids are unique
    ids = list(concept_index.keys())
    if len(ids) != len(set(ids)):
        raise IndexCorruptionError("Duplicate concept IDs detected")

    # Verify embedding dimensions
    for concept in concept_index.values():
        if concept.embedding and len(concept.embedding) != 768:
            raise IndexCorruptionError(f"Invalid embedding dimension for {concept.concept_id}")

    # Verify graph consistency
    for edge in knowledge_graph.edges():
        if edge.source not in concept_index and edge.source not in figure_index:
            raise IndexCorruptionError(f"Orphan edge source: {edge.source}")
```

**Mitigation**:
- Regular integrity checks (hourly)
- Transactional writes with rollback
- Checksums on persisted index files

**Recovery**:
1. Stop all write operations
2. Load last known good snapshot
3. Replay research log from snapshot
4. Re-verify integrity
5. Resume operations

---

### F-002: Embedding Model Failure

**Description**: Sentence transformer model fails to load or generate embeddings.

**Symptoms**:
- `generate_embedding()` raises exceptions
- RAG queries return empty results
- Memory integration fails

**Detection**:
```python
async def verify_embedding_model():
    try:
        test_embedding = await generate_embedding("test")
        if len(test_embedding) != 768:
            raise EmbeddingModelError("Wrong embedding dimension")
        if not all(isinstance(x, float) for x in test_embedding):
            raise EmbeddingModelError("Invalid embedding type")
    except Exception as e:
        raise EmbeddingModelError(f"Model verification failed: {e}")
```

**Mitigation**:
- Fallback model: `all-MiniLM-L6-v2`
- Model health check on startup
- Cached embeddings for common queries

**Recovery**:
1. Attempt model reload
2. If fails, switch to fallback model
3. Re-embed critical concepts with fallback
4. Log degraded mode, alert operators
5. Schedule primary model restoration

---

### F-003: Form 10 Memory System Unavailable

**Description**: Integration with Form 10 (Memory) fails, breaking memory storage/retrieval.

**Symptoms**:
- `store_in_memory()` fails silently or raises
- `retrieve_from_memory()` returns empty
- Cross-form queries incomplete

**Detection**:
```python
async def check_form_10_health():
    if not form_10_interface:
        return FormHealthStatus.DISCONNECTED

    try:
        result = await form_10_interface.ping()
        if not result.healthy:
            return FormHealthStatus.UNHEALTHY
    except TimeoutError:
        return FormHealthStatus.TIMEOUT
    except Exception:
        return FormHealthStatus.ERROR

    return FormHealthStatus.HEALTHY
```

**Mitigation**:
- Local cache for recent memories
- Graceful degradation (Form 28 only mode)
- Automatic reconnection with backoff

**Recovery**:
1. Enter Form 28 standalone mode
2. Queue memory operations for later
3. Monitor Form 10 health
4. On recovery, flush queued operations
5. Verify synchronization

---

## Major Failures (P1)

### F-004: Research Agent Source Unavailable

**Description**: Primary research sources (SEP, PhilPapers) become unavailable.

**Symptoms**:
- Research tasks fail or timeout
- Knowledge expansion stalls
- Query misses not resolved

**Detection**:
```python
async def check_source_availability():
    status = {}
    for source in ResearchSource:
        try:
            response = await source.health_check()
            status[source] = response.available
        except:
            status[source] = False
    return status
```

**Mitigation**:
- Multiple source redundancy
- Local corpus as fallback
- Cached results for common queries
- Rate limit respect to avoid bans

**Recovery**:
1. Mark source as unavailable
2. Redirect to alternate sources
3. Increase local corpus priority
4. Monitor source recovery
5. Resume normal operation when available

---

### F-005: Cross-Tradition Synthesis Failure

**Description**: Synthesis algorithm produces incoherent or low-fidelity results.

**Symptoms**:
- Fidelity scores below threshold (<0.5)
- Contradictory synthesis statements
- Missing tradition perspectives

**Detection**:
```python
def validate_synthesis(synthesis: CrossTraditionSynthesis) -> bool:
    # Check minimum fidelity
    if any(score < 0.5 for score in synthesis.fidelity_scores.values()):
        return False

    # Check coherence
    if synthesis.coherence_score < 0.5:
        return False

    # Check all traditions represented
    if len(synthesis.fidelity_scores) != len(synthesis.traditions):
        return False

    return True
```

**Mitigation**:
- Fallback to individual tradition responses
- Synthesis quality gating
- Human review flagging for edge cases

**Recovery**:
1. Return individual tradition perspectives instead
2. Flag topic as synthesis-incompatible
3. Log for algorithm improvement
4. Consider manual synthesis entry

---

### F-006: Maturity State Inconsistency

**Description**: Maturity tracking state becomes inconsistent with actual index state.

**Symptoms**:
- Maturity scores don't reflect actual knowledge
- Growth not properly tracked
- Gap detection inaccurate

**Detection**:
```python
def verify_maturity_consistency():
    # Recalculate from index
    calculated = calculate_maturity_from_index()

    # Compare to stored state
    discrepancy = abs(calculated - maturity_state.overall_maturity)

    if discrepancy > 0.1:
        raise MaturityInconsistencyError(
            f"Maturity discrepancy: stored={maturity_state.overall_maturity}, "
            f"calculated={calculated}"
        )
```

**Mitigation**:
- Regular recalculation (daily)
- Transactional maturity updates
- Snapshot-based recovery

**Recovery**:
1. Recalculate maturity from index state
2. Update all tradition depths
3. Recalculate domain coverages
4. Reset synthesis counts if needed
5. Take new snapshot

---

## Moderate Failures (P2)

### F-007: Message Bus Disconnection

**Description**: Temporary loss of message bus connection affects integration.

**Symptoms**:
- Wisdom broadcasts not delivered
- Workspace integration fails
- Form coordination disrupted

**Detection**:
```python
async def check_message_bus():
    try:
        await message_bus.ping()
    except ConnectionError:
        return BusStatus.DISCONNECTED
    return BusStatus.CONNECTED
```

**Mitigation**:
- Automatic reconnection
- Local operation buffer
- Graceful standalone mode

**Recovery**:
1. Attempt reconnection with exponential backoff
2. Buffer outgoing messages
3. On reconnect, replay buffered messages
4. Re-subscribe to channels

---

### F-008: Knowledge Graph Cycle Detection

**Description**: Circular relationships introduced in knowledge graph.

**Symptoms**:
- Infinite loops in graph traversal
- Memory consumption spikes
- Query timeouts

**Detection**:
```python
def detect_cycles():
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in graph.nodes():
        if node not in visited:
            if dfs(node):
                return True
    return False
```

**Mitigation**:
- Cycle check before edge insertion
- Max traversal depth limits
- Visited set in traversals

**Recovery**:
1. Identify cycle-creating edges
2. Remove or mark as weak edges
3. Rebuild affected subgraph
4. Re-verify acyclicity

---

### F-009: Embedding Cache Exhaustion

**Description**: Embedding cache fills, causing cache misses and regeneration overhead.

**Symptoms**:
- Increased latency
- Higher compute usage
- Memory pressure

**Detection**:
```python
def check_cache_health():
    return {
        "size": len(embedding_cache),
        "max_size": EMBEDDING_CACHE.max_size,
        "utilization": len(embedding_cache) / EMBEDDING_CACHE.max_size,
        "hit_rate": embedding_cache.hit_rate(),
    }
```

**Mitigation**:
- LRU eviction policy
- Pre-warm common embeddings
- Dynamic cache sizing

**Recovery**:
1. Evict cold entries (LRU)
2. Pre-warm high-frequency queries
3. Consider cache size increase
4. Monitor hit rate

---

### F-010: Research Task Queue Overflow

**Description**: Research task queue exceeds capacity during high demand.

**Symptoms**:
- New tasks rejected
- Research requests fail
- Knowledge expansion blocked

**Detection**:
```python
def check_queue_health():
    return {
        "queue_depth": len(task_queue),
        "max_capacity": MAX_QUEUE_SIZE,
        "utilization": len(task_queue) / MAX_QUEUE_SIZE,
        "oldest_task_age": get_oldest_task_age(),
    }
```

**Mitigation**:
- Priority-based eviction
- Request rate limiting
- Queue persistence

**Recovery**:
1. Process high-priority tasks first
2. Merge duplicate queries
3. Evict stale low-priority tasks
4. Temporary rate limiting
5. Scale processing capacity

---

## Minor Failures (P3)

### F-011: Concept Duplicate Detection Miss

**Description**: Same concept indexed under different IDs.

**Symptoms**:
- Redundant concepts in results
- Inconsistent cross-references
- Inflated concept count

**Detection**:
```python
def detect_duplicates():
    embeddings = [(c.concept_id, c.embedding) for c in concept_index.values()
                  if c.embedding]

    duplicates = []
    for i, (id1, emb1) in enumerate(embeddings):
        for id2, emb2 in embeddings[i+1:]:
            similarity = cosine_similarity(emb1, emb2)
            if similarity > 0.95:
                duplicates.append((id1, id2, similarity))

    return duplicates
```

**Mitigation**:
- Pre-insertion duplicate check
- Periodic deduplication scans
- Merge interface for operators

**Recovery**:
1. Identify duplicate pairs
2. Select canonical entry
3. Merge metadata and relationships
4. Update references
5. Remove duplicate

---

### F-012: Tradition Misclassification

**Description**: Concept assigned to incorrect philosophical tradition.

**Symptoms**:
- Tradition queries return unexpected results
- Cross-tradition synthesis uses wrong perspectives
- User confusion

**Detection**:
```python
def verify_tradition_assignments():
    issues = []
    for concept in concept_index.values():
        # Check if key figures match tradition
        tradition_figures = TRADITION_FIGURES[concept.tradition]
        if not any(f in tradition_figures for f in concept.key_figures):
            issues.append({
                "concept": concept.concept_id,
                "issue": "key_figures_mismatch",
                "expected_figures": tradition_figures,
                "actual_figures": concept.key_figures
            })
    return issues
```

**Mitigation**:
- Cross-reference with SEP categories
- Multi-source validation
- Manual review flagging

**Recovery**:
1. Flag concept for review
2. Research correct classification
3. Update tradition assignment
4. Rebuild tradition index
5. Log correction

---

## Failure Response Matrix

| Failure ID | Auto-Detect | Auto-Recover | Manual Required | Alerting |
|------------|-------------|--------------|-----------------|----------|
| F-001 | Yes | Partial | Yes | P0 Alert |
| F-002 | Yes | Yes | No | P0 Alert |
| F-003 | Yes | Yes | No | P1 Alert |
| F-004 | Yes | Yes | No | P1 Alert |
| F-005 | Yes | Partial | No | Logged |
| F-006 | Yes | Yes | No | Logged |
| F-007 | Yes | Yes | No | Logged |
| F-008 | Yes | Yes | No | Logged |
| F-009 | Yes | Yes | No | Metrics |
| F-010 | Yes | Yes | No | Metrics |
| F-011 | Periodic | Yes | No | Logged |
| F-012 | Periodic | No | Yes | Logged |

---

## Health Check Endpoint

```python
async def comprehensive_health_check() -> HealthReport:
    """Run all health checks and return aggregated report."""
    checks = {
        "index_integrity": await check_index_integrity(),
        "embedding_model": await verify_embedding_model(),
        "form_10_health": await check_form_10_health(),
        "source_availability": await check_source_availability(),
        "maturity_consistency": verify_maturity_consistency(),
        "message_bus": await check_message_bus(),
        "cache_health": check_cache_health(),
        "queue_health": check_queue_health(),
    }

    overall_status = HealthStatus.HEALTHY
    for check, result in checks.items():
        if result.status == HealthStatus.CRITICAL:
            overall_status = HealthStatus.CRITICAL
            break
        elif result.status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED

    return HealthReport(
        overall_status=overall_status,
        checks=checks,
        timestamp=datetime.now(timezone.utc)
    )
```

---

## Alerting Configuration

```yaml
# alerting.yaml
alerts:
  p0_critical:
    channels: [pagerduty, slack_urgent]
    conditions:
      - failure_ids: [F-001, F-002]
      - consecutive_failures: 1

  p1_major:
    channels: [slack_alerts]
    conditions:
      - failure_ids: [F-003, F-004, F-005, F-006]
      - consecutive_failures: 3

  degraded_mode:
    channels: [slack_alerts]
    conditions:
      - overall_status: DEGRADED
      - duration: 5m
```
