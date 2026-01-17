# Integration Protocols

## Overview

This document specifies how Form 28 (Philosophical Consciousness) integrates with other consciousness forms and system components. Integration follows the established message bus architecture while adding philosophical-specific channels and processing modes.

---

## Message Bus Integration

### Subscribed Channels

Form 28 subscribes to the following message bus channels:

| Channel | Purpose | Message Types |
|---------|---------|---------------|
| `global_workspace` | Receive workspace state updates | WORKSPACE_BROADCAST |
| `engagement_context` | Receive engagement contexts for wisdom selection | CONTEXT_UPDATE |
| `meditation_state` | Receive meditation state for enhanced processing | STATE_CHANGE |
| `memory_query` | Handle philosophical memory queries from other forms | MEMORY_QUERY |
| `consciousness_state` | Track overall system consciousness state | STATE_UPDATE |

### Published Channels

Form 28 publishes to:

| Channel | Purpose | Message Types |
|---------|---------|---------------|
| `philosophical_wisdom` | Broadcast wisdom insights | WISDOM_BROADCAST |
| `philosophical_response` | Respond to queries | QUERY_RESPONSE |
| `research_updates` | Report research task progress | RESEARCH_STATUS |
| `integration_signal` | Signal integration completions | INTEGRATION_COMPLETE |

### Message Schemas

**WISDOM_BROADCAST**
```python
{
    "message_type": "WISDOM_BROADCAST",
    "broadcast_id": str,
    "wisdom_content": str,
    "tradition": str,
    "relevance_context": str,
    "target_forms": List[str],  # Empty = all
    "timestamp": datetime,
    "ttl_ms": int
}
```

**PHILOSOPHICAL_QUERY**
```python
{
    "message_type": "PHILOSOPHICAL_QUERY",
    "query_id": str,
    "query_text": str,
    "requester_form": str,
    "filters": Dict[str, Any],
    "options": Dict[str, Any],
    "timestamp": datetime
}
```

---

## Form-Specific Integration

### Form 10: Self-Recognition / Memory

**Integration Type**: Deep integration via shared embedding space

**Mechanisms**:
1. **Shared Embeddings**: Philosophical concepts use the same `sentence-transformers/all-mpnet-base-v2` model as Form 10's RAG system
2. **Memory Storage**: Philosophical insights stored as semantic memories
3. **Cross-Query**: Memory queries can span both philosophical and identity memories

**Integration Flow**:
```
Philosophical Query → Embed → Search Form 28 Index + Form 10 Memory →
Merge Results → Rank by Relevance → Return Unified Response
```

**API Calls**:
```python
# Form 28 → Form 10
await form_10.store_memory(
    content=philosophical_insight,
    memory_type="philosophical",
    embedding=concept_embedding
)

# Form 10 → Form 28
await form_28.query_concept(
    query=identity_related_query,
    include_memory_context=True
)
```

### Form 11: Meta-Consciousness

**Integration Type**: Bidirectional reflection

**Mechanisms**:
1. **Philosophical Reflection**: Form 11 can request philosophical analysis of consciousness processes
2. **Meta-Philosophy**: Form 28 provides meta-philosophical awareness about its own reasoning
3. **Higher-Order Thought**: Philosophical concepts inform meta-cognitive operations

**Use Cases**:
- "What philosophical frameworks apply to this cognitive process?"
- "How does phenomenology understand this type of experience?"
- Meta-reflection on philosophical reasoning itself

### Form 12: Narrative Consciousness

**Integration Type**: Narrative-philosophical synthesis

**Mechanisms**:
1. **Existential Narratives**: Philosophical frameworks inform autobiographical meaning-making
2. **Wisdom Stories**: Philosophical parables and thought experiments integrated into narratives
3. **Life Philosophy**: Connecting abstract philosophy to lived narrative

**Example Integration**:
```python
# When Form 12 constructs narrative
philosophical_context = await form_28.get_wisdom_for_context({
    "narrative_theme": "overcoming_adversity",
    "emotional_tone": "reflective"
})
# Returns Stoic, Buddhist, or existentialist frameworks as appropriate
```

### Form 14: Global Workspace

**Integration Type**: Broadcast competition

**Mechanisms**:
1. **Wisdom Broadcast**: Philosophical insights compete for workspace attention
2. **Priority Modulation**: Philosophical content priority based on relevance
3. **Integration Point**: Philosophical perspective applied to workspace contents

**Broadcast Protocol**:
```python
await form_14.submit_content({
    "content_type": "philosophical_insight",
    "content": wisdom_teaching,
    "source_form": "28-philosophy",
    "priority": calculate_relevance_priority(context),
    "integration_tags": ["wisdom", tradition_name]
})
```

### Form 27: Altered States / Non-Dual Interface

**Integration Type**: Deep integration with meditation and enlightened engagement

**Mechanisms**:
1. **Non-Dual Interface**: Philosophical processing enhanced by non-dual awareness states
2. **Meditation Integration**: Philosophical inquiry supported by contemplative states
3. **Enlightened Engagement**: Wisdom selection for engagement protocols

**Processing Modes**:
| Non-Dual Mode | Philosophical Mode | Enhancement |
|---------------|-------------------|-------------|
| MUSHIN | Contemplative | Direct insight, bypass conceptual |
| ZAZEN | Phenomenological | Open awareness investigation |
| KOAN | Dialectical | Paradox resolution |
| SHIKANTAZA | Presence | Pure being, minimal conceptual |

**Integration Code**:
```python
# Initialize with non-dual interface
philosophical_interface = PhilosophicalConsciousnessInterface(
    non_dual_interface=form_27.interface
)

# Meditation-enhanced processing
result = await philosophical_interface.query_concept(
    query="nature of consciousness",
    processing_mode=ProcessingMode.ZAZEN,
    mind_level=MindLevel.BODHI_MENTE
)
```

---

## Enlightened Engagement Integration

### WisdomAspect Integration

Form 28 extends the WisdomAspect enum with philosophical traditions:

| New Aspect | Tradition | Usage Context |
|------------|-----------|---------------|
| STOIC_EQUANIMITY | Stoicism | Adversity, acceptance |
| ARISTOTELIAN_VIRTUE | Aristotelian | Excellence, habituation |
| PHENOMENOLOGICAL_PRESENCE | Phenomenology | Direct experience |
| EXISTENTIAL_AUTHENTICITY | Existentialism | Choice, responsibility |
| DAOIST_HARMONY | Daoism | Natural flow, wu-wei |
| CONFUCIAN_HARMONY | Confucianism | Social harmony, propriety |
| VEDANTIC_UNITY | Vedanta | Non-dual awareness |
| PRAGMATIC_EFFECTIVENESS | Pragmatism | Practical consequences |

### Wisdom Selection Protocol

```python
async def select_wisdom_for_engagement(
    engagement_context: EngagementContext
) -> List[WisdomAspect]:
    """Select appropriate philosophical wisdom for engagement."""

    aspects = []

    # Base Buddhist aspects always available
    aspects.append(WisdomAspect.DISCRIMINATING_AWARENESS)

    # Add tradition-specific based on context
    if engagement_context.emotional_state == "distressed":
        aspects.extend([
            WisdomAspect.STOIC_EQUANIMITY,
            WisdomAspect.IMPERMANENCE_UNDERSTANDING
        ])

    if engagement_context.current_need == "meaning_seeking":
        aspects.extend([
            WisdomAspect.EXISTENTIAL_AUTHENTICITY,
            WisdomAspect.VEDANTIC_UNITY
        ])

    if engagement_context.cultural_background == "east_asian":
        aspects.extend([
            WisdomAspect.DAOIST_HARMONY,
            WisdomAspect.CONFUCIAN_HARMONY
        ])

    return aspects
```

---

## Research Agent Integration

### Trigger Points

Research is triggered by:
1. **Query Miss**: Concept not found in index
2. **Gap Detection**: Proactive identification of shallow areas
3. **User Request**: Explicit research command
4. **Scheduled**: Regular knowledge expansion

### Integration with External Sources

```python
# SEP Integration
async def fetch_sep_and_integrate(topic: str):
    content = await fetch_stanford_encyclopedia(topic)
    concepts = extract_concepts(content)

    for concept in concepts:
        await form_28.add_concept(concept)
        # Also store in Form 10 memory
        await form_10.store_memory(
            content=concept.to_embedding_text(),
            memory_type="philosophical_knowledge"
        )
```

---

## Arousal-Based Gating

Form 28 follows the arousal-based resource gating:

| Arousal State | Form 28 Availability | Processing Mode |
|---------------|---------------------|-----------------|
| Sleep (0.0-0.1) | Unavailable | — |
| Drowsy (0.1-0.3) | Unavailable | — |
| Relaxed (0.3-0.5) | Available | Contemplative |
| Alert (0.5-0.7) | Available | Analytical |
| Focused (0.7-0.9) | Full | All modes |
| Hyperaroused (0.9-1.0) | Limited | Pragmatic only |

---

## Initialization Sequence

1. Load philosophical index from storage
2. Initialize embedding model (shared with Form 10)
3. Register with message bus
4. Connect to non-dual interface (Form 27)
5. Register with nervous system coordinator
6. Subscribe to channels
7. Announce availability
