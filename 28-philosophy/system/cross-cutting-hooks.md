# Cross-Cutting Hooks

## Overview

This document details the specific integration hooks between Form 28 (Philosophical Consciousness) and Forms 10-12, 14, and 27, as well as the enlightened engagement protocols.

---

## Form 10: Self-Recognition / Memory

### Hook Points

#### 1. Shared Embedding Infrastructure

**Location**: `neural_network/core/model_registry.py`

```python
# Both forms share this embedding model
SHARED_EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "dimensions": 768,
    "forms_using": ["10-self-recognition", "28-philosophy"]
}
```

#### 2. Memory Storage Hook

**Location**: `28-philosophy/interface/philosophical_consciousness_interface.py`

```python
async def store_in_memory(self, concept: PhilosophicalConcept, context: str):
    """Hook: Store philosophical insight in Form 10 memory."""
    if self.form_10_interface:
        await self.form_10_interface.store_memory(
            content=concept.to_embedding_text(),
            memory_type="philosophical",
            embedding=concept.embedding,
            metadata={
                "concept_id": concept.concept_id,
                "tradition": concept.tradition.value,
                "domain": concept.domain.value,
            }
        )
```

#### 3. Memory Retrieval Hook

```python
async def retrieve_from_memory(self, query: str) -> List[Dict]:
    """Hook: Retrieve philosophical memories from Form 10."""
    if self.form_10_interface:
        return await self.form_10_interface.search_memories(
            query=query,
            memory_type="philosophical",
            max_results=5
        )
    return []
```

#### 4. Identity-Philosophy Integration

```python
async def get_philosophy_for_identity_query(self, identity_aspect: str):
    """Hook: Provide philosophical context for identity questions."""
    # Map identity aspects to philosophical concepts
    mappings = {
        "self": ["no_self", "atman", "personal_identity"],
        "purpose": ["telos", "existential_meaning", "dharma"],
        "values": ["virtue_ethics", "categorical_imperative"],
    }

    concepts = []
    for concept_id in mappings.get(identity_aspect, []):
        if concept_id in self.concept_index:
            concepts.append(self.concept_index[concept_id])

    return concepts
```

---

## Form 11: Meta-Consciousness

### Hook Points

#### 1. Philosophical Reflection Request

**Message Type**: `PHILOSOPHICAL_REFLECTION_REQUEST`

```python
# Form 11 requests philosophical analysis of a cognitive process
@message_handler("PHILOSOPHICAL_REFLECTION_REQUEST")
async def handle_reflection_request(self, message: Dict):
    process_type = message["cognitive_process"]
    context = message["context"]

    # Find relevant philosophical frameworks
    frameworks = await self.query_concept(
        query=f"philosophical analysis of {process_type}",
        filters=QueryFilters(
            domains=[PhilosophicalDomain.PHILOSOPHY_OF_MIND]
        )
    )

    return {
        "frameworks": frameworks["concepts"],
        "analysis": self._generate_philosophical_analysis(process_type, frameworks)
    }
```

#### 2. Meta-Philosophical Awareness

```python
async def provide_meta_philosophical_context(self):
    """Hook: Provide meta-philosophical awareness about reasoning."""
    return {
        "current_reasoning_mode": self._reasoning_mode,
        "active_traditions": self._active_traditions,
        "assumptions_in_play": self._identify_philosophical_assumptions(),
        "alternative_frameworks": self._suggest_alternative_frameworks(),
    }
```

#### 3. Higher-Order Thought Integration

```python
async def integrate_with_hot(self, thought_content: Dict):
    """Hook: Enhance higher-order thought with philosophical context."""
    # Add philosophical dimension to meta-cognitive content
    return {
        "thought": thought_content,
        "philosophical_classification": self._classify_thought_philosophically(thought_content),
        "relevant_traditions": self._identify_relevant_traditions(thought_content),
    }
```

---

## Form 12: Narrative Consciousness

### Hook Points

#### 1. Existential Narrative Framework

```python
async def provide_narrative_framework(self, narrative_context: Dict):
    """Hook: Provide philosophical framework for narrative construction."""
    theme = narrative_context.get("theme", "")
    emotional_tone = narrative_context.get("emotional_tone", "")

    # Select appropriate philosophical traditions for narrative
    if "adversity" in theme or "challenge" in theme:
        traditions = [PhilosophicalTradition.STOICISM, PhilosophicalTradition.EXISTENTIALISM]
    elif "meaning" in theme or "purpose" in theme:
        traditions = [PhilosophicalTradition.EXISTENTIALISM, PhilosophicalTradition.CONFUCIAN]
    else:
        traditions = [PhilosophicalTradition.ARISTOTELIAN]

    wisdom = self.get_wisdom_for_context({
        "current_need": theme,
        "emotional_state": emotional_tone
    })

    return {
        "traditions": traditions,
        "wisdom_teachings": wisdom["teachings"],
        "narrative_guidance": self._generate_narrative_guidance(traditions)
    }
```

#### 2. Philosophical Story Integration

```python
async def get_philosophical_stories(self, theme: str) -> List[Dict]:
    """Hook: Provide philosophical parables and thought experiments."""
    stories = {
        "choice": [
            {"name": "Ring of Gyges", "tradition": "platonic", "lesson": "Would you be just if invisible?"},
            {"name": "Trolley Problem", "tradition": "analytic", "lesson": "Active vs passive harm"},
        ],
        "meaning": [
            {"name": "Myth of Sisyphus", "tradition": "existentialism", "lesson": "Finding meaning in absurdity"},
            {"name": "Allegory of Cave", "tradition": "platonic", "lesson": "Reality vs appearance"},
        ],
        "identity": [
            {"name": "Ship of Theseus", "tradition": "presocratic", "lesson": "Identity through change"},
            {"name": "Brain in Vat", "tradition": "analytic", "lesson": "Knowledge and reality"},
        ],
    }

    return stories.get(theme, [])
```

#### 3. Autobiographical Philosophy Integration

```python
async def enrich_autobiographical_narrative(self, life_event: Dict):
    """Hook: Add philosophical depth to autobiographical narratives."""
    event_type = life_event.get("type", "")

    philosophical_lens = {
        "loss": "impermanence_understanding",
        "achievement": "virtue_cultivation",
        "relationship": "intersubjectivity",
        "crisis": "existential_authenticity",
    }

    concept_id = philosophical_lens.get(event_type)
    if concept_id and concept_id in self.concept_index:
        return {
            "philosophical_context": self.concept_index[concept_id],
            "meaning_framework": self._generate_meaning_framework(event_type)
        }
    return None
```

---

## Form 14: Global Workspace

### Hook Points

#### 1. Wisdom Broadcast

```python
async def broadcast_philosophical_insight(self, insight: str, tradition: str, context: str):
    """Hook: Broadcast philosophical wisdom to global workspace."""
    await self.message_bus.publish(
        channel="global_workspace",
        message={
            "message_type": "WORKSPACE_CONTENT_SUBMISSION",
            "content_type": "philosophical_insight",
            "content": insight,
            "source_form": self.FORM_ID,
            "metadata": {
                "tradition": tradition,
                "context": context,
            },
            "priority": self._calculate_insight_priority(context),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
```

#### 2. Workspace State Integration

```python
@message_handler("WORKSPACE_BROADCAST")
async def handle_workspace_broadcast(self, message: Dict):
    """Hook: Respond to global workspace broadcasts with philosophical perspective."""
    workspace_contents = message.get("contents", [])

    # Check if philosophical input could enhance understanding
    for content in workspace_contents:
        if self._is_philosophically_relevant(content):
            # Submit philosophical enhancement
            enhancement = await self._generate_philosophical_enhancement(content)
            await self.broadcast_philosophical_insight(
                insight=enhancement,
                tradition=self._select_tradition_for_content(content),
                context=content.get("context", "")
            )
```

#### 3. Priority Calculation

```python
def _calculate_insight_priority(self, context: str) -> int:
    """Calculate priority for workspace submission."""
    base_priority = 5

    # Boost for existential/meaning contexts
    if any(word in context.lower() for word in ["meaning", "purpose", "crisis", "death"]):
        base_priority += 2

    # Boost for ethical contexts
    if any(word in context.lower() for word in ["should", "right", "wrong", "ethical"]):
        base_priority += 1

    return min(10, base_priority)
```

---

## Form 27: Altered States / Non-Dual Interface

### Hook Points

#### 1. Non-Dual Interface Connection

```python
def __init__(self, non_dual_interface: Optional[NonDualConsciousnessInterface] = None):
    """Initialize with Form 27 connection."""
    self.non_dual_interface = non_dual_interface

    if self.non_dual_interface:
        self._register_with_non_dual()
```

#### 2. Processing Mode Integration

```python
async def process_with_meditation_mode(
    self,
    query: str,
    processing_mode: ProcessingMode,
    mind_level: MindLevel
) -> Dict:
    """Hook: Process philosophical query with meditation-enhanced awareness."""
    if not self.non_dual_interface:
        return await self.query_concept(query)

    # Coordinate with Form 27
    coordination_result = await self.non_dual_interface.coordinate_consciousness_form({
        "consciousness_form": self.NAME,
        "form_type": "theoretical",
        "processing_mode": processing_mode.value,
        "mind_level": mind_level.value,
        "query": query
    })

    # Apply mode-specific processing
    if processing_mode == ProcessingMode.MUSHIN:
        # Direct insight, bypass conceptual
        self._reasoning_mode = "contemplative"
    elif processing_mode == ProcessingMode.ZAZEN:
        # Open awareness investigation
        self._reasoning_mode = "phenomenological"
    elif processing_mode == ProcessingMode.KOAN:
        # Paradox resolution
        self._reasoning_mode = "dialectical"

    result = await self.query_concept(query)
    result["coordination"] = coordination_result

    return result
```

#### 3. Karmic Seed Integration

```python
async def integrate_philosophical_karma(self, philosophical_insight: str):
    """Hook: Plant philosophical insight as karmic seed."""
    if self.non_dual_interface:
        from interface.non_dual_consciousness_interface import KarmicSeed

        seed = KarmicSeed(
            imprint=philosophical_insight,
            strength=0.5,
            originated_from=ConsciousnessLayer.MENTAL_CONSCIOUSNESS,
            dharma_polarity="skillful",
            timestamp=time.time()
        )

        await self.non_dual_interface.alaya_vijnana.store_seed(seed)
```

---

## Enlightened Engagement Protocols

### Hook Points

#### 1. WisdomAspect Selection

**Location**: `enlightened_engagement_protocols.py`

```python
async def _select_wisdom_aspects_with_philosophy(
    self,
    context: EngagementContext,
    mode: EngagementMode
) -> List[WisdomAspect]:
    """Hook: Include philosophical tradition aspects in selection."""
    aspects = [WisdomAspect.DISCRIMINATING_AWARENESS]

    # Add tradition-specific aspects based on context
    if context.emotional_state == "distressed":
        aspects.append(WisdomAspect.STOIC_EQUANIMITY)
        aspects.append(WisdomAspect.IMPERMANENCE_UNDERSTANDING)

    if context.current_need == "meaning_seeking":
        aspects.append(WisdomAspect.EXISTENTIAL_AUTHENTICITY)

    if context.spiritual_orientation == "eastern":
        aspects.append(WisdomAspect.DAOIST_HARMONY)
        aspects.append(WisdomAspect.VEDANTIC_UNITY)

    if context.recipient_capacity == "advanced":
        aspects.append(WisdomAspect.PHENOMENOLOGICAL_PRESENCE)

    return list(set(aspects))
```

#### 2. Wisdom Teaching Retrieval

```python
def _get_philosophical_teaching(self, aspect: WisdomAspect) -> str:
    """Hook: Retrieve teaching for philosophical wisdom aspect."""
    aspect_to_category = {
        WisdomAspect.STOIC_EQUANIMITY: "stoicism",
        WisdomAspect.ARISTOTELIAN_VIRTUE: "aristotelian_virtue",
        WisdomAspect.PHENOMENOLOGICAL_PRESENCE: "phenomenology",
        WisdomAspect.EXISTENTIAL_AUTHENTICITY: "existentialism",
        WisdomAspect.DAOIST_HARMONY: "daoist",
        WisdomAspect.CONFUCIAN_HARMONY: "confucian",
        WisdomAspect.VEDANTIC_UNITY: "vedantic",
        WisdomAspect.PRAGMATIC_EFFECTIVENESS: "pragmatic",
    }

    category = aspect_to_category.get(aspect)
    if category and category in self.wisdom_teachings_database:
        teachings = self.wisdom_teachings_database[category]
        return random.choice(teachings) if teachings else ""

    return ""
```

#### 3. Skillful Means Extension

```python
def _get_philosophical_skillful_means(self, context: EngagementContext) -> Dict:
    """Hook: Provide tradition-specific skillful means."""
    means_by_tradition = {
        "stoicism": {
            "approach": "dichotomy_of_control",
            "methods": ["reframing", "premeditation", "reflection"],
            "language_style": "firm_compassionate"
        },
        "existentialism": {
            "approach": "authentic_confrontation",
            "methods": ["ownership", "choice_emphasis", "meaning_creation"],
            "language_style": "honest_direct"
        },
        "phenomenology": {
            "approach": "direct_investigation",
            "methods": ["epoché", "description", "essence_seeking"],
            "language_style": "attentive_precise"
        },
    }

    # Select based on context
    if context.emotional_state == "distressed":
        return means_by_tradition["stoicism"]
    elif context.current_need == "meaning_seeking":
        return means_by_tradition["existentialism"]

    return means_by_tradition.get("phenomenology", {})
```
