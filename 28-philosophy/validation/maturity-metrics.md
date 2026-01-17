# Maturity Metrics

## Overview

This document defines the metrics, measurement approaches, and evaluation criteria for tracking the maturation of philosophical understanding within Form 28. The maturity system reflects how philosophical knowledge deepens and becomes more nuanced through continuous engagement, research, and synthesis.

---

## Maturity Philosophy

The maturity model is inspired by the philosophical concept of *phronesis* (practical wisdom) and the contemplative traditions' notion of deepening understanding. True philosophical maturity is not merely the accumulation of facts but the development of:

1. **Depth**: Understanding the subtleties within a tradition
2. **Breadth**: Familiarity across multiple traditions and domains
3. **Integration**: Ability to synthesize across traditions
4. **Nuance**: Recognition of internal debates and context-sensitivity
5. **Wisdom**: Application of philosophy to lived experience

---

## Overall Maturity Score

### Calculation

```python
def calculate_overall_maturity() -> float:
    """
    Overall maturity is a weighted combination of:
    - Tradition depth (how deeply we know each tradition)
    - Domain coverage (how well we cover philosophical domains)
    - Synthesis capability (how well we can integrate traditions)
    """
    # Tradition depth average (40% weight)
    tradition_scores = list(maturity_state.traditions_depth.values())
    tradition_avg = sum(tradition_scores) / len(tradition_scores)

    # Domain coverage average (30% weight)
    domain_scores = list(maturity_state.domains_coverage.values())
    domain_avg = sum(domain_scores) / len(domain_scores)

    # Synthesis factor (30% weight)
    # Based on successful cross-tradition syntheses
    synthesis_count = maturity_state.cross_tradition_syntheses
    synthesis_factor = min(1.0, synthesis_count / 100)  # Cap at 100 syntheses

    return (
        tradition_avg * 0.4 +
        domain_avg * 0.3 +
        synthesis_factor * 0.3
    )
```

### Maturity Levels

| Level | Score Range | Description | Characteristics |
|-------|-------------|-------------|-----------------|
| **NASCENT** | 0.0 - 0.2 | Beginning awareness | Basic definitions, surface understanding |
| **DEVELOPING** | 0.2 - 0.4 | Growing familiarity | Some connections, key figures known |
| **COMPETENT** | 0.4 - 0.6 | Solid foundation | Can explain and apply, knows arguments |
| **PROFICIENT** | 0.6 - 0.8 | Deep understanding | Nuanced awareness, internal debates |
| **MASTERFUL** | 0.8 - 1.0 | Expert integration | Cross-tradition synthesis, wisdom application |

---

## Tradition Depth Metrics

### Per-Tradition Scoring

Each philosophical tradition is tracked independently:

```python
traditions_depth: Dict[str, float] = {
    # Western
    "stoicism": 0.0,
    "aristotelian": 0.0,
    "platonic": 0.0,
    "kantian": 0.0,
    "phenomenology": 0.0,
    "existentialism": 0.0,
    "analytic": 0.0,
    "pragmatism": 0.0,
    # Eastern
    "buddhist_theravada": 0.0,
    "buddhist_mahayana": 0.0,
    "buddhist_zen": 0.0,
    "daoist": 0.0,
    "confucian": 0.0,
    "vedantic_advaita": 0.0,
    # ... all traditions
}
```

### Depth Factors

| Factor | Weight | Measurement |
|--------|--------|-------------|
| Concepts indexed | 40% | Count of concepts in tradition |
| Concept maturity | 25% | Average concept maturity score |
| Key figures known | 15% | Coverage of major philosophers |
| Primary texts | 10% | Number of primary texts indexed |
| Cross-references | 10% | Internal relationship density |

### Depth Calculation

```python
def calculate_tradition_depth(tradition: PhilosophicalTradition) -> float:
    concepts = get_concepts_by_tradition(tradition)
    figures = get_figures_by_tradition(tradition)
    texts = get_texts_by_tradition(tradition)

    # Concept coverage (40%)
    concept_coverage = min(1.0, len(concepts) / TRADITION_CONCEPT_TARGETS[tradition])

    # Concept maturity (25%)
    if concepts:
        concept_maturity_avg = sum(c.maturity_level for c in concepts) / len(concepts)
    else:
        concept_maturity_avg = 0.0

    # Key figures (15%)
    expected_figures = TRADITION_KEY_FIGURES[tradition]
    figure_coverage = len(set(figures) & expected_figures) / len(expected_figures)

    # Primary texts (10%)
    text_coverage = min(1.0, len(texts) / TRADITION_TEXT_TARGETS[tradition])

    # Cross-references (10%)
    if concepts:
        avg_relationships = sum(len(c.related_concepts) for c in concepts) / len(concepts)
        relationship_density = min(1.0, avg_relationships / 5)  # Target 5 per concept
    else:
        relationship_density = 0.0

    return (
        concept_coverage * 0.40 +
        concept_maturity_avg * 0.25 +
        figure_coverage * 0.15 +
        text_coverage * 0.10 +
        relationship_density * 0.10
    )
```

### Tradition Targets

| Tradition | Concept Target | Key Figure Target | Text Target |
|-----------|---------------|-------------------|-------------|
| Stoicism | 50 | 6 (Zeno, Cleanthes, Chrysippus, Seneca, Epictetus, Marcus Aurelius) | 10 |
| Aristotelian | 80 | 5 (Aristotle, Aquinas, MacIntyre, Anscombe, Foot) | 15 |
| Kantian | 60 | 4 (Kant, Rawls, Korsgaard, O'Neill) | 12 |
| Phenomenology | 70 | 6 (Husserl, Heidegger, Merleau-Ponty, Sartre, Levinas, Henry) | 15 |
| Buddhist Zen | 60 | 5 (Bodhidharma, Huineng, Dogen, Hakuin, Suzuki) | 10 |
| Vedantic Advaita | 50 | 4 (Shankara, Ramana Maharshi, Nisargadatta, Vivekananda) | 8 |

---

## Domain Coverage Metrics

### Per-Domain Scoring

```python
domains_coverage: Dict[str, float] = {
    "metaphysics": 0.0,
    "epistemology": 0.0,
    "ethics": 0.0,
    "aesthetics": 0.0,
    "logic": 0.0,
    "philosophy_of_mind": 0.0,
    "philosophy_of_language": 0.0,
    "political_philosophy": 0.0,
    "philosophy_of_science": 0.0,
    "philosophy_of_religion": 0.0,
    "consciousness_studies": 0.0,
}
```

### Coverage Calculation

```python
def calculate_domain_coverage(domain: PhilosophicalDomain) -> float:
    concepts = get_concepts_by_domain(domain)
    traditions = set(c.tradition for c in concepts)

    # Concept count factor (60%)
    concept_factor = min(1.0, len(concepts) / DOMAIN_CONCEPT_TARGETS[domain])

    # Tradition diversity (40%)
    tradition_factor = min(1.0, len(traditions) / 10)  # Target 10 traditions per domain

    return concept_factor * 0.6 + tradition_factor * 0.4
```

---

## Concept-Level Metrics

### Individual Concept Maturity

Each concept has its own maturity score:

```python
def calculate_concept_maturity(concept: PhilosophicalConcept) -> float:
    scores = {
        "definition_completeness": assess_definition(concept),
        "relationships": assess_relationships(concept),
        "attribution": assess_attribution(concept),
        "arguments": assess_arguments(concept),
        "source_quality": assess_sources(concept),
        "research_depth": assess_research(concept),
    }

    weights = {
        "definition_completeness": 0.20,
        "relationships": 0.15,
        "attribution": 0.15,
        "arguments": 0.20,
        "source_quality": 0.15,
        "research_depth": 0.15,
    }

    return sum(scores[k] * weights[k] for k in scores)
```

### Metric Definitions

#### Definition Completeness (0-1)

```python
def assess_definition(concept: PhilosophicalConcept) -> float:
    score = 0.0

    # Length threshold
    if len(concept.definition) > 200:
        score += 0.5
    elif len(concept.definition) > 100:
        score += 0.3

    # Extended description
    if concept.extended_description:
        score += 0.3

    # Key terms defined
    if concept.key_terms:
        score += 0.2

    return min(1.0, score)
```

#### Relationships (0-1)

```python
def assess_relationships(concept: PhilosophicalConcept) -> float:
    score = 0.0

    # Related concepts
    if len(concept.related_concepts) >= 5:
        score += 0.5
    else:
        score += len(concept.related_concepts) * 0.1

    # Opposed concepts
    if concept.opposed_concepts:
        score += 0.2

    # Prerequisites
    if concept.prerequisites:
        score += 0.3

    return min(1.0, score)
```

#### Attribution (0-1)

```python
def assess_attribution(concept: PhilosophicalConcept) -> float:
    score = 0.0

    # Key figures
    if len(concept.key_figures) >= 2:
        score += 0.4
    elif len(concept.key_figures) == 1:
        score += 0.2

    # Primary texts
    if len(concept.primary_texts) >= 2:
        score += 0.4
    elif len(concept.primary_texts) == 1:
        score += 0.2

    # Historical context
    if concept.historical_context:
        score += 0.2

    return min(1.0, score)
```

#### Arguments (0-1)

```python
def assess_arguments(concept: PhilosophicalConcept) -> float:
    score = 0.0

    # Key arguments
    if len(concept.key_arguments) >= 2:
        score += 0.4
    elif len(concept.key_arguments) == 1:
        score += 0.2

    # Counter-arguments
    if concept.counter_arguments:
        score += 0.3

    # Argument analysis
    if concept.argument_analysis:
        score += 0.3

    return min(1.0, score)
```

#### Source Quality (0-1)

```python
def assess_sources(concept: PhilosophicalConcept) -> float:
    score = 0.0
    sources = concept.sources

    # Stanford Encyclopedia source
    if any("plato.stanford.edu" in s for s in sources):
        score += 0.5

    # PhilPapers citations
    if any("philpapers.org" in s for s in sources):
        score += 0.3

    # Multiple sources
    if len(sources) >= 3:
        score += 0.2

    return min(1.0, score)
```

---

## Synthesis Metrics

### Synthesis Quality Score

```python
def assess_synthesis_quality(synthesis: CrossTraditionSynthesis) -> float:
    # Average fidelity to source traditions (50%)
    avg_fidelity = sum(synthesis.fidelity_scores.values()) / len(synthesis.fidelity_scores)

    # Coherence of synthesis (30%)
    coherence = synthesis.coherence_score

    # Breadth bonus (20%)
    # More traditions = harder to synthesize well
    breadth_bonus = min(0.2, len(synthesis.traditions) * 0.05)

    return avg_fidelity * 0.5 + coherence * 0.3 + breadth_bonus
```

### Fidelity Assessment

Fidelity measures how well the synthesis represents each tradition:

```python
def assess_fidelity(synthesis_statement: str, tradition: PhilosophicalTradition) -> float:
    """
    Check if synthesis accurately represents tradition.
    Uses both keyword matching and semantic similarity.
    """
    # Get tradition's core concepts
    core_concepts = get_core_concepts(tradition)

    # Keyword presence
    keyword_score = count_tradition_keywords(synthesis_statement, tradition) / 10

    # Semantic alignment
    synthesis_embedding = generate_embedding(synthesis_statement)
    tradition_embedding = get_tradition_embedding(tradition)
    semantic_score = cosine_similarity(synthesis_embedding, tradition_embedding)

    # No contradiction check
    contradictions = check_tradition_contradictions(synthesis_statement, tradition)
    contradiction_penalty = len(contradictions) * 0.2

    return max(0.0, (keyword_score * 0.3 + semantic_score * 0.7) - contradiction_penalty)
```

---

## Nuance Metrics

### Nuance Development Score

As maturity increases, the system develops nuanced understanding:

```python
def calculate_nuance_score() -> float:
    """
    Nuance score measures:
    1. Disambiguation capability
    2. Context sensitivity
    3. Internal debate awareness
    4. Historical sensitivity
    5. Cross-cultural awareness
    """
    scores = {
        "disambiguation": assess_disambiguation_capability(),
        "context_sensitivity": assess_context_sensitivity(),
        "debate_awareness": assess_internal_debate_awareness(),
        "historical_sensitivity": assess_historical_sensitivity(),
        "cross_cultural": assess_cross_cultural_awareness(),
    }

    return sum(scores.values()) / len(scores)
```

### Nuance Components

#### Disambiguation Capability

Recognizing distinct uses of the same term:

```python
def assess_disambiguation_capability() -> float:
    # Count concepts with multiple senses tracked
    multi_sense_concepts = [c for c in concept_index.values()
                           if len(c.sense_variations) > 1]

    return min(1.0, len(multi_sense_concepts) / 50)
```

#### Context Sensitivity

Adjusting interpretation based on context:

```python
def assess_context_sensitivity() -> float:
    # Measure how well queries adapt to context
    test_queries = [
        ("What is virtue?", "aristotelian"),  # Should emphasize habit/excellence
        ("What is virtue?", "stoicism"),      # Should emphasize wisdom/nature
        ("What is virtue?", "confucian"),     # Should emphasize ren/li
    ]

    correct = 0
    for query, context in test_queries:
        result = query_concept(query, tradition_context=context)
        if result["primary_tradition"].value == context:
            correct += 1

    return correct / len(test_queries)
```

#### Internal Debate Awareness

Knowledge of disagreements within traditions:

```python
def assess_internal_debate_awareness() -> float:
    # Count concepts with internal_debates field populated
    concepts_with_debates = [c for c in concept_index.values()
                            if c.internal_debates]

    return min(1.0, len(concepts_with_debates) / 30)
```

---

## Growth Tracking

### Growth Events

| Event | Impact |
|-------|--------|
| Concept query | +0.001 tradition depth |
| New concept added | +0.01 tradition depth |
| Synthesis performed | +0.005 all involved traditions |
| Research completed | +0.002 target tradition |
| Deep research | +0.01 target tradition |

### Growth Recording

```python
@dataclass
class GrowthEvent:
    event_type: str
    tradition: Optional[str]
    domain: Optional[str]
    magnitude: float
    timestamp: datetime
    details: Dict[str, Any]

def record_growth(event: GrowthEvent):
    maturity_state.growth_history.append(event)

    # Apply growth
    if event.tradition:
        current = maturity_state.traditions_depth.get(event.tradition, 0.0)
        # Diminishing returns
        increment = event.magnitude * (1 - current)
        maturity_state.traditions_depth[event.tradition] = min(1.0, current + increment)

    # Recalculate overall
    maturity_state.overall_maturity = calculate_overall_maturity()
    maturity_state.maturity_level = get_maturity_level(maturity_state.overall_maturity)
```

---

## Reporting

### Maturity Dashboard

```python
async def generate_maturity_report() -> MaturityReport:
    return MaturityReport(
        overall_maturity=maturity_state.overall_maturity,
        maturity_level=maturity_state.maturity_level,

        # Tradition breakdown
        traditions={
            t: {
                "depth": maturity_state.traditions_depth[t],
                "concept_count": len(get_concepts_by_tradition(t)),
                "growth_30d": calculate_growth(t, days=30),
            }
            for t in PhilosophicalTradition
        },

        # Domain breakdown
        domains={
            d: {
                "coverage": maturity_state.domains_coverage[d],
                "concept_count": len(get_concepts_by_domain(d)),
            }
            for d in PhilosophicalDomain
        },

        # Synthesis stats
        synthesis_count=maturity_state.cross_tradition_syntheses,
        synthesis_quality_avg=calculate_avg_synthesis_quality(),

        # Nuance score
        nuance_score=calculate_nuance_score(),

        # Growth trajectory
        growth_rate_30d=calculate_overall_growth(days=30),

        # Top gaps
        gaps=identify_growth_opportunities()[:5],

        # Timestamp
        generated_at=datetime.now(timezone.utc),
    )
```

### Gap Analysis

```python
async def identify_growth_opportunities() -> List[GrowthOpportunity]:
    opportunities = []

    # Shallow traditions
    for tradition, depth in maturity_state.traditions_depth.items():
        if depth < 0.3:
            opportunities.append(GrowthOpportunity(
                type="tradition_depth",
                target=tradition,
                current=depth,
                goal=0.5,
                suggested_action=f"Research core concepts of {tradition}",
                priority=(0.5 - depth) * 10
            ))

    # Uncovered domains
    for domain, coverage in maturity_state.domains_coverage.items():
        if coverage < 0.3:
            opportunities.append(GrowthOpportunity(
                type="domain_coverage",
                target=domain,
                current=coverage,
                goal=0.5,
                suggested_action=f"Expand {domain} across traditions",
                priority=(0.5 - coverage) * 8
            ))

    # Low synthesis count
    if maturity_state.cross_tradition_syntheses < 10:
        opportunities.append(GrowthOpportunity(
            type="synthesis_practice",
            target="cross_tradition",
            current=maturity_state.cross_tradition_syntheses,
            goal=10,
            suggested_action="Practice cross-tradition synthesis",
            priority=5
        ))

    return sorted(opportunities, key=lambda x: -x.priority)
```

---

## Visualization

### Maturity Radar Chart

```
                    Metaphysics
                         │
            Stoicism     │      Aristotelian
                 ╲       │       ╱
                  ╲      │      ╱
                   ╲     │     ╱
     Ethics ───────── ◉ ─────────── Epistemology
                   ╱     │     ╲
                  ╱      │      ╲
                 ╱       │       ╲
           Phenomenology │     Kantian
                         │
                    Existentialism
```

### Growth Timeline

```
Maturity
1.0 ┤
    │                                    ╭──────
0.8 ┤                              ╭─────╯
    │                        ╭─────╯
0.6 ┤                  ╭─────╯
    │            ╭─────╯
0.4 ┤      ╭─────╯
    │╭─────╯
0.2 ┤
    │
0.0 ┼────────────────────────────────────────────
    Day 0    30      60      90     120    150
```
