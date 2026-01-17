# Temporal Dynamics

## Overview

This document describes how philosophical understanding matures over time within Form 28. The system tracks, measures, and facilitates the deepening of philosophical knowledge through continuous engagement, research, and synthesis.

---

## Maturity Model

### Maturity Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| NASCENT | 0.0-0.2 | Basic awareness, surface definitions |
| DEVELOPING | 0.2-0.4 | Growing understanding, some connections |
| COMPETENT | 0.4-0.6 | Solid grasp, can explain and apply |
| PROFICIENT | 0.6-0.8 | Deep understanding, nuanced awareness |
| MASTERFUL | 0.8-1.0 | Expert-level, cross-tradition synthesis |

### Maturity Factors

Overall maturity is calculated from:

```python
def calculate_overall_maturity() -> float:
    # Tradition depth (40% weight)
    tradition_avg = mean(traditions_depth.values())

    # Domain coverage (30% weight)
    domain_avg = mean(domains_coverage.values())

    # Synthesis capability (30% weight)
    synthesis_factor = min(1.0, cross_tradition_syntheses / 100)

    return (
        tradition_avg * 0.4 +
        domain_avg * 0.3 +
        synthesis_factor * 0.3
    )
```

---

## Concept-Level Maturity

### Individual Concept Scoring

Each concept has its own maturity score based on:

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

### Definition Completeness (0-1)

- Length > 200 chars: +0.5
- Extended description present: +0.3
- Key terms defined: +0.2

### Relationships (0-1)

- Related concepts >= 5: +0.5
- Opposed concepts identified: +0.2
- Prerequisites mapped: +0.3

### Attribution (0-1)

- Key figures >= 2: +0.4
- Primary texts >= 2: +0.4
- Historical context: +0.2

### Arguments (0-1)

- Key arguments >= 2: +0.4
- Counter-arguments >= 1: +0.3
- Argument analysis: +0.3

### Source Quality (0-1)

- SEP source: +0.5
- PhilPapers citations: +0.3
- Multiple sources: +0.2

### Research Depth (0-1)

- Research sessions >= 3: +0.5
- Deep research >= 1: +0.3
- Recent update: +0.2

---

## Tradition-Level Maturity

### Tracking Tradition Depth

```python
traditions_depth: Dict[str, float] = {
    "stoicism": 0.0,
    "aristotelian": 0.0,
    "kantian": 0.0,
    "phenomenology": 0.0,
    "existentialism": 0.0,
    "buddhist_zen": 0.0,
    "daoist": 0.0,
    "vedantic_advaita": 0.0,
    # ... all traditions
}
```

### Depth Increment Formula

```python
def update_tradition_depth(tradition: str, concept_added: bool):
    current = traditions_depth.get(tradition, 0.0)

    # Diminishing returns as depth increases
    increment = 0.01 * (1 - current)

    if concept_added:
        increment *= 2  # Bonus for new content

    traditions_depth[tradition] = min(1.0, current + increment)
```

### Tradition Maturity Criteria

| Depth | Requirements |
|-------|--------------|
| 0.2+ | >= 10 concepts indexed |
| 0.4+ | >= 25 concepts, key figures |
| 0.6+ | >= 50 concepts, primary texts |
| 0.8+ | >= 100 concepts, cross-references |
| 1.0 | Comprehensive coverage, synthesis |

---

## Domain Coverage

### Tracking Domain Coverage

```python
domains_coverage: Dict[str, float] = {
    "metaphysics": 0.0,
    "epistemology": 0.0,
    "ethics": 0.0,
    "philosophy_of_mind": 0.0,
    "consciousness_studies": 0.0,
    # ... all domains
}
```

### Coverage Assessment

```python
def assess_domain_coverage(domain: str) -> float:
    concepts_in_domain = count_concepts_by_domain(domain)
    traditions_covering = count_traditions_covering_domain(domain)

    concept_factor = min(1.0, concepts_in_domain / 50)
    tradition_factor = min(1.0, traditions_covering / 10)

    return concept_factor * 0.6 + tradition_factor * 0.4
```

---

## Maturation Events

### Growth Triggers

| Event | Maturity Impact |
|-------|-----------------|
| Concept query | +0.001 tradition depth |
| Concept added | +0.01 tradition depth |
| Synthesis performed | +0.005 all involved traditions |
| Research completed | +0.002 target tradition |
| Deep research | +0.01 target tradition |

### Decay (Optional)

For systems requiring decay:

```python
def apply_decay(days_since_access: int):
    # Slow decay for unused knowledge
    decay_factor = 0.999 ** days_since_access

    for tradition in traditions_depth:
        traditions_depth[tradition] *= decay_factor
```

---

## Synthesis Maturation

### Cross-Tradition Synthesis Tracking

```python
@dataclass
class SynthesisRecord:
    synthesis_id: str
    topic: str
    traditions: List[str]
    quality_score: float
    created_at: datetime
```

### Synthesis Quality Factors

```python
def assess_synthesis_quality(synthesis: CrossTraditionSynthesis) -> float:
    # Fidelity to each tradition
    avg_fidelity = mean(synthesis.fidelity_scores.values())

    # Coherence of synthesis statement
    coherence = synthesis.coherence_score

    # Number of traditions integrated
    breadth_bonus = min(0.2, len(synthesis.traditions) * 0.05)

    return avg_fidelity * 0.5 + coherence * 0.3 + breadth_bonus
```

### Synthesis Impact on Maturity

```python
def apply_synthesis_maturity(synthesis: CrossTraditionSynthesis):
    quality = assess_synthesis_quality(synthesis)

    for tradition in synthesis.traditions:
        # Bonus proportional to quality
        increment = 0.005 * quality
        traditions_depth[tradition.value] += increment

    # Global synthesis counter
    maturity_state.cross_tradition_syntheses += 1
```

---

## Nuance Development

### Nuance Metrics

As maturity increases, the system develops:

1. **Disambiguation**: Recognizing distinct uses of terms
2. **Context Sensitivity**: Adjusting interpretation to context
3. **Internal Debate Awareness**: Knowing disagreements within traditions
4. **Historical Sensitivity**: Understanding conceptual evolution
5. **Cross-Cultural Awareness**: Avoiding false equivalences

### Nuance Scoring

```python
def calculate_nuance_score() -> float:
    disambiguation = assess_disambiguation_capability()
    context_sensitivity = assess_context_sensitivity()
    debate_awareness = assess_internal_debate_awareness()
    historical_sensitivity = assess_historical_sensitivity()
    cross_cultural = assess_cross_cultural_awareness()

    return (
        disambiguation * 0.2 +
        context_sensitivity * 0.2 +
        debate_awareness * 0.2 +
        historical_sensitivity * 0.2 +
        cross_cultural * 0.2
    )
```

---

## Maturity History

### Tracking Changes Over Time

```python
@dataclass
class MaturitySnapshot:
    timestamp: datetime
    overall_maturity: float
    maturity_level: MaturityLevel
    traditions_depth: Dict[str, float]
    domains_coverage: Dict[str, float]
    concepts_count: int
    syntheses_count: int
```

### Periodic Snapshots

```python
async def take_maturity_snapshot():
    snapshot = MaturitySnapshot(
        timestamp=datetime.now(timezone.utc),
        overall_maturity=maturity_state.get_overall_maturity(),
        maturity_level=maturity_state.get_maturity_level(),
        traditions_depth=dict(maturity_state.traditions_depth),
        domains_coverage=dict(maturity_state.domains_coverage),
        concepts_count=len(concept_index),
        syntheses_count=maturity_state.cross_tradition_syntheses,
    )

    maturity_state.maturity_history.append(snapshot.to_dict())

    # Keep last 100 snapshots
    if len(maturity_state.maturity_history) > 100:
        maturity_state.maturity_history = maturity_state.maturity_history[-100:]
```

---

## Maturation Guidance

### Gap-Directed Growth

```python
async def identify_growth_opportunities() -> List[GrowthOpportunity]:
    opportunities = []

    # Find shallow traditions
    for tradition, depth in traditions_depth.items():
        if depth < 0.3:
            opportunities.append(GrowthOpportunity(
                type="tradition_depth",
                target=tradition,
                current=depth,
                goal=0.5,
                suggested_action=f"Research core concepts of {tradition}"
            ))

    # Find uncovered domains
    for domain, coverage in domains_coverage.items():
        if coverage < 0.3:
            opportunities.append(GrowthOpportunity(
                type="domain_coverage",
                target=domain,
                current=coverage,
                goal=0.5,
                suggested_action=f"Expand {domain} across traditions"
            ))

    # Synthesis opportunities
    if maturity_state.cross_tradition_syntheses < 10:
        opportunities.append(GrowthOpportunity(
            type="synthesis_practice",
            target="cross_tradition",
            current=maturity_state.cross_tradition_syntheses,
            goal=10,
            suggested_action="Practice cross-tradition synthesis"
        ))

    return sorted(opportunities, key=lambda x: x.current)
```
