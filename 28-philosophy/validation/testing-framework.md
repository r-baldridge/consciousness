# Testing Framework

## Overview

This document describes the comprehensive testing approach for Form 28 (Philosophical Consciousness), including unit tests, integration tests, philosophical accuracy validation, and performance benchmarks.

---

## Test Structure

```
consciousness/28-philosophy/tests/
├── unit/
│   ├── test_data_models.py
│   ├── test_concept_index.py
│   ├── test_embedding_generation.py
│   ├── test_reasoning_engine.py
│   └── test_maturity_tracking.py
├── integration/
│   ├── test_form_10_integration.py
│   ├── test_form_27_integration.py
│   ├── test_message_bus.py
│   ├── test_research_agent.py
│   └── test_wisdom_selection.py
├── philosophical/
│   ├── test_tradition_accuracy.py
│   ├── test_concept_definitions.py
│   ├── test_synthesis_quality.py
│   └── test_nuance_preservation.py
├── performance/
│   ├── test_query_latency.py
│   ├── test_embedding_throughput.py
│   └── test_scalability.py
├── fixtures/
│   ├── concepts.json
│   ├── figures.json
│   └── test_queries.json
└── conftest.py
```

---

## Unit Tests

### Data Model Tests

**File**: `test_data_models.py`

```python
import pytest
from interface.philosophical_consciousness_interface import (
    PhilosophicalTradition,
    PhilosophicalDomain,
    PhilosophicalConcept,
    PhilosophicalFigure,
    CrossTraditionSynthesis,
)

class TestPhilosophicalTradition:
    def test_all_traditions_have_values(self):
        """Verify all traditions have unique string values."""
        values = [t.value for t in PhilosophicalTradition]
        assert len(values) == len(set(values))

    def test_tradition_categories(self):
        """Verify Western/Eastern categorization."""
        western = [
            PhilosophicalTradition.STOICISM,
            PhilosophicalTradition.ARISTOTELIAN,
            PhilosophicalTradition.KANTIAN,
        ]
        eastern = [
            PhilosophicalTradition.BUDDHIST_ZEN,
            PhilosophicalTradition.DAOIST,
            PhilosophicalTradition.VEDANTIC_ADVAITA,
        ]
        for t in western:
            assert t in PhilosophicalTradition
        for t in eastern:
            assert t in PhilosophicalTradition

class TestPhilosophicalConcept:
    def test_concept_creation(self):
        """Test basic concept instantiation."""
        concept = PhilosophicalConcept(
            concept_id="test_001",
            name="Test Concept",
            tradition=PhilosophicalTradition.STOICISM,
            domain=PhilosophicalDomain.ETHICS,
            definition="A test definition",
            related_concepts=[],
            key_figures=[],
            primary_texts=[],
        )
        assert concept.concept_id == "test_001"
        assert concept.tradition == PhilosophicalTradition.STOICISM

    def test_embedding_text_generation(self):
        """Test to_embedding_text produces valid output."""
        concept = PhilosophicalConcept(
            concept_id="amor_fati",
            name="Amor Fati",
            tradition=PhilosophicalTradition.STOICISM,
            domain=PhilosophicalDomain.ETHICS,
            definition="Love of fate",
            related_concepts=["acceptance", "will"],
            key_figures=["Nietzsche", "Marcus Aurelius"],
            primary_texts=[],
        )
        text = concept.to_embedding_text()
        assert "Amor Fati" in text
        assert "Stoicism" in text or "stoicism" in text.lower()
        assert "Love of fate" in text

    def test_required_fields(self):
        """Verify required fields raise on None."""
        with pytest.raises(TypeError):
            PhilosophicalConcept(
                concept_id=None,  # Required
                name="Test",
                tradition=PhilosophicalTradition.STOICISM,
            )

class TestCrossTraditionSynthesis:
    def test_synthesis_fidelity_bounds(self):
        """Fidelity scores must be 0.0-1.0."""
        synthesis = CrossTraditionSynthesis(
            topic="consciousness",
            traditions=[
                PhilosophicalTradition.PHENOMENOLOGY,
                PhilosophicalTradition.BUDDHIST_ZEN,
            ],
            synthesis_statement="Test synthesis",
            fidelity_scores={
                "phenomenology": 0.85,
                "buddhist_zen": 0.80,
            },
            coherence_score=0.75,
        )
        for score in synthesis.fidelity_scores.values():
            assert 0.0 <= score <= 1.0
```

### Concept Index Tests

**File**: `test_concept_index.py`

```python
import pytest
from interface.philosophical_consciousness_interface import (
    PhilosophicalConsciousnessInterface,
)

@pytest.fixture
def interface():
    """Create interface with test data."""
    return PhilosophicalConsciousnessInterface(
        use_test_data=True
    )

class TestConceptIndex:
    def test_index_initialization(self, interface):
        """Verify index loads correctly."""
        assert len(interface.concept_index) > 0

    def test_concept_retrieval_by_id(self, interface):
        """Retrieve concept by ID."""
        concept = interface.get_concept_by_id("stoic_apatheia")
        assert concept is not None
        assert concept.tradition == PhilosophicalTradition.STOICISM

    def test_concept_not_found(self, interface):
        """Non-existent concept returns None."""
        concept = interface.get_concept_by_id("nonexistent_123")
        assert concept is None

    def test_filter_by_tradition(self, interface):
        """Filter concepts by tradition."""
        stoic_concepts = interface.get_concepts_by_tradition(
            PhilosophicalTradition.STOICISM
        )
        for concept in stoic_concepts:
            assert concept.tradition == PhilosophicalTradition.STOICISM

    def test_filter_by_domain(self, interface):
        """Filter concepts by domain."""
        ethics_concepts = interface.get_concepts_by_domain(
            PhilosophicalDomain.ETHICS
        )
        for concept in ethics_concepts:
            assert concept.domain == PhilosophicalDomain.ETHICS
```

### Embedding Tests

**File**: `test_embedding_generation.py`

```python
import pytest
import numpy as np

class TestEmbeddingGeneration:
    @pytest.fixture
    def interface(self):
        return PhilosophicalConsciousnessInterface()

    async def test_embedding_dimensions(self, interface):
        """Verify embedding has correct dimensions."""
        text = "The nature of consciousness"
        embedding = await interface.generate_embedding(text)
        assert len(embedding) == 768

    async def test_embedding_normalization(self, interface):
        """Verify L2 normalization."""
        text = "Metaphysical inquiry"
        embedding = await interface.generate_embedding(text)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    async def test_similar_texts_close_embeddings(self, interface):
        """Similar concepts have similar embeddings."""
        emb1 = await interface.generate_embedding("virtue ethics")
        emb2 = await interface.generate_embedding("ethical virtue")
        emb3 = await interface.generate_embedding("quantum mechanics")

        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13  # Related more similar than unrelated

    async def test_embedding_caching(self, interface):
        """Verify embeddings are cached."""
        text = "Cache test text"
        emb1 = await interface.generate_embedding(text)
        emb2 = await interface.generate_embedding(text)
        assert emb1 == emb2  # Same object from cache
```

### Reasoning Engine Tests

**File**: `test_reasoning_engine.py`

```python
import pytest
from interface.philosophical_reasoning_engine import (
    PhilosophicalReasoningEngine,
    ArgumentStructure,
    ReasoningMode,
)

class TestArgumentAnalysis:
    @pytest.fixture
    def engine(self):
        return PhilosophicalReasoningEngine()

    def test_identify_premises(self, engine):
        """Identify premises in argument text."""
        argument = """
        All humans are mortal.
        Socrates is a human.
        Therefore, Socrates is mortal.
        """
        analysis = engine.analyze_argument(argument)
        assert len(analysis.premises) >= 2

    def test_identify_conclusion(self, engine):
        """Identify conclusion in argument."""
        argument = "If P then Q. P. Therefore Q."
        analysis = engine.analyze_argument(argument)
        assert analysis.conclusion is not None
        assert "Q" in analysis.conclusion

    def test_identify_argument_type(self, engine):
        """Classify argument type correctly."""
        deductive = "All A are B. All B are C. Therefore all A are C."
        analysis = engine.analyze_argument(deductive)
        assert analysis.argument_type == "deductive"

class TestDialecticalSynthesis:
    @pytest.fixture
    def engine(self):
        return PhilosophicalReasoningEngine()

    def test_thesis_antithesis_synthesis(self, engine):
        """Generate synthesis from thesis and antithesis."""
        result = engine.dialectical_synthesis(
            thesis="Free will exists absolutely",
            antithesis="All actions are determined",
        )
        assert result.synthesis is not None
        assert result.preserves_thesis_elements
        assert result.preserves_antithesis_elements

    def test_synthesis_acknowledges_tension(self, engine):
        """Synthesis should acknowledge unresolved tensions."""
        result = engine.dialectical_synthesis(
            thesis="Knowledge requires certainty",
            antithesis="Certainty is impossible",
        )
        assert len(result.unresolved_tensions) > 0
```

---

## Integration Tests

### Form 10 Integration

**File**: `test_form_10_integration.py`

```python
import pytest

class TestForm10Integration:
    @pytest.fixture
    async def interfaces(self):
        phil_interface = PhilosophicalConsciousnessInterface()
        # Mock Form 10 interface
        form_10 = MockForm10Interface()
        phil_interface.form_10_interface = form_10
        return phil_interface, form_10

    async def test_store_in_memory(self, interfaces):
        """Philosophical concepts stored in Form 10."""
        phil, form_10 = interfaces
        concept = phil.concept_index["stoic_apatheia"]

        await phil.store_in_memory(concept, "test context")

        assert form_10.stored_memories[-1]["memory_type"] == "philosophical"
        assert form_10.stored_memories[-1]["embedding"] is not None

    async def test_retrieve_from_memory(self, interfaces):
        """Retrieve philosophical memories from Form 10."""
        phil, form_10 = interfaces

        # Pre-populate mock
        form_10.mock_memories = [
            {"content": "Stoic acceptance", "similarity": 0.9}
        ]

        results = await phil.retrieve_from_memory("acceptance")
        assert len(results) > 0

    async def test_cross_form_query(self, interfaces):
        """Query spans philosophical index and Form 10 memory."""
        phil, form_10 = interfaces

        result = await phil.query_concept(
            query="nature of self",
            include_memory_context=True
        )

        assert result["concepts"] is not None
        assert "memory_context" in result
```

### Form 27 Integration

**File**: `test_form_27_integration.py`

```python
import pytest
from interface.philosophical_consciousness_interface import ProcessingMode

class TestForm27Integration:
    @pytest.fixture
    async def interfaces(self):
        phil_interface = PhilosophicalConsciousnessInterface()
        non_dual = MockNonDualInterface()
        phil_interface.non_dual_interface = non_dual
        return phil_interface, non_dual

    async def test_meditation_enhanced_query(self, interfaces):
        """Query processing enhanced by meditation state."""
        phil, non_dual = interfaces

        result = await phil.process_with_meditation_mode(
            query="nature of consciousness",
            processing_mode=ProcessingMode.ZAZEN,
            mind_level=MindLevel.BODHI_MENTE
        )

        assert result["coordination"] is not None
        assert phil._reasoning_mode == "phenomenological"

    async def test_karmic_seed_integration(self, interfaces):
        """Philosophical insights planted as karmic seeds."""
        phil, non_dual = interfaces

        await phil.integrate_philosophical_karma(
            "All phenomena are empty of inherent existence"
        )

        assert len(non_dual.stored_seeds) > 0
        seed = non_dual.stored_seeds[-1]
        assert seed.dharma_polarity == "skillful"
```

### Message Bus Tests

**File**: `test_message_bus.py`

```python
import pytest
from unittest.mock import AsyncMock

class TestMessageBusIntegration:
    @pytest.fixture
    def interface(self):
        interface = PhilosophicalConsciousnessInterface()
        interface.message_bus = AsyncMock()
        return interface

    async def test_wisdom_broadcast(self, interface):
        """Broadcast wisdom to global workspace."""
        await interface.broadcast_philosophical_insight(
            insight="The unexamined life is not worth living",
            tradition="platonic",
            context="self-reflection"
        )

        interface.message_bus.publish.assert_called_once()
        call_args = interface.message_bus.publish.call_args
        assert call_args[1]["channel"] == "global_workspace"
        assert call_args[1]["message"]["message_type"] == "WORKSPACE_CONTENT_SUBMISSION"

    async def test_subscribe_to_channels(self, interface):
        """Form 28 subscribes to required channels."""
        await interface.initialize()

        expected_channels = [
            "global_workspace",
            "engagement_context",
            "meditation_state",
        ]
        for channel in expected_channels:
            interface.message_bus.subscribe.assert_any_call(
                channel=channel,
                handler=pytest.ANY
            )

    async def test_handle_workspace_broadcast(self, interface):
        """Respond to workspace broadcasts appropriately."""
        message = {
            "contents": [
                {"type": "query", "context": "meaning of life"}
            ]
        }

        await interface.handle_workspace_broadcast(message)

        # Should have attempted to enhance with philosophy
        interface.message_bus.publish.assert_called()
```

### Research Agent Tests

**File**: `test_research_agent.py`

```python
import pytest
from interface.research_agent_coordinator import (
    ResearchAgentCoordinator,
    ResearchTask,
    ResearchSource,
    ResearchStatus,
)

class TestResearchAgent:
    @pytest.fixture
    def coordinator(self):
        coord = ResearchAgentCoordinator()
        coord.sep_fetcher = MockSEPFetcher()
        coord.philpapers_fetcher = MockPhilPapersFetcher()
        return coord

    async def test_create_research_task(self, coordinator):
        """Create and queue research task."""
        task = await coordinator.create_task(
            query="phenomenological reduction",
            sources=[ResearchSource.SEP, ResearchSource.PHILPAPERS],
            priority=7
        )

        assert task.task_id is not None
        assert task.status == ResearchStatus.QUEUED

    async def test_task_execution(self, coordinator):
        """Execute research task and get results."""
        task = await coordinator.create_task(
            query="test topic",
            sources=[ResearchSource.SEP],
            priority=8
        )

        await coordinator.execute_task(task.task_id)

        updated_task = coordinator.get_task(task.task_id)
        assert updated_task.status == ResearchStatus.COMPLETED

    async def test_quality_validation(self, coordinator):
        """Extracted content must pass quality threshold."""
        # Mock low-quality extraction
        coordinator.sep_fetcher.mock_quality = 0.3

        task = await coordinator.create_task(
            query="low quality test",
            sources=[ResearchSource.SEP],
            priority=5
        )

        await coordinator.execute_task(task.task_id)

        # Should not integrate low-quality content
        assert coordinator.integration_count == 0

    async def test_rate_limiting(self, coordinator):
        """Respect rate limits for sources."""
        tasks = [
            await coordinator.create_task(f"query_{i}", [ResearchSource.SEP], 5)
            for i in range(5)
        ]

        # Execute all
        for task in tasks:
            await coordinator.execute_task(task.task_id)

        # Verify minimum delay between requests
        delays = coordinator.sep_fetcher.request_delays
        for delay in delays[1:]:
            assert delay >= 2.0  # 2 second minimum for SEP
```

---

## Philosophical Accuracy Tests

### Tradition Accuracy

**File**: `test_tradition_accuracy.py`

```python
import pytest

class TestTraditionAccuracy:
    @pytest.fixture
    def interface(self):
        return PhilosophicalConsciousnessInterface()

    def test_stoic_concepts_correct(self, interface):
        """Verify Stoic concepts are accurate."""
        apatheia = interface.get_concept_by_id("stoic_apatheia")

        assert apatheia.tradition == PhilosophicalTradition.STOICISM
        assert "passion" in apatheia.definition.lower() or "emotion" in apatheia.definition.lower()
        assert any(f in apatheia.key_figures for f in ["Zeno", "Epictetus", "Marcus Aurelius", "Seneca"])

    def test_buddhist_concepts_correct(self, interface):
        """Verify Buddhist concepts are accurate."""
        sunyata = interface.get_concept_by_id("buddhist_sunyata")

        assert "BUDDHIST" in sunyata.tradition.value.upper()
        assert "empty" in sunyata.definition.lower() or "void" in sunyata.definition.lower()

    def test_no_anachronisms(self, interface):
        """Verify no anachronistic attributions."""
        # Kant should not be attributed to pre-18th century concepts
        kant_concepts = [c for c in interface.concept_index.values()
                        if "Kant" in c.key_figures]

        for concept in kant_concepts:
            assert concept.tradition != PhilosophicalTradition.PRESOCRATIC
            assert concept.tradition != PhilosophicalTradition.STOICISM

    def test_tradition_internal_consistency(self, interface):
        """Verify concepts within tradition are internally consistent."""
        stoic_concepts = interface.get_concepts_by_tradition(
            PhilosophicalTradition.STOICISM
        )

        # Stoic concepts should reference Stoic figures
        stoic_figures = {"Zeno", "Cleanthes", "Chrysippus", "Seneca",
                        "Epictetus", "Marcus Aurelius"}

        for concept in stoic_concepts:
            figures = set(concept.key_figures)
            assert len(figures & stoic_figures) > 0 or len(concept.key_figures) == 0
```

### Concept Definition Accuracy

**File**: `test_concept_definitions.py`

```python
import pytest

class TestConceptDefinitions:
    """Cross-reference definitions with Stanford Encyclopedia."""

    @pytest.fixture
    def interface(self):
        return PhilosophicalConsciousnessInterface()

    def test_categorical_imperative_definition(self, interface):
        """Verify categorical imperative correctly defined."""
        concept = interface.get_concept_by_id("kantian_categorical_imperative")

        # Must mention universalizability or duty
        definition = concept.definition.lower()
        assert "universal" in definition or "duty" in definition or "maxim" in definition

    def test_cogito_definition(self, interface):
        """Verify cogito correctly defined."""
        concept = interface.get_concept_by_id("cartesian_cogito")

        definition = concept.definition.lower()
        assert "think" in definition or "exist" in definition or "doubt" in definition

    def test_atman_brahman_distinction(self, interface):
        """Verify Atman and Brahman properly distinguished."""
        atman = interface.get_concept_by_id("vedantic_atman")
        brahman = interface.get_concept_by_id("vedantic_brahman")

        # Should be related but distinct
        assert atman.concept_id != brahman.concept_id
        assert brahman.concept_id in atman.related_concepts or "brahman" in atman.definition.lower()
```

### Synthesis Quality Tests

**File**: `test_synthesis_quality.py`

```python
import pytest

class TestSynthesisQuality:
    @pytest.fixture
    def interface(self):
        return PhilosophicalConsciousnessInterface()

    async def test_synthesis_maintains_fidelity(self, interface):
        """Synthesis must maintain fidelity to source traditions."""
        result = await interface.synthesize_across_traditions(
            topic="nature of self",
            traditions=[
                PhilosophicalTradition.BUDDHIST_ZEN,
                PhilosophicalTradition.VEDANTIC_ADVAITA,
            ]
        )

        # Both traditions should be well-represented
        assert result.fidelity_scores["buddhist_zen"] >= 0.7
        assert result.fidelity_scores["vedantic_advaita"] >= 0.7

    async def test_synthesis_acknowledges_differences(self, interface):
        """Synthesis must not erase genuine differences."""
        result = await interface.synthesize_across_traditions(
            topic="nature of self",
            traditions=[
                PhilosophicalTradition.BUDDHIST_ZEN,  # No-self
                PhilosophicalTradition.VEDANTIC_ADVAITA,  # Atman = Brahman
            ]
        )

        # Should acknowledge the tension
        assert len(result.acknowledged_tensions) > 0
        tension_text = " ".join(result.acknowledged_tensions).lower()
        assert "self" in tension_text or "atman" in tension_text

    async def test_no_false_equivalence(self, interface):
        """Synthesis should not create false equivalences."""
        result = await interface.synthesize_across_traditions(
            topic="ultimate reality",
            traditions=[
                PhilosophicalTradition.THOMISM,  # God as Being
                PhilosophicalTradition.BUDDHIST_ZEN,  # Emptiness
            ]
        )

        # Should flag incompatibility
        assert result.compatibility_score < 0.8
```

---

## Performance Tests

### Query Latency

**File**: `test_query_latency.py`

```python
import pytest
import time

class TestQueryLatency:
    @pytest.fixture
    def interface(self):
        return PhilosophicalConsciousnessInterface()

    async def test_simple_query_under_100ms(self, interface):
        """Simple queries complete in under 100ms."""
        start = time.perf_counter()
        await interface.query_concept("virtue")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Query took {elapsed:.1f}ms, expected <100ms"

    async def test_synthesis_under_500ms(self, interface):
        """Cross-tradition synthesis under 500ms."""
        start = time.perf_counter()
        await interface.synthesize_across_traditions(
            topic="consciousness",
            traditions=[
                PhilosophicalTradition.PHENOMENOLOGY,
                PhilosophicalTradition.BUDDHIST_ZEN,
            ]
        )
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 500, f"Synthesis took {elapsed:.1f}ms, expected <500ms"

    async def test_p95_latency(self, interface):
        """95th percentile latency under SLA."""
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await interface.query_concept("ethics")
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p95 = latencies[94]
        assert p95 < 150, f"P95 latency {p95:.1f}ms, expected <150ms"
```

### Scalability Tests

**File**: `test_scalability.py`

```python
import pytest

class TestScalability:
    async def test_index_10k_concepts(self):
        """Index handles 10K concepts without degradation."""
        interface = PhilosophicalConsciousnessInterface()

        # Generate test concepts
        for i in range(10000):
            concept = create_test_concept(f"concept_{i}")
            await interface.add_concept(concept)

        # Query should still be fast
        start = time.perf_counter()
        await interface.query_concept("test query")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 200  # Still fast with 10K concepts

    async def test_concurrent_queries(self):
        """Handle concurrent queries without errors."""
        interface = PhilosophicalConsciousnessInterface()

        async def query():
            return await interface.query_concept("virtue")

        # 50 concurrent queries
        results = await asyncio.gather(*[query() for _ in range(50)])

        assert all(r is not None for r in results)
        assert len(results) == 50
```

---

## Running Tests

### Commands

```bash
# Run all tests
python -m pytest consciousness/28-philosophy/tests/ -v

# Run unit tests only
python -m pytest consciousness/28-philosophy/tests/unit/ -v

# Run integration tests
python -m pytest consciousness/28-philosophy/tests/integration/ -v

# Run philosophical accuracy tests
python -m pytest consciousness/28-philosophy/tests/philosophical/ -v

# Run performance tests
python -m pytest consciousness/28-philosophy/tests/performance/ -v

# Run with coverage
python -m pytest consciousness/28-philosophy/tests/ --cov=consciousness/28-philosophy

# Run specific test class
python -m pytest consciousness/28-philosophy/tests/unit/test_data_models.py::TestPhilosophicalConcept -v
```

### CI/CD Integration

```yaml
# .github/workflows/form28-tests.yml
name: Form 28 Tests

on:
  push:
    paths:
      - 'consciousness/28-philosophy/**'
  pull_request:
    paths:
      - 'consciousness/28-philosophy/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest consciousness/28-philosophy/tests/ -v --cov
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```
