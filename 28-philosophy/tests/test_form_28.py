"""
Form 28 (Philosophical Consciousness) Test Suite

Tests for verifying the philosophical consciousness module works correctly.
"""
import sys
import types
from pathlib import Path


def load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get the base path
BASE_PATH = Path(__file__).parent.parent
INTERFACE_PATH = BASE_PATH / "interface"

# Load the main interface module
interface_module = load_module_from_path(
    "philosophical_consciousness_interface",
    INTERFACE_PATH / "philosophical_consciousness_interface.py"
)
sys.modules['philosophical_consciousness_interface'] = interface_module

# Load the reasoning engine - patch relative imports
reasoning_code = (INTERFACE_PATH / "philosophical_reasoning_engine.py").read_text()
reasoning_code = reasoning_code.replace(
    "from .philosophical_consciousness_interface import",
    "from philosophical_consciousness_interface import"
)
reasoning_module = types.ModuleType("philosophical_reasoning_engine")
sys.modules["philosophical_reasoning_engine"] = reasoning_module
exec(reasoning_code, reasoning_module.__dict__)

# Load research agent - patch relative imports
research_code = (INTERFACE_PATH / "research_agent_coordinator.py").read_text()
research_code = research_code.replace(
    "from .philosophical_consciousness_interface import",
    "from philosophical_consciousness_interface import"
)
research_module = types.ModuleType("research_agent_coordinator")
sys.modules["research_agent_coordinator"] = research_module
exec(research_code, research_module.__dict__)

# Import classes from loaded modules
PhilosophicalTradition = interface_module.PhilosophicalTradition
PhilosophicalDomain = interface_module.PhilosophicalDomain
PhilosophicalConcept = interface_module.PhilosophicalConcept
PhilosophicalFigure = interface_module.PhilosophicalFigure
PhilosophicalText = interface_module.PhilosophicalText
QueryFilters = interface_module.QueryFilters
CrossTraditionSynthesis = interface_module.CrossTraditionSynthesis
MaturityState = interface_module.PhilosophicalMaturityState
MaturityLevel = interface_module.MaturityLevel
ArgumentType = interface_module.ArgumentType
PhilosophicalConsciousnessInterface = interface_module.PhilosophicalConsciousnessInterface

PhilosophicalReasoningEngine = reasoning_module.PhilosophicalReasoningEngine
ArgumentAnalysis = reasoning_module.ArgumentAnalysis
ReasoningMode = reasoning_module.ReasoningMode

ResearchSource = research_module.ResearchSource
ResearchStatus = research_module.ResearchStatus
ResearchTask = research_module.ResearchTask
ResearchAgentCoordinator = research_module.PhilosophicalResearchAgentCoordinator


class TestPhilosophicalTradition:
    """Test the PhilosophicalTradition enum."""

    def test_all_traditions_exist(self):
        """Verify all expected traditions are defined."""
        western = [
            PhilosophicalTradition.STOICISM,
            PhilosophicalTradition.ARISTOTELIAN,
            PhilosophicalTradition.PLATONIC,
            PhilosophicalTradition.KANTIAN,
            PhilosophicalTradition.PHENOMENOLOGY,
            PhilosophicalTradition.EXISTENTIALISM,
            PhilosophicalTradition.ANALYTIC,
            PhilosophicalTradition.PRAGMATISM,
        ]
        eastern = [
            PhilosophicalTradition.BUDDHIST_THERAVADA,
            PhilosophicalTradition.BUDDHIST_MAHAYANA,
            PhilosophicalTradition.BUDDHIST_ZEN,
            PhilosophicalTradition.DAOIST,
            PhilosophicalTradition.CONFUCIAN,
            PhilosophicalTradition.VEDANTIC_ADVAITA,
        ]

        for tradition in western + eastern:
            assert tradition is not None
            assert tradition.value is not None

        print(f"✓ All {len(western)} Western traditions defined")
        print(f"✓ All {len(eastern)} Eastern traditions defined")

    def test_tradition_values_unique(self):
        """All tradition values should be unique."""
        values = [t.value for t in PhilosophicalTradition]
        assert len(values) == len(set(values)), "Duplicate tradition values found"
        print(f"✓ All {len(values)} tradition values are unique")

    def test_tradition_count(self):
        """Verify expected number of traditions."""
        count = len(list(PhilosophicalTradition))
        assert count >= 20, f"Expected at least 20 traditions, got {count}"
        print(f"✓ {count} traditions defined (expected >= 20)")


class TestPhilosophicalDomain:
    """Test the PhilosophicalDomain enum."""

    def test_core_domains_exist(self):
        """Verify core philosophical domains are defined."""
        core_domains = [
            PhilosophicalDomain.METAPHYSICS,
            PhilosophicalDomain.EPISTEMOLOGY,
            PhilosophicalDomain.ETHICS,
            PhilosophicalDomain.AESTHETICS,
            PhilosophicalDomain.LOGIC,
            PhilosophicalDomain.PHILOSOPHY_OF_MIND,
            PhilosophicalDomain.CONSCIOUSNESS_STUDIES,
        ]

        for domain in core_domains:
            assert domain is not None
            assert domain.value is not None

        print(f"✓ All {len(core_domains)} core domains defined")

    def test_domain_values_unique(self):
        """All domain values should be unique."""
        values = [d.value for d in PhilosophicalDomain]
        assert len(values) == len(set(values)), "Duplicate domain values found"
        print(f"✓ All {len(values)} domain values are unique")


class TestPhilosophicalConcept:
    """Test the PhilosophicalConcept dataclass."""

    def test_concept_creation(self):
        """Test creating a philosophical concept."""
        concept = PhilosophicalConcept(
            concept_id="test_apatheia",
            name="Apatheia",
            tradition=PhilosophicalTradition.STOICISM,
            domain=PhilosophicalDomain.ETHICS,
            definition="Freedom from destructive passions through rational control.",
            related_concepts=["ataraxia", "eudaimonia", "virtue"],
            key_figures=["Zeno of Citium", "Epictetus", "Marcus Aurelius"],
            primary_texts=["Meditations", "Discourses"],
        )

        assert concept.concept_id == "test_apatheia"
        assert concept.name == "Apatheia"
        assert concept.tradition == PhilosophicalTradition.STOICISM
        assert concept.domain == PhilosophicalDomain.ETHICS
        assert len(concept.related_concepts) == 3
        assert len(concept.key_figures) == 3
        print("✓ Concept creation works correctly")

    def test_concept_to_embedding_text(self):
        """Test embedding text generation."""
        concept = PhilosophicalConcept(
            concept_id="test_cogito",
            name="Cogito",
            tradition=PhilosophicalTradition.RATIONALISM,
            domain=PhilosophicalDomain.EPISTEMOLOGY,
            definition="I think, therefore I am - the foundational certainty.",
            related_concepts=["doubt", "certainty"],
            key_figures=["René Descartes"],
            primary_texts=["Meditations on First Philosophy"],
        )

        embedding_text = concept.to_embedding_text()

        assert "Cogito" in embedding_text
        assert "think" in embedding_text or "certainty" in embedding_text
        print(f"✓ Embedding text generated: {len(embedding_text)} chars")

    def test_concept_maturity_default(self):
        """Test default maturity score."""
        concept = PhilosophicalConcept(
            concept_id="test_concept",
            name="Test",
            tradition=PhilosophicalTradition.ANALYTIC,
            domain=PhilosophicalDomain.LOGIC,
            definition="A test concept.",
            related_concepts=[],
            key_figures=[],
            primary_texts=[],
        )

        assert concept.maturity_score == 0.0
        assert concept.maturity_level == MaturityLevel.NASCENT
        print("✓ Default maturity score is 0.0")


class TestPhilosophicalFigure:
    """Test the PhilosophicalFigure dataclass."""

    def test_figure_creation(self):
        """Test creating a philosophical figure."""
        figure = PhilosophicalFigure(
            figure_id="aristotle",
            name="Aristotle",
            birth_year=-384,
            death_year=-322,
            era="Classical Greek",
            traditions=[PhilosophicalTradition.ARISTOTELIAN],
            key_works=["Nicomachean Ethics", "Metaphysics", "Politics"],
            core_ideas=["eudaimonia", "virtue", "substance", "potentiality"],
            influences=["Plato"],
            influenced=["Aquinas", "Averroes"],
        )

        assert figure.figure_id == "aristotle"
        assert figure.name == "Aristotle"
        assert len(figure.key_works) == 3
        assert len(figure.core_ideas) == 4
        print("✓ Figure creation works correctly")


class TestMaturityState:
    """Test the MaturityState dataclass."""

    def test_maturity_state_creation(self):
        """Test creating a maturity state."""
        state = MaturityState()

        assert state.total_concepts_integrated == 0
        assert state.cross_tradition_syntheses == 0
        assert state.get_overall_maturity() == 0.0
        print("✓ Maturity state initialized correctly")

    def test_maturity_levels(self):
        """Test all maturity levels are defined."""
        levels = [
            MaturityLevel.NASCENT,
            MaturityLevel.DEVELOPING,
            MaturityLevel.COMPETENT,
            MaturityLevel.PROFICIENT,
            MaturityLevel.MASTERFUL,
        ]

        for level in levels:
            assert level is not None
        print(f"✓ All {len(levels)} maturity levels defined")

    def test_maturity_calculation(self):
        """Test maturity calculation with data."""
        state = MaturityState(
            traditions_depth={"stoicism": 0.5, "buddhist_zen": 0.3},
            domains_coverage={"ethics": 0.4, "metaphysics": 0.2},
            cross_tradition_syntheses=10
        )

        maturity = state.get_overall_maturity()
        assert 0.0 < maturity < 1.0
        print(f"✓ Maturity calculation works: {maturity:.2f}")


class TestQueryFilters:
    """Test the QueryFilters dataclass."""

    def test_filter_creation(self):
        """Test creating query filters."""
        filters = QueryFilters(
            traditions=[PhilosophicalTradition.STOICISM, PhilosophicalTradition.BUDDHIST_ZEN],
            domains=[PhilosophicalDomain.ETHICS],
            min_maturity=0.5,
        )

        assert len(filters.traditions) == 2
        assert len(filters.domains) == 1
        assert filters.min_maturity == 0.5
        print("✓ Query filters work correctly")


class TestCrossTraditionSynthesis:
    """Test cross-tradition synthesis dataclass."""

    def test_synthesis_creation(self):
        """Test creating a cross-tradition synthesis."""
        synthesis = CrossTraditionSynthesis(
            synthesis_id="synth_001",
            topic="nature of suffering",
            traditions=[
                PhilosophicalTradition.STOICISM,
                PhilosophicalTradition.BUDDHIST_THERAVADA,
            ],
            convergent_insights=["attachment as source", "acceptance as path"],
            divergent_positions=["metaphysical commitments differ"],
            synthesis_statement="Both traditions recognize suffering as arising from attachment...",
            fidelity_scores={
                "stoicism": 0.85,
                "buddhist_theravada": 0.90,
            },
            coherence_score=0.80,
        )

        assert synthesis.topic == "nature of suffering"
        assert len(synthesis.traditions) == 2
        assert synthesis.fidelity_scores["stoicism"] == 0.85
        assert synthesis.coherence_score == 0.80
        print("✓ Cross-tradition synthesis works correctly")


class TestPhilosophicalReasoningEngine:
    """Test the reasoning engine."""

    def test_engine_creation(self):
        """Test creating reasoning engine."""
        interface = PhilosophicalConsciousnessInterface()
        engine = PhilosophicalReasoningEngine(consciousness_interface=interface)
        assert engine is not None
        assert engine.reasoning_mode == ReasoningMode.ANALYTICAL
        print("✓ Reasoning engine created")

    def test_reasoning_modes(self):
        """Test all reasoning modes exist."""
        modes = [
            ReasoningMode.ANALYTICAL,
            ReasoningMode.DIALECTICAL,
            ReasoningMode.PHENOMENOLOGICAL,
            ReasoningMode.CONTEMPLATIVE,
            ReasoningMode.PRAGMATIC,
        ]

        for mode in modes:
            assert mode is not None
        print(f"✓ All {len(modes)} reasoning modes defined")

    def test_argument_types(self):
        """Test all argument types exist."""
        types = [
            ArgumentType.DEDUCTIVE,
            ArgumentType.INDUCTIVE,
            ArgumentType.ABDUCTIVE,
            ArgumentType.ANALOGY,
            ArgumentType.TRANSCENDENTAL,
        ]

        for arg_type in types:
            assert arg_type is not None
        print(f"✓ All {len(types)} argument types defined")

    def test_argument_analysis(self):
        """Test creating argument analysis."""
        analysis = ArgumentAnalysis(
            argument_id="modus_ponens_1",
            validity_assessment="Valid deductive argument",
            validity_score=1.0,
            premise_assessments=[{"premise": "If P then Q", "status": "accepted"}],
            hidden_assumptions=[],
            soundness_assessment="Sound if premises are true",
            soundness_score=0.9,
            related_arguments=["modus_tollens"],
            objections_summary=[],
            overall_evaluation="Valid and likely sound",
        )

        assert analysis.argument_id == "modus_ponens_1"
        assert analysis.validity_score == 1.0
        print("✓ Argument analysis works correctly")


class TestResearchAgentCoordinator:
    """Test the research agent coordinator."""

    def test_coordinator_creation(self):
        """Test creating the coordinator."""
        interface = PhilosophicalConsciousnessInterface()
        coordinator = ResearchAgentCoordinator(consciousness_interface=interface)
        assert coordinator is not None
        assert coordinator.max_concurrent == 3
        print("✓ Research coordinator created")

    def test_research_sources(self):
        """Test all research sources exist."""
        sources = [
            ResearchSource.STANFORD_ENCYCLOPEDIA,
            ResearchSource.PHILPAPERS,
            ResearchSource.INTERNET_ENCYCLOPEDIA,
            ResearchSource.LOCAL_CORPUS,
            ResearchSource.WEB_SEARCH,
        ]

        for source in sources:
            assert source is not None
        print(f"✓ All {len(sources)} research sources defined")

    def test_research_statuses(self):
        """Test all research statuses exist."""
        statuses = [
            ResearchStatus.PENDING,
            ResearchStatus.IN_PROGRESS,
            ResearchStatus.COMPLETED,
            ResearchStatus.FAILED,
        ]

        for status in statuses:
            assert status is not None
        print(f"✓ All {len(statuses)} research statuses defined")

    def test_task_creation(self):
        """Test creating a research task."""
        task = ResearchTask(
            task_id="task_001",
            query="phenomenological reduction",
            sources=[ResearchSource.STANFORD_ENCYCLOPEDIA, ResearchSource.PHILPAPERS],
            priority=7,
            status=ResearchStatus.PENDING,
        )

        assert task.task_id == "task_001"
        assert len(task.sources) == 2
        assert task.priority == 7
        assert task.status == ResearchStatus.PENDING
        print("✓ Research task creation works correctly")


class TestPhilosophicalConsciousnessInterface:
    """Test the main interface."""

    def test_interface_creation(self):
        """Test creating the interface."""
        interface = PhilosophicalConsciousnessInterface()

        assert interface is not None
        assert interface.FORM_ID == "28-philosophy"
        assert interface.NAME == "Philosophical Consciousness"
        print("✓ Interface created with correct FORM_ID and NAME")

    def test_interface_has_indexes(self):
        """Test interface has index structures."""
        interface = PhilosophicalConsciousnessInterface()

        assert hasattr(interface, 'concept_index')
        assert hasattr(interface, 'figure_index')
        assert hasattr(interface, 'text_index')
        assert hasattr(interface, 'maturity_state')
        print("✓ Interface has required index structures")

    def test_interface_has_maturity_state(self):
        """Test interface initializes maturity state."""
        interface = PhilosophicalConsciousnessInterface()

        assert interface.maturity_state is not None
        # MaturityState doesn't have maturity_level, it has get_overall_maturity()
        assert interface.maturity_state.get_overall_maturity() == 0.0
        print("✓ Interface has maturity state initialized")


class TestIntegration:
    """Integration tests for Form 28."""

    def test_concept_embedding_text_usable(self):
        """Test that concept embedding text is suitable for embedding."""
        concept = PhilosophicalConcept(
            concept_id="categorical_imperative",
            name="Categorical Imperative",
            tradition=PhilosophicalTradition.KANTIAN,
            domain=PhilosophicalDomain.ETHICS,
            definition="Act only according to that maxim whereby you can at the same time will that it should become a universal law.",
            related_concepts=["duty", "universalizability", "moral law"],
            key_figures=["Immanuel Kant"],
            primary_texts=["Groundwork of the Metaphysics of Morals"],
        )

        embedding_text = concept.to_embedding_text()

        # Should be non-empty and reasonable length
        assert len(embedding_text) > 50
        assert len(embedding_text) < 5000

        # Should contain key information
        assert "Categorical Imperative" in embedding_text
        print(f"✓ Embedding text is usable ({len(embedding_text)} chars)")

    def test_full_workflow(self):
        """Test a typical workflow."""
        # 1. Create interface
        interface = PhilosophicalConsciousnessInterface()

        # 2. Create a concept
        concept = PhilosophicalConcept(
            concept_id="test_workflow",
            name="Test Concept",
            tradition=PhilosophicalTradition.PHENOMENOLOGY,
            domain=PhilosophicalDomain.PHILOSOPHY_OF_MIND,
            definition="A test concept for workflow validation.",
            related_concepts=[],
            key_figures=["Edmund Husserl"],
            primary_texts=[],
        )

        # 3. Add to index
        interface.concept_index[concept.concept_id] = concept

        # 4. Verify retrieval
        retrieved = interface.concept_index.get("test_workflow")
        assert retrieved is not None
        assert retrieved.name == "Test Concept"

        # 5. Create filters
        filters = QueryFilters(
            traditions=[PhilosophicalTradition.PHENOMENOLOGY],
        )

        # 6. Filter concepts
        filtered = [c for c in interface.concept_index.values()
                   if c.tradition in filters.traditions]
        assert len(filtered) >= 1

        print("✓ Full workflow completed successfully")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Form 28 (Philosophical Consciousness) Test Suite")
    print("=" * 60)
    print()

    test_classes = [
        TestPhilosophicalTradition,
        TestPhilosophicalDomain,
        TestPhilosophicalConcept,
        TestPhilosophicalFigure,
        TestMaturityState,
        TestQueryFilters,
        TestCrossTraditionSynthesis,
        TestPhilosophicalReasoningEngine,
        TestResearchAgentCoordinator,
        TestPhilosophicalConsciousnessInterface,
        TestIntegration,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
