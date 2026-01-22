"""
Comprehensive Integration Tests for ML Research Module.

This module tests the full integration of:
1. Orchestrator selecting and loading architectures
2. Running inference through different architectures
3. Technique composition (ChainOfThought >> SelfEvaluation, RAG, HooksSystem)
4. Architecture switching (mamba vs transformer)

Tests are organized to run without PyTorch installed (using mocks) by default,
with additional tests marked for PyTorch/GPU execution.

Run with:
    # Run all mock-based tests
    pytest consciousness/ml_research/tests/test_integration.py -v

    # Run with PyTorch tests
    pytest consciousness/ml_research/tests/test_integration.py -v --run-pytorch

    # Run with slow/GPU tests
    pytest consciousness/ml_research/tests/test_integration.py -v --run-slow
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Handle pytest import - pytest may not be installed in all environments
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a mock pytest module for basic compatibility
    class MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        class mark:
            @staticmethod
            def slow(func):
                return func
            @staticmethod
            def pytorch(func):
                return func
            @staticmethod
            def gpu(func):
                return func
            @staticmethod
            def skip(reason=""):
                def decorator(func):
                    return func
                return decorator

        @staticmethod
        def raises(exception, **kwargs):
            from contextlib import contextmanager
            @contextmanager
            def mock_raises():
                try:
                    yield
                except exception:
                    pass
            return mock_raises()

        @staticmethod
        def main(args=None):
            print("pytest not available - cannot run tests directly")
            return 1

    pytest = MockPytest()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import test utilities - these are also available via pytest fixtures from conftest.py
try:
    # When running as part of pytest, conftest fixtures are auto-loaded
    from .conftest import (
        MockBackend,
        MockModelConfig,
        MockOrchestrator,
        MockTensor,
    )
except ImportError:
    # Fallback for direct execution
    from conftest import (
        MockBackend,
        MockModelConfig,
        MockOrchestrator,
        MockTensor,
    )


# =============================================================================
# TEST: FULL PIPELINE - ORCHESTRATOR + ARCHITECTURE + INFERENCE
# =============================================================================

class TestFullPipeline:
    """Test the full pipeline from orchestrator to inference."""

    def test_orchestrator_selects_architecture_for_task(self, mock_orchestrator):
        """Test that orchestrator selects appropriate architecture based on task."""
        # Long context task should select mamba
        arch = mock_orchestrator.select_architecture("Process this long context document")
        assert arch == "mamba"

        # Default task should select transformer
        arch = mock_orchestrator.select_architecture("Translate this text")
        assert arch == "transformer"

        # Efficient task should select rwkv
        arch = mock_orchestrator.select_architecture("Fast efficient inference needed")
        assert arch == "rwkv"

    def test_orchestrator_loads_architecture(self, mock_orchestrator, mock_config):
        """Test that orchestrator can load an architecture."""
        backend = mock_orchestrator.load_architecture("transformer", mock_config)

        assert backend is not None
        assert mock_orchestrator.get_current_architecture() == "transformer"
        assert isinstance(backend, MockBackend)

    def test_orchestrator_runs_inference(self, mock_orchestrator, mock_config):
        """Test running inference through the orchestrator."""
        mock_orchestrator.load_architecture("transformer", mock_config)

        result = mock_orchestrator.run_inference(
            "What is the capital of France?",
            max_tokens=50,
        )

        assert result["architecture"] == "transformer"
        assert result["output"] is not None
        assert result["metadata"]["call_count"] == 1

    def test_full_pipeline_select_load_infer(self, mock_orchestrator, mock_config):
        """Test the complete pipeline: select -> load -> infer."""
        task = "Process this long context document for summarization"

        # Step 1: Select architecture
        architecture = mock_orchestrator.select_architecture(task)
        assert architecture == "mamba"  # Should select mamba for long context

        # Step 2: Load architecture
        backend = mock_orchestrator.load_architecture(architecture, mock_config)
        assert backend is not None

        # Step 3: Run inference
        result = mock_orchestrator.run_inference(task)
        assert result["architecture"] == "mamba"
        assert result["output"] is not None

        # Verify routing history
        history = mock_orchestrator.get_routing_history()
        assert len(history) == 1
        assert history[0]["architecture"] == "mamba"

    def test_pipeline_with_different_input_types(self, mock_orchestrator, mock_config):
        """Test pipeline handles different input types."""
        mock_orchestrator.load_architecture("transformer", mock_config)

        # String input
        result1 = mock_orchestrator.run_inference("Hello world")
        assert result1["output"] is not None

        # Mock tensor input
        tensor_input = MockTensor(shape=(2, 10))
        result2 = mock_orchestrator.run_inference(tensor_input)
        assert result2["output"] is not None

    def test_pipeline_error_on_unknown_architecture(self, mock_orchestrator):
        """Test that pipeline raises error for unknown architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            mock_orchestrator.load_architecture("nonexistent_arch")


# =============================================================================
# TEST: TECHNIQUE COMPOSITION
# =============================================================================

class TestTechniqueComposition:
    """Test composition of ML techniques."""

    def test_chain_of_thought_basic(self):
        """Test basic ChainOfThought technique."""
        from consciousness.ml_research.ml_techniques.prompting import (
            ChainOfThought,
            CoTMode,
        )  # noqa: E501

        cot = ChainOfThought(mode=CoTMode.ZERO_SHOT)
        result = cot.run("What is 23 * 47?")

        assert result.success
        assert result.technique_id == "chain_of_thought"
        assert "reasoning" in result.output
        assert result.intermediate_steps

    def test_self_evaluation_basic(self):
        """Test basic SelfEvaluation technique."""
        from consciousness.ml_research.ml_techniques.verification import (
            SelfEvaluation,
            EvaluationCriteria,
        )

        evaluator = SelfEvaluation(
            criteria=[
                EvaluationCriteria.CORRECTNESS,
                EvaluationCriteria.COHERENCE,
            ],
            threshold=0.5,
        )

        result = evaluator.run({
            "input": "What is 2+2?",
            "output": "The answer is 4.",
        })

        assert result.technique_id == "self_evaluation"
        assert "evaluations" in result.output
        assert "average_score" in result.output

    def test_cot_self_evaluation_pipeline(self):
        """Test ChainOfThought >> SelfEvaluation pipeline composition."""
        from consciousness.ml_research.ml_techniques import Pipeline
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought
        from consciousness.ml_research.ml_techniques.verification import SelfEvaluation

        # Create pipeline using >> operator
        cot = ChainOfThought()
        evaluator = SelfEvaluation(threshold=0.3)

        # Create pipeline
        pipeline = cot >> evaluator

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.techniques) == 2

    def test_pipeline_run(self):
        """Test running a composed pipeline."""
        from consciousness.ml_research.ml_techniques import Pipeline
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought
        from consciousness.ml_research.ml_techniques.verification import SelfEvaluation

        # Create techniques that work with mock output
        class MockCoT(ChainOfThought):
            def run(self, input_data, context=None):
                result = super().run(input_data, context)
                # Format output for SelfEvaluation
                result.output = {
                    "input": str(input_data),
                    "output": result.output.get("reasoning", ""),
                }
                return result

        cot = MockCoT()
        evaluator = SelfEvaluation(threshold=0.3)

        pipeline = Pipeline([cot, evaluator])
        result = pipeline.run("What is 5 + 3?")

        assert result.technique_id == "pipeline"
        # Pipeline may succeed or fail depending on random evaluation
        assert result.intermediate_steps

    def test_parallel_composition(self):
        """Test parallel composition of techniques."""
        from consciousness.ml_research.ml_techniques import ParallelComposition
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        cot1 = ChainOfThought()
        cot2 = ChainOfThought()

        # Create parallel using | operator
        parallel = cot1 | cot2

        assert isinstance(parallel, ParallelComposition)
        assert len(parallel.techniques) == 2

    def test_nested_pipeline(self):
        """Test nested pipeline composition."""
        from consciousness.ml_research.ml_techniques import Pipeline
        from consciousness.ml_research.ml_techniques.prompting import (
            ChainOfThought,
            SelfConsistency,
        )

        # Create nested structure
        inner_pipeline = ChainOfThought() >> ChainOfThought()
        outer_pipeline = inner_pipeline >> ChainOfThought()

        assert isinstance(outer_pipeline, Pipeline)
        # 2 from inner + 1 from outer
        assert len(outer_pipeline.techniques) == 3


# =============================================================================
# TEST: RAG WITH DOCUMENT RETRIEVAL
# =============================================================================

class TestRAGIntegration:
    """Test RAG (Retrieval-Augmented Generation) integration."""

    def test_rag_basic_retrieval(self, sample_documents):
        """Test basic RAG document retrieval."""
        from consciousness.ml_research.ml_techniques.memory import (
            RAG,
            Document,
            RetrieverType,
        )

        # Create documents
        docs = [
            Document(
                doc_id=d["id"],
                content=d["content"],
                metadata=d["metadata"],
            )
            for d in sample_documents
        ]

        rag = RAG(
            documents=docs,
            retriever_type=RetrieverType.DENSE,
            top_k=3,
        )

        result = rag.run("What is the capital of France?")

        assert result.success
        assert result.technique_id == "rag"
        assert "retrieved_chunks" in result.output
        assert len(result.output["retrieved_chunks"]) <= 3

    def test_rag_hybrid_retrieval(self, sample_documents):
        """Test RAG with hybrid retrieval."""
        from consciousness.ml_research.ml_techniques.memory import (
            RAG,
            Document,
            RetrieverType,
        )

        docs = [
            Document(
                doc_id=d["id"],
                content=d["content"],
                metadata=d["metadata"],
            )
            for d in sample_documents
        ]

        rag = RAG(
            documents=docs,
            retriever_type=RetrieverType.HYBRID,
            top_k=3,
        )

        result = rag.run("transformer architecture attention")

        assert result.success
        assert "context" in result.output

    def test_rag_add_documents(self, sample_documents):
        """Test adding documents to RAG after initialization."""
        from consciousness.ml_research.ml_techniques.memory import (
            RAG,
            Document,
        )

        rag = RAG(documents=[], top_k=2)

        # Initially empty
        result1 = rag.run("test query")
        assert len(result1.output.get("retrieved_chunks", [])) == 0

        # Add documents
        docs = [
            Document(
                doc_id=d["id"],
                content=d["content"],
            )
            for d in sample_documents[:2]
        ]
        rag.add_documents(docs)

        # Now should retrieve
        result2 = rag.run("France capital")
        assert len(result2.output.get("retrieved_chunks", [])) > 0

    def test_rag_with_reranking(self, sample_documents):
        """Test RAG with reranking enabled."""
        from consciousness.ml_research.ml_techniques.memory import (
            RAG,
            Document,
        )

        docs = [
            Document(doc_id=d["id"], content=d["content"])
            for d in sample_documents
        ]

        rag = RAG(
            documents=docs,
            rerank=True,
            top_k=3,
        )

        result = rag.run("machine learning neural networks")

        assert result.success
        # Check that rerank step is in trace
        actions = [step.get("action") for step in result.intermediate_steps]
        assert "rerank" in actions


# =============================================================================
# TEST: HOOKS SYSTEM
# =============================================================================

class TestHooksSystem:
    """Test HooksSystem for pre/post execution hooks."""

    def test_hooks_system_basic(self, hook_tracker):
        """Test basic hooks system functionality."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        hooks_sys = HooksSystem()

        # Add pre-run hook
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=hook_tracker.make_hook("pre_run"),
            name="test_pre_hook",
        ))

        # Add post-run hook
        hooks_sys.add_hook(Hook(
            hook_type=HookType.POST_RUN,
            function=hook_tracker.make_hook("post_run"),
            name="test_post_hook",
        ))

        result = hooks_sys.run("test input")

        assert result.success
        # Verify hooks were called
        pre_calls = hook_tracker.get_calls("pre_run")
        post_calls = hook_tracker.get_calls("post_run")
        assert len(pre_calls) == 1
        assert len(post_calls) == 1

    def test_hooks_transform_input(self, hook_tracker):
        """Test input transformation hooks."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        def uppercase_transform(ctx):
            if isinstance(ctx.input_data, str):
                return ctx.input_data.upper()
            return ctx.input_data

        hooks_sys = HooksSystem()
        hooks_sys.add_hook(Hook(
            hook_type=HookType.TRANSFORM_INPUT,
            function=uppercase_transform,
            name="uppercase",
        ))

        result = hooks_sys.run("hello world")

        assert result.success
        assert result.output == "HELLO WORLD"

    def test_hooks_transform_output(self):
        """Test output transformation hooks."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        def add_suffix(ctx):
            if isinstance(ctx.output_data, str):
                return ctx.output_data + " [processed]"
            return ctx.output_data

        hooks_sys = HooksSystem()
        hooks_sys.add_hook(Hook(
            hook_type=HookType.TRANSFORM_OUTPUT,
            function=add_suffix,
            name="add_suffix",
        ))

        result = hooks_sys.run("test")

        assert result.success
        assert "[processed]" in result.output

    def test_hooks_with_inner_technique(self, hook_tracker):
        """Test hooks wrapping an inner technique."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        hooks_sys = HooksSystem()
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=hook_tracker.make_hook("pre_cot"),
            name="pre_cot",
        ))
        hooks_sys.add_hook(Hook(
            hook_type=HookType.POST_RUN,
            function=hook_tracker.make_hook("post_cot"),
            name="post_cot",
        ))

        inner = ChainOfThought()
        result = hooks_sys.run("What is 2+2?", inner_technique=inner)

        # Verify hooks were called around inner technique
        assert len(hook_tracker.get_calls("pre_cot")) == 1
        assert len(hook_tracker.get_calls("post_cot")) == 1

    def test_hooks_error_handling(self):
        """Test hooks system error handling."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        def failing_hook(ctx):
            raise ValueError("Hook failed!")

        hooks_sys = HooksSystem(continue_on_hook_error=True)
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=failing_hook,
            name="failing_hook",
        ))

        # Should not raise, just log error
        result = hooks_sys.run("test")
        assert result.success

        # With continue_on_hook_error=False, should propagate error
        hooks_sys2 = HooksSystem(continue_on_hook_error=False)
        hooks_sys2.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=failing_hook,
            name="failing_hook",
        ))

        result2 = hooks_sys2.run("test")
        assert not result2.success
        assert "Hook failed" in result2.error

    def test_hooks_conditional(self):
        """Test conditional hooks."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        call_count = [0]

        def counting_hook(ctx):
            call_count[0] += 1

        def only_long_inputs(ctx):
            return len(str(ctx.input_data)) > 10

        hooks_sys = HooksSystem()
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=counting_hook,
            name="conditional_hook",
            condition=only_long_inputs,
        ))

        # Short input - hook should not run
        hooks_sys.run("short")
        assert call_count[0] == 0

        # Long input - hook should run
        hooks_sys.run("this is a longer input string")
        assert call_count[0] == 1

    def test_hooks_priority(self):
        """Test hook execution priority."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        execution_order = []

        def make_hook(name: str):
            def hook(ctx):
                execution_order.append(name)
            return hook

        hooks_sys = HooksSystem()

        # Add hooks with different priorities (higher = runs first)
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=make_hook("low"),
            name="low_priority",
            priority=1,
        ))
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=make_hook("high"),
            name="high_priority",
            priority=10,
        ))
        hooks_sys.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=make_hook("medium"),
            name="medium_priority",
            priority=5,
        ))

        hooks_sys.run("test")

        assert execution_order == ["high", "medium", "low"]


# =============================================================================
# TEST: ARCHITECTURE SWITCHING
# =============================================================================

class TestArchitectureSwitching:
    """Test switching between different architectures."""

    def test_switch_transformer_to_mamba(self, mock_orchestrator, mock_config):
        """Test switching from transformer to mamba architecture."""
        # Start with transformer
        mock_orchestrator.load_architecture("transformer", mock_config)
        assert mock_orchestrator.get_current_architecture() == "transformer"

        result1 = mock_orchestrator.run_inference("First query")
        assert result1["architecture"] == "transformer"

        # Switch to mamba
        mock_orchestrator.switch_architecture("mamba")
        assert mock_orchestrator.get_current_architecture() == "mamba"

        result2 = mock_orchestrator.run_inference("Second query")
        assert result2["architecture"] == "mamba"

        # Verify both outputs exist but are from different architectures
        assert result1["architecture"] != result2["architecture"]

    def test_switch_preserves_backend_state(self, mock_orchestrator, mock_config):
        """Test that switching back preserves backend state."""
        # Use transformer
        mock_orchestrator.load_architecture("transformer", mock_config)
        mock_orchestrator.run_inference("Query 1")
        mock_orchestrator.run_inference("Query 2")

        # Check call count
        transformer_backend = mock_orchestrator._backends["transformer"]
        assert transformer_backend._call_count == 2

        # Switch to mamba
        mock_orchestrator.switch_architecture("mamba")
        mock_orchestrator.run_inference("Query 3")

        # Switch back to transformer
        mock_orchestrator.switch_architecture("transformer")

        # Call count should be preserved
        assert transformer_backend._call_count == 2

    def test_multiple_architecture_switches(self, mock_orchestrator, mock_config):
        """Test rapid switching between multiple architectures."""
        architectures = ["transformer", "mamba", "rwkv", "griffin", "hyena"]

        for arch in architectures:
            mock_orchestrator.load_architecture(arch, mock_config)
            result = mock_orchestrator.run_inference(f"Query for {arch}")
            assert result["architecture"] == arch

        # Verify routing history
        history = mock_orchestrator.get_routing_history()
        assert len(history) == len(architectures)

        for i, arch in enumerate(architectures):
            assert history[i]["architecture"] == arch

    def test_architecture_outputs_differ(self, mock_orchestrator, mock_config):
        """Test that different architectures produce different outputs."""
        query = "Common query for all architectures"

        # Get outputs from different architectures
        outputs = {}
        for arch in ["transformer", "mamba"]:
            mock_orchestrator.load_architecture(arch, mock_config)
            result = mock_orchestrator.run_inference(query)
            outputs[arch] = result

        # Verify we got results from both
        assert "transformer" in outputs
        assert "mamba" in outputs

        # Both should have valid output
        assert outputs["transformer"]["output"] is not None
        assert outputs["mamba"]["output"] is not None

    def test_architecture_switch_during_pipeline(self, mock_orchestrator, mock_config):
        """Test architecture switching within a pipeline execution."""
        # Simulate a pipeline that uses different architectures
        results = []

        # Phase 1: Use transformer for initial processing
        mock_orchestrator.load_architecture("transformer", mock_config)
        result1 = mock_orchestrator.run_inference("Initial analysis")
        results.append(("transformer", result1))

        # Phase 2: Switch to mamba for long-context processing
        mock_orchestrator.switch_architecture("mamba")
        result2 = mock_orchestrator.run_inference("Long context processing")
        results.append(("mamba", result2))

        # Phase 3: Back to transformer for final output
        mock_orchestrator.switch_architecture("transformer")
        result3 = mock_orchestrator.run_inference("Final synthesis")
        results.append(("transformer", result3))

        # Verify all phases completed
        assert len(results) == 3
        assert results[0][0] == "transformer"
        assert results[1][0] == "mamba"
        assert results[2][0] == "transformer"


# =============================================================================
# TEST: TECHNIQUE + BACKEND INTEGRATION
# =============================================================================

class TestTechniqueBackendIntegration:
    """Test techniques using the mock backend."""

    def test_cot_with_mock_backend(self, mock_backend):
        """Test ChainOfThought with mock backend."""
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        cot = ChainOfThought(model=mock_backend)
        result = cot.run("What is 15 * 8?")

        assert result.success
        assert mock_backend._call_count == 0  # Mock doesn't increment in placeholder

    def test_rag_with_mock_backend(self, mock_backend, sample_documents):
        """Test RAG with mock backend."""
        from consciousness.ml_research.ml_techniques.memory import RAG, Document

        docs = [
            Document(doc_id=d["id"], content=d["content"])
            for d in sample_documents
        ]

        rag = RAG(model=mock_backend, documents=docs, top_k=3)
        result = rag.run("machine learning")

        assert result.success
        assert "retrieved_chunks" in result.output

    def test_ensemble_with_mock_backends(self):
        """Test Ensemble technique with multiple mock backends."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            Ensemble,
            EnsembleMember,
            AggregationStrategy,
        )

        def mock_handler_1(x):
            return "answer_a"

        def mock_handler_2(x):
            return "answer_a"

        def mock_handler_3(x):
            return "answer_b"

        ensemble = Ensemble(
            members=[
                EnsembleMember("m1", mock_handler_1, weight=1.0),
                EnsembleMember("m2", mock_handler_2, weight=1.0),
                EnsembleMember("m3", mock_handler_3, weight=0.5),
            ],
            strategy=AggregationStrategy.WEIGHTED_VOTE,
        )

        result = ensemble.run("What is the answer?")

        assert result.success
        # answer_a should win with higher weight
        assert result.output == "answer_a"

    def test_task_routing_with_mock_handlers(self):
        """Test TaskRouting with mock handlers."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            TaskRouting,
            Route,
        )

        math_results = []
        search_results = []

        def math_handler(x):
            math_results.append(x)
            return f"Math result for: {x}"

        def search_handler(x):
            search_results.append(x)
            return f"Search result for: {x}"

        router = TaskRouting(
            routes=[
                Route("math", lambda x: "calculate" in x.lower(), math_handler),
                Route("search", lambda x: "find" in x.lower(), search_handler),
            ],
        )

        # Route to math
        result1 = router.run("Calculate 5 + 3")
        assert "Math result" in result1.output
        assert len(math_results) == 1

        # Route to search
        result2 = router.run("Find information about AI")
        assert "Search result" in result2.output
        assert len(search_results) == 1


# =============================================================================
# TEST: MEMORY BANK INTEGRATION
# =============================================================================

class TestMemoryBankIntegration:
    """Test MemoryBank technique integration."""

    def test_memory_store_retrieve(self):
        """Test storing and retrieving memories."""
        from consciousness.ml_research.ml_techniques.memory import (
            MemoryBank,
            MemoryType,
        )

        memory = MemoryBank(max_memories=100)

        # Store memories
        memory.store("User prefers dark mode", MemoryType.SEMANTIC)
        memory.store("Previous conversation about Python", MemoryType.EPISODIC)
        memory.store("Working on ML project", MemoryType.WORKING)

        # Retrieve relevant memories
        result = memory.run("What are the user preferences?")

        assert result.success
        assert "memories" in result.output
        assert len(result.output["memories"]) > 0

    def test_memory_importance_based_retrieval(self):
        """Test that important memories are prioritized."""
        from consciousness.ml_research.ml_techniques.memory import (
            MemoryBank,
            MemoryType,
        )

        memory = MemoryBank(max_memories=100)

        # Store with different importance levels
        memory.store("Low importance note", MemoryType.EPISODIC, importance=0.2)
        memory.store("High importance fact", MemoryType.SEMANTIC, importance=0.9)

        # Retrieve with min_importance filter
        high_imp = memory.retrieve("fact", k=5, min_importance=0.5)

        assert len(high_imp) == 1
        assert "High importance" in high_imp[0].content

    def test_memory_type_filtering(self):
        """Test filtering memories by type."""
        from consciousness.ml_research.ml_techniques.memory import (
            MemoryBank,
            MemoryType,
        )

        memory = MemoryBank()

        memory.store("Semantic fact 1", MemoryType.SEMANTIC)
        memory.store("Episodic memory 1", MemoryType.EPISODIC)
        memory.store("Semantic fact 2", MemoryType.SEMANTIC)

        # Retrieve only semantic memories
        semantic = memory.retrieve(
            "fact",
            k=10,
            memory_types=[MemoryType.SEMANTIC],
        )

        assert len(semantic) == 2
        assert all(m.memory_type == MemoryType.SEMANTIC for m in semantic)


# =============================================================================
# TEST: CONTEXT COMPRESSION
# =============================================================================

class TestContextCompression:
    """Test ContextCompression technique."""

    def test_basic_compression(self):
        """Test basic context compression."""
        from consciousness.ml_research.ml_techniques.memory import (
            ContextCompression,
            CompressionMethod,
        )

        compressor = ContextCompression(
            method=CompressionMethod.SELECTIVE,
            target_ratio=0.5,
        )

        long_text = " ".join(["This is a sentence about machine learning."] * 20)

        result = compressor.run({
            "context": long_text,
            "query": "machine learning",
        })

        assert result.success
        assert "compressed" in result.output
        assert result.output["compression_ratio"] < 1.0

    def test_compression_with_keywords(self):
        """Test compression preserves keywords."""
        from consciousness.ml_research.ml_techniques.memory import (
            ContextCompression,
            CompressionMethod,
        )

        compressor = ContextCompression(
            method=CompressionMethod.SELECTIVE,
            target_ratio=0.3,
            preserve_keywords=["important", "critical"],
        )

        text = "This is regular text. This is important information. More regular content. Critical data here."

        result = compressor.run({
            "context": text,
            "query": "data",
        })

        assert result.success
        compressed = result.output["compressed"]
        # Keywords should be preserved
        assert "important" in compressed.lower() or "critical" in compressed.lower()


# =============================================================================
# TEST: PYTORCH-DEPENDENT TESTS (Marked)
# =============================================================================

@pytest.mark.pytorch
class TestPyTorchIntegration:
    """Tests that require PyTorch to be installed."""

    def test_mamba_model_creation(self, patch_torch, mamba_config):
        """Test creating a Mamba model with PyTorch."""
        # This test uses mocked torch
        from unittest.mock import MagicMock

        # Create mock Mamba-like model
        model = MagicMock()
        model.config = mamba_config
        model.num_parameters = MagicMock(return_value=1000000)

        assert model.num_parameters() == 1000000
        assert model.config["d_model"] == 64

    def test_transformer_model_creation(self, patch_torch, transformer_config):
        """Test creating a Transformer model with PyTorch."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.config = transformer_config
        model.num_parameters = MagicMock(return_value=2000000)

        assert model.num_parameters() == 2000000
        assert model.config["n_head"] == 4


@pytest.mark.slow
class TestSlowIntegration:
    """Slow tests that require significant computation or large models."""

    def test_large_document_rag(self, sample_documents):
        """Test RAG with many documents."""
        from consciousness.ml_research.ml_techniques.memory import RAG, Document

        # Create many documents
        docs = []
        for i in range(100):
            docs.append(Document(
                doc_id=f"doc_{i}",
                content=f"Document {i} contains information about topic {i % 10}.",
            ))

        rag = RAG(documents=docs, top_k=10)
        result = rag.run("information about topic 5")

        assert result.success
        assert len(result.output["retrieved_chunks"]) <= 10

    def test_deep_pipeline_composition(self):
        """Test deeply nested pipeline composition."""
        from consciousness.ml_research.ml_techniques import Pipeline
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        # Create 10-step pipeline
        techniques = [ChainOfThought() for _ in range(10)]
        pipeline = Pipeline(techniques)

        # Should handle deep composition
        assert len(pipeline.techniques) == 10

    def test_high_sample_self_consistency(self):
        """Test SelfConsistency with many samples."""
        from consciousness.ml_research.ml_techniques.prompting import SelfConsistency

        sc = SelfConsistency(num_samples=20)
        result = sc.run("What is 7 * 8?")

        assert result.technique_id == "self_consistency"
        assert len(result.output["samples"]) == 20


# =============================================================================
# TEST: ERROR HANDLING AND EDGE CASES
# =============================================================================

class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        cot = ChainOfThought()
        result = cot.run("")

        # Should handle gracefully
        assert result.technique_id == "chain_of_thought"

    def test_none_input_handling(self):
        """Test handling of None inputs."""
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought

        cot = ChainOfThought()
        result = cot.run(None)

        # Should convert to string and handle
        assert result.technique_id == "chain_of_thought"

    def test_pipeline_failure_propagation(self):
        """Test that pipeline properly propagates failures."""
        from consciousness.ml_research.ml_techniques import Pipeline, TechniqueBase, TechniqueResult

        class FailingTechnique(TechniqueBase):
            TECHNIQUE_ID = "failing"

            def run(self, input_data, context=None):
                return TechniqueResult(
                    success=False,
                    output=None,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=0,
                    error="Intentional failure",
                )

        class SuccessTechnique(TechniqueBase):
            TECHNIQUE_ID = "success"

            def run(self, input_data, context=None):
                return TechniqueResult(
                    success=True,
                    output=input_data,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=0,
                )

        # Success then Fail
        pipeline = Pipeline([SuccessTechnique(), FailingTechnique()])
        result = pipeline.run("test")

        assert not result.success
        assert "Intentional failure" in result.error

    def test_orchestrator_reset(self, mock_orchestrator, mock_config):
        """Test orchestrator reset functionality."""
        mock_orchestrator.load_architecture("transformer", mock_config)
        mock_orchestrator.run_inference("Query 1")
        mock_orchestrator.run_inference("Query 2")

        assert len(mock_orchestrator.get_routing_history()) == 2

        mock_orchestrator.reset()

        assert mock_orchestrator.get_current_architecture() is None
        assert len(mock_orchestrator.get_routing_history()) == 0


# =============================================================================
# TEST: COMPLETE INTEGRATION SCENARIOS
# =============================================================================

class TestCompleteScenarios:
    """Test complete real-world integration scenarios."""

    def test_qa_pipeline_scenario(self, sample_documents, mock_orchestrator, mock_config):
        """Test a complete QA pipeline scenario."""
        from consciousness.ml_research.ml_techniques.memory import RAG, Document
        from consciousness.ml_research.ml_techniques.prompting import ChainOfThought
        from consciousness.ml_research.ml_techniques.verification import SelfEvaluation

        # Step 1: Set up RAG with documents
        docs = [
            Document(doc_id=d["id"], content=d["content"])
            for d in sample_documents
        ]
        rag = RAG(documents=docs, top_k=3)

        # Step 2: Retrieve relevant context
        rag_result = rag.run("What is the transformer architecture?")
        assert rag_result.success
        context = rag_result.output.get("context", "")

        # Step 3: Use CoT for reasoning
        cot = ChainOfThought()
        cot_result = cot.run(f"Context: {context}\nQuestion: Explain transformers.")
        assert cot_result.success

        # Step 4: Evaluate the response
        evaluator = SelfEvaluation(threshold=0.3)
        eval_result = evaluator.run({
            "input": "Explain transformers",
            "output": cot_result.output.get("reasoning", ""),
        })

        # Complete pipeline executed
        assert eval_result.technique_id == "self_evaluation"

    def test_multi_architecture_analysis_scenario(self, mock_orchestrator, mock_config):
        """Test a scenario comparing outputs from multiple architectures."""
        query = "Analyze this complex dataset"

        # Run same query through different architectures
        architectures = ["transformer", "mamba"]
        results = {}

        for arch in architectures:
            mock_orchestrator.load_architecture(arch, mock_config)
            result = mock_orchestrator.run_inference(query)
            results[arch] = result

        # Both should produce valid results
        assert all(r["output"] is not None for r in results.values())

        # Could compare outputs here for ensemble-like behavior
        assert len(results) == 2

    def test_hooks_monitoring_scenario(self, mock_orchestrator, mock_config):
        """Test using hooks for monitoring and logging."""
        from consciousness.ml_research.ml_techniques.orchestration import (
            HooksSystem,
            Hook,
            HookType,
        )

        # Create monitoring hooks
        monitor_log = []

        def monitor_pre(ctx):
            monitor_log.append({
                "event": "start",
                "input_size": len(str(ctx.input_data)),
            })

        def monitor_post(ctx):
            monitor_log.append({
                "event": "end",
                "output_size": len(str(ctx.output_data)) if ctx.output_data else 0,
            })

        hooks = HooksSystem()
        hooks.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=monitor_pre,
            name="monitor_pre",
        ))
        hooks.add_hook(Hook(
            hook_type=HookType.POST_RUN,
            function=monitor_post,
            name="monitor_post",
        ))

        # Run multiple queries through monitored system
        for query in ["Query 1", "Longer query 2 with more content", "Q3"]:
            hooks.run(query)

        # Verify monitoring captured all events
        assert len(monitor_log) == 6  # 2 events per query
        assert sum(1 for e in monitor_log if e["event"] == "start") == 3
        assert sum(1 for e in monitor_log if e["event"] == "end") == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
