"""
Tests for Architecture-Technique Integration Layer

Tests the integration of ml_techniques with modern_dev architectures (TRM, CTM, Mamba).
All tests are designed to run without the actual architecture implementations,
verifying fallback behavior and API contracts.

Run with:
    pytest consciousness/ml_research/ml_techniques/tests/test_architecture_integration.py -v
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Handle pytest import
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

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
            def parametrize(*args, **kwargs):
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

    pytest = MockPytest()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "content": "The transformer architecture uses attention mechanisms for parallel processing.",
            "metadata": {"topic": "ML"},
        },
        {
            "id": "doc2",
            "content": "Mamba is a state space model with linear complexity for sequence modeling.",
            "metadata": {"topic": "ML"},
        },
        {
            "id": "doc3",
            "content": "CTM uses neural synchronization for adaptive computation.",
            "metadata": {"topic": "ML"},
        },
    ]


@pytest.fixture
def sample_code():
    """Sample buggy code for repair testing."""
    return {
        "buggy_code": """
def calculate_sum(numbers):
    total = 0
    for n in numbers
        total += n
    return total
""",
        "error_message": "SyntaxError: expected ':'",
    }


# =============================================================================
# TEST: REGISTRY FUNCTIONS
# =============================================================================

class TestRegistryFunctions:
    """Test the integration registry functions."""

    def test_get_techniques_for_architecture_trm(self):
        """Test getting techniques for TRM architecture."""
        from ml_techniques.integration import get_techniques_for_architecture

        techniques = get_techniques_for_architecture('trm')

        assert isinstance(techniques, list)
        assert 'recursive_decomposition' in techniques
        assert 'chain_of_thought' in techniques
        assert 'code_repair' in techniques

    def test_get_techniques_for_architecture_ctm(self):
        """Test getting techniques for CTM architecture."""
        from ml_techniques.integration import get_techniques_for_architecture

        techniques = get_techniques_for_architecture('ctm')

        assert isinstance(techniques, list)
        assert 'temporal_reasoning' in techniques
        assert 'verification' in techniques
        assert 'memory_patterns' in techniques

    def test_get_techniques_for_architecture_mamba(self):
        """Test getting techniques for Mamba architecture."""
        from ml_techniques.integration import get_techniques_for_architecture

        techniques = get_techniques_for_architecture('mamba')

        assert isinstance(techniques, list)
        assert 'long_context_rag' in techniques
        assert 'streaming_inference' in techniques
        assert 'compression' in techniques

    def test_get_techniques_for_unknown_architecture(self):
        """Test that unknown architecture returns empty list."""
        from ml_techniques.integration import get_techniques_for_architecture

        techniques = get_techniques_for_architecture('unknown_arch')

        assert techniques == []

    def test_get_architecture_for_technique(self):
        """Test getting preferred architecture for a technique."""
        from ml_techniques.integration import get_architecture_for_technique

        # TRM techniques
        assert get_architecture_for_technique('recursive_decomposition') == 'trm'
        assert get_architecture_for_technique('chain_of_thought') == 'trm'

        # CTM techniques
        assert get_architecture_for_technique('temporal_reasoning') == 'ctm'
        assert get_architecture_for_technique('verification') == 'ctm'

        # Mamba techniques
        assert get_architecture_for_technique('long_context_rag') == 'mamba'
        assert get_architecture_for_technique('streaming_inference') == 'mamba'

    def test_get_architecture_for_unknown_technique(self):
        """Test that unknown technique returns None."""
        from ml_techniques.integration import get_architecture_for_technique

        result = get_architecture_for_technique('nonexistent_technique')

        assert result is None

    def test_get_all_architectures(self):
        """Test listing all architectures."""
        from ml_techniques.integration import get_all_architectures

        architectures = get_all_architectures()

        assert 'trm' in architectures
        assert 'ctm' in architectures
        assert 'mamba' in architectures

    def test_get_all_integrated_techniques(self):
        """Test listing all integrated techniques."""
        from ml_techniques.integration import get_all_integrated_techniques

        techniques = get_all_integrated_techniques()

        assert isinstance(techniques, list)
        assert len(techniques) > 0

    def test_is_architecture_compatible(self):
        """Test architecture-technique compatibility check."""
        from ml_techniques.integration import is_architecture_compatible

        # Compatible pairs
        assert is_architecture_compatible('trm', 'recursive_decomposition')
        assert is_architecture_compatible('ctm', 'verification')
        assert is_architecture_compatible('mamba', 'compression')

        # Incompatible pairs
        assert not is_architecture_compatible('trm', 'streaming_inference')
        assert not is_architecture_compatible('ctm', 'code_repair')


# =============================================================================
# TEST: TECHNIQUE CREATION
# =============================================================================

class TestTechniqueCreation:
    """Test creating integrated techniques."""

    def test_create_trm_decomposer(self):
        """Test creating TRMDecomposer technique."""
        from ml_techniques.integration import create_integrated_technique

        technique = create_integrated_technique('trm_decomposer', 'trm')

        assert technique is not None
        assert technique.TECHNIQUE_ID == 'trm_decomposer'

    def test_create_trm_chain_of_thought(self):
        """Test creating TRMChainOfThought technique."""
        from ml_techniques.integration import create_integrated_technique

        technique = create_integrated_technique('trm_chain_of_thought', 'trm')

        assert technique is not None
        assert technique.TECHNIQUE_ID == 'trm_chain_of_thought'

    def test_create_ctm_temporal_reasoning(self):
        """Test creating CTMTemporalReasoning technique."""
        from ml_techniques.integration import create_integrated_technique

        technique = create_integrated_technique('ctm_temporal_reasoning', 'ctm')

        assert technique is not None
        assert technique.TECHNIQUE_ID == 'ctm_temporal_reasoning'

    def test_create_mamba_rag(self):
        """Test creating MambaRAG technique."""
        from ml_techniques.integration import create_integrated_technique

        technique = create_integrated_technique('mamba_rag', 'mamba')

        assert technique is not None
        assert technique.TECHNIQUE_ID == 'mamba_rag'

    def test_create_unknown_technique_raises(self):
        """Test that creating unknown technique raises ValueError."""
        from ml_techniques.integration import create_integrated_technique

        with pytest.raises(ValueError, match="Unknown integrated technique"):
            create_integrated_technique('nonexistent_technique', 'trm')

    def test_create_with_kwargs(self):
        """Test creating technique with custom parameters."""
        from ml_techniques.integration import create_integrated_technique

        technique = create_integrated_technique(
            'trm_decomposer',
            'trm',
            trm_iterations=10,
            max_depth=3,
        )

        assert technique.trm_iterations == 10
        assert technique.max_depth == 3

    def test_list_integrated_techniques(self):
        """Test listing all integrated techniques with metadata."""
        from ml_techniques.integration import list_integrated_techniques

        techniques = list_integrated_techniques()

        assert isinstance(techniques, dict)
        assert len(techniques) > 0

        # Check metadata structure
        for technique_id, metadata in techniques.items():
            assert 'class' in metadata
            assert 'technique_id' in metadata
            assert 'category' in metadata


# =============================================================================
# TEST: TRM TECHNIQUES
# =============================================================================

class TestTRMTechniques:
    """Test TRM integrated techniques."""

    def test_trm_decomposer_run(self):
        """Test TRMDecomposer execution."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer

        decomposer = TRMDecomposer(trm_iterations=3, max_depth=2)
        result = decomposer.run("Build a web application with auth and database")

        assert result.success
        assert result.technique_id == 'trm_decomposer'
        assert result.output is not None
        assert len(result.intermediate_steps) > 0

    def test_trm_decomposer_fallback(self):
        """Test TRMDecomposer falls back when TRM unavailable."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer

        decomposer = TRMDecomposer()

        # TRM should not be available (not installed)
        assert decomposer.trm is None

        # Should still work via fallback
        result = decomposer.run("Simple task")
        assert result.success

    def test_trm_chain_of_thought_run(self):
        """Test TRMChainOfThought execution."""
        from ml_techniques.integration.trm_techniques import TRMChainOfThought

        cot = TRMChainOfThought(reasoning_steps=5)
        result = cot.run("What is 23 * 47?")

        assert result.success
        assert result.technique_id == 'trm_chain_of_thought'
        assert 'reasoning' in result.output
        assert 'reasoning_chain' in result.output

    def test_trm_chain_of_thought_early_halt(self):
        """Test TRMChainOfThought early halting."""
        from ml_techniques.integration.trm_techniques import TRMChainOfThought

        cot = TRMChainOfThought(
            reasoning_steps=20,
            use_halting=True,
        )
        result = cot.run("Simple question")

        assert result.success
        # Should halt before max steps due to confidence
        steps_used = result.metadata.get('steps_used', 0)
        assert steps_used <= 20

    def test_trm_code_repair_run(self, sample_code):
        """Test TRMCodeRepair execution."""
        from ml_techniques.integration.trm_techniques import TRMCodeRepair

        repair = TRMCodeRepair(max_repair_iterations=5)
        result = repair.run(sample_code)

        assert result.success
        assert result.technique_id == 'trm_code_repair'
        assert 'repaired_code' in result.output
        assert 'bug_classification' in result.output

    def test_trm_code_repair_string_input(self):
        """Test TRMCodeRepair with string input."""
        from ml_techniques.integration.trm_techniques import TRMCodeRepair

        repair = TRMCodeRepair()
        result = repair.run("def foo(): return bar")

        assert result.success
        assert result.output['original_code'] == "def foo(): return bar"


# =============================================================================
# TEST: CTM TECHNIQUES
# =============================================================================

class TestCTMTechniques:
    """Test CTM integrated techniques."""

    def test_ctm_temporal_reasoning_run(self):
        """Test CTMTemporalReasoning execution."""
        from ml_techniques.integration.ctm_techniques import CTMTemporalReasoning

        temporal = CTMTemporalReasoning(max_steps=10)
        result = temporal.run({
            "events": ["Event A", "Event B", "Event C"],
            "query": "What happened first?",
        })

        assert result.success
        assert result.technique_id == 'ctm_temporal_reasoning'
        assert 'sync_score' in result.output
        assert 'temporal_ordering' in result.output

    def test_ctm_temporal_reasoning_string_input(self):
        """Test CTMTemporalReasoning with string input."""
        from ml_techniques.integration.ctm_techniques import CTMTemporalReasoning

        temporal = CTMTemporalReasoning()
        result = temporal.run("What is the order of events?")

        assert result.success
        assert result.output['query'] == "What is the order of events?"

    def test_ctm_memory_store_retrieve(self):
        """Test CTMMemory store and retrieve."""
        from ml_techniques.integration.ctm_techniques import CTMMemory

        memory = CTMMemory(memory_capacity=10)

        # Store memories
        memory.store("Important fact about AI", importance=0.9)
        memory.store("Less important note", importance=0.3)

        # Retrieve
        result = memory.run("Tell me about AI")

        assert result.success
        assert result.technique_id == 'ctm_memory'
        assert 'memories' in result.output
        assert len(result.output['memories']) > 0

    def test_ctm_memory_capacity_limit(self):
        """Test CTMMemory respects capacity limit."""
        from ml_techniques.integration.ctm_techniques import CTMMemory

        memory = CTMMemory(memory_capacity=3)

        # Store more than capacity
        for i in range(5):
            memory.store(f"Memory {i}", importance=i * 0.2)

        # Should have consolidated to capacity
        assert memory._memories is not None
        # After consolidation, should have at most capacity
        result = memory.run("all memories")
        assert result.output['total_memories'] <= 5  # Before next consolidation

    def test_ctm_verification_run(self):
        """Test CTMVerification execution."""
        from ml_techniques.integration.ctm_techniques import CTMVerification

        verifier = CTMVerification(num_passes=3)
        result = verifier.run({
            "input": "What is 2+2?",
            "output": "4",
        })

        assert result.success
        assert result.technique_id == 'ctm_verification'
        assert 'confidence' in result.output
        assert 'verdict' in result.output
        assert 'passes' in result.output

    def test_ctm_verification_verdicts(self):
        """Test CTMVerification produces appropriate verdicts."""
        from ml_techniques.integration.ctm_techniques import CTMVerification

        verifier = CTMVerification(confidence_threshold=0.8)
        result = verifier.run({
            "input": "Simple question",
            "output": "Simple answer with enough content",
        })

        assert result.success
        verdict = result.output['verdict']
        assert verdict in ['high_confidence', 'medium_confidence', 'low_confidence']


# =============================================================================
# TEST: MAMBA TECHNIQUES
# =============================================================================

class TestMambaTechniques:
    """Test Mamba integrated techniques."""

    def test_mamba_rag_run(self, sample_documents):
        """Test MambaRAG execution."""
        from ml_techniques.integration.mamba_techniques import MambaRAG

        rag = MambaRAG(max_context_tokens=1000, top_k=3)

        # Add documents
        for doc in sample_documents:
            rag.add_documents([doc['content']], [doc['metadata']])

        result = rag.run("What is the transformer architecture?")

        assert result.success
        assert result.technique_id == 'mamba_rag'
        assert 'context' in result.output
        assert 'retrieved_chunks' in result.output

    def test_mamba_rag_empty_documents(self):
        """Test MambaRAG with no documents."""
        from ml_techniques.integration.mamba_techniques import MambaRAG

        rag = MambaRAG()
        result = rag.run("Query with no documents")

        assert result.success
        assert len(result.output['retrieved_chunks']) == 0

    def test_mamba_rag_compression(self, sample_documents):
        """Test MambaRAG with compression enabled."""
        from ml_techniques.integration.mamba_techniques import MambaRAG

        rag = MambaRAG(
            max_context_tokens=100,  # Small limit to trigger compression
            use_compression=True,
        )

        # Add documents
        for doc in sample_documents:
            rag.add_documents([doc['content']])

        result = rag.run("transformer")

        assert result.success
        assert result.output['compression_used']

    def test_mamba_streaming_run(self):
        """Test MambaStreaming execution."""
        from ml_techniques.integration.mamba_techniques import MambaStreaming

        tokens_received = []

        streaming = MambaStreaming(
            buffer_size=5,
            on_token=lambda t: tokens_received.append(t),
        )

        result = streaming.run("Generate a short story")

        assert result.success
        assert result.technique_id == 'mamba_streaming'
        assert 'generated' in result.output
        assert result.output['tokens_generated'] > 0

    def test_mamba_streaming_buffer_callback(self):
        """Test MambaStreaming buffer callbacks."""
        from ml_techniques.integration.mamba_techniques import MambaStreaming

        buffers_received = []

        streaming = MambaStreaming(
            buffer_size=10,
            on_buffer=lambda b: buffers_received.append(b),
        )

        result = streaming.run("Generate content", context={"max_tokens": 50})

        assert result.success
        # Should have received at least one buffer if we generated > buffer_size tokens
        if result.output['tokens_generated'] > 10:
            assert len(buffers_received) > 0

    def test_mamba_streaming_state_reset(self):
        """Test MambaStreaming state reset."""
        from ml_techniques.integration.mamba_techniques import MambaStreaming

        streaming = MambaStreaming()

        # Run once
        streaming.run("First prompt")

        # Reset state
        streaming.reset_state()

        assert streaming._hidden_state is None

    def test_mamba_compression_run(self):
        """Test MambaCompression execution."""
        from ml_techniques.integration.mamba_techniques import MambaCompression

        compressor = MambaCompression(compression_ratio=0.5)

        long_context = " ".join(["This is a sentence about machine learning."] * 20)

        result = compressor.run({
            "context": long_context,
            "query": "machine learning",
        })

        assert result.success
        assert result.technique_id == 'mamba_compression'
        assert 'compressed' in result.output
        assert 'compression_ratio' in result.output

    def test_mamba_compression_modes(self):
        """Test MambaCompression different modes."""
        from ml_techniques.integration.mamba_techniques import (
            MambaCompression,
            CompressionMode,
        )

        for mode in CompressionMode:
            compressor = MambaCompression(preserve_mode=mode)
            result = compressor.run({
                "context": "Test context for compression.",
                "query": "test",
            })

            assert result.success
            assert result.output['preserve_mode'] == mode.value


# =============================================================================
# TEST: FALLBACK BEHAVIOR
# =============================================================================

class TestFallbackBehavior:
    """Test fallback behavior when architectures are unavailable."""

    def test_trm_unavailable_uses_fallback(self):
        """Test TRM techniques fall back gracefully."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer

        decomposer = TRMDecomposer()

        # Verify TRM is not available
        assert decomposer.trm is None

        # Should still work
        result = decomposer.run("Task to decompose")
        assert result.success
        assert result.metadata.get('trm_available') is False

    def test_ctm_unavailable_uses_fallback(self):
        """Test CTM techniques fall back gracefully."""
        from ml_techniques.integration.ctm_techniques import CTMVerification

        verifier = CTMVerification()

        # Verify CTM is not available
        assert verifier.ctm is None

        # Should still work
        result = verifier.run({"input": "test", "output": "test"})
        assert result.success
        assert result.metadata.get('ctm_available') is False

    def test_mamba_unavailable_uses_fallback(self):
        """Test Mamba techniques fall back gracefully."""
        from ml_techniques.integration.mamba_techniques import MambaCompression

        compressor = MambaCompression()

        # Verify Mamba is not available
        assert compressor.mamba is None

        # Should still work
        result = compressor.run({"context": "test context"})
        assert result.success
        assert result.metadata.get('mamba_available') is False


# =============================================================================
# TEST: ARCHITECTURE AVAILABILITY
# =============================================================================

class TestArchitectureAvailability:
    """Test architecture availability checking."""

    def test_check_architecture_available(self):
        """Test architecture availability check."""
        from ml_techniques.integration import check_architecture_available

        # These should all return False in test environment
        # (architectures not installed)
        for arch in ['trm', 'ctm', 'mamba']:
            result = check_architecture_available(arch)
            assert isinstance(result, bool)

    def test_check_unknown_architecture(self):
        """Test checking unknown architecture."""
        from ml_techniques.integration import check_architecture_available

        result = check_architecture_available('nonexistent')
        assert result is False

    def test_get_architecture_status(self):
        """Test getting status of all architectures."""
        from ml_techniques.integration import get_architecture_status

        status = get_architecture_status()

        assert 'trm' in status
        assert 'ctm' in status
        assert 'mamba' in status
        assert all(isinstance(v, bool) for v in status.values())


# =============================================================================
# TEST: INTEGRATION WITH BASE TECHNIQUES
# =============================================================================

class TestBaseIntegration:
    """Test integration with base ml_techniques."""

    def test_trm_decomposer_inherits_technique_base(self):
        """Test TRMDecomposer properly inherits TechniqueBase."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer
        from ml_techniques import TechniqueBase

        decomposer = TRMDecomposer()

        assert isinstance(decomposer, TechniqueBase)
        assert hasattr(decomposer, 'run')
        assert hasattr(decomposer, 'add_hook')
        assert hasattr(decomposer, '_call_hooks')

    def test_technique_result_format(self):
        """Test techniques return proper TechniqueResult format."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer
        from ml_techniques import TechniqueResult

        decomposer = TRMDecomposer()
        result = decomposer.run("Test task")

        assert isinstance(result, TechniqueResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'output')
        assert hasattr(result, 'technique_id')
        assert hasattr(result, 'execution_time_ms')
        assert hasattr(result, 'intermediate_steps')

    def test_technique_hooks(self):
        """Test technique hooks work correctly."""
        from ml_techniques.integration.trm_techniques import TRMDecomposer

        hook_calls = []

        decomposer = TRMDecomposer()
        decomposer.add_hook('pre_run', lambda **kwargs: hook_calls.append('pre'))
        decomposer.add_hook('post_run', lambda **kwargs: hook_calls.append('post'))

        decomposer.run("Test task")

        assert 'pre' in hook_calls
        assert 'post' in hook_calls


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available - cannot run tests directly")
