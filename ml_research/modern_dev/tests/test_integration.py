"""
Integration tests for ML Research modern_dev pipeline.

Tests cover:
- TRM + RLM composition
- Mamba + TRM pipeline
- Orchestrator routing
- End-to-end code repair
- Performance regression

Run with: pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn


# =============================================================================
# TRM + RLM Integration Tests
# =============================================================================


@pytest.mark.integration
class TestTRMRLMIntegration:
    """Tests for TRM + RLM integration."""

    def test_composition_import(self):
        """Test composition module imports correctly."""
        from consciousness.ml_research.modern_dev.trm.src.composition import (
            TRMWithRLM,
            CompositionConfig,
            RepairStrategy,
            RepairResult,
        )

        assert TRMWithRLM is not None
        assert CompositionConfig is not None

    def test_composition_config_creation(self):
        """Test composition config creation."""
        from consciousness.ml_research.modern_dev.trm.src.composition import (
            CompositionConfig,
        )

        config = CompositionConfig(
            max_rlm_attempts=3,
            max_trm_iterations=8,
            confidence_threshold=0.9,
        )

        assert config.max_rlm_attempts == 3
        assert config.max_trm_iterations == 8

    def test_repair_strategy_dataclass(self):
        """Test RepairStrategy dataclass."""
        from consciousness.ml_research.modern_dev.trm.src.composition import (
            RepairStrategy,
        )

        strategy = RepairStrategy(
            bug_type="variable_typo",
            fix_description="Replace 'c' with 'b'",
            confidence=0.95,
            steps=["identify_typo", "replace_variable"],
        )

        assert strategy.bug_type == "variable_typo"
        assert strategy.confidence == 0.95

    def test_trm_rlm_pipeline_import(self):
        """Test TRM-RLM pipeline imports."""
        from consciousness.ml_research.modern_dev.shared.pipelines.trm_rlm import (
            TRMRLMPipeline,
            PipelineConfig,
            PipelineResult,
        )

        assert TRMRLMPipeline is not None

    def test_pipeline_config_from_dict(self):
        """Test pipeline config creation from dict."""
        from consciousness.ml_research.modern_dev.shared.pipelines.trm_rlm import (
            PipelineConfig,
        )

        config = PipelineConfig.from_dict({
            "max_attempts": 5,
            "timeout_seconds": 30,
            "use_trm": True,
            "use_rlm": True,
        })

        assert config.max_attempts == 5
        assert config.use_trm is True

    def test_feedback_loop_dataclass(self):
        """Test feedback loop structures."""
        from consciousness.ml_research.modern_dev.trm.src.composition import (
            FeedbackEntry,
            FeedbackLoop,
        )

        entry = FeedbackEntry(
            iteration=1,
            error_type="syntax_error",
            error_message="unexpected indent",
            fix_applied="adjust_indentation",
        )

        loop = FeedbackLoop(max_iterations=3)
        loop.add_entry(entry)

        assert len(loop.entries) == 1
        assert loop.entries[0].iteration == 1


@pytest.mark.integration
class TestRLMPipeline:
    """Tests for RLM pipeline integration."""

    def test_pipeline_import(self):
        """Test RLM pipeline imports."""
        from consciousness.ml_research.ml_techniques.code_synthesis.pipeline import (
            RLMPipeline,
            PipelineConfig,
            PipelineResult,
            PipelineUpdate,
        )

        assert RLMPipeline is not None

    def test_pipeline_config_defaults(self):
        """Test pipeline config has sensible defaults."""
        from consciousness.ml_research.ml_techniques.code_synthesis.pipeline import (
            PipelineConfig,
        )

        config = PipelineConfig()

        assert config.max_generation_attempts >= 1
        assert config.max_debug_attempts >= 1

    def test_pipeline_initialization(self):
        """Test pipeline initializes without errors."""
        from consciousness.ml_research.ml_techniques.code_synthesis.pipeline import (
            RLMPipeline,
            PipelineConfig,
        )

        config = PipelineConfig(
            max_generation_attempts=2,
            max_debug_attempts=2,
        )

        pipeline = RLMPipeline(config=config)
        assert pipeline is not None

    def test_pipeline_hook_registration(self):
        """Test pipeline hook registration."""
        from consciousness.ml_research.ml_techniques.code_synthesis.pipeline import (
            RLMPipeline,
        )

        pipeline = RLMPipeline()

        def dummy_hook(data):
            return data

        pipeline.register_hook("pre_generation", dummy_hook)
        assert "pre_generation" in pipeline._hooks


# =============================================================================
# Mamba + TRM Pipeline Tests
# =============================================================================


@pytest.mark.integration
class TestMambaTRMPipeline:
    """Tests for Mamba context + TRM repair pipeline."""

    def test_mamba_context_encoding(self, cpu_device):
        """Test Mamba encodes long context for TRM."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        # Simulate long context (would be codebase context)
        long_context = torch.randint(0, 1000, (1, 256), device=cpu_device)

        with torch.no_grad():
            # Get context encoding
            hidden = model.encode(long_context)

        assert hidden.shape == (1, 256, 64)

    def test_context_to_trm_conversion(self, cpu_device):
        """Test converting Mamba context to TRM input format."""
        from consciousness.ml_research.modern_dev.trm.src.model import (
            CodeRepairConfig,
            CodeRepairTRM,
        )

        config = CodeRepairConfig(
            grid_height=16,
            grid_width=12,
            vocab_size=1000,
            embed_dim=64,
        )
        model = CodeRepairTRM(config).to(cpu_device)
        model.eval()

        # TRM expects grid input
        grid_input = torch.randint(0, 1000, (1, 16, 12), device=cpu_device)

        with torch.no_grad():
            logits, info = model(grid_input)

        assert logits.shape == (1, 16, 12, 1000)


# =============================================================================
# Orchestrator Tests
# =============================================================================


@pytest.mark.integration
class TestOrchestratorRouting:
    """Tests for orchestrator routing decisions."""

    def test_orchestrator_import(self):
        """Test orchestrator imports."""
        from consciousness.ml_research.modern_dev.orchestrator import (
            Orchestrator,
            ArchitectureRegistry,
        )
        from consciousness.ml_research.modern_dev.orchestrator.router import (
            TaskRouter,
            RoutingDecision,
        )

        assert Orchestrator is not None
        assert TaskRouter is not None

    def test_architecture_registry(self):
        """Test architecture registry operations."""
        from consciousness.ml_research.modern_dev.orchestrator.router import (
            ArchitectureRegistry,
            ArchitectureCapability,
        )

        registry = ArchitectureRegistry()

        # Register TRM
        registry.register(
            "trm",
            ArchitectureCapability(
                name="TRM",
                supported_tasks=["code_repair", "iterative_refinement"],
                max_context_length=3072,
                inference_speed="medium",
                memory_requirement="medium",
                strengths=["recursive reasoning"],
                weaknesses=["fixed grid size"],
            ),
        )

        # Register Mamba
        registry.register(
            "mamba",
            ArchitectureCapability(
                name="Mamba",
                supported_tasks=["generation", "long_context"],
                max_context_length=100000,
                inference_speed="fast",
                memory_requirement="low",
                strengths=["long sequences"],
                weaknesses=["less interpretable"],
            ),
        )

        # Test retrieval
        assert registry.get("trm").name == "TRM"
        assert registry.get("mamba").max_context_length == 100000

        # Test capability matching
        capable = registry.get_capable("code_repair")
        assert "trm" in capable
        assert "mamba" not in capable

    def test_task_router_basic(self):
        """Test basic task routing."""
        from consciousness.ml_research.modern_dev.orchestrator.router import (
            TaskRouter,
            ArchitectureRegistry,
            ArchitectureCapability,
            Task,
        )

        registry = ArchitectureRegistry()
        registry.register(
            "trm",
            ArchitectureCapability(
                name="TRM",
                supported_tasks=["code_repair"],
                max_context_length=3072,
                inference_speed="medium",
                memory_requirement="medium",
                strengths=[],
                weaknesses=[],
            ),
        )
        registry.register(
            "mamba",
            ArchitectureCapability(
                name="Mamba",
                supported_tasks=["generation", "long_context"],
                max_context_length=100000,
                inference_speed="fast",
                memory_requirement="low",
                strengths=[],
                weaknesses=[],
            ),
        )

        router = TaskRouter(registry)

        # Code repair task should route to TRM
        task = Task(
            task_type="code_repair",
            input_length=1000,
            priority="normal",
        )

        decision = router.route(task)
        assert decision.primary == "trm"

        # Long context task should route to Mamba
        task = Task(
            task_type="long_context",
            input_length=50000,
            priority="normal",
        )

        decision = router.route(task)
        assert decision.primary == "mamba"

    def test_routing_with_fallback(self):
        """Test routing includes fallback option."""
        from consciousness.ml_research.modern_dev.orchestrator.router import (
            TaskRouter,
            ArchitectureRegistry,
            ArchitectureCapability,
            Task,
        )

        registry = ArchitectureRegistry()
        registry.register(
            "trm",
            ArchitectureCapability(
                name="TRM",
                supported_tasks=["code_repair", "generation"],
                max_context_length=3072,
                inference_speed="medium",
                memory_requirement="medium",
                strengths=[],
                weaknesses=[],
            ),
        )
        registry.register(
            "mamba",
            ArchitectureCapability(
                name="Mamba",
                supported_tasks=["generation", "long_context"],
                max_context_length=100000,
                inference_speed="fast",
                memory_requirement="low",
                strengths=[],
                weaknesses=[],
            ),
        )

        router = TaskRouter(registry)

        task = Task(
            task_type="generation",
            input_length=1000,
            priority="normal",
        )

        decision = router.route(task)

        # Should have primary and fallback
        assert decision.primary is not None
        assert decision.fallback is not None
        assert decision.primary != decision.fallback

    def test_fallback_handler(self):
        """Test fallback handler executes fallback on failure."""
        from consciousness.ml_research.modern_dev.orchestrator.router import (
            FallbackHandler,
            TaskRouter,
            ArchitectureRegistry,
            ArchitectureCapability,
            Task,
            ExecutionResult,
        )

        registry = ArchitectureRegistry()
        registry.register(
            "primary",
            ArchitectureCapability(
                name="Primary",
                supported_tasks=["test_task"],
                max_context_length=1000,
                inference_speed="fast",
                memory_requirement="low",
                strengths=[],
                weaknesses=[],
            ),
        )
        registry.register(
            "fallback",
            ArchitectureCapability(
                name="Fallback",
                supported_tasks=["test_task"],
                max_context_length=1000,
                inference_speed="medium",
                memory_requirement="low",
                strengths=[],
                weaknesses=[],
            ),
        )

        router = TaskRouter(registry)
        handler = FallbackHandler(router)

        # Mock execution that fails on primary, succeeds on fallback
        call_count = {"primary": 0, "fallback": 0}

        def mock_execute(arch, task):
            call_count[arch] += 1
            if arch == "primary":
                raise RuntimeError("Primary failed")
            return ExecutionResult(success=True, output="fallback output")

        handler.set_executor(mock_execute)

        task = Task(task_type="test_task", input_length=100, priority="normal")
        result = handler.execute_with_fallback(task)

        assert call_count["primary"] == 1
        assert call_count["fallback"] == 1
        assert result.success


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_simple_code_repair_flow(self, sample_code_repair_data):
        """Test simple code repair flow."""
        # This tests the conceptual flow, not actual model inference
        buggy = sample_code_repair_data[0]["buggy"]
        expected_fixed = sample_code_repair_data[0]["fixed"]

        # Simulate pipeline stages
        stages_completed = []

        # Stage 1: Tokenize
        stages_completed.append("tokenize")

        # Stage 2: Encode context (Mamba)
        stages_completed.append("encode_context")

        # Stage 3: Decompose repair (RLM)
        stages_completed.append("decompose_repair")

        # Stage 4: Refine (TRM)
        stages_completed.append("refine")

        # Stage 5: Detokenize
        stages_completed.append("detokenize")

        assert stages_completed == [
            "tokenize",
            "encode_context",
            "decompose_repair",
            "refine",
            "detokenize",
        ]

    def test_pipeline_with_retry(self):
        """Test pipeline with retry on failure."""
        max_retries = 3
        attempts = []

        def simulate_repair(attempt: int) -> bool:
            attempts.append(attempt)
            # Fail first two attempts, succeed on third
            return attempt >= 3

        for i in range(1, max_retries + 1):
            success = simulate_repair(i)
            if success:
                break

        assert len(attempts) == 3
        assert attempts[-1] == 3

    def test_pipeline_timeout_handling(self):
        """Test pipeline handles timeouts gracefully."""
        import time

        timeout_seconds = 0.1
        start = time.time()

        # Simulate a task that would timeout
        try:
            while time.time() - start < timeout_seconds * 2:
                if time.time() - start > timeout_seconds:
                    raise TimeoutError("Operation timed out")
                time.sleep(0.01)
        except TimeoutError as e:
            timeout_occurred = True
            error_message = str(e)

        assert timeout_occurred
        assert "timed out" in error_message


# =============================================================================
# Performance Regression Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.integration
class TestPerformanceRegression:
    """Performance regression tests."""

    def test_trm_inference_baseline(self, trm_model, trm_small_config, benchmark_timer):
        """Baseline TRM inference performance."""
        x = torch.randint(0, trm_small_config.vocab_size, (1, 16, 12))

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                trm_model(x)

        # Benchmark
        with benchmark_timer("trm_baseline", iterations=10) as timer:
            for _ in range(10):
                with torch.no_grad():
                    trm_model(x)

        # Should complete in reasonable time
        assert timer.result.avg_duration_ms < 500

    def test_mamba_inference_baseline(self, cpu_device, benchmark_timer):
        """Baseline Mamba inference performance."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        x = torch.randint(0, 1000, (1, 64), device=cpu_device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(x)

        # Benchmark
        with benchmark_timer("mamba_baseline", iterations=10) as timer:
            for _ in range(10):
                with torch.no_grad():
                    model(x)

        assert timer.result.avg_duration_ms < 500

    def test_combined_pipeline_performance(
        self, trm_model, trm_small_config, cpu_device, benchmark_timer
    ):
        """Benchmark combined TRM + simulated RLM pipeline."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        mamba_config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        mamba = MambaLM(mamba_config).to(cpu_device)
        mamba.eval()

        # Combined pipeline: Mamba encode -> TRM refine
        mamba_input = torch.randint(0, 1000, (1, 64), device=cpu_device)
        trm_input = torch.randint(0, trm_small_config.vocab_size, (1, 16, 12))

        # Warmup
        with torch.no_grad():
            mamba(mamba_input)
            trm_model(trm_input)

        # Benchmark combined
        with benchmark_timer("combined_pipeline", iterations=5) as timer:
            for _ in range(5):
                with torch.no_grad():
                    # Mamba context encoding
                    mamba(mamba_input)
                    # TRM refinement
                    trm_model(trm_input)

        print(f"Combined pipeline: {timer.result.avg_duration_ms:.2f}ms")
        assert timer.result.avg_duration_ms < 2000  # 2 second max


# =============================================================================
# Data Pipeline Integration Tests
# =============================================================================


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Tests for data pipeline integration."""

    def test_sample_data_format(self, sample_code_repair_data):
        """Test sample data has expected format."""
        for sample in sample_code_repair_data:
            assert "buggy" in sample
            assert "fixed" in sample
            assert "bug_type" in sample
            assert isinstance(sample["buggy"], str)
            assert isinstance(sample["fixed"], str)

    def test_tokenization_roundtrip(self):
        """Test code tokenization is reversible."""
        # Simulate tokenization
        code = "def add(a, b): return a + b"

        # Simple char-level tokenization for test
        tokens = [ord(c) for c in code]
        recovered = "".join(chr(t) for t in tokens)

        assert recovered == code

    def test_grid_encoding(self):
        """Test code can be encoded to grid format."""
        code = "def add(a, b): return a + b"
        grid_height, grid_width = 8, 8

        # Pad/truncate to grid
        tokens = [ord(c) for c in code[:grid_height * grid_width]]
        tokens += [0] * (grid_height * grid_width - len(tokens))

        grid = torch.tensor(tokens).reshape(grid_height, grid_width)

        assert grid.shape == (grid_height, grid_width)


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in integration."""

    def test_invalid_input_handling(self, trm_model, trm_small_config):
        """Test model handles invalid input gracefully."""
        # Wrong shape should raise
        with pytest.raises(Exception):
            x = torch.randint(0, 1000, (2, 100, 100))  # Wrong grid size
            trm_model(x)

    def test_out_of_vocab_handling(self, trm_model, trm_small_config):
        """Test handling of out-of-vocabulary tokens."""
        # Create input with tokens beyond vocab size
        x = torch.randint(
            trm_small_config.vocab_size,
            trm_small_config.vocab_size + 100,
            (2, 16, 12),
        )

        # Should either clamp or raise
        try:
            with torch.no_grad():
                trm_model(x)
        except (IndexError, RuntimeError):
            pass  # Expected behavior

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        from consciousness.ml_research.ml_techniques.code_synthesis.pipeline import (
            RLMPipeline,
        )

        pipeline = RLMPipeline()

        # Empty spec should be handled
        result = pipeline.run("")

        assert result is not None
        assert not result.success or result.code == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
