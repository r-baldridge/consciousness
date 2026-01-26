"""
Unit Tests for Code Repair Pipelines (INTEG-002)

Tests for:
- Configuration loading and serialization
- MambaTRMRLMPipeline
- CodeRepairPipeline
- CLI interface
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.code_repair import (
    CodeRepairConfig,
    CodeRepairPipeline,
    FileRepairResult,
    ModelPreset,
    RepairRequest,
    RepairResult,
    RepairStatus,
    repair_code,
)

from pipelines.mamba_trm_rlm import (
    MambaConfig,
    MambaTRMRLMPipeline,
    PipelineConfig,
    PipelineRequest,
    PipelineResponse,
    RLMConfig,
    TaskDecomposition,
    TaskType,
    TRMConfig,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestCodeRepairConfig:
    """Tests for CodeRepairConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CodeRepairConfig()

        assert config.model_preset == "base"
        assert config.max_context_length == 4096
        assert config.max_repair_attempts == 3
        assert config.use_mamba_context is True
        assert config.use_rlm_decomposition is True
        assert config.device == "auto"
        assert config.validate_output is True
        assert config.language == "python"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CodeRepairConfig(model_preset="small", max_repair_attempts=5)
        d = config.to_dict()

        assert d["model_preset"] == "small"
        assert d["max_repair_attempts"] == 5
        assert "device" in d

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_preset": "large",
            "max_context_length": 8192,
            "validate_output": False,
        }
        config = CodeRepairConfig.from_dict(data)

        assert config.model_preset == "large"
        assert config.max_context_length == 8192
        assert config.validate_output is False

    def test_config_yaml_roundtrip(self):
        """Test saving and loading config from YAML."""
        config = CodeRepairConfig(
            model_preset="small",
            max_repair_attempts=5,
            language="javascript",
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save_yaml(f.name)
            loaded = CodeRepairConfig.from_yaml(f.name)
            os.unlink(f.name)

        assert loaded.model_preset == config.model_preset
        assert loaded.max_repair_attempts == config.max_repair_attempts
        assert loaded.language == config.language

    def test_invalid_preset_raises(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError):
            CodeRepairConfig(model_preset="invalid")


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()

        assert isinstance(config.mamba, MambaConfig)
        assert isinstance(config.trm, TRMConfig)
        assert isinstance(config.rlm, RLMConfig)
        assert config.device == "auto"

    def test_nested_config_from_dict(self):
        """Test creating config with nested components."""
        data = {
            "mamba": {"d_model": 512, "n_layers": 12},
            "trm": {"max_iterations": 4},
            "rlm": {"use_templates": False},
            "device": "cpu",
        }
        config = PipelineConfig.from_dict(data)

        assert config.mamba.d_model == 512
        assert config.mamba.n_layers == 12
        assert config.trm.max_iterations == 4
        assert config.rlm.use_templates is False
        assert config.device == "cpu"

    def test_config_yaml_roundtrip(self):
        """Test YAML serialization."""
        config = PipelineConfig(
            mamba=MambaConfig(d_model=256),
            trm=TRMConfig(max_iterations=4),
            device="cpu",
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save_yaml(f.name)
            loaded = PipelineConfig.from_yaml(f.name)
            os.unlink(f.name)

        assert loaded.mamba.d_model == config.mamba.d_model
        assert loaded.trm.max_iterations == config.trm.max_iterations
        assert loaded.device == config.device


class TestMambaConfig:
    """Tests for MambaConfig."""

    def test_defaults(self):
        """Test default values."""
        config = MambaConfig()

        assert config.d_model == 768
        assert config.n_layers == 24
        assert config.d_state == 16
        assert config.max_context_length == 100000

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = MambaConfig(d_model=512)
        d = config.to_dict()

        assert d["d_model"] == 512
        assert "n_layers" in d


class TestTRMConfig:
    """Tests for TRMConfig."""

    def test_defaults(self):
        """Test default values."""
        config = TRMConfig()

        assert config.grid_height == 64
        assert config.grid_width == 48
        assert config.max_iterations == 8
        assert config.use_halting is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = TRMConfig(max_iterations=4)
        d = config.to_dict()

        assert d["max_iterations"] == 4


class TestRLMConfig:
    """Tests for RLMConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RLMConfig()

        assert config.max_generation_attempts == 3
        assert config.use_templates is True
        assert config.variable_extraction_mode == "hybrid"


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestRepairRequest:
    """Tests for RepairRequest."""

    def test_basic_request(self):
        """Test basic request creation."""
        request = RepairRequest(code="def foo(): retrun 1")

        assert request.code == "def foo(): retrun 1"
        assert request.error_message is None
        assert request.context is None

    def test_full_request(self):
        """Test request with all fields."""
        request = RepairRequest(
            code="def foo(): retrun 1",
            error_message="SyntaxError",
            context="# File context",
            test_cases=[{"inputs": {}, "expected": 1}],
            file_path="/path/to/file.py",
            line_range=(10, 20),
            language="python",
        )

        assert request.error_message == "SyntaxError"
        assert request.line_range == (10, 20)

    def test_to_dict(self):
        """Test dictionary conversion."""
        request = RepairRequest(code="test", error_message="error")
        d = request.to_dict()

        assert d["code"] == "test"
        assert d["error_message"] == "error"


class TestRepairResult:
    """Tests for RepairResult."""

    def test_successful_result(self):
        """Test successful repair result."""
        result = RepairResult(
            original_code="def foo(): retrun 1",
            repaired_code="def foo(): return 1",
            status=RepairStatus.SUCCESS,
            confidence=0.95,
        )

        assert result.is_success is True
        assert result.confidence == 0.95

    def test_failed_result(self):
        """Test failed repair result."""
        result = RepairResult(
            original_code="broken",
            repaired_code="broken",
            status=RepairStatus.FAILED,
            confidence=0.0,
        )

        assert result.is_success is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = RepairResult(
            original_code="a",
            repaired_code="b",
            status=RepairStatus.SUCCESS,
            confidence=0.8,
            explanation="Fixed typo",
        )
        d = result.to_dict()

        assert d["status"] == "success"
        assert d["confidence"] == 0.8
        assert d["explanation"] == "Fixed typo"


class TestTaskDecomposition:
    """Tests for TaskDecomposition."""

    def test_basic_decomposition(self):
        """Test basic decomposition."""
        decomp = TaskDecomposition(
            task_type=TaskType.CODE_REPAIR,
            subproblems=["Identify bug", "Fix syntax"],
            variables=[{"name": "x", "type": "int"}],
            constraints=[],
            dependencies=[],
            reasoning_trace=["Step 1", "Step 2"],
            confidence=0.7,
        )

        assert decomp.task_type == TaskType.CODE_REPAIR
        assert len(decomp.subproblems) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        decomp = TaskDecomposition(
            task_type=TaskType.CODE_REPAIR,
            subproblems=["Fix bug"],
            variables=[],
            constraints=[],
            dependencies=[],
            reasoning_trace=[],
            confidence=0.5,
        )
        d = decomp.to_dict()

        assert d["task_type"] == "code_repair"
        assert d["confidence"] == 0.5


class TestPipelineRequest:
    """Tests for PipelineRequest."""

    def test_basic_request(self):
        """Test basic request."""
        request = PipelineRequest(
            task_type="code_repair",
            input_data={"buggy_code": "test"},
        )

        assert request.task_type == "code_repair"
        assert request.input_data["buggy_code"] == "test"

    def test_to_dict(self):
        """Test dictionary conversion."""
        request = PipelineRequest(
            task_type="code_repair",
            input_data={"code": "test"},
            context={"file": "example.py"},
        )
        d = request.to_dict()

        assert d["task_type"] == "code_repair"
        assert d["context"]["file"] == "example.py"


class TestPipelineResponse:
    """Tests for PipelineResponse."""

    def test_successful_response(self):
        """Test successful response."""
        response = PipelineResponse(
            success=True,
            output="def foo(): return 1",
            confidence=0.9,
            components_used=["mamba", "trm"],
        )

        assert response.success is True
        assert response.confidence == 0.9
        assert "mamba" in response.components_used

    def test_to_dict(self):
        """Test dictionary conversion."""
        response = PipelineResponse(
            success=True,
            output="result",
            confidence=0.8,
            execution_time_ms=100.0,
        )
        d = response.to_dict()

        assert d["success"] is True
        assert d["confidence"] == 0.8


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestMambaTRMRLMPipeline:
    """Tests for MambaTRMRLMPipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation with default config."""
        pipeline = MambaTRMRLMPipeline()

        assert pipeline.config is not None
        assert pipeline.device in ["cpu", "cuda", "mps"]

    def test_pipeline_from_pretrained(self):
        """Test loading pretrained configurations."""
        for preset in ["tiny", "small", "base", "large"]:
            pipeline = MambaTRMRLMPipeline.from_pretrained(preset)
            assert pipeline is not None

    def test_pipeline_process(self):
        """Test basic pipeline processing."""
        pipeline = MambaTRMRLMPipeline.from_pretrained("tiny")

        request = PipelineRequest(
            task_type="code_repair",
            input_data={"buggy_code": "def foo(): retrun 1"},
        )

        response = pipeline.process(request)

        assert isinstance(response, PipelineResponse)
        assert response.output is not None

    def test_pipeline_with_context(self):
        """Test pipeline with context."""
        pipeline = MambaTRMRLMPipeline.from_pretrained("tiny")

        request = PipelineRequest(
            task_type="code_repair",
            input_data={"buggy_code": "x = calculate(1, 2)"},
            context={"file_content": "def calculate(a, b): return a + b"},
        )

        response = pipeline.process(request)

        assert response is not None

    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        pipeline = MambaTRMRLMPipeline.from_pretrained("tiny")

        # Process a request
        request = PipelineRequest(
            task_type="code_repair",
            input_data={"buggy_code": "test"},
        )
        pipeline.process(request)

        stats = pipeline.get_stats()

        assert stats["total_requests"] == 1
        assert "success_rate" in stats

    def test_pipeline_reset_stats(self):
        """Test resetting statistics."""
        pipeline = MambaTRMRLMPipeline.from_pretrained("tiny")

        # Process and reset
        request = PipelineRequest(task_type="test", input_data={})
        pipeline.process(request)
        pipeline.reset_stats()

        stats = pipeline.get_stats()
        assert stats["total_requests"] == 0


class TestCodeRepairPipeline:
    """Tests for CodeRepairPipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = CodeRepairPipeline()

        assert pipeline.config is not None
        assert pipeline.pipeline is not None

    def test_pipeline_from_pretrained(self):
        """Test loading pretrained pipeline."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        assert pipeline.config.model_preset == "tiny"

    def test_basic_repair(self):
        """Test basic code repair."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        result = pipeline.repair(
            buggy_code="def foo(): retrun 1",
            error_message="SyntaxError: invalid syntax",
        )

        assert isinstance(result, RepairResult)
        assert result.original_code == "def foo(): retrun 1"

    def test_repair_with_test_cases(self):
        """Test repair with test case validation."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        result = pipeline.repair(
            buggy_code="def add(a, b): return a - b",  # Bug: should be +
            test_cases=[
                {"inputs": {"a": 1, "b": 2}, "expected": 3},
            ],
        )

        assert result.test_results is not None

    def test_repair_file(self):
        """Test file-based repair."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        # Create a temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write("def foo(): retrun 1\n")
            temp_path = f.name

        try:
            result = pipeline.repair_file(temp_path)

            assert isinstance(result, FileRepairResult)
            assert result.file_path == temp_path
            assert result.original_content == "def foo(): retrun 1\n"
        finally:
            os.unlink(temp_path)

    def test_repair_file_with_line_range(self):
        """Test file repair with line range."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        # Create a temp file with multiple lines
        content = """def first():
    return 1

def buggy():
    retrun 2

def last():
    return 3
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = pipeline.repair_file(temp_path, line_range=(4, 5))

            assert result.line_range == (4, 5)
        finally:
            os.unlink(temp_path)

    def test_repair_batch(self):
        """Test batch repair."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        requests = [
            RepairRequest(code="def foo(): retrun 1"),
            RepairRequest(code="def bar(): prin('hi')"),
        ]

        results = pipeline.repair_batch(requests)

        assert len(results) == 2
        assert all(isinstance(r, RepairResult) for r in results)

    def test_validation(self):
        """Test code validation."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")
        pipeline.config.validate_output = True

        # Valid code
        result1 = pipeline.repair(buggy_code="x = 1")
        # Invalid code (if repair doesn't fix it)
        # result2 = pipeline.repair(buggy_code="def (:")

        assert result1.validation_result is not None

    def test_stats(self):
        """Test statistics tracking."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        pipeline.repair(buggy_code="x = 1")
        stats = pipeline.get_stats()

        assert stats["total_repairs"] == 1
        assert "success_rate" in stats

    def test_reset_stats(self):
        """Test resetting statistics."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        pipeline.repair(buggy_code="x = 1")
        pipeline.reset_stats()

        stats = pipeline.get_stats()
        assert stats["total_repairs"] == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_repair_code_function(self):
        """Test repair_code convenience function."""
        result = repair_code(
            buggy_code="def foo(): retrun 1",
            model="tiny",
        )

        assert isinstance(result, RepairResult)

    def test_repair_code_with_error(self):
        """Test repair_code with error message."""
        result = repair_code(
            buggy_code="def foo(): retrun 1",
            error_message="SyntaxError",
            model="tiny",
        )

        assert result is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEndToEndFlow:
    """End-to-end integration tests."""

    def test_simple_syntax_fix(self):
        """Test fixing a simple syntax error."""
        pipeline = CodeRepairPipeline.from_pretrained("tiny")

        result = pipeline.repair(
            buggy_code="def foo(): retrun 1",
            error_message="SyntaxError: invalid syntax"
        )

        assert result.status in [RepairStatus.SUCCESS, RepairStatus.PARTIAL, RepairStatus.FAILED]
        assert result.execution_time_ms > 0

    def test_full_pipeline_flow(self):
        """Test the complete pipeline flow."""
        # Create pipeline
        pipeline = MambaTRMRLMPipeline.from_pretrained("tiny")

        # Create request with context
        request = PipelineRequest(
            task_type="code_repair",
            input_data={
                "buggy_code": """
def calculate_sum(numbers):
    total = 0
    for n in numbrs:  # typo
        total += n
    return total
""",
            },
            context={
                "file_content": "# Utility functions\nimport math\n",
            },
        )

        # Process
        response = pipeline.process(request)

        # Verify response structure
        assert isinstance(response, PipelineResponse)
        assert response.output is not None
        assert response.execution_time_ms > 0
        assert len(response.components_used) > 0

    def test_config_driven_pipeline(self):
        """Test pipeline driven by YAML config."""
        # Create temp config file
        config_content = """
pipeline:
  model_preset: tiny
  max_repair_attempts: 2
  validate_output: true

mamba:
  d_model: 64
  n_layers: 2

trm:
  max_iterations: 2

rlm:
  max_generation_attempts: 1
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(config_content)
            config_path = f.name

        try:
            pipeline = CodeRepairPipeline.from_yaml(config_path)

            result = pipeline.repair(buggy_code="x = 1")

            assert pipeline.config.model_preset == "tiny"
            assert pipeline.config.max_repair_attempts == 2
        finally:
            os.unlink(config_path)


# =============================================================================
# RUN TESTS
# =============================================================================


def run_tests():
    """Run all tests with verbose output."""
    print("=" * 70)
    print("CODE REPAIR PIPELINE TEST SUITE (INTEG-002)")
    print("=" * 70)

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
