"""
Code Repair Pipeline (INTEG-002)

End-to-end code repair pipeline combining Mamba, TRM, and RLM.

This module provides the high-level interface for code repair tasks:
- Single code snippet repair
- File-based repair with line range selection
- Batch repair for multiple snippets
- Configuration via YAML or Python

The pipeline flow:
1. Input: buggy code + optional error message + optional context
2. Mamba: encode surrounding context (file, imports, etc.)
3. RLM: decompose repair strategy
4. TRM: iterative refinement
5. Output: fixed code + confidence + explanation

Example:
    from modern_dev.pipelines.code_repair import CodeRepairPipeline

    # Quick usage
    pipeline = CodeRepairPipeline()
    result = pipeline.repair(
        buggy_code="def foo(): retrun 1",
        error_message="SyntaxError: invalid syntax"
    )
    print(result.repaired_code)

    # From config file
    pipeline = CodeRepairPipeline.from_yaml("config/code_repair.yaml")

    # File repair
    result = pipeline.repair_file("src/buggy.py", line_range=(10, 20))

    # Batch repair
    results = pipeline.repair_batch([
        RepairRequest(code="..."),
        RepairRequest(code="...", error="..."),
    ])
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

# Handle both package and standalone imports
try:
    from .mamba_trm_rlm import (
        MambaTRMRLMPipeline,
        PipelineConfig,
        PipelineRequest,
        PipelineResponse,
        TaskDecomposition,
    )
except ImportError:
    from mamba_trm_rlm import (
        MambaTRMRLMPipeline,
        PipelineConfig,
        PipelineRequest,
        PipelineResponse,
        TaskDecomposition,
    )


logger = logging.getLogger(__name__)


# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================


class ModelPreset(Enum):
    """Available model presets."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


@dataclass
class CodeRepairConfig:
    """Configuration for code repair pipeline.

    Attributes:
        model_preset: Model size preset ("tiny", "small", "base", "large").
        max_context_length: Maximum context length in tokens.
        max_repair_attempts: Maximum repair attempts before giving up.
        use_mamba_context: Whether to use Mamba for context encoding.
        use_rlm_decomposition: Whether to use RLM for task decomposition.
        device: Device to run on ("auto", "cpu", "cuda", "mps").
        timeout_seconds: Maximum time for repair in seconds.
        validate_output: Whether to validate repaired code (syntax check).
        language: Target language for repair ("python", "javascript", etc.).
        include_explanation: Whether to include explanation in output.
        log_level: Logging level.
    """
    model_preset: str = "base"
    max_context_length: int = 4096
    max_repair_attempts: int = 3
    use_mamba_context: bool = True
    use_rlm_decomposition: bool = True
    device: str = "auto"
    timeout_seconds: float = 30.0
    validate_output: bool = True
    language: str = "python"
    include_explanation: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration."""
        valid_presets = [p.value for p in ModelPreset]
        if self.model_preset not in valid_presets:
            raise ValueError(
                f"Invalid model_preset: {self.model_preset}. "
                f"Must be one of: {valid_presets}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_preset": self.model_preset,
            "max_context_length": self.max_context_length,
            "max_repair_attempts": self.max_repair_attempts,
            "use_mamba_context": self.use_mamba_context,
            "use_rlm_decomposition": self.use_rlm_decomposition,
            "device": self.device,
            "timeout_seconds": self.timeout_seconds,
            "validate_output": self.validate_output,
            "language": self.language,
            "include_explanation": self.include_explanation,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeRepairConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CodeRepairConfig":
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle nested pipeline config
        if "pipeline" in data:
            return cls.from_dict(data["pipeline"])

        return cls.from_dict(data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({"pipeline": self.to_dict()}, f, default_flow_style=False)

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to full PipelineConfig."""
        return PipelineConfig(
            device=self.device,
            max_context_length=self.max_context_length,
            timeout_ms=self.timeout_seconds * 1000,
            log_level=self.log_level,
        )


# =============================================================================
# DATA CLASSES
# =============================================================================


class RepairStatus(Enum):
    """Status of a repair attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class RepairRequest:
    """Request for code repair.

    Attributes:
        code: Buggy code to repair.
        error_message: Optional error message.
        context: Optional surrounding context.
        test_cases: Optional test cases for validation.
        file_path: Optional source file path (for context).
        line_range: Optional line range within file.
        language: Target language (defaults to config).
    """
    code: str
    error_message: Optional[str] = None
    context: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    file_path: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "error_message": self.error_message,
            "context": self.context,
            "test_cases": self.test_cases,
            "file_path": self.file_path,
            "line_range": self.line_range,
            "language": self.language,
        }


@dataclass
class RepairResult:
    """Result of a code repair attempt.

    Attributes:
        original_code: The original buggy code.
        repaired_code: The repaired code (or original if failed).
        status: Status of the repair.
        confidence: Confidence in the repair (0.0 to 1.0).
        explanation: Optional explanation of changes.
        decomposition: Task decomposition from RLM.
        iterations: Number of TRM refinement iterations.
        execution_time_ms: Total execution time.
        validation_result: Result of syntax validation.
        test_results: Results of running test cases.
        metadata: Additional metadata.
    """
    original_code: str
    repaired_code: str
    status: RepairStatus
    confidence: float
    explanation: Optional[str] = None
    decomposition: Optional[TaskDecomposition] = None
    iterations: int = 0
    execution_time_ms: float = 0.0
    validation_result: Optional[Dict[str, Any]] = None
    test_results: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Whether the repair was successful."""
        return self.status == RepairStatus.SUCCESS

    @property
    def is_valid(self) -> bool:
        """Whether the repaired code is syntactically valid."""
        if self.validation_result:
            return self.validation_result.get("is_valid", False)
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_code": self.original_code,
            "repaired_code": self.repaired_code,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "decomposition": self.decomposition.to_dict() if self.decomposition else None,
            "iterations": self.iterations,
            "execution_time_ms": self.execution_time_ms,
            "validation_result": self.validation_result,
            "test_results": self.test_results,
            "metadata": self.metadata,
        }


@dataclass
class FileRepairResult:
    """Result of a file repair operation.

    Attributes:
        file_path: Path to the repaired file.
        original_content: Original file content.
        repaired_content: Repaired file content.
        line_range: Lines that were repaired.
        repair_result: Underlying RepairResult.
        changes_made: Whether any changes were made.
    """
    file_path: str
    original_content: str
    repaired_content: str
    line_range: Optional[Tuple[int, int]]
    repair_result: RepairResult
    changes_made: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_range": self.line_range,
            "repair_result": self.repair_result.to_dict(),
            "changes_made": self.changes_made,
        }


# =============================================================================
# CODE REPAIR PIPELINE
# =============================================================================


class CodeRepairPipeline:
    """
    End-to-end code repair pipeline.

    Flow:
    1. Input: buggy code + optional error message + optional context
    2. Mamba: encode surrounding context (file, imports, etc.)
    3. RLM: decompose repair strategy
    4. TRM: iterative refinement
    5. Output: fixed code + confidence + explanation

    Example:
        >>> pipeline = CodeRepairPipeline()
        >>>
        >>> # Simple repair
        >>> result = pipeline.repair(
        ...     buggy_code="def foo(): retrun 1",
        ...     error_message="SyntaxError: invalid syntax"
        ... )
        >>> print(result.repaired_code)
        def foo(): return 1
        >>>
        >>> # File repair
        >>> result = pipeline.repair_file("buggy.py", line_range=(10, 20))
        >>>
        >>> # Batch repair
        >>> results = pipeline.repair_batch([
        ...     RepairRequest(code="def foo(): retrun 1"),
        ...     RepairRequest(code="def bar(): prin('hi')"),
        ... ])

    Attributes:
        config: Code repair configuration.
        pipeline: Underlying MambaTRMRLMPipeline.
    """

    def __init__(
        self,
        config: Optional[CodeRepairConfig] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the code repair pipeline.

        Args:
            config: Code repair configuration.
            config_path: Path to YAML configuration file.
        """
        # Load configuration
        if config_path:
            self.config = CodeRepairConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = CodeRepairConfig()

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO)
        )

        # Initialize underlying pipeline
        try:
            self.pipeline = MambaTRMRLMPipeline.from_pretrained(
                self.config.model_preset
            )
        except Exception as e:
            logger.warning(f"Could not load pretrained pipeline: {e}")
            logger.info("Using default pipeline configuration")
            self.pipeline = MambaTRMRLMPipeline(
                config=self.config.to_pipeline_config()
            )

        # Statistics
        self._stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "partial_repairs": 0,
            "failed_repairs": 0,
            "avg_confidence": [],
            "avg_time_ms": [],
        }

        logger.info(
            f"Initialized CodeRepairPipeline with preset: {self.config.model_preset}"
        )

    def repair(
        self,
        buggy_code: str,
        error_message: Optional[str] = None,
        context: Optional[str] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> RepairResult:
        """Repair buggy code.

        Args:
            buggy_code: The buggy code to repair.
            error_message: Optional error message providing hints.
            context: Optional surrounding context (file content, etc.).
            test_cases: Optional test cases for validation.

        Returns:
            RepairResult with repaired code and metadata.
        """
        start_time = time.time()
        self._stats["total_repairs"] += 1

        try:
            # Build context for pipeline
            pipeline_context = {}
            if context:
                pipeline_context["file_content"] = context
            if error_message:
                pipeline_context["error_message"] = error_message

            # Create pipeline request
            request = PipelineRequest(
                task_type="code_repair",
                input_data={
                    "buggy_code": buggy_code,
                    "error_message": error_message,
                },
                context=pipeline_context if pipeline_context else None,
            )

            # Run pipeline
            response = self.pipeline.process(request)

            # Process response
            repaired_code = response.output if response.output else buggy_code

            # Validate if configured
            validation_result = None
            if self.config.validate_output:
                validation_result = self._validate_code(
                    repaired_code,
                    self.config.language
                )

            # Run test cases if provided
            test_results = None
            if test_cases:
                test_results = self._run_tests(repaired_code, test_cases)

            # Determine status
            status = self._determine_status(
                response, validation_result, test_results
            )

            # Generate explanation if configured
            explanation = None
            if self.config.include_explanation:
                explanation = self._generate_explanation(
                    buggy_code, repaired_code, response
                )

            execution_time_ms = (time.time() - start_time) * 1000

            result = RepairResult(
                original_code=buggy_code,
                repaired_code=repaired_code,
                status=status,
                confidence=response.confidence,
                explanation=explanation,
                decomposition=response.decomposition,
                iterations=response.refinement_iterations,
                execution_time_ms=execution_time_ms,
                validation_result=validation_result,
                test_results=test_results,
                metadata={
                    "components_used": response.components_used,
                    "trace": response.trace,
                },
            )

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            logger.exception(f"Repair error: {e}")

            execution_time_ms = (time.time() - start_time) * 1000
            self._stats["failed_repairs"] += 1

            return RepairResult(
                original_code=buggy_code,
                repaired_code=buggy_code,
                status=RepairStatus.ERROR,
                confidence=0.0,
                execution_time_ms=execution_time_ms,
                metadata={"error": str(e)},
            )

    def repair_file(
        self,
        file_path: str,
        line_range: Optional[Tuple[int, int]] = None,
        write_back: bool = False,
    ) -> FileRepairResult:
        """Repair code in a file.

        Args:
            file_path: Path to the file to repair.
            line_range: Optional (start, end) line range to repair.
                       Lines are 1-indexed. If None, repairs entire file.
            write_back: Whether to write repaired code back to file.

        Returns:
            FileRepairResult with repair details.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(path, 'r') as f:
            original_content = f.read()

        lines = original_content.split('\n')

        # Extract target code
        if line_range:
            start, end = line_range
            # Convert to 0-indexed
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)
            target_lines = lines[start_idx:end_idx]
            buggy_code = '\n'.join(target_lines)
            # Use surrounding lines as context
            context_lines = lines[:start_idx] + lines[end_idx:]
            context = '\n'.join(context_lines)
        else:
            buggy_code = original_content
            context = None
            line_range = (1, len(lines))

        # Repair the code
        repair_result = self.repair(
            buggy_code=buggy_code,
            context=context,
        )

        # Reconstruct file content
        if line_range and repair_result.is_success:
            start_idx = max(0, line_range[0] - 1)
            end_idx = min(len(lines), line_range[1])
            repaired_lines = repair_result.repaired_code.split('\n')
            new_lines = lines[:start_idx] + repaired_lines + lines[end_idx:]
            repaired_content = '\n'.join(new_lines)
        else:
            repaired_content = repair_result.repaired_code

        # Write back if requested
        if write_back and repair_result.is_success:
            with open(path, 'w') as f:
                f.write(repaired_content)
            logger.info(f"Wrote repaired code to {file_path}")

        return FileRepairResult(
            file_path=str(path.absolute()),
            original_content=original_content,
            repaired_content=repaired_content,
            line_range=line_range,
            repair_result=repair_result,
            changes_made=original_content != repaired_content,
        )

    def repair_batch(
        self,
        items: List[RepairRequest],
        parallel: bool = False,
    ) -> List[RepairResult]:
        """Batch repair multiple code snippets.

        Args:
            items: List of RepairRequest objects.
            parallel: Whether to process in parallel (if available).

        Returns:
            List of RepairResult objects.
        """
        results = []

        for item in items:
            result = self.repair(
                buggy_code=item.code,
                error_message=item.error_message,
                context=item.context,
                test_cases=item.test_cases,
            )
            results.append(result)

        return results

    def _validate_code(
        self,
        code: str,
        language: str,
    ) -> Dict[str, Any]:
        """Validate code syntax.

        Args:
            code: Code to validate.
            language: Programming language.

        Returns:
            Validation result dict.
        """
        if language.lower() == "python":
            try:
                import ast
                ast.parse(code)
                return {"is_valid": True, "errors": []}
            except SyntaxError as e:
                return {
                    "is_valid": False,
                    "errors": [{
                        "message": str(e.msg),
                        "line": e.lineno,
                        "column": e.offset,
                    }],
                }

        # For other languages, assume valid (would need language-specific parsers)
        return {"is_valid": True, "errors": [], "note": "validation not available"}

    def _run_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run test cases against code.

        Args:
            code: Code to test.
            test_cases: List of test case dicts with "inputs" and "expected".

        Returns:
            List of test result dicts.
        """
        results = []

        for i, test in enumerate(test_cases):
            try:
                # Create execution namespace
                namespace = {}
                exec(code, namespace)

                # Get the function (assume first callable)
                func = None
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith('_'):
                        func = obj
                        break

                if func is None:
                    results.append({
                        "test_index": i,
                        "passed": False,
                        "error": "No function found",
                    })
                    continue

                # Run the test
                inputs = test.get("inputs", {})
                expected = test.get("expected")

                if isinstance(inputs, dict):
                    actual = func(**inputs)
                elif isinstance(inputs, list):
                    actual = func(*inputs)
                else:
                    actual = func(inputs)

                passed = actual == expected

                results.append({
                    "test_index": i,
                    "passed": passed,
                    "expected": expected,
                    "actual": actual,
                })

            except Exception as e:
                results.append({
                    "test_index": i,
                    "passed": False,
                    "error": str(e),
                })

        return results

    def _determine_status(
        self,
        response: PipelineResponse,
        validation_result: Optional[Dict[str, Any]],
        test_results: Optional[List[Dict[str, Any]]],
    ) -> RepairStatus:
        """Determine repair status from response and validation."""
        if not response.success:
            return RepairStatus.FAILED

        # Check validation
        is_valid = True
        if validation_result:
            is_valid = validation_result.get("is_valid", True)

        # Check tests
        tests_passed = True
        if test_results:
            tests_passed = all(t.get("passed", False) for t in test_results)

        if is_valid and tests_passed:
            return RepairStatus.SUCCESS
        elif is_valid or tests_passed:
            return RepairStatus.PARTIAL
        else:
            return RepairStatus.FAILED

    def _generate_explanation(
        self,
        original: str,
        repaired: str,
        response: PipelineResponse,
    ) -> str:
        """Generate explanation of changes."""
        if original == repaired:
            return "No changes were made."

        parts = ["Changes made:"]

        # Analyze differences (simple line-by-line)
        orig_lines = original.split('\n')
        new_lines = repaired.split('\n')

        changes = []
        for i, (orig, new) in enumerate(zip(orig_lines, new_lines)):
            if orig != new:
                changes.append(f"  Line {i + 1}: '{orig.strip()}' -> '{new.strip()}'")

        if len(new_lines) > len(orig_lines):
            for i in range(len(orig_lines), len(new_lines)):
                changes.append(f"  Line {i + 1}: Added '{new_lines[i].strip()}'")
        elif len(orig_lines) > len(new_lines):
            for i in range(len(new_lines), len(orig_lines)):
                changes.append(f"  Line {i + 1}: Removed '{orig_lines[i].strip()}'")

        if changes:
            parts.extend(changes[:5])  # Limit to first 5 changes
            if len(changes) > 5:
                parts.append(f"  ... and {len(changes) - 5} more changes")

        # Add decomposition info if available
        if response.decomposition:
            parts.append("\nRepair strategy:")
            for step in response.decomposition.subproblems[:3]:
                parts.append(f"  - {step}")

        return '\n'.join(parts)

    def _update_stats(self, result: RepairResult) -> None:
        """Update internal statistics."""
        if result.status == RepairStatus.SUCCESS:
            self._stats["successful_repairs"] += 1
        elif result.status == RepairStatus.PARTIAL:
            self._stats["partial_repairs"] += 1
        else:
            self._stats["failed_repairs"] += 1

        self._stats["avg_confidence"].append(result.confidence)
        self._stats["avg_time_ms"].append(result.execution_time_ms)

    def get_stats(self) -> Dict[str, Any]:
        """Get repair statistics."""
        stats = dict(self._stats)

        if stats["total_repairs"] > 0:
            stats["success_rate"] = (
                stats["successful_repairs"] / stats["total_repairs"]
            )
        else:
            stats["success_rate"] = 0.0

        if stats["avg_confidence"]:
            stats["mean_confidence"] = (
                sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
            )
        else:
            stats["mean_confidence"] = 0.0

        if stats["avg_time_ms"]:
            stats["mean_time_ms"] = (
                sum(stats["avg_time_ms"]) / len(stats["avg_time_ms"])
            )
        else:
            stats["mean_time_ms"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "partial_repairs": 0,
            "failed_repairs": 0,
            "avg_confidence": [],
            "avg_time_ms": [],
        }

    @classmethod
    def from_yaml(cls, path: str) -> "CodeRepairPipeline":
        """Load from YAML config.

        Example YAML:
            pipeline:
              model_preset: base
              max_context_length: 4096
              max_repair_attempts: 3
              use_mamba_context: true
              use_rlm_decomposition: true
              validate_output: true
              language: python

            mamba:
              d_model: 768
              n_layers: 24

            trm:
              max_iterations: 8

            rlm:
              max_generation_attempts: 3

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configured CodeRepairPipeline instance.
        """
        return cls(config_path=path)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "CodeRepairPipeline":
        """Load pretrained pipeline.

        Available presets:
        - "tiny": Minimal for testing
        - "small": Small for development
        - "base": Base for production
        - "large": Large for best quality

        Args:
            model_name: Name of pretrained configuration.

        Returns:
            Configured CodeRepairPipeline instance.
        """
        config = CodeRepairConfig(model_preset=model_name)
        return cls(config=config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def repair_code(
    buggy_code: str,
    error_message: Optional[str] = None,
    context: Optional[str] = None,
    model: str = "base",
) -> RepairResult:
    """Convenience function to repair code.

    Args:
        buggy_code: Code to repair.
        error_message: Optional error message.
        context: Optional context.
        model: Model preset to use.

    Returns:
        RepairResult with repaired code.
    """
    pipeline = CodeRepairPipeline.from_pretrained(model)
    return pipeline.repair(buggy_code, error_message, context)


def repair_file(
    file_path: str,
    line_range: Optional[Tuple[int, int]] = None,
    write_back: bool = False,
    model: str = "base",
) -> FileRepairResult:
    """Convenience function to repair a file.

    Args:
        file_path: Path to file.
        line_range: Optional line range.
        write_back: Whether to write back.
        model: Model preset to use.

    Returns:
        FileRepairResult with repair details.
    """
    pipeline = CodeRepairPipeline.from_pretrained(model)
    return pipeline.repair_file(file_path, line_range, write_back)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI entry point for code repair."""
    parser = argparse.ArgumentParser(
        description="Code Repair Pipeline - Fix buggy code using ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair from stdin
  echo "def foo(): retrun 1" | python -m pipelines.code_repair

  # Repair a file
  python -m pipelines.code_repair -i buggy.py -o fixed.py

  # Repair with error message
  python -m pipelines.code_repair -i buggy.py --error "SyntaxError: line 10"

  # Use specific model preset
  python -m pipelines.code_repair -i buggy.py -m large

  # Repair specific lines
  python -m pipelines.code_repair -i large_file.py --lines 10-20
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Input file or code string (use - for stdin)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "small", "base", "large"],
        help="Model preset (default: base)",
    )
    parser.add_argument(
        "--error", "-e",
        help="Error message to help with repair",
    )
    parser.add_argument(
        "--lines",
        help="Line range to repair (e.g., '10-20')",
    )
    parser.add_argument(
        "--context",
        help="Additional context file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--write-back", "-w",
        action="store_true",
        help="Write repaired code back to input file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after repair",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )

    # Load configuration
    if args.config:
        pipeline = CodeRepairPipeline.from_yaml(args.config)
    else:
        pipeline = CodeRepairPipeline.from_pretrained(args.model)

    # Read input
    if args.input == "-" or (args.input is None and not sys.stdin.isatty()):
        buggy_code = sys.stdin.read()
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(input_path, 'r') as f:
            buggy_code = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Read context if provided
    context = None
    if args.context:
        with open(args.context, 'r') as f:
            context = f.read()

    # Parse line range
    line_range = None
    if args.lines:
        try:
            start, end = args.lines.split('-')
            line_range = (int(start), int(end))
        except ValueError:
            print(f"Error: Invalid line range: {args.lines}", file=sys.stderr)
            print("Expected format: START-END (e.g., 10-20)", file=sys.stderr)
            sys.exit(1)

    # Perform repair
    try:
        if args.input and args.input != "-" and (line_range or args.write_back):
            # File-based repair
            result = pipeline.repair_file(
                args.input,
                line_range=line_range,
                write_back=args.write_back,
            )
            repair_result = result.repair_result
            output_code = result.repaired_content
        else:
            # Direct code repair
            repair_result = pipeline.repair(
                buggy_code=buggy_code,
                error_message=args.error,
                context=context,
            )
            output_code = repair_result.repaired_code

        # Output result
        if args.json:
            import json
            print(json.dumps(repair_result.to_dict(), indent=2))
        else:
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_code)
                if args.verbose:
                    print(f"Wrote repaired code to {args.output}")
            else:
                print(output_code)

            if args.verbose:
                print(f"\n--- Repair Status: {repair_result.status.value} ---")
                print(f"Confidence: {repair_result.confidence:.2f}")
                print(f"Time: {repair_result.execution_time_ms:.1f}ms")
                if repair_result.explanation:
                    print(f"\n{repair_result.explanation}")

        # Show stats if requested
        if args.stats:
            stats = pipeline.get_stats()
            print("\n--- Statistics ---")
            for key, value in stats.items():
                if not isinstance(value, list):
                    print(f"{key}: {value}")

        # Exit with appropriate code
        if repair_result.status == RepairStatus.SUCCESS:
            sys.exit(0)
        elif repair_result.status == RepairStatus.PARTIAL:
            sys.exit(2)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Configuration
    "CodeRepairConfig",
    "ModelPreset",
    # Data classes
    "RepairRequest",
    "RepairResult",
    "FileRepairResult",
    "RepairStatus",
    # Main pipeline
    "CodeRepairPipeline",
    # Convenience functions
    "repair_code",
    "repair_file",
    # CLI
    "main",
]


if __name__ == "__main__":
    main()
