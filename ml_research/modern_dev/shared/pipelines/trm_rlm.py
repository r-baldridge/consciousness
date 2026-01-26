"""
TRM + RLM Pipeline (TRM-005)

Full pipeline for code repair using TRM (Tiny Recursive Model) combined
with RLM (Recursive Language Model).

This module provides a high-level interface for:
- Single file repair
- Batch processing
- Streaming output
- Configurable retry logic
- Model loading and management

The pipeline orchestrates the composition of TRM and RLM, handling
tokenization, validation, and result formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Union,
    Iterator,
    Callable,
    Tuple,
    TypeVar,
)
from enum import Enum
from pathlib import Path
import json
import time
import logging
import os

import numpy as np

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

# Local imports
try:
    from ...trm.src.model import CodeRepairTRM, CodeRepairConfig
    from ...trm.src.composition import (
        TRMWithRLM,
        CompositionConfig,
        RepairResult,
        RepairStatus,
        RepairStrategy,
        TRMInput,
        TRMOutput,
        BatchTRMWithRLM,
    )
except ImportError:
    # Handle direct imports
    try:
        from consciousness.ml_research.modern_dev.trm.src.model import (
            CodeRepairTRM,
            CodeRepairConfig,
        )
        from consciousness.ml_research.modern_dev.trm.src.composition import (
            TRMWithRLM,
            CompositionConfig,
            RepairResult,
            RepairStatus,
            RepairStrategy,
            TRMInput,
            TRMOutput,
            BatchTRMWithRLM,
        )
    except ImportError:
        # Stubs for when imports fail
        CodeRepairTRM = None
        CodeRepairConfig = None
        TRMWithRLM = None
        CompositionConfig = None
        RepairResult = None
        RepairStatus = None

try:
    from ....ml_techniques.code_synthesis.rlm import RLMExtractor, RLMDecomposition
except ImportError:
    try:
        from consciousness.ml_research.ml_techniques.code_synthesis.rlm import (
            RLMExtractor,
            RLMDecomposition,
        )
    except ImportError:
        RLMExtractor = None
        RLMDecomposition = None


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


class StreamingUpdateType(str, Enum):
    """Type of streaming update."""
    START = "start"
    DECOMPOSITION = "decomposition"
    TRM_ITERATION = "trm_iteration"
    FEEDBACK = "feedback"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamingUpdate:
    """
    Streaming update during pipeline processing.

    Provides real-time progress updates during repair operations.

    Attributes:
        update_type: Type of update
        progress: Progress percentage (0-100)
        message: Human-readable status message
        data: Optional structured data
        timestamp_ms: Timestamp in milliseconds
    """
    update_type: StreamingUpdateType
    progress: float
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.update_type.value,
            "progress": self.progress,
            "message": self.message,
            "data": self.data,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class PipelineResult:
    """
    Result from pipeline processing.

    Wraps RepairResult with additional pipeline-specific metadata.

    Attributes:
        repair_result: The underlying RepairResult
        input_file: Optional input file path
        output_file: Optional output file path
        pipeline_time_ms: Total pipeline processing time
        retry_count: Number of retries performed
        streaming_updates: List of streaming updates
        metadata: Additional pipeline metadata
    """
    repair_result: Optional[RepairResult]
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    pipeline_time_ms: float = 0.0
    retry_count: int = 0
    streaming_updates: List[StreamingUpdate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if repair was successful."""
        if self.repair_result is None:
            return False
        return self.repair_result.status == RepairStatus.SUCCESS

    @property
    def repaired_code(self) -> Optional[str]:
        """Get repaired code if available."""
        if self.repair_result:
            return self.repair_result.repaired_code
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repair_result": self.repair_result.to_dict() if self.repair_result else None,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "pipeline_time_ms": self.pipeline_time_ms,
            "retry_count": self.retry_count,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class PipelineConfig:
    """
    Configuration for the TRM+RLM pipeline.

    Attributes:
        model_path: Path to TRM model checkpoint
        tokenizer_path: Path to tokenizer file
        config_path: Optional path to configuration JSON
        composition_config: TRM+RLM composition configuration
        max_retries: Maximum retries on failure
        retry_delay_ms: Delay between retries
        batch_size: Batch size for batch processing
        enable_streaming: Enable streaming updates
        cache_models: Cache loaded models in memory
        validate_syntax: Validate Python syntax
        validate_types: Run type checking on repairs
        output_format: Output format (code, json, diff)
        device: Compute device (cpu, cuda, mps)
    """
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None
    composition_config: Optional[CompositionConfig] = None

    # Retry configuration
    max_retries: int = 2
    retry_delay_ms: float = 100.0

    # Processing configuration
    batch_size: int = 8
    enable_streaming: bool = True
    cache_models: bool = True

    # Validation
    validate_syntax: bool = True
    validate_types: bool = False

    # Output
    output_format: str = "code"  # code, json, diff

    # Hardware
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "config_path": self.config_path,
            "composition_config": self.composition_config.to_dict() if self.composition_config else None,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "batch_size": self.batch_size,
            "enable_streaming": self.enable_streaming,
            "cache_models": self.cache_models,
            "validate_syntax": self.validate_syntax,
            "validate_types": self.validate_types,
            "output_format": self.output_format,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        composition_data = data.pop("composition_config", None)
        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if composition_data:
            config.composition_config = CompositionConfig.from_dict(composition_data)
        return config

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================


class TRMRLMPipeline:
    """
    Full pipeline for code repair using TRM + RLM.

    Provides a complete solution for code repair with:
    - Model loading and management
    - Single file and batch processing
    - Streaming output for real-time updates
    - Configurable retry logic
    - Validation integration

    Example:
        >>> # Initialize pipeline
        >>> pipeline = TRMRLMPipeline(config_path="config.json")
        >>>
        >>> # Single repair
        >>> result = pipeline.process(buggy_code)
        >>> print(result.repaired_code)
        >>>
        >>> # Batch repair
        >>> results = pipeline.process_batch(buggy_codes)
        >>>
        >>> # Streaming
        >>> for update in pipeline.process_streaming(buggy_code):
        ...     print(f"{update.progress}%: {update.message}")

    Attributes:
        config: Pipeline configuration
        trm_model: Loaded TRM model
        rlm_pipeline: RLM extractor instance
        tokenizer: Code tokenizer
        composer: TRMWithRLM composition instance
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[str] = None,
        trm_model: Optional[Any] = None,
        rlm_pipeline: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the TRM+RLM pipeline.

        Args:
            config: Pipeline configuration object
            config_path: Path to configuration JSON file
            trm_model: Pre-loaded TRM model (optional)
            rlm_pipeline: Pre-loaded RLM pipeline (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = PipelineConfig.from_file(config_path)
        else:
            self.config = PipelineConfig()

        # Initialize components
        self._trm_model = trm_model
        self._rlm_pipeline = rlm_pipeline
        self._tokenizer = tokenizer
        self._composer: Optional[TRMWithRLM] = None
        self._batch_composer: Optional[BatchTRMWithRLM] = None

        # Model cache
        self._model_cache: Dict[str, Any] = {}

        # Statistics
        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time_ms": 0.0,
            "retries": 0,
        }

        logger.info(f"TRMRLMPipeline initialized with device={self.config.device}")

    @property
    def trm_model(self) -> Optional[Any]:
        """Get TRM model, loading if necessary."""
        if self._trm_model is None and self.config.model_path:
            self._trm_model = self._load_trm_model()
        return self._trm_model

    @property
    def rlm_pipeline(self) -> Optional[Any]:
        """Get RLM pipeline, creating if necessary."""
        if self._rlm_pipeline is None:
            self._rlm_pipeline = self._create_rlm_pipeline()
        return self._rlm_pipeline

    @property
    def tokenizer(self) -> Optional[Any]:
        """Get tokenizer, loading if necessary."""
        if self._tokenizer is None and self.config.tokenizer_path:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    @property
    def composer(self) -> Optional[TRMWithRLM]:
        """Get composition instance, creating if necessary."""
        if self._composer is None:
            self._composer = self._create_composer()
        return self._composer

    def _load_trm_model(self) -> Optional[Any]:
        """Load TRM model from checkpoint."""
        if not self.config.model_path:
            logger.warning("No model_path configured, TRM model not loaded")
            return None

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot load TRM model")
            return None

        if CodeRepairTRM is None:
            logger.error("CodeRepairTRM not available")
            return None

        path = Path(self.config.model_path)
        if not path.exists():
            logger.error(f"Model path does not exist: {path}")
            return None

        # Check cache
        cache_key = str(path.absolute())
        if self.config.cache_models and cache_key in self._model_cache:
            logger.debug(f"Using cached TRM model: {cache_key}")
            return self._model_cache[cache_key]

        logger.info(f"Loading TRM model from {path}")
        try:
            model = CodeRepairTRM.from_pretrained(str(path))

            # Move to device
            device = torch.device(self.config.device)
            model = model.to(device)
            model.eval()

            if self.config.cache_models:
                self._model_cache[cache_key] = model

            return model

        except Exception as e:
            logger.exception(f"Failed to load TRM model: {e}")
            return None

    def _load_tokenizer(self) -> Optional[Any]:
        """Load tokenizer from file."""
        if not self.config.tokenizer_path:
            logger.warning("No tokenizer_path configured")
            return None

        if not TOKENIZERS_AVAILABLE:
            logger.warning("tokenizers library not available")
            return None

        path = Path(self.config.tokenizer_path)
        if not path.exists():
            logger.error(f"Tokenizer path does not exist: {path}")
            return None

        logger.info(f"Loading tokenizer from {path}")
        try:
            return Tokenizer.from_file(str(path))
        except Exception as e:
            logger.exception(f"Failed to load tokenizer: {e}")
            return None

    def _create_rlm_pipeline(self) -> Optional[Any]:
        """Create RLM pipeline instance."""
        if RLMExtractor is None:
            logger.warning("RLMExtractor not available")
            return None

        return RLMExtractor(
            use_llm=False,  # Use pattern-based extraction
            max_variables=20,
            infer_intermediates=True,
        )

    def _create_composer(self) -> Optional[TRMWithRLM]:
        """Create TRMWithRLM composition instance."""
        if TRMWithRLM is None:
            logger.error("TRMWithRLM not available")
            return None

        composition_config = self.config.composition_config or CompositionConfig()

        # Create validator if configured
        validator = None
        if self.config.validate_syntax:
            validator = self._create_validator()

        return TRMWithRLM(
            trm_model=self.trm_model,
            rlm_pipeline=self.rlm_pipeline,
            config=composition_config,
            tokenizer=self.tokenizer,
            validator=validator,
        )

    def _create_validator(self) -> Callable[[str], Tuple[bool, str]]:
        """Create code validator function."""
        def validate(code: str) -> Tuple[bool, str]:
            import ast
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        return validate

    def process(
        self,
        input_data: Union[str, Path],
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process a single code repair.

        Args:
            input_data: Buggy code string or path to file
            error_message: Optional error message
            context: Optional additional context

        Returns:
            PipelineResult with repair outcome
        """
        start_time = time.time()
        streaming_updates: List[StreamingUpdate] = []
        retry_count = 0
        input_file = None

        # Handle file input
        if isinstance(input_data, Path) or (isinstance(input_data, str) and os.path.isfile(input_data)):
            input_file = str(input_data)
            with open(input_data, 'r') as f:
                buggy_code = f.read()
        else:
            buggy_code = input_data

        # Add start update
        if self.config.enable_streaming:
            streaming_updates.append(StreamingUpdate(
                update_type=StreamingUpdateType.START,
                progress=0.0,
                message="Starting code repair",
                data={"code_length": len(buggy_code)},
            ))

        # Process with retries
        last_result: Optional[RepairResult] = None
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if self.composer is None:
                    raise RuntimeError("Composer not initialized")

                # Run repair
                result = self.composer.repair(
                    buggy_code,
                    error_message=error_message,
                    context=context,
                )

                last_result = result

                if result.status == RepairStatus.SUCCESS:
                    break

                # Retry on partial success
                if result.status != RepairStatus.PARTIAL:
                    break

                retry_count += 1
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_ms / 1000)
                    self._stats["retries"] += 1

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                retry_count += 1
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_ms / 1000)

        # Update statistics
        self._stats["total_processed"] += 1
        if last_result and last_result.status == RepairStatus.SUCCESS:
            self._stats["successful"] += 1
        else:
            self._stats["failed"] += 1

        elapsed_ms = (time.time() - start_time) * 1000
        self._stats["total_time_ms"] += elapsed_ms

        # Add completion update
        if self.config.enable_streaming:
            streaming_updates.append(StreamingUpdate(
                update_type=StreamingUpdateType.COMPLETE,
                progress=100.0,
                message="Repair complete" if last_result else "Repair failed",
                data={
                    "success": last_result.status == RepairStatus.SUCCESS if last_result else False,
                    "retry_count": retry_count,
                },
            ))

        return PipelineResult(
            repair_result=last_result,
            input_file=input_file,
            pipeline_time_ms=elapsed_ms,
            retry_count=retry_count,
            streaming_updates=streaming_updates,
            metadata={
                "error": last_error,
                "attempts": retry_count + 1,
            },
        )

    def process_batch(
        self,
        input_data: Union[List[str], List[Path]],
        error_messages: Optional[List[Optional[str]]] = None,
        progress_callback: Optional[Callable[[int, int, PipelineResult], None]] = None,
    ) -> List[PipelineResult]:
        """
        Process a batch of code repairs.

        Args:
            input_data: List of buggy codes or file paths
            error_messages: Optional list of error messages
            progress_callback: Optional callback(completed, total, result)

        Returns:
            List of PipelineResult objects
        """
        if error_messages is None:
            error_messages = [None] * len(input_data)

        results = []
        total = len(input_data)

        for i, (code, error) in enumerate(zip(input_data, error_messages)):
            result = self.process(code, error)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results

    def process_streaming(
        self,
        input_data: Union[str, Path],
        error_message: Optional[str] = None,
    ) -> Iterator[StreamingUpdate]:
        """
        Process with streaming updates.

        Yields real-time progress updates during repair.

        Args:
            input_data: Buggy code string or file path
            error_message: Optional error message

        Yields:
            StreamingUpdate objects with progress information
        """
        start_time = time.time()

        # Handle file input
        if isinstance(input_data, Path) or (isinstance(input_data, str) and os.path.isfile(input_data)):
            with open(input_data, 'r') as f:
                buggy_code = f.read()
        else:
            buggy_code = input_data

        # Start
        yield StreamingUpdate(
            update_type=StreamingUpdateType.START,
            progress=0.0,
            message="Starting code repair",
            data={"code_length": len(buggy_code)},
        )

        # Check components
        if self.composer is None:
            yield StreamingUpdate(
                update_type=StreamingUpdateType.ERROR,
                progress=0.0,
                message="Pipeline not properly initialized",
            )
            return

        # Decomposition phase
        yield StreamingUpdate(
            update_type=StreamingUpdateType.DECOMPOSITION,
            progress=10.0,
            message="Analyzing code with RLM",
        )

        try:
            # Run repair (we can't easily intercept TRM iterations from here,
            # so we'll provide a simplified streaming view)
            result = self.composer.repair(buggy_code, error_message)

            # Report TRM iterations
            if result.trm_output:
                for i in range(result.trm_output.iterations_used):
                    progress = 20.0 + (i / result.trm_output.iterations_used) * 60.0
                    yield StreamingUpdate(
                        update_type=StreamingUpdateType.TRM_ITERATION,
                        progress=progress,
                        message=f"TRM iteration {i + 1}/{result.trm_output.iterations_used}",
                        data={"iteration": i + 1},
                    )

            # Feedback rounds
            for i in range(result.feedback_rounds):
                yield StreamingUpdate(
                    update_type=StreamingUpdateType.FEEDBACK,
                    progress=80.0 + (i / max(result.feedback_rounds, 1)) * 10.0,
                    message=f"Feedback round {i + 1}",
                )

            # Validation
            if result.validation_result:
                yield StreamingUpdate(
                    update_type=StreamingUpdateType.VALIDATION,
                    progress=95.0,
                    message="Validating repair",
                    data=result.validation_result,
                )

            # Complete
            yield StreamingUpdate(
                update_type=StreamingUpdateType.COMPLETE,
                progress=100.0,
                message="Repair complete",
                data={
                    "success": result.status == RepairStatus.SUCCESS,
                    "confidence": result.confidence,
                    "time_ms": (time.time() - start_time) * 1000,
                },
            )

        except Exception as e:
            yield StreamingUpdate(
                update_type=StreamingUpdateType.ERROR,
                progress=0.0,
                message=f"Error: {str(e)}",
            )

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """
        Process a file and optionally write the result.

        Args:
            input_path: Path to input file
            output_path: Optional path to write repaired code

        Returns:
            PipelineResult with repair outcome
        """
        result = self.process(input_path)

        if output_path and result.success and result.repaired_code:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(result.repaired_code)

            result.output_file = str(output_path)

        return result

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.py",
        recursive: bool = True,
    ) -> Dict[str, PipelineResult]:
        """
        Process all matching files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Optional output directory
            pattern: File pattern to match
            recursive: Whether to search recursively

        Returns:
            Dictionary mapping file paths to results
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)

        if recursive:
            files = list(input_dir.rglob(pattern))
        else:
            files = list(input_dir.glob(pattern))

        results = {}
        for input_file in files:
            # Compute output path
            if output_dir:
                relative = input_file.relative_to(input_dir)
                output_file = output_dir / relative
            else:
                output_file = None

            result = self.process_file(input_file, output_file)
            results[str(input_file)] = result

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = dict(self._stats)

        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_processed"]
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_time_ms"] = 0.0

        # Include composer stats if available
        if self._composer:
            stats["composer_stats"] = self._composer.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time_ms": 0.0,
            "retries": 0,
        }
        if self._composer:
            self._composer.reset_stats()

    def clear_cache(self) -> None:
        """Clear model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def __enter__(self) -> "TRMRLMPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if not self.config.cache_models:
            self.clear_cache()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_pipeline(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs,
) -> TRMRLMPipeline:
    """
    Create a TRM+RLM pipeline with the given configuration.

    Convenience function for quick pipeline creation.

    Args:
        model_path: Path to TRM model checkpoint
        tokenizer_path: Path to tokenizer file
        config_path: Path to configuration JSON
        **kwargs: Additional configuration options

    Returns:
        Configured TRMRLMPipeline instance
    """
    if config_path:
        config = PipelineConfig.from_file(config_path)
    else:
        config = PipelineConfig(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            **kwargs,
        )

    return TRMRLMPipeline(config=config)


def repair_code(
    buggy_code: str,
    error_message: Optional[str] = None,
    pipeline: Optional[TRMRLMPipeline] = None,
    **kwargs,
) -> Optional[str]:
    """
    Simple function to repair buggy code.

    Creates a pipeline if not provided.

    Args:
        buggy_code: The buggy code to repair
        error_message: Optional error message
        pipeline: Optional pre-created pipeline
        **kwargs: Pipeline configuration options

    Returns:
        Repaired code string, or None if repair failed
    """
    if pipeline is None:
        pipeline = create_pipeline(**kwargs)

    result = pipeline.process(buggy_code, error_message)
    return result.repaired_code if result.success else None


def repair_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    pipeline: Optional[TRMRLMPipeline] = None,
    **kwargs,
) -> PipelineResult:
    """
    Repair a code file.

    Args:
        input_path: Path to buggy code file
        output_path: Optional path to write repaired code
        pipeline: Optional pre-created pipeline
        **kwargs: Pipeline configuration options

    Returns:
        PipelineResult with repair outcome
    """
    if pipeline is None:
        pipeline = create_pipeline(**kwargs)

    return pipeline.process_file(input_path, output_path)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "StreamingUpdateType",
    # Data classes
    "StreamingUpdate",
    "PipelineResult",
    "PipelineConfig",
    # Main class
    "TRMRLMPipeline",
    # Convenience functions
    "create_pipeline",
    "repair_code",
    "repair_file",
]
