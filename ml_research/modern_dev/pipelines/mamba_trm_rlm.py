"""
Mamba-TRM-RLM Pipeline (INTEG-002)

Full Mamba-TRM-RLM pipeline for code understanding and repair.

Components:
- Mamba: Long-context understanding with O(N) complexity for encoding
  surrounding context (file, imports, repository structure)
- RLM: Task decomposition and code synthesis with variable extraction
  and constraint satisfaction
- TRM: Recursive refinement with 8 iterations achieving 42 effective depth
  for iterative code repair

The pipeline flow:
1. Mamba encodes long context (full file, imports, related files)
2. RLM decomposes the repair task into sub-problems
3. TRM iteratively refines the solution
4. Orchestrator coordinates fallback and retry logic

Reference:
    - TRM: Tiny Recursive Model for code repair
    - RLM: Recursive Language Model for decomposition
    - Mamba: Linear-Time Sequence Modeling

Example:
    from modern_dev.pipelines.mamba_trm_rlm import MambaTRMRLMPipeline

    pipeline = MambaTRMRLMPipeline()
    response = pipeline.process(PipelineRequest(
        task_type="code_repair",
        input_data={"buggy_code": "def foo(): retrun 1"},
        context={"file_content": "...full file..."}
    ))
    print(response.output)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# Optional imports with availability flags
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


@dataclass
class MambaConfig:
    """Configuration for Mamba component.

    Mamba handles long-context understanding with O(N) complexity,
    making it suitable for encoding entire files and related context.

    Attributes:
        d_model: Model dimension (embedding size). Default: 768.
        n_layers: Number of Mamba blocks. Default: 24.
        d_state: SSM state dimension. Default: 16.
        d_conv: Convolution kernel width. Default: 4.
        expand: Expansion factor for inner dimension. Default: 2.
        max_context_length: Maximum context length in tokens. Default: 100000.
        use_cache: Whether to use inference caching. Default: True.
        dtype: Data type for model. Default: "float32".
    """
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_context_length: int = 100000
    use_cache: bool = True
    dtype: str = "float32"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "max_context_length": self.max_context_length,
            "use_cache": self.use_cache,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MambaConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model) component.

    TRM performs iterative refinement through recursive processing,
    achieving high effective depth with limited parameters.

    Attributes:
        grid_height: Height of the token grid (lines). Default: 64.
        grid_width: Width of the token grid (columns). Default: 48.
        d_model: Model dimension. Default: 256.
        n_heads: Number of attention heads. Default: 8.
        max_iterations: Maximum refinement iterations. Default: 8.
        early_stop_threshold: Confidence threshold for early stopping. Default: 0.95.
        use_halting: Whether to use adaptive halting. Default: True.
    """
    grid_height: int = 64
    grid_width: int = 48
    d_model: int = 256
    n_heads: int = 8
    max_iterations: int = 8
    early_stop_threshold: float = 0.95
    use_halting: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "max_iterations": self.max_iterations,
            "early_stop_threshold": self.early_stop_threshold,
            "use_halting": self.use_halting,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TRMConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RLMConfig:
    """Configuration for RLM (Recursive Language Model) component.

    RLM decomposes tasks into sub-problems and handles code synthesis
    with variable extraction and constraint satisfaction.

    Attributes:
        max_generation_attempts: Maximum code generation attempts. Default: 3.
        max_debug_attempts: Maximum debugging iterations. Default: 5.
        use_templates: Whether to use code templates. Default: True.
        enable_type_inference: Whether to infer types. Default: True.
        variable_extraction_mode: Mode for variable extraction. Default: "hybrid".
    """
    max_generation_attempts: int = 3
    max_debug_attempts: int = 5
    use_templates: bool = True
    enable_type_inference: bool = True
    variable_extraction_mode: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_generation_attempts": self.max_generation_attempts,
            "max_debug_attempts": self.max_debug_attempts,
            "use_templates": self.use_templates,
            "enable_type_inference": self.enable_type_inference,
            "variable_extraction_mode": self.variable_extraction_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLMConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    Combines configuration for all components (Mamba, TRM, RLM)
    with pipeline-level settings.

    Attributes:
        mamba: Mamba component configuration.
        trm: TRM component configuration.
        rlm: RLM component configuration.
        device: Device to run on ("auto", "cpu", "cuda", "mps"). Default: "auto".
        max_context_length: Maximum context length. Default: 4096.
        timeout_ms: Maximum processing time in milliseconds. Default: 30000.
        enable_fallback: Whether to enable fallback strategies. Default: True.
        log_level: Logging level. Default: "INFO".
    """
    mamba: MambaConfig = field(default_factory=MambaConfig)
    trm: TRMConfig = field(default_factory=TRMConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    device: str = "auto"
    max_context_length: int = 4096
    timeout_ms: float = 30000.0
    enable_fallback: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Ensure nested configs are proper types."""
        if isinstance(self.mamba, dict):
            self.mamba = MambaConfig.from_dict(self.mamba)
        if isinstance(self.trm, dict):
            self.trm = TRMConfig.from_dict(self.trm)
        if isinstance(self.rlm, dict):
            self.rlm = RLMConfig.from_dict(self.rlm)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mamba": self.mamba.to_dict(),
            "trm": self.trm.to_dict(),
            "rlm": self.rlm.to_dict(),
            "device": self.device,
            "max_context_length": self.max_context_length,
            "timeout_ms": self.timeout_ms,
            "enable_fallback": self.enable_fallback,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        mamba_config = data.get("mamba", {})
        trm_config = data.get("trm", {})
        rlm_config = data.get("rlm", {})

        return cls(
            mamba=MambaConfig.from_dict(mamba_config) if mamba_config else MambaConfig(),
            trm=TRMConfig.from_dict(trm_config) if trm_config else TRMConfig(),
            rlm=RLMConfig.from_dict(rlm_config) if rlm_config else RLMConfig(),
            device=data.get("device", "auto"),
            max_context_length=data.get("max_context_length", 4096),
            timeout_ms=data.get("timeout_ms", 30000.0),
            enable_fallback=data.get("enable_fallback", True),
            log_level=data.get("log_level", "INFO"),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle both flat and nested YAML structures
        if "pipeline" in data:
            pipeline_data = data["pipeline"]
            pipeline_data["mamba"] = data.get("mamba", {})
            pipeline_data["trm"] = data.get("trm", {})
            pipeline_data["rlm"] = data.get("rlm", {})
            return cls.from_dict(pipeline_data)

        return cls.from_dict(data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# =============================================================================
# DATA CLASSES
# =============================================================================


class TaskType(Enum):
    """Types of tasks the pipeline can handle."""
    CODE_REPAIR = "code_repair"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    REFACTORING = "refactoring"


@dataclass
class TaskDecomposition:
    """Decomposition of a task into sub-problems.

    Created by RLM to guide the repair process.

    Attributes:
        task_type: Type of the overall task.
        subproblems: List of identified sub-problems.
        variables: Extracted variables and their types.
        constraints: Identified constraints.
        dependencies: Dependencies between sub-problems.
        reasoning_trace: Step-by-step reasoning.
        confidence: Confidence in this decomposition.
    """
    task_type: TaskType
    subproblems: List[str]
    variables: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    dependencies: List[str]
    reasoning_trace: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "subproblems": self.subproblems,
            "variables": self.variables,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class PipelineRequest:
    """Request to the Mamba-TRM-RLM pipeline.

    Attributes:
        task_type: Type of task to perform.
        input_data: Input data (code, specification, etc.).
        context: Additional context (file content, imports, etc.).
        constraints: Resource/quality constraints.
        options: Additional options.
    """
    task_type: str
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "input_data": self.input_data,
            "context": self.context,
            "constraints": self.constraints,
            "options": self.options,
        }


@dataclass
class PipelineResponse:
    """Response from the Mamba-TRM-RLM pipeline.

    Attributes:
        success: Whether the pipeline succeeded.
        output: Output from the pipeline.
        decomposition: Task decomposition from RLM.
        context_embedding: Context embedding from Mamba (optional).
        refinement_iterations: Number of TRM refinement iterations.
        confidence: Overall confidence in the result.
        execution_time_ms: Total execution time.
        components_used: List of components used.
        trace: Execution trace for debugging.
        metadata: Additional response metadata.
    """
    success: bool
    output: Any
    decomposition: Optional[TaskDecomposition] = None
    context_embedding: Optional[Any] = None  # torch.Tensor when available
    refinement_iterations: int = 0
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    components_used: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "decomposition": self.decomposition.to_dict() if self.decomposition else None,
            "refinement_iterations": self.refinement_iterations,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "components_used": self.components_used,
            "trace": self.trace,
            "metadata": self.metadata,
        }


# =============================================================================
# COMPONENT LOADERS
# =============================================================================


def load_mamba(config: Optional[MambaConfig] = None, device: str = "cpu") -> Any:
    """Load Mamba model.

    Args:
        config: Mamba configuration.
        device: Device to load model on.

    Returns:
        Loaded Mamba model or None if unavailable.
    """
    config = config or MambaConfig()

    try:
        from ..mamba_impl.src.mamba_model import MambaLM, MambaLMConfig

        mamba_config = MambaLMConfig(
            vocab_size=50257,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

        model = MambaLM(mamba_config)

        if TORCH_AVAILABLE:
            model = model.to(device)
            model.eval()

        logger.info(f"Loaded Mamba model with {model.num_parameters():,} parameters")
        return model

    except ImportError as e:
        logger.warning(f"Could not load Mamba model: {e}")
        return None


def load_trm(config: Optional[TRMConfig] = None, device: str = "cpu") -> Any:
    """Load TRM (Tiny Recursive Model).

    Args:
        config: TRM configuration.
        device: Device to load model on.

    Returns:
        Loaded TRM model or None if unavailable.
    """
    config = config or TRMConfig()

    try:
        from ..trm.src.model import TRM, TRMConfig as TRMModelConfig

        trm_config = TRMModelConfig(
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_iterations=config.max_iterations,
        )

        model = TRM(trm_config)

        if TORCH_AVAILABLE:
            model = model.to(device)
            model.eval()

        logger.info(f"Loaded TRM model with grid {config.grid_height}x{config.grid_width}")
        return model

    except ImportError as e:
        logger.warning(f"Could not load TRM model: {e}")
        return None


def load_rlm(config: Optional[RLMConfig] = None) -> Any:
    """Load RLM (Recursive Language Model) pipeline.

    Args:
        config: RLM configuration.

    Returns:
        Loaded RLM pipeline or None if unavailable.
    """
    config = config or RLMConfig()

    try:
        from ...ml_techniques.code_synthesis.pipeline import RLMPipeline, PipelineConfig

        rlm_config = PipelineConfig(
            max_generation_attempts=config.max_generation_attempts,
            max_debug_attempts=config.max_debug_attempts,
            enable_type_inference=config.enable_type_inference,
            variable_extraction_mode=config.variable_extraction_mode,
        )

        pipeline = RLMPipeline(config=rlm_config)
        logger.info("Loaded RLM pipeline")
        return pipeline

    except ImportError as e:
        logger.warning(f"Could not load RLM pipeline: {e}")
        return None


def _get_device(device: str = "auto") -> str:
    """Determine the device to use.

    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps").

    Returns:
        Resolved device string.
    """
    if device != "auto":
        return device

    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# =============================================================================
# MAIN PIPELINE
# =============================================================================


class MambaTRMRLMPipeline:
    """
    Full Mamba-TRM-RLM pipeline for code understanding and repair.

    Components:
    - Mamba: Long-context understanding (O(N) complexity)
    - RLM: Task decomposition and code synthesis
    - TRM: Recursive refinement (8 iterations, 42 effective depth)

    The pipeline coordinates these components through an orchestrator
    that handles routing, fallback, and error recovery.

    Example:
        >>> pipeline = MambaTRMRLMPipeline()
        >>>
        >>> # Simple code repair
        >>> response = pipeline.process(PipelineRequest(
        ...     task_type="code_repair",
        ...     input_data={"buggy_code": "def foo(): retrun 1"}
        ... ))
        >>>
        >>> # With full context
        >>> response = pipeline.process(PipelineRequest(
        ...     task_type="code_repair",
        ...     input_data={"buggy_code": "..."},
        ...     context={"file_content": "...full file..."}
        ... ))

    Attributes:
        config: Pipeline configuration.
        mamba: Mamba model for context encoding.
        trm: TRM model for iterative refinement.
        rlm: RLM pipeline for task decomposition.
        orchestrator: Orchestrator for coordination.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the Mamba-TRM-RLM pipeline.

        Args:
            config: Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO)
        )

        # Determine device
        self.device = _get_device(self.config.device)
        logger.info(f"Using device: {self.device}")

        # Load components
        self.mamba = self._load_mamba()
        self.trm = self._load_trm()
        self.rlm = self._load_rlm()
        self.orchestrator = self._load_orchestrator()

        # Track statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "mamba_calls": 0,
            "trm_calls": 0,
            "rlm_calls": 0,
            "avg_latency_ms": [],
        }

        logger.info(
            f"Initialized MambaTRMRLMPipeline with components: "
            f"mamba={'loaded' if self.mamba else 'unavailable'}, "
            f"trm={'loaded' if self.trm else 'unavailable'}, "
            f"rlm={'loaded' if self.rlm else 'unavailable'}"
        )

    def _load_mamba(self) -> Any:
        """Load Mamba model."""
        return load_mamba(self.config.mamba, self.device)

    def _load_trm(self) -> Any:
        """Load TRM model."""
        return load_trm(self.config.trm, self.device)

    def _load_rlm(self) -> Any:
        """Load RLM pipeline."""
        return load_rlm(self.config.rlm)

    def _load_orchestrator(self) -> Any:
        """Load orchestrator for coordination."""
        try:
            from ..orchestrator.router import MLOrchestrator
            return MLOrchestrator()
        except ImportError:
            logger.warning("Could not load orchestrator, using direct routing")
            return None

    def process(self, request: PipelineRequest) -> PipelineResponse:
        """Process a request through the full pipeline.

        Flow:
        1. Encode context with Mamba (if context provided)
        2. Decompose task with RLM
        3. Refine output with TRM
        4. Return final result

        Args:
            request: Pipeline request with task and input data.

        Returns:
            PipelineResponse with result and metadata.
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        trace = []
        components_used = []

        try:
            # Step 1: Encode context with Mamba
            context_embedding = None
            if request.context and self.mamba is not None:
                trace.append({"stage": "mamba_encode", "start_ms": self._elapsed_ms(start_time)})
                context_embedding = self.encode_context(
                    request.context.get("file_content", "")
                )
                components_used.append("mamba")
                self._stats["mamba_calls"] += 1
                trace[-1]["end_ms"] = self._elapsed_ms(start_time)

            # Step 2: Decompose task with RLM
            decomposition = None
            if self.rlm is not None:
                trace.append({"stage": "rlm_decompose", "start_ms": self._elapsed_ms(start_time)})
                decomposition = self.decompose_task(
                    request.input_data.get("buggy_code", ""),
                    context_embedding
                )
                components_used.append("rlm")
                self._stats["rlm_calls"] += 1
                trace[-1]["end_ms"] = self._elapsed_ms(start_time)

            # Step 3: Refine with TRM
            output = request.input_data.get("buggy_code", "")
            refinement_iterations = 0
            confidence = 0.0

            if self.trm is not None:
                trace.append({"stage": "trm_refine", "start_ms": self._elapsed_ms(start_time)})
                refined_output = self.refine_output(output, context_embedding)
                if refined_output:
                    output = refined_output
                    refinement_iterations = self.config.trm.max_iterations
                components_used.append("trm")
                self._stats["trm_calls"] += 1
                trace[-1]["end_ms"] = self._elapsed_ms(start_time)
                confidence = 0.7  # Placeholder confidence

            # Compute final metrics
            execution_time_ms = self._elapsed_ms(start_time)
            self._stats["avg_latency_ms"].append(execution_time_ms)
            self._stats["successful_requests"] += 1

            return PipelineResponse(
                success=True,
                output=output,
                decomposition=decomposition,
                context_embedding=context_embedding,
                refinement_iterations=refinement_iterations,
                confidence=confidence,
                execution_time_ms=execution_time_ms,
                components_used=components_used,
                trace=trace,
                metadata={
                    "device": self.device,
                    "request": request.to_dict(),
                },
            )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")

            return PipelineResponse(
                success=False,
                output=None,
                execution_time_ms=self._elapsed_ms(start_time),
                components_used=components_used,
                trace=trace,
                metadata={"error": str(e)},
            )

    def encode_context(self, context: str) -> Optional[Any]:
        """Use Mamba to encode long context.

        Mamba's O(N) complexity makes it suitable for encoding
        entire files, import graphs, and related context.

        Args:
            context: Context string to encode (file content, imports, etc.).

        Returns:
            Context embedding tensor, or None if Mamba unavailable.
        """
        if self.mamba is None:
            return None

        if not TORCH_AVAILABLE:
            return None

        try:
            # Tokenize context (simple character-level for now)
            # In production, use a proper tokenizer
            max_len = min(len(context), self.config.max_context_length)
            tokens = [ord(c) % 256 for c in context[:max_len]]

            # Pad to minimum length
            if len(tokens) < 16:
                tokens.extend([0] * (16 - len(tokens)))

            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Get embeddings from Mamba
            with torch.no_grad():
                if hasattr(self.mamba, 'embedding'):
                    embedding = self.mamba.embedding(input_ids)
                    # Average pool over sequence
                    context_embedding = embedding.mean(dim=1)
                    return context_embedding
                else:
                    # Forward pass and extract hidden states
                    output = self.mamba(input_ids, return_hidden_states=True)
                    if isinstance(output, dict) and 'hidden_states' in output:
                        # Use last hidden state, average pooled
                        last_hidden = output['hidden_states'][-1]
                        return last_hidden.mean(dim=1)
                    return None

        except Exception as e:
            logger.warning(f"Failed to encode context: {e}")
            return None

    def decompose_task(
        self,
        task: str,
        context_embedding: Optional[Any] = None
    ) -> Optional[TaskDecomposition]:
        """Use RLM to decompose task into sub-problems.

        RLM analyzes the buggy code and error message to produce
        a structured decomposition that guides TRM refinement.

        Args:
            task: Task description or buggy code.
            context_embedding: Optional context embedding from Mamba.

        Returns:
            TaskDecomposition or None if RLM unavailable.
        """
        if self.rlm is None:
            return None

        try:
            # Build specification for RLM
            spec = f"Analyze and repair the following code:\n```python\n{task}\n```"

            # Run RLM pipeline
            if hasattr(self.rlm, 'run'):
                result = self.rlm.run(spec)

                # Extract decomposition from result
                variables = []
                constraints = []
                subproblems = []

                if hasattr(result, 'variables'):
                    variables = [
                        {"name": v.name, "type": str(v.var_type)}
                        for v in result.variables
                    ]

                if hasattr(result, 'constraints'):
                    constraints = [c.to_dict() for c in result.constraints]

                if hasattr(result, 'decomposition') and result.decomposition:
                    if hasattr(result.decomposition, 'subproblems'):
                        subproblems = result.decomposition.subproblems

                return TaskDecomposition(
                    task_type=TaskType.CODE_REPAIR,
                    subproblems=subproblems or ["Analyze code", "Identify bug", "Apply fix"],
                    variables=variables,
                    constraints=constraints,
                    dependencies=[],
                    reasoning_trace=subproblems[:5] if subproblems else [],
                    confidence=0.7,
                    metadata={
                        "has_context": context_embedding is not None,
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to decompose task: {e}")

        return None

    def refine_output(
        self,
        draft: str,
        context: Optional[Any] = None
    ) -> Optional[str]:
        """Use TRM for iterative refinement.

        TRM performs up to 8 iterations of refinement, achieving
        42 effective depth through recursive processing.

        Args:
            draft: Draft code to refine.
            context: Optional context embedding from Mamba.

        Returns:
            Refined code string or None if TRM unavailable.
        """
        if self.trm is None:
            return None

        if not TORCH_AVAILABLE:
            return None

        try:
            # Encode draft to grid format (64x48)
            grid_height = self.config.trm.grid_height
            grid_width = self.config.trm.grid_width

            grid = torch.zeros(1, grid_height, grid_width, dtype=torch.long, device=self.device)

            lines = draft.split('\n')[:grid_height]
            for i, line in enumerate(lines):
                for j, char in enumerate(line[:grid_width]):
                    grid[0, i, j] = ord(char)

            # Run TRM refinement
            with torch.no_grad():
                if hasattr(self.trm, 'generate'):
                    result = self.trm.generate(
                        grid,
                        max_iterations=self.config.trm.max_iterations,
                    )

                    if isinstance(result, dict) and 'output' in result:
                        output_grid = result['output']
                    else:
                        output_grid = result
                elif hasattr(self.trm, 'forward'):
                    output_grid = self.trm(grid)
                else:
                    return None

            # Decode grid back to code
            if isinstance(output_grid, torch.Tensor):
                output_grid = output_grid.squeeze(0).cpu().numpy()

            lines = []
            for row in output_grid:
                line_chars = []
                for token in row:
                    if token > 0:
                        try:
                            line_chars.append(chr(int(token)))
                        except (ValueError, OverflowError):
                            line_chars.append(' ')
                lines.append(''.join(line_chars).rstrip())

            # Remove trailing empty lines
            while lines and not lines[-1]:
                lines.pop()

            return '\n'.join(lines)

        except Exception as e:
            logger.warning(f"Failed to refine output: {e}")
            return None

    def _elapsed_ms(self, start_time: float) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = dict(self._stats)

        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )
        else:
            stats["success_rate"] = 0.0

        if stats["avg_latency_ms"]:
            stats["mean_latency_ms"] = (
                sum(stats["avg_latency_ms"]) / len(stats["avg_latency_ms"])
            )
        else:
            stats["mean_latency_ms"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "mamba_calls": 0,
            "trm_calls": 0,
            "rlm_calls": 0,
            "avg_latency_ms": [],
        }

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MambaTRMRLMPipeline":
        """Create pipeline from YAML configuration.

        Example YAML:
            pipeline:
              device: auto
              max_context_length: 4096
              timeout_ms: 30000

            mamba:
              d_model: 768
              n_layers: 24
              use_cache: true

            trm:
              grid_height: 64
              grid_width: 48
              max_iterations: 8

            rlm:
              max_generation_attempts: 3
              use_templates: true

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configured MambaTRMRLMPipeline instance.
        """
        config = PipelineConfig.from_yaml(path)
        return cls(config=config)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MambaTRMRLMPipeline":
        """Load a pretrained pipeline configuration.

        Available presets:
        - "tiny": Minimal configuration for testing
        - "small": Small configuration for development
        - "base": Base configuration for production
        - "large": Large configuration for best quality

        Args:
            model_name: Name of the pretrained configuration.

        Returns:
            Configured MambaTRMRLMPipeline instance.
        """
        presets = {
            "tiny": PipelineConfig(
                mamba=MambaConfig(d_model=64, n_layers=2),
                trm=TRMConfig(d_model=64, max_iterations=2),
                rlm=RLMConfig(max_generation_attempts=1),
            ),
            "small": PipelineConfig(
                mamba=MambaConfig(d_model=256, n_layers=8),
                trm=TRMConfig(d_model=128, max_iterations=4),
                rlm=RLMConfig(max_generation_attempts=2),
            ),
            "base": PipelineConfig(
                mamba=MambaConfig(d_model=768, n_layers=24),
                trm=TRMConfig(d_model=256, max_iterations=8),
                rlm=RLMConfig(max_generation_attempts=3),
            ),
            "large": PipelineConfig(
                mamba=MambaConfig(d_model=1024, n_layers=48),
                trm=TRMConfig(d_model=512, max_iterations=8),
                rlm=RLMConfig(max_generation_attempts=5),
            ),
        }

        if model_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Unknown preset: {model_name}. Available: {available}"
            )

        return cls(config=presets[model_name])


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Configuration
    "PipelineConfig",
    "MambaConfig",
    "TRMConfig",
    "RLMConfig",
    # Request/Response
    "PipelineRequest",
    "PipelineResponse",
    "TaskDecomposition",
    "TaskType",
    # Main pipeline
    "MambaTRMRLMPipeline",
    # Component loaders
    "load_mamba",
    "load_trm",
    "load_rlm",
]
