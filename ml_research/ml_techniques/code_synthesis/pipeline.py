"""
RLM Pipeline - Unified Code Synthesis Pipeline

Integrates all code synthesis components into a unified pipeline:
- Variable extraction from natural language specifications
- Constraint analysis and validation
- Code generation with templates
- Self-debugging with iterative refinement

Provides:
- PipelineConfig: Configuration loaded from YAML
- PipelineResult: Comprehensive result from pipeline execution
- PipelineUpdate: Streaming update for intermediate results
- RLMPipeline: Main orchestrator for end-to-end code synthesis
- Hook system for TRM/Mamba integration

Usage:
    # Basic usage
    pipeline = RLMPipeline()
    result = pipeline.run("Create a function that sorts a list of numbers")
    print(result.code)

    # With configuration
    pipeline = RLMPipeline.from_yaml("config.yaml")
    result = pipeline.run(specification, test_cases=tests)

    # With streaming
    for update in pipeline.run_streaming(specification):
        print(f"Stage: {update.stage}, Progress: {update.progress}")

    # With hooks for TRM integration
    pipeline = RLMPipeline()
    pipeline.register_hook("refinement", trm_refine_fn)
    pipeline.register_hook("context", mamba_context_fn)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
)
import time
import yaml
import logging

# Handle both package and standalone imports
try:
    from .variable_extractor import (
        VariableExtractor,
        ExtractedVariable,
        VariableType,
    )
    from .constraints import (
        ConstraintAnalyzer,
        Constraint,
    )
    from .generator import (
        CodeGenerator,
        GeneratedCode,
        LLMCodeGenerator,
        MultiStepGenerator,
        TemplateLibrary,
    )
    from .debugger import (
        SelfDebugger,
        ExecutionSandbox,
        ErrorAnalyzer,
        DebugResult,
        ExecutionResult,
        ErrorAnalysis,
    )
    from .rlm import (
        RLMExtractor,
        RLMDecomposition,
    )
except ImportError:
    from variable_extractor import (
        VariableExtractor,
        ExtractedVariable,
        VariableType,
    )
    from constraints import (
        ConstraintAnalyzer,
        Constraint,
    )
    from generator import (
        CodeGenerator,
        GeneratedCode,
        LLMCodeGenerator,
        MultiStepGenerator,
        TemplateLibrary,
    )
    from debugger import (
        SelfDebugger,
        ExecutionSandbox,
        ErrorAnalyzer,
        DebugResult,
        ExecutionResult,
        ErrorAnalysis,
    )
    from rlm import (
        RLMExtractor,
        RLMDecomposition,
    )


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class PipelineStage(Enum):
    """Stages in the RLM pipeline."""
    INITIALIZATION = "initialization"
    EXTRACTION = "extraction"
    CONSTRAINT_ANALYSIS = "constraint_analysis"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    DEBUGGING = "debugging"
    REFINEMENT = "refinement"
    COMPLETION = "completion"
    ERROR = "error"


class BackendType(Enum):
    """Supported backend types."""
    LOCAL = "local"           # No LLM, rule-based only
    OPENAI = "openai"         # OpenAI API
    ANTHROPIC = "anthropic"   # Anthropic API
    CUSTOM = "custom"         # Custom backend via callable


class HookType(Enum):
    """Types of hooks available in the pipeline."""
    PRE_EXTRACTION = "pre_extraction"
    POST_EXTRACTION = "post_extraction"
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    PRE_DEBUG = "pre_debug"
    POST_DEBUG = "post_debug"
    REFINEMENT = "refinement"
    CONTEXT = "context"
    ON_ERROR = "on_error"
    ON_STAGE_CHANGE = "on_stage_change"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the RLM pipeline.

    Can be loaded from YAML configuration files for declarative setup.

    Attributes:
        max_generation_attempts: Maximum code generation attempts
        max_debug_attempts: Maximum debugging iterations
        enable_streaming: Whether to yield intermediate results
        backend: Backend type for LLM operations
        model_name: Specific model name for the backend
        execution_timeout: Timeout for code execution in seconds
        enable_validation: Whether to validate generated code
        enable_debugging: Whether to enable self-debugging
        enable_type_inference: Whether to infer types from context
        template_library: Path to custom template library
        model_hooks: Configuration for model hooks
        sandbox_config: Configuration for execution sandbox
        logging_level: Logging verbosity level
    """
    max_generation_attempts: int = 3
    max_debug_attempts: int = 5
    enable_streaming: bool = False
    backend: str = "local"
    model_name: Optional[str] = None
    execution_timeout: float = 5.0
    enable_validation: bool = True
    enable_debugging: bool = True
    enable_type_inference: bool = True
    template_library: Optional[str] = None
    model_hooks: Dict[str, Any] = field(default_factory=dict)
    sandbox_config: Dict[str, Any] = field(default_factory=dict)
    logging_level: str = "INFO"

    # Advanced configuration
    variable_extraction_mode: str = "hybrid"  # "explicit", "inferred", "hybrid"
    constraint_validation_mode: str = "strict"  # "strict", "lenient"
    code_style: str = "clean"  # "clean", "verbose", "minimal"
    include_docstrings: bool = True
    include_type_hints: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        # Handle nested pipeline config
        if "pipeline" in data:
            pipeline_config = data["pipeline"]
        else:
            pipeline_config = data

        # Handle backend config
        backend_config = data.get("backend", {})
        if isinstance(backend_config, dict):
            backend = backend_config.get("type", "local")
            model_name = backend_config.get("model")
        else:
            backend = backend_config
            model_name = None

        # Handle hooks config
        hooks_config = data.get("hooks", {})

        return cls(
            max_generation_attempts=pipeline_config.get("max_generation_attempts", 3),
            max_debug_attempts=pipeline_config.get("max_debug_attempts", 5),
            enable_streaming=pipeline_config.get("enable_streaming", False),
            backend=backend,
            model_name=model_name,
            execution_timeout=pipeline_config.get("execution_timeout", 5.0),
            enable_validation=pipeline_config.get("enable_validation", True),
            enable_debugging=pipeline_config.get("enable_debugging", True),
            enable_type_inference=pipeline_config.get("enable_type_inference", True),
            template_library=pipeline_config.get("template_library"),
            model_hooks=hooks_config,
            sandbox_config=data.get("sandbox", {}),
            logging_level=data.get("logging_level", "INFO"),
            variable_extraction_mode=pipeline_config.get("variable_extraction_mode", "hybrid"),
            constraint_validation_mode=pipeline_config.get("constraint_validation_mode", "strict"),
            code_style=pipeline_config.get("code_style", "clean"),
            include_docstrings=pipeline_config.get("include_docstrings", True),
            include_type_hints=pipeline_config.get("include_type_hints", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "pipeline": {
                "max_generation_attempts": self.max_generation_attempts,
                "max_debug_attempts": self.max_debug_attempts,
                "enable_streaming": self.enable_streaming,
                "execution_timeout": self.execution_timeout,
                "enable_validation": self.enable_validation,
                "enable_debugging": self.enable_debugging,
                "enable_type_inference": self.enable_type_inference,
                "variable_extraction_mode": self.variable_extraction_mode,
                "constraint_validation_mode": self.constraint_validation_mode,
                "code_style": self.code_style,
                "include_docstrings": self.include_docstrings,
                "include_type_hints": self.include_type_hints,
            },
            "backend": {
                "type": self.backend,
                "model": self.model_name,
            },
            "hooks": self.model_hooks,
            "sandbox": self.sandbox_config,
            "logging_level": self.logging_level,
        }


@dataclass
class PipelineUpdate:
    """
    Update emitted during streaming pipeline execution.

    Attributes:
        stage: Current pipeline stage
        progress: Progress within stage (0.0 to 1.0)
        message: Human-readable status message
        data: Stage-specific data (variables, code, errors, etc.)
        timestamp: When this update was generated
        elapsed_ms: Milliseconds since pipeline start
    """
    stage: PipelineStage
    progress: float
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert update to dictionary."""
        return {
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class PipelineResult:
    """
    Comprehensive result from RLM pipeline execution.

    Attributes:
        success: Whether the pipeline completed successfully
        code: Final generated code (if successful)
        is_valid: Whether the code passes validation
        all_tests_passed: Whether all provided tests pass
        specification: Original input specification
        decomposition: RLM decomposition result
        variables: Extracted variables
        constraints: Analyzed constraints
        generation_result: Result from code generation
        debug_result: Result from debugging (if debugging was performed)
        execution_time_ms: Total pipeline execution time
        stages_completed: List of completed stages
        errors: List of error messages (if any)
        metadata: Additional metadata
        trace: Full execution trace for debugging
    """
    success: bool
    code: Optional[str] = None
    is_valid: bool = False
    all_tests_passed: bool = False
    specification: str = ""
    decomposition: Optional[RLMDecomposition] = None
    variables: List[ExtractedVariable] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    generation_result: Optional[GeneratedCode] = None
    debug_result: Optional[DebugResult] = None
    execution_time_ms: float = 0.0
    stages_completed: List[PipelineStage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "code": self.code,
            "is_valid": self.is_valid,
            "all_tests_passed": self.all_tests_passed,
            "specification": self.specification[:500] if self.specification else None,
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "execution_time_ms": self.execution_time_ms,
            "stages_completed": [s.value for s in self.stages_completed],
            "errors": self.errors,
            "metadata": self.metadata,
        }


# =============================================================================
# HOOK RESULT
# =============================================================================

@dataclass
class HookResult:
    """Result from a hook invocation."""
    success: bool
    modified_data: Optional[Any] = None
    should_continue: bool = True
    message: Optional[str] = None


# =============================================================================
# RLM PIPELINE
# =============================================================================

class RLMPipeline:
    """
    Full RLM pipeline for code synthesis.

    Orchestrates the complete workflow from natural language specification
    to working, validated code:

    1. Parse specification -> Extract variables and constraints
    2. Generate code candidates using templates or LLM
    3. Validate generated code (syntax, type checking)
    4. Debug and fix errors iteratively
    5. Return best result

    The pipeline supports:
    - Configuration via YAML files
    - Streaming intermediate results
    - Pluggable hooks for TRM/Mamba integration
    - Multiple backend types (local, OpenAI, Anthropic, custom)

    Example:
        >>> pipeline = RLMPipeline()
        >>> result = pipeline.run(
        ...     "Create a function that takes a list of integers "
        ...     "and returns the sum of all even numbers"
        ... )
        >>> if result.success:
        ...     print(result.code)

        >>> # With test cases
        >>> test_cases = [
        ...     {"inputs": {"numbers": [1, 2, 3, 4]}, "expected": 6},
        ...     {"inputs": {"numbers": []}, "expected": 0},
        ... ]
        >>> result = pipeline.run(spec, test_cases=test_cases)

        >>> # Streaming mode
        >>> for update in pipeline.run_streaming(spec):
        ...     print(f"{update.stage.value}: {update.message}")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[str] = None,
        backend: Optional[Any] = None,
    ):
        """
        Initialize the RLM pipeline.

        Args:
            config: Pipeline configuration object
            config_path: Path to YAML configuration file
            backend: Optional LLM backend (callable or object with generate method)
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfig()

        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.logging_level, logging.INFO))

        # Initialize backend
        self.backend = backend or self._create_backend()

        # Initialize components
        self.extractor = VariableExtractor(
            use_llm=self.backend is not None
        )
        self.constraint_analyzer = ConstraintAnalyzer()
        self.generator = CodeGenerator(
            backend=self.backend,
            auto_format=True,
        )

        # Initialize debugger with sandbox
        sandbox_timeout = self.config.sandbox_config.get(
            "timeout", self.config.execution_timeout
        )
        self.debugger = SelfDebugger(
            backend=self.backend,
            max_attempts=self.config.max_debug_attempts,
            sandbox_timeout=sandbox_timeout,
        )

        # Initialize RLM extractor for decomposition
        self.rlm_extractor = RLMExtractor()

        # Hook registry
        self._hooks: Dict[str, List[Callable]] = {
            hook_type.value: [] for hook_type in HookType
        }

        # Register hooks from config
        self._register_config_hooks()

        # Execution state
        self._current_stage: PipelineStage = PipelineStage.INITIALIZATION
        self._start_time: float = 0.0

    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            return PipelineConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return PipelineConfig()

    def _create_backend(self) -> Optional[Any]:
        """Create backend based on configuration."""
        backend_type = self.config.backend.lower()

        if backend_type == "local":
            return None

        elif backend_type == "openai":
            try:
                import openai
                # Create OpenAI client wrapper
                return self._create_openai_backend()
            except ImportError:
                logger.warning("OpenAI package not installed, falling back to local")
                return None

        elif backend_type == "anthropic":
            try:
                import anthropic
                return self._create_anthropic_backend()
            except ImportError:
                logger.warning("Anthropic package not installed, falling back to local")
                return None

        elif backend_type == "custom":
            # Custom backend should be provided via constructor
            return None

        return None

    def _create_openai_backend(self) -> Optional[Callable]:
        """Create OpenAI backend wrapper."""
        try:
            import openai
            client = openai.OpenAI()
            model = self.config.model_name or "gpt-4"

            def generate(prompt: str, **kwargs) -> str:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.2),
                )
                return response.choices[0].message.content or ""

            return generate
        except Exception as e:
            logger.error(f"Failed to create OpenAI backend: {e}")
            return None

    def _create_anthropic_backend(self) -> Optional[Callable]:
        """Create Anthropic backend wrapper."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            model = self.config.model_name or "claude-3-sonnet-20240229"

            def generate(prompt: str, **kwargs) -> str:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            return generate
        except Exception as e:
            logger.error(f"Failed to create Anthropic backend: {e}")
            return None

    def _register_config_hooks(self) -> None:
        """Register hooks from configuration."""
        hooks_config = self.config.model_hooks

        # Register refinement hook (for TRM integration)
        if "refinement" in hooks_config:
            hook_type = hooks_config["refinement"]
            if hook_type == "trm":
                logger.info("TRM refinement hook registered (placeholder)")
                # Placeholder for TRM integration

        # Register context hook (for Mamba integration)
        if "context" in hooks_config:
            hook_type = hooks_config["context"]
            if hook_type == "mamba":
                logger.info("Mamba context hook registered (placeholder)")
                # Placeholder for Mamba integration

    def register_hook(
        self,
        hook_name: str,
        hook_fn: Callable[..., Optional[HookResult]],
    ) -> None:
        """
        Register a hook function.

        Available hooks:
        - pre_extraction: Called before variable extraction
        - post_extraction: Called after extraction with variables
        - pre_generation: Called before code generation
        - post_generation: Called after generation with result
        - pre_debug: Called before debugging
        - post_debug: Called after debugging
        - refinement: Called when refinement is needed (can invoke TRM)
        - context: Called to get context (can invoke Mamba)
        - on_error: Called when an error occurs
        - on_stage_change: Called when pipeline stage changes

        Args:
            hook_name: Name of the hook (from HookType enum values)
            hook_fn: Callable that receives hook-specific arguments
                     and optionally returns HookResult
        """
        if hook_name in self._hooks:
            self._hooks[hook_name].append(hook_fn)
            logger.debug(f"Registered hook: {hook_name}")
        else:
            logger.warning(f"Unknown hook type: {hook_name}")

    def unregister_hook(
        self,
        hook_name: str,
        hook_fn: Callable,
    ) -> bool:
        """
        Unregister a hook function.

        Args:
            hook_name: Name of the hook
            hook_fn: The function to unregister

        Returns:
            True if hook was found and removed
        """
        if hook_name in self._hooks and hook_fn in self._hooks[hook_name]:
            self._hooks[hook_name].remove(hook_fn)
            return True
        return False

    def _call_hooks(
        self,
        hook_type: str,
        **kwargs,
    ) -> Tuple[bool, Optional[Any]]:
        """
        Call all hooks of a given type.

        Args:
            hook_type: Type of hook to call
            **kwargs: Arguments to pass to hooks

        Returns:
            Tuple of (should_continue, modified_data)
        """
        modified_data = kwargs.get("data")

        for hook in self._hooks.get(hook_type, []):
            try:
                result = hook(**kwargs)
                if isinstance(result, HookResult):
                    if not result.should_continue:
                        return False, result.modified_data
                    if result.modified_data is not None:
                        modified_data = result.modified_data
                elif result is not None:
                    modified_data = result
            except Exception as e:
                logger.error(f"Hook {hook_type} failed: {e}")

        return True, modified_data

    def _set_stage(self, stage: PipelineStage) -> None:
        """Set the current pipeline stage and notify hooks."""
        old_stage = self._current_stage
        self._current_stage = stage
        self._call_hooks(
            HookType.ON_STAGE_CHANGE.value,
            old_stage=old_stage,
            new_stage=stage,
        )

    def _elapsed_ms(self) -> float:
        """Get elapsed milliseconds since pipeline start."""
        return (time.time() - self._start_time) * 1000

    def run(
        self,
        specification: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline from specification to working code.

        Args:
            specification: Natural language description of the desired code
            test_cases: Optional list of test cases for validation
                       Each test case should have "inputs" and optionally "expected"
            context: Optional additional context for generation

        Returns:
            PipelineResult with generated code and metadata
        """
        self._start_time = time.time()
        trace: List[Dict[str, Any]] = []
        stages_completed: List[PipelineStage] = []
        errors: List[str] = []

        # Initialize result
        result = PipelineResult(
            success=False,
            specification=specification,
        )

        try:
            # =================================================================
            # Stage 1: Initialization
            # =================================================================
            self._set_stage(PipelineStage.INITIALIZATION)
            trace.append({
                "stage": "initialization",
                "spec_length": len(specification),
                "has_test_cases": test_cases is not None,
                "elapsed_ms": self._elapsed_ms(),
            })

            # Get context from hook if available
            should_continue, hook_context = self._call_hooks(
                HookType.CONTEXT.value,
                specification=specification,
                test_cases=test_cases,
            )
            if not should_continue:
                result.errors.append("Pipeline aborted by context hook")
                return result

            # Merge hook context with provided context
            full_context = context or {}
            if hook_context:
                full_context.update(hook_context)

            stages_completed.append(PipelineStage.INITIALIZATION)

            # =================================================================
            # Stage 2: Variable Extraction
            # =================================================================
            self._set_stage(PipelineStage.EXTRACTION)

            # Pre-extraction hook
            should_continue, modified_spec = self._call_hooks(
                HookType.PRE_EXTRACTION.value,
                specification=specification,
                data=specification,
            )
            if not should_continue:
                result.errors.append("Pipeline aborted by pre-extraction hook")
                return result
            if modified_spec:
                specification = modified_spec

            # Extract variables
            variables = self.extractor.extract(specification)

            trace.append({
                "stage": "extraction",
                "num_variables": len(variables),
                "variable_names": [v.name for v in variables],
                "elapsed_ms": self._elapsed_ms(),
            })

            result.variables = variables

            # Post-extraction hook
            should_continue, modified_vars = self._call_hooks(
                HookType.POST_EXTRACTION.value,
                variables=variables,
                specification=specification,
                data=variables,
            )
            if not should_continue:
                result.errors.append("Pipeline aborted by post-extraction hook")
                return result
            if modified_vars:
                variables = modified_vars
                result.variables = variables

            stages_completed.append(PipelineStage.EXTRACTION)

            # =================================================================
            # Stage 3: Constraint Analysis
            # =================================================================
            self._set_stage(PipelineStage.CONSTRAINT_ANALYSIS)

            constraints = self.constraint_analyzer.analyze(variables)
            dependency_order = self.constraint_analyzer.get_computation_order(variables)

            trace.append({
                "stage": "constraint_analysis",
                "num_constraints": len(constraints),
                "dependency_order": dependency_order,
                "elapsed_ms": self._elapsed_ms(),
            })

            result.constraints = constraints

            # Create decomposition for result
            decomposition = RLMDecomposition(
                variables=variables,
                constraints=constraints,
                dependency_order=dependency_order,
                subproblems=[],  # Will be filled by RLM extractor if needed
                original_spec=specification,
            )
            result.decomposition = decomposition

            stages_completed.append(PipelineStage.CONSTRAINT_ANALYSIS)

            # =================================================================
            # Stage 4: Code Generation
            # =================================================================
            self._set_stage(PipelineStage.CODE_GENERATION)

            # Pre-generation hook
            should_continue, _ = self._call_hooks(
                HookType.PRE_GENERATION.value,
                variables=variables,
                constraints=constraints,
                specification=specification,
            )
            if not should_continue:
                result.errors.append("Pipeline aborted by pre-generation hook")
                return result

            # Generate code (with multiple attempts if configured)
            best_code: Optional[GeneratedCode] = None
            generation_errors: List[str] = []

            for attempt in range(self.config.max_generation_attempts):
                try:
                    generated = self.generator.generate_from_spec(specification)

                    if generated.is_valid:
                        best_code = generated
                        break
                    else:
                        generation_errors.extend(generated.validation_errors)
                        if best_code is None or len(generated.validation_errors) < len(best_code.validation_errors):
                            best_code = generated

                except Exception as e:
                    generation_errors.append(f"Generation attempt {attempt + 1} failed: {e}")

            if best_code is None:
                result.errors.append("All code generation attempts failed")
                result.errors.extend(generation_errors)
                return result

            trace.append({
                "stage": "code_generation",
                "is_valid": best_code.is_valid,
                "code_lines": len(best_code.code.split('\n')),
                "template_used": best_code.template_used,
                "elapsed_ms": self._elapsed_ms(),
            })

            result.generation_result = best_code
            result.code = best_code.code

            # Post-generation hook
            should_continue, modified_code = self._call_hooks(
                HookType.POST_GENERATION.value,
                code=best_code,
                variables=variables,
                specification=specification,
                data=best_code,
            )
            if not should_continue:
                result.errors.append("Pipeline aborted by post-generation hook")
                return result
            if modified_code and isinstance(modified_code, GeneratedCode):
                best_code = modified_code
                result.generation_result = best_code
                result.code = best_code.code

            stages_completed.append(PipelineStage.CODE_GENERATION)

            # =================================================================
            # Stage 5: Validation
            # =================================================================
            if self.config.enable_validation:
                self._set_stage(PipelineStage.VALIDATION)

                # Basic syntax validation is already done
                result.is_valid = best_code.is_valid

                trace.append({
                    "stage": "validation",
                    "is_valid": result.is_valid,
                    "validation_errors": best_code.validation_errors,
                    "elapsed_ms": self._elapsed_ms(),
                })

                stages_completed.append(PipelineStage.VALIDATION)

            # =================================================================
            # Stage 6: Debugging (if enabled and tests provided)
            # =================================================================
            if (
                self.config.enable_debugging
                and test_cases
                and (not result.is_valid or test_cases)
            ):
                self._set_stage(PipelineStage.DEBUGGING)

                # Pre-debug hook
                should_continue, _ = self._call_hooks(
                    HookType.PRE_DEBUG.value,
                    code=result.code,
                    test_cases=test_cases,
                    specification=specification,
                )
                if not should_continue:
                    result.errors.append("Pipeline aborted by pre-debug hook")
                    return result

                # Run debugger
                debug_result = self.debugger.debug(
                    code=result.code or "",
                    spec=specification,
                    test_cases=test_cases,
                )

                trace.append({
                    "stage": "debugging",
                    "success": debug_result.success,
                    "attempts": debug_result.attempts,
                    "all_tests_passed": debug_result.all_tests_passed,
                    "elapsed_ms": self._elapsed_ms(),
                })

                result.debug_result = debug_result
                result.code = debug_result.final_code
                result.all_tests_passed = debug_result.all_tests_passed

                # Update validity based on debugging
                if debug_result.success:
                    result.is_valid = True

                # Post-debug hook
                should_continue, _ = self._call_hooks(
                    HookType.POST_DEBUG.value,
                    code=result.code,
                    debug_result=debug_result,
                    specification=specification,
                )

                stages_completed.append(PipelineStage.DEBUGGING)

            # =================================================================
            # Stage 7: Refinement (if needed)
            # =================================================================
            if not result.is_valid or (test_cases and not result.all_tests_passed):
                self._set_stage(PipelineStage.REFINEMENT)

                # Call refinement hook (for TRM integration)
                should_continue, refined_code = self._call_hooks(
                    HookType.REFINEMENT.value,
                    code=result.code,
                    specification=specification,
                    variables=variables,
                    constraints=constraints,
                    errors=result.errors,
                    data=result.code,
                )

                if refined_code and isinstance(refined_code, str):
                    result.code = refined_code
                    # Re-validate refined code
                    try:
                        import ast
                        ast.parse(refined_code)
                        result.is_valid = True
                    except SyntaxError:
                        pass

                trace.append({
                    "stage": "refinement",
                    "refinement_applied": refined_code is not None,
                    "elapsed_ms": self._elapsed_ms(),
                })

                stages_completed.append(PipelineStage.REFINEMENT)

            # =================================================================
            # Stage 8: Completion
            # =================================================================
            self._set_stage(PipelineStage.COMPLETION)

            result.success = result.is_valid
            if test_cases:
                result.success = result.success and result.all_tests_passed

            result.execution_time_ms = self._elapsed_ms()
            result.stages_completed = stages_completed
            result.trace = trace
            result.metadata = {
                "config": self.config.to_dict(),
                "backend": self.config.backend,
                "num_variables": len(variables),
                "num_constraints": len(constraints),
            }

            stages_completed.append(PipelineStage.COMPLETION)

            trace.append({
                "stage": "completion",
                "success": result.success,
                "total_time_ms": result.execution_time_ms,
            })

            return result

        except Exception as e:
            self._set_stage(PipelineStage.ERROR)

            # Call error hook
            self._call_hooks(
                HookType.ON_ERROR.value,
                error=e,
                stage=self._current_stage,
            )

            logger.error(f"Pipeline failed: {e}")

            result.success = False
            result.errors.append(str(e))
            result.execution_time_ms = self._elapsed_ms()
            result.stages_completed = stages_completed
            result.trace = trace

            return result

    def run_streaming(
        self,
        specification: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[PipelineUpdate]:
        """
        Stream intermediate results during pipeline execution.

        Yields PipelineUpdate objects at each stage, allowing real-time
        monitoring of pipeline progress.

        Args:
            specification: Natural language description of the desired code
            test_cases: Optional list of test cases for validation
            context: Optional additional context

        Yields:
            PipelineUpdate objects with progress information
        """
        self._start_time = time.time()

        # Yield initialization update
        yield PipelineUpdate(
            stage=PipelineStage.INITIALIZATION,
            progress=0.0,
            message="Initializing pipeline",
            data={"spec_length": len(specification)},
            elapsed_ms=self._elapsed_ms(),
        )

        # Stage 2: Extraction
        yield PipelineUpdate(
            stage=PipelineStage.EXTRACTION,
            progress=0.1,
            message="Extracting variables from specification",
            elapsed_ms=self._elapsed_ms(),
        )

        variables = self.extractor.extract(specification)

        yield PipelineUpdate(
            stage=PipelineStage.EXTRACTION,
            progress=0.2,
            message=f"Extracted {len(variables)} variables",
            data={"variables": [v.name for v in variables]},
            elapsed_ms=self._elapsed_ms(),
        )

        # Stage 3: Constraint Analysis
        yield PipelineUpdate(
            stage=PipelineStage.CONSTRAINT_ANALYSIS,
            progress=0.3,
            message="Analyzing constraints",
            elapsed_ms=self._elapsed_ms(),
        )

        constraints = self.constraint_analyzer.analyze(variables)

        yield PipelineUpdate(
            stage=PipelineStage.CONSTRAINT_ANALYSIS,
            progress=0.4,
            message=f"Found {len(constraints)} constraints",
            data={"num_constraints": len(constraints)},
            elapsed_ms=self._elapsed_ms(),
        )

        # Stage 4: Code Generation
        yield PipelineUpdate(
            stage=PipelineStage.CODE_GENERATION,
            progress=0.5,
            message="Generating code",
            elapsed_ms=self._elapsed_ms(),
        )

        generated = self.generator.generate_from_spec(specification)

        yield PipelineUpdate(
            stage=PipelineStage.CODE_GENERATION,
            progress=0.6,
            message="Code generated" if generated.is_valid else "Code generated with validation errors",
            data={
                "is_valid": generated.is_valid,
                "code_preview": generated.code[:200] if generated.code else None,
            },
            elapsed_ms=self._elapsed_ms(),
        )

        # Stage 5: Validation
        if self.config.enable_validation:
            yield PipelineUpdate(
                stage=PipelineStage.VALIDATION,
                progress=0.7,
                message="Validating generated code",
                data={"validation_errors": generated.validation_errors},
                elapsed_ms=self._elapsed_ms(),
            )

        # Stage 6: Debugging
        if self.config.enable_debugging and test_cases:
            yield PipelineUpdate(
                stage=PipelineStage.DEBUGGING,
                progress=0.8,
                message="Running debugger with test cases",
                data={"num_test_cases": len(test_cases)},
                elapsed_ms=self._elapsed_ms(),
            )

            debug_result = self.debugger.debug(
                code=generated.code,
                spec=specification,
                test_cases=test_cases,
            )

            yield PipelineUpdate(
                stage=PipelineStage.DEBUGGING,
                progress=0.9,
                message=f"Debugging {'succeeded' if debug_result.success else 'completed'}",
                data={
                    "success": debug_result.success,
                    "attempts": debug_result.attempts,
                    "all_tests_passed": debug_result.all_tests_passed,
                },
                elapsed_ms=self._elapsed_ms(),
            )

            final_code = debug_result.final_code
            success = debug_result.success
        else:
            final_code = generated.code
            success = generated.is_valid

        # Stage 7: Completion
        yield PipelineUpdate(
            stage=PipelineStage.COMPLETION,
            progress=1.0,
            message="Pipeline completed",
            data={
                "success": success,
                "code": final_code,
            },
            elapsed_ms=self._elapsed_ms(),
        )

    @classmethod
    def from_yaml(cls, path: str, backend: Optional[Any] = None) -> "RLMPipeline":
        """
        Load pipeline from YAML configuration file.

        Example YAML:
            pipeline:
              max_generation_attempts: 3
              max_debug_attempts: 5
              enable_streaming: true
              enable_debugging: true

            backend:
              type: anthropic
              model: claude-3-sonnet

            hooks:
              refinement: trm
              context: mamba

            sandbox:
              timeout: 10
              max_output_length: 5000

        Args:
            path: Path to YAML configuration file
            backend: Optional override for backend

        Returns:
            Configured RLMPipeline instance
        """
        return cls(config_path=path, backend=backend)

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, Any], PipelineConfig],
        backend: Optional[Any] = None,
    ) -> "RLMPipeline":
        """
        Create pipeline from configuration dict or object.

        Args:
            config: Configuration dict or PipelineConfig object
            backend: Optional override for backend

        Returns:
            Configured RLMPipeline instance
        """
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)
        return cls(config=config, backend=backend)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    backend: Optional[Any] = None,
    enable_debugging: bool = True,
    max_attempts: int = 3,
    **kwargs,
) -> RLMPipeline:
    """
    Create a pipeline with common defaults.

    Args:
        backend: Optional LLM backend
        enable_debugging: Whether to enable debugging
        max_attempts: Maximum generation/debug attempts
        **kwargs: Additional config parameters

    Returns:
        Configured RLMPipeline
    """
    config = PipelineConfig(
        enable_debugging=enable_debugging,
        max_generation_attempts=max_attempts,
        max_debug_attempts=max_attempts,
        **kwargs,
    )
    return RLMPipeline(config=config, backend=backend)


def run_pipeline(
    specification: str,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    backend: Optional[Any] = None,
    **kwargs,
) -> PipelineResult:
    """
    Run the full pipeline in one call.

    Args:
        specification: Natural language specification
        test_cases: Optional test cases
        backend: Optional LLM backend
        **kwargs: Additional config parameters

    Returns:
        PipelineResult
    """
    pipeline = create_pipeline(backend=backend, **kwargs)
    return pipeline.run(specification, test_cases=test_cases)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PipelineStage",
    "BackendType",
    "HookType",
    # Data classes
    "PipelineConfig",
    "PipelineUpdate",
    "PipelineResult",
    "HookResult",
    # Main class
    "RLMPipeline",
    # Convenience functions
    "create_pipeline",
    "run_pipeline",
]
