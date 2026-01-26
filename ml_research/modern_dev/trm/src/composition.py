"""
TRM + RLM Composition Module (TRM-005)

Combines TRM (Tiny Recursive Model) with RLM (Recursive Language Model)
for enhanced code repair through decomposition and iterative refinement.

The key insight is that RLM's ability to decompose problems complements
TRM's iterative refinement capabilities:

1. RLM analyzes buggy code and decomposes the repair strategy
2. Strategy is converted to TRM's input format (64x48 grid)
3. TRM performs iterative refinement (up to 8 iterations)
4. If TRM fails, RLM re-decomposes with error feedback

This creates a powerful feedback loop where high-level reasoning (RLM)
guides low-level refinement (TRM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Union,
    Tuple,
    Iterator,
    Callable,
)
from enum import Enum
import time
import logging
from pathlib import Path

import numpy as np

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local imports - handle both package and direct imports
try:
    from .model import CodeRepairTRM, CodeRepairConfig
except ImportError:
    CodeRepairTRM = None
    CodeRepairConfig = None


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


class RepairStatus(str, Enum):
    """Status of a repair attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"


class DecompositionType(str, Enum):
    """Type of RLM decomposition for code repair."""
    SYNTAX_FIX = "syntax_fix"           # Simple syntax errors
    SEMANTIC_FIX = "semantic_fix"       # Logic/semantic errors
    TYPE_FIX = "type_fix"               # Type-related errors
    REFACTORING = "refactoring"         # Code structure improvements
    PATTERN_MATCH = "pattern_match"     # Known bug patterns
    UNKNOWN = "unknown"                 # Requires analysis


@dataclass
class RepairStrategy:
    """
    RLM's decomposition of how to fix a bug.

    This represents the high-level repair strategy that RLM extracts
    from analyzing the buggy code and error message. It guides TRM's
    iterative refinement process.

    Attributes:
        decomposition_type: Type of bug/repair identified
        bug_locations: List of (line, column) positions of likely bugs
        suggested_fixes: Natural language descriptions of fixes
        variables_affected: Variable names that are affected
        dependencies: Code dependencies that matter for the fix
        confidence: RLM's confidence in this strategy (0-1)
        reasoning_trace: Step-by-step reasoning from RLM
        metadata: Additional strategy metadata
    """
    decomposition_type: DecompositionType
    bug_locations: List[Tuple[int, int]]
    suggested_fixes: List[str]
    variables_affected: List[str]
    dependencies: List[str]
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decomposition_type": self.decomposition_type.value,
            "bug_locations": self.bug_locations,
            "suggested_fixes": self.suggested_fixes,
            "variables_affected": self.variables_affected,
            "dependencies": self.dependencies,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepairStrategy":
        """Create from dictionary."""
        return cls(
            decomposition_type=DecompositionType(data.get("decomposition_type", "unknown")),
            bug_locations=data.get("bug_locations", []),
            suggested_fixes=data.get("suggested_fixes", []),
            variables_affected=data.get("variables_affected", []),
            dependencies=data.get("dependencies", []),
            confidence=data.get("confidence", 0.0),
            reasoning_trace=data.get("reasoning_trace", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TRMInput:
    """
    Formatted input for TRM model.

    TRM operates on a 64x48 grid of token IDs. This class encapsulates
    the grid along with masks that highlight areas of interest.

    Attributes:
        token_grid: 2D array of token IDs [64, 48]
        attention_mask: Valid token positions [64, 48]
        focus_mask: Positions to focus on (bug locations) [64, 48]
        strategy_embedding: Optional embedding of repair strategy
        metadata: Additional input metadata
    """
    token_grid: np.ndarray  # [64, 48] int32
    attention_mask: np.ndarray  # [64, 48] bool
    focus_mask: Optional[np.ndarray] = None  # [64, 48] float32
    strategy_embedding: Optional[np.ndarray] = None  # [embed_dim]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes."""
        assert self.token_grid.shape == (64, 48), \
            f"Expected grid shape (64, 48), got {self.token_grid.shape}"
        assert self.attention_mask.shape == (64, 48), \
            f"Expected mask shape (64, 48), got {self.attention_mask.shape}"
        if self.focus_mask is not None:
            assert self.focus_mask.shape == (64, 48), \
                f"Expected focus mask shape (64, 48), got {self.focus_mask.shape}"

    def to_torch(self) -> Dict[str, "torch.Tensor"]:
        """Convert to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        result = {
            "token_grid": torch.from_numpy(self.token_grid).long(),
            "attention_mask": torch.from_numpy(self.attention_mask).float(),
        }
        if self.focus_mask is not None:
            result["focus_mask"] = torch.from_numpy(self.focus_mask).float()
        if self.strategy_embedding is not None:
            result["strategy_embedding"] = torch.from_numpy(self.strategy_embedding).float()
        return result


@dataclass
class TRMOutput:
    """
    TRM's output with confidence scores.

    Captures the result of TRM's iterative refinement process.

    Attributes:
        output_grid: Repaired token grid [64, 48]
        confidence: Overall confidence in the repair
        iterations_used: Number of iterations TRM performed
        per_position_confidence: Confidence at each position [64, 48]
        trajectory: Optional list of intermediate states
        early_stopped: Whether TRM stopped early due to high confidence
        metadata: Additional output metadata
    """
    output_grid: np.ndarray  # [64, 48] int32
    confidence: float
    iterations_used: int
    per_position_confidence: Optional[np.ndarray] = None  # [64, 48]
    trajectory: Optional[List[np.ndarray]] = None
    early_stopped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output_grid": self.output_grid.tolist(),
            "confidence": self.confidence,
            "iterations_used": self.iterations_used,
            "early_stopped": self.early_stopped,
            "metadata": self.metadata,
        }


@dataclass
class RepairResult:
    """
    Final repair result with metadata.

    The complete result of the TRM+RLM repair pipeline.

    Attributes:
        original_code: The buggy input code
        repaired_code: The fixed output code
        status: Overall repair status
        confidence: Confidence in the repair
        strategy: RLM strategy that guided the repair
        trm_output: Raw TRM output
        feedback_rounds: Number of RLM feedback iterations
        total_time_ms: Total processing time in milliseconds
        validation_result: Optional validation result
        metadata: Additional result metadata
    """
    original_code: str
    repaired_code: str
    status: RepairStatus
    confidence: float
    strategy: Optional[RepairStrategy] = None
    trm_output: Optional[TRMOutput] = None
    feedback_rounds: int = 0
    total_time_ms: float = 0.0
    validation_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_code": self.original_code,
            "repaired_code": self.repaired_code,
            "status": self.status.value,
            "confidence": self.confidence,
            "strategy": self.strategy.to_dict() if self.strategy else None,
            "trm_output": self.trm_output.to_dict() if self.trm_output else None,
            "feedback_rounds": self.feedback_rounds,
            "total_time_ms": self.total_time_ms,
            "validation_result": self.validation_result,
            "metadata": self.metadata,
        }


@dataclass
class CompositionConfig:
    """
    Configuration for TRM + RLM composition.

    Attributes:
        max_trm_iterations: Maximum TRM refinement iterations
        max_feedback_rounds: Maximum RLM feedback rounds
        confidence_threshold: Minimum confidence to accept repair
        early_stop_threshold: TRM early stopping confidence threshold
        use_focus_mask: Whether to use focus masks from RLM
        validate_repairs: Whether to validate repairs (compile/test)
        fallback_to_rlm: If TRM fails, use pure RLM repair
        trm_config: Optional TRM-specific configuration
        timeout_ms: Maximum total processing time
    """
    max_trm_iterations: int = 8
    max_feedback_rounds: int = 3
    confidence_threshold: float = 0.7
    early_stop_threshold: float = 0.95
    use_focus_mask: bool = True
    validate_repairs: bool = True
    fallback_to_rlm: bool = True
    trm_config: Optional[Dict[str, Any]] = None
    timeout_ms: float = 30000.0  # 30 seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_trm_iterations": self.max_trm_iterations,
            "max_feedback_rounds": self.max_feedback_rounds,
            "confidence_threshold": self.confidence_threshold,
            "early_stop_threshold": self.early_stop_threshold,
            "use_focus_mask": self.use_focus_mask,
            "validate_repairs": self.validate_repairs,
            "fallback_to_rlm": self.fallback_to_rlm,
            "trm_config": self.trm_config,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositionConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# MAIN COMPOSITION CLASS
# =============================================================================


class TRMWithRLM:
    """
    Combines RLM decomposition with TRM refinement for code repair.

    This class implements the core composition logic that leverages
    the complementary strengths of RLM (high-level decomposition) and
    TRM (iterative refinement).

    Pipeline:
        1. RLM analyzes buggy code and decomposes repair strategy
        2. Strategy converted to TRM input format
        3. TRM performs iterative refinement (up to 8 iterations)
        4. If TRM fails, RLM re-decomposes with error feedback

    Example:
        >>> from composition import TRMWithRLM, CompositionConfig
        >>> config = CompositionConfig(max_trm_iterations=8)
        >>> composer = TRMWithRLM(trm_model, rlm_pipeline, config)
        >>> result = composer.repair(buggy_code, error_message)
        >>> print(result.repaired_code)

    Attributes:
        trm_model: Trained TRM model for iterative refinement
        rlm_pipeline: RLM pipeline for decomposition
        config: Composition configuration
        tokenizer: Tokenizer for code encoding/decoding
        validator: Optional code validator
    """

    def __init__(
        self,
        trm_model: Any,
        rlm_pipeline: Any,
        config: Optional[CompositionConfig] = None,
        tokenizer: Any = None,
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
    ):
        """
        Initialize TRM + RLM composition.

        Args:
            trm_model: Trained CodeRepairTRM model
            rlm_pipeline: RLM decomposition pipeline (RLMExtractor or similar)
            config: Composition configuration
            tokenizer: Tokenizer for code encoding/decoding
            validator: Optional function to validate repaired code
                       Should return (is_valid, error_message)
        """
        self.trm_model = trm_model
        self.rlm_pipeline = rlm_pipeline
        self.config = config or CompositionConfig()
        self.tokenizer = tokenizer
        self.validator = validator

        # Statistics tracking
        self._stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "feedback_rounds_used": [],
            "trm_iterations_used": [],
            "avg_confidence": [],
        }

        logger.info(
            f"Initialized TRMWithRLM composition with config: "
            f"max_trm_iterations={self.config.max_trm_iterations}, "
            f"max_feedback_rounds={self.config.max_feedback_rounds}"
        )

    def repair(
        self,
        buggy_code: str,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RepairResult:
        """
        End-to-end repair from buggy code to fixed code.

        This is the main entry point for the composition pipeline.

        Args:
            buggy_code: The buggy code to repair
            error_message: Optional error message providing hints
            context: Optional additional context (file path, language, etc.)

        Returns:
            RepairResult with the repaired code and metadata
        """
        start_time = time.time()
        self._stats["total_repairs"] += 1

        context = context or {}
        feedback_rounds = 0
        best_result: Optional[RepairResult] = None

        try:
            # Step 1: Initial RLM decomposition
            strategy = self._rlm_decompose(buggy_code, error_message, context)

            logger.debug(
                f"RLM decomposition: type={strategy.decomposition_type.value}, "
                f"confidence={strategy.confidence:.2f}, "
                f"locations={len(strategy.bug_locations)}"
            )

            # Step 2: Convert strategy to TRM input
            trm_input = self._strategy_to_trm_input(buggy_code, strategy)

            # Feedback loop
            for round_idx in range(self.config.max_feedback_rounds):
                feedback_rounds = round_idx + 1

                # Step 3: Run TRM refinement
                trm_output = self._trm_refine(trm_input)

                logger.debug(
                    f"TRM refinement round {round_idx + 1}: "
                    f"iterations={trm_output.iterations_used}, "
                    f"confidence={trm_output.confidence:.2f}"
                )

                # Decode output to code
                repaired_code = self._decode_grid(trm_output.output_grid)

                # Check confidence threshold
                if trm_output.confidence >= self.config.confidence_threshold:
                    # Validate if configured
                    validation_result = None
                    if self.config.validate_repairs and self.validator:
                        is_valid, val_msg = self.validator(repaired_code)
                        validation_result = {"is_valid": is_valid, "message": val_msg}

                        if not is_valid:
                            # Validation failed - try feedback loop
                            logger.debug(f"Validation failed: {val_msg}")
                            strategy = self._feedback_loop(
                                trm_output, strategy, val_msg, buggy_code
                            )
                            trm_input = self._strategy_to_trm_input(buggy_code, strategy)
                            continue

                    # Success!
                    elapsed_ms = (time.time() - start_time) * 1000
                    result = RepairResult(
                        original_code=buggy_code,
                        repaired_code=repaired_code,
                        status=RepairStatus.SUCCESS,
                        confidence=trm_output.confidence,
                        strategy=strategy,
                        trm_output=trm_output,
                        feedback_rounds=feedback_rounds,
                        total_time_ms=elapsed_ms,
                        validation_result=validation_result,
                        metadata={"context": context},
                    )

                    self._update_stats(result)
                    return result

                # Low confidence - generate new strategy with feedback
                strategy = self._feedback_loop(
                    trm_output, strategy, None, buggy_code
                )
                trm_input = self._strategy_to_trm_input(buggy_code, strategy)

                # Track best result so far
                if best_result is None or trm_output.confidence > best_result.confidence:
                    best_result = RepairResult(
                        original_code=buggy_code,
                        repaired_code=repaired_code,
                        status=RepairStatus.PARTIAL,
                        confidence=trm_output.confidence,
                        strategy=strategy,
                        trm_output=trm_output,
                        feedback_rounds=feedback_rounds,
                        total_time_ms=(time.time() - start_time) * 1000,
                    )

            # Max feedback rounds reached
            if best_result:
                best_result.total_time_ms = (time.time() - start_time) * 1000
                self._update_stats(best_result)
                return best_result

            # Fallback to RLM-only repair if configured
            if self.config.fallback_to_rlm:
                return self._rlm_only_repair(buggy_code, error_message, context, start_time)

            # Complete failure
            elapsed_ms = (time.time() - start_time) * 1000
            return RepairResult(
                original_code=buggy_code,
                repaired_code=buggy_code,  # Return original
                status=RepairStatus.FAILED,
                confidence=0.0,
                feedback_rounds=feedback_rounds,
                total_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.exception(f"Error during repair: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return RepairResult(
                original_code=buggy_code,
                repaired_code=buggy_code,
                status=RepairStatus.ERROR,
                confidence=0.0,
                total_time_ms=elapsed_ms,
                metadata={"error": str(e)},
            )

    def _rlm_decompose(
        self,
        code: str,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RepairStrategy:
        """
        Use RLM to decompose repair into strategy.

        Analyzes the buggy code and error message to produce a
        structured repair strategy that guides TRM.

        Args:
            code: Buggy code to analyze
            error: Optional error message
            context: Optional additional context

        Returns:
            RepairStrategy with decomposition results
        """
        # Build specification for RLM
        spec_parts = [
            f"Analyze the following buggy code and identify the repair strategy:",
            f"```python",
            code,
            f"```",
        ]

        if error:
            spec_parts.extend([
                "",
                f"Error message: {error}",
            ])

        spec = "\n".join(spec_parts)

        # Run RLM decomposition
        if hasattr(self.rlm_pipeline, 'decompose'):
            # Direct RLMExtractor
            decomposition = self.rlm_pipeline.decompose(spec)
            return self._decomposition_to_strategy(decomposition, code, error)
        elif hasattr(self.rlm_pipeline, 'run'):
            # TechniqueBase interface
            result = self.rlm_pipeline.run(spec, context)
            if result.success and result.output:
                return self._decomposition_to_strategy(result.output, code, error)

        # Fallback: basic strategy
        return self._create_basic_strategy(code, error)

    def _decomposition_to_strategy(
        self,
        decomposition: Any,
        code: str,
        error: Optional[str],
    ) -> RepairStrategy:
        """Convert RLM decomposition to RepairStrategy."""
        # Determine decomposition type from variables/subproblems
        decomp_type = DecompositionType.UNKNOWN

        if hasattr(decomposition, 'subproblems'):
            subproblems_text = " ".join(decomposition.subproblems).lower()
            if "syntax" in subproblems_text:
                decomp_type = DecompositionType.SYNTAX_FIX
            elif "type" in subproblems_text:
                decomp_type = DecompositionType.TYPE_FIX
            elif "logic" in subproblems_text or "semantic" in subproblems_text:
                decomp_type = DecompositionType.SEMANTIC_FIX
            elif "refactor" in subproblems_text:
                decomp_type = DecompositionType.REFACTORING

        # Extract bug locations from error message if possible
        bug_locations = []
        if error:
            bug_locations = self._parse_error_locations(error)

        # Extract variable names
        variables = []
        if hasattr(decomposition, 'variables'):
            variables = [v.name for v in decomposition.variables]

        # Extract dependencies
        dependencies = []
        if hasattr(decomposition, 'dependency_order'):
            dependencies = decomposition.dependency_order

        # Build suggested fixes from subproblems
        suggested_fixes = []
        if hasattr(decomposition, 'subproblems'):
            suggested_fixes = decomposition.subproblems

        # Compute confidence based on decomposition quality
        confidence = 0.5  # Base confidence
        if decomp_type != DecompositionType.UNKNOWN:
            confidence += 0.2
        if bug_locations:
            confidence += 0.1
        if suggested_fixes:
            confidence += 0.1
        if hasattr(decomposition, 'metadata'):
            if decomposition.metadata.get('num_constraints', 0) > 0:
                confidence += 0.1

        return RepairStrategy(
            decomposition_type=decomp_type,
            bug_locations=bug_locations,
            suggested_fixes=suggested_fixes,
            variables_affected=variables,
            dependencies=dependencies,
            confidence=min(confidence, 1.0),
            reasoning_trace=suggested_fixes[:5],  # First 5 subproblems as trace
            metadata={
                "error": error,
                "num_variables": len(variables),
            },
        )

    def _create_basic_strategy(
        self,
        code: str,
        error: Optional[str],
    ) -> RepairStrategy:
        """Create basic strategy when RLM decomposition fails."""
        bug_locations = []
        decomp_type = DecompositionType.UNKNOWN

        if error:
            bug_locations = self._parse_error_locations(error)
            if "syntax" in error.lower():
                decomp_type = DecompositionType.SYNTAX_FIX
            elif "type" in error.lower():
                decomp_type = DecompositionType.TYPE_FIX

        return RepairStrategy(
            decomposition_type=decomp_type,
            bug_locations=bug_locations,
            suggested_fixes=["Analyze and fix the code"],
            variables_affected=[],
            dependencies=[],
            confidence=0.3,
        )

    def _parse_error_locations(self, error: str) -> List[Tuple[int, int]]:
        """Parse line/column locations from error message."""
        import re
        locations = []

        # Common patterns: "line X", "line X, column Y", "X:Y:"
        patterns = [
            r"line\s+(\d+)(?:,?\s*col(?:umn)?\s+(\d+))?",
            r"(\d+):(\d+):",
            r"at\s+line\s+(\d+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error.lower())
            for match in matches:
                if isinstance(match, tuple):
                    line = int(match[0])
                    col = int(match[1]) if len(match) > 1 and match[1] else 0
                else:
                    line = int(match)
                    col = 0
                locations.append((line - 1, col))  # 0-indexed

        return locations

    def _strategy_to_trm_input(
        self,
        code: str,
        strategy: RepairStrategy,
    ) -> TRMInput:
        """
        Convert RLM strategy to TRM input format.

        Encodes the code into a 64x48 grid and creates masks
        based on the repair strategy.

        Args:
            code: Source code to encode
            strategy: RLM repair strategy

        Returns:
            TRMInput ready for TRM model
        """
        # Encode code to grid
        token_grid, attention_mask = self._encode_code_to_grid(code)

        # Create focus mask from bug locations
        focus_mask = None
        if self.config.use_focus_mask and strategy.bug_locations:
            focus_mask = np.zeros((64, 48), dtype=np.float32)
            for line, col in strategy.bug_locations:
                if 0 <= line < 64:
                    # Highlight the line with Gaussian-like distribution
                    focus_mask[line, :] = 0.5
                    if 0 <= col < 48:
                        focus_mask[line, col] = 1.0
                        # Also highlight neighbors
                        if col > 0:
                            focus_mask[line, col - 1] = 0.8
                        if col < 47:
                            focus_mask[line, col + 1] = 0.8

        # Create strategy embedding if configured
        strategy_embedding = None
        if strategy.decomposition_type != DecompositionType.UNKNOWN:
            # Simple one-hot encoding of decomposition type
            embedding = np.zeros(len(DecompositionType), dtype=np.float32)
            type_idx = list(DecompositionType).index(strategy.decomposition_type)
            embedding[type_idx] = 1.0
            strategy_embedding = embedding

        return TRMInput(
            token_grid=token_grid,
            attention_mask=attention_mask,
            focus_mask=focus_mask,
            strategy_embedding=strategy_embedding,
            metadata={
                "strategy_confidence": strategy.confidence,
                "decomposition_type": strategy.decomposition_type.value,
            },
        )

    def _encode_code_to_grid(
        self,
        code: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode code string to token grid."""
        grid = np.zeros((64, 48), dtype=np.int32)
        mask = np.zeros((64, 48), dtype=bool)

        if self.tokenizer is None:
            # Simple character-level encoding fallback
            lines = code.split('\n')[:64]
            for i, line in enumerate(lines):
                chars = list(line[:48])
                for j, char in enumerate(chars):
                    grid[i, j] = ord(char)
                    mask[i, j] = True
        else:
            # Use tokenizer
            encoding = self.tokenizer.encode(code)
            tokens = encoding.ids

            # Split on newline tokens (typically token ID 2)
            newline_token = 2
            row, col = 0, 0

            for token in tokens:
                if token == newline_token:
                    row += 1
                    col = 0
                    if row >= 64:
                        break
                else:
                    if col < 48:
                        grid[row, col] = token
                        mask[row, col] = True
                        col += 1

        return grid, mask

    def _trm_refine(self, trm_input: TRMInput) -> TRMOutput:
        """
        Run TRM refinement iterations.

        Args:
            trm_input: Formatted TRM input

        Returns:
            TRMOutput with refinement results
        """
        if not TORCH_AVAILABLE:
            # Fallback: return input unchanged
            return TRMOutput(
                output_grid=trm_input.token_grid.copy(),
                confidence=0.5,
                iterations_used=0,
                metadata={"error": "PyTorch not available"},
            )

        if self.trm_model is None:
            # No model: return input unchanged
            return TRMOutput(
                output_grid=trm_input.token_grid.copy(),
                confidence=0.5,
                iterations_used=0,
                metadata={"error": "No TRM model provided"},
            )

        # Convert to torch tensors
        inputs = trm_input.to_torch()

        # Add batch dimension
        for key in inputs:
            inputs[key] = inputs[key].unsqueeze(0)

        # Run model
        self.trm_model.eval()
        with torch.no_grad():
            result = self.trm_model.generate(
                inputs["token_grid"],
                mask=inputs.get("attention_mask"),
                max_iterations=self.config.max_trm_iterations,
            )

        # Extract results
        output_grid = result["output"].squeeze(0).cpu().numpy()
        confidence = float(result["confidence"].mean().cpu().numpy())
        iterations = result.get("iterations", self.config.max_trm_iterations)

        # Get trajectory if available
        trajectory = None
        if "trajectory" in result:
            trajectory = [t.squeeze(0).cpu().numpy() for t in result["trajectory"]]

        return TRMOutput(
            output_grid=output_grid,
            confidence=confidence,
            iterations_used=iterations,
            trajectory=trajectory,
            early_stopped=iterations < self.config.max_trm_iterations,
        )

    def _decode_grid(self, grid: np.ndarray) -> str:
        """Decode token grid back to code string."""
        if self.tokenizer is None:
            # Simple character-level decoding
            lines = []
            for row in grid:
                line_chars = []
                for token in row:
                    if token > 0:
                        try:
                            line_chars.append(chr(token))
                        except (ValueError, OverflowError):
                            line_chars.append(' ')
                lines.append(''.join(line_chars).rstrip())

            # Remove trailing empty lines
            while lines and not lines[-1]:
                lines.pop()

            return '\n'.join(lines)
        else:
            # Use tokenizer to decode
            tokens = []
            newline_token = 2

            for row in grid:
                row_tokens = [int(t) for t in row if t != 0]
                if row_tokens:
                    tokens.extend(row_tokens)
                    tokens.append(newline_token)

            # Remove trailing newlines
            while tokens and tokens[-1] == newline_token:
                tokens.pop()

            return self.tokenizer.decode(tokens)

    def _feedback_loop(
        self,
        trm_output: TRMOutput,
        original_strategy: RepairStrategy,
        validation_error: Optional[str],
        original_code: str,
    ) -> RepairStrategy:
        """
        If TRM fails, generate new strategy with error feedback.

        Args:
            trm_output: TRM's output
            original_strategy: Previous strategy
            validation_error: Error from validation (if any)
            original_code: Original buggy code

        Returns:
            New RepairStrategy incorporating feedback
        """
        # Build feedback context
        feedback_parts = [
            f"Previous repair attempt with confidence {trm_output.confidence:.2f} was insufficient.",
            f"Previous strategy type: {original_strategy.decomposition_type.value}",
        ]

        if validation_error:
            feedback_parts.append(f"Validation error: {validation_error}")

        if trm_output.early_stopped:
            feedback_parts.append("TRM stopped early, suggesting high local confidence but global failure.")

        feedback_parts.extend([
            "",
            "Please re-analyze with this feedback in mind.",
        ])

        feedback = "\n".join(feedback_parts)

        # Re-run RLM with feedback
        return self._rlm_decompose(
            original_code,
            error=feedback,
            context={"previous_strategy": original_strategy.to_dict()},
        )

    def _rlm_only_repair(
        self,
        code: str,
        error: Optional[str],
        context: Optional[Dict[str, Any]],
        start_time: float,
    ) -> RepairResult:
        """Fallback to RLM-only repair when TRM fails."""
        logger.info("Falling back to RLM-only repair")

        # This would use RLM's code generation capabilities
        # For now, return the original code with low confidence
        elapsed_ms = (time.time() - start_time) * 1000

        return RepairResult(
            original_code=code,
            repaired_code=code,
            status=RepairStatus.PARTIAL,
            confidence=0.3,
            total_time_ms=elapsed_ms,
            metadata={"fallback": "rlm_only"},
        )

    def _update_stats(self, result: RepairResult) -> None:
        """Update internal statistics."""
        if result.status == RepairStatus.SUCCESS:
            self._stats["successful_repairs"] += 1

        self._stats["feedback_rounds_used"].append(result.feedback_rounds)
        self._stats["avg_confidence"].append(result.confidence)

        if result.trm_output:
            self._stats["trm_iterations_used"].append(result.trm_output.iterations_used)

    def get_stats(self) -> Dict[str, Any]:
        """Get composition statistics."""
        stats = dict(self._stats)

        if stats["total_repairs"] > 0:
            stats["success_rate"] = stats["successful_repairs"] / stats["total_repairs"]
        else:
            stats["success_rate"] = 0.0

        if stats["avg_confidence"]:
            stats["mean_confidence"] = np.mean(stats["avg_confidence"])
        else:
            stats["mean_confidence"] = 0.0

        if stats["trm_iterations_used"]:
            stats["mean_trm_iterations"] = np.mean(stats["trm_iterations_used"])
        else:
            stats["mean_trm_iterations"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset internal statistics."""
        self._stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "feedback_rounds_used": [],
            "trm_iterations_used": [],
            "avg_confidence": [],
        }


# =============================================================================
# BATCH PROCESSING
# =============================================================================


class BatchTRMWithRLM:
    """
    Batch processing wrapper for TRMWithRLM.

    Provides efficient batch processing with configurable parallelism.

    Example:
        >>> batch_composer = BatchTRMWithRLM(composer, batch_size=16)
        >>> results = batch_composer.repair_batch(buggy_codes)
    """

    def __init__(
        self,
        composer: TRMWithRLM,
        batch_size: int = 8,
        num_workers: int = 1,
    ):
        """
        Initialize batch processor.

        Args:
            composer: TRMWithRLM instance
            batch_size: Number of samples to process together
            num_workers: Number of parallel workers (1 = sequential)
        """
        self.composer = composer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def repair_batch(
        self,
        buggy_codes: List[str],
        error_messages: Optional[List[Optional[str]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[RepairResult]:
        """
        Repair a batch of buggy codes.

        Args:
            buggy_codes: List of buggy code strings
            error_messages: Optional list of error messages
            progress_callback: Optional callback(completed, total)

        Returns:
            List of RepairResult objects
        """
        if error_messages is None:
            error_messages = [None] * len(buggy_codes)

        results = []
        total = len(buggy_codes)

        for i, (code, error) in enumerate(zip(buggy_codes, error_messages)):
            result = self.composer.repair(code, error)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def repair_streaming(
        self,
        buggy_codes: Iterator[str],
        error_messages: Optional[Iterator[Optional[str]]] = None,
    ) -> Iterator[RepairResult]:
        """
        Stream repair results.

        Args:
            buggy_codes: Iterator of buggy code strings
            error_messages: Optional iterator of error messages

        Yields:
            RepairResult for each input
        """
        if error_messages is None:
            error_messages = iter(lambda: None, None)  # Infinite None generator

        for code, error in zip(buggy_codes, error_messages):
            yield self.composer.repair(code, error)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "RepairStatus",
    "DecompositionType",
    # Data classes
    "RepairStrategy",
    "TRMInput",
    "TRMOutput",
    "RepairResult",
    "CompositionConfig",
    # Main classes
    "TRMWithRLM",
    "BatchTRMWithRLM",
]
