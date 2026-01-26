"""
RLM (Recursive Language Model) Core Implementation

Decomposes programming problems into:
1. Variables (inputs, outputs, intermediates)
2. Constraints (type requirements, value bounds)
3. Relationships (dependencies between variables)
4. Subproblems (decomposed tasks)

This enables systematic code generation by solving subproblems recursively.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time

# Try to import from parent package
try:
    from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory
except ImportError:
    # Fallback definitions for standalone use
    from abc import ABC, abstractmethod

    class TechniqueCategory:
        """Technique categories."""
        CODE_SYNTHESIS = "code_synthesis"
        DECOMPOSITION = "decomposition"
        PROMPTING = "prompting"

    @dataclass
    class TechniqueResult:
        """Result from executing a technique."""
        success: bool
        output: Any
        technique_id: str
        execution_time_ms: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
        intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class TechniqueConfig:
        """Configuration for a technique."""
        technique_id: str
        parameters: Dict[str, Any] = field(default_factory=dict)

    class TechniqueBase(ABC):
        """Base class for techniques."""
        TECHNIQUE_ID: str = "base"
        CATEGORY = TechniqueCategory.PROMPTING

        def __init__(self, config: Optional[TechniqueConfig] = None, **kwargs):
            self.config = config or TechniqueConfig(
                technique_id=self.TECHNIQUE_ID,
                parameters=kwargs,
            )
            self._hooks: Dict[str, List] = {
                "pre_run": [],
                "post_run": [],
                "on_error": [],
            }

        @abstractmethod
        def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
            pass

        def _call_hooks(self, hook_type: str, **kwargs) -> None:
            for hook in self._hooks.get(hook_type, []):
                try:
                    hook(**kwargs)
                except Exception:
                    pass


# Handle both package and standalone imports
try:
    from .variable_extractor import VariableExtractor, ExtractedVariable, VariableType
    from .constraints import ConstraintAnalyzer, Constraint
except ImportError:
    from variable_extractor import VariableExtractor, ExtractedVariable, VariableType
    from constraints import ConstraintAnalyzer, Constraint


@dataclass
class RLMDecomposition:
    """
    Result of RLM decomposition.

    Contains all structured information extracted from the specification
    for code synthesis.

    Attributes:
        variables: List of extracted variables with types and constraints
        constraints: Formal constraints between variables
        dependency_order: Order in which to compute variables (topological)
        subproblems: Natural language descriptions of subproblems
        original_spec: The original specification text
        metadata: Additional metadata from the decomposition process
    """
    variables: List[ExtractedVariable]
    constraints: List[Constraint]
    dependency_order: List[str]
    subproblems: List[str]
    original_spec: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "dependency_order": self.dependency_order,
            "subproblems": self.subproblems,
            "original_spec": self.original_spec,
            "metadata": self.metadata,
        }

    def get_inputs(self) -> List[ExtractedVariable]:
        """Get input variables."""
        return [v for v in self.variables if v.is_input]

    def get_outputs(self) -> List[ExtractedVariable]:
        """Get output variables."""
        return [v for v in self.variables if v.is_output]

    def get_intermediates(self) -> List[ExtractedVariable]:
        """Get intermediate (non-input, non-output) variables."""
        return [v for v in self.variables if not v.is_input and not v.is_output]

    def generate_signature(self, function_name: str = "solve") -> str:
        """
        Generate Python function signature from decomposition.

        Args:
            function_name: Name for the generated function

        Returns:
            Python function signature as string
        """
        inputs = self.get_inputs()
        outputs = self.get_outputs()

        # Build parameter list
        params = []
        for var in inputs:
            annotation = var.to_python_annotation()
            params.append(f"{var.name}: {annotation}")

        # Build return type
        if len(outputs) == 0:
            return_type = "None"
        elif len(outputs) == 1:
            return_type = outputs[0].to_python_annotation()
        else:
            return_type = f"Tuple[{', '.join(v.to_python_annotation() for v in outputs)}]"

        return f"def {function_name}({', '.join(params)}) -> {return_type}:"

    def generate_docstring(self) -> str:
        """Generate docstring from decomposition."""
        lines = ['"""']

        # Brief description from spec
        brief = self.original_spec[:100]
        if len(self.original_spec) > 100:
            brief += "..."
        lines.append(brief)
        lines.append("")

        # Args section
        inputs = self.get_inputs()
        if inputs:
            lines.append("Args:")
            for var in inputs:
                lines.append(f"    {var.name}: {var.description}")

        # Returns section
        outputs = self.get_outputs()
        if outputs:
            lines.append("")
            lines.append("Returns:")
            for var in outputs:
                lines.append(f"    {var.to_python_annotation()}: {var.description}")

        lines.append('"""')
        return "\n".join(lines)


class RLMExtractor(TechniqueBase):
    """
    RLM Variable Extractor - Specialized for decomposition phase.

    This is a focused implementation that handles only the decomposition
    and variable extraction phase of RLM, producing structured RLMDecomposition
    objects for downstream code synthesis.

    The main RLM class in __init__.py handles full code generation;
    this class is for when you need just the decomposition.

    Example:
        >>> rlm = RLMExtractor()
        >>> result = rlm.run("Takes a list of integers, returns sum of even numbers")
        >>> decomp = result.output
        >>> print(decomp.variables)
        >>> print(decomp.subproblems)
    """

    TECHNIQUE_ID = "rlm_extractor"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        use_llm: bool = False,
        max_variables: int = 20,
        infer_intermediates: bool = True,
        **kwargs
    ):
        """
        Initialize the RLM Extractor.

        Args:
            use_llm: Whether to use LLM for enhanced extraction
            max_variables: Maximum number of variables to extract
            infer_intermediates: Whether to infer intermediate variables
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.use_llm = use_llm
        self.max_variables = max_variables
        self.infer_intermediates = infer_intermediates
        self.extractor = VariableExtractor(use_llm=use_llm)
        self.analyzer = ConstraintAnalyzer()

    def decompose(self, specification: str) -> RLMDecomposition:
        """
        Decompose a specification into structured components.

        Args:
            specification: Natural language description of programming task

        Returns:
            RLMDecomposition with variables, constraints, and subproblems
        """
        # 1. Extract variables
        variables = self.extractor.extract(specification)

        # Limit number of variables
        if len(variables) > self.max_variables:
            # Prioritize inputs and outputs
            inputs = [v for v in variables if v.is_input]
            outputs = [v for v in variables if v.is_output]
            others = [v for v in variables if not v.is_input and not v.is_output]

            remaining = self.max_variables - len(inputs) - len(outputs)
            variables = inputs + outputs + others[:max(0, remaining)]

        # 2. Analyze constraints
        constraints = self.analyzer.analyze(variables)

        # 3. Build dependency order
        try:
            graph = self.analyzer.build_dependency_graph(variables)
            order = self.analyzer.topological_sort(graph)
        except ValueError:
            # Cycle detected, use simple ordering
            order = [v.name for v in variables]

        # 4. Identify subproblems
        subproblems = self._identify_subproblems(variables, constraints, specification)

        # 5. Build metadata
        metadata = {
            "num_inputs": len([v for v in variables if v.is_input]),
            "num_outputs": len([v for v in variables if v.is_output]),
            "num_constraints": len(constraints),
            "has_cycles": len(order) != len(set(v.name for v in variables)),
        }

        return RLMDecomposition(
            variables=variables,
            constraints=constraints,
            dependency_order=order,
            subproblems=subproblems,
            original_spec=specification,
            metadata=metadata,
        )

    def _identify_subproblems(
        self,
        variables: List[ExtractedVariable],
        constraints: List[Constraint],
        specification: str
    ) -> List[str]:
        """
        Break down the specification into subproblems.

        Each subproblem represents a discrete computation step.

        Args:
            variables: Extracted variables
            constraints: Formal constraints
            specification: Original specification

        Returns:
            List of subproblem descriptions
        """
        subproblems: List[str] = []

        # 1. Input validation subproblem
        input_vars = [v for v in variables if v.is_input]
        if input_vars:
            input_constraints = [c for c in constraints if c.variable in [v.name for v in input_vars]]
            if input_constraints:
                subproblems.append(
                    f"Validate inputs: {', '.join(v.name for v in input_vars)} "
                    f"({len(input_constraints)} constraints)"
                )

        # 2. Intermediate computation subproblems
        for var in variables:
            if not var.is_input and not var.is_output:
                deps = ', '.join(var.dependencies) if var.dependencies else 'inputs'
                subproblems.append(
                    f"Compute {var.name}: {var.description} (depends on: {deps})"
                )

        # 3. Output computation subproblems
        output_vars = [v for v in variables if v.is_output]
        for var in output_vars:
            deps = ', '.join(var.dependencies) if var.dependencies else 'inputs'
            subproblems.append(
                f"Compute output {var.name}: {var.description} (depends on: {deps})"
            )

        # 4. Look for explicit steps in specification
        import re
        step_matches = re.findall(
            r'(\d+)\.\s*([^0-9]+?)(?=\d+\.|$)',
            specification,
            re.DOTALL
        )
        for num, step_text in step_matches:
            step_text = step_text.strip()
            if step_text and len(step_text) > 5:
                # Check if this step is already covered
                is_covered = any(step_text[:20].lower() in sp.lower() for sp in subproblems)
                if not is_covered:
                    subproblems.append(f"Step {num}: {step_text[:100]}")

        return subproblems

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run RLM decomposition on a specification.

        Args:
            input_data: Natural language specification string
            context: Optional context with additional info

        Returns:
            TechniqueResult with RLMDecomposition as output
        """
        start = time.time()
        trace: List[Dict[str, Any]] = []

        self._call_hooks("pre_run", input_data=input_data)

        if not isinstance(input_data, str):
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                error="Input must be a string specification",
            )

        try:
            # Perform decomposition
            trace.append({"action": "start_decomposition", "spec_length": len(input_data)})

            decomposition = self.decompose(input_data)

            trace.append({
                "action": "extraction_complete",
                "num_variables": len(decomposition.variables),
                "num_constraints": len(decomposition.constraints),
                "num_subproblems": len(decomposition.subproblems),
            })

            self._call_hooks("post_run", result=decomposition)

            return TechniqueResult(
                success=True,
                output=decomposition,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "variables": [v.name for v in decomposition.variables],
                    "has_inputs": any(v.is_input for v in decomposition.variables),
                    "has_outputs": any(v.is_output for v in decomposition.variables),
                },
            )

        except Exception as e:
            self._call_hooks("on_error", error=e)
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# Alias for backward compatibility and convenience
RLM = RLMExtractor
