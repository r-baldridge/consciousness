"""
Code Synthesis Techniques

Methods for generating, debugging, and executing code as part of
AI application pipelines. Includes RLM (Recursive Language Model)
for algorithmic prompt dissection.

=============================================================================
TECHNIQUES
=============================================================================

1. RLM (Recursive Language Model)
   - Algorithmic prompt dissection into code variables
   - Extracts structure from natural language specs
   - Generates executable code capturing the specification

2. ProgramSynthesis
   - Generate programs from specifications
   - Can use examples, natural language, or formal specs

3. SelfDebugging
   - Run code, observe errors, generate fixes
   - Iterative refinement until correct

4. CodeAsPolicy
   - Use code as intermediate representation for actions
   - Enables composable, precise, verifiable behaviors

=============================================================================
RLM CONCEPT
=============================================================================

RLM transforms natural language into structured code through:

1. VARIABLE EXTRACTION
   - Identify entities, quantities, constraints
   - Map to typed variables

2. RELATIONSHIP MAPPING
   - Extract relationships between variables
   - Convert to functions/expressions

3. CONSTRAINT ENCODING
   - Capture requirements as assertions/conditions
   - Generate validation code

4. RECURSIVE REFINEMENT
   - If spec is complex, decompose and recurse
   - Combine sub-specifications into coherent code

Example:
    Input: "Create a function that takes a list of numbers,
            filters out negatives, squares the rest, and returns
            the sum if it's greater than 100, otherwise returns 0"

    RLM Output:
        def process_numbers(numbers: List[float]) -> float:
            # Variable: filtered - positive numbers only
            filtered = [n for n in numbers if n >= 0]

            # Variable: squared - each number squared
            squared = [n ** 2 for n in filtered]

            # Variable: total - sum of squared values
            total = sum(squared)

            # Constraint: return based on threshold
            return total if total > 100 else 0
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import time

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


class VariableType(Enum):
    """Types of variables extracted from specifications."""
    ENTITY = "entity"           # Named things (users, products, etc.)
    QUANTITY = "quantity"       # Numbers, measurements
    COLLECTION = "collection"   # Lists, sets, mappings
    CONSTRAINT = "constraint"   # Rules, conditions
    FUNCTION = "function"       # Operations, transformations
    RELATIONSHIP = "relationship"  # Connections between entities


@dataclass
class ExtractedVariable:
    """A variable extracted from natural language specification."""
    name: str
    var_type: VariableType
    description: str
    python_type: str = "Any"
    source_text: str = ""
    dependencies: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class CodeBlock:
    """A generated code block with metadata."""
    code: str
    language: str = "python"
    variables_used: List[str] = field(default_factory=list)
    variables_defined: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RLMResult:
    """Result of RLM processing."""
    variables: List[ExtractedVariable]
    code_blocks: List[CodeBlock]
    final_code: str
    execution_result: Optional[Any] = None
    validation_passed: bool = False


class RLM(TechniqueBase):
    """
    Recursive Language Model - Algorithmic Prompt Dissection into Code Variables.

    Transforms natural language specifications into executable code through
    systematic extraction of variables, relationships, and constraints.

    Algorithm:
        1. PARSE: Analyze specification structure
        2. EXTRACT: Identify variables, types, constraints
        3. RELATE: Map relationships between variables
        4. GENERATE: Create code for each component
        5. COMPOSE: Combine into complete program
        6. VALIDATE: Test against specification
        7. REFINE: If validation fails, recurse on subproblems

    Configuration:
        language: Target programming language (default: python)
        max_recursion: Maximum decomposition depth (default: 5)
        variable_extraction: Method for extracting variables
        execute_and_validate: Whether to run generated code
        type_inference: Whether to infer Python types

    Usage:
        rlm = RLM(model=my_model, language="python")
        result = rlm.run('''
            Create a user authentication system that:
            - Validates email format
            - Hashes passwords with bcrypt
            - Generates JWT tokens with 24h expiry
            - Supports refresh tokens
        ''')
        print(result.output)  # Generated auth system code
    """

    TECHNIQUE_ID = "rlm"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        model: Optional[Any] = None,
        language: str = "python",
        max_recursion: int = 5,
        execute_and_validate: bool = False,
        type_inference: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.language = language
        self.max_recursion = max_recursion
        self.execute_and_validate = execute_and_validate
        self.type_inference = type_inference

    def _extract_variables(self, spec: str) -> List[ExtractedVariable]:
        """
        Extract variables from natural language specification.

        Looks for:
            - Nouns (entities, collections)
            - Numbers/measurements (quantities)
            - Verbs (functions/operations)
            - Conditions (constraints)
            - Relationships (dependencies)
        """
        # Placeholder - real implementation uses LLM
        variables = []

        # Pattern matching for common structures
        import re

        # Look for "a/an/the X" patterns (entities)
        entity_pattern = r'\b(?:a|an|the)\s+(\w+)'
        for match in re.finditer(entity_pattern, spec.lower()):
            variables.append(ExtractedVariable(
                name=match.group(1),
                var_type=VariableType.ENTITY,
                description=f"Entity: {match.group(1)}",
                python_type="Any",
                source_text=match.group(0),
            ))

        # Look for "list/array/collection of X" (collections)
        collection_pattern = r'\b(?:list|array|collection|set)\s+of\s+(\w+)'
        for match in re.finditer(collection_pattern, spec.lower()):
            variables.append(ExtractedVariable(
                name=f"{match.group(1)}_list",
                var_type=VariableType.COLLECTION,
                description=f"Collection of {match.group(1)}",
                python_type=f"List[Any]",
                source_text=match.group(0),
            ))

        # Look for numeric constraints
        number_pattern = r'\b(\d+(?:\.\d+)?)\s*(\w+)?'
        for match in re.finditer(number_pattern, spec):
            unit = match.group(2) or "units"
            variables.append(ExtractedVariable(
                name=f"threshold_{len(variables)}",
                var_type=VariableType.QUANTITY,
                description=f"Numeric value: {match.group(1)} {unit}",
                python_type="float",
                source_text=match.group(0),
            ))

        return variables

    def _extract_relationships(
        self,
        spec: str,
        variables: List[ExtractedVariable],
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between variables.

        Returns list of (var1, relationship, var2) tuples.
        """
        relationships = []

        # Placeholder - real implementation uses LLM
        # Look for patterns like "X depends on Y", "X contains Y", etc.

        return relationships

    def _generate_code_block(
        self,
        variable: ExtractedVariable,
        context: Dict[str, Any],
    ) -> CodeBlock:
        """Generate code for a single variable/component."""

        if variable.var_type == VariableType.ENTITY:
            code = f"# {variable.description}\n{variable.name} = None  # TODO: Initialize"
        elif variable.var_type == VariableType.COLLECTION:
            code = f"# {variable.description}\n{variable.name}: {variable.python_type} = []"
        elif variable.var_type == VariableType.QUANTITY:
            code = f"# {variable.description}\n{variable.name}: {variable.python_type} = 0.0"
        elif variable.var_type == VariableType.FUNCTION:
            code = f"def {variable.name}():\n    # {variable.description}\n    pass"
        else:
            code = f"# {variable.description}\n{variable.name} = None"

        return CodeBlock(
            code=code,
            language=self.language,
            variables_defined=[variable.name],
            description=variable.description,
        )

    def _compose_code(
        self,
        spec: str,
        variables: List[ExtractedVariable],
        code_blocks: List[CodeBlock],
    ) -> str:
        """Compose code blocks into complete program."""

        header = f'''"""
Auto-generated by RLM (Recursive Language Model)

Original Specification:
{spec[:500]}{"..." if len(spec) > 500 else ""}

Extracted Variables:
{chr(10).join(f"  - {v.name}: {v.var_type.value} ({v.python_type})" for v in variables)}
"""

from typing import Any, List, Dict, Optional

'''
        body = "\n\n".join(block.code for block in code_blocks)

        return header + body

    def _validate_code(self, code: str, spec: str) -> Tuple[bool, Optional[str]]:
        """Validate generated code against specification."""
        if not self.execute_and_validate:
            return True, None

        try:
            # Try to compile
            compile(code, "<rlm_generated>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def _recursive_refine(
        self,
        spec: str,
        code: str,
        error: str,
        depth: int,
    ) -> str:
        """Recursively refine code based on validation errors."""
        if depth >= self.max_recursion:
            return code

        # Placeholder - real implementation uses LLM to fix errors
        return code

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        spec = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", spec=spec)

        try:
            # Step 1: Extract variables
            variables = self._extract_variables(spec)
            trace.append({
                "action": "extract_variables",
                "count": len(variables),
                "variables": [v.name for v in variables],
            })

            # Step 2: Extract relationships
            relationships = self._extract_relationships(spec, variables)
            trace.append({
                "action": "extract_relationships",
                "count": len(relationships),
            })

            # Step 3: Generate code blocks
            code_blocks = []
            for var in variables:
                block = self._generate_code_block(var, context or {})
                code_blocks.append(block)
                trace.append({
                    "action": "generate_code_block",
                    "variable": var.name,
                })

            # Step 4: Compose
            final_code = self._compose_code(spec, variables, code_blocks)
            trace.append({
                "action": "compose_code",
                "lines": len(final_code.split("\n")),
            })

            # Step 5: Validate
            valid, error = self._validate_code(final_code, spec)
            trace.append({
                "action": "validate",
                "passed": valid,
                "error": error,
            })

            # Step 6: Refine if needed
            if not valid and error:
                final_code = self._recursive_refine(spec, final_code, error, depth=0)
                trace.append({"action": "refine"})

            # Create result
            rlm_result = RLMResult(
                variables=variables,
                code_blocks=code_blocks,
                final_code=final_code,
                validation_passed=valid,
            )

            self._call_hooks("post_run", result=rlm_result)

            return TechniqueResult(
                success=True,
                output=final_code,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "variables": [v.name for v in variables],
                    "validation_passed": valid,
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


class SelfDebugging(TechniqueBase):
    """
    Self-Debugging: Model debugs its own generated code.

    Paper: "Self-Debugging: Teaching LLMs to Debug Their Own Code"

    Algorithm:
        1. Generate initial code
        2. Execute code
        3. If error, analyze error message
        4. Generate fix
        5. Repeat until correct or max iterations

    Configuration:
        max_iterations: Maximum debug attempts (default: 5)
        execution_timeout: Timeout for code execution (default: 30s)
        include_traceback: Include full traceback in analysis
        sandbox: Execute in sandboxed environment
    """

    TECHNIQUE_ID = "self_debugging"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        model: Optional[Any] = None,
        max_iterations: int = 5,
        execution_timeout: int = 30,
        include_traceback: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.include_traceback = include_traceback

    def _execute_code(self, code: str) -> Tuple[bool, Any, Optional[str]]:
        """Execute code and return (success, result, error)."""
        try:
            # Create isolated namespace
            namespace: Dict[str, Any] = {}
            exec(code, namespace)
            return True, namespace, None
        except Exception as e:
            import traceback
            error = traceback.format_exc() if self.include_traceback else str(e)
            return False, None, error

    def _analyze_error(self, code: str, error: str) -> str:
        """Analyze error and suggest fix (placeholder)."""
        # Real implementation uses LLM
        return f"Error analysis: {error[:100]}"

    def _generate_fix(self, code: str, error: str, analysis: str) -> str:
        """Generate fixed code (placeholder)."""
        # Real implementation uses LLM
        return code  # Return original as placeholder

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        code = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        current_code = code
        for iteration in range(self.max_iterations):
            # Execute
            success, result, error = self._execute_code(current_code)
            trace.append({
                "action": "execute",
                "iteration": iteration,
                "success": success,
            })

            if success:
                return TechniqueResult(
                    success=True,
                    output=current_code,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={"iterations": iteration + 1},
                )

            # Analyze and fix
            analysis = self._analyze_error(current_code, error or "")
            trace.append({"action": "analyze", "analysis": analysis[:100]})

            current_code = self._generate_fix(current_code, error or "", analysis)
            trace.append({"action": "fix"})

        # Max iterations reached
        return TechniqueResult(
            success=False,
            output=current_code,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=f"Max iterations ({self.max_iterations}) reached",
        )


class CodeAsPolicy(TechniqueBase):
    """
    Use code as intermediate representation for actions.

    Instead of outputting actions directly, generates code that
    executes to produce actions. Enables:
        - Composable behaviors (functions, loops)
        - Precise parameters
        - Verifiable before execution
        - Reusable policies

    Common in robotics and tool-use scenarios.
    """

    TECHNIQUE_ID = "code_as_policy"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        model: Optional[Any] = None,
        available_primitives: Optional[List[str]] = None,
        execute_policy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.available_primitives = available_primitives or []
        self.execute_policy = execute_policy

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        # Placeholder implementation
        return TechniqueResult(
            success=True,
            output={"placeholder": "CodeAsPolicy not yet implemented"},
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=[],
        )


class ProgramSynthesis(TechniqueBase):
    """
    Generate programs from specifications.

    Supports multiple specification types:
        - Natural language descriptions
        - Input/output examples
        - Formal specifications
        - Test cases

    Uses search + verification to find correct programs.
    """

    TECHNIQUE_ID = "program_synthesis"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        model: Optional[Any] = None,
        spec_type: str = "natural_language",
        max_candidates: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.spec_type = spec_type
        self.max_candidates = max_candidates

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        # Placeholder implementation
        return TechniqueResult(
            success=True,
            output={"placeholder": "ProgramSynthesis not yet implemented"},
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=[],
        )


# =============================================================================
# PROGRAM OF THOUGHTS (PoT)
# =============================================================================

class ProgramOfThoughts(TechniqueBase):
    """
    Program-of-Thoughts (PoT): Generate code to solve reasoning problems.

    Paper: "Program of Thoughts Prompting: Disentangling Computation from
           Reasoning for Numerical Reasoning Tasks" (Chen et al., 2022)
    https://arxiv.org/abs/2211.12588

    Instead of reasoning in natural language (like Chain-of-Thought), PoT
    generates executable code that computes the answer. This is particularly
    effective for:
        - Mathematical reasoning
        - Logical reasoning
        - Multi-step calculations
        - Problems requiring precise computation

    Process:
        1. Parse the problem statement
        2. Generate code that solves the problem
        3. Execute the code
        4. Return the computed answer

    Advantages over CoT:
        - More accurate for numerical computations
        - Explicit variable tracking
        - Verifiable intermediate steps
        - Can leverage external libraries (math, numpy, etc.)

    Configuration:
        language: Target language for code generation (default: python)
        execution_timeout: Max time for code execution in seconds
        code_prompt: Template for code generation
        allowed_imports: List of allowed module imports

    Usage:
        pot = ProgramOfThoughts(
            model=my_model,
            language="python",
            execution_timeout=10,
        )
        result = pot.run(
            "A train travels at 60 mph for 2.5 hours, then at 80 mph for "
            "1.5 hours. What is the total distance traveled?"
        )
    """

    TECHNIQUE_ID = "program_of_thoughts"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    DEFAULT_CODE_PROMPT = """
Solve the following problem by writing Python code.
Write clean, well-commented code that computes the answer step by step.
The final answer should be stored in a variable called 'answer'.

Problem: {problem}

Python code:
```python
"""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        language: str = "python",
        execution_timeout: int = 10,
        code_prompt: Optional[str] = None,
        allowed_imports: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.language = language
        self.execution_timeout = execution_timeout
        self.code_prompt = code_prompt or self.DEFAULT_CODE_PROMPT
        self.allowed_imports = allowed_imports or ["math", "statistics", "fractions", "decimal"]

    def _generate_code(self, problem: str) -> str:
        """Generate code to solve the problem (placeholder)."""
        # Real implementation uses LLM
        return f'''# Problem: {problem[:100]}
# Generated code to solve the problem

# Step 1: Parse the problem
# Step 2: Set up variables
# Step 3: Compute the answer

answer = 0  # Placeholder
'''

    def _execute_code(self, code: str) -> Tuple[bool, Any, Optional[str]]:
        """Execute generated code safely."""
        try:
            # Create restricted namespace with allowed imports
            namespace: Dict[str, Any] = {"__builtins__": {}}

            # Add allowed imports
            import importlib
            for module_name in self.allowed_imports:
                try:
                    namespace[module_name] = importlib.import_module(module_name)
                except ImportError:
                    pass

            # Add safe builtins
            safe_builtins = [
                'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
                'float', 'int', 'len', 'list', 'map', 'max', 'min', 'pow',
                'range', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
            ]
            import builtins
            for name in safe_builtins:
                namespace[name] = getattr(builtins, name)

            # Execute code
            exec(code, namespace)

            # Get answer
            answer = namespace.get('answer', None)
            return True, answer, None

        except Exception as e:
            import traceback
            return False, None, traceback.format_exc()

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()
        trace: List[Dict] = []

        problem = input_data if isinstance(input_data, str) else str(input_data)

        # Generate code
        code = self._generate_code(problem)
        trace.append({
            "action": "generate_code",
            "code_lines": len(code.split('\n')),
        })

        # Execute code
        success, answer, error = self._execute_code(code)
        trace.append({
            "action": "execute",
            "success": success,
            "error": error[:200] if error else None,
        })

        return TechniqueResult(
            success=success,
            output={
                "answer": answer,
                "code": code,
                "execution_success": success,
                "error": error,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=error if not success else None,
        )


# =============================================================================
# SCRATCHPAD
# =============================================================================

@dataclass
class ScratchpadEntry:
    """An entry in the scratchpad."""
    key: str
    value: Any
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    step: int = 0


class ScratchpadFormat(Enum):
    """Formats for scratchpad representation."""
    TEXT = "text"          # Plain text key-value pairs
    JSON = "json"          # JSON format
    CODE = "code"          # Code-like variable assignments
    MARKDOWN = "markdown"  # Markdown formatted


class Scratchpad(TechniqueBase):
    """
    Scratchpad: Working memory for intermediate computations.

    Provides a structured way to store and retrieve intermediate results
    during multi-step reasoning or computation. Acts as external working
    memory that the model can read from and write to.

    Used in various papers for:
        - Algorithmic reasoning (tracking state)
        - Multi-step math (storing intermediate values)
        - Program execution traces
        - Chain-of-thought with explicit state

    Configuration:
        scratchpad_format: How to format the scratchpad (text/json/code/markdown)
        max_entries: Maximum number of entries to maintain

    Usage:
        scratchpad = Scratchpad(
            model=my_model,
            scratchpad_format=ScratchpadFormat.CODE,
            max_entries=20,
        )
        result = scratchpad.run({
            "task": "Calculate compound interest over 5 years",
            "initial_values": {"principal": 1000, "rate": 0.05},
        })
    """

    TECHNIQUE_ID = "scratchpad"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        scratchpad_format: ScratchpadFormat = ScratchpadFormat.TEXT,
        max_entries: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.scratchpad_format = scratchpad_format
        self.max_entries = max_entries
        self._entries: List[ScratchpadEntry] = []
        self._step_counter = 0

    def write(self, key: str, value: Any, description: str = "") -> None:
        """Write a value to the scratchpad."""
        self._step_counter += 1
        entry = ScratchpadEntry(
            key=key,
            value=value,
            description=description,
            step=self._step_counter,
        )
        self._entries.append(entry)

        # Enforce max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def read(self, key: str) -> Optional[Any]:
        """Read the most recent value for a key."""
        for entry in reversed(self._entries):
            if entry.key == key:
                return entry.value
        return None

    def get_all(self, key: str) -> List[Any]:
        """Get all values for a key (history)."""
        return [e.value for e in self._entries if e.key == key]

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        self._step_counter = 0

    def format_scratchpad(self) -> str:
        """Format the scratchpad for inclusion in prompts."""
        if not self._entries:
            return "[Scratchpad is empty]"

        if self.scratchpad_format == ScratchpadFormat.TEXT:
            lines = []
            for e in self._entries:
                desc = f" # {e.description}" if e.description else ""
                lines.append(f"Step {e.step}: {e.key} = {e.value}{desc}")
            return "\n".join(lines)

        elif self.scratchpad_format == ScratchpadFormat.JSON:
            import json
            return json.dumps([
                {"step": e.step, "key": e.key, "value": e.value, "description": e.description}
                for e in self._entries
            ], indent=2)

        elif self.scratchpad_format == ScratchpadFormat.CODE:
            lines = []
            for e in self._entries:
                desc = f"  # {e.description}" if e.description else ""
                lines.append(f"{e.key} = {repr(e.value)}{desc}")
            return "\n".join(lines)

        elif self.scratchpad_format == ScratchpadFormat.MARKDOWN:
            lines = ["| Step | Variable | Value | Description |", "|------|----------|-------|-------------|"]
            for e in self._entries:
                lines.append(f"| {e.step} | `{e.key}` | `{e.value}` | {e.description} |")
            return "\n".join(lines)

        return str(self._entries)

    def _process_task(self, task: str, initial_values: Dict[str, Any]) -> Any:
        """Process task using scratchpad (placeholder)."""
        # Real implementation uses LLM with scratchpad context
        # Initialize with provided values
        for key, value in initial_values.items():
            self.write(key, value, "initial value")

        # Placeholder: just return initial values
        return {"result": "Processed", "steps": len(self._entries)}

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()
        trace: List[Dict] = []

        # Clear previous state
        self.clear()

        # Parse input
        if isinstance(input_data, dict):
            task = input_data.get("task", "")
            initial_values = input_data.get("initial_values", {})
        else:
            task = str(input_data)
            initial_values = {}

        trace.append({
            "action": "initialize",
            "initial_values": list(initial_values.keys()),
        })

        # Process task
        result = self._process_task(task, initial_values)
        trace.append({
            "action": "process",
            "entries_created": len(self._entries),
        })

        return TechniqueResult(
            success=True,
            output={
                "result": result,
                "scratchpad": self.format_scratchpad(),
                "entries": [
                    {"key": e.key, "value": e.value, "step": e.step}
                    for e in self._entries
                ],
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# PAL (PROGRAM-AIDED LANGUAGE)
# =============================================================================

class PAL(TechniqueBase):
    """
    PAL: Program-Aided Language Models.

    Paper: "PAL: Program-aided Language Models" (Gao et al., 2022)
    https://arxiv.org/abs/2211.10435

    Similar to Program-of-Thoughts, PAL uses code as a reasoning medium.
    The key insight is that LLMs are better at generating programs that
    compute answers than computing answers directly in natural language.

    Process:
        1. Read natural language problem
        2. Generate Python program that solves it
        3. Execute the program
        4. Return the result

    PAL is particularly effective for:
        - Math word problems
        - Symbolic reasoning
        - Algorithmic tasks
        - Problems requiring precise computation

    Configuration:
        runtime: Execution runtime (python/javascript/etc.)
        allowed_imports: Modules the generated code can import

    Usage:
        pal = PAL(
            model=my_model,
            runtime="python",
            allowed_imports=["math", "datetime"],
        )
        result = pal.run(
            "If John has 3 apples and buys 2 more bags with 6 apples each, "
            "how many apples does he have in total?"
        )
    """

    TECHNIQUE_ID = "pal"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    DEFAULT_PROMPT = """
Read the following problem and write a Python program that computes the answer.
Use clear variable names that match the problem description.
Store the final answer in a variable called 'answer'.

Problem: {problem}

Python program:
```python
"""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        runtime: str = "python",
        allowed_imports: Optional[List[str]] = None,
        execution_timeout: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.runtime = runtime
        self.allowed_imports = allowed_imports or ["math", "datetime", "collections", "itertools"]
        self.execution_timeout = execution_timeout

    def _generate_program(self, problem: str) -> str:
        """Generate a program to solve the problem (placeholder)."""
        # Real implementation uses LLM
        return f'''# Problem: {problem[:100]}
# PAL-generated solution

# Define variables from the problem
initial_amount = 0
additional_amount = 0

# Compute the answer
answer = initial_amount + additional_amount
'''

    def _execute_program(self, program: str) -> Tuple[bool, Any, Optional[str]]:
        """Execute the generated program."""
        try:
            namespace: Dict[str, Any] = {}

            # Add allowed imports
            import importlib
            for module_name in self.allowed_imports:
                try:
                    namespace[module_name] = importlib.import_module(module_name)
                except ImportError:
                    pass

            # Add safe builtins
            import builtins
            safe_builtins = [
                'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
                'float', 'int', 'len', 'list', 'map', 'max', 'min', 'pow',
                'range', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
                'print',
            ]
            for name in safe_builtins:
                if hasattr(builtins, name):
                    namespace[name] = getattr(builtins, name)

            exec(program, namespace)
            answer = namespace.get('answer', None)
            return True, answer, None

        except Exception as e:
            import traceback
            return False, None, traceback.format_exc()

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()
        trace: List[Dict] = []

        problem = input_data if isinstance(input_data, str) else str(input_data)

        # Generate program
        program = self._generate_program(problem)
        trace.append({
            "action": "generate_program",
            "lines": len(program.split('\n')),
        })

        # Execute program
        success, answer, error = self._execute_program(program)
        trace.append({
            "action": "execute",
            "success": success,
        })

        return TechniqueResult(
            success=success,
            output={
                "answer": answer,
                "program": program,
                "runtime": self.runtime,
                "execution_success": success,
                "error": error,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=error if not success else None,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VariableType",
    "ScratchpadFormat",
    # Data classes
    "ExtractedVariable",
    "CodeBlock",
    "RLMResult",
    "ScratchpadEntry",
    # Techniques
    "RLM",
    "SelfDebugging",
    "CodeAsPolicy",
    "ProgramSynthesis",
    "ProgramOfThoughts",
    "Scratchpad",
    "PAL",
]
