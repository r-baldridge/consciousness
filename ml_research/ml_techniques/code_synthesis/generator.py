"""
Code Generator for RLM (Recursive Language Model)

Generates executable Python code from:
- Natural language specifications
- Extracted variables and constraints
- Predefined templates

This module provides:
- CodeTemplate: Reusable code pattern with placeholders
- TemplateLibrary: Collection of built-in code templates
- CodeGenerator: Main class for generating code from specs
- LLMCodeGenerator: LLM-backed code generation
- MultiStepGenerator: Step-by-step code generation with verification
- GeneratedCode: Result of code generation with validation info
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
import ast
import re
import textwrap

# Handle both package and standalone imports
try:
    from .variable_extractor import VariableExtractor, ExtractedVariable, VariableType
    from .constraints import ConstraintAnalyzer, Constraint
except ImportError:
    from variable_extractor import VariableExtractor, ExtractedVariable, VariableType
    from constraints import ConstraintAnalyzer, Constraint


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CodeTemplate:
    """
    A reusable code pattern with placeholders for code generation.

    Attributes:
        template_id: Unique identifier for the template
        name: Human-readable name
        pattern: String pattern with {placeholders}
        parameters: List of placeholder names that must be filled
        language: Target programming language
        description: Optional description of the template's purpose

    Example:
        >>> template = CodeTemplate(
        ...     template_id="function_def",
        ...     name="Function Definition",
        ...     pattern="def {func_name}({params}):\\n    {body}",
        ...     parameters=["func_name", "params", "body"],
        ... )
        >>> code = template.render(func_name="add", params="a, b", body="return a + b")
        >>> print(code)
        def add(a, b):
            return a + b
    """
    template_id: str
    name: str
    pattern: str
    parameters: List[str]
    language: str = "python"
    description: str = ""

    def render(self, **kwargs) -> str:
        """
        Render the template with provided values.

        Args:
            **kwargs: Values for each placeholder

        Returns:
            Rendered code string

        Raises:
            ValueError: If required parameters are missing
        """
        missing = set(self.parameters) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return self.pattern.format(**kwargs)

    def validate_parameters(self, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate that all required parameters are provided.

        Returns:
            Tuple of (is_valid, missing_parameters)
        """
        missing = [p for p in self.parameters if p not in kwargs]
        return len(missing) == 0, missing


@dataclass
class GeneratedCode:
    """
    Result of code generation.

    Attributes:
        code: The generated Python code
        language: Programming language of the code
        is_valid: Whether the code passes syntax validation
        validation_errors: List of validation error messages
        variables_used: List of variable names used in the code
        template_used: ID of template used (if template-based)
        metadata: Additional generation metadata
    """
    code: str
    language: str = "python"
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    template_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate code after initialization."""
        if self.language == "python" and not self.validation_errors:
            is_valid, error = self._validate_python_syntax()
            if not is_valid and error:
                self.is_valid = False
                self.validation_errors.append(error)

    def _validate_python_syntax(self) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        try:
            ast.parse(self.code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "language": self.language,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "variables_used": self.variables_used,
            "template_used": self.template_used,
            "metadata": self.metadata,
        }


# =============================================================================
# TEMPLATE LIBRARY
# =============================================================================


class TemplateLibrary:
    """
    Collection of built-in code templates for common patterns.

    Provides templates for:
    - Function definitions
    - Class definitions
    - Loop constructs
    - Conditional statements
    - Exception handling
    - List comprehensions
    - Async functions
    - Decorators
    - Context managers

    Example:
        >>> library = TemplateLibrary()
        >>> template = library.get_template("function_def")
        >>> code = template.render(func_name="greet", params="name", body='return f"Hello, {name}!"')
    """

    def __init__(self):
        """Initialize the template library with built-in templates."""
        self._templates: Dict[str, CodeTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load all built-in templates."""
        builtin_templates = [
            # Function definition
            CodeTemplate(
                template_id="function_def",
                name="Function Definition",
                pattern=textwrap.dedent('''
                    def {func_name}({params}):
                        """{docstring}"""
                        {body}
                ''').strip(),
                parameters=["func_name", "params", "docstring", "body"],
                description="Basic function definition with docstring",
            ),

            # Simple function (no docstring)
            CodeTemplate(
                template_id="function_simple",
                name="Simple Function",
                pattern="def {func_name}({params}):\n    {body}",
                parameters=["func_name", "params", "body"],
                description="Simple function without docstring",
            ),

            # Class definition
            CodeTemplate(
                template_id="class_def",
                name="Class Definition",
                pattern=textwrap.dedent('''
                    class {class_name}{inheritance}:
                        """{docstring}"""

                        def __init__(self{init_params}):
                            {init_body}

                        {methods}
                ''').strip(),
                parameters=["class_name", "inheritance", "docstring", "init_params", "init_body", "methods"],
                description="Class definition with init and methods",
            ),

            # Simple class
            CodeTemplate(
                template_id="class_simple",
                name="Simple Class",
                pattern=textwrap.dedent('''
                    class {class_name}:
                        def __init__(self{init_params}):
                            {init_body}
                ''').strip(),
                parameters=["class_name", "init_params", "init_body"],
                description="Simple class with just init",
            ),

            # For loop
            CodeTemplate(
                template_id="loop_for",
                name="For Loop",
                pattern="for {variable} in {iterable}:\n    {body}",
                parameters=["variable", "iterable", "body"],
                description="Standard for loop",
            ),

            # For loop with index
            CodeTemplate(
                template_id="loop_for_enumerate",
                name="For Loop with Index",
                pattern="for {index}, {variable} in enumerate({iterable}):\n    {body}",
                parameters=["index", "variable", "iterable", "body"],
                description="For loop with enumerate",
            ),

            # While loop
            CodeTemplate(
                template_id="loop_while",
                name="While Loop",
                pattern="while {condition}:\n    {body}",
                parameters=["condition", "body"],
                description="Standard while loop",
            ),

            # If statement
            CodeTemplate(
                template_id="conditional_if",
                name="If Statement",
                pattern="if {condition}:\n    {body}",
                parameters=["condition", "body"],
                description="Simple if statement",
            ),

            # If-else statement
            CodeTemplate(
                template_id="conditional_if_else",
                name="If-Else Statement",
                pattern="if {condition}:\n    {if_body}\nelse:\n    {else_body}",
                parameters=["condition", "if_body", "else_body"],
                description="If-else statement",
            ),

            # If-elif-else
            CodeTemplate(
                template_id="conditional_if_elif_else",
                name="If-Elif-Else Statement",
                pattern="if {condition1}:\n    {body1}\nelif {condition2}:\n    {body2}\nelse:\n    {else_body}",
                parameters=["condition1", "body1", "condition2", "body2", "else_body"],
                description="If-elif-else statement",
            ),

            # Try-except
            CodeTemplate(
                template_id="try_except",
                name="Try-Except Block",
                pattern="try:\n    {try_body}\nexcept {exception_type} as {exception_var}:\n    {except_body}",
                parameters=["try_body", "exception_type", "exception_var", "except_body"],
                description="Try-except block",
            ),

            # Try-except-finally
            CodeTemplate(
                template_id="try_except_finally",
                name="Try-Except-Finally Block",
                pattern="try:\n    {try_body}\nexcept {exception_type} as {exception_var}:\n    {except_body}\nfinally:\n    {finally_body}",
                parameters=["try_body", "exception_type", "exception_var", "except_body", "finally_body"],
                description="Try-except-finally block",
            ),

            # List comprehension
            CodeTemplate(
                template_id="list_comprehension",
                name="List Comprehension",
                pattern="[{expression} for {variable} in {iterable}]",
                parameters=["expression", "variable", "iterable"],
                description="Simple list comprehension",
            ),

            # List comprehension with condition
            CodeTemplate(
                template_id="list_comprehension_conditional",
                name="Conditional List Comprehension",
                pattern="[{expression} for {variable} in {iterable} if {condition}]",
                parameters=["expression", "variable", "iterable", "condition"],
                description="List comprehension with filter condition",
            ),

            # Dict comprehension
            CodeTemplate(
                template_id="dict_comprehension",
                name="Dict Comprehension",
                pattern="{{{key_expr}: {value_expr} for {variable} in {iterable}}}",
                parameters=["key_expr", "value_expr", "variable", "iterable"],
                description="Dictionary comprehension",
            ),

            # Async function
            CodeTemplate(
                template_id="async_function",
                name="Async Function",
                pattern=textwrap.dedent('''
                    async def {func_name}({params}):
                        """{docstring}"""
                        {body}
                ''').strip(),
                parameters=["func_name", "params", "docstring", "body"],
                description="Async function definition",
            ),

            # Decorator
            CodeTemplate(
                template_id="decorator",
                name="Decorator",
                pattern=textwrap.dedent('''
                    def {decorator_name}({decorator_params}):
                        def decorator(func):
                            def wrapper(*args, **kwargs):
                                {before_call}
                                result = func(*args, **kwargs)
                                {after_call}
                                return result
                            return wrapper
                        return decorator
                ''').strip(),
                parameters=["decorator_name", "decorator_params", "before_call", "after_call"],
                description="Decorator function pattern",
            ),

            # Simple decorator
            CodeTemplate(
                template_id="decorator_simple",
                name="Simple Decorator",
                pattern=textwrap.dedent('''
                    def {decorator_name}(func):
                        def wrapper(*args, **kwargs):
                            {body}
                            return func(*args, **kwargs)
                        return wrapper
                ''').strip(),
                parameters=["decorator_name", "body"],
                description="Simple decorator without parameters",
            ),

            # Context manager
            CodeTemplate(
                template_id="context_manager",
                name="Context Manager Class",
                pattern=textwrap.dedent('''
                    class {class_name}:
                        def __init__(self{init_params}):
                            {init_body}

                        def __enter__(self):
                            {enter_body}
                            return self

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            {exit_body}
                            return False
                ''').strip(),
                parameters=["class_name", "init_params", "init_body", "enter_body", "exit_body"],
                description="Context manager class pattern",
            ),

            # Context manager function
            CodeTemplate(
                template_id="context_manager_func",
                name="Context Manager Function",
                pattern=textwrap.dedent('''
                    from contextlib import contextmanager

                    @contextmanager
                    def {func_name}({params}):
                        {setup}
                        try:
                            yield {yield_value}
                        finally:
                            {cleanup}
                ''').strip(),
                parameters=["func_name", "params", "setup", "yield_value", "cleanup"],
                description="Context manager using contextmanager decorator",
            ),

            # Property
            CodeTemplate(
                template_id="property_def",
                name="Property Definition",
                pattern=textwrap.dedent('''
                    @property
                    def {prop_name}(self):
                        """{docstring}"""
                        return self._{prop_name}

                    @{prop_name}.setter
                    def {prop_name}(self, value):
                        {validation}
                        self._{prop_name} = value
                ''').strip(),
                parameters=["prop_name", "docstring", "validation"],
                description="Property with getter and setter",
            ),

            # Generator function
            CodeTemplate(
                template_id="generator",
                name="Generator Function",
                pattern=textwrap.dedent('''
                    def {func_name}({params}):
                        """{docstring}"""
                        for {loop_var} in {iterable}:
                            {process}
                            yield {yield_expr}
                ''').strip(),
                parameters=["func_name", "params", "docstring", "loop_var", "iterable", "process", "yield_expr"],
                description="Generator function pattern",
            ),

            # Lambda
            CodeTemplate(
                template_id="lambda",
                name="Lambda Expression",
                pattern="lambda {params}: {expression}",
                parameters=["params", "expression"],
                description="Lambda expression",
            ),

            # Assert statement
            CodeTemplate(
                template_id="assert",
                name="Assert Statement",
                pattern='assert {condition}, "{message}"',
                parameters=["condition", "message"],
                description="Assert statement with message",
            ),

            # Raise exception
            CodeTemplate(
                template_id="raise_exception",
                name="Raise Exception",
                pattern='raise {exception_type}("{message}")',
                parameters=["exception_type", "message"],
                description="Raise an exception",
            ),
        ]

        for template in builtin_templates:
            self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[CodeTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: Unique identifier of the template

        Returns:
            CodeTemplate if found, None otherwise
        """
        return self._templates.get(template_id)

    def register_template(self, template: CodeTemplate) -> None:
        """
        Register a new template.

        Args:
            template: CodeTemplate to register

        Raises:
            ValueError: If template_id already exists
        """
        if template.template_id in self._templates:
            raise ValueError(f"Template '{template.template_id}' already exists")
        self._templates[template.template_id] = template

    def update_template(self, template: CodeTemplate) -> None:
        """
        Update an existing template.

        Args:
            template: CodeTemplate with updated values
        """
        self._templates[template.template_id] = template

    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template by ID.

        Args:
            template_id: ID of template to remove

        Returns:
            True if removed, False if not found
        """
        if template_id in self._templates:
            del self._templates[template_id]
            return True
        return False

    def list_templates(self) -> List[str]:
        """Get list of all template IDs."""
        return list(self._templates.keys())

    def match_pattern(self, spec_text: str) -> List[CodeTemplate]:
        """
        Find templates that match the given specification.

        Args:
            spec_text: Natural language specification

        Returns:
            List of matching templates, ordered by relevance
        """
        spec_lower = spec_text.lower()
        matches: List[Tuple[CodeTemplate, int]] = []

        # Keywords for each pattern type
        pattern_keywords = {
            "function_def": ["function", "def", "method", "procedure", "takes", "returns"],
            "function_simple": ["function", "def", "simple"],
            "class_def": ["class", "object", "type", "model", "entity"],
            "class_simple": ["class", "simple"],
            "loop_for": ["for", "each", "iterate", "loop", "every"],
            "loop_for_enumerate": ["for", "index", "enumerate", "position"],
            "loop_while": ["while", "until", "repeat", "continue"],
            "conditional_if": ["if", "when", "check", "condition"],
            "conditional_if_else": ["if", "else", "otherwise"],
            "conditional_if_elif_else": ["if", "elif", "else", "multiple conditions"],
            "try_except": ["try", "except", "catch", "error", "exception", "handle"],
            "try_except_finally": ["try", "finally", "cleanup"],
            "list_comprehension": ["list", "comprehension", "map", "transform"],
            "list_comprehension_conditional": ["filter", "where", "condition", "comprehension"],
            "dict_comprehension": ["dict", "dictionary", "mapping", "comprehension"],
            "async_function": ["async", "await", "asynchronous", "concurrent"],
            "decorator": ["decorator", "wrap", "enhance", "modify behavior"],
            "decorator_simple": ["decorator", "simple", "wrap"],
            "context_manager": ["context", "with", "resource", "cleanup"],
            "context_manager_func": ["context", "contextmanager", "yield"],
            "property_def": ["property", "getter", "setter", "attribute"],
            "generator": ["generator", "yield", "iterate", "stream"],
            "lambda": ["lambda", "anonymous", "inline function"],
        }

        for template_id, keywords in pattern_keywords.items():
            template = self._templates.get(template_id)
            if template:
                score = sum(1 for kw in keywords if kw in spec_lower)
                if score > 0:
                    matches.append((template, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]


# =============================================================================
# CODE GENERATOR
# =============================================================================


class CodeGenerator:
    """
    Main code generator class.

    Generates Python code from:
    - Natural language specifications
    - Extracted variables
    - Templates

    Example:
        >>> generator = CodeGenerator()
        >>> result = generator.generate_from_spec("Create a function that adds two numbers")
        >>> print(result.code)
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        template_library: Optional[TemplateLibrary] = None,
        auto_format: bool = True,
    ):
        """
        Initialize the code generator.

        Args:
            backend: Optional LLM backend for enhanced generation
            template_library: Optional custom template library
            auto_format: Whether to auto-format generated code
        """
        self.backend = backend
        self.template_library = template_library or TemplateLibrary()
        self.auto_format = auto_format
        self._extractor = VariableExtractor()
        self._analyzer = ConstraintAnalyzer()

    def generate_from_spec(self, spec: str) -> GeneratedCode:
        """
        Generate code from a natural language specification.

        Args:
            spec: Natural language description of the desired code

        Returns:
            GeneratedCode with the generated Python code
        """
        # 1. Extract variables from specification
        variables = self._extractor.extract(spec)

        # 2. Analyze constraints
        constraints = self._analyzer.analyze(variables)

        # 3. Determine code structure based on spec patterns
        matched_templates = self.template_library.match_pattern(spec)

        # 4. Generate code
        if matched_templates:
            # Use best matching template
            template = matched_templates[0]
            code = self._generate_from_template_and_variables(
                template, variables, constraints, spec
            )
            template_used = template.template_id
        else:
            # Generate basic function structure
            code = self._generate_basic_structure(variables, constraints, spec)
            template_used = None

        # 5. Format code if enabled
        if self.auto_format:
            code = self._format_code(code)

        # 6. Validate
        is_valid, validation_error = self._validate_syntax(code)

        return GeneratedCode(
            code=code,
            language="python",
            is_valid=is_valid,
            validation_errors=[validation_error] if validation_error else [],
            variables_used=[v.name for v in variables],
            template_used=template_used,
            metadata={
                "spec": spec[:200],
                "num_variables": len(variables),
                "num_constraints": len(constraints),
            },
        )

    def generate_from_variables(
        self,
        variables: List[ExtractedVariable],
        function_name: str = "process",
    ) -> str:
        """
        Generate code from extracted variables.

        Args:
            variables: List of ExtractedVariable objects
            function_name: Name for the generated function

        Returns:
            Generated Python code string
        """
        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        # Build function signature
        params = ", ".join(
            f"{v.name}: {v.to_python_annotation()}"
            for v in inputs
        )

        # Return type
        if len(outputs) == 1:
            return_type = outputs[0].to_python_annotation()
        elif len(outputs) > 1:
            return_type = f"Tuple[{', '.join(v.to_python_annotation() for v in outputs)}]"
        else:
            return_type = "None"

        # Build docstring
        docstring_lines = ['"""', "Auto-generated function.", ""]
        if inputs:
            docstring_lines.append("Args:")
            for v in inputs:
                docstring_lines.append(f"    {v.name}: {v.description}")
        if outputs:
            docstring_lines.append("")
            docstring_lines.append("Returns:")
            for v in outputs:
                docstring_lines.append(f"    {v.description}")
        docstring_lines.append('"""')
        docstring = "\n    ".join(docstring_lines)

        # Generate constraints as assertions
        constraints_code = []
        for v in inputs:
            for constraint in v.constraints:
                assertion = self._constraint_to_assertion(v.name, constraint)
                if assertion:
                    constraints_code.append(assertion)

        # Generate body
        body_lines = []
        if constraints_code:
            body_lines.extend(constraints_code)
            body_lines.append("")

        # Add placeholder body
        body_lines.append("# TODO: Implement logic")

        if outputs:
            if len(outputs) == 1:
                body_lines.append(f"result = {outputs[0].var_type.default_value()}")
                body_lines.append("return result")
            else:
                results = [v.var_type.default_value() for v in outputs]
                body_lines.append(f"return {', '.join(results)}")
        else:
            body_lines.append("pass")

        body = "\n    ".join(body_lines)

        # Assemble function
        code = f"""def {function_name}({params}) -> {return_type}:
    {docstring}
    {body}
"""
        return code

    def generate_from_template(
        self,
        template_id: str,
        **kwargs
    ) -> str:
        """
        Generate code from a template.

        Args:
            template_id: ID of template to use
            **kwargs: Parameters for the template

        Returns:
            Rendered code string

        Raises:
            ValueError: If template not found or parameters missing
        """
        template = self.template_library.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        return template.render(**kwargs)

    def _generate_from_template_and_variables(
        self,
        template: CodeTemplate,
        variables: List[ExtractedVariable],
        constraints: List[Constraint],
        spec: str,
    ) -> str:
        """Generate code using template and extracted information."""
        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        # Build template parameters based on template type
        params_dict: Dict[str, str] = {}

        if template.template_id.startswith("function"):
            params_dict["func_name"] = self._generate_function_name(spec)
            params_dict["params"] = ", ".join(
                f"{v.name}: {v.to_python_annotation()}"
                for v in inputs
            ) if inputs else ""
            params_dict["docstring"] = spec[:100] + ("..." if len(spec) > 100 else "")

            # Generate body with constraints
            body_lines = []
            for c in constraints:
                if c.constraint_type != "type":  # Skip type constraints for brevity
                    body_lines.append(f'assert {c.expression}, "{c.error_message}"')

            body_lines.append("# TODO: Implement")
            if outputs:
                body_lines.append(f"return {outputs[0].var_type.default_value()}")
            else:
                body_lines.append("pass")

            params_dict["body"] = "\n    ".join(body_lines)

        elif template.template_id.startswith("class"):
            params_dict["class_name"] = self._generate_class_name(spec)
            params_dict["inheritance"] = ""
            params_dict["docstring"] = spec[:100] + ("..." if len(spec) > 100 else "")
            params_dict["init_params"] = ", " + ", ".join(v.name for v in inputs) if inputs else ""
            params_dict["init_body"] = "\n            ".join(
                f"self.{v.name} = {v.name}" for v in inputs
            ) if inputs else "pass"
            params_dict["methods"] = "pass"

        elif template.template_id.startswith("loop"):
            params_dict["variable"] = "item"
            params_dict["iterable"] = inputs[0].name if inputs else "items"
            params_dict["body"] = "pass"
            if "enumerate" in template.template_id:
                params_dict["index"] = "i"

        elif "conditional" in template.template_id:
            params_dict["condition"] = "True"
            params_dict["body"] = "pass"
            params_dict["if_body"] = "pass"
            params_dict["else_body"] = "pass"
            if "elif" in template.template_id:
                params_dict["condition1"] = "True"
                params_dict["body1"] = "pass"
                params_dict["condition2"] = "False"
                params_dict["body2"] = "pass"

        elif "try_except" in template.template_id:
            params_dict["try_body"] = "pass"
            params_dict["exception_type"] = "Exception"
            params_dict["exception_var"] = "e"
            params_dict["except_body"] = "pass"
            if "finally" in template.template_id:
                params_dict["finally_body"] = "pass"

        elif "comprehension" in template.template_id:
            params_dict["expression"] = "x"
            params_dict["variable"] = "x"
            params_dict["iterable"] = inputs[0].name if inputs else "items"
            if "conditional" in template.template_id:
                params_dict["condition"] = "True"
            if "dict" in template.template_id:
                params_dict["key_expr"] = "k"
                params_dict["value_expr"] = "v"

        elif template.template_id == "async_function":
            params_dict["func_name"] = self._generate_function_name(spec)
            params_dict["params"] = ", ".join(v.name for v in inputs) if inputs else ""
            params_dict["docstring"] = spec[:100]
            params_dict["body"] = "pass"

        elif "decorator" in template.template_id:
            params_dict["decorator_name"] = "my_decorator"
            params_dict["decorator_params"] = ""
            params_dict["before_call"] = "pass"
            params_dict["after_call"] = "pass"
            params_dict["body"] = "pass"

        elif "context_manager" in template.template_id:
            params_dict["class_name"] = self._generate_class_name(spec)
            params_dict["func_name"] = self._generate_function_name(spec)
            params_dict["params"] = ""
            params_dict["init_params"] = ""
            params_dict["init_body"] = "pass"
            params_dict["enter_body"] = "pass"
            params_dict["exit_body"] = "pass"
            params_dict["setup"] = "pass"
            params_dict["yield_value"] = "None"
            params_dict["cleanup"] = "pass"

        # Fill in any missing parameters with defaults
        for param in template.parameters:
            if param not in params_dict:
                params_dict[param] = "pass" if param == "body" else ""

        return template.render(**params_dict)

    def _generate_basic_structure(
        self,
        variables: List[ExtractedVariable],
        constraints: List[Constraint],
        spec: str,
    ) -> str:
        """Generate basic function structure when no template matches."""
        func_name = self._generate_function_name(spec)
        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        # Parameters
        params = ", ".join(
            f"{v.name}: {v.to_python_annotation()}"
            for v in inputs
        ) if inputs else ""

        # Return type
        if outputs:
            if len(outputs) == 1:
                return_type = outputs[0].to_python_annotation()
            else:
                return_type = f"Tuple[{', '.join(v.to_python_annotation() for v in outputs)}]"
        else:
            return_type = "Any"

        # Build the function
        lines = [
            f"def {func_name}({params}) -> {return_type}:",
            '    """',
            f"    {spec[:100]}{'...' if len(spec) > 100 else ''}",
            '    """',
        ]

        # Add constraint checks
        for c in constraints:
            if c.constraint_type not in ("type", "optional"):
                lines.append(f'    assert {c.expression}, "{c.error_message}"')

        lines.append("    ")
        lines.append("    # TODO: Implement logic here")

        # Return statement
        if outputs:
            if len(outputs) == 1:
                lines.append(f"    return {outputs[0].var_type.default_value()}")
            else:
                values = ", ".join(v.var_type.default_value() for v in outputs)
                lines.append(f"    return {values}")
        else:
            lines.append("    pass")

        return "\n".join(lines)

    def _generate_function_name(self, spec: str) -> str:
        """Generate a function name from specification."""
        # Extract key words
        spec_lower = spec.lower()

        # Common patterns
        if "calculate" in spec_lower:
            base = "calculate"
        elif "compute" in spec_lower:
            base = "compute"
        elif "get" in spec_lower:
            base = "get"
        elif "find" in spec_lower:
            base = "find"
        elif "check" in spec_lower:
            base = "check"
        elif "validate" in spec_lower:
            base = "validate"
        elif "process" in spec_lower:
            base = "process"
        elif "convert" in spec_lower:
            base = "convert"
        elif "create" in spec_lower:
            base = "create"
        else:
            base = "process"

        # Try to extract a noun for the object
        words = re.findall(r'\b[a-z]+\b', spec_lower)
        stop_words = {'a', 'an', 'the', 'of', 'for', 'to', 'and', 'or', 'that', 'which', 'is', 'are'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]

        if len(meaningful) > 1:
            return f"{base}_{meaningful[1]}"
        return base

    def _generate_class_name(self, spec: str) -> str:
        """Generate a class name from specification."""
        spec_lower = spec.lower()
        words = re.findall(r'\b[a-z]+\b', spec_lower)
        stop_words = {'a', 'an', 'the', 'of', 'for', 'to', 'and', 'or', 'class', 'create'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]

        if meaningful:
            # CamelCase the first meaningful word
            return meaningful[0].capitalize()
        return "MyClass"

    def _constraint_to_assertion(self, var_name: str, constraint: str) -> Optional[str]:
        """Convert a constraint string to an assertion."""
        if "positive" in constraint:
            return f'assert {var_name} > 0, "{var_name} must be positive"'
        elif "non_negative" in constraint:
            return f'assert {var_name} >= 0, "{var_name} must be non-negative"'
        elif "non_empty" in constraint:
            return f'assert len({var_name}) > 0, "{var_name} must not be empty"'
        elif constraint.startswith("min_inclusive:"):
            value = constraint.split(":")[1]
            return f'assert {var_name} >= {value}, "{var_name} must be at least {value}"'
        elif constraint.startswith("max_inclusive:"):
            value = constraint.split(":")[1]
            return f'assert {var_name} <= {value}, "{var_name} must be at most {value}"'
        elif constraint.startswith("min_exclusive:"):
            value = constraint.split(":")[1]
            return f'assert {var_name} > {value}, "{var_name} must be greater than {value}"'
        elif constraint.startswith("max_exclusive:"):
            value = constraint.split(":")[1]
            return f'assert {var_name} < {value}, "{var_name} must be less than {value}"'
        return None

    def _validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def _format_code(self, code: str) -> str:
        """
        Format Python code using black if available.

        Args:
            code: Python code string

        Returns:
            Formatted code string
        """
        try:
            import black
            mode = black.Mode(line_length=88)
            return black.format_str(code, mode=mode)
        except ImportError:
            # black not installed, return original
            return code
        except Exception:
            # Formatting failed, return original
            return code


# =============================================================================
# LLM CODE GENERATOR
# =============================================================================


class LLMCodeGenerator:
    """
    LLM-backed code generator for enhanced code synthesis.

    Uses a language model backend to generate code from
    natural language descriptions and examples.

    Example:
        >>> generator = LLMCodeGenerator(backend=my_llm)
        >>> code = generator.generate("Create a function to sort a list")
    """

    def __init__(
        self,
        backend: Any,
        language: str = "python",
        temperature: float = 0.2,
    ):
        """
        Initialize the LLM code generator.

        Args:
            backend: LLM backend with generate() method or callable
            language: Target programming language
            temperature: Sampling temperature for generation
        """
        self.backend = backend
        self.language = language
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate code from a prompt.

        Args:
            prompt: Natural language description
            context: Optional context with additional information

        Returns:
            Generated code string
        """
        full_prompt = self._build_prompt(prompt, context)
        response = self._call_backend(full_prompt)
        return self._extract_code(response)

    def generate_with_examples(
        self,
        spec: str,
        examples: List[Tuple[str, str]],
    ) -> str:
        """
        Generate code with few-shot examples.

        Args:
            spec: Specification for the code to generate
            examples: List of (specification, code) tuples as examples

        Returns:
            Generated code string
        """
        prompt_parts = [
            "Generate Python code based on the specification.",
            "Here are some examples:",
            "",
        ]

        for i, (ex_spec, ex_code) in enumerate(examples, 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"Specification: {ex_spec}",
                "Code:",
                "```python",
                ex_code,
                "```",
                "",
            ])

        prompt_parts.extend([
            "Now generate code for:",
            f"Specification: {spec}",
            "Code:",
        ])

        full_prompt = "\n".join(prompt_parts)
        response = self._call_backend(full_prompt)
        return self._extract_code(response)

    def complete_partial(
        self,
        partial_code: str,
        spec: str,
    ) -> str:
        """
        Complete partial code based on specification.

        Args:
            partial_code: Partial code to complete
            spec: Specification describing what to complete

        Returns:
            Completed code string
        """
        prompt = f"""Complete the following Python code based on the specification.

Specification: {spec}

Partial code:
```python
{partial_code}
```

Complete the code (include the partial code):
```python"""

        response = self._call_backend(prompt)
        completed = self._extract_code(response)

        # Ensure partial code is included
        if partial_code.strip() not in completed:
            completed = partial_code.rstrip() + "\n" + completed

        return completed

    def _build_prompt(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the generation prompt."""
        parts = [
            f"Generate {self.language} code for the following task:",
            "",
            description,
            "",
        ]

        if context:
            if "input_types" in context:
                parts.append(f"Input types: {context['input_types']}")
            if "output_type" in context:
                parts.append(f"Output type: {context['output_type']}")
            if "constraints" in context:
                parts.append(f"Constraints: {context['constraints']}")

        parts.extend([
            "",
            "Provide only the code, no explanations.",
            f"```{self.language}",
        ])

        return "\n".join(parts)

    def _call_backend(self, prompt: str) -> str:
        """Call the LLM backend."""
        if hasattr(self.backend, 'generate'):
            return self.backend.generate(prompt, temperature=self.temperature)
        elif callable(self.backend):
            return self.backend(prompt)
        else:
            raise ValueError("Backend must have generate() method or be callable")

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Look for def keyword
        if "def " in response:
            lines = response.split('\n')
            code_lines = []
            in_function = False
            indent_level = 0

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("def "):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())

                if in_function:
                    current_indent = len(line) - len(line.lstrip())
                    if stripped and current_indent < indent_level and not stripped.startswith("def "):
                        break
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines).strip()

        return response.strip()


# =============================================================================
# MULTI-STEP GENERATOR
# =============================================================================


class MultiStepGenerator:
    """
    Generates code in steps with verification at each step.

    Steps:
    1. Generate function signature
    2. Generate docstring
    3. Generate function body
    4. Verify and fix issues

    This approach allows for better control and error handling
    compared to single-shot generation.

    Example:
        >>> generator = MultiStepGenerator()
        >>> result = generator.generate("Add two numbers and return the sum")
        >>> print(result.code)
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        max_fix_attempts: int = 3,
    ):
        """
        Initialize the multi-step generator.

        Args:
            backend: Optional LLM backend for enhanced generation
            max_fix_attempts: Maximum attempts to fix validation errors
        """
        self.backend = backend
        self.max_fix_attempts = max_fix_attempts
        self._extractor = VariableExtractor()
        self._analyzer = ConstraintAnalyzer()

    def generate(self, spec: str) -> GeneratedCode:
        """
        Generate code through multiple steps.

        Args:
            spec: Natural language specification

        Returns:
            GeneratedCode with the final generated code
        """
        # Extract information
        variables = self._extractor.extract(spec)
        constraints = self._analyzer.analyze(variables)

        # Step 1: Generate signature
        signature = self._step1_generate_signature(variables, spec)

        # Step 2: Generate docstring
        docstring = self._step2_generate_docstring(variables, spec)

        # Step 3: Generate body
        body = self._step3_generate_body(variables, constraints, spec)

        # Assemble code
        code = self._assemble_code(signature, docstring, body)

        # Step 4: Verify and fix
        code, validation_errors = self._step4_verify_and_fix(code, spec)

        return GeneratedCode(
            code=code,
            language="python",
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            variables_used=[v.name for v in variables],
            template_used=None,
            metadata={
                "generation_method": "multi_step",
                "steps_completed": 4,
            },
        )

    def _step1_generate_signature(
        self,
        variables: List[ExtractedVariable],
        spec: str,
    ) -> str:
        """Step 1: Generate function signature."""
        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        # Generate function name
        func_name = self._generate_function_name(spec)

        # Build parameters
        params = ", ".join(
            f"{v.name}: {v.to_python_annotation()}"
            for v in inputs
        ) if inputs else ""

        # Build return type
        if outputs:
            if len(outputs) == 1:
                return_type = outputs[0].to_python_annotation()
            else:
                return_type = f"Tuple[{', '.join(v.to_python_annotation() for v in outputs)}]"
        else:
            return_type = "Any"

        return f"def {func_name}({params}) -> {return_type}:"

    def _step2_generate_docstring(
        self,
        variables: List[ExtractedVariable],
        spec: str,
    ) -> str:
        """Step 2: Generate docstring."""
        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        lines = ['"""']

        # Brief description
        brief = spec[:100]
        if len(spec) > 100:
            brief += "..."
        lines.append(brief)
        lines.append("")

        # Args
        if inputs:
            lines.append("Args:")
            for v in inputs:
                lines.append(f"    {v.name}: {v.description}")

        # Returns
        if outputs:
            lines.append("")
            lines.append("Returns:")
            for v in outputs:
                lines.append(f"    {v.description}")

        lines.append('"""')
        return "\n".join(lines)

    def _step3_generate_body(
        self,
        variables: List[ExtractedVariable],
        constraints: List[Constraint],
        spec: str,
    ) -> str:
        """Step 3: Generate function body."""
        outputs = [v for v in variables if v.is_output]
        lines = []

        # Add constraint checks
        for c in constraints:
            if c.constraint_type not in ("type", "optional"):
                lines.append(c.to_assertion())

        # Add placeholder implementation
        lines.append("")
        lines.append("# TODO: Implement the logic described in the specification")

        # Return statement
        if outputs:
            if len(outputs) == 1:
                lines.append(f"return {outputs[0].var_type.default_value()}")
            else:
                values = ", ".join(v.var_type.default_value() for v in outputs)
                lines.append(f"return {values}")
        else:
            lines.append("pass")

        return "\n".join(lines)

    def _assemble_code(
        self,
        signature: str,
        docstring: str,
        body: str,
    ) -> str:
        """Assemble the final code from components."""
        # Indent docstring and body
        docstring_indented = textwrap.indent(docstring, "    ")
        body_indented = textwrap.indent(body, "    ")

        return f"{signature}\n{docstring_indented}\n{body_indented}"

    def _step4_verify_and_fix(
        self,
        code: str,
        spec: str,
    ) -> Tuple[str, List[str]]:
        """Step 4: Verify and fix issues."""
        errors: List[str] = []

        for attempt in range(self.max_fix_attempts):
            try:
                ast.parse(code)
                return code, []
            except SyntaxError as e:
                error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
                errors.append(error_msg)

                # Try to fix common issues
                code = self._attempt_fix(code, e)

        return code, errors

    def _attempt_fix(self, code: str, error: SyntaxError) -> str:
        """Attempt to fix common syntax errors."""
        lines = code.split('\n')

        if error.lineno and error.lineno <= len(lines):
            line_idx = error.lineno - 1
            problem_line = lines[line_idx]

            # Fix missing colons
            if "expected ':'" in str(error.msg):
                if not problem_line.rstrip().endswith(':'):
                    lines[line_idx] = problem_line.rstrip() + ':'

            # Fix unmatched quotes
            if "unterminated string" in str(error.msg).lower():
                quote_count = problem_line.count('"') + problem_line.count("'")
                if quote_count % 2 == 1:
                    # Add missing quote
                    if '"' in problem_line:
                        lines[line_idx] = problem_line + '"'
                    else:
                        lines[line_idx] = problem_line + "'"

            # Fix missing indentation
            if "expected an indented block" in str(error.msg).lower():
                if not problem_line.startswith(' ') and not problem_line.startswith('\t'):
                    lines[line_idx] = "    " + problem_line

        return '\n'.join(lines)

    def _generate_function_name(self, spec: str) -> str:
        """Generate a function name from specification."""
        spec_lower = spec.lower()
        words = re.findall(r'\b[a-z]+\b', spec_lower)
        stop_words = {'a', 'an', 'the', 'of', 'for', 'to', 'and', 'or', 'that', 'which', 'is', 'are', 'create', 'function'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]

        if meaningful:
            return '_'.join(meaningful[:3])
        return "process"


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Data classes
    "CodeTemplate",
    "GeneratedCode",
    # Template library
    "TemplateLibrary",
    # Generators
    "CodeGenerator",
    "LLMCodeGenerator",
    "MultiStepGenerator",
]
