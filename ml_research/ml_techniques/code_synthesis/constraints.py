"""
Constraint Analyzer for RLM (Recursive Language Model)

Analyzes and validates constraints between variables extracted from
natural language specifications.

This module provides:
- Constraint dataclass for formal constraint representation
- ConstraintAnalyzer for building and validating constraint systems
- Dependency graph construction and topological sorting
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import re

# Handle both package and standalone imports
try:
    from .variable_extractor import ExtractedVariable, VariableType
except ImportError:
    from variable_extractor import ExtractedVariable, VariableType


@dataclass
class Constraint:
    """
    A formal constraint on a variable.

    Attributes:
        variable: Name of the constrained variable
        constraint_type: Category of constraint (type, range, relationship, format, custom)
        expression: Python expression that evaluates to True if constraint is satisfied
        is_hard: Whether this is a hard constraint (must satisfy) or soft (preference)
        error_message: Human-readable error message when constraint is violated
        depends_on: Other variables this constraint references
        priority: Evaluation priority (higher = evaluated first)
    """
    variable: str
    constraint_type: str  # "type", "range", "relationship", "format", "custom"
    expression: str
    is_hard: bool = True
    error_message: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variable": self.variable,
            "constraint_type": self.constraint_type,
            "expression": self.expression,
            "is_hard": self.is_hard,
            "error_message": self.error_message,
            "depends_on": self.depends_on,
            "priority": self.priority,
        }

    def to_assertion(self) -> str:
        """Convert to Python assertion statement."""
        msg = self.error_message or f"Constraint failed: {self.expression}"
        return f'assert {self.expression}, "{msg}"'

    def to_if_check(self) -> str:
        """Convert to if statement with raise."""
        msg = self.error_message or f"Constraint violated: {self.expression}"
        return f'if not ({self.expression}):\n    raise ValueError("{msg}")'


class ConstraintAnalyzer:
    """
    Analyzes and validates constraints between variables.

    Builds formal constraints from extracted variables and their
    constraint strings, and provides validation capabilities.

    Example:
        >>> from variable_extractor import VariableExtractor
        >>> extractor = VariableExtractor()
        >>> variables = extractor.extract("Takes a positive integer greater than 0")
        >>> analyzer = ConstraintAnalyzer()
        >>> constraints = analyzer.analyze(variables)
        >>> for c in constraints:
        ...     print(f"{c.variable}: {c.expression}")
    """

    # Type to Python isinstance check mapping
    TYPE_CHECKS: Dict[VariableType, str] = {
        VariableType.INTEGER: "int",
        VariableType.FLOAT: "(int, float)",
        VariableType.STRING: "str",
        VariableType.BOOLEAN: "bool",
        VariableType.LIST: "list",
        VariableType.DICT: "dict",
        VariableType.SET: "set",
        VariableType.TUPLE: "tuple",
        VariableType.FUNCTION: "callable",
        VariableType.NONE: "type(None)",
    }

    def __init__(self):
        """Initialize the constraint analyzer."""
        self.constraints: List[Constraint] = []
        self._dependency_graph: Dict[str, List[str]] = {}

    def analyze(self, variables: List[ExtractedVariable]) -> List[Constraint]:
        """
        Analyze variables and generate formal constraints.

        Args:
            variables: List of extracted variables

        Returns:
            List of Constraint objects
        """
        constraints: List[Constraint] = []

        for var in variables:
            # 1. Type constraint
            if var.var_type != VariableType.UNKNOWN:
                type_constraint = self._create_type_constraint(var)
                if type_constraint:
                    constraints.append(type_constraint)

            # 2. Parse string constraints from extraction
            for constraint_str in var.constraints:
                parsed = self._parse_constraint(var.name, constraint_str)
                if parsed:
                    constraints.append(parsed)

            # 3. Element type constraints for collections
            if var.element_type and var.var_type in (
                VariableType.LIST, VariableType.SET, VariableType.TUPLE
            ):
                element_constraint = self._create_element_type_constraint(var)
                if element_constraint:
                    constraints.append(element_constraint)

        self.constraints = constraints
        return constraints

    def _create_type_constraint(self, var: ExtractedVariable) -> Optional[Constraint]:
        """Create a type checking constraint for a variable."""
        python_type = self.TYPE_CHECKS.get(var.var_type)
        if not python_type:
            return None

        if python_type == "callable":
            expression = f"callable({var.name})"
        else:
            expression = f"isinstance({var.name}, {python_type})"

        return Constraint(
            variable=var.name,
            constraint_type="type",
            expression=expression,
            error_message=f"{var.name} must be of type {var.var_type.value}",
            priority=100,  # Type constraints evaluated first
        )

    def _create_element_type_constraint(
        self,
        var: ExtractedVariable
    ) -> Optional[Constraint]:
        """Create element type constraint for collections."""
        if not var.element_type:
            return None

        python_type = self.TYPE_CHECKS.get(var.element_type)
        if not python_type:
            return None

        expression = f"all(isinstance(e, {python_type}) for e in {var.name})"

        return Constraint(
            variable=var.name,
            constraint_type="element_type",
            expression=expression,
            error_message=f"All elements of {var.name} must be of type {var.element_type.value}",
            priority=90,
            depends_on=[var.name],
        )

    def _parse_constraint(
        self,
        var_name: str,
        constraint_str: str
    ) -> Optional[Constraint]:
        """
        Parse a constraint string into a Constraint object.

        Constraint string format: "constraint_type:value" or just "constraint_type"
        """
        parts = constraint_str.split(':', 1)
        constraint_type = parts[0]
        value = parts[1] if len(parts) > 1 else ""

        # Map constraint types to expressions
        expressions: Dict[str, Tuple[str, str, int]] = {
            'min_exclusive': (
                f"{var_name} > {self._extract_number(value)}",
                f"{var_name} must be greater than {self._extract_number(value)}",
                50
            ),
            'max_exclusive': (
                f"{var_name} < {self._extract_number(value)}",
                f"{var_name} must be less than {self._extract_number(value)}",
                50
            ),
            'min_inclusive': (
                f"{var_name} >= {self._extract_number(value)}",
                f"{var_name} must be at least {self._extract_number(value)}",
                50
            ),
            'max_inclusive': (
                f"{var_name} <= {self._extract_number(value)}",
                f"{var_name} must be at most {self._extract_number(value)}",
                50
            ),
            'non_negative': (
                f"{var_name} >= 0",
                f"{var_name} must be non-negative",
                50
            ),
            'positive': (
                f"{var_name} > 0",
                f"{var_name} must be positive",
                50
            ),
            'non_empty': (
                f"len({var_name}) > 0",
                f"{var_name} must not be empty",
                60
            ),
            'unique': (
                f"len({var_name}) == len(set({var_name}))",
                f"{var_name} must contain unique elements",
                40
            ),
            'sorted': (
                f"{var_name} == sorted({var_name})",
                f"{var_name} must be sorted",
                40
            ),
            'sorted_asc': (
                f"{var_name} == sorted({var_name})",
                f"{var_name} must be sorted in ascending order",
                40
            ),
            'sorted_desc': (
                f"{var_name} == sorted({var_name}, reverse=True)",
                f"{var_name} must be sorted in descending order",
                40
            ),
            'not_null': (
                f"{var_name} is not None",
                f"{var_name} must not be None",
                80
            ),
            'required': (
                f"{var_name} is not None",
                f"{var_name} is required",
                80
            ),
            'optional': (
                "True",  # Optional constraints always pass
                f"{var_name} is optional",
                0
            ),
        }

        # Handle range constraint specially
        if constraint_type == 'range':
            numbers = self._extract_numbers(value)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} <= {var_name} <= {numbers[1]}"
                msg = f"{var_name} must be between {numbers[0]} and {numbers[1]}"
                return Constraint(
                    variable=var_name,
                    constraint_type="range",
                    expression=expr,
                    error_message=msg,
                    priority=50,
                )
            return None

        if constraint_type in expressions:
            expr, msg, priority = expressions[constraint_type]
            return Constraint(
                variable=var_name,
                constraint_type=constraint_type,
                expression=expr,
                error_message=msg,
                priority=priority,
            )

        return None

    def _extract_number(self, text: str) -> str:
        """Extract first number from text."""
        match = re.search(r'-?\d+\.?\d*', text)
        return match.group(0) if match else "0"

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text."""
        return re.findall(r'-?\d+\.?\d*', text)

    def build_dependency_graph(
        self,
        variables: List[ExtractedVariable]
    ) -> Dict[str, List[str]]:
        """
        Build a directed acyclic graph of variable dependencies.

        Args:
            variables: List of extracted variables

        Returns:
            Dictionary mapping variable names to their dependencies
        """
        graph: Dict[str, List[str]] = {}

        for var in variables:
            graph[var.name] = list(var.dependencies)

        self._dependency_graph = graph
        return graph

    def topological_sort(
        self,
        graph: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Return variables in order of computation (dependencies first).

        Uses Kahn's algorithm for topological sorting.

        Args:
            graph: Dependency graph (uses internal if not provided)

        Returns:
            List of variable names in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        if graph is None:
            graph = self._dependency_graph

        if not graph:
            return []

        # Calculate in-degrees
        in_degree: Dict[str, int] = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    # Don't count if we know about this dependency
                    pass
                else:
                    # External dependency, add to graph with 0 in-degree
                    in_degree[dep] = 0
                    graph[dep] = []

        # Count incoming edges
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] = in_degree.get(node, 0)
                    # This node depends on dep, so when we process,
                    # we need dep first

        # Recalculate: in_degree counts how many nodes depend on each node
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in graph:
                    in_degree[node] += 1

        # Start with nodes that have no dependencies
        queue = [node for node, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # For each node that depends on this one
            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(result) != len(graph):
            # Cycle detected
            remaining = set(graph.keys()) - set(result)
            raise ValueError(f"Dependency cycle detected involving: {remaining}")

        return result

    def get_computation_order(
        self,
        variables: List[ExtractedVariable]
    ) -> List[str]:
        """
        Get the order in which variables should be computed.

        Convenience method that builds graph and sorts in one call.

        Args:
            variables: List of extracted variables

        Returns:
            Variable names in computation order (dependencies first)
        """
        graph = self.build_dependency_graph(variables)
        return self.topological_sort(graph)

    def validate(
        self,
        values: Dict[str, Any],
        constraints: Optional[List[Constraint]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate variable values against constraints.

        Args:
            values: Dictionary mapping variable names to their values
            constraints: Constraints to validate (uses internal if not provided)

        Returns:
            Tuple of (all_valid, error_messages)
        """
        if constraints is None:
            constraints = self.constraints

        errors: List[str] = []

        # Sort constraints by priority (highest first)
        sorted_constraints = sorted(
            constraints,
            key=lambda c: c.priority,
            reverse=True
        )

        for constraint in sorted_constraints:
            if constraint.variable not in values:
                continue

            try:
                # Create local namespace with variable values
                local_vars = dict(values)

                # Add safe builtins
                safe_builtins = {
                    'len': len,
                    'set': set,
                    'sorted': sorted,
                    'all': all,
                    'any': any,
                    'isinstance': isinstance,
                    'callable': callable,
                    'int': int,
                    'float': float,
                    'str': str,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'type': type,
                    'None': None,
                    'True': True,
                    'False': False,
                }

                result = eval(
                    constraint.expression,
                    {"__builtins__": safe_builtins},
                    local_vars
                )

                if not result:
                    error_msg = constraint.error_message or f"Constraint failed: {constraint.expression}"
                    errors.append(error_msg)

                    # Stop on first hard constraint failure
                    if constraint.is_hard:
                        break

            except Exception as e:
                errors.append(f"Error evaluating constraint '{constraint.expression}': {e}")

        return len(errors) == 0, errors

    def generate_validation_code(
        self,
        variables: List[ExtractedVariable],
        function_name: str = "validate_inputs"
    ) -> str:
        """
        Generate Python validation function from constraints.

        Args:
            variables: Variables to validate
            function_name: Name for generated function

        Returns:
            Python code as string
        """
        constraints = self.analyze(variables)

        lines = [
            f"def {function_name}(",
            "    " + ", ".join(v.name for v in variables if v.is_input),
            ") -> bool:",
            '    """Validate input parameters."""',
            "    errors = []",
            "",
        ]

        for constraint in sorted(constraints, key=lambda c: c.priority, reverse=True):
            lines.append(f"    # {constraint.constraint_type}: {constraint.variable}")
            lines.append(f"    if not ({constraint.expression}):")
            msg = constraint.error_message or constraint.expression
            lines.append(f'        errors.append("{msg}")')
            lines.append("")

        lines.extend([
            "    if errors:",
            "        raise ValueError(f\"Validation failed: {errors}\")",
            "    return True",
        ])

        return "\n".join(lines)

    def get_constraints_for_variable(self, var_name: str) -> List[Constraint]:
        """Get all constraints for a specific variable."""
        return [c for c in self.constraints if c.variable == var_name]

    def get_hard_constraints(self) -> List[Constraint]:
        """Get only hard (required) constraints."""
        return [c for c in self.constraints if c.is_hard]

    def get_soft_constraints(self) -> List[Constraint]:
        """Get only soft (preference) constraints."""
        return [c for c in self.constraints if not c.is_hard]
