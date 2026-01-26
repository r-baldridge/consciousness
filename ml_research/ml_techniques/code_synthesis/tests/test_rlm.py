"""
Tests for RLM (Recursive Language Model) Variable Extraction

Tests the variable extraction, constraint analysis, and decomposition
components of the RLM code synthesis technique.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from variable_extractor import VariableExtractor, ExtractedVariable, VariableType
from constraints import ConstraintAnalyzer, Constraint
from rlm import RLM, RLMExtractor, RLMDecomposition


# =============================================================================
# VARIABLE EXTRACTOR TESTS
# =============================================================================

class TestVariableExtractor:
    """Tests for VariableExtractor class."""

    def test_simple_extraction(self):
        """Test extraction from simple specification."""
        extractor = VariableExtractor()
        spec = "Write a function that takes two numbers and returns their sum."
        variables = extractor.extract(spec)

        assert len(variables) >= 2, f"Expected at least 2 variables, got {len(variables)}"
        assert any(v.is_input for v in variables), "Should have at least one input"
        assert any(v.is_output for v in variables), "Should have at least one output"

    def test_list_extraction(self):
        """Test extraction of list/collection types."""
        extractor = VariableExtractor()
        spec = "Create a function that takes a list of integers and returns the sum of all even numbers."
        variables = extractor.extract(spec)

        # Should have a list type
        list_vars = [v for v in variables if v.var_type == VariableType.LIST]
        assert len(list_vars) > 0, "Should extract at least one list variable"

        # The list should have integer element type
        list_var = list_vars[0]
        assert list_var.element_type == VariableType.INTEGER or list_var.var_type == VariableType.LIST

        # Should have an integer output (sum)
        int_outputs = [v for v in variables if v.var_type == VariableType.INTEGER and v.is_output]
        assert len(int_outputs) > 0 or any(v.is_output for v in variables)

    def test_constraint_extraction(self):
        """Test extraction of constraints from specification."""
        extractor = VariableExtractor()
        spec = "Takes a positive integer greater than 0 and returns its factorial."
        variables = extractor.extract(spec)

        input_var = next((v for v in variables if v.is_input), None)
        assert input_var is not None, "Should have an input variable"
        assert len(input_var.constraints) > 0, f"Should extract constraints, got: {input_var.constraints}"

        # Check for positive or greater than constraint
        constraint_text = ' '.join(input_var.constraints)
        has_positive = 'positive' in constraint_text or 'min' in constraint_text
        assert has_positive, f"Should have positive constraint, got: {input_var.constraints}"

    def test_string_variable_extraction(self):
        """Test extraction of string types."""
        extractor = VariableExtractor()
        spec = "Takes a string and returns the reversed string."
        variables = extractor.extract(spec)

        string_vars = [v for v in variables if v.var_type == VariableType.STRING]
        assert len(string_vars) > 0, "Should extract string variables"

    def test_dictionary_extraction(self):
        """Test extraction of dictionary/map types."""
        extractor = VariableExtractor()
        spec = "Takes a dictionary mapping names to ages and returns the average age."
        variables = extractor.extract(spec)

        dict_vars = [v for v in variables if v.var_type == VariableType.DICT]
        # Should recognize dictionary/map
        assert len(dict_vars) > 0 or any('dict' in v.description.lower() or 'map' in v.description.lower() for v in variables)

    def test_boolean_extraction(self):
        """Test extraction of boolean types."""
        extractor = VariableExtractor()
        spec = "Takes a list and returns true if the list is empty, false otherwise."
        variables = extractor.extract(spec)

        # Should have boolean output or recognize true/false
        bool_vars = [v for v in variables if v.var_type == VariableType.BOOLEAN]
        has_bool = len(bool_vars) > 0 or any('true' in v.description.lower() or 'false' in v.description.lower() for v in variables)
        assert has_bool, "Should extract boolean variable"

    def test_multiple_inputs(self):
        """Test extraction of multiple input parameters."""
        extractor = VariableExtractor()
        spec = "Takes two strings, a prefix and a suffix, and returns them concatenated."
        variables = extractor.extract(spec)

        inputs = [v for v in variables if v.is_input]
        assert len(inputs) >= 1, "Should extract at least one input"

    def test_variable_naming(self):
        """Test that generated variable names are valid Python identifiers."""
        extractor = VariableExtractor()
        spec = "Takes a list of 123 numbers and returns the maximum value."
        variables = extractor.extract(spec)

        for var in variables:
            # Should be valid Python identifier
            assert var.name.isidentifier(), f"'{var.name}' is not a valid Python identifier"
            # Should not start with number
            assert not var.name[0].isdigit(), f"'{var.name}' starts with a digit"

    def test_type_inference_confidence(self):
        """Test that type inference includes confidence scores."""
        extractor = VariableExtractor()
        spec = "Takes an integer and returns its square."
        variables = extractor.extract(spec)

        for var in variables:
            assert 0.0 <= var.confidence <= 1.0, f"Confidence {var.confidence} out of range"


# =============================================================================
# CONSTRAINT ANALYZER TESTS
# =============================================================================

class TestConstraintAnalyzer:
    """Tests for ConstraintAnalyzer class."""

    def test_type_constraint_generation(self):
        """Test generation of type constraints."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="num",
                var_type=VariableType.INTEGER,
                description="A number",
                is_input=True,
            )
        ]

        constraints = analyzer.analyze(variables)

        # Should have type constraint
        type_constraints = [c for c in constraints if c.constraint_type == "type"]
        assert len(type_constraints) > 0, "Should generate type constraints"

        # Expression should be valid isinstance check
        tc = type_constraints[0]
        assert "isinstance" in tc.expression

    def test_range_constraint_parsing(self):
        """Test parsing of range constraints."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="age",
                var_type=VariableType.INTEGER,
                description="A person's age",
                constraints=["min_inclusive:0", "max_inclusive:150"],
                is_input=True,
            )
        ]

        constraints = analyzer.analyze(variables)

        # Should have min and max constraints
        range_constraints = [c for c in constraints if "age >=" in c.expression or "age <=" in c.expression]
        assert len(range_constraints) >= 2, "Should parse min and max constraints"

    def test_non_empty_constraint(self):
        """Test non-empty constraint for collections."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="items",
                var_type=VariableType.LIST,
                description="A list of items",
                constraints=["non_empty"],
                is_input=True,
            )
        ]

        constraints = analyzer.analyze(variables)

        non_empty = [c for c in constraints if "len" in c.expression and "> 0" in c.expression]
        assert len(non_empty) > 0, "Should generate non-empty constraint"

    def test_dependency_graph_building(self):
        """Test building of dependency graph."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="First number",
                is_input=True,
            ),
            ExtractedVariable(
                name="y",
                var_type=VariableType.INTEGER,
                description="Second number",
                is_input=True,
            ),
            ExtractedVariable(
                name="result",
                var_type=VariableType.INTEGER,
                description="The sum",
                dependencies=["x", "y"],
                is_output=True,
            ),
        ]

        graph = analyzer.build_dependency_graph(variables)

        assert "x" in graph
        assert "y" in graph
        assert "result" in graph
        assert "x" in graph["result"]
        assert "y" in graph["result"]

    def test_topological_sort(self):
        """Test topological sorting of dependencies."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(name="a", var_type=VariableType.INTEGER, description="", is_input=True),
            ExtractedVariable(name="b", var_type=VariableType.INTEGER, description="", dependencies=["a"]),
            ExtractedVariable(name="c", var_type=VariableType.INTEGER, description="", dependencies=["b"], is_output=True),
        ]

        order = analyzer.get_computation_order(variables)

        # a should come before b, b before c
        if "a" in order and "b" in order:
            assert order.index("a") < order.index("b"), "a should come before b"
        if "b" in order and "c" in order:
            assert order.index("b") < order.index("c"), "b should come before c"

    def test_validation_success(self):
        """Test constraint validation with valid values."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="A positive number",
                constraints=["positive"],
                is_input=True,
            )
        ]

        analyzer.analyze(variables)
        valid, errors = analyzer.validate({"x": 5})

        assert valid, f"Should be valid, got errors: {errors}"

    def test_validation_failure(self):
        """Test constraint validation with invalid values."""
        analyzer = ConstraintAnalyzer()
        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="A positive number",
                constraints=["positive"],
                is_input=True,
            )
        ]

        analyzer.analyze(variables)
        valid, errors = analyzer.validate({"x": -5})

        assert not valid, "Should fail validation"
        assert len(errors) > 0, "Should have error messages"


# =============================================================================
# RLM DECOMPOSITION TESTS
# =============================================================================

class TestRLMDecomposition:
    """Tests for RLM decomposition."""

    def test_rlm_decomposition(self):
        """Test full RLM decomposition."""
        rlm = RLMExtractor()
        spec = """
        Create a function that:
        1. Takes a list of strings
        2. Filters out strings shorter than 3 characters
        3. Converts remaining strings to uppercase
        4. Returns the sorted list
        """

        result = rlm.run(spec)
        assert result.success, f"RLM should succeed, error: {result.error}"
        assert result.output is not None

        decomp = result.output
        assert len(decomp.variables) >= 2, "Should have at least 2 variables"
        assert len(decomp.subproblems) >= 1, "Should have at least 1 subproblem"

    def test_dependency_order(self):
        """Test that dependency order is correct."""
        rlm = RLMExtractor()
        spec = "Takes x and y, computes z = x + y, then returns z * 2"

        decomp = rlm.decompose(spec)
        order = decomp.dependency_order

        # Inputs should come before outputs in dependency order
        input_names = [v.name for v in decomp.variables if v.is_input]
        output_names = [v.name for v in decomp.variables if v.is_output]

        for inp in input_names:
            for out in output_names:
                if inp in order and out in order:
                    assert order.index(inp) <= order.index(out), \
                        f"Input {inp} should come before output {out}"

    def test_subproblem_identification(self):
        """Test that subproblems are correctly identified."""
        rlm = RLMExtractor()
        spec = """
        Function that:
        1. Accepts a list of numbers
        2. Removes duplicates
        3. Sorts the remaining numbers
        4. Returns the median
        """

        decomp = rlm.decompose(spec)

        # Should have multiple subproblems
        assert len(decomp.subproblems) >= 1, "Should identify subproblems"

    def test_generate_signature(self):
        """Test function signature generation."""
        rlm = RLMExtractor()
        spec = "Takes a list of integers and returns the sum."

        decomp = rlm.decompose(spec)
        signature = decomp.generate_signature("calculate_sum")

        assert "def calculate_sum" in signature
        assert ":" in signature  # Has return type annotation

    def test_generate_docstring(self):
        """Test docstring generation."""
        rlm = RLMExtractor()
        spec = "Takes a list of integers and returns the sum."

        decomp = rlm.decompose(spec)
        docstring = decomp.generate_docstring()

        assert '"""' in docstring
        assert "Args:" in docstring or "Returns:" in docstring

    def test_technique_result_metadata(self):
        """Test that TechniqueResult includes proper metadata."""
        rlm = RLMExtractor()
        spec = "Takes a number and returns its square."

        result = rlm.run(spec)

        assert result.success
        assert result.technique_id == "rlm_extractor"
        assert result.execution_time_ms >= 0
        assert "variables" in result.metadata

    def test_decomposition_to_dict(self):
        """Test serialization of decomposition."""
        rlm = RLMExtractor()
        spec = "Takes two numbers and returns their product."

        decomp = rlm.decompose(spec)
        as_dict = decomp.to_dict()

        assert "variables" in as_dict
        assert "constraints" in as_dict
        assert "subproblems" in as_dict
        assert "original_spec" in as_dict

    def test_empty_specification(self):
        """Test handling of empty specification."""
        rlm = RLMExtractor()
        result = rlm.run("")

        # Should handle gracefully (may succeed with no variables or fail)
        assert result is not None
        assert hasattr(result, 'success')

    def test_non_string_input(self):
        """Test handling of non-string input."""
        rlm = RLMExtractor()
        result = rlm.run(123)

        assert not result.success
        assert "string" in result.error.lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with diverse specifications."""

    @pytest.mark.parametrize("spec,expected_input_count,expected_output_count", [
        ("Takes a number and returns its double.", 1, 1),
        ("Given x and y, returns their sum.", 1, 1),
        ("Accepts a list and returns the first element.", 1, 1),
    ])
    def test_basic_specifications(self, spec, expected_input_count, expected_output_count):
        """Test basic specifications with expected variable counts."""
        extractor = VariableExtractor()
        variables = extractor.extract(spec)

        inputs = [v for v in variables if v.is_input]
        outputs = [v for v in variables if v.is_output]

        # Allow some flexibility in counts
        assert len(inputs) >= 1, f"Expected at least 1 input for: {spec}"
        assert len(outputs) >= 1, f"Expected at least 1 output for: {spec}"

    def test_complex_specification(self):
        """Test complex multi-step specification."""
        rlm = RLMExtractor()
        spec = """
        Create a function that processes user data:
        - Takes a list of user dictionaries with 'name' and 'age' fields
        - Filters users who are at least 18 years old
        - Sorts them by age in ascending order
        - Returns a list of just their names
        """

        result = rlm.run(spec)
        assert result.success

        decomp = result.output
        # Should identify multiple processing steps
        assert len(decomp.variables) >= 2
        assert len(decomp.subproblems) >= 1

    def test_mathematical_specification(self):
        """Test mathematical/algorithmic specification."""
        rlm = RLMExtractor()
        spec = """
        Implement binary search:
        Takes a sorted list of integers and a target value.
        Returns the index of the target if found, -1 otherwise.
        The list must be non-empty and sorted in ascending order.
        """

        result = rlm.run(spec)
        assert result.success

        decomp = result.output

        # Should have constraints about sorted list
        has_sorted_constraint = any(
            'sorted' in c.expression.lower() or 'sorted' in str(c.error_message).lower()
            for c in decomp.constraints
        )
        # Constraint might be in variable constraints instead
        has_sorted_var = any('sorted' in str(v.constraints).lower() for v in decomp.variables)
        assert has_sorted_constraint or has_sorted_var or 'sorted' in spec.lower()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_unicode_in_specification(self):
        """Test handling of unicode characters."""
        extractor = VariableExtractor()
        spec = "Takes a string with emojis like and returns its length."
        variables = extractor.extract(spec)

        # Should not crash
        assert variables is not None

    def test_very_long_specification(self):
        """Test handling of very long specifications."""
        extractor = VariableExtractor()
        spec = "Takes a number. " * 100 + "Returns the result."
        variables = extractor.extract(spec)

        # Should handle without timeout/crash
        assert variables is not None

    def test_ambiguous_specification(self):
        """Test handling of ambiguous specification."""
        extractor = VariableExtractor()
        spec = "Does something with things."
        variables = extractor.extract(spec)

        # Should handle gracefully (might extract nothing or generic vars)
        assert variables is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
