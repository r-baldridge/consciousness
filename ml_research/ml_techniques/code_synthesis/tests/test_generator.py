"""
Tests for Code Generator

Tests the code generation components:
- CodeTemplate and TemplateLibrary
- CodeGenerator
- LLMCodeGenerator
- MultiStepGenerator
- GeneratedCode
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator import (
    CodeTemplate,
    GeneratedCode,
    TemplateLibrary,
    CodeGenerator,
    LLMCodeGenerator,
    MultiStepGenerator,
)
from variable_extractor import ExtractedVariable, VariableType


# =============================================================================
# CODE TEMPLATE TESTS
# =============================================================================


class TestCodeTemplate:
    """Tests for CodeTemplate class."""

    def test_template_creation(self):
        """Test basic template creation."""
        template = CodeTemplate(
            template_id="test_template",
            name="Test Template",
            pattern="def {func_name}():\n    pass",
            parameters=["func_name"],
        )

        assert template.template_id == "test_template"
        assert template.name == "Test Template"
        assert template.language == "python"

    def test_template_render(self):
        """Test template rendering with parameters."""
        template = CodeTemplate(
            template_id="simple_func",
            name="Simple Function",
            pattern="def {func_name}({params}):\n    {body}",
            parameters=["func_name", "params", "body"],
        )

        result = template.render(
            func_name="add",
            params="a, b",
            body="return a + b",
        )

        assert "def add(a, b):" in result
        assert "return a + b" in result

    def test_template_render_missing_params(self):
        """Test template rendering with missing parameters raises error."""
        template = CodeTemplate(
            template_id="test",
            name="Test",
            pattern="{a} + {b}",
            parameters=["a", "b"],
        )

        with pytest.raises(ValueError) as exc_info:
            template.render(a="1")

        assert "Missing required parameters" in str(exc_info.value)
        assert "b" in str(exc_info.value)

    def test_validate_parameters(self):
        """Test parameter validation."""
        template = CodeTemplate(
            template_id="test",
            name="Test",
            pattern="{x} {y}",
            parameters=["x", "y"],
        )

        valid, missing = template.validate_parameters(x="1", y="2")
        assert valid
        assert len(missing) == 0

        valid, missing = template.validate_parameters(x="1")
        assert not valid
        assert "y" in missing


# =============================================================================
# GENERATED CODE TESTS
# =============================================================================


class TestGeneratedCode:
    """Tests for GeneratedCode class."""

    def test_valid_code(self):
        """Test GeneratedCode with valid Python."""
        code = GeneratedCode(
            code="def add(a, b):\n    return a + b",
            language="python",
        )

        assert code.is_valid
        assert len(code.validation_errors) == 0

    def test_invalid_code(self):
        """Test GeneratedCode with invalid Python."""
        code = GeneratedCode(
            code="def add(a, b)\n    return a + b",  # Missing colon
            language="python",
        )

        assert not code.is_valid
        assert len(code.validation_errors) > 0
        assert "SyntaxError" in code.validation_errors[0]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        code = GeneratedCode(
            code="x = 1",
            language="python",
            variables_used=["x"],
            template_used="assignment",
        )

        as_dict = code.to_dict()

        assert as_dict["code"] == "x = 1"
        assert as_dict["language"] == "python"
        assert as_dict["is_valid"] is True
        assert "x" in as_dict["variables_used"]
        assert as_dict["template_used"] == "assignment"


# =============================================================================
# TEMPLATE LIBRARY TESTS
# =============================================================================


class TestTemplateLibrary:
    """Tests for TemplateLibrary class."""

    def test_builtin_templates_exist(self):
        """Test that built-in templates are loaded."""
        library = TemplateLibrary()

        # Check core templates exist
        assert library.get_template("function_def") is not None
        assert library.get_template("class_def") is not None
        assert library.get_template("loop_for") is not None
        assert library.get_template("loop_while") is not None
        assert library.get_template("conditional_if") is not None
        assert library.get_template("try_except") is not None
        assert library.get_template("list_comprehension") is not None
        assert library.get_template("async_function") is not None
        assert library.get_template("decorator") is not None
        assert library.get_template("context_manager") is not None

    def test_list_templates(self):
        """Test listing all templates."""
        library = TemplateLibrary()
        templates = library.list_templates()

        assert len(templates) > 10
        assert "function_def" in templates
        assert "class_def" in templates

    def test_register_custom_template(self):
        """Test registering a custom template."""
        library = TemplateLibrary()

        custom = CodeTemplate(
            template_id="custom_pattern",
            name="Custom Pattern",
            pattern="# Custom: {content}",
            parameters=["content"],
        )

        library.register_template(custom)

        retrieved = library.get_template("custom_pattern")
        assert retrieved is not None
        assert retrieved.name == "Custom Pattern"

    def test_register_duplicate_raises_error(self):
        """Test registering duplicate template raises error."""
        library = TemplateLibrary()

        custom = CodeTemplate(
            template_id="function_def",  # Already exists
            name="Duplicate",
            pattern="test",
            parameters=[],
        )

        with pytest.raises(ValueError):
            library.register_template(custom)

    def test_remove_template(self):
        """Test removing a template."""
        library = TemplateLibrary()

        # Add a custom template
        custom = CodeTemplate(
            template_id="to_remove",
            name="To Remove",
            pattern="test",
            parameters=[],
        )
        library.register_template(custom)

        assert library.get_template("to_remove") is not None

        # Remove it
        result = library.remove_template("to_remove")
        assert result is True
        assert library.get_template("to_remove") is None

        # Try to remove non-existent
        result = library.remove_template("nonexistent")
        assert result is False

    def test_match_pattern_function(self):
        """Test pattern matching for function specs."""
        library = TemplateLibrary()

        matches = library.match_pattern(
            "Create a function that takes two numbers and returns their sum"
        )

        assert len(matches) > 0
        template_ids = [m.template_id for m in matches]
        assert any("function" in tid for tid in template_ids)

    def test_match_pattern_loop(self):
        """Test pattern matching for loop specs."""
        library = TemplateLibrary()

        matches = library.match_pattern(
            "Iterate through each item in the list"
        )

        assert len(matches) > 0
        template_ids = [m.template_id for m in matches]
        assert any("loop" in tid or "for" in tid for tid in template_ids)

    def test_match_pattern_class(self):
        """Test pattern matching for class specs."""
        library = TemplateLibrary()

        matches = library.match_pattern(
            "Create a class to represent a user with name and email"
        )

        assert len(matches) > 0
        template_ids = [m.template_id for m in matches]
        assert any("class" in tid for tid in template_ids)

    def test_match_pattern_exception(self):
        """Test pattern matching for exception handling."""
        library = TemplateLibrary()

        matches = library.match_pattern(
            "Handle the error if the file cannot be opened"
        )

        assert len(matches) > 0
        template_ids = [m.template_id for m in matches]
        assert any("except" in tid or "try" in tid for tid in template_ids)


# =============================================================================
# CODE GENERATOR TESTS
# =============================================================================


class TestCodeGenerator:
    """Tests for CodeGenerator class."""

    def test_generate_from_spec_simple(self):
        """Test generating code from simple specification."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that takes two numbers and returns their sum"
        )

        assert isinstance(result, GeneratedCode)
        assert result.is_valid
        assert "def " in result.code
        assert "return" in result.code

    def test_generate_from_spec_with_constraints(self):
        """Test generating code with constraints in spec."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that takes a positive integer and returns its factorial"
        )

        assert result.is_valid
        assert "def " in result.code
        # Should include some form of constraint or assertion
        assert "assert" in result.code or "if" in result.code or "positive" in result.code.lower()

    def test_generate_from_variables(self):
        """Test generating code from extracted variables."""
        generator = CodeGenerator()

        variables = [
            ExtractedVariable(
                name="numbers",
                var_type=VariableType.LIST,
                description="List of numbers to sum",
                is_input=True,
                element_type=VariableType.INTEGER,
            ),
            ExtractedVariable(
                name="total",
                var_type=VariableType.INTEGER,
                description="Sum of all numbers",
                is_output=True,
            ),
        ]

        code = generator.generate_from_variables(variables, "sum_numbers")

        assert "def sum_numbers" in code
        assert "numbers" in code
        assert "List" in code
        assert "return" in code

    def test_generate_from_template(self):
        """Test generating code from template."""
        generator = CodeGenerator()

        code = generator.generate_from_template(
            "function_simple",
            func_name="greet",
            params="name",
            body='return f"Hello, {name}!"',
        )

        assert "def greet(name):" in code
        assert 'return f"Hello, {name}!"' in code

    def test_generate_from_template_not_found(self):
        """Test error when template not found."""
        generator = CodeGenerator()

        with pytest.raises(ValueError) as exc_info:
            generator.generate_from_template("nonexistent_template")

        assert "not found" in str(exc_info.value)

    def test_syntax_validation(self):
        """Test syntax validation."""
        generator = CodeGenerator()

        valid, error = generator._validate_syntax("def foo(): pass")
        assert valid
        assert error is None

        valid, error = generator._validate_syntax("def foo() pass")  # Missing colon
        assert not valid
        assert "SyntaxError" in error

    def test_generated_code_includes_metadata(self):
        """Test that generated code includes metadata."""
        generator = CodeGenerator()
        result = generator.generate_from_spec("Add two numbers")

        assert "spec" in result.metadata
        assert "num_variables" in result.metadata


# =============================================================================
# LLM CODE GENERATOR TESTS
# =============================================================================


class MockLLMBackend:
    """Mock LLM backend for testing."""

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate mock response."""
        if "add" in prompt.lower() or "sum" in prompt.lower():
            return """```python
def add(a, b):
    return a + b
```"""
        elif "sort" in prompt.lower():
            return """```python
def sort_list(items):
    return sorted(items)
```"""
        else:
            return """```python
def process():
    pass
```"""


class TestLLMCodeGenerator:
    """Tests for LLMCodeGenerator class."""

    def test_generate_basic(self):
        """Test basic generation with mock backend."""
        backend = MockLLMBackend()
        generator = LLMCodeGenerator(backend=backend)

        code = generator.generate("Create a function to add two numbers")

        assert "def add" in code
        assert "return a + b" in code

    def test_generate_with_context(self):
        """Test generation with context."""
        backend = MockLLMBackend()
        generator = LLMCodeGenerator(backend=backend)

        code = generator.generate(
            "Create a function to sum values",
            context={"input_types": {"a": "int", "b": "int"}, "output_type": "int"},
        )

        assert "def " in code

    def test_generate_with_examples(self):
        """Test few-shot generation with examples."""
        backend = MockLLMBackend()
        generator = LLMCodeGenerator(backend=backend)

        examples = [
            ("Double a number", "def double(x):\n    return x * 2"),
            ("Triple a number", "def triple(x):\n    return x * 3"),
        ]

        code = generator.generate_with_examples(
            "Add two numbers",
            examples=examples,
        )

        assert "def " in code

    def test_complete_partial(self):
        """Test completing partial code."""
        backend = MockLLMBackend()
        generator = LLMCodeGenerator(backend=backend)

        partial = "def calculate_sum("
        code = generator.complete_partial(partial, "Sum two numbers")

        # Should include the partial code
        assert "calculate_sum" in code or "add" in code

    def test_extract_code_from_markdown(self):
        """Test code extraction from markdown blocks."""
        generator = LLMCodeGenerator(backend=MockLLMBackend())

        response = """Here's the code:
```python
def foo():
    return 42
```
That's all!"""

        code = generator._extract_code(response)

        assert "def foo():" in code
        assert "return 42" in code
        assert "Here's the code" not in code


# =============================================================================
# MULTI-STEP GENERATOR TESTS
# =============================================================================


class TestMultiStepGenerator:
    """Tests for MultiStepGenerator class."""

    def test_generate_basic(self):
        """Test basic multi-step generation."""
        generator = MultiStepGenerator()
        result = generator.generate("Add two numbers and return the sum")

        assert isinstance(result, GeneratedCode)
        assert result.is_valid
        assert "def " in result.code
        assert '"""' in result.code  # Has docstring

    def test_generate_with_constraints(self):
        """Test generation with constraints."""
        generator = MultiStepGenerator()
        result = generator.generate(
            "Takes a positive integer greater than 0 and returns its square"
        )

        assert result.is_valid
        assert "def " in result.code
        # Should have some constraint handling
        assert "assert" in result.code or "positive" in result.code.lower()

    def test_step1_signature(self):
        """Test signature generation step."""
        generator = MultiStepGenerator()

        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="Input number",
                is_input=True,
            ),
            ExtractedVariable(
                name="result",
                var_type=VariableType.INTEGER,
                description="Squared value",
                is_output=True,
            ),
        ]

        signature = generator._step1_generate_signature(variables, "Square a number")

        assert "def " in signature
        assert "x" in signature
        assert "int" in signature
        assert "->" in signature

    def test_step2_docstring(self):
        """Test docstring generation step."""
        generator = MultiStepGenerator()

        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="The number to square",
                is_input=True,
            ),
            ExtractedVariable(
                name="result",
                var_type=VariableType.INTEGER,
                description="The squared value",
                is_output=True,
            ),
        ]

        docstring = generator._step2_generate_docstring(variables, "Square a number")

        assert '"""' in docstring
        assert "Args:" in docstring
        assert "x" in docstring
        assert "Returns:" in docstring

    def test_generation_metadata(self):
        """Test that generation includes method metadata."""
        generator = MultiStepGenerator()
        result = generator.generate("Simple function")

        assert result.metadata["generation_method"] == "multi_step"
        assert result.metadata["steps_completed"] == 4


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_simple_function(self):
        """Test full generation pipeline for simple function."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that takes a string and returns it in uppercase"
        )

        assert result.is_valid
        assert "def " in result.code
        assert "str" in result.code.lower() or "string" in result.code.lower()

    def test_full_pipeline_list_operation(self):
        """Test full generation pipeline for list operation."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that takes a list of integers and returns the sum of all even numbers"
        )

        assert result.is_valid
        assert "def " in result.code
        assert "list" in result.code.lower() or "List" in result.code

    def test_template_and_generator_consistency(self):
        """Test that template library and generator work together."""
        library = TemplateLibrary()
        generator = CodeGenerator(template_library=library)

        # Generate using spec that should match function template
        result = generator.generate_from_spec("Function to calculate area")

        assert result.is_valid
        # Template should be used when matched
        assert result.template_used is not None or "def " in result.code

    def test_custom_template_integration(self):
        """Test using custom template with generator."""
        library = TemplateLibrary()

        # Register custom template
        library.register_template(CodeTemplate(
            template_id="api_endpoint",
            name="API Endpoint",
            pattern='@app.route("{route}")\ndef {func_name}():\n    {body}',
            parameters=["route", "func_name", "body"],
        ))

        generator = CodeGenerator(template_library=library)

        code = generator.generate_from_template(
            "api_endpoint",
            route="/users",
            func_name="get_users",
            body="return []",
        )

        assert '@app.route("/users")' in code
        assert "def get_users" in code


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_specification(self):
        """Test handling empty specification."""
        generator = CodeGenerator()
        result = generator.generate_from_spec("")

        # Should still produce some code (may be minimal)
        assert result is not None
        assert isinstance(result, GeneratedCode)

    def test_very_long_specification(self):
        """Test handling very long specification."""
        generator = CodeGenerator()
        long_spec = "Create a function that " + "processes data " * 100 + "and returns result"
        result = generator.generate_from_spec(long_spec)

        assert result is not None
        assert isinstance(result, GeneratedCode)

    def test_special_characters_in_spec(self):
        """Test handling special characters in specification."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that handles strings with quotes \" and 'apostrophes'"
        )

        assert result is not None
        assert isinstance(result, GeneratedCode)

    def test_unicode_in_specification(self):
        """Test handling unicode in specification."""
        generator = CodeGenerator()
        result = generator.generate_from_spec(
            "Create a function that handles unicode characters like alpha beta gamma"
        )

        assert result is not None
        assert isinstance(result, GeneratedCode)

    def test_no_variables_extracted(self):
        """Test handling spec with no extractable variables."""
        generator = CodeGenerator()
        result = generator.generate_from_spec("Do something")

        assert result is not None
        assert "def " in result.code

    def test_multiple_outputs(self):
        """Test handling multiple output variables."""
        generator = CodeGenerator()

        variables = [
            ExtractedVariable(
                name="x",
                var_type=VariableType.INTEGER,
                description="Input",
                is_input=True,
            ),
            ExtractedVariable(
                name="quotient",
                var_type=VariableType.INTEGER,
                description="Quotient",
                is_output=True,
            ),
            ExtractedVariable(
                name="remainder",
                var_type=VariableType.INTEGER,
                description="Remainder",
                is_output=True,
            ),
        ]

        code = generator.generate_from_variables(variables, "divmod_custom")

        assert "def divmod_custom" in code
        assert "Tuple" in code
        assert "return" in code


# =============================================================================
# TEMPLATE RENDERING TESTS
# =============================================================================


class TestTemplateRendering:
    """Tests for specific template rendering."""

    def test_function_def_template(self):
        """Test function definition template."""
        library = TemplateLibrary()
        template = library.get_template("function_def")

        code = template.render(
            func_name="calculate",
            params="x: int, y: int",
            docstring="Calculate something",
            body="return x + y",
        )

        assert "def calculate(x: int, y: int):" in code
        assert '"""Calculate something"""' in code
        assert "return x + y" in code

    def test_class_simple_template(self):
        """Test simple class template."""
        library = TemplateLibrary()
        template = library.get_template("class_simple")

        code = template.render(
            class_name="Person",
            init_params=", name, age",
            init_body="self.name = name\n            self.age = age",
        )

        assert "class Person:" in code
        assert "def __init__" in code
        assert "self.name = name" in code

    def test_try_except_template(self):
        """Test try-except template."""
        library = TemplateLibrary()
        template = library.get_template("try_except")

        code = template.render(
            try_body="result = risky_operation()",
            exception_type="ValueError",
            exception_var="e",
            except_body='print(f"Error: {e}")',
        )

        assert "try:" in code
        assert "except ValueError as e:" in code

    def test_list_comprehension_conditional_template(self):
        """Test conditional list comprehension template."""
        library = TemplateLibrary()
        template = library.get_template("list_comprehension_conditional")

        code = template.render(
            expression="x * 2",
            variable="x",
            iterable="numbers",
            condition="x > 0",
        )

        assert "[x * 2 for x in numbers if x > 0]" in code

    def test_async_function_template(self):
        """Test async function template."""
        library = TemplateLibrary()
        template = library.get_template("async_function")

        code = template.render(
            func_name="fetch_data",
            params="url: str",
            docstring="Fetch data from URL",
            body="response = await client.get(url)\n        return response",
        )

        assert "async def fetch_data" in code
        assert "await" in code

    def test_context_manager_template(self):
        """Test context manager class template."""
        library = TemplateLibrary()
        template = library.get_template("context_manager")

        code = template.render(
            class_name="FileHandler",
            init_params=", filename",
            init_body="self.filename = filename",
            enter_body="self.file = open(self.filename)",
            exit_body="self.file.close()",
        )

        assert "class FileHandler:" in code
        assert "__enter__" in code
        assert "__exit__" in code


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
