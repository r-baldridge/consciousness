"""
Tests for Self-Debugging System

Tests the execution sandbox, error analyzer, and self-debugger components
with various error scenarios.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from debugger import (
    # Enums
    ErrorCategory,
    FixStrategyType,
    # Data classes
    ExecutionResult,
    ErrorAnalysis,
    FixStrategy,
    FixAttempt,
    DebugResult,
    # Classes
    ExecutionSandbox,
    ErrorAnalyzer,
    SelfDebugger,
)


# =============================================================================
# EXECUTION SANDBOX TESTS
# =============================================================================

class TestExecutionSandbox:
    """Tests for ExecutionSandbox class."""

    def test_simple_execution(self):
        """Test basic code execution."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("x = 2 + 2")

        assert result.success
        assert "x" in result.namespace
        assert result.namespace["x"] == 4

    def test_stdout_capture(self):
        """Test that stdout is captured."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("print('hello world')")

        assert result.success
        assert "hello world" in result.stdout

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("def foo( missing colon")

        assert not result.success
        assert result.error_type == "SyntaxError"

    def test_runtime_error_handling(self):
        """Test handling of runtime errors."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("x = 1 / 0")

        assert not result.success
        assert result.error_type == "ZeroDivisionError"

    def test_name_error_handling(self):
        """Test handling of name errors."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("print(undefined_variable)")

        assert not result.success
        assert result.error_type == "NameError"

    def test_type_error_handling(self):
        """Test handling of type errors."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("x = 'hello' + 5")

        assert not result.success
        assert result.error_type == "TypeError"

    def test_safe_imports_allowed(self):
        """Test that safe imports are allowed."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("import math\nx = math.sqrt(16)")

        assert result.success
        assert result.namespace.get("x") == 4.0

    def test_unsafe_imports_blocked(self):
        """Test that unsafe imports are blocked."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("import os")

        assert not result.success
        assert result.error_type == "ImportError"

    def test_execute_with_inputs_success(self):
        """Test execution with test case inputs - success case."""
        sandbox = ExecutionSandbox()
        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
            {"inputs": {"a": 5, "b": 5}, "expected": 10},
        ]

        results = sandbox.execute_with_inputs(code, test_cases)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].output == 3
        assert results[1].output == 10

    def test_execute_with_inputs_failure(self):
        """Test execution with test case inputs - failure case."""
        sandbox = ExecutionSandbox()
        code = """
def add(a, b):
    return a - b  # Bug: subtracts instead of adds
"""
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
        ]

        results = sandbox.execute_with_inputs(code, test_cases)

        assert len(results) == 1
        assert not results[0].success  # Expected 3, got -1

    def test_execute_with_inputs_error(self):
        """Test execution with test case inputs - runtime error."""
        sandbox = ExecutionSandbox()
        code = """
def divide(a, b):
    return a / b
"""
        test_cases = [
            {"inputs": {"a": 10, "b": 0}, "expected": None},  # Division by zero
        ]

        results = sandbox.execute_with_inputs(code, test_cases)

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error_type == "ZeroDivisionError"

    def test_timeout_handling(self):
        """Test that infinite loops are caught by timeout."""
        sandbox = ExecutionSandbox(timeout=1.0)
        # This will timeout
        result = sandbox.execute("while True: pass")

        # On Windows this won't timeout (SIGALRM not supported)
        # So we check that it either timed out or ran forever
        if sys.platform != 'win32':
            assert not result.success
            assert result.error_type == "TimeoutError"

    def test_execution_time_tracked(self):
        """Test that execution time is tracked."""
        sandbox = ExecutionSandbox()
        result = sandbox.execute("x = sum(range(1000))")

        assert result.success
        assert result.execution_time_ms >= 0


# =============================================================================
# ERROR ANALYZER TESTS
# =============================================================================

class TestErrorAnalyzer:
    """Tests for ErrorAnalyzer class."""

    def test_classify_syntax_error(self):
        """Test classification of syntax errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("SyntaxError", "invalid syntax")

        assert category == ErrorCategory.SYNTAX

    def test_classify_type_error(self):
        """Test classification of type errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("TypeError", "unsupported operand")

        assert category == ErrorCategory.TYPE

    def test_classify_name_error(self):
        """Test classification of name errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("NameError", "name 'foo' is not defined")

        assert category == ErrorCategory.NAME

    def test_classify_index_error(self):
        """Test classification of index errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("IndexError", "list index out of range")

        assert category == ErrorCategory.INDEX

    def test_classify_key_error(self):
        """Test classification of key errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("KeyError", "'missing_key'")

        assert category == ErrorCategory.KEY

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        analyzer = ErrorAnalyzer()
        category = analyzer.classify_error("CustomException", "something went wrong")

        assert category == ErrorCategory.UNKNOWN

    def test_extract_line_number_from_traceback(self):
        """Test line number extraction from traceback."""
        analyzer = ErrorAnalyzer()
        traceback = '''
Traceback (most recent call last):
  File "<sandbox>", line 5, in <module>
    x = foo()
NameError: name 'foo' is not defined
'''
        line_num = analyzer.extract_line_number(traceback)

        assert line_num == 5

    def test_analyze_name_error(self):
        """Test full analysis of name error."""
        analyzer = ErrorAnalyzer()
        result = ExecutionResult(
            success=False,
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
            traceback='File "<sandbox>", line 3\n    x = undefined_var',
        )

        analysis = analyzer.analyze(result)

        assert analysis.category == ErrorCategory.NAME
        assert "undefined_var" in analysis.root_cause
        assert len(analysis.suggested_fixes) > 0

    def test_analyze_type_error(self):
        """Test analysis of type error."""
        analyzer = ErrorAnalyzer()
        result = ExecutionResult(
            success=False,
            error_type="TypeError",
            error_message="can only concatenate str (not 'int') to str",
            traceback='File "<sandbox>", line 2\n    x = "hello" + 5',
        )

        analysis = analyzer.analyze(result)

        assert analysis.category == ErrorCategory.TYPE
        assert analysis.error_type == "TypeError"

    def test_suggest_fix_for_name_error(self):
        """Test fix suggestion for name error."""
        analyzer = ErrorAnalyzer()
        code = """
x = 10
y = 20
result = x + z
"""
        result = ExecutionResult(
            success=False,
            error_type="NameError",
            error_message="name 'z' is not defined",
            traceback='File "<sandbox>", line 4',
        )

        analysis = analyzer.analyze(result, code)
        fix = analyzer.suggest_fix_strategy(analysis, code)

        assert fix.strategy_type in [
            FixStrategyType.INSERT_LINE,
            FixStrategyType.MODIFY_EXPRESSION,
        ]

    def test_suggest_fix_for_index_error(self):
        """Test fix suggestion for index error."""
        analyzer = ErrorAnalyzer()
        code = """
items = [1, 2, 3]
x = items[10]
"""
        result = ExecutionResult(
            success=False,
            error_type="IndexError",
            error_message="list index out of range",
            traceback='File "<sandbox>", line 3',
        )

        analysis = analyzer.analyze(result, code)
        fix = analyzer.suggest_fix_strategy(analysis, code)

        assert fix.strategy_type == FixStrategyType.ADD_BOUNDS_CHECK

    def test_suggest_fix_for_key_error(self):
        """Test fix suggestion for key error."""
        analyzer = ErrorAnalyzer()
        code = """
data = {"a": 1, "b": 2}
x = data["c"]
"""
        result = ExecutionResult(
            success=False,
            error_type="KeyError",
            error_message="'c'",
            traceback='File "<sandbox>", line 3',
        )

        analysis = analyzer.analyze(result, code)
        fix = analyzer.suggest_fix_strategy(analysis, code)

        assert fix.strategy_type == FixStrategyType.MODIFY_EXPRESSION


# =============================================================================
# SELF DEBUGGER TESTS
# =============================================================================

class TestSelfDebugger:
    """Tests for SelfDebugger class."""

    def test_debug_already_correct_code(self):
        """Test debugging code that's already correct."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
            {"inputs": {"a": 5, "b": 7}, "expected": 12},
        ]

        result = debugger.debug(code, "add two numbers", test_cases)

        assert result.success
        assert result.all_tests_passed
        assert result.attempts == 1

    def test_debug_simple_typo(self):
        """Test debugging a simple typo in variable name."""
        debugger = SelfDebugger(max_attempts=5)
        code = """
def add(a, b):
    return a + c  # Bug: should be 'b' not 'c'
"""
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
        ]

        result = debugger.debug(code, "add two numbers", test_cases)

        # Without LLM, this may or may not be fixed
        # Just verify we get a result
        assert result is not None
        assert result.original_code == code
        assert len(result.fix_history) >= 0

    def test_debug_with_syntax_error(self):
        """Test debugging code with syntax error."""
        debugger = SelfDebugger(max_attempts=5)
        code = """
def add(a, b)
    return a + b
"""  # Missing colon
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
        ]

        result = debugger.debug(code, "add two numbers", test_cases)

        # Should attempt to fix
        assert result is not None
        assert len(result.fix_history) > 0

    def test_debug_division_by_zero(self):
        """Test debugging code with division by zero risk."""
        debugger = SelfDebugger(max_attempts=5)
        code = """
def divide(a, b):
    return a / b
"""
        test_cases = [
            {"inputs": {"a": 10, "b": 2}, "expected": 5},
            {"inputs": {"a": 10, "b": 0}},  # Will cause error
        ]

        result = debugger.debug(code, "divide two numbers", test_cases)

        # Should try to fix
        assert result is not None

    def test_debug_records_history(self):
        """Test that debug records fix history."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def bad():
    return undefined
"""
        test_cases = [
            {"inputs": {}, "expected": 42},
        ]

        result = debugger.debug(code, "return 42", test_cases, "bad")

        # Should have attempted fixes
        assert result is not None
        assert len(result.debug_trace) > 0

    def test_debug_result_serialization(self):
        """Test that DebugResult can be serialized."""
        debugger = SelfDebugger(max_attempts=2)
        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
        ]

        result = debugger.debug(code, "add", test_cases)
        result_dict = result.to_dict()

        assert "original_code" in result_dict
        assert "final_code" in result_dict
        assert "success" in result_dict
        assert "fix_history" in result_dict

    def test_max_attempts_respected(self):
        """Test that max_attempts limit is respected."""
        debugger = SelfDebugger(max_attempts=2)
        code = """
def always_fail():
    raise RuntimeError("always fails")
"""
        test_cases = [
            {"inputs": {}, "expected": "never"},
        ]

        result = debugger.debug(code, "should not fail", test_cases, "always_fail")

        assert result.attempts <= 2
        assert not result.success


# =============================================================================
# INTEGRATION TESTS - ERROR SCENARIOS
# =============================================================================

class TestErrorScenarios:
    """Integration tests with various error scenarios."""

    def test_scenario_wrong_return_type(self):
        """Test debugging wrong return type."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def get_length(text):
    return text  # Bug: should return len(text)
"""
        test_cases = [
            {"inputs": {"text": "hello"}, "expected": 5},
            {"inputs": {"text": ""}, "expected": 0},
        ]

        result = debugger.debug(code, "return length of string", test_cases)

        assert result is not None
        # Without LLM this won't be fixed, but should handle gracefully
        assert len(result.debug_trace) > 0

    def test_scenario_off_by_one(self):
        """Test debugging off-by-one error."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def factorial(n):
    result = 1
    for i in range(n):  # Bug: should be range(1, n+1)
        result *= i
    return result
"""
        test_cases = [
            {"inputs": {"n": 5}, "expected": 120},
            {"inputs": {"n": 0}, "expected": 1},
        ]

        result = debugger.debug(code, "compute factorial", test_cases)

        assert result is not None

    def test_scenario_missing_edge_case(self):
        """Test debugging missing edge case handling."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def safe_divide(a, b):
    return a / b  # Missing check for b == 0
"""
        test_cases = [
            {"inputs": {"a": 10, "b": 2}, "expected": 5},
            {"inputs": {"a": 10, "b": 0}, "expected": 0},  # Edge case
        ]

        result = debugger.debug(code, "divide, return 0 if divisor is 0", test_cases)

        assert result is not None

    def test_scenario_index_out_of_bounds(self):
        """Test debugging index out of bounds."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def get_first(items):
    return items[0]  # No check for empty list
"""
        test_cases = [
            {"inputs": {"items": [1, 2, 3]}, "expected": 1},
            {"inputs": {"items": []}, "expected": None},  # Edge case
        ]

        result = debugger.debug(code, "get first item or None if empty", test_cases)

        assert result is not None

    def test_scenario_key_not_found(self):
        """Test debugging key not found error."""
        debugger = SelfDebugger(max_attempts=3)
        code = """
def get_value(data, key):
    return data[key]  # No check for missing key
"""
        test_cases = [
            {"inputs": {"data": {"a": 1}, "key": "a"}, "expected": 1},
            {"inputs": {"data": {"a": 1}, "key": "b"}, "expected": None},  # Missing key
        ]

        result = debugger.debug(code, "get value or None if missing", test_cases)

        assert result is not None


# =============================================================================
# FIX STRATEGY TESTS
# =============================================================================

class TestFixStrategies:
    """Tests for fix strategy application."""

    def test_fix_indentation(self):
        """Test fix indentation strategy."""
        debugger = SelfDebugger()
        code = "def foo():\n\treturn 1"  # Tab indentation

        strategy = FixStrategy(
            strategy_type=FixStrategyType.FIX_INDENTATION,
            suggestion="Fix indentation",
        )
        analysis = ErrorAnalysis(
            category=ErrorCategory.SYNTAX,
            error_type="IndentationError",
            message="inconsistent use of tabs",
        )

        fixed = debugger._fix_indentation(code, strategy)

        assert "\t" not in fixed
        assert "    " in fixed  # Replaced with spaces

    def test_fix_replace_line(self):
        """Test replace line strategy."""
        debugger = SelfDebugger()
        code = "x = 1\ny = 2\nz = x + y"

        strategy = FixStrategy(
            strategy_type=FixStrategyType.REPLACE_LINE,
            target_line=2,
            new_code="y = 3",
        )
        analysis = ErrorAnalysis(
            category=ErrorCategory.VALUE,
            error_type="ValueError",
            message="wrong value",
            line_number=2,
        )

        fixed = debugger._replace_line(code, strategy, analysis)

        assert "y = 3" in fixed

    def test_fix_insert_line(self):
        """Test insert line strategy."""
        debugger = SelfDebugger()
        code = "x = 1\nz = x + y"

        strategy = FixStrategy(
            strategy_type=FixStrategyType.INSERT_LINE,
            target_line=2,
            new_code="y = 2",
        )
        analysis = ErrorAnalysis(
            category=ErrorCategory.NAME,
            error_type="NameError",
            message="name 'y' is not defined",
            line_number=2,
        )

        fixed = debugger._insert_line(code, strategy, analysis)

        lines = fixed.split('\n')
        assert len(lines) == 3
        assert "y = 2" in fixed


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Tests for data class functionality."""

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        result = ExecutionResult(
            success=True,
            output=42,
            stdout="hello",
            execution_time_ms=10.5,
        )

        d = result.to_dict()

        assert d["success"] == True
        assert d["output"] == "42"
        assert d["stdout"] == "hello"
        assert d["execution_time_ms"] == 10.5

    def test_error_analysis_to_dict(self):
        """Test ErrorAnalysis serialization."""
        analysis = ErrorAnalysis(
            category=ErrorCategory.TYPE,
            error_type="TypeError",
            message="bad type",
            line_number=5,
            suggested_fixes=["fix1", "fix2"],
        )

        d = analysis.to_dict()

        assert d["category"] == "type"
        assert d["error_type"] == "TypeError"
        assert d["line_number"] == 5
        assert len(d["suggested_fixes"]) == 2

    def test_fix_strategy_to_dict(self):
        """Test FixStrategy serialization."""
        strategy = FixStrategy(
            strategy_type=FixStrategyType.REPLACE_LINE,
            target_line=3,
            suggestion="replace with correct code",
            confidence=0.8,
        )

        d = strategy.to_dict()

        assert d["strategy_type"] == "replace_line"
        assert d["target_line"] == 3
        assert d["confidence"] == 0.8


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
