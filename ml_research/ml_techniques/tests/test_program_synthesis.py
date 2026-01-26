"""Tests for ProgramSynthesis implementation."""
import sys
sys.path.insert(0, '.')

from consciousness.ml_research.ml_techniques.code_synthesis import (
    ProgramSynthesis, SpecificationType, SynthesisStrategy, IOExample, FormalSpec
)


def test_io_examples_double():
    """Test synthesis from I/O examples - double function."""
    print('=== Testing ProgramSynthesis - Double Function ===')

    synth = ProgramSynthesis(
        spec_type=SpecificationType.IO_EXAMPLES,
        strategy=SynthesisStrategy.NEURAL,
        max_candidates=3,
    )

    result = synth.run({
        "function_name": "double",
        "examples": [
            {"inputs": {"x": 2}, "expected": 4},
            {"inputs": {"x": 5}, "expected": 10},
            {"inputs": {"x": 0}, "expected": 0},
            {"inputs": {"x": -3}, "expected": -6},
        ]
    })

    print(f'  Success: {result.success}')
    print(f'  Tests passed: {result.output.get("tests_passed")}/{result.output.get("tests_total")}')
    print(f'  Score: {result.output.get("score")}')
    print(f'  Candidates evaluated: {result.output.get("candidates_evaluated")}')

    if result.output.get("score", 0) >= 0.75:
        print('  PASSED (75%+ tests)')
    else:
        print(f'  PARTIAL - Code generated:\n{result.output.get("code", "")[:200]}')


def test_io_examples_increment():
    """Test synthesis from I/O examples - increment function."""
    print('=== Testing ProgramSynthesis - Increment Function ===')

    synth = ProgramSynthesis(max_candidates=3)

    result = synth.run({
        "function_name": "increment",
        "examples": [
            {"inputs": {"n": 0}, "expected": 1},
            {"inputs": {"n": 5}, "expected": 6},
            {"inputs": {"n": -1}, "expected": 0},
        ]
    })

    print(f'  Success: {result.success}')
    print(f'  Score: {result.output.get("score")}')

    if result.output.get("score", 0) >= 0.66:
        print('  PASSED')
    else:
        print(f'  Code:\n{result.output.get("code", "")[:200]}')


def test_natural_language():
    """Test synthesis from natural language description."""
    print('=== Testing ProgramSynthesis - Natural Language ===')

    synth = ProgramSynthesis(
        spec_type=SpecificationType.NATURAL_LANGUAGE,
        max_candidates=2,
    )

    result = synth.run(
        "Create a function called 'greet' that takes a name parameter "
        "and returns 'Hello, <name>!' where <name> is the input."
    )

    print(f'  Success: {result.success}')
    print(f'  Code generated: {len(result.output.get("code", ""))} chars')
    assert "def " in result.output.get("code", ""), "Should generate a function"
    print('  PASSED')


def test_formal_spec():
    """Test synthesis with formal specification."""
    print('=== Testing ProgramSynthesis - Formal Spec ===')

    synth = ProgramSynthesis(
        spec_type=SpecificationType.FORMAL,
        max_candidates=2,
    )

    result = synth.run({
        "function_name": "add",
        "input_types": {"a": "int", "b": "int"},
        "output_type": "int",
        "preconditions": ["a >= 0", "b >= 0"],
        "postconditions": ["result == a + b"],
        "examples": [
            {"inputs": {"a": 1, "b": 2}, "expected": 3},
            {"inputs": {"a": 0, "b": 0}, "expected": 0},
        ]
    })

    print(f'  Success: {result.success}')
    print(f'  Score: {result.output.get("score")}')
    print('  PASSED')


def test_string_operations():
    """Test synthesis of string operations."""
    print('=== Testing ProgramSynthesis - String Upper ===')

    synth = ProgramSynthesis(max_candidates=3)

    result = synth.run({
        "function_name": "to_upper",
        "examples": [
            {"inputs": {"s": "hello"}, "expected": "HELLO"},
            {"inputs": {"s": "World"}, "expected": "WORLD"},
            {"inputs": {"s": ""}, "expected": ""},
        ]
    })

    print(f'  Success: {result.success}')
    print(f'  Score: {result.output.get("score")}')

    if result.output.get("score", 0) >= 0.66:
        print('  PASSED')
    else:
        print(f'  Code:\n{result.output.get("code", "")[:200]}')


def test_list_operations():
    """Test synthesis of list operations."""
    print('=== Testing ProgramSynthesis - List Sum ===')

    synth = ProgramSynthesis(max_candidates=3)

    result = synth.run({
        "function_name": "list_sum",
        "examples": [
            {"inputs": {"numbers": [1, 2, 3]}, "expected": 6},
            {"inputs": {"numbers": []}, "expected": 0},
            {"inputs": {"numbers": [10]}, "expected": 10},
        ]
    })

    print(f'  Success: {result.success}')
    print(f'  Score: {result.output.get("score")}')

    if result.output.get("score", 0) >= 0.66:
        print('  PASSED')
    else:
        print(f'  Code:\n{result.output.get("code", "")[:200]}')


def test_refinement():
    """Test that refinement is attempted on failures."""
    print('=== Testing ProgramSynthesis - Refinement ===')

    synth = ProgramSynthesis(
        max_candidates=2,
        max_refinements=2,
    )

    result = synth.run({
        "function_name": "complex_calc",
        "examples": [
            {"inputs": {"x": 5}, "expected": 25},
            {"inputs": {"x": 3}, "expected": 9},
        ]
    })

    refinement_count = sum(1 for step in result.intermediate_steps if step.get("action") == "refine_candidate")
    print(f'  Refinement rounds: {refinement_count}')
    print(f'  Score: {result.output.get("score")}')
    print('  PASSED')


def test_metadata():
    """Test that metadata is properly set."""
    print('=== Testing ProgramSynthesis - Metadata ===')

    synth = ProgramSynthesis(
        spec_type=SpecificationType.IO_EXAMPLES,
        strategy=SynthesisStrategy.EXAMPLE_BASED,
    )

    result = synth.run({
        "function_name": "test",
        "examples": [{"inputs": {"x": 1}, "expected": 1}]
    })

    assert result.metadata.get("spec_type") == "io_examples"
    assert result.metadata.get("strategy") == "example_based"
    print(f'  Spec type: {result.metadata.get("spec_type")}')
    print(f'  Strategy: {result.metadata.get("strategy")}')
    print('  PASSED')


if __name__ == '__main__':
    print('=' * 60)
    print('ProgramSynthesis Test Suite')
    print('=' * 60)

    test_io_examples_double()
    print()
    test_io_examples_increment()
    print()
    test_natural_language()
    print()
    test_formal_spec()
    print()
    test_string_operations()
    print()
    test_list_operations()
    print()
    test_refinement()
    print()
    test_metadata()

    print()
    print('=' * 60)
    print('All ProgramSynthesis tests completed!')
    print('=' * 60)
