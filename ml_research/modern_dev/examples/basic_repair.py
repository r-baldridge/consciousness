"""
Basic Code Repair Example

This example shows how to repair simple bugs using the default pipeline.

Features demonstrated:
- MLOrchestrator initialization
- Basic code repair requests
- Handling responses
- Error message context

Usage:
    python -m modern_dev.examples.basic_repair
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modern_dev.orchestrator.router import MLOrchestrator, Request, Response


def repair_variable_typo() -> None:
    """Example 1: Repair a variable name typo."""
    print("=" * 60)
    print("Example 1: Variable Typo Repair")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = MLOrchestrator()

    # Buggy code with a typo: 'c' instead of 'b'
    buggy_code = """def add(a, b):
    return a + c
"""

    print(f"Buggy code:\n{buggy_code}")

    # Create repair request
    response = orchestrator.process(Request(
        task_type="code_repair",
        input_data={
            "buggy_code": buggy_code,
            "error_message": "NameError: name 'c' is not defined",
        },
    ))

    # Handle response
    print(f"\nSuccess: {response.success}")
    print(f"Architecture used: {response.architecture_used}")
    print(f"Execution time: {response.execution_time_ms:.2f}ms")

    if response.success:
        print(f"\nRepair output:\n{response.output}")
    else:
        print(f"\nError: {response.metadata}")

    print()


def repair_with_error_context() -> None:
    """Example 2: Repair with error message context."""
    print("=" * 60)
    print("Example 2: Division by Zero Fix")
    print("=" * 60)

    orchestrator = MLOrchestrator()

    # Code that can cause division by zero
    buggy_code = """def divide(a, b):
    return a / b
"""

    print(f"Buggy code:\n{buggy_code}")

    # Provide error context to help the model
    response = orchestrator.process(Request(
        task_type="code_repair",
        input_data={
            "buggy_code": buggy_code,
            "error_message": "ZeroDivisionError: division by zero when b=0",
        },
    ))

    print(f"\nSuccess: {response.success}")
    print(f"Architecture used: {response.architecture_used}")

    if response.success:
        print(f"\nRepair output:\n{response.output}")

    print()


def repair_with_test_cases() -> None:
    """Example 3: Repair with test cases for validation."""
    print("=" * 60)
    print("Example 3: Repair with Test Cases")
    print("=" * 60)

    orchestrator = MLOrchestrator()

    # Off-by-one error in range
    buggy_code = """def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""

    print(f"Buggy code:\n{buggy_code}")

    # Provide test cases to validate the fix
    response = orchestrator.process(Request(
        task_type="code_repair",
        input_data={
            "buggy_code": buggy_code,
            "error_message": "sum_range(5) returns 10, expected 15",
            "test_cases": [
                {"input": {"n": 5}, "expected": 15},  # 1+2+3+4+5 = 15
                {"input": {"n": 3}, "expected": 6},   # 1+2+3 = 6
                {"input": {"n": 1}, "expected": 1},
            ],
        },
    ))

    print(f"\nSuccess: {response.success}")
    print(f"Architecture used: {response.architecture_used}")

    if response.success:
        print(f"\nRepair output:\n{response.output}")

    print()


def repair_syntax_error() -> None:
    """Example 4: Repair a syntax error."""
    print("=" * 60)
    print("Example 4: Syntax Error Repair")
    print("=" * 60)

    orchestrator = MLOrchestrator()

    # Common typo: 'retrun' instead of 'return'
    buggy_code = """def greet(name):
    message = "Hello, " + name
    retrun message
"""

    print(f"Buggy code:\n{buggy_code}")

    response = orchestrator.process(Request(
        task_type="code_repair",
        input_data={
            "buggy_code": buggy_code,
            "error_message": "SyntaxError: invalid syntax",
        },
    ))

    print(f"\nSuccess: {response.success}")
    print(f"Architecture used: {response.architecture_used}")

    if response.success:
        print(f"\nRepair output:\n{response.output}")

    print()


def repair_with_context() -> None:
    """Example 5: Repair with surrounding code context."""
    print("=" * 60)
    print("Example 5: Repair with Code Context")
    print("=" * 60)

    orchestrator = MLOrchestrator()

    # Buggy function
    buggy_code = """def process_item(item):
    return item.value * 2
"""

    # Surrounding context provides class definition
    context = """class Item:
    def __init__(self, val):
        self.val = val  # Note: attribute is 'val', not 'value'

def create_item(x):
    return Item(x)
"""

    print(f"Context:\n{context}")
    print(f"Buggy code:\n{buggy_code}")

    response = orchestrator.process(Request(
        task_type="code_repair",
        input_data={
            "buggy_code": buggy_code,
            "context": context,
            "error_message": "AttributeError: 'Item' object has no attribute 'value'",
        },
    ))

    print(f"\nSuccess: {response.success}")
    print(f"Architecture used: {response.architecture_used}")

    if response.success:
        print(f"\nRepair output:\n{response.output}")

    print()


def check_orchestrator_status() -> None:
    """Show orchestrator capabilities and status."""
    print("=" * 60)
    print("Orchestrator Status")
    print("=" * 60)

    orchestrator = MLOrchestrator()
    status = orchestrator.get_status()

    print(f"Initialized: {status['initialized']}")
    print(f"Architectures: {', '.join(status['architectures'])}")
    print(f"Techniques: {', '.join(status['techniques'])}")
    print(f"Total routings: {status['routing_stats'].get('total_routings', 0)}")

    print("\nArchitecture details:")
    for name in status['architectures']:
        info = orchestrator.get_architecture_info(name)
        if info:
            print(f"\n  {info['name']}:")
            print(f"    Tasks: {', '.join(info['supported_tasks'])}")
            print(f"    Max context: {info['max_context_length']}")
            print(f"    Speed: {info['inference_speed']}")
            print(f"    Memory: {info['memory_requirement']}")

    print()


def main():
    """Run all basic repair examples."""
    print("\n" + "=" * 60)
    print("ML Research Code Repair - Basic Examples")
    print("=" * 60 + "\n")

    # Show status first
    check_orchestrator_status()

    # Run examples
    repair_variable_typo()
    repair_with_error_context()
    repair_with_test_cases()
    repair_syntax_error()
    repair_with_context()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
